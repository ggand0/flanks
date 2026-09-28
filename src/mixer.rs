//! Positional battle mixer. One Bevy audio player runs [`MixerStream`], a
//! source that mixes every positional battle sound itself: clips are
//! decoded once to mono PCM, and each voice carries its own left/right
//! gain, pitch and start delay. The main thread ranks each frame's sound
//! requests the way M2TW's sound banks do (priority, minus a per-metre
//! distance penalty) and hands the winners to the audio thread through a
//! queue that thread drains once per block.
//!
//! Why not Bevy's spatial audio: it is rodio's `Spatial` source, whose gain
//! falls as 1/d² and whose left/right term favours the far ear. And every
//! Bevy one-shot decodes its mp3 on the single mixer thread, which is what
//! capped the old one-shot pool at 64 voices.
//!
//! Distance model (M2TW descr_sounds.txt): the listener is the camera,
//! gain = min(1, mindist / d) (DirectSound inverse distance, rolloff 1),
//! and a sound whose distance gain falls under 1 percent is not played.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use bevy::asset::Asset;
use bevy::audio::{AudioSource, ChannelCount, Decodable, SampleRate, Source};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on};

/// Output rate of the mix. Every clip is resampled to it at decode time;
/// rodio converts the mix to the device rate.
pub const OUT_RATE: u32 = 48_000;
/// Frames mixed per block (5.3 ms at 48 kHz). Commands land on block
/// boundaries.
const BLOCK: usize = 256;
/// Voices the allocator keeps playing. Mixing decoded PCM costs a few
/// multiply-adds per voice per frame, so the budget is set by what stays
/// audible, not by CPU.
pub const MAX_VOICES: usize = 256;
/// M2TW `volume_cutoff .01`: a sound under 1 percent of full level is off.
const DIST_CUTOFF: f32 = 0.01;
/// Share of full left/right separation a hard-side source gets. Full
/// separation (one ear silent) reads as a sound inside one ear on
/// headphones.
const PAN_WIDTH: f32 = 0.8;
/// Fade applied when the allocator takes a voice for a stronger sound.
const STEAL_FADE_S: f32 = 0.03;
/// A sound must outrank a live voice by this much to take it, so two
/// near-equal sounds do not trade a voice back and forth (0.1 is 10 m of
/// the nearer-wins term).
const STEAL_MARGIN: f32 = 0.1;

/// One M2TW sound bank's playback rules (descr_sounds_*.txt columns).
#[derive(Clone, Copy, Debug)]
pub struct Bank {
    /// Names the bank in FL_LOG_AUDIO and for `max_live`.
    pub name: &'static str,
    /// Higher wins a voice.
    pub priority: f32,
    /// Priority change per metre from the listener (M2TW
    /// `distancepriority`, read as per metre: -2 for blows).
    pub dist_priority: f32,
    /// Full volume inside this distance, then gain = mindist / d.
    pub mindist: f32,
    /// Level of this bank's clips at mindist, in dB. Calibrated per pool
    /// so our clip loudness lands where M2TW's lands for the same bank.
    pub vol_db: f32,
    /// M2TW's config volume for the bank (dB). With `volume_cutoff .01`
    /// it sets the range: the sound is off where this level times the
    /// distance gain falls under 1 percent, so a -20 dB grunt carries
    /// 7.5 m and a 0 dB blow 75 m.
    pub cut_db: f32,
    /// Playback speed range (pitch and length together).
    pub speed: (f32, f32),
    /// Most voices this bank may hold at once (0: no limit).
    pub max_live: u16,
}

impl Bank {
    /// Distance where the bank falls under M2TW's volume cutoff.
    pub fn max_dist(&self) -> f32 {
        self.mindist * db_to_lin(self.cut_db) / DIST_CUTOFF
    }
}

/// A decoded clip in [`Clips`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipId(u16);

/// One sound the game wants this frame.
pub struct Request {
    pub clip: ClipId,
    pub bank: Bank,
    /// World position of the emitter (metres).
    pub pos: Vec3,
    /// 0..1 roll for the speed within the bank's range.
    pub speed_roll: f32,
    /// Seconds before the voice starts. Sim events arrive once per tick,
    /// so a tick's blows are spread over the tick instead of stacking on
    /// one frame.
    pub delay: f32,
}

/// Where the camera hears from: position and its right vector.
#[derive(Clone, Copy, Debug, Default)]
pub struct Listener {
    pub pos: Vec3,
    pub right: Vec3,
}

impl Listener {
    /// Left/right gain and the distance gain for a source at `pos`, or
    /// None when the source is past the cutoff.
    fn gains(&self, bank: &Bank, pos: Vec3) -> Option<([f32; 2], f32)> {
        let to = pos - self.pos;
        let d = to.length();
        let dist_gain = (bank.mindist / d.max(1e-3)).min(1.0);
        if d > bank.max_dist() {
            return None;
        }
        // Equal-power pan on the source's side of the listener.
        let side = if d > 1e-3 { to.dot(self.right) / d } else { 0.0 };
        let a = (side * PAN_WIDTH + 1.0) * std::f32::consts::FRAC_PI_4;
        let g = db_to_lin(bank.vol_db) * dist_gain;
        Some(([g * a.cos(), g * a.sin()], d))
    }
}

pub fn db_to_lin(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

// ---------------------------------------------------------------- audio thread

enum Cmd {
    Start {
        id: u32,
        pcm: Arc<[i16]>,
        gain: [f32; 2],
        speed: f32,
        looped: bool,
        delay: u32,
    },
    Gain {
        id: u32,
        gain: [f32; 2],
    },
    Stop {
        id: u32,
        fade: u32,
    },
    StopAll {
        fade: u32,
    },
}

#[derive(Default)]
struct Shared {
    cmds: Vec<Cmd>,
    finished: Vec<u32>,
    /// Audio-thread mixing time and blocks mixed since last read
    /// (FL_LOG_AUDIO).
    mix_ns: u64,
    mix_blocks: u32,
    /// Most voices mixed in one block since last read.
    mix_peak: u32,
}

/// Fractional bits of a voice's read position.
const FRAC_BITS: u32 = 32;

struct Voice {
    id: u32,
    pcm: Arc<[i16]>,
    /// Read position in clip frames, fixed point with FRAC_BITS.
    pos: u64,
    /// Frames advanced per output frame (the pitch), same fixed point.
    step: u64,
    looped: bool,
    gain: [f32; 2],
    target: [f32; 2],
    delay: u32,
    /// Fade envelope 0..1; `env_step` < 0 while fading out.
    env: f32,
    env_step: f32,
}

/// The asset Bevy plays: a handle to the shared command queue.
#[derive(Asset, TypePath)]
pub struct MixerStream {
    shared: Arc<Mutex<Shared>>,
}

impl Decodable for MixerStream {
    type Decoder = MixerSource;

    fn decoder(&self) -> MixerSource {
        MixerSource {
            shared: self.shared.clone(),
            voices: Vec::with_capacity(MAX_VOICES * 2),
            buf: vec![0.0; BLOCK * 2],
            at: BLOCK * 2,
            done: Vec::new(),
            limit: 1.0,
            stats: (0, 0, 0),
        }
    }
}

/// The endless stereo stream rodio pulls samples from.
pub struct MixerSource {
    shared: Arc<Mutex<Shared>>,
    voices: Vec<Voice>,
    buf: Vec<f32>,
    at: usize,
    done: Vec<u32>,
    /// Limiter gain: drops at once to keep a block's peak under 1,
    /// recovers over about 0.2 s.
    limit: f32,
    /// Mixing time not yet reported (ns, blocks, peak voices).
    stats: (u64, u32, u32),
}

impl MixerSource {
    fn apply(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Start {
                id,
                pcm,
                gain,
                speed,
                looped,
                delay,
            } => self.voices.push(Voice {
                id,
                pcm,
                pos: 0,
                step: (speed as f64 * (1u64 << FRAC_BITS) as f64) as u64,
                looped,
                gain,
                target: gain,
                delay,
                env: 1.0,
                env_step: 0.0,
            }),
            Cmd::Gain { id, gain } => {
                if let Some(v) = self.voices.iter_mut().find(|v| v.id == id) {
                    v.target = gain;
                }
            }
            Cmd::Stop { id, fade } => {
                if let Some(v) = self.voices.iter_mut().find(|v| v.id == id) {
                    v.env_step = -1.0 / fade.max(1) as f32;
                }
            }
            Cmd::StopAll { fade } => {
                for v in &mut self.voices {
                    v.env_step = -1.0 / fade.max(1) as f32;
                }
            }
        }
    }

    fn render(&mut self) {
        // Never block the audio thread: a busy lock just defers the
        // commands to the next block.
        let t0 = std::time::Instant::now();
        let cmds = match self.shared.try_lock() {
            Ok(mut sh) => {
                sh.finished.append(&mut self.done);
                sh.mix_ns += self.stats.0;
                sh.mix_blocks += self.stats.1;
                sh.mix_peak = sh.mix_peak.max(self.stats.2);
                self.stats = (0, 0, 0);
                std::mem::take(&mut sh.cmds)
            }
            Err(_) => Vec::new(),
        };
        for c in cmds {
            self.apply(c);
        }
        self.buf.fill(0.0);
        let inv_n = 1.0 / BLOCK as f32;
        let frac_scale = 1.0 / (1u64 << FRAC_BITS) as f32;
        for v in &mut self.voices {
            let pcm: &[i16] = &v.pcm;
            let mut ended = pcm.len() < 2;
            let mut f = 0usize;
            if !ended && v.delay > 0 {
                let d = (v.delay as usize).min(BLOCK);
                v.delay -= d as u32;
                f = d;
            }
            // The last read starts at frame len - 2 (it interpolates to
            // len - 1).
            let last = (pcm.len().max(2) as u64 - 1) << FRAC_BITS;
            // Gain glides to its target over the block (no zipper noise
            // when the camera moves).
            let (gl, gr) = (v.gain[0], v.gain[1]);
            let (dgl, dgr) = ((v.target[0] - gl) * inv_n, (v.target[1] - gr) * inv_n);
            let (mut pos, step) = (v.pos, v.step);
            let (mut env, env_step) = (v.env, v.env_step);
            while !ended && f < BLOCK {
                if pos >= last {
                    if v.looped {
                        pos -= last;
                        continue;
                    }
                    ended = true;
                    break;
                }
                let i = (pos >> FRAC_BITS) as usize;
                let frac = (pos & ((1u64 << FRAC_BITS) - 1)) as f32 * frac_scale;
                let a = pcm[i] as f32;
                let b = pcm[i + 1] as f32;
                let s = (a + (b - a) * frac) * (env * (1.0 / 32768.0));
                let t = f as f32;
                self.buf[2 * f] += s * (gl + dgl * t);
                self.buf[2 * f + 1] += s * (gr + dgr * t);
                pos += step;
                if env_step != 0.0 {
                    env = (env + env_step).clamp(0.0, 1.0);
                    ended = env <= 0.0;
                }
                f += 1;
            }
            v.pos = pos;
            v.env = env;
            v.gain = v.target;
            if ended {
                v.env = 0.0;
                v.env_step = -1.0;
            }
        }
        let done = &mut self.done;
        self.voices.retain(|v| {
            let keep = !(v.env <= 0.0 && v.env_step < 0.0);
            if !keep {
                done.push(v.id);
            }
            keep
        });

        // Peak limiter: a dense near fight can sum past full scale.
        let peak = self.buf.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        let want = if peak * self.limit > 0.97 {
            0.97 / peak
        } else {
            (self.limit + (1.0 - self.limit) * 0.03).min(1.0)
        };
        let from = self.limit;
        for (f, pair) in self.buf.chunks_exact_mut(2).enumerate() {
            let g = from + (want - from) * (f as f32 * inv_n);
            pair[0] = (pair[0] * g).clamp(-1.0, 1.0);
            pair[1] = (pair[1] * g).clamp(-1.0, 1.0);
        }
        self.limit = want;
        self.stats.0 += t0.elapsed().as_nanos() as u64;
        self.stats.1 += 1;
        self.stats.2 = self.stats.2.max(self.voices.len() as u32);
    }
}

impl Iterator for MixerSource {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.at >= self.buf.len() {
            self.render();
            self.at = 0;
        }
        let s = self.buf[self.at];
        self.at += 1;
        Some(s)
    }
}

impl Source for MixerSource {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> ChannelCount {
        ChannelCount::new(2).unwrap()
    }

    fn sample_rate(&self) -> SampleRate {
        SampleRate::new(OUT_RATE).unwrap()
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

// ---------------------------------------------------------------- main thread

struct Live {
    id: u32,
    bank: Bank,
    pos: Vec3,
    gain: [f32; 2],
    /// Real time the clip ends (f64::MAX for loops).
    ends: f64,
    /// The caller's key for a tracked loop.
    key: Option<u64>,
}

/// Fade of a tracked loop whose source is gone (M2TW ARROW_FLY
/// `fadeout .2`).
const TRACK_FADE_S: f32 = 0.2;

/// The main thread's side of the mixer: this frame's requests and the
/// voices it believes are playing.
#[derive(Resource)]
pub struct Mixer {
    shared: Arc<Mutex<Shared>>,
    live: Vec<Live>,
    requests: Vec<Request>,
    /// Tracked loops submitted this frame, by key.
    tracked: Vec<(u64, Request)>,
    cmds: Vec<Cmd>,
    next_id: u32,
    pub listener: Listener,
    /// Voices started, and requests dropped for want of a voice, since
    /// the counters were last read (FL_LOG_AUDIO).
    pub started: u32,
    pub dropped: u32,
}

impl Mixer {
    pub fn request(&mut self, r: Request) {
        self.requests.push(r);
    }

    /// A looped voice that follows a moving source (an arrow in flight).
    /// Submit every frame while the source lives, under a stable key; a
    /// key missing from a frame fades out.
    pub fn track(&mut self, key: u64, r: Request) {
        self.tracked.push((key, r));
    }

    /// FL_LOG_AUDIO line: live voices per bank, voices started and
    /// requests dropped since the last line.
    fn log_line(&mut self) -> String {
        let mut by_bank: Vec<(&str, u32)> = Vec::new();
        for l in &self.live {
            match by_bank.iter_mut().find(|b| b.0 == l.bank.name) {
                Some(b) => b.1 += 1,
                None => by_bank.push((l.bank.name, 1)),
            }
        }
        by_bank.sort_by_key(|b| std::cmp::Reverse(b.1));
        let banks: Vec<String> = by_bank.iter().map(|(n, c)| format!("{n} {c}")).collect();
        let (ns, blocks, peak) = {
            let mut sh = self.shared.lock().unwrap();
            let r = (sh.mix_ns, sh.mix_blocks, sh.mix_peak);
            (sh.mix_ns, sh.mix_blocks, sh.mix_peak) = (0, 0, 0);
            r
        };
        // Share of real time the audio thread spent mixing.
        let load = ns as f64 / (blocks.max(1) as f64 * BLOCK as f64 / OUT_RATE as f64 * 1e9);
        let line = format!(
            "mixer: {} live, {} started, {} dropped, mix {:.1}% of a core (peak {} voices) | {}",
            self.live.len(),
            self.started,
            self.dropped,
            100.0 * load,
            peak,
            banks.join(", ")
        );
        self.started = 0;
        self.dropped = 0;
        line
    }

    /// Stop every voice (leaving the battle).
    pub fn stop_all(&mut self) {
        self.requests.clear();
        self.tracked.clear();
        self.live.clear();
        self.cmds.push(Cmd::StopAll {
            fade: (0.1 * OUT_RATE as f32) as u32,
        });
    }

    /// M2TW's effective priority (priority + distancepriority x metres),
    /// with a small nearer-wins term so that among equal priorities the
    /// closer, louder sound keeps the voice.
    fn rank(bank: &Bank, d: f32) -> f32 {
        bank.priority + bank.dist_priority * d - 0.01 * d
    }

    /// Retire finished voices, re-aim the live ones at the moved listener,
    /// then give free or weaker voices to the strongest requests.
    fn flush(&mut self, clips: &Clips, now: f64) {
        {
            let mut sh = self.shared.lock().unwrap();
            let mut finished = std::mem::take(&mut sh.finished);
            sh.cmds.append(&mut self.cmds);
            drop(sh);
            finished.sort_unstable();
            self.live.retain(|l| finished.binary_search(&l.id).is_err());
        }
        self.live.retain(|l| l.ends > now);

        // Tracked loops: move the ones still reported, fade the rest.
        let mut tracked = std::mem::take(&mut self.tracked);
        tracked.sort_unstable_by_key(|t| t.0);
        let fade = (TRACK_FADE_S * OUT_RATE as f32) as u32;
        let cmds = &mut self.cmds;
        self.live.retain_mut(|l| {
            let Some(key) = l.key else { return true };
            match tracked.binary_search_by_key(&key, |t| t.0) {
                Ok(i) => {
                    l.pos = tracked[i].1.pos;
                    tracked[i].0 = u64::MAX;
                    true
                }
                Err(_) => {
                    cmds.push(Cmd::Stop { id: l.id, fade });
                    false
                }
            }
        });

        let listener = self.listener;
        for l in &mut self.live {
            let gain = listener.gains(&l.bank, l.pos).map(|(g, _)| g).unwrap_or([0.0; 2]);
            let moved = (gain[0] - l.gain[0]).abs() + (gain[1] - l.gain[1]).abs();
            if moved > 0.02 * (l.gain[0] + l.gain[1]).max(1e-4) {
                l.gain = gain;
                self.cmds.push(Cmd::Gain { id: l.id, gain });
            }
        }

        let new_tracked = tracked.into_iter().filter(|t| t.0 != u64::MAX).map(|(k, r)| (Some(k), r));
        let mut ranked: Vec<(f32, [f32; 2], Option<u64>, Request)> = std::mem::take(&mut self.requests)
            .into_iter()
            .map(|r| (None, r))
            .chain(new_tracked)
            .filter_map(|(key, r)| {
                let (gain, d) = listener.gains(&r.bank, r.pos)?;
                Some((Self::rank(&r.bank, d), gain, key, r))
            })
            .collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        // Once a request fails to take a voice from a full pool, every
        // lower-ranked one would fail against the same weakest voice.
        let mut pool_closed = false;
        for (rank, gain, key, r) in ranked {
            let Some(pcm) = clips.pcm(r.clip) else { continue };
            // A capped bank competes only with itself once full; otherwise
            // every voice is fair game.
            let capped = r.bank.max_live > 0
                && self.live.iter().filter(|l| l.bank.name == r.bank.name).count()
                    >= r.bank.max_live as usize;
            if !capped && pool_closed {
                if key.is_none() {
                    self.dropped += 1;
                }
                continue;
            }
            if capped || self.live.len() >= MAX_VOICES {
                // Take the weakest voice if this sound outranks it.
                let weakest = self
                    .live
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| !capped || l.bank.name == r.bank.name)
                    .map(|(i, l)| (i, Self::rank(&l.bank, l.pos.distance(listener.pos))))
                    .min_by(|a, b| a.1.total_cmp(&b.1));
                let Some((wi, weakest)) = weakest else {
                    if key.is_none() {
                        self.dropped += 1;
                    }
                    continue;
                };
                if rank <= weakest + STEAL_MARGIN {
                    // A tracked loop asks again next frame; only one-shots
                    // are lost.
                    if key.is_none() {
                        self.dropped += 1;
                    }
                    pool_closed |= !capped;
                    continue;
                }
                let victim = self.live.swap_remove(wi);
                self.cmds.push(Cmd::Stop {
                    id: victim.id,
                    fade: (STEAL_FADE_S * OUT_RATE as f32) as u32,
                });
            }
            let (lo, hi) = r.bank.speed;
            let speed = lo + (hi - lo) * r.speed_roll;
            let delay = (r.delay.max(0.0) * OUT_RATE as f32) as u32;
            let looped = key.is_some();
            let secs = if looped {
                f64::MAX
            } else {
                pcm.len() as f64 / (OUT_RATE as f64 * speed as f64) + r.delay as f64 + 0.05
            };
            self.next_id = self.next_id.wrapping_add(1);
            let id = self.next_id;
            self.cmds.push(Cmd::Start {
                id,
                pcm: pcm.clone(),
                gain,
                speed,
                looped,
                delay,
            });
            self.live.push(Live {
                id,
                bank: r.bank,
                pos: r.pos,
                gain,
                ends: now + secs,
                key,
            });
            self.started += 1;
        }
        if !self.cmds.is_empty() {
            self.shared.lock().unwrap().cmds.append(&mut self.cmds);
        }
    }
}

/// The mixer's Bevy player.
#[derive(Component)]
pub struct MixerPlayer;

/// Clips decoded to mono PCM at [`OUT_RATE`], loaded through the asset
/// server and decoded off the main thread as they arrive.
#[derive(Resource, Default)]
pub struct Clips {
    slots: Vec<ClipSlot>,
}

struct ClipSlot {
    handle: Handle<AudioSource>,
    pcm: Option<Arc<[i16]>>,
    task: Option<Task<Option<Arc<[i16]>>>>,
    failed: bool,
}

impl Clips {
    pub fn load(&mut self, assets: &AssetServer, path: &str) -> ClipId {
        let id = ClipId(self.slots.len() as u16);
        self.slots.push(ClipSlot {
            handle: assets.load(path.to_string()),
            pcm: None,
            task: None,
            failed: false,
        });
        id
    }

    /// Load `<name>.mp3` for every name.
    pub fn pool(&mut self, assets: &AssetServer, names: &[&str]) -> Vec<ClipId> {
        names.iter().map(|n| self.load(assets, &format!("{n}.mp3"))).collect()
    }

    fn pcm(&self, id: ClipId) -> Option<&Arc<[i16]>> {
        self.slots.get(id.0 as usize)?.pcm.as_ref()
    }
}

/// Decode to mono i16 at OUT_RATE (channel average, linear resample).
fn decode_mono(src: AudioSource) -> Option<Arc<[i16]>> {
    let dec = src.decoder();
    let ch = dec.channels().get() as usize;
    let rate = dec.sample_rate().get();
    let raw: Vec<f32> = dec.collect();
    if raw.is_empty() {
        return None;
    }
    let mono: Vec<f32> = raw.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect();
    let out: Vec<f32> = if rate == OUT_RATE {
        mono
    } else {
        let step = rate as f64 / OUT_RATE as f64;
        let n = ((mono.len() - 1) as f64 / step) as usize;
        (0..n)
            .map(|k| {
                let p = k as f64 * step;
                let i = p as usize;
                let t = (p - i as f64) as f32;
                mono[i] + (mono[(i + 1).min(mono.len() - 1)] - mono[i]) * t
            })
            .collect()
    };
    Some(out.iter().map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16).collect())
}

fn decode_clips(mut clips: ResMut<Clips>, sources: Res<Assets<AudioSource>>) {
    for slot in &mut clips.slots {
        if slot.pcm.is_some() || slot.failed {
            continue;
        }
        if let Some(task) = &slot.task {
            if task.is_finished() {
                let task = slot.task.take().unwrap();
                match block_on(task) {
                    Some(pcm) => slot.pcm = Some(pcm),
                    None => slot.failed = true,
                }
            }
            continue;
        }
        if let Some(src) = sources.get(&slot.handle) {
            let src = src.clone();
            slot.task = Some(AsyncComputeTaskPool::get().spawn(async move { decode_mono(src) }));
        }
    }
}

fn setup_mixer(mut commands: Commands, mut streams: ResMut<Assets<MixerStream>>) {
    let shared = Arc::new(Mutex::new(Shared::default()));
    let handle = streams.add(MixerStream {
        shared: shared.clone(),
    });
    commands.spawn((AudioPlayer::<MixerStream>(handle), MixerPlayer));
    commands.insert_resource(Mixer {
        shared,
        live: Vec::with_capacity(MAX_VOICES),
        requests: Vec::new(),
        tracked: Vec::new(),
        cmds: Vec::new(),
        next_id: 0,
        listener: Listener::default(),
        started: 0,
        dropped: 0,
    });
}

/// Aim the listener at the camera, set the master volume, and hand this
/// frame's requests to the audio thread. Runs after every system that
/// requests sounds.
#[allow(clippy::too_many_arguments)] // bevy system params
pub fn flush_mixer(
    mut mixer: ResMut<Mixer>,
    clips: Res<Clips>,
    camera: Query<&Transform, With<crate::camera::RtsCamera>>,
    time: Res<Time<Real>>,
    settings: Res<crate::settings::Settings>,
    mut sink: Query<&mut bevy::audio::AudioSink, With<MixerPlayer>>,
    virt_time: Res<Time<Virtual>>,
    mut next_log: Local<f64>,
) {
    if let Ok(t) = camera.single() {
        mixer.listener = Listener {
            pos: t.translation,
            right: *t.right(),
        };
    }
    if let Ok(mut s) = sink.single_mut() {
        use bevy::audio::AudioSinkPlayback;
        let v = crate::audio::battle_vol(&settings);
        if (s.volume().to_linear() - v).abs() > 1e-3 {
            s.set_volume(bevy::audio::Volume::Linear(v));
        }
        // A paused battle holds every battlefield voice where it is.
        let paused = virt_time.is_paused();
        if paused != s.is_paused() {
            if paused {
                s.pause();
            } else {
                s.play();
            }
        }
    }
    let now = time.elapsed_secs_f64();
    mixer.flush(&clips, now);
    static LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *LOG.get_or_init(|| std::env::var("FL_LOG_AUDIO").is_ok()) && now >= *next_log {
        *next_log = now + 1.0;
        info!("{}", mixer.log_line());
    }
}

fn stop_mixer(mut mixer: ResMut<Mixer>) {
    mixer.stop_all();
    let cmds = std::mem::take(&mut mixer.cmds);
    mixer.shared.lock().unwrap().cmds.extend(cmds);
}

pub struct MixerPlugin;

impl Plugin for MixerPlugin {
    fn build(&self, app: &mut App) {
        use bevy::audio::AddAudioSource;
        app.add_audio_source::<MixerStream>()
            .init_resource::<Clips>()
            .add_systems(Startup, setup_mixer)
            .add_systems(Update, decode_clips)
            .add_systems(OnExit(crate::game_state::GameState::Battle), stop_mixer);
    }
}
