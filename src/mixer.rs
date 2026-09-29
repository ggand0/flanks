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
//! Levels: every clip plays at a common loudness (the manifest written by
//! tools/audio_loudness.py), each bank sets its layer's level on top, and
//! each bank group caps how many voices it may hold.
//!
//! Distance: level falls from the point the camera looks at, full inside a
//! bank's `mindist` and mindist / d beyond (-6 dB per doubling), times the
//! zoom fade. Left/right comes from the source's direction from the
//! camera, so the stereo image follows the screen.

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
/// A sound whose distance gain falls under 1 percent is not played.
const DIST_CUTOFF: f32 = 0.01;
/// Most bank groups the output meter tracks.
const METER_GROUPS: usize = 32;
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

/// One sound bank's playback rules. Priority and distance priority are
/// M2TW's (descr_sounds_*.txt).
#[derive(Clone, Copy, Debug)]
pub struct Bank {
    /// Names the bank in FL_LOG_AUDIO.
    pub name: &'static str,
    /// Higher wins a voice.
    pub priority: f32,
    /// Priority change per metre from the listener (M2TW
    /// `distancepriority`, read as per metre: -2 for blows).
    pub dist_priority: f32,
    /// Full level inside this distance from the look point, then
    /// mindist / d.
    pub mindist: f32,
    /// Level of a normalized clip of this bank at full level, in dB.
    pub vol_db: f32,
    /// Least zoom fade the bank keeps (a regiment's roar still reaches a
    /// zoomed-out camera; one man's grunt does not).
    pub zoom_floor: f32,
    /// Playback speed range (pitch and length together).
    pub speed: (f32, f32),
    /// Banks sharing a group share its voice cap and its meter.
    pub group: &'static str,
    /// Most voices the group may hold at once (0: no limit). The nearest
    /// sounds keep the voices.
    pub max_live: u16,
    /// Part in the crowd dip under a nearby volley.
    pub duck: Duck,
}

/// The crowd dip: while a `Trigger` sound plays within DUCK_RADIUS_M of
/// the look point, every `Target` sound dips by up to DUCK_DB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Duck {
    None,
    /// Sets the dip off (the volley, the bow strings).
    Trigger,
    /// Dips under it (the charge yells, grunts and screams).
    Target,
}

/// How far the crowd dips under a nearby volley (dB).
const DUCK_DB: f32 = 6.0;
/// A volley counts as nearby within this distance of the look point (m).
const DUCK_RADIUS_M: f32 = 30.0;
/// The dip's attack and release (s).
const DUCK_ATTACK_S: f32 = 0.05;
const DUCK_RELEASE_S: f32 = 0.5;

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
    /// The caller's tag for voices it may fade out together later
    /// (`Mixer::fade_owner`); 0 for none.
    pub owner: u64,
    /// The regiment the sound comes from (`NO_UNIT` for none), for the
    /// FL_LOG_AUDIO clip log.
    pub unit: u32,
}

/// `Request::unit` for a sound that belongs to no regiment.
pub const NO_UNIT: u32 = u32::MAX;

/// A voice start, kept for the FL_LOG_AUDIO clip log.
pub struct StartRecord {
    pub clip: ClipId,
    pub bank: &'static str,
    /// Metres from the look point.
    pub dist: f32,
    pub unit: u32,
}

/// FL_LOG_AUDIO is set.
pub fn log_enabled() -> bool {
    static LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LOG.get_or_init(|| std::env::var("FL_LOG_AUDIO").is_ok())
}

/// Write one FL_LOG_AUDIO line to the console and to this launch's audio
/// log, tmp/runs/audio/audio-<unix seconds>.log in the repo, stamped with
/// seconds since launch.
pub fn audio_log(t: f64, line: &str) {
    use std::io::Write;
    static FILE: std::sync::OnceLock<Option<Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    info!("{line}");
    let file = FILE.get_or_init(|| {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tmp/runs/audio");
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let path = dir.join(format!("audio-{stamp}.log"));
        let file = std::fs::create_dir_all(&dir).and_then(|_| std::fs::File::create(&path));
        match file {
            Ok(f) => {
                info!("audio log: {}", path.display());
                Some(Mutex::new(f))
            }
            Err(e) => {
                warn!("audio log: cannot write {}: {e}", path.display());
                None
            }
        }
    });
    if let Some(f) = file
        && let Ok(mut f) = f.lock()
    {
        let _ = writeln!(f, "{t:8.1} {line}");
    }
}

/// Where the battle is heard from: level by distance from the point the
/// camera looks at, left/right by direction from the camera.
#[derive(Clone, Copy, Debug)]
pub struct Listener {
    /// The camera's focus on the ground.
    pub pos: Vec3,
    /// The camera itself.
    pub eye: Vec3,
    pub right: Vec3,
    /// Zoom fade, 1 at close zoom.
    pub zoom: f32,
    /// The crowd dip's current gain on `Duck::Target` sounds.
    pub duck: f32,
}

impl Default for Listener {
    fn default() -> Self {
        Self {
            pos: Vec3::ZERO,
            eye: Vec3::ZERO,
            right: Vec3::X,
            zoom: 1.0,
            duck: 1.0,
        }
    }
}

impl Listener {
    /// Ground distance from the look point.
    pub fn dist(&self, pos: Vec3) -> f32 {
        pos.xz().distance(self.pos.xz())
    }

    /// Left/right gain and the distance for a source at `pos`, or None
    /// when the source is past the cutoff.
    fn gains(&self, bank: &Bank, pos: Vec3) -> Option<([f32; 2], f32)> {
        let d = self.dist(pos);
        let dist_gain = (bank.mindist / d.max(1e-3)).min(1.0);
        if dist_gain < DIST_CUTOFF {
            return None;
        }
        // Equal-power pan by the source's direction from the camera.
        let side = (pos - self.eye).normalize_or_zero().dot(self.right);
        let a = (side * PAN_WIDTH + 1.0) * std::f32::consts::FRAC_PI_4;
        let mut g = db_to_lin(bank.vol_db) * dist_gain * self.zoom.max(bank.zoom_floor);
        if bank.duck == Duck::Target {
            g *= self.duck;
        }
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
        meter: u8,
        /// Not placed in the world (UI, stings): kept when a battle's
        /// placed voices are stopped.
        flat: bool,
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
        keep_flat: bool,
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
    /// Output energy per bank group and in total, and the frames it was
    /// summed over, since last read.
    meter: [f64; METER_GROUPS],
    meter_total: f64,
    meter_frames: u64,
    /// Mean output power over the last ~0.3 s (the beds dip under it).
    recent_ms: f32,
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
    meter: u8,
    flat: bool,
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
            meter: [0.0; METER_GROUPS],
            meter_total: 0.0,
            meter_frames: 0,
            recent_ms: 0.0,
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
    /// Output energy not yet reported, per group and in total, and frames.
    meter: [f64; METER_GROUPS],
    meter_total: f64,
    meter_frames: u64,
    recent_ms: f32,
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
                meter,
                flat,
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
                meter,
                flat,
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
            Cmd::StopAll { fade, keep_flat } => {
                for v in self.voices.iter_mut().filter(|v| !(keep_flat && v.flat)) {
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
                for (a, b) in sh.meter.iter_mut().zip(self.meter.iter_mut()) {
                    *a += *b;
                    *b = 0.0;
                }
                sh.meter_total += self.meter_total;
                sh.meter_frames += self.meter_frames;
                sh.recent_ms = self.recent_ms;
                (self.meter_total, self.meter_frames) = (0.0, 0);
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
            let mut energy = 0.0f32;
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
                let (l, r) = (s * (gl + dgl * t), s * (gr + dgr * t));
                self.buf[2 * f] += l;
                self.buf[2 * f + 1] += r;
                energy += l * l + r * r;
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
            self.meter[v.meter as usize % METER_GROUPS] += energy as f64;
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

        // Output power for the meter and the bed dip.
        let block_energy: f32 = self.buf.iter().map(|s| s * s).sum();
        self.meter_total += block_energy as f64;
        self.meter_frames += BLOCK as u64;
        let ms = block_energy / (2 * BLOCK) as f32;
        // About 0.3 s to follow (56 blocks of 5.3 ms).
        self.recent_ms += (ms - self.recent_ms) * (1.0 / 56.0);

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
    /// The clip's normalization gain (linear).
    clip_gain: f32,
    owner: u64,
}

/// Fade of a tracked loop whose source is gone (M2TW ARROW_FLY
/// `fadeout .2`).
const TRACK_FADE_S: f32 = 0.2;

/// The main thread's side of the mixer: this frame's requests and the
/// voices it believes are playing.
#[derive(Resource)]
pub struct Mixer {
    shared: Arc<Mutex<Shared>>,
    /// The UI channel's queue (its own player, under the UI volume).
    ui_shared: Arc<Mutex<Shared>>,
    /// Unplaced sounds asked for this frame.
    flat: Vec<(Bus, ClipId, Bank)>,
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
    /// Bank group names by meter index.
    meter_names: Vec<&'static str>,
    /// Voice starts since the clip log last read them (FL_LOG_AUDIO).
    pub start_log: Vec<StartRecord>,
    /// Banks of playing voices cut for a stronger sound since the clip
    /// log last read them (FL_LOG_AUDIO).
    pub cut_log: Vec<&'static str>,
    /// Mean power of the positional mix over the last ~0.3 s.
    recent_ms: f32,
    /// The crowd dip now (dB), and when it was last updated.
    duck_db: f32,
    duck_at: f64,
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
        let (ns, blocks, peak, meter, total, frames) = {
            let mut sh = self.shared.lock().unwrap();
            let r = (sh.mix_ns, sh.mix_blocks, sh.mix_peak, sh.meter, sh.meter_total, sh.meter_frames);
            (sh.mix_ns, sh.mix_blocks, sh.mix_peak) = (0, 0, 0);
            sh.meter = [0.0; METER_GROUPS];
            (sh.meter_total, sh.meter_frames) = (0.0, 0);
            r
        };
        // Metered output level per group over the line's second, in dBFS
        // (power per channel, before the limiter).
        let db = |e: f64| 10.0 * (e / (2.0 * frames.max(1) as f64)).max(1e-12).log10();
        let mut levels: Vec<(f64, &str)> = self
            .meter_names
            .iter()
            .enumerate()
            .filter(|(i, _)| meter[*i] > 0.0)
            .map(|(i, n)| (db(meter[i]), *n))
            .collect();
        levels.sort_by(|a, b| b.0.total_cmp(&a.0));
        let levels: Vec<String> = levels.iter().map(|(d, n)| format!("{n} {d:.0}")).collect();
        // Share of real time the audio thread spent mixing.
        let load = ns as f64 / (blocks.max(1) as f64 * BLOCK as f64 / OUT_RATE as f64 * 1e9);
        let line = format!(
            "mixer: {} live, {} started, {} dropped, mix {:.1}% of a core (peak {} voices), crowd dip {:.1} dB | {} | out {:.0} dBFS: {}",
            self.live.len(),
            self.started,
            self.dropped,
            100.0 * load,
            peak,
            self.duck_db,
            banks.join(", "),
            db(total),
            levels.join(", ")
        );
        self.started = 0;
        self.dropped = 0;
        line
    }

    /// Fade out, over `secs`, every playing voice the caller tagged with
    /// `owner`.
    pub fn fade_owner(&mut self, owner: u64, secs: f32) {
        let fade = (secs * OUT_RATE as f32) as u32;
        let cmds = &mut self.cmds;
        self.live.retain(|l| {
            if l.owner == owner {
                cmds.push(Cmd::Stop { id: l.id, fade });
                false
            } else {
                true
            }
        });
    }

    /// Loudness of the positional mix over the last ~0.3 s, in dBFS.
    pub fn recent_db(&self) -> f32 {
        10.0 * self.recent_ms.max(1e-12).log10()
    }

    fn meter_index(&mut self, group: &'static str) -> u8 {
        match self.meter_names.iter().position(|g| *g == group) {
            Some(i) => i as u8,
            None => {
                self.meter_names.push(group);
                (self.meter_names.len() - 1) as u8
            }
        }
    }

    /// An unplaced sound (UI click, sting) on `bus`, at the bank's level,
    /// centred.
    pub fn play_flat(&mut self, bus: Bus, clip: ClipId, bank: Bank) {
        self.flat.push((bus, clip, bank));
    }

    /// Stop every placed voice (leaving the battle). Unplaced ones, the
    /// outcome sting among them, play on into the results screen.
    pub fn stop_all(&mut self) {
        self.requests.clear();
        self.tracked.clear();
        self.live.clear();
        self.cmds.push(Cmd::StopAll {
            fade: (0.1 * OUT_RATE as f32) as u32,
            keep_flat: true,
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
            self.recent_ms = sh.recent_ms;
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

        // The crowd dip: set off by a volley or bow strings playing near the
        // look point, following them in fast and letting go slowly.
        let dt = (now - self.duck_at).clamp(0.0, 0.1) as f32;
        self.duck_at = now;
        let near_volley = self
            .live
            .iter()
            .any(|l| l.bank.duck == Duck::Trigger && self.listener.dist(l.pos) < DUCK_RADIUS_M);
        let want = if near_volley { DUCK_DB } else { 0.0 };
        let tau = if want > self.duck_db { DUCK_ATTACK_S } else { DUCK_RELEASE_S };
        self.duck_db += (want - self.duck_db) * (dt / tau).min(1.0);
        self.listener.duck = db_to_lin(-self.duck_db);

        let listener = self.listener;
        for l in &mut self.live {
            let gain = listener
                .gains(&l.bank, l.pos)
                .map(|(g, _)| [g[0] * l.clip_gain, g[1] * l.clip_gain])
                .unwrap_or([0.0; 2]);
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
        // Live voices per cap group.
        let mut groups: Vec<(&'static str, u16)> = Vec::new();
        for l in &self.live {
            group_add(&mut groups, l.bank.group, 1);
        }
        for (rank, gain, key, r) in ranked {
            let Some((pcm, clip_gain)) = clips.pcm(r.clip) else { continue };
            let gain = [gain[0] * clip_gain, gain[1] * clip_gain];
            // A capped group competes only with itself once full;
            // otherwise every voice is fair game.
            let capped = r.bank.max_live > 0 && group_count(&groups, r.bank.group) >= r.bank.max_live;
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
                    .filter(|(_, l)| !capped || l.bank.group == r.bank.group)
                    .map(|(i, l)| (i, Self::rank(&l.bank, listener.dist(l.pos))))
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
                group_add(&mut groups, victim.bank.group, -1);
                if log_enabled() && victim.key.is_none() {
                    self.cut_log.push(victim.bank.name);
                }
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
            let meter = self.meter_index(r.bank.group);
            self.cmds.push(Cmd::Start {
                id,
                pcm: pcm.clone(),
                gain,
                speed,
                looped,
                delay,
                meter,
                flat: false,
            });
            group_add(&mut groups, r.bank.group, 1);
            if log_enabled() {
                self.start_log.push(StartRecord {
                    clip: r.clip,
                    bank: r.bank.name,
                    dist: listener.dist(r.pos),
                    unit: r.unit,
                });
            }
            self.live.push(Live {
                id,
                bank: r.bank,
                pos: r.pos,
                gain,
                ends: now + secs,
                key,
                clip_gain,
                owner: r.owner,
            });
            self.started += 1;
        }
        // Unplaced sounds: no distance, no pan, no voice ranking.
        let mut ui_cmds = Vec::new();
        for (bus, clip, bank) in std::mem::take(&mut self.flat) {
            let Some((pcm, clip_gain)) = clips.pcm(clip) else { continue };
            let g = db_to_lin(bank.vol_db) * clip_gain;
            self.next_id = self.next_id.wrapping_add(1);
            let meter = self.meter_index(bank.group);
            let cmd = Cmd::Start {
                id: self.next_id,
                pcm: pcm.clone(),
                gain: [g, g],
                speed: 1.0,
                looped: false,
                delay: 0,
                meter,
                flat: true,
            };
            match bus {
                Bus::Battle => self.cmds.push(cmd),
                Bus::Ui => ui_cmds.push(cmd),
            }
        }
        {
            let mut ui = self.ui_shared.lock().unwrap();
            ui.cmds.append(&mut ui_cmds);
            // Nothing on the UI channel is tracked; drop its reports.
            ui.finished.clear();
            ui.meter = [0.0; METER_GROUPS];
            (ui.meter_total, ui.meter_frames) = (0.0, 0);
        }
        if !self.cmds.is_empty() {
            self.shared.lock().unwrap().cmds.append(&mut self.cmds);
        }
    }
}

fn group_add(groups: &mut Vec<(&'static str, u16)>, g: &'static str, n: i32) {
    match groups.iter_mut().find(|e| e.0 == g) {
        Some(e) => e.1 = (e.1 as i32 + n).max(0) as u16,
        None => groups.push((g, n.max(0) as u16)),
    }
}

fn group_count(groups: &[(&'static str, u16)], g: &'static str) -> u16 {
    groups.iter().find(|e| e.0 == g).map_or(0, |e| e.1)
}

/// The mixer's two output channels: placed and unplaced battle sounds
/// under the battle volume, UI clicks under the UI volume.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bus {
    Battle,
    Ui,
}

/// A mixer channel's Bevy player.
#[derive(Component)]
pub struct MixerPlayer(Bus);

/// Clips decoded to mono PCM at [`OUT_RATE`], loaded through the asset
/// server and decoded off the main thread as they arrive.
#[derive(Resource, Default)]
pub struct Clips {
    slots: Vec<ClipSlot>,
}

struct ClipSlot {
    handle: Handle<AudioSource>,
    /// Path under assets/.
    path: String,
    /// Normalization gain from the loudness manifest (linear).
    gain: f32,
    pcm: Option<Arc<[i16]>>,
    task: Option<Task<Option<Arc<[i16]>>>>,
    failed: bool,
}

impl Clips {
    /// Load a clip at `path` (under assets/), played with `gain_db` so it
    /// lands on the common loudness.
    pub fn load(&mut self, assets: &AssetServer, path: &str, gain_db: f32) -> ClipId {
        let id = ClipId(self.slots.len() as u16);
        self.slots.push(ClipSlot {
            handle: assets.load(path.to_string()),
            path: path.to_string(),
            gain: db_to_lin(gain_db),
            pcm: None,
            task: None,
            failed: false,
        });
        id
    }

    /// The clip's path under assets/.
    pub fn path(&self, id: ClipId) -> &str {
        self.slots.get(id.0 as usize).map_or("?", |s| s.path.as_str())
    }

    fn pcm(&self, id: ClipId) -> Option<(&Arc<[i16]>, f32)> {
        let slot = self.slots.get(id.0 as usize)?;
        Some((slot.pcm.as_ref()?, slot.gain))
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
    let ui_shared = Arc::new(Mutex::new(Shared::default()));
    for (bus, sh) in [(Bus::Battle, &shared), (Bus::Ui, &ui_shared)] {
        let handle = streams.add(MixerStream { shared: sh.clone() });
        commands.spawn((AudioPlayer::<MixerStream>(handle), MixerPlayer(bus)));
    }
    commands.insert_resource(Mixer {
        shared,
        ui_shared,
        flat: Vec::new(),
        live: Vec::with_capacity(MAX_VOICES),
        requests: Vec::new(),
        tracked: Vec::new(),
        cmds: Vec::new(),
        next_id: 0,
        listener: Listener::default(),
        started: 0,
        dropped: 0,
        meter_names: Vec::new(),
        start_log: Vec::new(),
        cut_log: Vec::new(),
        recent_ms: 0.0,
        duck_db: 0.0,
        duck_at: 0.0,
    });
}

/// Aim the listener at the camera, set the master volume, and hand this
/// frame's requests to the audio thread. Runs after every system that
/// requests sounds.
#[allow(clippy::too_many_arguments)] // bevy system params
pub fn flush_mixer(
    mut mixer: ResMut<Mixer>,
    clips: Res<Clips>,
    camera: Query<(&Transform, &crate::camera::RtsCamera)>,
    time: Res<Time<Real>>,
    settings: Res<crate::settings::Settings>,
    mut sinks: Query<(&MixerPlayer, &mut bevy::audio::AudioSink)>,
    virt_time: Res<Time<Virtual>>,
    mut next_log: Local<f64>,
) {
    if let Ok((t, cam)) = camera.single() {
        let duck = mixer.listener.duck;
        mixer.listener = Listener {
            pos: cam.focus,
            eye: t.translation,
            right: *t.right(),
            zoom: crate::audio::zoom_attenuation(cam.distance),
            duck,
        };
    }
    for (player, mut s) in &mut sinks {
        use bevy::audio::AudioSinkPlayback;
        let v = match player.0 {
            Bus::Battle => crate::audio::battle_vol(&settings),
            Bus::Ui => crate::audio::ui_vol(&settings),
        };
        if (s.volume().to_linear() - v).abs() > 1e-3 {
            s.set_volume(bevy::audio::Volume::Linear(v));
        }
        // A paused battle holds every battlefield voice where it is; UI
        // clicks still sound.
        let paused = player.0 == Bus::Battle && virt_time.is_paused();
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
    if log_enabled() && now >= *next_log {
        *next_log = now + 1.0;
        let line = mixer.log_line();
        audio_log(now, &line);
    }
}

fn stop_mixer(mut mixer: ResMut<Mixer>) {
    mixer.stop_all();
    let cmds = std::mem::take(&mut mixer.cmds);
    mixer.shared.lock().unwrap().cmds.extend(cmds);
}

/// Back at the menu: stop the unplaced voices too (a sting's tail). The
/// first menu comes before Startup has made the mixer.
fn stop_flat(mixer: Option<Res<Mixer>>) {
    let Some(mixer) = mixer else { return };
    for sh in [&mixer.shared, &mixer.ui_shared] {
        sh.lock().unwrap().cmds.push(Cmd::StopAll {
            fade: (0.1 * OUT_RATE as f32) as u32,
            keep_flat: false,
        });
    }
}

pub struct MixerPlugin;

impl Plugin for MixerPlugin {
    fn build(&self, app: &mut App) {
        use bevy::audio::AddAudioSource;
        app.add_audio_source::<MixerStream>()
            .init_resource::<Clips>()
            .add_systems(Startup, setup_mixer)
            .add_systems(Update, decode_clips)
            .add_systems(OnExit(crate::game_state::GameState::Battle), stop_mixer)
            .add_systems(OnEnter(crate::game_state::GameState::Menu), stop_flat);
    }
}
