//! Battle audio: aggregate beds + rate-limited one-shots. NEVER per-unit
//! sound — at 200k units the mix is driven by sim statistics near the
//! camera (hits/tick, engaged regiments, morale transitions), exactly the
//! signals the overlay already trusts.
//!
//! Asset list, generation prompts, and retry notes: tmp/audio-plan.md.
//! Missing files degrade gracefully (their triggers just stay silent).

use bevy::audio::{AudioSink, AudioSinkPlayback, PlaybackSettings, Volume};
use bevy::prelude::*;

use crate::camera::RtsCamera;
use crate::combat::CombatStats;
use crate::movement::SimStats;
use crate::orders::{Groups, Order, RegState, Selection};
use crate::units::hash01;

/// Env master override (FL_VOLUME, linear). Multiplies the settings
/// volumes so scripted test runs stay muted regardless of the saved
/// settings file.
pub fn env_master() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::util::env_or("FL_VOLUME", 1.0))
}

/// Effective battle-sound volume (beds, combat, vox, horns, stings).
fn battle_vol(s: &crate::settings::Settings) -> f32 {
    s.audio.master * s.audio.battle * env_master()
}

/// Effective UI-sound volume (selection/order clicks).
fn ui_vol(s: &crate::settings::Settings) -> f32 {
    s.audio.master * s.audio.ui * env_master()
}

/// Bed smoothing time constant (seconds to ~2/3 of the way to target).
const BED_SMOOTH: f32 = 0.35;

pub struct BattleAudioPlugin;

impl Plugin for BattleAudioPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_audio).add_systems(
            Update,
            (update_beds, combat_one_shots, event_cues).chain(),
        );
    }
}

#[derive(Resource)]
struct AudioBank {
    clang: Vec<Handle<AudioSource>>,
    shield: Vec<Handle<AudioSource>>,
    /// Flesh/armor damage connects (blunt, spear, sword) — mixed into the
    /// hit pool at low probability so not every hit rings like a bell.
    damage: Vec<Handle<AudioSource>>,
    death: Vec<Handle<AudioSource>>,
    vox_rout: Vec<Handle<AudioSource>>,
    vox_rally: Vec<Handle<AudioSource>>,
    /// Victory cheer when an ENEMY regiment breaks (owner: the
    /// _celebrate takes are "we broke them", not rally-from-rout).
    vox_cheer: Vec<Handle<AudioSource>>,
    /// Played once per regiment charge (the latched `charging` state).
    vox_warcry: Vec<Handle<AudioSource>>,
    horn_charge: Vec<Handle<AudioSource>>,
    horn_rout: Handle<AudioSource>,
    ui_select: Handle<AudioSource>,
    /// Click feedback for a move order (quieter than select per owner).
    ui_order: Handle<AudioSource>,
    /// Click feedback for an attack order on an enemy regiment.
    ui_attack: Handle<AudioSource>,
    sting_victory: Handle<AudioSource>,
    sting_defeat: Handle<AudioSource>,
}

/// Looping bed entities, indexed by `Bed`.
#[derive(Component, Clone, Copy, PartialEq)]
enum Bed {
    Far,
    Mid,
    Close,
    Drums,
    /// Massed boots (freesound loop) — plays with the drums on the march.
    March,
}

fn setup_audio(mut commands: Commands, assets: Res<AssetServer>) {
    let load_set = |names: &[&str]| -> Vec<Handle<AudioSource>> {
        names.iter().map(|n| assets.load(format!("{n}.mp3"))).collect()
    };

    commands.insert_resource(AudioBank {
        // Owner-preferred second batch (sfx_new/) over the first clangs.
        clang: load_set(&[
            "sfx_new/sword_clang_06",
            "sfx_new/sword_clang_07",
            "sfx_new/sword_clang_08",
            "sfx_new/sword_clang_09",
            "sfx_new/armor_clang_01",
            "sfx_new/armor_clang_02",
        ]),
        shield: load_set(&["sfx_shield_01", "sfx_shield_02", "sfx_shield_03"]),
        damage: load_set(&[
            "sfx_new/sfx_blunt_damage_01",
            "sfx_new/sfx_blunt_damage_02",
            "sfx_new/sfx_spear_damage_01",
            "sfx_new/sfx_sword_damage_01",
            "sfx_new/sfx_sword_damage_02",
        ]),
        death: load_set(&[
            "sfx_death_01",
            "sfx_death_02",
            "sfx_death_03",
            "sfx_death_04",
            "sfx_death_05",
        ]),
        vox_rout: load_set(&["vox_rout_01", "vox_rout_02", "vox_rout_03"]),
        vox_rally: load_set(&["vox_rally_01", "vox_rally_02"]),
        vox_cheer: load_set(&[
            "sfx_new/vox_rally_03_celebrate",
            "sfx_new/vox_rally_04_celebrate",
        ]),
        // Owner benched vox_warcry_02 — 01 only for now.
        vox_warcry: load_set(&["sfx_new/vox_warcry_01"]),
        horn_charge: load_set(&["sig_horn_charge", "sig_horn_charge_02"]),
        horn_rout: assets.load("sfx_new/sig_horn_rout.mp3"),
        ui_select: assets.load("sfx_new/ui_select1.mp3"),
        ui_order: assets.load("sfx_new/ui_order0.mp3"),
        ui_attack: assets.load("sfx_new/ui_attack.mp3"),
        sting_victory: assets.load("sting_victory.mp3"),
        sting_defeat: assets.load("sting_defeat.mp3"),
    });

    let bed = |name: &str| {
        (
            AudioPlayer::new(assets.load(format!("{name}.mp3"))),
            PlaybackSettings {
                volume: Volume::Linear(0.0),
                ..PlaybackSettings::LOOP
            },
        )
    };
    commands.spawn((bed("bed_battle_far"), Bed::Far));
    commands.spawn((bed("bed_battle_mid0"), Bed::Mid));
    commands.spawn((bed("bed_melee_close0"), Bed::Close));
    commands.spawn((bed("sig_drums_march"), Bed::Drums));
    commands.spawn((bed("sfx_new/bed_march_freesound_loop_14.5s"), Bed::March));
}

/// Crossfade the beds from battle state around the camera focus.
fn update_beds(
    groups: Res<Groups>,
    stats: Res<SimStats>,
    camera: Query<&RtsCamera>,
    time: Res<Time<Real>>,
    virt_time: Res<Time<Virtual>>,
    settings: Res<crate::settings::Settings>,
    mut sinks: Query<(&Bed, &mut AudioSink)>,
) {
    let Ok(cam) = camera.single() else { return };
    let paused = virt_time.is_paused();
    let focus = Vec2::new(cam.focus.x, cam.focus.z);

    let mut engaged_total = 0usize;
    let mut engaged_near = 0usize;
    let mut min_dist = f32::MAX;
    let mut marching_own = false;
    for g in &groups.list {
        if g.count == 0 {
            continue;
        }
        if g.engaged {
            engaged_total += 1;
            let d = g.centroid.distance(focus);
            min_dist = min_dist.min(d);
            if d < 300.0 {
                engaged_near += 1;
            }
        } else if g.team == 0 && g.order.is_some() && !g.state.is_broken() {
            marching_own = true;
        }
    }

    // Zooming out raises the "how far can you hear" floor a bit.
    let hear = 220.0 + cam.distance * 0.5;
    let prox = if min_dist == f32::MAX {
        0.0
    } else {
        (1.0 - (min_dist / hear)).clamp(0.0, 1.0)
    };
    let hits = (stats.events as f32 / 40.0).clamp(0.0, 1.0);

    // Zoomed out you should hear the DIN of battle, not individual steel:
    // the close-melee layer fades toward the mid/far beds with distance.
    let zoom_att = zoom_attenuation(cam.distance);

    let m = battle_vol(&settings);
    let target = |bed: &Bed| -> f32 {
        if paused {
            return 0.0;
        }
        match bed {
            Bed::Far => 0.22 * ((engaged_total as f32) / 8.0).clamp(0.0, 1.0) * m,
            Bed::Mid => 0.45 * ((engaged_near as f32) / 5.0).clamp(0.0, 1.0) * prox.sqrt() * m,
            Bed::Close => {
                0.60 * prox * prox * (0.25 + 0.75 * hits) * (0.35 + 0.65 * zoom_att) * m
            }
            Bed::Drums => {
                if marching_own {
                    0.30 * m
                } else {
                    0.0
                }
            }
            Bed::March => {
                if marching_own {
                    0.28 * m
                } else {
                    0.0
                }
            }
        }
    };

    let blend = (time.delta_secs() / BED_SMOOTH).min(1.0);
    for (bed, mut sink) in &mut sinks {
        let cur = sink.volume().to_linear();
        let v = cur + (target(bed) - cur) * blend;
        sink.set_volume(Volume::Linear(v));
    }
}

/// Poor-man's proximity: how "inside the battle" the camera is by zoom.
/// 1.0 at RTS close-up (<= 90 m), fading to a floor when surveying the
/// whole map. Real per-source spatial audio is a future item; this alone
/// stops individual clangs/screams from following you to max zoom.
fn zoom_attenuation(cam_distance: f32) -> f32 {
    (90.0 / cam_distance.max(90.0)).clamp(0.12, 1.0)
}

/// Spawn a fire-and-forget one-shot with volume/pitch jitter. `vol` is
/// the final linear volume; callers scale by `battle_vol`/`ui_vol`.
fn one_shot(commands: &mut Commands, h: Handle<AudioSource>, vol: f32, speed: f32) {
    commands.spawn((
        AudioPlayer::new(h),
        PlaybackSettings {
            volume: Volume::Linear(vol),
            speed,
            ..PlaybackSettings::DESPAWN
        },
    ));
}

fn pick(v: &[Handle<AudioSource>], seed: u32) -> Option<Handle<AudioSource>> {
    if v.is_empty() {
        return None;
    }
    Some(v[(hash01(seed) * v.len() as f32) as usize % v.len()].clone())
}

/// Clangs and death screams, budgeted from global sim stats scaled by
/// how close the camera is to the nearest engagement.
#[allow(clippy::too_many_arguments)] // bevy system params
fn combat_one_shots(
    mut commands: Commands,
    bank: Option<Res<AudioBank>>,
    stats: Res<SimStats>,
    combat: Res<CombatStats>,
    groups: Res<Groups>,
    camera: Query<&RtsCamera>,
    time: Res<Time>,
    settings: Res<crate::settings::Settings>,
    mut clang_acc: Local<f32>,
    mut prev_kills: Local<u64>,
    mut death_cooldown: Local<f32>,
    mut frame: Local<u32>,
) {
    *frame = frame.wrapping_add(1);
    let Some(bank) = bank else { return };
    let Ok(cam) = camera.single() else { return };
    let bv = battle_vol(&settings);
    let focus = Vec2::new(cam.focus.x, cam.focus.z);
    let min_dist = groups
        .list
        .iter()
        .filter(|g| g.engaged && g.count > 0)
        .map(|g| g.centroid.distance(focus))
        .fold(f32::MAX, f32::min);
    let hear = 200.0 + cam.distance * 0.4;
    let prox = if min_dist == f32::MAX {
        0.0
    } else {
        (1.0 - min_dist / hear).clamp(0.0, 1.0)
    };

    let zoom_att = zoom_attenuation(cam.distance);

    // Clang budget: fraction of actual hits, close-up only — both the
    // RATE and the volume fall away as the camera zooms out.
    *clang_acc +=
        stats.events as f32 * 30.0 * time.delta_secs() * 0.02 * prox * prox * zoom_att;
    let mut n = (*clang_acc).floor() as u32;
    *clang_acc -= n as f32;
    n = n.min(2); // hard cap per frame
    for k in 0..n {
        let seed = frame.wrapping_mul(31) ^ k;
        let r = hash01(seed ^ 0xA5);
        let set = if r < 0.2 {
            &bank.shield
        } else if r < 0.4 {
            &bank.damage
        } else {
            &bank.clang
        };
        if let Some(h) = pick(set, seed) {
            one_shot(
                &mut commands,
                h,
                (0.16 + 0.10 * hash01(seed ^ 0x11)) * zoom_att * bv,
                0.92 + 0.16 * hash01(seed ^ 0x22),
            );
        }
    }

    // Death screams: on kill deltas, rate-limited, and strictly a
    // close-up sound — a scream you can pick out from a hilltop is wrong.
    *death_cooldown -= time.delta_secs();
    let kills: u64 = combat.kills[0] + combat.kills[1];
    if kills > *prev_kills && *death_cooldown <= 0.0 && prox > 0.25 && zoom_att > 0.35 {
        let seed = frame.wrapping_mul(97);
        if let Some(h) = pick(&bank.death, seed) {
            one_shot(
                &mut commands,
                h,
                (0.18 + 0.12 * hash01(seed ^ 0x33)) * prox * zoom_att * bv,
                0.9 + 0.2 * hash01(seed ^ 0x44),
            );
        }
        *death_cooldown = 0.5 + 0.5 * hash01(seed ^ 0x55);
    }
    *prev_kills = kills;
}

/// Edge-detection state for `event_cues`, bundled into one Local (the
/// bare-Local version blew past Bevy's 16 system-param limit).
#[derive(Default)]
struct CueState {
    prev_sel: usize,
    /// Last tick's order per regiment (edge = order CHANGED, so
    /// re-orders click and horn too).
    prev_order: Vec<Option<Order>>,
    prev_state: Vec<u8>,
    /// Last tick's per-regiment "attack target within charge range".
    prev_cry: Vec<bool>,
    roar_budget: u32,
    prev_outcome: bool,
    horn_gate: f32,
    vox_gate: f32,
    frame: u32,
}

/// Discrete cues: selection click, charge horn on new orders, war cry
/// acknowledgment on attack orders, sustained war cries while regiments
/// charge home, rout/rally vox + horn, victory/defeat stings.
#[allow(clippy::too_many_arguments)] // bevy system params
fn event_cues(
    mut commands: Commands,
    bank: Option<Res<AudioBank>>,
    groups: Res<Groups>,
    selection: Res<Selection>,
    outcome: Res<crate::ai::BattleOutcome>,
    camera: Query<&RtsCamera>,
    time: Res<Time>,
    settings: Res<crate::settings::Settings>,
    mut st: Local<CueState>,
) {
    let Some(bank) = bank else { return };
    let bv = battle_vol(&settings);
    let uv = ui_vol(&settings);
    st.frame = st.frame.wrapping_add(1);
    st.horn_gate -= time.delta_secs();
    st.vox_gate -= time.delta_secs();
    let n = groups.list.len();
    st.prev_order.resize(n, None);
    st.prev_state.resize(n, 0);
    st.prev_cry.resize(n, false);

    // Selection click on a changed non-empty selection.
    if selection.count_units > 0 && selection.count_units != st.prev_sel {
        one_shot(&mut commands, bank.ui_select.clone(), 0.5 * uv, 1.0);
    }
    st.prev_sel = selection.count_units;

    let mut new_own_move = false;
    let mut new_own_attack = false;
    let mut new_break_own = false;
    let mut new_break_enemy = false;
    let mut new_break_any = false;
    let mut new_rally = false;
    // War cry rule (ONE rule): a regiment cries when "attack target
    // within CHARGE_RANGE" newly becomes true — ordered at close range,
    // closed to range on an approach, or retargeted to another nearby
    // enemy while fighting (M2TW). Far attack orders get the horn only.
    // The cry edge plays immediately; while an unengaged run-in lasts,
    // the roar re-fires on the vox gate up to WARCRY_CLIPS total (one
    // clip died before impact on long run-ins; unlimited rolling looped
    // forever on pursuits).
    const WARCRY_CLIPS: u32 = 3;
    let mut charge_dist = f32::MAX;
    let mut cry_dist = f32::MAX;
    let focus = camera
        .single()
        .map(|c| Vec2::new(c.focus.x, c.focus.z))
        .unwrap_or_default();
    let hear = camera.single().map(|c| 220.0 + c.distance * 0.5).unwrap_or(220.0);
    for (g, gd) in groups.list.iter().enumerate() {
        let prev_atk = match st.prev_order[g] {
            Some(Order::Attack(t)) => t,
            _ => u32::MAX,
        };
        if gd.team == 0
            && gd.order != st.prev_order[g]
            && gd.count > 0
            && !gd.state.is_broken()
            // At-ease self-engagement is not a player click: no UI
            // feedback (the war cry below still fires — that one is
            // battlefield sound, and a regiment roaring as it takes
            // matters into its own hands is correct).
            && !gd.auto_order
        {
            match gd.order {
                Some(Order::Attack(_)) => new_own_attack = true,
                Some(Order::Move(_)) => new_own_move = true,
                None => {}
            }
        }
        st.prev_order[g] = gd.order;

        let state = match gd.state {
            RegState::Steady => 0u8,
            RegState::Routing { .. } => 1,
            RegState::Shattered => 2,
        };
        if state >= 1 && st.prev_state[g] == 0 && gd.count > 0 {
            new_break_any = true;
            if gd.team == 0 {
                new_break_own = true;
            } else {
                new_break_enemy = true;
            }
        }
        if state == 0 && st.prev_state[g] == 1 {
            new_rally = true;
        }
        st.prev_state[g] = state;

        if gd.charging && gd.count > 0 {
            charge_dist = charge_dist.min(gd.centroid.distance(focus));
        }

        // Cry-edge detection: attack target alive and within charge
        // range; a target CHANGE while in range also counts (retarget
        // mid-melee, M2TW).
        let (atk, in_range) = match gd.order {
            Some(Order::Attack(t)) if groups.list[t as usize].count > 0 => {
                let d = gd.centroid.distance(groups.list[t as usize].centroid);
                (t, d < crate::frontline::CHARGE_RANGE)
            }
            _ => (u32::MAX, false),
        };
        let cry = in_range && gd.count > 0 && !gd.state.is_broken();
        if cry && (!st.prev_cry[g] || atk != prev_atk) {
            cry_dist = cry_dist.min(gd.centroid.distance(focus));
        }
        st.prev_cry[g] = cry;
    }

    let seed = st.frame.wrapping_mul(211);
    // Edge cry: plays NOW (order/charge feedback beats the vox gate) and
    // opens a fresh roar budget for the run-in.
    if cry_dist < f32::MAX {
        let prox = (1.0 - cry_dist / hear).clamp(0.0, 1.0);
        if prox > 0.05 {
            if let Some(h) = pick(&bank.vox_warcry, seed ^ 0xAB) {
                let vol = 0.55 * (0.25 + 0.75 * prox) * bv;
                one_shot(&mut commands, h, vol, 0.94 + 0.12 * hash01(seed ^ 0xAC));
                info!("war cry (edge, vol {vol:.2}, prox {prox:.2})");
            }
            st.roar_budget = WARCRY_CLIPS - 1;
            st.vox_gate = 2.5;
        }
    }

    // Order-click feedback: immediate and ungated, like the selection
    // click — this is UI, not battlefield sound.
    if new_own_attack {
        one_shot(&mut commands, bank.ui_attack.clone(), 0.45 * uv, 1.0);
    }
    if new_own_move {
        one_shot(&mut commands, bank.ui_order.clone(), 0.35 * uv, 1.0);
    }
    if (new_own_attack || new_own_move) && st.horn_gate <= 0.0 {
        if let Some(h) = pick(&bank.horn_charge, seed) {
            one_shot(&mut commands, h, 0.55 * bv, 1.0);
        }
        st.horn_gate = 3.0;
    }
    if st.vox_gate <= 0.0 {
        // One vox per gate window, most dramatic first.
        if new_break_any {
            if let Some(h) = pick(&bank.vox_rout, seed ^ 0x66) {
                one_shot(&mut commands, h, 0.55 * bv, 1.0);
            }
            if new_break_own {
                one_shot(&mut commands, bank.horn_rout.clone(), 0.5 * bv, 1.0);
            }
            // Victors roar over the enemy's panic (TW moment).
            if new_break_enemy
                && let Some(h) = pick(&bank.vox_cheer, seed ^ 0xBB)
            {
                one_shot(&mut commands, h, 0.5 * bv, 1.0);
            }
            st.vox_gate = 1.5;
        } else if new_rally {
            if let Some(h) = pick(&bank.vox_rally, seed ^ 0x77) {
                one_shot(&mut commands, h, 0.5 * bv, 1.0);
            }
            st.vox_gate = 1.5;
        } else {
            // Rolling war cry: one clip per gate window while any charge
            // runs within earshot AND the onset budget lasts, volume by
            // proximity. The ~3 s clips on a 2.5 s gate overlap into a
            // continuous roar from the charge onset to contact.
            let prox = (1.0 - charge_dist / hear).clamp(0.0, 1.0);
            if prox > 0.05 && st.roar_budget > 0 {
                if let Some(h) = pick(&bank.vox_warcry, seed ^ 0x88) {
                    let vol = 0.55 * (0.25 + 0.75 * prox) * bv;
                    one_shot(&mut commands, h, vol, 0.94 + 0.12 * hash01(seed ^ 0x99));
                    st.roar_budget -= 1;
                    info!(
                        "war cry (vol {vol:.2}, prox {prox:.2}, {} clips left)",
                        st.roar_budget
                    );
                }
                st.vox_gate = 2.5;
            }
        }
    }

    // Outcome sting, once.
    if !st.prev_outcome && outcome.0.is_some() {
        let h = match outcome.0 {
            Some(0) => bank.sting_victory.clone(),
            _ => bank.sting_defeat.clone(),
        };
        one_shot(&mut commands, h, 0.8 * bv, 1.0);
        st.prev_outcome = true;
    }
}
