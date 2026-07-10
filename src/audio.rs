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
use crate::orders::{Groups, RegState, Selection};
use crate::units::hash01;

/// Master volume (FL_VOLUME overrides, linear).
fn master() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::util::env_or("FL_VOLUME", 1.0))
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
    /// Empty until vox_warcry assets exist (see audio-plan retry notes).
    vox_warcry: Vec<Handle<AudioSource>>,
    horn_charge: Vec<Handle<AudioSource>>,
    horn_rout: Handle<AudioSource>,
    ui_select: Handle<AudioSource>,
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
        vox_rally: load_set(&[
            "vox_rally_01",
            "vox_rally_02",
            "sfx_new/vox_rally_03",
            "sfx_new/vox_rally_04_celebrate",
        ]),
        vox_warcry: load_set(&["sfx_new/vox_warcry_01", "sfx_new/vox_warcry_02"]),
        horn_charge: load_set(&["sig_horn_charge", "sig_horn_charge_02"]),
        horn_rout: assets.load("sfx_new/sig_horn_rout.mp3"),
        ui_select: assets.load("sfx_new/ui_select.mp3"),
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
}

/// Crossfade the beds from battle state around the camera focus.
fn update_beds(
    groups: Res<Groups>,
    stats: Res<SimStats>,
    camera: Query<&RtsCamera>,
    time: Res<Time>,
    mut sinks: Query<(&Bed, &mut AudioSink)>,
) {
    let Ok(cam) = camera.single() else { return };
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

    let m = master();
    let target = |bed: &Bed| -> f32 {
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

/// Spawn a fire-and-forget one-shot with volume/pitch jitter.
fn one_shot(commands: &mut Commands, h: Handle<AudioSource>, vol: f32, speed: f32) {
    commands.spawn((
        AudioPlayer::new(h),
        PlaybackSettings {
            volume: Volume::Linear(vol * master()),
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
    bank: Res<AudioBank>,
    stats: Res<SimStats>,
    combat: Res<CombatStats>,
    groups: Res<Groups>,
    camera: Query<&RtsCamera>,
    time: Res<Time>,
    mut clang_acc: Local<f32>,
    mut prev_kills: Local<u64>,
    mut death_cooldown: Local<f32>,
    mut frame: Local<u32>,
) {
    *frame = frame.wrapping_add(1);
    let Ok(cam) = camera.single() else { return };
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
                (0.16 + 0.10 * hash01(seed ^ 0x11)) * zoom_att,
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
                (0.18 + 0.12 * hash01(seed ^ 0x33)) * prox * zoom_att,
                0.9 + 0.2 * hash01(seed ^ 0x44),
            );
        }
        *death_cooldown = 0.5 + 0.5 * hash01(seed ^ 0x55);
    }
    *prev_kills = kills;
}

/// Discrete cues: selection click, charge horn on new orders, war cries on
/// first contact, rout/rally vox + horn, victory/defeat stings.
#[allow(clippy::too_many_arguments)] // bevy system params
fn event_cues(
    mut commands: Commands,
    bank: Res<AudioBank>,
    groups: Res<Groups>,
    selection: Res<Selection>,
    outcome: Res<crate::ai::BattleOutcome>,
    time: Res<Time>,
    mut prev_sel: Local<usize>,
    mut prev_orders: Local<Vec<bool>>,
    mut prev_state: Local<Vec<u8>>,
    mut prev_engaged: Local<Vec<bool>>,
    mut prev_outcome: Local<bool>,
    mut horn_gate: Local<f32>,
    mut vox_gate: Local<f32>,
    mut frame: Local<u32>,
) {
    *frame = frame.wrapping_add(1);
    *horn_gate -= time.delta_secs();
    *vox_gate -= time.delta_secs();
    let n = groups.list.len();
    prev_orders.resize(n, false);
    prev_state.resize(n, 0);
    prev_engaged.resize(n, false);

    // Selection click on a changed non-empty selection.
    if selection.count_units > 0 && selection.count_units != *prev_sel {
        one_shot(&mut commands, bank.ui_select.clone(), 0.5, 1.0);
    }
    *prev_sel = selection.count_units;

    let mut new_own_order = false;
    let mut new_break_own = false;
    let mut new_break_any = false;
    let mut new_rally = false;
    let mut new_contact = false;
    for (g, gd) in groups.list.iter().enumerate() {
        let has_order = gd.order.is_some();
        if gd.team == 0 && has_order && !prev_orders[g] && !gd.state.is_broken() {
            new_own_order = true;
        }
        prev_orders[g] = has_order;

        let state = match gd.state {
            RegState::Steady => 0u8,
            RegState::Routing { .. } => 1,
            RegState::Shattered => 2,
        };
        if state >= 1 && prev_state[g] == 0 && gd.count > 0 {
            new_break_any = true;
            if gd.team == 0 {
                new_break_own = true;
            }
        }
        if state == 0 && prev_state[g] == 1 {
            new_rally = true;
        }
        prev_state[g] = state;

        if gd.engaged && !prev_engaged[g] && gd.count > 0 {
            new_contact = true;
        }
        prev_engaged[g] = gd.engaged;
    }

    let seed = frame.wrapping_mul(211);
    if new_own_order && *horn_gate <= 0.0 {
        if let Some(h) = pick(&bank.horn_charge, seed) {
            one_shot(&mut commands, h, 0.55, 1.0);
        }
        *horn_gate = 3.0;
    }
    if *vox_gate <= 0.0 {
        // One vox per gate window, most dramatic first.
        if new_break_any {
            if let Some(h) = pick(&bank.vox_rout, seed ^ 0x66) {
                one_shot(&mut commands, h, 0.55, 1.0);
            }
            if new_break_own {
                one_shot(&mut commands, bank.horn_rout.clone(), 0.5, 1.0);
            }
            *vox_gate = 1.5;
        } else if new_rally {
            if let Some(h) = pick(&bank.vox_rally, seed ^ 0x77) {
                one_shot(&mut commands, h, 0.5, 1.0);
            }
            *vox_gate = 1.5;
        } else if new_contact {
            if let Some(h) = pick(&bank.vox_warcry, seed ^ 0x88) {
                one_shot(&mut commands, h, 0.55, 1.0);
            }
            *vox_gate = 1.2;
        }
    }

    // Outcome sting, once.
    if !*prev_outcome && outcome.0.is_some() {
        let h = match outcome.0 {
            Some(0) => bank.sting_victory.clone(),
            _ => bank.sting_defeat.clone(),
        };
        one_shot(&mut commands, h, 0.8, 1.0);
        *prev_outcome = true;
    }
}
