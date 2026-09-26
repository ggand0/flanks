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
use crate::game_state::GameState;
use crate::sim::SimStats;
use crate::orders::{Groups, RegState};
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

/// A player UI action that wants click feedback, written by the input
/// systems (lasso release, card clicks, order clicks). One clip plays
/// per message: repeating the same selection or re-ordering the same
/// attack clicks every time, which the old state-diff detection never
/// could. Deploy is a placement during the deployment phase — click
/// only, no charge horn.
#[derive(Message)]
pub enum UiCue {
    Select,
    Move,
    Attack,
    Deploy,
}

/// Bed smoothing time constant (seconds to ~2/3 of the way to target).
const BED_SMOOTH: f32 = 0.35;

/// Hard ceiling on simultaneously playing one-shots. Every one-shot is
/// a live decoder on rodio's single mixer thread; the underruns
/// ("Buffer underrun/overrun" errors, audible dropouts) appeared north
/// of ~100 concurrent. The first guess of 40 audibly starved the loose
/// and impact layers themselves (a volley's looses alone stack ~50
/// clips), so the ceiling sits between the two: high enough for the
/// full volley chorus, under the measured breaking point. The budgets
/// shape the sound; this guard protects the mixer.
const MAX_LIVE_ONE_SHOTS: usize = 64;

/// Massed-volley sheet cues: BENCHED — the pre-rendered 2.5 s sheets
/// don't track the actual flight (they fire on rate edges, so onset,
/// length, and position all drift from what's on screen). The likely
/// replacement is many INDIVIDUAL whoosh/swish one-shots budgeted from
/// the live arrow cloud so the volley layers itself organically, like
/// every other sound here. Set to true to hear the sheets again.
const VOLLEY_SHEETS_ON: bool = false;

pub struct BattleAudioPlugin;

impl Plugin for BattleAudioPlugin {
    fn build(&self, app: &mut App) {
        // Battle-only: outside the battle the sim stats these systems
        // read are stale, and quitting to the menu must not leave the
        // beds ringing (or stale stats spawning clangs) behind it.
        app.add_message::<UiCue>()
            .add_systems(Startup, setup_audio)
            .add_systems(
                Update,
                (
                    update_beds,
                    combat_one_shots,
                    melee_vox,
                    archer_one_shots,
                    arrow_fly_loops,
                    event_cues,
                    charge_vox,
                    celebrate_vox,
                    rout_vox,
                )
                    .chain()
                    .run_if(in_state(GameState::Battle)),
            )
            // Beds cut on leaving battle; one-shots survive into the
            // results screen (the victory/defeat sting must finish)
            // and are only culled when the menu comes up.
            .add_systems(OnExit(GameState::Battle), (silence_beds, stop_fly_loops))
            .add_systems(OnEnter(GameState::Menu), stop_one_shots);
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
    /// Rout soundscape (sfx_rout/, owner's movie-research direction +
    /// M2TW Individual_Retreat): commanders shouting Retreat/Withdraw
    /// over a mostly silent fleeing mass, panic screams only in the
    /// first moments of a break, massed running feet underneath.
    rout_shout: Vec<Handle<AudioSource>>,
    rout_panic: Vec<Handle<AudioSource>>,
    feet_wash: Vec<Handle<AudioSource>>,
    /// Rolling group cheers while a regiment CELEBRATES (sfx_celebrate/,
    /// M2TW unit_celebrate state bank), size-banded like the charge
    /// sheets, plus single-man victory whoops over them.
    celebrate_small: Vec<Handle<AudioSource>>,
    celebrate_large: Vec<Handle<AudioSource>>,
    celebrate_whoop: Vec<Handle<AudioSource>>,
    /// Melee human layer (sfx_melee/, M2TW soldier_voice vocals):
    /// attacker effort grunts and screams, victim hit grunts, and
    /// sustained battle screams over a locked fight.
    melee_attack_grunt: Vec<Handle<AudioSource>>,
    melee_attack_scream: Vec<Handle<AudioSource>>,
    melee_hit_grunt: Vec<Handle<AudioSource>>,
    melee_battle_scream: Vec<Handle<AudioSource>>,
    /// Single-man charge yells (sfx_charge/, M2TW Individual_Charge:
    /// one in five charging soldiers screams his own clip).
    yell_charge: Vec<Handle<AudioSource>>,
    /// Group charge sheets (M2TW unit_charge bank), size-banded:
    /// medium under 300 men, large above.
    group_charge_medium: Vec<Handle<AudioSource>>,
    group_charge_large: Vec<Handle<AudioSource>>,
    horn_charge: Vec<Handle<AudioSource>>,
    horn_rout: Handle<AudioSource>,
    ui_select: Handle<AudioSource>,
    /// Click feedback for a move order (mixed quieter than select).
    ui_order: Handle<AudioSource>,
    /// Click feedback for an attack order on an enemy regiment.
    ui_attack: Handle<AudioSource>,
    sting_victory: Handle<AudioSource>,
    sting_defeat: Handle<AudioSource>,
    // --- archer set (sfx_bow/, prompts in tmp/archer-sfx-prompts.md) ---
    /// Single string-snap releases, budgeted from ArrowStats.loosed.
    bow_loose: Vec<Handle<AudioSource>>,
    /// Body hits; the wood knocks mix in at low odds (shielded men).
    arrow_flesh: Vec<Handle<AudioSource>>,
    arrow_wood: Vec<Handle<AudioSource>>,
    /// Shafts thudding into dirt (the miss patter).
    arrow_ground: Vec<Handle<AudioSource>>,
    /// Near-miss whistle for close-up incoming fire.
    arrow_flyby: Vec<Handle<AudioSource>>,
    /// Seamless in-flight air loop, attached per arrow (the M2TW
    /// ARROW_FLY model). Missing files stay silent until generated.
    arrow_fly_loop: Vec<Handle<AudioSource>>,
    /// Massed-volley sheets: loosed away / falling on the camera.
    volley_away: Vec<Handle<AudioSource>>,
    volley_incoming: Vec<Handle<AudioSource>>,
}

/// Looping bed entities, indexed by `Bed`.
#[derive(Component, Clone, Copy, PartialEq)]
enum Bed {
    Far,
    Mid,
    Close,
    Drums,
    /// Massed boots (Pixabay loop) — plays with the drums on the march.
    March,
}

fn setup_audio(mut commands: Commands, assets: Res<AssetServer>) {
    let load_set = |names: &[&str]| -> Vec<Handle<AudioSource>> {
        names.iter().map(|n| assets.load(format!("{n}.mp3"))).collect()
    };

    commands.insert_resource(AudioBank {
        // The second batch (sfx_new/) won out over the first clangs.
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
        // vox_rally_01/02 are benched (owner: unusable, use nowhere).
        vox_rout: load_set(&[
            "vox_rout_01",
            "vox_rout_02",
            "vox_rout_03",
            "sfx_rout/vox_rout_04",
            "sfx_rout/vox_rout_05",
        ]),
        rout_shout: load_set(&[
            "sfx_rout/fallback0",
            "sfx_rout/fallback1",
            "sfx_rout/fallback2",
            "sfx_rout/retreat0",
            "sfx_rout/retreat1",
            "sfx_rout/run_away!0",
            "sfx_rout/run_away!1_funny",
            "sfx_rout/withdraw0",
            "sfx_rout/withdraw1",
            "sfx_rout/withdraw2",
        ]),
        rout_panic: load_set(&[
            "sfx_rout/vox_panic_01",
            "sfx_rout/vox_panic_02",
            "sfx_rout/vox_panic_03",
        ]),
        // Massed washes layered from the single-man source loops by
        // tmp/build-feet-wash.sh (ElevenLabs would only produce one
        // or two runners per take).
        feet_wash: load_set(&[
            "sfx_rout/feet_run_wash_mass_01",
            "sfx_rout/feet_run_wash_mass_02",
        ]),
        // The sfx_new/vox_rally_03/04_celebrate one-shots are benched
        // (owner: superseded); the celebrate state cheers from these.
        celebrate_small: load_set(&[
            "sfx_celebrate/group_cheer_small_01_mocking",
            "sfx_celebrate/group_cheer_small_02_mocking",
            "sfx_celebrate/group_cheer_small_04_laughing",
            "sfx_celebrate/group_cheer_small_05_laughing",
            "sfx_celebrate/group_cheer_small_06_laughing",
            "sfx_celebrate/group_cheer_small_07_joyous_shouts",
            "sfx_celebrate/group_cheer_small_08_joyous_shouts",
        ]),
        celebrate_large: load_set(&[
            "sfx_celebrate/group_cheer_large_01",
            "sfx_celebrate/group_cheer_large_02_short",
            "sfx_celebrate/group_cheer_large_03_ok",
            "sfx_celebrate/group_cheer_large_04",
            "sfx_celebrate/group_cheer_large_05_ok",
        ]),
        // vox_whoop_04 benched by its own filename (skipfornow).
        celebrate_whoop: load_set(&[
            "sfx_celebrate/vox_whoop_01",
            "sfx_celebrate/vox_whoop_02_joyous_shout",
            "sfx_celebrate/vox_whoop_03_laugh",
            "sfx_celebrate/vox_whoop_05_knight_laugh",
            "sfx_celebrate/vox_whoop_06_cheer_yeah",
            "sfx_celebrate/vox_whoop_07_roar_yeah",
            "sfx_celebrate/vox_whoop_08_knight_shout_yes",
            "sfx_celebrate/vox_whoop_09_knight_shout_yeaa",
        ]),
        melee_attack_grunt: load_set(&[
            "sfx_melee/attack_grunts/attack_grunt0",
            "sfx_melee/attack_grunts/attack_grunt_knight0",
            "sfx_melee/attack_grunts/attack_grunt_knight1",
            "sfx_melee/attack_grunts/attack_grunt_knight2",
            "sfx_melee/attack_grunts/attack_grunt_knight3",
            "sfx_melee/attack_grunts/attack_grunt_knight4",
            "sfx_melee/attack_grunts/attack_grunt_knight5",
            "sfx_melee/attack_grunts/attack_grunt_knight6",
            "sfx_melee/attack_grunts/attack_grunt_knight7",
            "sfx_melee/attack_grunts/attack_grunt_knight8",
            "sfx_melee/attack_grunts/attack_grunt_knight9",
            "sfx_melee/attack_grunts/attack_grunt_knight10",
            "sfx_melee/attack_grunts/attack_grunt_knight11",
            "sfx_melee/attack_grunts/attack_grunt_old_knight0",
            "sfx_melee/attack_grunts/attack_grunt_old_knight1",
            "sfx_melee/attack_grunts/attack_grunt_old_knight2",
            "sfx_melee/attack_grunts/attack_grunt_young_knight0",
            "sfx_melee/attack_grunts/attack_grunt_young_knight1",
            "sfx_melee/attack_grunts/attack_grunt_young_knight2",
            "sfx_melee/attack_grunts/attack_grunt_young_knight3",
            "sfx_melee/attack_grunts/attack_grunt_young_knight4",
        ]),
        melee_attack_scream: load_set(&[
            "sfx_melee/attack_screams/attack_scream0_bitfunny",
            "sfx_melee/attack_screams/attack_scream1_young",
            "sfx_melee/attack_screams/attack_scream2_good",
            "sfx_melee/attack_screams/attack_scream3",
            "sfx_melee/attack_screams/attack_scream4",
            "sfx_melee/attack_screams/attack_scream5",
            "sfx_melee/attack_screams/attack_scream6",
            "sfx_melee/attack_screams/attack_scream7",
            "sfx_melee/attack_screams/attack_scream_young0",
            "sfx_melee/attack_screams/attack_scream_young1",
            "sfx_melee/attack_screams/attack_scream_young2",
            "sfx_melee/attack_screams/attack_scream_young3",
        ]),
        melee_hit_grunt: load_set(&[
            "sfx_melee/hit_grunts/choked_groan(big_damage)1",
            "sfx_melee/hit_grunts/damage_gasp0",
            "sfx_melee/hit_grunts/damage_gasp1",
            "sfx_melee/hit_grunts/damage_gasp2",
            "sfx_melee/hit_grunts/damage_grunt0",
            "sfx_melee/hit_grunts/damage_grunt2",
            "sfx_melee/hit_grunts/damage_grunt3",
            "sfx_melee/hit_grunts/damage_grunt4",
            "sfx_melee/hit_grunts/damage_grunt6",
            "sfx_melee/hit_grunts/damage_grunt7",
            "sfx_melee/hit_grunts/damage_grunt8",
            "sfx_melee/hit_grunts/damage_grunt9",
            "sfx_melee/hit_grunts/damage_grunt10",
            "sfx_melee/hit_grunts/damage_grunt11",
            "sfx_melee/hit_grunts/damage_grunt12",
            "sfx_melee/hit_grunts/stabbed0",
        ]),
        melee_battle_scream: load_set(&[
            "sfx_melee/battle_screams/battle_scream0",
            "sfx_melee/battle_screams/battle_scream1",
            "sfx_melee/battle_screams/battle_scream3",
            "sfx_melee/battle_screams/battle_scream4",
            "sfx_melee/battle_screams/battle_scream6",
            "sfx_melee/battle_screams/battle_scream7",
            "sfx_melee/battle_screams/battle_scream8",
            "sfx_melee/battle_screams/battle_scream9",
        ]),
        // The old vox_warcry crowd clips are benched: the charge is
        // layered from these pools now (devlog 0069).
        yell_charge: load_set(&[
            "sfx_charge/vox_yell_01",
            "sfx_charge/vox_yell_01b",
            "sfx_charge/vox_yell_02",
            "sfx_charge/vox_yell_02b",
            "sfx_charge/vox_yell_03a",
            "sfx_charge/vox_yell_03b",
            "sfx_charge/vox_yell_03c",
            "sfx_charge/vox_yell_03d",
            "sfx_charge/vox_yell_04",
            "sfx_charge/vox_yell_04b",
            "sfx_charge/vox_yell_05a",
            "sfx_charge/vox_yell_05b",
            "sfx_charge/vox_yell_06a",
            "sfx_charge/vox_yell_06b",
            "sfx_charge/vox_yell_07",
            "sfx_charge/vox_yell_08",
            "sfx_charge/vox_yell_09a",
            "sfx_charge/vox_yell_09b",
            "sfx_charge/vox_yell_09c",
            "sfx_charge/vox_yell_09d",
            "sfx_charge/vox_yell_09_young_a",
            "sfx_charge/vox_yell_09_young_b",
            "sfx_charge/vox_yell_09_young_c",
            "sfx_charge/vox_yell_09_young_d",
            "sfx_charge/vox_yell_10a",
            "sfx_charge/vox_yell_10b",
            "sfx_charge/vox_yell_10c",
            "sfx_charge/vox_yell_10_young_a",
            "sfx_charge/vox_yell_10_young_b",
        ]),
        group_charge_medium: load_set(&["sfx_charge/group_charge_medium"]),
        group_charge_large: load_set(&[
            "sfx_charge/group_charge_large_01",
            "sfx_charge/group_charge_large_02",
        ]),
        horn_charge: load_set(&["sig_horn_charge", "sig_horn_charge_02"]),
        horn_rout: assets.load("sfx_new/sig_horn_rout.mp3"),
        ui_select: assets.load("sfx_new/ui_select1.mp3"),
        ui_order: assets.load("sfx_new/ui_order0.mp3"),
        ui_attack: assets.load("sfx_new/ui_attack.mp3"),
        sting_victory: assets.load("sting_victory.mp3"),
        sting_defeat: assets.load("sting_defeat.mp3"),
        bow_loose: load_set(&[
            "sfx_bow/sfx_bow_loose_01",
            "sfx_bow/sfx_bow_loose_02",
            "sfx_bow/sfx_bow_loose_03",
            "sfx_bow/sfx_bow_loose_04",
        ]),
        arrow_flesh: load_set(&[
            "sfx_bow/sfx_arrow_flesh_01",
            "sfx_bow/sfx_arrow_flesh_02",
            "sfx_bow/sfx_arrow_flesh_03",
        ]),
        arrow_wood: load_set(&["sfx_bow/sfx_arrow_wood_01", "sfx_bow/sfx_arrow_wood_02"]),
        arrow_ground: load_set(&[
            "sfx_bow/sfx_arrow_ground_01",
            "sfx_bow/sfx_arrow_ground_02",
            "sfx_bow/sfx_arrow_ground_03",
        ]),
        arrow_flyby: load_set(&["sfx_bow/sfx_arrow_flyby_01", "sfx_bow/sfx_arrow_flyby_02"]),
        arrow_fly_loop: load_set(&[
            "sfx_bow/sfx_arrow_fly_loop_01",
            "sfx_bow/sfx_arrow_fly_loop_02",
        ]),
        volley_away: vec![
            assets.load("sfx_bow/sfx_volley_away_01.wav"),
            assets.load("sfx_bow/sfx_volley_away_02.mp3"),
            assets.load("sfx_bow/sfx_volley_away_03.mp3"),
        ],
        volley_incoming: vec![
            assets.load("sfx_bow/volley_02_flight_incoming.wav"),
            assets.load("sfx_bow/arrow_incoming_02_several.mp3"),
            assets.load("sfx_bow/arrow_incoming_03.mp3"),
        ],
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
    commands.spawn((bed("sfx_new/bed_march_loop_14.5s"), Bed::March));
}

fn silence_beds(mut sinks: Query<&mut AudioSink, With<Bed>>) {
    for mut sink in &mut sinks {
        sink.set_volume(Volume::Linear(0.0));
    }
}

/// Looping fly sounds must not ring into the results screen (one-shots
/// may — the sting must finish; a loop never finishes).
fn stop_fly_loops(mut commands: Commands, loops: Query<Entity, With<ArrowFlyLoop>>) {
    for e in &loops {
        commands.entity(e).despawn();
    }
}

/// Despawn every in-flight one-shot (war cries, horns, sting tails).
fn stop_one_shots(
    mut commands: Commands,
    playing: Query<Entity, (With<AudioPlayer>, Without<Bed>)>,
) {
    for e in &playing {
        commands.entity(e).despawn();
    }
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

/// Mean XZ of the live arrow pool — where "the volley" is for
/// proximity purposes. None when nothing is in the air.
fn arrow_cloud(arrows: &crate::arrows::Arrows) -> Option<Vec2> {
    if arrows.len() == 0 {
        return None;
    }
    let mut sum = Vec2::ZERO;
    for p in &arrows.pos {
        sum += p.xz();
    }
    Some(sum / arrows.len() as f32)
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
    (astats, arrows): (
        Res<crate::arrows::ArrowStats>,
        Res<crate::arrows::Arrows>,
    ),
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
    let hear = 200.0 + cam.distance * 0.4;
    // M2TW fires EVERY hit's weapon sound positionally (weapon_hit
    // bank: probability 1, no rate knob anywhere in the format) and
    // thins by distance ATTENUATION and voice priority, never by
    // rate. Emulated here: hits are attributed to engaged regiments
    // by strength and weighted by earshot, giving the audible share
    // of this tick's hits; the rate cap below stands in for the
    // voice limit. Camera distance shapes the VOLUME only.
    let (mut min_dist, mut engaged_total, mut engaged_audible) = (f32::MAX, 0.0f32, 0.0f32);
    for g in groups.list.iter().filter(|g| g.engaged && g.count > 0) {
        let d = g.centroid.distance(focus);
        min_dist = min_dist.min(d);
        engaged_total += g.count as f32;
        engaged_audible += g.count as f32 * (1.0 - d / hear).clamp(0.0, 1.0);
    }
    let audible = if engaged_total > 0.0 {
        engaged_audible / engaged_total
    } else {
        0.0
    };
    let prox = if min_dist == f32::MAX {
        0.0
    } else {
        (1.0 - min_dist / hear).clamp(0.0, 1.0)
    };

    let zoom_att = zoom_attenuation(cam.distance);

    // Steel: every audible hit wants its clang, up to the rate cap
    // (~24 live 0.8 s clips at the cap — the voice-limit analog).
    const STEEL_RATE_CAP: f32 = 30.0;
    let near_hit_rate = stats.events as f32 * 30.0 * audible;
    *clang_acc += near_hit_rate.min(STEEL_RATE_CAP) * time.delta_secs();
    *clang_acc = (*clang_acc).min(4.0);
    let mut n = (*clang_acc).floor() as u32;
    *clang_acc -= n as f32;
    n = n.min(3); // hard cap per frame
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
                (0.16 + 0.10 * hash01(seed ^ 0x11))
                    * (0.3 + 0.7 * prox)
                    * zoom_att
                    * bv,
                0.92 + 0.16 * hash01(seed ^ 0x22),
            );
        }
    }

    // Death screams: on kill deltas, rate-limited, and strictly a
    // close-up sound — a scream you can pick out from a hilltop is wrong.
    // Men dying under arrows die far from any ENGAGED regiment, so the
    // melee-proximity gate alone kept volley kills silent: when shafts
    // are landing on bodies this tick, the falling cloud's proximity
    // counts too.
    let mut prox = prox;
    if astats.hits > 0
        && let Some(cloud) = arrow_cloud(&arrows)
    {
        prox = prox.max((1.0 - cloud.distance(focus) / hear).clamp(0.0, 1.0));
    }
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

/// A looping fly sound riding ONE arrow (the M2TW ARROW_FLY model:
/// `looped, probability .3, fadein .2, fadeout .2` — a fraction of
/// arrows carry a positional air loop that dies WITH the arrow, so the
/// whoosh audibly resolves into its own impact). We track the stable
/// arrow id (pool indices reshuffle every tick) and steer the sink's
/// volume by live distance each frame.
#[derive(Component)]
struct ArrowFlyLoop {
    arrow_id: u32,
    /// Seconds of fade-out left once the arrow is gone; f32::MAX = live.
    fading: f32,
}

/// M2TW: probability .3 — only this fraction of arrows sing.
const FLY_LOOP_PROB: f32 = 0.3;
/// Concurrent fly loops (each is one more decoder on the mixer).
const MAX_FLY_LOOPS: usize = 12;
/// M2TW: fadeout .2 s at the arrow's death.
const FLY_LOOP_FADE: f32 = 0.2;

/// Attach, follow, and retire the per-arrow fly loops.
#[allow(clippy::too_many_arguments)] // bevy system params
fn arrow_fly_loops(
    mut commands: Commands,
    bank: Option<Res<AudioBank>>,
    arrows: Res<crate::arrows::Arrows>,
    camera: Query<&RtsCamera>,
    time: Res<Time>,
    virt_time: Res<Time<Virtual>>,
    settings: Res<crate::settings::Settings>,
    mut loops: Query<(Entity, &mut ArrowFlyLoop, Option<&mut AudioSink>)>,
) {
    let Some(bank) = bank else { return };
    let Ok(cam) = camera.single() else { return };
    let dt = time.delta_secs();
    let paused = virt_time.is_paused();
    let bv = battle_vol(&settings);
    let zoom_att = zoom_attenuation(cam.distance);
    let focus = Vec2::new(cam.focus.x, cam.focus.z);
    let hear = 140.0 + cam.distance * 0.3;

    // One pass over the pool: positions of tracked ids + spawn candidates.
    let mut tracked: Vec<u32> = loops.iter().map(|(_, l, _)| l.arrow_id).collect();
    let mut found: Vec<(u32, f32)> = Vec::with_capacity(tracked.len()); // (id, dist)
    let mut budget = MAX_FLY_LOOPS.saturating_sub(tracked.len());
    for i in 0..arrows.len() {
        let id = arrows.id[i];
        let d = arrows.pos[i].xz().distance(focus);
        if tracked.contains(&id) {
            found.push((id, d));
            continue;
        }
        // New candidates: the M2TW probability gate (stable per id, so
        // an arrow is either a singer for its whole flight or never),
        // in earshot, budget-capped.
        if budget > 0
            && !paused
            && d < hear
            && hash01(id.wrapping_mul(0x9E37_79B1)) < FLY_LOOP_PROB
            && let Some(h) = pick(&bank.arrow_fly_loop, id)
        {
            commands.spawn((
                AudioPlayer::new(h),
                PlaybackSettings {
                    volume: Volume::Linear(0.0),
                    speed: 0.85 + 0.3 * hash01(id.wrapping_mul(0x85EB)),
                    ..PlaybackSettings::LOOP
                },
                ArrowFlyLoop {
                    arrow_id: id,
                    fading: f32::MAX,
                },
            ));
            tracked.push(id);
            budget -= 1;
        }
    }

    for (e, mut fl, sink) in &mut loops {
        let live = found.iter().find(|(id, _)| *id == fl.arrow_id);
        if live.is_none() && fl.fading == f32::MAX {
            // Its arrow landed this frame: begin the M2TW fade-out (the
            // impact one-shot is taking over right now).
            fl.fading = FLY_LOOP_FADE;
        }
        let Some(mut sink) = sink else { continue };
        if fl.fading < f32::MAX {
            fl.fading -= dt;
            if fl.fading <= 0.0 {
                commands.entity(e).despawn();
                continue;
            }
            let cur = sink.volume().to_linear();
            sink.set_volume(Volume::Linear(cur * (fl.fading / FLY_LOOP_FADE).clamp(0.0, 1.0)));
        } else if paused {
            sink.set_volume(Volume::Linear(0.0));
        } else if let Some((_, d)) = live {
            let prox = (1.0 - d / hear).clamp(0.0, 1.0);
            sink.set_volume(Volume::Linear(0.16 * prox * prox * zoom_att.max(0.3) * bv));
        }
    }
}

/// State for `archer_one_shots`, bundled into one Local.
#[derive(Default)]
struct ArrowAudioState {
    loose_acc: f32,
    impact_acc: f32,
    ground_acc: f32,
    /// Loosed-per-second EMA (the volley-onset detector).
    rate_ema: f32,
    away_high: bool,
    away_gate: f32,
    incoming_high: bool,
    incoming_gate: f32,
    flyby_acc: f32,
    frame: u32,
}

/// Archer sound: string snaps and impact thuds budgeted from the
/// per-tick ArrowStats counters (same fractional-accumulator pattern as
/// the melee clangs), plus the two massed-volley sheets as edge cues —
/// `volley_away` when the loose rate spikes near friendly bows,
/// `volley_incoming` when a cloud of falling arrows is about to land
/// near the camera. Assets: sfx_bow/ (tmp/archer-sfx-prompts.md).
#[allow(clippy::too_many_arguments)] // bevy system params
fn archer_one_shots(
    mut commands: Commands,
    bank: Option<Res<AudioBank>>,
    astats: Res<crate::arrows::ArrowStats>,
    arrows: Res<crate::arrows::Arrows>,
    groups: Res<Groups>,
    camera: Query<&RtsCamera>,
    time: Res<Time>,
    virt_time: Res<Time<Virtual>>,
    settings: Res<crate::settings::Settings>,
    playing: Query<(), (With<AudioPlayer>, Without<Bed>)>,
    mut st: Local<ArrowAudioState>,
) {
    let Some(bank) = bank else { return };
    let Ok(cam) = camera.single() else { return };
    // Mixer-protection allowance for this frame (see MAX_LIVE_ONE_SHOTS).
    let mut allowance = MAX_LIVE_ONE_SHOTS.saturating_sub(playing.iter().count()) as u32;
    // A paused sim freezes the per-tick counters at their last values:
    // integrating them while paused would drip phantom arrows forever.
    if virt_time.is_paused() {
        return;
    }
    st.frame = st.frame.wrapping_add(1);
    let dt = time.delta_secs();
    st.away_gate -= dt;
    st.incoming_gate -= dt;
    let bv = battle_vol(&settings);
    let zoom_att = zoom_attenuation(cam.distance);
    let focus = Vec2::new(cam.focus.x, cam.focus.z);
    let hear = 200.0 + cam.distance * 0.4;

    // Where the bows are: nearest archer regiment still carrying arrows.
    let shooter_dist = groups
        .list
        .iter()
        .filter(|g| {
            g.kind == crate::unit_types::KIND_ARCHER && g.count > 0 && g.ammo_left > 0
        })
        .map(|g| g.centroid.distance(focus))
        .fold(f32::MAX, f32::min);
    let shooter_prox = if shooter_dist == f32::MAX {
        0.0
    } else {
        (1.0 - shooter_dist / hear).clamp(0.0, 1.0)
    };

    // Where the arrows are: the volley cloud, and how many shafts are
    // dropping around the camera (the flyby feed).
    let cloud_prox = arrow_cloud(&arrows)
        .map(|c| (1.0 - c.distance(focus) / hear).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let mut falling_near = 0usize;
    for i in 0..arrows.len() {
        if arrows.vel[i].y < 0.0 && arrows.pos[i].xz().distance_squared(focus) < 80.0 * 80.0 {
            falling_near += 1;
        }
    }

    // String snaps: a volley is the EVENT, not garnish over a din (the
    // melee-clang tuning read as "a dozen archers", owner-tested) — a
    // large share of looses each spawn a jittered snap and the clips
    // LAYER into the ripple of a massed release. Distance thins the
    // count only mildly; it mostly quiets the clips.
    st.loose_acc +=
        astats.loosed as f32 * 30.0 * dt * 0.35 * shooter_prox * (0.4 + 0.6 * zoom_att);
    st.loose_acc = st.loose_acc.min(5.0);
    let mut n = (st.loose_acc).floor() as u32;
    st.loose_acc -= n as f32;
    n = n.min(4).min(allowance);
    allowance -= n;
    for k in 0..n {
        let seed = st.frame.wrapping_mul(53) ^ k ^ 0x51;
        if let Some(h) = pick(&bank.bow_loose, seed) {
            one_shot(
                &mut commands,
                h,
                (0.09 + 0.06 * hash01(seed ^ 0x12)) * zoom_att * bv,
                0.88 + 0.24 * hash01(seed ^ 0x23),
            );
        }
    }

    // Body hits (wood knock at low odds — the shaft that found a shield)
    // and the dirt patter of the misses, near the falling cloud. Same
    // mass-scale budgeting as the looses.
    st.impact_acc += astats.hits as f32 * 30.0 * dt * 0.30 * cloud_prox * (0.4 + 0.6 * zoom_att);
    st.impact_acc = st.impact_acc.min(4.0);
    let mut n = (st.impact_acc).floor() as u32;
    st.impact_acc -= n as f32;
    n = n.min(3).min(allowance);
    allowance -= n;
    for k in 0..n {
        let seed = st.frame.wrapping_mul(71) ^ k ^ 0x62;
        let set = if hash01(seed ^ 0xA7) < 0.2 {
            &bank.arrow_wood
        } else {
            &bank.arrow_flesh
        };
        if let Some(h) = pick(set, seed) {
            one_shot(
                &mut commands,
                h,
                (0.11 + 0.08 * hash01(seed ^ 0x13)) * zoom_att * bv,
                0.88 + 0.24 * hash01(seed ^ 0x24),
            );
        }
    }
    let ground = astats.landed.saturating_sub(astats.hits);
    st.ground_acc += ground as f32 * 30.0 * dt * 0.10 * cloud_prox * (0.4 + 0.6 * zoom_att);
    st.ground_acc = st.ground_acc.min(3.0);
    let mut n = (st.ground_acc).floor() as u32;
    st.ground_acc -= n as f32;
    n = n.min(2).min(allowance);
    allowance -= n;
    for k in 0..n {
        let seed = st.frame.wrapping_mul(89) ^ k ^ 0x73;
        if let Some(h) = pick(&bank.arrow_ground, seed) {
            one_shot(
                &mut commands,
                h,
                (0.07 + 0.05 * hash01(seed ^ 0x14)) * zoom_att * bv,
                0.86 + 0.28 * hash01(seed ^ 0x25),
            );
        }
    }

    // Volley-away sheet: the loose rate spiking = a massed release.
    let rate = astats.loosed as f32 * 30.0;
    st.rate_ema += (rate - st.rate_ema) * (dt / 0.25).min(1.0);
    let high = st.rate_ema > 60.0;
    if VOLLEY_SHEETS_ON && high && !st.away_high && st.away_gate <= 0.0 && shooter_prox > 0.05 {
        let seed = st.frame.wrapping_mul(131);
        if let Some(h) = pick(&bank.volley_away, seed) {
            let vol = 0.5 * (0.3 + 0.7 * shooter_prox) * bv;
            one_shot(&mut commands, h, vol, 0.96 + 0.08 * hash01(seed ^ 0x15));
            info!("volley away (vol {vol:.2}, rate {:.0}/s)", st.rate_ema);
        }
        st.away_gate = 2.0;
    }
    st.away_high = high;

    // Incoming sheet: a cloud of descending shafts over the camera.
    let inc = falling_near > 25;
    if VOLLEY_SHEETS_ON && inc && !st.incoming_high && st.incoming_gate <= 0.0 && zoom_att > 0.2
    {
        let seed = st.frame.wrapping_mul(151);
        if let Some(h) = pick(&bank.volley_incoming, seed) {
            let vol = 0.45 * zoom_att.max(0.5) * bv;
            one_shot(&mut commands, h, vol, 0.96 + 0.08 * hash01(seed ^ 0x16));
            info!("volley incoming (vol {vol:.2}, {falling_near} falling near)");
        }
        st.incoming_gate = 2.5;
    }
    st.incoming_high = inc;

    // Flyby whooshes: budgeted from the COUNT of shafts currently
    // falling around the camera (an arrow spends ~1.5 s in the zone, so
    // ~0.5/s per shaft means most of them whistle once) — a volley
    // passing overhead layers itself into a rushing stream that starts,
    // thickens, and fades with the actual flight; a lone skirmish
    // arrow stays a lone whistle. Camera must be down in it (zoom).
    if zoom_att > 0.35 {
        st.flyby_acc += falling_near as f32 * dt * 0.28 * zoom_att;
    }
    st.flyby_acc = st.flyby_acc.min(3.0);
    let mut n = (st.flyby_acc).floor() as u32;
    st.flyby_acc -= n as f32;
    n = n.min(2).min(allowance);
    for k in 0..n {
        let seed = st.frame.wrapping_mul(173) ^ k ^ 0x84;
        if let Some(h) = pick(&bank.arrow_flyby, seed) {
            one_shot(
                &mut commands,
                h,
                (0.05 + 0.04 * hash01(seed ^ 0x17)) * zoom_att * bv,
                0.86 + 0.28 * hash01(seed ^ 0x28),
            );
        }
    }
}

/// Edge-detection state for `event_cues`, bundled into one Local (the
/// bare-Local version blew past Bevy's 16 system-param limit).
#[derive(Default)]
struct CueState {
    prev_state: Vec<u8>,
    prev_outcome: bool,
    horn_gate: f32,
    vox_gate: f32,
    frame: u32,
}

/// Discrete cues: selection and order clicks, charge horn on new
/// orders, rout/rally vox + horn, victory/defeat stings. The charge
/// war cries live in `charge_vox`.
#[allow(clippy::too_many_arguments)] // bevy system params
fn event_cues(
    mut commands: Commands,
    bank: Option<Res<AudioBank>>,
    groups: Res<Groups>,
    mut cues: MessageReader<UiCue>,
    outcome: Res<crate::ai::BattleOutcome>,
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
    st.prev_state.resize(groups.list.len(), 0);

    // UI click feedback: one clip per action kind per frame, straight
    // from the input systems — every click sounds, repeats included.
    let (mut cue_select, mut cue_move, mut cue_attack, mut cue_deploy) =
        (false, false, false, false);
    for cue in cues.read() {
        match cue {
            UiCue::Select => cue_select = true,
            UiCue::Move => cue_move = true,
            UiCue::Attack => cue_attack = true,
            UiCue::Deploy => cue_deploy = true,
        }
    }
    if cue_select {
        one_shot(&mut commands, bank.ui_select.clone(), 0.5 * uv, 1.0);
    }

    let mut new_break_own = false;
    let mut new_break_any = false;
    for (g, gd) in groups.list.iter().enumerate() {
        let state = match gd.state {
            RegState::Steady => 0u8,
            RegState::Routing { .. } => 1,
            RegState::Shattered => 2,
        };
        if state >= 1 && st.prev_state[g] == 0 && gd.count > 0 {
            new_break_any = true;
            if gd.team == 0 {
                new_break_own = true;
            }
        }
        st.prev_state[g] = state;
    }

    let seed = st.frame.wrapping_mul(211);

    // Order-click feedback: immediate and ungated, like the selection
    // click — this is UI, not battlefield sound. Deployment placements
    // click like a move but never horn.
    if cue_attack {
        one_shot(&mut commands, bank.ui_attack.clone(), 0.45 * uv, 1.0);
    }
    if cue_move || cue_deploy {
        one_shot(&mut commands, bank.ui_order.clone(), 0.35 * uv, 1.0);
    }
    if (cue_attack || cue_move) && st.horn_gate <= 0.0 {
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
            // The victors' roar moved to celebrate_vox: the M2TW cheer
            // is a STATE (the last nearby foe gone), not a break edge.
            // Rally has no vox for now (owner benched the old clips);
            // a rally cue needs a fresh asset first.
            st.vox_gate = 1.5;
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

/// State for `charge_vox`, bundled into one Local.
#[derive(Default)]
struct ChargeVoxState {
    yell_acc: f32,
    /// Per-regiment group-sheet clock: seconds until the next sheet.
    sheet_t: Vec<f32>,
    frame: u32,
}

/// The M2TW charge soundscape, two layers (mined configs, devlog
/// 0069). Foreground: single-man charge yells — M2TW plays
/// `Individual_Charge` on one in five charging soldiers, each his own
/// positional clip — budgeted here from the men charging near the
/// camera, the same fractional-accumulator pattern as the clangs and
/// bow looses. Background: the `unit_charge` group sheet — one quiet
/// size-banded crowd clip per charging regiment, retriggered every
/// 2.0..2.5 s while the charge state lasts, overlapping itself into a
/// wash. The yells carry the moment; the sheet is texture under them.
#[allow(clippy::too_many_arguments)] // bevy system params
fn charge_vox(
    mut commands: Commands,
    bank: Option<Res<AudioBank>>,
    groups: Res<Groups>,
    camera: Query<&RtsCamera>,
    time: Res<Time>,
    virt_time: Res<Time<Virtual>>,
    settings: Res<crate::settings::Settings>,
    playing: Query<(), (With<AudioPlayer>, Without<Bed>)>,
    mut st: Local<ChargeVoxState>,
) {
    let Some(bank) = bank else { return };
    let Ok(cam) = camera.single() else { return };
    if virt_time.is_paused() {
        return;
    }
    st.frame = st.frame.wrapping_add(1);
    let dt = time.delta_secs();
    let bv = battle_vol(&settings);
    let zoom_att = zoom_attenuation(cam.distance);
    let focus = Vec2::new(cam.focus.x, cam.focus.z);
    let hear = 220.0 + cam.distance * 0.5;
    let mut allowance = MAX_LIVE_ONE_SHOTS.saturating_sub(playing.iter().count()) as u32;
    st.sheet_t.resize(groups.list.len(), 0.0);

    // One pass: the yell feed (charging men, proximity-squared
    // weighted) and the per-regiment sheet clocks.
    let mut men_near = 0.0f32;
    let mut best_prox = 0.0f32;
    for (g, gd) in groups.list.iter().enumerate() {
        if !gd.charging || gd.count == 0 {
            // A finished charge resets its clock: the next one opens
            // with an immediate sheet.
            st.sheet_t[g] = 0.0;
            continue;
        }
        let prox = (1.0 - gd.centroid.distance(focus) / hear).clamp(0.0, 1.0);
        men_near += gd.count as f32 * prox * prox;
        best_prox = best_prox.max(prox);

        st.sheet_t[g] -= dt;
        if st.sheet_t[g] <= 0.0 {
            let seed = st.frame.wrapping_mul(191) ^ (g as u32).wrapping_mul(0x9E37);
            if prox > 0.05 && allowance > 0 {
                let set = if gd.count >= 300 {
                    &bank.group_charge_large
                } else {
                    &bank.group_charge_medium
                };
                if let Some(h) = pick(set, seed) {
                    // Sits under the yell layer but must stay audible
                    // through it: a sustained wash reads quieter than
                    // its gain suggests (first cut at 0.10 vanished
                    // entirely under yells + beds).
                    one_shot(
                        &mut commands,
                        h,
                        0.22 * (0.3 + 0.7 * prox) * zoom_att.max(0.55) * bv,
                        0.94 + 0.12 * hash01(seed ^ 0x31),
                    );
                    allowance -= 1;
                }
            }
            // M2TW: delay 2.0 randomdelay .5 — the 4 s clips overlap.
            st.sheet_t[g] = 2.0 + 0.5 * hash01(seed ^ 0x42);
        }
    }

    // Yell budget: 0.2 yells per man (M2TW probability) spread over the
    // measured 2.8..7 s charge window (~4.5 s mid, devlog 0056). Our
    // regiments run far bigger than M2TW's 60-150 men, so the caps and
    // the one-shot allowance keep a 1000-man charge a chorus instead
    // of a decoder flood.
    if best_prox > 0.0 {
        st.yell_acc += men_near * (0.2 / 4.5) * dt * (0.4 + 0.6 * zoom_att);
    }
    st.yell_acc = st.yell_acc.min(3.0);
    let mut n = st.yell_acc.floor() as u32;
    st.yell_acc -= n as f32;
    n = n.min(2).min(allowance);
    for k in 0..n {
        let seed = st.frame.wrapping_mul(227) ^ k ^ 0x95;
        if let Some(h) = pick(&bank.yell_charge, seed) {
            one_shot(
                &mut commands,
                h,
                (0.20 + 0.10 * hash01(seed ^ 0x18))
                    * (0.25 + 0.75 * best_prox)
                    * zoom_att
                    * bv,
                0.90 + 0.20 * hash01(seed ^ 0x29),
            );
        }
    }
}

/// State for `rout_vox`, bundled into one Local.
#[derive(Default)]
struct RoutVoxState {
    prev_broken: Vec<bool>,
    /// Seconds of the initial-panic window left, per regiment.
    panic_left: Vec<f32>,
    /// Per-regiment feet-wash clock (seconds until the next wash).
    wash_t: Vec<f32>,
    panic_acc: f32,
    shout_acc: f32,
    frame: u32,
}

/// The rout soundscape. Owner's direction from film retreats
/// (Napoleon 2023, The Patriot), matching the M2TW config: after the
/// first panicked moments men flee mostly SILENT — what carries is
/// officers shouting Retreat / Withdraw / Fall back (M2TW
/// Individual_Retreat is exactly such voice lines, sparse at p .08)
/// and the drumming feet of the running mass (unit_run bank; M2TW's
/// own group retreat sheet is benched in its config, so none here
/// either). Three layers per BROKEN regiment: panic screams only
/// inside a 7 s window after the break, sparse command shouts while
/// any rout runs in earshot, and a rolling feet wash underneath.
/// The break EDGE itself (crowd vox + horn) stays in event_cues.
#[allow(clippy::too_many_arguments)] // bevy system params
fn rout_vox(
    mut commands: Commands,
    bank: Option<Res<AudioBank>>,
    groups: Res<Groups>,
    camera: Query<&RtsCamera>,
    time: Res<Time>,
    virt_time: Res<Time<Virtual>>,
    settings: Res<crate::settings::Settings>,
    playing: Query<(), (With<AudioPlayer>, Without<Bed>)>,
    mut st: Local<RoutVoxState>,
) {
    let Some(bank) = bank else { return };
    let Ok(cam) = camera.single() else { return };
    if virt_time.is_paused() {
        return;
    }
    st.frame = st.frame.wrapping_add(1);
    let dt = time.delta_secs();
    let bv = battle_vol(&settings);
    let zoom_att = zoom_attenuation(cam.distance);
    let focus = Vec2::new(cam.focus.x, cam.focus.z);
    let hear = 220.0 + cam.distance * 0.5;
    let mut allowance = MAX_LIVE_ONE_SHOTS.saturating_sub(playing.iter().count()) as u32;
    let n_groups = groups.list.len();
    st.prev_broken.resize(n_groups, false);
    st.panic_left.resize(n_groups, 0.0);
    st.wash_t.resize(n_groups, 0.0);

    let mut panic_men = 0.0f32;
    let mut best_prox = 0.0f32;
    for (g, gd) in groups.list.iter().enumerate() {
        let broken = gd.state.is_broken() && gd.count > 0;
        if broken && !st.prev_broken[g] {
            st.panic_left[g] = 7.0;
        }
        st.prev_broken[g] = broken;
        if !broken {
            st.panic_left[g] = 0.0;
            st.wash_t[g] = 0.0;
            continue;
        }
        st.panic_left[g] = (st.panic_left[g] - dt).max(0.0);
        let prox = (1.0 - gd.centroid.distance(focus) / hear).clamp(0.0, 1.0);
        best_prox = best_prox.max(prox);
        if st.panic_left[g] > 0.0 {
            panic_men += gd.count as f32 * prox * prox;
        }

        // Feet wash: rolling 6 s clips per fleeing regiment, first on
        // the break edge, sized by regiment strength.
        st.wash_t[g] -= dt;
        if st.wash_t[g] <= 0.0 {
            let seed = st.frame.wrapping_mul(331) ^ (g as u32).wrapping_mul(0x9E37);
            if prox > 0.05
                && allowance > 0
                && let Some(h) = pick(&bank.feet_wash, seed)
            {
                let size = (gd.count as f32 / 300.0).min(1.0);
                one_shot(
                    &mut commands,
                    h,
                    0.20 * (0.5 + 0.5 * size)
                        * (0.3 + 0.7 * prox)
                        * zoom_att.max(0.5)
                        * bv,
                    0.94 + 0.12 * hash01(seed ^ 0x91),
                );
                allowance -= 1;
            }
            st.wash_t[g] = 3.0 + 1.0 * hash01(seed ^ 0xA2);
        }
    }

    // Initial panic: screams only in the first moments of a break,
    // then the mass goes quiet (the film observation).
    st.panic_acc += (panic_men * 0.00066).min(1.5) * dt;
    st.panic_acc = st.panic_acc.min(2.0);
    let mut n = st.panic_acc.floor() as u32;
    st.panic_acc -= n as f32;
    n = n.min(1).min(allowance);
    allowance -= n;
    for k in 0..n {
        let seed = st.frame.wrapping_mul(347) ^ k ^ 0xB3;
        if let Some(h) = pick(&bank.rout_panic, seed) {
            one_shot(
                &mut commands,
                h,
                (0.24 + 0.08 * hash01(seed ^ 0xC4)) * (0.25 + 0.75 * best_prox) * zoom_att * bv,
                0.92 + 0.16 * hash01(seed ^ 0xD5),
            );
        }
    }

    // Command shouts: one officer voice every ~3 s while any rout
    // runs in earshot — not per man, and never a machine gun during
    // a mass collapse.
    st.shout_acc += best_prox * 0.35 * dt;
    st.shout_acc = st.shout_acc.min(1.5);
    let mut n = st.shout_acc.floor() as u32;
    st.shout_acc -= n as f32;
    n = n.min(1).min(allowance);
    for k in 0..n {
        let seed = st.frame.wrapping_mul(353) ^ k ^ 0xE6;
        if let Some(h) = pick(&bank.rout_shout, seed) {
            one_shot(
                &mut commands,
                h,
                (0.30 + 0.08 * hash01(seed ^ 0xF7)) * (0.25 + 0.75 * best_prox) * zoom_att * bv,
                0.94 + 0.12 * hash01(seed ^ 0x19),
            );
        }
    }
}

/// State for `celebrate_vox`, bundled into one Local.
#[derive(Default)]
struct CelebrateVoxState {
    whoop_acc: f32,
    /// Per-regiment cheer-sheet clock (seconds until the next sheet).
    sheet_t: Vec<f32>,
    frame: u32,
}

/// Victory celebration (M2TW unit_celebrate state bank, devlog 0070):
/// while a regiment's `celebrate` window runs (the last nearby foe
/// routed or died, ~5 s, frontline.rs), it cheers as a STATE — a
/// rolling size-banded group cheer per regiment, first sheet on the
/// edge, retriggered so the 4-6 s clips overlap (M2TW: fadein 1
/// fadeout 3, randomdelay 1, FULL volume — the loudest group event
/// in the bank set), with single-man whoops budgeted over it
/// (Individual_Celebrate p .08).
#[allow(clippy::too_many_arguments)] // bevy system params
fn celebrate_vox(
    mut commands: Commands,
    bank: Option<Res<AudioBank>>,
    groups: Res<Groups>,
    camera: Query<&RtsCamera>,
    time: Res<Time>,
    virt_time: Res<Time<Virtual>>,
    settings: Res<crate::settings::Settings>,
    playing: Query<(), (With<AudioPlayer>, Without<Bed>)>,
    mut st: Local<CelebrateVoxState>,
) {
    let Some(bank) = bank else { return };
    let Ok(cam) = camera.single() else { return };
    if virt_time.is_paused() {
        return;
    }
    st.frame = st.frame.wrapping_add(1);
    let dt = time.delta_secs();
    let bv = battle_vol(&settings);
    let zoom_att = zoom_attenuation(cam.distance);
    let focus = Vec2::new(cam.focus.x, cam.focus.z);
    let hear = 220.0 + cam.distance * 0.5;
    let mut allowance = MAX_LIVE_ONE_SHOTS.saturating_sub(playing.iter().count()) as u32;
    st.sheet_t.resize(groups.list.len(), 0.0);

    let mut men_near = 0.0f32;
    let mut best_prox = 0.0f32;
    for (g, gd) in groups.list.iter().enumerate() {
        if gd.celebrate == 0 || gd.count == 0 {
            // A finished cheer resets its clock: the next one opens
            // with an immediate sheet on the celebrate edge.
            st.sheet_t[g] = 0.0;
            continue;
        }
        let prox = (1.0 - gd.centroid.distance(focus) / hear).clamp(0.0, 1.0);
        men_near += gd.count as f32 * prox * prox;
        best_prox = best_prox.max(prox);

        st.sheet_t[g] -= dt;
        if st.sheet_t[g] <= 0.0 {
            let seed = st.frame.wrapping_mul(307) ^ (g as u32).wrapping_mul(0x9E37);
            if prox > 0.05 && allowance > 0 {
                let set = if gd.count >= 300 {
                    &bank.celebrate_large
                } else {
                    &bank.celebrate_small
                };
                if let Some(h) = pick(set, seed) {
                    // M2TW plays this bank at full volume: the loudest
                    // sheet in our ledger, above the charge sheets.
                    one_shot(
                        &mut commands,
                        h,
                        0.40 * (0.3 + 0.7 * prox) * zoom_att.max(0.5) * bv,
                        0.96 + 0.08 * hash01(seed ^ 0x51),
                    );
                    allowance -= 1;
                }
            }
            // Retrigger under the clip length so the cheers roll.
            st.sheet_t[g] = 2.5 + 1.0 * hash01(seed ^ 0x62);
        }
    }

    // Individual whoops: M2TW p .08 per man, spread over the 5 s
    // celebrate window, proximity-thinned.
    const WHOOP_RATE_CAP: f32 = 10.0;
    st.whoop_acc += (men_near * (0.08 / 5.0)).min(WHOOP_RATE_CAP) * dt;
    st.whoop_acc = st.whoop_acc.min(2.0);
    let mut n = st.whoop_acc.floor() as u32;
    st.whoop_acc -= n as f32;
    n = n.min(2).min(allowance);
    for k in 0..n {
        let seed = st.frame.wrapping_mul(311) ^ k ^ 0xA7;
        if let Some(h) = pick(&bank.celebrate_whoop, seed) {
            one_shot(
                &mut commands,
                h,
                (0.22 + 0.08 * hash01(seed ^ 0x73))
                    * (0.25 + 0.75 * best_prox)
                    * zoom_att
                    * bv,
                0.92 + 0.16 * hash01(seed ^ 0x84),
            );
        }
    }
}

/// State for `melee_vox`, bundled into one Local.
#[derive(Default)]
struct MeleeVoxState {
    hit_acc: f32,
    ambient_acc: f32,
    frame: u32,
}

/// The melee human layer (M2TW soldier_voice vocals, devlog 0070):
/// men, not steel, carry the sound of a fight. Two feeds. Per-HIT
/// vocals ride the same sim hit counter as the clangs — each spawn
/// picks the attacker's grunt, the victim's grunt, or the attacker's
/// scream at the M2TW probability ratio (.4 / .25 / .25). A slow
/// ambient stream of battle screams (M2TW p .2, the loudest clips)
/// comes from the mass of ENGAGED men near the camera. M2TW plays
/// all of these at tiny mindist (0.75-2 m) — a strictly close-up
/// layer, so zoom attenuates it harder than the beds.
#[allow(clippy::too_many_arguments)] // bevy system params
fn melee_vox(
    mut commands: Commands,
    bank: Option<Res<AudioBank>>,
    stats: Res<SimStats>,
    groups: Res<Groups>,
    camera: Query<&RtsCamera>,
    time: Res<Time>,
    virt_time: Res<Time<Virtual>>,
    settings: Res<crate::settings::Settings>,
    playing: Query<(), (With<AudioPlayer>, Without<Bed>)>,
    mut st: Local<MeleeVoxState>,
) {
    let Some(bank) = bank else { return };
    let Ok(cam) = camera.single() else { return };
    if virt_time.is_paused() {
        return;
    }
    st.frame = st.frame.wrapping_add(1);
    let dt = time.delta_secs();
    let bv = battle_vol(&settings);
    let zoom_att = zoom_attenuation(cam.distance);
    let focus = Vec2::new(cam.focus.x, cam.focus.z);
    let hear = 200.0 + cam.distance * 0.4;

    // Same audible-share model as the clangs (see combat_one_shots):
    // hits attributed to engaged regiments by strength, weighted by
    // earshot. Camera distance shapes volume, never rate.
    let (mut min_dist, mut engaged_total, mut engaged_audible) = (f32::MAX, 0.0f32, 0.0f32);
    let mut men_near = 0.0f32;
    for gd in groups.list.iter().filter(|g| g.engaged && g.count > 0) {
        let d = gd.centroid.distance(focus);
        min_dist = min_dist.min(d);
        let p = (1.0 - d / hear).clamp(0.0, 1.0);
        engaged_total += gd.count as f32;
        engaged_audible += gd.count as f32 * p;
        men_near += gd.count as f32 * p * p;
    }
    let audible = if engaged_total > 0.0 {
        engaged_audible / engaged_total
    } else {
        0.0
    };
    let prox = if min_dist == f32::MAX {
        0.0
    } else {
        (1.0 - min_dist / hear).clamp(0.0, 1.0)
    };
    let mut allowance = MAX_LIVE_ONE_SHOTS.saturating_sub(playing.iter().count()) as u32;

    // Per-hit vocals at 0.6 per steel: M2TW rides ~.9 vocals per hit
    // but each QUIETER than the material hit — steel is the carrier,
    // voices the garnish (weapon_hit p 1 vol -10..0 vs vocals
    // p .4/.25/.25 vol -20/-15). Whiffed swings go unheard (the sim
    // only counts hits); acceptable undercount.
    const VOCAL_RATE_CAP: f32 = 18.0;
    let near_hit_rate = stats.events as f32 * 30.0 * audible;
    st.hit_acc += (near_hit_rate * 0.6).min(VOCAL_RATE_CAP) * dt;
    st.hit_acc = st.hit_acc.min(3.0);
    let mut n = st.hit_acc.floor() as u32;
    st.hit_acc -= n as f32;
    n = n.min(2).min(allowance);
    allowance -= n;
    for k in 0..n {
        let seed = st.frame.wrapping_mul(241) ^ k ^ 0xC3;
        let r = hash01(seed ^ 0xD1);
        let (set, vol) = if r < 0.45 {
            (&bank.melee_attack_grunt, 0.13 + 0.06 * hash01(seed ^ 0x1A))
        } else if r < 0.73 {
            (&bank.melee_hit_grunt, 0.15 + 0.06 * hash01(seed ^ 0x2B))
        } else {
            (&bank.melee_attack_scream, 0.22 + 0.08 * hash01(seed ^ 0x3C))
        };
        if let Some(h) = pick(set, seed) {
            one_shot(
                &mut commands,
                h,
                vol * (0.25 + 0.75 * prox) * zoom_att * bv,
                0.92 + 0.16 * hash01(seed ^ 0x4D),
            );
        }
    }

    // Ambient battle screams: ~one every 1-2 s over a close 1000-man
    // melee, fading out fast with distance and zoom (zoom shapes the
    // volume below, not this rate).
    st.ambient_acc += men_near * dt * 0.0008;
    st.ambient_acc = st.ambient_acc.min(2.0);
    let mut n = st.ambient_acc.floor() as u32;
    st.ambient_acc -= n as f32;
    n = n.min(1).min(allowance);
    for k in 0..n {
        let seed = st.frame.wrapping_mul(263) ^ k ^ 0xE5;
        if let Some(h) = pick(&bank.melee_battle_scream, seed) {
            one_shot(
                &mut commands,
                h,
                (0.26 + 0.08 * hash01(seed ^ 0x5E)) * (0.25 + 0.75 * prox) * zoom_att * bv,
                0.92 + 0.16 * hash01(seed ^ 0x6F),
            );
        }
    }
}
