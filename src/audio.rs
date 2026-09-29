//! Battle audio. Battlefield sounds are positional: the sim reports each
//! landed blow, loosed arrow and arrow strike where it happens, regiments
//! in a charge, celebration or rout emit their group sheets and single
//! voices from their own ground, and the mixer (mixer.rs) plays each clip
//! at a common loudness, levels it by the mix table below, fades it with
//! distance from the point the camera looks at, and caps each layer's
//! voices. Beds, UI clicks, horns and stings stay plain Bevy players.
//!
//! Missing files degrade gracefully (their triggers just stay silent).

use bevy::audio::{AudioSink, AudioSinkPlayback, PlaybackSettings, Volume};
use bevy::prelude::*;

use crate::camera::RtsCamera;
use crate::game_state::GameState;
use crate::mixer::{Bank, Bus, ClipId, Clips, Duck, Mixer, NO_UNIT, Request};
use crate::orders::{Groups, RegState};
use crate::sim::SimStats;
use crate::units::hash01;

/// Env master override (FL_VOLUME, linear). Multiplies the settings
/// volumes so scripted test runs stay muted regardless of the saved
/// settings file.
pub fn env_master() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::util::env_or("FL_VOLUME", 1.0))
}

/// Effective battle-sound volume (beds, combat, vox, horns, stings).
pub(crate) fn battle_vol(s: &crate::settings::Settings) -> f32 {
    s.audio.master * s.audio.battle * env_master()
}

/// Effective UI-sound volume (selection/order clicks).
pub(crate) fn ui_vol(s: &crate::settings::Settings) -> f32 {
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
/// The battle beds start to dip when the positional mix runs louder than
/// this (dBFS, per channel)...
const BED_DUCK_FROM_DB: f32 = -40.0;
/// ...and dip at most this much (dB).
const BED_DUCK_MAX_DB: f32 = 6.0;

/// Per-clip loudness manifest (tools/audio_loudness.py): the gain in dB
/// that brings each clip to the common loudness, and its length in
/// seconds, by path under assets/.
fn clip_info(path: &str) -> (f32, f32) {
    type Manifest = std::collections::HashMap<String, (f32, f32)>;
    static CLIPS: std::sync::OnceLock<Manifest> = std::sync::OnceLock::new();
    let clips = CLIPS.get_or_init(|| {
        let v: serde_json::Value =
            serde_json::from_str(include_str!("../assets/audio_levels.json")).unwrap_or_default();
        v["clips"]
            .as_object()
            .map(|clips| {
                clips
                    .iter()
                    .filter_map(|(k, c)| {
                        Some((k.clone(), (c["gain_db"].as_f64()? as f32, c["seconds"].as_f64()? as f32)))
                    })
                    .collect()
            })
            .unwrap_or_default()
    });
    match clips.get(path) {
        Some(c) => *c,
        None => {
            warn!("audio: {path} is not in assets/audio_levels.json; run tools/audio_loudness.py");
            (0.0, 0.0)
        }
    }
}

fn norm_db(path: &str) -> f32 {
    clip_info(path).0
}

/// Load `<name>.mp3` for every name, each at the common loudness.
fn pool(clips: &mut Clips, assets: &AssetServer, names: &[&str]) -> Vec<ClipId> {
    names.iter().map(|n| load(clips, assets, &format!("{n}.mp3"))).collect()
}

fn load(clips: &mut Clips, assets: &AssetServer, path: &str) -> ClipId {
    clips.load(assets, path, norm_db(path))
}

pub struct BattleAudioPlugin;

impl Plugin for BattleAudioPlugin {
    fn build(&self, app: &mut App) {
        // Battle-only: outside the battle the sim stats these systems
        // read are stale, and quitting to the menu must not leave the
        // beds ringing behind it.
        app.add_message::<UiCue>()
            .add_plugins(crate::mixer::MixerPlugin)
            .init_resource::<SoundEvents>()
            .init_resource::<MemberSamples>()
            .add_systems(Startup, setup_audio)
            .add_systems(
                Update,
                (
                    update_beds,
                    refresh_member_samples,
                    blow_sounds,
                    arrow_sounds,
                    regiment_sounds,
                    event_cues,
                    crate::mixer::flush_mixer,
                    clip_log,
                )
                    .chain()
                    .run_if(in_state(GameState::Battle)),
            )
            // Beds cut on leaving battle; one-shots survive into the
            // results screen (the victory/defeat sting must finish)
            // and are only culled when the menu comes up.
            .add_systems(OnExit(GameState::Battle), (silence_beds, clear_sound_events));
    }
}

// ------------------------------------------------------------ sim events

/// Sim sounds are kept within this distance of the point the camera looks
/// at, plus half the camera's distance (a zoomed-out view hears farther).
const EVENT_CULL_M: f32 = 150.0;
/// Cap per event list, so a frame that never drains cannot grow them.
const EVENT_CAP: usize = 4096;

/// What a blow or an arrow struck, which picks its sound (M2TW
/// weapon_hit `hit metal / wood / flesh`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Material {
    /// Blade on blade: the blow was parried.
    Steel,
    /// Metal armour.
    Metal,
    /// A shield.
    Wood,
    /// Flesh or leather: M2TW plays one sample set for both.
    Flesh,
}

/// The material a landed blow struck, drawn in proportion to the defence
/// points that met it: parry skill (blade on blade), shield (wood) and
/// armour (the victim's armour sound). No defence in play: flesh. `roll`
/// is uniform in 0..1. M2TW picks the material per hit too, by a rule its
/// config does not show.
pub fn struck_material(skill: f32, shield: f32, armour: f32, kind: u8, roll: f32) -> Material {
    let total = skill.max(0.0) + shield.max(0.0) + armour.max(0.0);
    if total <= 0.0 {
        return Material::Flesh;
    }
    let x = roll * total;
    if x < skill {
        Material::Steel
    } else if x < skill + shield {
        Material::Wood
    } else {
        armour_material(kind)
    }
}

/// The armour sound of a kind. M2TW's EDU gives every unit one
/// (`stat_pri_armour`): armour 0-1 is flesh, 2-4 mostly leather, 5 and
/// up mostly metal.
pub fn armour_material(kind: u8) -> Material {
    use crate::unit_types::{KIND_HEAVY, KIND_SPEAR};
    match kind {
        KIND_HEAVY | KIND_SPEAR => Material::Metal,
        _ => Material::Flesh,
    }
}

/// A landed melee blow.
pub struct Blow {
    pub victim: Vec3,
    pub attacker: Vec3,
    pub victim_group: u32,
    pub attacker_group: u32,
    pub material: Material,
    pub killed: bool,
}

/// An arrow that struck a man.
pub struct ArrowHit {
    pub pos: Vec3,
    /// The struck man's regiment.
    pub group: u32,
    pub material: Material,
    pub killed: bool,
}

/// Sim events near the camera's look point, written where they happen
/// (the damage pass, the arrow flight) and drained by the audio systems
/// each frame.
#[derive(Resource, Default)]
pub struct SoundEvents {
    /// The camera's look point, written by the audio each frame.
    pub listener: Vec3,
    /// Events farther than this from the look point are dropped.
    pub cull_r: f32,
    pub blows: Vec<Blow>,
    pub arrow_hits: Vec<ArrowHit>,
    pub arrow_ground: Vec<Vec3>,
    /// Loosed arrows: the shooter's regiment and the launch point.
    pub looses: Vec<(u32, Vec3)>,
}

impl SoundEvents {
    fn hears(&self, p: Vec3) -> bool {
        p.xz().distance_squared(self.listener.xz()) < self.cull_r * self.cull_r
    }

    pub fn push_blow(&mut self, b: Blow) {
        if self.blows.len() < EVENT_CAP && self.hears(b.victim) {
            self.blows.push(b);
        }
    }

    pub fn push_arrow_hit(&mut self, h: ArrowHit) {
        if self.arrow_hits.len() < EVENT_CAP && self.hears(h.pos) {
            self.arrow_hits.push(h);
        }
    }

    pub fn push_arrow_ground(&mut self, p: Vec3) {
        if self.arrow_ground.len() < EVENT_CAP && self.hears(p) {
            self.arrow_ground.push(p);
        }
    }

    pub fn push_loose(&mut self, group: u32, p: Vec3) {
        if self.looses.len() < EVENT_CAP && self.hears(p) {
            self.looses.push((group, p));
        }
    }
}

fn clear_sound_events(mut ev: ResMut<SoundEvents>) {
    ev.blows.clear();
    ev.arrow_hits.clear();
    ev.arrow_ground.clear();
    ev.looses.clear();
}

// ------------------------------------------------------------ banks

/// The mix table. Each layer's level in dB relative to a death scream, at
/// full level (inside the bank's mindist of the look point, close zoom).
/// Starting values: the balance of the mix approved by ear before
/// positional audio, measured as each pool's median loudness plus its old
/// gain; the blows keep the material balance of the first positional mix
/// (the metal ring 19 dB under the shield). Priorities and distance
/// priorities are M2TW's. Group caps are the voices the approved mix held
/// (about 31 for the steel, 18 for the grunts and screams together, 1-2
/// death screams, 14 bow strings), nearest first. The charge sheets, the
/// volley, the arrow air and the rout cry sit 7 dB over that balance,
/// which cancels MIX_GAIN_DB for them: they play at the approved mix's
/// absolute level while the melee plays 7 dB under it. The charge yells
/// sit 5 dB over it with a larger cap (more yells, each a little
/// quieter), and the bow strings 10 dB over it so the volley does not
/// cover them.
mod mix {
    /// A death scream's output level in the approved mix: pool median
    /// -7.7 LUFS at gain 0.24 (-12.4 dB).
    pub const DEATH_SCREAM_OUT: f32 = -20.1;
    /// Normalized clip loudness (tools/audio_loudness.py targets).
    pub const ONE_SHOT: f32 = -20.0;
    pub const SUSTAINED: f32 = -23.0;
    /// An equal-power pan puts a centred mono voice 3 dB down per channel;
    /// the approved mix played it at full level in both.
    pub const PAN_CENTRE: f32 = 3.0;
    /// Full level within this distance of the look point: one man, and a
    /// whole regiment's sheet.
    pub const MAN_M: f32 = 20.0;
    pub const SHEET_M: f32 = 30.0;
}

/// Every positional level shifts by this (dB): the one by-ear knob over the
/// table. -7 puts a dense melee at the look point near -16 dBFS, where the
/// approved mix sat, instead of -9 dBFS on the limiter.
const MIX_GAIN_DB: f32 = -7.0;

#[allow(clippy::too_many_arguments)]
const fn bank(
    name: &'static str,
    group: &'static str,
    max_live: u16,
    priority: f32,
    dist_priority: f32,
    rel_db: f32,
    sustained: bool,
    speed: (f32, f32),
) -> Bank {
    let clip = if sustained { mix::SUSTAINED } else { mix::ONE_SHOT };
    Bank {
        name,
        priority,
        dist_priority,
        mindist: if sustained { mix::SHEET_M } else { mix::MAN_M },
        vol_db: mix::DEATH_SCREAM_OUT + rel_db - clip + mix::PAN_CENTRE + MIX_GAIN_DB,
        zoom_floor: if sustained { 0.5 } else { 0.0 },
        speed,
        group,
        max_live,
        duck: Duck::None,
    }
}

/// An unplaced sound's bank (UI click, sting): the table level without
/// the centre-pan compensation, since it plays at full level in both
/// channels.
const fn flat_bank(name: &'static str, rel_db: f32, sustained: bool) -> Bank {
    let b = bank(name, name, 0, 250.0, 0.0, rel_db, sustained, (1.0, 1.0));
    Bank {
        vol_db: b.vol_db - mix::PAN_CENTRE,
        ..b
    }
}

// Horns, UI clicks and stings at their approved level relative to the
// soldiers (old gain on the pool's loudness, relative to the death
// scream).
/// M2TW `war_horn`: placed, full level within 40 m, priority 240, fixed
/// pitch, carried across the field.
const HORN_CHARGE: Bank = Bank {
    mindist: 40.0,
    zoom_floor: 1.0,
    ..bank("horn", "horn", 2, 240.0, 0.0, 5.0, true, (1.0, 1.0))
};
const HORN_ROUT: Bank = Bank {
    mindist: 40.0,
    zoom_floor: 1.0,
    ..bank("horn", "horn", 2, 240.0, 0.0, 9.5, false, (1.0, 1.0))
};
/// M2TW `unit_warhorns_delay 9`: one war horn per army per 9 s.
const WAR_HORN_GAP_S: f32 = 9.0;
const UI_SELECT: Bank = flat_bank("ui", -8.8, false);
/// A move order clicks 5.6 dB under an attack order.
const UI_ORDER: Bank = flat_bank("ui", -1.9, false);
const UI_ATTACK: Bank = flat_bank("ui", 3.7, false);
const STING_VICTORY: Bank = flat_bank("sting", 3.5, true);
const STING_DEFEAT: Bank = flat_bank("sting", -1.1, true);

// Blows: one shared group, the shield thud loudest, the rings quiet.
const HIT_WOOD: Bank = bank("hit wood", "blows", 32, 90.0, -2.0, -5.7, false, (0.8, 1.2));
const HIT_FLESH: Bank = bank("hit flesh", "blows", 32, 90.0, -2.0, -11.6, false, (0.8, 1.2));
const HIT_METAL: Bank = bank("hit metal", "blows", 32, 90.0, -2.0, -24.7, false, (0.6, 1.1));
const HIT_STEEL: Bank = bank("hit steel", "blows", 32, 90.0, -2.0, -24.9, false, (0.6, 1.1));
/// Killing blow. Our flesh-connect pool stands in until a death-hit pool
/// exists.
const DEATH_HIT: Bank = bank("death hit", "blows", 32, 180.0, -2.0, -4.9, false, (0.8, 1.2));
const DEATH_SCREAM: Bank = bank("death scream", "death scream", 3, 130.0, 0.0, 0.0, false, (0.9, 1.1));
const ATTACK_GRUNT: Bank = Bank {
    duck: Duck::Target,
    ..bank("attack grunt", "voices", 18, 120.0, 0.0, -5.5, false, (0.92, 1.08))
};
const ATTACK_SCREAM: Bank = Bank {
    duck: Duck::Target,
    ..bank("attack scream", "voices", 18, 120.0, 0.0, 1.6, false, (0.92, 1.08))
};
const VICTIM_GRUNT: Bank = Bank {
    duck: Duck::Target,
    ..bank("victim grunt", "voices", 18, 120.0, 0.0, -8.2, false, (0.92, 1.08))
};
const BATTLE_SCREAM: Bank = Bank {
    duck: Duck::Target,
    ..bank("battle scream", "battle scream", 8, 100.0, 0.0, 2.2, false, (0.92, 1.08))
};
/// Bow string on each loose, at the archer.
const BOW_STRING: Bank = Bank {
    duck: Duck::Trigger,
    ..bank("bow string", "bow string", 16, 110.0, 0.0, -9.4, false, (0.88, 1.12))
};
const ARROW_FLESH: Bank = bank("arrow flesh", "arrow strike", 8, 90.0, -2.0, -16.3, false, (0.8, 1.2));
const ARROW_WOOD: Bank = bank("arrow wood", "arrow strike", 8, 90.0, -2.0, -14.5, false, (0.8, 1.2));
const ARROW_DEATH_HIT: Bank = bank("death hit", "arrow strike", 8, 180.0, -2.0, -4.9, false, (0.8, 1.2));
const ARROW_GROUND: Bank = bank("arrow ground", "arrow ground", 6, 90.0, -2.0, -30.7, false, (0.8, 1.2));
/// A looped air sound on three arrows in ten, followed in flight.
const ARROW_FLY: Bank = bank("arrow fly", "arrow fly", 12, 170.0, 0.0, -19.7, true, (0.5, 1.5));
/// A shaft dropping past the look point.
const ARROW_WHIZZ: Bank = bank("arrow whizz", "arrow whizz", 4, 170.0, 0.0, -22.9, false, (0.7, 1.3));
/// One group release per volley share (M2TW unit_missile_attack).
const VOLLEY: Bank = Bank {
    duck: Duck::Trigger,
    ..bank("volley", "volley", 4, 190.0, -1.0, 3.6, true, (0.9, 1.1))
};
/// A charge yell is heard across the charging block (about 60 m wide):
/// full level within a sheet's distance.
const CHARGE_YELL: Bank = Bank {
    mindist: mix::SHEET_M,
    duck: Duck::Target,
    ..bank("charge yell", "charge yell", 64, 80.0, 0.0, 4.6, false, (0.9, 1.1))
};
const CHARGE_SHEET: Bank = bank("charge sheet", "charge sheet", 0, 170.0, -1.0, 3.6, true, (0.8, 1.1));
const CHEER_SHEET: Bank = bank("cheer sheet", "cheer sheet", 0, 170.0, -1.0, -2.0, true, (0.9, 1.0));
const WHOOP: Bank = bank("whoop", "whoop", 10, 100.0, 0.0, 1.1, false, (0.92, 1.08));
const ROUT_SHOUT: Bank = bank("rout shout", "rout shout", 3, 120.0, 0.0, 3.9, false, (0.94, 1.06));
const ROUT_PANIC: Bank = bank("rout panic", "rout panic", 4, 110.0, 0.0, -6.4, true, (0.92, 1.08));
const FEET: Bank = bank("feet", "feet", 6, 70.0, 0.0, -11.3, true, (0.94, 1.06));
/// A crowd cry from a regiment the moment it breaks, one at a time, 2 dB
/// over the approved break cue's absolute level (0.55 on clips of -12.5
/// LUFS), so it clears the melee around the breaking regiment.
const ROUT_CROWD: Bank = bank("rout crowd", "rout crowd", 1, 150.0, -1.0, 11.4, true, (1.0, 1.0));

/// One sim tick in seconds: a tick's events are spread over it.
const TICK_S: f32 = 1.0 / 30.0;

/// The positional clip pools, decoded into the mixer.
#[derive(Resource)]
struct Pools {
    bow_string: Vec<ClipId>,
    hit_steel: Vec<ClipId>,
    hit_metal: Vec<ClipId>,
    hit_wood: Vec<ClipId>,
    hit_flesh: Vec<ClipId>,
    death_scream: Vec<ClipId>,
    attack_grunt: Vec<ClipId>,
    attack_scream: Vec<ClipId>,
    victim_grunt: Vec<ClipId>,
    battle_scream: Vec<ClipId>,
    arrow_flesh: Vec<ClipId>,
    arrow_wood: Vec<ClipId>,
    arrow_ground: Vec<ClipId>,
    arrow_fly: Vec<ClipId>,
    arrow_whizz: Vec<ClipId>,
    volley: Vec<ClipId>,
    charge_yell: Vec<ClipId>,
    charge_medium: Vec<ClipId>,
    charge_large: Vec<ClipId>,
    cheer_small: Vec<ClipId>,
    cheer_large: Vec<ClipId>,
    whoop: Vec<ClipId>,
    rout_shout: Vec<ClipId>,
    rout_panic: Vec<ClipId>,
    rout_crowd: Vec<ClipId>,
    feet: Vec<ClipId>,
    horn_charge: Vec<ClipId>,
    horn_rout: ClipId,
    ui_select: ClipId,
    /// Click feedback for a move order.
    ui_order: ClipId,
    /// Click feedback for an attack order on an enemy regiment.
    ui_attack: ClipId,
    sting_victory: ClipId,
    sting_defeat: ClipId,
}

/// The looping bed layers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Bed {
    Far,
    Mid,
    Close,
    /// Massed boots (Pixabay loop) while an own regiment marches.
    March,
}

impl Bed {
    const ALL: [Bed; 4] = [Bed::Far, Bed::Mid, Bed::Close, Bed::March];

    /// The layer's takes under assets/. The first one's loudness is the
    /// layer's level; the others are matched to it.
    fn takes(self) -> &'static [&'static str] {
        match self {
            Bed::Far => &["bed_battle_far.mp3"],
            Bed::Mid => &["bed_battle_mid0.mp3", "bed_battle_mid1.mp3"],
            Bed::Close => &[
                "bed_melee_close0.mp3",
                "bed_melee_close1.mp3",
                "bed_melee_close2.mp3",
                "bed_melee_close3.mp3",
                "bed_melee_close4.mp3",
                "bed_melee_close5.mp3",
            ],
            Bed::March => &["sfx_new/bed_march_loop_14.5s.mp3"],
        }
    }

    /// Gain that matches take `i` to the layer's first take.
    fn take_gain(self, i: usize) -> f32 {
        let t = self.takes();
        crate::mixer::db_to_lin(norm_db(t[i]) - norm_db(t[0]))
    }
}

/// One playing take of a bed layer. A layer with several takes hands over
/// to another one before the current take ends, crossfading.
#[derive(Component)]
struct BedTake {
    bed: Bed,
    take: usize,
    /// Crossfade position 0..1: rises while the take is current, falls
    /// once it has been handed over.
    xfade: f32,
    current: bool,
}

/// Crossfade between two takes of a bed layer (s).
const BED_XFADE_S: f32 = 3.0;

fn spawn_bed_take(commands: &mut Commands, assets: &AssetServer, bed: Bed, take: usize, xfade: f32) {
    commands.spawn((
        AudioPlayer::new(assets.load(bed.takes()[take])),
        PlaybackSettings {
            volume: Volume::Linear(0.0),
            ..PlaybackSettings::LOOP
        },
        BedTake {
            bed,
            take,
            xfade,
            current: true,
        },
    ));
}

fn setup_audio(mut commands: Commands, assets: Res<AssetServer>, mut clips: ResMut<Clips>) {
    let clips = &mut *clips;

    commands.insert_resource(Pools {
        bow_string: pool(clips, &assets, &[
            "sfx_bow/sfx_bow_loose_01",
            "sfx_bow/sfx_bow_loose_02",
            "sfx_bow/sfx_bow_loose_03",
            "sfx_bow/sfx_bow_loose_04",
        ]),
        // The second batch (sfx_new/) won out over the first clangs.
        hit_steel: pool(clips, &assets, &[
            "sfx_new/sword_clang_06",
            "sfx_new/sword_clang_07",
            "sfx_new/sword_clang_08",
            "sfx_new/sword_clang_09",
        ]),
        hit_metal: pool(clips, &assets, &["sfx_new/armor_clang_01", "sfx_new/armor_clang_02"]),
        hit_wood: pool(clips, &assets, &["sfx_shield_01", "sfx_shield_02", "sfx_shield_03"]),
        hit_flesh: pool(clips, &assets, &[
            "sfx_new/sfx_blunt_damage_01",
            "sfx_new/sfx_blunt_damage_02",
            "sfx_new/sfx_sword_damage_01",
            "sfx_new/sfx_sword_damage_02",
        ]),
        death_scream: pool(clips, &assets, &[
            "sfx_death_01",
            "sfx_death_02",
            "sfx_death_03",
            "sfx_death_04",
            "sfx_death_05",
        ]),
        attack_grunt: pool(clips, &assets, &[
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
        attack_scream: pool(clips, &assets, &[
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
        victim_grunt: pool(clips, &assets, &[
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
        battle_scream: pool(clips, &assets, &[
            "sfx_melee/battle_screams/battle_scream0",
            "sfx_melee/battle_screams/battle_scream1",
            "sfx_melee/battle_screams/battle_scream3",
            "sfx_melee/battle_screams/battle_scream4",
            "sfx_melee/battle_screams/battle_scream6",
            "sfx_melee/battle_screams/battle_scream7",
            "sfx_melee/battle_screams/battle_scream8",
            "sfx_melee/battle_screams/battle_scream9",
        ]),
        arrow_flesh: pool(clips, &assets, &[
            "sfx_bow/sfx_arrow_flesh_01",
            "sfx_bow/sfx_arrow_flesh_02",
            "sfx_bow/sfx_arrow_flesh_03",
        ]),
        arrow_wood: pool(clips, &assets, &["sfx_bow/sfx_arrow_wood_01", "sfx_bow/sfx_arrow_wood_02"]),
        arrow_ground: pool(clips, &assets, &[
            "sfx_bow/sfx_arrow_ground_01",
            "sfx_bow/sfx_arrow_ground_02",
            "sfx_bow/sfx_arrow_ground_03",
        ]),
        arrow_fly: pool(clips, &assets, &["sfx_bow/sfx_arrow_fly_loop_01", "sfx_bow/sfx_arrow_fly_loop_02"]),
        arrow_whizz: pool(clips, &assets, &["sfx_bow/sfx_arrow_flyby_01", "sfx_bow/sfx_arrow_flyby_02"]),
        volley: vec![
            load(clips, &assets, "sfx_bow/sfx_volley_away_01.wav"),
            load(clips, &assets, "sfx_bow/sfx_volley_away_02.mp3"),
            load(clips, &assets, "sfx_bow/sfx_volley_away_03.mp3"),
        ],
        charge_yell: pool(
            clips,
            &assets,
            &[
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
            ],
        ),
        charge_medium: pool(clips, &assets, &["sfx_charge/group_charge_medium"]),
        charge_large: pool(
            clips,
            &assets,
            &["sfx_charge/group_charge_large_01", "sfx_charge/group_charge_large_02"],
        ),
        cheer_small: pool(
            clips,
            &assets,
            &[
                "sfx_celebrate/group_cheer_small_01_mocking",
                "sfx_celebrate/group_cheer_small_02_mocking",
                "sfx_celebrate/group_cheer_small_04_laughing",
                "sfx_celebrate/group_cheer_small_05_laughing",
                "sfx_celebrate/group_cheer_small_06_laughing",
                "sfx_celebrate/group_cheer_small_07_joyous_shouts",
                "sfx_celebrate/group_cheer_small_08_joyous_shouts",
            ],
        ),
        cheer_large: pool(
            clips,
            &assets,
            &[
                "sfx_celebrate/group_cheer_large_01",
                "sfx_celebrate/group_cheer_large_02_short",
                "sfx_celebrate/group_cheer_large_03_ok",
                "sfx_celebrate/group_cheer_large_04",
                "sfx_celebrate/group_cheer_large_05_ok",
            ],
        ),
        // vox_whoop_04 benched by its own filename (skipfornow).
        whoop: pool(
            clips,
            &assets,
            &[
                "sfx_celebrate/vox_whoop_01",
                "sfx_celebrate/vox_whoop_02_joyous_shout",
                "sfx_celebrate/vox_whoop_03_laugh",
                "sfx_celebrate/vox_whoop_05_knight_laugh",
                "sfx_celebrate/vox_whoop_06_cheer_yeah",
                "sfx_celebrate/vox_whoop_07_roar_yeah",
                "sfx_celebrate/vox_whoop_08_knight_shout_yes",
                "sfx_celebrate/vox_whoop_09_knight_shout_yeaa",
            ],
        ),
        rout_shout: pool(
            clips,
            &assets,
            &[
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
            ],
        ),
        rout_panic: pool(
            clips,
            &assets,
            &["sfx_rout/vox_panic_01", "sfx_rout/vox_panic_02", "sfx_rout/vox_panic_03"],
        ),
        // vox_rally_01/02 are benched: unusable, use nowhere.
        rout_crowd: pool(
            clips,
            &assets,
            &[
                "vox_rout_01",
                "vox_rout_02",
                "vox_rout_03",
                "sfx_rout/vox_rout_04",
                "sfx_rout/vox_rout_05",
            ],
        ),
        // Massed washes layered from single-man source loops by
        // work/scripts/build-feet-wash.sh (ElevenLabs would only produce
        // one or two runners per take).
        feet: pool(
            clips,
            &assets,
            &["sfx_rout/feet_run_wash_mass_01", "sfx_rout/feet_run_wash_mass_02"],
        ),
        horn_charge: pool(clips, &assets, &["sig_horn_charge", "sig_horn_charge_02"]),
        horn_rout: load(clips, &assets, "sfx_new/sig_horn_rout.mp3"),
        ui_select: load(clips, &assets, "sfx_new/ui_select1.mp3"),
        ui_order: load(clips, &assets, "sfx_new/ui_order0.mp3"),
        ui_attack: load(clips, &assets, "sfx_new/ui_attack.mp3"),
        sting_victory: load(clips, &assets, "sting_victory.mp3"),
        sting_defeat: load(clips, &assets, "sting_defeat.mp3"),
    });

    for bed in Bed::ALL {
        spawn_bed_take(&mut commands, &assets, bed, 0, 1.0);
    }
}

fn silence_beds(mut sinks: Query<&mut AudioSink, With<BedTake>>) {
    for mut sink in &mut sinks {
        sink.set_volume(Volume::Linear(0.0));
    }
}

/// Crossfade the beds from battle state around the camera focus, and
/// rotate each layer's takes.
#[allow(clippy::too_many_arguments)] // bevy system params
fn update_beds(
    mut commands: Commands,
    assets: Res<AssetServer>,
    groups: Res<Groups>,
    stats: Res<SimStats>,
    camera: Query<&RtsCamera>,
    time: Res<Time<Real>>,
    virt_time: Res<Time<Virtual>>,
    settings: Res<crate::settings::Settings>,
    mixer: Res<Mixer>,
    mut takes: Query<(Entity, &mut BedTake, Option<&mut AudioSink>)>,
    mut clock: Local<([f32; 4], u32)>,
    mut next_log: Local<f32>,
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

    // The battle beds dip while the fight at the look point is loud, so a
    // bed never masks the men in front of the camera: 0.5 dB per dB the
    // positional mix runs over BED_DUCK_FROM_DB, at most BED_DUCK_MAX_DB.
    let duck_db = ((mixer.recent_db() - BED_DUCK_FROM_DB) * 0.5).clamp(0.0, BED_DUCK_MAX_DB);
    let duck = crate::mixer::db_to_lin(-duck_db);
    // Level before the master volume; the sink gets it times `m`.
    let m = battle_vol(&settings);
    let level = |bed: &Bed| -> f32 {
        if paused {
            return 0.0;
        }
        match bed {
            Bed::Far => 0.22 * ((engaged_total as f32) / 8.0).clamp(0.0, 1.0) * duck,
            Bed::Mid => {
                0.45 * ((engaged_near as f32) / 5.0).clamp(0.0, 1.0) * prox.sqrt() * duck
            }
            Bed::Close => {
                0.60 * prox * prox * (0.25 + 0.75 * hits) * (0.35 + 0.65 * zoom_att) * duck
            }
            Bed::March => {
                if marching_own {
                    0.28
                } else {
                    0.0
                }
            }
        }
    };

    // Hand each multi-take layer over to another take before the current
    // one ends (a paused battle holds the clocks).
    let dt = if paused { 0.0 } else { time.delta_secs() };
    let (left, roll) = &mut *clock;
    for (i, bed) in Bed::ALL.into_iter().enumerate() {
        let n = bed.takes().len();
        if n < 2 {
            continue;
        }
        left[i] -= dt;
        if left[i] > 0.0 {
            continue;
        }
        let mut current = 0;
        for (_, mut t, _) in &mut takes {
            if t.bed == bed && t.current {
                t.current = false;
                current = t.take;
            }
        }
        *roll = roll.wrapping_add(1);
        let next = (current + 1 + (hash01(roll.wrapping_mul(0x9E37_79B1)) * (n - 1) as f32) as usize % (n - 1)) % n;
        spawn_bed_take(&mut commands, &assets, bed, next, 0.0);
        left[i] = clip_info(bed.takes()[next]).1 - BED_XFADE_S;
    }

    let blend = (time.delta_secs() / BED_SMOOTH).min(1.0);
    let step = time.delta_secs() / BED_XFADE_S;
    let mut levels = Vec::new();
    for (e, mut t, sink) in &mut takes {
        t.xfade = if t.current { (t.xfade + step).min(1.0) } else { t.xfade - step };
        if !t.current && t.xfade <= 0.0 {
            commands.entity(e).despawn();
            continue;
        }
        // Equal-power crossfade.
        let w = (t.xfade.clamp(0.0, 1.0) * std::f32::consts::FRAC_PI_2).sin();
        let lv = level(&t.bed) * t.bed.take_gain(t.take) * w;
        if let Some(mut sink) = sink {
            let cur = sink.volume().to_linear();
            sink.set_volume(Volume::Linear(cur + (lv * m - cur) * blend));
        }
        levels.push((t.bed.takes()[t.take], lv));
    }

    // FL_LOG_AUDIO: each bed's level, its clip's loudness plus its volume
    // (LUFS, comparable to the mixer's per-group dBFS within a few dB).
    static LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *LOG.get_or_init(|| std::env::var("FL_LOG_AUDIO").is_ok()) && time.elapsed_secs() >= *next_log {
        *next_log = time.elapsed_secs() + 1.0;
        let line: Vec<String> = levels
            .iter()
            .filter(|(_, v)| *v > 1e-4)
            .map(|(path, v)| format!("{path} {:.0}", -23.0 - norm_db(path) + 20.0 * v.log10()))
            .collect();
        crate::mixer::audio_log(
            time.elapsed_secs_f64(),
            &format!("beds: dip {duck_db:.1} dB | {}", line.join(", ")),
        );
    }
}

/// How "inside the battle" the camera is by zoom, for the beds and every
/// positional sound: 1.0 at RTS close-up (<= 90 m), fading to a floor when
/// surveying the whole map.
pub(crate) fn zoom_attenuation(cam_distance: f32) -> f32 {
    (90.0 / cam_distance.max(90.0)).clamp(0.12, 1.0)
}

// ------------------------------------------------------------ positional sounds

/// Ask the mixer for a clip from `pool` at `pos`, a sound of regiment
/// `unit` (NO_UNIT for none).
fn play(mixer: &mut Mixer, pool: &[ClipId], bank: Bank, pos: Vec3, seed: u32, delay: f32, unit: u32) {
    play_owned(mixer, pool, bank, pos, seed, delay, unit, 0);
}

/// `play`, tagging the voice with `owner` so it can be faded out later.
#[allow(clippy::too_many_arguments)]
fn play_owned(
    mixer: &mut Mixer,
    pool: &[ClipId],
    bank: Bank,
    pos: Vec3,
    seed: u32,
    delay: f32,
    unit: u32,
    owner: u64,
) {
    if pool.is_empty() {
        return;
    }
    let clip = pool[(hash01(seed) * pool.len() as f32) as usize % pool.len()];
    mixer.request(Request {
        clip,
        bank,
        pos,
        speed_roll: hash01(seed ^ 0x5BD1_E995),
        delay,
        owner,
        unit,
    });
}

/// Melee blows (M2TW weapon_hit + soldier_voice, per blow): at the victim
/// the struck material, or on a killing blow the death hit and the death
/// scream; the attacker's grunt (p .4) and scream (p .25), the victim's
/// grunt and groan (p .25 each).
fn blow_sounds(
    mut mixer: ResMut<Mixer>,
    mut ev: ResMut<SoundEvents>,
    pools: Option<Res<Pools>>,
    camera: Query<&RtsCamera>,
    mut frame: Local<u32>,
) {
    if let Ok(cam) = camera.single() {
        ev.listener = cam.focus;
        ev.cull_r = EVENT_CULL_M + 0.5 * cam.distance;
    }
    let Some(pools) = pools else { return };
    *frame = frame.wrapping_add(1);
    let base = frame.wrapping_mul(0x9E37_79B1);
    for (k, b) in ev.blows.drain(..).enumerate() {
        let s = base ^ (k as u32).wrapping_mul(0x85EB_CA6B);
        let r = |salt: u32| hash01(s ^ salt);
        let delay = TICK_S * r(0x11);
        let (pool, bank) = match b.material {
            Material::Steel => (&pools.hit_steel, HIT_STEEL),
            Material::Metal => (&pools.hit_metal, HIT_METAL),
            Material::Wood => (&pools.hit_wood, HIT_WOOD),
            Material::Flesh => (&pools.hit_flesh, HIT_FLESH),
        };
        if b.killed {
            play(&mut mixer, &pools.hit_flesh, DEATH_HIT, b.victim, s ^ 0x31, delay, b.victim_group);
            play(&mut mixer, &pools.death_scream, DEATH_SCREAM, b.victim, s ^ 0x41, delay, b.victim_group);
        } else {
            play(&mut mixer, pool, bank, b.victim, s ^ 0x21, delay, b.victim_group);
            if r(0x51) < 0.25 {
                play(&mut mixer, &pools.victim_grunt, VICTIM_GRUNT, b.victim, s ^ 0x61, delay, b.victim_group);
            }
            if r(0x71) < 0.25 {
                play(&mut mixer, &pools.victim_grunt, VICTIM_GRUNT, b.victim, s ^ 0x81, delay, b.victim_group);
            }
        }
        if r(0x91) < 0.4 {
            play(&mut mixer, &pools.attack_grunt, ATTACK_GRUNT, b.attacker, s ^ 0xA1, delay, b.attacker_group);
        }
        if r(0xB1) < 0.25 {
            play(&mut mixer, &pools.attack_scream, ATTACK_SCREAM, b.attacker, s ^ 0xC1, delay, b.attacker_group);
        }
    }
}

/// Seconds a whizzed arrow id is remembered (one whizz per pass).
const WHIZZ_MEMORY_S: f32 = 3.0;
/// A falling shaft this close to the look point may whistle past it.
const WHIZZ_R: f32 = 25.0;
/// Arrows this close to the look point may carry an air loop.
const FLY_R: f32 = 40.0;

#[derive(Default)]
struct ArrowSoundState {
    whizzed: Vec<(u32, f32)>,
    frame: u32,
}

/// Arrow strikes and misses where they land, the air loop on three
/// arrows in ten (ARROW_FLY, followed in flight), and the whizz of three
/// falling shafts in ten near the look point.
#[allow(clippy::too_many_arguments)] // bevy system params
fn arrow_sounds(
    mut mixer: ResMut<Mixer>,
    mut ev: ResMut<SoundEvents>,
    pools: Option<Res<Pools>>,
    arrows: Res<crate::arrows::Arrows>,
    time: Res<Time>,
    virt_time: Res<Time<Virtual>>,
    mut st: Local<ArrowSoundState>,
) {
    let Some(pools) = pools else { return };
    st.frame = st.frame.wrapping_add(1);
    let base = st.frame.wrapping_mul(0x9E37_79B1);
    for (k, h) in ev.arrow_hits.drain(..).enumerate() {
        let s = base ^ (k as u32).wrapping_mul(0x85EB_CA6B) ^ 0x1234;
        let delay = TICK_S * hash01(s ^ 0x11);
        let (pool, bank) = match h.material {
            Material::Wood => (&pools.arrow_wood, ARROW_WOOD),
            // No arrow-on-metal clips yet: the body strike stands in.
            Material::Steel | Material::Metal | Material::Flesh => (&pools.arrow_flesh, ARROW_FLESH),
        };
        if h.killed {
            play(&mut mixer, &pools.hit_flesh, ARROW_DEATH_HIT, h.pos, s ^ 0x31, delay, h.group);
            play(&mut mixer, &pools.death_scream, DEATH_SCREAM, h.pos, s ^ 0x41, delay, h.group);
        } else {
            play(&mut mixer, pool, bank, h.pos, s ^ 0x21, delay, h.group);
        }
    }
    for (k, p) in ev.arrow_ground.drain(..).enumerate() {
        let s = base ^ (k as u32).wrapping_mul(0x85EB_CA6B) ^ 0x5678;
        play(&mut mixer, &pools.arrow_ground, ARROW_GROUND, p, s, TICK_S * hash01(s ^ 0x11), NO_UNIT);
    }

    // A paused battle holds the whizz memory with the arrows.
    let dt = if virt_time.is_paused() { 0.0 } else { time.delta_secs() };
    st.whizzed.retain_mut(|w| {
        w.1 -= dt;
        w.1 > 0.0
    });
    let listener = ev.listener.xz();
    let (fly_r2, whizz_r2) = (FLY_R * FLY_R, WHIZZ_R * WHIZZ_R);
    for i in 0..arrows.len() {
        let p = arrows.pos[i];
        let d2 = p.xz().distance_squared(listener);
        if d2 > fly_r2.max(whizz_r2) {
            continue;
        }
        let id = arrows.id[i];
        if d2 < fly_r2 && !pools.arrow_fly.is_empty() && hash01(id.wrapping_mul(0x9E37_79B1)) < 0.3 {
            let clip = pools.arrow_fly[id as usize % pools.arrow_fly.len()];
            mixer.track(
                id as u64,
                Request {
                    clip,
                    bank: ARROW_FLY,
                    pos: p,
                    speed_roll: hash01(id.wrapping_mul(0x85EB)),
                    delay: 0.0,
                    owner: 0,
                    unit: arrows.group[i],
                },
            );
        }
        if d2 < whizz_r2
            && arrows.vel[i].y < 0.0
            && hash01(id.wrapping_mul(0x27D4_EB2F)) < 0.3
            && !st.whizzed.iter().any(|w| w.0 == id)
        {
            st.whizzed.push((id, WHIZZ_MEMORY_S));
            play(&mut mixer, &pools.arrow_whizz, ARROW_WHIZZ, p, id ^ 0x77, 0.0, arrows.group[i]);
        }
    }
}

/// A few living men of each regiment, refreshed a slice of the army per
/// frame (a full pass every ~0.4 s at 200k), so single voices can come
/// from real soldiers without a per-regiment member list.
#[derive(Resource, Default)]
struct MemberSamples {
    cursor: usize,
    /// Per regiment: sampled unit indices of the last complete pass.
    active: Vec<Vec<u32>>,
    building: Vec<Vec<u32>>,
    seen: Vec<u32>,
}

const MEMBER_SAMPLES: usize = 16;
const MEMBER_SCAN_PER_FRAME: usize = 4096;

fn refresh_member_samples(
    units: Res<crate::units::Units>,
    groups: Res<Groups>,
    mut ms: ResMut<MemberSamples>,
) {
    let n = units.group.len();
    let ng = groups.list.len();
    let ms = &mut *ms;
    if ms.building.len() != ng {
        ms.active = vec![Vec::new(); ng];
        ms.building = vec![Vec::new(); ng];
        ms.seen = vec![0; ng];
        ms.cursor = 0;
    }
    if n == 0 {
        return;
    }
    let end = (ms.cursor + MEMBER_SCAN_PER_FRAME).min(n);
    for i in ms.cursor..end {
        let g = units.group[i] as usize;
        if g >= ng || units.death_t[i] != 0 {
            continue;
        }
        // Reservoir sample: every living man of the regiment equally likely.
        ms.seen[g] += 1;
        let b = &mut ms.building[g];
        if b.len() < MEMBER_SAMPLES {
            b.push(i as u32);
        } else {
            let r = (hash01(i as u32 ^ ms.seen[g].wrapping_mul(0x9E37)) * ms.seen[g] as f32) as usize;
            if r < MEMBER_SAMPLES {
                b[r] = i as u32;
            }
        }
    }
    ms.cursor = end;
    if ms.cursor >= n {
        ms.cursor = 0;
        std::mem::swap(&mut ms.active, &mut ms.building);
        for b in &mut ms.building {
            b.clear();
        }
        ms.seen.fill(0);
    }
}

impl MemberSamples {
    /// A living man of regiment `g` (his current position).
    fn man(&self, units: &crate::units::Units, g: usize, roll: f32) -> Option<Vec3> {
        let s = self.active.get(g)?;
        if s.is_empty() {
            return None;
        }
        let start = (roll * s.len() as f32) as usize % s.len();
        (0..s.len()).find_map(|k| {
            let i = s[(start + k) % s.len()] as usize;
            (i < units.group.len() && units.group[i] as usize == g && units.death_t[i] == 0)
                .then(|| units.pos[i])
        })
    }
}

/// Per-regiment clocks for the group sheets and single voices.
#[derive(Default)]
struct RegimentSoundState {
    charge_sheet_t: Vec<f32>,
    yell_acc: Vec<f32>,
    cheer_sheet_t: Vec<f32>,
    whoop_acc: Vec<f32>,
    prev_broken: Vec<bool>,
    prev_charging: Vec<bool>,
    panic_left: Vec<f32>,
    panic_acc: Vec<f32>,
    shout_acc: Vec<f32>,
    feet_t: Vec<f32>,
    scream_acc: Vec<f32>,
    volley_n: Vec<u32>,
    volley_at: Vec<Vec3>,
    volley_cool: Vec<f32>,
    frame: u32,
}

/// Charge yell window: M2TW's p .2 per man spread over the ~4.5 s middle
/// of the measured 2.8..7 s charge.
const CHARGE_YELL_WINDOW_S: f32 = 4.5;
/// Single voices a regiment may ask for in one frame (the allocator
/// keeps the strongest; this only bounds the work).
const VOICES_PER_FRAME: u32 = 6;
/// Charge yells a regiment may ask for in one frame: a 1000-man charge
/// wants about 44 a second.
const YELLS_PER_FRAME: u32 = 12;
/// The charge group sound's fade when the charge ends (M2TW
/// `unit_charge fadeout 1`).
const CHARGE_FADE_OUT_S: f32 = 1.0;
/// Owner tag of a regiment's charge group sounds (the regiment index in
/// the low bits).
const CHARGE_OWNER: u64 = 1 << 40;
/// The celebrate window is 150 ticks (frontline.rs).
const CELEBRATE_S: f32 = 5.0;
/// A new volley sound when this share of the regiment's living men has
/// loosed since the last one.
const VOLLEY_SHARE: f32 = 0.15;
/// Shortest gap between two volley sounds of one regiment (s).
const VOLLEY_GAP_S: f32 = 1.0;
/// Bow strings asked for per frame, nearest looses first.
const STRINGS_PER_FRAME: usize = 32;

/// Regiment-level sounds, each from the regiment's own ground: the charge
/// (group sheet on the M2TW 2.0 + 0.5 s clock, one yell per five men),
/// the celebration (rolling cheer sheets, whoops from one man in twelve),
/// the rout (panic at the break, officers shouting, running feet), battle
/// screams over a melee, the bow string on each loose, and the archer
/// volley (one group release each time a share of the regiment has
/// loosed).
#[allow(clippy::too_many_arguments)] // bevy system params
fn regiment_sounds(
    mut mixer: ResMut<Mixer>,
    mut ev: ResMut<SoundEvents>,
    pools: Option<Res<Pools>>,
    groups: Res<Groups>,
    units: Res<crate::units::Units>,
    members: Res<MemberSamples>,
    terrain: Res<crate::terrain::Terrain>,
    time: Res<Time>,
    virt_time: Res<Time<Virtual>>,
    mut st: Local<RegimentSoundState>,
) {
    let Some(pools) = pools else { return };
    let ng = groups.list.len();
    let st = &mut *st;
    for v in [
        &mut st.charge_sheet_t,
        &mut st.yell_acc,
        &mut st.cheer_sheet_t,
        &mut st.whoop_acc,
        &mut st.panic_left,
        &mut st.panic_acc,
        &mut st.shout_acc,
        &mut st.feet_t,
        &mut st.scream_acc,
        &mut st.volley_cool,
    ] {
        v.resize(ng, 0.0);
    }
    st.prev_broken.resize(ng, false);
    st.prev_charging.resize(ng, false);
    st.volley_n.resize(ng, 0);
    st.volley_at.resize(ng, Vec3::ZERO);
    if virt_time.is_paused() {
        ev.looses.clear();
        return;
    }
    st.frame = st.frame.wrapping_add(1);
    let dt = time.delta_secs();
    let listener = ev.listener;

    // A string snap per loose, at the archer: the nearest few per frame
    // (a volley tick looses hundreds; the nearest take the free bow-string
    // voices).
    let mut looses = std::mem::take(&mut ev.looses);
    for &(g, p) in &looses {
        let g = g as usize;
        if g < ng && st.volley_cool[g] <= 0.0 {
            st.volley_n[g] += 1;
            st.volley_at[g] += p;
        }
    }
    let near = |p: &Vec3| p.xz().distance_squared(listener.xz());
    if looses.len() > STRINGS_PER_FRAME {
        looses.select_nth_unstable_by(STRINGS_PER_FRAME, |a, b| near(&a.1).total_cmp(&near(&b.1)));
        looses.truncate(STRINGS_PER_FRAME);
    }
    for (k, &(lg, p)) in looses.iter().enumerate() {
        let s = st.frame.wrapping_mul(0x9E37_79B1) ^ (k as u32).wrapping_mul(0x85EB_CA6B) ^ 0x4242;
        play(&mut mixer, &pools.bow_string, BOW_STRING, p, s, TICK_S * hash01(s ^ 0x11), lg);
    }
    looses.clear();
    ev.looses = looses;

    for (g, gd) in groups.list.iter().enumerate() {
        let seed = st.frame.wrapping_mul(0x9E37_79B1) ^ (g as u32).wrapping_mul(0x85EB_CA6B);
        let r = |salt: u32| hash01(seed ^ salt);
        let alive = gd.count > 0;
        let broken = alive && gd.state.is_broken();
        let new_break = broken && !st.prev_broken[g];
        st.prev_broken[g] = broken;
        if new_break {
            st.panic_left[g] = ROUT_PANIC_S;
        }
        let break_cry = new_break;
        st.volley_cool[g] -= dt;

        // Out of earshot: keep the clocks honest, ask for nothing.
        let far = !alive || gd.centroid.distance(listener.xz()) - gd.radius > ev.cull_r;
        // Group sheets sound from the regiment's centre, at head height.
        let (cx, cz) = (gd.centroid.x, gd.centroid.y);
        let centre = Vec3::new(cx, terrain.height_at(cx, cz) + 1.5, cz);
        let men = gd.count as f32;

        // Charge. When the charge ends its group sound fades out over
        // M2TW's 1 s (`unit_charge fadeout 1`) instead of playing its
        // clip out; yells already started play to their end.
        let charging = gd.charging && alive;
        let charge_owner = CHARGE_OWNER | g as u64;
        if st.prev_charging[g] && !charging {
            mixer.fade_owner(charge_owner, CHARGE_FADE_OUT_S);
        }
        st.prev_charging[g] = charging;
        if charging {
            st.charge_sheet_t[g] -= dt;
            if st.charge_sheet_t[g] <= 0.0 {
                if !far {
                    let set = if gd.count >= 300 { &pools.charge_large } else { &pools.charge_medium };
                    play_owned(&mut mixer, set, CHARGE_SHEET, centre, seed ^ 0x13, 0.0, g as u32, charge_owner);
                }
                st.charge_sheet_t[g] = 2.0 + 0.5 * r(0x23);
            }
            st.yell_acc[g] += men * (0.2 / CHARGE_YELL_WINDOW_S) * dt;
            let n = (st.yell_acc[g] as u32).min(YELLS_PER_FRAME);
            st.yell_acc[g] -= st.yell_acc[g].floor();
            if !far {
                for k in 0..n {
                    if let Some(at) = members.man(&units, g, r(0x33 + k)) {
                        play(&mut mixer, &pools.charge_yell, CHARGE_YELL, at, seed ^ (0x43 + k), 0.2 * r(0x53 + k), g as u32);
                    }
                }
            }
        } else {
            st.charge_sheet_t[g] = 0.0;
            st.yell_acc[g] = 0.0;
        }

        // Celebration.
        if gd.celebrate > 0 && alive {
            st.cheer_sheet_t[g] -= dt;
            if st.cheer_sheet_t[g] <= 0.0 {
                if !far {
                    let set = if gd.count >= 300 { &pools.cheer_large } else { &pools.cheer_small };
                    play(&mut mixer, set, CHEER_SHEET, centre, seed ^ 0x63, 0.0, g as u32);
                }
                // M2TW randomdelay 1: the 4-6 s clips roll into each other.
                st.cheer_sheet_t[g] = 2.5 + 1.0 * r(0x73);
            }
            st.whoop_acc[g] += men * (0.08 / CELEBRATE_S) * dt;
            let n = (st.whoop_acc[g] as u32).min(VOICES_PER_FRAME);
            st.whoop_acc[g] -= st.whoop_acc[g].floor();
            if !far {
                for k in 0..n {
                    if let Some(at) = members.man(&units, g, r(0x83 + k)) {
                        play(&mut mixer, &pools.whoop, WHOOP, at, seed ^ (0x93 + k), 0.3 * r(0xA3 + k), g as u32);
                    }
                }
            }
        } else {
            st.cheer_sheet_t[g] = 0.0;
            st.whoop_acc[g] = 0.0;
        }

        // Rout: the crowd cry at the break, then panic, shouts and feet.
        if break_cry && !far {
            play(&mut mixer, &pools.rout_crowd, ROUT_CROWD, centre, seed ^ 0x163, 0.0, g as u32);
        }
        if broken {
            st.panic_left[g] = (st.panic_left[g] - dt).max(0.0);
            if st.panic_left[g] > 0.0 {
                st.panic_acc[g] += men * ROUT_PANIC_RATE * dt;
            }
            st.shout_acc[g] += ROUT_SHOUT_RATE * dt;
            st.feet_t[g] -= dt;
            if !far {
                let n = (st.panic_acc[g] as u32).min(VOICES_PER_FRAME);
                for k in 0..n {
                    if let Some(at) = members.man(&units, g, r(0xB3 + k)) {
                        play(&mut mixer, &pools.rout_panic, ROUT_PANIC, at, seed ^ (0xC3 + k), 0.2 * r(0xD3 + k), g as u32);
                    }
                }
                if st.shout_acc[g] >= 1.0
                    && let Some(at) = members.man(&units, g, r(0xE3))
                {
                    play(&mut mixer, &pools.rout_shout, ROUT_SHOUT, at, seed ^ 0xF3, 0.0, g as u32);
                }
                if st.feet_t[g] <= 0.0 {
                    play(&mut mixer, &pools.feet, FEET, centre, seed ^ 0x103, 0.0, g as u32);
                }
            }
            st.panic_acc[g] -= st.panic_acc[g].floor();
            st.shout_acc[g] -= st.shout_acc[g].floor();
            if st.feet_t[g] <= 0.0 {
                // The 6 s washes overlap into a steady drumming.
                st.feet_t[g] = 3.0 + 1.0 * r(0x113);
            }
        } else {
            st.panic_left[g] = 0.0;
            st.panic_acc[g] = 0.0;
            st.shout_acc[g] = 0.0;
            st.feet_t[g] = 0.0;
        }

        // Battle screams over a regiment in melee. M2TW's trigger for
        // Individual_Battle_Scream is not in its config; the rate is
        // devlog 0072's (about one scream every 1-2 s over a close
        // 1000-man melee), each from one of the regiment's men.
        if gd.engaged && alive {
            st.scream_acc[g] += men * BATTLE_SCREAM_RATE * dt;
            let n = (st.scream_acc[g] as u32).min(VOICES_PER_FRAME);
            st.scream_acc[g] -= st.scream_acc[g].floor();
            if !far {
                for k in 0..n {
                    if let Some(at) = members.man(&units, g, r(0x133 + k)) {
                        play(&mut mixer, &pools.battle_scream, BATTLE_SCREAM, at, seed ^ (0x143 + k), 0.3 * r(0x153 + k), g as u32);
                    }
                }
            }
        } else {
            st.scream_acc[g] = 0.0;
        }

        // Archer volley.
        let volley_n = st.volley_n[g];
        if volley_n > 0 && volley_n as f32 >= (men * VOLLEY_SHARE).max(5.0) {
            let at = st.volley_at[g] / volley_n as f32;
            play(&mut mixer, &pools.volley, VOLLEY, at, seed ^ 0x123, 0.0, g as u32);
            st.volley_n[g] = 0;
            st.volley_at[g] = Vec3::ZERO;
            st.volley_cool[g] = VOLLEY_GAP_S;
        }
    }
}

/// Panic at a break lasts this long, then the fleeing mass goes quiet
/// (film retreats, devlog 0074).
const ROUT_PANIC_S: f32 = 7.0;
/// Panic screams per man per second inside that window.
const ROUT_PANIC_RATE: f32 = 0.00066;
/// Battle screams per engaged man per second.
const BATTLE_SCREAM_RATE: f32 = 0.0008;
/// Officer shouts per routing regiment per second (devlog 0074: one voice
/// every ~3 s, not a machine gun).
const ROUT_SHOUT_RATE: f32 = 0.35;

/// Edge-detection state for `event_cues`, bundled into one Local (the
/// bare-Local version blew past Bevy's 16 system-param limit).
#[derive(Default)]
struct CueState {
    prev_state: Vec<u8>,
    prev_outcome: bool,
    /// Seconds until the army may sound another war horn.
    horn_gate: f32,
    frame: u32,
}

/// Discrete cues: selection and order clicks (unplaced, UI channel), the
/// war horn on new orders at the ordered regiments and on an own break at
/// the breaking regiment (placed, one per army per 9 s), and the outcome
/// sting (unplaced).
#[allow(clippy::too_many_arguments)] // bevy system params
fn event_cues(
    mut mixer: ResMut<Mixer>,
    pools: Option<Res<Pools>>,
    groups: Res<Groups>,
    selection: Res<crate::selection::Selection>,
    terrain: Res<crate::terrain::Terrain>,
    mut cues: MessageReader<UiCue>,
    outcome: Res<crate::ai::BattleOutcome>,
    time: Res<Time>,
    mut st: Local<CueState>,
) {
    let Some(pools) = pools else { return };
    st.frame = st.frame.wrapping_add(1);
    st.horn_gate -= time.delta_secs();
    st.prev_state.resize(groups.list.len(), 0);

    // UI click feedback: one clip per action kind per frame, straight
    // from the input systems; every click sounds, repeats included.
    // Deployment placements click like a move but never horn.
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
        mixer.play_flat(Bus::Ui, pools.ui_select, UI_SELECT);
    }
    if cue_attack {
        mixer.play_flat(Bus::Ui, pools.ui_attack, UI_ATTACK);
    }
    if cue_move || cue_deploy {
        mixer.play_flat(Bus::Ui, pools.ui_order, UI_ORDER);
    }

    let at = |c: Vec2| Vec3::new(c.x, terrain.height_at(c.x, c.y) + 1.5, c.y);
    let seed = st.frame.wrapping_mul(211);
    // The war horn sounds from the regiments the order went to.
    if (cue_attack || cue_move) && st.horn_gate <= 0.0 {
        let (mut sum, mut n, mut first) = (Vec2::ZERO, 0.0, NO_UNIT);
        for (g, gd) in groups.list.iter().enumerate() {
            if selection.regiments.get(g).copied().unwrap_or(false) && gd.team == 0 && gd.count > 0 {
                sum += gd.centroid;
                n += 1.0;
                first = first.min(g as u32);
            }
        }
        if n > 0.0 {
            play(&mut mixer, &pools.horn_charge, HORN_CHARGE, at(sum / n), seed, 0.0, first);
            st.horn_gate = WAR_HORN_GAP_S;
        }
    }

    // The rout horn from an own regiment that breaks. Its crowd cry plays
    // in regiment_sounds; rally has no cue until a fresh asset exists.
    let mut broke: Option<usize> = None;
    for (g, gd) in groups.list.iter().enumerate() {
        let state = match gd.state {
            RegState::Steady => 0u8,
            RegState::Routing { .. } => 1,
            RegState::Shattered => 2,
        };
        if state >= 1 && st.prev_state[g] == 0 && gd.count > 0 && gd.team == 0 {
            broke = Some(g);
        }
        st.prev_state[g] = state;
    }
    if let Some(g) = broke
        && st.horn_gate <= 0.0
    {
        play(&mut mixer, &[pools.horn_rout], HORN_ROUT, at(groups.list[g].centroid), seed ^ 0x77, 0.0, g as u32);
        st.horn_gate = WAR_HORN_GAP_S;
    }

    // Outcome sting, once.
    if !st.prev_outcome && outcome.0.is_some() {
        let (clip, bank) = match outcome.0 {
            Some(0) => (pools.sting_victory, STING_VICTORY),
            _ => (pools.sting_defeat, STING_DEFEAT),
        };
        mixer.play_flat(Bus::Battle, clip, bank);
        st.prev_outcome = true;
    }
}

/// FL_LOG_AUDIO: once a second, the clips started most often, each with
/// its sound, its median distance from the look point, and the regiments
/// it came from with their state. Finds a clip repeating out of one spot.
fn clip_log(
    mut mixer: ResMut<Mixer>,
    clips: Res<Clips>,
    groups: Res<Groups>,
    time: Res<Time<Real>>,
    mut next: Local<f32>,
) {
    if !crate::mixer::log_enabled() {
        return;
    }
    if time.elapsed_secs() < *next {
        return;
    }
    *next = time.elapsed_secs() + 1.0;
    let starts = std::mem::take(&mut mixer.start_log);
    let cuts = std::mem::take(&mut mixer.cut_log);
    // Per clip: its sound, distances, and starts per regiment.
    type ClipStarts<'a> = (ClipId, &'a str, Vec<f32>, Vec<(u32, u32)>);
    let mut per: Vec<ClipStarts> = Vec::new();
    for r in &starts {
        let i = match per.iter().position(|e| e.0 == r.clip) {
            Some(i) => i,
            None => {
                per.push((r.clip, r.bank, Vec::new(), Vec::new()));
                per.len() - 1
            }
        };
        per[i].2.push(r.dist);
        match per[i].3.iter_mut().find(|u| u.0 == r.unit) {
            Some(u) => u.1 += 1,
            None => per[i].3.push((r.unit, 1)),
        }
    }
    per.sort_by_key(|e| std::cmp::Reverse(e.2.len()));
    let state = |g: u32| -> String {
        let Some(gd) = groups.list.get(g as usize) else { return "no unit".into() };
        let mut s = vec![if gd.team == 0 { "own" } else { "enemy" }, crate::unit_types::kind_name(gd.kind)];
        for (on, name) in [
            (gd.charging, "charging"),
            (gd.crashing, "crashing"),
            (gd.engaged, "engaged"),
            (gd.contact, "contact"),
            (gd.state.is_broken(), "broken"),
            (gd.celebrate > 0, "celebrating"),
        ] {
            if on {
                s.push(name);
            }
        }
        format!("g{g} {}", s.join(" "))
    };
    let lines: Vec<String> = per
        .iter_mut()
        .take(6)
        .map(|(clip, bank, d, units)| {
            d.sort_by(|a, b| a.total_cmp(b));
            units.sort_by_key(|u| std::cmp::Reverse(u.1));
            let from: Vec<String> = units.iter().take(3).map(|(g, n)| format!("{} x{n}", state(*g))).collect();
            format!(
                "{} x{} ({bank}, {:.0} m; {})",
                clips.path(*clip),
                d.len(),
                d[d.len() / 2],
                from.join(", ")
            )
        })
        .collect();
    if !lines.is_empty() {
        crate::mixer::audio_log(time.elapsed_secs_f64(), &format!("clips: {}", lines.join(" | ")));
    }

    // The voices alone: per voice sound, its starts, how many playing
    // voices of it were cut for a stronger sound, its most repeated clip
    // and median distance, and the regiments it came from.
    let mut voices: Vec<String> = Vec::new();
    for bank in VOICE_BANKS {
        let mine: Vec<&crate::mixer::StartRecord> = starts.iter().filter(|r| r.bank == bank).collect();
        if mine.is_empty() {
            continue;
        }
        let cut = cuts.iter().filter(|c| **c == bank).count();
        let mut by_clip: Vec<(ClipId, u32)> = Vec::new();
        let mut by_unit: Vec<(u32, u32)> = Vec::new();
        let mut d: Vec<f32> = Vec::new();
        for r in &mine {
            match by_clip.iter_mut().find(|c| c.0 == r.clip) {
                Some(c) => c.1 += 1,
                None => by_clip.push((r.clip, 1)),
            }
            match by_unit.iter_mut().find(|u| u.0 == r.unit) {
                Some(u) => u.1 += 1,
                None => by_unit.push((r.unit, 1)),
            }
            d.push(r.dist);
        }
        by_clip.sort_by_key(|c| std::cmp::Reverse(c.1));
        by_unit.sort_by_key(|u| std::cmp::Reverse(u.1));
        d.sort_by(|a, b| a.total_cmp(b));
        let from: Vec<String> = by_unit.iter().take(3).map(|(g, n)| format!("{} x{n}", state(*g))).collect();
        voices.push(format!(
            "{bank} {} starts, {cut} cut, top {} x{}, {:.0} m; {}",
            mine.len(),
            clips.path(by_clip[0].0),
            by_clip[0].1,
            d[d.len() / 2],
            from.join(", ")
        ));
    }
    if !voices.is_empty() {
        crate::mixer::audio_log(time.elapsed_secs_f64(), &format!("voices: {}", voices.join(" | ")));
    }
}

/// The human-voice banks, for the FL_LOG_AUDIO voice line.
const VOICE_BANKS: [&str; 10] = [
    "attack grunt",
    "attack scream",
    "victim grunt",
    "battle scream",
    "death scream",
    "charge yell",
    "whoop",
    "rout shout",
    "rout panic",
    "rout crowd",
];
