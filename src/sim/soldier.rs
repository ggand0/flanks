//! One soldier's tick, stage by stage. `tick_chunk` runs every soldier
//! of a chunk through the stages below in order; `Step` carries what one
//! stage hands the next. The rules live in the stage comments.

use bevy::math::Vec3Swizzles;
use bevy::prelude::*;

use crate::sim::damage::DamageEvent;
use crate::spatial::SpatialGrid;
use crate::terrain::Terrain;
use crate::unit_types::TYPES;

pub(crate) const SEP_RADIUS: f32 = 1.4;
const SEP_STRENGTH: f32 = 60.0;
/// Base neighbor query radius: covers separation and sword reach. Kinds
/// with longer reach (spears) widen their own scan per unit — the sword
/// majority must not pay for the spear's reach.
const QUERY_RADIUS: f32 = 2.0;
/// Cap on summed separation push to avoid explosive forces deep in a crowd.
const SEP_PUSH_MAX: f32 = 2.5;
/// Below this distance overlap is resolved positionally, for a man in
/// his ready stance (at ease `body_ease`). A soldier is 0.75 to 1.1 m
/// across with his arms, shield and weapon. Never more than a pair's own
/// rest distance, so the men of a wall keep their 1.05 m files.
/// FL_BODY=<metres> sets it, up to the separation distance.
const BODY_DISTANCE: f32 = 1.2;
pub(crate) fn body_distance() -> f32 {
    static D: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *D.get_or_init(|| crate::util::env_or("FL_BODY", BODY_DISTANCE).clamp(0.5, SEP_RADIUS))
}
/// Crowd-density yield: a man's drive fades to nothing as the crowd around
/// him (his neighbours' closeness summed, one each at a touch) rises
/// between the two values of `crowd_brake`. Scalar density can't cancel
/// out the way opposing push vectors do, so this is what stops a
/// goal-seeking crowd from compressing itself into overlap: packed
/// interior units genuinely stop shoving. CROWD_SLOW, six neighbours at
/// 1.12 m, is the crowd above which a pressing man with nobody in reach
/// stops looking for a far enemy (`acquire`).
const CROWD_SLOW: f32 = 1.2;
/// The fade is set by the men's real size, not by the body distance: in
/// his ready stance a man's drive eases from six neighbours at arm's length, BRAKE_FAR of
/// the 1.70 m man, to nothing when six are pressed against him at
/// BRAKE_NEAR, his width with arms and shield. Both scale with the unit
/// scale. A band set from the body distance would be 11 cm of squeeze at
/// a 1.2 m body, and the drive of a whole lane would come back at once
/// whenever a gap opened at its front.
const BRAKE_FAR: f32 = 1.3;
const BRAKE_NEAR: f32 = 0.9;
/// The crowd brake of a man in his ready stance.
pub(crate) fn crowd_brake() -> (f32, f32) {
    let w = &widths()[3];
    (w.slow, w.stop)
}
/// The body distance at ease: a man standing, walking or running with
/// his arms down is narrower than one in his ready stance with weapon
/// and shield up (BODY_DISTANCE). The spring rest at ease stays the same
/// 0.2 m over the body (`Width`). FL_BODY_EASE=<metres> sets it, up to
/// the body in the ready stance.
const BODY_EASE: f32 = 0.9;
pub(crate) fn body_ease() -> f32 {
    static D: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *D.get_or_init(|| crate::util::env_or("FL_BODY_EASE", BODY_EASE).clamp(0.5, body_distance()))
}
/// The spring rest at ease (metres).
pub(crate) fn sep_ease() -> f32 {
    body_ease() + SEP_RADIUS - BODY_DISTANCE
}
/// Ticks a man takes from at ease into his ready stance and back, the
/// change of his width: half a second, the length of the stance clip,
/// so a block that turns to fight widens as a step and not a jolt.
pub(crate) const POSE_READY: u8 = 15;
/// A regiment's stance (sim/job.rs `stance_of`), as the pose pass draws
/// its men: at ease; in the ready stance while a man stands (an enemy
/// near, an attack, a charge); in it whatever he does (fighting, a
/// wall).
pub(crate) const STANCE_EASE: u8 = 0;
pub(crate) const STANCE_READY: u8 = 1;
pub(crate) const STANCE_FIGHT: u8 = 2;
/// His pose as the grid carries it (spatial.rs META_POSE): 0 at ease to
/// 3 ready, the ramp in thirds. Both men of a pair read each other's
/// width from it and their own from the same rounding, so the pair's
/// distances are the same from either side.
pub(crate) fn pose_level(pose: u8) -> usize {
    (pose as usize * 3 + POSE_READY as usize / 2) / POSE_READY as usize
}
/// A man's size at one pose level: his body distance and spring rest
/// (a pair's are the mean of the two men's), and the crowd at which his
/// drive starts to fade and is gone, from six men at his body plus
/// 0.1 m to six at his body less 0.2 m at ease, and BRAKE_FAR to
/// BRAKE_NEAR in his ready stance, each against his own spring rest.
pub(crate) struct Width {
    pub body: f32,
    pub sep: f32,
    pub slow: f32,
    pub stop: f32,
}
pub(crate) fn widths() -> &'static [Width; 4] {
    static W: std::sync::OnceLock<[Width; 4]> = std::sync::OnceLock::new();
    W.get_or_init(|| {
        let scale = crate::unit_types::unit_scale();
        std::array::from_fn(|level| {
            let t = level as f32 / 3.0;
            let mix = |ease: f32, ready: f32| ease * (1.0 - t) + ready * t;
            let body = mix(body_ease(), body_distance());
            let sep = mix(sep_ease(), SEP_RADIUS);
            let at = |d: f32| 6.0 * (1.0 - (d * scale).min(sep) / sep);
            Width {
                body,
                sep,
                slow: at(mix(body_ease() + 0.1, BRAKE_FAR)),
                stop: at(mix(body_ease() - 0.2, BRAKE_NEAR)),
            }
        })
    })
}
/// The contact between a man and the body he runs into. By default the
/// two share the closing speed by weight (half each between equal men),
/// so a shove travels through a press and a deep mass pushes a line.
/// FL_CONTACT=wall plays the soft wall instead: the runner's legs lose
/// the closing speed over his give and the man hit is handed none of it.
fn share_contact() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| !std::env::var("FL_CONTACT").is_ok_and(|v| v == "wall"))
}
const STEER_GAIN: f32 = 3.0;
const MAX_ACCEL: f32 = 50.0;
/// What a man's legs can do: the steering toward his wish is capped at
/// this acceleration (m/s^2), about what a man in armour with a shield
/// manages from a stand. The gain alone would give 28 m/s^2, three g, at
/// every start and stop, and a lane behind an opening gap would surge
/// and stop as one body. The spring's push and the contact are not legs
/// and keep MAX_ACCEL.
pub(crate) const LEG_ACCEL: f32 = 4.0;
/// Facing turn smoothing (fraction/s of the remaining angle) — the
/// settle behavior for SMALL corrections.
const YAW_RATE: f32 = 10.0;
/// A blooded soldier (one whose combat memo holds a living enemy) keeps
/// facing the fight while that enemy stands within this range, instead
/// of dressing back to the ordered facing between swing cycles. Fresh
/// men still hold the line — this is memory of a fight, not proximity
/// awareness, so rear ambushes keep their first-blood advantage.
const KEEP_FACING_R: f32 = 6.0;
/// Hard angular speed cap (rad/s). The smoothing above is exponential,
/// so before this cap a 180° about-face finished in ~0.2 s — rear-
/// attacked soldiers whipped around between two swings and rear
/// attacks played out as frontal fights. A burdened man in ranks
/// turns ~180° in about a second: pi rad/s. Small angles never hit
/// the cap, so marching wheels and duel tracking feel unchanged.
const TURN_SPEED_MAX: f32 = std::f32::consts::PI;
/// Positional overlap resolution: the share of a pair's overlap each man
/// is moved back per tick, and the per-tick cap on total correction
/// distance. A packed man overlaps up to six bodies and their corrections
/// add up, so the share stays under one sixth: over it, six bodies throw
/// him past the middle of them and he comes back on the next tick, every
/// tick. 90% of an overlap is gone in 7 ticks.
const CORR_GAIN: f32 = 0.15;
const CORR_MAX: f32 = 0.1;
/// Units slow down proportionally inside this distance to the target.
const ARRIVE_RADIUS: f32 = 35.0;
/// How close to his slot a parked man stands without stepping (metres):
/// rigid slots sit exactly at the separation rest distance, so a dressed
/// rank is force-free and can afford a tight tolerance; with 1.5 the
/// ranks never finished dressing. At 0.7 a man brushed by a passer
/// stayed wherever he landed and the next passer shoved him further.
pub(crate) const HOLD_DEADZONE: f32 = 0.3;
/// A holding soldier who leaves his mark by more than the hold deadzone
/// walks back at this pace at least, and keeps going until he is within
/// STEP_STOP of it: a real step, never a creep. M2TW's slowest
/// locomotion is its shuffle, 0.75 to 0.9 m/s.
pub(crate) const STEP_PACE: f32 = 0.8;
pub(crate) const STEP_STOP: f32 = 0.35;
/// Below this ground speed a standing formation's soldier keeps his
/// facing while he moves: he steps sideways or back, he does not turn
/// to walk (M2TW's shuffle keeps facing). The renderer plays a skinned
/// kind's step clips below it.
pub(crate) const STEP_FACE_SPEED: f32 = 1.05;
/// Going to an enemy he can see: M2TW's ready-stance `advance`
/// (1.06 m/s) for the last few meters or with a comrade close ahead,
/// its `combat_jog` (2.87 m/s) from further out over open ground.
pub(crate) const ADVANCE_PACE: f32 = 1.06;
const COMBAT_JOG_PACE: f32 = 2.87;
const JOG_BEYOND: f32 = 3.0;
/// A man blocked by a comrade on the way to his enemy sidesteps toward
/// an open lane in some one-second windows (M2TW's isSideStepping),
/// and waits in the others.
const SIDESTEP_WINDOW: u32 = 30;
/// A man fights at his weapon's length. A blade lands with its far end,
/// so he stands where that end meets his enemy: he closes to this share
/// of his reach, and steps back to it from an enemy who has come inside
/// it when the ground behind him is open.
const FIGHT_SHARE: f32 = 0.9;
/// He lets an enemy this far inside his fighting distance before he
/// steps back, and then steps all the way out to it: a real step, as on
/// the way back to his mark.
const BACK_SLACK: f32 = 0.3;
/// Man-ticks of that step back since the room log last read them
/// (FL_LOG_ROOM).
pub(crate) static STEP_BACK_TICKS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);
/// Share of those windows in which a blocked man sidesteps
/// (FL_SIDESTEP, 0 to 1). A play-testing knob.
fn sidestep_chance() -> f32 {
    static P: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *P.get_or_init(|| crate::util::env_or("FL_SIDESTEP", 0.2))
}
/// A standing man plants his feet: a push must exceed this (m/s^2 of
/// separation push) before it moves him. About a 10 cm squeeze from one
/// neighbor. A man holding his place (`planted` in steer) keeps the grip
/// while he steps back to it, and his feet take the first STAND_GRIP x dt
/// of a shared shove too, each man 0.7 to 1.3 of it: a man brushed by a
/// passer stays in his file instead of being carried off. Running men
/// are unaffected, a charge's shove is far above it, and true overlap is
/// still corrected.
pub(crate) const STAND_GRIP: f32 = 6.0;
/// The soft wall (FL_CONTACT=wall): a man's legs carry him this far into
/// the body he runs into (his own give, 0.7 to 1.3 of it), losing their
/// closing speed on the way, and nothing of his speed is handed to the
/// man he hits: the press and that man's feet take it. A stop on one tick
/// is the twitch (he is set back and walks in again).
const GIVE: f32 = 0.08;
/// A standing man's stance (his lean, arms and shield) takes the first
/// STANCE metres of a net squeeze before he gives ground to it.
const STANCE: f32 = 0.10;
#[inline]
fn give(i: usize) -> f32 {
    GIVE * own_spread(i)
}
/// His own 0.7 to 1.3 of a man-to-man quantity (his give, his grip).
#[inline]
fn own_spread(i: usize) -> f32 {
    0.7 + 0.6 * crate::units::hash01((i as u32).wrapping_mul(0x51ED).wrapping_add(77))
}

/// A man of a regiment in melee who sees no enemy joins the fight when
/// he sees the comrade beside him run toward it: in each half-second
/// window he reacts with this chance (FL_JOIN_REACT overrides), so the
/// line rolls up from the contact outward, a man at a time, as in
/// M2TW when a deep unit hits a wide line. Once going he keeps going.
const JOIN_WINDOW: u32 = 15;
fn join_react_chance() -> f32 {
    static P: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *P.get_or_init(|| crate::util::env_or("FL_JOIN_REACT", 0.2_f32).clamp(0.0, 1.0))
}
/// Ground speed toward the fight that reads as "going to it". A regiment
/// starts a charge only when its centre closes on its target this fast
/// (frontline.rs).
pub(crate) const GOING_SPEED: f32 = 1.0;
/// How far a man notices a comrade of his own regiment running to the
/// fight (on his far look, below).
const JOIN_SEE_R: f32 = 6.0;
/// A man does not re-read the comrades around him 30 times a second.
/// He looks at the comrades beside him and in his way every LOOK_TICKS
/// (about a quarter second, a human reaction time), in his own rhythm,
/// and acts on what he last saw (the `sight` column) in between. His
/// far look (enemies in sight, comrades running to the fight) keeps the
/// acquisition's own 1-in-8 rhythm.
const LOOK_TICKS: u32 = 8;
/// `sight` bits: a comrade blocks his way, his left, his right; a
/// comrade ahead; a comrade between him and his mark; a comrade close
/// by running to the fight; one further out (far look); a man close
/// behind him.
const SIGHT_WAY: u16 = 1;
pub(crate) const SIGHT_LEFT: u16 = 1 << 1;
pub(crate) const SIGHT_RIGHT: u16 = 1 << 2;
const SIGHT_AHEAD: u16 = 1 << 3;
const SIGHT_MARK: u16 = 1 << 4;
const SIGHT_GO: u16 = 1 << 5;
const SIGHT_GO_FAR: u16 = 1 << 6;
const SIGHT_BACK: u16 = 1 << 7;
/// An enemy stands in his weapon's reach (the scan's finding, every
/// tick): the archers' search never aims at him (sim/mod.rs
/// `clear_members`).
pub(crate) const SIGHT_FOE: u16 = 1 << 8;
/// Bit 9 is free.
/// An archer's knife: bits 10 to 14 count down (one every fourth tick)
/// from the last tick an enemy came close to him (`KNIFE_HOLD`), and
/// `SIGHT_KNIFE` is set while the knife is out, by that count or by his
/// regiment's swords-out order. With it out he draws no bow, and the
/// renderer shows the knife and hides the bow (render_units_gpu.rs).
const SIGHT_KNIFE_SHIFT: u16 = 10;
const SIGHT_KNIFE_COUNT: u16 = 0x1f << SIGHT_KNIFE_SHIFT;
pub(crate) const SIGHT_KNIFE: u16 = 1 << 15;
/// The count an enemy close to him sets: 31 steps of four ticks, about
/// 4 s at 30 ticks a second, before he puts the knife away.
const KNIFE_HOLD: u16 = 31;
/// How close an enemy block's edge comes before he takes his knife out:
/// about two seconds of a charge, one bow draw.
pub(crate) const KNIFE_EDGE: f32 = 10.0;
/// Even with nobody going in sight, a man joins once his regiment has
/// been in melee longer than his own patience, spread between these
/// (seconds; FL_JOIN_PATIENCE sets the upper end): the fight's noise and
/// the officers carry further than sight. Far men join late, never not.
const JOIN_PATIENCE_MIN: f32 = 25.0;
fn join_patience_max() -> f32 {
    static D: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *D.get_or_init(|| crate::util::env_or("FL_JOIN_PATIENCE", 50.0_f32).max(JOIN_PATIENCE_MIN))
}

/// Sight radius for a man of a fighting regiment looking for an enemy
/// to go to (FL_SEEK_R, meters). A play-testing knob.
fn seek_radius() -> f32 {
    static R: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *R.get_or_init(|| crate::util::env_or("FL_SEEK_R", 15.0))
}

/// How near an enemy of any regiment must be, coming at a man whose
/// regiment fights one ordered target (sim/job.rs `focus`), before he
/// turns to fight him (FL_THREAT_R, meters). Beyond it he goes after the
/// target's men only. A play-testing knob: M2TW's seek radius is not
/// known (devlog 0120). At the combat jog a man covers 4 m in 1.4 s, a
/// quarter second of which is his next look and a second a full turn.
fn threat_radius() -> f32 {
    static R: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *R.get_or_init(|| crate::util::env_or("FL_THREAT_R", 4.0))
}

/// Routing units flee at this fraction of their speed: fleeing at
/// exactly max speed made pursuit a zero-kill treadmill.
const ROUT_FLEE_FRAC: f32 = 0.9;
/// Sparse-fight acquisition radius. Two depleted formations interleave
/// with gaps wider than QUERY_RADIUS and the fight stalls (hits/tick 0
/// with both sides standing at "the front"). Pressing units whose normal
/// scan finds no enemy look this far on a staggered 1-in-8-tick cadence
/// and lock a closing target; swings still require reach.
const WIDE_ACQUIRE_R: f32 = 4.0;
/// Speed fraction above which a newly started swing counts as a charge.
/// The momentum bonus (attacker's charge_bonus attack points) resolves in
/// the damage pass — a braced SPEARWALL nullifies it (points beat momentum).
const CHARGE_SPEED_FRAC: f32 = 0.6;
/// A regiment in its charge phase runs the last stretch home.
const CHARGE_SPEED_BOOST: f32 = 1.15;
/// Walls advance at a deliberate pace — breaking into a run breaks the wall.
const WALL_SPEED_FRAC: f32 = 0.72;
/// Separation rest distance between two SAME-TEAM units who are both in a
/// wall stance: shoulder to shoulder. Without this the physics pushes a
/// wall back out to normal spacing and the tight slots never happen.
const WALL_SEP_RADIUS: f32 = 1.05;
/// The braced spear is a PHYSICAL hazard, not a stance rule: a
/// spearwall soldier's leveled point covers a line along his facing,
/// from POINT_MIN (get inside it and the spear is useless) out to his
/// reach, this wide. A body crossing that line at charge speed runs
/// onto the point — a damage event from the spearman powered by the
/// CHARGER's own momentum, and a stop (stagger). A charger who slips
/// between the lines (rank edge, dead wielder, hole in the wall)
/// reaches the spearman and swings with his full charge bonus: gaps
/// are real, coverage at 1.05 m wall pitch is ~2/3 of the frontage.
const SPEAR_POINT_MIN: f32 = 0.9;
pub(crate) const SPEAR_LINE_HALF_W: f32 = 0.35;

/// Global damage multiplier. FL_COMBAT_SCALE overrides (fast test battles).
/// What every soldier reads and none writes: the job's shared inputs.
#[derive(Clone, Copy)]
pub(crate) struct Field<'a> {
    pub grid: &'a SpatialGrid,
    pub terrain: &'a Terrain,
    /// Positions at the tick's start.
    pub pos_prev: &'a [Vec3],
    pub speed: &'a [f32],
    pub team: &'a [u8],
    pub kind: &'a [u8],
    pub group: &'a [u32],
    pub home: &'a [Vec2],
    /// Facings at the tick's start, only when a spearwall stands on the field.
    pub yaw_snap: &'a [f32],
    // Per regiment.
    pub orders: &'a [Option<Vec2>],
    pub anchors: &'a [Vec2],
    pub broken: &'a [bool],
    pub press: &'a [bool],
    /// Per regiment: under a Move order (sim/job.rs `moving`).
    pub moving: &'a [bool],
    /// Per regiment: the stance its men are drawn in (STANCE_*).
    pub stance: &'a [u8],
    /// Per regiment: the one enemy regiment its men fight (sim/job.rs
    /// `focus`), None for any.
    pub focus: &'a [Option<u32>],
    pub engaged: &'a [bool],
    pub contact: &'a [bool],
    pub fight_point: &'a [Option<Vec2>],
    pub melee_ticks: &'a [u32],
    pub hold: &'a [bool],
    pub threat: &'a [Vec2],
    pub form_face: &'a [Vec2],
    pub wall: &'a [u8],
    pub charging: &'a [bool],
    pub fat_speed: &'a [f32],
    pub fat_nocharge: &'a [bool],
    pub shoot_at: &'a [Option<crate::sim::job::ShootAt>],
    pub target_members: &'a [Vec<u32>],
    /// Per regiment under search: its clear men, ascending (sim/mod.rs
    /// `clear_members`).
    pub clear_members: &'a [Vec<u32>],
    /// Per archer regiment: the blocks its men search in, nearest first.
    pub search_blocks: &'a [Vec<u32>],
    /// Per regiment: why its men take their knives out (archers only).
    pub sidearm: &'a [crate::sim::job::Sidearm],
    /// The death column at the tick's start: another man's life, read
    /// by the archer who picks him.
    pub death_snap: &'a [u8],
    pub blocks: &'a [Vec<(Vec2, f32, f32)>; 2],
    pub faces_spearwall: [bool; 2],
    pub dt: f32,
    pub tick_seed: u32,
    pub tick: u32,
    pub bounds_min: Vec2,
    pub bounds_max: Vec2,
    /// Per regiment: its route, and the grid (sim/route.rs).
    pub paths: crate::sim::route::KernelRoutes<'a>,
}

/// One soldier's own row: the columns his tick writes.
pub(crate) struct Soldier<'a> {
    pub i: usize,
    /// His new position (the output column).
    pub pos: &'a mut Vec3,
    pub vel: &'a mut Vec3,
    pub yaw: &'a mut f32,
    pub yaw_prev: &'a mut f32,
    pub target: &'a mut u32,
    pub swing: &'a mut u8,
    pub swing_t: &'a mut u8,
    pub flash: &'a mut u8,
    pub death_t: &'a mut u8,
    pub ammo: &'a mut u8,
    pub out_form: &'a mut bool,
    pub sight: &'a mut u16,
    pub way: &'a mut Vec2,
    pub chain: &'a mut u8,
    pub pose: &'a mut u8,
    pub quarry: &'a mut u32,
    pub stuck: &'a mut u16,
}

/// The rows of one chunk of soldiers, in index order.
pub(crate) struct Rows<'a> {
    pub start: usize,
    pub pos: &'a mut [Vec3],
    pub vel: &'a mut [Vec3],
    pub yaw: &'a mut [f32],
    pub yaw_prev: &'a mut [f32],
    pub target: &'a mut [u32],
    pub swing: &'a mut [u8],
    pub swing_t: &'a mut [u8],
    pub flash: &'a mut [u8],
    pub death_t: &'a mut [u8],
    pub ammo: &'a mut [u8],
    pub out_form: &'a mut [bool],
    pub sight: &'a mut [u16],
    pub way: &'a mut [Vec2],
    pub chain: &'a mut [u8],
    pub pose: &'a mut [u8],
    pub quarry: &'a mut [u32],
    pub stuck: &'a mut [u16],
}

/// What a chunk emits beside its rows: landed swings and loosed arrows.
pub(crate) struct ChunkOut<'a> {
    pub events: &'a mut Vec<DamageEvent>,
    pub arrows: &'a mut Vec<crate::arrows::ArrowSpawn>,
}

/// What one stage of a soldier's tick hands the next.
struct Step {
    /// His position at the tick's start, on the ground plane.
    p: Vec2,
    /// His regiment.
    gi: usize,
    my_kind: usize,
    dying: bool,
    routed: bool,
    /// Where he wants to go, as a velocity; stages add to it and scale it.
    desired: Vec2,
    // The scan.
    push: Vec2,
    corr: Vec2,
    /// Which way the overlap correction points, before its deadband and
    /// cap, how deep the net overlap is along it in metres, and the ground
    /// velocity of the bodies he overlaps, weighted by how far each sets
    /// him back.
    overlap_dir: Vec2,
    overlap_depth: f32,
    overlap_vel: Vec2,
    /// His share by weight of a closing speed on them (the shared speed).
    overlap_share: f32,
    crowd: f32,
    /// The crowd ahead of him on his way: the neighbours within the
    /// spring whose direction is within 60 degrees of his wish, by their
    /// closeness (`try_a_side`).
    ahead: f32,
    /// The crowd at which his drive starts to fade and is gone, for his
    /// width (`Width`).
    brake: (f32, f32),
    best_idx: u32,
    sticky: bool,
    prev_target: u32,
    impale_idx: u32,
    at_charge_speed: bool,
    scan_r: f32,
    my_team_bit: u32,
    // His regiment's melee and his part in it.
    in_melee: bool,
    may_leave: bool,
    committed: bool,
    join_fp: Option<Vec2>,
    // What he remembers and sees.
    memo_valid: bool,
    memo_d2: f32,
    memo_dir: Vec2,
    side_dir: Vec2,
    watch_go: bool,
    way_blocked: bool,
    left_blocked: bool,
    right_blocked: bool,
    comrade_ahead: bool,
    slot_blocked: bool,
    back_blocked: bool,
    // The swing and the steering.
    face_target: Option<Vec2>,
    /// His drive before the crowd's brake faded it (`try_a_side`).
    wish: Vec2,
    jam: f32,
    new_v: Vec2,
}

/// Run every soldier of a chunk through his tick.
pub(crate) fn tick_chunk(f: &Field, rows: Rows, out: &mut ChunkOut) {
    for j in 0..rows.pos.len() {
        let mut s = Soldier {
            i: rows.start + j,
            pos: &mut rows.pos[j],
            vel: &mut rows.vel[j],
            yaw: &mut rows.yaw[j],
            yaw_prev: &mut rows.yaw_prev[j],
            target: &mut rows.target[j],
            swing: &mut rows.swing[j],
            swing_t: &mut rows.swing_t[j],
            flash: &mut rows.flash[j],
            death_t: &mut rows.death_t[j],
            ammo: &mut rows.ammo[j],
            out_form: &mut rows.out_form[j],
            sight: &mut rows.sight[j],
            way: &mut rows.way[j],
            chain: &mut rows.chain[j],
            pose: &mut rows.pose[j],
            quarry: &mut rows.quarry[j],
            stuck: &mut rows.stuck[j],
        };
        let mut st = begin(f, &mut s);
        drive(f, &mut s, &mut st);
        scan(f, &mut s, &mut st, out);
        look_around(f, &mut s, &mut st);
        hold_the_frame(f, &mut st);
        acquire(f, &mut s, &mut st);
        decide_to_join(f, &mut s, &mut st);
        swing(f, &mut s, &mut st, out);
        yield_to_crowd(&mut st);
        try_a_side(f, &mut s, &mut st);
        close_in(f, &mut s, &mut st);
        steer(f, &mut s, &mut st);
        face(f, &mut s, &mut st);
        integrate(f, &mut s, &mut st);
    }
}

/// The tick's bookkeeping before any decision: who he is, whether he is
/// dying or broken, the hit flash and the death countdown, and his part
/// in his regiment's melee.
#[inline]
fn begin(f: &Field, s: &mut Soldier) -> Step {
    let Field { pos_prev, kind, group, broken, fight_point, hold, moving, .. } = *f;
    let i = s.i;
    let p = pos_prev[i].xz();
    let my_kind = kind[i] as usize;
    let dying = *s.death_t > 0;
    // Hit flash decays here (set by the serial apply pass). A twitch code
    // (sim/damage.rs TWITCH_TICKS) counts its low nibble down and goes
    // at once.
    *s.flash = if *s.flash >= 16 {
        if *s.flash & 15 > 1 { *s.flash - 1 } else { 0 }
    } else {
        s.flash.saturating_sub(1)
    };
    // Corpses play out their death anim: no orders, no combat; they stay as
    // an obstacle until swept.
    if dying && *s.death_t > 1 {
        *s.death_t -= 1;
    }
    let gi = group[i] as usize;
    let routed = broken[gi];
    // In melee: his regiment has a fight. Out of formation (out_form): he
    // left his slot to fight, his own state, not his regiment's (M2TW's
    // isInFormation). It sticks while he sees an enemy or his regiment has
    // a fight to head for (decide_to_join); a Move order, a hold order, a
    // rout and death bring him back. Fighting takes him out of formation.
    // Keyed on the regiment's melee instead, a man whose regiment had no
    // fight on its clock, with an enemy standing off his flank, went to
    // him and was pulled back to his slot, over and over (devlog 0212).
    let in_melee = fight_point[gi].is_some() && !routed && !dying;
    let may_leave = !routed && !dying && !hold[gi] && !moving[gi];
    if !may_leave {
        *s.out_form = false;
    } else if *s.swing & crate::units::SWING_STATE_MASK
        != crate::units::SWING_READY
        && *s.swing & crate::units::SWING_STATE_MASK
            != crate::units::SWING_DEFEND
        && *s.swing & crate::units::SWING_RANGED == 0
    {
        *s.out_form = true;
    }
    if !in_melee {
        *s.sight &= !(SIGHT_GO | SIGHT_GO_FAR);
    }
    let committed = *s.out_form;
    let join_fp = if in_melee { fight_point[gi] } else { None };
    Step {
        p,
        gi,
        my_kind,
        dying,
        routed,
        desired: Vec2::ZERO,
        push: Vec2::ZERO,
        corr: Vec2::ZERO,
        overlap_dir: Vec2::ZERO,
        overlap_depth: 0.0,
        overlap_vel: Vec2::ZERO,
        overlap_share: 0.0,
        crowd: 0.0,
        ahead: 0.0,
        brake: (0.0, 0.0),
        best_idx: u32::MAX,
        sticky: false,
        prev_target: u32::MAX,
        impale_idx: u32::MAX,
        at_charge_speed: false,
        scan_r: 0.0,
        my_team_bit: 0,
        in_melee,
        may_leave,
        committed,
        join_fp,
        memo_valid: false,
        memo_d2: f32::MAX,
        memo_dir: Vec2::ZERO,
        side_dir: Vec2::ZERO,
        watch_go: false,
        way_blocked: false,
        left_blocked: false,
        right_blocked: false,
        comrade_ahead: false,
        slot_blocked: false,
        back_blocked: false,
        face_target: None,
        wish: Vec2::ZERO,
        jam: 0.0,
        new_v: Vec2::ZERO,
    }
}

/// Where he wants to go on his own account: a broken man flees for his
/// map edge, a formed man walks to his slot under his regiment's order, at
/// a pace set by slope and water, with a deadzone so a parked man does not
/// jitter and a step that finishes once begun.
#[inline]
fn drive(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { terrain, speed, team, home, orders, anchors, paths, .. } = *f;
    let i = s.i;
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    // Units move ONLY under orders. A regiment order is one point rigidly
    // translated by each unit's `home` offset (the block moves; it never
    // converges). No order = hold at the anchor — a standing order: units
    // drift back to their slot at reduced gain inside the block. Enemy
    // contact is pure physics: cross-team separation blocks, crowd yield
    // stops the shove, combat thins the block. The "front line" is where that
    // collision is.
    let mut desired = Vec2::ZERO;
    if !dying && routed {
        // Broken: flee toward the own map edge with a per-unit lateral
        // scatter — slightly SLOWER than formed pursuers (0.9x): fleeing at
        // exactly max speed made pursuit a zero-kill treadmill (gap frozen
        // forever, counts flat for 30-45 s).
        let flee_z: f32 = if team[i] == 0 { -1.0 } else { 1.0 };
        let lat = (crate::units::hash01((i as u32).wrapping_mul(17) + 3) - 0.5)
            * 0.7;
        let dir = Vec2::new(lat, flee_z).normalize_or_zero();
        let slope = terrain.slope_at(p.x, p.y);
        desired = dir
            * speed[i]
            * ROUT_FLEE_FRAC
            * terrain.wade_mult(p.x, p.y)
            / (1.0 + 3.0 * slope * slope);
    } else if !dying {
        let holding = orders[gi].is_none();
        let mut goal = orders[gi].unwrap_or(anchors[gi]) + home[i];
        // On a route, a man whose straight way to his slot is blocked
        // follows the route instead (sim/route.rs); short of the last leg
        // he keeps his pace. He reads the route every LOOK_TICKS, when he
        // reaches the point he chose, or when he has none, and walks on
        // what he chose between reads.
        let mut through = false;
        match (paths.routes.get(gi).and_then(Option::as_ref), paths.nav) {
            (Some(route), Some(nav)) => {
                let frame = orders[gi].unwrap_or(anchors[gi]);
                let way = *s.way;
                let kept = !way.x.is_nan();
                let at_way = kept && way.x.is_finite() && p.distance_squared(way) < 4.0;
                if !kept || at_way || (i as u32).wrapping_add(f.tick).is_multiple_of(LOOK_TICKS) {
                    let (t, th) = crate::sim::route::man_target(nav, terrain, route, frame, p, goal, i);
                    *s.way = if t == goal && !th { Vec2::INFINITY } else if t == goal { Vec2::NEG_INFINITY } else { t };
                    (goal, through) = (t, th);
                } else if way.x.is_infinite() {
                    through = way.x < 0.0;
                } else {
                    (goal, through) = (way, true);
                }
            }
            _ => *s.way = Vec2::NAN,
        }
        let to_goal = goal - p;
        let dist = to_goal.length();
        // Hold deadzone: parked units don't jitter around their slot point
        // (HOLD_DEADZONE). A soldier already stepping finishes the step
        // (STEP_STOP) instead of stalling at the deadzone's edge.
        let stepping = s.vel.xz().length_squared()
            > (0.5 * STEP_PACE) * (0.5 * STEP_PACE);
        let deadzone = if stepping { STEP_STOP } else { HOLD_DEADZONE };
        if !(holding && dist < deadzone) {
            // Slope penalty: steep ground is slow ground. Wading the river is
            // slow too (the bridge deck is dry: full speed).
            let slope = terrain.slope_at(p.x, p.y);
            let slope_mult = 1.0 / (1.0 + 3.0 * slope * slope);
            let (gain, arrive) = if holding {
                (0.6, 10.0)
            } else {
                (1.0, ARRIVE_RADIUS)
            };
            let mut desired_speed = speed[i]
                * slope_mult
                * terrain.wade_mult(p.x, p.y)
                * gain
                * if through { 1.0 } else { (dist / arrive).min(1.0) };
            if holding {
                desired_speed = desired_speed
                    .max(STEP_PACE * slope_mult * terrain.wade_mult(p.x, p.y));
            }
            if dist > 1e-3 {
                desired = to_goal * (desired_speed / dist);
            }
        }
    }
    st.desired = desired;
}

/// The fused neighbour scan every man runs every tick over the 2 m box:
/// separation pushes and overlap corrections, the crowd around him, the
/// nearest living enemy in his weapon's reach (sticky to the one he already
/// fights), and, at charge speed against a spearwall, the braced point his
/// body runs onto, which is a damage event from the spearman.
#[inline]
fn scan(f: &Field, s: &mut Soldier, st: &mut Step, out: &mut ChunkOut) {
    let Field { grid, speed, team, yaw_snap, wall, stance, faces_spearwall, tick_seed, .. } = *f;
    let i = s.i;
    let params = &TYPES[st.my_kind];
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    // Fused neighbor scan: separation physics + nearest living enemy in reach
    // (swing targeting). Scalar over cache-ordered SortedUnits — an 8-wide
    // SIMD variant measured slower here: candidate runs are too short for
    // lane occupancy.
    let mut push = Vec2::ZERO;
    let mut corr = Vec2::ZERO;
    let mut overlap_vel = Vec2::ZERO;
    let mut overlap_share = 0.0f32;
    let mut overlap = 0.0f32;
    let mut crowd = 0.0f32;
    let mut ahead = 0.0f32;
    let wish_dir = st.desired.normalize_or_zero();
    let my_team_bit = (team[i] as u32) * crate::spatial::META_TEAM;
    let my_wall = wall[gi] != 0;
    let my_mass = params.mass;
    // His width and each neighbour's by their poses at the tick's start.
    let widths = widths();
    let mine = &widths[pose_level(*s.pose)];
    let gain = CORR_GAIN;
    let reach2 = params.reach * params.reach;
    let prev_target = *s.target;
    let mut best_d2 = f32::MAX;
    let mut best_idx = u32::MAX;
    let mut sticky = false;
    // Moving at charge speed with unspent momentum: braced enemy spears in
    // the path are a collision hazard, and the scan must see out to SPEAR
    // reach, not just mine. A man already run through (flash; a twitch
    // code of 16 up is a render cue, not a wound) or reeling is not
    // re-impaled every tick: a spear is a point, not an aura.
    let spear_reach = TYPES[crate::unit_types::KIND_SPEAR as usize].reach;
    let cs = speed[i] * CHARGE_SPEED_FRAC;
    let at_charge_speed = faces_spearwall[team[i] as usize]
        && !dying
        && (*s.flash == 0 || *s.flash >= 16)
        && *s.swing & crate::units::SWING_STAGGERED == 0
        && s.vel.xz().length_squared() > cs * cs;
    let mut impale_idx = u32::MAX;
    let mut impale_d = f32::MAX;
    let scan_r = if at_charge_speed {
        QUERY_RADIUS.max(params.reach).max(spear_reach)
    } else {
        QUERY_RADIUS.max(params.reach)
    };
    grid.for_each_candidate_vel(p, scan_r, |o, o_vel| {
        if o.idx as usize == i {
            return;
        }
        let d = p - o.xz();
        let d2 = d.length_squared();
        let enemy = (o.meta & crate::spatial::META_TEAM) != my_team_bit
            && (o.meta & crate::spatial::META_DYING) == 0;
        if enemy && d2 < reach2 {
            sticky |= o.idx == prev_target;
            if d2 < best_d2 {
                best_d2 = d2;
                best_idx = o.idx;
            }
        }
        // Spear-line collision: he is a braced enemy spearman, and my body is
        // crossing his leveled point while I close at speed.
        if at_charge_speed
            && enemy
            && (o.meta & crate::spatial::META_WALL) != 0
            && crate::spatial::meta_kind(o.meta)
                == crate::unit_types::KIND_SPEAR as usize
        {
            let ys = yaw_snap[o.idx as usize];
            let f = Vec2::new(ys.sin(), ys.cos());
            let rel = d; // spearman -> me (d = p - o.xz())
            let fwd_d = rel.dot(f);
            let lat = (rel.x * f.y - rel.y * f.x).abs();
            let closing = s.vel.xz().dot(d) < 0.0;
            if fwd_d > SPEAR_POINT_MIN
                && fwd_d < spear_reach
                && lat < SPEAR_LINE_HALF_W
                && closing
                && fwd_d < impale_d
            {
                impale_d = fwd_d;
                impale_idx = o.idx;
            }
        }
        // Two same-team units both in a wall STANCE rest shoulder to shoulder
        // (symmetric predicate: both sides compute the same radius). Ordinary
        // ranks keep parade spacing even while fighting — the seams between
        // files are what enemy bodies flow into.
        let cross = (o.meta & crate::spatial::META_TEAM) != my_team_bit;
        // No packing rule for fighting regiments: M2TW keeps its formation
        // grid during melee, and observably loosens it. The fighting crowd's
        // spacing is slots plus body collision, nothing else. A
        // shoulder-to-shoulder press rest for fighting pairs sealed the very
        // seams the intermix needs: a pressed front's gaps shrank to about
        // 1.05 m against 0.95 m bodies and symmetric fights collapsed to a
        // two-rank duel line. It must not come back.
        let theirs = &widths[crate::spatial::meta_pose(o.meta)];
        let sep_r = if !cross && my_wall && (o.meta & crate::spatial::META_WALL) != 0 {
            WALL_SEP_RADIUS
        } else {
            0.5 * (mine.sep + theirs.sep)
        };
        if d2 < sep_r * sep_r && d2 > 1e-8 {
            // Mass-weighted: heavies shove lights aside.
            let o_mass = TYPES[crate::spatial::meta_kind(o.meta)].mass;
            let mw = 2.0 * o_mass / (my_mass + o_mass);
            let len = d2.sqrt();
            let w = 1.0 - len / sep_r;
            let hard = (0.5 * (mine.body + theirs.body)).min(sep_r);
            if len < hard {
                // Overlap is resolved POSITIONALLY only. A hard force boost
                // here as well made packed crowds oscillate at the
                // acceleration cap, two solvers fighting over the same
                // overlap, every frame.
                let back = (hard - len) * gain * mw;
                corr += d * (back / len);
                overlap_vel += o_vel * back;
                overlap_share += o_mass / (my_mass + o_mass) * back;
                overlap += back;
            }
            push += d * (w * mw / len);
            ahead += w * (d.dot(wish_dir) < -0.5 * len) as u32 as f32;
            // Jam density: COMPRESSED pairs only — a neighbor resting AT his
            // pair's rest distance contests nothing, so a formed man's
            // slot-keeping is never faded by the settled enemies one stride
            // away, which made a line yield and stay ragged.
            crowd += w;
        }
    });

    // Ran onto a braced spear: the collision is a damage event from the
    // SPEARMAN, resolved with everything else in the serial apply (which also
    // stops the runner). One point, one wound — the nearest line crossed this
    // tick.
    if impale_idx != u32::MAX {
        let jit = 0.85
            + 0.3
                * crate::units::hash01(
                    tick_seed ^ (i as u32).wrapping_mul(0x7A31),
                );
        out.events.push(DamageEvent {
            victim: i as u32,
            attacker: impale_idx,
            jit,
            charge: false,
            impale: true,
            warn: 0,
            chain: 0,
        });
    }
    // His mark for the archers (sim/mod.rs `clear_members`): every tick
    // whether an enemy stands in his reach.
    if !dying {
        if best_idx != u32::MAX {
            *s.sight |= SIGHT_FOE;
        } else {
            *s.sight &= !SIGHT_FOE;
        }
    }
    // His stance for the next tick, as he is drawn: into the ready stance
    // while his regiment fights, while it stands ready and he stands (the
    // pose pass's guard, below the speed of his steps), or with an enemy
    // in his reach; else back at ease, walking or running. A step of the
    // ramp a tick.
    let ready = match stance[gi] {
        STANCE_FIGHT => true,
        STANCE_READY => s.vel.xz().length_squared() < STEP_FACE_SPEED * STEP_FACE_SPEED,
        _ => false,
    };
    *s.pose = if ready || best_idx != u32::MAX {
        (*s.pose + 1).min(POSE_READY)
    } else {
        s.pose.saturating_sub(1)
    };
    st.brake = (mine.slow, mine.stop);
    // An archer's knife: out while his regiment has its swords out, and
    // for a few seconds after an enemy came close to him, one in his
    // reach, an enemy block's edge within KNIFE_EDGE of him, or with his
    // quiver empty an enemy block in his regiment's bow range.
    if !dying && st.my_kind == crate::unit_types::KIND_ARCHER as usize {
        let arm = &f.sidearm[gi];
        let close = best_idx != u32::MAX
            || (*s.ammo == 0 && arm.foe_in_range)
            || arm.blocks.iter().any(|b| b.distance(p) <= KNIFE_EDGE);
        let mut count = (*s.sight & SIGHT_KNIFE_COUNT) >> SIGHT_KNIFE_SHIFT;
        if close {
            count = KNIFE_HOLD;
        } else if count > 0 && f.tick.wrapping_add(i as u32).is_multiple_of(4) {
            count -= 1;
        }
        let out = if arm.swords || count > 0 { SIGHT_KNIFE } else { 0 };
        *s.sight = (*s.sight & !(SIGHT_KNIFE_COUNT | SIGHT_KNIFE)) | count << SIGHT_KNIFE_SHIFT | out;
    }
    st.push = push;
    st.corr = corr;
    if overlap > 0.0 {
        st.overlap_vel = overlap_vel / overlap;
        st.overlap_share = overlap_share / overlap;
    }
    st.crowd = crowd;
    st.ahead = ahead;
    st.best_idx = best_idx;
    st.sticky = sticky;
    st.prev_target = prev_target;
    st.impale_idx = impale_idx;
    st.at_charge_speed = at_charge_speed;
    st.scan_r = scan_r;
    st.my_team_bit = my_team_bit;
}

/// What he remembers and what he sees of his comrades: the enemy he
/// remembers if still in sight, the way he is going, and every eighth tick
/// a look at the men around him (a comrade in his way, to his left or
/// right, ahead, between him and his mark, or running to the fight; with
/// an enemy in reach, a comrade to either side of his way to him and a
/// man close behind him), remembered in the sight bits until the next
/// look.
#[inline]
fn look_around(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { grid, pos_prev, team, group, orders, engaged, focus, tick, .. } = *f;
    let i = s.i;
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    let in_melee = st.in_melee;
    let committed = st.committed;
    let join_fp = st.join_fp;
    let desired = st.desired;
    let prev_target = st.prev_target;
    let best_idx = st.best_idx;
    let scan_r = st.scan_r;
    let my_team_bit = st.my_team_bit;
    // The enemy he remembers, if still in sight: he tracks him while he has
    // no enemy in reach, and seeing him is what makes a man still in
    // formation react. A man fighting someone in reach out of formation needs
    // neither, and skips the lookup.
    // With his regiment focused on one target (sim/job.rs `focus`), a man
    // of another regiment is remembered only within his threat distance:
    // one who is not on him he lets go, and follows his orders.
    let memo = prev_target as usize;
    let mut memo_d2 = f32::MAX;
    let memo_valid = ((best_idx == u32::MAX && !dying && !routed)
        || (in_melee && !committed))
        && memo < pos_prev.len()
        && team[memo] != team[i]
        && {
            let d2 = pos_prev[memo].xz().distance_squared(p);
            memo_d2 = d2;
            d2 < (seek_radius() + 1.0) * (seek_radius() + 1.0)
                && focus[gi].is_none_or(|t| {
                    group[memo] == t || d2 < threat_radius() * threat_radius()
                })
        };
    // Which way he is going: to that enemy, else to his regiment's fight.
    let memo_dir = if memo_valid {
        (pos_prev[memo].xz() - p).normalize_or_zero()
    } else if let Some(fp) = join_fp {
        (fp - p).normalize_or_zero()
    } else {
        Vec2::ZERO
    };
    // The blocking test toward a holding man's own mark.
    let slot_dir = if orders[gi].is_none() && engaged[gi] && !routed {
        desired.normalize_or_zero()
    } else {
        Vec2::ZERO
    };
    // With an enemy in reach his way is to that enemy, and he looks at the
    // room he has to fight him.
    let fighter = best_idx != u32::MAX && !dying && !routed;
    let to_enemy = if fighter { pos_prev[best_idx as usize].xz() - p } else { Vec2::ZERO };
    let fight_dir = to_enemy.normalize_or_zero();
    // Open lanes to either side of his way.
    let side_dir = if fighter {
        Vec2::new(-fight_dir.y, fight_dir.x)
    } else {
        Vec2::new(-memo_dir.y, memo_dir.x)
    };
    // His look at the comrades around him on his way (to his enemy, to the
    // fight, or back to his mark), in his own rhythm: a separate pass over
    // the neighbors of the separation scan, run only by men who are going
    // somewhere. He looks for a lane only while he has no enemy in reach, and
    // watches his comrades go only while he sees no enemy of his own.
    let look_lanes = memo_dir != Vec2::ZERO && best_idx == u32::MAX && !dying && !routed;
    let watch_go = in_melee && !committed && !memo_valid;
    let slot_on = slot_dir != Vec2::ZERO;
    if (i as u32).wrapping_add(tick).is_multiple_of(LOOK_TICKS) {
        let mut seen = 0u16;
        if look_lanes || watch_go || slot_on {
            let (mut way, mut left, mut right, mut ahead_any, mut mark, mut go) =
                (false, false, false, false, false, false);
            // Branch-free: neighbors of both teams interleave in a melee, and
            // branching on each one mispredicts.
            grid.for_each_candidate_vel(p, scan_r, |o, ov| {
                let comrade = (o.idx as usize != i)
                    & ((o.meta & crate::spatial::META_TEAM) == my_team_bit);
                let d = p - o.xz();
                let d2 = d.length_squared();
                let reach = 0.707 * d2.sqrt();
                let near = d2 < SEP_RADIUS * SEP_RADIUS;
                let going = ov.dot(memo_dir);
                // A comrade ahead: he walks up to him instead of jogging, and
                // at arm's length the man blocks his way. One already walking
                // away the same way is neither: men heading for the same
                // fight move together instead of each waiting for the next.
                let away = going > 0.5;
                let ahead = comrade & look_lanes & ((-d).dot(memo_dir) > reach) & !away;
                ahead_any |= ahead;
                way |= ahead & near;
                // A comrade of his own regiment close by, running to the
                // fight: the sight that makes him follow.
                go |= comrade
                    & watch_go
                    & (crate::spatial::meta_group(o.meta) == gi)
                    & (going > GOING_SPEED);
                let lateral = (-d).dot(side_dir);
                let beside = comrade & look_lanes & near;
                left |= beside & (lateral > reach);
                right |= beside & (lateral < -reach);
                mark |= comrade & slot_on & near & ((-d).dot(slot_dir) > reach);
            });
            seen = (way as u16 * SIGHT_WAY)
                | (left as u16 * SIGHT_LEFT)
                | (right as u16 * SIGHT_RIGHT)
                | (ahead_any as u16 * SIGHT_AHEAD)
                | (mark as u16 * SIGHT_MARK)
                | (go as u16 * SIGHT_GO);
        }
        if fighter {
            // The room he has to fight the enemy in reach: a comrade close
            // to either side of his way to the enemy, and any man close
            // behind him.
            let (mut left, mut right, mut back) = (false, false, false);
            grid.for_each_candidate(p, scan_r, |o| {
                let other = o.idx as usize != i;
                let comrade = other & ((o.meta & crate::spatial::META_TEAM) == my_team_bit);
                let to = o.xz() - p;
                let d2 = to.length_squared();
                let len = d2.sqrt();
                let quarter = 0.707 * len;
                let near = d2 < SEP_RADIUS * SEP_RADIUS;
                let along = to.dot(fight_dir);
                let lateral = to.dot(side_dir);
                left |= comrade & near & (lateral > quarter);
                right |= comrade & near & (lateral < -quarter);
                back |= other & near & (along < -quarter);
            });
            seen |= (left as u16 * SIGHT_LEFT)
                | (right as u16 * SIGHT_RIGHT)
                | (back as u16 * SIGHT_BACK);
        }
        *s.sight = (*s.sight
            & (SIGHT_GO_FAR | SIGHT_FOE | SIGHT_KNIFE_COUNT | SIGHT_KNIFE))
            | seen;
    }
    let way_blocked = *s.sight & SIGHT_WAY != 0;
    let left_blocked = *s.sight & SIGHT_LEFT != 0;
    let right_blocked = *s.sight & SIGHT_RIGHT != 0;
    let comrade_ahead = *s.sight & SIGHT_AHEAD != 0;
    let slot_blocked = *s.sight & SIGHT_MARK != 0;
    st.back_blocked = *s.sight & SIGHT_BACK != 0;
    st.memo_valid = memo_valid;
    st.memo_d2 = memo_d2;
    st.memo_dir = memo_dir;
    st.side_dir = side_dir;
    st.watch_go = watch_go;
    st.way_blocked = way_blocked;
    st.left_blocked = left_blocked;
    st.right_blocked = right_blocked;
    st.comrade_ahead = comrade_ahead;
    st.slot_blocked = slot_blocked;
}

/// Stuck ticks before a man tries a side (half a second); how long a try
/// lasts at most (a second) and at least (a real step, not a twitch);
/// how long he remembers the side he tried, in eighths of a second (four
/// seconds).
const STUCK_TICKS: u16 = 15;
const TRY_TICKS: u16 = 30;
const TRY_MIN: u16 = 6;
const SIDE_MEMORY: u16 = 15;
/// The crowd ahead of him (`Step::ahead`) that counts as someone in his
/// way, and the sideways lean of his separation push that picks his
/// side over his memory and his coin.
const AHEAD_MIN: f32 = 0.1;
const LEAN_MIN: f32 = 0.05;
/// A holding man this far from his slot is walking somewhere too, and
/// tries a side when stuck: a man left inside a friend that closed, or
/// carried off by a friend walking through, works his way out instead of
/// standing against the gripping men for good.
const TRY_FROM_SLOT: f32 = 5.0;
/// The `stuck` column: the stuck ticks (bits 0 to 3), the try's ticks
/// left (4 to 8), the side memory in eighths of a second (9 to 12), and
/// the remembered side (13, set for his left).
const STUCK_COUNT: u16 = 0x000f;
const STUCK_TRY_SHIFT: u16 = 4;
const STUCK_TRY: u16 = 0x1f << STUCK_TRY_SHIFT;
const STUCK_MEM_SHIFT: u16 = 9;
const STUCK_MEM: u16 = 0x0f << STUCK_MEM_SHIFT;
pub(crate) const STUCK_LEFT: u16 = 1 << 13;
/// The stuck ticks and the try's ticks left packed in `stuck`.
pub(crate) fn stuck_count(v: u16) -> u16 {
    v & STUCK_COUNT
}
pub(crate) fn stuck_trying(v: u16) -> u16 {
    (v & STUCK_TRY) >> STUCK_TRY_SHIFT
}

/// A man going somewhere who finds himself stuck tries a side. Stuck is
/// felt, not seen: he wants to walk and has moved at under half his wish
/// for STUCK_TICKS with bodies pressed against him ahead (the net push
/// would not do: in a press the men behind cancel the men in front). He
/// tries the side his separation push leans to, the side with more room;
/// when it leans nowhere, the side he tried within the last SIDE_MEMORY,
/// else his own coin: a sidestep at STEP_PACE on top of whatever the
/// crowd leaves of his drive, not faded by the crowd's brake, so a man
/// packed in still shuffles sideways. The step ends as soon as the way
/// ahead of him is clear, after TRY_MIN ticks, and at TRY_TICKS at the
/// latest: he steps into the gap and walks, he does not step through it
/// into the next man (that was the pinball between the columns). Pressed
/// forward and free to slide sideways, he slides into the gap beside the
/// man in his way, and the man behind him does the same a moment later.
/// Only a man walking somewhere, under his regiment's order or back to a
/// slot more than TRY_FROM_SLOT off, with no enemy in mind and his
/// regiment not fighting: men walking to a fight and the rear ranks of a
/// fighting regiment keep their own rules (`close_in`), so the melee is
/// as it was. His state rides the `stuck` column (STUCK_*).
#[inline]
fn try_a_side(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { orders, engaged, anchors, home, tick, .. } = *f;
    let i = s.i;
    let wish = st.wish;
    let want = wish.length();
    let walking = orders[st.gi].is_some()
        || (anchors[st.gi] + home[i]).distance_squared(st.p) > TRY_FROM_SLOT * TRY_FROM_SLOT;
    let going = walking
        && !engaged[st.gi]
        && !st.dying
        && !st.routed
        && !st.in_melee
        && !st.committed
        && st.best_idx == u32::MAX
        && !st.memo_valid
        && st.join_fp.is_none()
        && want > 0.5;
    if !going {
        *s.stuck = 0;
        return;
    }
    let dir = wish / want;
    let left_dir = Vec2::new(-dir.y, dir.x);
    let v = *s.stuck;
    let mut count = stuck_count(v);
    let mut trying = stuck_trying(v);
    let mut mem = (v & STUCK_MEM) >> STUCK_MEM_SHIFT;
    let mut left = v & STUCK_LEFT != 0;
    if trying > 0 {
        trying -= 1;
        // The way ahead clear after the first TRY_MIN ticks: the step is
        // done, he walks.
        if trying <= TRY_TICKS - TRY_MIN && st.ahead < AHEAD_MIN {
            trying = 0;
        }
        let side = if left { left_dir } else { -left_dir };
        st.desired += side * STEP_PACE;
        mem = SIDE_MEMORY;
    } else {
        let pressed = st.ahead > AHEAD_MIN;
        let stuck = pressed && s.vel.xz().dot(dir) < 0.5 * want;
        count = if stuck { (count + 1).min(STUCK_TICKS) } else { count.saturating_sub(1) };
        if count >= STUCK_TICKS {
            let lean = st.push.dot(left_dir);
            left = if lean.abs() > LEAN_MIN {
                lean > 0.0
            } else if mem > 0 {
                left
            } else {
                crate::units::hash01((i as u32).wrapping_mul(0x2545).wrapping_add(tick)) > 0.5
            };
            trying = TRY_TICKS;
            count = 0;
            mem = SIDE_MEMORY;
        } else if mem > 0 && tick.is_multiple_of(8) {
            mem -= 1;
        }
    }
    *s.stuck = count | (trying << STUCK_TRY_SHIFT) | (mem << STUCK_MEM_SHIFT) | (left as u16 * STUCK_LEFT);
}

/// Two things a man in melee does not do for his slot: step back away
/// from his regiment's fight to dress, or press into the comrade between
/// him and his mark.
#[inline]
fn hold_the_frame(f: &Field, st: &mut Step) {
    let Field { contact, form_face, .. } = *f;
    let gi = st.gi;
    let routed = st.routed;
    let slot_blocked = st.slot_blocked;
    let mut desired = st.desired;
    // A regiment fighting on its contact frame never steps backward, away
    // from its fight, to dress its ranks: a man ahead of his mark (bunched up
    // behind a stopped front, or shoved forward) stays; he only steps forward
    // or sideways to it. Without this a charging block walked backward to
    // reopen its ranks the moment it made contact. A regiment struck from
    // behind has its fight at its back, so it keeps stepping back to hold its
    // ground.
    if contact[gi] && !routed && form_face[gi] != Vec2::ZERO {
        let back = desired.dot(form_face[gi]);
        if back < 0.0 {
            desired -= form_face[gi] * back;
        }
    }
    // In melee, holding men step back to their marks only through open
    // ground: shoved off his mark with a comrade between him and it, a man
    // stands where he is until the way clears (the comrade dies, steps up or
    // moves back to his own mark) instead of pressing into the man's back.
    if slot_blocked {
        desired = Vec2::ZERO;
    }
    st.desired = desired;
}

/// The far look, every eighth tick: a pressing man with nobody in reach
/// and open ground around him remembers the nearest enemy within his sight
/// (15 m for a fighting regiment, 4 m on the approach; with his regiment
/// focused on one target, its men, and any man within his threat distance
/// coming at him), and a man still in formation notices a comrade of his
/// regiment running to the fight within 6 m.
#[inline]
fn acquire(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { grid, press, engaged, focus, tick_seed, .. } = *f;
    let i = s.i;
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    let best_idx = st.best_idx;
    let crowd = st.crowd;
    let watch_go = st.watch_go;
    let memo_dir = st.memo_dir;
    let my_team_bit = st.my_team_bit;
    // Sparse-fight acquisition (see WIDE_ACQUIRE_R): a pressing unit with an
    // empty scan and open space around it memoizes a farther enemy in
    // `target` so the closing drive below can restore contact. Gated hard
    // (press + no near enemy + low crowd + 1/8 cadence) to stay off the 200k
    // hot path.
    let acquire_crowd_lim = CROWD_SLOW;
    // A man of a fighting regiment looks as far as he can see (seek_radius)
    // on his far look; on the approach the old short scan stands.
    let far_look = (i as u32).wrapping_add(tick_seed).is_multiple_of(8);
    if far_look {
        *s.sight &= !SIGHT_GO_FAR;
    }
    if best_idx == u32::MAX
        && !dying
        && !routed
        && press[gi]
        && crowd < acquire_crowd_lim
        && far_look
    {
        let look = if engaged[gi] { seek_radius() } else { WIDE_ACQUIRE_R };
        let mut far_d2 = look * look;
        if let Some(t) = focus[gi] {
            // His regiment fights one ordered target (sim/job.rs `focus`):
            // he looks for its men anywhere in his sight, and for any
            // enemy within his threat distance who is coming at him,
            // closing faster than GOING_SPEED. A man of another regiment
            // standing off or going elsewhere is not his business. The
            // same cells as the look below, with each man's velocity.
            let threat2 = threat_radius() * threat_radius();
            grid.for_each_candidate_vel(p, look, |o, ov| {
                let enemy = (o.meta & crate::spatial::META_TEAM) != my_team_bit
                    && (o.meta & crate::spatial::META_DYING) == 0;
                if enemy {
                    let d = p - o.xz();
                    let d2 = d.length_squared();
                    let closing = ov.dot(d);
                    let counts = crate::spatial::meta_group(o.meta) as u32 == t
                        || (d2 < threat2
                            && closing > 0.0
                            && closing * closing > GOING_SPEED * GOING_SPEED * d2);
                    if counts && d2 < far_d2 {
                        far_d2 = d2;
                        *s.target = o.idx;
                    }
                }
            });
        } else {
            grid.for_each_candidate(p, look, |o| {
                let enemy = (o.meta & crate::spatial::META_TEAM) != my_team_bit
                    && (o.meta & crate::spatial::META_DYING) == 0;
                if enemy {
                    let d2 = (p - o.xz()).length_squared();
                    if d2 < far_d2 {
                        far_d2 = d2;
                        *s.target = o.idx;
                    }
                }
            });
        }
        // Further than the neighbor scan, within JOIN_SEE_R: a comrade of his
        // own regiment running to the fight.
        if watch_go
            && grid.any_candidate_vel(p, look.min(JOIN_SEE_R), |o, ov| {
                o.idx as usize != i
                    && crate::spatial::meta_group(o.meta) == gi
                    && (p - o.xz()).length_squared() < JOIN_SEE_R * JOIN_SEE_R
                    && ov.dot(memo_dir) > GOING_SPEED
            })
        {
            *s.sight |= SIGHT_GO_FAR;
        }
    }
}

/// Leaving formation to fight, in his own time: a chance per half-second
/// window while he sees an enemy or a comrade going, and patience as the
/// backstop. Once out he never dresses on his slot while he sees an enemy
/// or his regiment's melee runs.
#[inline]
fn decide_to_join(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { melee_ticks, engaged, tick, .. } = *f;
    let i = s.i;
    let in_melee = st.in_melee;
    let may_leave = st.may_leave;
    let gi = st.gi;
    let memo_valid = st.memo_valid;
    let memo_d2 = st.memo_d2;
    let best_idx = st.best_idx;
    let mut desired = st.desired;
    let committed = st.committed;
    let mut committed = committed;
    // Out of formation with his regiment's melee over, or never begun: he
    // comes back to his slot once he sees no enemy, nobody in reach and no
    // remembered enemy within his closing range (close_in). While his
    // regiment's melee runs he stays out until it ends, heading for the
    // fight when he sees nobody.
    if committed && !in_melee {
        let look = if engaged[gi] { seek_radius() } else { WIDE_ACQUIRE_R };
        let sees = best_idx != u32::MAX
            || (memo_valid && memo_d2 < (look + 1.0) * (look + 1.0));
        if !sees {
            committed = false;
            *s.out_form = false;
        }
    }
    // Leaving formation to fight, in his own time: he reacts to an enemy in
    // sight or a comrade seen running to the fight (a chance per half-second
    // window), or his patience with the fight's noise runs out. Near men go
    // first and the line rolls up outward. With no melee on his regiment's
    // clock (a lone enemy at the flank, a thin contact) he goes only for an
    // enemy within the short look (WIDE_ACQUIRE_R): the men nearest him
    // leave the ranks for him, the rest stay in formation.
    if may_leave && !committed {
        let patience = (JOIN_PATIENCE_MIN
            + (join_patience_max() - JOIN_PATIENCE_MIN)
                * crate::units::hash01((i as u32).wrapping_mul(0x3C6E) ^ 0xF372))
            * 30.0;
        let saw_comrade_go = *s.sight & (SIGHT_GO | SIGHT_GO_FAR) != 0;
        let sees = if in_melee {
            memo_valid || saw_comrade_go
        } else {
            memo_valid && memo_d2 < WIDE_ACQUIRE_R * WIDE_ACQUIRE_R
        };
        let react = sees && {
            let window = (tick.wrapping_add((i as u32).wrapping_mul(11)) / JOIN_WINDOW)
                .wrapping_mul(0x9E37_79B1);
            crate::units::hash01(window ^ (i as u32).wrapping_mul(0x6A09))
                < join_react_chance()
        };
        if react || (in_melee && melee_ticks[gi] as f32 > patience) {
            committed = true;
            *s.out_form = true;
        }
    }
    // Out of formation he never dresses on his slot: he fights, goes for an
    // enemy he sees, heads for the fight, or stands where he is when
    // blocked. Inferring this from his speed made a man who slowed down run
    // back to his slot and come out again, over and over.
    if committed {
        desired = Vec2::ZERO;
    }
    st.committed = committed;
    st.desired = desired;
}

/// The swing state machine: an archer's draw and loose, a wind-up and
/// its strike (a damage event, resolved by the serial apply), the recovery,
/// and a ready man's choice of target in his forward half-plane.
#[inline]
fn swing(f: &Field, s: &mut Soldier, st: &mut Step, out: &mut ChunkOut) {
    let Field { terrain, pos_prev, speed, team, group, broken, shoot_at, target_members, blocks, tick_seed, .. } = *f;
    let i = s.i;
    let params = &TYPES[st.my_kind];
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    let sticky = st.sticky;
    let prev_target = st.prev_target;
    let best_idx = st.best_idx;
    let my_kind = st.my_kind;
    let mut desired = st.desired;
    // Swing state machine. All writes are to this unit's own row; damage goes
    // through the chunk event buffer.
    let mut face_target = None;
    if !dying {
        match *s.swing & crate::units::SWING_STATE_MASK {
            crate::units::SWING_WINDUP
                if *s.swing & crate::units::SWING_RANGED != 0 =>
            {
                // Drawing the bow: feet planted, eyes on the target block.
                // A man whose knife came out lets the draw go.
                desired = Vec2::ZERO;
                let knife = *s.sight & SIGHT_KNIFE != 0;
                if let Some(shot) = shoot_at[gi].as_ref().filter(|_| !knife) {
                    let q = *s.quarry as usize;
                    face_target =
                        Some(if q < pos_prev.len() { pos_prev[q].xz() } else { shot.c });
                    if *s.swing_t == 0 {
                        let h = |k: u32| {
                            crate::units::hash01(
                                tick_seed
                                    ^ (i as u32).wrapping_mul(k).wrapping_add(k),
                            )
                        };
                        // Aim: the man he drew at (every man has his own
                        // aim target) while he lives, else another man of
                        // the block; the footprint-disc spot stands in only
                        // if every member died this tick. Then the range-
                        // INDEPENDENT landing scatter and a lead for the
                        // block's drift over the flight.
                        let members = &target_members[shot.t];
                        let q = *s.quarry as usize;
                        let base = if q < pos_prev.len()
                            && team[q] != team[i]
                            && f.death_snap[q] == 0
                        {
                            pos_prev[q].xz()
                        } else if members.is_empty() {
                            let ang =
                                h(0x1F3B) * std::f32::consts::TAU;
                            let rad = shot.r * 0.85 * h(0x2E5D).sqrt();
                            shot.c + Vec2::new(ang.cos(), ang.sin()) * rad
                        } else {
                            let m = members[(h(0x1F3B)
                                * members.len() as f32)
                                as usize
                                % members.len()]
                                as usize;
                            pos_prev[m].xz()
                        };
                        let sigma = crate::unit_types::missile::AIM_SPREAD;
                        let mut aim = base
                            + Vec2::new(
                                (h(0x3B7F) + h(0x45A3) - 1.0) * sigma,
                                (h(0x5DC1) + h(0x6B8D) - 1.0) * sigma,
                            );
                        aim += shot.vel * (aim.distance(p) / 30.0);
                        // Launch above the body-hit band (arrows.rs): a
                        // shaft leaving at head height is inside its own
                        // shooter's hit cylinder at flight-time zero and
                        // kills him.
                        let over = crate::unit_types::half_height(my_kind)
                            + crate::arrows::LAUNCH_OVER_HEAD;
                        let from = Vec3::new(p.x, pos_prev[i].y + over, p.y);
                        let to = Vec3::new(
                            aim.x,
                            terrain.height_at(aim.x, aim.y)
                                + crate::arrows::AIM_HEIGHT * crate::unit_types::tallest(),
                            aim.y,
                        );
                        out.arrows.push(crate::arrows::ArrowSpawn {
                            pos: from,
                            vel: crate::arrows::solve_launch(
                                from,
                                to,
                                &blocks[team[i] as usize],
                                terrain,
                            ),
                            team: team[i],
                            group: group[i],
                        });
                        *s.ammo = s.ammo.saturating_sub(1);
                        *s.swing = crate::units::SWING_RECOVER;
                        // Reload: the M2TW volley cycle is animation-bound at
                        // ~10 s; the jitter keeps later volleys ragged.
                        *s.swing_t =
                            (crate::unit_types::missile::RELOAD_TICKS as f32
                                * (0.8 + 0.25 * h(0x77F1)))
                                .min(255.0)
                                as u8;
                    } else {
                        *s.swing_t -= 1;
                    }
                } else {
                    // Target gone mid-draw, or the knife out: ease off and
                    // reassess shortly.
                    *s.swing = crate::units::SWING_RECOVER;
                    *s.swing_t = crate::unit_types::missile::CANCEL_TICKS;
                }
            }
            crate::units::SWING_WINDUP => {
                // Feet planted while winding up — EXCEPT against a routing
                // target: the cut-down happens at a run, or the runner is 3 m
                // gone by the strike tick and every blow whiffs (the pursuit
                // treadmill).
                let t = *s.target as usize;
                let target_routed =
                    t < pos_prev.len() && broken[group[t] as usize];
                if !target_routed {
                    desired *= 0.25;
                }
                if t < pos_prev.len() {
                    face_target = Some(pos_prev[t].xz());
                }
                if *s.swing_t == 0 {
                    // Strike lands; validity (still alive, still in reach,
                    // still an enemy) is checked in the apply pass — a dodged
                    // or dead target is a whiff. A cut lands harder.
                    let cut = params.is_cut(*s.swing);
                    let mut jit =
                        0.85 + 0.3 * crate::units::hash01(tick_seed ^ (i as u32));
                    if cut {
                        jit *= params.cut_weight();
                    }
                    out.events.push(DamageEvent {
                        victim: *s.target,
                        attacker: i as u32,
                        jit,
                        charge: *s.swing & crate::units::SWING_CHARGE != 0,
                        impale: false,
                        warn: 0,
                        chain: 0,
                    });
                    *s.swing = crate::units::SWING_RECOVER;
                    let cjit = (1.0 - crate::unit_types::COOLDOWN_JITTER)
                        + 2.0
                            * crate::unit_types::COOLDOWN_JITTER
                            * crate::units::hash01(
                                tick_seed ^ (i as u32).wrapping_mul(0x9E37),
                            );
                    *s.swing_t = (params.cooldown_ticks as f32 * cjit) as u8;
                    if cut {
                        *s.swing_t = (*s.swing_t).max(params.cut_return_ticks);
                    }
                } else {
                    *s.swing_t -= 1;
                }
            }
            crate::units::SWING_DEFEND => {
                // Answering a swing he saw coming (sim/damage.rs): feet
                // planted, eyes on the attacker, the shield or the blade
                // up to meet the blow, then back on guard. The damage
                // apply set this and may end it early: a parry that
                // stopped the blow turns into the counter's wind-up.
                desired *= 0.25;
                let t = *s.target as usize;
                if t < pos_prev.len() {
                    face_target = Some(pos_prev[t].xz());
                }
                // A parry that stopped the blow counters: a stab at
                // the man he parried. In his kind's window he launches
                // it with a chance each tick by his skill, at the
                // window's end for sure.
                let stopped = *s.swing & crate::sim::damage::GUARD_STOPPED != 0;
                let elapsed = crate::sim::damage::PARRY_LEN.saturating_sub(*s.swing_t);
                let (from, by) = crate::sim::damage::COUNTER_WINDOW[st.my_kind];
                let launch = stopped
                    && elapsed >= from
                    && (elapsed >= by
                        || crate::units::hash01(tick_seed ^ (i as u32).wrapping_mul(0x7F4A_7C15))
                            < params.defence_skill * crate::sim::damage::counter_per_tick());
                if launch {
                    *s.swing = crate::units::SWING_WINDUP
                        | (*s.swing & crate::units::SWING_STAGGER_IMMUNE);
                    *s.swing_t = ((params.windup_ticks as f32
                        * crate::sim::damage::counter_windup())
                        .ceil() as u8)
                        .clamp(1, params.windup_ticks);
                    // The man he parried is warned like any target, one
                    // deeper in the exchange: he may answer from the
                    // bind, with less chance each time (sim/damage.rs).
                    out.events.push(DamageEvent {
                        victim: *s.target,
                        attacker: i as u32,
                        jit: 0.0,
                        charge: false,
                        impale: false,
                        warn: *s.swing_t,
                        chain: s.chain.saturating_add(1),
                    });
                } else if *s.swing_t == 0 {
                    *s.swing = crate::units::SWING_READY
                        | (*s.swing & crate::units::SWING_STAGGER_IMMUNE);
                } else {
                    *s.swing_t -= 1;
                }
            }
            crate::units::SWING_RECOVER => {
                if *s.swing_t == 0 {
                    // A stagger that just wore off leaves one free pass
                    // against the next one (anti-stunlock); a plain recovery
                    // carries an unspent pass forward.
                    let immune = if *s.swing
                        & crate::units::SWING_STAGGERED
                        != 0
                    {
                        crate::units::SWING_STAGGER_IMMUNE
                    } else {
                        *s.swing & crate::units::SWING_STAGGER_IMMUNE
                    };
                    *s.swing = crate::units::SWING_READY | immune;
                } else {
                    *s.swing_t -= 1;
                }
            }
            _ => {
                // Ready: pick a target from the scan. Stick with the previous
                // one when still in reach (duels), else nearest. Routing
                // units never start attacks (they still defend nothing —
                // pursuit is free hits).
                let chosen = if sticky { prev_target } else { best_idx };
                // No eyes in the back of his head: a man only opens on a
                // target in his forward half-plane. Being struck tells him
                // where to turn (the flash facing below), but it does NOT let
                // him swing backward over his shoulder — he attacks once he
                // has turned far enough, at the human turn-speed cap. Without
                // this gate every rear-approached victim counter-wound-up on
                // proximity and was face-on before the first blow landed; the
                // rear sector never fired in practice.
                let aware = chosen != u32::MAX && {
                    let t = chosen as usize;
                    t < pos_prev.len() && {
                        let to_t = pos_prev[t].xz() - p;
                        let fwd = Vec2::new(
                            s.yaw.sin(),
                            s.yaw.cos(),
                        );
                        fwd.dot(to_t) >= 0.0
                    }
                };
                if chosen != u32::MAX && !routed && aware {
                    *s.target = chosen;
                    // Arriving at speed = a charging blow: momentum converts
                    // to damage + a bigger lunge (render reads the flag).
                    let v2 = s.vel.xz().length_squared();
                    let cs = speed[i] * CHARGE_SPEED_FRAC;
                    // Attack style for this swing: 0 = stab, 1 = the classic
                    // overhead swing (2 = slash exists in the shader but
                    // benched). A kind with a cut draws 1 as its slow cut;
                    // its 2 is the overhead swing at the stab's pace, drawn
                    // once the overhead is a clip of its own (the code's
                    // swing over the guard turns the blade sideways). Spears
                    // only ever thrust, and a charging man with a cut stabs:
                    // a slow cut at a run carries him into the man before it
                    // lands.
                    let charge = v2 > cs * cs;
                    let style = if my_kind
                        == crate::unit_types::KIND_SPEAR as usize
                        || (charge && params.cut_windup_ticks > 0)
                    {
                        0
                    } else {
                        (crate::units::hash01(
                            tick_seed ^ (i as u32).wrapping_mul(0x51ED),
                        ) * 2.0) as u8
                    };
                    let style = style << crate::units::SWING_STYLE_SHIFT;
                    *s.swing = if charge {
                        crate::units::SWING_WINDUP
                            | crate::units::SWING_CHARGE
                            | style
                    } else {
                        crate::units::SWING_WINDUP | style
                    };
                    *s.swing_t = params.windup_for(*s.swing);
                    // The target is told of the swing now, so that he
                    // can answer it with a block or a parry timed to the
                    // strike (sim/damage.rs).
                    out.events.push(DamageEvent {
                        victim: chosen,
                        attacker: i as u32,
                        jit: 0.0,
                        charge: false,
                        impale: false,
                        warn: *s.swing_t,
                        chain: 0,
                    });
                } else if my_kind == crate::unit_types::KIND_ARCHER as usize
                    && !routed
                    && *s.ammo > 0
                    && *s.sight & SIGHT_KNIFE == 0
                    && s.vel.xz().length_squared() < 4.0
                    && let Some(shot) = &shoot_at[gi]
                    && let Some(q) = pick_quarry(f, s, p, gi, shot)
                {
                    // Nock and draw at his man (foot archers shoot standing
                    // only; the speed test keeps a marching or skirmishing
                    // man's bow on his back). Style bits stay 0: the stab
                    // pull-back IS the string draw.
                    *s.quarry = q;
                    *s.swing = crate::units::SWING_WINDUP
                        | crate::units::SWING_RANGED;
                    *s.swing_t = crate::unit_types::missile::DRAW_TICKS
                        + (crate::units::hash01(
                            tick_seed ^ (i as u32).wrapping_mul(0x2C9F),
                        ) * 20.0) as u8;
                }
            }
        }
        // A reloading archer contacted in melee drops the reload: he defends
        // at knife tempo instead of standing through the 8 s bow cycle.
        if my_kind == crate::unit_types::KIND_ARCHER as usize
            && best_idx != u32::MAX
            && *s.swing & crate::units::SWING_STATE_MASK
                == crate::units::SWING_RECOVER
            && *s.swing_t > params.cooldown_ticks
        {
            *s.swing_t = params.cooldown_ticks;
        }
    }
    st.desired = desired;
    st.face_target = face_target;
}

/// The share of the archers who keep shooting at the same man while he
/// lives and stays in reach; the rest pick afresh for every shot, as men
/// in the chaos of a battle do.
const KEEP_QUARRY_SHARE: f32 = 1.0 / 3.0;
/// An archer without a man to shoot looks for one every this many ticks.
const QUARRY_LOOK_TICKS: u32 = 8;
/// Random members he tries in one look.
const QUARRY_TRIES: u32 = 3;

/// Whether `m` is a man this archer may shoot: an enemy of a block his
/// regiment shoots at, alive, within his reach and, in a search, among
/// his block's clear men (sim/mod.rs `clear_members`).
#[inline]
fn quarry_ok(
    f: &Field,
    p: Vec2,
    my_team: u8,
    reach2: f32,
    shot: &crate::sim::job::ShootAt,
    blocks: &[u32],
    m: u32,
) -> bool {
    let Field { pos_prev, team, group, death_snap, clear_members, .. } = *f;
    let mi = m as usize;
    mi < pos_prev.len()
        && team[mi] != my_team
        && death_snap[mi] == 0
        && blocks.contains(&group[mi])
        && pos_prev[mi].xz().distance_squared(p) <= reach2
        && (!shot.search || clear_members[group[mi] as usize].binary_search(&m).is_ok())
}

/// The man an archer draws at: the one he keeps, when he is one of the
/// men who keep theirs and the man still serves; else a new pick from
/// his regiment's blocks (sim/job.rs `shoot_at`, `search_blocks`): a
/// random living member, or in a search a random clear member, the
/// nearest block first, a few tries a look and a look every few ticks.
/// None: he waits with the bow down.
#[inline]
fn pick_quarry(
    f: &Field,
    s: &Soldier,
    p: Vec2,
    gi: usize,
    shot: &crate::sim::job::ShootAt,
) -> Option<u32> {
    let Field { team, kind, target_members, clear_members, search_blocks, tick, tick_seed, .. } =
        *f;
    let i = s.i;
    let my_team = team[i];
    let reach = crate::unit_types::bow_reach(kind[i]);
    let reach2 = reach * reach;
    let single = [shot.t as u32];
    let blocks: &[u32] = if shot.search && !search_blocks[gi].is_empty() {
        &search_blocks[gi]
    } else {
        &single
    };
    let keeps =
        crate::units::hash01((i as u32).wrapping_mul(0x9E37) ^ 0x1B3D) < KEEP_QUARRY_SHARE;
    if keeps && quarry_ok(f, p, my_team, reach2, shot, blocks, *s.quarry) {
        return Some(*s.quarry);
    }
    if !tick.wrapping_add((i as u32).wrapping_mul(7)).is_multiple_of(QUARRY_LOOK_TICKS) {
        return None;
    }
    for k in 0..QUARRY_TRIES {
        let b = blocks[k as usize % blocks.len()] as usize;
        let list: &[u32] = if shot.search { &clear_members[b] } else { &target_members[b] };
        if list.is_empty() {
            continue;
        }
        let h = crate::units::hash01(
            tick_seed
                ^ (i as u32).wrapping_mul(0x7F4A_7C15).wrapping_add(k.wrapping_mul(0x2545_F491)),
        );
        let m = list[((h * list.len() as f32) as usize).min(list.len() - 1)];
        if quarry_ok(f, p, my_team, reach2, shot, blocks, m) {
            return Some(m);
        }
    }
    None
}

/// What the crowd does to his drive: overlap corrections and pushes are
/// clamped, and in a packed crowd the drive fades out so the mass cannot
/// keep compressing itself.
#[inline]
fn yield_to_crowd(st: &mut Step) {
    let crowd = st.crowd;
    let mut corr = st.corr;
    let mut push = st.push;
    let mut desired = st.desired;
    let corr_len2 = corr.length_squared();
    if corr_len2 > 1e-12 {
        st.overlap_dir = corr / corr_len2.sqrt();
        st.overlap_depth = corr_len2.sqrt() / CORR_GAIN;
    }
    // Every correction is applied, the sub-centimetre ones too: a threshold
    // turns a steady lean into jumps of a centimetre.
    if corr_len2 > CORR_MAX * CORR_MAX {
        corr *= CORR_MAX / corr_len2.sqrt();
    }
    let push_len = push.length();
    if push_len > SEP_PUSH_MAX {
        push *= SEP_PUSH_MAX / push_len;
    }
    // Yield in dense crowds: goal drive fades out entirely so the mass can't
    // keep compressing itself; `jam` (0 = free, 1 = packed) also damps the
    // response below.
    let (slow, stop) = st.brake;
    let jam = ((crowd - slow) / (stop - slow)).clamp(0.0, 1.0);
    st.wish = desired;
    desired *= 1.0 - jam;
    st.corr = corr;
    st.push = push;
    st.jam = jam;
    st.desired = desired;
}

/// A waiting man's sidestep: in some one-second windows he steps toward
/// whichever side is open, at the shuffle's pace, and in the others he
/// stands.
#[inline]
fn sidestep(i: usize, tick: u32, left_blocked: bool, right_blocked: bool, side_dir: Vec2) -> Vec2 {
    let window = (tick.wrapping_add((i as u32).wrapping_mul(7)) / SIDESTEP_WINDOW)
        .wrapping_mul(0x2545_F491);
    let roll = crate::units::hash01(window ^ (i as u32).wrapping_mul(0x9E37));
    let chance = sidestep_chance();
    let side = match (left_blocked, right_blocked) {
        _ if roll >= chance => Vec2::ZERO,
        (false, true) => side_dir,
        (true, false) => -side_dir,
        (false, false) if roll < 0.5 * chance => side_dir,
        (false, false) => -side_dir,
        (true, true) => Vec2::ZERO,
    };
    side * STEP_PACE
}

/// His footwork around the enemy in reach: closing to his weapon's
/// length and stepping back to it from an enemy who has come inside it.
/// With nobody in reach, going to the enemy he
/// remembers through open ground (waiting behind a comrade, sidestepping
/// now and then), or, with nobody to close on, joining his regiment's
/// fight at the jog.
#[inline]
fn close_in(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { pos_prev, speed, team, group, broken, wall, engaged, hold, moving, focus, tick, .. } = *f;
    let i = s.i;
    let params = &TYPES[st.my_kind];
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    let committed = st.committed;
    let memo_valid = st.memo_valid;
    let join_fp = st.join_fp;
    let best_idx = st.best_idx;
    let crowd = st.crowd;
    let jam = st.jam;
    let way_blocked = st.way_blocked;
    let left_blocked = st.left_blocked;
    let right_blocked = st.right_blocked;
    let comrade_ahead = st.comrade_ahead;
    let side_dir = st.side_dir;
    let mut desired = st.desired;
    let mut d_surge = Vec2::ZERO;
    // Fighters close the last meter to swing range. Only active when an enemy
    // is ALREADY in reach — this is combat execution (like the wind-up foot
    // plant), not steering; it bypasses the jam yield on purpose so front
    // lines stay joined instead of settling at the separation standoff just
    // outside sword range. Close toward the LOCKED swing target when there is
    // one — the per-tick nearest enemy flips in a clog and flip-flopping the
    // close direction reads as twitch. Falls back to the far-acquisition memo
    // in `target` when the near scan is empty AND the unit is in open space —
    // in a dense press the memo would let second ranks drive through the jam
    // and compress the crowd into overlap. The memo may be stale after
    // death sweeps reindex, so it is validated as "some enemy within closing
    // range" — a legitimate closing target regardless of identity. Only a
    // man out of formation goes to a remembered enemy: a formed man seeing
    // one leaves formation first (decide_to_join), so the slot never pulls
    // against the chase.
    let (close_to, memo_close) = if *s.swing & crate::units::SWING_STATE_MASK
        != crate::units::SWING_READY
        && (*s.target as usize) < pos_prev.len()
    {
        (*s.target, false)
    } else if best_idx != u32::MAX {
        (best_idx, false)
    } else if crowd < st.brake.0 && committed {
        (*s.target, true)
    } else {
        (u32::MAX, false)
    };
    // A man sent somewhere by a Move order closes on nobody: he walks on,
    // striking only a man in front of him (the swing above). With his
    // regiment focused on one target he closes on a man of another
    // regiment only within his threat distance (look_around).
    if !dying
        && !routed
        && !moving[gi]
        && (close_to as usize) < pos_prev.len()
        && team[close_to as usize] != team[i]
        && focus[gi].is_none_or(|t| {
            group[close_to as usize] == t
                || pos_prev[close_to as usize].xz().distance_squared(p)
                    < threat_radius() * threat_radius()
        })
    {
        let to_enemy = pos_prev[close_to as usize].xz() - p;
        let dist = to_enemy.length();
        // Held regiments fight at arm's length only.
        let look = if engaged[gi] { seek_radius() } else { WIDE_ACQUIRE_R };
        let max_close = if hold[gi] { 2.2 } else { look + 1.0 };
        // He closes on an enemy in reach to his weapon's length. Going to
        // a seen enemy, each man stops at his own distance, between that
        // and just inside his reach.
        let own = crate::units::hash01((i as u32).wrapping_mul(0x3C1B) ^ 0x51F7);
        let fight = FIGHT_SHARE * params.reach;
        let stop = if memo_close { fight + 0.8 * (params.reach - fight) * own } else { fight };
        if dist > stop && dist < max_close {
            let mut urge = ((dist - stop) / 0.8).clamp(0.0, 1.0);
            // The surge toward a REMEMBERED enemy (no one in reach yet) is
            // steering, not combat execution: it yields to the jam like all
            // steering, so the press brakes on genuine body-pack. Ungated
            // this factor is always 1 (the memo gate above already required
            // a crowd under the brake's start, i.e. jam == 0).
            if memo_close {
                urge *= 1.0 - jam;
                // He steps toward a remembered enemy only through open
                // ground. With a comrade in the way he stands and waits for
                // room instead of leaning on the man's back (M2TW's crowded
                // soldier).
                if way_blocked {
                    urge = 0.0;
                    // Waiting; in some windows he sidesteps toward whichever
                    // side is open.
                    desired += sidestep(i, tick, left_blocked, right_blocked, side_dir);
                }
            }
            let pace = if !memo_close {
                speed[i] * 0.35
            } else if dist > JOG_BEYOND && !comrade_ahead {
                COMBAT_JOG_PACE
            } else {
                ADVANCE_PACE
            };
            desired += to_enemy * (pace * urge / dist);
            d_surge = to_enemy * (pace * urge / dist);
        } else if !memo_close
            && committed
            && wall[gi] == 0
            && !st.back_blocked
            && *s.swing & crate::units::SWING_STATE_MASK != crate::units::SWING_WINDUP
            && !broken[group[close_to as usize] as usize]
            && dist > 1e-3
        {
            // The enemy has come inside his weapon's length: out of
            // formation, with open ground behind him and no blow under way,
            // he steps back to it. He lets the man well inside first, and a
            // step once begun goes all the way out. A man in formation holds
            // his place, and nobody backs off a broken enemy.
            let stepping = s.vel.xz().dot(to_enemy) < -0.5 * STEP_PACE * dist;
            let inside = if stepping { fight } else { fight - BACK_SLACK };
            if dist < inside {
                desired -= to_enemy * (STEP_PACE / dist);
                STEP_BACK_TICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    // Joining: no enemy to close on, so he heads for the enemy unit his
    // regiment fights, jogging over open ground, walking with a comrade close
    // ahead, waiting or sidestepping when blocked. He picks a soldier to
    // fight once one is in sight (the acquisition above).
    if committed
        && !memo_valid
        && let Some(fp) = join_fp
        && best_idx == u32::MAX
        && d_surge == Vec2::ZERO
        && *s.swing & crate::units::SWING_STATE_MASK == crate::units::SWING_READY
    {
        let to_fp = fp - p;
        let dist = to_fp.length();
        if dist > 1.0 {
            let dir = to_fp / dist;
            desired = if !way_blocked {
                let pace = if comrade_ahead { ADVANCE_PACE } else { COMBAT_JOG_PACE };
                dir * pace * (1.0 - jam)
            } else {
                sidestep(i, tick, left_blocked, right_blocked, side_dir)
            };
        }
    }
    st.desired = desired;
}

/// From drive to velocity: the wall and charge paces, fatigue, the
/// stagger, the crowd's damping, a standing man's grip on the ground and
/// his stance against a squeeze, the acceleration and speed caps, and the
/// give of his legs at a body he runs into.
#[inline]
fn steer(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { speed, orders, wall, charging, fat_speed, fat_nocharge, dt, .. } = *f;
    let i = s.i;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    // Holding his place: his regiment has no order and he is not broken,
    // dying, in its melee or out of formation for a fight.
    let planted = orders[gi].is_none() && !dying && !routed && !st.in_melee && !st.committed;
    let push = st.push;
    let jam = st.jam;
    let mut desired = st.desired;
    // Formation pace: walls advance deliberately (running breaks a wall), the
    // charge phase runs the last stretch home. Broken/dying already excluded
    // from both states by construction.
    if wall[gi] != 0 {
        desired *= WALL_SPEED_FRAC;
    } else if charging[gi] && !dying && !routed && !fat_nocharge[gi] {
        desired *= CHARGE_SPEED_BOOST;
    }
    desired *= fat_speed[gi];
    // A staggered man reels where the blow left him: no steering, no closing,
    // until the stun runs out. The shove that staggered him still resolves
    // through separation — he is a body, not an actor.
    if *s.swing & crate::units::SWING_STAGGERED != 0 {
        desired = Vec2::ZERO;
    }

    let v = s.vel.xz();
    // Jammed units stop shoving entirely: at full jam the crowd is
    // quasi-static and overlap resolution is purely positional — force-based
    // separation in a wedged mass only produces bang-bang oscillation.
    let mut push_a = push * (SEP_STRENGTH * (1.0 - jam));
    // A standing man plants his feet: small pushes do not move him, real
    // shoves do (less the grip). Without it any squeeze turned straight into
    // sliding, and a fight's jostle rippled back through packed ranks. His
    // stance takes the first STANCE of a net squeeze the same way: he gives
    // ground only to what is beyond it. A man holding his place keeps the
    // grip while he steps back to it.
    let standing = desired.length_squared() < 1e-6;
    if standing || planted {
        let pl = push_a.length();
        push_a *= if pl > STAND_GRIP { (pl - STAND_GRIP) / pl } else { 0.0 };
    }
    if standing && st.overlap_depth > 0.0 && *s.swing & crate::units::SWING_STAGGERED == 0 {
        st.corr *= ((st.overlap_depth - STANCE) / st.overlap_depth).max(0.0);
    }
    let mut legs = (desired - v) * STEER_GAIN;
    let l2 = legs.length_squared();
    if l2 > LEG_ACCEL * LEG_ACCEL {
        legs *= LEG_ACCEL / l2.sqrt();
    }
    let mut accel = legs + push_a;
    let a2 = accel.length_squared();
    if a2 > MAX_ACCEL * MAX_ACCEL {
        accel *= MAX_ACCEL / a2.sqrt();
    }

    let mut new_v = v + accel * dt;
    // Viscous damping in the press: bleeds the spring energy that otherwise
    // ping-pongs between neighbors every tick.
    new_v *= 1.0 - 0.4 * jam;
    let vmax = speed[i] * 1.15; // slight overspeed under crowd pressure
    let v2 = new_v.length_squared();
    if v2 > vmax * vmax {
        new_v *= vmax / v2.sqrt();
    }
    // The contact. Shared speed: a man gives up his share by weight of the
    // closing speed on the bodies he overlaps, measured against their own
    // movement, and each of the two does the same, so a runner and the man
    // he hits move on together and a shove travels through a press. Soft
    // wall (FL_CONTACT=wall): toward the bodies he overlaps a man's legs
    // lose their closing speed over his give, all of it left at the touch,
    // none at the give, never carrying him past it in one tick, and never
    // more than his own speed into them, so a column on the march is not
    // held up; the man he hits is handed nothing and is moved by the
    // setting apart alone, as pressure.
    let dir = st.overlap_dir;
    if dir != Vec2::ZERO && share_contact() {
        let into = (new_v - st.overlap_vel).dot(-dir);
        if into > 0.0 {
            // A man holding his place loses less of it: his feet take the
            // first part of the shove.
            let mut lost = into * st.overlap_share;
            if planted {
                lost = (lost - STAND_GRIP * dt * own_spread(i)).max(0.0);
            }
            new_v += dir * lost;
        }
    } else if dir != Vec2::ZERO {
        let own = new_v.dot(-dir);
        let into = (new_v - st.overlap_vel).dot(-dir);
        let closing = into.min(own);
        if closing > 0.0 {
            let give = give(i);
            let left = (give - st.overlap_depth).max(0.0);
            let allowed = (closing * left / give).min(left / dt);
            new_v += dir * (closing - allowed);
        }
    }
    st.new_v = new_v;
}

/// Which way he looks: the man he swings at, the enemy in reach when he
/// is fighting or just struck, where he is going when out of formation,
/// the foe he remembers close by, else the line's facing, at the human
/// turn rate.
#[inline]
fn face(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { pos_prev, team, threat, form_face, dt, .. } = *f;
    let i = s.i;
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    let committed = st.committed;
    let memo_dir = st.memo_dir;
    let best_idx = st.best_idx;
    let face_target = st.face_target;
    let new_v = st.new_v;
    // Facing priority: locked wind-up target > nearest enemy in reach >
    // movement direction. Fighters keep eyes on the enemy even while the
    // crowd shoves them; only routing/unengaged units face their velocity.
    // yaw_prev snapshots the pre-update angle so the renderer can interpolate
    // (yaw stepped once per tick otherwise — visible facing snaps at high
    // fps).
    *s.yaw_prev = *s.yaw;
    let face_dir = match face_target {
        Some(t) => t - p,
        // A man IN his swing cycle faces the fight (a formed one too — he is
        // the fighting rim), and a man JUST STRUCK turns toward the blow
        // (flash). A man merely NEAR an enemy does not: turning on proximity
        // raced the attacker's wind-up and had every rear-approached victim
        // frontal by first blood — the whole point of facing, gone. So the
        // first hit lands in the back, spins its victim, and THEN he answers.
        // Unformed units (Blob, no facing claim) still turn on proximity.
        None if !routed
            && best_idx != u32::MAX
            && ((*s.swing & crate::units::SWING_STATE_MASK)
                != crate::units::SWING_READY
                || *s.flash > 0
                || form_face[gi] == Vec2::ZERO) =>
        {
            pos_prev[best_idx as usize].xz() - p
        }
        // Out of formation with no enemy in reach: he faces where he is
        // going, the enemy he remembers or the fight he heads for, standing
        // or walking. Without this a blocked joiner fell through to the
        // formed man's rule below and stood facing the line's front with the
        // fight beside him.
        None if !routed && committed && memo_dir != Vec2::ZERO => memo_dir,
        // Blooded and the enemy still close: a man who has traded blows keeps
        // facing the fight while his last foe (combat memo, validated by team
        // and distance like the closing drive) stands within KEEP_FACING_R —
        // no parade dressing with a sword a few strides away. A player reform
        // takes hold once the ground near him clears. Fresh men fall through
        // and hold the ordered line.
        None if !routed
            && new_v.length_squared() < STEP_FACE_SPEED * STEP_FACE_SPEED
            && form_face[gi] != Vec2::ZERO
            && (*s.target as usize) < pos_prev.len()
            && team[*s.target as usize] != team[i]
            && pos_prev[*s.target as usize]
                .xz()
                .distance_squared(p)
                < KEEP_FACING_R * KEEP_FACING_R =>
        {
            pos_prev[*s.target as usize].xz() - p
        }
        // Standing in formation: HOLD the ordered facing. M2TW rule — a
        // formed unit never rotates itself toward a threat (it goes "ready"
        // in place; the render brace pose keys off enemy_near, not yaw);
        // facing is the player's job, and leaving a flank open is supposed to
        // cost.
        None if !routed
            && new_v.length_squared() < STEP_FACE_SPEED * STEP_FACE_SPEED
            && form_face[gi] != Vec2::ZERO =>
        {
            form_face[gi]
        }
        // Standing watch WITHOUT a formation claim (Blob mobs, rallied
        // remnants): face the enemy mass instead of keeping a stale yaw.
        None if !routed
            && new_v.length_squared() < STEP_FACE_SPEED * STEP_FACE_SPEED
            && threat[gi] != Vec2::ZERO =>
        {
            threat[gi]
        }
        None => new_v,
    };
    let min_len2 = if face_target.is_some() || best_idx != u32::MAX {
        1e-4 // enemies can be close; still face them
    } else {
        0.25 // velocity facing ignores micro-drift
    };
    // A staggered man cannot even turn — the stun freezes his facing, so a
    // charge's second blow finds the same back the first one hit.
    if !dying
        && *s.swing & crate::units::SWING_STAGGERED == 0
        && face_dir.length_squared() > min_len2
    {
        let target_yaw = face_dir.x.atan2(face_dir.y);
        let diff = (target_yaw - *s.yaw + std::f32::consts::PI)
            .rem_euclid(std::f32::consts::TAU)
            - std::f32::consts::PI;
        let step = diff * (YAW_RATE * dt).min(1.0);
        *s.yaw += step.clamp(-TURN_SPEED_MAX * dt, TURN_SPEED_MAX * dt);
    }
}

/// The step: velocity into position, clamped to the field, sliding along
/// impassable ground, set down on the terrain.
#[inline]
fn integrate(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { terrain, pos_prev, kind, dt, bounds_min, bounds_max, .. } = *f;
    let i = s.i;
    let new_v = st.new_v;
    let corr = st.corr;
    *s.vel = Vec3::new(new_v.x, 0.0, new_v.y);
    let mut nx = (pos_prev[i].x + new_v.x * dt + corr.x)
        .clamp(bounds_min.x, bounds_max.x);
    let mut nz = (pos_prev[i].z + new_v.y * dt + corr.y)
        .clamp(bounds_min.y, bounds_max.y);
    // Impassable ground (terrace risers, gorge walls, crater lips):
    // wall-slide — keep the axis that stays on walkable ground, drop the one
    // that doesn't, so crowds flow along the obstacle.
    if terrain.blocked_at(nx, nz) {
        let (px, pz) = (pos_prev[i].x, pos_prev[i].z);
        if !terrain.blocked_at(nx, pz) {
            nz = pz;
        } else if !terrain.blocked_at(px, nz) {
            nx = px;
        } else {
            nx = px;
            nz = pz;
        }
    }
    *s.pos = Vec3::new(
        nx,
        terrain.height_at(nx, nz) + crate::unit_types::half_height(kind[i] as usize),
        nz,
    );
}

