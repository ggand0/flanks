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
/// Below this distance overlap is resolved positionally (bodies are
/// ~0.6 m wide).
const HARD_RADIUS: f32 = 0.9;
/// Crowd-density yield: goal drive fades to zero between these two local
/// density values. Scalar density can't cancel out the way opposing push
/// vectors do, so this is what stops a goal-seeking crowd from compressing
/// itself into overlap: packed interior units genuinely stop shoving.
const CROWD_SLOW: f32 = 1.2;
const CROWD_STOP: f32 = 2.5;
const STEER_GAIN: f32 = 3.0;
const MAX_ACCEL: f32 = 50.0;
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
/// Positional overlap resolution: fraction of pair overlap corrected per
/// tick per unit, and the per-tick cap on total correction distance.
/// UNDER-relaxed on purpose: resolving overlap over ~3-4 ticks reads as
/// bodies settling; resolving it in 1-2 reads as twitching.
const CORR_GAIN: f32 = 0.3;
const CORR_MAX: f32 = 0.1;
/// Units slow down proportionally inside this distance to the target.
const ARRIVE_RADIUS: f32 = 35.0;
/// A holding soldier who leaves his mark by more than the hold deadzone
/// walks back at this pace at least, and keeps going until he is within
/// STEP_STOP of it: a real step, never a creep. M2TW's slowest
/// locomotion is its shuffle, 0.75 to 0.9 m/s.
const STEP_PACE: f32 = 0.8;
const STEP_STOP: f32 = 0.35;
/// Below this ground speed a standing formation's soldier keeps his
/// facing while he moves: he steps sideways or back, he does not turn
/// to walk (M2TW's shuffle keeps facing).
const STEP_FACE_SPEED: f32 = 1.05;
/// Going to an enemy he can see: M2TW's ready-stance `advance`
/// (1.06 m/s) for the last few meters or with a comrade close ahead,
/// its `combat_jog` (2.87 m/s) from further out over open ground.
const ADVANCE_PACE: f32 = 1.06;
const COMBAT_JOG_PACE: f32 = 2.87;
const JOG_BEYOND: f32 = 3.0;
/// Each man stops short of the enemy he goes to at his own distance,
/// 1.2 m plus up to this much: ranks are men, not a machine.
const SEEK_STOP_SPREAD: f32 = 0.4;
/// A man blocked by a comrade on the way to his enemy sidesteps toward
/// an open lane in some one-second windows (M2TW's isSideStepping),
/// and waits in the others.
const SIDESTEP_WINDOW: u32 = 30;

/// Share of those windows in which a blocked man sidesteps
/// (FL_SIDESTEP, 0 to 1). A play-testing knob.
fn sidestep_chance() -> f32 {
    static P: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *P.get_or_init(|| crate::util::env_or("FL_SIDESTEP", 0.2))
}
/// A standing man plants his feet: a push must exceed this (m/s^2 of
/// separation push) before it moves him. About a 10 cm squeeze from one
/// neighbor. Stepping and running men, and charge knockback, are
/// unaffected; true overlap is still corrected.
const STAND_GRIP: f32 = 6.0;

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
/// Ground speed toward the fight that reads as "going to it".
const GOING_SPEED: f32 = 1.0;
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
/// by running to the fight; one further out (far look).
const SIGHT_WAY: u8 = 1;
const SIGHT_LEFT: u8 = 1 << 1;
const SIGHT_RIGHT: u8 = 1 << 2;
const SIGHT_AHEAD: u8 = 1 << 3;
const SIGHT_MARK: u8 = 1 << 4;
const SIGHT_GO: u8 = 1 << 5;
const SIGHT_GO_FAR: u8 = 1 << 6;
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
const SPEAR_LINE_HALF_W: f32 = 0.35;

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
    pub blocks: &'a [Vec<(Vec2, f32, f32)>; 2],
    pub faces_spearwall: [bool; 2],
    pub dt: f32,
    pub tick_seed: u32,
    pub tick: u32,
    pub bounds_min: Vec2,
    pub bounds_max: Vec2,
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
    pub sight: &'a mut u8,
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
    pub sight: &'a mut [u8],
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
    crowd: f32,
    best_idx: u32,
    sticky: bool,
    prev_target: u32,
    impale_idx: u32,
    at_charge_speed: bool,
    scan_r: f32,
    my_team_bit: u32,
    // His regiment's melee and his part in it.
    in_melee: bool,
    committed: bool,
    join_fp: Option<Vec2>,
    // What he remembers and sees.
    memo_valid: bool,
    memo_dir: Vec2,
    side_dir: Vec2,
    watch_go: bool,
    way_blocked: bool,
    left_blocked: bool,
    right_blocked: bool,
    comrade_ahead: bool,
    slot_blocked: bool,
    // The swing and the steering.
    face_target: Option<Vec2>,
    corr_len2: f32,
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
    let Field { pos_prev, kind, group, broken, fight_point, .. } = *f;
    let i = s.i;
    let p = pos_prev[i].xz();
    let my_kind = kind[i] as usize;
    let dying = *s.death_t > 0;
    // Hit flash decays here (set by the serial apply pass).
    *s.flash = s.flash.saturating_sub(1);
    // Corpses play out their death anim: no orders, no combat; they stay as
    // an obstacle until swept.
    if dying && *s.death_t > 1 {
        *s.death_t -= 1;
    }
    let gi = group[i] as usize;
    let routed = broken[gi];
    // In melee: his regiment has a fight. He is either still in formation or
    // out of it (out_form: he left his slot to fight, a state that sticks
    // until the melee ends; M2TW's isInFormation). Fighting takes him out of
    // formation.
    let in_melee = fight_point[gi].is_some() && !routed && !dying;
    if !in_melee {
        *s.out_form = false;
        *s.sight &= !(SIGHT_GO | SIGHT_GO_FAR);
    } else if *s.swing & crate::units::SWING_STATE_MASK
        != crate::units::SWING_READY
        && *s.swing & crate::units::SWING_RANGED == 0
    {
        *s.out_form = true;
    }
    let committed = in_melee && *s.out_form;
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
        crowd: 0.0,
        best_idx: u32::MAX,
        sticky: false,
        prev_target: u32::MAX,
        impale_idx: u32::MAX,
        at_charge_speed: false,
        scan_r: 0.0,
        my_team_bit: 0,
        in_melee,
        committed,
        join_fp,
        memo_valid: false,
        memo_dir: Vec2::ZERO,
        side_dir: Vec2::ZERO,
        watch_go: false,
        way_blocked: false,
        left_blocked: false,
        right_blocked: false,
        comrade_ahead: false,
        slot_blocked: false,
        face_target: None,
        corr_len2: 0.0,
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
    let Field { terrain, speed, team, home, orders, anchors, .. } = *f;
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
        let goal = orders[gi].unwrap_or(anchors[gi]) + home[i];
        let to_goal = goal - p;
        let dist = to_goal.length();
        // Hold deadzone: parked units don't jitter around their slot point.
        // 0.7 m (was 1.5 when homes were jittered spawn offsets): rigid slots
        // sit exactly at the separation rest distance, so a dressed rank is
        // force-free and can afford tight tolerance — with 1.5 the ranks
        // never finished dressing. A soldier already stepping finishes the
        // step (STEP_STOP) instead of stalling at the deadzone's edge.
        let stepping = s.vel.xz().length_squared()
            > (0.5 * STEP_PACE) * (0.5 * STEP_PACE);
        let deadzone = if stepping { STEP_STOP } else { 0.7 };
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
                * (dist / arrive).min(1.0);
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
    let Field { grid, speed, team, yaw_snap, wall, faces_spearwall, tick_seed, .. } = *f;
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
    let mut crowd = 0.0f32;
    let my_team_bit = (team[i] as u32) * crate::spatial::META_TEAM;
    let my_wall = wall[gi] != 0;
    let my_mass = params.mass;
    let reach2 = params.reach * params.reach;
    let prev_target = *s.target;
    let mut best_d2 = f32::MAX;
    let mut best_idx = u32::MAX;
    let mut sticky = false;
    // Moving at charge speed with unspent momentum: braced enemy spears in
    // the path are a collision hazard, and the scan must see out to SPEAR
    // reach, not just mine. A man already run through (flash) or reeling is
    // not re-impaled every tick — a spear is a point, not an aura.
    let spear_reach = TYPES[crate::unit_types::KIND_SPEAR as usize].reach;
    let cs = speed[i] * CHARGE_SPEED_FRAC;
    let at_charge_speed = faces_spearwall[team[i] as usize]
        && !dying
        && *s.flash == 0
        && *s.swing & crate::units::SWING_STAGGERED == 0
        && s.vel.xz().length_squared() > cs * cs;
    let mut impale_idx = u32::MAX;
    let mut impale_d = f32::MAX;
    let scan_r = if at_charge_speed {
        QUERY_RADIUS.max(params.reach).max(spear_reach)
    } else {
        QUERY_RADIUS.max(params.reach)
    };
    grid.for_each_candidate(p, scan_r, |o| {
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
        let sep_r = if !cross && my_wall && (o.meta & crate::spatial::META_WALL) != 0 {
            WALL_SEP_RADIUS
        } else {
            SEP_RADIUS
        };
        if d2 < sep_r * sep_r && d2 > 1e-8 {
            // Mass-weighted: heavies shove lights aside.
            let o_mass = TYPES[crate::spatial::meta_kind(o.meta)].mass;
            let mw = 2.0 * o_mass / (my_mass + o_mass);
            let len = d2.sqrt();
            let w = 1.0 - len / sep_r;
            if len < HARD_RADIUS {
                // Overlap is resolved POSITIONALLY only. A hard force boost
                // here as well made packed crowds oscillate at the
                // acceleration cap, two solvers fighting over the same
                // overlap, every frame.
                corr += d * ((HARD_RADIUS - len) * CORR_GAIN * mw / len);
            }
            push += d * (w * mw / len);
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
        });
    }
    st.push = push;
    st.corr = corr;
    st.crowd = crowd;
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
/// right, ahead, between him and his mark, or running to the fight),
/// remembered in the sight bits until the next look.
#[inline]
fn look_around(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { grid, pos_prev, team, orders, engaged, tick, .. } = *f;
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
    let memo = prev_target as usize;
    let memo_valid = ((best_idx == u32::MAX && !dying && !routed)
        || (in_melee && !committed))
        && memo < pos_prev.len()
        && team[memo] != team[i]
        && pos_prev[memo].xz().distance_squared(p)
            < (seek_radius() + 1.0) * (seek_radius() + 1.0);
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
    // Open lanes to either side of his way.
    let side_dir = Vec2::new(-memo_dir.y, memo_dir.x);
    // His look at the comrades around him on his way (to his enemy, to the
    // fight, or back to his mark), in his own rhythm: a separate pass over
    // the neighbors of the separation scan, run only by men who are going
    // somewhere. He looks for a lane only while he has no enemy in reach, and
    // watches his comrades go only while he sees no enemy of his own.
    let look_lanes = memo_dir != Vec2::ZERO && best_idx == u32::MAX && !dying && !routed;
    let watch_go = in_melee && !committed && !memo_valid;
    let slot_on = slot_dir != Vec2::ZERO;
    if (i as u32).wrapping_add(tick).is_multiple_of(LOOK_TICKS) {
        let mut seen = 0u8;
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
            seen = (way as u8 * SIGHT_WAY)
                | (left as u8 * SIGHT_LEFT)
                | (right as u8 * SIGHT_RIGHT)
                | (ahead_any as u8 * SIGHT_AHEAD)
                | (mark as u8 * SIGHT_MARK)
                | (go as u8 * SIGHT_GO);
        }
        *s.sight = (*s.sight & SIGHT_GO_FAR) | seen;
    }
    let way_blocked = *s.sight & SIGHT_WAY != 0;
    let left_blocked = *s.sight & SIGHT_LEFT != 0;
    let right_blocked = *s.sight & SIGHT_RIGHT != 0;
    let comrade_ahead = *s.sight & SIGHT_AHEAD != 0;
    let slot_blocked = *s.sight & SIGHT_MARK != 0;
    st.memo_valid = memo_valid;
    st.memo_dir = memo_dir;
    st.side_dir = side_dir;
    st.watch_go = watch_go;
    st.way_blocked = way_blocked;
    st.left_blocked = left_blocked;
    st.right_blocked = right_blocked;
    st.comrade_ahead = comrade_ahead;
    st.slot_blocked = slot_blocked;
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
/// (15 m for a fighting regiment, 4 m on the approach), and a man still in
/// formation notices a comrade of his regiment running to the fight within
/// 6 m.
#[inline]
fn acquire(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { grid, press, engaged, tick_seed, .. } = *f;
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
/// backstop. Once out he never dresses on his slot until the melee ends.
#[inline]
fn decide_to_join(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { melee_ticks, tick, .. } = *f;
    let i = s.i;
    let in_melee = st.in_melee;
    let gi = st.gi;
    let memo_valid = st.memo_valid;
    let mut desired = st.desired;
    let committed = st.committed;
    // Leaving formation to fight, in his own time: he reacts to an enemy in
    // sight or a comrade seen running to the fight (a chance per half-second
    // window), or his patience with the fight's noise runs out. Near men go
    // first and the line rolls up outward.
    let mut committed = committed;
    if in_melee && !committed {
        let patience = (JOIN_PATIENCE_MIN
            + (join_patience_max() - JOIN_PATIENCE_MIN)
                * crate::units::hash01((i as u32).wrapping_mul(0x3C6E) ^ 0xF372))
            * 30.0;
        let saw_comrade_go = *s.sight & (SIGHT_GO | SIGHT_GO_FAR) != 0;
        let react = (memo_valid || saw_comrade_go) && {
            let window = (tick.wrapping_add((i as u32).wrapping_mul(11)) / JOIN_WINDOW)
                .wrapping_mul(0x9E37_79B1);
            crate::units::hash01(window ^ (i as u32).wrapping_mul(0x6A09))
                < join_react_chance()
        };
        if react || melee_ticks[gi] as f32 > patience {
            committed = true;
            *s.out_form = true;
        }
    }
    // Out of formation he never dresses on his slot until the melee ends and
    // the regiment re-forms: he fights, goes for an enemy he sees, heads for
    // the fight, or stands where he is when blocked. Inferring this from his
    // speed made a man who slowed down run back to his slot and come out
    // again, over and over.
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
                desired = Vec2::ZERO;
                if let Some(shot) = &shoot_at[gi] {
                    face_target = Some(shot.c);
                    if *s.swing_t == 0 {
                        let h = |k: u32| {
                            crate::units::hash01(
                                tick_seed
                                    ^ (i as u32).wrapping_mul(k).wrapping_add(k),
                            )
                        };
                        // Aim: an actual soldier of the target regiment (M2TW
                        // keeps a per-soldier aim target), the M2TW range-
                        // INDEPENDENT landing scatter, and a lead for the
                        // block's drift over the flight. The footprint-disc
                        // spot stands in only if every member died this tick.
                        let members = &target_members[shot.t];
                        let base = if members.is_empty() {
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
                        let sigma =
                            crate::unit_types::missile::SCATTER_SIGMA * 1.75;
                        let mut aim = base
                            + Vec2::new(
                                (h(0x3B7F) + h(0x45A3) - 1.0) * sigma,
                                (h(0x5DC1) + h(0x6B8D) - 1.0) * sigma,
                            );
                        aim += shot.vel * (aim.distance(p) / 30.0);
                        // Launch ABOVE the body-hit band (arrows.rs tops out
                        // at ground + 1.15): a shaft leaving at head height
                        // is inside its own shooter's hit cylinder at
                        // flight-time zero and kills him.
                        let from =
                            Vec3::new(p.x, pos_prev[i].y + 0.75, p.y);
                        let to = Vec3::new(
                            aim.x,
                            terrain.height_at(aim.x, aim.y) + 0.7,
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
                    // Target gone mid-draw: ease off and reassess shortly.
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
                    // or dead target is a whiff.
                    let jit =
                        0.85 + 0.3 * crate::units::hash01(tick_seed ^ (i as u32));
                    out.events.push(DamageEvent {
                        victim: *s.target,
                        attacker: i as u32,
                        jit,
                        charge: *s.swing & crate::units::SWING_CHARGE != 0,
                        impale: false,
                    });
                    *s.swing = crate::units::SWING_RECOVER;
                    let cjit = 0.75
                        + 0.5
                            * crate::units::hash01(
                                tick_seed ^ (i as u32).wrapping_mul(0x9E37),
                            );
                    *s.swing_t = (params.cooldown_ticks as f32 * cjit) as u8;
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
                    // Attack style for this swing (render variety only): 0 =
                    // stab, 1 = the classic swing. (2 = slash exists in the
                    // shader but benched.) Spears only ever thrust.
                    let style = if my_kind
                        == crate::unit_types::KIND_SPEAR as usize
                    {
                        0
                    } else {
                        (crate::units::hash01(
                            tick_seed ^ (i as u32).wrapping_mul(0x51ED),
                        ) * 2.0) as u8
                    };
                    let style = style << crate::units::SWING_STYLE_SHIFT;
                    *s.swing = if v2 > cs * cs {
                        crate::units::SWING_WINDUP
                            | crate::units::SWING_CHARGE
                            | style
                    } else {
                        crate::units::SWING_WINDUP | style
                    };
                    *s.swing_t = params.windup_ticks;
                } else if my_kind == crate::unit_types::KIND_ARCHER as usize
                    && !routed
                    && *s.ammo > 0
                    && s.vel.xz().length_squared() < 4.0
                    && let Some(shot) = &shoot_at[gi]
                    && p.distance_squared(shot.c)
                        < crate::unit_types::missile::RANGE
                            * crate::unit_types::missile::RANGE
                {
                    // Nock and draw (foot archers shoot standing only; the
                    // walk gate keeps a marching or skirmishing man's bow on
                    // his back). Style bits stay 0: the stab pull-back IS the
                    // string draw.
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

/// What the crowd does to his drive: overlap corrections and pushes are
/// clamped, and in a packed crowd the drive fades out so the mass cannot
/// keep compressing itself.
#[inline]
fn yield_to_crowd(st: &mut Step) {
    let crowd = st.crowd;
    let mut corr = st.corr;
    let mut push = st.push;
    let mut desired = st.desired;
    let mut corr_len2 = corr.length_squared();
    if corr_len2 < 1e-4 {
        // Sub-centimeter corrections are settle noise, not overlap: applying
        // them is pure micro-twitch.
        corr = Vec2::ZERO;
        corr_len2 = 0.0;
    } else if corr_len2 > CORR_MAX * CORR_MAX {
        corr *= CORR_MAX / corr_len2.sqrt();
    }
    let push_len = push.length();
    if push_len > SEP_PUSH_MAX {
        push *= SEP_PUSH_MAX / push_len;
    }
    // Yield in dense crowds: goal drive fades out entirely so the mass can't
    // keep compressing itself; `jam` (0 = free, 1 = packed) also damps the
    // response below.
    let jam = ((crowd - CROWD_SLOW) / (CROWD_STOP - CROWD_SLOW)).clamp(0.0, 1.0);
    desired *= 1.0 - jam;
    st.corr = corr;
    st.corr_len2 = corr_len2;
    st.push = push;
    st.jam = jam;
    st.desired = desired;
}

/// Closing the last metre on the enemy in reach, going to the enemy he
/// remembers through open ground (waiting behind a comrade, sidestepping
/// now and then), or, with nobody to close on, joining his regiment's
/// fight at the jog.
#[inline]
fn close_in(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { pos_prev, speed, team, engaged, hold, tick, .. } = *f;
    let i = s.i;
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    let in_melee = st.in_melee;
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
    // range" — a legitimate closing target regardless of identity.
    let (close_to, memo_close) = if *s.swing & crate::units::SWING_STATE_MASK
        != crate::units::SWING_READY
        && (*s.target as usize) < pos_prev.len()
    {
        (*s.target, false)
    } else if best_idx != u32::MAX {
        (best_idx, false)
    } else if crowd < CROWD_SLOW && (!in_melee || committed) {
        (*s.target, true)
    } else {
        (u32::MAX, false)
    };
    if !dying
        && !routed
        && (close_to as usize) < pos_prev.len()
        && team[close_to as usize] != team[i]
    {
        let to_enemy = pos_prev[close_to as usize].xz() - p;
        let dist = to_enemy.length();
        // Held regiments fight at arm's length only.
        let look = if engaged[gi] { seek_radius() } else { WIDE_ACQUIRE_R };
        let max_close = if hold[gi] { 2.2 } else { look + 1.0 };
        // Going to a seen enemy, each man stops at his own distance; closing
        // the last meter to a man already in reach stays as it was.
        let stop = if memo_close {
            1.2 + SEEK_STOP_SPREAD
                * crate::units::hash01((i as u32).wrapping_mul(0x3C1B) ^ 0x51F7)
        } else {
            1.2
        };
        if dist > stop && dist < max_close {
            let mut urge = ((dist - stop) / 0.8).clamp(0.0, 1.0);
            // The surge toward a REMEMBERED enemy (no one in reach yet) is
            // steering, not combat execution: it yields to the jam like all
            // steering, so the press brakes on genuine body-pack. Ungated
            // this factor is always 1 (the memo gate above already required
            // crowd < CROWD_SLOW, i.e. jam == 0).
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
                    let window = (tick.wrapping_add((i as u32).wrapping_mul(7))
                        / SIDESTEP_WINDOW)
                        .wrapping_mul(0x2545_F491);
                    let roll = crate::units::hash01(window ^ (i as u32).wrapping_mul(0x9E37));
                    let chance = sidestep_chance();
                    if roll < chance {
                        let side = match (left_blocked, right_blocked) {
                            (false, true) => side_dir,
                            (true, false) => -side_dir,
                            (false, false) if roll < 0.5 * chance => side_dir,
                            (false, false) => -side_dir,
                            (true, true) => Vec2::ZERO,
                        };
                        desired += side * STEP_PACE;
                    }
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
                let window = (tick.wrapping_add((i as u32).wrapping_mul(7))
                    / SIDESTEP_WINDOW)
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
            };
        }
    }
    st.desired = desired;
}

/// From drive to velocity: the wall and charge paces, fatigue, the
/// stagger, the crowd's damping, a standing man's grip on the ground, the
/// acceleration and speed caps, and the overlap correction's kill of the
/// velocity still driving into it.
#[inline]
fn steer(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { speed, wall, charging, fat_speed, fat_nocharge, dt, .. } = *f;
    let i = s.i;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    let push = st.push;
    let jam = st.jam;
    let corr = st.corr;
    let corr_len2 = st.corr_len2;
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
    // sliding, and a fight's jostle rippled back through packed ranks.
    if desired.length_squared() < 1e-6 {
        let pl = push_a.length();
        push_a *= if pl > STAND_GRIP { (pl - STAND_GRIP) / pl } else { 0.0 };
    }
    let mut accel = (desired - v) * STEER_GAIN + push_a;
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
    // After a positional correction, kill the velocity component still
    // driving into the overlap or it re-penetrates next tick.
    if corr_len2 > 1e-12 {
        let cn = corr.normalize_or_zero();
        let into = new_v.dot(-cn);
        if into > 0.0 {
            new_v += cn * into;
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

