//! One soldier's tick, stage by stage. `tick_chunk` runs every soldier
//! of a chunk through the stages below in order; `Step` carries what one
//! stage hands the next. The rules live in the stage comments.

use bevy::math::Vec3Swizzles;
use bevy::prelude::*;

use crate::sim::damage::DamageEvent;
use crate::spatial::SpatialGrid;
use crate::terrain::Terrain;

mod body;
mod contact;
mod fight;
mod motion;
mod passing;
mod steer;

use body::*;
use contact::*;
use fight::*;
use motion::*;
use passing::*;
use steer::*;

pub(crate) use body::{
    body_distance, crowd_brake, friction_contact, pose_level, ready_body, sep_ease, turn_level, width_level, widths,
    POSE_READY, SEP_RADIUS, STANCE_EASE, STANCE_FIGHT, STANCE_READY,
};
pub(crate) use contact::KNIFE_EDGE;
pub(crate) use fight::{GOING_SPEED, STEP_BACK_TICKS};
pub(crate) use motion::{ADVANCE_PACE, HOLD_DEADZONE, STEP_PACE, STEP_STOP};
pub(crate) use passing::{squeezing, stuck_count, stuck_trying, RING_AIMS, STUCK_LEFT};
pub(crate) use steer::{shove_speed, LEG_ACCEL, STAND_GRIP, STEP_FACE_SPEED};

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
/// An archer's knife: bits 10 to 14 count down (one every fourth tick)
/// from the last tick an enemy came close to him (`KNIFE_HOLD`), and
/// `SIGHT_KNIFE` is set while the knife is out, by that count or by his
/// regiment's swords-out order. With it out he draws no bow, and the
/// renderer shows the knife and hides the bow (render_units_gpu.rs).
const SIGHT_KNIFE_SHIFT: u16 = 10;
/// His aim is a ring aim (`choose_opening`): kept beside or behind him
/// until he is on it, unlike a wide-look aim, which is dropped once past.
const SIGHT_RING: u16 = 1 << 9;
const SIGHT_KNIFE_COUNT: u16 = 0x1f << SIGHT_KNIFE_SHIFT;
pub(crate) const SIGHT_KNIFE: u16 = 1 << 15;

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
    /// Per regiment: walking through a friend that opened for it.
    pub passing: &'a [bool],
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
    /// Per regiment, its fighting line along `form_face` (NaN none;
    /// orders.rs `fight_line`).
    pub fight_line: &'a [f32],
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
    pub aim: &'a mut Vec2,
    pub imposed: &'a mut Vec2,
    /// He stood last tick (`steer`); read by the scan, then written anew.
    pub stood: &'a mut bool,
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
    pub aim: &'a mut [Vec2],
    pub imposed: &'a mut [Vec2],
    pub stood: &'a mut [bool],
}

/// What a chunk emits beside its rows: landed swings and loosed arrows.
pub(crate) struct ChunkOut<'a> {
    pub events: &'a mut Vec<DamageEvent>,
    pub arrows: &'a mut Vec<crate::arrows::ArrowSpawn>,
}

/// What one stage of a soldier's tick hands the next: who he is and his
/// wish at the top, the rest grouped by the stage that writes it.
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
    /// What the scan measured around him (`scan`, the overlap read in
    /// `yield_to_crowd`).
    scan: Scanned,
    /// The enemy in his reach and the one he remembers (`scan`).
    foe: Foe,
    /// What he sees of his comrades and the way he is going (`look_around`).
    sight: Sight,
    /// The choice of opening and the squeeze (`scan`, `choose_opening`).
    open: Opening,
    /// His regiment's melee and his part in it (`begin`, `drive`,
    /// `decide_to_join`, `swing`).
    fight: Fight,
    /// From his wish to his step (`drive`, `yield_to_crowd`, `try_a_side`,
    /// `steer`, `slide`).
    walk: Walk,
}

/// What the scan measured around him.
struct Scanned {
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
    /// The still men ahead of him (the friction contact's brake): the
    /// neighbours within the spring on the half-plane ahead of his wish,
    /// by their closeness and by how little they moved along his way.
    crowd_still: f32,
    /// The crowd ahead of him on his way: the neighbours within the
    /// spring whose direction is within 60 degrees of his wish, by their
    /// closeness (`try_a_side`).
    ahead: f32,
    /// How deep a body is in his on each of his four sides about his wish
    /// (ahead, behind, left, right; 0 none, 1 in full; `press_shares`, the
    /// crowd walk in `steer`).
    press4: [f32; 4],
    /// The crowd at which his drive starts to fade and is gone, for his
    /// width (`Width`).
    brake: (f32, f32),
    scan_r: f32,
    my_team_bit: u32,
}
impl Scanned {
    /// Nothing measured yet.
    const NONE: Scanned = Scanned {
        push: Vec2::ZERO,
        corr: Vec2::ZERO,
        overlap_dir: Vec2::ZERO,
        overlap_depth: 0.0,
        overlap_vel: Vec2::ZERO,
        overlap_share: 0.0,
        crowd: 0.0,
        crowd_still: 0.0,
        ahead: 0.0,
        press4: [0.0; 4],
        brake: (0.0, 0.0),
        scan_r: 0.0,
        my_team_bit: 0,
    };
}

/// The enemy in his reach and the one he remembers.
struct Foe {
    best_idx: u32,
    sticky: bool,
    prev_target: u32,
    memo_valid: bool,
    memo_d2: f32,
}
impl Foe {
    /// Nobody in reach and nobody remembered.
    const NONE: Foe =
        Foe { best_idx: u32::MAX, sticky: false, prev_target: u32::MAX, memo_valid: false, memo_d2: f32::MAX };
}

/// What he remembers and sees.
struct Sight {
    memo_dir: Vec2,
    side_dir: Vec2,
    watch_go: bool,
    way_blocked: bool,
    left_blocked: bool,
    right_blocked: bool,
    comrade_ahead: bool,
    slot_blocked: bool,
    back_blocked: bool,
}
impl Sight {
    /// Nothing seen yet.
    const NONE: Sight = Sight {
        memo_dir: Vec2::ZERO,
        side_dir: Vec2::ZERO,
        watch_go: false,
        way_blocked: false,
        left_blocked: false,
        right_blocked: false,
        comrade_ahead: false,
        slot_blocked: false,
        back_blocked: false,
    };
}

/// The choice of opening and the squeeze (`choose_opening`,
/// `try_a_side`): whether he reads the comrades ahead this tick, his
/// normal body and spring (not squeezed), whether the scan saw an
/// obstacle ahead and the opening it found among them (across and
/// along his way, its fit), a body at each of his sides, and the
/// opening he aims at (its fit: 0 none, 1 his depth, 2 his width;
/// `gap`: it fits his depth only; `gap_near`: that crack within his
/// spring).
struct Opening {
    open_on: bool,
    body0: f32,
    sep0: f32,
    saw: bool,
    near: (f32, f32, u8),
    /// The near opening's air across his way and the far side of its
    /// passage along it (`choose_opening`'s taken crack).
    near_span: (f32, f32, f32),
    between: bool,
    aim_fit: u8,
    gap: bool,
    gap_near: bool,
}
impl Opening {
    /// No opening read.
    const NONE: Opening = Opening {
        open_on: false,
        body0: 0.0,
        sep0: 0.0,
        saw: false,
        near: (0.0, 0.0, 0),
        near_span: (0.0, 0.0, 0.0),
        between: false,
        aim_fit: 0,
        gap: false,
        gap_near: false,
    };
}

/// His regiment's melee and his part in it.
struct Fight {
    in_melee: bool,
    may_leave: bool,
    committed: bool,
    join_fp: Option<Vec2>,
    /// He walks to the fight (`joins_as_walker`): out of formation with no
    /// enemy in his reach, his wish from `drive` is the way to the enemy
    /// he remembers, to his regiment's fighting line or to its fight, not
    /// to his slot (`join_goal`).
    joining: bool,
    /// He left his place to join the fight on this tick (`decide_to_join`):
    /// he sets off as a walker, so his stood bit is cleared (`steer`) and he
    /// walks up to the men ahead before he stands again (`drive`).
    set_off: bool,
    /// His wish to his slot from `drive`, kept when the join replaced it,
    /// for the tick he comes back into formation (`decide_to_join`).
    slot_wish: Vec2,
    /// Where the swing turns him (`swing`, `face`).
    face_target: Option<Vec2>,
}

/// From his wish to his step.
struct Walk {
    /// His drive before the crowd's brake faded it (`try_a_side`).
    wish: Vec2,
    /// What the crowd's brake has taken off his wish since `yield_to_crowd`,
    /// so that `desired + fade` is his own wish unfaded (`steer`).
    fade: Vec2,
    /// The sidestep of his try at a side (`try_a_side`), part of `desired`;
    /// left out of whether he stands (`steer`).
    side: Vec2,
    /// How far he is from his place (`drive`): his regiment's order or
    /// anchor plus his slot, never a route's waypoint; and whether he is a
    /// holding soldier inside the hold deadzone. Either within STEP_STOP or
    /// parked, he has arrived (`steer`).
    place_dist: f32,
    parked: bool,
    jam: f32,
    /// The press's hold on his body this tick (`steer`): the brake's value
    /// before a charge takes his own stopping away.
    body_jam: f32,
    new_v: Vec2,
    /// Where his step ends on the ground plane (`slide`).
    next: Vec2,
}
impl Walk {
    /// No wish and no step yet, and no place.
    const NONE: Walk = Walk {
        wish: Vec2::ZERO,
        fade: Vec2::ZERO,
        side: Vec2::ZERO,
        place_dist: f32::INFINITY,
        parked: false,
        jam: 0.0,
        body_jam: 0.0,
        new_v: Vec2::ZERO,
        next: Vec2::ZERO,
    };
}

/// Run every soldier of a chunk through his tick.
pub(crate) fn tick_chunk(f: &Field, rows: Rows, out: &mut ChunkOut) {
    // The scan's scratch: the ranks of a man's candidates within its cut.
    let mut cands: Vec<u32> = Vec::new();
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
            aim: &mut rows.aim[j],
            imposed: &mut rows.imposed[j],
            stood: &mut rows.stood[j],
        };
        let mut st = begin(f, &mut s);
        drive(f, &mut s, &mut st);
        scan(f, &mut s, &mut st, out, &mut cands);
        look_around(f, &mut s, &mut st);
        choose_opening(f, &mut s, &mut st);
        hold_the_frame(f, &mut st);
        acquire(f, &mut s, &mut st);
        decide_to_join(f, &mut s, &mut st);
        swing(f, &mut s, &mut st, out);
        yield_to_crowd(f, &mut st);
        try_a_side(f, &mut s, &mut st);
        close_in(f, &mut s, &mut st);
        steer(f, &mut s, &mut st);
        slide(f, s.i, &mut st);
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
    // him and was pulled back to his slot, over and over.
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
        scan: Scanned::NONE,
        foe: Foe::NONE,
        sight: Sight::NONE,
        open: Opening::NONE,
        fight: Fight {
            in_melee,
            may_leave,
            committed,
            join_fp,
            joining: false,
            set_off: false,
            slot_wish: Vec2::ZERO,
            face_target: None,
        },
        walk: Walk::NONE,
    }
}
