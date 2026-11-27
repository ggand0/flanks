//! Where he wants to go: the drive to his place or away in a rout, the
//! walk to the fight, and what the crowd's brake leaves of it.

use bevy::math::Vec3Swizzles;
use bevy::prelude::*;

use super::*;
use crate::unit_types::TYPES;

/// Cap on summed separation push to avoid explosive forces deep in a crowd.
const SEP_PUSH_MAX: f32 = 2.5;
/// Under the friction contact's brake a soldier out of formation with no
/// enemy in his reach walks to the fight like any walker: his wish to go
/// to the enemy he remembers, straight ahead to his regiment's fighting
/// line, or to his regiment's fight is made in `drive` (`join_goal`) in
/// place of the walk to his slot, so the scan reads the men ahead of him
/// on that way, and the still men ahead stop him there (`yield_to_crowd`)
/// as they stop every walker. He does not wait at arm's length behind a
/// comrade and move up in hops. Under the arm's length rule (`keep_arm`)
/// his walk shrinks to nothing as he comes to an arm's length from the
/// comrade in his way, and he moves off any body nearer than that
/// (`keep_arm_length`). Without it, once the men ahead have stopped him
/// behind a comrade in his way he stands, braced, with no wish, until a
/// short step of room opens ahead (`drive`, `steer`). Without that brake
/// he joins through `close_in`.
#[inline]
pub(super) fn joins_as_walker() -> bool {
    still_brake()
}
/// Units slow down proportionally inside this distance to the target.
pub(super) const ARRIVE_RADIUS: f32 = 35.0;

/// The share of his pace a soldier wants at `dist` metres from where he
/// walks to, slowing over the last `arrive` metres. A soldier on a route
/// (`through`) keeps his pace, and so does a soldier whose regiment is in
/// its charge (`charge_run`): he runs on until bodies stop him.
fn arrive_share(dist: f32, arrive: f32, through: bool, charge_run: bool) -> f32 {
    if through || charge_run { 1.0 } else { (dist / arrive).min(1.0) }
}

/// Under the friction contact a soldier of a regiment in its charge does
/// not slow for his place: his regiment's places lie at the enemy's
/// outline, often inside a friend that fights there, and he would slow to
/// a walk short of that friend's back while the charge cry plays.
fn charge_runs_home() -> bool {
    friction_contact()
}
/// Under the friction contact a soldier of a regiment in its charge keeps
/// his whole wish to run in: the soldiers ahead of him do not fade it
/// (`yield_to_crowd`), so the regiment does not halt as one when its first
/// soldiers touch a standing friend. The still men ahead still slow his
/// body as they slow anyone's (`body_jam`, `steer`): he runs until he
/// reaches them and then presses on them, so two charging regiments do not
/// run through each other's ranks.
fn charge_runs_in() -> bool {
    friction_contact()
}
/// How close to his slot a parked man stands without stepping (metres):
/// rigid slots sit exactly at the separation rest distance, so a dressed
/// rank is force-free and can afford a tight tolerance; with 1.5 the
/// ranks never finished dressing. At 0.7 a man brushed by a passer
/// stays wherever he lands and the next passer shoves him further.
pub(crate) const HOLD_DEADZONE: f32 = 0.3;
/// A holding soldier who leaves his mark by more than the hold deadzone
/// walks back at this pace at least, and keeps going until he is within
/// STEP_STOP of it: a real step, never a creep. M2TW's slowest
/// locomotion is its shuffle, 0.75 to 0.9 m/s.
pub(crate) const STEP_PACE: f32 = 0.8;
pub(crate) const STEP_STOP: f32 = 0.35;
/// Going to an enemy he can see: M2TW's ready-stance `advance`
/// (1.06 m/s) for the last few meters or with a comrade close ahead,
/// its `combat_jog` (2.87 m/s) from further out over open ground.
pub(crate) const ADVANCE_PACE: f32 = 1.06;
pub(super) const COMBAT_JOG_PACE: f32 = 2.87;
pub(super) const JOG_BEYOND: f32 = 3.0;

/// Routing units flee at this fraction of their speed: fleeing at
/// exactly max speed made pursuit a zero-kill treadmill.
const ROUT_FLEE_FRAC: f32 = 0.9;

/// Where he wants to go on his own account: a broken man flees for his
/// map edge, a formed man walks to his slot under his regiment's order, at
/// a pace set by slope and water, with a deadzone so a parked man does not
/// jitter and a step that finishes once begun. Under the friction
/// contact's brake a man out of formation with nobody in his reach walks
/// to the fight instead (`joins_as_walker`, `join_goal`).
#[inline]
pub(super) fn drive(f: &Field, s: &mut Soldier, st: &mut Step, asks: &mut Vec<crate::sim::route::WayAsk>) {
    let Field { terrain, speed, home, orders, anchors, .. } = *f;
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
        desired = flee(f, i, p);
    } else if !dying {
        let holding = orders[gi].is_none();
        let goal = orders[gi].unwrap_or(anchors[gi]) + home[i];
        st.walk.place_dist = goal.distance(p);
        let place = goal;
        let (goal, through) = follow_route(f, s, st, goal, asks);
        st.walk.rounding = goal != place;
        let to_goal = goal - p;
        let dist = to_goal.length();
        // Hold deadzone: parked units don't jitter around their slot point
        // (HOLD_DEADZONE). A soldier already stepping finishes the step
        // (STEP_STOP) instead of stalling at the deadzone's edge.
        let stepping = s.vel.xz().length_squared()
            > (0.5 * STEP_PACE) * (0.5 * STEP_PACE);
        let deadzone = if stepping { STEP_STOP } else { HOLD_DEADZONE };
        st.walk.parked = holding && dist < deadzone;
        if !st.walk.parked {
            // Slope penalty: steep ground is slow ground. Wading the river is
            // slow too (the bridge deck is dry: full speed).
            let slope = terrain.slope_at(p.x, p.y);
            let slope_mult = 1.0 / (1.0 + 3.0 * slope * slope);
            let (gain, arrive) = if holding {
                (0.6, 10.0)
            } else {
                (1.0, ARRIVE_RADIUS)
            };
            // In his regiment's charge (as `steer` gives the charge pace;
            // dying and routed men never come here).
            let charge_run = charge_runs_home() && f.charging[gi] && !f.fat_nocharge[gi];
            let mut desired_speed = speed[i]
                * slope_mult
                * terrain.wade_mult(p.x, p.y)
                * gain
                * arrive_share(dist, arrive, through, charge_run);
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
    // Out of formation, with no enemy in his reach last tick and no blow
    // under way, he walks to the fight instead of to his slot
    // (`joins_as_walker`): the scan then reads the men ahead of him on
    // that way, and the brake stops him at the still men ahead like any
    // walker. His slot wish is kept for the tick he comes back into
    // formation.
    if joins_as_walker()
        && !dying
        && !routed
        && st.fight.committed
        && *s.sight & SIGHT_FOE == 0
        && *s.swing & crate::units::SWING_STATE_MASK == crate::units::SWING_READY
    {
        st.fight.slot_wish = desired;
        st.fight.joining = true;
        // Under the arm's length rule (`keep_arm`) he walks to the fight
        // every tick, and how near the comrade in his way stands shrinks
        // that walk after the scan, to nothing at arm's length
        // (`keep_arm_length`). Without it: arrived at the men ahead, a
        // comrade in his way by his last look and himself standing since
        // the still men ahead stopped him (`steer` sets his stood bit
        // then), he stands with no wish until a step of room opens ahead
        // (JOIN_ROOM). His own standing is read, never his speed: in a
        // packed regiment the bodies around him move him more than a
        // centimetre a tick, and read from his speed that push would make
        // him a walker again, unbraced, and the ranks behind a front would
        // shake. A joiner who kept his wish while the soldiers ahead hold
        // him would never brace, and any creep ahead would let the rear
        // walk into the backs ahead and drift back. Keeping his elbow room
        // (`elbow_room`), he stops on his look alone, an arm's length
        // behind that comrade.
        let arrived = *s.sight & SIGHT_WAY != 0 && (*s.stood || elbow_room());
        // An opening ahead that fits him (`look_around`): he steps into it
        // at the step pace, and once there he stands.
        if join_open() && s.aim.x.is_finite() {
            if keep_arm() {
                st.fight.join_dir = join_goal(f, s, st).0;
            }
            let to = *s.aim - p;
            let d = to.length();
            if d > 0.15 {
                st.desired = to * (STEP_PACE / d);
                st.walk.open_step = true;
            } else {
                *s.aim = Vec2::NAN;
                st.desired = Vec2::ZERO;
            }
        } else if keep_arm() {
            // His walk to the fight. How near the comrade in his way stands
            // shrinks it after the scan (`keep_arm_length`).
            let (way, wish) = join_goal(f, s, st);
            st.fight.join_dir = way;
            st.desired = wish;
        } else {
            st.desired = if arrived { Vec2::ZERO } else { join_goal(f, s, st).1 };
        }
    }
}

/// A step into an opening that fits him (`drive`) is not taken by the
/// crowd's brake or the press's hold; bodies still stop him through the
/// setting apart.
#[inline]
pub(super) fn step_into_opening(st: &mut Step) {
    if st.walk.open_step && st.fight.joining {
        st.desired = st.walk.wish;
        st.walk.fade = Vec2::ZERO;
        st.walk.jam = 0.0;
        st.walk.body_jam = 0.0;
    } else {
        st.walk.open_step = false;
    }
}

/// The pace at which a crowded soldier moves off the bodies inside his
/// arm's length, per unit of crowding (m/s): one comrade at the touch of
/// bodies moves him at 0.2 m/s. His legs follow a wish with a lag of a
/// third of a second (steer.rs STEER_GAIN), and a faster pace would carry
/// him past the place where the crowding ends.
const EASE_PACE: f32 = 1.5;

/// The share of his walk to the fight a soldier keeps with the comrade in
/// his way `gap` metres off: none at his arm's length `arm` or nearer, all
/// of it at SEP_RADIUS or beyond, and in between the square of the share
/// of that stretch he still has, so he slows the more the nearer he comes.
#[inline]
fn way_share(gap: f32, arm: f32) -> f32 {
    let u = ((gap - arm) / (SEP_RADIUS - arm)).clamp(0.0, 1.0);
    u * u
}

/// His move off the bodies inside his arm's length: along their sum
/// (`Scanned::crowding`), EASE_PACE for each unit of it, a step's pace at
/// most.
#[inline]
fn ease_off(crowding: Vec2) -> Vec2 {
    (crowding * EASE_PACE).clamp_length_max(STEP_PACE)
}

/// The arm's length rule (`keep_arm`): a soldier walking to the fight keeps
/// an arm's length (`arm_length`) from the bodies round him, read every
/// tick. Farther than an arm's length from the comrade in his way he walks
/// up, and his pace shrinks to nothing as he comes to arm's length. Nearer
/// than an arm's length to any body he moves away from it, the nearer the
/// faster, at a step's pace at most. He takes no step of a set length, and
/// he stands when nobody crowds him and the comrade in his way is at arm's
/// length. A step into an opening that fits him is left as it is.
#[inline]
pub(super) fn keep_arm_length(st: &mut Step) {
    if !keep_arm() || !st.fight.joining || st.walk.open_step {
        return;
    }
    // His walk to the fight, as the crowd's brake left it, by how far the
    // comrade in his way stands beyond his arm's length.
    let share = way_share(st.scan.way_gap, arm_length());
    st.desired *= share;
    st.walk.fade *= share;
    st.walk.wish *= share;
    // His move off the bodies inside his arm's length, which the brake does
    // not take: it leads away from them. The press's hold on his body is
    // for his walk into the men ahead, so a crowded soldier's is none.
    let ease = ease_off(st.scan.crowding);
    if ease != Vec2::ZERO {
        st.desired += ease;
        st.walk.wish += ease;
        st.walk.body_jam = 0.0;
    }
}

/// A broken man's flight (`drive`): his wish toward his own map edge.
#[inline]
fn flee(f: &Field, i: usize, p: Vec2) -> Vec2 {
    let Field { terrain, speed, team, paths, .. } = *f;
    // Broken: flee toward the own map edge with a per-unit lateral
    // scatter — slightly SLOWER than formed pursuers (0.9x): fleeing at
    // exactly max speed made pursuit a zero-kill treadmill (gap frozen
    // forever, counts flat for 30-45 s).
    let flee_z: f32 = if team[i] == 0 { -1.0 } else { 1.0 };
    let lat = (crate::units::hash01((i as u32).wrapping_mul(17) + 3) - 0.5)
        * 0.7;
    // On a map with a grid he runs the way a man walks to his edge (nav.rs
    // `flee_dir`): straight at it over open ground, along the wall's walk,
    // down a stair and round the wall where stone stands in the way. His
    // scatter turns that way to the same side as on open ground, where
    // it is `(lat, flee_z)`.
    let ahead = paths.nav.and_then(|nav| nav.flee_dir(team[i], p)).unwrap_or(Vec2::new(0.0, flee_z));
    let side = Vec2::new(ahead.y, -ahead.x) * flee_z;
    let dir = (ahead + side * lat).normalize_or_zero();
    let slope = terrain.slope_at(p.x, p.y);
    dir
        * speed[i]
        * ROUT_FLEE_FRAC
        * terrain.wade_mult(p.x, p.y)
        / (1.0 + 3.0 * slope * slope)
}

/// The goal of his walk to his place (`drive`), the route's point when
/// his regiment has a route, and whether he walks through it. Cut off
/// from the route with no way of his own in sight, he asks for one
/// (`asks`, sim/route.rs `WayAsk`).
#[inline]
fn follow_route(f: &Field, s: &mut Soldier, st: &Step, goal: Vec2, asks: &mut Vec<crate::sim::route::WayAsk>) -> (Vec2, bool) {
    let Field { orders, anchors, paths, .. } = *f;
    let i = s.i;
    let p = st.p;
    let gi = st.gi;
    let mut goal = goal;
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
                let aim = crate::sim::route::man_target(nav, route, frame, p, goal, way.is_finite().then_some(way), i);
                if aim.ask {
                    asks.push(crate::sim::route::WayAsk { g: gi as u32, p, goal: frame });
                }
                let (t, th) = (aim.at, aim.through);
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
    (goal, through)
}

/// Where a soldier walking to the fight goes (`joins_as_walker`): the way
/// (a unit vector, zero for none) and his wish along it. `drive` makes his
/// wish from it and `look_around` looks along the way, so his looks and
/// his facing follow the way he walks. In this order:
/// - an enemy he remembers (`remembered_foe`) within the short look
///   (WIDE_ACQUIRE_R): toward him;
/// - his regiment facing a way and holding a fighting line, and himself
///   more than a ready body behind that line along the facing
///   (`behind_the_line`): straight ahead along the facing, so a fighting
///   regiment's rear walks up as a block and not as a clump about one
///   point;
/// - the enemy he remembers within his closing range (his look,
///   `look_radius`, plus a metre; when the battle's set-up turns on
///   "Auto-engage nearby units", a man out of his rank looks as far as
///   he sees, `seek_radius`): toward him;
/// - his regiment's fight point while more than a metre off;
/// - with no melee on his regiment's clock (he left his rank on touch,
///   `join_touch`): the way his regiment faces, or for a regiment with no
///   facing the way his slot wish points;
/// - else nowhere.
///
/// Toward an enemy he goes at the jog over open ground more than
/// JOG_BEYOND from him, else at the advance, slowing over his last 0.8 m
/// and standing at his own stopping distance, between nine tenths of his
/// reach and just inside it. Straight ahead, to the fight point and on the
/// touch joiner's way he goes at the jog, or at the advance with a comrade
/// ahead. "A comrade ahead" is
/// his last look (`look_around`).
///
/// He goes that way only where he can walk it straight: on a map with
/// blocked ground, a fight behind a wall, on its walk above him or across
/// a cliff is reached by his regiment's route, so there he walks his slot
/// wish, which follows the route (`follow_route`), instead. One straight
/// line read on the planner's grid a call; none where no ground blocks.
#[inline]
pub(super) fn join_goal(f: &Field, s: &Soldier, st: &Step) -> (Vec2, Vec2) {
    let (way, wish, toward) = join_way(f, s, st);
    match f.paths.nav {
        Some(nav) if toward != st.p && !nav.walk_line(st.p, toward) => {
            (st.fight.slot_wish.normalize_or_zero(), st.fight.slot_wish)
        }
        _ => (way, wish),
    }
}

/// `join_goal`'s way, wish and the point it heads for, before the test
/// that he can walk there straight; the point is where he stands when he
/// heads nowhere.
#[inline]
fn join_way(f: &Field, s: &Soldier, st: &Step) -> (Vec2, Vec2, Vec2) {
    let Field { pos_prev, engaged, hold, form_face, fight_line, .. } = *f;
    let i = s.i;
    let p = st.p;
    let gi = st.gi;
    let ahead = *s.sight & SIGHT_AHEAD != 0;
    let memo = *s.target as usize;
    let (d2, counts) = remembered_foe(f, i, p, gi, memo);
    let dist = d2.sqrt();
    let to_foe = |max_close: f32| {
        let params = &TYPES[st.my_kind];
        let to_enemy = pos_prev[memo].xz() - p;
        let dir = to_enemy / dist.max(1e-6);
        let own = crate::units::hash01((i as u32).wrapping_mul(0x3C1B) ^ 0x51F7);
        let fight = FIGHT_SHARE * params.reach;
        let stop = fight + 0.8 * (params.reach - fight) * own;
        let at = pos_prev[memo].xz();
        if dist <= stop || dist >= max_close {
            return (dir, Vec2::ZERO, at);
        }
        let urge = ((dist - stop) / 0.8).clamp(0.0, 1.0);
        let pace = if dist > JOG_BEYOND && !ahead { COMBAT_JOG_PACE } else { ADVANCE_PACE };
        (dir, to_enemy * (pace * urge / dist), at)
    };
    let look = look_radius(f.engage_near, engaged[gi], st);
    let max_close = if hold[gi] { 2.2 } else { look + 1.0 };
    if counts && dist < WIDE_ACQUIRE_R {
        return to_foe(max_close);
    }
    let face = form_face[gi];
    let line = fight_line[gi];
    if behind_the_line(face, line, p, ready_body()) {
        let pace = if ahead { ADVANCE_PACE } else { COMBAT_JOG_PACE };
        return (face, face * pace, p + face * (line - p.dot(face)));
    }
    if counts {
        return to_foe(max_close);
    }
    if let Some(fp) = st.fight.join_fp {
        let to_fp = fp - p;
        let d = to_fp.length();
        if d > 1.0 {
            let pace = if ahead { ADVANCE_PACE } else { COMBAT_JOG_PACE };
            return (to_fp / d, to_fp * (pace / d), fp);
        }
        return (to_fp.normalize_or_zero(), Vec2::ZERO, fp);
    }
    // With no melee on his regiment's clock a soldier walking to the fight
    // walks toward it (`join_touch`): the way his regiment faces, and the
    // way his slot wish points only for a regiment with no facing. His
    // place can lie behind him (he ran past it in the charge), so his slot
    // wish is not the way to the fight. Only a soldier walking to the
    // fight this tick has this way: one out of his rank for a blow or its
    // recovery, as every soldier is on the battle's first ticks, has none,
    // sees no comrade ahead on it and is back in his rank at once
    // (`decide_to_join`).
    let toward = if face != Vec2::ZERO { face } else { st.fight.slot_wish.normalize_or_zero() };
    if join_touch() && !st.fight.in_melee && st.fight.joining && toward != Vec2::ZERO {
        let pace = if ahead { ADVANCE_PACE } else { COMBAT_JOG_PACE };
        return (toward, toward * pace, p + toward * 2.0);
    }
    (Vec2::ZERO, Vec2::ZERO, p)
}

/// Whether a soldier at `p` stands more than `body` behind his regiment's
/// fighting line `line` along its facing `face` (`join_goal`): never with
/// no facing or no line (NaN).
#[inline]
fn behind_the_line(face: Vec2, line: f32, p: Vec2, body: f32) -> bool {
    face != Vec2::ZERO && !line.is_nan() && line - p.dot(face) > body
}

/// What the crowd does to his drive: overlap corrections and pushes are
/// clamped, and in a packed crowd the drive fades out so the mass cannot
/// keep compressing itself.
#[inline]
pub(super) fn yield_to_crowd(f: &Field, st: &mut Step) {
    let crowd = st.scan.crowd;
    let mut corr = st.scan.corr;
    let mut push = st.scan.push;
    let mut desired = st.desired;
    let corr_len2 = corr.length_squared();
    if corr_len2 > 1e-12 {
        st.scan.overlap_dir = corr / corr_len2.sqrt();
        st.scan.overlap_depth = corr_len2.sqrt() / CORR_GAIN;
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
    let (slow, stop) = st.scan.brake;
    let jam = if still_brake() {
        // The friction contact's brake: the still men ahead of him against
        // a band half as wide, for that smaller count.
        let (slow, stop) = (0.5 * slow, 0.5 * stop);
        ((st.scan.crowd_still - slow) / (stop - slow)).clamp(0.0, 1.0)
    } else {
        ((crowd - slow) / (stop - slow)).clamp(0.0, 1.0)
    };
    // A soldier of a regiment in its charge keeps his whole wish to run in
    // (`charge_runs_in`): the soldiers ahead of him do not fade it. Else the
    // first soldiers to reach a standing friend stand, the rank behind them
    // stands for them, and the whole regiment halts within a second of the
    // first touch. His body is still slowed by the still men ahead as
    // anyone's is (`body_jam`).
    let pressed = jam;
    let runs_in = charge_runs_in() && f.charging[st.gi] && !f.fat_nocharge[st.gi] && !st.dying && !st.routed;
    let jam = if runs_in { 0.0 } else { jam };
    st.walk.wish = desired;
    st.walk.fade = desired * jam;
    desired *= 1.0 - jam;
    st.scan.corr = corr;
    st.scan.push = push;
    st.walk.jam = jam;
    st.walk.body_jam = if runs_in { pressed } else { jam };
    st.desired = desired;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A soldier walking to his place slows over the last metres; one on a
    /// route, or in his regiment's charge, keeps his whole pace up to it.
    #[test]
    fn a_charging_soldier_does_not_slow_for_his_place() {
        assert_eq!(arrive_share(70.0, ARRIVE_RADIUS, false, false), 1.0);
        assert!((arrive_share(10.0, ARRIVE_RADIUS, false, false) - 10.0 / ARRIVE_RADIUS).abs() < 1e-6);
        assert_eq!(arrive_share(10.0, ARRIVE_RADIUS, true, false), 1.0);
        assert_eq!(arrive_share(10.0, ARRIVE_RADIUS, false, true), 1.0);
        assert_eq!(arrive_share(0.5, ARRIVE_RADIUS, false, true), 1.0);
    }

    /// A joiner more than a ready body behind his regiment's fighting line
    /// along its facing walks straight ahead; within a ready body of it, past
    /// it, or with no line or no facing he does not.
    #[test]
    fn a_joiner_walks_straight_ahead_only_from_behind_the_line() {
        let face = Vec2::new(0.0, 1.0);
        let body = 0.9;
        assert!(behind_the_line(face, 10.0, Vec2::new(3.0, 5.0), body));
        assert!(behind_the_line(face, 10.0, Vec2::new(-40.0, 9.0), body));
        assert!(!behind_the_line(face, 10.0, Vec2::new(0.0, 9.1), body));
        assert!(!behind_the_line(face, 10.0, Vec2::new(0.0, 12.0), body));
        assert!(!behind_the_line(face, f32::NAN, Vec2::new(0.0, 0.0), body));
        assert!(!behind_the_line(Vec2::ZERO, 10.0, Vec2::new(0.0, 0.0), body));
        let east = Vec2::new(1.0, 0.0);
        assert!(behind_the_line(east, -20.0, Vec2::new(-25.0, 100.0), body));
        assert!(!behind_the_line(east, -20.0, Vec2::new(-20.5, 100.0), body));
    }

    /// A joiner keeps none of his walk with the comrade in his way at his
    /// arm's length or nearer, all of it at SEP_RADIUS and beyond or with
    /// nobody in his way, a quarter half way, and never less as the gap grows.
    #[test]
    fn way_share_shrinks_the_walk_to_nothing_at_arms_length() {
        let arm = 1.05;
        assert_eq!(way_share(arm, arm), 0.0);
        assert_eq!(way_share(0.5, arm), 0.0);
        assert_eq!(way_share(0.0, arm), 0.0);
        assert_eq!(way_share(SEP_RADIUS, arm), 1.0);
        assert_eq!(way_share(3.0, arm), 1.0);
        assert_eq!(way_share(f32::INFINITY, arm), 1.0);
        assert!((way_share(0.5 * (arm + SEP_RADIUS), arm) - 0.25).abs() < 1e-5);
        let mut last = 0.0;
        for k in 0..=200 {
            let share = way_share(k as f32 * 0.01, arm);
            assert!(share >= last, "{k}");
            last = share;
        }
    }

    /// A soldier with no crowding does not move off; with crowding he moves
    /// along it, the faster the more of it, never past STEP_PACE.
    #[test]
    fn ease_off_leads_away_from_the_crowding_at_a_step_at_most() {
        assert_eq!(ease_off(Vec2::ZERO), Vec2::ZERO);
        let dir = Vec2::new(0.6, -0.8);
        let mut last = 0.0;
        for k in 1..=100 {
            let ease = ease_off(dir * (k as f32 * 0.05));
            assert!(ease.normalize().dot(dir) > 0.9999, "{k}");
            assert!(ease.length() >= last - 1e-6, "{k}");
            assert!(ease.length() <= STEP_PACE + 1e-6, "{k}");
            last = ease.length();
        }
        assert!(ease_off(dir * 0.1).length() < ease_off(dir * 0.2).length());
        assert!((ease_off(dir * 100.0).length() - STEP_PACE).abs() < 1e-5);
    }
}
