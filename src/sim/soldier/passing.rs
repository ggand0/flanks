//! Getting past comrades: the look around him, the choice of opening, the
//! try at a side, the squeeze and the frame a fighting regiment keeps.

use bevy::math::Vec3Swizzles;
use bevy::prelude::*;

use super::*;

/// The room (m) a standing joiner wants behind the man ahead before he
/// steps up: his look calls a comrade "in his way" only within a ready
/// body plus this (`look_around`), so he steps up once a short step of
/// room has opened, and from formation spacing every rank first closes up
/// to the bodies ahead.
const JOIN_ROOM: f32 = 0.3;
/// Under the friction contact the soldiers of a regiment under an Attack
/// order do not step backward to their places (`hold_the_frame`), so a
/// regiment that ran in behind a friend stays piled up. A soldier who
/// follows the route round a wall or a block walks it whichever way it
/// goes.
fn pile_stays() -> bool {
    friction_contact()
}
/// A man does not re-read the comrades around him 30 times a second.
/// He looks at the comrades beside him and in his way every LOOK_TICKS
/// (about a quarter second, a human reaction time), in his own rhythm,
/// and acts on what he last saw (the `sight` column) in between. His
/// far look (enemies in sight, comrades running to the fight) keeps the
/// acquisition's own 1-in-8 rhythm.
pub(super) const LOOK_TICKS: u32 = 8;
/// Ring aims set this tick, for the straggler log (scenes/logs.rs).
pub(crate) static RING_AIMS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The ring read (`choose_opening`): how far round him a hemmed man reads
/// the men for a crack, and how close to a ring aim counts as on it.
const RING_SEE: f32 = 2.0;
const RING_ON: f32 = 0.3;
/// The widest open side (radians between two neighbours round him) a man
/// may have and still count as hemmed in: a third of a turn.
const RING_OPEN_SPAN: f32 = 2.0 * std::f32::consts::FRAC_PI_3;
/// FL_OPEN_RING=0 turns the ring read off.
fn open_ring() -> bool {
    static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *K.get_or_init(|| crate::util::env_or("FL_OPEN_RING", 1u32) != 0)
}

/// The side a man leans to among equal openings: the side he chose last
/// while he remembers it, and his own coin, the same every tick.
#[inline]
pub(super) fn side_pref(i: usize, stuck: u16) -> (f32, f32) {
    let pref = if (stuck & STUCK_MEM) != 0 { if stuck & STUCK_LEFT != 0 { 1.0 } else { -1.0 } } else { 0.0 };
    let coin = if crate::units::hash01((i as u32).wrapping_mul(0x7A31).wrapping_add(13)) > 0.5 { 1.0 } else { -1.0 };
    (pref, coin)
}

/// The choice of opening (OPEN_SEE above): from the obstacles ahead his
/// scan saw, and on his look tick from a wider look, the opening he aims
/// at; his drive turned toward it at its pace; the side he chose
/// remembered as the try remembers its side. Only a man walking somewhere
/// with no enemy in mind and his regiment not fighting (the try's test):
/// fighting men keep their own footwork.
#[inline]
pub(super) fn choose_opening(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { orders, tick, anchors, home, .. } = *f;
    let i = s.i;
    let going = st.open.open_on && st.foe.best_idx == u32::MAX && !st.foe.memo_valid;
    let ring_flag = *s.sight & SIGHT_RING != 0;
    *s.sight &= !SIGHT_RING;
    if !going {
        // A joiner keeps the aim of his opening ahead (`look_around`).
        if !(join_open() && st.fight.joining) {
            *s.aim = Vec2::NAN;
        }
        return;
    }
    let p = st.p;
    let want = st.desired.length();
    let wd = st.desired / want;
    let perp = Vec2::new(-wd.y, wd.x);
    let mut v = *s.stuck;
    // The near read, from the scan.
    let (mut aim_lat, mut aim_along, mut fit) = st.open.near;
    if fit > 0 {
        *s.aim = Vec2::NAN;
    }
    // A man he follows still takes up his space: a near crack that fits
    // only his depth is taken by any comrade ahead of him, one he follows
    // or not, whose body overlaps its air short of its passage's far side.
    // He may still aim at a taken crack, which is following, but never
    // turns his shoulders for it. Only for a man with such a crack in his
    // aim, so a second look at the comrades within the scan's box.
    let taken = fit == 1 && crack_taken(f, i, st, p, wd);
    // The ring (FL_OPEN_RING): a man of a holding regiment far from his
    // place with nobody ahead fitting him reads the men all round him within RING_SEE on his look
    // tick, the crack between each two neighbours round the ring, and aims
    // at the one that fits him nearest his way, whichever side it lies.
    // A man hemmed in a dressed block at an angle to his way sees no pair
    // ahead of his line, and the cracks round him run between a man ahead
    // and a man abeam or behind.
    let look = tick.wrapping_add((i as u32).wrapping_mul(7)).is_multiple_of(OPEN_LOOK_TICKS);
    // Only a man whose regiment holds: a marching man far from his place
    // is the whole regiment on the march, and a lane walker who reads the
    // ring aims beside himself and blocks the lane.
    let hemmed = open_ring()
        && orders[st.gi].is_none()
        && (anchors[st.gi] + home[i]).distance_squared(p) > TRY_FROM_SLOT * TRY_FROM_SLOT;
    let ring_aim = fit == 0 && st.open.saw && hemmed && look && read_the_ring(f, s, st, p, wd, perp, &mut v);
    // The wide look: nothing near fits and he saw somebody in his way.
    if fit == 0 && !ring_aim && st.open.saw && look {
        wide_look(f, s, st, p, wd, perp, &mut v);
    }
    // A near fit wins; else the wide aim while it is still ahead of him,
    // or a ring aim within RING_SEE whichever way it lies, until he is on
    // it.
    let mut ring_kept = false;
    if fit == 0 && s.aim.x.is_finite() {
        let rel = *s.aim - p;
        let r_along = rel.dot(wd);
        let d2 = rel.length_squared();
        let near_ring = (ring_flag || ring_aim) && d2 < RING_SEE * RING_SEE && d2 > RING_ON * RING_ON;
        if r_along > OPEN_AHEAD || near_ring {
            aim_lat = rel.dot(perp);
            aim_along = r_along;
            fit = if v & STUCK_WIDE_DEPTH != 0 { 1 } else { 2 };
            ring_kept = r_along <= OPEN_AHEAD;
            if near_ring {
                *s.sight |= SIGHT_RING;
            }
        } else {
            *s.aim = Vec2::NAN;
        }
    }
    st.open.aim_fit = fit;
    st.open.gap = fit == 1 && !taken;
    st.open.gap_near = st.open.gap && aim_along < st.open.sep0;
    if fit > 0 {
        // The side he chose, remembered as the try remembers its side.
        if aim_lat.abs() > 0.1 {
            v = (v & !(STUCK_MEM | STUCK_LEFT)) | (SIDE_MEMORY << STUCK_MEM_SHIFT) | ((aim_lat > 0.0) as u16 * STUCK_LEFT);
        }
        // His drive turned toward the aim, at its pace; a ring aim beside
        // or behind him is driven at straight.
        let to = if ring_kept { wd * aim_along + perp * aim_lat } else { wd * aim_along.max(OPEN_MIN_AHEAD) + perp * aim_lat };
        let len = to.length();
        if len > 1e-6 {
            st.desired = to * (want / len);
        }
    }
    *s.stuck = v;
}

/// Whether the near crack that fits only his depth is taken (`choose_opening`).
#[inline]
fn crack_taken(f: &Field, i: usize, st: &Step, p: Vec2, wd: Vec2) -> bool {
    let Field { grid, .. } = *f;
    let mut taken = false;

    let (lo, hi, far) = st.open.near_span;
    let my_team_bit = st.scan.my_team_bit;
    let widths = widths();
    grid.for_each_candidate(p, QUERY_RADIUS, |o| {
        if taken
            || o.idx as usize == i
            || (o.meta & crate::spatial::META_TEAM) != my_team_bit
            || (o.meta & crate::spatial::META_DYING) != 0
        {
            return;
        }
        let to = o.xz() - p;
        let along = to.dot(wd);
        if along <= OPEN_AHEAD || to.length_squared() >= QUERY_RADIUS * QUERY_RADIUS {
            return;
        }
        let lat = wd.x * to.y - wd.y * to.x;
        let r = 0.5 * widths[crate::spatial::meta_pose(o.meta)].body;
        taken = along - r < far && lat + r > lo && lat - r < hi;
    });
    taken
}

/// The ring read (`choose_opening`): the crack round a hemmed man nearest
/// his way that fits him and that his body can walk straight to
/// (`way_free`), as his aim. Returns whether he aims at one.
#[inline]
fn read_the_ring(f: &Field, s: &mut Soldier, st: &Step, p: Vec2, wd: Vec2, perp: Vec2, v: &mut u16) -> bool {
    let Field { grid, orders, terrain, .. } = *f;
    let i = s.i;
    let mut ring_aim = false;
    let my_team_bit = st.scan.my_team_bit;
    let widths = widths();
    let mut men: [(f32, Vec2, f32); 16] = [(0.0, Vec2::ZERO, 0.0); 16];
    let mut n = 0usize;
    let mut own = 0usize;
    grid.for_each_candidate_vel(p, RING_SEE, |o, o_vel| {
        if n == men.len()
            || o.idx as usize == i
            || (o.meta & crate::spatial::META_TEAM) != my_team_bit
            || (o.meta & crate::spatial::META_DYING) != 0
        {
            return;
        }
        let to = o.xz() - p;
        if to.length_squared() >= RING_SEE * RING_SEE {
            return;
        }
        let holding = orders[crate::spatial::meta_group(o.meta)].is_none();
        if !holding && o_vel.dot(wd) > QUEUE_SPEED {
            return;
        }
        let ang = (wd.x * to.y - wd.y * to.x).atan2(to.dot(wd));
        men[n] = (ang, to, 0.5 * widths[crate::spatial::meta_pose(o.meta)].body);
        n += 1;
        own += usize::from(crate::spatial::meta_group(o.meta) == st.gi);
    });
    // Hemmed in: men on every side of him, no open side wider than
    // RING_OPEN_SPAN, and most of them another regiment's. A man in a
    // lane or at a block's edge has an open side and walks; a man
    // inside his own block (an attacking block the moment it arrives)
    // is among his comrades; the ring is for the man left in a cell of
    // another regiment's planted men.
    let mut open_span = std::f32::consts::TAU;
    if n >= 3 {
        men[..n].sort_by(|a, b| a.0.total_cmp(&b.0));
        open_span = (0..n)
            .map(|k| if k + 1 == n { men[0].0 + std::f32::consts::TAU - men[k].0 } else { men[k + 1].0 - men[k].0 })
            .fold(0.0f32, f32::max);
    }
    if n >= 3 && open_span <= RING_OPEN_SPAN && 2 * own < n {
        let mut best = f32::INFINITY;
        let mut best_point = Vec2::ZERO;
        let mut best_fit = 0u8;
        for k in 0..n {
            let (a0, q0, h0) = men[k];
            let (a1, q1, h1) = men[(k + 1) % n];
            let span = if k + 1 == n { a1 + std::f32::consts::TAU - a0 } else { a1 - a0 };
            let (point, wf) = if span > std::f32::consts::FRAC_PI_2 {
                // Open ground between them: the way between at the
                // nearer man's distance.
                let mid = a0 + 0.5 * span;
                let dir = wd * mid.cos() + perp * mid.sin();
                (dir * q0.length().min(q1.length()).max(st.open.sep0), 2u8)
            } else {
                let chord = q1 - q0;
                let len = chord.length();
                let air = len - h0 - h1;
                let wf = if air >= st.open.body0 - OPEN_SLACK {
                    2
                } else if air >= SQUEEZE_DEPTH {
                    1
                } else {
                    0
                };
                if wf == 0 || len < 1e-3 {
                    continue;
                }
                (q0 + chord * ((h0 + 0.5 * air) / len), wf)
            };
            let off = (wd.x * point.y - wd.y * point.x).atan2(point.dot(wd)).abs();
            let cost = off + if wf == 1 { SQUEEZE_DETOUR } else { 0.0 };
            if cost < best && way_free(terrain, p, point) {
                best = cost;
                best_point = point;
                best_fit = wf;
            }
        }
        if best_fit > 0 {
            *s.aim = p + best_point;
            *v = (*v & !STUCK_WIDE_DEPTH) | ((best_fit == 1) as u16 * STUCK_WIDE_DEPTH);
            ring_aim = true;
            RING_AIMS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    ring_aim
}

/// The wide look (`choose_opening`): out to OPEN_FAR, the opening in the
/// row nearest him that fits him and that his body can walk straight to
/// (`way_free`), as his aim.
#[inline]
fn wide_look(f: &Field, s: &mut Soldier, st: &Step, p: Vec2, wd: Vec2, perp: Vec2, v: &mut u16) {
    let Field { grid, orders, terrain, .. } = *f;
    let i = s.i;
    let (pref, coin) = side_pref(i, *v);
    let my_team_bit = st.scan.my_team_bit;
    let widths = widths();
    let read = |o: &crate::spatial::SortedUnit, o_vel: Vec2| -> Option<(f32, f32, f32)> {
        if o.idx as usize == i
            || (o.meta & crate::spatial::META_TEAM) != my_team_bit
            || (o.meta & crate::spatial::META_DYING) != 0
        {
            return None;
        }
        let to = o.xz() - p;
        let along = to.dot(wd);
        if along <= OPEN_AHEAD || to.length_squared() >= OPEN_FAR * OPEN_FAR {
            return None;
        }
        let holding = orders[crate::spatial::meta_group(o.meta)].is_none();
        if !holding && o_vel.dot(wd) > QUEUE_SPEED {
            return None;
        }
        let lat = wd.x * to.y - wd.y * to.x;
        Some((along, lat, 0.5 * widths[crate::spatial::meta_pose(o.meta)].body))
    };
    // One pass over the grid: the comrades in his sight into a buffer,
    // then the nearest and the row from it. Two passes cost 3.0 ms of
    // the 200k step; the buffer overflowing (a press denser than
    // WIDE_SEEN men in 4 m) falls back to the second pass.
    let mut wide_at = f32::INFINITY;
    let mut seen: [(f32, f32, f32); WIDE_SEEN] = [(0.0, 0.0, 0.0); WIDE_SEEN];
    let mut n_seen = 0usize;
    let mut overflow = false;
    grid.for_each_candidate_vel(p, OPEN_FAR, |o, o_vel| {
        if let Some(m) = read(o, o_vel) {
            wide_at = wide_at.min(m.0);
            if n_seen < WIDE_SEEN {
                seen[n_seen] = m;
                n_seen += 1;
            } else {
                overflow = true;
            }
        }
    });
    let mut wide = Row::default();
    if overflow {
        grid.for_each_candidate_vel(p, OPEN_FAR, |o, o_vel| {
            if let Some((along, lat, half)) = read(o, o_vel)
                && along < wide_at + OPEN_BAND
            {
                wide.add(lat, half, along);
            }
        });
    } else {
        for &(along, lat, half) in &seen[..n_seen] {
            if along < wide_at + OPEN_BAND {
                wide.add(lat, half, along);
            }
        }
    }
    let free = |lat: f32, along: f32| way_free(terrain, p, wd * along + perp * lat);
    let (w_lat, w_along, w_fit, _) =
        openings(&wide, wide_at, OPEN_FAR, st.open.body0, SQUEEZE_DEPTH, true, pref, coin, free);
    *s.aim = if w_fit > 0 { p + wd * w_along + perp * w_lat } else { Vec2::NAN };
    *v = (*v & !STUCK_WIDE_DEPTH) | ((w_fit == 1) as u16 * STUCK_WIDE_DEPTH);
}

/// The depth bodies already give in a press (contact.rs CROWD_DEPTH): an
/// opening this much narrower than his body still fits him.
const JOIN_OPEN_GIVE: f32 = 0.1;
/// His elbow room (`open_roomy`): an opening fits him only this much wider
/// than his body.
const JOIN_OPEN_ELBOW: f32 = 0.3;

/// The opening among the bodies just ahead of him along `wd` (both teams,
/// within OPEN_SEE, on his level; a comrade walking away along his way is
/// none) that fits his width less JOIN_OPEN_GIVE (`join_open`): the point
/// he aims at in the world, None when nothing fits. With `open_side` he
/// reads the bodies out to QUERY_RADIUS, takes the nearest rank of them as
/// the row and counts the open ground beside it too, and an aim another
/// body stands on is none; with `open_roomy` the opening must fit his width
/// and JOIN_OPEN_ELBOW.
#[inline]
pub(super) fn opening_ahead(f: &Field, i: usize, st: &Step, wd: Vec2, body: f32) -> Option<Vec2> {
    if wd == Vec2::ZERO {
        return None;
    }
    let p = st.p;
    let my_team_bit = st.scan.my_team_bit;
    let widths = widths();
    let levels = Levels::new(f, i);
    let mut row = Row::default();
    let mut row_at = f32::INFINITY;
    let side = open_side();
    let see = if side { QUERY_RADIUS } else { OPEN_SEE };
    let mut kept = [(0.0f32, 0.0f32, 0.0f32); 24];
    let mut n_kept = 0usize;
    f.grid.for_each_candidate_vel(p, QUERY_RADIUS, |o, ov| {
        if o.idx as usize == i || (o.meta & crate::spatial::META_DYING) != 0 || levels.apart(f, o.idx) {
            return;
        }
        let to = o.xz() - p;
        if to.length_squared() >= see * see {
            return;
        }
        let along = to.dot(wd);
        if along <= OPEN_AHEAD {
            return;
        }
        if (o.meta & crate::spatial::META_TEAM) == my_team_bit && ov.dot(wd) > 0.5 {
            return;
        }
        let lat = wd.x * to.y - wd.y * to.x;
        let half = 0.5 * widths[crate::spatial::meta_pose(o.meta)].body;
        if side {
            if n_kept < kept.len() {
                kept[n_kept] = (lat, half, along);
                n_kept += 1;
            }
        } else {
            row.add(lat, half, along);
        }
        row_at = row_at.min(along);
    });
    // The nearest rank: the bodies within 0.6 m along of the nearest one.
    if side {
        for &(lat, half, along) in &kept[..n_kept] {
            if along - row_at <= 0.6 {
                row.add(lat, half, along);
            }
        }
    }
    if row.n_left == 0 && row.n_right == 0 {
        return None;
    }
    let perp = Vec2::new(-wd.y, wd.x);
    let coin = side_pref(i, 0).1;
    let free = |lat: f32, along: f32| way_free(f.terrain, p, wd * along + perp * lat);
    let fit_w = if open_roomy() { body + JOIN_OPEN_ELBOW } else { body - JOIN_OPEN_GIVE };
    let (aim_lat, aim_along, fit, _) = openings(&row, row_at, see, fit_w, f32::INFINITY, side, 0.0, coin, free);
    let aim = p + wd * aim_along.max(OPEN_MIN_AHEAD) + perp * aim_lat;
    (fit == 2 && !(side && aim_taken(f, i, aim, body))).then_some(aim)
}

/// Whether another living body on his level stands on `aim`, within his
/// and its half bodies less JOIN_OPEN_GIVE, or with `open_roomy` within
/// them and half his elbow room.
fn aim_taken(f: &Field, i: usize, aim: Vec2, body: f32) -> bool {
    let widths = widths();
    let levels = Levels::new(f, i);
    let mut taken = false;
    f.grid.for_each_candidate(aim, QUERY_RADIUS, |o| {
        if taken || o.idx as usize == i || (o.meta & crate::spatial::META_DYING) != 0 || levels.apart(f, o.idx) {
            return;
        }
        let o_body = widths[crate::spatial::meta_pose(o.meta)].body;
        let r = if open_roomy() { 0.5 * (body + o_body) + 0.5 * JOIN_OPEN_ELBOW } else { 0.5 * (body + o_body) - JOIN_OPEN_GIVE };
        taken = o.xz().distance_squared(aim) < r * r;
    });
    taken
}

/// What he remembers and what he sees of his comrades: the enemy he
/// remembers if still in sight, the way he is going, and every eighth tick
/// a look at the men around him (a comrade in his way, to his left or
/// right, ahead, between him and his mark, or running to the fight; with
/// an enemy in reach, a comrade to either side of his way to him and a
/// man close behind him), remembered in the sight bits until the next
/// look.
#[inline]
pub(super) fn look_around(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { pos_prev, orders, engaged, tick, .. } = *f;
    let i = s.i;
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    let in_melee = st.fight.in_melee;
    let committed = st.fight.committed;
    let join_fp = st.fight.join_fp;
    let desired = st.desired;
    let best_idx = st.foe.best_idx;
    // The enemy he remembers, as the scan read it (`scan`).
    let memo = st.foe.prev_target as usize;
    let memo_valid = st.foe.memo_valid;
    // Which way he is going: to that enemy, else to his regiment's fight.
    // A man walking to the fight with nobody in reach looks along the way
    // he walks (`join_goal`): straight ahead, to an enemy or to the fight.
    let memo_dir = if joins_as_walker() && committed && best_idx == u32::MAX && !dying && !routed {
        join_goal(f, s, st).0
    } else if memo_valid {
        (pos_prev[memo].xz() - p).normalize_or_zero()
    } else if let Some(fp) = join_fp {
        (fp - p).normalize_or_zero()
    } else {
        Vec2::ZERO
    };
    // The blocking test toward a holding man's own mark. A man walking to
    // the fight (`joining`) has his slot wish kept apart; the test reads
    // that, so it stays about his mark.
    let slot_dir = if orders[gi].is_none() && engaged[gi] && !routed {
        if st.fight.joining { st.fight.slot_wish.normalize_or_zero() } else { desired.normalize_or_zero() }
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
    // A joiner's comrade "in his way" stands within a ready body plus a
    // short step of room (JOIN_ROOM), or within SEP_RADIUS when he keeps
    // his elbow room (`elbow_room`); for everyone else, and for his sides
    // and mark, within SEP_RADIUS.
    let way_r = if st.fight.joining && !elbow_room() { ready_body() + JOIN_ROOM } else { SEP_RADIUS };
    if (i as u32).wrapping_add(tick).is_multiple_of(LOOK_TICKS) {
        let mut seen = 0u16;
        if look_lanes || watch_go || slot_on {
            let ways = OnHisWay { memo_dir, side_dir, slot_dir, look_lanes, watch_go, slot_on, way_r };
            seen = look_on_his_way(f, i, st, &ways);
        }
        if fighter {
            seen |= look_at_the_room(f, i, st, fight_dir, side_dir);
        }
        // A joiner stopped behind comrades reads the row of bodies just
        // ahead of him and aims at an opening that fits him (`join_open`,
        // `drive`). An aim still ahead of him that nobody has taken is
        // kept.
        if join_open() && st.fight.joining && !dying && !routed && best_idx == u32::MAX {
            let blocked = seen & SIGHT_WAY != 0;
            if blocked && *s.stood && memo_dir != Vec2::ZERO {
                let body = st.open.body0;
                let keep = s.aim.x.is_finite() && s.aim.distance_squared(p) > 0.15 * 0.15 && !aim_taken(f, i, *s.aim, body);
                if !keep {
                    *s.aim = opening_ahead(f, i, st, memo_dir, body).unwrap_or(Vec2::NAN);
                }
            } else {
                *s.aim = Vec2::NAN;
            }
        }
        // A soldier in his rank walking to his place, stopped behind a
        // comrade who has left his rank and stands, leaves his rank too
        // (`join_touch`): he sees that comrade go and has him in his way.
        if join_touch()
            && st.fight.may_leave
            && !committed
            && !in_melee
            && best_idx == u32::MAX
            && !dying
            && !routed
            && desired.length_squared() > 0.25
            && st.scan.crowd_still > 0.0
            && touched(f, i, st, desired.normalize_or_zero())
        {
            seen |= SIGHT_GO | SIGHT_WAY | SIGHT_AHEAD;
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
    st.sight.back_blocked = *s.sight & SIGHT_BACK != 0;
    st.sight.memo_dir = memo_dir;
    st.sight.side_dir = side_dir;
    st.sight.watch_go = watch_go;
    st.sight.way_blocked = way_blocked;
    st.sight.left_blocked = left_blocked;
    st.sight.right_blocked = right_blocked;
    st.sight.comrade_ahead = comrade_ahead;
    st.sight.slot_blocked = slot_blocked;
}

/// Whether a living comrade out of his rank and standing is within
/// SEP_RADIUS of him, within 45 degrees of his wish `wish_dir`
/// (`join_touch`).
fn touched(f: &Field, i: usize, st: &Step, wish_dir: Vec2) -> bool {
    let p = st.p;
    let my_team_bit = st.scan.my_team_bit;
    let mut found = false;
    f.grid.for_each_candidate(p, st.scan.scan_r, |o| {
        if found
            || o.idx as usize == i
            || (o.meta & crate::spatial::META_TEAM) != my_team_bit
            || (o.meta & crate::spatial::META_DYING) != 0
            || (o.meta & crate::spatial::META_STOOD) == 0
            || !f.out_form[o.idx as usize]
        {
            return;
        }
        let to = o.xz() - p;
        let d2 = to.length_squared();
        found = d2 < SEP_RADIUS * SEP_RADIUS && to.dot(wish_dir) > 0.707 * d2.sqrt();
    });
    found
}

/// What his look on his way looks for (`look_around`): the way he is
/// going, his sides and the way to his mark, whether he looks for a lane,
/// watches his comrades go or tests his mark, and how near a comrade in
/// his way stands.
#[derive(Clone, Copy)]
struct OnHisWay {
    memo_dir: Vec2,
    side_dir: Vec2,
    slot_dir: Vec2,
    look_lanes: bool,
    watch_go: bool,
    slot_on: bool,
    way_r: f32,
}

/// His look at the comrades around him on his way (`look_around`): the
/// sight bits of a comrade in his way, to his left or right, ahead,
/// between him and his mark, and running to the fight.
#[inline]
fn look_on_his_way(f: &Field, i: usize, st: &Step, ways: &OnHisWay) -> u16 {
    let Field { grid, .. } = *f;
    let OnHisWay { memo_dir, side_dir, slot_dir, look_lanes, watch_go, slot_on, way_r } = *ways;
    let p = st.p;
    let gi = st.gi;
    let scan_r = st.scan.scan_r;
    let my_team_bit = st.scan.my_team_bit;
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
        way |= ahead & (d2 < way_r * way_r);
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
    (way as u16 * SIGHT_WAY)
        | (left as u16 * SIGHT_LEFT)
        | (right as u16 * SIGHT_RIGHT)
        | (ahead_any as u16 * SIGHT_AHEAD)
        | (mark as u16 * SIGHT_MARK)
        | (go as u16 * SIGHT_GO)
}

/// The room he has to fight the enemy in reach (`look_around`): the
/// sight bits of a comrade close to either side of his way to the enemy
/// and of any man close behind him.
#[inline]
fn look_at_the_room(f: &Field, i: usize, st: &Step, fight_dir: Vec2, side_dir: Vec2) -> u16 {
    let Field { grid, .. } = *f;
    let p = st.p;
    let scan_r = st.scan.scan_r;
    let my_team_bit = st.scan.my_team_bit;
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
    (left as u16 * SIGHT_LEFT)
        | (right as u16 * SIGHT_RIGHT)
        | (back as u16 * SIGHT_BACK)
}

/// Stuck ticks before a man tries a side (half a second); how long a try
/// lasts at most (a second) and at least (a real step, not a twitch);
/// how long he remembers the side he tried, in steps of 8 ticks (four
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
pub(super) const TRY_FROM_SLOT: f32 = 5.0;
/// The `stuck` column: the stuck ticks (bits 0 to 3), the try's ticks
/// left (4 to 8), the side memory in steps of 8 ticks (9 to 12), the
/// remembered side (13, set for his left), squeezing (14; then bits 0 to
/// 8 count his ticks in the squeeze instead, since a squeezing man is
/// neither counting nor trying), and whether the opening of his last wide
/// look fits only his depth (15, `choose_opening`).
const STUCK_COUNT: u16 = 0x000f;
const STUCK_TRY_SHIFT: u16 = 4;
const STUCK_TRY: u16 = 0x1f << STUCK_TRY_SHIFT;
const STUCK_MEM_SHIFT: u16 = 9;
const STUCK_MEM: u16 = 0x0f << STUCK_MEM_SHIFT;
pub(crate) const STUCK_LEFT: u16 = 1 << 13;
const STUCK_SQUEEZE: u16 = 1 << 14;
const STUCK_SQUEEZE_TICKS: u16 = 0x01ff;
const STUCK_WIDE_DEPTH: u16 = 1 << 15;
/// The stuck ticks and the try's ticks left packed in `stuck`.
pub(crate) fn stuck_count(v: u16) -> u16 {
    if squeezing(v) { 0 } else { v & STUCK_COUNT }
}
pub(crate) fn stuck_trying(v: u16) -> u16 {
    if squeezing(v) { 0 } else { (v & STUCK_TRY) >> STUCK_TRY_SHIFT }
}
/// Whether he is squeezing through a crack, and for how many ticks.
#[inline]
pub(crate) fn squeezing(v: u16) -> bool {
    v & STUCK_SQUEEZE != 0
}
fn squeeze_ticks(v: u16) -> u16 {
    if squeezing(v) { v & STUCK_SQUEEZE_TICKS } else { 0 }
}

/// The squeeze: a man going through a crack narrower
/// than he is wide turns his shoulders and is SQUEEZE_DEPTH across
/// (`turn_level`). It lasts from the crack's mouth until he is through:
/// it ends once he has no body within his normal width at each side
/// (less BETWEEN_SLACK, within a metre along his way) and either he aims
/// at no crack or he moves freely at nine tenths of the advance pace with
/// no crack within his spring, after TRY_MIN ticks, or after
/// SQUEEZE_STALL ticks with no progress, whatever is beside him: a man
/// wedged in a press turns back square. While
/// it lasts his legs push along his wish with their full effort, not
/// faded by the crowd's brake, and his speed is capped at SQUEEZE_CAP,
/// a walk.
pub(super) const BETWEEN_SLACK: f32 = 0.05;
const SQUEEZE_STALL: u16 = 60;
pub(super) const SQUEEZE_CAP: f32 = 2.0;
/// The choice of opening: a man going somewhere, with no
/// enemy in his reach and his regiment not fighting (the try's own test),
/// reads the comrades ahead of him within OPEN_SEE (the scan's same-team
/// reach) and aims at an opening between them that fits him, before he
/// reaches them. A comrade of a regiment under an order who moves his way
/// faster than QUEUE_SPEED is a queue he follows, not an obstacle; a
/// holding man is an obstacle whatever a shove does to him, and a man
/// squeezing who has stopped is a body like any other. Of the obstacles he takes the row nearest him (the men within
/// OPEN_BAND along of the nearest one, a rank's pitch and a bit, so the
/// two files of a pair offset by a rank read as one row) and in it the
/// nearest two on each side of his way. The air between neighbouring men
/// of the row fits his width (his normal body, less OPEN_SLACK), else his
/// depth (the squeeze), else nothing. For each fitting opening his centre
/// goes to its point nearest his way; the nearest wins, a squeeze counting
/// as SQUEEZE_DETOUR metres of detour, the side he chose last worth
/// OPEN_SIDE_KEEP, his own coin breaking a tie. His drive keeps its pace
/// and turns toward that point, OPEN_MIN_AHEAD ahead at least. The near
/// read takes only openings between two men it has seen. When nothing
/// near fits and he saw an obstacle, he looks out to OPEN_FAR once in
/// OPEN_LOOK_TICKS, in his own rhythm, where the open ground beyond the
/// last man on each side counts too, and keeps what he found (`aim`)
/// until his next look, until it is behind him or until his near read
/// fits. A man he follows still takes up his space: a crack that fits
/// only his depth with any comrade inside its passage is taken, and he
/// does not turn his shoulders for it, though he may aim at it, which is
/// following.
pub(super) const OPEN_SEE: f32 = SEP_RADIUS;
const OPEN_FAR: f32 = 4.0;
const OPEN_BAND: f32 = 1.5;
pub(super) const OPEN_AHEAD: f32 = 0.05;
pub(super) const QUEUE_SPEED: f32 = 0.3;
const SQUEEZE_DETOUR: f32 = 1.0;
const OPEN_LOOK_TICKS: u32 = 8;
/// The comrades the wide look keeps from its one pass over the grid.
const WIDE_SEEN: usize = 96;
const OPEN_SLACK: f32 = 0.02;
const OPEN_SIDE_KEEP: f32 = 0.2;
const OPEN_MIN_AHEAD: f32 = 0.3;

#[inline]
pub(super) fn opening_on() -> bool {
    static O: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *O.get_or_init(|| crate::util::env_or("FL_OPENING", 1u32) != 0)
}

/// The row nearest a man: the nearest two obstacles on each side of his
/// way by their offset across it (left positive), each with his half
/// body and his far side along the way (his offset along it plus his half
/// body).
#[derive(Default)]
pub(super) struct Row {
    left: [(f32, f32, f32); 2],
    right: [(f32, f32, f32); 2],
    n_left: u8,
    n_right: u8,
}
impl Row {
    #[inline]
    pub(super) fn add(&mut self, lat: f32, half: f32, along: f32) {
        let (side, n, key) = if lat >= 0.0 {
            (&mut self.left, &mut self.n_left, lat)
        } else {
            (&mut self.right, &mut self.n_right, -lat)
        };
        let man = (key, half, along + half);
        if *n == 0 || key < side[0].0 {
            side[1] = side[0];
            side[0] = man;
            *n = (*n + 1).min(2);
        } else if *n == 1 || key < side[1].0 {
            side[1] = man;
            *n = 2;
        }
    }
}

/// The openings a man sees in a row at `row_at` along his way, out to
/// `see`: ahead between the nearest man on each side, left between the
/// two nearest on his left, right between the two on his right, and, when
/// `beyond` (the wide look), the open ground past the last man seen on a
/// side up to the edge of his sight at that distance. An opening counts
/// only where `free(across, along)` says his body can walk the straight
/// way to its aim (`way_free`): men standing beside a tower leave cracks
/// that lead into its stone. Returns the aim across and along his way,
/// its fit (0 none, 1 his depth, 2 his width), and the picked opening's
/// air across his way with the far side of its passage along it (the
/// farther of its two men's far sides).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn openings(
    row: &Row,
    row_at: f32,
    see: f32,
    body: f32,
    depth: f32,
    beyond: bool,
    pref: f32,
    coin: f32,
    free: impl Fn(f32, f32) -> bool,
) -> (f32, f32, u8, (f32, f32, f32)) {
    let none = (f32::NAN, f32::NAN, f32::NAN);
    if row.n_left == 0 && row.n_right == 0 {
        return (0.0, 0.0, 0, none);
    }
    let edge = (see * see - row_at.min(see) * row_at.min(see)).max(0.0).sqrt();
    let far = if beyond { edge } else { f32::NAN };
    let (l1, l1r, l1f) = if row.n_left >= 1 { row.left[0] } else { none };
    let (r1, r1r, r1f) = if row.n_right >= 1 { row.right[0] } else { none };
    let (hi_l, l2f) = if row.n_left >= 2 { (row.left[1].0 - row.left[1].1, row.left[1].2) } else { (far, f32::NAN) };
    let (lo_r, r2f) = if row.n_right >= 2 { (-(row.right[1].0 - row.right[1].1), row.right[1].2) } else { (-far, f32::NAN) };
    // f32::max takes the number over a NaN: a side with one man seen has
    // his far side.
    let ahead = (
        if row.n_right >= 1 { -(r1 - r1r) } else { -far },
        if row.n_left >= 1 { l1 - l1r } else { far },
        true,
        l1f.max(r1f),
    );
    let left = (l1 + l1r, hi_l, row.n_left >= 1, l1f.max(l2f));
    let right = (lo_r, -(r1 + r1r), row.n_right >= 1, r1f.max(r2f));
    let sign = |x: f32| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 };
    let (mut best, mut aim_lat, mut fit, mut span) = (f32::INFINITY, 0.0, 0u8, none);
    for (lo, hi, ok, pass) in [ahead, left, right] {
        if !ok {
            continue;
        }
        // NaN air (an edge the near read does not see) fits nothing.
        let air = hi - lo;
        let fw = air >= body - OPEN_SLACK;
        let fd = !fw && air >= depth - OPEN_SLACK;
        if !(fw || fd) {
            continue;
        }
        let r = 0.5 * if fw { body } else { depth };
        let aim = (lo + r).max(0.0).min(hi - r);
        let aim = if aim.is_finite() { aim } else { 0.0 };
        let cost = aim.abs() + if fd { SQUEEZE_DETOUR } else { 0.0 }
            - if pref != 0.0 && sign(aim) == pref { OPEN_SIDE_KEEP } else { 0.0 }
            - if sign(aim) == coin { 0.01 } else { 0.0 };
        if cost < best && free(aim, row_at.min(see)) {
            best = cost;
            aim_lat = aim;
            fit = if fw { 2 } else { 1 };
            span = (lo, hi, pass);
        }
    }
    (aim_lat, if fit > 0 { row_at.min(see) } else { 0.0 }, fit, span)
}

/// Whether a body of his radius can walk the straight way from `p` to
/// `p + to` without meeting blocked ground: `Terrain::body_free` at every
/// fine cell along it (every half metre where `p` is off the wall's fine
/// ground, whose blocked cells are 2 m wide), toward its end. Where the
/// fine ground says nothing blocked is nearer `p` than the way and his
/// radius, one read answers. On a map with no blocked ground every way is free and
/// nothing is read. The choice of opening asks it of the opening it would
/// take (`openings`, the ring read): up to 16 reads of a body's few cells
/// for a crack 4 m off beside stone.
#[inline]
pub(super) fn way_free(terrain: &crate::terrain::Terrain, p: Vec2, to: Vec2) -> bool {
    let r = terrain.body_radius();
    let len = to.length();
    if r == 0.0 || len < 1e-6 {
        return true;
    }
    let on_fine = terrain.fine().and_then(|f| f.line_step(p));
    if let Some(Some(step)) = on_fine
        && step >= len + r
    {
        return true;
    }
    let dir = to / len;
    let stride = if on_fine.is_some() { crate::town_layout::FINE } else { 0.5 };
    let n = (len / stride).ceil().max(1.0) as usize;
    (1..=n).all(|k| {
        let q = p + to * (k as f32 / n as f32);
        terrain.body_free(q.x, q.y, r, dir.x, dir.y)
    })
}

/// A man going somewhere who finds himself stuck tries a side. Stuck is
/// felt, not seen: he wants to walk and has moved at under half his wish,
/// and slower than a step, for STUCK_TICKS with bodies pressed against
/// him ahead (the net push would not do: in a press the men behind cancel
/// the men in front). He
/// tries the side his separation push leans to, the side with more room;
/// when it leans nowhere, the side he tried within the last SIDE_MEMORY,
/// else his own coin: a sidestep at STEP_PACE on top of whatever the
/// crowd leaves of his drive, not faded by the crowd's brake, so a man
/// packed in still shuffles sideways. The step ends as soon as the way
/// ahead of him is clear, after TRY_MIN ticks, and at TRY_TICKS at the
/// latest: he steps into the gap and walks, he does not step through it
/// into the next man, which would bounce him from column to column. Pressed
/// forward and free to slide sideways, he slides into the gap beside the
/// man in his way, and the man behind him does the same a moment later.
/// The blind sidestep is only for a man with no opening to aim at
/// (`choose_opening`): reaching a crack he aims at that fits only his
/// depth, he turns his shoulders at once and squeezes through
/// (`turn_level`). Only a man walking somewhere, under his regiment's
/// order or back to a slot more than TRY_FROM_SLOT off, with no enemy in
/// mind and his regiment not fighting: men walking to a fight and the
/// rear ranks of a fighting regiment keep their own rules (`close_in`),
/// so the melee keeps its own footwork, and a squeeze ends when the
/// fight reaches him. His state rides the `stuck` column (STUCK_*).
#[inline]
pub(super) fn try_a_side(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { orders, engaged, anchors, home, tick, .. } = *f;
    let i = s.i;
    let wish = st.walk.wish;
    let want = wish.length();
    let walking = orders[st.gi].is_some()
        || (anchors[st.gi] + home[i]).distance_squared(st.p) > TRY_FROM_SLOT * TRY_FROM_SLOT;
    let going = walking
        && !engaged[st.gi]
        && !st.dying
        && !st.routed
        && !st.fight.in_melee
        && !st.fight.committed
        && st.foe.best_idx == u32::MAX
        && !st.foe.memo_valid
        && st.fight.join_fp.is_none()
        && want > 0.5;
    if !going {
        *s.stuck = 0;
        // A falling soldier keeps the turn of his shoulders (`fall`).
        if !(st.dying && fall()) {
            turn_shoulders(s);
        }
        return;
    }
    let dir = wish / want;
    let left_dir = Vec2::new(-dir.y, dir.x);
    let v = *s.stuck;
    let mut squeeze = squeeze_ticks(v);
    let mut count = stuck_count(v);
    let mut trying = stuck_trying(v);
    let mut mem = (v & STUCK_MEM) >> STUCK_MEM_SHIFT;
    let mut left = v & STUCK_LEFT != 0;
    let v_along = s.vel.xz().dot(dir);
    // The squeeze (SQUEEZE_STALL above): it ends once no body is at each
    // of his sides and either no crack is ahead or he walks freely, or
    // once he has made no progress for SQUEEZE_STALL ticks, whatever is
    // beside him: a man wedged in a press turns back square.
    if squeeze > 0 {
        let free = v_along >= 0.9 * ADVANCE_PACE && !st.open.gap_near;
        let stalled = squeeze >= SQUEEZE_STALL && v_along < 0.1;
        let through = !st.open.between && squeeze >= TRY_MIN && (!st.open.gap || free);
        squeeze = if through || stalled { 0 } else { (squeeze + 1).min(STUCK_SQUEEZE_TICKS) };
    }
    if squeeze > 0 {
        // His legs at their full effort along his wish: he pushes with
        // all he has, at a walk at most (`steer`).
        st.desired = wish;
        st.walk.fade = Vec2::ZERO;
    } else if trying > 0 {
        trying -= 1;
        // The way ahead clear after the first TRY_MIN ticks: the step is
        // done, he walks.
        if trying <= TRY_TICKS - TRY_MIN && st.scan.ahead < AHEAD_MIN {
            trying = 0;
        }
        let side = if left { left_dir } else { -left_dir };
        st.desired += side * STEP_PACE;
        st.walk.side = side * STEP_PACE;
        mem = SIDE_MEMORY;
    } else {
        // Stuck: pressed, and not even stepping. Under half his wish
        // alone would count a lane walker at the sprint behind a slower man
        // as stuck at a jog, and he would zigzag down the lane. A crack he aims at
        // that fits only his depth presses him too.
        let pressed = st.scan.ahead > AHEAD_MIN || st.open.gap;
        let stuck = pressed && v_along < (0.5 * want).min(STEP_PACE);
        count = if stuck { (count + 1).min(STUCK_TICKS) } else { count.saturating_sub(1) };
        // He turns his shoulders as he reaches the crack he aims at, no
        // stuck wait; the blind sidestep only with no opening to aim at.
        // A straggler of a holding regiment whose aim has not moved him
        // (stuck at its top, a ring or wide aim into a standing man's
        // spring) drops it and tries a side; the ring reads again after
        // the step. Passers keep their aims: a lane queue stands still too.
        let dead_aim = st.open.aim_fit != 0
            && orders[st.gi].is_none()
            && (anchors[st.gi] + home[i]).distance_squared(st.p) > TRY_FROM_SLOT * TRY_FROM_SLOT;
        let start = count >= STUCK_TICKS && !st.open.gap_near && (st.open.aim_fit == 0 || dead_aim);
        if start {
            if dead_aim {
                *s.aim = Vec2::NAN;
                *s.sight &= !SIGHT_RING;
            }
            let lean = st.scan.push.dot(left_dir);
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
        } else {
            if st.open.gap_near {
                squeeze = 1;
                count = 0;
            }
            if mem > 0 && tick.is_multiple_of(8) {
                mem -= 1;
            }
        }
    }
    let low = if squeeze > 0 { STUCK_SQUEEZE | squeeze } else { count | (trying << STUCK_TRY_SHIFT) };
    *s.stuck = low | (mem << STUCK_MEM_SHIFT) | (left as u16 * STUCK_LEFT) | (v & STUCK_WIDE_DEPTH);
    turn_shoulders(s);
}

/// The turn of his shoulders follows his squeeze a step a tick
/// (`turn_level`): toward turned while he squeezes, back toward square
/// once he does not.
#[inline]
fn turn_shoulders(s: &mut Soldier) {
    let turn = *s.pose >> POSE_TURN_SHIFT;
    let turn = if squeezing(*s.stuck) { (turn + 1).min(POSE_READY) } else { turn.saturating_sub(1) };
    *s.pose = (*s.pose & POSE_STANCE) | (turn << POSE_TURN_SHIFT);
}

/// Two things a man in melee does not do for his slot: step back away
/// from his regiment's fight to dress, or press into the comrade between
/// him and his mark. A man walking to the fight (`joining`) is not going
/// to his slot, so neither touches his wish; they act on his kept slot
/// wish, for the tick he comes back into formation.
#[inline]
pub(super) fn hold_the_frame(f: &Field, st: &mut Step) {
    let Field { contact, form_face, orders, moving, .. } = *f;
    let gi = st.gi;
    let routed = st.routed;
    let slot_blocked = st.sight.slot_blocked;
    let mut desired = if st.fight.joining { st.fight.slot_wish } else { st.desired };
    // A regiment fighting on its contact frame never steps backward, away
    // from its fight, to dress its ranks: a man ahead of his mark (bunched up
    // behind a stopped front, or shoved forward) stays; he only steps forward
    // or sideways to it. Without this a charging block walked backward to
    // reopen its ranks the moment it made contact. A regiment struck from
    // behind has its fight at its back, so it keeps stepping back to hold its
    // ground.
    // Under the friction contact the same holds for a regiment under an
    // Attack order that has not reached its fight yet: its soldiers who
    // ran in and piled up behind a friend stay where they are, they do not
    // walk backward to their places and open the pile again (`pile_stays`).
    // A man walking the route round blocked ground (`Walk::rounding`) is
    // not going to his place: it is the way, not a step back to dress.
    let attacking = pile_stays() && orders[gi].is_some() && !moving[gi] && !st.walk.rounding;
    if (contact[gi] || attacking) && !routed && form_face[gi] != Vec2::ZERO {
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
    if st.fight.joining {
        st.fight.slot_wish = desired;
    } else {
        st.desired = desired;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::{build_terrain, MapKind};
    use crate::town_layout::{TOWERS_X, WALL_Z};

    /// A man standing a metre off the east end tower's drum on the outer
    /// side cannot walk the straight way to an aim 3 m on, into the drum;
    /// the way along the drum's side, and the way back out, are open.
    #[test]
    fn a_way_into_a_tower_is_not_free() {
        let terrain = build_terrain(MapKind::WallDebug);
        let centre = Vec2::new(TOWERS_X[3], WALL_Z);
        let out = Vec2::new(0.6, 0.8);
        let p = centre + out * 9.3;
        assert!(terrain.body_free(p.x, p.y, terrain.body_radius(), 0.0, 0.0), "he stands on open ground");
        assert!(!way_free(&terrain, p, -out * 3.0), "the way into the drum is free");
        let along = Vec2::new(-out.y, out.x);
        assert!(way_free(&terrain, p + out * 0.5, along * 2.0), "the way along the drum is blocked");
        assert!(way_free(&terrain, p, out * 3.0), "the way out is blocked");
    }
}
