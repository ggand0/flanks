//! The neighbour scan: the setting apart, the crowd around him, the enemy
//! in his reach and the spear points he runs onto.

use bevy::math::Vec3Swizzles;
use bevy::prelude::*;

use super::*;
use crate::sim::damage::DamageEvent;
use crate::unit_types::TYPES;

/// Base neighbor query radius: covers separation and sword reach. Kinds
/// with longer reach (spears) widen their own scan per unit — the sword
/// majority must not pay for the spear's reach.
pub(super) const QUERY_RADIUS: f32 = 2.0;
/// The widest pair's spring rest over the pair table (`pair_widths`),
/// for the scan's cut of its foes.
fn widest_rest() -> f32 {
    static R: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *R.get_or_init(|| pair_widths().iter().flatten().fold(0.0f32, |r, w| r.max(w.sep)))
}
/// The scan's cut of its foes (`scan`): metres past the farthest a foe can
/// be and still count, far above the float rounding of a position on the map.
const CUT_MARGIN: f32 = 0.01;
/// The enemy he remembers (`target` at the tick's start) as one still to
/// go for: a living index of the other team within his sight plus a
/// metre, and with his regiment focused on one target, a man of it or
/// one within his threat distance. Returns the squared distance to him
/// (f32::MAX for an index out of range or of his own team) and whether he
/// counts. The scan's `memo_valid` and the join in `drive` both read it.
#[inline]
pub(super) fn remembered_foe(f: &Field, i: usize, p: Vec2, gi: usize, memo: usize) -> (f32, bool) {
    let Field { pos_prev, team, group, focus, .. } = *f;
    if memo >= pos_prev.len() || team[memo] == team[i] {
        return (f32::MAX, false);
    }
    let d2 = pos_prev[memo].xz().distance_squared(p);
    let ok = d2 < (seek_radius() + 1.0) * (seek_radius() + 1.0)
        && focus[gi].is_none_or(|t| group[memo] == t || d2 < threat_radius() * threat_radius());
    (d2, ok)
}
/// How deep another body is in his own, from the touch (the pair's body
/// distance) on, at which it presses on him in full for the crowd walk (m).
/// Of the soldiers running in a packed mass, 85 in 100 have a body at
/// least this deep in theirs on every side; of the soldiers moving on a
/// stair or a wall crossing, 2 to 12 in 100.
const CROWD_DEPTH: f32 = 0.1;
/// How much of a body lies on each of his four sides about his wish
/// (ahead, behind, left, right): 1 within 30 degrees of the side's
/// direction, falling to nothing at 60 degrees, so a body between two
/// sides counts in part for both and none of the four jumps as it moves
/// around him. `n` is the unit vector from the body to him, `wish_dir` his
/// wish's direction.
#[inline]
fn press_shares(n: Vec2, wish_dir: Vec2) -> [f32; 4] {
    let along = -n.dot(wish_dir);
    let lat = wish_dir.y * n.x - wish_dir.x * n.y;
    let share = |c: f32| ((c - 0.5) * (1.0 / (0.866_025_4 - 0.5))).clamp(0.0, 1.0);
    [share(along), share(-along), share(lat), share(-lat)]
}
/// The still-man brake's crawl (m/s): a man ahead who moved along the
/// reader's way at this speed or faster last tick counts nothing.
const CRAWL: f32 = 0.3;
/// How still a man ahead stands for the brake: 1 when he stood or came at
/// the reader, nothing when he moved along the reader's way at CRAWL.
#[inline]
fn still_weight(o_vel: Vec2, wish_dir: Vec2) -> f32 {
    (1.0 - o_vel.dot(wish_dir) / CRAWL).clamp(0.0, 1.0)
}
/// Positional overlap resolution: the share of a pair's overlap each man
/// is moved back per tick, and the per-tick cap on total correction
/// distance. A packed man overlaps up to six bodies and their corrections
/// add up, so the share stays under one sixth: over it, six bodies throw
/// him past the middle of them and he comes back on the next tick, every
/// tick. 90% of an overlap is gone in 7 ticks.
pub(super) const CORR_GAIN: f32 = 0.15;
pub(super) const CORR_MAX: f32 = 0.1;
/// The count an enemy close to him sets: 31 steps of four ticks, about
/// 4 s at 30 ticks a second, before he puts the knife away.
const KNIFE_HOLD: u16 = 31;
/// How close an enemy block's edge comes before he takes his knife out:
/// about two seconds of a charge, one bow draw.
pub(crate) const KNIFE_EDGE: f32 = 10.0;
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

/// The fused neighbour scan every man runs every tick over the 2 m box:
/// separation pushes and overlap corrections, the crowd around him, the
/// nearest living enemy in his weapon's reach (sticky to the one he already
/// fights), and, at charge speed against a spearwall, the braced point his
/// body runs onto, which is a damage event from the spearman.
#[inline]
pub(super) fn scan(f: &Field, s: &mut Soldier, st: &mut Step, out: &mut ChunkOut, cands: &mut Vec<u32>) {
    let Field {
        grid,
        speed,
        team,
        yaw_snap,
        wall,
        stance,
        faces_spearwall,
        tick_seed,
        orders,
        engaged,
        anchors,
        home,
        passing,
        moving,
        ..
    } = *f;
    let i = s.i;
    let params = &TYPES[st.my_kind];
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    // Fused neighbor scan: separation physics + nearest living enemy in reach
    // (swing targeting), over the grid's cache-ordered units within the cut.
    let mut push = Vec2::ZERO;
    let mut corr = Vec2::ZERO;
    let mut overlap_vel = Vec2::ZERO;
    let mut overlap_share = 0.0f32;
    let mut overlap = 0.0f32;
    let mut crowd = 0.0f32;
    let mut crowd_still = 0.0f32;
    let mut ahead = 0.0f32;
    let wish_dir = st.desired.normalize_or_zero();
    // The still men ahead are read only for the friction contact's brake,
    // and only by a man who wishes to go somewhere.
    let still_on = still_brake() && st.desired.length_squared() >= 1e-12;
    // The sides pressed against him, for the crowd walk under the
    // friction contact: only a soldier who wishes to go somewhere has sides
    // about his wish.
    let sides_on = friction_contact() && st.desired.length_squared() >= 1e-12;
    let mut press4 = [0.0f32; 4];
    let inv_depth = 1.0 / CROWD_DEPTH;
    let my_team_bit = (team[i] as u32) * crate::spatial::META_TEAM;
    let my_wall = wall[gi] != 0;
    // His width and each neighbour's by their poses and the turn of their
    // shoulders at the tick's start, the pair's rest, its inverse and its
    // overlap distance from a table, and the pair's mass weights by kind.
    let widths = widths();
    let squeeze_on = squeezing(*s.stuck);
    let pose_lv = pose_level(*s.pose);
    let my_level = width_level(*s.pose);
    let mine = &widths[my_level];
    // The choice of opening's read (`choose_opening`): only a man walking
    // somewhere, not fighting, reads the comrades ahead; a man squeezing
    // feels the bodies at his sides.
    let body0 = widths[pose_lv].body;
    let far = (anchors[gi] + home[i]).distance_squared(p) > TRY_FROM_SLOT * TRY_FROM_SLOT;
    let walking = orders[gi].is_some() || far;
    // Only a man with a reason to pick an opening: his regiment walking
    // through a friend that opened for it, or himself a straggler of a
    // holding regiment. Read for every walking man, it costs 5.6 ms of
    // the 200k step in a front battle where nobody passes anyone.
    let reason = passing[gi] || (orders[gi].is_none() && far);
    let open_on = opening_on()
        && walking
        && reason
        && !engaged[gi]
        && !dying
        && !st.routed
        && !st.in_melee
        && !st.committed
        && st.join_fp.is_none()
        && st.desired.length_squared() > 0.25;
    // Every obstacle within his reach is in the row nearest him (the band
    // reaches past OPEN_SEE), so the scan keeps the nearest two on each
    // side as it goes.
    let mut row = Row::default();
    let mut row_at = f32::INFINITY;
    let (mut side_left, mut side_right) = (false, false);
    let pairs = &pair_widths()[my_level];
    let masses = &mass_pairs()[st.my_kind];
    // Under the friction contact the setting apart weighs each man by how
    // braced he is (`brace_pairs`): his own level from his pose and his
    // stood bit at the tick's start, the ones the grid carries for him, and
    // each neighbour's from the grid, so both men of a pair read the same
    // two levels. The walking bracing counts only against the other side
    // (`brace_level`), so his level against a friend and against a foe are
    // kept apart.
    let friction = friction_contact();
    // A falling soldier's own share of his body, from his clock at the
    // tick's start as the grid carries it (his own column has already
    // counted down this tick, `begin`).
    let fall_on = fall();
    let my_share = if fall_on { fall_share(f.death_snap[i]) } else { 1.0 };
    let my_stood = *s.stood;
    let (my_brace_same, my_brace_cross) = if friction {
        (
            brace_level(pose_lv, my_stood, dying, fall_on, false, || moving[gi]),
            brace_level(pose_lv, my_stood, dying, fall_on, true, || moving[gi]),
        )
    } else {
        (0, 0)
    };
    let braces = if friction { Some(&brace_pairs()[st.my_kind]) } else { None };
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
    // The farthest a foe can be and still count for anything below: the
    // widest pair's spring rest, his weapon's reach, or at charge speed a
    // braced spear's point. A friend counts only within SEP_RADIUS.
    let spear_cut = if at_charge_speed {
        (spear_reach * spear_reach + SPEAR_LINE_HALF_W * SPEAR_LINE_HALF_W).sqrt()
    } else {
        0.0
    };
    let foe_cut = widest_rest().max(params.reach).max(spear_cut) + CUT_MARGIN;
    // First the candidates within the cut, by rank, in the walk's order:
    // in a press, whether each passes is hard to predict, so each is written and
    // kept by the test's outcome instead of by a branch on it. A friend
    // beyond SEP_RADIUS can touch nothing of his, and most of a scan's
    // candidates in a press are such men (the cells span more than the
    // circle); a foe beyond `foe_cut` fails every foe test. The box stays
    // `scan_r`'s, which the look around him reads again (`look_around`).
    cands.clear();
    cands.reserve(grid.box_len(p, scan_r));
    let cut = crate::spatial::KeepCut {
        team_bit: my_team_bit,
        me: i as u32,
        same_cut2: SEP_RADIUS * SEP_RADIUS,
        foe_cut2: foe_cut * foe_cut,
    };
    // SAFETY: `box_len` counts every man of the box the walk visits, and
    // the list has room for all of them; the first `n_kept` are written.
    unsafe {
        let n_kept = grid.kept_in_box(p, scan_r, &cut, cands.as_mut_ptr());
        cands.set_len(n_kept);
    }
    let sorted = grid.sorted();
    for &o_rank in cands.iter() {
        let o = &sorted[o_rank as usize];
        let o_vel = grid.sorted_vel(o_rank as usize);
        let d = p - o.xz();
        let d2 = d.length_squared();
        let same = (o.meta & crate::spatial::META_TEAM) == my_team_bit;
        let enemy = !same && (o.meta & crate::spatial::META_DYING) == 0;
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
        // The comrades ahead for the choice of opening: a queue (a man of
        // a regiment under an order moving his way) is not an obstacle; a
        // holding man always is.
        if same && (o.meta & crate::spatial::META_DYING) == 0 && (open_on | squeeze_on) {
            let to = -d;
            let along = to.dot(wish_dir);
            let lat = wish_dir.x * to.y - wish_dir.y * to.x;
            let o_body = widths[crate::spatial::meta_pose(o.meta)].body;
            if open_on && along > OPEN_AHEAD {
                let holding = orders[crate::spatial::meta_group(o.meta)].is_none();
                let queue = !holding && o_vel.dot(wish_dir) > QUEUE_SPEED;
                if !queue {
                    row.add(lat, 0.5 * o_body, along);
                    row_at = row_at.min(along);
                }
            }
            if squeeze_on {
                let near = 0.5 * (body0 + o_body) - BETWEEN_SLACK;
                if d2 < near * near && along.abs() < 1.0 {
                    side_left |= lat > 0.0;
                    side_right |= lat < 0.0;
                }
            }
        }
        // A pair with a falling soldier in it (`fall`) is as far apart as
        // what is left of his body: each man's body and rest by his share,
        // a wall pair's rest too. Two living soldiers read the table.
        let o_dying = (o.meta & crate::spatial::META_DYING) != 0;
        let falling = fall_on && (dying | o_dying);
        let fallen;
        let (pair, wall_rest) = if falling {
            let o_share = fall_share(crate::spatial::meta_fall(o.meta));
            fallen = pair_width_of(my_level, my_share, crate::spatial::meta_width(o.meta), o_share);
            (&fallen, WALL_SEP_RADIUS * (0.5 * (my_share + o_share)))
        } else {
            (&pairs[crate::spatial::meta_width(o.meta)], WALL_SEP_RADIUS)
        };
        let (sep_r, inv_sep, hard) = if !cross && my_wall && (o.meta & crate::spatial::META_WALL) != 0 {
            (wall_rest, 1.0 / wall_rest, pair.hard.min(wall_rest))
        } else {
            (pair.sep, pair.inv_sep, pair.hard)
        };
        if d2 < sep_r * sep_r && d2 > 1e-8 {
            // Mass-weighted: heavies shove lights aside.
            let (mw, share) = masses[crate::spatial::meta_kind(o.meta)];
            let len = d2.sqrt();
            let inv_len = 1.0 / len;
            let w = 1.0 - len * inv_sep;
            if len < hard && sides_on {
                // The crowd walk: how deep this body is in his, on the
                // sides of him it lies on (`d` points from it to him).
                let depth = ((hard - len) * inv_depth).min(1.0);
                let shares = press_shares(d * inv_len, wish_dir);
                for q in 0..4 {
                    press4[q] = press4[q].max(depth * shares[q]);
                }
            }
            if len < hard {
                // Overlap is resolved POSITIONALLY only. A hard force boost
                // here as well made packed crowds oscillate at the
                // acceleration cap, two solvers fighting over the same
                // overlap, every frame.
                let back = (hard - len) * gain * mw;
                // The friction contact sets him apart by the braced
                // weights; what the contact speed reads below stays by
                // the plain ones.
                let set = match braces {
                    Some(braces) => {
                        // `moving` is read only for a walking foe
                        // (`brace_level`).
                        let o_brace = brace_level(
                            crate::spatial::meta_pose(o.meta),
                            (o.meta & crate::spatial::META_STOOD) != 0,
                            o_dying,
                            fall_on,
                            cross,
                            || moving[crate::spatial::meta_group(o.meta)],
                        );
                        let my_brace = if cross { my_brace_cross } else { my_brace_same };
                        (hard - len) * gain * braces[crate::spatial::meta_kind(o.meta)][my_brace][o_brace]
                    }
                    None => back,
                };
                corr += d * (set * inv_len);
                overlap_vel += o_vel * back;
                overlap_share += share * back;
                overlap += back;
            }
            push += d * (w * mw * inv_len);
            ahead += w * (d.dot(wish_dir) < -0.5 * len) as u32 as f32;
            // Jam density: COMPRESSED pairs only — a neighbor resting AT his
            // pair's rest distance contests nothing, so a formed man's
            // slot-keeping is never faded by the settled enemies one stride
            // away, which made a line yield and stay ragged.
            crowd += w;
            // The still men ahead: a neighbour on the half-plane ahead of
            // his wish (`d` points from the neighbour to him), by how little
            // he moved along it last tick.
            if still_on {
                let on = (d.dot(wish_dir) < 0.0) as u32 as f32;
                crowd_still += w * on * still_weight(o_vel, wish_dir);
            }
        }
    }

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
    // The enemy he remembers, if still in sight: he tracks him while he has
    // no enemy in reach, and seeing him is what makes a man still in
    // formation react. A man fighting someone in reach out of formation needs
    // neither, and skips the lookup.
    // With his regiment focused on one target (sim/job.rs `focus`), a man
    // of another regiment is remembered only within his threat distance:
    // one who is not on him he lets go, and follows his orders. Read here
    // rather than in `look_around` so that his stance below knows whether
    // he is on his way to his place (`on_his_way`).
    let memo = prev_target as usize;
    let (memo_d2, memo_valid) = if (best_idx == u32::MAX && !dying && !st.routed)
        || (st.in_melee && !st.committed)
    {
        remembered_foe(f, i, p, gi, memo)
    } else {
        (f32::MAX, false)
    };
    st.memo_valid = memo_valid;
    st.memo_d2 = memo_d2;
    st.best_idx = best_idx;
    // His stance for the next tick, as he is drawn: into the ready stance
    // while his regiment fights, while it stands ready and he stands (the
    // pose pass's guard, below the speed of his steps; under the friction
    // contact a push does not count, only his own wish to walk, `stands`),
    // or with an enemy in his reach; else back at ease, walking or
    // running. A step of the ramp a tick. Under the friction contact a
    // soldier on the walk to his place with nothing else to do who was a
    // walker last tick (`steer` keeps it in `stood` until he arrives,
    // `still_stands`) is on his way and stays at ease, whatever his wish
    // this tick. In a fighting regiment (STANCE_FIGHT) every soldier is
    // ready, walking or standing, so a walker there is in the ready stance,
    // his weight his own against his side and the walking bracing against
    // the other side.
    let ready = match stance[gi] {
        STANCE_FIGHT => true,
        STANCE_READY => {
            stands(friction, s.vel.xz(), st.desired) && !(friction && !*s.stood && on_his_way(s, st))
        }
        _ => false,
    };
    // A falling soldier keeps the stance he was struck in (`fall`).
    if !(dying && fall_on) {
        let stance_ramp = *s.pose & POSE_STANCE;
        let stance_ramp = if ready || best_idx != u32::MAX {
            (stance_ramp + 1).min(POSE_READY)
        } else {
            stance_ramp.saturating_sub(1)
        };
        *s.pose = (*s.pose & !POSE_STANCE) | stance_ramp;
    }
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
    st.crowd_still = crowd_still;
    st.ahead = ahead;
    st.press4 = press4;
    st.best_idx = best_idx;
    st.sticky = sticky;
    st.prev_target = prev_target;
    st.impale_idx = impale_idx;
    st.at_charge_speed = at_charge_speed;
    st.scan_r = scan_r;
    st.my_team_bit = my_team_bit;
    st.open_on = open_on;
    st.body0 = body0;
    st.sep0 = widths[pose_lv].sep;
    st.saw = row_at.is_finite();
    if open_on && st.saw {
        let (pref, coin) = side_pref(i, *s.stuck);
        let (aim_lat, aim_along, fit, span) = openings(&row, row_at, OPEN_SEE, body0, SQUEEZE_DEPTH, false, pref, coin);
        st.near = (aim_lat, aim_along, fit);
        st.near_span = span;
    }
    st.between = side_left && side_right;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A body straight ahead, behind, left and right lies on its own side
    /// in full; one between two sides lies on both in part and on the far
    /// sides not at all. `n` points from the body to him, his wish is +y.
    #[test]
    fn press_shares_split_a_body_between_the_sides() {
        let wish = Vec2::Y;
        assert_eq!(press_shares(Vec2::new(0.0, -1.0), wish), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(press_shares(Vec2::new(0.0, 1.0), wish), [0.0, 1.0, 0.0, 0.0]);
        let left = press_shares(Vec2::new(1.0, 0.0), wish);
        let right = press_shares(Vec2::new(-1.0, 0.0), wish);
        assert_eq!(left[0] + left[1], 0.0);
        assert_eq!(right[0] + right[1], 0.0);
        assert_eq!(left[2] + left[3], 1.0);
        assert_eq!(right[2] + right[3], 1.0);
        assert_ne!(left[2], right[2]);
        let d = std::f32::consts::FRAC_1_SQRT_2;
        let between = press_shares(Vec2::new(-d, -d), wish);
        assert!(between[0] > 0.4 && between[0] < 0.7);
        assert_eq!(between[1], 0.0);
        assert!((between[2] + between[3] - between[0]).abs() < 1e-6);
    }

    /// A man ahead who stood or came at the reader counts in full, one who
    /// moved along the reader's way at CRAWL or faster not at all, one at
    /// half the crawl half.
    #[test]
    fn still_weight_reads_the_move_along_the_way() {
        let wish = Vec2::new(0.0, 1.0);
        assert_eq!(still_weight(Vec2::ZERO, wish), 1.0);
        assert_eq!(still_weight(Vec2::new(0.0, -1.0), wish), 1.0);
        assert_eq!(still_weight(Vec2::new(0.0, CRAWL), wish), 0.0);
        assert_eq!(still_weight(Vec2::new(0.0, 2.0), wish), 0.0);
        assert!((still_weight(Vec2::new(5.0, 0.5 * CRAWL), wish) - 0.5).abs() < 1e-6);
    }
}
