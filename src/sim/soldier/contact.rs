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
/// one within his threat distance, and on his own level as the scan reads
/// levels (`Levels`): an enemy who climbed onto the wall above him, or one
/// at its foot below a man on its walk, is out of his reach, and he lets
/// him go. Returns the squared distance to him (f32::MAX for an index out
/// of range or of his own team) and whether he counts. The scan's
/// `memo_valid` and the join in `drive` both read it.
#[inline]
pub(super) fn remembered_foe(f: &Field, i: usize, p: Vec2, gi: usize, memo: usize) -> (f32, bool) {
    let Field { pos_prev, team, group, focus, .. } = *f;
    if memo >= pos_prev.len() || team[memo] == team[i] {
        return (f32::MAX, false);
    }
    let d2 = pos_prev[memo].xz().distance_squared(p);
    let ok = d2 < (seek_radius() + 1.0) * (seek_radius() + 1.0)
        && focus[gi].is_none_or(|t| group[memo] == t || d2 < threat_radius() * threat_radius())
        && !Levels::new(f, i).apart(f, memo as u32);
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
/// body runs onto, which is a damage event from the spearman. One loop over
/// his neighbours does every job: the contact between bodies (`Contact`),
/// the crowd around him (`Crowd`), the enemy in his reach (`Reach`), the
/// spear points (`Spear`), the obstacles for the choice of opening
/// (`Obstacles`) and, for a walker to the fight under the arm's length
/// rule, the bodies inside his arm's length and the comrade in his way
/// (`Arm`).
#[inline]
pub(super) fn scan(f: &Field, s: &mut Soldier, st: &mut Step, out: &mut ChunkOut, cands: &mut Vec<u32>) {
    let i = s.i;
    let params = &TYPES[st.my_kind];
    let p = st.p;
    let dying = st.dying;
    // Fused neighbor scan: separation physics + nearest living enemy in reach
    // (swing targeting), over the grid's cache-ordered units within the cut.
    let wish_dir = st.desired.normalize_or_zero();
    let my_team_bit = (f.team[i] as u32) * crate::spatial::META_TEAM;
    // His width and each neighbour's by their poses and the turn of their
    // shoulders at the tick's start, the pair's rest, its inverse and its
    // overlap distance from a table, and the pair's mass weights by kind.
    let widths = widths();
    let pose_lv = pose_level(*s.pose);
    let my_level = width_level(*s.pose);
    let friction = friction_contact();
    let fall_on = fall();
    let mut contact = Contact::new(f, s, st, my_level, pose_lv, friction, fall_on);
    let mut crowd = Crowd::new(st, wish_dir, friction);
    let mut reach = Reach::new(params, *s.target);
    let mut spear = Spear::new(f, s, dying);
    let mut obstacles = Obstacles::new(f, s, st, wish_dir, widths[pose_lv].body, widths);
    let mut arm = Arm::new(st);
    let scan_r = spear.scan_radius(params);
    gather_candidates(f.grid, p, i, my_team_bit, scan_r, spear.foe_cut(params), cands);
    let grid = f.grid;
    let sorted = grid.sorted();
    let levels = Levels::new(f, i);
    for &o_rank in cands.iter() {
        let o = &sorted[o_rank as usize];
        if levels.apart(f, o.idx) {
            continue;
        }
        let o_vel = grid.sorted_vel(o_rank as usize);
        let d = p - o.xz();
        let d2 = d.length_squared();
        let same = (o.meta & crate::spatial::META_TEAM) == my_team_bit;
        let o_dying = (o.meta & crate::spatial::META_DYING) != 0;
        let n = Near { o, o_vel, d, d2, same, o_dying, enemy: !same && !o_dying };
        reach.see(&n);
        spear.see(&n, f.yaw_snap);
        obstacles.see(&n);
        arm.see(&n);
        let (sep_r, inv_sep, hard) = contact.rest(&n);
        if d2 < sep_r * sep_r && d2 > 1e-8 {
            let len = d2.sqrt();
            let inv_len = 1.0 / len;
            let w = 1.0 - len * inv_sep;
            contact.touch(&n, len, inv_len, w, hard);
            crowd.count(&n, len, inv_len, w, hard);
        }
    }
    spear.impale(f, i, out);
    mark_foe_in_reach(s, dying, reach.best_idx);
    recall_foe(f, st, i, reach.best_idx, reach.prev_target);
    ramp_stance(f, s, st, friction, fall_on, reach.best_idx);
    let mine = &widths[my_level];
    st.scan.brake = (mine.slow, mine.stop);
    draw_knife(f, s, st, reach.best_idx);
    contact.hand_on(st);
    crowd.hand_on(st);
    reach.hand_on(st);
    arm.hand_on(st);
    st.scan.scan_r = scan_r;
    st.scan.my_team_bit = my_team_bit;
    obstacles.hand_on(s, st, widths[pose_lv].sep);
}

/// Whether two things whose ground heights differ by `dy` stand on
/// different levels: the difference is a standing soldier's height or
/// more, the half heights `half_a` and `half_b` of the two men standing
/// there added. Two men on different levels do not overlap in height.
#[inline(always)]
pub(crate) fn levels_apart(dy: f32, half_a: f32, half_b: f32) -> bool {
    dy.abs() >= half_a + half_b
}

/// Two men on different levels: on a map with cliffs (a wall's walk and
/// the ground at its foot, a raised block), a neighbour close in plan may
/// stand a storey above or below. Their bodies meet only where they
/// overlap in height, so a man whose middle is more than the two half
/// heights above or below his own (`levels_apart`) is no body he touches,
/// no man in his reach and no part of the crowd round him. Where no
/// ground blocks there is no cliff and no such pair, and nothing is read.
pub(super) struct Levels {
    on: bool,
    y: f32,
    hh: f32,
}

impl Levels {
    #[inline]
    pub(super) fn new(f: &Field, i: usize) -> Self {
        let on = f.terrain.kind.has_blocked_ground();
        let hh = if on { crate::unit_types::half_height(f.kind[i] as usize) } else { 0.0 };
        Levels { on, y: f.pos_prev[i].y, hh }
    }

    /// Whether man `o` stands on another level from this one.
    #[inline(always)]
    pub(super) fn apart(&self, f: &Field, o: u32) -> bool {
        self.on && {
            let o = o as usize;
            levels_apart(f.pos_prev[o].y - self.y, self.hh, crate::unit_types::half_height(f.kind[o] as usize))
        }
    }
}

/// One neighbour as the scan reads him: his row in the grid and his
/// velocity at the tick's start, `d` from him to the scanning man and its
/// square, whether he is of the man's team, whether he is dying, and
/// whether he is a living enemy.
struct Near<'a> {
    o: &'a crate::spatial::SortedUnit,
    o_vel: Vec2,
    d: Vec2,
    d2: f32,
    same: bool,
    o_dying: bool,
    enemy: bool,
}

/// The candidates of his scan within the cut, by rank, in the walk's
/// order.
#[inline]
fn gather_candidates(
    grid: &crate::spatial::SpatialGrid,
    p: Vec2,
    i: usize,
    my_team_bit: u32,
    scan_r: f32,
    foe_cut: f32,
    cands: &mut Vec<u32>,
) {
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
}

/// The contact between bodies (`scan`): every neighbour within the pair's
/// spring rest pushes him away by their masses, every body that overlaps
/// his sets him back by the overlap correction, by their masses or under
/// the friction contact by how braced each man is, and the bodies he
/// overlaps give the velocity and his share by weight of the shared speed.
struct Contact<'a> {
    pairs: &'a [PairWidth; WIDTH_LEVELS],
    masses: &'a [(f32, f32); crate::unit_types::NUM_KINDS],
    braces: Option<&'a BraceRow>,
    moving: &'a [bool],
    my_level: usize,
    my_share: f32,
    my_wall: bool,
    my_brace_same: usize,
    my_brace_cross: usize,
    dying: bool,
    fall_on: bool,
    push: Vec2,
    corr: Vec2,
    overlap_vel: Vec2,
    overlap_share: f32,
    overlap: f32,
}

impl<'a> Contact<'a> {
    /// His side of every pair: his width level, his share of his body, his
    /// wall stance and his brace levels.
    #[inline]
    fn new(
        f: &Field<'a>,
        s: &Soldier,
        st: &Step,
        my_level: usize,
        pose_lv: usize,
        friction: bool,
        fall_on: bool,
    ) -> Self {
        let Field { wall, moving, .. } = *f;
        let gi = st.gi;
        let dying = st.dying;
        // Under the friction contact the setting apart weighs each man by how
        // braced he is (`brace_pairs`): his own level from his pose and his
        // stood bit at the tick's start, the ones the grid carries for him, and
        // each neighbour's from the grid, so both men of a pair read the same
        // two levels. The walking bracing counts only against the other side
        // (`brace_level`), so his level against a friend and against a foe are
        // kept apart.
        // A falling soldier's own share of his body, from his clock at the
        // tick's start as the grid carries it (his own column has already
        // counted down this tick, `begin`).
        let my_share = if fall_on { fall_share(f.death_snap[s.i]) } else { 1.0 };
        let my_stood = *s.stood;
        let (my_brace_same, my_brace_cross) = if friction {
            (
                brace_level(pose_lv, my_stood, dying, fall_on, false, || moving[gi]),
                brace_level(pose_lv, my_stood, dying, fall_on, true, || moving[gi]),
            )
        } else {
            (0, 0)
        };
        Contact {
            pairs: &pair_widths()[my_level],
            masses: &mass_pairs()[st.my_kind],
            braces: if friction { Some(&brace_pairs()[st.my_kind]) } else { None },
            moving,
            my_level,
            my_share,
            my_wall: wall[gi] != 0,
            my_brace_same,
            my_brace_cross,
            dying,
            fall_on,
            push: Vec2::ZERO,
            corr: Vec2::ZERO,
            overlap_vel: Vec2::ZERO,
            overlap_share: 0.0,
            overlap: 0.0,
        }
    }

    /// The pair's spring rest, its inverse and the distance under which
    /// their bodies overlap.
    #[inline(always)]
    fn rest(&self, n: &Near) -> (f32, f32, f32) {
        let o = n.o;
        // Two same-team units both in a wall STANCE rest shoulder to shoulder
        // (symmetric predicate: both sides compute the same radius). Ordinary
        // ranks keep parade spacing even while fighting — the seams between
        // files are what enemy bodies flow into.
        let cross = !n.same;
        // No packing rule for fighting regiments: M2TW keeps its formation
        // grid during melee, and observably loosens it. The fighting crowd's
        // spacing is slots plus body collision, nothing else. A
        // shoulder-to-shoulder press rest for fighting pairs sealed the very
        // seams the intermix needs: a pressed front's gaps shrank to about
        // 1.05 m against 0.95 m bodies and symmetric fights collapsed to a
        // two-rank duel line. It must not come back.
        // A pair with a falling soldier in it (`fall`) is as far apart as
        // what is left of his body: each man's body and rest by his share,
        // a wall pair's rest too. Two living soldiers read the table.
        let falling = self.fall_on && (self.dying | n.o_dying);
        let fallen;
        let (pair, wall_rest) = if falling {
            let o_share = fall_share(crate::spatial::meta_fall(o.meta));
            fallen = pair_width_of(self.my_level, self.my_share, crate::spatial::meta_width(o.meta), o_share);
            (&fallen, WALL_SEP_RADIUS * (0.5 * (self.my_share + o_share)))
        } else {
            (&self.pairs[crate::spatial::meta_width(o.meta)], WALL_SEP_RADIUS)
        };
        if !cross && self.my_wall && (o.meta & crate::spatial::META_WALL) != 0 {
            (wall_rest, 1.0 / wall_rest, pair.hard.min(wall_rest))
        } else {
            (pair.sep, pair.inv_sep, pair.hard)
        }
    }

    /// A neighbour within the pair's rest, `len` from him: his push, and
    /// with the bodies overlapping, the correction and the overlap's
    /// velocity and share.
    #[inline(always)]
    fn touch(&mut self, n: &Near, len: f32, inv_len: f32, w: f32, hard: f32) {
        let o = n.o;
        let d = n.d;
        let cross = !n.same;
        // Mass-weighted: heavies shove lights aside.
        let (mw, share) = self.masses[crate::spatial::meta_kind(o.meta)];
        if len < hard {
            // Overlap is resolved POSITIONALLY only. A hard force boost
            // here as well made packed crowds oscillate at the
            // acceleration cap, two solvers fighting over the same
            // overlap, every frame.
            let back = (hard - len) * CORR_GAIN * mw;
            // The friction contact sets him apart by the braced
            // weights; what the contact speed reads below stays by
            // the plain ones.
            let set = match self.braces {
                Some(braces) => {
                    // `moving` is read only for a walking foe
                    // (`brace_level`).
                    let o_brace = brace_level(
                        crate::spatial::meta_pose(o.meta),
                        (o.meta & crate::spatial::META_STOOD) != 0,
                        n.o_dying,
                        self.fall_on,
                        cross,
                        || self.moving[crate::spatial::meta_group(o.meta)],
                    );
                    let my_brace = if cross { self.my_brace_cross } else { self.my_brace_same };
                    (hard - len) * CORR_GAIN * braces[crate::spatial::meta_kind(o.meta)][my_brace][o_brace]
                }
                None => back,
            };
            self.corr += d * (set * inv_len);
            self.overlap_vel += n.o_vel * back;
            self.overlap_share += share * back;
            self.overlap += back;
        }
        self.push += d * (w * mw * inv_len);
    }

    /// The push and the correction, and the overlap's velocity and share as
    /// means over the overlap.
    #[inline]
    fn hand_on(&self, st: &mut Step) {
        st.scan.push = self.push;
        st.scan.corr = self.corr;
        if self.overlap > 0.0 {
            st.scan.overlap_vel = self.overlap_vel / self.overlap;
            st.scan.overlap_share = self.overlap_share / self.overlap;
        }
    }
}

/// The crowd around him (`scan`): how close his neighbours stand within
/// the pair's rest (`crowd`), the crowd ahead of him on his way (`ahead`),
/// the still men ahead (`crowd_still`, the friction contact's brake) and
/// how deep a body is in his on each of his four sides about his wish
/// (`press4`, the crowd walk).
struct Crowd {
    wish_dir: Vec2,
    still_on: bool,
    sides_on: bool,
    inv_depth: f32,
    crowd: f32,
    crowd_still: f32,
    ahead: f32,
    press4: [f32; 4],
}

impl Crowd {
    #[inline]
    fn new(st: &Step, wish_dir: Vec2, friction: bool) -> Self {
        // The still men ahead are read only for the friction contact's brake,
        // and only by a man who wishes to go somewhere.
        let still_on = still_brake() && st.desired.length_squared() >= 1e-12;
        // The sides pressed against him, for the crowd walk under the
        // friction contact: only a soldier who wishes to go somewhere has sides
        // about his wish.
        let sides_on = friction && st.desired.length_squared() >= 1e-12;
        Crowd {
            wish_dir,
            still_on,
            sides_on,
            inv_depth: 1.0 / CROWD_DEPTH,
            crowd: 0.0,
            crowd_still: 0.0,
            ahead: 0.0,
            press4: [0.0; 4],
        }
    }

    /// A neighbour within the pair's rest, `len` from him.
    #[inline(always)]
    fn count(&mut self, n: &Near, len: f32, inv_len: f32, w: f32, hard: f32) {
        let d = n.d;
        let wish_dir = self.wish_dir;
        if len < hard && self.sides_on {
            // The crowd walk: how deep this body is in his, on the
            // sides of him it lies on (`d` points from it to him).
            let depth = ((hard - len) * self.inv_depth).min(1.0);
            let shares = press_shares(d * inv_len, wish_dir);
            let press4 = &mut self.press4;
            for q in 0..4 {
                press4[q] = press4[q].max(depth * shares[q]);
            }
        }
        self.ahead += w * (d.dot(wish_dir) < -0.5 * len) as u32 as f32;
        // Jam density: COMPRESSED pairs only — a neighbor resting AT his
        // pair's rest distance contests nothing, so a formed man's
        // slot-keeping is never faded by the settled enemies one stride
        // away, which made a line yield and stay ragged.
        self.crowd += w;
        // The still men ahead: a neighbour on the half-plane ahead of
        // his wish (`d` points from the neighbour to him), by how little
        // he moved along it last tick.
        if self.still_on {
            let on = (d.dot(wish_dir) < 0.0) as u32 as f32;
            self.crowd_still += w * on * still_weight(n.o_vel, wish_dir);
        }
    }

    #[inline]
    fn hand_on(&self, st: &mut Step) {
        st.scan.crowd = self.crowd;
        st.scan.crowd_still = self.crowd_still;
        st.scan.ahead = self.ahead;
        st.scan.press4 = self.press4;
    }
}

/// The arm's length rule's read (`scan`, `keep_arm_length`), only for a
/// living walker to the fight under `keep_arm`: the bodies of either team
/// inside his arm's length, summed as `Scanned::crowding`, and how far off
/// the comrade in his way stands (`Scanned::way_gap`): the nearest living
/// comrade within SEP_RADIUS and 45 degrees of his way who is not walking
/// away along it, as his look reads a comrade ahead (`look_on_his_way`).
/// With no way it only sums the crowding. It also keeps how near his
/// nearest living body of either team stands (`Scanned::room`).
struct Arm {
    on: bool,
    way: Vec2,
    arm: f32,
    way_gap: f32,
    crowding: Vec2,
    near2: f32,
}

impl Arm {
    #[inline]
    fn new(st: &Step) -> Self {
        Arm {
            on: keep_arm() && st.fight.joining && !st.dying,
            way: st.fight.join_dir,
            arm: arm_length(),
            way_gap: f32::INFINITY,
            crowding: Vec2::ZERO,
            near2: f32::INFINITY,
        }
    }

    /// A neighbour on his level (`d` points from the neighbour to him).
    #[inline(always)]
    fn see(&mut self, n: &Near) {
        if !self.on || n.o_dying || n.d2 <= 1e-8 {
            return;
        }
        self.near2 = self.near2.min(n.d2);
        let arm = self.arm;
        let inside = n.d2 < arm * arm;
        let ahead = n.same && self.way != Vec2::ZERO && n.d2 < SEP_RADIUS * SEP_RADIUS;
        if !(inside || ahead) {
            return;
        }
        let len = n.d2.sqrt();
        if inside {
            self.crowding += n.d / len * ((arm - len) / arm);
        }
        if ahead && (-n.d).dot(self.way) > 0.707 * len && n.o_vel.dot(self.way) <= 0.5 {
            self.way_gap = self.way_gap.min(len);
        }
    }

    #[inline]
    fn hand_on(&self, st: &mut Step) {
        st.scan.way_gap = self.way_gap;
        st.scan.crowding = self.crowding;
        st.scan.room = self.near2.sqrt().min(self.arm);
    }
}

/// The enemy in his weapon's reach (`scan`): the nearest living one, and
/// whether the one he fought last tick (`prev_target`) is still in reach,
/// which keeps him on that one (sticky).
struct Reach {
    reach2: f32,
    prev_target: u32,
    best_d2: f32,
    best_idx: u32,
    sticky: bool,
}

impl Reach {
    #[inline]
    fn new(params: &crate::unit_types::UnitTypeParams, prev_target: u32) -> Self {
        Reach { reach2: params.reach * params.reach, prev_target, best_d2: f32::MAX, best_idx: u32::MAX, sticky: false }
    }

    #[inline(always)]
    fn see(&mut self, n: &Near) {
        if n.enemy && n.d2 < self.reach2 {
            self.sticky |= n.o.idx == self.prev_target;
            if n.d2 < self.best_d2 {
                self.best_d2 = n.d2;
                self.best_idx = n.o.idx;
            }
        }
    }

    #[inline]
    fn hand_on(&self, st: &mut Step) {
        st.foe.best_idx = self.best_idx;
        st.foe.sticky = self.sticky;
        st.foe.prev_target = self.prev_target;
    }
}

/// The braced spear points he runs onto (`scan`): moving at charge speed
/// at a side with a spearwall on the field, the nearest leveled point his
/// body crosses this tick.
struct Spear {
    at_charge_speed: bool,
    spear_reach: f32,
    vel: Vec2,
    impale_idx: u32,
    impale_d: f32,
}

impl Spear {
    #[inline]
    fn new(f: &Field, s: &Soldier, dying: bool) -> Self {
        let Field { speed, team, faces_spearwall, .. } = *f;
        let i = s.i;
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
        Spear { at_charge_speed, spear_reach, vel: s.vel.xz(), impale_idx: u32::MAX, impale_d: f32::MAX }
    }

    /// The radius of his scan's box: the base, his weapon's reach, and at
    /// charge speed a braced spear's.
    #[inline]
    fn scan_radius(&self, params: &crate::unit_types::UnitTypeParams) -> f32 {
        if self.at_charge_speed {
            QUERY_RADIUS.max(params.reach).max(self.spear_reach)
        } else {
            QUERY_RADIUS.max(params.reach)
        }
    }

    /// The farthest a foe can be and still count for anything in the scan.
    #[inline]
    fn foe_cut(&self, params: &crate::unit_types::UnitTypeParams) -> f32 {
        // The farthest a foe can be and still count for anything in the scan: the
        // widest pair's spring rest, his weapon's reach, or at charge speed a
        // braced spear's point. A friend counts only within SEP_RADIUS.
        let spear_cut = if self.at_charge_speed {
            (self.spear_reach * self.spear_reach + SPEAR_LINE_HALF_W * SPEAR_LINE_HALF_W).sqrt()
        } else {
            0.0
        };
        widest_rest().max(params.reach).max(spear_cut) + CUT_MARGIN
    }

    #[inline(always)]
    fn see(&mut self, n: &Near, yaw_snap: &[f32]) {
        let o = n.o;
        let d = n.d;
        let spear_reach = self.spear_reach;
        // Spear-line collision: he is a braced enemy spearman, and my body is
        // crossing his leveled point while I close at speed.
        if self.at_charge_speed
            && n.enemy
            && (o.meta & crate::spatial::META_WALL) != 0
            && crate::spatial::meta_kind(o.meta)
                == crate::unit_types::KIND_SPEAR as usize
        {
            let ys = yaw_snap[o.idx as usize];
            let f = Vec2::new(ys.sin(), ys.cos());
            let rel = d; // spearman -> me (d = p - o.xz())
            let fwd_d = rel.dot(f);
            let lat = (rel.x * f.y - rel.y * f.x).abs();
            let closing = self.vel.dot(d) < 0.0;
            if fwd_d > SPEAR_POINT_MIN
                && fwd_d < spear_reach
                && lat < SPEAR_LINE_HALF_W
                && closing
                && fwd_d < self.impale_d
            {
                self.impale_d = fwd_d;
                self.impale_idx = o.idx;
            }
        }
    }

    /// Ran onto a braced spear: the wound, a damage event from the spearman.
    #[inline]
    fn impale(&self, f: &Field, i: usize, out: &mut ChunkOut) {
        // Ran onto a braced spear: the collision is a damage event from the
        // SPEARMAN, resolved with everything else in the serial apply (which also
        // stops the runner). One point, one wound — the nearest line crossed this
        // tick.
        if self.impale_idx != u32::MAX {
            let jit = 0.85
                + 0.3
                    * crate::units::hash01(
                        f.tick_seed ^ (i as u32).wrapping_mul(0x7A31),
                    );
            out.events.push(DamageEvent {
                victim: i as u32,
                attacker: self.impale_idx,
                jit,
                charge: false,
                impale: true,
                warn: 0,
                chain: 0,
            });
        }
    }
}

/// The obstacles for the choice of opening and the squeeze (`scan`): the
/// row nearest him of the comrades ahead who are not a queue he follows,
/// and, while he squeezes, whether a body is at each of his sides.
struct Obstacles<'a> {
    open_on: bool,
    squeeze_on: bool,
    wish_dir: Vec2,
    body0: f32,
    widths: &'a [Width; WIDTH_LEVELS],
    orders: &'a [Option<Vec2>],
    terrain: &'a crate::terrain::Terrain,
    row: Row,
    row_at: f32,
    side_left: bool,
    side_right: bool,
}

impl<'a> Obstacles<'a> {
    #[inline]
    fn new(
        f: &Field<'a>,
        s: &Soldier,
        st: &Step,
        wish_dir: Vec2,
        body0: f32,
        widths: &'a [Width; WIDTH_LEVELS],
    ) -> Self {
        // The choice of opening's read (`choose_opening`): only a man walking
        // somewhere, not fighting, reads the comrades ahead; a man squeezing
        // feels the bodies at his sides.
        // Every obstacle within his reach is in the row nearest him (the band
        // reaches past OPEN_SEE), so the scan keeps the nearest two on each
        // side as it goes.
        Obstacles {
            open_on: reads_openings(f, s, st),
            squeeze_on: squeezing(*s.stuck),
            wish_dir,
            body0,
            widths,
            orders: f.orders,
            terrain: f.terrain,
            row: Row::default(),
            row_at: f32::INFINITY,
            side_left: false,
            side_right: false,
        }
    }

    #[inline(always)]
    fn see(&mut self, n: &Near) {
        let o = n.o;
        let wish_dir = self.wish_dir;
        // The comrades ahead for the choice of opening: a queue (a man of
        // a regiment under an order moving his way) is not an obstacle; a
        // holding man always is.
        if n.same && (o.meta & crate::spatial::META_DYING) == 0 && (self.open_on | self.squeeze_on) {
            let to = -n.d;
            let along = to.dot(wish_dir);
            let lat = wish_dir.x * to.y - wish_dir.y * to.x;
            let o_body = self.widths[crate::spatial::meta_pose(o.meta)].body;
            if self.open_on && along > OPEN_AHEAD {
                let holding = self.orders[crate::spatial::meta_group(o.meta)].is_none();
                let queue = !holding && n.o_vel.dot(wish_dir) > QUEUE_SPEED;
                if !queue {
                    self.row.add(lat, 0.5 * o_body, along);
                    self.row_at = self.row_at.min(along);
                }
            }
            if self.squeeze_on {
                let near = 0.5 * (self.body0 + o_body) - BETWEEN_SLACK;
                if n.d2 < near * near && along.abs() < 1.0 {
                    self.side_left |= lat > 0.0;
                    self.side_right |= lat < 0.0;
                }
            }
        }
    }

    /// His normal body and spring, whether he saw an obstacle and the
    /// opening he found among them (one his body can walk straight to,
    /// `way_free`), and whether a body is at both sides.
    #[inline]
    fn hand_on(&self, s: &Soldier, st: &mut Step, sep0: f32) {
        st.open.open_on = self.open_on;
        st.open.body0 = self.body0;
        st.open.sep0 = sep0;
        st.open.saw = self.row_at.is_finite();
        if self.open_on && st.open.saw {
            let (pref, coin) = side_pref(s.i, *s.stuck);
            let (p, wd) = (st.p, self.wish_dir);
            let perp = Vec2::new(-wd.y, wd.x);
            let free = |lat: f32, along: f32| way_free(self.terrain, p, wd * along + perp * lat);
            let (aim_lat, aim_along, fit, span) =
                openings(&self.row, self.row_at, OPEN_SEE, self.body0, SQUEEZE_DEPTH, false, pref, coin, free);
            st.open.near = (aim_lat, aim_along, fit);
            st.open.near_span = span;
        }
        st.open.between = self.side_left && self.side_right;
    }
}

/// Whether he reads the comrades ahead for the choice of opening this tick.
#[inline]
fn reads_openings(f: &Field, s: &Soldier, st: &Step) -> bool {
    let Field { orders, engaged, anchors, home, passing, .. } = *f;
    let i = s.i;
    let gi = st.gi;
    let p = st.p;
    let far = (anchors[gi] + home[i]).distance_squared(p) > TRY_FROM_SLOT * TRY_FROM_SLOT;
    let walking = orders[gi].is_some() || far;
    // Only a man with a reason to pick an opening: his regiment walking
    // through a friend that opened for it, or himself a straggler of a
    // holding regiment. Read for every walking man, it costs 5.6 ms of
    // the 200k step in a front battle where nobody passes anyone.
    let reason = passing[gi] || (orders[gi].is_none() && far);
    opening_on()
        && walking
        && reason
        && !engaged[gi]
        && !st.dying
        && !st.routed
        && !st.fight.in_melee
        && !st.fight.committed
        && st.fight.join_fp.is_none()
        && st.desired.length_squared() > 0.25
}

/// His mark for the archers (sim/mod.rs `clear_members`): every tick
/// whether an enemy stands in his reach.
#[inline]
fn mark_foe_in_reach(s: &mut Soldier, dying: bool, best_idx: u32) {
    if !dying {
        if best_idx != u32::MAX {
            *s.sight |= SIGHT_FOE;
        } else {
            *s.sight &= !SIGHT_FOE;
        }
    }
}

/// The enemy he remembers (`remembered_foe`), read when he has nobody in
/// reach or is a formed man of a fighting regiment.
#[inline]
fn recall_foe(f: &Field, st: &mut Step, i: usize, best_idx: u32, prev_target: u32) {
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
    let (memo_d2, memo_valid) = if (best_idx == u32::MAX && !st.dying && !st.routed)
        || (st.fight.in_melee && !st.fight.committed)
    {
        remembered_foe(f, i, st.p, st.gi, memo)
    } else {
        (f32::MAX, false)
    };
    st.foe.memo_valid = memo_valid;
    st.foe.memo_d2 = memo_d2;
    st.foe.best_idx = best_idx;
}

/// His stance for the next tick, a step of the ramp toward ready or back
/// at ease.
#[inline]
fn ramp_stance(f: &Field, s: &mut Soldier, st: &Step, friction: bool, fall_on: bool, best_idx: u32) {
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
    let ready = match f.stance[st.gi] {
        STANCE_FIGHT => true,
        STANCE_READY => {
            stands(friction, s.vel.xz(), st.desired) && !(friction && !*s.stood && on_his_way(s, st))
        }
        _ => false,
    };
    // A falling soldier keeps the stance he was struck in (`fall`).
    if !(st.dying && fall_on) {
        let stance_ramp = *s.pose & POSE_STANCE;
        let stance_ramp = if ready || best_idx != u32::MAX {
            (stance_ramp + 1).min(POSE_READY)
        } else {
            stance_ramp.saturating_sub(1)
        };
        *s.pose = (*s.pose & !POSE_STANCE) | stance_ramp;
    }
}

/// An archer's knife, out or away for the next tick.
#[inline]
fn draw_knife(f: &Field, s: &mut Soldier, st: &Step, best_idx: u32) {
    let i = s.i;
    let gi = st.gi;
    // An archer's knife: out while his regiment has its swords out, and
    // for a few seconds after an enemy came close to him, one in his
    // reach, an enemy block's edge within KNIFE_EDGE of him, or with his
    // quiver empty an enemy block in his regiment's bow range.
    if !st.dying && st.my_kind == crate::unit_types::KIND_ARCHER as usize {
        let arm = &f.sidearm[gi];
        let close = best_idx != u32::MAX
            || (*s.ammo == 0 && arm.foe_in_range)
            || arm.blocks.iter().any(|b| b.distance(st.p) <= KNIFE_EDGE);
        let mut count = (*s.sight & SIGHT_KNIFE_COUNT) >> SIGHT_KNIFE_SHIFT;
        if close {
            count = KNIFE_HOLD;
        } else if count > 0 && f.tick.wrapping_add(i as u32).is_multiple_of(4) {
            count -= 1;
        }
        let out = if arm.swords || count > 0 { SIGHT_KNIFE } else { 0 };
        *s.sight = (*s.sight & !(SIGHT_KNIFE_COUNT | SIGHT_KNIFE)) | count << SIGHT_KNIFE_SHIFT | out;
    }
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
