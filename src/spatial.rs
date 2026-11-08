//! Uniform spatial hash grid over the unit positions, rebuilt every fixed
//! tick with a parallel counting sort. Cell size ~2x unit radius so range
//! queries only touch a 3x3 cell neighborhood.
//!
//! The rebuild also emits `sorted`: per-unit (position, meta, index) packed
//! in grid-cell order. Neighbor queries iterate it LINEARLY — the hot
//! integrate loop reads contiguous memory instead of gathering pos/team
//! through random unit indices (the cache misses dominated the old cost).
//!
//! NOTE: an 8-wide SIMD variant of this layout (split x/z/meta
//! lane arrays + f32x8 kernel) was built and measured SLOWER overall — the
//! candidate runs are too short (~3-15 units) for lane occupancy, and the
//! split arrays cost extra cache lines on the short-run majority. Don't
//! retry without an AoSoA block layout and vectorizing the rest of the
//! integrate body too.

use bevy::prelude::*;

pub const CELL_SIZE: f32 = 1.5;
/// Grid never exceeds this many cells per axis (memory guard).
const MAX_DIM: usize = 2048;
/// Units per parallel rebuild chunk.
const REBUILD_CHUNK: usize = 32_768;
// The count pass keeps a chunk's men of a cell in a histogram entry's low
// 16 bits and its men of team 1 in the high 16 (`rebuild`).
const _: () = assert!(REBUILD_CHUNK < 1 << 16);
/// Cells per task of the rebuild's pass over the cells (`cell_vel2`).
const CELL_CHUNK: usize = 65_536;

/// Meta bits carried by each sorted unit. Kind is a 2-bit FIELD (up to 4
/// unit kinds), not a flag — read it with `meta_kind`, never as a mask test.
pub const META_TEAM: u32 = 1 << 0;
pub const META_KIND_SHIFT: u32 = 1;
pub const META_KIND: u32 = 0b11 << META_KIND_SHIFT;
pub const META_DYING: u32 = 1 << 3;
/// Unit's regiment is in a wall stance: same-team wall pairs pack to a
/// tighter separation rest distance (a shieldwall the physics would
/// otherwise push back out to normal spacing).
pub const META_WALL: u32 = 1 << 4;
/// How far the unit is into his ready stance, a 2-bit FIELD (0 at ease
/// to 3 ready, sim/soldier.rs `pose_level`): read it with `meta_pose`.
pub const META_POSE_SHIFT: u32 = 5;
pub const META_POSE: u32 = 0b11 << META_POSE_SHIFT;
/// How far the unit has turned his shoulders to squeeze through a crack,
/// a 2-bit FIELD (0 square to 3 turned, sim/soldier.rs `turn_level`), right
/// above the pose so the two read together as his width level
/// (`meta_width`).
pub const META_TURN_SHIFT: u32 = 7;
pub const META_TURN: u32 = 0b11 << META_TURN_SHIFT;
/// The unit's regiment (index into `Groups::list`) fills the bits above
/// the flags: read it with `meta_group`.
pub const META_GROUP_SHIFT: u32 = 9;

/// The kind field of a `SortedUnit::meta` (index into `unit_types::TYPES`).
#[inline]
pub fn meta_kind(meta: u32) -> usize {
    ((meta & META_KIND) >> META_KIND_SHIFT) as usize
}

/// The pose field of a `SortedUnit::meta`.
#[inline]
pub fn meta_pose(meta: u32) -> usize {
    ((meta & META_POSE) >> META_POSE_SHIFT) as usize
}

/// His width level, the pose and the turn of his shoulders together
/// (sim/soldier.rs `widths`: pose level + 4 x turn level).
#[inline]
pub fn meta_width(meta: u32) -> usize {
    ((meta & (META_POSE | META_TURN)) >> META_POSE_SHIFT) as usize
}

/// The regiment field of a `SortedUnit::meta`.
#[inline]
pub fn meta_group(meta: u32) -> usize {
    (meta >> META_GROUP_SHIFT) as usize
}

/// One unit in grid order: everything a neighbor query needs, in 16 bytes.
#[derive(Clone, Copy, Default)]
pub struct SortedUnit {
    pub x: f32,
    pub z: f32,
    pub idx: u32,
    /// META_* bit flags (team, kind, dying, ...) and the regiment.
    pub meta: u32,
}

impl SortedUnit {
    #[inline]
    pub fn xz(&self) -> Vec2 {
        Vec2::new(self.x, self.z)
    }
}

/// Raw-pointer wrapper so scatter tasks can write disjoint slots of one
/// output slice. SAFETY: only used with counting-sort offsets, which
/// partition [0, n) — each slot is written by exactly one task.
struct SharedOut(*mut SortedUnit);
unsafe impl Send for SharedOut {}
unsafe impl Sync for SharedOut {}
/// The same for the velocity column, written at the same offsets.
struct SharedVel(*mut Vec2);
unsafe impl Send for SharedVel {}
unsafe impl Sync for SharedVel {}

#[derive(Resource, Default)]
pub struct SpatialGrid {
    origin: Vec2,
    dims: (usize, usize),
    /// Prefix sums: units of cell c are sorted[starts[c]..starts[c + 1]].
    starts: Vec<u32>,
    /// The same prefix sums over the men of team 1 only: cells a..b hold
    /// starts_t1[b] - starts_t1[a] of them (`box_foes_vel2`).
    starts_t1: Vec<u32>,
    /// The largest squared speed of `sorted_vel` in each cell, 0 for an
    /// empty one (`box_foes_vel2`).
    cell_vel2: Vec<f32>,
    /// Units in cell order (the counting-sort payload).
    sorted: Vec<SortedUnit>,
    /// Tick-start ground velocity of `sorted[k]`, in the same order: a
    /// separate column, so the separation scan's 16-byte stride stays
    /// as it is and only the queries that look at motion pay for it.
    sorted_vel: Vec<Vec2>,
    /// The ground velocity of `sorted[k]`'s actual move over the last
    /// tick, (position - position a tick before) / dt, in the same order:
    /// the still-man brake's read (sim/soldier.rs `scan`). Under the
    /// yield-and-collide contact a pressed man's kept velocity is his
    /// legs', not his move.
    sorted_move: Vec<Vec2>,
    /// The largest of `sorted_vel`'s speeds: no neighbour moves faster
    /// (the scan's cut of its candidates, sim/soldier.rs `scan`).
    vel_max: f32,
    /// The field spans more than MAX_DIM cells on an axis, so the last
    /// cells hold men beyond their bounds (`kept_in_disc`).
    clamped: bool,
    /// Scratch: per-unit cell index from the count pass.
    cell_of: Vec<u32>,
    /// Each unit's rank, his slot in `sorted`, by his index.
    rank_of: Vec<u32>,
    /// Scratch: per-chunk histogram / write cursors.
    hists: Vec<Vec<u32>>,
}

impl SpatialGrid {
    #[allow(clippy::too_many_arguments)] // parallel SoA columns
    pub fn rebuild(
        &mut self,
        positions: &[Vec3],
        prev_positions: &[Vec3],
        dt: f32,
        vels: &[Vec3],
        teams: &[u8],
        kinds: &[u8],
        groups: &[u32],
        death_t: &[u8],
        poses: &[u8],
        group_wall: &[bool],
    ) {
        let n = positions.len();
        if n == 0 {
            self.dims = (0, 0);
            self.sorted.clear();
            self.sorted_vel.clear();
            self.sorted_move.clear();
            self.vel_max = 0.0;
            return;
        }

        let mut min = Vec2::splat(f32::MAX);
        let mut max = Vec2::splat(f32::MIN);
        for p in positions {
            min = min.min(Vec2::new(p.x, p.z));
            max = max.max(Vec2::new(p.x, p.z));
        }
        self.origin = min - CELL_SIZE;
        let span = max - self.origin + CELL_SIZE;
        self.clamped = (span.x / CELL_SIZE) as usize + 1 > MAX_DIM || (span.y / CELL_SIZE) as usize + 1 > MAX_DIM;
        self.dims = (
            ((span.x / CELL_SIZE) as usize + 1).min(MAX_DIM),
            ((span.y / CELL_SIZE) as usize + 1).min(MAX_DIM),
        );
        let cells = self.dims.0 * self.dims.1;
        let (origin, dims) = (self.origin, self.dims);

        let n_chunks = n.div_ceil(REBUILD_CHUNK);
        self.cell_of.resize(n, 0);
        self.hists.resize(n_chunks, Vec::new());

        // Count pass: per-chunk histograms + cached per-unit cell index.
        // An entry counts the chunk's men of the cell in its low 16 bits
        // and those of team 1 in its high 16.
        crate::util::sim_scope(|scope| {
            for ((pos_chunk, team_chunk), (cell_chunk, hist)) in positions
                .chunks(REBUILD_CHUNK)
                .zip(teams.chunks(REBUILD_CHUNK))
                .zip(self.cell_of.chunks_mut(REBUILD_CHUNK).zip(&mut self.hists))
            {
                scope.spawn(async move {
                    hist.clear();
                    hist.resize(cells, 0);
                    for (j, p) in pos_chunk.iter().enumerate() {
                        let c = cell_index_for(origin, dims, Vec2::new(p.x, p.z));
                        cell_chunk[j] = c as u32;
                        hist[c] += 1 + ((team_chunk[j] as u32) << 16);
                    }
                });
            }
        });

        // Merge: rewrite each chunk histogram into that chunk's write
        // cursors, and build the cell prefix sums. Deterministic layout:
        // cell-major, chunk order within a cell.
        self.starts.clear();
        self.starts.resize(cells + 1, 0);
        self.starts_t1.clear();
        self.starts_t1.resize(cells + 1, 0);
        let (mut acc, mut acc_t1) = (0u32, 0u32);
        for c in 0..cells {
            for hist in &mut self.hists {
                let cnt = hist[c];
                hist[c] = acc;
                acc += cnt & 0xFFFF;
                acc_t1 += cnt >> 16;
            }
            self.starts[c + 1] = acc;
            self.starts_t1[c + 1] = acc_t1;
        }

        // Scatter pass: chunks write their units to precomputed disjoint
        // offsets. SAFETY (SharedOut): the merged cursors partition [0, n),
        // so every slot of `sorted` is written exactly once, by one task.
        self.sorted.resize(n, SortedUnit::default());
        self.sorted_vel.resize(n, Vec2::ZERO);
        self.sorted_move.resize(n, Vec2::ZERO);
        self.rank_of.resize(n, 0);
        // A column of a tick before that does not match (none yet) reads
        // as no move.
        let prev = if prev_positions.len() == n { prev_positions } else { positions };
        let out = SharedOut(self.sorted.as_mut_ptr());
        let out = &out;
        let out_vel = SharedVel(self.sorted_vel.as_mut_ptr());
        let out_vel = &out_vel;
        let out_move = SharedVel(self.sorted_move.as_mut_ptr());
        let out_move = &out_move;
        let vel_max2: Vec<f32> = crate::util::sim_scope(|scope| {
            for (t, ((pos_chunk, rank_chunk), (cell_chunk, hist))) in positions
                .chunks(REBUILD_CHUNK)
                .zip(self.rank_of.chunks_mut(REBUILD_CHUNK))
                .zip(self.cell_of.chunks(REBUILD_CHUNK).zip(&mut self.hists))
                .enumerate()
            {
                let start = t * REBUILD_CHUNK;
                scope.spawn(async move {
                    let mut vel_max2 = 0.0f32;
                    for (j, p) in pos_chunk.iter().enumerate() {
                        let i = start + j;
                        let c = cell_chunk[j] as usize;
                        let k = hist[c] as usize;
                        hist[c] += 1;
                        rank_chunk[j] = k as u32;
                        let meta = ((teams[i] as u32) * META_TEAM)
                            | ((kinds[i] as u32) << META_KIND_SHIFT)
                            | (((death_t[i] > 0) as u32) * META_DYING)
                            | ((group_wall[groups[i] as usize] as u32) * META_WALL)
                            | ((crate::sim::soldier::pose_level(poses[i]) as u32) << META_POSE_SHIFT)
                            | ((crate::sim::soldier::turn_level(poses[i]) as u32) << META_TURN_SHIFT)
                            | (groups[i] << META_GROUP_SHIFT);
                        unsafe {
                            *out.0.add(k) = SortedUnit {
                                x: p.x,
                                z: p.z,
                                idx: i as u32,
                                meta,
                            };
                            *out_vel.0.add(k) = Vec2::new(vels[i].x, vels[i].z);
                            *out_move.0.add(k) = Vec2::new(p.x - prev[i].x, p.z - prev[i].z) / dt;
                        }
                        vel_max2 = vel_max2.max(Vec2::new(vels[i].x, vels[i].z).length_squared());
                    }
                    vel_max2
                });
            }
        });
        self.vel_max = vel_max2.into_iter().fold(0.0, f32::max).sqrt();

        // Each cell's fastest man, from the sorted column.
        self.cell_vel2.resize(cells, 0.0);
        let (starts, sorted_vel) = (&self.starts[..], &self.sorted_vel[..]);
        crate::util::sim_scope(|scope| {
            for (k, out) in self.cell_vel2.chunks_mut(CELL_CHUNK).enumerate() {
                scope.spawn(async move {
                    let c0 = k * CELL_CHUNK;
                    for (j, o) in out.iter_mut().enumerate() {
                        let (s, e) = (starts[c0 + j] as usize, starts[c0 + j + 1] as usize);
                        *o = sorted_vel[s..e].iter().fold(0.0f32, |m, v| m.max(v.length_squared()));
                    }
                });
            }
        });
    }

    /// Over the box of cells `kept_in_disc` reads for
    /// `center` and `radius` (before it trims the rows to its disc):
    /// whether any man of the other team than `team1` stands in them,
    /// dying or not, the largest squared speed of `sorted_vel` among all
    /// the men in them, and how many men they hold (the scan's cut,
    /// sim/soldier.rs `scan`).
    #[inline]
    pub fn box_foes_vel2(&self, center: Vec2, radius: f32, team1: bool) -> (bool, f32, usize) {
        if self.dims.0 == 0 {
            return (false, 0.0, 0);
        }
        let (cx0, cy0) = self.cell_coords(center - radius);
        let (cx1, cy1) = self.cell_coords(center + radius);
        let (mut foes, mut vel2, mut men) = (0u32, 0.0f32, 0usize);
        for cy in cy0..=cy1 {
            let row = cy * self.dims.0;
            let (a, b) = (row + cx0, row + cx1 + 1);
            let all = self.starts[b] - self.starts[a];
            let t1 = self.starts_t1[b] - self.starts_t1[a];
            foes |= if team1 { all - t1 } else { t1 };
            men += all as usize;
            for v in &self.cell_vel2[a..b] {
                vel2 = vel2.max(*v);
            }
        }
        (foes != 0, vel2, men)
    }

    /// Every unit in grid order: his rank is his slot here.
    #[inline]
    pub fn sorted(&self) -> &[SortedUnit] {
        &self.sorted
    }

    /// The tick-start ground velocity and the last tick's move of the
    /// unit at rank `r` (`sorted_vel`, `sorted_move`).
    #[inline]
    pub fn vel_move(&self, r: usize) -> (Vec2, Vec2) {
        (self.sorted_vel[r], self.sorted_move[r])
    }

    /// Each unit's rank in `sorted`, by his index.
    #[inline]
    pub fn rank_of(&self) -> &[u32] {
        &self.rank_of
    }

    /// The fastest neighbour's speed at the tick's start (`sorted_vel`).
    #[inline]
    pub fn vel_max(&self) -> f32 {
        self.vel_max
    }

    #[inline]
    fn cell_coords(&self, p: Vec2) -> (usize, usize) {
        cell_coords_for(self.origin, self.dims, p)
    }

    /// Visit all units in cells overlapping the disc at `center` with
    /// `radius` (candidates only — caller does the distance test). Units
    /// arrive as contiguous `SortedUnit`s: position + meta + index without
    /// touching the SoA arrays.
    #[inline]
    pub fn for_each_candidate(&self, center: Vec2, radius: f32, mut f: impl FnMut(&SortedUnit)) {
        if self.dims.0 == 0 {
            return;
        }
        let (cx0, cy0) = self.cell_coords(center - radius);
        let (cx1, cy1) = self.cell_coords(center + radius);
        for cy in cy0..=cy1 {
            let row = cy * self.dims.0;
            let s = self.starts[row + cx0] as usize;
            let e = self.starts[row + cx1 + 1] as usize;
            // Consecutive cells of a row are adjacent in `sorted`, so [s, e)
            // covers exactly cells cx0..=cx1 of this row, linearly.
            for u in &self.sorted[s..e] {
                f(u);
            }
        }
    }

    /// `for_each_candidate` that also hands each unit's tick-start ground
    /// velocity, read from the grid-ordered column next to it.
    #[inline]
    pub fn for_each_candidate_vel(
        &self,
        center: Vec2,
        radius: f32,
        mut f: impl FnMut(&SortedUnit, Vec2),
    ) {
        if self.dims.0 == 0 {
            return;
        }
        let (cx0, cy0) = self.cell_coords(center - radius);
        let (cx1, cy1) = self.cell_coords(center + radius);
        for cy in cy0..=cy1 {
            let row = cy * self.dims.0;
            let s = self.starts[row + cx0] as usize;
            let e = self.starts[row + cx1 + 1] as usize;
            for (u, v) in self.sorted[s..e].iter().zip(&self.sorted_vel[s..e]) {
                f(u, *v);
            }
        }
    }

    /// The scan's cut over its walk (sim/soldier.rs `scan`): the walk
    /// visits the units of the box of `radius` with each row of cells cut
    /// to the cells the disc of `disc` reaches, in `sorted`'s order (a man
    /// in a skipped cell is farther than `disc` from `center`), and the
    /// rank of every man it visits who is not `cut.me` and is within reach
    /// is written to `out`, in that order, and the count returned. A man
    /// is out of reach when d2 >= near2 and (d2 >= cut2 or his squared
    /// speed <= fast2), (near2, cut2) being the friends' or the foes' by
    /// his team. Four men at a time, with the same arithmetic as one at a
    /// time.
    ///
    /// # Safety
    /// `out` has room for every man in the cells of the box of `radius`.
    #[inline]
    pub unsafe fn kept_in_disc(&self, center: Vec2, radius: f32, disc: f32, cut: &KeepCut, out: *mut u32) -> usize {
        if self.dims.0 == 0 {
            return 0;
        }
        let (cx0, cy0) = self.cell_coords(center - radius);
        let (cx1, cy1) = self.cell_coords(center + radius);
        let r2 = disc * disc;
        let mut n = 0;
        for cy in cy0..=cy1 {
            let (a, b) = if self.clamped {
                (cx0, cx1)
            } else {
                // The row's nearest edge to the centre across the rows.
                let lo = self.origin.y + cy as f32 * CELL_SIZE;
                let gy = (lo - center.y).max(center.y - (lo + CELL_SIZE)).max(0.0);
                let half = (r2 - gy * gy).max(0.0).sqrt();
                (
                    self.cell_coords(Vec2::new(center.x - half, center.y)).0.max(cx0),
                    self.cell_coords(Vec2::new(center.x + half, center.y)).0.min(cx1),
                )
            };
            let row = cy * self.dims.0;
            let s = self.starts[row + a] as usize;
            let e = self.starts[row + b + 1] as usize;
            // SAFETY: the caller's.
            n = unsafe { keep_run(&self.sorted[s..e], &self.sorted_vel[s..e], s as u32, center, cut, out, n) };
        }
        n
    }

    /// Does any candidate satisfy `f`? Stops at the first that does.
    #[inline]
    pub fn any_candidate_vel(
        &self,
        center: Vec2,
        radius: f32,
        mut f: impl FnMut(&SortedUnit, Vec2) -> bool,
    ) -> bool {
        if self.dims.0 == 0 {
            return false;
        }
        let (cx0, cy0) = self.cell_coords(center - radius);
        let (cx1, cy1) = self.cell_coords(center + radius);
        for cy in cy0..=cy1 {
            let row = cy * self.dims.0;
            let s = self.starts[row + cx0] as usize;
            let e = self.starts[row + cx1 + 1] as usize;
            for (u, v) in self.sorted[s..e].iter().zip(&self.sorted_vel[s..e]) {
                if f(u, *v) {
                    return true;
                }
            }
        }
        false
    }
}

/// The scan's cut for `SpatialGrid::kept_in_disc`: the man's own team
/// bit and index, and the squared reaches.
pub struct KeepCut {
    pub team_bit: u32,
    pub me: u32,
    pub same_near2: f32,
    pub same_cut2: f32,
    pub foe_near2: f32,
    pub foe_cut2: f32,
    pub fast2: f32,
}

/// `kept_in_disc` over one run of `sorted` from rank `start`, writing
/// from `out[n]` on; returns the new count.
///
/// # Safety
/// `out` has room for `n` plus every man of the run.
#[inline]
unsafe fn keep_run(units: &[SortedUnit], vels: &[Vec2], start: u32, center: Vec2, cut: &KeepCut, out: *mut u32, mut n: usize) -> usize {
    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
    let mut k = 0;
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 is part of every x86_64; the loads read units
        // k..k + 4 and their velocities, within the run.
        unsafe {
            let (px, pz) = (_mm_set1_ps(center.x), _mm_set1_ps(center.y));
            let team = _mm_set1_epi32(cut.team_bit as i32);
            let one = _mm_set1_epi32(META_TEAM as i32);
            let me = _mm_set1_epi32(cut.me as i32);
            let (sn2, sc2) = (_mm_set1_ps(cut.same_near2), _mm_set1_ps(cut.same_cut2));
            let (fn2, fc2) = (_mm_set1_ps(cut.foe_near2), _mm_set1_ps(cut.foe_cut2));
            let fast2 = _mm_set1_ps(cut.fast2);
            let (ub, vb) = (units.as_ptr() as *const f32, vels.as_ptr() as *const f32);
            while k + 4 <= units.len() {
                // Four men's x, z, index and meta, and their velocities.
                let p0 = _mm_loadu_ps(ub.add(4 * k));
                let p1 = _mm_loadu_ps(ub.add(4 * k + 4));
                let p2 = _mm_loadu_ps(ub.add(4 * k + 8));
                let p3 = _mm_loadu_ps(ub.add(4 * k + 12));
                let (t0, t1) = (_mm_unpacklo_ps(p0, p1), _mm_unpacklo_ps(p2, p3));
                let (t2, t3) = (_mm_unpackhi_ps(p0, p1), _mm_unpackhi_ps(p2, p3));
                let (x, z) = (_mm_movelh_ps(t0, t1), _mm_movehl_ps(t1, t0));
                let idx = _mm_castps_si128(_mm_movelh_ps(t2, t3));
                let meta = _mm_castps_si128(_mm_movehl_ps(t3, t2));
                let v01 = _mm_loadu_ps(vb.add(2 * k));
                let v23 = _mm_loadu_ps(vb.add(2 * k + 4));
                let vx = _mm_shuffle_ps::<0b10_00_10_00>(v01, v23);
                let vz = _mm_shuffle_ps::<0b11_01_11_01>(v01, v23);
                let (dx, dz) = (_mm_sub_ps(px, x), _mm_sub_ps(pz, z));
                let d2 = _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dz, dz));
                let v2 = _mm_add_ps(_mm_mul_ps(vx, vx), _mm_mul_ps(vz, vz));
                let same = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(meta, one), team));
                let near2 = _mm_or_ps(_mm_and_ps(same, sn2), _mm_andnot_ps(same, fn2));
                let cut2 = _mm_or_ps(_mm_and_ps(same, sc2), _mm_andnot_ps(same, fc2));
                let out_of_reach = _mm_and_ps(_mm_cmpge_ps(d2, near2), _mm_or_ps(_mm_cmpge_ps(d2, cut2), _mm_cmple_ps(v2, fast2)));
                let is_me = _mm_castsi128_ps(_mm_cmpeq_epi32(idx, me));
                let keep = _mm_movemask_ps(_mm_or_ps(out_of_reach, is_me)) ^ 0b1111;
                for lane in 0..4 {
                    *out.add(n) = start + (k + lane) as u32;
                    n += (keep >> lane) as usize & 1;
                }
                k += 4;
            }
        }
    }
    for (j, (u, v)) in units.iter().zip(vels).enumerate().skip(k) {
        let d2 = (center - u.xz()).length_squared();
        let same = (u.meta & META_TEAM) == cut.team_bit;
        let (near2, cut2) = if same { (cut.same_near2, cut.same_cut2) } else { (cut.foe_near2, cut.foe_cut2) };
        let out_of_reach = (d2 >= near2) & ((d2 >= cut2) | (v.length_squared() <= cut.fast2));
        // SAFETY: the caller's.
        unsafe { *out.add(n) = start + j as u32 };
        n += ((u.idx != cut.me) & !out_of_reach) as usize;
    }
    n
}

#[inline]
fn cell_coords_for(origin: Vec2, dims: (usize, usize), p: Vec2) -> (usize, usize) {
    let g = (p - origin) / CELL_SIZE;
    (
        (g.x as usize).min(dims.0 - 1),
        (g.y as usize).min(dims.1 - 1),
    )
}

#[inline]
fn cell_index_for(origin: Vec2, dims: (usize, usize), p: Vec2) -> usize {
    let (cx, cy) = cell_coords_for(origin, dims, p);
    cy * dims.0 + cx
}
