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
/// Units per parallel rebuild chunk. A man's slot in his cell is his
/// place among its men in index order whatever the chunks, so their size
/// changes nothing but the work's share across the threads.
const REBUILD_CHUNK: usize = 16_384;
/// Cells per task of the rebuild's merge.
const CELL_CHUNK: usize = 4096;

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

/// One unit in grid order: everything a neighbor query needs, in 16 bytes,
/// in this order (the scan's cut loads four f32 a man, `keep_run`).
#[derive(Clone, Copy, Default)]
#[repr(C)]
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
/// The same for a chunk's histogram in the merge, whose tasks each read
/// and write their own run of cells of every chunk's histogram.
struct SharedHist(*mut u32);
unsafe impl Send for SharedHist {}
unsafe impl Sync for SharedHist {}

#[derive(Resource, Default)]
pub struct SpatialGrid {
    origin: Vec2,
    dims: (usize, usize),
    /// Prefix sums: units of cell c are sorted[starts[c]..starts[c + 1]].
    starts: Vec<u32>,
    /// Units in cell order (the counting-sort payload).
    sorted: Vec<SortedUnit>,
    /// Tick-start ground velocity of `sorted[k]`, in the same order: a
    /// separate column, so the separation scan's 16-byte stride stays
    /// as it is and only the queries that look at motion pay for it.
    sorted_vel: Vec<Vec2>,
    /// Scratch: per-unit cell index from the count pass.
    cell_of: Vec<u32>,
    /// Scratch: per-chunk histogram / write cursors.
    hists: Vec<Vec<u32>>,
}

impl SpatialGrid {
    #[allow(clippy::too_many_arguments)] // parallel SoA columns
    pub fn rebuild(
        &mut self,
        positions: &[Vec3],
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
            return;
        }

        // The field's bounds, chunk by chunk in parallel: the smallest and
        // the largest place on each axis are the same in any order.
        let bounds: Vec<(Vec2, Vec2)> = crate::util::sim_scope(|scope| {
            for pos_chunk in positions.chunks(REBUILD_CHUNK) {
                scope.spawn(async move {
                    let mut min = Vec2::splat(f32::MAX);
                    let mut max = Vec2::splat(f32::MIN);
                    for p in pos_chunk {
                        min = min.min(Vec2::new(p.x, p.z));
                        max = max.max(Vec2::new(p.x, p.z));
                    }
                    (min, max)
                });
            }
        });
        let (min, max) = bounds
            .into_iter()
            .fold((Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)), |(min, max), (lo, hi)| (min.min(lo), max.max(hi)));
        self.origin = min - CELL_SIZE;
        let span = max - self.origin + CELL_SIZE;
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
        crate::util::sim_scope(|scope| {
            for (pos_chunk, (cell_chunk, hist)) in positions
                .chunks(REBUILD_CHUNK)
                .zip(self.cell_of.chunks_mut(REBUILD_CHUNK).zip(&mut self.hists))
            {
                scope.spawn(async move {
                    hist.clear();
                    hist.resize(cells, 0);
                    for (j, p) in pos_chunk.iter().enumerate() {
                        let c = cell_index_for(origin, dims, Vec2::new(p.x, p.z));
                        cell_chunk[j] = c as u32;
                        hist[c] += 1;
                    }
                });
            }
        });

        // Merge: rewrite each chunk histogram into that chunk's write
        // cursors, and build the cell prefix sums. Deterministic layout:
        // cell-major, chunk order within a cell. In parallel over runs of
        // cells: each run's men first, then the runs' offsets one after the
        // other, then each run's prefix sums and cursors from its offset,
        // the same numbers one pass over the cells gives. SAFETY
        // (SharedHist): every histogram holds `cells` slots, and each task
        // reads and writes only its own run of them.
        self.starts.clear();
        self.starts.resize(cells + 1, 0);
        let hist_at: Vec<SharedHist> = self.hists.iter_mut().map(|h| SharedHist(h.as_mut_ptr())).collect();
        let hist_at = &hist_at[..];
        let run_men: Vec<u32> = crate::util::sim_scope(|scope| {
            for c0 in (0..cells).step_by(CELL_CHUNK) {
                scope.spawn(async move {
                    let c1 = (c0 + CELL_CHUNK).min(cells);
                    let mut men = 0u32;
                    for h in hist_at {
                        men += (c0..c1).map(|c| unsafe { *h.0.add(c) }).sum::<u32>();
                    }
                    men
                });
            }
        });
        let mut acc = 0u32;
        let run_from: Vec<u32> = run_men
            .into_iter()
            .map(|men| {
                let from = acc;
                acc += men;
                from
            })
            .collect();
        crate::util::sim_scope(|scope| {
            for ((k, starts), &from) in self.starts[1..].chunks_mut(CELL_CHUNK).enumerate().zip(&run_from) {
                scope.spawn(async move {
                    let c0 = k * CELL_CHUNK;
                    let mut acc = from;
                    for (j, start) in starts.iter_mut().enumerate() {
                        for h in hist_at {
                            let at = unsafe { h.0.add(c0 + j) };
                            let cnt = unsafe { *at };
                            unsafe { *at = acc };
                            acc += cnt;
                        }
                        *start = acc;
                    }
                });
            }
        });

        // Scatter pass: chunks write their units to precomputed disjoint
        // offsets. SAFETY (SharedOut): the merged cursors partition [0, n),
        // so every slot of `sorted` is written exactly once, by one task.
        self.sorted.resize(n, SortedUnit::default());
        self.sorted_vel.resize(n, Vec2::ZERO);
        let out = SharedOut(self.sorted.as_mut_ptr());
        let out = &out;
        let out_vel = SharedVel(self.sorted_vel.as_mut_ptr());
        let out_vel = &out_vel;
        crate::util::sim_scope(|scope| {
            for (t, (pos_chunk, (cell_chunk, hist))) in positions
                .chunks(REBUILD_CHUNK)
                .zip(self.cell_of.chunks(REBUILD_CHUNK).zip(&mut self.hists))
                .enumerate()
            {
                let start = t * REBUILD_CHUNK;
                scope.spawn(async move {
                    for (j, p) in pos_chunk.iter().enumerate() {
                        let i = start + j;
                        let c = cell_chunk[j] as usize;
                        let k = hist[c] as usize;
                        hist[c] += 1;
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
                        }
                    }
                });
            }
        });
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

    /// How many units the cells `for_each_candidate` reads for `center`
    /// and `radius` hold.
    #[inline]
    pub fn box_len(&self, center: Vec2, radius: f32) -> usize {
        if self.dims.0 == 0 {
            return 0;
        }
        let (cx0, cy0) = self.cell_coords(center - radius);
        let (cx1, cy1) = self.cell_coords(center + radius);
        let mut men = 0;
        for cy in cy0..=cy1 {
            let row = cy * self.dims.0;
            men += (self.starts[row + cx1 + 1] - self.starts[row + cx0]) as usize;
        }
        men
    }

    /// Every unit in grid order: his rank is his slot here.
    #[inline]
    pub fn sorted(&self) -> &[SortedUnit] {
        &self.sorted
    }

    /// The tick-start ground velocity of the unit at rank `r`.
    #[inline]
    pub fn sorted_vel(&self, r: usize) -> Vec2 {
        self.sorted_vel[r]
    }

    /// The scan's cut over its walk (sim/soldier.rs `scan`): the walk
    /// visits the units of the cells `for_each_candidate` reads for
    /// `center` and `radius`, in the same order, and the rank of every
    /// man it visits who is not `cut.me` and is within his team's cut (not
    /// d2 >= `same_cut2` for a friend, `foe_cut2` for a foe) is written to
    /// `out`, in that order, and the count returned. Eight or four men at
    /// a time, with the same arithmetic as one at a time.
    ///
    /// # Safety
    /// `out` has room for every man in the cells of the box (`box_len`).
    #[inline]
    pub unsafe fn kept_in_box(&self, center: Vec2, radius: f32, cut: &KeepCut, out: *mut u32) -> usize {
        #[cfg(target_arch = "x86_64")]
        if simd_avx2() {
            // SAFETY: the CPU has AVX2; the rest is the caller's.
            return unsafe { self.kept_in_box_avx2(center, radius, cut, out) };
        }
        let lanes = KeepLanes::new(center, cut);
        // SAFETY: the caller's.
        self.kept_rows(center, radius, |s, e, n| unsafe {
            keep_run(&self.sorted[s..e], s as u32, center, cut, &lanes, out, n)
        })
    }

    /// `kept_in_box` eight candidates at a time, on a CPU with AVX2.
    ///
    /// # Safety
    /// The CPU has AVX2, and `out` is as for `kept_in_box`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn kept_in_box_avx2(&self, center: Vec2, radius: f32, cut: &KeepCut, out: *mut u32) -> usize {
        let lanes = KeepLanes::new(center, cut);
        let lanes8 = KeepLanes8::new(center, cut);
        // SAFETY: the caller's.
        self.kept_rows(center, radius, |s, e, n| unsafe {
            keep_run8(&self.sorted[s..e], s as u32, center, cut, &lanes, &lanes8, out, n)
        })
    }

    /// The walk of `kept_in_box`: each row's run of `sorted` over the box,
    /// `run(start, end, count)` returning the new count.
    #[inline(always)]
    fn kept_rows(&self, center: Vec2, radius: f32, mut run: impl FnMut(usize, usize, usize) -> usize) -> usize {
        if self.dims.0 == 0 {
            return 0;
        }
        let (cx0, cy0) = self.cell_coords(center - radius);
        let (cx1, cy1) = self.cell_coords(center + radius);
        let mut n = 0;
        for cy in cy0..=cy1 {
            let row = cy * self.dims.0;
            let s = self.starts[row + cx0] as usize;
            let e = self.starts[row + cx1 + 1] as usize;
            n = run(s, e, n);
        }
        n
    }
}

/// The scan's cut for `SpatialGrid::kept_in_box`: the man's own team bit
/// and index, and the squared cuts of a friend and of a foe.
pub struct KeepCut {
    pub team_bit: u32,
    pub me: u32,
    pub same_cut2: f32,
    pub foe_cut2: f32,
}

/// The cut runs eight men at a time where the CPU has AVX2, unless
/// FL_SIMD=sse2 keeps it on four at a time (the same numbers either way;
/// the switch is for checking that).
#[cfg(target_arch = "x86_64")]
#[inline]
fn simd_avx2() -> bool {
    static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *A.get_or_init(|| std::arch::is_x86_feature_detected!("avx2") && !std::env::var("FL_SIMD").is_ok_and(|v| v == "sse2"))
}

/// `KeepCut` in all four lanes, set once for a man's walk.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct KeepLanes {
    px: std::arch::x86_64::__m128,
    pz: std::arch::x86_64::__m128,
    team: std::arch::x86_64::__m128i,
    me: std::arch::x86_64::__m128i,
    sc2: std::arch::x86_64::__m128,
    fc2: std::arch::x86_64::__m128,
}
#[cfg(not(target_arch = "x86_64"))]
struct KeepLanes;

impl KeepLanes {
    #[inline]
    fn new(center: Vec2, cut: &KeepCut) -> KeepLanes {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: SSE2 is part of every x86_64.
        unsafe {
            use std::arch::x86_64::*;
            KeepLanes {
                px: _mm_set1_ps(center.x),
                pz: _mm_set1_ps(center.y),
                team: _mm_set1_epi32(cut.team_bit as i32),
                me: _mm_set1_epi32(cut.me as i32),
                sc2: _mm_set1_ps(cut.same_cut2),
                fc2: _mm_set1_ps(cut.foe_cut2),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (center, cut);
            KeepLanes
        }
    }
}

/// `kept_in_box` over one run of `sorted` from rank `start`, writing from
/// `out[n]` on; returns the new count. Each man's rank is written and the
/// count advanced by his outcome, without a branch on it. The distance is
/// the scalar's, (x - x') squared plus (z - z') squared, in that order and
/// without a fused multiply-add.
///
/// # Safety
/// `out` has room for `n` plus every man of the run.
#[inline]
unsafe fn keep_run(units: &[SortedUnit], start: u32, center: Vec2, cut: &KeepCut, lanes: &KeepLanes, out: *mut u32, mut n: usize) -> usize {
    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
    let mut k = 0;
    #[cfg(not(target_arch = "x86_64"))]
    let _ = lanes;
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 is part of every x86_64; the loads read units
        // k..k + 4, within the run.
        unsafe {
            let KeepLanes { px, pz, team, me, sc2, fc2 } = *lanes;
            let one = _mm_set1_epi32(META_TEAM as i32);
            let ub = units.as_ptr() as *const f32;
            while k + 4 <= units.len() {
                // Four men's x, z, index and meta, one man a column.
                let p0 = _mm_loadu_ps(ub.add(4 * k));
                let p1 = _mm_loadu_ps(ub.add(4 * k + 4));
                let p2 = _mm_loadu_ps(ub.add(4 * k + 8));
                let p3 = _mm_loadu_ps(ub.add(4 * k + 12));
                let (t0, t1) = (_mm_unpacklo_ps(p0, p1), _mm_unpacklo_ps(p2, p3));
                let (t2, t3) = (_mm_unpackhi_ps(p0, p1), _mm_unpackhi_ps(p2, p3));
                let (x, z) = (_mm_movelh_ps(t0, t1), _mm_movehl_ps(t1, t0));
                let idx = _mm_castps_si128(_mm_movelh_ps(t2, t3));
                let meta = _mm_castps_si128(_mm_movehl_ps(t3, t2));
                let (dx, dz) = (_mm_sub_ps(px, x), _mm_sub_ps(pz, z));
                let d2 = _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dz, dz));
                let same = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(meta, one), team));
                let cut2 = _mm_or_ps(_mm_and_ps(same, sc2), _mm_andnot_ps(same, fc2));
                let is_me = _mm_castsi128_ps(_mm_cmpeq_epi32(idx, me));
                let keep = _mm_movemask_ps(_mm_or_ps(_mm_cmpge_ps(d2, cut2), is_me)) ^ 0b1111;
                for lane in 0..4 {
                    *out.add(n) = start + (k + lane) as u32;
                    n += (keep >> lane) as usize & 1;
                }
                k += 4;
            }
        }
    }
    for (j, u) in units.iter().enumerate().skip(k) {
        let d2 = (center - u.xz()).length_squared();
        let cut2 = if (u.meta & META_TEAM) == cut.team_bit { cut.same_cut2 } else { cut.foe_cut2 };
        let beyond = d2 >= cut2;
        // SAFETY: the caller's.
        unsafe { *out.add(n) = start + j as u32 };
        n += ((u.idx != cut.me) & !beyond) as usize;
    }
    n
}

/// `KeepCut` in all eight lanes, for `keep_run8`.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct KeepLanes8 {
    px: std::arch::x86_64::__m256,
    pz: std::arch::x86_64::__m256,
    team: std::arch::x86_64::__m256i,
    me: std::arch::x86_64::__m256i,
    sc2: std::arch::x86_64::__m256,
    fc2: std::arch::x86_64::__m256,
}

#[cfg(target_arch = "x86_64")]
impl KeepLanes8 {
    #[target_feature(enable = "avx2")]
    fn new(center: Vec2, cut: &KeepCut) -> KeepLanes8 {
        use std::arch::x86_64::*;
        KeepLanes8 {
            px: _mm256_set1_ps(center.x),
            pz: _mm256_set1_ps(center.y),
            team: _mm256_set1_epi32(cut.team_bit as i32),
            me: _mm256_set1_epi32(cut.me as i32),
            sc2: _mm256_set1_ps(cut.same_cut2),
            fc2: _mm256_set1_ps(cut.foe_cut2),
        }
    }
}

/// `keep_run` eight men at a time, the same arithmetic in each lane, and
/// the rest of the run four at a time and one at a time as there.
///
/// # Safety
/// The CPU has AVX2, and `out` has room for `n` plus every man of the
/// run.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)] // the run and the man's cut
unsafe fn keep_run8(
    units: &[SortedUnit],
    start: u32,
    center: Vec2,
    cut: &KeepCut,
    lanes: &KeepLanes,
    lanes8: &KeepLanes8,
    out: *mut u32,
    mut n: usize,
) -> usize {
    use std::arch::x86_64::*;
    let KeepLanes8 { px, pz, team, me, sc2, fc2 } = *lanes8;
    let one = _mm256_set1_epi32(META_TEAM as i32);
    let ub = units.as_ptr() as *const f32;
    let mut k = 0;
    // SAFETY: the loads read units k..k + 8, within the run; the stores
    // are the caller's.
    unsafe {
        while k + 8 <= units.len() {
            // Men k..k + 4 in the low half of each register, k + 4..k + 8
            // in the high half, so lane j holds man k + j.
            let pair = |a: usize| _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(_mm_loadu_ps(ub.add(4 * (k + a)))), _mm_loadu_ps(ub.add(4 * (k + a + 4))));
            let (p0, p1, p2, p3) = (pair(0), pair(1), pair(2), pair(3));
            let (t0, t1) = (_mm256_unpacklo_ps(p0, p1), _mm256_unpacklo_ps(p2, p3));
            let (t2, t3) = (_mm256_unpackhi_ps(p0, p1), _mm256_unpackhi_ps(p2, p3));
            let (x, z) = (_mm256_shuffle_ps::<0x44>(t0, t1), _mm256_shuffle_ps::<0xEE>(t0, t1));
            let idx = _mm256_castps_si256(_mm256_shuffle_ps::<0x44>(t2, t3));
            let meta = _mm256_castps_si256(_mm256_shuffle_ps::<0xEE>(t2, t3));
            let (dx, dz) = (_mm256_sub_ps(px, x), _mm256_sub_ps(pz, z));
            let d2 = _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dz, dz));
            let same = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(meta, one), team));
            let cut2 = _mm256_or_ps(_mm256_and_ps(same, sc2), _mm256_andnot_ps(same, fc2));
            let is_me = _mm256_castsi256_ps(_mm256_cmpeq_epi32(idx, me));
            let keep = _mm256_movemask_ps(_mm256_or_ps(_mm256_cmp_ps::<_CMP_GE_OQ>(d2, cut2), is_me)) ^ 0xFF;
            for lane in 0..8 {
                *out.add(n) = start + (k + lane) as u32;
                n += (keep >> lane) as usize & 1;
            }
            k += 8;
        }
        keep_run(&units[k..], start + k as u32, center, cut, lanes, out, n)
    }
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
