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
/// The unit's regiment (index into `Groups::list`) fills the bits above
/// the flags: read it with `meta_group`.
pub const META_GROUP_SHIFT: u32 = 7;

/// The kind field of a `SortedUnit::meta` (index into `unit_types::TYPES`).
#[inline]
pub fn meta_kind(meta: u32) -> usize {
    ((meta & META_KIND) >> META_KIND_SHIFT) as usize
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
        walled: &[bool],
    ) {
        let n = positions.len();
        if n == 0 {
            self.dims = (0, 0);
            self.sorted.clear();
            self.sorted_vel.clear();
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
        // cell-major, chunk order within a cell.
        self.starts.clear();
        self.starts.resize(cells + 1, 0);
        let mut acc = 0u32;
        for c in 0..cells {
            for hist in &mut self.hists {
                let cnt = hist[c];
                hist[c] = acc;
                acc += cnt;
            }
            self.starts[c + 1] = acc;
        }

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
                            | ((walled[i] as u32) * META_WALL)
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
