//! Uniform spatial hash grid over the unit positions, rebuilt every fixed
//! tick with a counting sort. Cell size ~2x unit radius so range queries
//! only touch a 3x3 cell neighborhood.

use bevy::prelude::*;

pub const CELL_SIZE: f32 = 1.5;
/// Grid never exceeds this many cells per axis (memory guard).
const MAX_DIM: usize = 2048;

#[derive(Resource, Default)]
pub struct SpatialGrid {
    origin: Vec2,
    dims: (usize, usize),
    /// Prefix sums: units of cell c are entries[starts[c]..starts[c + 1]].
    starts: Vec<u32>,
    /// Unit indices grouped by cell.
    entries: Vec<u32>,
    /// Scratch: write cursor per cell during scatter.
    cursor: Vec<u32>,
}

impl SpatialGrid {
    pub fn rebuild(&mut self, positions: &[Vec3]) {
        let n = positions.len();
        self.entries.resize(n, 0);
        if n == 0 {
            self.dims = (0, 0);
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
        self.starts.clear();
        self.starts.resize(cells + 1, 0);

        // Count per cell (starts shifted by one so the prefix sum lands right).
        for p in positions {
            let c = self.cell_index(Vec2::new(p.x, p.z));
            self.starts[c + 1] += 1;
        }
        for c in 0..cells {
            self.starts[c + 1] += self.starts[c];
        }
        // Scatter.
        self.cursor.clear();
        self.cursor.extend_from_slice(&self.starts[..cells]);
        for (i, p) in positions.iter().enumerate() {
            let c = self.cell_index(Vec2::new(p.x, p.z));
            self.entries[self.cursor[c] as usize] = i as u32;
            self.cursor[c] += 1;
        }
    }

    #[inline]
    fn cell_coords(&self, p: Vec2) -> (usize, usize) {
        let g = (p - self.origin) / CELL_SIZE;
        (
            (g.x as usize).min(self.dims.0 - 1),
            (g.y as usize).min(self.dims.1 - 1),
        )
    }

    #[inline]
    fn cell_index(&self, p: Vec2) -> usize {
        let (cx, cy) = self.cell_coords(p);
        cy * self.dims.0 + cx
    }

    /// Visit indices of all units in cells overlapping the disc at `center`
    /// with `radius` (candidates only — caller does the distance test).
    #[inline]
    pub fn for_each_candidate(&self, center: Vec2, radius: f32, mut f: impl FnMut(u32)) {
        if self.dims.0 == 0 {
            return;
        }
        let (cx0, cy0) = self.cell_coords(center - radius);
        let (cx1, cy1) = self.cell_coords(center + radius);
        for cy in cy0..=cy1 {
            let row = cy * self.dims.0;
            let s = self.starts[row + cx0] as usize;
            let e = self.starts[row + cx1 + 1] as usize;
            // Cells in a row are contiguous in `entries` only per cell, but
            // consecutive cells share boundaries, so [s, e) covers exactly
            // cells cx0..=cx1 of this row.
            for &idx in &self.entries[s..e] {
                f(idx);
            }
        }
    }
}
