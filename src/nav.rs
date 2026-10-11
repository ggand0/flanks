//! The walkability grid and the regiment planner.
//!
//! The grid is the terrain's own vertex grid: one cell per height vertex,
//! 2 m apart. A soldier collides with the nearest vertex's blocked bit
//! (`Terrain::blocked_at`), so planning on the same grid finds every gap
//! a man can walk through: the narrowest the obstacle map asks for is
//! 6 m, which an 8 m grid erases and a 4 m grid finds only when its cells
//! happen to line up. Each cell keeps
//!
//! - whether it is walkable: not blocked by the terrain and not inside
//!   the field's edge margin, where the sim clamps soldiers;
//! - its cost, the time a man takes per metre there against flat dry
//!   ground, from the same slope and wading factors the soldier's `drive`
//!   stage applies to his pace;
//! - its clearance, the distance to the nearest unwalkable cell (exact up
//!   to `CLEAR_CAP`);
//! - its region: cells joined by walkable steps share one, so a goal in
//!   another region is known to be out of reach before any search.
//!
//! Blocked ground has one source today, the terrain's mask, and the map
//! edge. Another source (tree trunks) would mark more cells unwalkable in
//! `NavGrid::mark_blocked` before the clearance and regions are derived.

use bevy::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

use crate::terrain::{MapKind, Terrain, CELL};

/// Clearance is exact up to this distance (m) and capped there. Wider
/// than the widest regiment's half width the planner ever asks for.
pub const CLEAR_CAP: f32 = 64.0;
/// The sim keeps soldiers this far inside the field (sim/job.rs
/// `bounds_min`): the cells beyond are not walkable.
const EDGE: f32 = 4.0;
/// The cost of giving up all the room a passage offers: walking at a
/// wall that could have been kept a half width away costs like walking
/// three times as far (see `NavGrid::costs`).
const ROOM: f32 = 2.0;
/// A smoothed segment may cost this much more than the grid path it
/// replaces (the sampled cost and the stepped cost differ slightly).
const SMOOTH_SLACK: f32 = 1.02;
/// How far ahead a man looks along his way for blocked ground (m). What
/// lies beyond he meets when he gets there; his regiment's waypoint
/// leads him meanwhile. Bounds every line check a man makes.
pub const SIGHT: f32 = 48.0;
/// How far round an unreachable goal the planner looks for reachable
/// ground to end the route at (m).
const NEAR_GOAL: f32 = 400.0;

/// The grid. See the module docs.
pub struct NavGrid {
    /// The map it was built from: another map means a new grid, not a
    /// refresh, even at the same size.
    pub kind: MapKind,
    pub origin: Vec2,
    pub nx: usize,
    pub nz: usize,
    walk: Vec<bool>,
    cost: Vec<f32>,
    wet: Vec<bool>,
    clear: Vec<f32>,
    region: Vec<u32>,
    /// The largest region: the field the armies stand on. Cells of every
    /// other region (the flat tops of blocks, closed rooms) are cut off.
    main_region: u32,
    /// The map has water a man wades through: the straight check then
    /// looks for it too.
    pub has_water: bool,
    /// Heights the grid was built from, to find what a crater changed.
    heights: Vec<f32>,
    /// The cost tables built so far, one per regiment width (see
    /// `Costs`). Emptied whenever the grid changes.
    costs: std::sync::Mutex<Vec<std::sync::Arc<Costs>>>,
}

impl Clone for NavGrid {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind,
            origin: self.origin,
            nx: self.nx,
            nz: self.nz,
            walk: self.walk.clone(),
            cost: self.cost.clone(),
            wet: self.wet.clone(),
            clear: self.clear.clone(),
            region: self.region.clone(),
            main_region: self.main_region,
            has_water: self.has_water,
            heights: self.heights.clone(),
            costs: Default::default(),
        }
    }
}

/// The cost per metre of every cell in each step class for regiments of
/// one width (`NavGrid::costs` gives the rule). Built once per width
/// and shared by every plan for a regiment that wide.
pub struct Costs {
    span: Span,
    w: Vec<[f32; 4]>,
}

/// What a regiment asks the planner for.
#[derive(Clone, Copy, Debug)]
pub struct PlanRequest {
    /// The regiment (index into `Groups::list`).
    pub g: u32,
    /// Serial of the request: a regiment's newer request supersedes it.
    pub serial: u32,
    pub start: Vec2,
    pub goal: Vec2,
    /// Half the regiment's width across its way (m).
    pub half_w: f32,
}

/// The planner's answer.
#[derive(Clone, Debug)]
pub struct Plan {
    pub req: PlanRequest,
    /// The route from the start (first point) to the end (last point),
    /// empty when the start has no walkable ground near it.
    pub points: Vec<Vec2>,
    /// The goal itself is reachable; else the route ends at the nearest
    /// reachable ground with room for the regiment.
    pub reached: bool,
    /// Cells the search closed, the ticks it ran over, and its total time.
    pub cells: u32,
    pub ticks: u32,
    pub ms: f32,
}

impl Plan {
    /// The route's length (m).
    pub fn length(&self) -> f32 {
        self.points.windows(2).map(|w| w[0].distance(w[1])).sum()
    }
}

impl NavGrid {
    /// The grid of `terrain`, or None for a map with no blocked ground
    /// and no water: there every straight way is clear and nothing plans.
    pub fn build(terrain: &Terrain) -> Option<Self> {
        if !terrain.kind.has_blocked_ground() && terrain.classic {
            return None;
        }
        let (nx, nz) = terrain.grid_verts();
        let n = nx * nz;
        let mut grid = Self {
            kind: terrain.kind,
            origin: terrain.min(),
            nx,
            nz,
            walk: vec![false; n],
            cost: vec![1.0; n],
            wet: vec![false; n],
            clear: vec![0.0; n],
            region: vec![u32::MAX; n],
            main_region: 0,
            has_water: false,
            heights: terrain.heights().to_vec(),
            costs: Default::default(),
        };
        grid.fill_cells(terrain, 0, 0, nx - 1, nz - 1);
        grid.has_water = grid.wet.iter().any(|&w| w);
        grid.clearance_window(0, 0, nx - 1, nz - 1, 0, 0, nx - 1, nz - 1);
        grid.label_regions();
        Some(grid)
    }

    /// Bring the grid up to date with `terrain` after its heights changed
    /// (a crater): the cells whose heights or blocked bit may have moved
    /// are refilled, the clearance is recomputed around them, the regions
    /// relabelled. Returns false when nothing changed.
    pub fn refresh(&mut self, terrain: &Terrain) -> bool {
        let h = terrain.heights();
        if h.len() != self.heights.len() {
            if let Some(g) = Self::build(terrain) {
                *self = g;
            }
            return true;
        }
        let (mut x0, mut z0, mut x1, mut z1) = (usize::MAX, usize::MAX, 0, 0);
        for z in 0..self.nz {
            for x in 0..self.nx {
                let i = z * self.nx + x;
                if h[i] != self.heights[i] {
                    x0 = x0.min(x);
                    z0 = z0.min(z);
                    x1 = x1.max(x);
                    z1 = z1.max(z);
                }
            }
        }
        if x0 == usize::MAX {
            return false;
        }
        self.heights.copy_from_slice(h);
        // The blocked rule reads 4-neighbours and the slope one metre
        // away: two cells of margin.
        let (x0, z0) = (x0.saturating_sub(2), z0.saturating_sub(2));
        let (x1, z1) = ((x1 + 2).min(self.nx - 1), (z1 + 2).min(self.nz - 1));
        self.fill_cells(terrain, x0, z0, x1, z1);
        self.has_water = self.wet.iter().any(|&w| w);
        // Clearance up to the cap can change that far around the edit,
        // and depends on cells up to the cap beyond that.
        let r = (CLEAR_CAP / CELL).ceil() as usize + 1;
        let (ox0, oz0) = (x0.saturating_sub(r), z0.saturating_sub(r));
        let (ox1, oz1) = ((x1 + r).min(self.nx - 1), (z1 + r).min(self.nz - 1));
        let (wx0, wz0) = (ox0.saturating_sub(r), oz0.saturating_sub(r));
        let (wx1, wz1) = ((ox1 + r).min(self.nx - 1), (oz1 + r).min(self.nz - 1));
        self.clearance_window(wx0, wz0, wx1, wz1, ox0, oz0, ox1, oz1);
        self.label_regions();
        self.costs.lock().unwrap().clear();
        true
    }

    /// Walkable, cost and water for the cells in `[x0, x1] x [z0, z1]`.
    fn fill_cells(&mut self, terrain: &Terrain, x0: usize, z0: usize, x1: usize, z1: usize) {
        let lo = terrain.min() + EDGE;
        let hi = terrain.max() - EDGE;
        for z in z0..=z1 {
            for x in x0..=x1 {
                let i = z * self.nx + x;
                let p = self.pos(i);
                let inside = p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y;
                self.walk[i] = inside && !terrain.blocked_at(p.x, p.y);
                let slope = terrain.slope_at(p.x, p.y);
                let wade = terrain.wade_mult(p.x, p.y);
                self.wet[i] = wade < 1.0;
                // drive: pace = speed / (1 + 3 slope^2) * wade.
                self.cost[i] = (1.0 + 3.0 * slope * slope) / wade;
            }
        }
        self.mark_blocked(x0, z0, x1, z1);
    }

    /// Further sources of blocked ground, applied after the terrain's.
    /// None yet; tree trunks would mark their cells here.
    fn mark_blocked(&mut self, _x0: usize, _z0: usize, _x1: usize, _z1: usize) {}

    /// Euclidean distance to the nearest unwalkable cell, computed over
    /// the window `[wx0, wx1] x [wz0, wz1]` (cells outside it are taken
    /// as open) and written for `[ox0, ox1] x [oz0, oz1]`, capped at
    /// `CLEAR_CAP`. Felzenszwalb and Huttenlocher's two-pass transform.
    #[allow(clippy::too_many_arguments)]
    fn clearance_window(
        &mut self,
        wx0: usize,
        wz0: usize,
        wx1: usize,
        wz1: usize,
        ox0: usize,
        oz0: usize,
        ox1: usize,
        oz1: usize,
    ) {
        let w = wx1 - wx0 + 1;
        let h = wz1 - wz0 + 1;
        let big = 1e12f64;
        let mut f = vec![0.0f64; w * h];
        for z in 0..h {
            for x in 0..w {
                let i = (wz0 + z) * self.nx + wx0 + x;
                f[z * w + x] = if self.walk[i] { big } else { 0.0 };
            }
        }
        let mut col = vec![0.0f64; h.max(w)];
        let mut out = vec![0.0f64; h.max(w)];
        let mut v = vec![0usize; h.max(w)];
        let mut zz = vec![0.0f64; h.max(w) + 1];
        for x in 0..w {
            for z in 0..h {
                col[z] = f[z * w + x];
            }
            edt_1d(&col[..h], &mut out[..h], &mut v, &mut zz);
            for z in 0..h {
                f[z * w + x] = out[z];
            }
        }
        for z in 0..h {
            col[..w].copy_from_slice(&f[z * w..z * w + w]);
            edt_1d(&col[..w], &mut out[..w], &mut v, &mut zz);
            f[z * w..z * w + w].copy_from_slice(&out[..w]);
        }
        for z in oz0..=oz1 {
            for x in ox0..=ox1 {
                let d2 = f[(z - wz0) * w + (x - wx0)];
                self.clear[z * self.nx + x] = ((d2.sqrt() as f32) * CELL).min(CLEAR_CAP);
            }
        }
    }

    /// Label the regions: walkable cells joined by the planner's steps
    /// (`neighbours`) share a label.
    fn label_regions(&mut self) {
        self.region.fill(u32::MAX);
        let mut next = 0u32;
        let mut stack = Vec::new();
        for s in 0..self.walk.len() {
            if !self.walk[s] || self.region[s] != u32::MAX {
                continue;
            }
            self.region[s] = next;
            stack.push(s as u32);
            while let Some(i) = stack.pop() {
                let mut nb = [(0u32, 0.0f32, 0u8); 8];
                let k = self.neighbours(i as usize, &mut nb);
                for &(j, _, _) in &nb[..k] {
                    if self.region[j as usize] == u32::MAX {
                        self.region[j as usize] = next;
                        stack.push(j);
                    }
                }
            }
            next += 1;
        }
        let mut size = vec![0u32; next as usize];
        for &r in &self.region {
            if r != u32::MAX {
                size[r as usize] += 1;
            }
        }
        self.main_region = (0..next).max_by_key(|&r| (size[r as usize], u32::MAX - r)).unwrap_or(0);
    }

    /// The walkable cells one step from walkable cell `i`, with the step's
    /// length and class: 0 along x, 1 along z, 2 along the rising
    /// diagonal, 3 along the other. A diagonal step needs both cells beside
    /// it open: the planner never cuts a corner a block of men cannot.
    #[inline]
    fn neighbours(&self, i: usize, out: &mut [(u32, f32, u8); 8]) -> usize {
        let x = (i % self.nx) as i32;
        let z = (i / self.nx) as i32;
        let open = |x: i32, z: i32| {
            x >= 0
                && z >= 0
                && (x as usize) < self.nx
                && (z as usize) < self.nz
                && self.walk[z as usize * self.nx + x as usize]
        };
        let mut k = 0;
        for (dx, dz, class) in [(1, 0, 0), (-1, 0, 0), (0, 1, 1), (0, -1, 1)] {
            if open(x + dx, z + dz) {
                out[k] = (((z + dz) as usize * self.nx + (x + dx) as usize) as u32, CELL, class);
                k += 1;
            }
        }
        for (dx, dz, class) in [(1, 1, 2), (1, -1, 3), (-1, 1, 3), (-1, -1, 2)] {
            if open(x + dx, z + dz) && open(x + dx, z) && open(x, z + dz) {
                out[k] = (
                    ((z + dz) as usize * self.nx + (x + dx) as usize) as u32,
                    CELL * std::f32::consts::SQRT_2,
                    class,
                );
                k += 1;
            }
        }
        k
    }

    #[inline]
    fn pos(&self, i: usize) -> Vec2 {
        self.origin + Vec2::new((i % self.nx) as f32, (i / self.nx) as f32) * CELL
    }

    /// The cell whose vertex is nearest `p`, None off the field.
    #[inline]
    fn cell_at(&self, p: Vec2) -> Option<usize> {
        let g = ((p - self.origin) / CELL).round();
        if g.x < 0.0 || g.y < 0.0 || g.x >= self.nx as f32 || g.y >= self.nz as f32 {
            return None;
        }
        Some(g.y as usize * self.nx + g.x as usize)
    }

    /// The clearance at `p` (m), zero off the field.
    #[inline]
    pub fn clearance_at(&self, p: Vec2) -> f32 {
        self.cell_at(p).map_or(0.0, |i| self.clear[i])
    }

    /// A man at `a` sees no unwalkable ground on the straight line to `b`,
    /// as far as he looks (`SIGHT`). Marches along the line by the
    /// clearance: in open ground a few lookups cover the whole way, next
    /// to a wall it steps half a metre, so a look is at most about a
    /// hundred lookups.
    pub fn walk_line(&self, a: Vec2, b: Vec2) -> bool {
        let d = b - a;
        let len = d.length();
        let dir = if len > 1e-4 { d / len } else { Vec2::ZERO };
        let len = len.min(SIGHT);
        // A point is within half a cell's diagonal of its vertex, and an
        // unwalkable cell reaches as far from its own: the free radius
        // around a point is its vertex's clearance less a full diagonal.
        let slack = CELL * std::f32::consts::SQRT_2;
        let mut t = 0.0f32;
        loop {
            let q = a + dir * t.min(len);
            let Some(i) = self.cell_at(q) else { return false };
            if !self.walk[i] {
                return false;
            }
            if t >= len {
                return true;
            }
            t += (self.clear[i] - slack).max(CELL * 0.25);
        }
    }

    /// Every point of the line from `a` to `b` stands at least `w` from
    /// unwalkable ground: a block of that half width walks it straight.
    pub fn band_clear(&self, a: Vec2, b: Vec2, w: f32) -> bool {
        let d = b - a;
        let len = d.length();
        let dir = if len > 1e-4 { d / len } else { Vec2::ZERO };
        let need = w.min(CLEAR_CAP - CELL);
        let mut t = 0.0f32;
        loop {
            let q = a + dir * t.min(len);
            let c = self.clearance_at(q);
            if c < need {
                return false;
            }
            if t >= len {
                return true;
            }
            t += (c - need).max(CELL * 0.5);
        }
    }

    /// No water along the line from `a` to `b`.
    pub fn dry_line(&self, a: Vec2, b: Vec2) -> bool {
        if !self.has_water {
            return true;
        }
        let len = a.distance(b);
        let steps = (len / (CELL * 0.5)).ceil().max(1.0) as usize;
        (0..=steps).all(|k| {
            let q = a.lerp(b, k as f32 / steps as f32);
            self.cell_at(q).is_none_or(|i| !self.wet[i])
        })
    }

    /// The straight way from `a` to `b` needs no route: a block of half
    /// width `w` walks it without meeting blocked ground or water.
    pub fn straight_ok(&self, a: Vec2, b: Vec2, w: f32) -> bool {
        self.band_clear(a, b, w) && self.dry_line(a, b)
    }

    /// A block of half extents `he` (axis-aligned, m) centred at `c`
    /// stands wholly on the field's open ground: every point of it,
    /// sampled every cell, is walkable and in the field's region (not a
    /// roof, not a closed room).
    pub fn footprint_clear(&self, c: Vec2, he: Vec2) -> bool {
        let nx = (he.x / CELL).ceil().max(1.0) as i32;
        let nz = (he.y / CELL).ceil().max(1.0) as i32;
        for iz in -nz..=nz {
            for ix in -nx..=nx {
                let p = c + Vec2::new(ix as f32 * he.x / nx as f32, iz as f32 * he.y / nz as f32);
                match self.cell_at(p) {
                    Some(i) if self.walk[i] && self.region[i] == self.main_region => {}
                    _ => return false,
                }
            }
        }
        true
    }

    /// The nearest centre to `c`, within `max_r` metres and passing `ok`,
    /// where a block of half extents `he` stands clear (`footprint_clear`).
    /// Cells with less clearance than the block's narrower half extent
    /// are skipped without the full check.
    pub fn nearest_footprint(&self, c: Vec2, he: Vec2, max_r: f32, ok: impl Fn(Vec2) -> bool) -> Option<Vec2> {
        let need = he.min_element().min(CLEAR_CAP - CELL);
        self.nearest_where(c, max_r, |i| {
            let p = self.pos(i);
            self.clear[i] >= need && ok(p) && self.footprint_clear(p, he)
        })
        .map(|i| self.pos(i))
    }

    /// The unwalkable cells next to walkable ground within `radius` of
    /// `center`: the outline of the blocked ground, for the debug view.
    pub fn edge_cells(&self, center: Vec2, radius: f32) -> Vec<Vec2> {
        let lo = ((center - radius - self.origin) / CELL).floor().max(Vec2::ZERO);
        let hi = ((center + radius - self.origin) / CELL).ceil();
        let (x1, z1) = ((hi.x as usize).min(self.nx - 1), (hi.y as usize).min(self.nz - 1));
        let mut out = Vec::new();
        for z in lo.y as usize..=z1 {
            for x in lo.x as usize..=x1 {
                let i = z * self.nx + x;
                if self.walk[i] {
                    continue;
                }
                let near_open = (x > 0 && self.walk[i - 1])
                    || (x + 1 < self.nx && self.walk[i + 1])
                    || (z > 0 && self.walk[i - self.nx])
                    || (z + 1 < self.nz && self.walk[i + self.nx]);
                if near_open {
                    out.push(self.pos(i));
                }
            }
        }
        out
    }

    /// The walkable cell nearest `p` within `max_r` metres, by distance.
    fn nearest_walkable(&self, p: Vec2, max_r: f32) -> Option<usize> {
        self.nearest_where(p, max_r, |i| self.walk[i])
    }

    /// The cell nearest `p` that passes `ok`, searched ring by ring out
    /// to `max_r` metres.
    fn nearest_where(&self, p: Vec2, max_r: f32, ok: impl Fn(usize) -> bool) -> Option<usize> {
        let g = ((p - self.origin) / CELL).round();
        let (cx, cz) = (g.x as i32, g.y as i32);
        let rings = (max_r / CELL).ceil() as i32;
        let mut best: Option<(f32, usize)> = None;
        for r in 0..=rings {
            // A cell on ring r is at least (r - 1) cells from p.
            if let Some((d, _)) = best
                && d < (r as f32 - 1.0) * CELL
            {
                break;
            }
            for dz in -r..=r {
                for dx in -r..=r {
                    if dx.abs() != r && dz.abs() != r {
                        continue;
                    }
                    let (x, z) = (cx + dx, cz + dz);
                    if x < 0 || z < 0 || x >= self.nx as i32 || z >= self.nz as i32 {
                        continue;
                    }
                    let i = z as usize * self.nx + x as usize;
                    if !ok(i) {
                        continue;
                    }
                    let d = self.pos(i).distance(p);
                    if best.is_none_or(|(bd, bi)| d < bd || (d == bd && i < bi)) {
                        best = Some((d, i));
                    }
                }
            }
        }
        best.map(|(_, i)| i)
    }

    /// The cost table for regiments of half width `span`, built on first
    /// use. Every cell's cost per metre in each step class is the
    /// ground's own cost times two factors for the room a regiment has
    /// there, read off the line through the cell straight across its way
    /// (the regiment's width long):
    ///
    /// - The squeeze: the regiment's width over the open width on that
    ///   line, when that is less. A block that must pour through less
    ///   width than it has takes that many times as long to pass, its men
    ///   streaming through the lanes there are. A wall gap counts its one
    ///   lane, a field of posts every lane between them. The open width is
    ///   the narrowest of the line straight across and the two at 45
    ///   degrees to it, so a step cannot look roomy by crossing a thin wall
    ///   at a slant. Cells cut off from the field (a block's flat top) are
    ///   not open.
    /// - The room given up: where the line has a spot with more clearance
    ///   (up to the regiment's half width) than this cell, the share of the
    ///   half width given up, times `ROOM`. The men nearest the wall would
    ///   walk into it and wait, so a route keeps to the middle of a passage
    ///   that is wide enough, and hugs nothing it does not have to.
    fn costs(&self, span: Span) -> std::sync::Arc<Costs> {
        if let Some(t) = self.cached_costs(span) {
            return t;
        }
        let passes: Vec<_> = (0..4u8).map(|class| self.class_pass(class, span)).collect();
        self.store_costs(span, &passes, false)
    }

    fn cached_costs(&self, span: Span) -> Option<std::sync::Arc<Costs>> {
        let tables = self.costs.lock().unwrap();
        tables.iter().find(|t| t.span.k_axis == span.k_axis).cloned()
    }

    /// Build the cost tables the plans for `requests` need and do not have
    /// yet, on the pool. Returns whether any had to be built.
    pub fn prepare_costs(&self, requests: &[PlanRequest]) -> bool {
        let mut spans: Vec<Span> = Vec::new();
        for r in requests {
            let span = Span::of(r.half_w.clamp(CELL, CLEAR_CAP - CELL));
            if !spans.iter().any(|s| s.k_axis == span.k_axis) && self.cached_costs(span).is_none() {
                spans.push(span);
            }
        }
        let built = !spans.is_empty();
        for span in spans {
            let t0 = Instant::now();
            let passes = crate::util::sim_scope(|scope| {
                for class in 0..4u8 {
                    scope.spawn(async move { self.class_pass(class, span) });
                }
            });
            self.store_costs(span, &passes, true);
            debug!(
                "[path] cost table for a {:.1} m half width built in {:.2} ms",
                span.half_w,
                t0.elapsed().as_secs_f32() * 1000.0
            );
        }
        built
    }

    /// One step class's lines: per cell, the open share of the line across
    /// it and the largest clearance of the open cells on that line.
    fn class_pass(&self, class: u8, span: Span) -> (Vec<f32>, Vec<f32>) {
        let n = self.walk.len();
        let open = |j: usize| self.region[j] == self.main_region;
        let mut share = vec![1.0f32; n];
        let mut widest = vec![0.0f32; n];
        let mut line = Vec::new();
        let mut pref = Vec::new();
        let mut deque = std::collections::VecDeque::new();
        let (lx, lz) = line_dir(class);
        let k = if class < 2 { span.k_axis } else { span.k_diag } as i64;
        let all = (2 * k + 1) as f32;
        for start in self.line_starts(lx, lz) {
            line.clear();
            let (mut x, mut z) = ((start % self.nx) as i64, (start / self.nx) as i64);
            while x >= 0 && z >= 0 && x < self.nx as i64 && z < self.nz as i64 {
                line.push(z as usize * self.nx + x as usize);
                x += lx;
                z += lz;
            }
            let len = line.len() as i64;
            pref.clear();
            pref.push(0u32);
            for &j in &line {
                pref.push(pref.last().unwrap() + open(j) as u32);
            }
            // Sliding maximum of the open cells' clearance over [t - k, t + k].
            let v = |t: i64| {
                let j = line[t as usize];
                if open(j) { self.clear[j] } else { 0.0 }
            };
            deque.clear();
            for r in 0..len + k {
                if r < len {
                    while deque.back().is_some_and(|&b| v(b) <= v(r)) {
                        deque.pop_back();
                    }
                    deque.push_back(r);
                }
                let t = r - k;
                if t < 0 {
                    continue;
                }
                while deque.front().is_some_and(|&f| f < t - k) {
                    deque.pop_front();
                }
                let j = line[t as usize];
                let (lo, hi) = ((t - k).max(0) as usize, (t + k).min(len - 1) as usize);
                share[j] = (pref[hi + 1] - pref[lo]) as f32 / all;
                widest[j] = v(*deque.front().unwrap());
            }
        }
        (share, widest)
    }

    /// Combine the four classes' passes into the table and keep it, in
    /// parallel strips on the pool when `pooled` (from the sim worker).
    fn store_costs(&self, span: Span, passes: &[(Vec<f32>, Vec<f32>)], pooled: bool) -> std::sync::Arc<Costs> {
        let n = self.walk.len();
        let mut w = vec![[f32::INFINITY; 4]; n];
        let fill = |start: usize, strip: &mut [[f32; 4]]| {
            for (j, wi) in strip.iter_mut().enumerate() {
                let i = start + j;
                if !self.walk[i] {
                    continue;
                }
                for class in 0..4usize {
                    let slants: [usize; 2] = if class < 2 { [2, 3] } else { [0, 1] };
                    let sh = passes[class].0[i].min(passes[slants[0]].0[i]).min(passes[slants[1]].0[i]);
                    let squeeze = (1.0 / sh.max(1e-3)).max(1.0);
                    let lost = ((passes[class].1[i].min(span.half_w) - self.clear[i]) / span.half_w).max(0.0);
                    wi[class] = self.cost[i] * squeeze * (1.0 + ROOM * lost);
                }
            }
        };
        if pooled {
            const STRIP: usize = 16384;
            let fill = &fill;
            crate::util::sim_scope(|scope| {
                for (k, strip) in w.chunks_mut(STRIP).enumerate() {
                    scope.spawn(async move { fill(k * STRIP, strip) });
                }
            });
        } else {
            fill(0, &mut w);
        }
        let table = std::sync::Arc::new(Costs { span, w });
        let mut tables = self.costs.lock().unwrap();
        if let Some(t) = tables.iter().find(|t| t.span.k_axis == span.k_axis) {
            return t.clone();
        }
        tables.push(table.clone());
        table
    }

    /// The first cell of every line of direction `(lx, lz)` across the
    /// grid: the cells whose step back along it leaves the field.
    fn line_starts(&self, lx: i64, lz: i64) -> Vec<usize> {
        let (nx, nz) = (self.nx as i64, self.nz as i64);
        let mut out = Vec::new();
        for z in 0..nz {
            for x in 0..nx {
                let (px, pz) = (x - lx, z - lz);
                if px < 0 || pz < 0 || px >= nx || pz >= nz {
                    out.push((z * nx + x) as usize);
                }
            }
        }
        out
    }

    /// Plan a route for `req` in one go: a search begun and run to its end.
    #[cfg(test)]
    pub fn plan(&self, req: &PlanRequest) -> Plan {
        let mut search = self.begin(req);
        self.advance(&mut search, u32::MAX).expect("an unbounded search finishes")
    }

    /// Begin a route search for `req`: A* over the grid from the cell
    /// nearest the start to the goal's cell, or, when the goal is out of
    /// reach, to the nearest reachable cell with room for the regiment.
    /// `advance` runs it, a budget of cells at a time, so a long search
    /// spreads over ticks and no tick waits for it.
    pub fn begin(&self, req: &PlanRequest) -> Search {
        let t0 = Instant::now();
        let need = req.half_w.clamp(CELL, CLEAR_CAP - CELL);
        let costs = self.costs(Span::of(need));
        let mut search = Search {
            req: *req,
            need,
            costs,
            start: 0,
            target: 0,
            reached: false,
            heap: BinaryHeap::new(),
            seen: std::collections::HashMap::new(),
            cells: 0,
            ticks: 0,
            ms: 0.0,
            done: false,
        };
        let Some(start) = self.nearest_walkable(req.start, CLEAR_CAP) else {
            search.done = true;
            search.ms = t0.elapsed().as_secs_f32() * 1000.0;
            return search;
        };
        let region = self.region[start];
        let goal_cell = self.cell_at(req.goal).filter(|&i| self.region[i] == region);
        let (target, reached) = match goal_cell {
            Some(i) => (i, true),
            None => {
                // The nearest reachable ground where the block fits, else
                // the nearest reachable ground at all, within `NEAR_GOAL`
                // (a ring search costs by the square of its radius); none
                // in reach makes a route of one point, the start.
                let roomy = self.nearest_where(req.goal, NEAR_GOAL, |i| {
                    self.region[i] == region && self.clear[i] >= need
                });
                let any = || self.nearest_where(req.goal, NEAR_GOAL, |i| self.region[i] == region);
                match roomy.or_else(any) {
                    Some(i) => (i, false),
                    None => (start, false),
                }
            }
        };
        search.start = start;
        search.target = target;
        search.reached = reached;
        let goal_p = self.pos(target);
        search.seen.insert(start as u32, Seen { g: 0.0, parent: u32::MAX, closed: false });
        search.heap.push(Node { f: octile(self.pos(start), goal_p), i: start as u32 });
        search.ms = t0.elapsed().as_secs_f32() * 1000.0;
        search
    }

    /// Run `search` for at most `budget` more cells. Returns the plan once
    /// the search is over: the grid path pulled straight wherever a
    /// straight segment keeps the same clearance and costs no more.
    pub fn advance(&self, search: &mut Search, budget: u32) -> Option<Plan> {
        let t0 = Instant::now();
        search.ticks += 1;
        if !search.done {
            let goal_p = self.pos(search.target);
            let h = |i: usize| octile(self.pos(i), goal_p);
            let mut nb = [(0u32, 0.0f32, 0u8); 8];
            let mut spent = 0u32;
            loop {
                if spent >= budget {
                    search.ms += t0.elapsed().as_secs_f32() * 1000.0;
                    return None;
                }
                let Some(Node { i, .. }) = search.heap.pop() else { break };
                let seen = search.seen.get_mut(&i).expect("a heap entry was seen");
                if seen.closed {
                    continue;
                }
                seen.closed = true;
                let gi = seen.g;
                spent += 1;
                search.cells += 1;
                if i as usize == search.target {
                    break;
                }
                let k = self.neighbours(i as usize, &mut nb);
                for &(j, len, class) in &nb[..k] {
                    let class = class as usize;
                    let gj = gi + len * 0.5 * (search.costs.w[i as usize][class] + search.costs.w[j as usize][class]);
                    match search.seen.get_mut(&j) {
                        Some(sj) if sj.closed => continue,
                        Some(sj) if gj >= sj.g => continue,
                        Some(sj) => {
                            sj.g = gj;
                            sj.parent = i;
                        }
                        None => {
                            search.seen.insert(j, Seen { g: gj, parent: i, closed: false });
                        }
                    }
                    search.heap.push(Node { f: gj + h(j as usize), i: j });
                }
            }
            search.done = true;
        }
        let req = search.req;
        let mut out = Plan {
            req,
            points: Vec::new(),
            reached: search.reached,
            cells: search.cells,
            ticks: search.ticks,
            ms: search.ms,
        };
        if search.seen.is_empty() {
            // No walkable ground near the start.
            out.ms += t0.elapsed().as_secs_f32() * 1000.0;
            return Some(out);
        }
        // The grid path, start to target.
        let mut path = vec![search.target];
        let mut i = search.target as u32;
        while i as usize != search.start {
            let Some(sn) = search.seen.get(&i) else { break };
            if sn.parent == u32::MAX {
                break;
            }
            i = sn.parent;
            path.push(i as usize);
        }
        path.reverse();
        let g_of = |i: usize| search.seen.get(&(i as u32)).map_or(0.0, |sn| sn.g);
        let kept = self.pull_straight(&path, &g_of, search.need, &search.costs);
        out.points.push(req.start);
        if kept.len() > 2 {
            for &k in &kept[1..kept.len() - 1] {
                out.points.push(self.pos(path[k]));
            }
        }
        out.points.push(if search.reached { req.goal } else { self.pos(search.target) });
        out.ms += t0.elapsed().as_secs_f32() * 1000.0;
        Some(out)
    }

    /// The indices of `path` kept as waypoints: from each kept cell, the
    /// farthest later cell reachable by a straight segment that is
    /// walkable, keeps the clearance the grid path had between them (up to
    /// the regiment's half width) and costs no more than that stretch.
    fn pull_straight(&self, path: &[usize], g_of: &dyn Fn(usize) -> f32, need: f32, costs: &Costs) -> Vec<usize> {
        let last = path.len() - 1;
        let mut kept = vec![0];
        let mut i = 0;
        while i < last {
            let mut best = i + 1;
            let mut min_clear = self.clear[path[i]].min(self.clear[path[i + 1]]);
            for j in i + 2..=last {
                min_clear = min_clear.min(self.clear[path[j]]);
                let want = min_clear.min(need);
                let budget = (g_of(path[j]) - g_of(path[i])) * SMOOTH_SLACK + 0.1;
                if self.segment_fits(self.pos(path[i]), self.pos(path[j]), want, costs, budget) {
                    best = j;
                } else if j > best + 12 {
                    break;
                }
            }
            kept.push(best);
            i = best;
        }
        kept
    }

    /// The straight segment from `a` to `b` is walkable, keeps clearance
    /// `want` everywhere, and costs at most `budget` by `costs`.
    fn segment_fits(&self, a: Vec2, b: Vec2, want: f32, costs: &Costs, budget: f32) -> bool {
        let len = a.distance(b);
        let d = (b - a).abs();
        let class: usize = if d.x > 2.414 * d.y {
            0
        } else if d.y > 2.414 * d.x {
            1
        } else if (b.x - a.x) * (b.y - a.y) > 0.0 {
            2
        } else {
            3
        };
        let steps = (len / (CELL * 0.5)).ceil().max(1.0) as usize;
        let step = len / steps as f32;
        let mut cost = 0.0;
        for k in 0..=steps {
            let q = a.lerp(b, k as f32 / steps as f32);
            let Some(i) = self.cell_at(q) else { return false };
            if !self.walk[i] || self.clear[i] + 0.01 < want {
                return false;
            }
            let wgt = if k == 0 || k == steps { 0.5 } else { 1.0 };
            cost += costs.w[i][class] * step * wgt;
            if cost > budget {
                return false;
            }
        }
        true
    }
}

/// A regiment's half width in cells, across an axis step and across a
/// diagonal one (diagonal cells lie a cell's diagonal apart). Widths are
/// rounded up to `SPAN_STEP` cells, so a battle's regiments share a few
/// cost tables (one is 16 bytes a cell) instead of one per facing.
#[derive(Clone, Copy)]
pub struct Span {
    half_w: f32,
    k_axis: usize,
    k_diag: usize,
}

/// Cells per width class of the cost tables (6 m of half width).
const SPAN_STEP: usize = 3;

impl Span {
    fn of(half_w: f32) -> Self {
        let k_axis = ((half_w / CELL).ceil() as usize).div_ceil(SPAN_STEP).max(1) * SPAN_STEP;
        let half_w = k_axis as f32 * CELL;
        Self {
            half_w,
            k_axis,
            k_diag: (half_w / (CELL * std::f32::consts::SQRT_2)).round() as usize,
        }
    }
}

/// The direction of the line straight across a step of class `class`:
/// across an x step it runs along z, across a z step along x, across a
/// rising diagonal along the falling one, and the other way round.
#[inline]
fn line_dir(class: u8) -> (i64, i64) {
    match class {
        0 => (0, 1),
        1 => (1, 0),
        2 => (-1, 1),
        _ => (1, 1),
    }
}

/// The 8-way grid distance between two points (m).
#[inline]
fn octile(a: Vec2, b: Vec2) -> f32 {
    let d = (a - b).abs();
    let (lo, hi) = if d.x < d.y { (d.x, d.y) } else { (d.y, d.x) };
    hi + (std::f32::consts::SQRT_2 - 1.0) * lo
}

/// One-dimensional squared distance transform (Felzenszwalb and
/// Huttenlocher): `out[q] = min_p (q - p)^2 + f[p]`.
fn edt_1d(f: &[f64], out: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let inf = f64::INFINITY;
    let meet = |q: usize, p: usize| {
        ((f[q] + (q * q) as f64) - (f[p] + (p * p) as f64)) / (2.0 * (q as f64 - p as f64))
    };
    let mut k = 0usize;
    v[0] = 0;
    z[0] = -inf;
    z[1] = inf;
    for q in 1..n {
        // z[0] is minus infinity, so the first parabola is never popped.
        let mut s = meet(q, v[k]);
        while s <= z[k] {
            k -= 1;
            s = meet(q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = inf;
    }
    k = 0;
    for (q, o) in out.iter_mut().enumerate() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let p = v[k];
        let d = q as f64 - p as f64;
        *o = d * d + f[p];
    }
}

/// An open cell of the search, ordered so the heap pops the lowest `f`,
/// ties to the lower index (the same order on every run).
#[derive(Clone, Copy)]
struct Node {
    f: f32,
    i: u32,
}

impl PartialEq for Node {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Node {}
impl PartialOrd for Node {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Node {
    fn cmp(&self, o: &Self) -> Ordering {
        o.f.total_cmp(&self.f).then_with(|| o.i.cmp(&self.i))
    }
}

/// A route search in progress (`NavGrid::begin`, `advance`), carried
/// across ticks by sim/route.rs. The visited cells live in a map, so a
/// search costs memory for the cells it touched, not for the grid.
pub struct Search {
    pub req: PlanRequest,
    need: f32,
    costs: std::sync::Arc<Costs>,
    start: usize,
    target: usize,
    reached: bool,
    heap: BinaryHeap<Node>,
    seen: std::collections::HashMap<u32, Seen>,
    cells: u32,
    ticks: u32,
    ms: f32,
    done: bool,
}

/// What the search knows of a cell it has touched.
struct Seen {
    g: f32,
    parent: u32,
    closed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obstacle_map::CASES;
    use crate::terrain::{build_terrain, MapKind};

    fn grid() -> NavGrid {
        NavGrid::build(&build_terrain(MapKind::Obstacles)).expect("the obstacle map has a grid")
    }

    fn req(start: Vec2, goal: Vec2) -> PlanRequest {
        PlanRequest { g: 0, serial: 0, start, goal, half_w: 14.3 }
    }

    /// Every segment of a route is walkable for a man.
    fn assert_walkable(g: &NavGrid, plan: &Plan) {
        for w in plan.points.windows(2) {
            assert!(g.walk_line(w[0], w[1]), "segment {} -> {} crosses blocked ground", w[0], w[1]);
        }
    }

    #[test]
    fn open_maps_build_no_grid() {
        assert!(NavGrid::build(&build_terrain(MapKind::Grassland)).is_none());
        assert!(NavGrid::build(&build_terrain(MapKind::BigGrassland)).is_none());
        assert!(NavGrid::build(&build_terrain(MapKind::River)).is_some());
    }

    #[test]
    fn clearance_matches_a_brute_force_distance() {
        let g = grid();
        let blocked: Vec<Vec2> = (0..g.walk.len()).filter(|&i| !g.walk[i]).map(|i| g.pos(i)).collect();
        for p in [Vec2::new(-341.0, -100.0), Vec2::new(0.0, -10.0), Vec2::new(340.0, -36.0), Vec2::new(10.0, -320.0)] {
            let i = g.cell_at(p).unwrap();
            let q = g.pos(i);
            let brute = blocked.iter().map(|b| b.distance(q)).fold(f32::MAX, f32::min).min(CLEAR_CAP);
            assert!((g.clear[i] - brute).abs() < 1e-3, "at {q}: {} vs {}", g.clear[i], brute);
        }
    }

    #[test]
    fn every_case_plans_a_walkable_route_and_the_box_is_out_of_reach() {
        let g = grid();
        for case in &CASES {
            let plan = g.plan(&req(case.start, case.goal));
            assert!(plan.points.len() >= 2, "{}: no route", case.name);
            assert_walkable(&g, &plan);
            if case.name == "box" {
                assert!(!plan.reached, "the box's goal was reached");
                let end = *plan.points.last().unwrap();
                assert!(end.distance(case.goal) < 70.0, "box: ended at {end}");
                assert!(g.clearance_at(end) >= 14.0);
            } else {
                assert!(plan.reached, "{}: goal not reached", case.name);
                assert!(!g.straight_ok(case.start, case.goal, 14.3), "{}: straight way clear", case.name);
            }
        }
    }

    #[test]
    fn the_narrow_gap_is_taken_and_the_wide_gap_route_keeps_room() {
        let g = grid();
        let narrow = &CASES[2];
        let plan = g.plan(&req(narrow.start, narrow.goal));
        assert!(plan.length() < narrow.start.distance(narrow.goal) * 1.2, "narrow: {}", plan.length());
        let wide = &CASES[1];
        let plan = g.plan(&req(wide.start, wide.goal));
        // It crosses the wall inside the wide gap, with room.
        let cross = plan
            .points
            .windows(2)
            .find_map(|w| {
                let (a, b) = (w[0], w[1]);
                ((a.y - -291.0) * (b.y - -291.0) <= 0.0).then(|| a.lerp(b, (-291.0 - a.y) / (b.y - a.y)))
            })
            .expect("the route crosses the wall");
        assert!(g.clearance_at(cross) >= 12.0, "wide: crossed at {cross}");
    }

    #[test]
    fn the_same_request_plans_the_same_route() {
        let g = grid();
        let a = g.plan(&req(CASES[4].start, CASES[4].goal));
        let b = g.plan(&req(CASES[4].start, CASES[4].goal));
        assert_eq!(a.points, b.points);
        assert_eq!(a.cells, b.cells);
        // The same search a budget at a time gives the same route.
        let mut search = g.begin(&req(CASES[4].start, CASES[4].goal));
        let mut ticks = 0;
        let c = loop {
            ticks += 1;
            if let Some(plan) = g.advance(&mut search, 4000) {
                break plan;
            }
        };
        assert_eq!(a.points, c.points);
        assert!(ticks > 5, "{ticks} ticks");
    }
}
