//! The stone wall across the hill map's plateau, on two maps: the walled
//! town (`FL_MAP=walled_town`), with a town of houses behind the wall, and
//! Wall Debug (`FL_MAP=wall_debug`), the same wall with no houses. Every
//! piece is listed once here (`pieces`, per map), and both the drawn
//! models (town.rs) and the sim's ground (`sim_ground`) come from that
//! list, so the two cannot drift.
//!
//! The sim reads the ground on a 2 m vertex grid: a soldier stands on
//! the vertex nearest him, and a vertex is open or blocked
//! (`Terrain::blocked_at`). Natural ground is blocked by the step rule
//! (a 4-neighbour 2 m or more away in height, `terrain::SLOPE_BLOCK`),
//! which cannot see a kerb, a rail or a stair leaning on a wall. So the
//! pieces state outright which vertices are open: each piece names its
//! clear surface (`Piece::surface`), and a vertex within a cell of a
//! piece is open when the 2 m square around it meets nothing solid, and
//! blocked otherwise. The ground beside a vertical face is open: a man
//! can stand at the foot of a wall. A soldier's body stops at the
//! boundary of the open vertices (`Terrain::body_free`), half a cell
//! from the blocked row less his own radius.
//!
//! - Wall units (2 m along the wall, 8 m deep, the walk at 8 m): the
//!   walk's clear 6 m is three open rows, the parapet and kerb strips
//!   are solid.
//! - Landing units: the kerb removed, so the clear surface reaches the
//!   stair's platform. Three per stair, the last two over the platform
//!   and one beside the flight's top, so the way onto the walk is two
//!   open columns wide.
//! - Drum towers: the passage floor through the disc at 8 m is clear,
//!   the rest of the disc solid (the roof no one reaches).
//! - The gatehouse: the bridge and the drums' passages at 8 m are clear,
//!   the drums' rest and the gate block solid. The ground passage is
//!   under the strip, so it is closed.
//! - Stairs: an 8 m rise over 12 m along the flight (1.33 m per cell,
//!   so the slope slows men but never blocks), then the platform at 8 m;
//!   the treads and the platform are clear, the rail along the open side
//!   and across the far end solid. `STAIR_W` is the model's width.
//! - Houses: their 10 by 14 m rectangle raised 6 m, solid.
//!
//! Pieces stand on levelled ground: the wall run on one flat terrace,
//! each house on its own pad, blended out into the hill's slopes. The
//! drawn ground is that levelled ground; the sim's copy adds the pieces.
//!
//! Vertices sit on even metres. The wall's line, its ends and the towers'
//! centres are even, so a strip's faces fall on vertex rows; the houses'
//! centres are odd, so their faces (5 and 7 m out) do too.
//!
//! The walk is also a place units are put: `walk_lane` is where a man's
//! centre may stand on it, and orders along it take the walk's own rows
//! (orders.rs `lane_layout`).

use bevy::math::Vec2;

use crate::terrain::{MapKind, CELL};

/// The wall's centre line (z, m): its outer face at +4 looks south down
/// the climb, the town lies north of its inner face at -4.
pub const WALL_Z: f32 = -130.0;
/// Half the wall's thickness (m): its faces at `WALL_Z` -4 and +4.
pub const WALL_HALF: f32 = 4.0;
/// The walk's height over the levelled ground (m).
pub const WALK: f32 = 8.0;
/// The drum roofs' height over the levelled ground (m).
pub const ROOF: f32 = 12.0;
/// The houses' sim height over their pads (m): the eaves.
pub const EAVES: f32 = 6.0;
/// The drum towers' centres along the wall (x, m), the gatehouse at 0.
pub const TOWERS_X: [f32; 4] = [-152.0, -80.0, 80.0, 152.0];
/// The stairs' feet (x, m), one per curtain, each rising toward +x.
pub const STAIRS_X: [f32; 4] = [-124.0, -52.0, 36.0, 108.0];
/// The stair model's width across the flight (m).
pub const STAIR_W: f32 = 4.0;
/// The stair's rail: this thick, along the flight's open side and across
/// the platform's far end.
const RAIL: f32 = 0.5;
/// How far the stair's rail, the walk's inner kerb and its outer parapet
/// rise over the floor beside them (m), and the merlons' tips over the
/// walk: one merlon 1 m wide at the middle of each 2 m wall unit, 1 m
/// gaps between, as in the wall's models. A drum's roof has the
/// same parapet and merlons around its rim.
const KERB: f32 = 0.5;
const MERLON: f32 = 1.5;
/// Half a merlon's width along the wall and the gatehouse's bridge (m).
const MERLON_HALF: f32 = 0.5;
/// How deep the wall's edge strips are, the parapet with its merlons and
/// the kerb, and a drum roof's rim (m).
const EDGE: f32 = 1.0;
/// Half the walk's clear width between the strips (m).
const WALK_HALF: f32 = WALL_HALF - EDGE;
/// The highest any piece reaches over its ground (m): the merlons on a
/// drum's roof.
pub const TALLEST: f32 = ROOF + MERLON;
/// A drum tower's radius (m) and the merlons around its roof's rim.
const TOWER_R: f32 = 8.0;
const TOWER_MERLONS: f32 = 24.0;
/// Half the width of a drum's passage at the walk's height, the towers'
/// and the gatehouse drums', and of the gatehouse's bridge (m).
const PASSAGE_HALF: f32 = 4.0;
/// The gatehouse's drums: their centres either side of the gate (x, m),
/// their radius (m) and the merlons around each roof's rim.
const GATE_DRUM_X: f32 = 10.0;
const GATE_DRUM_R: f32 = 6.0;
const GATE_DRUM_MERLONS: f32 = 18.0;
/// Half the gatehouse's gate block (m): it reaches 7 m either side of the
/// gate into the drums and 5 m out from the wall's line on both sides,
/// its edge strips past the bridge's 8 m.
const GATE_BLOCK: Vec2 = Vec2::new(7.0, 5.0);
/// The merlons on the gate block's outer strip (x, m).
const GATE_MERLONS_X: [f32; 4] = [-3.0, -1.0, 1.0, 3.0];
/// The stair's flight along its length (m), its platform after it.
const FLIGHT: f32 = 12.0;
/// The house grid: columns and rows of centres on a 30 m pitch.
const HOUSE_PITCH: f32 = 30.0;
const HOUSE_COLS: i32 = 10;
const HOUSE_ROWS: i32 = 5;
/// The first column's centre and the first row's (31 m behind the wall's
/// line: 16 m of street between the stairs and the houses' faces).
const HOUSE_X0: f32 = -135.0;
const HOUSE_Z0: f32 = WALL_Z - 31.0;
/// How far the levelled ground blends back into the hill (m).
const WALL_BLEND: f32 = 12.0;
const HOUSE_BLEND: f32 = 5.0;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum House {
    Humble,
    Market,
    Workshop,
    Merchant,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Kind {
    Wall,
    /// A wall unit with no inner kerb, where a stair's platform meets the
    /// walk.
    Landing,
    Stair,
    Tower,
    Gatehouse,
    House(House),
}

/// One placed piece: its model's origin on the ground and its turn about
/// the vertical in quarter turns (counter-clockwise seen from above, so
/// one turn faces the model's +z toward +x).
#[derive(Clone, Copy, Debug)]
pub struct Piece {
    pub kind: Kind,
    pub pos: Vec2,
    pub turns: u8,
}

/// What a point of a piece's footprint is to a soldier.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Surface {
    /// Floor a man walks on: the walk, a passage, a tread, the platform.
    Clear,
    /// A kerb, a parapet, a rail, a face, a roof: nothing stands there.
    Solid,
}

impl Piece {
    pub fn yaw(&self) -> f32 {
        self.turns as f32 * std::f32::consts::FRAC_PI_2
    }

    /// `(cos, sin)` of the yaw, exact.
    fn rot(&self) -> (f32, f32) {
        match self.turns % 4 {
            0 => (1.0, 0.0),
            1 => (0.0, 1.0),
            2 => (-1.0, 0.0),
            _ => (0.0, -1.0),
        }
    }

    /// The model's local x and z of the world point `p`.
    fn local(&self, p: Vec2) -> Vec2 {
        let (c, s) = self.rot();
        let d = p - self.pos;
        Vec2::new(d.x * c - d.y * s, d.x * s + d.y * c)
    }

    /// The model's local bounds on the ground, min and max.
    fn local_bounds(&self) -> (Vec2, Vec2) {
        match self.kind {
            Kind::Wall | Kind::Landing => (Vec2::new(-1.0, -4.0), Vec2::new(1.0, 4.0)),
            Kind::Stair => (Vec2::new(0.0, -STAIR_W), Vec2::new(16.0, 0.0)),
            Kind::Tower => (Vec2::splat(-8.0), Vec2::splat(8.0)),
            Kind::Gatehouse => (Vec2::new(-16.0, -6.0), Vec2::new(16.0, 6.0)),
            Kind::House(_) => (Vec2::new(-5.0, -7.0), Vec2::new(5.0, 7.0)),
        }
    }

    /// The world rectangle the piece stands on.
    pub fn bounds(&self) -> (Vec2, Vec2) {
        let (lo, hi) = self.local_bounds();
        let (c, s) = self.rot();
        let to_world = |l: Vec2| self.pos + Vec2::new(l.x * c + l.y * s, -l.x * s + l.y * c);
        let (a, b) = (to_world(lo), to_world(hi));
        (a.min(b), a.max(b))
    }

    /// The piece's height over the ground at the world point `p`, None
    /// off its footprint.
    fn raise(&self, p: Vec2) -> Option<f32> {
        const E: f32 = 0.01;
        let l = self.local(p);
        let (lo, hi) = self.local_bounds();
        if l.x < lo.x - E || l.x > hi.x + E || l.y < lo.y - E || l.y > hi.y + E {
            return None;
        }
        self.raise_local(l)
    }

    /// `raise` at the point of the footprint nearest `p`: the height of a
    /// part that reaches past the footprint, the stair's return rail over
    /// the wall's strip. Nothing where even that point is off the piece
    /// (a tower's square corner outside its disc).
    pub fn raise_clamped(&self, p: Vec2) -> f32 {
        let (lo, hi) = self.local_bounds();
        self.raise_local(self.local(p).clamp(lo, hi)).unwrap_or(0.0)
    }

    /// The top of the piece's solid part at `p` over the ground: the walk's
    /// inner kerb and outer parapet, the merlons, the stair's rail, a
    /// drum's roof and the parapet and merlons around its rim, the gate
    /// block's kerb, parapet and merlons, a house's eaves. What an arrow
    /// meets there (arrows.rs `arc_clears`).
    pub fn top(&self, p: Vec2) -> f32 {
        let (lo, hi) = self.local_bounds();
        let l = self.local(p).clamp(lo, hi);
        match self.kind {
            Kind::Wall | Kind::Landing if l.y > WALK_HALF => {
                WALK + if l.x.abs() <= MERLON_HALF { MERLON } else { KERB }
            }
            Kind::Wall | Kind::Landing => WALK + KERB,
            Kind::Stair => WALK * (l.x / FLIGHT).clamp(0.0, 1.0) + KERB,
            Kind::Tower if l.y.abs() > PASSAGE_HALF && l.length() <= TOWER_R => drum_top(l, TOWER_R, TOWER_MERLONS),
            Kind::Gatehouse => {
                let drum = |cx: f32| {
                    let d = l - Vec2::new(cx, 0.0);
                    (l.y.abs() > PASSAGE_HALF && d.length() <= GATE_DRUM_R).then(|| drum_top(d, GATE_DRUM_R, GATE_DRUM_MERLONS))
                };
                if let Some(h) = drum(-GATE_DRUM_X).or_else(|| drum(GATE_DRUM_X)) {
                    h
                } else if l.y > PASSAGE_HALF && l.x.abs() <= GATE_BLOCK.x {
                    // The outer strip: the parapet and its merlons.
                    let merlon = GATE_MERLONS_X.iter().any(|&x| (l.x - x).abs() <= MERLON_HALF);
                    WALK + if merlon { MERLON } else { KERB }
                } else if l.y < -PASSAGE_HALF && l.x.abs() <= GATE_BLOCK.x {
                    WALK + KERB
                } else {
                    self.raise_clamped(p)
                }
            }
            _ => self.raise_clamped(p),
        }
    }

    /// The rise per metre of the piece's floor at `p`, as a world vector:
    /// up the stair's flight, nothing on every level floor.
    pub fn rise(&self, p: Vec2) -> Vec2 {
        let l = self.local(p);
        if self.kind == Kind::Stair && (0.0..FLIGHT).contains(&l.x) {
            let (c, s) = self.rot();
            Vec2::new(c, -s) * (WALK / FLIGHT)
        } else {
            Vec2::ZERO
        }
    }

    /// `raise` at the local point `l`, which is on the footprint.
    fn raise_local(&self, l: Vec2) -> Option<f32> {
        const E: f32 = 0.01;
        match self.kind {
            Kind::Wall | Kind::Landing => Some(WALK),
            Kind::House(_) => Some(EAVES),
            // The flight, then the platform from 12 m on.
            Kind::Stair => Some(WALK * (l.x / FLIGHT).clamp(0.0, 1.0)),
            Kind::Tower => {
                if l.y.abs() <= PASSAGE_HALF + E {
                    Some(WALK)
                } else if l.length() <= TOWER_R + E {
                    Some(ROOF)
                } else {
                    None
                }
            }
            Kind::Gatehouse => {
                let drum = |cx: f32| (l - Vec2::new(cx, 0.0)).length() <= GATE_DRUM_R + E;
                if l.y.abs() <= PASSAGE_HALF + E {
                    Some(WALK)
                } else if drum(-GATE_DRUM_X) || drum(GATE_DRUM_X) {
                    Some(ROOF)
                } else if l.x.abs() <= GATE_BLOCK.x + E && l.y.abs() <= GATE_BLOCK.y + E {
                    // The gate block's edge strips, at the walk's height.
                    Some(WALK)
                } else {
                    None
                }
            }
        }
    }

    /// What the world point `p` is on this piece, None off it. The clear
    /// surfaces are the models': the walk's 6 m between the parapet and the
    /// kerb, the landing's 7 m, the tower passage's 8 m, the gatehouse's 8 m
    /// bridge and passages, the stair's treads and platform inside its rail.
    pub fn surface(&self, p: Vec2) -> Option<Surface> {
        let l = self.local(p);
        let (lo, hi) = self.local_bounds();
        let inside = l.x >= lo.x && l.x <= hi.x && l.y >= lo.y && l.y <= hi.y;
        match self.kind {
            Kind::Wall => inside.then_some(if l.y.abs() <= WALK_HALF { Surface::Clear } else { Surface::Solid }),
            Kind::Landing => inside.then_some(if l.y <= WALK_HALF { Surface::Clear } else { Surface::Solid }),
            Kind::House(_) => inside.then_some(Surface::Solid),
            Kind::Stair => {
                // The return rail across the far end reaches a metre over
                // the wall's inner strip.
                if l.x >= 16.0 - RAIL && l.x <= 16.0 && l.y >= -STAIR_W && l.y <= 1.0 {
                    return Some(Surface::Solid);
                }
                if !inside {
                    return None;
                }
                let rail = l.y <= -STAIR_W + RAIL || l.x >= 16.0 - RAIL;
                Some(if rail { Surface::Solid } else { Surface::Clear })
            }
            Kind::Tower => {
                if l.length() > TOWER_R {
                    None
                } else if l.y.abs() <= PASSAGE_HALF {
                    Some(Surface::Clear)
                } else {
                    Some(Surface::Solid)
                }
            }
            Kind::Gatehouse => {
                let drum = |cx: f32| (l - Vec2::new(cx, 0.0)).length() <= GATE_DRUM_R;
                let strip = inside && l.y.abs() <= PASSAGE_HALF;
                let block = l.x.abs() <= GATE_BLOCK.x && l.y.abs() <= GATE_BLOCK.y;
                if strip {
                    Some(Surface::Clear)
                } else if drum(-GATE_DRUM_X) || drum(GATE_DRUM_X) || block {
                    Some(Surface::Solid)
                } else {
                    None
                }
            }
        }
    }
}

/// The top of a drum's roof at `d` from its centre, inside its radius `r`:
/// the roof, and on the rim's outer `EDGE` the parapet with `merlons`
/// merlons, the first toward +x, each as wide as the gap after it (the
/// models' battlement, 24 merlons on a tower and 18 on a gatehouse drum).
fn drum_top(d: Vec2, r: f32, merlons: f32) -> f32 {
    if d.length() < r - EDGE {
        return ROOF;
    }
    let pitch = std::f32::consts::TAU / merlons;
    let off = (d.y.atan2(d.x) / pitch).round() * pitch - d.y.atan2(d.x);
    ROOF + if off.abs() <= pitch * 0.25 { MERLON } else { KERB }
}

/// The wall units of one curtain: centres every 2 m from `a` to `b`.
fn curtain(out: &mut Vec<Piece>, a: f32, b: f32) {
    let mut x = a;
    while x <= b + 0.01 {
        // A stair's platform spans 12 to 16 m from its foot: the two units
        // there are landings, and the one before them, beside the flight's
        // top, so the way onto the walk is two open columns wide (the
        // module docs).
        let landing = STAIRS_X.iter().any(|&s| x > s + 10.0 && x < s + 16.0);
        let kind = if landing { Kind::Landing } else { Kind::Wall };
        out.push(Piece { kind, pos: Vec2::new(x, WALL_Z), turns: 0 });
        x += 2.0;
    }
}

/// Every piece of the map `kind`, a map with the wall: the curtains west
/// to east, the towers, the gatehouse, the stairs, then on the walled
/// town the houses row by row.
pub fn pieces(kind: MapKind) -> Vec<Piece> {
    let mut out = Vec::new();
    // The first wall centres sit 7 m from a tower's centre and 15 m from
    // the gatehouse's, as the models join.
    let gate = Piece { kind: Kind::Gatehouse, pos: Vec2::new(0.0, WALL_Z), turns: 0 };
    let ends = [TOWERS_X[0], TOWERS_X[1], 0.0, TOWERS_X[2], TOWERS_X[3]];
    for w in ends.windows(2) {
        let from = w[0] + if w[0] == 0.0 { 15.0 } else { 7.0 };
        let to = w[1] - if w[1] == 0.0 { 15.0 } else { 7.0 };
        curtain(&mut out, from, to);
    }
    for &x in &TOWERS_X {
        out.push(Piece { kind: Kind::Tower, pos: Vec2::new(x, WALL_Z), turns: 0 });
    }
    out.push(gate);
    // The stair's origin is its foot on the wall's inner face.
    for &x in &STAIRS_X {
        out.push(Piece { kind: Kind::Stair, pos: Vec2::new(x, WALL_Z - 4.0), turns: 0 });
    }
    if kind == MapKind::WalledTown {
        houses(&mut out);
    }
    out
}

/// The houses: the grid less the plaza inside the gate and one in eight
/// for yards. The plaza's sides get merchant and market fronts facing
/// it, and the avenue north from it merchant and market houses; the rest
/// face one of the four ways at random.
fn houses(out: &mut Vec<Piece>) {
    let mut k = 0u32;
    for row in 0..HOUSE_ROWS {
        for col in 0..HOUSE_COLS {
            k += 1;
            let pos = Vec2::new(HOUSE_X0 + col as f32 * HOUSE_PITCH, HOUSE_Z0 - row as f32 * HOUSE_PITCH);
            let plaza = row < 2;
            let avenue = pos.x.abs() < 20.0;
            if plaza && avenue {
                continue;
            }
            let roll = |salt: u32| crate::units::hash01(k.wrapping_mul(0x9E37_79B1) ^ salt);
            let beside_plaza = plaza && pos.x.abs() < 50.0;
            if !avenue && !beside_plaza && roll(1) < 0.125 {
                continue;
            }
            let (kind, turns) = if beside_plaza {
                // Fronts toward the plaza: the east side faces west.
                let kind = if row == 0 { House::Merchant } else { House::Market };
                (kind, if pos.x > 0.0 { 3 } else { 1 })
            } else if avenue {
                // Fronts toward the avenue would put a house's 14 m length
                // across it; these face its ends, toward a neighbour, so
                // the avenue stays 20 m wide.
                let kind = if row % 2 == 0 { House::Merchant } else { House::Market };
                (kind, if roll(2) < 0.5 { 0 } else { 2 })
            } else {
                let kind = match (roll(3) * 5.0) as u32 {
                    0 | 1 => House::Humble,
                    2 => House::Workshop,
                    3 => House::Market,
                    _ => House::Merchant,
                };
                (kind, (roll(4) * 4.0) as u8 % 4)
            };
            out.push(Piece { kind: Kind::House(kind), pos, turns });
        }
    }
}

/// A flat pad the ground is levelled to, blended back over `blend` m.
struct Pad {
    lo: Vec2,
    hi: Vec2,
    blend: f32,
}

impl Pad {
    /// Vertex index ranges of the pad with its blend, clipped to the grid.
    fn span(&self, origin: Vec2, (vx, vz): (usize, usize), extra: f32) -> (usize, usize, usize, usize) {
        let lo = ((self.lo - extra - origin) / CELL).floor().max(Vec2::ZERO);
        let hi = ((self.hi + extra - origin) / CELL).ceil();
        (lo.x as usize, lo.y as usize, (hi.x as usize).min(vx - 1), (hi.y as usize).min(vz - 1))
    }

    /// Level `heights` to `level` on the pad, blending back out.
    fn apply(&self, heights: &mut [f32], origin: Vec2, dims: (usize, usize), level: f32) {
        let (x0, z0, x1, z1) = self.span(origin, dims, self.blend);
        for z in z0..=z1 {
            for x in x0..=x1 {
                let p = origin + Vec2::new(x as f32, z as f32) * CELL;
                let d = (self.lo - p).max(p - self.hi).max(Vec2::ZERO).length();
                let t = (d / self.blend).clamp(0.0, 1.0);
                let w = 1.0 - t * t * (3.0 - 2.0 * t);
                let h = &mut heights[z * dims.0 + x];
                *h += (level - *h) * w;
            }
        }
    }

    /// The lowest and highest height on the pad, and the mean.
    fn stats(&self, heights: &[f32], origin: Vec2, dims: (usize, usize)) -> (f32, f32, f32) {
        let (x0, z0, x1, z1) = self.span(origin, dims, 0.0);
        let (mut lo, mut hi, mut sum, mut n) = (f32::INFINITY, f32::NEG_INFINITY, 0.0, 0.0);
        for z in z0..=z1 {
            for x in x0..=x1 {
                let h = heights[z * dims.0 + x];
                lo = lo.min(h);
                hi = hi.max(h);
                sum += h;
                n += 1.0;
            }
        }
        (lo, hi, sum / n)
    }
}

/// The wall run's pad: everything on the wall line, from the west
/// tower's edge to the east one's and from the stairs to the drums'
/// fronts, with a cell of margin.
fn wall_pad(pieces: &[Piece]) -> Pad {
    let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
    for p in pieces.iter().filter(|p| !matches!(p.kind, Kind::House(_))) {
        let (a, b) = p.bounds();
        lo = lo.min(a);
        hi = hi.max(b);
    }
    Pad { lo: lo - CELL, hi: hi + CELL, blend: WALL_BLEND }
}

/// Level the hill's heights for the town: the wall run to the middle of
/// the heights under it (so it cuts and fills the least), then each house
/// to the mean under its own pad. Returns the wall run's level.
pub fn level_ground(heights: &mut [f32], origin: Vec2, dims: (usize, usize), pieces: &[Piece]) -> f32 {
    let pad = wall_pad(pieces);
    let (lo, hi, _) = pad.stats(heights, origin, dims);
    let wall_level = (lo + hi) * 0.5;
    pad.apply(heights, origin, dims, wall_level);
    for p in pieces.iter().filter(|p| matches!(p.kind, Kind::House(_))) {
        let (a, b) = p.bounds();
        let pad = Pad { lo: a - CELL, hi: b + CELL, blend: HOUSE_BLEND };
        let (_, _, mean) = pad.stats(heights, origin, dims);
        pad.apply(heights, origin, dims, mean);
    }
    wall_level
}

/// The sim's ground over the town: the heights with every piece raised
/// on the levelled ground, and per vertex whether it is open or blocked
/// where a piece decides that (None where the step rule does).
pub struct SimGround {
    pub heights: Vec<f32>,
    pub open: Vec<Option<bool>>,
}

/// The sim's ground: the levelled ground with every piece raised on it
/// (where pieces overlap the higher wins, a drum over the walk's strip),
/// and the open vertices by the pieces' surfaces (the module docs): a
/// vertex within a cell of a piece is open unless the 2 m square around
/// it meets something solid.
pub fn sim_ground(ground: &[f32], origin: Vec2, dims: (usize, usize), pieces: &[Piece]) -> SimGround {
    let mut raise = vec![0.0f32; ground.len()];
    let mut near: Vec<Vec<u16>> = vec![Vec::new(); ground.len()];
    for (k, p) in pieces.iter().enumerate() {
        let (a, b) = p.bounds();
        let pad = Pad { lo: a, hi: b, blend: 0.0 };
        let (x0, z0, x1, z1) = pad.span(origin, dims, 0.0);
        for z in z0..=z1 {
            for x in x0..=x1 {
                let i = z * dims.0 + x;
                if let Some(r) = p.raise(origin + Vec2::new(x as f32, z as f32) * CELL) {
                    raise[i] = raise[i].max(r);
                }
            }
        }
        let (x0, z0, x1, z1) = pad.span(origin, dims, CELL);
        for z in z0..=z1 {
            for x in x0..=x1 {
                near[z * dims.0 + x].push(k as u16);
            }
        }
    }
    let heights: Vec<f32> = ground.iter().zip(&raise).map(|(g, r)| g + r).collect();
    // The square's samples stop just short of its edges, so a surface
    // that ends exactly on the neighbouring vertex's line is not read.
    const D: f32 = 0.99;
    let solid: Vec<Option<bool>> = near
        .iter()
        .enumerate()
        .map(|(i, cands)| {
            if cands.is_empty() {
                return None;
            }
            let v = origin + Vec2::new((i % dims.0) as f32, (i / dims.0) as f32) * CELL;
            Some([-D, 0.0, D].iter().any(|&dx| {
                [-D, 0.0, D].iter().any(|&dz| {
                    let s = v + Vec2::new(dx, dz);
                    cands.iter().any(|&k| pieces[k as usize].surface(s) == Some(Surface::Solid))
                })
            }))
        })
        .collect();
    // A vertex that meets nothing solid is still blocked where a cliff
    // runs between it and a neighbour that meets nothing solid either:
    // an end tower's passage opening onto air, a platform's edge. Solid
    // neighbours do not count, so the ground at a face stays open.
    let cliff = crate::terrain::SLOPE_BLOCK * CELL;
    let open = (0..heights.len())
        .map(|i| {
            let s = solid[i]?;
            if s {
                return Some(false);
            }
            let (x, z) = (i % dims.0, i / dims.0);
            let mut around = Vec::with_capacity(4);
            if x > 0 {
                around.push(i - 1);
            }
            if x + 1 < dims.0 {
                around.push(i + 1);
            }
            if z > 0 {
                around.push(i - dims.0);
            }
            if z + 1 < dims.1 {
                around.push(i + dims.0);
            }
            let drop = around
                .iter()
                .any(|&j| solid[j] != Some(true) && (heights[i] - heights[j]).abs() >= cliff);
            Some(!drop)
        })
        .collect();
    SimGround { heights, open }
}

/// The fine ground's cell (m): a quarter of a metre, so a man's body
/// stops within an eighth of a metre of a rail, a kerb or a face.
pub const FINE: f32 = 0.25;
/// The highest step a man takes from one floor onto the next (m): two of
/// the stair model's 0.2 m risers. Between two fine cells a man could
/// stand on, a rise this high or more is a cliff.
const STEP: f32 = 0.5;
/// A fine cell on no piece.
const NO_PIECE: u16 = u16::MAX;
/// A fine cell's flags: no man stands there; a piece's solid part.
const F_BLOCKED: u8 = 1;
const F_SOLID: u8 = 2;

/// The ground over the pieces as the sim reads it, on a `FINE` grid: per
/// cell, whether a man may stand there and which piece's surface it is.
/// The 2 m vertex grid can open only whole 2 m squares, so on a stair
/// whose clear tread is 3.5 m wide it opens one row of vertices and gives
/// a man's centre 0.75 m of it, and a point beside a face blends the two
/// levels across the cell. Read here, a man stands where the model's
/// floor is, at the floor's own height.
///
/// A cell is solid where its centre is on a piece's solid part, clear
/// where it is on a piece's clear surface and on no solid part, and
/// otherwise off the pieces, where the drawn ground's own step rule
/// (`natural_blocked`) decides. A clear or natural cell is also blocked
/// where it lies a `STEP` or more above or below a 4-neighbour that is
/// not solid: an end tower's passage onto air. Its height is the drawn
/// ground plus the piece's (`Piece::raise`), exact at any point, so the
/// stair is one slope from its foot to its platform.
#[derive(Clone)]
pub struct FineGround {
    origin: Vec2,
    nx: usize,
    nz: usize,
    piece: Vec<u16>,
    flags: Vec<u8>,
    /// Per cell, the distance to the nearest blocked cell (m), capped at
    /// `CLEAR_CAP`: how far a straight line steps from it.
    clear: Vec<f32>,
    pieces: Vec<Piece>,
    /// Per `BLOCK` of cells, row-major `bx` wide: whether a piece's bounds,
    /// a metre and a cell more, reach into it.
    by_piece: Vec<bool>,
    bx: usize,
    bz: usize,
}

/// The fine ground's clearance cap (m).
const CLEAR_CAP: f32 = 8.0;
/// The side of the fine ground's square blocks of cells (cells): 4 m.
const BLOCK: usize = 16;

impl FineGround {
    /// The cell holding `(x, z)`, None off the grid.
    #[inline]
    fn index(&self, x: f32, z: f32) -> Option<usize> {
        let g = (Vec2::new(x, z) - self.origin) / FINE;
        if g.x < 0.0 || g.y < 0.0 {
            return None;
        }
        let (i, k) = (g.x as usize, g.y as usize);
        (i < self.nx && k < self.nz).then(|| k * self.nx + i)
    }

    /// Whether `(x, z)` is on the grid, and if so whether a man may not
    /// stand there.
    #[inline]
    pub fn blocked(&self, x: f32, z: f32) -> Option<bool> {
        self.index(x, z).map(|i| self.flags[i] & F_BLOCKED != 0)
    }

    /// Whether `(x, z)` is in a block a piece reaches into (false off the
    /// grid): where a reader of the pieces' heights reads every cell.
    #[inline]
    pub fn by_a_piece(&self, x: f32, z: f32) -> bool {
        self.index(x, z).is_some_and(|_| {
            let b = ((Vec2::new(x, z) - self.origin) / (FINE * BLOCK as f32)).floor();
            self.by_piece[b.y as usize * self.bx + b.x as usize]
        })
    }

    /// How far along `dir` (a unit vector) from `p` the line reaches the
    /// first block a piece reaches into (`by_a_piece`), within `max` m: 0
    /// where `p` is in one, None where the line meets none by then. A
    /// reader that strides over open ground stops there, so no stride
    /// steps over a piece. Walks the line block by block: one or two
    /// blocks per 4 m of `max`.
    pub fn to_piece(&self, p: Vec2, dir: Vec2, max: f32) -> Option<f32> {
        let size = FINE * BLOCK as f32;
        let mut t = if self.index(p.x, p.y).is_some() { 0.0 } else { self.entry(p, dir)? };
        if t > max {
            return None;
        }
        let g = (p + dir * t - self.origin) / size;
        let (mut bx, mut bz) = ((g.x.floor().max(0.0) as usize).min(self.bx - 1), (g.y.floor().max(0.0) as usize).min(self.bz - 1));
        // Per axis: the distance along the line to the block's next edge,
        // and between edges; infinite along an axis the line keeps to.
        let axis = |p: f32, d: f32, o: f32, b: usize| {
            if d.abs() < 1e-9 {
                (f32::INFINITY, f32::INFINITY)
            } else {
                let edge = o + (b as f32 + if d > 0.0 { 1.0 } else { 0.0 }) * size;
                ((edge - p) / d, size / d.abs())
            }
        };
        let (mut next_x, step_x) = axis(p.x, dir.x, self.origin.x, bx);
        let (mut next_z, step_z) = axis(p.y, dir.y, self.origin.y, bz);
        loop {
            if self.by_piece[bz * self.bx + bx] {
                return Some(t);
            }
            t = next_x.min(next_z);
            if t > max {
                return None;
            }
            // Into the neighbouring block, None past the grid's.
            let (b, n, d, step, next) =
                if next_x < next_z { (&mut bx, self.bx, dir.x, step_x, &mut next_x) } else { (&mut bz, self.bz, dir.y, step_z, &mut next_z) };
            *next += step;
            if d > 0.0 {
                *b += 1;
                if *b >= n {
                    return None;
                }
            } else if *b == 0 {
                return None;
            } else {
                *b -= 1;
            }
        }
    }

    /// How far along `dir` (a unit vector) from `p` the grid begins: None
    /// where `p` is on it already or the line never meets it.
    fn entry(&self, p: Vec2, dir: Vec2) -> Option<f32> {
        let (lo, hi) = (self.origin, self.origin + Vec2::new(self.nx as f32, self.nz as f32) * FINE);
        let (mut t0, mut t1) = (0.0f32, f32::INFINITY);
        for (p, d, lo, hi) in [(p.x, dir.x, lo.x, hi.x), (p.y, dir.y, lo.y, hi.y)] {
            if d.abs() < 1e-9 {
                if p < lo || p >= hi {
                    return None;
                }
                continue;
            }
            let (a, b) = ((lo - p) / d, (hi - p) / d);
            t0 = t0.max(a.min(b));
            t1 = t1.min(a.max(b));
        }
        (t0 > 0.0 && t0 <= t1).then_some(t0)
    }

    /// Whether `(x, z)` is on a piece's solid part (false off the grid).
    #[inline]
    pub fn solid(&self, x: f32, z: f32) -> bool {
        self.index(x, z).is_some_and(|i| self.flags[i] & F_SOLID != 0)
    }

    /// On the grid: the height the pieces add over the drawn ground at
    /// `(x, z)`, nothing off them: a floor's height on a clear cell, the
    /// top of the solid part on a solid one (`Piece::top`).
    #[inline]
    pub fn raise(&self, x: f32, z: f32) -> Option<f32> {
        let i = self.index(x, z)?;
        let k = self.piece[i];
        Some(if k == NO_PIECE {
            0.0
        } else if self.flags[i] & F_SOLID != 0 {
            self.pieces[k as usize].top(Vec2::new(x, z))
        } else {
            self.pieces[k as usize].raise_clamped(Vec2::new(x, z))
        })
    }

    /// On the grid: the rise per metre the pieces add at `(x, z)`, as a
    /// world vector: the stair's flight, nothing elsewhere.
    #[inline]
    pub fn rise(&self, x: f32, z: f32) -> Option<Vec2> {
        let i = self.index(x, z)?;
        let k = self.piece[i];
        Some(if k == NO_PIECE || self.flags[i] & F_SOLID != 0 {
            Vec2::ZERO
        } else {
            self.pieces[k as usize].rise(Vec2::new(x, z))
        })
    }

    /// On the grid: None where `p` is blocked, else how far a straight
    /// line from `p` may step before it can meet a blocked cell (its
    /// cell's clearance less a cell's diagonal, at least a quarter cell).
    #[inline]
    pub fn line_step(&self, p: Vec2) -> Option<Option<f32>> {
        let i = self.index(p.x, p.y)?;
        Some((self.flags[i] & F_BLOCKED == 0).then(|| (self.clear[i] - FINE * std::f32::consts::SQRT_2).max(FINE * 0.25)))
    }

    /// The cell's edges along `axis` (0 for x, 1 for z) at `(x, z)`, None
    /// off the grid: where a body stops at a blocked cell.
    #[inline]
    pub fn edges(&self, x: f32, z: f32, axis: usize) -> Option<(f32, f32)> {
        self.index(x, z)?;
        let (p, o) = if axis == 0 { (x, self.origin.x) } else { (z, self.origin.y) };
        let lo = o + ((p - o) / FINE).floor() * FINE;
        Some((lo, lo + FINE))
    }
}

/// The fine ground over `pieces` (`FineGround`). `ground` is the drawn
/// ground's height at a point and `natural_blocked` its step rule there,
/// for the cells off the pieces.
pub fn fine_ground(
    pieces: &[Piece],
    ground: impl Fn(Vec2) -> f32,
    natural_blocked: impl Fn(Vec2) -> bool,
) -> FineGround {
    // The pieces' bounds and 2 m more on every side: the stair's return
    // rail reaches a metre past its own.
    let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
    for p in pieces {
        let (a, b) = p.bounds();
        lo = lo.min(a);
        hi = hi.max(b);
    }
    let origin = lo - Vec2::splat(2.0);
    let (nx, nz) = (((hi.x - origin.x + 2.0) / FINE).ceil() as usize, ((hi.y - origin.y + 2.0) / FINE).ceil() as usize);
    let centre = |i: usize, k: usize| origin + (Vec2::new(i as f32, k as f32) + 0.5) * FINE;
    // The pieces near each 4 m block of cells, so a cell asks a few.
    let (bx, bz) = (nx.div_ceil(BLOCK), nz.div_ceil(BLOCK));
    let mut near: Vec<Vec<u16>> = vec![Vec::new(); bx * bz];
    for (k, p) in pieces.iter().enumerate() {
        let (a, b) = p.bounds();
        let (a, b) = (a - Vec2::splat(1.0 + FINE), b + Vec2::splat(1.0 + FINE));
        let to = |v: Vec2| (((v - origin) / (FINE * BLOCK as f32)).max(Vec2::ZERO)).floor();
        let (ia, ib) = (to(a), to(b));
        for zb in ia.y as usize..=(ib.y as usize).min(bz - 1) {
            for xb in ia.x as usize..=(ib.x as usize).min(bx - 1) {
                near[zb * bx + xb].push(k as u16);
            }
        }
    }
    let mut piece = vec![NO_PIECE; nx * nz];
    let mut flags = vec![0u8; nx * nz];
    for k in 0..nz {
        for i in 0..nx {
            let c = centre(i, k);
            // The highest solid part and the highest clear surface there.
            let (mut solid, mut clear) = (None::<(f32, u16)>, None::<(f32, u16)>);
            for &j in &near[(k / BLOCK) * bx + i / BLOCK] {
                let p = &pieces[j as usize];
                let h = p.raise_clamped(c);
                let best = |b: Option<(f32, u16)>| if b.is_none_or(|(bh, _)| h > bh) { Some((h, j)) } else { b };
                match p.surface(c) {
                    Some(Surface::Solid) => solid = best(solid),
                    Some(Surface::Clear) => clear = best(clear),
                    None => {}
                }
            }
            let at = k * nx + i;
            if let Some((_, j)) = solid {
                piece[at] = j;
                flags[at] = F_SOLID | F_BLOCKED;
            } else if let Some((_, j)) = clear {
                piece[at] = j;
            } else if natural_blocked(c) {
                flags[at] = F_BLOCKED;
            }
        }
    }
    // A cliff between two cells a man could stand on blocks both: the side
    // of the flight's top beside the landing, a tower's passage onto air.
    let height = |i: usize, k: usize| {
        let c = centre(i, k);
        let j = piece[k * nx + i];
        ground(c) + if j == NO_PIECE { 0.0 } else { pieces[j as usize].raise_clamped(c) }
    };
    let cliff = STEP;
    let mut drop = vec![false; nx * nz];
    for k in 0..nz {
        for i in 0..nx {
            if flags[k * nx + i] & F_SOLID != 0 {
                continue;
            }
            let h = height(i, k);
            let mut look = |i2: usize, k2: usize| {
                if flags[k2 * nx + i2] & F_SOLID == 0 && (height(i2, k2) - h).abs() >= cliff {
                    drop[k * nx + i] = true;
                }
            };
            if i > 0 {
                look(i - 1, k);
            }
            if i + 1 < nx {
                look(i + 1, k);
            }
            if k > 0 {
                look(i, k - 1);
            }
            if k + 1 < nz {
                look(i, k + 1);
            }
        }
    }
    for (f, d) in flags.iter_mut().zip(&drop) {
        if *d {
            *f |= F_BLOCKED;
        }
    }
    let open: Vec<bool> = flags.iter().map(|f| f & F_BLOCKED == 0).collect();
    let clear = crate::nav::clearance(&open, nx, nz, FINE, CLEAR_CAP);
    let by_piece = near.iter().map(|n| !n.is_empty()).collect();
    FineGround { origin, nx, nz, piece, flags, clear, pieces: pieces.to_vec(), by_piece, bx, bz }
}

/// A straight stretch of ground with fixed rows a unit stands in: the
/// wall's walk. Along x, facing +z.
#[derive(Clone, Copy, Debug)]
pub struct Lane {
    /// Where a man's centre may stand across the lane (z), back to front.
    pub across: (f32, f32),
    /// Where a man's centre may stand along it (x).
    pub along: (f32, f32),
}

impl Lane {
    /// `p` is on the lane: on the walk's open rows, between its ends.
    pub fn contains(&self, p: Vec2) -> bool {
        (p.y - WALL_Z).abs() <= 3.0 && p.x >= self.along.0 && p.x <= self.along.1
    }

    /// An order at `p` means the lane: anywhere on the wall, its strips
    /// and faces included, since a click or a drag end on the parapet,
    /// the kerb or a face is meant for the walk. The picked point of a
    /// click on a face lies on the cell beside it, a cell further out
    /// than the wall's 8 m.
    pub fn catches(&self, p: Vec2) -> bool {
        (p.y - WALL_Z).abs() <= 6.0 && p.x >= self.along.0 && p.x <= self.along.1
    }

    /// The yaw of a unit standing on the lane: facing +z, out over the
    /// wall toward the plain.
    pub fn facing(&self) -> f32 {
        crate::formation::facing_of(Vec2::Y)
    }
}

/// The walk as a lane for a man of radius `r`: its three open rows give
/// his centre 6 m, less his radius at the kerb and the parapet; its open
/// columns run from the west tower's far side to the east one's.
pub fn walk_lane(r: f32) -> Lane {
    let (west, east) = (TOWERS_X[0] - 7.0, TOWERS_X[3] + 7.0);
    Lane { across: (WALL_Z - 3.0 + r, WALL_Z + 3.0 - r), along: (west + r, east - r) }
}

/// Points the scripted scenes and the tests use.
pub mod spots {
    use super::*;

    /// Open plateau north of the town, where the defender stands.
    pub const BEHIND_TOWN: Vec2 = Vec2::new(0.0, -330.0);
    /// The plaza inside the gate.
    pub const PLAZA: Vec2 = Vec2::new(0.0, -175.0);
    /// The plain at the foot of the climb.
    pub const PLAIN: Vec2 = Vec2::new(0.0, 200.0);
    /// On the walk of the eastern curtains, between the towers at x = 80
    /// and x = 152.
    pub const WALK_EAST: Vec2 = Vec2::new(130.0, WALL_Z);
    /// On the walk of the western curtains, west of the gatehouse.
    pub const WALK_WEST: Vec2 = Vec2::new(-30.0, WALL_Z);
    /// The foot of the stair east of the gate.
    pub const STAIR_FOOT_EAST: Vec2 = Vec2::new(STAIRS_X[2], WALL_Z - 4.0 - STAIR_W * 0.5);
    /// On the gatehouse's bridge.
    #[cfg(test)]
    pub const BRIDGE: Vec2 = Vec2::new(0.0, WALL_Z);
    /// Right outside the gate, on the attackers' side.
    #[cfg(test)]
    pub const OUTSIDE_GATE: Vec2 = Vec2::new(0.0, WALL_Z + 20.0);
    /// A stair's middle row, halfway up its flight.
    #[cfg(test)]
    pub fn stair_mid(i: usize) -> Vec2 {
        Vec2::new(STAIRS_X[i] + 6.0, WALL_Z - 6.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::build_terrain;

    /// Wall Debug is the walled town less its houses: the same wall pieces
    /// in the same order, so its ground and models are the town's wall.
    #[test]
    fn the_wall_debug_pieces_are_the_walled_town_less_its_houses() {
        let key = |p: &Piece| (p.kind, p.pos.x.to_bits(), p.pos.y.to_bits(), p.turns);
        let town: Vec<_> = pieces(MapKind::WalledTown)
            .iter()
            .filter(|p| !matches!(p.kind, Kind::House(_)))
            .map(key)
            .collect();
        let debug = pieces(MapKind::WallDebug);
        assert_eq!(debug.iter().map(key).collect::<Vec<_>>(), town);
        let count = |k: Kind| debug.iter().filter(|p| p.kind == k).count();
        assert!(!debug.iter().any(|p| matches!(p.kind, Kind::House(_))));
        assert_eq!(count(Kind::Tower), 4);
        assert_eq!(count(Kind::Stair), 4);
        assert_eq!(count(Kind::Gatehouse), 1);
    }

    /// A stair's floor is one slope from its foot to its platform at
    /// every point across its tread: the drawn ground plus 8 m over the
    /// 12 m flight, with no step where the 2 m grid's cells meet.
    #[test]
    fn a_stairs_floor_is_one_slope_across_its_whole_tread() {
        let t = build_terrain(MapKind::WallDebug);
        for &s in &STAIRS_X {
            let foot = WALL_Z - 4.0;
            let mut z = foot - STAIR_W + RAIL + 0.05;
            while z < foot {
                let mut lx = 0.05f32;
                while lx < 12.0 {
                    let (x, h) = (s + lx, t.height_at(s + lx, z));
                    let want = t.ground_at(x, z) + WALK * lx / 12.0;
                    assert!((h - want).abs() < 1e-3, "stair at {s}: height {h} at ({x}, {z}), the slope gives {want}");
                    lx += 0.05;
                }
                z += 0.1;
            }
        }
    }

    /// On a stair a man's centre may stand anywhere on the tread less his
    /// body's radius on either side: from the rail to the wall's face, as
    /// wide as the model's 3.5 m tread, to within a fine cell.
    #[test]
    fn a_man_stands_across_a_stairs_whole_tread() {
        let t = build_terrain(MapKind::WallDebug);
        let r = t.body_radius();
        let foot = WALL_Z - 4.0;
        for &s in &STAIRS_X {
            for lx in [1.0f32, 4.0, 8.0] {
                let free = |z: f32| t.body_free(s + lx, z, r, 0.0, 1.0) && t.body_free(s + lx, z, r, 0.0, -1.0);
                let (mut lo, mut hi) = (foot - 2.0, foot - 2.0);
                assert!(free(lo), "stair at {s}: the tread's middle at {lx} m is not free");
                while free(lo - 0.01) {
                    lo -= 0.01;
                }
                while free(hi + 0.01) {
                    hi += 0.01;
                }
                let (want_lo, want_hi) = (foot - STAIR_W + RAIL + r, foot - r);
                assert!((lo - want_lo).abs() <= FINE && (hi - want_hi).abs() <= FINE,
                    "stair at {s}, {lx} m up: centres from {lo} to {hi}, the tread gives {want_lo} to {want_hi}");
            }
        }
    }

    /// A man pushed at a stair's rail stops with his body's edge at the
    /// rail, not with his edge past it on the ground beyond: the rail is
    /// thinner than his radius.
    #[test]
    fn a_man_pushed_at_a_rail_stops_at_the_rail() {
        let t = build_terrain(MapKind::WallDebug);
        let r = t.body_radius();
        let rail = WALL_Z - 4.0 - STAIR_W + RAIL;
        for &s in &STAIRS_X {
            for lx in [3.0f32, 9.0, 14.0] {
                let x = s + lx;
                assert!(!t.body_free(x, rail + r - 0.05, r, 0.0, -1.0), "stair at {s}, {lx} m: a body steps into the rail");
                assert!(t.body_free(x, rail + r + 0.05, r, 0.0, -1.0), "stair at {s}, {lx} m: a body stops short of the rail");
                let off = t.body_overlap(x, rail + r - 0.2, r);
                assert!((off.y - 0.2).abs() < 1e-3, "stair at {s}, {lx} m: a body 0.2 m into the rail moves {off}");
            }
        }
    }

    /// The wall's terrace is levelled before the house pads, so the wall
    /// stands at the same height on both maps: the walk, the bridge, a
    /// tower's passage and the stairs, in the sim's ground and the drawn.
    #[test]
    fn the_wall_stands_at_the_same_height_on_both_maps() {
        let (town, debug) = (build_terrain(MapKind::WalledTown), build_terrain(MapKind::WallDebug));
        let mut points = vec![spots::WALK_WEST, spots::WALK_EAST, spots::BRIDGE, Vec2::new(TOWERS_X[1], WALL_Z)];
        points.extend((0..STAIRS_X.len()).map(spots::stair_mid));
        points.push(spots::STAIR_FOOT_EAST);
        for p in points {
            assert_eq!(town.height_at(p.x, p.y), debug.height_at(p.x, p.y), "the sim's height at {p}");
            assert_eq!(town.ground_at(p.x, p.y), debug.ground_at(p.x, p.y), "the drawn ground at {p}");
            assert_eq!(town.blocked_at(p.x, p.y), debug.blocked_at(p.x, p.y), "blocked at {p}");
        }
    }
}
