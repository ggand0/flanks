//! The walled town (`FL_MAP=walled_town`): the hill map's ground with a
//! stone wall across the plateau and a town of houses behind it. Every
//! piece is listed once here (`pieces`), and both the drawn models
//! (town.rs) and the sim's ground (`sim_ground`) come from that list, so
//! the two cannot drift.
//!
//! The sim reads the ground on a 2 m vertex grid: a soldier stands on
//! the vertex nearest him, and a vertex is open or blocked
//! (`Terrain::blocked_at`). Natural ground is blocked by the step rule
//! (a 4-neighbour 2 m or more away in height, `terrain::SLOPE_BLOCK`),
//! which cannot see a kerb, a rail or a stair leaning on a wall. So the
//! pieces say outright which vertices are open: each piece names its
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

use crate::terrain::CELL;

/// The wall's centre line (z, m): its outer face at +4 looks south down
/// the climb, the town lies north of its inner face at -4.
pub const WALL_Z: f32 = -130.0;
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
/// The stair model's width across the flight (m): 4 m as delivered
/// (handoff 152), 6 m once the wider one lands (handoff 155).
pub const STAIR_W: f32 = 4.0;
/// The stair's rail: this thick, along the flight's open side and across
/// the platform's far end.
const RAIL: f32 = 0.5;
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
        match self.kind {
            Kind::Wall | Kind::Landing => Some(WALK),
            Kind::House(_) => Some(EAVES),
            // The flight, then the platform from 12 m on.
            Kind::Stair => Some(WALK * (l.x / 12.0).clamp(0.0, 1.0)),
            Kind::Tower => {
                if l.y.abs() <= 4.0 + E {
                    Some(WALK)
                } else if l.length() <= 8.0 + E {
                    Some(ROOF)
                } else {
                    None
                }
            }
            Kind::Gatehouse => {
                let drum = |cx: f32| (l - Vec2::new(cx, 0.0)).length() <= 6.0 + E;
                if l.y.abs() <= 4.0 + E {
                    Some(WALK)
                } else if drum(-10.0) || drum(10.0) {
                    Some(ROOF)
                } else {
                    None
                }
            }
        }
    }

    /// What the world point `p` is on this piece, None off it. The clear
    /// surfaces are the models' (handoffs 152 and 153): the walk's 6 m
    /// between the parapet and the kerb, the landing's 7 m, the tower
    /// passage's 8 m, the gatehouse's 8 m bridge and passages, the stair's
    /// treads and platform inside its rail.
    pub fn surface(&self, p: Vec2) -> Option<Surface> {
        let l = self.local(p);
        let (lo, hi) = self.local_bounds();
        let inside = l.x >= lo.x && l.x <= hi.x && l.y >= lo.y && l.y <= hi.y;
        match self.kind {
            Kind::Wall => inside.then_some(if l.y.abs() <= 3.0 { Surface::Clear } else { Surface::Solid }),
            Kind::Landing => inside.then_some(if l.y <= 3.0 { Surface::Clear } else { Surface::Solid }),
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
                if l.length() > 8.0 {
                    None
                } else if l.y.abs() <= 4.0 {
                    Some(Surface::Clear)
                } else {
                    Some(Surface::Solid)
                }
            }
            Kind::Gatehouse => {
                let drum = |cx: f32| (l - Vec2::new(cx, 0.0)).length() <= 6.0;
                let strip = inside && l.y.abs() <= 4.0;
                let block = l.x.abs() <= 5.0 && l.y.abs() <= 5.0;
                if strip {
                    Some(Surface::Clear)
                } else if drum(-10.0) || drum(10.0) || block {
                    Some(Surface::Solid)
                } else {
                    None
                }
            }
        }
    }
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

/// Every piece of the map: the wall run west to east, the stairs, then
/// the houses row by row.
pub fn pieces() -> Vec<Piece> {
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
    houses(&mut out);
    out
}

/// The houses: the grid less the plaza inside the gate and one in eight
/// for yards. The plaza's sides and the avenue north from it get merchant
/// and market fronts facing them; the rest stand every way round.
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
                // Fronts toward the avenue's ends would face a neighbour;
                // these turn their narrow side to it, so it stays 20 m.
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

    /// The lane's direction and its facing (yaw).
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
    /// On the walk of the eastern curtains, east of the second tower.
    pub const WALK_EAST: Vec2 = Vec2::new(130.0, WALL_Z);
    /// On the walk of the western curtains, west of the gatehouse.
    pub const WALK_WEST: Vec2 = Vec2::new(-30.0, WALL_Z);
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
