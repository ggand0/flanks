//! The walled town (`FL_MAP=walled_town`): the hill map's ground with a
//! stone wall across the plateau and a town of houses behind it. Every
//! piece is listed once here (`pieces`), and both the drawn models
//! (town.rs) and the sim's heights (`sim_heights`) come from that list, so
//! the two cannot drift.
//!
//! The sim sees a piece as raised ground, like the Town map's blocks: a
//! vertex with a 4-neighbour 2 m or more away in height is blocked
//! (`terrain::SLOPE_BLOCK`), and a soldier stands on the vertex nearest
//! him. So a raised top loses its edge rows, a lane on it is its width
//! less 2 m, and the ground row beside a piece is blocked too: men stop
//! about 3 m off a face.
//!
//! - Wall units (2 m along the wall, 8 m deep): an 8 m strip whose lane is
//!   the middle 6 m, the model's walk.
//! - Drum towers: the strip goes on through the passage at 8 m, the rest
//!   of the disc stands at 12 m, the roof no one reaches.
//! - The gatehouse: the strip across all of it at 8 m, the drums' rest at
//!   12 m. Its ground passage is under the strip, so it is closed.
//! - Stairs: an 8 m rise over 12 m (1.33 m per cell, under the 2 m that
//!   blocks), then the platform at 8 m. The flight touches the wall's
//!   inner face, and the face's vertex row takes the flight's height
//!   there: otherwise that row would stand 8 m over the middle row of a
//!   4 m stair and block it, and nothing could climb.
//! - Houses: their 10 by 14 m rectangle raised 6 m, the eaves.
//!
//! Pieces stand on levelled ground: the wall run on one flat terrace,
//! each house on its own pad, blended out into the hill's slopes. The
//! drawn ground is that levelled ground; the sim's copy adds the pieces.
//!
//! Vertices sit on even metres. The wall's line, its ends and the towers'
//! centres are even, so a strip's faces fall on vertex rows; the houses'
//! centres are odd, so their faces (5 and 7 m out) do too.

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
            Kind::Stair => (Vec2::new(0.0, -4.0), Vec2::new(16.0, 0.0)),
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
}

/// The wall units of one curtain: centres every 2 m from `a` to `b`.
fn curtain(out: &mut Vec<Piece>, a: f32, b: f32) {
    let mut x = a;
    while x <= b + 0.01 {
        // A stair's platform spans 12 to 16 m from its foot: the two units
        // there are landings.
        let landing = STAIRS_X.iter().any(|&s| x > s + 12.0 && x < s + 16.0);
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

/// The sim's heights: the levelled ground with every piece raised on it.
/// Where pieces overlap the higher wins (a drum over the walk's strip),
/// except that a stair's footprint takes the stair's height, the wall's
/// inner face row included (see the module docs).
pub fn sim_heights(ground: &[f32], origin: Vec2, dims: (usize, usize), pieces: &[Piece]) -> Vec<f32> {
    let mut raise = vec![0.0f32; ground.len()];
    let mut stair = vec![None::<f32>; ground.len()];
    for p in pieces {
        let (a, b) = p.bounds();
        let pad = Pad { lo: a, hi: b, blend: 0.0 };
        let (x0, z0, x1, z1) = pad.span(origin, dims, 0.0);
        for z in z0..=z1 {
            for x in x0..=x1 {
                let i = z * dims.0 + x;
                let Some(r) = p.raise(origin + Vec2::new(x as f32, z as f32) * CELL) else { continue };
                if p.kind == Kind::Stair {
                    stair[i] = Some(r);
                } else {
                    raise[i] = raise[i].max(r);
                }
            }
        }
    }
    ground
        .iter()
        .zip(raise.iter().zip(&stair))
        .map(|(g, (r, s))| g + s.unwrap_or(*r))
        .collect()
}

/// Points the scripted scenes and the tests use.
pub mod spots {
    use super::*;

    /// Open plateau north of the town, where the defender stands.
    #[cfg(test)]
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
