//! The obstacle map (`FL_MAP=obstacles`): flat ground with giant raised
//! blocks, the proving ground for pathfinding. Also the town
//! (`FL_MAP=town`, `town_blocks`): Big Grassland's ground with a walled
//! town of streets and houses across the middle, the map that shows what
//! many obstacles cost in a full battle. Every block is a 6 m step
//! between neighbouring vertices, so the terrain's own blocked rule
//! (`SLOPE_BLOCK`) closes every side of it and the soldiers' wall slide
//! applies with no new collision code. Each test case stands in its own
//! part of the field; the scripted scenes (path_test.rs) read the start
//! and goal points from here, so the map and the scenes cannot drift.
//!
//! Vertices sit on even metres (the field is 1024 by 768 m centred on the
//! origin, 2 m cells), so every edge below is an even number. A block
//! whose top spans the vertices `a..=b` blocks the vertices `a - 2` to
//! `b + 2`; a soldier is on blocked ground when his nearest vertex is
//! blocked. A gap between tops ending at `a` and starting at `b` is
//! therefore walkable over `b - a - 6` metres.

use bevy::math::Vec2;

/// Height of every block's top above the ground (m).
pub const BLOCK_HEIGHT: f32 = 6.0;

/// One block's top: `[x0, x1] x [z0, z1]`, inclusive, in metres.
pub struct Block {
    pub x0: f32,
    pub x1: f32,
    pub z0: f32,
    pub z1: f32,
}

const fn block(x0: f32, x1: f32, z0: f32, z1: f32) -> Block {
    Block { x0, x1, z0, z1 }
}

/// A test case's ground: where its regiment starts and where it is sent.
pub struct Case {
    pub name: &'static str,
    pub start: Vec2,
    pub goal: Vec2,
}

// The long wall across the whole field at z -294..-288, from edge to edge,
// so its three gaps are the only ways from the south strip to the north.
const WALL_Z0: f32 = -294.0;
const WALL_Z1: f32 = -288.0;
/// The wide gap: tops end at -324 and start at -276, 42 m walkable,
/// wider than any regiment.
pub const WIDE_GAP_X: f32 = -300.0;
/// The narrow gap: tops end at -6 and start at 6, 6 m walkable, four men
/// abreast.
pub const NARROW_GAP_X: f32 = 0.0;
/// The shared gap: tops end at 290 and start at 310, 14 m walkable.
pub const SHARED_GAP_X: f32 = 300.0;

/// The south strip (below the wall) and the north side of the wall band.
pub const SOUTH_Z: f32 = -350.0;
pub const NORTH_Z: f32 = -230.0;

/// The first row of cases (z -180..100) and the second (z 100..380).
const CUBE_X: f32 = -341.0;
const POCKET_X: f32 = 0.0;
const PILLARS_X: f32 = 340.0;

/// The zig-zag enclosure: its outer walls, the entrance in its south
/// wall near the east end, and the last corridor along its north wall.
const ZIG_X0: f32 = -480.0;
const ZIG_X1: f32 = -200.0;
const ZIG_Z0: f32 = 120.0;
const ZIG_Z1: f32 = 370.0;
const ZIG_DOOR_X: f32 = -240.0;

/// The closed box, centred on the goal.
const BOX_C: Vec2 = Vec2::new(40.0, 240.0);

/// The cases, one regiment each.
pub const CASES: [Case; 8] = [
    Case { name: "cube", start: Vec2::new(CUBE_X, -150.0), goal: Vec2::new(CUBE_X, 70.0) },
    Case {
        name: "wide",
        start: Vec2::new(WIDE_GAP_X + 80.0, SOUTH_Z),
        goal: Vec2::new(WIDE_GAP_X + 80.0, NORTH_Z),
    },
    Case {
        name: "narrow",
        start: Vec2::new(NARROW_GAP_X, SOUTH_Z),
        goal: Vec2::new(NARROW_GAP_X, NORTH_Z),
    },
    Case { name: "pocket", start: Vec2::new(POCKET_X, -150.0), goal: Vec2::new(POCKET_X, 75.0) },
    Case {
        name: "zigzag",
        start: Vec2::new(ZIG_DOOR_X, 92.0),
        goal: Vec2::new(ZIG_DOOR_X, 336.0),
    },
    Case { name: "box", start: Vec2::new(BOX_C.x, 130.0), goal: BOX_C },
    Case { name: "pillars", start: Vec2::new(PILLARS_X, -150.0), goal: Vec2::new(PILLARS_X, 75.0) },
    // The shared gap's two regiments trade places through it; this is
    // the one coming from the north. Its partner starts at the goal.
    Case {
        name: "shared",
        start: Vec2::new(SHARED_GAP_X, NORTH_Z),
        goal: Vec2::new(SHARED_GAP_X, SOUTH_Z),
    },
];

/// Every block's top.
pub fn blocks() -> Vec<Block> {
    let mut b = vec![
        // The long wall and its three gaps.
        block(-512.0, -324.0, WALL_Z0, WALL_Z1),
        block(-276.0, NARROW_GAP_X - 6.0, WALL_Z0, WALL_Z1),
        block(NARROW_GAP_X + 6.0, 290.0, WALL_Z0, WALL_Z1),
        block(310.0, 512.0, WALL_Z0, WALL_Z1),
        // The cube: 80 m, nine metres east of its regiment's line so the
        // west way round is the shorter one.
        block(-372.0, -292.0, -80.0, 0.0),
        // The pocket: a U 100 m wide inside, open toward its regiment,
        // the goal behind its closed end.
        block(-58.0, -50.0, -60.0, 40.0),
        block(50.0, 58.0, -60.0, 40.0),
        block(-58.0, 58.0, 32.0, 40.0),
        // The zig-zag: an enclosure entered through its south wall, with
        // three inner walls that turn the way back and forth.
        block(ZIG_X0, ZIG_DOOR_X - 22.0, ZIG_Z0, ZIG_Z0 + 6.0),
        block(ZIG_DOOR_X + 22.0, ZIG_X1, ZIG_Z0, ZIG_Z0 + 6.0),
        block(ZIG_X0, ZIG_X1, ZIG_Z1 - 6.0, ZIG_Z1),
        block(ZIG_X0, ZIG_X0 + 6.0, ZIG_Z0, ZIG_Z1),
        block(ZIG_X1 - 6.0, ZIG_X1, ZIG_Z0, ZIG_Z1),
        block(ZIG_X0 + 60.0, ZIG_X1, 178.0, 184.0),
        block(ZIG_X0, ZIG_X1 - 60.0, 238.0, 244.0),
        block(ZIG_X0 + 60.0, ZIG_X1, 298.0, 304.0),
        // The closed box: a 60 m room with no way in.
        block(BOX_C.x - 40.0, BOX_C.x + 40.0, BOX_C.y - 40.0, BOX_C.y - 34.0),
        block(BOX_C.x - 40.0, BOX_C.x + 40.0, BOX_C.y + 34.0, BOX_C.y + 40.0),
        block(BOX_C.x - 40.0, BOX_C.x - 34.0, BOX_C.y - 40.0, BOX_C.y + 40.0),
        block(BOX_C.x + 34.0, BOX_C.x + 40.0, BOX_C.y - 40.0, BOX_C.y + 40.0),
    ];
    // The pillars: 4 m posts 16 m apart (6 m walkable between them), every
    // other row shifted half a pitch so no lane runs straight through,
    // across the whole zone so the way through is the only short one.
    for row in 0..6 {
        let z = -60.0 + 16.0 * row as f32;
        let shift = if row % 2 == 0 { 0.0 } else { 8.0 };
        let mut x = 180.0 + shift;
        while x <= 500.0 {
            b.push(block(x - 2.0, x + 2.0, z - 2.0, z + 2.0));
            x += 16.0;
        }
    }
    b
}

/// The height of the obstacle map at `p`: the ground at zero, a block's
/// top at `BLOCK_HEIGHT`.
pub fn height(p: Vec2, blocks: &[Block]) -> f32 {
    let on_top = blocks
        .iter()
        .any(|b| p.x >= b.x0 && p.x <= b.x1 && p.y >= b.z0 && p.y <= b.z1);
    if on_top { BLOCK_HEIGHT } else { 0.0 }
}


/// The town's half size (m): the wall ring stands at x ±600, z ±300.
pub const TOWN_HALF: Vec2 = Vec2::new(600.0, 300.0);

/// The walled town across the middle of Big Grassland: a 6 m wall with
/// three gates on each long side and one on each short side (24 m
/// between the wall's tops, 18 m walkable), avenues 30 m wide through the
/// gates, a plaza 100 m across at the centre, and 14 m houses on a 30 m
/// pitch (16 m streets, 10 m walkable) everywhere else, about one in eight
/// left out for yards. Armies stand north and south of it
/// (`FL_ARMY_GAP=700` keeps the spawn clear of the wall).
pub fn town_blocks() -> Vec<Block> {
    let (hx, hz) = (TOWN_HALF.x, TOWN_HALF.y);
    let gate = 12.0; // half the gap between the wall's tops
    let gates_x = [-400.0, 0.0, 400.0];
    let mut b = Vec::new();
    // The north and south walls, cut at the gates.
    for z0 in [-hz, hz - 6.0] {
        let mut x = -hx;
        for &g in &gates_x {
            b.push(block(x, g - gate, z0, z0 + 6.0));
            x = g + gate;
        }
        b.push(block(x, hx, z0, z0 + 6.0));
    }
    // The west and east walls, one gate each at z 0.
    for x0 in [-hx, hx - 6.0] {
        b.push(block(x0, x0 + 6.0, -hz, -gate));
        b.push(block(x0, x0 + 6.0, gate, hz));
    }
    // Houses: 14 m tops on a 30 m pitch, skipping the avenues (30 m about
    // the gates' lines), the plaza and a wall margin, and one in eight by
    // hash for yards.
    let avenue = 15.0;
    let plaza = 50.0;
    let mut k = 0u32;
    let mut z = -hz + 30.0;
    while z + 14.0 <= hz - 30.0 {
        let mut x = -hx + 30.0;
        while x + 14.0 <= hx - 30.0 {
            let (cx, cz) = (x + 7.0, z + 7.0);
            let on_avenue = gates_x.iter().any(|&g| (cx - g).abs() < avenue + 7.0) || cz.abs() < avenue + 7.0;
            let on_plaza = cx.abs() < plaza + 7.0 && cz.abs() < plaza + 7.0;
            k += 1;
            let yard = crate::units::hash01(k.wrapping_mul(0x9E37_79B1)) < 0.125;
            if !on_avenue && !on_plaza && !yard {
                b.push(block(x, x + 14.0, z, z + 14.0));
            }
            x += 30.0;
        }
        z += 30.0;
    }
    b
}

/// Raise every vertex inside a block's top by `BLOCK_HEIGHT`, so the
/// block stands on whatever ground is there. Walks each block's own
/// vertices, so a town of hundreds of blocks costs nothing per vertex.
pub fn raise_blocks(heights: &mut [f32], origin: Vec2, (verts_x, verts_z): (usize, usize), blocks: &[Block]) {
    for b in blocks {
        let x0 = (((b.x0 - origin.x) / crate::terrain::CELL).ceil().max(0.0)) as usize;
        let z0 = (((b.z0 - origin.y) / crate::terrain::CELL).ceil().max(0.0)) as usize;
        let x1 = (((b.x1 - origin.x) / crate::terrain::CELL).floor().max(0.0) as usize).min(verts_x - 1);
        let z1 = (((b.z1 - origin.y) / crate::terrain::CELL).floor().max(0.0) as usize).min(verts_z - 1);
        for z in z0..=z1 {
            for x in x0..=x1 {
                heights[z * verts_x + x] += BLOCK_HEIGHT;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::terrain::{build_terrain, MapKind, CELL};
    use bevy::math::Vec2;

    /// Every vertex a soldier could stand on, flooded from open ground
    /// with diagonal steps allowed (a man moving diagonally can cross a
    /// corner point): no block's top is reached, and the closed box's
    /// room is not either. The wall slide refuses blocked ground, so a
    /// soldier can only ever stand where this flood goes.
    #[test]
    fn nothing_walks_up_a_block_and_the_box_stays_closed() {
        let t = build_terrain(MapKind::Obstacles);
        let (vx, vz) = t.grid_verts();
        let pos = |x: usize, z: usize| t.min() + Vec2::new(x as f32, z as f32) * CELL;
        let open = |x: usize, z: usize| {
            let p = pos(x, z);
            !t.blocked_at(p.x, p.y)
        };
        let to_idx = |p: Vec2| {
            let g = ((p - t.min()) / CELL).round();
            (g.x as usize, g.y as usize)
        };
        let (sx, sz) = to_idx(Vec2::new(0.0, 100.0));
        assert!(open(sx, sz));
        let mut seen = vec![false; vx * vz];
        let mut stack = vec![(sx, sz)];
        seen[sz * vx + sx] = true;
        while let Some((x, z)) = stack.pop() {
            for dz in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = x as i32 + dx;
                    let nz = z as i32 + dz;
                    if nx < 0 || nz < 0 || nx >= vx as i32 || nz >= vz as i32 {
                        continue;
                    }
                    let (nx, nz) = (nx as usize, nz as usize);
                    if !seen[nz * vx + nx] && open(nx, nz) {
                        seen[nz * vx + nx] = true;
                        stack.push((nx, nz));
                    }
                }
            }
        }
        let mut tops = 0;
        for z in 0..vz {
            for x in 0..vx {
                let p = pos(x, z);
                if t.height_at(p.x, p.y) > 0.5 {
                    tops += 1;
                    assert!(!seen[z * vx + x], "a block's top at {p} is reachable");
                }
            }
        }
        assert!(tops > 1000);
        let (bx, bz) = to_idx(super::CASES[5].goal);
        assert!(open(bx, bz), "the box's room is open ground");
        assert!(!seen[bz * vx + bx], "the closed box can be entered");
        // Every case's start and goal (bar the box's) is reachable open
        // ground.
        for case in &super::CASES {
            for p in [case.start, case.goal] {
                let (x, z) = to_idx(p);
                if case.name != "box" || p == case.start {
                    assert!(seen[z * vx + x], "{} point {p} is not reachable", case.name);
                }
            }
        }
    }

    /// The wall's gaps have the walkable widths their docs give: 42, 6
    /// and 14 m (21, 3 and 7 open vertices across).
    #[test]
    fn wall_gaps_have_their_widths() {
        let t = build_terrain(MapKind::Obstacles);
        let z = -292.0;
        let open_run = |cx: f32| {
            let mut n = 0;
            let mut x = cx;
            while !t.blocked_at(x, z) {
                n += 1;
                x += CELL;
            }
            let mut x = cx - CELL;
            while !t.blocked_at(x, z) {
                n += 1;
                x -= CELL;
            }
            n
        };
        assert_eq!(open_run(super::WIDE_GAP_X), 21);
        assert_eq!(open_run(super::NARROW_GAP_X), 3);
        assert_eq!(open_run(super::SHARED_GAP_X), 7);
    }

    /// The town: every street is reachable from outside through the gates,
    /// no house top is, and the plaza at the centre is open ground.
    #[test]
    fn the_town_is_open_through_its_gates_and_its_houses_are_closed() {
        let t = build_terrain(MapKind::Town);
        let (vx, vz) = t.grid_verts();
        let pos = |x: usize, z: usize| t.min() + Vec2::new(x as f32, z as f32) * CELL;
        let open = |x: usize, z: usize| {
            let p = pos(x, z);
            !t.blocked_at(p.x, p.y)
        };
        let to_idx = |p: Vec2| {
            let g = ((p - t.min()) / CELL).round();
            (g.x as usize, g.y as usize)
        };
        let (sx, sz) = to_idx(Vec2::new(0.0, -500.0));
        assert!(open(sx, sz));
        let mut seen = vec![false; vx * vz];
        let mut stack = vec![(sx, sz)];
        seen[sz * vx + sx] = true;
        while let Some((x, z)) = stack.pop() {
            for (dx, dz) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                let (nx, nz) = (x as i32 + dx, z as i32 + dz);
                if nx < 0 || nz < 0 || nx >= vx as i32 || nz >= vz as i32 {
                    continue;
                }
                let (nx, nz) = (nx as usize, nz as usize);
                if !seen[nz * vx + nx] && open(nx, nz) {
                    seen[nz * vx + nx] = true;
                    stack.push((nx, nz));
                }
            }
        }
        let blocks = super::town_blocks();
        assert!(blocks.len() > 400, "{} blocks", blocks.len());
        for b in &blocks {
            let (x, z) = to_idx(Vec2::new((b.x0 + b.x1) * 0.5, (b.z0 + b.z1) * 0.5));
            assert!(!seen[z * vx + x], "a top at ({}, {}) is reachable", b.x0, b.z0);
        }
        let (px, pz) = to_idx(Vec2::ZERO);
        assert!(seen[pz * vx + px], "the plaza is cut off");
        // A street between two houses deep inside.
        let (qx, qz) = to_idx(Vec2::new(-485.0, -215.0));
        assert!(seen[qz * vx + qx], "a street is cut off");
    }
}
