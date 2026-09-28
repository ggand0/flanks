//! Authored grassland vegetation and the river's procedural plant meshes.
//! Vegetation is visual only and does not affect terrain or pathfinding.

use bevy::asset::RenderAssetUsages;
use bevy::camera::primitives::MeshAabb;
use bevy::image::{
    CompressedImageFormats, ImageAddressMode, ImageSampler, ImageSamplerDescriptor, ImageType,
};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{Face, TextureFormat};

use crate::terrain::{
    CELL, CHUNK_CELLS, CHUNKS_X, CHUNKS_Z, MapKind, Terrain, fbm, river_center_x, river_half_width,
};
use crate::units::hash01;

pub struct VegetationPlugin;

impl Plugin for VegetationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, load_trees)
            .add_systems(Update, respawn_vegetation.after(crate::terrain::MapRebuild))
            .add_systems(
                PostUpdate,
                select_tree_level
                    .before(bevy::transform::TransformSystems::Propagate)
                    .before(bevy::camera::visibility::VisibilitySystems::CalculateBounds),
            );
    }
}

/// Flat-shaded triangle builder shared with the river bridge.
pub(crate) struct Soup {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
}

impl Soup {
    pub(crate) fn new() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            colors: Vec::new(),
        }
    }

    pub(crate) fn tri(&mut self, a: Vec3, b: Vec3, c: Vec3, color: [f32; 4]) {
        let n = (b - a).cross(c - a).normalize_or_zero();
        for v in [a, b, c] {
            self.positions.push(v.to_array());
            self.normals.push(n.to_array());
            self.colors.push(color);
        }
    }

    pub(crate) fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, color: [f32; 4]) {
        self.tri(a, b, c, color);
        self.tri(a, c, d, color);
    }

    /// Axis-aligned cuboid (pre-rotation); `c` = center, `h` = half extents.
    pub(crate) fn cuboid(&mut self, c: Vec3, h: Vec3, color: [f32; 4]) {
        let p = |x: f32, y: f32, z: f32| c + Vec3::new(x * h.x, y * h.y, z * h.z);
        // 8 corners.
        let v = [
            p(-1.0, -1.0, -1.0),
            p(1.0, -1.0, -1.0),
            p(1.0, -1.0, 1.0),
            p(-1.0, -1.0, 1.0),
            p(-1.0, 1.0, -1.0),
            p(1.0, 1.0, -1.0),
            p(1.0, 1.0, 1.0),
            p(-1.0, 1.0, 1.0),
        ];
        // Outward-wound faces (bottom skipped: buried).
        self.quad(v[3], v[2], v[6], v[7], color); // +Z
        self.quad(v[1], v[0], v[4], v[5], color); // -Z
        self.quad(v[2], v[1], v[5], v[6], color); // +X
        self.quad(v[0], v[3], v[7], v[4], color); // -X
        self.quad(v[7], v[6], v[5], v[4], color); // +Y
    }

    /// 4-sided pyramid: square base half-width `hw` at y0, apex at y1.
    /// Bottom face skipped (hidden).
    pub(crate) fn pyramid(&mut self, c: Vec3, hw: f32, y0: f32, y1: f32, color: [f32; 4]) {
        let b = [
            Vec3::new(c.x - hw, y0, c.z - hw),
            Vec3::new(c.x + hw, y0, c.z - hw),
            Vec3::new(c.x + hw, y0, c.z + hw),
            Vec3::new(c.x - hw, y0, c.z + hw),
        ];
        let apex = Vec3::new(c.x, y1, c.z);
        for i in 0..4 {
            let (a, b2) = (b[i], b[(i + 1) % 4]);
            // Winding varies per face; emit both orders and let the
            // cross product give the geometric normal either way by
            // picking the outward one.
            let n = (b2 - a).cross(apex - a);
            let mid = (a + b2) * 0.5;
            let outward = Vec3::new(mid.x - c.x, 0.0, mid.z - c.z);
            if n.dot(outward) > 0.0 {
                self.tri(a, b2, apex, color);
            } else {
                self.tri(b2, a, apex, color);
            }
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub(crate) fn into_mesh(self) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.colors)
    }
}

pub(crate) fn srgb(r: f32, g: f32, b: f32) -> [f32; 4] {
    Color::srgb(r, g, b).to_linear().to_f32_array()
}

/// Rotate+scale+translate a local-space point into world space.
fn xform(p: Vec3, yaw: f32, scale: f32, at: Vec3) -> Vec3 {
    let (s, c) = yaw.sin_cos();
    let q = Vec3::new(p.x * c + p.z * s, p.y, -p.x * s + p.z * c) * scale;
    at + q
}

enum Kind {
    Pine,
    Broadleaf,
    Bush,
}

/// Append one plant (local archetype transformed by yaw/scale/at) to the soup.
fn add_plant(soup: &mut Soup, kind: Kind, yaw: f32, scale: f32, at: Vec3, seed: u32) {
    let jitter = |i: u32, base: [f32; 4]| {
        let k = 0.9 + 0.2 * hash01(seed.wrapping_mul(7).wrapping_add(i));
        [base[0] * k, base[1] * k, base[2] * k, base[3]]
    };
    let trunk = srgb(0.33, 0.23, 0.13);
    // Cuboid/pyramid helpers applied through the plant transform.
    fn cub(soup: &mut Soup, c: Vec3, h: Vec3, col: [f32; 4], yaw: f32, scale: f32, at: Vec3) {
        let mut tmp = Soup::new();
        tmp.cuboid(Vec3::ZERO, h, col);
        push_transformed(soup, &tmp, c, yaw, scale, at);
    }
    #[allow(clippy::too_many_arguments)] // primitive builder, all scalars
    fn pyr(soup: &mut Soup, hw: f32, y0: f32, y1: f32, col: [f32; 4], yaw: f32, scale: f32, at: Vec3) {
        let mut tmp = Soup::new();
        tmp.pyramid(Vec3::ZERO, hw, y0, y1, col);
        push_transformed(soup, &tmp, Vec3::ZERO, yaw, scale, at);
    }
    match kind {
        Kind::Pine => {
            let dark = jitter(1, srgb(0.17, 0.34, 0.17));
            cub(soup, Vec3::new(0.0, 0.8, 0.0), Vec3::new(0.25, 0.8, 0.25), trunk, yaw, scale, at);
            for i in 0..3 {
                let fi = i as f32;
                pyr(
                    soup,
                    2.1 - fi * 0.55,
                    1.2 + fi * 1.5,
                    3.6 + fi * 1.5,
                    jitter(2 + i, dark),
                    yaw,
                    scale,
                    at,
                );
            }
        }
        Kind::Broadleaf => {
            let leaf = jitter(1, srgb(0.28, 0.46, 0.18));
            cub(soup, Vec3::new(0.0, 1.1, 0.0), Vec3::new(0.3, 1.1, 0.3), trunk, yaw, scale, at);
            cub(soup, Vec3::new(0.0, 3.4, 0.0), Vec3::new(1.9, 1.4, 1.9), jitter(2, leaf), yaw, scale, at);
            cub(soup, Vec3::new(1.2, 2.8, 0.5), Vec3::new(1.2, 0.9, 1.2), jitter(3, leaf), yaw, scale, at);
            cub(soup, Vec3::new(-1.0, 3.0, -0.6), Vec3::new(1.1, 0.8, 1.1), jitter(4, leaf), yaw, scale, at);
        }
        Kind::Bush => {
            let olive = jitter(1, srgb(0.30, 0.38, 0.16));
            cub(soup, Vec3::new(0.0, 0.5, 0.0), Vec3::new(0.9, 0.55, 0.9), olive, yaw, scale, at);
            cub(soup, Vec3::new(0.5, 0.35, 0.4), Vec3::new(0.6, 0.4, 0.6), jitter(2, olive), yaw, scale, at);
        }
    }
}

/// Re-emit `src` triangles transformed: local offset `c`, then yaw,
/// scale, translate to `at`. Normals recomputed from world positions.
fn push_transformed(dst: &mut Soup, src: &Soup, c: Vec3, yaw: f32, scale: f32, at: Vec3) {
    for (ti, t) in src.positions.chunks_exact(3).enumerate() {
        let p = |i: usize| {
            let lp = Vec3::from_array(t[i]) + c;
            xform(lp, yaw, scale, at)
        };
        dst.tri(p(0), p(1), p(2), src.colors[ti * 3]);
    }
}

/// The plants of the current map: one entity per authored plant, or one
/// merged mesh per terrain chunk on the river map.
#[derive(Component)]
struct Plants;

/// Plants follow the terrain's map: the first frame grows them, and a
/// map change clears the old ones and grows the new map's.
fn respawn_vegetation(
    mut seen: Local<Option<MapKind>>,
    old: Query<Entity, With<Plants>>,
    mut commands: Commands,
    terrain: Res<Terrain>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    trees: Res<TreeAssets>,
) {
    if *seen == Some(terrain.kind) {
        return;
    }
    *seen = Some(terrain.kind);
    for e in &old {
        commands.entity(e).despawn();
    }
    match terrain.kind {
        MapKind::Grassland => {
            let specs: Vec<_> = trees.0.iter().map(|tree| (tree.name, tree.reach)).collect();
            let planting = grassland_planting(&specs, &terrain, crate::regiments::army_gap());
            for plant in &planting.plants {
                let (x, z) = (plant.pos.x, plant.pos.y);
                let placement = (x, z, plant.yaw, plant.scale);
                spawn_authored_plant(&mut commands, &terrain, &trees.0[plant.asset], plant.asset, placement);
            }
            planting.log(&specs);
        }
        MapKind::Sandbox => {
            for &(name, x, z, yaw, scale) in SANDBOX_COMPOSITION {
                let Some((asset, tree)) = trees
                    .0
                    .iter()
                    .enumerate()
                    .find(|(_, tree)| tree.name == name)
                else {
                    warn!("vegetation: sandbox composition is missing {name}");
                    continue;
                };
                spawn_authored_plant(&mut commands, &terrain, tree, asset, (x, z, yaw, scale));
            }
            info!(
                "vegetation: sandbox composition, {} plants",
                SANDBOX_COMPOSITION.len()
            );
        }
        MapKind::River => plant(&mut commands, &terrain, &mut meshes, &mut materials),
        MapKind::Classic => {}
    }
}

// Asset, world X/Z in metres, yaw in radians, uniform scale.
// Two loose groups leave a broad central opening and an uneven woodland edge.
const SANDBOX_COMPOSITION: &[(&str, f32, f32, f32, f32)] = &[
    ("mature_oak_pale", -16.0, -3.0, 0.45, 1.00),
    ("oak_pale", -26.0, 1.0, 2.10, 0.90),
    ("leaning_oak_lighter", -19.0, 10.0, -1.10, 0.88),
    ("oak_lighter", -5.0, -18.0, -0.70, 0.95),
    ("mature_oak_lighter", 14.0, -12.0, 2.60, 0.90),
    ("leaning_oak_pale", 23.0, 12.0, 0.80, 1.08),
    ("silver_birch_warm", 9.0, -4.0, 1.30, 0.96),
    ("silver_birch_warm", 27.0, 1.0, -0.40, 0.83),
    ("shrub_b_sandbox", -19.0, -7.0, 0.20, 1.18),
    ("shrub_b_sandbox", -21.0, -4.0, 2.40, 0.82),
    ("shrub_b_sandbox", -15.0, -7.0, 4.80, 1.02),
    ("shrub_b_sandbox", -30.0, 4.0, 1.70, 0.93),
    ("shrub_b_sandbox", -27.0, 7.0, 3.80, 1.15),
    ("shrub_b_sandbox", -21.0, 15.0, 5.50, 0.90),
    ("shrub_b_sandbox", -17.0, 14.0, 0.90, 1.24),
    ("shrub_b_sandbox", -14.0, 16.0, 3.10, 0.76),
    ("shrub_b_sandbox", -3.0, -12.0, 4.10, 0.88),
    ("shrub_b_sandbox", -1.0, -15.0, 1.10, 1.06),
    ("shrub_b_sandbox", 9.0, -14.0, 2.70, 1.20),
    ("shrub_b_sandbox", 12.0, -7.0, 5.80, 0.86),
    ("shrub_b_sandbox", 18.0, -5.0, 0.50, 1.03),
    ("shrub_b_sandbox", 21.0, 18.0, 1.90, 1.16),
    ("shrub_b_sandbox", 25.0, 18.0, 4.50, 0.74),
    ("shrub_b_sandbox", 28.0, 12.0, 3.30, 0.96),
    ("shrub_b_sandbox", 30.0, 1.0, 5.10, 1.08),
    ("shrub_b_sandbox", 29.0, -3.0, 2.20, 0.80),
    ("shrub_b_sandbox", 9.0, 14.0, 0.70, 1.12),
    ("shrub_b_sandbox", 11.0, 16.0, 3.60, 0.72),
    ("shrub_a", 14.0, 14.0, 0.30, 1.00),
];

fn spawn_authored_plant(
    commands: &mut Commands,
    terrain: &Terrain,
    tree: &TreeAsset,
    asset: usize,
    (x, z, yaw, scale): (f32, f32, f32, f32),
) {
    let near = &tree.levels[0];
    let sink = (near.height * scale * 0.01).clamp(0.01, 0.08);
    commands
        .spawn((
            Transform::from_xyz(x, terrain.height_at(x, z) - sink, z)
                .with_rotation(Quat::from_rotation_y(yaw))
                .with_scale(Vec3::splat(scale)),
            Visibility::Inherited,
            Plants,
            TreeLevel { asset, level: 0 },
        ))
        .with_children(|parent| {
            for (slot, part) in near.parts.iter().enumerate() {
                parent.spawn((
                    Mesh3d(part.mesh.clone()),
                    MeshMaterial3d(part.material.clone()),
                    TreePartSlot(slot),
                ));
            }
        });
}

// The grassland planting. The two deployment strips cover most of the
// field, so the plants stand in what is left: the 30 m side margins, the
// outer ends of the army gap and the 8 m rear margins. Every plant keeps
// its whole crown out of both strips, out of the open centre of the gap
// and inside the field, checked at the largest scale its kind is drawn
// at. Plants are scenery: nothing in the sim reads them.

/// Half-width of the open centre of the army gap, kept free of plants.
const OPEN_CENTRE: f32 = 320.0;
/// Scale ranges; clearance is checked at the upper end.
const TREE_SCALE: (f32, f32) = (0.85, 1.10);
const SHRUB_SCALE: (f32, f32) = (0.75, 1.20);
/// Steepest ground a plant stands on, rise per metre.
const MAX_SLOPE: f32 = 0.35;
/// Candidate positions tried for each plant before it is given up.
const TRIES: u32 = 10;

/// The grassland's assets by role. Each oak shape comes in two foliage
/// colours, picked at random per tree.
const MATURE_OAKS: &[&str] = &["mature_oak_pale", "mature_oak_lighter"];
const UPRIGHT_OAKS: &[&str] = &["oak_pale", "oak_lighter"];
const LEANING_OAKS: &[&str] = &["leaning_oak_pale", "leaning_oak_lighter"];
const BIRCHES: &[&str] = &["silver_birch_warm"];
const SHRUB_A: &str = "shrub_a";
const SHRUB_B: &str = "shrub_b_sandbox";

/// Where a stand draws its candidates, in world x and z.
enum Area {
    Box(Rect),
    Ellipse { centre: Vec2, radii: Vec2 },
    /// A hedge: shrubs step along the line from `a` to `b`.
    Hedge { a: Vec2, b: Vec2 },
}

impl Area {
    fn sample(&self, u: f32, v: f32) -> Vec2 {
        match *self {
            Area::Box(r) => r.min + (r.max - r.min) * Vec2::new(u, v),
            Area::Ellipse { centre, radii } => centre + radii * disc(u, v),
            Area::Hedge { a, b } => a.lerp(b, u),
        }
    }
}

const fn strip(x0: f32, x1: f32, z0: f32, z1: f32) -> Area {
    Area::Box(Rect { min: Vec2::new(x0, z0), max: Vec2::new(x1, z1) })
}

/// A point in the unit disc, uniform over its area.
fn disc(u: f32, v: f32) -> Vec2 {
    Vec2::from_angle(v * std::f32::consts::TAU) * u.sqrt()
}

/// One planting group. Trees come in clusters, most opening on a mature
/// oak with a few more around it, some alone. Shrubs come in small groups,
/// most of them at the edge of the stand's trees.
struct Stand {
    /// Name in the placement log.
    name: &'static str,
    area: Area,
    trees: u32,
    shrubs: u32,
    /// Share of the trees that are birches.
    birch: f32,
    /// Share of the other trees, cluster anchors aside, that are leaning oaks.
    leaning: f32,
    /// Share of the shrubs that are upright A; the rest are low B.
    shrub_a: f32,
    /// When set, the stand's crowns stay inside this rectangle, which may
    /// lie in a deployment zone, instead of outside the zones.
    inside: Option<Rect>,
}

// West is -x. The west margin carries the main broken oak edge, the east
// margin fewer and shorter groups with more leaning oaks, over drier
// ground. Openings between the stretches keep it from reading as a wall.
// The dry ground crosses the army gap at x = 310 to 335 in the layout
// image, so the eastern hedge stands there, just outside the open centre,
// with an opening at z = -5 to 7.
/// The rear corners of the two deployment zones that no rank reaches.
const PLAYER_REAR_CORNER: Rect = Rect { min: Vec2::new(-482.0, -376.0), max: Vec2::new(-335.0, -327.0) };
const ENEMY_REAR_CORNER: Rect = Rect { min: Vec2::new(335.0, 327.0), max: Vec2::new(482.0, 376.0) };

const GRASSLAND_STANDS: &[Stand] = &[
    Stand { name: "west 1", area: strip(-512.0, -482.0, -372.0, -300.0), trees: 12, shrubs: 10, birch: 0.22, leaning: 0.20, shrub_a: 0.24, inside: None },
    Stand { name: "west 2", area: strip(-512.0, -482.0, -262.0, -205.0), trees: 8, shrubs: 8, birch: 0.25, leaning: 0.20, shrub_a: 0.24, inside: None },
    Stand { name: "west 3", area: strip(-512.0, -482.0, -170.0, -70.0), trees: 16, shrubs: 12, birch: 0.20, leaning: 0.20, shrub_a: 0.24, inside: None },
    Stand { name: "west 4", area: strip(-512.0, -482.0, 45.0, 150.0), trees: 16, shrubs: 12, birch: 0.20, leaning: 0.20, shrub_a: 0.24, inside: None },
    Stand { name: "west 5", area: strip(-512.0, -482.0, 190.0, 250.0), trees: 8, shrubs: 8, birch: 0.25, leaning: 0.20, shrub_a: 0.24, inside: None },
    Stand { name: "west 6", area: strip(-512.0, -482.0, 290.0, 372.0), trees: 12, shrubs: 10, birch: 0.22, leaning: 0.20, shrub_a: 0.24, inside: None },
    // The gap copses belong at the outer ends of the army gap, where the
    // flank fights reach. Soldiers pass through trunks until trees collide
    // with them, so for now the copses stand in the rear corners of the
    // deployment zones, beyond the ranks: with 100 regiments of 1,000 the
    // full ranks end by |z| = 327 at any army gap from 20 to 120 m, and the
    // last, partial rank stays within |x| = 330. With the collisions,
    // restore the three commented stands and drop the rear-corner ones.
    // Stand { name: "west gap copse", area: Area::Ellipse { centre: Vec2::new(-435.0, 0.0), radii: Vec2::new(80.0, 30.0) }, trees: 18, shrubs: 16, birch: 0.20, leaning: 0.25, shrub_a: 0.24, inside: None },
    // Stand { name: "west gap outlier", area: Area::Ellipse { centre: Vec2::new(-362.0, -4.0), radii: Vec2::new(20.0, 22.0) }, trees: 4, shrubs: 6, birch: 0.25, leaning: 0.35, shrub_a: 0.24, inside: None },
    // Stand { name: "east gap copse", area: Area::Ellipse { centre: Vec2::new(430.0, 5.0), radii: Vec2::new(65.0, 30.0) }, trees: 12, shrubs: 12, birch: 0.18, leaning: 0.40, shrub_a: 0.24, inside: None },
    Stand { name: "player rear copse", area: Area::Ellipse { centre: Vec2::new(-410.0, -352.0), radii: Vec2::new(70.0, 22.0) }, trees: 18, shrubs: 16, birch: 0.20, leaning: 0.25, shrub_a: 0.24, inside: Some(PLAYER_REAR_CORNER) },
    Stand { name: "player rear outlier", area: Area::Ellipse { centre: Vec2::new(-350.0, -350.0), radii: Vec2::new(14.0, 18.0) }, trees: 4, shrubs: 6, birch: 0.25, leaning: 0.35, shrub_a: 0.24, inside: Some(PLAYER_REAR_CORNER) },
    Stand { name: "east 1", area: strip(482.0, 512.0, -330.0, -290.0), trees: 5, shrubs: 6, birch: 0.15, leaning: 0.45, shrub_a: 0.24, inside: None },
    Stand { name: "east 2", area: strip(482.0, 512.0, -190.0, -160.0), trees: 3, shrubs: 4, birch: 0.15, leaning: 0.50, shrub_a: 0.24, inside: None },
    Stand { name: "east 3", area: strip(482.0, 512.0, -95.0, -60.0), trees: 4, shrubs: 5, birch: 0.15, leaning: 0.45, shrub_a: 0.24, inside: None },
    Stand { name: "east 4", area: strip(482.0, 512.0, 80.0, 120.0), trees: 5, shrubs: 5, birch: 0.15, leaning: 0.45, shrub_a: 0.24, inside: None },
    Stand { name: "east 5", area: strip(482.0, 512.0, 215.0, 240.0), trees: 3, shrubs: 3, birch: 0.15, leaning: 0.50, shrub_a: 0.24, inside: None },
    Stand { name: "east 6", area: strip(482.0, 512.0, 300.0, 350.0), trees: 5, shrubs: 5, birch: 0.15, leaning: 0.45, shrub_a: 0.24, inside: None },
    // In the rear corner for now, as the note above the player's rear copse says.
    Stand { name: "enemy rear copse", area: Area::Ellipse { centre: Vec2::new(410.0, 352.0), radii: Vec2::new(68.0, 22.0) }, trees: 12, shrubs: 12, birch: 0.18, leaning: 0.40, shrub_a: 0.24, inside: Some(ENEMY_REAR_CORNER) },
    Stand { name: "east hedge south", area: Area::Hedge { a: Vec2::new(329.0, -26.0), b: Vec2::new(334.0, -5.0) }, trees: 0, shrubs: 9, birch: 0.0, leaning: 0.0, shrub_a: 0.35, inside: None },
    Stand { name: "east hedge north", area: Area::Hedge { a: Vec2::new(334.0, 7.0), b: Vec2::new(328.0, 27.0) }, trees: 0, shrubs: 8, birch: 0.0, leaning: 0.0, shrub_a: 0.35, inside: None },
    Stand { name: "player rear 1", area: strip(-310.0, -265.0, -384.0, -376.0), trees: 0, shrubs: 5, birch: 0.0, leaning: 0.0, shrub_a: 0.0, inside: None },
    Stand { name: "player rear 2", area: strip(-60.0, -20.0, -384.0, -376.0), trees: 0, shrubs: 4, birch: 0.0, leaning: 0.0, shrub_a: 0.0, inside: None },
    Stand { name: "player rear 3", area: strip(210.0, 255.0, -384.0, -376.0), trees: 0, shrubs: 5, birch: 0.0, leaning: 0.0, shrub_a: 0.0, inside: None },
    Stand { name: "enemy rear 1", area: strip(-200.0, -160.0, 376.0, 384.0), trees: 0, shrubs: 4, birch: 0.0, leaning: 0.0, shrub_a: 0.0, inside: None },
    Stand { name: "enemy rear 2", area: strip(70.0, 115.0, 376.0, 384.0), trees: 0, shrubs: 5, birch: 0.0, leaning: 0.0, shrub_a: 0.0, inside: None },
    Stand { name: "enemy rear 3", area: strip(340.0, 380.0, 376.0, 384.0), trees: 0, shrubs: 4, birch: 0.0, leaning: 0.0, shrub_a: 0.0, inside: None },
];

/// The ground the grassland plants must keep clear of.
struct Clearance {
    /// The two deployment strips and the open centre between them.
    keep_out: [Rect; 3],
    field: Rect,
}

impl Clearance {
    fn new(terrain: &Terrain, army_gap: f32) -> Self {
        let (lo, hi) = crate::orders::deploy_zone_for(terrain, army_gap);
        Self {
            keep_out: [
                Rect::from_corners(lo, hi),
                Rect::from_corners(Vec2::new(lo.x, -hi.y), Vec2::new(hi.x, -lo.y)),
                Rect::from_corners(Vec2::new(-OPEN_CENTRE, hi.y), Vec2::new(OPEN_CENTRE, -hi.y)),
            ],
            field: Rect::from_corners(terrain.min(), terrain.max()),
        }
    }

    /// Whether a crown of radius `r` around `p` stays inside the field
    /// and out of every keep-out rectangle, or, for a stand confined to
    /// `inside`, within that rectangle.
    fn clear(&self, p: Vec2, r: f32, inside: Option<Rect>) -> bool {
        match inside {
            Some(rect) => rect.inflate(-r).contains(p) && self.field.inflate(-r).contains(p),
            None => {
                self.field.inflate(-r).contains(p)
                    && self.keep_out.iter().all(|k| distance_to_rect(p, *k) >= r)
            }
        }
    }
}

fn distance_to_rect(p: Vec2, r: Rect) -> f32 {
    (r.min - p).max(p - r.max).max(Vec2::ZERO).length()
}

/// One grassland plant: an index into the asset list, its root on the
/// ground plane, yaw, scale, and its crown radius at that scale.
struct Placed {
    asset: usize,
    pos: Vec2,
    yaw: f32,
    scale: f32,
    radius: f32,
    tree: bool,
    stand: usize,
}

#[derive(Default)]
struct Planting {
    plants: Vec<Placed>,
    /// Candidates turned down, by reason.
    rejected_clearance: u32,
    rejected_slope: u32,
    rejected_spacing: u32,
    /// Role names with no loaded asset.
    missing: Vec<&'static str>,
}

/// A stream of vegetation-only draws: the stateless `hash01` on seeds of
/// its own, so no sim random source is touched.
struct Draws(u32);

impl Draws {
    fn new(stand: usize) -> Self {
        Self(0x7e6e_0000_u32.wrapping_add((stand as u32).wrapping_mul(0x0001_3579)))
    }

    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(1);
        hash01(self.0.wrapping_mul(0x9e37_79b9) ^ 0x5eed_7ee5)
    }

    fn pick<'a>(&mut self, names: &[&'a str]) -> &'a str {
        names[((self.next() * names.len() as f32) as usize).min(names.len() - 1)]
    }
}

impl Planting {
    /// Place one plant if its crown is clear, its ground is gentle and it
    /// keeps its distance from the plants already placed.
    #[allow(clippy::too_many_arguments)] // one candidate, all of it
    fn try_place(
        &mut self,
        specs: &[(&'static str, f32)],
        clearance: &Clearance,
        terrain: &Terrain,
        name: &'static str,
        pos: Vec2,
        yaw: f32,
        scale: f32,
        stand: usize,
    ) -> bool {
        let Some(asset) = specs.iter().position(|(n, _)| *n == name) else {
            if !self.missing.contains(&name) {
                self.missing.push(name);
            }
            return false;
        };
        let reach = specs[asset].1;
        let tree = !name.starts_with("shrub");
        let largest = if tree { TREE_SCALE.1 } else { SHRUB_SCALE.1 };
        if !clearance.clear(pos, reach * largest, GRASSLAND_STANDS[stand].inside) {
            self.rejected_clearance += 1;
            return false;
        }
        if terrain.slope_at(pos.x, pos.y) > MAX_SLOPE {
            self.rejected_slope += 1;
            return false;
        }
        let radius = reach * scale;
        // Crowns of neighbouring trees may overlap a little, shrubs sit
        // under a tree's crown edge, and shrubs keep off each other.
        let crowded = self.plants.iter().any(|p| {
            let share = match (tree, p.tree) {
                (true, true) => 0.55,
                (false, false) => 0.75,
                _ => 0.5,
            };
            pos.distance(p.pos) < share * (radius + p.radius)
        });
        if crowded {
            self.rejected_spacing += 1;
            return false;
        }
        self.plants.push(Placed { asset, pos, yaw, scale, radius, tree, stand });
        true
    }

    fn log(&self, specs: &[(&'static str, f32)]) {
        for p in &self.plants {
            debug!(
                "vegetation: grassland plant {} at {:.1} {:.1} yaw {:.2} scale {:.2} crown {:.2} m, {}",
                specs[p.asset].0,
                p.pos.x,
                p.pos.y,
                p.yaw,
                p.scale,
                p.radius,
                GRASSLAND_STANDS[p.stand].name
            );
        }
        for (s, stand) in GRASSLAND_STANDS.iter().enumerate() {
            let here = self.plants.iter().filter(|p| p.stand == s);
            let trees = here.clone().filter(|p| p.tree).count();
            let shrubs = here.count() - trees;
            info!(
                "vegetation: grassland {}: {trees} of {} trees, {shrubs} of {} shrubs",
                stand.name, stand.trees, stand.shrubs
            );
        }
        for (asset, (name, reach)) in specs.iter().enumerate() {
            let n = self.plants.iter().filter(|p| p.asset == asset).count();
            if n > 0 {
                info!("vegetation: grassland {name}: {n}, crown reach {reach:.2} m at scale 1");
            }
        }
        for name in &self.missing {
            warn!("vegetation: grassland has no loaded {name}");
        }
        let trees = self.plants.iter().filter(|p| p.tree).count();
        info!(
            "vegetation: grassland planted {trees} trees and {} shrubs; candidates turned down: {} for clearance, {} for slope, {} for spacing",
            self.plants.len() - trees,
            self.rejected_clearance,
            self.rejected_slope,
            self.rejected_spacing
        );
    }
}

/// The grassland's plants for this field and army gap, the same on every
/// run. `specs` pairs each loaded asset's name with its crown reach.
fn grassland_planting(specs: &[(&'static str, f32)], terrain: &Terrain, army_gap: f32) -> Planting {
    let clearance = Clearance::new(terrain, army_gap);
    let mut out = Planting::default();
    for (s, stand) in GRASSLAND_STANDS.iter().enumerate() {
        let mut d = Draws::new(s);
        let tree_scale = |d: &mut Draws| TREE_SCALE.0 + (TREE_SCALE.1 - TREE_SCALE.0) * d.next();
        let shrub_scale = |d: &mut Draws| SHRUB_SCALE.0 + (SHRUB_SCALE.1 - SHRUB_SCALE.0) * d.next();

        let mut trees = 0;
        let mut clusters = 0;
        while trees < stand.trees && clusters < stand.trees * 3 {
            clusters += 1;
            let centre = stand.area.sample(d.next(), d.next());
            // One cluster in five is a lone tree; the rest gather two to five.
            let size = if d.next() < 0.2 { 1 } else { 2 + (d.next() * 4.0) as u32 };
            let spread = 7.0 + 9.0 * d.next();
            for k in 0..size.min(stand.trees - trees) {
                let names = if k == 0 && size > 1 {
                    MATURE_OAKS
                } else if d.next() < stand.birch {
                    BIRCHES
                } else if d.next() < stand.leaning {
                    LEANING_OAKS
                } else if d.next() < 0.15 {
                    MATURE_OAKS
                } else {
                    UPRIGHT_OAKS
                };
                let name = d.pick(names);
                let scale = tree_scale(&mut d);
                let yaw = d.next() * std::f32::consts::TAU;
                // The anchor stands near the centre, the rest spread around it.
                let scatter = if k == 0 { 3.0 } else { spread };
                for _ in 0..TRIES {
                    let pos = centre + disc(d.next(), d.next()) * scatter;
                    if out.try_place(specs, &clearance, terrain, name, pos, yaw, scale, s) {
                        trees += 1;
                        break;
                    }
                }
            }
        }

        let mut shrubs = 0;
        if let Area::Hedge { a, b } = stand.area {
            // A broken run: a shrub every 2.4 to 3.4 m, a little off the line.
            let length = a.distance(b);
            let side = (b - a).normalize().perp();
            let mut t = 0.0;
            while t <= length && shrubs < stand.shrubs {
                let name = if d.next() < stand.shrub_a { SHRUB_A } else { SHRUB_B };
                let pos = a.lerp(b, t / length) + side * (d.next() - 0.5) * 1.6;
                let (yaw, scale) = (d.next() * std::f32::consts::TAU, shrub_scale(&mut d));
                if out.try_place(specs, &clearance, terrain, name, pos, yaw, scale, s) {
                    shrubs += 1;
                }
                t += 2.4 + d.next();
            }
            continue;
        }
        let stand_trees: Vec<(Vec2, f32)> = out
            .plants
            .iter()
            .filter(|p| p.stand == s && p.tree)
            .map(|p| (p.pos, p.radius))
            .collect();
        let mut groups = 0;
        while shrubs < stand.shrubs && groups < stand.shrubs * 3 {
            groups += 1;
            // Most groups gather at the crown edge of one of the stand's trees.
            let hub = if !stand_trees.is_empty() && d.next() < 0.7 {
                let i = ((d.next() * stand_trees.len() as f32) as usize).min(stand_trees.len() - 1);
                let (root, radius) = stand_trees[i];
                root + Vec2::from_angle(d.next() * std::f32::consts::TAU) * radius * (0.9 + 0.5 * d.next())
            } else {
                stand.area.sample(d.next(), d.next())
            };
            let size = 2 + (d.next() * 4.0) as u32;
            for _ in 0..size.min(stand.shrubs - shrubs) {
                let name = if d.next() < stand.shrub_a { SHRUB_A } else { SHRUB_B };
                let (yaw, scale) = (d.next() * std::f32::consts::TAU, shrub_scale(&mut d));
                for _ in 0..TRIES {
                    let pos = hub + disc(d.next(), d.next()) * 3.5;
                    if out.try_place(specs, &clearance, terrain, name, pos, yaw, scale, s) {
                        shrubs += 1;
                        break;
                    }
                }
            }
        }
    }
    out
}

fn plant(
    commands: &mut Commands,
    terrain: &Terrain,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    if terrain.classic {
        return;
    }
    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 1.0,
        reflectance: 0.05,
        ..default()
    });

    let min = terrain.min();
    let max = terrain.max();
    let chunk_world = CELL * CHUNK_CELLS as f32;
    let mut chunks: Vec<Soup> = (0..CHUNKS_X * CHUNKS_Z).map(|_| Soup::new()).collect();

    // Jittered grid candidates; noise decides the forest patches.
    const STEP: f32 = 6.5;
    let nx = ((max.x - min.x) / STEP) as usize;
    let nz = ((max.y - min.y) / STEP) as usize;
    let mut count = 0u32;
    for gz in 0..nz {
        for gx in 0..nx {
            let seed = (gz as u32) * 65_537 + gx as u32;
            let jx = (hash01(seed * 3 + 1) - 0.5) * STEP * 0.9;
            let jz = (hash01(seed * 3 + 2) - 0.5) * STEP * 0.9;
            let p =
                Vec2::new(min.x + gx as f32 * STEP, min.y + gz as f32 * STEP) + Vec2::new(jx, jz);
            let r = p.length();
            // Patchy forest noise; bushes spill past the forest edge.
            let forest = fbm(p / 100.0 + Vec2::splat(211.3));
            let kind = if forest > 0.55 && r > 200.0 {
                if hash01(seed * 5 + 3) < 0.55 {
                    Kind::Pine
                } else {
                    Kind::Broadleaf
                }
            } else if forest > 0.47 && r > 150.0 && hash01(seed * 5 + 4) < 0.35 {
                Kind::Bush
            } else {
                continue;
            };
            let h = terrain.height_at(p.x, p.y);
            if !(1.0..13.0).contains(&h) {
                continue;
            }
            if terrain.slope_at(p.x, p.y) > 0.45 {
                continue;
            }
            // Clear of the river corridor (incl. its banks).
            let river_d = (p.x - river_center_x(p.y)).abs();
            if river_d < river_half_width(p.y) * 2.3 + 8.0 {
                continue;
            }
            // Pines take over on higher ground.
            let kind = if h > 7.0 && matches!(kind, Kind::Broadleaf) {
                Kind::Pine
            } else {
                kind
            };
            let yaw = hash01(seed * 11 + 5) * std::f32::consts::TAU;
            let scale = 1.0 + 0.6 * hash01(seed * 11 + 6);
            let at = Vec3::new(p.x, h - 0.15, p.y); // sink slightly into ground
            let cx = (((p.x - min.x) / chunk_world) as usize).min(CHUNKS_X - 1);
            let cz = (((p.y - min.y) / chunk_world) as usize).min(CHUNKS_Z - 1);
            add_plant(&mut chunks[cz * CHUNKS_X + cx], kind, yaw, scale, at, seed);
            count += 1;
        }
    }

    let mut spawned = 0u32;
    for soup in chunks.into_iter() {
        if soup.is_empty() {
            continue;
        }
        let mesh = soup.into_mesh();
        let aabb = mesh.compute_aabb();
        let handle = meshes.add(mesh);
        let mut e = commands.spawn((Mesh3d(handle), MeshMaterial3d(material.clone()), Plants));
        if let Some(aabb) = aabb {
            e.insert(aabb);
        }
        spawned += 1;
    }
    info!("vegetation: {count} plants in {spawned} chunks");
}

struct TreePart {
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
}

struct TreeMesh {
    parts: Vec<TreePart>,
    height: f32,
    detail_size: f32,
}

#[derive(Component)]
struct TreePartSlot(usize);

#[derive(Resource, Default)]
struct TreeAssets(Vec<TreeAsset>);

struct TreeAsset {
    name: &'static str,
    /// Crown radius in metres at scale 1: the farthest horizontal distance
    /// of any near or middle vertex from the trunk base, so the circle
    /// covers the plant at every yaw, leaning crowns included. The far
    /// card is left out: its planes carry transparent padding.
    reach: f32,
    levels: [TreeMesh; 3],
}

#[derive(Component)]
struct TreeLevel {
    asset: usize,
    level: usize,
}

fn load_trees(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut trees = Vec::new();
    for (name, fallback, budgets) in [
        (
            "oak",
            "oak_trunks_light_v1/natural/oak/tree.glb",
            [3000, 700, 4],
        ),
        (
            "oak_lighter",
            "oak_trunks_light_v1/lighter/oak/tree.glb",
            [3000, 700, 4],
        ),
        (
            "mature_oak",
            "oak_trunks_light_v1/natural/mature_oak/tree.glb",
            [3000, 700, 4],
        ),
        (
            "mature_oak_lighter",
            "oak_trunks_light_v1/lighter/mature_oak/tree.glb",
            [3000, 700, 4],
        ),
        (
            "leaning_oak",
            "oak_trunks_light_v1/natural/leaning_oak/tree.glb",
            [3000, 700, 4],
        ),
        (
            "leaning_oak_lighter",
            "oak_trunks_light_v1/lighter/leaning_oak/tree.glb",
            [3000, 700, 4],
        ),
        (
            "shrub_b_v4",
            "shrub_b_v4_fuller/shrub_b/tree.glb",
            [1900, 120, 4],
        ),
        (
            "silver_birch_warm",
            "birch_warm_v1/birch_warm/silver_birch/tree.glb",
            [2000, 450, 4],
        ),
        (
            "oak_pale",
            "oak_pale_v1/oak/tree.glb",
            [3000, 700, 4],
        ),
        (
            "mature_oak_pale",
            "oak_pale_v1/mature_oak/tree.glb",
            [3000, 700, 4],
        ),
        (
            "leaning_oak_pale",
            "oak_pale_v1/leaning_oak/tree.glb",
            [3000, 700, 4],
        ),
        (
            "shrub_a",
            "shrub_a_v1/shrub_a/tree.glb",
            [2800, 900, 4],
        ),
        (
            "shrub_b_sandbox",
            "shrub_b_lod_v1/shrub_b/tree.glb",
            [1900, 600, 4],
        ),
    ] {
        let shipped = root.join(format!("assets/vegetation/{name}.glb"));
        let path = if shipped.exists() {
            shipped
        } else {
            root.join("assets_dev/vegetation").join(fallback)
        };
        match read_tree(&path, budgets) {
            Ok((levels, textures)) => {
                let reach = horizontal_reach(&levels);
                let material_handles: Vec<_> = textures
                    .into_iter()
                    .map(|(image, normal, opaque)| {
                        materials.add(StandardMaterial {
                            base_color_texture: Some(images.add(image)),
                            normal_map_texture: normal.map(|image| images.add(image)),
                            alpha_mode: if opaque {
                                AlphaMode::Opaque
                            } else {
                                AlphaMode::Mask(0.5)
                            },
                            perceptual_roughness: 0.92,
                            reflectance: 0.15,
                            cull_mode: if opaque { Some(Face::Back) } else { None },
                            // Normals describe the crown volume, not each card face.
                            double_sided: false,
                            ..default()
                        })
                    })
                    .collect();
                let levels = levels.map(|(parts, height)| {
                    // Broad, low shrubs need detail while their crown is still wide on screen.
                    let detail_size = parts
                        .iter()
                        .filter_map(|(mesh, _)| mesh.compute_aabb())
                        .map(|bounds| (bounds.half_extents * 2.0).max_element())
                        .fold(height, f32::max);
                    TreeMesh {
                        parts: parts
                            .into_iter()
                            .map(|(mesh, material)| TreePart {
                                mesh: meshes.add(mesh),
                                material: material_handles[material].clone(),
                            })
                            .collect(),
                        height,
                        detail_size,
                    }
                });
                info!(
                    "vegetation: loaded {name} L0/L1/card from {}",
                    path.display()
                );
                trees.push(TreeAsset {
                    name,
                    reach,
                    levels,
                });
            }
            Err(error) => warn!("vegetation: cannot load {}: {error}", path.display()),
        }
    }
    commands.insert_resource(TreeAssets(trees));
}

type TreeLevelData = (Vec<(Mesh, usize)>, f32);

/// `TreeAsset::reach` of a decoded tree.
fn horizontal_reach(levels: &[TreeLevelData; 3]) -> f32 {
    levels[..2]
        .iter()
        .flat_map(|(parts, _)| parts)
        .filter_map(|(mesh, _)| match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(bevy::mesh::VertexAttributeValues::Float32x3(positions)) => Some(positions),
            _ => None,
        })
        .flatten()
        .map(|p| Vec2::new(p[0], p[2]).length())
        .fold(0.0, f32::max)
}
type TreeData = ([TreeLevelData; 3], Vec<(Image, Option<Image>, bool)>);

/// Decode the whole asset before publishing handles, so a failed import leaves no assets behind.
fn read_tree(path: &std::path::Path, budgets: [usize; 3]) -> Result<TreeData, String> {
    let glb = gltf::Gltf::open(path).map_err(|e| e.to_string())?;
    let blob = glb.blob.as_deref().ok_or("missing binary buffer")?;
    for node in glb.nodes() {
        let matrix = Mat4::from_cols_array_2d(&node.transform().matrix());
        if !matrix.abs_diff_eq(Mat4::IDENTITY, 0.00001) || node.skin().is_some() {
            return Err("tree nodes must have applied transforms and no skin".into());
        }
    }
    let mut textures = Vec::new();
    let mut material_indices = std::collections::HashMap::new();
    let mut levels = Vec::new();
    for (name, budget) in ["L0", "L1", "card"].into_iter().zip(budgets) {
        let mut nodes = glb.nodes().filter(|node| node.name() == Some(name));
        let node = nodes.next().ok_or_else(|| format!("missing {name}"))?;
        if nodes.next().is_some() {
            return Err(format!("duplicate {name}"));
        }
        let source = node.mesh().ok_or_else(|| format!("{name} has no mesh"))?;
        let expected_parts = if name == "card" { 1 } else { 2 };
        let mut primitives: Vec<_> = source.primitives().collect();
        if primitives.is_empty() || primitives.len() > expected_parts {
            return Err(format!(
                "{name} must have one cutout primitive and at most one opaque bark primitive"
            ));
        }
        // Slot zero remains the cutout surface when switching to the far card.
        primitives.sort_by_key(|p| p.material().alpha_mode() == gltf::material::AlphaMode::Opaque);
        let mut parts = Vec::new();
        let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
        let mut triangle_count = 0;
        for (slot, primitive) in primitives.into_iter().enumerate() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                return Err(format!("{name} must contain indexed triangles"));
            }
            let reader = primitive.reader(|buffer| (buffer.index() == 0).then_some(blob));
            let positions: Vec<_> = reader
                .read_positions()
                .ok_or("missing positions")?
                .collect();
            let normals: Vec<_> = reader.read_normals().ok_or("missing normals")?.collect();
            let uv: Vec<_> = reader
                .read_tex_coords(0)
                .ok_or("missing UV0")?
                .into_f32()
                .collect();
            let wind: Vec<_> = reader
                .read_tex_coords(1)
                .ok_or("missing wind UV1")?
                .into_f32()
                .collect();
            let colors: Vec<_> = reader
                .read_colors(0)
                .ok_or("missing vertex colours")?
                .into_rgba_f32()
                .collect();
            let indices: Vec<_> = reader
                .read_indices()
                .ok_or("missing indices")?
                .into_u32()
                .collect();
            let count = positions.len();
            if count == 0
                || [normals.len(), uv.len(), wind.len(), colors.len()]
                    .iter()
                    .any(|n| *n != count)
                || indices.is_empty()
                || !indices.len().is_multiple_of(3)
                || indices.len() / 3 > budget
                || indices.iter().any(|i| *i as usize >= count)
                || positions
                    .iter()
                    .flatten()
                    .chain(normals.iter().flatten())
                    .chain(uv.iter().flatten())
                    .chain(wind.iter().flatten())
                    .chain(colors.iter().flatten())
                    .any(|v| !v.is_finite())
            {
                return Err(format!(
                    "{name} has invalid geometry or exceeds its {budget} triangle budget"
                ));
            }
            if colors.iter().any(|c| (c[3] - 1.0).abs() > 0.001) {
                return Err(format!("{name} vertex alpha must stay opaque"));
            }
            triangle_count += indices.len() / 3;
            let material = primitive.material();
            let opaque = slot == 1;
            if opaque {
                if material.alpha_mode() != gltf::material::AlphaMode::Opaque
                    || material.double_sided()
                {
                    return Err(format!("{name} bark must be opaque and outward-facing"));
                }
            } else if material.alpha_mode() != gltf::material::AlphaMode::Mask
                || !material.double_sided()
                || (material.alpha_cutoff().unwrap_or(0.5) - 0.5).abs() > 0.001
            {
                return Err(format!(
                    "{name} foliage must use a double-sided alpha mask with cutoff 0.5"
                ));
            }
            let source_index = material.index().ok_or("missing material")?;
            let material_index = if let Some(&index) = material_indices.get(&source_index) {
                index
            } else {
                let texture = material
                    .pbr_metallic_roughness()
                    .base_color_texture()
                    .ok_or("missing atlas")?;
                if texture.tex_coord() != 0 {
                    return Err("atlas must use UV0".into());
                }
                let mut image = read_tree_image(texture.texture().source(), blob, false, opaque)?;
                // Tiled bark declares mirrored wrapping; atlas textures retain clamped edges.
                let sampler = texture.texture().sampler();
                if let ImageSampler::Descriptor(ref mut descriptor) = image.sampler {
                    if sampler.wrap_s() == gltf::texture::WrappingMode::MirroredRepeat {
                        descriptor.address_mode_u = ImageAddressMode::MirrorRepeat;
                    }
                    if sampler.wrap_t() == gltf::texture::WrappingMode::MirroredRepeat {
                        descriptor.address_mode_v = ImageAddressMode::MirrorRepeat;
                    }
                }
                let normal = material
                    .normal_texture()
                    .map(|texture| {
                        if texture.tex_coord() != 0 || (texture.scale() - 1.0).abs() > 0.001 {
                            return Err("normal atlas must use UV0 and unit strength".into());
                        }
                        read_tree_image(texture.texture().source(), blob, true, false)
                    })
                    .transpose()?;
                let index = textures.len();
                textures.push((image, normal, opaque));
                material_indices.insert(source_index, index);
                index
            };
            min_y = positions.iter().map(|p| p[1]).fold(min_y, f32::min);
            max_y = positions.iter().map(|p| p[1]).fold(max_y, f32::max);
            let mut mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uv)
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, wind)
            .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
            .with_inserted_indices(Indices::U32(indices));
            if material.normal_texture().is_some() {
                mesh.generate_tangents().map_err(|e| e.to_string())?;
            }
            parts.push((mesh, material_index));
        }
        if triangle_count > budget {
            return Err(format!("{name} exceeds its {budget} triangle budget"));
        }
        info!(
            "vegetation: tree {name}: {triangle_count} triangles, height {:.3} m",
            max_y - min_y
        );
        levels.push((parts, max_y - min_y));
    }
    let levels = levels
        .try_into()
        .map_err(|_| "expected three tree levels")?;
    Ok((levels, textures))
}

fn read_tree_image(
    source: gltf::Image<'_>,
    blob: &[u8],
    normal: bool,
    opaque: bool,
) -> Result<Image, String> {
    let gltf::image::Source::View { view, mime_type } = source.source() else {
        return Err("atlas must be embedded".into());
    };
    let bytes = blob
        .get(view.offset()..view.offset() + view.length())
        .ok_or("invalid atlas range")?;
    let mut image = Image::from_buffer(
        bytes,
        ImageType::MimeType(mime_type),
        CompressedImageFormats::NONE,
        !normal,
        ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::ClampToEdge,
            address_mode_v: ImageAddressMode::ClampToEdge,
            ..ImageSamplerDescriptor::linear()
        }),
        RenderAssetUsages::RENDER_WORLD,
    )
    .map_err(|e| e.to_string())?;
    let format = if normal {
        TextureFormat::Rgba8Unorm
    } else {
        TextureFormat::Rgba8UnormSrgb
    };
    if image.texture_descriptor.format != format {
        return Err("atlas must be RGBA8".into());
    }
    if opaque {
        // Opaque bark is isolated from leaf alpha before mip generation.
        for pixel in image
            .data
            .as_mut()
            .ok_or("atlas has no pixels")?
            .chunks_exact_mut(4)
        {
            pixel[3] = 255;
        }
    }
    foliage_mips(&mut image, normal, !opaque)?;
    Ok(image)
}

/// Alpha-weighted linear colour avoids dark borders; coverage scaling retains distant leaves.
fn foliage_mips(image: &mut Image, normal: bool, preserve_coverage: bool) -> Result<(), String> {
    let size = image.texture_descriptor.size;
    let (mut w, mut h) = (size.width as usize, size.height as usize);
    let mut previous = image.data.take().ok_or("atlas has no CPU pixels")?;
    if previous.len() != w * h * 4 {
        return Err("atlas dimensions do not match its pixels".into());
    }
    let lut: [f32; 256] = std::array::from_fn(|i| {
        let v = i as f32 / 255.0;
        if normal {
            v
        } else if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    });
    let encode = |v: f32| -> u8 {
        let s = if normal {
            v
        } else if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    };
    let coverage = previous.chunks_exact(4).filter(|p| p[3] >= 128).count() as f32 / (w * h) as f32;
    let mut data = previous.clone();
    let mut levels = 1;
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut next = vec![0u8; nw * nh * 4];
        for y in 0..nh {
            for x in 0..nw {
                let (mut sum, mut weight, mut samples) = ([0.0; 3], 0.0, 0.0);
                // Area bins include the final row/column of non-power-of-two atlases.
                for sy in y * h / nh..(y + 1) * h / nh {
                    for sx in x * w / nw..(x + 1) * w / nw {
                        let pixel = &previous[(sy * w + sx) * 4..][..4];
                        let alpha = pixel[3] as f32 / 255.0;
                        for c in 0..3 {
                            sum[c] += lut[pixel[c] as usize] * alpha;
                        }
                        weight += alpha;
                        samples += 1.0;
                    }
                }
                let pixel = &mut next[(y * nw + x) * 4..][..4];
                let mut average = Vec3::from_array(sum) / weight.max(0.0001);
                if normal {
                    average = ((average * 2.0 - Vec3::ONE).normalize_or_zero() + Vec3::ONE) * 0.5;
                }
                for c in 0..3 {
                    pixel[c] = encode(average[c]);
                }
                pixel[3] = (weight / samples * 255.0 + 0.5) as u8;
            }
        }
        if preserve_coverage {
            let target = (coverage * (nw * nh) as f32).round() as usize;
            let (mut lo, mut hi) = (0.0, 4.0);
            for _ in 0..12 {
                let scale = (lo + hi) * 0.5;
                let count = next
                    .chunks_exact(4)
                    .filter(|p| p[3] as f32 * scale >= 127.5)
                    .count();
                if count < target {
                    lo = scale;
                } else {
                    hi = scale;
                }
            }
            for pixel in next.chunks_exact_mut(4) {
                pixel[3] = (pixel[3] as f32 * hi).clamp(0.0, 255.0).round() as u8;
            }
        }
        data.extend_from_slice(&next);
        previous = next;
        (w, h) = (nw, nh);
        levels += 1;
    }
    image.data = Some(data);
    image.texture_descriptor.mip_level_count = levels;
    Ok(())
}

fn select_tree_level(
    asset: Res<TreeAssets>,
    cameras: Query<(&Camera, &Projection, &Transform), With<crate::camera::RtsCamera>>,
    mut plants: Query<(&Transform, &mut TreeLevel, &Children), With<Plants>>,
    mut parts: Query<(
        &TreePartSlot,
        &mut Mesh3d,
        &mut MeshMaterial3d<StandardMaterial>,
        &mut Visibility,
    )>,
) {
    let Ok((camera, projection, camera_transform)) = cameras.single() else {
        return;
    };
    let Projection::Perspective(perspective) = projection else {
        return;
    };
    let viewport = camera.logical_viewport_size().map_or(900.0, |s| s.y);
    for (transform, mut selected, children) in &mut plants {
        let Some(tree) = asset.0.get(selected.asset) else {
            continue;
        };
        let levels = &tree.levels;
        let centre = transform.transform_point(Vec3::Y * levels[0].height * 0.5);
        let size = levels[0].detail_size * transform.scale.abs().max_element();
        let distance = camera_transform.translation.distance(centre).max(1.0);
        let pixels = size * viewport / (2.0 * (perspective.fov * 0.5).tan() * distance);
        let mut next = match selected.level {
            0 if pixels < 16.0 => 2,
            0 if pixels < 108.0 => 1,
            1 if pixels > 132.0 => 0,
            1 if pixels < 16.0 => 2,
            2 if pixels > 132.0 => 0,
            2 if pixels > 20.0 => 1,
            level => level,
        };
        // Vertical cards lose their crown footprint from above. Keep the mesh
        // for steep views, with angular hysteresis around the card limit.
        let elevation = (camera_transform.translation.y - centre.y).abs() / distance;
        let card_limit = if selected.level == 2 { 0.84 } else { 0.80 };
        if next == 2 && elevation > card_limit {
            next = 1;
        }
        if next != selected.level {
            debug!(
                "vegetation: {} level {next}, projected size {pixels:.1} px",
                tree.name
            );
            selected.level = next;
            for child in children {
                let Ok((slot, mut mesh, mut material, mut visibility)) = parts.get_mut(*child)
                else {
                    continue;
                };
                if let Some(part) = levels[next].parts.get(slot.0) {
                    mesh.0 = part.mesh.clone();
                    material.0 = part.material.clone();
                    *visibility = Visibility::Inherited;
                } else {
                    *visibility = Visibility::Hidden;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::render::render_resource::{Extent3d, TextureDimension};

    fn image(width: u32, height: u32, pixels: Vec<u8>) -> Image {
        Image::new(
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            pixels,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        )
    }

    #[test]
    fn mip_colour_ignores_transparent_rgb() {
        let mut atlas = image(
            2,
            2,
            vec![255, 0, 0, 255, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0],
        );
        foliage_mips(&mut atlas, false, true).unwrap();
        let data = atlas.data.unwrap();
        assert_eq!(&data[16..19], &[255, 0, 0]);
        assert_eq!(atlas.texture_descriptor.mip_level_count, 2);
    }

    #[test]
    fn opaque_bark_stays_opaque_through_every_mip() {
        let mut atlas = image(8, 8, [170, 160, 150, 255].repeat(64));
        foliage_mips(&mut atlas, false, false).unwrap();
        assert_eq!(atlas.texture_descriptor.mip_level_count, 4);
        for pixel in atlas.data.unwrap().chunks_exact(4) {
            assert_eq!(pixel, &[170, 160, 150, 255]);
        }
    }

    #[test]
    fn odd_atlas_edges_contribute_to_mips() {
        let mut pixels = [255, 0, 0, 255].repeat(9);
        pixels[32..36].copy_from_slice(&[0, 0, 255, 255]);
        let mut atlas = image(3, 3, pixels);
        foliage_mips(&mut atlas, false, true).unwrap();
        let data = atlas.data.unwrap();
        // The blue bottom-right texel must survive the 3x3-to-1x1 average.
        assert!(data[38] > 80);
        assert!(data[36] > 200);
        assert!(data[39] >= 128);
    }

    /// The grassland's assets and their crown reach, read from the shipped files.
    fn grassland_specs() -> Vec<(&'static str, f32)> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let names = MATURE_OAKS
            .iter()
            .chain(UPRIGHT_OAKS)
            .chain(LEANING_OAKS)
            .chain(BIRCHES)
            .chain(&[SHRUB_A, SHRUB_B]);
        names
            .map(|&name| {
                let path = root.join(format!("assets/vegetation/{name}.glb"));
                let (levels, _) =
                    read_tree(&path, [3000, 900, 4]).unwrap_or_else(|e| panic!("{name}: {e}"));
                (name, horizontal_reach(&levels))
            })
            .collect()
    }

    #[test]
    fn grassland_crowns_keep_clear_of_deployment_and_the_open_centre() {
        let terrain = crate::terrain::build_terrain(MapKind::Grassland);
        let specs = grassland_specs();
        let (min, max) = (terrain.min(), terrain.max());
        for army_gap in [20.0, 60.0, 120.0] {
            let planting = grassland_planting(&specs, &terrain, army_gap);
            assert!(planting.missing.is_empty(), "missing {:?}", planting.missing);
            let trees = planting.plants.iter().filter(|p| p.tree).count();
            assert!(trees > 50 && planting.plants.len() - trees > 50, "gap {army_gap}: {trees} trees");
            let (lo, hi) = crate::orders::deploy_zone_for(&terrain, army_gap);
            let keep_out = [
                (lo, hi),
                (Vec2::new(lo.x, -hi.y), Vec2::new(hi.x, -lo.y)),
                (Vec2::new(-OPEN_CENTRE, hi.y), Vec2::new(OPEN_CENTRE, -hi.y)),
            ];
            for p in &planting.plants {
                let (low, high) = if p.tree { TREE_SCALE } else { SHRUB_SCALE };
                assert!((low..=high).contains(&p.scale));
                // Checked at the largest scale the kind is drawn at.
                let r = specs[p.asset].1 * high;
                let name = specs[p.asset].0;
                if let Some(rect) = GRASSLAND_STANDS[p.stand].inside {
                    let (a, b) = (rect.min, rect.max);
                    assert!(
                        p.pos.x - r >= a.x && p.pos.x + r <= b.x && p.pos.y - r >= a.y && p.pos.y + r <= b.y,
                        "gap {army_gap}: {name} at {} leaves {a}..{b}", p.pos
                    );
                    continue;
                }
                assert!(
                    p.pos.x - r >= min.x && p.pos.x + r <= max.x && p.pos.y - r >= min.y && p.pos.y + r <= max.y,
                    "gap {army_gap}: {name} at {} leaves the field", p.pos
                );
                for (a, b) in keep_out {
                    let nearest = p.pos.clamp(a, b);
                    assert!(
                        p.pos.distance(nearest) >= r - 1e-4,
                        "gap {army_gap}: {name} at {} reaches into {a}..{b}", p.pos
                    );
                }
            }
        }
    }

    #[test]
    fn grassland_planting_is_the_same_every_run() {
        let terrain = crate::terrain::build_terrain(MapKind::Grassland);
        let specs = grassland_specs();
        let key = |p: &Placed| (p.asset, p.pos.to_array(), p.yaw, p.scale);
        let first: Vec<_> = grassland_planting(&specs, &terrain, 60.0).plants.iter().map(key).collect();
        let second: Vec<_> = grassland_planting(&specs, &terrain, 60.0).plants.iter().map(key).collect();
        assert_eq!(first, second);
    }
}
