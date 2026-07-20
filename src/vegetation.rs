//! Static low-poly vegetation: code-built trees and bushes scattered by
//! noise into patchy forests, merged into one flat-shaded vertex-colored
//! mesh per terrain chunk (one draw call per non-empty chunk, frustum
//! culling for free). Visual only — no gameplay effect.

use bevy::asset::RenderAssetUsages;
use bevy::camera::primitives::MeshAabb;
use bevy::mesh::PrimitiveTopology;
use bevy::prelude::*;

use crate::terrain::{
    CELL, CHUNK_CELLS, CHUNKS_X, CHUNKS_Z, Terrain, fbm, river_center_x, river_half_width,
};
use crate::units::hash01;

pub struct VegetationPlugin;

impl Plugin for VegetationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_vegetation);
    }
}

/// Flat-shaded triangle soup builder (positions/normals/colors), same
/// style as the terrain chunks.
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

fn spawn_vegetation(
    mut commands: Commands,
    terrain: Res<Terrain>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
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
            let p = Vec2::new(min.x + gx as f32 * STEP, min.y + gz as f32 * STEP)
                + Vec2::new(jx, jz);
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
        let mut e = commands.spawn((Mesh3d(handle), MeshMaterial3d(material.clone())));
        if let Some(aabb) = aabb {
            e.insert(aabb);
        }
        spawned += 1;
    }
    info!("vegetation: {count} plants in {spawned} chunks");
}
