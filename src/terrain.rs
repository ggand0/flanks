//! Chunked deformable heightmap terrain. Heights live in one big vertex grid;
//! chunks are 32x32-cell mesh entities rebuilt when a crater dirties them.
//! Look: flat-shaded triangle soup with hard height-banded colors.

use bevy::asset::RenderAssetUsages;
use bevy::camera::primitives::{Aabb, MeshAabb};
use bevy::mesh::PrimitiveTopology;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use std::time::Instant;

use crate::units::hash01;

pub const CELL: f32 = 2.0;
pub const CHUNK_CELLS: usize = 32;
pub const CHUNKS_X: usize = 16;
pub const CHUNKS_Z: usize = 12;
const VERTS_X: usize = CHUNKS_X * CHUNK_CELLS + 1;
const VERTS_Z: usize = CHUNKS_Z * CHUNK_CELLS + 1;

#[derive(Resource)]
pub struct Terrain {
    /// Vertex heights, row-major [z][x].
    heights: Vec<f32>,
    /// World-space min corner.
    pub origin: Vec2,
    dirty: Vec<bool>,
}

impl Terrain {
    pub fn min(&self) -> Vec2 {
        self.origin
    }

    pub fn max(&self) -> Vec2 {
        self.origin
            + Vec2::new(
                (VERTS_X - 1) as f32 * CELL,
                (VERTS_Z - 1) as f32 * CELL,
            )
    }

    #[inline]
    fn h(&self, x: usize, z: usize) -> f32 {
        self.heights[z * VERTS_X + x]
    }

    /// Bilinear height sample, clamped to the field.
    #[inline]
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let g = (Vec2::new(x, z) - self.origin) / CELL;
        let gx = g.x.clamp(0.0, (VERTS_X - 2) as f32);
        let gz = g.y.clamp(0.0, (VERTS_Z - 2) as f32);
        let x0 = gx as usize;
        let z0 = gz as usize;
        let fx = gx - x0 as f32;
        let fz = gz - z0 as f32;
        let h00 = self.h(x0, z0);
        let h10 = self.h(x0 + 1, z0);
        let h01 = self.h(x0, z0 + 1);
        let h11 = self.h(x0 + 1, z0 + 1);
        h00 * (1.0 - fx) * (1.0 - fz)
            + h10 * fx * (1.0 - fz)
            + h01 * (1.0 - fx) * fz
            + h11 * fx * fz
    }

    /// |gradient| of the height field (rise per meter).
    #[inline]
    pub fn slope_at(&self, x: f32, z: f32) -> f32 {
        const E: f32 = 1.0;
        let h = self.height_at(x, z);
        let dx = (self.height_at(x + E, z) - h) / E;
        let dz = (self.height_at(x, z + E) - h) / E;
        (dx * dx + dz * dz).sqrt()
    }

    /// Carve a crater: smooth depression + small raised rim. Marks chunks dirty.
    pub fn carve_crater(&mut self, center: Vec2, radius: f32, depth: f32) {
        let rim_r = radius * 1.35;
        let gmin = ((center - rim_r - self.origin) / CELL).floor();
        let gmax = ((center + rim_r - self.origin) / CELL).ceil();
        let x0 = (gmin.x.max(0.0)) as usize;
        let z0 = (gmin.y.max(0.0)) as usize;
        let x1 = (gmax.x as usize).min(VERTS_X - 1);
        let z1 = (gmax.y as usize).min(VERTS_Z - 1);
        for z in z0..=z1 {
            for x in x0..=x1 {
                let p = self.origin + Vec2::new(x as f32, z as f32) * CELL;
                let d = p.distance(center);
                let dh = if d < radius {
                    let t = 1.0 - (d / radius) * (d / radius);
                    -depth * t * t
                } else if d < rim_r {
                    let t = (d - radius) / (rim_r - radius);
                    depth * 0.12 * (1.0 - t * t)
                } else {
                    continue;
                };
                self.heights[z * VERTS_X + x] += dh;
            }
        }
        // Chunk c spans verts [c*32, c*32+32]; a vertex touches up to 2 chunks.
        let cx0 = x0.saturating_sub(1) / CHUNK_CELLS;
        let cz0 = z0.saturating_sub(1) / CHUNK_CELLS;
        let cx1 = (x1 / CHUNK_CELLS).min(CHUNKS_X - 1);
        let cz1 = (z1 / CHUNK_CELLS).min(CHUNKS_Z - 1);
        for cz in cz0..=cz1 {
            for cx in cx0..=cx1 {
                self.dirty[cz * CHUNKS_X + cx] = true;
            }
        }
    }

    /// March a ray to the surface. Returns the hit point.
    pub fn raycast(&self, ray: Ray3d) -> Option<Vec3> {
        let mut t = 0.0f32;
        let mut prev_t = 0.0f32;
        let dir = ray.direction.as_vec3();
        for _ in 0..1500 {
            let p = ray.origin + dir * t;
            if p.y < self.height_at(p.x, p.z) {
                // Bisect between prev_t and t.
                let (mut lo, mut hi) = (prev_t, t);
                for _ in 0..10 {
                    let mid = 0.5 * (lo + hi);
                    let q = ray.origin + dir * mid;
                    if q.y < self.height_at(q.x, q.z) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                return Some(ray.origin + dir * (0.5 * (lo + hi)));
            }
            prev_t = t;
            t += 1.5;
            if t > 2500.0 {
                break;
            }
        }
        None
    }
}

/// Mesh handle + entity per chunk, indexed [cz * CHUNKS_X + cx].
#[derive(Resource, Default)]
struct TerrainChunks(Vec<Handle<Mesh>>);

pub struct TerrainPlugin;

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainChunks>()
            .add_systems(PreStartup, generate_terrain)
            .add_systems(Startup, spawn_chunks)
            .add_systems(Update, (crater_tool, auto_test_craters, remesh_dirty).chain());
    }
}

fn fbm(p: Vec2) -> f32 {
    fn lattice(xi: i32, zi: i32) -> f32 {
        let ux = xi as u32;
        let uz = zi as u32;
        hash01(ux.wrapping_mul(0x9E37_79B1) ^ uz.wrapping_mul(0x85EB_CA77))
    }
    fn value_noise(p: Vec2) -> f32 {
        let x0 = p.x.floor();
        let z0 = p.y.floor();
        let fx = p.x - x0;
        let fz = p.y - z0;
        let sx = fx * fx * (3.0 - 2.0 * fx);
        let sz = fz * fz * (3.0 - 2.0 * fz);
        let (xi, zi) = (x0 as i32, z0 as i32);
        let v00 = lattice(xi, zi);
        let v10 = lattice(xi + 1, zi);
        let v01 = lattice(xi, zi + 1);
        let v11 = lattice(xi + 1, zi + 1);
        let a = v00 + (v10 - v00) * sx;
        let b = v01 + (v11 - v01) * sx;
        (a + (b - a) * sz) * 2.0 - 1.0
    }
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut sum = 0.0;
    for _ in 0..4 {
        sum += value_noise(p * freq) * amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum
}

fn generate_terrain(mut commands: Commands) {
    let origin = Vec2::new(
        -(VERTS_X as f32 - 1.0) * CELL * 0.5,
        -(VERTS_Z as f32 - 1.0) * CELL * 0.5,
    );
    let mut heights = vec![0.0f32; VERTS_X * VERTS_Z];
    for z in 0..VERTS_Z {
        for x in 0..VERTS_X {
            let p = origin + Vec2::new(x as f32, z as f32) * CELL;
            // Large rolling landforms + medium detail + ridged peaks.
            let base = fbm(p / 320.0) * 22.0;
            let detail = fbm(p / 90.0 + Vec2::splat(37.7)) * 4.5;
            let r = 1.0 - fbm(p / 260.0 + Vec2::splat(91.3)).abs().min(1.0);
            let ridged = r * r * 16.0;
            let mut h = base + detail + ridged;
            // Soften (not flatten) the central battlefield: rolling and
            // readable in the middle, dramatic on the outskirts.
            let center_dist = (p.length() - 140.0).max(0.0) / 260.0;
            h *= 0.35 + 0.65 * center_dist.min(1.0);
            // Bias up so the battlefield sits in the grass bands; dirt only
            // in real hollows and crater floors.
            heights[z * VERTS_X + x] = h + 2.6;
        }
    }
    commands.insert_resource(Terrain {
        heights,
        origin,
        dirty: vec![false; CHUNKS_X * CHUNKS_Z],
    });
}

fn spawn_chunks(
    mut commands: Commands,
    terrain: Res<Terrain>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut chunks: ResMut<TerrainChunks>,
) {
    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 1.0,
        reflectance: 0.05,
        ..default()
    });
    for cz in 0..CHUNKS_Z {
        for cx in 0..CHUNKS_X {
            let mesh = build_chunk_mesh(&terrain, cx, cz);
            let aabb = mesh.compute_aabb();
            let handle = meshes.add(mesh);
            chunks.0.push(handle.clone());
            let mut e = commands.spawn((Mesh3d(handle), MeshMaterial3d(material.clone())));
            if let Some(aabb) = aabb {
                e.insert(aabb);
            }
        }
    }
}

fn band_color(h: f32, slope: f32) -> [f32; 4] {
    let c = if slope > 0.75 {
        Color::srgb(0.46, 0.42, 0.36) // scree on steep faces
    } else if h < -2.5 {
        Color::srgb(0.33, 0.25, 0.17) // crater floor / deep dirt
    } else if h < 0.0 {
        Color::srgb(0.43, 0.34, 0.22) // dirt
    } else if h < 5.0 {
        Color::srgb(0.34, 0.43, 0.22) // low grass
    } else if h < 11.0 {
        Color::srgb(0.42, 0.50, 0.26) // grass
    } else if h < 17.0 {
        Color::srgb(0.52, 0.52, 0.33) // dry highland
    } else if h < 24.0 {
        Color::srgb(0.52, 0.48, 0.42) // rock
    } else {
        Color::srgb(0.78, 0.79, 0.82) // snowcap
    };
    c.to_linear().to_f32_array()
}

/// Flat-shaded triangle soup for one chunk: 2 triangles per cell, per-face
/// normal and one hard-banded color per triangle. World coords baked in.
fn build_chunk_mesh(terrain: &Terrain, cx: usize, cz: usize) -> Mesh {
    let n_tris = CHUNK_CELLS * CHUNK_CELLS * 2;
    let mut positions = Vec::with_capacity(n_tris * 3);
    let mut normals = Vec::with_capacity(n_tris * 3);
    let mut colors = Vec::with_capacity(n_tris * 3);

    let vx0 = cx * CHUNK_CELLS;
    let vz0 = cz * CHUNK_CELLS;
    for dz in 0..CHUNK_CELLS {
        for dx in 0..CHUNK_CELLS {
            let (x, z) = (vx0 + dx, vz0 + dz);
            let wp = |xx: usize, zz: usize| -> Vec3 {
                let w = terrain.origin + Vec2::new(xx as f32, zz as f32) * CELL;
                Vec3::new(w.x, terrain.h(xx, zz), w.y)
            };
            let p00 = wp(x, z);
            let p10 = wp(x + 1, z);
            let p01 = wp(x, z + 1);
            let p11 = wp(x + 1, z + 1);
            // Alternate the quad split diagonal for a less regular look.
            let tris = if (x + z) % 2 == 0 {
                [[p00, p01, p11], [p00, p11, p10]]
            } else {
                [[p00, p01, p10], [p10, p01, p11]]
            };
            for tri in tris {
                let n = (tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize_or_zero();
                let hc = (tri[0].y + tri[1].y + tri[2].y) / 3.0;
                let slope = (1.0 - n.y * n.y).sqrt() / n.y.max(0.1);
                let col = band_color(hc, slope);
                for v in tri {
                    positions.push(v);
                    normals.push(n);
                    colors.push(col);
                }
            }
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
}

fn remesh_dirty(
    mut terrain: ResMut<Terrain>,
    chunks: Res<TerrainChunks>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut chunk_entities: Query<(&Mesh3d, &mut Aabb)>,
) {
    if !terrain.dirty.iter().any(|d| *d) {
        return;
    }
    let t0 = Instant::now();
    let mut rebuilt = 0;
    for cz in 0..CHUNKS_Z {
        for cx in 0..CHUNKS_X {
            let ci = cz * CHUNKS_X + cx;
            if !terrain.dirty[ci] {
                continue;
            }
            let mesh = build_chunk_mesh(&terrain, cx, cz);
            let aabb = mesh.compute_aabb();
            let _ = meshes.insert(&chunks.0[ci], mesh);
            if let Some(new_aabb) = aabb {
                for (m, mut old) in &mut chunk_entities {
                    if m.0 == chunks.0[ci] {
                        *old = new_aabb;
                    }
                }
            }
            rebuilt += 1;
        }
    }
    terrain.dirty.fill(false);
    debug!(
        "remeshed {rebuilt} chunks in {:.2} ms",
        t0.elapsed().as_secs_f32() * 1000.0
    );
}

/// Debug tool: X carves a crater under the cursor.
fn crater_tool(
    keys: Res<ButtonInput<KeyCode>>,
    mut terrain: ResMut<Terrain>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform)>,
    time: Res<Time>,
    mut cooldown: Local<f32>,
) {
    *cooldown -= time.delta_secs();
    if !keys.pressed(KeyCode::KeyX) || *cooldown > 0.0 {
        return;
    }
    let Ok(window) = window.single() else { return };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, cam_tf)) = camera.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(cam_tf, cursor) else {
        return;
    };
    if let Some(hit) = terrain.raycast(ray) {
        terrain.carve_crater(Vec2::new(hit.x, hit.z), 11.0, 5.0);
        *cooldown = 0.15;
        info!("crater at ({:.0}, {:.0})", hit.x, hit.z);
    }
}

/// FL_TEST_CRATERS=1: carve a crater near the field center every 2 s
/// (screenshot/perf verification without input injection).
fn auto_test_craters(
    mut terrain: ResMut<Terrain>,
    time: Res<Time>,
    mut next: Local<f32>,
    mut n: Local<u32>,
) {
    if std::env::var("FL_TEST_CRATERS").is_err() {
        return;
    }
    if time.elapsed_secs() < *next {
        return;
    }
    *next = time.elapsed_secs() + 2.0;
    *n += 1;
    let a = hash01(*n * 7 + 1) * std::f32::consts::TAU;
    let r = hash01(*n * 7 + 2).sqrt() * 140.0;
    let center = Vec2::new(a.cos(), a.sin()) * r;
    let radius = 8.0 + hash01(*n * 7 + 3) * 6.0;
    terrain.carve_crater(center, radius, radius * 0.45);
    info!("test crater #{} at ({:.0}, {:.0}) r={radius:.1}", *n, center.x, center.y);
}
