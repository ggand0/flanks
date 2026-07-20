//! River water surface: an indexed strip mesh following the river
//! centerline (terrain.rs), rendered with a custom unlit-ish material
//! (assets/shaders/water.wgsl). Chunky quantized vertex waves + flat
//! derivative normals keep the faceted low-poly art direction.

use bevy::asset::RenderAssetUsages;
use bevy::camera::primitives::MeshAabb;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

use crate::terrain::{
    self, river_bed_depth, river_center_x, river_half_width, river_water_level,
};

pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<WaterMaterial>::default())
            .add_systems(Startup, spawn_water);
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct WaterMaterial {
    /// xyz = direction toward the sun (matches the scene light), w spare.
    #[uniform(0)]
    pub sun_dir: Vec4,
}

impl Material for WaterMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/water.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/water.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

/// Direction toward the sun, matching `setup_world`'s DirectionalLight.
fn sun_dir() -> Vec3 {
    let shine = Quat::from_euler(EulerRot::YXZ, 0.7, -0.75, 0.0) * Vec3::NEG_Z;
    -shine
}

/// Strip grid along the river: rows step in z, columns span t in
/// [-1.05, 1.05] so the edges tuck slightly under the banks. uv =
/// (t, bed depth fraction) for shore foam / deep-water coloring.
fn build_water_mesh(terrain: &terrain::Terrain) -> Mesh {
    const ROW_STEP: f32 = 6.0;
    const COLS: usize = 9;
    const T_MAX: f32 = 1.05; // just past the shoreline (t=1)

    let z0 = terrain.min().y;
    let z1 = terrain.max().y;
    let rows = ((z1 - z0) / ROW_STEP) as usize + 1;

    let mut positions = Vec::with_capacity(rows * COLS);
    let mut uvs = Vec::with_capacity(rows * COLS);
    for r in 0..rows {
        let z = (z0 + r as f32 * ROW_STEP).min(z1);
        let cx = river_center_x(z);
        let w = river_half_width(z);
        let y = river_water_level(z);
        for c in 0..COLS {
            let t = -T_MAX + 2.0 * T_MAX * c as f32 / (COLS - 1) as f32;
            positions.push([cx + t * w, y, z]);
            let depth_frac = if t.abs() <= 1.0 {
                river_bed_depth(t) / terrain::RIVER_DEPTH
            } else {
                0.0
            };
            uvs.push([t, depth_frac]);
        }
    }

    let mut indices = Vec::with_capacity((rows - 1) * (COLS - 1) * 6);
    for r in 0..rows - 1 {
        for c in 0..COLS - 1 {
            let a = (r * COLS + c) as u32;
            let b = a + COLS as u32;
            // Wound for an upward (+Y) front face.
            indices.extend_from_slice(&[a, b, a + 1]);
            indices.extend_from_slice(&[b, b + 1, a + 1]);
        }
    }

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
}

fn spawn_water(
    mut commands: Commands,
    terrain: Res<terrain::Terrain>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<WaterMaterial>>,
) {
    let mesh = build_water_mesh(&terrain);
    let aabb = mesh.compute_aabb();
    let handle = meshes.add(mesh);
    let material = materials.add(WaterMaterial {
        sun_dir: sun_dir().extend(0.0),
    });
    let mut e = commands.spawn((Mesh3d(handle), MeshMaterial3d(material)));
    if let Some(aabb) = aabb {
        e.insert(aabb);
    }
}
