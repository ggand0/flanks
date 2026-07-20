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
    self, BRIDGE_DECK_LIFT, BRIDGE_HALF_SPAN, BRIDGE_Z, RIVER_BANK, river_bed_depth,
    river_center_x, river_half_width, river_water_level,
};
use crate::vegetation::{Soup, srgb};

pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<WaterMaterial>::default())
            .add_systems(Startup, (spawn_water, spawn_bridge));
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
    if terrain.classic {
        return;
    }
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

/// The stone bridge at BRIDGE_Z: flat deck slab + parapets + two piers
/// standing in the channel. Low-poly soup, same style as everything
/// else. The walkable surface is terrain.rs's deck height override;
/// this mesh is purely the visual.
fn spawn_bridge(
    mut commands: Commands,
    terrain: Res<terrain::Terrain>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if terrain.classic {
        return;
    }
    let cx = river_center_x(BRIDGE_Z);
    let hw = river_half_width(BRIDGE_Z);
    let wl = river_water_level(BRIDGE_Z);
    let deck_y = wl + BRIDGE_DECK_LIFT;
    // Matches bridge_deck_contains: the deck lands on the bank profile.
    let deck_half = (1.0 + BRIDGE_DECK_LIFT / RIVER_BANK) * hw;

    let stone = srgb(0.52, 0.50, 0.46);
    let stone_light = srgb(0.58, 0.56, 0.52);
    let stone_dark = srgb(0.44, 0.42, 0.38);

    let mut soup = Soup::new();
    // Deck slab (top at deck_y).
    soup.cuboid(
        Vec3::new(cx, deck_y - 0.2, BRIDGE_Z),
        Vec3::new(deck_half, 0.2, BRIDGE_HALF_SPAN),
        stone_light,
    );
    // Parapets along both long edges.
    for side in [-1.0f32, 1.0] {
        soup.cuboid(
            Vec3::new(cx, deck_y + 0.45, BRIDGE_Z + side * (BRIDGE_HALF_SPAN - 0.3)),
            Vec3::new(deck_half, 0.45, 0.3),
            stone,
        );
    }
    // Piers down into the channel (bed is wl - RIVER_DEPTH at center).
    for off in [-0.55f32, 0.55] {
        let px = cx + off * hw;
        let bed = wl - terrain::RIVER_DEPTH;
        soup.cuboid(
            Vec3::new(px, (bed + deck_y - 0.4) * 0.5, BRIDGE_Z),
            Vec3::new(1.5, (deck_y - 0.4 - bed) * 0.5, BRIDGE_HALF_SPAN - 0.6),
            stone_dark,
        );
    }

    let mesh = soup.into_mesh();
    let aabb = mesh.compute_aabb();
    let handle = meshes.add(mesh);
    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 1.0,
        reflectance: 0.05,
        ..default()
    });
    let mut e = commands.spawn((Mesh3d(handle), MeshMaterial3d(material)));
    if let Some(aabb) = aabb {
        e.insert(aabb);
    }
}
