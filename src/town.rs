//! The walled town's models: the wall, its towers, stairs and gatehouse,
//! and the four houses, placed from town_layout.rs's list on the drawn
//! ground. Static, one distance level.
//! The nine GLBs reference three shared atlases by file name
//! (tools/blender/town/promote_assets.py), so each atlas is decoded once
//! and one material per atlas serves every instance, which Bevy batches.
//! Loaded the first time a map with the wall is shown, so other maps pay
//! nothing.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use bevy::asset::RenderAssetUsages;
use bevy::image::{CompressedImageFormats, ImageAddressMode, ImageSampler, ImageSamplerDescriptor, ImageType};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;

use crate::terrain::{MapKind, Terrain};
use crate::town_layout::{House, Kind};

/// The models' `KHR_materials_specular` factor is 0.4, so F0 = 0.04 x
/// 0.4. Bevy's reflectance r gives F0 = 0.16 r^2.
const REFLECTANCE: f32 = 0.316;

pub struct TownPlugin;

impl Plugin for TownPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TownModels>()
            .add_systems(Update, respawn_town.after(crate::terrain::MapRebuild));
    }
}

/// The root entity of one placed piece.
#[derive(Component)]
struct TownPiece;

/// One drawn part of a model: a node of its GLB.
struct Part {
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    transform: Transform,
}

/// The decoded models by kind, empty until the map is first shown.
#[derive(Resource, Default)]
struct TownModels(HashMap<Kind, Vec<Part>>);

const FILES: [(Kind, &str); 9] = [
    (Kind::Wall, "wall_unit.glb"),
    (Kind::Landing, "wall_unit_landing.glb"),
    (Kind::Stair, "stair_ramp.glb"),
    (Kind::Tower, "drum_tower.glb"),
    (Kind::Gatehouse, "gatehouse.glb"),
    (Kind::House(House::Humble), "humble_house.glb"),
    (Kind::House(House::Market), "market_house.glb"),
    (Kind::House(House::Workshop), "workshop_house.glb"),
    (Kind::House(House::Merchant), "merchant_house.glb"),
];

/// The pieces follow the terrain's map: shown on the maps with the wall
/// (the walled town's houses only there), gone on every other.
#[allow(clippy::too_many_arguments)]
fn respawn_town(
    mut seen: Local<Option<MapKind>>,
    old: Query<Entity, With<TownPiece>>,
    mut commands: Commands,
    terrain: Res<Terrain>,
    mut models: ResMut<TownModels>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    if *seen == Some(terrain.kind) {
        return;
    }
    *seen = Some(terrain.kind);
    for e in &old {
        commands.entity(e).despawn();
    }
    if !terrain.kind.has_wall() {
        return;
    }
    if models.0.is_empty() {
        let t0 = Instant::now();
        match load_models(&mut meshes, &mut materials, &mut images) {
            Ok(loaded) => models.0 = loaded,
            Err(e) => {
                error!("town: cannot load the models: {e}");
                return;
            }
        }
        info!("town: models loaded in {:.0} ms", t0.elapsed().as_secs_f32() * 1000.0);
    }
    let pieces = crate::town_layout::pieces(terrain.kind);
    for piece in &pieces {
        let Some(parts) = models.0.get(&piece.kind) else { continue };
        let ground = terrain.ground_at(piece.pos.x, piece.pos.y);
        commands
            .spawn((
                Transform::from_xyz(piece.pos.x, ground, piece.pos.y)
                    .with_rotation(Quat::from_rotation_y(piece.yaw())),
                Visibility::Inherited,
                TownPiece,
            ))
            .with_children(|parent| {
                for part in parts {
                    parent.spawn((
                        Mesh3d(part.mesh.clone()),
                        MeshMaterial3d(part.material.clone()),
                        part.transform,
                    ));
                }
            });
    }
    info!("town: {} pieces placed", pieces.len());
}

/// Decode every model and its atlases. Nothing is published unless all
/// of them read, so a failed import leaves no assets behind.
fn load_models(
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
) -> Result<HashMap<Kind, Vec<Part>>, String> {
    let dir = crate::util::game_root().join("assets/town");
    let mut atlases: HashMap<String, Image> = HashMap::new();
    let mut decoded = Vec::new();
    for (kind, file) in FILES {
        let path = dir.join(file);
        let parts = read_model(&path, &dir, &mut atlases).map_err(|e| format!("{}: {e}", path.display()))?;
        decoded.push((kind, parts));
    }
    // One material per atlas and roughness, shared by every instance.
    let textures: HashMap<String, Handle<Image>> =
        atlases.into_iter().map(|(name, image)| (name, images.add(image))).collect();
    let mut shared: HashMap<(String, u32), Handle<StandardMaterial>> = HashMap::new();
    let mut out = HashMap::new();
    for (kind, parts) in decoded {
        let parts = parts
            .into_iter()
            .map(|(mesh, atlas, roughness, transform)| {
                let material = shared
                    .entry((atlas.clone(), roughness.to_bits()))
                    .or_insert_with(|| {
                        materials.add(StandardMaterial {
                            base_color_texture: Some(textures[&atlas].clone()),
                            perceptual_roughness: roughness,
                            metallic: 0.0,
                            reflectance: REFLECTANCE,
                            ..default()
                        })
                    })
                    .clone();
                Part { mesh: meshes.add(mesh), material, transform }
            })
            .collect();
        out.insert(kind, parts);
    }
    Ok(out)
}

type ModelPart = (Mesh, String, f32, Transform);

/// One GLB's parts: each node's mesh with its atlas's file name, its
/// roughness and the node's transform as exported (the gatehouse's gates
/// carry their pivots there).
fn read_model(path: &Path, dir: &Path, atlases: &mut HashMap<String, Image>) -> Result<Vec<ModelPart>, String> {
    let bytes = crate::game_files::read(path).map_err(|e| e.to_string())?;
    let glb = gltf::Gltf::from_slice(&bytes).map_err(|e| e.to_string())?;
    let blob = glb.blob.as_deref().ok_or("missing binary buffer")?;
    let scene = glb.default_scene().or_else(|| glb.scenes().next()).ok_or("no scene")?;
    let mut parts = Vec::new();
    let mut stack: Vec<_> = scene.nodes().map(|n| (n, Mat4::IDENTITY)).collect();
    while let Some((node, parent)) = stack.pop() {
        let matrix = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        stack.extend(node.children().map(|c| (c, matrix)));
        let Some(mesh) = node.mesh() else { continue };
        if node.skin().is_some() {
            return Err("town models must carry no skin".into());
        }
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                return Err("expected triangles".into());
            }
            let reader = primitive.reader(|buffer| (buffer.index() == 0).then_some(blob));
            let positions: Vec<_> = reader.read_positions().ok_or("missing positions")?.collect();
            let normals: Vec<_> = reader.read_normals().ok_or("missing normals")?.collect();
            let uv: Vec<_> = reader.read_tex_coords(0).ok_or("missing UV0")?.into_f32().collect();
            let indices: Vec<_> = reader.read_indices().ok_or("missing indices")?.into_u32().collect();
            let count = positions.len();
            if count == 0
                || normals.len() != count
                || uv.len() != count
                || indices.is_empty()
                || !indices.len().is_multiple_of(3)
                || indices.iter().any(|&i| i as usize >= count)
            {
                return Err(format!("{} has invalid geometry", mesh.name().unwrap_or("a mesh")));
            }
            let material = primitive.material();
            let texture = material.pbr_metallic_roughness().base_color_texture().ok_or("missing atlas")?;
            if texture.tex_coord() != 0 {
                return Err("atlas must use UV0".into());
            }
            let gltf::image::Source::Uri { uri, .. } = texture.texture().source().source() else {
                return Err("atlas must be a separate file (tools/blender/town/promote_assets.py)".into());
            };
            if !atlases.contains_key(uri) {
                atlases.insert(uri.to_string(), read_atlas(&dir.join(uri))?);
            }
            let roughness = material.pbr_metallic_roughness().roughness_factor();
            let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
                .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
                .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uv)
                .with_inserted_indices(Indices::U32(indices));
            parts.push((mesh, uri.to_string(), roughness, Transform::from_matrix(matrix)));
        }
    }
    Ok(parts)
}

/// An sRGB base colour atlas with its mip chain built on the CPU.
fn read_atlas(path: &Path) -> Result<Image, String> {
    let bytes = crate::game_files::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut image = Image::from_buffer(
        &bytes,
        ImageType::Extension("png"),
        CompressedImageFormats::NONE,
        true,
        ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::ClampToEdge,
            address_mode_v: ImageAddressMode::ClampToEdge,
            anisotropy_clamp: 8,
            ..ImageSamplerDescriptor::linear()
        }),
        RenderAssetUsages::RENDER_WORLD,
    )
    .map_err(|e| format!("{}: {e}", path.display()))?;
    if image.texture_descriptor.format != TextureFormat::Rgba8UnormSrgb {
        return Err(format!("{}: atlas must be 8-bit colour", path.display()));
    }
    crate::vegetation::opaque_mips(&mut image)?;
    Ok(image)
}
