//! Imported soldier models: one glTF file per unit kind, turned into the
//! same four-level mesh set `unit_meshes` builds in code.
//!
//! The file contract is docs/plans/unit-asset-spec.md: metres, origin on
//! the ground, +Y up, +Z forward, one node per level named `L0` to `L3`.
//! Per vertex, `TEXCOORD_1` holds (part id, pivot height) and `COLOR_0`
//! holds the material colour with the team colour amount in alpha. Those
//! are the channels the vertex shader already reads.
//!
//! On import the model is scaled to `2 * half_height` and dropped so the
//! feet sit at `-half_height`. It is authored anatomically, weapon in the
//! right hand on -X and shield in the left on +X, which is the engine's
//! convention too.
//!
//! A model may carry one base colour texture, an atlas read through
//! `TEXCOORD_0`. Its rgb is the colour and its alpha the team tint mask,
//! and the vertex colour of a textured model is white. The atlas gets a
//! mip chain at load and the shader tints it per pixel.
//!
//! Levels the file lacks are derived from its finest level: each part is
//! cut into slabs along its longest axis and each slab becomes one box.
//! Box colours are weighted by what the battle camera sees, because an
//! area average lets the hidden mail under a surcoat turn a blue
//! regiment grey. On a textured model they come from the atlas.
//!
//! A weapon arm split into upper arm, forearm and hand is posed by
//! attack tables that ship next to the model, `<kind>.arm_stab.json`,
//! which the build scripts in tools/blender write. (`<kind>.stab.json` is
//! a skeleton's stab clip.) Without them the arm bends
//! at the elbow as one piece. An archer with jointed arms, a strung bow,
//! a held arrow and a separate head and torso shoots from the tables in
//! `<kind>.shoot.json`. Without them the parts join the arm, bow arm or
//! body they belong to, and the archer shoots as a rigid figure.
//!
//! A model with a glTF skin and the clips of `SKEL_CLIPS` next to it
//! (`<kind>.<clip>.json`, the one clip format of docs/plans/022) is posed
//! by its skeleton at the near levels (render_units_gpu.rs `SKEL_LODS`)
//! unless `FL_CLIPS=0`: every vertex follows a joint pair
//! (`ATTRIBUTE_SKIN`), and the pose pass plays the clips (unit_pose.wgsl
//! `put_skeleton`). At the far levels, and everywhere with `FL_CLIPS=0`,
//! the same file draws by its parts. An archer's skin and bow rig work
//! together: his near levels shoot from the clips, his far levels from
//! the shot tables. A clip's tracks go to the model's joints by name, so
//! a kind plays another's clips on the joints the two share: the
//! man-at-arms has the knight's skeleton and plays the knight's clips,
//! and the archer plays them for the slots his own files leave empty
//! (`CLIP_KIND`).
//!
//! `FL_UNIT_MESH=code` keeps the code-built meshes and `FL_GLB_FAR=code`
//! fills only the missing levels from them. `FL_GLB_<KIND>=path` loads
//! another file for one kind, for example `FL_GLB_KNIGHT=path/to/knight.glb`.

use std::path::{Path, PathBuf};

use bevy::asset::RenderAssetUsages;
use bevy::image::{
    CompressedImageFormats, ImageFilterMode, ImageSampler, ImageSamplerDescriptor, ImageType,
};
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;

use crate::render_units::NUM_LODS;
use crate::unit_meshes::{MeshBuf, Rig, SKEL_CLIP_SLOTS, SKEL_JOINTS, SKEL_NO_JOINT, Skel, blend};
use crate::unit_types::NUM_KINDS;

type Fallible<T> = Result<T, String>;

/// One file per kind, named as the asset spec names them.
const KIND_FILE: [&str; NUM_KINDS] = ["knight", "man_at_arms", "spearman", "archer"];

/// The kind whose clips a skinned kind plays where he has no file of his
/// own, read from `<kind>.<clip>.json` beside his model: the man-at-arms
/// is built on the knight's body and skeleton, and the archer's body
/// joints are the knight's by name, so both play the knight's clips.
const CLIP_KIND: [usize; NUM_KINDS] = [0, 0, 2, 0];

/// Part id to the name its pivot empty carries (`pivot_<part>`), in the
/// order unit_meshes.rs and the vertex shader use. Id 7 is the arrow in
/// flight, which no model carries. Id 20 is the archer's knife, which
/// moves with the hand that holds it and shows only while it is out
/// (unit_pose.wgsl `put_away`).
const PART_NAMES: [&str; 21] = [
    "body",
    "arm_weapon",
    "leg_l",
    "leg_r",
    "arm_spear",
    "arm_shield",
    "arm_bow",
    "",
    "weapon",
    "forearm_spear",
    "hand_spear",
    "forearm_bow",
    "hand_bow",
    "bow_string",
    "bow_upper",
    "bow_lower",
    "arrow",
    "bow_string_lower",
    "head",
    "torso",
    "knife",
];

/// Other names a part goes by: an archer's drawing forearm and hand take
/// the ids of a spearman's.
const PART_ALIASES: [(&str, usize); 2] = [("forearm_draw", 9), ("hand_draw", 10)];

/// The part id a `pivot_<name>` empty belongs to.
fn part_id(name: &str) -> Option<usize> {
    PART_NAMES
        .iter()
        .position(|q| !q.is_empty() && *q == name)
        .or_else(|| PART_ALIASES.iter().find(|(n, _)| *n == name).map(|(_, id)| *id))
}

/// The held weapon's part id (unit_meshes.rs PART_WEAPON).
const WEAPON: usize = 8;
/// A jointed weapon arm's forearm and hand. Its upper arm keeps the arm's
/// id.
const FOREARM: usize = 9;
const HAND: usize = 10;

/// Triangle budget per level (docs/plans/unit-asset-spec.md). Only used
/// to flag a level that will cost more than the plan assumed.
const TRI_BUDGET: [usize; NUM_LODS] = [3_000, 800, 250, 60];

/// Slabs a derived level cuts each part into: the body carries the head
/// and the headgear, so it gets more than a limb does. The last level is
/// all one part (the spec: "L3 is all body"), cut into `.0` slabs.
const SLABS: [(usize, usize); NUM_LODS] = [(0, 0), (8, 4), (3, 2), (3, 0)];

/// Smallest half-extent of a derived box, in engine units. A shield plate
/// or a blade is thin, and a box with no thickness at all drops out of
/// the silhouette at some angles.
const MIN_HALF: f32 = 0.005;

/// A kind's meshes, the texture atlas its imported levels sample, its
/// rig, and its clip buffer: the shot tables of an archer's bow rig, then
/// a skeleton's clips (empty without either).
pub struct KindMeshes {
    pub lods: [Mesh; NUM_LODS],
    /// The model's base colour atlas: sRGB colour, alpha the team tint
    /// mask, mip chain included.
    pub atlas: Option<Image>,
    /// Levels that sample the atlas. Derived and code-built levels carry
    /// their colour per vertex.
    pub textured: [bool; NUM_LODS],
    pub rig: Rig,
    pub clips: Vec<[f32; 4]>,
    /// The last level is the light one (`light_last_level`), and
    /// `cast_last` is the file's own last level, which its shadow casters
    /// keep.
    pub light_last: bool,
    pub cast_last: Option<Mesh>,
    /// `FL_SPRITE`: the last level is a card (`card_level`) that shows the
    /// kind's picture in `sprite`, and `fallen_last` is the file's own last
    /// level, which the dying and the fallen at that level keep.
    pub sprite: Option<Image>,
    pub fallen_last: Option<Mesh>,
    /// `FL_SPRITE`: the card's table of the sheet's layout (`sprite_table`).
    pub sprite_table: Option<Vec<[f32; 4]>>,
}

impl KindMeshes {
    fn code_built(kind: usize) -> Self {
        Self {
            lods: crate::unit_meshes::build_kind_lods(kind),
            atlas: None,
            textured: [false; NUM_LODS],
            rig: crate::unit_meshes::code_rig(kind),
            clips: Vec::new(),
            light_last: false,
            cast_last: None,
            sprite: None,
            fallen_last: None,
            sprite_table: None,
        }
    }
}

/// The arrow in flight and in the ground: `arrow.glb` when it is there,
/// else the code-built arrow. The model is scaled to the code-built
/// arrow's length, which is longer than a real arrow beside a soldier so
/// a volley reads at battle zoom.
pub fn arrow_mesh() -> Mesh {
    let code = crate::unit_meshes::build_arrow();
    let code_only = std::env::var("FL_UNIT_MESH").is_ok_and(|v| v == "code");
    let Some(path) = (!code_only).then(|| model_file("arrow")).flatten() else {
        return code;
    };
    let length = mesh_length(&code);
    match import_arrow(&path, length) {
        Ok(mesh) => {
            info!("arrow: imported {}, scaled to {length:.2} long", path.display());
            mesh
        }
        Err(e) => {
            error!("{} is not usable, the code-built arrow stands in: {e}", path.display());
            code
        }
    }
}

/// A mesh's extent along +Z, the arrow's flight axis.
fn mesh_length(mesh: &Mesh) -> f32 {
    use bevy::mesh::VertexAttributeValues as V;
    let Some(V::Float32x3(pos)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else {
        return 0.0;
    };
    let (lo, hi) = pos.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p[2]), hi.max(p[2])));
    hi - lo
}

/// Read the arrow's `L0`: one rigid part, id 7, vertex coloured, metres
/// along +Z, origin at the shaft's middle. Scaled to `length`.
fn import_arrow(path: &Path, length: f32) -> Fallible<Mesh> {
    let bytes = crate::game_files::read(path).map_err(|e| format!("cannot read it: {e}"))?;
    let gltf = gltf::Gltf::from_slice(&bytes).map_err(|e| format!("not valid glTF: {e}"))?;
    let blob = gltf.blob.as_deref();
    let scene = gltf
        .document
        .default_scene()
        .or_else(|| gltf.document.scenes().next())
        .ok_or("the file has no scene")?;
    let mut level = Level::default();
    let mut stack: Vec<(gltf::Node, Mat4)> = scene.nodes().map(|n| (n, Mat4::IDENTITY)).collect();
    while let Some((node, parent)) = stack.pop() {
        let xf = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        for child in node.children() {
            stack.push((child, xf));
        }
        if node.name() != Some("L0") {
            continue;
        }
        let Some(mesh) = node.mesh() else { continue };
        for prim in mesh.primitives() {
            read_primitive(&mut level, &prim, blob, xf, None)?;
        }
    }
    if level.pos.is_empty() {
        return Err("no node named L0".into());
    }
    if level.part.iter().any(|p| (p - 7.0).abs() > 1e-3) {
        return Err("a vertex carries a part id other than 7, the arrow's".into());
    }
    let (lo, hi) = level.pos.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p.z), hi.max(p.z)));
    if hi - lo <= 0.0 {
        return Err("the arrow has no length along +Z".into());
    }
    let scale = length / (hi - lo);
    for p in &mut level.pos {
        *p *= scale;
    }
    Ok(level_mesh(&level))
}

/// The mesh set a kind renders with: the imported model when its file
/// is there, the code-built set otherwise. `FL_UNIT_MESH=code` keeps the
/// code-built set either way, which is the A/B against an import. A file
/// that is there but unusable logs an error and falls back, so a battle
/// still runs on a half-exported model. Called once per kind at startup.
/// `pulled`: the kind draws by the GPU path (render_units_gpu.rs), the only
/// one that draws the light last level (`l3_light`).
pub fn kind_lods(kind: usize, pulled: bool) -> KindMeshes {
    let code_only = std::env::var("FL_UNIT_MESH").is_ok_and(|v| v == "code");
    let path = (!code_only).then(|| model_path(kind)).flatten();
    let Some(path) = path else {
        return KindMeshes::code_built(kind);
    };
    match import(kind, &path, pulled) {
        Ok(model) => model,
        Err(e) => {
            error!("{} is not usable, the code-built mesh stands in: {e}", path.display());
            KindMeshes::code_built(kind)
        }
    }
}

/// The model file of a kind: the file `FL_GLB_<KIND>` names, then the
/// shipped asset, then the working copy the asset track writes. Paths are
/// anchored at the crate root, the way the asset plugin anchors `assets/`.
fn model_path(kind: usize) -> Option<PathBuf> {
    model_file(KIND_FILE[kind])
}

/// A model file by name: the one `FL_GLB_<NAME>` names, then the shipped
/// asset, then the working copy.
fn model_file(name: &str) -> Option<PathBuf> {
    let root = crate::util::game_root();
    let var = format!("FL_GLB_{}", name.to_uppercase());
    if let Ok(named) = std::env::var(&var) {
        let named = root.join(named);
        if crate::game_files::is_file(&named) {
            return Some(named);
        }
        warn!("{var}: {} is not a file", named.display());
    }
    let shipped = root.join("assets/units").join(format!("{name}.glb"));
    if crate::game_files::is_file(&shipped) {
        return Some(shipped);
    }
    let working = root.join("assets_dev").join(name).join(format!("{name}.glb"));
    crate::game_files::is_file(&working).then_some(working)
}

/// One level's geometry, in model space while it is read and in engine
/// local space after `to_local`.
#[derive(Default, Clone)]
struct Level {
    pos: Vec<Vec3>,
    nrm: Vec<Vec3>,
    part: Vec<f32>,
    pivot: Vec<f32>,
    col: Vec<[f32; 4]>,
    /// Atlas coordinates, zero on an untextured model.
    tex: Vec<[f32; 2]>,
    /// Per vertex of a skinned model, the two joints it follows and the
    /// weight of the second (`JOINTS_0`, `WEIGHTS_0`). Empty when a
    /// primitive of the level has no skin.
    skin: Vec<(u16, u16, f32)>,
    idx: Vec<u32>,
}

impl Level {
    fn tris(&self) -> usize {
        self.idx.len() / 3
    }

    /// Ground line and height, for the scale and the sanity checks. The
    /// height is the top of the trunk (`is_trunk`), not of whatever the
    /// soldier holds above his head: a spearman's spear stands 0.7 m
    /// higher.
    fn y_range(&self) -> (f32, f32) {
        let ground = self.pos.iter().fold(f32::MAX, |lo, p| lo.min(p.y));
        let top = |body: bool| {
            self.pos
                .iter()
                .zip(&self.part)
                .filter(|(_, part)| !body || is_trunk(**part))
                .fold(f32::MIN, |hi, (p, _)| hi.max(p.y))
        };
        let body = top(true);
        (ground, if body > f32::MIN { body } else { top(false) })
    }
}

/// Read a model and hand back its four levels, engine local space, ready
/// to hand to `Mesh3d`, with its atlas.
fn import(kind: usize, path: &Path, pulled: bool) -> Fallible<KindMeshes> {
    let bytes = crate::game_files::read(path).map_err(|e| format!("cannot read it: {e}"))?;
    let gltf = gltf::Gltf::from_slice(&bytes).map_err(|e| format!("not valid glTF: {e}"))?;
    let blob = gltf.blob.as_deref();
    let atlas = read_atlas(&gltf, blob, path)?;
    let atlas_image = atlas.as_ref().map(|(image, _)| *image);

    // Level nodes and pivot empties, with the parent chain applied. A
    // stock Blender export leaves them at the scene root, but a model
    // nested under an empty has to arrive in the same place.
    let mut levels: [Option<Level>; NUM_LODS] = Default::default();
    let mut pivot_nodes: Vec<(usize, f32)> = Vec::new();
    // Pivot and joint empties by name, in model space, for the arm rig.
    let mut empties: Vec<(String, Vec3)> = Vec::new();
    let scene = gltf
        .document
        .default_scene()
        .or_else(|| gltf.document.scenes().next())
        .ok_or("the file has no scene")?;
    // Every node's origin in model space, by node index, for the skeleton.
    let mut origins: Vec<Vec3> = vec![Vec3::ZERO; gltf.document.nodes().len()];
    let mut stack: Vec<(gltf::Node, Mat4)> = scene.nodes().map(|n| (n, Mat4::IDENTITY)).collect();
    while let Some((node, parent)) = stack.pop() {
        let xf = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        for child in node.children() {
            stack.push((child, xf));
        }
        origins[node.index()] = xf.w_axis.truncate();
        let Some(name) = node.name() else { continue };
        if let Some(part) = name
            .strip_prefix("pivot_")
            .and_then(part_id)
        {
            pivot_nodes.push((part, xf.w_axis.y));
        }
        if name.starts_with("pivot_") || name.starts_with("joint_") {
            empties.push((name.to_string(), xf.w_axis.truncate()));
        }
        let Some(lod) = level_index(name) else { continue };
        let Some(mesh) = node.mesh() else { continue };
        let level = levels[lod].get_or_insert_with(Level::default);
        for prim in mesh.primitives() {
            let tex_set =
                atlas_set(&prim, atlas_image).map_err(|e| format!("node {name}: {e}"))?;
            read_primitive(level, &prim, blob, xf, tex_set)
                .map_err(|e| format!("node {name}: {e}"))?;
        }
    }

    let finest = levels.iter().flatten().next().ok_or("no node named L0, L1, L2 or L3")?;
    let (ground, top) = finest.y_range();
    if top <= 0.0 {
        return Err("the model has no height".into());
    }
    let name = KIND_FILE[kind];
    if ground.abs() > 0.02 {
        warn!("{name}: the model's feet are at {ground:.3} m, the spec puts the origin on the ground");
    }
    if (top - 1.8).abs() > 0.2 {
        warn!("{name}: the model is {top:.2} m tall, the spec authors at 1.80 m");
    }
    check_parts(finest, &pivot_nodes, name);

    // True human metres to engine units: the soldier is 2 * half_height
    // tall and stands on the terrain at half_height.
    let half_height = crate::unit_types::half_height(kind);
    let scale = 2.0 * half_height / top;
    for level in levels.iter_mut().flatten() {
        to_local(level, scale, half_height);
    }
    let local = |p: Vec3| [scale * p.y - half_height, scale * p.z];
    let local3 = |p: Vec3| Vec3::new(scale * p.x, scale * p.y - half_height, scale * p.z);
    let (mut rig, mut clips) = match bow_rig(&mut levels, &empties, local3, scale, path, name) {
        Some(bow) => bow,
        None => (arm_rig(&mut levels, &empties, local, scale, path, name), Vec::new()),
    };
    // A skeleton's clips follow a bow rig's shots in the clip buffer.
    let skinned = clips_on() && gltf.document.skins().next().is_some();
    if skinned {
        let bow = rig.bow.params[3] > 0.5;
        match read_skeleton(&gltf, &origins, &mut levels, local3, scale, path, kind, bow) {
            Ok((mut skel, rows)) => {
                info!(
                    "{name}: skeleton of {} joints, clips {:?} ({} rows)",
                    skel.params[0],
                    SKEL_CLIPS,
                    rows.len()
                );
                for clip in &mut skel.clips {
                    clip[0] += clips.len() as u32;
                }
                rig.skel = skel;
                clips.extend(rows);
            }
            Err(e) => warn!("{name}: its skeleton is not usable, it draws by its parts: {e}"),
        }
    }
    if rig.skel.params[0] == 0 {
        for level in levels.iter_mut().flatten() {
            level.skin.clear();
        }
    }

    let imported: Vec<usize> = (0..NUM_LODS).filter(|l| levels[*l].is_some()).collect();
    let tris: Vec<usize> = levels.iter().map(|l| l.as_ref().map_or(0, Level::tris)).collect();
    for (lod, level) in levels.iter().enumerate() {
        if let Some(level) = level
            && level.tris() > TRI_BUDGET[lod]
        {
            warn!("{name} L{lod}: {} tris is over the {} budget", level.tris(), TRI_BUDGET[lod]);
        }
    }
    info!(
        "{name}: imported {} levels {imported:?} with {tris:?} tris, scaled {scale:.3} to {:.2} units",
        path.display(),
        2.0 * half_height,
    );

    // Levels the file does not carry are derived from the finest one it
    // does. That level is cloned because the array below empties `levels`
    // as it builds.
    let mut source = levels.iter().flatten().next().cloned().expect("checked above");
    let mut textured: [bool; NUM_LODS] =
        std::array::from_fn(|l| atlas.is_some() && levels[l].is_some());
    let atlas = atlas.map(|(_, image)| image);
    if let Some(image) = &atlas {
        // A textured model's vertex colour is white. The atlas supplies the
        // colour of the imported levels in the shader, and of the derived
        // levels here.
        source.col = atlas_vertex_colors(&source, image);
        for level in levels.iter_mut().flatten() {
            level.col.fill([1.0, 1.0, 1.0, 0.0]);
        }
        let size = image.texture_descriptor.size;
        let mips = image.texture_descriptor.mip_level_count;
        info!("{name}: atlas {}x{} with {mips} mip levels", size.width, size.height);
    }
    // FL_L3_LIGHT=1: the file's last level is drawn as three boxes coloured
    // from it, and keeps only its shadow casters.
    let last = NUM_LODS - 1;
    let mut cast_last = None;
    if l3_light() && !pulled && kind == 0 {
        warn!("FL_L3_LIGHT: the light last level draws only on the GPU path; drawing the file's last level");
    }
    // FL_SPRITE: the file's last level becomes a card, and keeps only its
    // shadow casters and the dying and the fallen.
    let mut sprite = None;
    let mut sprite_rows = None;
    let mut fallen_last = None;
    if sprites() && pulled && levels[last].is_some() {
        match sprite_table(kind, rig.bow.params[3] > 0.5).and_then(|rows| Ok((rows, sprite_sheet()?))) {
            Ok((rows, image)) => {
                sprite_rows = Some(rows);
                sprite = Some(image);
            }
            Err(e) => error!("FL_SPRITE: {e}; drawing the file's last level"),
        }
    }
    if sprite.is_some() && let Some(file) = levels[last].take() {
        levels[last] = Some(card_level());
        cast_last = Some(level_mesh(&file));
        fallen_last = Some(level_mesh(&file));
        info!("{name}: the last level is a card (FL_SPRITE)");
    }
    let light_last = l3_light() && sprite.is_none() && pulled && levels[last].is_some();
    if light_last && let Some(file) = levels[last].take() {
        levels[last] = Some(light_last_level(&file, atlas.as_ref(), kind, half_height, name));
        cast_last = Some(level_mesh(&file));
        textured[last] = false;
    }
    let weights = visible_weights(&source);
    let code_far = std::env::var("FL_GLB_FAR").is_ok_and(|v| v == "code");
    let mut code = code_far.then(|| crate::unit_meshes::build_kind_lods(kind).map(Some));
    let lods = std::array::from_fn(|lod| match levels[lod].take() {
        Some(level) => level_mesh(&level),
        None => match code.as_mut().and_then(|c| c[lod].take()) {
            Some(mesh) => mesh,
            None => derive_level(&source, &weights, lod),
        },
    });
    Ok(KindMeshes {
        lods,
        atlas,
        textured,
        rig,
        clips,
        light_last,
        cast_last,
        sprite,
        sprite_table: sprite_rows,
        fallen_last,
    })
}

/// `FL_SPRITE=1`: every standing soldier at the last level is drawn as a
/// card (off by default). It covers a pixel as the meshes do under the
/// view's anti-aliasing: with MSAA the samples its picture covers, without
/// it the pixels whose centre it covers (unit_instancing.wgsl
/// `SpriteFragment`). GPU path only. Read once, at import. The last
/// level then begins `FL_SPRITE_FROM` metres from the camera, 400 by
/// default (render_units.rs `LodConfig::sprite_from`), unless `FL_LOD_PX`
/// is given, which sets every switch in pixels as without the card.
pub fn sprites() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::util::env_or("FL_SPRITE", 0_u32) != 0)
}

/// The sprite sheet's layout (assets/units/sprites.json, written with the
/// sheet by tools/sprite_sheet.py). Per kind one cell per (camera height,
/// direction around the man, frame), each just large enough for him.
#[derive(serde::Deserialize)]
struct SpriteLayout {
    width: u32,
    height: u32,
    /// Camera heights above the horizon the cells are drawn from, degrees.
    elevations: Vec<f32>,
    /// Directions around the man, evenly spaced from his front.
    directions: u32,
    /// The family names, in the order of the shader's ids
    /// (`SPRITE_FAMILIES`).
    families: Vec<String>,
    speeds: SpriteSpeeds,
    kinds: Vec<SpriteKind>,
}

#[derive(serde::Deserialize)]
struct SpriteSpeeds {
    /// Below this ground speed, m/s, a man shows his standing frames.
    stand_below: f32,
    /// The ground speeds, m/s, at which the slow walk switches to the
    /// walk, the walk to the jog and the jog to the run.
    switch: [f32; 3],
}

#[derive(serde::Deserialize)]
struct SpriteKind {
    name: String,
    /// Frames per view.
    frames: u32,
    /// Per family, its first frame and its frame count.
    families: Vec<[u32; 2]>,
    /// The team cloth's hue, the cloth's mean colour over its grey with the
    /// largest channel 1: the shader tints the cloth's grey by the team
    /// colour times this.
    cloth: [f32; 3],
    /// Per cell, in the order (camera height, direction, frame): its rect
    /// in the sheet's left half in pixels (x, y, width, height), and what
    /// it shows in metres from his centre, from x0 to x1 along the
    /// camera's right and from y0 to y1 along its up.
    cells: Vec<[f32; 8]>,
}

/// The shader's family ids (unit_instancing.wgsl `sprite_frame`).
const SPRITE_FAMILIES: [&str; 23] = [
    "stand", "ready", "wall", "bowup", "stand_knife", "slow", "slow_level", "slow_knife", "walk", "walk_level",
    "walk_knife", "jog", "jog_level", "jog_knife", "run", "run_level", "run_knife", "stab", "swing", "shot",
    "stagger", "cheer", "charge",
];

/// Rows of a kind's sprite table before its cells (`sprite_table`), an
/// even count so each cell's two rows fill one card vertex.
pub const SPRITE_HEAD_ROWS: usize = 18;

/// `FL_SPRITE`: per kind, the rows the card's vertex shader reads the
/// sheet's layout from (unit_instancing.wgsl `sprite_row`): 0 frames per
/// view, directions and camera heights, 1 the camera heights in radians,
/// 2 the speed below which he stands and 1 for a bow kind, 3 the cloth's
/// hue, 4 the speeds the gaits switch at, 5 to 16 two families a row, each
/// its first frame and its frame count, 17 unused; then two rows a cell,
/// its rect in uv (u, v of its top left corner, its width and height) and
/// its box in metres.
fn sprite_table(kind: usize, bow: bool) -> Fallible<Vec<[f32; 4]>> {
    let path = crate::util::game_root().join("assets/units/sprites.json");
    let bytes = crate::game_files::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let layout: SpriteLayout =
        serde_json::from_slice(&bytes).map_err(|e| format!("cannot parse {}: {e}", path.display()))?;
    if layout.families.iter().map(String::as_str).ne(SPRITE_FAMILIES) {
        return Err(format!("{} lists other families than the shader picks", path.display()));
    }
    if layout.elevations.is_empty() || layout.elevations.len() > 4 || layout.directions == 0 {
        return Err(format!("{}: one to four camera heights and some directions", path.display()));
    }
    let k = layout
        .kinds
        .iter()
        .find(|k| k.name == KIND_FILE[kind])
        .ok_or(format!("{} has no {}", path.display(), KIND_FILE[kind]))?;
    let views = layout.elevations.len() * layout.directions as usize;
    if k.cells.len() != views * k.frames as usize || k.families.len() != SPRITE_FAMILIES.len() {
        return Err(format!("{}: {} has {} cells and {} families", path.display(), k.name, k.cells.len(), k.families.len()));
    }
    let (w, h) = (layout.width as f32, layout.height as f32);
    let mut elevations = [0.0; 4];
    for (e, deg) in elevations.iter_mut().zip(&layout.elevations) {
        *e = deg.to_radians();
    }
    let mut rows = vec![
        [k.frames as f32, layout.directions as f32, layout.elevations.len() as f32, 0.0],
        elevations,
        [layout.speeds.stand_below, if bow { 1.0 } else { 0.0 }, 0.0, 0.0],
        [k.cloth[0], k.cloth[1], k.cloth[2], 0.0],
        [layout.speeds.switch[0], layout.speeds.switch[1], layout.speeds.switch[2], 0.0],
    ];
    for pair in k.families.chunks(2) {
        let second = pair.get(1).copied().unwrap_or([0, 1]);
        rows.push([pair[0][0] as f32, pair[0][1] as f32, second[0] as f32, second[1] as f32]);
    }
    rows.resize(SPRITE_HEAD_ROWS, [0.0; 4]);
    for c in &k.cells {
        rows.push([c[0] / w, c[1] / h, c[2] / w, c[3] / h]);
        rows.push([c[4], c[5], c[6], c[7]]);
    }
    Ok(rows)
}

/// The sprite sheet (tools/sprite_sheet.py): rgb the colour outside the
/// team cloth and alpha the cut-out in its left half, and half a sheet to
/// the right of each texel its normal in rgb and the cloth's grey in
/// alpha. Read and given its mips as a model's atlas is.
fn sprite_sheet() -> Fallible<Image> {
    let path = crate::util::game_root().join("assets/units/sprites.png");
    let bytes = crate::game_files::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        mipmap_filter: ImageFilterMode::Linear,
        ..ImageSamplerDescriptor::linear()
    });
    let mut image = Image::from_buffer(
        &bytes,
        ImageType::MimeType("image/png"),
        CompressedImageFormats::NONE,
        true,
        sampler,
        RenderAssetUsages::RENDER_WORLD,
    )
    .map_err(|e| format!("cannot decode {}: {e}", path.display()))?;
    if image.texture_descriptor.format != TextureFormat::Rgba8UnormSrgb {
        return Err(format!("{} decodes to {:?}, it must be 8-bit RGBA", path.display(), image.texture_descriptor.format));
    }
    add_mips(&mut image)?;
    Ok(image)
}

/// The card: a unit square, x and y from 0 to 1, which the vertex shader
/// stretches over the cell's box and turns to face the camera about his
/// centre (unit_instancing.wgsl `sprite_output`). Every corner is on the
/// body, part 0.
fn card_level() -> Level {
    let corner = |x: f32, y: f32| Vec3::new(x, y, 0.0);
    Level {
        pos: vec![corner(0.0, 0.0), corner(1.0, 0.0), corner(1.0, 1.0), corner(0.0, 1.0)],
        nrm: vec![Vec3::Z; 4],
        part: vec![0.0; 4],
        pivot: vec![0.0; 4],
        col: vec![[1.0, 1.0, 1.0, 0.0]; 4],
        tex: vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        skin: Vec::new(),
        idx: vec![0, 1, 2, 0, 2, 3],
    }
}

/// `FL_L3_LIGHT=1`: every imported kind draws its last level from three
/// boxes (`light_last_level`), on the GPU path. Read once, at import.
pub fn l3_light() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::util::env_or("FL_L3_LIGHT", 0_u32) != 0)
}

/// Where the light last level splits head from body, as a share of the
/// soldier's height from the feet: the top of the shoulders in the files'
/// last levels (knight, man-at-arms, spearman 1.49 of 1.80 m, archer 1.46).
const NECK_SHARE: [f32; NUM_KINDS] = [0.83, 0.83, 0.83, 0.81];

/// Where the light last level splits the upper body from the lower, as a
/// share of the soldier's height: about the belt (the knight's 1.06 of
/// 1.80 m).
const WAIST_SHARE: f32 = 0.58;

/// Per kind, per box of the light last level (lower body, upper body,
/// head), the scales across and front to back that bring the box's
/// silhouettes seen from the front, the side and above nearest to those
/// of the file's last level in the box's height band, least squares on the
/// logs of the three areas, the silhouettes rasterized at 4 mm. The file's
/// silhouette counts every triangle up to the top of the head but the
/// archer's bow, string and arrows. Fitted once to the game's models: the
/// arms make the bodies wider than the trunk the boxes are measured from,
/// and the brimmed hats make the heads of the man-at-arms and the spearman
/// narrower than their brims.
const LIGHT_SCALES: [[[f32; 2]; 3]; NUM_KINDS] = [
    [[1.063, 1.331], [1.206, 1.201], [0.938, 0.938]],
    [[1.002, 1.368], [1.188, 1.170], [0.589, 0.587]],
    [[0.998, 1.365], [1.188, 1.178], [0.589, 0.587]],
    [[1.576, 2.322], [1.226, 1.108], [0.996, 0.987]],
];

/// The last level as 28 triangles: a lower body box, an upper body box and
/// a head box, built from the file's last level `file` at rest in engine
/// local space (feet at `-half_height`).
///
/// The boxes stand on the soldier's axis, each tapered between two
/// rectangles: the lower body from the feet to the waist (`WAIST_SHARE`),
/// the upper body from the waist to the neck (`NECK_SHARE`), the head from
/// the neck to the top of the head. A rectangle is the extent of the file's
/// vertices in a band of a tenth of his height there (the lowest tenth,
/// around the waist, below the neck, and above the neck for the head). Only
/// the trunk's vertices count (`is_trunk`), and each extent is the smaller
/// of the two sides of the axis, so an arm, a weapon or a bow held out to
/// one side does not shift the box; then each box is scaled by
/// `LIGHT_SCALES`. The archer's bow is not drawn at this level. The lower
/// box has no top or bottom face, the others no bottom face.
///
/// Each face takes the area-weighted mean of the colour the file's
/// triangles show where they face the same way: a triangle goes to the box
/// its centroid's height is in and to that box's face whose normal is
/// nearest its own, so each face shows its own band of the man. With an
/// atlas, the colour is the atlas sample at the triangle's atlas point,
/// rgb averaged in linear light and the team tint mask as it is. The
/// vertex colour holds the square root of the linear rgb and the mask
/// (unit_instancing.wgsl `UNIT_LIGHT_LEVEL`). Every corner is on the body,
/// part 0, with the pivot and skin of a body corner of the file, so the
/// level draws as the body alone.
fn light_last_level(file: &Level, atlas: Option<&Image>, kind: usize, half_height: f32, name: &str) -> Level {
    let height = 2.0 * half_height;
    let feet = -half_height;
    let neck = feet + NECK_SHARE[kind] * height;
    let waist = feet + WAIST_SHARE * height;
    let trunk: Vec<Vec3> = file
        .pos
        .iter()
        .zip(&file.part)
        .filter(|(p, part)| is_trunk(**part) && p.y <= half_height + 0.01 * height)
        .map(|(p, _)| *p)
        .collect();
    // Half the extent across and front to back, the smaller side of the
    // axis, of the trunk vertices between two heights.
    let extent = |lo: f32, hi: f32| -> Vec2 {
        let (mut min, mut max) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        for p in trunk.iter().filter(|p| p.y >= lo && p.y <= hi) {
            min = min.min(Vec2::new(p.x, p.z));
            max = max.max(Vec2::new(p.x, p.z));
        }
        if min.x > max.x {
            return Vec2::splat(0.05 * height);
        }
        (-min).min(max).max(Vec2::splat(0.01 * height))
    };
    let above = neck + 0.003 * height;
    let top = trunk.iter().filter(|p| p.y > above).fold(neck + 0.05 * height, |t, p| t.max(p.y));
    let [lower, upper, head] = LIGHT_SCALES[kind].map(Vec2::from);
    let foot = extent(feet, feet + 0.1 * height);
    let belt = extent(waist - 0.05 * height, waist + 0.05 * height);
    let shoulder = extent(neck - 0.1 * height, above);
    let crown = extent(above, f32::MAX);
    // Per box: the bottom height and half extents, the top height and half
    // extents, and whether it has a top face.
    let boxes = [
        (feet, foot * lower, waist, belt * lower, false),
        (waist, belt * upper, neck, shoulder * upper, true),
        (neck, crown * head, top, crown * head, true),
    ];
    info!(
        "{name}: light last level (engine units, across x front to back): lower body to {:.2}, {:.2} x {:.2} to \
         {:.2} x {:.2}; upper body to {:.2}, to {:.2} x {:.2}; head to {:.2}, {:.2} x {:.2}",
        waist - feet,
        2.0 * boxes[0].1.x,
        2.0 * boxes[0].1.y,
        2.0 * boxes[0].3.x,
        2.0 * boxes[0].3.y,
        neck - feet,
        2.0 * boxes[1].3.x,
        2.0 * boxes[1].3.y,
        top - feet,
        2.0 * boxes[2].1.x,
        2.0 * boxes[2].1.y,
    );
    // Faces: the top, then +x, -x, +z, -z, each four corners turned
    // outward, with the box they belong to.
    let mut faces: Vec<([Vec3; 4], Vec3, usize)> = Vec::new();
    for (k, &(y0, e0, y1, e1, has_top)) in boxes.iter().enumerate() {
        let b = |sx: f32, sz: f32| Vec3::new(sx * e0.x, y0, sz * e0.y);
        let t = |sx: f32, sz: f32| Vec3::new(sx * e1.x, y1, sz * e1.y);
        let centre = Vec3::new(0.0, (y0 + y1) / 2.0, 0.0);
        let roof = [t(-1.0, -1.0), t(1.0, -1.0), t(1.0, 1.0), t(-1.0, 1.0)];
        let sides = [
            [b(1.0, -1.0), b(1.0, 1.0), t(1.0, 1.0), t(1.0, -1.0)],
            [b(-1.0, 1.0), b(-1.0, -1.0), t(-1.0, -1.0), t(-1.0, 1.0)],
            [b(1.0, 1.0), b(-1.0, 1.0), t(-1.0, 1.0), t(1.0, 1.0)],
            [b(-1.0, -1.0), b(1.0, -1.0), t(1.0, -1.0), t(-1.0, -1.0)],
        ];
        for mut quad in has_top.then_some(roof).into_iter().chain(sides) {
            let mid = (quad[0] + quad[1] + quad[2] + quad[3]) / 4.0;
            if (quad[1] - quad[0]).cross(quad[2] - quad[0]).dot(mid - centre) < 0.0 {
                quad.reverse();
            }
            let n = (quad[1] - quad[0]).cross(quad[2] - quad[0]).normalize_or(Vec3::Y);
            faces.push((quad, n, k));
        }
    }
    // Area-weighted colour per face (linear rgb, mask), from the file's
    // triangles.
    let lut = srgb_to_linear_lut();
    let texel = |uv: Vec2| -> Option<[f32; 4]> {
        let image = atlas?;
        let size = image.texture_descriptor.size;
        let (w, h) = (size.width as usize, size.height as usize);
        let px = image.data.as_deref().filter(|d| d.len() >= w * h * 4)?;
        // One bilinear sample of the finest mip, as the GPU filters it.
        let x = uv.x * w as f32 - 0.5;
        let y = uv.y * h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |i: f32, n: usize| (i as i64).clamp(0, n as i64 - 1) as usize;
        let mut out = [0.0f32; 4];
        for (dx, dy, wt) in [
            (0.0, 0.0, (1.0 - fx) * (1.0 - fy)),
            (1.0, 0.0, fx * (1.0 - fy)),
            (0.0, 1.0, (1.0 - fx) * fy),
            (1.0, 1.0, fx * fy),
        ] {
            let p = &px[(at(y0 + dy, h) * w + at(x0 + dx, w)) * 4..][..4];
            for c in 0..3 {
                out[c] += wt * lut[p[c] as usize];
            }
            out[3] += wt * p[3] as f32 / 255.0;
        }
        Some(out)
    };
    let mut sums = vec![[0.0f32; 5]; faces.len()];
    for tri in file.idx.chunks_exact(3) {
        let [a, b, c] = [0, 1, 2].map(|k| file.pos[tri[k] as usize]);
        let cross = (b - a).cross(c - a);
        let area = cross.length() / 2.0;
        if area <= 0.0 {
            continue;
        }
        let normal = cross / (2.0 * area);
        let y = (a.y + b.y + c.y) / 3.0;
        let k = usize::from(y > waist) + usize::from(y > neck);
        let face = (0..faces.len())
            .filter(|&i| faces[i].2 == k)
            .max_by(|&i, &j| faces[i].1.dot(normal).total_cmp(&faces[j].1.dot(normal)))
            .expect("every box has faces");
        let uv = tri.iter().map(|&i| Vec2::from(file.tex[i as usize])).sum::<Vec2>() / 3.0;
        let colour = texel(uv).unwrap_or_else(|| {
            let mut c = [0.0f32; 4];
            for &i in tri {
                for (k, v) in file.col[i as usize].iter().enumerate() {
                    c[k] += v / 3.0;
                }
            }
            c
        });
        let s = &mut sums[face];
        for k in 0..4 {
            s[k] += area * colour[k];
        }
        s[4] += area;
    }
    // A face no triangle faced takes its box's mean, then the whole mean.
    let mean = |keep: &dyn Fn(usize) -> bool| -> [f32; 5] {
        let mut m = [0.0f32; 5];
        for (_, s) in sums.iter().enumerate().filter(|(i, _)| keep(*i)) {
            for k in 0..5 {
                m[k] += s[k];
            }
        }
        m
    };
    let whole = mean(&|_| true);
    let mut level = Level::default();
    let body = file.part.iter().position(|p| p.round() == 0.0).unwrap_or(0);
    let pivot = file.pivot.get(body).copied().unwrap_or(0.0);
    for (i, (quad, n, k)) in faces.iter().enumerate() {
        let boxed = mean(&|j| faces[j].2 == *k);
        let s = [sums[i], boxed, whole].into_iter().find(|s| s[4] > 0.0).unwrap_or([1.0, 1.0, 1.0, 0.0, 1.0]);
        let col = [(s[0] / s[4]).sqrt(), (s[1] / s[4]).sqrt(), (s[2] / s[4]).sqrt(), s[3] / s[4]];
        let first = level.pos.len() as u32;
        for p in quad {
            level.pos.push(*p);
            level.nrm.push(*n);
            level.part.push(0.0);
            level.pivot.push(pivot);
            level.col.push(col);
            level.tex.push([0.0, 0.0]);
            if let Some(skin) = file.skin.get(body) {
                level.skin.push(*skin);
            }
        }
        level.idx.extend([first, first + 1, first + 2, first, first + 2, first + 3]);
    }
    level
}

fn finest_local(levels: &[Option<Level>; NUM_LODS]) -> &Level {
    levels.iter().flatten().next().expect("checked by the caller")
}

/// The weapon arm's rig. A jointed arm needs its upper arm, forearm, hand
/// and weapon parts, their pivot empties and its attack tables. A bent arm
/// needs the `weapon` part, `pivot_weapon` and `joint_elbow`. Parts the
/// arm cannot move on their own join the arm, which then bends or turns
/// as one piece, and the log says why. `local` takes a model-space point
/// to the pitch plane of engine local space.
fn arm_rig(
    levels: &mut [Option<Level>; NUM_LODS],
    empties: &[(String, Vec3)],
    local: impl Fn(Vec3) -> [f32; 2],
    scale: f32,
    path: &Path,
    name: &str,
) -> Rig {
    let level = finest_local(levels);
    let has = |part: usize| level.part.iter().any(|p| p.round() as usize == part);
    let (arm, arm_name, hold) = if has(4) {
        (4, "arm_spear", crate::unit_meshes::HOLD_SPEAR)
    } else if has(1) {
        (1, "arm_weapon", crate::unit_meshes::HOLD_SWORD)
    } else {
        return Rig::default();
    };
    let find = |key: &str| empties.iter().find(|(n, _)| n == key).map(|(_, p)| *p);
    let jointed = has(FOREARM) || has(HAND);
    let chain = jointed.then(|| {
        let pivot = |part: usize| {
            find(&format!("pivot_{}", PART_NAMES[part]))
                .ok_or_else(|| format!("no pivot_{} empty", PART_NAMES[part]))
        };
        if !(has(FOREARM) && has(HAND) && has(WEAPON)) {
            return Err("the jointed arm lacks its forearm, hand or weapon part".to_string());
        }
        let joints = [pivot(arm)?, pivot(FOREARM)?, pivot(HAND)?, pivot(WEAPON)?];
        let attack = read_attack(&attack_path(path), &joints, [arm, FOREARM, HAND, WEAPON])?;
        let [shoulder, elbow, wrist, grip] = joints.map(&local);
        let (tip, rear) = weapon_axis(level, grip);
        Ok(Rig {
            shoulder,
            elbow,
            wrist,
            grip,
            tip,
            rear,
            arm: arm as f32,
            hold,
            slide: scale * attack.weapon_slide_levelled_m,
            chain: 1.0,
            windup: attack.windup,
            recover: attack.recover,
            ..Rig::default()
        })
    });
    match chain {
        Some(Ok(rig)) => {
            info!("{name}: jointed weapon arm, attack from {}", attack_path(path).display());
            return rig;
        }
        Some(Err(e)) => {
            warn!("{name}: {e}, so the forearm and hand join the arm and it bends at the elbow");
            fold(levels, &[FOREARM, HAND], arm);
        }
        None => {}
    }

    let level = finest_local(levels);
    let has_weapon = level.part.iter().any(|p| p.round() as usize == WEAPON);
    let (Some(shoulder), Some(elbow), Some(grip)) =
        (find(&format!("pivot_{arm_name}")), find("joint_elbow"), find("pivot_weapon"))
    else {
        if has_weapon {
            warn!("{name}: a weapon part without pivot_weapon and joint_elbow, the arm and weapon turn as one");
            fold(levels, &[WEAPON], arm);
        }
        return Rig::default();
    };
    if !has_weapon {
        warn!("{name}: no weapon part, the arm stays rigid");
        return Rig::default();
    }
    let grip = local(grip);
    let (tip, rear) = weapon_axis(level, grip);
    Rig {
        shoulder: local(shoulder),
        elbow: local(elbow),
        wrist: grip,
        grip,
        tip,
        rear,
        arm: arm as f32,
        hold,
        ..Rig::default()
    }
}

/// A model with a skeleton and its clips is posed by them; `FL_CLIPS=0`
/// draws it by its parts.
pub fn clips_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::util::env_or("FL_CLIPS", 1_u32) != 0)
}

/// The clips a skinned kind plays, by slot (unit_pose.wgsl `CLIP_STAND`
/// and on). Each is `<kind>.<name>.json` next to the model. The first
/// `SKEL_CLIPS_NEEDED` must be there; a missing later one leaves its slot
/// empty and the player does without it. Slots 10 to 13 are an archer's
/// shot: the draw from ready to the level aim, its release, the raise from
/// the level aim to the high one, and its release.
const SKEL_CLIPS: [&str; 19] = [
    "stand",
    "ready",
    "shuffle_left",
    "shuffle_right",
    "shuffle_forward",
    "shuffle_backward",
    "cut",
    "stab",
    "charge",
    "charge_attack",
    "draw",
    "release",
    "aim_up",
    "release_up",
    // The answers to a blow (sim/damage.rs SWING_DEFEND): the shield
    // block, the weapon parry.
    "block",
    "parry",
    // An archer's stand and ready with his knife out, which take the
    // place of his stand and ready while it is.
    "melee_stand",
    "melee_ready",
    // An archer's run with his knife out: the whole body, at the gait's
    // phase (unit_pose.wgsl CLIP_MELEE_RUN).
    "melee_run",
];
const SKEL_CLIPS_NEEDED: usize = 2;
/// The slot of the first step clip: left, right, forward and backward
/// follow in that order (unit_pose.wgsl CLIP_SHUFFLE_LEFT).
pub(crate) const SKEL_CLIP_STEPS: usize = 2;
/// The slots of the block and the parry (unit_pose.wgsl CLIP_BLOCK,
/// CLIP_PARRY).
pub(crate) const SKEL_CLIP_BLOCK: usize = 14;
pub(crate) const SKEL_CLIP_PARRY: usize = 15;
const _: () = assert!(SKEL_CLIPS[SKEL_CLIP_BLOCK].len() == 5 && SKEL_CLIPS[SKEL_CLIP_PARRY].len() == 5);
const _: () = assert!(SKEL_CLIPS.len() <= SKEL_CLIP_SLOTS);

/// The joints the player's gait, arm sway and attack write, in the order
/// of `Skel::roles`.
const SKEL_ROLES: [[&str; 4]; 6] = [
    ["pelvis", "chest", "head", "weapon"],
    ["upperarm_r", "forearm_r", "hand_r", "coat_r"],
    ["upperarm_l", "forearm_l", "hand_l", "coat_l"],
    ["thigh_r", "shin_r", "foot_r", ""],
    ["thigh_l", "shin_l", "foot_l", ""],
    ["arrow", "", "", ""],
];
/// Roles a skeleton may lack: an archer holds no weapon, and only he has
/// an arrow.
const SKEL_ROLES_OPTIONAL: [&str; 2] = ["weapon", "arrow"];

/// A clip in the one format (docs/plans/022): per sample the root's
/// offset from rest, then one quaternion per joint relative to its parent.
#[derive(serde::Deserialize)]
struct ClipFile {
    format: String,
    version: u32,
    joints: Vec<String>,
    fps: f32,
    #[serde(rename = "loop")]
    looped: bool,
    /// How far one play of the clip carries the soldier, model metres.
    #[serde(default)]
    travel: [f32; 3],
    #[serde(default)]
    events: ClipEvents,
    samples: Vec<Vec<f32>>,
}

#[derive(serde::Deserialize, Default)]
struct ClipEvents {
    /// When the blow lands, seconds from the clip's start.
    contact: Option<f32>,
    /// When a draw shows its arrow on the string.
    nock: Option<f32>,
    /// When a block's shield or a parry's blade is up to meet the blow.
    guard: Option<f32>,
}

/// The model's skeleton and its clips as rows of the clip buffer, and
/// every level's joint pairs put in parent then child order. Fails when
/// the skin, a level's weights or a clip file do not fit the player.
/// `bow`: the kind has a bow rig, so no weapon arm draws a blow.
#[allow(clippy::too_many_arguments)]
fn read_skeleton(
    gltf: &gltf::Gltf,
    origins: &[Vec3],
    levels: &mut [Option<Level>; NUM_LODS],
    local3: impl Fn(Vec3) -> Vec3,
    scale: f32,
    model: &Path,
    kind: usize,
    bow: bool,
) -> Fallible<(Skel, Vec<[f32; 4]>)> {
    let skin = gltf.document.skins().next().ok_or("no skin")?;
    let nodes: Vec<gltf::Node> = skin.joints().collect();
    let count = nodes.len();
    if count == 0 || count > SKEL_JOINTS {
        return Err(format!("{count} joints, the player poses up to {SKEL_JOINTS}"));
    }
    let names: Vec<String> = nodes.iter().map(|n| n.name().unwrap_or("").to_string()).collect();
    let mut parent = vec![-1_i32; count];
    for (i, node) in nodes.iter().enumerate() {
        for child in node.children() {
            if let Some(c) = nodes.iter().position(|n| n.index() == child.index()) {
                parent[c] = i as i32;
            }
        }
    }
    if parent.iter().enumerate().any(|(i, p)| *p >= i as i32) || parent.iter().filter(|p| **p < 0).count() != 1 {
        return Err("the joints are not one tree with parents first".into());
    }
    let mut skel = Skel::default();
    skel.params[0] = count as u32;
    for (i, node) in nodes.iter().enumerate() {
        let at = local3(origins[node.index()]);
        skel.joints[i] = [at.x, at.y, at.z, parent[i] as f32];
    }
    for (row, roles) in SKEL_ROLES.iter().enumerate() {
        for (k, role) in roles.iter().enumerate() {
            if role.is_empty() {
                continue;
            }
            skel.roles[row][k] = match names.iter().position(|n| n == role) {
                Some(j) => j as u32,
                None if SKEL_ROLES_OPTIONAL.contains(role) => SKEL_NO_JOINT,
                None => return Err(format!("no joint named {role}")),
            };
        }
    }
    // A vertex follows a joint and one of its children, the child second.
    for (lod, level) in levels.iter_mut().enumerate() {
        let Some(level) = level else { continue };
        if level.skin.is_empty() {
            return Err(format!("L{lod} has no JOINTS_0 and WEIGHTS_0"));
        }
        for v in &mut level.skin {
            let (a, b, w) = *v;
            if a as usize >= count || b as usize >= count {
                return Err(format!("L{lod}: a vertex names joint {} of {count}", a.max(b)));
            }
            *v = if w < 1e-4 {
                (a, a, 0.0)
            } else if parent[b as usize] == a as i32 {
                (a, b, w)
            } else if parent[a as usize] == b as i32 {
                (b, a, 1.0 - w)
            } else {
                return Err(format!(
                    "L{lod}: a vertex follows {} and {}, which are not parent and child",
                    names[a as usize], names[b as usize]
                ));
            };
        }
    }
    let mut rows: Vec<[f32; 4]> = Vec::new();
    let mut missing: Vec<&str> = Vec::new();
    let mut at_rest: Vec<&str> = Vec::new();
    // A cut clip plays the sim's slow cut; a kind whose sim has none
    // swings his style 1 quick, and the player draws it by the arm. An
    // archer has no weapon arm to draw it by, so the cut clip plays his
    // quick blow at its pace. A kind without a shield never blocks.
    let params = &crate::unit_types::TYPES[kind];
    let has_cut = params.cut_windup_ticks > 0 || bow;
    for (slot, clip) in SKEL_CLIPS.iter().enumerate() {
        if (*clip == "cut" && !has_cut) || (*clip == "block" && params.shield <= 0.0) {
            continue;
        }
        // His own file first; a kind that plays another's clips takes
        // that kind's file when he has none of his own.
        let own = model.with_extension(format!("{clip}.json"));
        let path = if CLIP_KIND[kind] == kind || crate::game_files::is_file(&own) {
            own
        } else {
            model.with_file_name(format!("{}.{clip}.json", KIND_FILE[CLIP_KIND[kind]]))
        };
        let text = match crate::game_files::read_to_string(&path) {
            Ok(text) => text,
            Err(_) if slot >= SKEL_CLIPS_NEEDED => {
                missing.push(*clip);
                continue;
            }
            Err(e) => return Err(format!("cannot read its clip {}: {e}", path.display())),
        };
        let file: ClipFile =
            serde_json::from_str(&text).map_err(|e| format!("{} is not a clip: {e}", path.display()))?;
        if file.format != "flanks-clip" || file.version != 1 {
            return Err(format!("{} is not a flanks-clip of version 1", path.display()));
        }
        // Each of the model's joints takes the clip's track of its name.
        // One the clip does not name keeps its rest pose, and a track for
        // a joint the model lacks is dropped, so another kind's clip plays
        // on the joints the two skeletons share. The root carries the
        // clip's offset, so the clip must name it.
        let track: Vec<Option<usize>> =
            names.iter().map(|n| file.joints.iter().position(|c| c == n)).collect();
        if track[0].is_none() {
            return Err(format!("{} does not name the model's root joint {}", path.display(), names[0]));
        }
        for (j, t) in track.iter().enumerate() {
            if t.is_none() && !at_rest.contains(&names[j].as_str()) {
                at_rest.push(&names[j]);
            }
        }
        if file.samples.is_empty() || !file.fps.is_finite() || file.fps <= 0.0 {
            return Err(format!("{} has no samples or no rate", path.display()));
        }
        skel.clips[slot] = [rows.len() as u32, file.samples.len() as u32, 0, file.looped as u32];
        let length = (file.samples.len() - 1) as f32 / file.fps;
        let contact = file.events.contact.or(file.events.nock).or(file.events.guard).unwrap_or(-1.0);
        if contact > length {
            return Err(format!("{}: its event at {contact} s is after its end at {length} s", path.display()));
        }
        let travel = scale * Vec3::from_array(file.travel).length();
        skel.rates[slot] = [file.fps, length, contact, travel];
        let width = 3 + 4 * file.joints.len();
        for sample in &file.samples {
            if sample.len() != width || sample.iter().any(|v| !v.is_finite()) {
                return Err(format!(
                    "{}: a sample is not 3 + 4 x {} numbers",
                    path.display(),
                    file.joints.len()
                ));
            }
            rows.push([scale * sample[0], scale * sample[1], scale * sample[2], 0.0]);
            for t in &track {
                let q = match t {
                    Some(c) => {
                        let q = &sample[3 + 4 * c..7 + 4 * c];
                        Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize()
                    }
                    None => Quat::IDENTITY,
                };
                rows.push(q.to_array());
            }
        }
    }
    if !missing.is_empty() {
        info!("{}: no clips {missing:?}, the player does without them", model.display());
    }
    if !at_rest.is_empty() {
        info!("{}: joints {at_rest:?} keep their rest pose in clips that do not name them", model.display());
    }
    Ok((skel, rows))
}

/// The weapon's direction from the grip to its point, and how far it
/// reaches behind the grip, in the pitch plane of the rest pose.
fn weapon_axis(level: &Level, grip: [f32; 2]) -> ([f32; 2], f32) {
    let weapon: Vec<usize> =
        (0..level.part.len()).filter(|&i| level.part[i].round() as usize == WEAPON).collect();
    let yz = |i: usize| Vec2::new(level.pos[i].y - grip[0], level.pos[i].z - grip[1]);
    let tip = weapon
        .iter()
        .map(|&i| yz(i))
        .max_by(|a, b| a.length_squared().total_cmp(&b.length_squared()))
        .unwrap_or(Vec2::Y)
        .normalize_or(Vec2::Y);
    let rear = weapon.iter().map(|&i| -yz(i).dot(tip)).fold(0.0f32, f32::max);
    (tip.to_array(), rear)
}

/// Merge parts into the arm: they take its id and pivot, so they move
/// with it.
fn fold(levels: &mut [Option<Level>; NUM_LODS], parts: &[usize], arm: usize) {
    for level in levels.iter_mut().flatten() {
        let Some(pivot) =
            level.part.iter().position(|p| p.round() as usize == arm).map(|i| level.pivot[i])
        else {
            continue;
        };
        for i in 0..level.part.len() {
            if parts.contains(&(level.part[i].round() as usize)) {
                level.part[i] = arm as f32;
                level.pivot[i] = pivot;
            }
        }
    }
}

/// Every part an archer's bow rig moves: the drawing arm, the bow arm,
/// the bow's grip, limbs and string halves, the held arrow, head and
/// torso.
const BOW_RIG: [usize; 14] = [1, 9, 10, 6, 11, 12, 8, 13, 14, 15, 16, 17, 18, 19];

/// Each string half from the nock to its tip, and how far the nocked
/// arrow passes beside the bow's grip, in authoring metres
/// (tools/blender/archer/motion.py `TOP` and `transforms`).
const STRING_HALF_M: f32 = 0.87;
const ARROW_BESIDE_GRIP_M: f32 = 0.020;

/// An archer's bow rig and its shot tables. None when the model has no
/// bow rig parts. A model that has them but cannot use them loses them
/// into the arm, bow arm or body they belong to, logs why, and gets None
/// as well.
fn bow_rig(
    levels: &mut [Option<Level>; NUM_LODS],
    empties: &[(String, Vec3)],
    local: impl Fn(Vec3) -> Vec3,
    scale: f32,
    path: &Path,
    name: &str,
) -> Option<(Rig, Vec<[f32; 4]>)> {
    let level = finest_local(levels);
    if !level.part.iter().any(|p| (11..=19).contains(&(p.round() as usize))) {
        return None;
    }
    match build_bow(level, empties, &local, scale, path) {
        Ok(bow) => {
            info!("{name}: bow rig, shots from {}", shoot_path(path).display());
            Some(bow)
        }
        Err(e) => {
            warn!("{name}: {e}, so the bow rig's parts join the arms and body");
            fold(levels, &[9, 10, 20], 1);
            fold(levels, &[11, 12, 8, 13, 14, 15, 16, 17], 6);
            fold(levels, &[18, 19], 0);
            None
        }
    }
}

fn build_bow(
    level: &Level,
    empties: &[(String, Vec3)],
    local: &impl Fn(Vec3) -> Vec3,
    scale: f32,
    path: &Path,
) -> Fallible<(Rig, Vec<[f32; 4]>)> {
    let has = |part: usize| level.part.iter().any(|p| p.round() as usize == part);
    if let Some(part) = BOW_RIG.iter().find(|&&p| !has(p)) {
        return Err(format!("the bow rig has no {} part", PART_NAMES[*part]));
    }
    let file_path = shoot_path(path);
    let shots = read_shoot(&file_path)?;
    // Each joint is the pivot empty the file names for its part, and it
    // has to sit where the file's tables were built.
    let joint = |part: usize| -> Fallible<Vec3> {
        let key = part.to_string();
        let part_name = shots
            .parts
            .get(&key)
            .ok_or_else(|| format!("{} names no part {part}", file_path.display()))?;
        let at = empties
            .iter()
            .find(|(n, _)| n.strip_prefix("pivot_") == Some(part_name.as_str()))
            .map(|(_, p)| *p)
            .ok_or_else(|| format!("no pivot_{part_name} empty"))?;
        let authored = shots
            .joints
            .get(&key)
            .ok_or_else(|| format!("{} has no joint for {part_name}", file_path.display()))?;
        if at.distance(Vec3::from_array(*authored)) > 1e-3 {
            return Err(format!(
                "{} puts the {part_name} joint at {authored:?}, the model at {at}",
                file_path.display()
            ));
        }
        Ok(at)
    };
    let v4 = |p: Vec3| {
        let q = local(p);
        [q.x, q.y, q.z, 0.0]
    };
    let nock = joint(13)?;
    // The string halves are rigid, so their length is the rig's.
    let reach = level
        .pos
        .iter()
        .zip(&level.part)
        .filter(|(_, p)| p.round() as usize == 13)
        .map(|(v, _)| v.distance(local(nock)))
        .fold(0.0f32, f32::max);
    if (reach - scale * STRING_HALF_M).abs() > scale * 0.02 {
        return Err(format!(
            "the upper string reaches {:.3} m from the nock, the rig's string half is {STRING_HALF_M} m",
            reach / scale
        ));
    }
    let half = Vec3::new(0.0, STRING_HALF_M, 0.0);
    let clips = shots.layout();
    let bow = crate::unit_meshes::Bow {
        draw: [v4(joint(1)?), v4(joint(9)?), v4(joint(10)?)],
        hold: [v4(joint(6)?), v4(joint(11)?), v4(joint(12)?), v4(joint(8)?)],
        limbs: [v4(joint(14)?), v4(joint(15)?)],
        tips: [v4(nock + half), v4(nock - half)],
        nock: v4(nock),
        neck: v4(joint(18)?),
        waist: v4(joint(19)?),
        params: [scale * STRING_HALF_M, scale * ARROW_BESIDE_GRIP_M, shots.pickup, 1.0],
        clips: clips.offsets,
    };
    let rig = Rig { bow, ..Rig::default() };
    Ok((rig, shots.buffer(&clips, local)))
}

/// The shot tables of a model with a bow rig, next to it.
fn shoot_path(model: &Path) -> PathBuf {
    model.with_extension("shoot.json")
}

/// A bow rig's shots as the build scripts write them
/// (tools/blender/archer/motion.py `export`). Each pose holds six local
/// rotations as xyzw quaternions (the drawing arm's shoulder, elbow and
/// wrist, then the bow arm's), and then bow yaw, limb bend, arrow shown,
/// body yaw, torso pitch and bow pitch.
#[derive(serde::Deserialize)]
struct ShootFile {
    version: u32,
    parts: std::collections::HashMap<String, String>,
    joints: std::collections::HashMap<String, [f32; 3]>,
    durations: std::collections::HashMap<String, f32>,
    events: std::collections::HashMap<String, Vec<ShootEvent>>,
    /// The reload's arrow while it is out of the quiver: nock position
    /// and orientation in the torso's frame, and how far it has settled
    /// onto the string.
    arrow_samples: std::collections::HashMap<String, Vec<[f32; 8]>>,
    samples: std::collections::HashMap<String, Vec<[f32; 30]>>,
}

#[derive(serde::Deserialize)]
struct ShootEvent {
    time: f32,
    event: String,
}

struct Shots {
    parts: std::collections::HashMap<String, String>,
    joints: std::collections::HashMap<String, [f32; 3]>,
    raise: Vec<[f32; 30]>,
    release: Vec<[f32; 30]>,
    reload: Vec<[f32; 30]>,
    arrow: Vec<[f32; 8]>,
    /// Share of the reload before the next arrow shows.
    pickup: f32,
}

/// Where each table sits in the clip buffer.
struct ClipLayout {
    offsets: [[u32; 4]; 2],
}

/// vec4s per pose in the clip buffer: six rotations and two of scalars.
const POSE_VEC4S: usize = 8;

impl Shots {
    fn layout(&self) -> ClipLayout {
        let raise = 0;
        let release = raise + self.raise.len() * POSE_VEC4S;
        let reload = release + self.release.len() * POSE_VEC4S;
        let arrow = reload + self.reload.len() * POSE_VEC4S;
        let n = |v: usize| v as u32;
        ClipLayout {
            offsets: [
                [n(raise), n(self.raise.len()), n(release), n(self.release.len())],
                [n(reload), n(self.reload.len()), n(arrow), n(self.arrow.len())],
            ],
        }
    }

    /// The tables as the shader reads them, positions in engine local
    /// space.
    fn buffer(&self, layout: &ClipLayout, local: &impl Fn(Vec3) -> Vec3) -> Vec<[f32; 4]> {
        let mut out = Vec::new();
        for table in [&self.raise, &self.release, &self.reload] {
            for row in table.iter() {
                for q in row[..24].chunks(4) {
                    out.push([q[0], q[1], q[2], q[3]]);
                }
                out.push([row[24], row[25], row[26], row[27]]);
                out.push([row[28], row[29], 0.0, 0.0]);
            }
        }
        for row in &self.arrow {
            let at = local(Vec3::new(row[0], row[1], row[2]));
            out.push([at.x, at.y, at.z, row[7]]);
            out.push([row[3], row[4], row[5], row[6]]);
        }
        debug_assert_eq!(out.len(), layout.offsets[1][2] as usize + self.arrow.len() * 2);
        out
    }
}

/// Read and check a bow rig's shot file.
fn read_shoot(path: &Path) -> Fallible<Shots> {
    let text = crate::game_files::read_to_string(path)
        .map_err(|e| format!("cannot read its shot tables {}: {e}", path.display()))?;
    let mut file: ShootFile = serde_json::from_str(&text)
        .map_err(|e| format!("{} is not a shot file: {e}", path.display()))?;
    if file.version != 3 {
        return Err(format!("{} is version {}, the engine reads 3", path.display(), file.version));
    }
    let mut table = |clip: &str| -> Fallible<Vec<[f32; 30]>> {
        let rows = file
            .samples
            .remove(clip)
            .ok_or_else(|| format!("{} has no {clip} table", path.display()))?;
        if rows.len() < 2 {
            return Err(format!("{}: the {clip} table has {} poses", path.display(), rows.len()));
        }
        if rows.iter().flatten().any(|v| !v.is_finite()) {
            return Err(format!("{}: the {clip} table has a pose that is not a number", path.display()));
        }
        Ok(rows)
    };
    let (raise, release, reload) = (table("raise")?, table("release")?, table("reload")?);
    let arrow = file
        .arrow_samples
        .remove("reload")
        .ok_or_else(|| format!("{} has no reload arrow table", path.display()))?;
    if arrow.len() != reload.len() || arrow.iter().flatten().any(|v| !v.is_finite()) {
        return Err(format!(
            "{}: the reload arrow table has {} rows for {} poses",
            path.display(),
            arrow.len(),
            reload.len()
        ));
    }
    // The shot is a loop: raise, release, reload, and the reload ends
    // where the next raise starts. Only the arrow is allowed to vanish at
    // the loose.
    let same = |a: &[f32; 30], b: &[f32; 30], skip: Option<usize>| {
        let turns = (0..6).all(|j| {
            let dot: f32 = (0..4).map(|k| a[j * 4 + k] * b[j * 4 + k]).sum();
            dot.abs() > 1.0 - 1e-4
        });
        turns && (24..30).all(|i| Some(i) == skip || (a[i] - b[i]).abs() < 1e-3)
    };
    let last = |t: &Vec<[f32; 30]>| *t.last().expect("two poses or more");
    for (from, to, a, b, skip) in [
        ("raise", "release", last(&raise), release[0], Some(26)),
        ("release", "reload", last(&release), reload[0], None),
        ("reload", "raise", last(&reload), raise[0], None),
    ] {
        if !same(&a, &b, skip) {
            return Err(format!("{}: the {from} table does not end where {to} starts", path.display()));
        }
    }
    for (clip, engine) in [
        ("release", crate::render_units::RELEASE_S),
        ("reload", crate::render_units::RELOAD_S),
    ] {
        match file.durations.get(clip) {
            Some(d) if (d - engine).abs() > 1e-3 => warn!(
                "{}: the {clip} is authored over {d} s and plays over {engine} s",
                path.display()
            ),
            Some(_) => {}
            None => return Err(format!("{} gives no {clip} duration", path.display())),
        }
    }
    let reload_s = file.durations.get("reload").copied().unwrap_or(crate::render_units::RELOAD_S);
    let pickup = file
        .events
        .get("reload")
        .and_then(|events| events.iter().find(|e| e.event == "arrow_emerges"))
        .map(|e| e.time / reload_s)
        .ok_or_else(|| format!("{} does not say when the next arrow shows", path.display()))?;
    Ok(Shots {
        parts: file.parts,
        joints: file.joints,
        raise,
        release,
        reload,
        arrow,
        pickup,
    })
}

/// The attack tables of a model with a jointed arm, next to it.
fn attack_path(model: &Path) -> PathBuf {
    model.with_extension("arm_stab.json")
}

/// A jointed arm's attack as the build scripts write it
/// (tools/blender/spearman/motion.py `export_clip`). Poses are shoulder,
/// elbow and wrist turns from the rest pose plus how far the weapon is
/// levelled, and the turns are pitches with + taking +Z toward +Y, the
/// shader's convention.
#[derive(serde::Deserialize)]
struct AttackFile {
    version: u32,
    /// Joint positions by part id, model space.
    joints_gltf_xyz: std::collections::HashMap<String, [f32; 3]>,
    windup_samples: Vec<[f32; 4]>,
    recovery_samples: Vec<[f32; 4]>,
    recovery_seconds: f32,
    weapon_slide_levelled_m: f32,
}

struct Attack {
    windup: [[f32; 4]; crate::unit_meshes::ATTACK_SAMPLES],
    recover: [[f32; 4]; crate::unit_meshes::ATTACK_SAMPLES],
    weapon_slide_levelled_m: f32,
}

/// Read and check an attack file against the model's joints.
fn read_attack(path: &Path, joints: &[Vec3; 4], parts: [usize; 4]) -> Fallible<Attack> {
    let text = crate::game_files::read_to_string(path)
        .map_err(|e| format!("cannot read its attack tables {}: {e}", path.display()))?;
    let file: AttackFile = serde_json::from_str(&text)
        .map_err(|e| format!("{} is not an attack file: {e}", path.display()))?;
    if file.version != 1 {
        return Err(format!("{} is version {}, the engine reads 1", path.display(), file.version));
    }
    let table = |samples: &[[f32; 4]]| -> Fallible<[[f32; 4]; crate::unit_meshes::ATTACK_SAMPLES]> {
        if samples.iter().flatten().any(|v| !v.is_finite()) {
            return Err(format!("{} has a pose that is not a number", path.display()));
        }
        samples.try_into().map_err(|_| {
            format!(
                "{} has {} poses in a table, the engine samples {}",
                path.display(),
                samples.len(),
                crate::unit_meshes::ATTACK_SAMPLES
            )
        })
    };
    let windup = table(&file.windup_samples)?;
    let recover = table(&file.recovery_samples)?;
    // The soldier stands in the guard before and after a blow, and the
    // follow-through starts where the wind-up struck.
    let apart = |a: [f32; 4], b: [f32; 4]| a.iter().zip(b).any(|(a, b)| (a - b).abs() > 1e-3);
    if apart(windup[0], recover[crate::unit_meshes::ATTACK_SAMPLES - 1]) {
        return Err(format!("{}: the follow-through does not end in the guard", path.display()));
    }
    if apart(windup[crate::unit_meshes::ATTACK_SAMPLES - 1], recover[0]) {
        return Err(format!("{}: the follow-through does not start at the strike", path.display()));
    }
    for (joint, part) in joints.iter().zip(parts) {
        let Some(at) = file.joints_gltf_xyz.get(&part.to_string()) else {
            return Err(format!("{} has no joint for part {}", path.display(), PART_NAMES[part]));
        };
        if joint.distance(Vec3::from_array(*at)) > 1e-3 {
            return Err(format!(
                "{} puts the {} joint at {at:?}, the model at {joint}",
                path.display(),
                PART_NAMES[part]
            ));
        }
    }
    if (file.recovery_seconds - crate::render_units::FOLLOW_S).abs() > 1e-3 {
        warn!(
            "{}: the follow-through is authored over {} s and plays over {} s",
            path.display(),
            file.recovery_seconds,
            crate::render_units::FOLLOW_S
        );
    }
    Ok(Attack { windup, recover, weapon_slide_levelled_m: file.weapon_slide_levelled_m })
}

/// The atlas a primitive samples, as its `TEXCOORD` set: None when the
/// model has no atlas. With an atlas, every primitive must read it.
fn atlas_set(prim: &gltf::Primitive, atlas: Option<usize>) -> Fallible<Option<u32>> {
    let Some(atlas) = atlas else {
        return Ok(None);
    };
    let info = prim.material().pbr_metallic_roughness().base_color_texture();
    match info {
        Some(info) if info.texture().source().index() == atlas => match info.tex_coord() {
            0 => Ok(Some(0)),
            n => Err(format!(
                "the atlas is read through TEXCOORD_{n}, the spec puts it in TEXCOORD_0"
            )),
        },
        _ => Err("a primitive does not sample the model's atlas, one texture per model".into()),
    }
}

/// `L0` to `L3` in a node name, the spec's level naming.
fn level_index(name: &str) -> Option<usize> {
    let rest = name.strip_prefix('L')?;
    let lod: usize = rest.parse().ok()?;
    (lod < NUM_LODS).then_some(lod)
}

/// Append one primitive's vertices, with the node transform applied.
fn read_primitive(
    level: &mut Level,
    prim: &gltf::Primitive,
    blob: Option<&[u8]>,
    xf: Mat4,
    tex_set: Option<u32>,
) -> Fallible<()> {
    use gltf::mesh::util::ReadColors;
    if prim.mode() != gltf::mesh::Mode::Triangles {
        return Err("the mesh is not triangles, export with the triangulate option".into());
    }
    let reader = prim.reader(|b| (b.index() == 0).then_some(blob).flatten());
    let pos = reader.read_positions().ok_or("no POSITION")?;
    let nrm = reader.read_normals().ok_or("no NORMAL, export with export_normals=True")?;
    let uv = reader
        .read_tex_coords(1)
        .ok_or("no TEXCOORD_1: the build script must bake (part id, pivot height) into a second UV map")?;
    // A textured model takes its colour and team mask from the atlas, so
    // only an untextured one needs COLOR_0, and needs its alpha.
    let colors = reader.read_colors(0);
    if tex_set.is_none() {
        match &colors {
            None => return Err("no COLOR_0: export with export_vertex_color=\"ACTIVE\"".into()),
            Some(ReadColors::RgbU8(_) | ReadColors::RgbU16(_) | ReadColors::RgbF32(_)) => {
                return Err(
                    "COLOR_0 is VEC3, so the team-colour amount in alpha was dropped: export with export_vertex_color=\"ACTIVE\""
                        .into(),
                );
            }
            Some(_) => {}
        }
    }
    let tex = match tex_set {
        Some(set) => Some(
            reader
                .read_tex_coords(set)
                .ok_or(format!("the atlas reads TEXCOORD_{set}, which the mesh does not have"))?,
        ),
        None => None,
    };
    let indices = reader.read_indices().ok_or("the mesh has no index buffer")?;
    let skin = reader.read_joints(0).zip(reader.read_weights(0));

    let base = level.pos.len() as u32;
    // A node scale would squash the normals; the inverse transpose is
    // the one that stays perpendicular.
    let normal_xf = Mat3::from_mat4(xf).inverse().transpose();
    level.pos.extend(pos.map(|p| xf.transform_point3(Vec3::from_array(p))));
    level.nrm.extend(nrm.map(|n| (normal_xf * Vec3::from_array(n)).normalize_or_zero()));
    for uv in uv.into_f32() {
        level.part.push(uv[0]);
        level.pivot.push(uv[1]);
    }
    let n = level.pos.len();
    match colors {
        Some(colors) => level.col.extend(colors.into_rgba_f32()),
        None => level.col.resize(n, [1.0; 4]),
    }
    match tex {
        Some(tex) => level.tex.extend(tex.into_f32()),
        None => level.tex.resize(n, [0.0; 2]),
    }
    // The two heaviest joints of each vertex. A level keeps its skin only
    // while every primitive read into it has one.
    match skin {
        Some((joints, weights)) if level.skin.len() == base as usize => {
            for (j, w) in joints.into_u16().zip(weights.into_f32()) {
                let mut order = [0, 1, 2, 3];
                order.sort_by(|a, b| w[*b].total_cmp(&w[*a]));
                let (first, second) = (order[0], order[1]);
                let sum = w[first] + w[second];
                let share = if sum > 0.0 { w[second] / sum } else { 0.0 };
                level.skin.push((j[first], j[second], share));
            }
        }
        _ => level.skin.clear(),
    }
    if level.nrm.len() != n
        || level.col.len() != n
        || level.part.len() != n
        || level.tex.len() != n
        || (!level.skin.is_empty() && level.skin.len() != n)
    {
        return Err("the attributes have different vertex counts".into());
    }
    // A malformed file fails here with a message, not with a panic in
    // the renderer.
    let count = n - base as usize;
    let indices: Vec<u32> = indices.into_u32().collect();
    if !indices.len().is_multiple_of(3) {
        return Err("the index list is not whole triangles".into());
    }
    if indices.iter().any(|i| *i as usize >= count) {
        return Err("an index points past the end of the vertex list".into());
    }
    level.idx.extend(indices.into_iter().map(|i| base + i));
    Ok(())
}

/// Checks for the export traps the spec lists. Blender shows none of
/// them.
fn check_parts(level: &Level, pivot_nodes: &[(usize, f32)], name: &str) {
    let mut pivots: Vec<Option<f32>> = vec![None; PART_NAMES.len()];
    let mut unknown = 0usize;
    let mut mixed: Vec<usize> = Vec::new();
    for (i, part) in level.part.iter().enumerate() {
        let id = part.round();
        if (part - id).abs() > 1e-3 || id < 0.0 || id as usize >= PART_NAMES.len() {
            unknown += 1;
            continue;
        }
        let id = id as usize;
        match pivots[id] {
            Some(v) if (v - level.pivot[i]).abs() > 1e-3 => {
                if !mixed.contains(&id) {
                    mixed.push(id);
                }
            }
            _ => pivots[id] = Some(level.pivot[i]),
        }
    }
    // One line per fault, not one per vertex.
    if unknown > 0 {
        warn!("{name}: {unknown} vertices carry a part id the engine has no part for");
    }
    for id in mixed {
        warn!("{name}: part {} carries more than one pivot height", PART_NAMES[id]);
    }
    // The exporter flips UV v, so a build script that fails to compensate
    // ships `1 - height` and every joint sits in the wrong place. The
    // pivot empties survive export untouched, so they are the check.
    for (part, y) in pivot_nodes {
        let Some(Some(uv)) = pivots.get(*part) else { continue };
        if (uv - y).abs() > 0.01 {
            warn!(
                "{name}: part {} has pivot height {uv:.3} in TEXCOORD_1 but its pivot_{} empty is at {y:.3}",
                PART_NAMES[*part], PART_NAMES[*part]
            );
        }
    }
    let mut spanning = 0;
    for t in level.idx.chunks(3) {
        if t.len() == 3 && (level.part[t[0] as usize] != level.part[t[1] as usize]
            || level.part[t[0] as usize] != level.part[t[2] as usize])
        {
            spanning += 1;
        }
    }
    if spanning > 0 {
        warn!("{name}: {spanning} triangles span two parts, so they tear when the parts rotate");
    }
}

/// Model space (metres, feet at 0) to engine local space (feet at
/// -half_height).
fn to_local(level: &mut Level, scale: f32, half_height: f32) {
    for p in &mut level.pos {
        *p = Vec3::new(scale * p.x, scale * p.y - half_height, scale * p.z);
    }
    for v in &mut level.pivot {
        *v = scale * *v - half_height;
    }
}

/// Per vertex of a skinned level: x = its two joints as `a + 256 * b`,
/// y = the weight of joint b. b is a or a child of a, and the vertex
/// turns about b's origin (unit_pose.wgsl `place`).
pub const ATTRIBUTE_SKIN: bevy::mesh::MeshVertexAttribute = bevy::mesh::MeshVertexAttribute::new(
    "UnitSkin",
    0x464c_534b,
    bevy::render::render_resource::VertexFormat::Float32x2,
);

fn level_mesh(level: &Level) -> Mesh {
    let mesh = level_mesh_unskinned(level);
    if level.skin.is_empty() {
        return mesh;
    }
    mesh.with_inserted_attribute(
        ATTRIBUTE_SKIN,
        level
            .skin
            .iter()
            .map(|(a, b, w)| [(*a as u32 + 256 * *b as u32) as f32, *w])
            .collect::<Vec<_>>(),
    )
}

fn level_mesh_unskinned(level: &Level) -> Mesh {
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            level.pos.iter().map(Vec3::to_array).collect::<Vec<_>>(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            level.nrm.iter().map(Vec3::to_array).collect::<Vec<_>>(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_UV_0,
            level
                .part
                .iter()
                .zip(&level.pivot)
                .map(|(p, v)| [*p, *v])
                .collect::<Vec<_>>(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, level.tex.clone())
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, level.col.clone())
        .with_inserted_indices(Indices::U32(level.idx.clone()))
}

/// The model's atlas: the base colour texture of its first textured
/// material, decoded, with a mip chain. The glTF image index comes with
/// it so every primitive can be checked against it.
fn read_atlas(
    gltf: &gltf::Gltf,
    blob: Option<&[u8]>,
    path: &Path,
) -> Fallible<Option<(usize, Image)>> {
    let Some(info) = gltf
        .document
        .materials()
        .find_map(|m| m.pbr_metallic_roughness().base_color_texture())
    else {
        return Ok(None);
    };
    let source = info.texture().source();
    let (bytes, mime) = match source.source() {
        gltf::image::Source::View { view, mime_type } => {
            let blob = blob
                .filter(|_| view.buffer().index() == 0)
                .ok_or("the atlas is not in the GLB's binary chunk")?;
            let bytes = blob
                .get(view.offset()..view.offset() + view.length())
                .ok_or("the atlas runs past the end of the file")?;
            (bytes.to_vec(), Some(mime_type))
        }
        gltf::image::Source::Uri { uri, mime_type } => {
            let file = path.parent().unwrap_or(Path::new(".")).join(uri);
            let bytes = crate::game_files::read(&file)
                .map_err(|e| format!("cannot read the atlas {}: {e}", file.display()))?;
            (bytes.into_owned(), mime_type)
        }
    };
    let sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: 8,
        ..ImageSamplerDescriptor::linear()
    });
    // sRGB colour. The mask in alpha stays linear, which is what an sRGB
    // format does with alpha.
    let mut image = Image::from_buffer(
        &bytes,
        ImageType::MimeType(mime.unwrap_or("image/png")),
        CompressedImageFormats::NONE,
        true,
        sampler,
        RenderAssetUsages::RENDER_WORLD,
    )
    .map_err(|e| format!("cannot decode the atlas: {e}"))?;
    if image.texture_descriptor.format != TextureFormat::Rgba8UnormSrgb {
        return Err(format!(
            "the atlas decodes to {:?}, it must be 8-bit RGBA",
            image.texture_descriptor.format
        ));
    }
    add_mips(&mut image)?;
    Ok(Some((source.index(), image)))
}

fn srgb_to_linear_lut() -> [f32; 256] {
    std::array::from_fn(|i| {
        let c = i as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    })
}

fn linear_to_srgb(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let c = if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (c * 255.0 + 0.5) as u8
}

/// A box-filtered mip chain down to one pixel, colour averaged in linear
/// light and the mask as it is. A soldier a few pixels tall samples the
/// small levels, and without them the atlas shimmers.
fn add_mips(image: &mut Image) -> Fallible<()> {
    let size = image.texture_descriptor.size;
    let (mut w, mut h) = (size.width as usize, size.height as usize);
    let Some(base) = image.data.take() else {
        return Err("the atlas has no pixel data".into());
    };
    if base.len() != w * h * 4 {
        return Err("the atlas pixel data does not match its size".into());
    }
    let lut = srgb_to_linear_lut();
    let mut data = base.clone();
    let mut prev = base;
    let mut levels = 1;
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut next = vec![0u8; nw * nh * 4];
        for y in 0..nh {
            for x in 0..nw {
                let mut sum = [0.0f32; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let sx = (2 * x + dx).min(w - 1);
                    let sy = (2 * y + dy).min(h - 1);
                    let p = &prev[(sy * w + sx) * 4..][..4];
                    for c in 0..3 {
                        sum[c] += lut[p[c] as usize];
                    }
                    sum[3] += p[3] as f32;
                }
                let out = &mut next[(y * nw + x) * 4..][..4];
                for c in 0..3 {
                    out[c] = linear_to_srgb(sum[c] * 0.25);
                }
                out[3] = (sum[3] * 0.25 + 0.5) as u8;
            }
        }
        data.extend_from_slice(&next);
        prev = next;
        (w, h) = (nw, nh);
        levels += 1;
    }
    image.data = Some(data);
    image.texture_descriptor.mip_level_count = levels;
    Ok(())
}

/// Vertex colours from the atlas, for the derived far levels, which are
/// boxes with nothing to sample. Each triangle is sampled at ten points
/// across it and each vertex averages the triangles around it. The result
/// is in the form the vertex path blends: the colour outside the team
/// mask, with the mask as alpha.
fn atlas_vertex_colors(level: &Level, image: &Image) -> Vec<[f32; 4]> {
    let size = image.texture_descriptor.size;
    let (w, h) = (size.width as usize, size.height as usize);
    let px = image.data.as_deref().unwrap_or(&[]);
    let n = level.pos.len();
    if px.len() < w * h * 4 {
        return vec![[1.0, 1.0, 1.0, 0.0]; n];
    }
    let lut = srgb_to_linear_lut();
    // Per vertex: linear colour outside the mask, inside it, triangles.
    let mut acc = vec![([0.0f32; 3], [0.0f32; 3], 0.0f32); n];
    for tri in level.idx.chunks_exact(3) {
        let uv = [0, 1, 2].map(|k| Vec2::from(level.tex[tri[k] as usize]));
        let mut plain = [0.0f32; 3];
        let mut team = [0.0f32; 3];
        for a in 0..=3 {
            for b in 0..=3 - a {
                let c = 3 - a - b;
                let p = (uv[0] * a as f32 + uv[1] * b as f32 + uv[2] * c as f32) / 3.0;
                let x = ((p.x.clamp(0.0, 1.0) * w as f32) as usize).min(w - 1);
                let y = ((p.y.clamp(0.0, 1.0) * h as f32) as usize).min(h - 1);
                let t = &px[(y * w + x) * 4..][..4];
                let mask = t[3] as f32 / 255.0;
                for ch in 0..3 {
                    let v = lut[t[ch] as usize];
                    plain[ch] += v * (1.0 - mask) / 10.0;
                    team[ch] += v * mask / 10.0;
                }
            }
        }
        for &i in tri {
            let e = &mut acc[i as usize];
            for ch in 0..3 {
                e.0[ch] += plain[ch];
                e.1[ch] += team[ch];
            }
            e.2 += 1.0;
        }
    }
    acc.iter()
        .map(|(plain, team, count)| {
            if *count == 0.0 {
                return [1.0, 1.0, 1.0, 0.0];
            }
            // The atlas shows `plain + tint * team` and the vertex path
            // shows `mix(rgb, tint, a)`. They agree exactly where the
            // masked surface is grey, which the spec asks team surfaces
            // to be.
            let a = ((team[0] + team[1] + team[2]) / (3.0 * count)).clamp(0.0, 1.0);
            let keep = (1.0 - a).max(1e-3) * count;
            [
                (plain[0] / keep).min(1.0),
                (plain[1] / keep).min(1.0),
                (plain[2] / keep).min(1.0),
                a,
            ]
        })
        .collect()
}

/// Facings the visibility pass renders. A soldier stands at any yaw, so
/// what a surface is worth is its average over a full turn.
const VIS_VIEWS: usize = 8;
/// Depth buffer edge in pixels, spanning the whole figure. Detail under
/// a pixel here is also under a pixel at the sizes a derived level is
/// drawn at.
const VIS_RES: usize = 96;
/// The battle camera looks down at about this angle.
const VIS_PITCH: f32 = 50.0;

/// How much of each vertex's surface a player sees, measured by
/// rendering the model into a depth buffer from `VIS_VIEWS` facings at
/// the battle camera's angle. A triangle is worth the pixels it wins,
/// split between its three corners, so hidden layers count for nothing.
fn visible_weights(level: &Level) -> Vec<f32> {
    let mut weights = vec![0.0; level.pos.len()];
    let (min, max) = level
        .pos
        .iter()
        .fold((Vec3::MAX, Vec3::MIN), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    let center = (min + max) * 0.5;
    let radius = ((max - min).length() * 0.5).max(1e-3);
    let scale = VIS_RES as f32 / (2.0 * radius);
    let pixel = 1.0 / (3.0 * scale * scale);
    let pitch = VIS_PITCH.to_radians();
    let mut depth = vec![0.0f32; VIS_RES * VIS_RES];
    let mut winner = vec![u32::MAX; VIS_RES * VIS_RES];
    let mut proj: Vec<Vec3> = Vec::with_capacity(level.pos.len());
    for view in 0..VIS_VIEWS {
        let yaw = std::f32::consts::TAU * view as f32 / VIS_VIEWS as f32;
        // Straight down the camera's line of sight, into the scene.
        let dir = Vec3::new(
            yaw.cos() * pitch.cos(),
            -pitch.sin(),
            yaw.sin() * pitch.cos(),
        );
        let right = dir.cross(Vec3::Y).normalize();
        let up = right.cross(dir);
        depth.fill(f32::MAX);
        winner.fill(u32::MAX);
        proj.clear();
        proj.extend(level.pos.iter().map(|p| {
            let q = *p - center;
            Vec3::new(
                (q.dot(right) + radius) * scale,
                (q.dot(up) + radius) * scale,
                q.dot(dir),
            )
        }));
        for (t, tri) in level.idx.chunks(3).enumerate() {
            if tri.len() < 3 {
                continue;
            }
            let (a, b, c) = (
                proj[tri[0] as usize],
                proj[tri[1] as usize],
                proj[tri[2] as usize],
            );
            // Back faces never show. The normals are the model's own, so
            // this holds whichever way the triangles wind.
            let facing: Vec3 = tri.iter().map(|i| level.nrm[*i as usize]).sum();
            if facing.dot(dir) >= 0.0 {
                continue;
            }
            let area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
            if area.abs() < 1e-12 {
                continue;
            }
            let inv = 1.0 / area;
            let lo_x = a.x.min(b.x).min(c.x).floor().max(0.0) as usize;
            let lo_y = a.y.min(b.y).min(c.y).floor().max(0.0) as usize;
            let hi_x = (a.x.max(b.x).max(c.x).ceil().max(0.0) as usize).min(VIS_RES);
            let hi_y = (a.y.max(b.y).max(c.y).ceil().max(0.0) as usize).min(VIS_RES);
            for py in lo_y..hi_y {
                for px in lo_x..hi_x {
                    let (fx, fy) = (px as f32 + 0.5, py as f32 + 0.5);
                    let w0 = ((b.x - fx) * (c.y - fy) - (c.x - fx) * (b.y - fy)) * inv;
                    let w1 = ((c.x - fx) * (a.y - fy) - (a.x - fx) * (c.y - fy)) * inv;
                    let w2 = 1.0 - w0 - w1;
                    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                        continue;
                    }
                    let d = w0 * a.z + w1 * b.z + w2 * c.z;
                    let slot = py * VIS_RES + px;
                    if d < depth[slot] {
                        depth[slot] = d;
                        winner[slot] = t as u32;
                    }
                }
            }
        }
        for t in winner.iter().filter(|t| **t != u32::MAX) {
            for i in &level.idx[*t as usize * 3..*t as usize * 3 + 3] {
                weights[*i as usize] += pixel;
            }
        }
    }
    weights
}

/// Body, legs, and an archer's separate head and torso: the parts a
/// figure's mass sits in. The last level is one block stack, and an arm
/// holding a sword out in front would double the width of every box if
/// it counted toward them.
fn is_trunk(part: f32) -> bool {
    matches!(part.round() as u32, 0 | 2 | 3 | 18 | 19)
}

/// Build a far level from a finer one: every part is cut into slabs
/// along its longest axis and each slab becomes one box of the size its
/// vertices span. Boxes keep the part id and pivot they came from, so
/// the pose is the same on both sides of a level switch, and take the
/// blended colour of the surfaces they replace.
fn derive_level(src: &Level, weights: &[f32], lod: usize) -> Mesh {
    let (body_slabs, limb_slabs) = SLABS[lod];
    let mut buf = MeshBuf::new();
    if lod + 1 == NUM_LODS {
        // The spec: "L3 is all body". The boxes take their size from
        // the trunk, so a weapon held out in front cannot widen them, and
        // their colour from every part, since at two pixels tall the hue
        // is all that is left.
        let trunk: Vec<usize> = (0..src.pos.len()).filter(|i| is_trunk(src.part[*i])).collect();
        let all: Vec<usize> = (0..src.pos.len()).collect();
        emit_slabs(&mut buf, src, weights, &trunk, &all, 0.0, 0.0, body_slabs);
        return crate::unit_meshes::build(buf);
    }
    let mut parts: Vec<(f32, f32, Vec<usize>)> = Vec::new();
    for i in 0..src.pos.len() {
        let part = src.part[i];
        match parts.iter_mut().find(|(p, _, _)| *p == part) {
            Some((_, _, list)) => list.push(i),
            None => parts.push((part, src.pivot[i], vec![i])),
        }
    }
    for (part, pivot, verts) in parts {
        let n = if part.round() as u32 == 0 { body_slabs } else { limb_slabs };
        emit_slabs(&mut buf, src, weights, &verts, &verts, part, pivot, n);
    }
    crate::unit_meshes::build(buf)
}

/// Cut `geom` into `n` slabs along its longest axis and emit one box per
/// slab. Each box takes the blended colour of the `color` vertices that
/// fall in its slab, which is the same set except on the last level.
#[allow(clippy::too_many_arguments)]
fn emit_slabs(
    buf: &mut MeshBuf,
    src: &Level,
    weights: &[f32],
    geom: &[usize],
    color: &[usize],
    part: f32,
    pivot: f32,
    n: usize,
) {
    if geom.is_empty() {
        return;
    }
    let n = n.max(1);
    let (min, max) = geom
        .iter()
        .fold((Vec3::MAX, Vec3::MIN), |(lo, hi), i| (lo.min(src.pos[*i]), hi.max(src.pos[*i])));
    let span = max - min;
    let axis = if span.y >= span.x && span.y >= span.z {
        1
    } else if span.x >= span.z {
        0
    } else {
        2
    };
    let thickness = (span[axis] / n as f32).max(1e-5);
    let slab_of = |p: Vec3| (((p[axis] - min[axis]) / thickness) as usize).min(n - 1);
    let mut slabs = vec![(Vec3::MAX, Vec3::MIN, Vec::new()); n];
    for i in geom {
        let (lo, hi, _) = &mut slabs[slab_of(src.pos[*i])];
        *lo = lo.min(src.pos[*i]);
        *hi = hi.max(src.pos[*i]);
    }
    for i in color {
        // A colour vertex outside the trunk lands in the slab nearest
        // its height, so a raised arm tints the shoulders and not the feet.
        let s = slab_of(src.pos[*i].clamp(min, max));
        // The floor keeps a slab that the visibility pass never saw from
        // blending nothing at all.
        slabs[s].2.push((src.col[*i], weights[*i] + 1e-6));
    }
    for (s, (mut lo, mut hi, cols)) in slabs.into_iter().enumerate() {
        if cols.is_empty() || lo.x > hi.x {
            continue;
        }
        // Along the split axis the box takes the slab's own bounds, not
        // its vertices': neighbours then meet instead of leaving a ring
        // of gaps up the figure.
        lo[axis] = min[axis] + s as f32 * thickness;
        hi[axis] = lo[axis] + thickness;
        let half = ((hi - lo) * 0.5).max(Vec3::splat(MIN_HALF));
        buf.cuboid((lo + hi) * 0.5, half, part, pivot, blend(&cols));
    }
}
