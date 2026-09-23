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
//! attack tables that ship next to the model, `<kind>.stab.json`, which
//! the build scripts in tools/blender write. Without them the arm bends
//! at the elbow as one piece. An archer with jointed arms, a strung bow,
//! a held arrow and a separate head and torso shoots from the tables in
//! `<kind>.shoot.json`. Without them the parts join the arm, bow arm or
//! body they belong to, and the archer shoots as a rigid figure.
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
use crate::unit_meshes::{MeshBuf, Rig, blend};
use crate::unit_types::NUM_KINDS;

type Fallible<T> = Result<T, String>;

/// One file per kind, named as the asset spec names them.
const KIND_FILE: [&str; NUM_KINDS] = ["knight", "man_at_arms", "spearman", "archer"];

/// Part id to the name its pivot empty carries (`pivot_<part>`), in the
/// order unit_meshes.rs and the vertex shader use. Id 7 is the arrow in
/// flight, which no model carries.
const PART_NAMES: [&str; 20] = [
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
/// rig, and the shot tables of an archer's bow rig (empty without one).
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
}

impl KindMeshes {
    fn code_built(kind: usize) -> Self {
        Self {
            lods: crate::unit_meshes::build_kind_lods(kind),
            atlas: None,
            textured: [false; NUM_LODS],
            rig: crate::unit_meshes::code_rig(kind),
            clips: Vec::new(),
        }
    }
}

/// The mesh set a kind renders with: the imported model when its file
/// is there, the code-built set otherwise. `FL_UNIT_MESH=code` keeps the
/// code-built set either way, which is the A/B against an import. A file
/// that is there but unusable logs an error and falls back, so a battle
/// still runs on a half-exported model. Called once per kind at startup.
pub fn kind_lods(kind: usize) -> KindMeshes {
    let code_only = std::env::var("FL_UNIT_MESH").is_ok_and(|v| v == "code");
    let path = (!code_only).then(|| model_path(kind)).flatten();
    let Some(path) = path else {
        return KindMeshes::code_built(kind);
    };
    match import(kind, &path) {
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
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let name = KIND_FILE[kind];
    let var = format!("FL_GLB_{}", name.to_uppercase());
    if let Ok(named) = std::env::var(&var) {
        let named = root.join(named);
        if named.is_file() {
            return Some(named);
        }
        warn!("{var}: {} is not a file", named.display());
    }
    let shipped = root.join("assets/units").join(format!("{name}.glb"));
    if shipped.is_file() {
        return Some(shipped);
    }
    let working = root.join("assets_dev").join(name).join(format!("{name}.glb"));
    working.is_file().then_some(working)
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
    idx: Vec<u32>,
}

impl Level {
    fn tris(&self) -> usize {
        self.idx.len() / 3
    }

    /// Ground line and height, for the scale and the sanity checks. The
    /// height is the top of the body part, not of whatever the soldier
    /// holds above his head: a spearman's spear stands 0.7 m higher.
    fn y_range(&self) -> (f32, f32) {
        let ground = self.pos.iter().fold(f32::MAX, |lo, p| lo.min(p.y));
        // The figure's height is its trunk's: a spear or a bow held
        // upright reaches higher.
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
fn import(kind: usize, path: &Path) -> Fallible<KindMeshes> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read it: {e}"))?;
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
    let mut stack: Vec<(gltf::Node, Mat4)> = scene.nodes().map(|n| (n, Mat4::IDENTITY)).collect();
    while let Some((node, parent)) = stack.pop() {
        let xf = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        for child in node.children() {
            stack.push((child, xf));
        }
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
    let (rig, clips) = match bow_rig(&mut levels, &empties, local3, scale, path, name) {
        Some(bow) => bow,
        None => (arm_rig(&mut levels, &empties, local, scale, path, name), Vec::new()),
    };

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
    let textured: [bool; NUM_LODS] =
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
    })
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
            fold(levels, &[9, 10], 1);
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
    let text = std::fs::read_to_string(path)
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
    model.with_extension("stab.json")
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
    let text = std::fs::read_to_string(path)
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
    if level.nrm.len() != n
        || level.col.len() != n
        || level.part.len() != n
        || level.tex.len() != n
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

fn level_mesh(level: &Level) -> Mesh {
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
            let bytes = std::fs::read(&file)
                .map_err(|e| format!("cannot read the atlas {}: {e}", file.display()))?;
            (bytes, mime_type)
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
