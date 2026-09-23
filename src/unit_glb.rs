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
//! feet sit at `-half_height`. It is also mirrored on X, because every
//! code-built mesh carries the shield on -X and the shieldwall pose
//! swings it from there. The sim credits shield cover on +X, a mismatch
//! older than this file.
//!
//! Levels the file lacks are derived from its finest level: each part is
//! cut into slabs along its longest axis and each slab becomes one box.
//! Box colours are weighted by what the battle camera sees, because an
//! area average lets the hidden mail under a surcoat turn a blue
//! regiment grey.
//!
//! `FL_UNIT_MESH=code` keeps the code-built meshes, `FL_GLB_FAR=code`
//! fills only the missing levels from them, and `FL_GLB_MIRROR=0`
//! imports the model as authored.

use std::path::{Path, PathBuf};

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;

use crate::render_units::NUM_LODS;
use crate::unit_meshes::{MeshBuf, blend};
use crate::unit_types::NUM_KINDS;

type Fallible<T> = Result<T, String>;

/// One file per kind, named as the asset spec names them.
const KIND_FILE: [&str; NUM_KINDS] = ["knight", "man_at_arms", "spearman", "archer"];

/// Part id to the name its pivot empty carries (`pivot_<part>`), in the
/// order unit_meshes.rs and the vertex shader use.
const PART_NAMES: [&str; 7] = [
    "body",
    "arm_weapon",
    "leg_l",
    "leg_r",
    "arm_spear",
    "arm_shield",
    "arm_bow",
];

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

/// The mesh set a kind renders with: the imported model when its file
/// is there, the code-built set otherwise. `FL_UNIT_MESH=code` keeps the
/// code-built set either way, which is the A/B against an import. A file
/// that is there but unusable logs an error and falls back, so a battle
/// still runs on a half-exported model. Called once per kind at startup.
pub fn kind_lods(kind: usize) -> [Mesh; NUM_LODS] {
    let code_only = std::env::var("FL_UNIT_MESH").is_ok_and(|v| v == "code");
    let path = (!code_only).then(|| model_path(kind)).flatten();
    let Some(path) = path else {
        return crate::unit_meshes::build_kind_lods(kind);
    };
    match import(kind, &path) {
        Ok(meshes) => meshes,
        Err(e) => {
            error!("{} is not usable, the code-built mesh stands in: {e}", path.display());
            crate::unit_meshes::build_kind_lods(kind)
        }
    }
}

/// The model file of a kind: the shipped asset first, then the working
/// copy the asset track writes. Paths are anchored at the crate root, the
/// way the asset plugin anchors `assets/`.
fn model_path(kind: usize) -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let name = KIND_FILE[kind];
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
    idx: Vec<u32>,
}

impl Level {
    fn tris(&self) -> usize {
        self.idx.len() / 3
    }

    /// Model height and ground line, for the scale and the sanity checks.
    fn y_range(&self) -> (f32, f32) {
        self.pos.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p.y), hi.max(p.y)))
    }
}

/// Read a model and hand back its four levels, engine local space, ready
/// to hand to `Mesh3d`.
fn import(kind: usize, path: &Path) -> Fallible<[Mesh; NUM_LODS]> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read it: {e}"))?;
    let gltf = gltf::Gltf::from_slice(&bytes).map_err(|e| format!("not valid glTF: {e}"))?;
    let blob = gltf.blob.as_deref();

    // Level nodes and pivot empties, with the parent chain applied. A
    // stock Blender export leaves them at the scene root, but a model
    // nested under an empty has to arrive in the same place.
    let mut levels: [Option<Level>; NUM_LODS] = Default::default();
    let mut pivot_nodes: Vec<(usize, f32)> = Vec::new();
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
            .and_then(|p| PART_NAMES.iter().position(|q| *q == p))
        {
            pivot_nodes.push((part, xf.w_axis.y));
        }
        let Some(lod) = level_index(name) else { continue };
        let Some(mesh) = node.mesh() else { continue };
        let level = levels[lod].get_or_insert_with(Level::default);
        for prim in mesh.primitives() {
            read_primitive(level, &prim, blob, xf).map_err(|e| format!("node {name}: {e}"))?;
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
    let mirror = !std::env::var("FL_GLB_MIRROR").is_ok_and(|v| v == "0");
    for level in levels.iter_mut().flatten() {
        to_local(level, scale, half_height, mirror);
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
    let source = levels.iter().flatten().next().cloned().expect("checked above");
    let weights = visible_weights(&source);
    let code_far = std::env::var("FL_GLB_FAR").is_ok_and(|v| v == "code");
    let mut code = code_far.then(|| crate::unit_meshes::build_kind_lods(kind).map(Some));
    Ok(std::array::from_fn(|lod| match levels[lod].take() {
        Some(level) => level_mesh(&level),
        None => match code.as_mut().and_then(|c| c[lod].take()) {
            Some(mesh) => mesh,
            None => derive_level(&source, &weights, lod),
        },
    }))
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
    let colors = reader
        .read_colors(0)
        .ok_or("no COLOR_0: export with export_vertex_color=\"ACTIVE\"")?;
    if matches!(colors, ReadColors::RgbU8(_) | ReadColors::RgbU16(_) | ReadColors::RgbF32(_)) {
        return Err(
            "COLOR_0 is VEC3, so the team-colour amount in alpha was dropped: export with export_vertex_color=\"ACTIVE\""
                .into(),
        );
    }
    let indices = reader.read_indices().ok_or("the mesh has no index buffer")?;

    let base = level.pos.len() as u32;
    // A node scale would squash the normals; the inverse transpose is
    // the one that stays perpendicular.
    let normal_xf = Mat3::from_mat4(xf).inverse().transpose();
    level.pos.extend(pos.map(|p| xf.transform_point3(Vec3::from_array(p))));
    level.nrm.extend(nrm.map(|n| (normal_xf * Vec3::from_array(n)).normalize_or_zero()));
    level.col.extend(colors.into_rgba_f32());
    for uv in uv.into_f32() {
        level.part.push(uv[0]);
        level.pivot.push(uv[1]);
    }
    let n = level.pos.len();
    if level.nrm.len() != n || level.col.len() != n || level.part.len() != n {
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

/// Model space (metres, feet at 0, shield on +X) to engine local space
/// (feet at -half_height, shield on -X).
fn to_local(level: &mut Level, scale: f32, half_height: f32, mirror: bool) {
    let sx = if mirror { -scale } else { scale };
    for p in &mut level.pos {
        *p = Vec3::new(sx * p.x, scale * p.y - half_height, scale * p.z);
    }
    if mirror {
        for n in &mut level.nrm {
            n.x = -n.x;
        }
        // Mirroring turns every triangle inside out, and back faces are
        // culled.
        for t in level.idx.chunks_mut(3) {
            t.swap(1, 2);
        }
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
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, level.col.clone())
        .with_inserted_indices(Indices::U32(level.idx.clone()))
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

/// Body and legs, the parts a figure's mass sits in. The last level is
/// one block stack, and an arm holding a sword out in front would double
/// the width of every box if it counted toward them.
fn is_trunk(part: f32) -> bool {
    matches!(part as u32, 0 | 2 | 3)
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
