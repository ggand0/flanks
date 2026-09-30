//! Unit render data built on the GPU (docs/plans/gpu-render-data-item2.md).
//!
//! The CPU packs one compact snapshot per soldier once per sim tick and
//! hands it to the render world by swapping buffers. Every frame a compute
//! pass (`shaders/unit_build.wgsl`) does what `sync_instance_data` does on
//! the CPU: interpolate, cull, pick the detail level, run the pose
//! smoothers, write the 64 byte instance record and append the soldier to
//! his kind-by-level index list. The list counts become indirect draw
//! arguments and each bucket draws with one pulled, non-instanced draw.
//!
//! A third pass (`shaders/unit_pose_pass.wgsl`) then poses every drawn
//! soldier once, so each corner of his mesh only reads his pose and places
//! itself, instead of working the gait, the joints and the bow out again
//! for every corner. `FL_POSE_PASS=0` poses per corner, for A/B runs.
//!
//! This is the default path. `FL_GPU_SYNC=0` keeps the CPU path in
//! render_units.rs, which stays complete as the A/B and the fallback.

use bevy::asset::{RenderAssetUsages, embedded_asset, load_embedded_asset};
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::diagnostic::FrameCount;
use bevy::math::primitives::ViewFrustum;
use bevy::prelude::*;
use bevy::render::{
    Extract, ExtractSchedule, MainWorld, Render, RenderApp, RenderStartup, RenderSystems,
    diagnostic::RecordDiagnostics,
    gpu_readback::{Readback, ReadbackComplete},
    render_asset::RenderAssets,
    render_resource::*,
    globals::{GlobalsBuffer, GlobalsUniform},
    renderer::{RenderAdapter, RenderContext, RenderDevice, RenderQueue},
    storage::{GpuShaderBuffer, ShaderBuffer},
    sync_world::RenderEntity,
    texture::GpuImage,
};
use bytemuck::{Pod, Zeroable};

use crate::render_units::{
    BAND_FIGHTING, BOW_FALL_S, BOW_RISE_S, BOW_WALK_MS, CELEBRATE_BASE, CORPSE_CAP, CULL_RADIUS,
    Corpses, CustomPipeline, ExtractedAtlas, FOLLOW_BASE, FOLLOW_S, FOLLOW_SPAN, HOLD_S,
    ExtractedBucket, InstanceBucket, InstanceData, LodBands, LodConfig, NUM_BUCKETS, NUM_LODS,
    RANGED_BASE, RELEASE_S, RELOAD_S, REWIND_S, RenderCounts, RigBuffer, SYNC_CHUNK, UnitRig,
    celebrate_progress,
    shadow_level, stance_tier, wall_signal,
};
use crate::render_units_shadow::{CAST_LODS, unit_shadows};
use crate::units::Units;
use crate::unit_types::NUM_KINDS;

/// Whether the GPU path is on. Decided once at startup from `FL_GPU_SYNC`
/// and the device's capabilities (`GpuUnitRenderPlugin::finish`).
#[derive(Resource, Clone, Copy)]
pub struct GpuSyncConfig {
    pub enabled: bool,
    /// FL_GPU_CHECK=1: the CPU sweep runs too and its per-bucket counts
    /// are compared with the GPU's on the same frame.
    pub check: bool,
    /// The pose pass poses each drawn soldier once. `FL_POSE_PASS=0`: each
    /// corner poses itself.
    pub pose_pass: bool,
}

pub fn gpu_sync(cfg: Res<GpuSyncConfig>) -> bool {
    cfg.enabled
}

/// The CPU sweep runs on the CPU path, and next to the GPU path in check mode.
pub fn cpu_sweep(cfg: Res<GpuSyncConfig>) -> bool {
    !cfg.enabled || cfg.check
}

/// Words of the counts readback: 16 bucket totals, 16 fallen per bucket,
/// the frame stamp, the soldier count, two spare, then the 16 caster list
/// counts (cascade * 4 + kind).
const READBACK_WORDS: usize = 52;

/// Sun shadow cascades the build fills caster lists for, Bevy's maximum.
pub const MAX_CASCADES: usize = bevy::pbr::MAX_CASCADES_PER_LIGHT;

/// Caster lists: one per cascade and kind.
const CASTER_LISTS: usize = MAX_CASCADES * NUM_KINDS;

/// The selection rings' draw arguments, after the camera's buckets and
/// the caster lists (selection_rings.rs).
pub const RING_ARG: usize = NUM_BUCKETS + CASTER_LISTS;

/// Indirect draw arguments: the camera's buckets, the caster lists, the
/// selection rings.
pub const DRAW_ARGS: usize = RING_ARG + 1;

/// The pose pass's dispatch arguments, one per kind, after the draws'.
const POSE_ARG: usize = DRAW_ARGS;

/// Build counters: per bucket the soldiers and the fallen, the caster
/// lists, the ring count, then the pose slots taken per kind.
const COUNTERS: usize = 2 * NUM_BUCKETS + CASTER_LISTS + 1 + NUM_KINDS;

/// Pose slots (four floats each) per soldier of a kind without a bow rig,
/// and of one with it (`POSE_SLOTS` and `POSE_SLOTS_BOW` in
/// shaders/unit_pose.wgsl).
const POSE_SLOTS: u32 = 17;
const POSE_SLOTS_BOW: u32 = 45;

/// Bytes of one set of per-bucket list entries in the bucket table: the
/// camera's set, then one per cascade. A pulled draw binds one set, at a
/// multiple of the storage offset alignment every device allows.
const BUCKET_SET_BYTES: u64 = 256;

/// Per-tick snapshot of one soldier, 56 bytes, every field exact. All
/// scalars, so the WGSL struct (`Soldier` in unit_build.wgsl) has the same
/// stride. `prev` rides along on purpose: the death sweep swap-removes
/// soldiers, so a previous position kept GPU-side by index would streak
/// swapped soldiers across the field for one tick.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuSoldier {
    pos: [f32; 3],
    prev: [f32; 3],
    yaw: f32,
    yaw_prev: f32,
    /// kind | swing << 8 | swing_t << 16 | flash << 24
    a: u32,
    /// group | death_t << 24
    b: u32,
    /// rgb = base color, a = the stable anim seed.
    color: [f32; 4],
}

/// Per-regiment pose signals for one frame (`Regiment` in unit_build.wgsl).
#[derive(Clone, Copy, Pod, Zeroable, Default)]
#[repr(C)]
pub struct RegimentRecord {
    stance: f32,
    celebrate: f32,
    walled: f32,
    flags: u32,
}

const REG_BROKEN: u32 = 1;
/// Selected and able to take orders: rings in the selection colour.
const REG_SELECTED: u32 = 2;
/// The enemy regiment under the cursor: red rings, the attack preview.
const REG_HOVERED: u32 = 4;
/// The player's own regiment under the cursor or its card: faint rings.
const REG_HOVER_OWN: u32 = 8;

/// The compute pass uniform (`Params` in unit_build.wgsl, same field order).
#[derive(ShaderType, Clone, Copy, Default)]
pub struct BuildParams {
    planes: [Vec4; 6],
    cam_pos: Vec3,
    alpha: f32,
    k_walk: f32,
    k_band: f32,
    k_wall: f32,
    inv_dt: f32,
    n: u32,
    n_regs: u32,
    cull: u32,
    corpse_base: u32,
    corpse_len: UVec4,
    corpse_cap: u32,
    /// The frame this pass belongs to, stamped into the readback.
    frame: u32,
    /// Seconds since the last frame, for the gait phase and the attack.
    dt: f32,
    /// render_units.rs BAND_FIGHTING.
    fighting: f32,
    /// [kind * 3 + set]: set 0 fine, 1 coarse, 2 plain.
    bands: [Vec4; 12],
    windup: Vec4,
    /// draw_ticks, death_ticks, hit_stagger_ticks, celebrate_base
    consts: Vec4,
    /// FOLLOW_S, FOLLOW_BASE, FOLLOW_SPAN, REWIND_S (render_units.rs).
    attack: Vec4,
    /// RELEASE_S, RELOAD_S, HOLD_S, BOW_WALK_MS (render_units.rs).
    shot: Vec4,
    /// BOW_RISE_S, BOW_FALL_S, CANCEL_TICKS, RANGED_BASE.
    bow: Vec4,
    /// x = first index slot of the bucket, y = mesh corners per soldier.
    buckets: [UVec4; NUM_BUCKETS],
    /// The camera's forward axis, for view depth.
    cam_fwd: Vec4,
    /// Where a soldier's shadow can fall: xz = the centre of the ground his
    /// cull sphere shades, from his position, w = its radius.
    shadow_reach: Vec4,
    /// The view depths each cascade is sampled at, blend band included.
    cascade_near: Vec4,
    cascade_far: Vec4,
    /// Per cascade, one entry per kind: the first slot of the caster list.
    shadow_lists: [UVec4; MAX_CASCADES],
    /// Per cascade and kind: corners per soldier of the level he casts with.
    shadow_corners: [UVec4; MAX_CASCADES],
    /// Cascades the build fills, 0 with shadows off.
    n_cascades: u32,
    /// render_units_shadow.rs CAST_LODS.
    cast_lods: u32,
    /// The first slot of the ring list in the index list.
    ring_base: u32,
    /// Per kind: the first entry of its region of the pose sources.
    pose_src_base: UVec4,
    /// 1 with the pose pass on.
    pose_pass: u32,
}

/// The pose pass uniform (`Params` in unit_pose_pass.wgsl).
#[derive(ShaderType, Clone, Copy, Default)]
struct PoseParams {
    src_base: UVec4,
    stride: UVec4,
}

const _: () = assert!(NUM_KINDS == 4, "BuildParams and PoseParams pack per-kind values in vec4s");
const _: () = assert!(COUNTERS == 53, "unit_build.wgsl sizes `counts` for 53");
const _: () = assert!(NUM_BUCKETS == 16, "unit_build.wgsl sizes its counters for 16 buckets");
const _: () = assert!(MAX_CASCADES == 4, "unit_build.wgsl sizes its cascades for 4");
const _: () = assert!(
    NUM_BUCKETS * 16 == BUCKET_SET_BYTES as usize,
    "one bucket table set is 16 entries of 16 bytes"
);

/// The soldier snapshot on the main world side. `records` is double
/// buffered against the render world: extract swaps the two, so the
/// handoff is a pointer exchange.
#[derive(Resource, Default)]
pub struct SoldierSnapshot {
    pub records: Vec<GpuSoldier>,
    pub n: u32,
    pub kind_counts: [u32; NUM_KINDS],
    /// A new snapshot waits for extract.
    pub fresh: bool,
    pub pack_ms: f32,
}

/// The per-frame inputs of the compute pass, main world side.
#[derive(Resource, Default)]
pub struct GpuFrameInput {
    pub params: BuildParams,
    pub regiments: Vec<RegimentRecord>,
    pub lod_debug: bool,
    /// Per cascade and kind: the detail level soldiers cast with.
    pub shadow_levels: [[u32; NUM_KINDS]; MAX_CASCADES],
}

/// Pack the live soldier columns into the snapshot. Runs after every
/// writer of the columns (the fixed loop, the deployment drag) and only
/// when `Units` changed, so frames without a tick pack nothing. Parallel
/// on the compute pool. Not reachable from the tick job, so a plain scope
/// is correct here.
fn pack_soldier_snapshot(
    units: Res<Units>,
    settings: Res<crate::settings::Settings>,
    mut snap: ResMut<SoldierSnapshot>,
) {
    if !units.is_changed() {
        return;
    }
    // The Hit flash setting: off packs every flash as zero.
    let flash = settings.interface.hit_flash;
    let t0 = std::time::Instant::now();
    let n = units.len();
    snap.records.resize(n, GpuSoldier::zeroed());
    let units = &*units;
    let records = &mut snap.records;
    let chunk_counts: Vec<[u32; NUM_KINDS]> = bevy::tasks::ComputeTaskPool::get().scope(|scope| {
        for (ci, out) in records.chunks_mut(SYNC_CHUNK).enumerate() {
            scope.spawn(async move { pack_chunk(units, ci * SYNC_CHUNK, out, flash) });
        }
    });
    let mut kind_counts = [0u32; NUM_KINDS];
    for counts in chunk_counts {
        for (total, c) in kind_counts.iter_mut().zip(counts) {
            *total += c;
        }
    }
    snap.n = n as u32;
    snap.kind_counts = kind_counts;
    snap.fresh = true;
    snap.pack_ms = t0.elapsed().as_secs_f32() * 1000.0;
}

fn pack_chunk(units: &Units, start: usize, out: &mut [GpuSoldier], flash: bool) -> [u32; NUM_KINDS] {
    let mut kind_counts = [0u32; NUM_KINDS];
    for (j, rec) in out.iter_mut().enumerate() {
        let i = start + j;
        let kind = units.kind[i] as u32;
        kind_counts[kind as usize] += 1;
        let group = units.group[i];
        debug_assert!(group < 1 << 24, "regiment index must fit 24 bits");
        *rec = GpuSoldier {
            pos: units.pos[i].to_array(),
            prev: units.pos_prev[i].to_array(),
            yaw: units.yaw[i],
            yaw_prev: units.yaw_prev[i],
            a: kind
                | (units.swing[i] as u32) << 8
                | (units.swing_t[i] as u32) << 16
                | if flash { (units.flash[i] as u32) << 24 } else { 0 },
            b: (group & 0x00ff_ffff) | (units.death_t[i] as u32) << 24,
            color: units.color[i],
        };
    }
    kind_counts
}

/// The per-frame parameters: camera, clock, level bands, one record per
/// regiment. This is the top of `sync_instance_data` without the per
/// soldier sweep.
#[allow(clippy::too_many_arguments)] // bevy system params
fn build_frame_params(
    units: Res<Units>,
    selection: Res<crate::orders::Selection>,
    hover: Res<crate::orders::Hover>,
    groups: Res<crate::orders::Groups>,
    time: Res<Time>,
    fixed_time: Res<Time<Fixed>>,
    lod_cfg: Res<LodConfig>,
    camera: Query<(Entity, &Camera, &Projection, &Transform), With<Camera3d>>,
    lights: Query<(
        &DirectionalLight,
        &GlobalTransform,
        &bevy::light::CascadeShadowConfig,
        &bevy::light::Cascades,
    )>,
    snap: Res<SoldierSnapshot>,
    mut frame: ResMut<GpuFrameInput>,
    mut counts: ResMut<RenderCounts>,
    mut no_cull: Local<Option<bool>>,
    frame_count: Res<FrameCount>,
    mut logged: Local<[Option<[u32; NUM_KINDS]>; MAX_CASCADES]>,
) {
    let t0 = std::time::Instant::now();
    let Ok((cam_entity, cam, projection, cam_tf)) = camera.single() else {
        return;
    };
    // Fresh frustum from THIS frame's camera state, as the CPU pass does.
    let clip_from_world = projection.get_clip_from_view() * cam_tf.to_matrix().inverse();
    let frustum = ViewFrustum::from_clip_from_world(&clip_from_world);
    let cull = !*no_cull.get_or_insert_with(|| std::env::var("FL_NO_CULL").is_ok());
    let px_per_unit = match (projection, cam.physical_viewport_size()) {
        (Projection::Perspective(p), Some(size)) => {
            size.y as f32 / (2.0 * (p.fov * 0.5).tan())
        }
        _ => 0.0,
    };
    let bands = LodBands::new(&lod_cfg, px_per_unit);

    let has_sel = selection.regiments.iter().any(|s| *s);
    frame.regiments.clear();
    frame
        .regiments
        .extend(groups.list.iter().enumerate().map(|(g, gd)| {
            let mut flags = 0;
            if gd.state.is_broken() {
                flags |= REG_BROKEN;
            }
            if has_sel
                && selection.regiments.get(g).copied().unwrap_or(false)
                && !gd.state.is_broken()
            {
                flags |= REG_SELECTED;
            }
            if hover.enemy == Some(g as u32) {
                flags |= REG_HOVERED;
            }
            if hover.own == Some(g as u32) {
                flags |= REG_HOVER_OWN;
            }
            RegimentRecord {
                stance: stance_tier(gd),
                celebrate: celebrate_progress(gd),
                walled: wall_signal(gd),
                flags,
            }
        }));

    let dt = time.delta_secs();
    let mut p = BuildParams::default();
    for (plane, half_space) in p.planes.iter_mut().zip(&frustum.half_spaces) {
        *plane = half_space.normal_d();
    }
    p.cam_pos = cam_tf.translation;
    p.alpha = fixed_time.overstep_fraction();
    p.k_walk = (dt / 0.25).min(1.0);
    p.k_band = (dt / 0.35).min(1.0);
    p.k_wall = (dt / 0.5).min(1.0);
    p.inv_dt = 1.0 / fixed_time.timestep().as_secs_f32().max(1e-6);
    p.n = snap.n;
    p.n_regs = groups.list.len() as u32;
    p.cull = cull as u32;
    for kind in 0..NUM_KINDS {
        for (set, src) in [&bands.fine, &bands.coarse, &bands.plain].into_iter().enumerate() {
            let t = src[kind];
            p.bands[kind * 3 + set] = Vec4::new(t[0], t[1], t[2], 0.0);
        }
    }
    p.windup = Vec4::from_array(std::array::from_fn(|k| {
        crate::unit_types::TYPES[k].windup_ticks as f32
    }));
    p.consts = Vec4::new(
        crate::unit_types::missile::DRAW_TICKS as f32,
        crate::sim::damage::DEATH_TICKS as f32,
        crate::sim::damage::HIT_STAGGER_TICKS as f32,
        CELEBRATE_BASE,
    );
    p.dt = dt;
    p.fighting = BAND_FIGHTING;
    p.attack = Vec4::new(FOLLOW_S, FOLLOW_BASE, FOLLOW_SPAN, REWIND_S);
    p.shot = Vec4::new(RELEASE_S, RELOAD_S, HOLD_S, BOW_WALK_MS);
    p.bow = Vec4::new(
        BOW_RISE_S,
        BOW_FALL_S,
        crate::unit_types::missile::CANCEL_TICKS as f32,
        RANGED_BASE,
    );
    p.frame = frame_count.0;
    // The sun's cascades for this camera, as the shadow pass draws them
    // this frame. A soldier casts into a cascade when the ground his shadow
    // can fall on is in view at depths the cascade serves, with the level
    // that shows all the detail the cascade's filtered texels can
    // (render_units.rs `shadow_level`).
    p.cast_lods = CAST_LODS as u32;
    p.cam_fwd = cam_tf.forward().as_vec3().extend(0.0);
    let sun = lights
        .iter()
        .find(|(light, ..)| light.shadow_maps_enabled && unit_shadows())
        .and_then(|(_, sun, config, c)| Some((sun, config, c.cascades.get(&cam_entity)?)));
    if let Some((sun, config, cascades)) = sun {
        // A point at height h above the ground shades the ground h * run
        // away from the sun. A soldier stands half his height above his
        // feet, so the top of his cull sphere is at most CULL_RADIUS plus
        // the tallest kind's half height above the ground, and the sphere's
        // shadow lies in a sphere around the midpoint of that run. A sun
        // under 5 degrees is taken at 5.
        let to_sun = sun.back().as_vec3();
        let run = -Vec2::new(to_sun.x, to_sun.z) / to_sun.y.max(5f32.to_radians().sin());
        let top = CULL_RADIUS
            + (0..NUM_KINDS).map(crate::unit_types::half_height).fold(0.0, f32::max);
        let half = run * (0.5 * top);
        p.shadow_reach = Vec4::new(half.x, 0.0, half.y, CULL_RADIUS + half.length());
        // Bevy samples cascade c up to its far bound, and blends in from
        // the previous one's far bound less the overlap.
        let mut near = 0.0;
        for (c, cascade) in cascades.iter().take(MAX_CASCADES).enumerate() {
            let far = config.bounds[c];
            p.cascade_near[c] = near;
            p.cascade_far[c] = far;
            let levels = std::array::from_fn(|kind| {
                shadow_level(&lod_cfg, kind, cascade.texel_size) as u32
            });
            if logged[c] != Some(levels) {
                logged[c] = Some(levels);
                info!(
                    "sun shadow cascade {c}: {near:.0} to {far:.0} m, texel {:.1} cm, soldiers cast with levels {levels:?}",
                    cascade.texel_size * 100.0
                );
            }
            frame.shadow_levels[c] = levels;
            p.n_cascades = c as u32 + 1;
            near = (1.0 - config.overlap_proportion) * far;
        }
    }
    frame.params = p;
    frame.lod_debug = lod_cfg.debug;

    counts.total = units.len();
    let pack_ms = if snap.fresh { snap.pack_ms } else { 0.0 };
    counts.sync_ms = pack_ms + t0.elapsed().as_secs_f32() * 1000.0;
}

/// The counts readback buffer: a storage buffer asset the compute pass
/// writes and Bevy's readback plugin copies back, one to two frames late.
/// Debug and overlay only, nothing in the game reads it.
#[derive(Resource)]
pub struct CountsReadback {
    handle: Handle<ShaderBuffer>,
}

fn setup_counts_readback(mut commands: Commands, mut buffers: ResMut<Assets<ShaderBuffer>>) {
    let mut buffer = ShaderBuffer::with_size(READBACK_WORDS * 4, RenderAssetUsages::RENDER_WORLD);
    buffer.buffer_description.label = Some("unit bucket counts readback");
    buffer.buffer_description.usage =
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
    let handle = buffers.add(buffer);
    commands
        .spawn(Readback::buffer(handle.clone()))
        .observe(on_counts_readback);
    commands.insert_resource(CountsReadback { handle });
}

#[derive(Default)]
struct CheckStats {
    compared: u32,
    mismatched: u32,
    worst: u32,
    detailed: u32,
    /// Readbacks since the caster counts were last logged.
    since_casters: u32,
}

/// A readback landed: fill the overlay counts, and in check mode compare
/// them with what the CPU sweep recorded for the same frame.
fn on_counts_readback(
    event: On<ReadbackComplete>,
    cfg: Res<GpuSyncConfig>,
    mut counts: ResMut<RenderCounts>,
    mut stats: Local<CheckStats>,
) {
    let words: Vec<u32> = event
        .data
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if words.len() < READBACK_WORDS {
        return;
    }
    let fallen: [u32; NUM_BUCKETS] = std::array::from_fn(|b| words[NUM_BUCKETS + b]);
    let living: [u32; NUM_BUCKETS] = std::array::from_fn(|b| words[b].saturating_sub(fallen[b]));
    let frame = words[2 * NUM_BUCKETS];
    counts.set_from_buckets(&living, &fallen);
    // The sun shadow casters per cascade, one count per kind, about every
    // two seconds next to the periodic log.
    stats.since_casters += 1;
    if stats.since_casters >= 120 && words[36..].iter().any(|&n| n > 0) {
        stats.since_casters = 0;
        let casters: Vec<&[u32]> = words[36..].chunks(NUM_KINDS).collect();
        info!("  unit shadow casters per cascade, per kind: {casters:?}");
    }
    if !cfg.check {
        return;
    }
    let Some(&(_, cpu_living, cpu_fallen)) = counts.check.iter().find(|e| e.0 == frame) else {
        return;
    };
    let mut worst = 0u32;
    for b in 0..NUM_BUCKETS {
        worst = worst
            .max(living[b].abs_diff(cpu_living[b]))
            .max(fallen[b].abs_diff(cpu_fallen[b]));
    }
    stats.compared += 1;
    if worst > 0 {
        stats.mismatched += 1;
        stats.worst = stats.worst.max(worst);
        if stats.detailed < 5 {
            stats.detailed += 1;
            warn!(
                "[gpu check] frame {frame}: gpu living {living:?} fallen {fallen:?} | cpu living {cpu_living:?} fallen {cpu_fallen:?}"
            );
        }
    }
    if stats.compared.is_multiple_of(300) {
        info!(
            "[gpu check] {} frames compared, {} differed, worst bucket difference {}",
            stats.compared, stats.mismatched, stats.worst
        );
    }
}

/// One mesh corner as the pulled draw reads it from a storage buffer
/// (unit_instancing.wgsl `PullVertex`, same field order and padding).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct PullVertex {
    position: [f32; 3],
    part: f32,
    normal: [f32; 3],
    pivot: f32,
    /// Vertex colour, unorm8 rgba.
    color: u32,
    /// Atlas coordinates, unorm16 each, then padding to the WGSL struct's
    /// 48 bytes.
    atlas_uv: u32,
    pad: [u32; 2],
}

/// A bucket's level mesh with its index list expanded: the vertex shader
/// finds a corner by `vertex_index % len`, no index buffer. Any `Mesh`
/// with position, normal, the part/pivot UV and vertex color qualifies,
/// so an imported model set plugs in unchanged.
#[derive(Component)]
pub struct PullMesh {
    corners: Vec<PullVertex>,
    bucket: usize,
}

fn pack_unorm8(v: [f32; 4]) -> u32 {
    v.iter()
        .enumerate()
        .map(|(k, c)| ((c.clamp(0.0, 1.0) * 255.0).round() as u32) << (8 * k))
        .sum()
}

fn pack_unorm16(v: [f32; 2]) -> u32 {
    let q = |c: f32| (c.clamp(0.0, 1.0) * 65535.0).round() as u32;
    q(v[0]) | q(v[1]) << 16
}

impl PullMesh {
    pub fn from_mesh(mesh: &Mesh, bucket: usize) -> Option<Self> {
        use bevy::mesh::VertexAttributeValues as V;
        let Some(V::Float32x3(pos)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else {
            return None;
        };
        let Some(V::Float32x3(nrm)) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL) else {
            return None;
        };
        let Some(V::Float32x2(uv)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0) else {
            return None;
        };
        let Some(V::Float32x4(col)) = mesh.attribute(Mesh::ATTRIBUTE_COLOR) else {
            return None;
        };
        let Some(V::Float32x2(atlas_uv)) = mesh.attribute(Mesh::ATTRIBUTE_UV_1) else {
            return None;
        };
        let corners = mesh.indices()?.iter().map(|i| PullVertex {
            position: pos[i],
            part: uv[i][0],
            normal: nrm[i],
            pivot: uv[i][1],
            color: pack_unorm8(col[i]),
            atlas_uv: pack_unorm16(atlas_uv[i]),
            pad: [0; 2],
        });
        Some(Self {
            corners: corners.collect(),
            bucket,
        })
    }
}

/// Render-world side of `PullMesh`: the corners in a storage buffer.
#[derive(Component)]
pub struct PullMeshGpu {
    vertices: Buffer,
    /// Corners per soldier (the level's index count).
    pub count: u32,
    pub bucket: usize,
}

/// Uploads each pulled bucket's mesh once. The render entity persists,
/// so later frames find the buffer in place and do nothing.
fn extract_pull_meshes(
    main_entities: Extract<Query<(&RenderEntity, &PullMesh)>>,
    uploaded: Query<(), With<PullMeshGpu>>,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    for (render_entity, mesh) in &main_entities {
        let e = render_entity.id();
        if uploaded.contains(e) {
            continue;
        }
        commands.entity(e).insert(PullMeshGpu {
            vertices: render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("unit pull mesh"),
                contents: bytemuck::cast_slice(&mesh.corners),
                usage: BufferUsages::STORAGE,
            }),
            count: mesh.corners.len() as u32,
            bucket: mesh.bucket,
        });
    }
}

/// Group 3 of a pulled bucket's draw, rebuilt whenever the shared buffers
/// it binds are reallocated.
#[derive(Component)]
pub struct PulledBucketGpu {
    pub bind_group: BindGroup,
    /// The same group for drawing into each sun shadow cascade: it binds
    /// that cascade's set of the bucket table, so the bucket's draw reads
    /// the cascade's caster list for its kind.
    pub shadow: Vec<BindGroup>,
    generation: u32,
    /// Bound to its own atlas, or has none. Otherwise it waits on the upload.
    atlas_settled: bool,
    pub bucket: u32,
}

/// The extracted inputs, render world side.
#[derive(Resource, Default)]
pub struct GpuUnitInput {
    enabled: bool,
    /// The pose pass is on (`GpuSyncConfig::pose_pass`).
    pub pose_pass: bool,
    records: Vec<GpuSoldier>,
    fresh: bool,
    n: u32,
    kind_counts: [u32; NUM_KINDS],
    params: BuildParams,
    regiments: Vec<RegimentRecord>,
    pub lod_debug: bool,
    /// Per cascade and kind: the detail level soldiers cast with.
    pub shadow_levels: [[u32; NUM_KINDS]; MAX_CASCADES],
    /// Bodies that fell since the last frame: slot in the corpse region
    /// and the frozen record. Sorted by slot in prepare.
    corpse_pending: Vec<(u32, InstanceData)>,
    corpse_len: [u32; NUM_KINDS],
    /// Scratch for coalesced corpse uploads.
    corpse_run: Vec<InstanceData>,
    readback: Option<Handle<ShaderBuffer>>,
}

/// Swap the snapshot into the render world and copy the small per-frame
/// inputs. Mutable main world access is what makes the swap possible.
fn extract_gpu_units(mut main_world: ResMut<MainWorld>, mut input: ResMut<GpuUnitInput>) {
    let cfg = *main_world.resource::<GpuSyncConfig>();
    let enabled = cfg.enabled;
    input.enabled = enabled;
    input.pose_pass = cfg.pose_pass;
    {
        // Drained in every mode, so the list cannot grow on the CPU path.
        let mut corpses = main_world.resource_mut::<Corpses>();
        input.corpse_pending.clear();
        std::mem::swap(&mut corpses.pending, &mut input.corpse_pending);
        input.corpse_len = std::array::from_fn(|k| corpses.len(k) as u32);
    }
    if !enabled {
        return;
    }
    {
        let mut snap = main_world.resource_mut::<SoldierSnapshot>();
        if snap.fresh {
            snap.fresh = false;
            std::mem::swap(&mut snap.records, &mut input.records);
            input.fresh = true;
            input.n = snap.n;
            input.kind_counts = snap.kind_counts;
        }
    }
    let frame = main_world.resource::<GpuFrameInput>();
    input.params = frame.params;
    input.regiments.clear();
    input.regiments.extend_from_slice(&frame.regiments);
    input.lod_debug = frame.lod_debug;
    input.shadow_levels = frame.shadow_levels;
    if input.readback.is_none() {
        input.readback = main_world
            .get_resource::<CountsReadback>()
            .map(|r| r.handle.clone());
    }
}

/// The compute pipelines and their bind group layouts.
#[derive(Resource)]
pub(crate) struct GpuUnitPipelines {
    build: CachedComputePipelineId,
    finalize: CachedComputePipelineId,
    layout: BindGroupLayoutDescriptor,
    pose: CachedComputePipelineId,
    /// Group 0 of the pose pass: its uniform, the records, the pose
    /// sources, the build counters, Bevy's globals.
    pose_layout: BindGroupLayoutDescriptor,
    /// Groups 1 and 2 of the pose pass: the pose shader binds its kind's
    /// rig at group 3, where the draws have it.
    empty_layout: BindGroupLayoutDescriptor,
    /// Group 3 of the pose pass, per kind: the rig, the shot tables, the
    /// kind's pose buffer, the kind.
    kind_layout: BindGroupLayoutDescriptor,
}

fn init_gpu_unit_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "unit build layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer::<BuildParams>(false),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
            ),
        ),
    );
    let shader = load_embedded_asset!(asset_server.as_ref(), "shaders/unit_build.wgsl");
    let descriptor = |label: &'static str, entry: &'static str| ComputePipelineDescriptor {
        label: Some(label.into()),
        layout: vec![layout.clone()],
        immediate_size: 0,
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some(entry.into()),
        zero_initialize_workgroup_memory: false,
    };
    let build = pipeline_cache.queue_compute_pipeline(descriptor("unit build", "build"));
    let finalize =
        pipeline_cache.queue_compute_pipeline(descriptor("unit build finalize", "finalize"));

    let pose_layout = BindGroupLayoutDescriptor::new(
        "unit pose layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer::<PoseParams>(false),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::uniform_buffer::<GlobalsUniform>(false),
            ),
        ),
    );
    let empty_layout = BindGroupLayoutDescriptor::new("unit pose empty layout", &[]);
    let kind_layout = BindGroupLayoutDescriptor::new(
        "unit pose kind layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (6, binding_types::uniform_buffer_sized(false, None)),
                (7, binding_types::storage_buffer_read_only_sized(false, None)),
                (9, binding_types::storage_buffer_sized(false, None)),
                (10, binding_types::uniform_buffer::<UVec4>(false)),
            ),
        ),
    );
    let pose = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("unit pose".into()),
        layout: vec![
            pose_layout.clone(),
            empty_layout.clone(),
            empty_layout.clone(),
            kind_layout.clone(),
        ],
        immediate_size: 0,
        shader: load_embedded_asset!(asset_server.as_ref(), "shaders/unit_pose_pass.wgsl"),
        shader_defs: vec!["UNIT_POSE_WRITE".into()],
        entry_point: Some("pose".into()),
        zero_initialize_workgroup_memory: false,
    });
    commands.insert_resource(GpuUnitPipelines {
        build,
        finalize,
        layout,
        pose,
        pose_layout,
        empty_layout,
        kind_layout,
    });
}

/// The GPU buffers of one battle size. Reallocated as a whole when a
/// bigger army arrives (a soldier count only shrinks within a battle).
pub struct UnitAlloc {
    soldiers: Buffer,
    live_cap: usize,
    /// `[0, live_cap)` is rewritten every frame, the region after it
    /// holds the fallen, one frozen record each, per kind.
    pub records: Buffer,
    smooth: Buffer,
    /// Per bucket, `kind_cap[kind] + CORPSE_CAP` slots: a soldier of a
    /// kind can only land in one of that kind's levels. Then as many per
    /// cascade and kind for the casters.
    pub index_list: Buffer,
    kind_cap: [usize; NUM_KINDS],
    bases: [u32; NUM_BUCKETS],
    shadow_bases: [[u32; NUM_KINDS]; MAX_CASCADES],
    /// The ring list: `live_cap` slots after the caster lists, a soldier
    /// of a selected or hovered regiment each (selection_rings.rs).
    pub ring_base: u32,
    counts: Buffer,
    pub args: Buffer,
    regiments: Buffer,
    regiments_cap: usize,
    pub bucket_info: Buffer,
    /// Per pose slot, the record it poses: a region per kind of
    /// `kind_cap + CORPSE_CAP` entries, since a soldier or a body takes at
    /// most one slot a frame (unit_build.wgsl `list_entry`).
    pose_src: Buffer,
    pose_src_base: [u32; NUM_KINDS],
    /// Per kind, each drawn soldier's pose (shaders/unit_pose.wgsl):
    /// `pose_stride` slots of four floats per pose slot. One buffer per
    /// kind keeps each binding a kind's size.
    pub poses: [Buffer; NUM_KINDS],
    pose_stride: [u32; NUM_KINDS],
}

fn storage_buffer(device: &RenderDevice, label: &str, bytes: usize, extra: BufferUsages) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.max(16) as u64,
        usage: BufferUsages::STORAGE | extra,
        mapped_at_creation: false,
    })
}

impl UnitAlloc {
    /// `pose_stride` is zero for every kind with the pose pass off.
    fn new(
        device: &RenderDevice,
        live_cap: usize,
        kind_cap: [usize; NUM_KINDS],
        pose_stride: [u32; NUM_KINDS],
    ) -> Self {
        let mut bases = [0u32; NUM_BUCKETS];
        let mut total = 0usize;
        for kind in 0..NUM_KINDS {
            for lod in 0..NUM_LODS {
                bases[kind * NUM_LODS + lod] = total as u32;
                total += kind_cap[kind] + CORPSE_CAP;
            }
        }
        let mut shadow_bases = [[0u32; NUM_KINDS]; MAX_CASCADES];
        for cascade in &mut shadow_bases {
            for (kind, base) in cascade.iter_mut().enumerate() {
                *base = total as u32;
                total += kind_cap[kind] + CORPSE_CAP;
            }
        }
        let ring_base = total as u32;
        total += live_cap;
        let regiments_cap = 256;
        let mut pose_src_base = [0u32; NUM_KINDS];
        let mut sources = 0usize;
        for (kind, base) in pose_src_base.iter_mut().enumerate() {
            *base = sources as u32;
            sources += kind_cap[kind] + CORPSE_CAP;
        }
        let limit = device.limits().max_storage_buffer_binding_size as usize;
        let poses = std::array::from_fn(|kind| {
            let bytes = (kind_cap[kind] + CORPSE_CAP) * pose_stride[kind] as usize * 16;
            if bytes > limit {
                error!("unit poses of kind {kind}: {bytes} bytes, past the device's {limit} per binding");
            }
            storage_buffer(device, "unit poses", bytes, BufferUsages::empty())
        });
        Self {
            soldiers: storage_buffer(
                device,
                "unit snapshot",
                live_cap * size_of::<GpuSoldier>(),
                BufferUsages::COPY_DST,
            ),
            live_cap,
            records: storage_buffer(
                device,
                "unit records",
                (live_cap + NUM_KINDS * CORPSE_CAP) * size_of::<crate::render_units::InstanceData>(),
                BufferUsages::COPY_DST,
            ),
            smooth: storage_buffer(device, "unit smoothing", live_cap * 40, BufferUsages::empty()),
            index_list: storage_buffer(device, "unit index list", total * 4, BufferUsages::empty()),
            kind_cap,
            bases,
            shadow_bases,
            ring_base,
            counts: storage_buffer(device, "unit bucket counts", COUNTERS * 4, BufferUsages::COPY_DST),
            args: storage_buffer(
                device,
                "unit draw args",
                (DRAW_ARGS + NUM_KINDS) * 16,
                BufferUsages::INDIRECT | BufferUsages::COPY_DST,
            ),
            regiments: storage_buffer(
                device,
                "unit regiments",
                regiments_cap * size_of::<RegimentRecord>(),
                BufferUsages::COPY_DST,
            ),
            regiments_cap,
            bucket_info: storage_buffer(
                device,
                "unit bucket info",
                (1 + MAX_CASCADES) * BUCKET_SET_BYTES as usize,
                BufferUsages::COPY_DST,
            ),
            pose_src: storage_buffer(device, "unit pose sources", sources * 4, BufferUsages::empty()),
            pose_src_base,
            poses,
            pose_stride,
        }
    }
}

#[derive(Resource, Default)]
pub struct GpuUnitBuffers {
    pub alloc: Option<UnitAlloc>,
    params: UniformBuffer<BuildParams>,
    bind_group: Option<BindGroup>,
    /// The readback buffer the bind group holds. The asset can be
    /// recreated, and then the bind group follows.
    bound_readback: Option<BufferId>,
    /// Bumped whenever a buffer bound by a draw bind group is recreated.
    pub generation: u32,
    /// Threads of this frame's build dispatch.
    threads: u32,
    /// The pose pass runs this frame: the build hands out pose slots.
    pose_pass: bool,
    pose_params: UniformBuffer<PoseParams>,
    /// The pose pass's group 0, and the globals buffer it holds.
    pose_group: Option<BindGroup>,
    bound_globals: Option<BufferId>,
    /// Groups 1 and 2 of the pose pass.
    empty_group: Option<BindGroup>,
    /// Per kind, the pose pass's group 3, made for `kind_generation`.
    kind_groups: Vec<BindGroup>,
    kind_generation: u32,
    /// Per kind, `UVec4(kind, 0, 0, 0)` for group 3 binding 10.
    kind_uniforms: Vec<Buffer>,
}

/// Upload the snapshot when a new one arrived, the small per-frame inputs
/// every frame, and grow the buffers when a bigger battle starts.
#[allow(clippy::too_many_arguments)] // bevy system params
fn prepare_gpu_units(
    mut input: ResMut<GpuUnitInput>,
    mut buffers: ResMut<GpuUnitBuffers>,
    meshes: Query<&PullMeshGpu>,
    rigs: Query<(&ExtractedBucket, &UnitRig, &RigBuffer)>,
    pipelines: Res<GpuUnitPipelines>,
    pipeline_cache: Res<PipelineCache>,
    (device, queue): (Res<RenderDevice>, Res<RenderQueue>),
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    globals: Res<GlobalsBuffer>,
) {
    if !input.enabled {
        return;
    }
    let t0 = std::time::Instant::now();
    let n = input.n as usize;
    let kind_needed = input.kind_counts.map(|c| c as usize);
    // Pose slots per soldier of each kind: more for a kind with a bow rig.
    let mut pose_stride = [0; NUM_KINDS];
    if input.pose_pass {
        pose_stride = [POSE_SLOTS; NUM_KINDS];
        for (bucket, rig, _) in &rigs {
            if rig.rig.bow.params[3] > 0.5 {
                pose_stride[bucket.0 / NUM_LODS] = POSE_SLOTS_BOW;
            }
        }
    }
    let grow = match &buffers.alloc {
        Some(a) => {
            a.live_cap < n
                || a.kind_cap.iter().zip(kind_needed).any(|(cap, need)| *cap < need)
                || a.pose_stride != pose_stride
        }
        None => true,
    };
    if grow {
        let live_cap = (n + n / 4).max(1024);
        let old = buffers.alloc.as_ref().map_or([0; NUM_KINDS], |a| a.kind_cap);
        let kind_cap = std::array::from_fn(|k| old[k].max(kind_needed[k]).max(256));
        info!(
            "unit gpu buffers: {live_cap} soldiers, index slots per kind {:?}, pose slots per soldier {pose_stride:?}",
            kind_cap.map(|c| c + CORPSE_CAP)
        );
        buffers.alloc = Some(UnitAlloc::new(&device, live_cap, kind_cap, pose_stride));
        buffers.generation += 1;
        buffers.bind_group = None;
        buffers.pose_group = None;
    }
    let buffers = &mut *buffers;
    let alloc = buffers.alloc.as_mut().expect("allocated above");

    if input.fresh {
        input.fresh = false;
        if n > 0 {
            queue.write_buffer(&alloc.soldiers, 0, bytemuck::cast_slice(&input.records[..n]));
        }
    }
    let n_regs = input.regiments.len();
    if n_regs > alloc.regiments_cap {
        alloc.regiments_cap = n_regs + n_regs / 2;
        alloc.regiments = storage_buffer(
            &device,
            "unit regiments",
            alloc.regiments_cap * size_of::<RegimentRecord>(),
            BufferUsages::COPY_DST,
        );
        buffers.bind_group = None;
    }
    if n_regs > 0 {
        queue.write_buffer(&alloc.regiments, 0, bytemuck::cast_slice(&input.regiments));
    }

    // The fallen: each new body lands in its ring slot of the corpse
    // region, once. Slots of one tick are mostly consecutive, so runs of
    // them go up in one write.
    if !input.corpse_pending.is_empty() {
        let record = size_of::<InstanceData>();
        let region = alloc.live_cap * record;
        let GpuUnitInput {
            corpse_pending,
            corpse_run,
            ..
        } = &mut *input;
        corpse_pending.sort_unstable_by_key(|(slot, _)| *slot);
        let mut start = 0;
        while start < corpse_pending.len() {
            let mut end = start + 1;
            while end < corpse_pending.len()
                && corpse_pending[end].0 == corpse_pending[end - 1].0 + 1
            {
                end += 1;
            }
            corpse_run.clear();
            corpse_run.extend(corpse_pending[start..end].iter().map(|(_, r)| *r));
            let offset = region + corpse_pending[start].0 as usize * record;
            queue.write_buffer(&alloc.records, offset as u64, bytemuck::cast_slice(corpse_run));
            start = end;
        }
        corpse_pending.clear();
    }

    // The bucket table: the camera's set, then one set per cascade whose
    // every level of a kind points at the cascade's caster list for it.
    let mut info = [[[0u32; 4]; NUM_BUCKETS]; 1 + MAX_CASCADES];
    for mesh in &meshes {
        let b = mesh.bucket;
        let stride = alloc.pose_stride[b / NUM_LODS];
        info[0][b] = [alloc.bases[b], mesh.count, stride, 0];
        for c in 0..MAX_CASCADES {
            info[1 + c][b] = [alloc.shadow_bases[c][b / NUM_LODS], mesh.count, stride, 0];
        }
    }
    queue.write_buffer(&alloc.bucket_info, 0, bytemuck::cast_slice(&info));
    let mut params = input.params;
    params.corpse_base = alloc.live_cap as u32;
    params.corpse_cap = CORPSE_CAP as u32;
    params.corpse_len = UVec4::from_array(input.corpse_len);
    params.buckets = info[0].map(UVec4::from_array);
    params.shadow_lists = alloc.shadow_bases.map(UVec4::from_array);
    params.ring_base = alloc.ring_base;
    params.shadow_corners = input.shadow_levels.map(|levels| {
        UVec4::from_array(std::array::from_fn(|kind| {
            info[0][kind * NUM_LODS + levels[kind] as usize][1]
        }))
    });
    params.pose_src_base = UVec4::from_array(alloc.pose_src_base);
    params.pose_pass = input.pose_pass as u32;
    buffers.params.set(params);
    buffers.params.write_buffer(&device, &queue);
    buffers.threads = n as u32 + input.corpse_len.iter().sum::<u32>();
    buffers.pose_pass = input.pose_pass;
    buffers.pose_params.set(PoseParams {
        src_base: UVec4::from_array(alloc.pose_src_base),
        stride: UVec4::from_array(alloc.pose_stride),
    });
    buffers.pose_params.write_buffer(&device, &queue);

    // The readback asset arrives a frame or two after startup. No bind
    // group until then, so the pass waits and nothing draws.
    let Some(readback) = input.readback.as_ref().and_then(|h| gpu_buffers.get(h)) else {
        buffers.bind_group = None;
        return;
    };
    if buffers.bound_readback != Some(readback.buffer.id()) {
        buffers.bound_readback = Some(readback.buffer.id());
        buffers.bind_group = None;
    }
    if buffers.bind_group.is_none() {
        buffers.bind_group = Some(device.create_bind_group(
            "unit build bind group",
            &pipeline_cache.get_bind_group_layout(&pipelines.layout),
            &BindGroupEntries::sequential((
                buffers.params.binding().expect("written above"),
                alloc.soldiers.as_entire_binding(),
                alloc.regiments.as_entire_binding(),
                alloc.smooth.as_entire_binding(),
                alloc.records.as_entire_binding(),
                alloc.index_list.as_entire_binding(),
                alloc.counts.as_entire_binding(),
                alloc.args.as_entire_binding(),
                readback.buffer.as_entire_binding(),
                alloc.pose_src.as_entire_binding(),
            )),
        ));
    }
    if buffers.pose_pass {
        prepare_pose_groups(buffers, &rigs, &pipelines, &pipeline_cache, &device, &globals);
    }
    crate::render_units::PREPARE_US.fetch_add(
        t0.elapsed().as_micros() as u32,
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// The pose pass's bind groups: group 0 when the buffers or Bevy's globals
/// moved, group 3 per kind when the pose buffers moved.
fn prepare_pose_groups(
    buffers: &mut GpuUnitBuffers,
    rigs: &Query<(&ExtractedBucket, &UnitRig, &RigBuffer)>,
    pipelines: &GpuUnitPipelines,
    pipeline_cache: &PipelineCache,
    device: &RenderDevice,
    globals: &GlobalsBuffer,
) {
    let Some(alloc) = &buffers.alloc else {
        return;
    };
    let Some(globals_binding) = globals.buffer.binding() else {
        return;
    };
    let globals_id = globals.buffer.buffer().map(|b| b.id());
    if buffers.pose_group.is_none() || buffers.bound_globals != globals_id {
        buffers.bound_globals = globals_id;
        buffers.pose_group = Some(device.create_bind_group(
            "unit pose bind group",
            &pipeline_cache.get_bind_group_layout(&pipelines.pose_layout),
            &BindGroupEntries::sequential((
                buffers.pose_params.binding().expect("written above"),
                alloc.records.as_entire_binding(),
                alloc.pose_src.as_entire_binding(),
                alloc.counts.as_entire_binding(),
                globals_binding,
            )),
        ));
    }
    if buffers.empty_group.is_none() {
        buffers.empty_group = Some(device.create_bind_group(
            "unit pose empty bind group",
            &pipeline_cache.get_bind_group_layout(&pipelines.empty_layout),
            &[],
        ));
    }
    if buffers.kind_uniforms.is_empty() {
        buffers.kind_uniforms = (0..NUM_KINDS as u32)
            .map(|kind| {
                device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("unit pose kind"),
                    contents: bytemuck::bytes_of(&[kind, 0, 0, 0]),
                    usage: BufferUsages::UNIFORM,
                })
            })
            .collect();
    }
    if buffers.kind_groups.len() == NUM_KINDS && buffers.kind_generation == buffers.generation {
        return;
    }
    // Every level of a kind binds the same rig: take the first found.
    let mut kind_rigs: [Option<&RigBuffer>; NUM_KINDS] = [None; NUM_KINDS];
    for (bucket, _, rig) in rigs {
        kind_rigs[bucket.0 / NUM_LODS].get_or_insert(rig);
    }
    let layout = pipeline_cache.get_bind_group_layout(&pipelines.kind_layout);
    let groups: Option<Vec<BindGroup>> = (0..NUM_KINDS)
        .map(|kind| {
            let rig = kind_rigs[kind]?;
            Some(device.create_bind_group(
                "unit pose kind bind group",
                &layout,
                &BindGroupEntries::with_indices((
                    (6, rig.rig.as_entire_binding()),
                    (7, rig.clips.as_entire_binding()),
                    (9, alloc.poses[kind].as_entire_binding()),
                    (10, buffers.kind_uniforms[kind].as_entire_binding()),
                )),
            ))
        })
        .collect();
    if let Some(groups) = groups {
        buffers.kind_groups = groups;
        buffers.kind_generation = buffers.generation;
    }
}

/// Group 3 of every pulled bucket, rebuilt when the shared buffers moved.
#[allow(clippy::too_many_arguments, clippy::type_complexity)] // bevy system params
fn prepare_pull_bind_groups(
    mut commands: Commands,
    buffers: Res<GpuUnitBuffers>,
    custom_pipeline: Res<CustomPipeline>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    images: Res<RenderAssets<GpuImage>>,
    sun: Res<crate::render_units_shadow::SunBuffer>,
    meshes: Query<(
        Entity,
        &PullMeshGpu,
        Option<&ExtractedAtlas>,
        &RigBuffer,
        Option<&PulledBucketGpu>,
    )>,
) {
    let (Some(alloc), Some(sun)) = (&buffers.alloc, sun.binding()) else {
        return;
    };
    for (entity, mesh, atlas, rig, existing) in &meshes {
        if existing.is_some_and(|b| b.generation == buffers.generation && b.atlas_settled) {
            continue;
        }
        let (view, sampler, atlas_settled) = custom_pipeline.atlas_for(atlas, &images);
        let layout = pipeline_cache.get_bind_group_layout(&custom_pipeline.pull_layout);
        // Set 0 of the bucket table is the camera's, set 1 + c cascade c's.
        let group = |set: u64| {
            device.create_bind_group(
                "unit pull bind group",
                &layout,
                &BindGroupEntries::with_indices((
                    (0, alloc.records.as_entire_binding()),
                    (1, alloc.index_list.as_entire_binding()),
                    (2, mesh.vertices.as_entire_binding()),
                    (
                        3,
                        BufferBinding {
                            buffer: &alloc.bucket_info,
                            offset: set * BUCKET_SET_BYTES,
                            size: BufferSize::new(BUCKET_SET_BYTES),
                        },
                    ),
                    (4, view),
                    (5, sampler),
                    (6, rig.rig.as_entire_binding()),
                    (7, rig.clips.as_entire_binding()),
                    (8, sun.clone()),
                    (9, alloc.poses[mesh.bucket / NUM_LODS].as_entire_binding()),
                )),
            )
        };
        commands.entity(entity).insert(PulledBucketGpu {
            bind_group: group(0),
            shadow: (1..=MAX_CASCADES as u64).map(group).collect(),
            generation: buffers.generation,
            atlas_settled,
            bucket: mesh.bucket as u32,
        });
    }
}

/// The compute passes, recorded into the frame's encoder before the sun's
/// shadow pass and the main passes of the view: clear the counters, build
/// every soldier, turn the counts into draw arguments, then pose each drawn
/// soldier once per kind. The shadow cascades draw from the same lists
/// (render_units_shadow.rs), so the passes must come first or they draw
/// last frame's. Until every pipeline and bind group is ready nothing runs,
/// and the draws keep last frame's lists, records and poses together.
pub(crate) fn run_unit_build_pass(
    buffers: Res<GpuUnitBuffers>,
    pipelines: Res<GpuUnitPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (Some(alloc), Some(bind_group)) = (&buffers.alloc, &buffers.bind_group) else {
        return;
    };
    let (Some(build), Some(finalize)) = (
        pipeline_cache.get_compute_pipeline(pipelines.build),
        pipeline_cache.get_compute_pipeline(pipelines.finalize),
    ) else {
        return;
    };
    let pose = match buffers.pose_pass {
        true => match (
            pipeline_cache.get_compute_pipeline(pipelines.pose),
            &buffers.pose_group,
            &buffers.empty_group,
        ) {
            (Some(pose), Some(group), Some(empty)) if buffers.kind_groups.len() == NUM_KINDS => {
                Some((pose, group, empty))
            }
            _ => return,
        },
        false => None,
    };
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    encoder.clear_buffer(&alloc.counts, 0, None);
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("unit build"),
        timestamp_writes: None,
    });
    // Shows up in the overlay's GPU list as `unit_build`.
    let span = diagnostics.pass_span(&mut pass, "unit_build");
    pass.set_bind_group(0, &**bind_group, &[]);
    if buffers.threads > 0 {
        pass.set_pipeline(build);
        pass.dispatch_workgroups(buffers.threads.div_ceil(64), 1, 1);
    }
    pass.set_pipeline(finalize);
    pass.dispatch_workgroups(1, 1, 1);
    span.end(&mut pass);
    drop(pass);

    let Some((pose, group, empty)) = pose else {
        return;
    };
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("unit pose"),
        timestamp_writes: None,
    });
    // Shows up in the overlay's GPU list as `unit_pose`.
    let span = diagnostics.pass_span(&mut pass, "unit_pose");
    pass.set_pipeline(pose);
    pass.set_bind_group(0, &**group, &[]);
    pass.set_bind_group(1, &**empty, &[]);
    pass.set_bind_group(2, &**empty, &[]);
    for (kind, kind_group) in buffers.kind_groups.iter().enumerate() {
        pass.set_bind_group(3, &**kind_group, &[]);
        pass.dispatch_workgroups_indirect(&alloc.args, ((POSE_ARG + kind) * 16) as u64);
    }
    span.end(&mut pass);
}

pub struct GpuUnitRenderPlugin;

impl Plugin for GpuUnitRenderPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/unit_build.wgsl");
        embedded_asset!(app, "shaders/unit_pose_pass.wgsl");
        app.init_resource::<SoldierSnapshot>()
            .init_resource::<GpuFrameInput>()
            .add_systems(Startup, setup_counts_readback.run_if(gpu_sync))
            .add_systems(
                PostUpdate,
                (pack_soldier_snapshot, build_frame_params)
                    .chain()
                    .after(bevy::light::SimulationLightSystems::UpdateDirectionalLightCascades)
                    .run_if(gpu_sync),
            );
        app.sub_app_mut(RenderApp)
            .init_resource::<GpuUnitInput>()
            .init_resource::<GpuUnitBuffers>()
            .add_systems(RenderStartup, init_gpu_unit_pipelines)
            .add_systems(ExtractSchedule, (extract_gpu_units, extract_pull_meshes))
            .add_systems(
                Render,
                (
                    prepare_gpu_units
                        .in_set(RenderSystems::PrepareResources)
                        .after(crate::render_units::prepare_instance_buffers),
                    prepare_pull_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                Core3d,
                run_unit_build_pass
                    .before(bevy::pbr::per_view_shadow_pass::<{ bevy::pbr::EARLY_SHADOW_PASS }>)
                    .before(Core3dSystems::MainPass),
            );
    }

    /// The device exists by now: decide the path once. Storage buffers in
    /// the vertex stage, atomics and indirect draws are core WebGPU, but a
    /// downlevel backend can lack the first, and then the CPU path stays.
    fn finish(&self, app: &mut App) {
        let requested = !std::env::var("FL_GPU_SYNC").is_ok_and(|v| v == "0");
        let check = std::env::var("FL_GPU_CHECK").is_ok();
        let pose_pass = !std::env::var("FL_POSE_PASS").is_ok_and(|v| v == "0");
        let supported = match (
            app.world().get_resource::<RenderAdapter>(),
            app.world().get_resource::<RenderDevice>(),
        ) {
            (Some(adapter), Some(device)) => {
                adapter
                    .get_downlevel_capabilities()
                    .flags
                    .contains(DownlevelFlags::VERTEX_STORAGE)
                    && device.limits().max_storage_buffers_per_shader_stage >= 8
            }
            _ => false,
        };
        if requested && !supported {
            warn!("this device lacks vertex storage buffers: the CPU unit path runs instead");
        }
        if requested && supported {
            info!("unit render data built on the GPU (FL_GPU_SYNC=0 for the CPU path)");
        }
        if check && requested && supported {
            info!("FL_GPU_CHECK: the CPU sweep runs too, counts compared per frame");
        }
        if requested && supported && !pose_pass {
            info!("FL_POSE_PASS=0: each corner poses itself");
        }
        app.insert_resource(GpuSyncConfig {
            enabled: requested && supported,
            check,
            pose_pass,
        });
    }
}

/// Which bucket entity draws pulled: every unit bucket in GPU mode.
pub fn pull_mesh_for(mesh: &Mesh, bucket: &InstanceBucket, cfg: &GpuSyncConfig) -> Option<PullMesh> {
    cfg.enabled.then(|| PullMesh::from_mesh(mesh, bucket.0)).flatten()
}
