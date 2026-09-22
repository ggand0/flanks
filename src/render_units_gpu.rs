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
    renderer::{RenderAdapter, RenderContext, RenderDevice, RenderQueue},
    storage::{GpuShaderBuffer, ShaderBuffer},
    sync_world::RenderEntity,
};
use bytemuck::{Pod, Zeroable};

use crate::render_units::{
    CELEBRATE_BASE, CORPSE_CAP, Corpses, CustomPipeline, InstanceBucket, InstanceData, LodBands,
    LodConfig, NUM_BUCKETS, NUM_LODS, RenderCounts, SYNC_CHUNK, celebrate_progress, march_signal,
    regiment_phase, stance_tier, wall_signal,
};
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
}

pub fn gpu_sync(cfg: Res<GpuSyncConfig>) -> bool {
    cfg.enabled
}

/// The CPU sweep runs on the CPU path, and next to the GPU path in check mode.
pub fn cpu_sweep(cfg: Res<GpuSyncConfig>) -> bool {
    !cfg.enabled || cfg.check
}

/// Words of the counts readback: 16 bucket totals, 16 fallen per bucket,
/// the frame stamp, the soldier count.
const READBACK_WORDS: usize = 36;

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
    marching: f32,
    walled: f32,
    phase: f32,
    flags: u32,
}

const REG_BROKEN: u32 = 1;
const REG_SELECTED: u32 = 2;
const REG_HOVERED: u32 = 4;

/// The compute pass uniform (`Params` in unit_build.wgsl, same field order).
#[derive(ShaderType, Clone, Copy, Default)]
pub struct BuildParams {
    planes: [Vec4; 6],
    cam_pos: Vec3,
    alpha: f32,
    k_walk: f32,
    k_band: f32,
    k_march: f32,
    inv_dt: f32,
    n: u32,
    n_regs: u32,
    cull: u32,
    corpse_base: u32,
    corpse_len: UVec4,
    corpse_cap: u32,
    /// The frame this pass belongs to, stamped into the readback.
    frame: u32,
    pad0: u32,
    pad1: u32,
    /// [kind * 3 + set]: set 0 fine, 1 coarse, 2 plain.
    bands: [Vec4; 12],
    windup: Vec4,
    /// draw_ticks, death_ticks, hit_stagger_ticks, celebrate_base
    consts: Vec4,
    /// x = first index slot of the bucket, y = mesh corners per soldier.
    buckets: [UVec4; NUM_BUCKETS],
}

const _: () = assert!(NUM_KINDS == 4, "BuildParams packs per-kind values in vec4s");
const _: () = assert!(NUM_BUCKETS == 16, "unit_build.wgsl sizes its counters for 16 buckets");

/// The soldier snapshot on the main world side. `records` is double
/// buffered against the render world: extract swaps the two, so the
/// handoff is a pointer exchange.
#[derive(Resource, Default)]
pub struct SoldierSnapshot {
    pub records: Vec<GpuSoldier>,
    pub n: u32,
    pub kind_counts: [u32; NUM_KINDS],
    pub generation: u64,
    pub tick: u32,
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
}

/// Pack the live soldier columns into the snapshot. Runs after every
/// writer of the columns (the fixed loop, the deployment drag) and only
/// when `Units` changed, so frames without a tick pack nothing. Parallel
/// on the compute pool. Not reachable from the tick job, so a plain scope
/// is correct here.
fn pack_soldier_snapshot(
    units: Res<Units>,
    pipeline: Res<crate::movement::TickPipeline>,
    mut snap: ResMut<SoldierSnapshot>,
) {
    if !units.is_changed() {
        return;
    }
    let t0 = std::time::Instant::now();
    let n = units.len();
    snap.records.resize(n, GpuSoldier::zeroed());
    let units = &*units;
    let records = &mut snap.records;
    let chunk_counts: Vec<[u32; NUM_KINDS]> = bevy::tasks::ComputeTaskPool::get().scope(|scope| {
        for (ci, out) in records.chunks_mut(SYNC_CHUNK).enumerate() {
            scope.spawn(async move { pack_chunk(units, ci * SYNC_CHUNK, out) });
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
    snap.generation = units.generation;
    snap.tick = pipeline.tick;
    snap.fresh = true;
    snap.pack_ms = t0.elapsed().as_secs_f32() * 1000.0;
}

fn pack_chunk(units: &Units, start: usize, out: &mut [GpuSoldier]) -> [u32; NUM_KINDS] {
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
                | (units.flash[i] as u32) << 24,
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
    camera: Query<(&Camera, &Projection, &Transform), With<Camera3d>>,
    snap: Res<SoldierSnapshot>,
    mut frame: ResMut<GpuFrameInput>,
    mut counts: ResMut<RenderCounts>,
    mut no_cull: Local<Option<bool>>,
    frame_count: Res<FrameCount>,
) {
    let t0 = std::time::Instant::now();
    let Ok((cam, projection, cam_tf)) = camera.single() else {
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
            if has_sel && selection.regiments.get(g).copied().unwrap_or(false) {
                flags |= REG_SELECTED;
            }
            if hover.enemy == Some(g as u32) {
                flags |= REG_HOVERED;
            }
            RegimentRecord {
                stance: stance_tier(gd),
                celebrate: celebrate_progress(gd),
                marching: march_signal(gd),
                walled: wall_signal(gd),
                phase: regiment_phase(g),
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
    p.k_march = (dt / 0.5).min(1.0);
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
        crate::movement::DEATH_TICKS as f32,
        crate::movement::HIT_STAGGER_TICKS as f32,
        CELEBRATE_BASE,
    );
    p.frame = frame_count.0;
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
    color: [f32; 4],
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
        let corners = mesh.indices()?.iter().map(|i| PullVertex {
            position: pos[i],
            part: uv[i][0],
            normal: nrm[i],
            pivot: uv[i][1],
            color: col[i],
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
    generation: u32,
    pub bucket: u32,
}

/// The extracted inputs, render world side.
#[derive(Resource, Default)]
pub struct GpuUnitInput {
    enabled: bool,
    records: Vec<GpuSoldier>,
    fresh: bool,
    n: u32,
    kind_counts: [u32; NUM_KINDS],
    params: BuildParams,
    regiments: Vec<RegimentRecord>,
    pub lod_debug: bool,
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
    let enabled = main_world.resource::<GpuSyncConfig>().enabled;
    input.enabled = enabled;
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
    if input.readback.is_none() {
        input.readback = main_world
            .get_resource::<CountsReadback>()
            .map(|r| r.handle.clone());
    }
}

/// The compute pipelines and their bind group layout.
#[derive(Resource)]
struct GpuUnitPipelines {
    build: CachedComputePipelineId,
    finalize: CachedComputePipelineId,
    layout: BindGroupLayoutDescriptor,
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
    commands.insert_resource(GpuUnitPipelines {
        build,
        finalize,
        layout,
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
    /// kind can only land in one of that kind's levels.
    pub index_list: Buffer,
    kind_cap: [usize; NUM_KINDS],
    bases: [u32; NUM_BUCKETS],
    counts: Buffer,
    pub args: Buffer,
    regiments: Buffer,
    regiments_cap: usize,
    pub bucket_info: Buffer,
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
    fn new(device: &RenderDevice, live_cap: usize, kind_cap: [usize; NUM_KINDS]) -> Self {
        let mut bases = [0u32; NUM_BUCKETS];
        let mut total = 0usize;
        for kind in 0..NUM_KINDS {
            for lod in 0..NUM_LODS {
                bases[kind * NUM_LODS + lod] = total as u32;
                total += kind_cap[kind] + CORPSE_CAP;
            }
        }
        let regiments_cap = 256;
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
            smooth: storage_buffer(device, "unit smoothing", live_cap * 20, BufferUsages::empty()),
            index_list: storage_buffer(device, "unit index list", total * 4, BufferUsages::empty()),
            kind_cap,
            bases,
            counts: storage_buffer(device, "unit bucket counts", 32 * 4, BufferUsages::COPY_DST),
            args: storage_buffer(
                device,
                "unit draw args",
                NUM_BUCKETS * 16,
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
                NUM_BUCKETS * 16,
                BufferUsages::COPY_DST,
            ),
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
}

/// Upload the snapshot when a new one arrived, the small per-frame inputs
/// every frame, and grow the buffers when a bigger battle starts.
#[allow(clippy::too_many_arguments)] // bevy system params
fn prepare_gpu_units(
    mut input: ResMut<GpuUnitInput>,
    mut buffers: ResMut<GpuUnitBuffers>,
    meshes: Query<&PullMeshGpu>,
    pipelines: Res<GpuUnitPipelines>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
) {
    if !input.enabled {
        return;
    }
    let t0 = std::time::Instant::now();
    let n = input.n as usize;
    let kind_needed = input.kind_counts.map(|c| c as usize);
    let grow = match &buffers.alloc {
        Some(a) => a.live_cap < n || a.kind_cap.iter().zip(kind_needed).any(|(cap, need)| *cap < need),
        None => true,
    };
    if grow {
        let live_cap = (n + n / 4).max(1024);
        let old = buffers.alloc.as_ref().map_or([0; NUM_KINDS], |a| a.kind_cap);
        let kind_cap = std::array::from_fn(|k| old[k].max(kind_needed[k]).max(256));
        info!(
            "unit gpu buffers: {live_cap} soldiers, index slots per kind {:?}",
            kind_cap.map(|c| c + CORPSE_CAP)
        );
        buffers.alloc = Some(UnitAlloc::new(&device, live_cap, kind_cap));
        buffers.generation += 1;
        buffers.bind_group = None;
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

    let mut info = [[0u32; 4]; NUM_BUCKETS];
    for mesh in &meshes {
        info[mesh.bucket] = [alloc.bases[mesh.bucket], mesh.count, 0, 0];
    }
    queue.write_buffer(&alloc.bucket_info, 0, bytemuck::cast_slice(&info));
    let mut params = input.params;
    params.corpse_base = alloc.live_cap as u32;
    params.corpse_cap = CORPSE_CAP as u32;
    params.corpse_len = UVec4::from_array(input.corpse_len);
    params.buckets = info.map(UVec4::from_array);
    buffers.params.set(params);
    buffers.params.write_buffer(&device, &queue);
    buffers.threads = n as u32 + input.corpse_len.iter().sum::<u32>();

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
            )),
        ));
    }
    crate::render_units::PREPARE_US.fetch_add(
        t0.elapsed().as_micros() as u32,
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Group 3 of every pulled bucket, rebuilt when the shared buffers moved.
fn prepare_pull_bind_groups(
    mut commands: Commands,
    buffers: Res<GpuUnitBuffers>,
    custom_pipeline: Res<CustomPipeline>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    meshes: Query<(Entity, &PullMeshGpu, Option<&PulledBucketGpu>)>,
) {
    let Some(alloc) = &buffers.alloc else {
        return;
    };
    for (entity, mesh, existing) in &meshes {
        if existing.is_some_and(|b| b.generation == buffers.generation) {
            continue;
        }
        let bind_group = device.create_bind_group(
            "unit pull bind group",
            &pipeline_cache.get_bind_group_layout(&custom_pipeline.pull_layout),
            &BindGroupEntries::sequential((
                alloc.records.as_entire_binding(),
                alloc.index_list.as_entire_binding(),
                mesh.vertices.as_entire_binding(),
                alloc.bucket_info.as_entire_binding(),
            )),
        );
        commands.entity(entity).insert(PulledBucketGpu {
            bind_group,
            generation: buffers.generation,
            bucket: mesh.bucket as u32,
        });
    }
}

/// The compute pass, recorded into the frame's encoder before the main
/// passes of the view: clear the counters, build every soldier, turn the
/// counts into draw arguments.
fn run_unit_build_pass(
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
}

pub struct GpuUnitRenderPlugin;

impl Plugin for GpuUnitRenderPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/unit_build.wgsl");
        app.init_resource::<SoldierSnapshot>()
            .init_resource::<GpuFrameInput>()
            .add_systems(Startup, setup_counts_readback.run_if(gpu_sync))
            .add_systems(
                PostUpdate,
                (pack_soldier_snapshot, build_frame_params)
                    .chain()
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
                    .after(Core3dSystems::Prepass)
                    .before(Core3dSystems::MainPass),
            );
    }

    /// The device exists by now: decide the path once. Storage buffers in
    /// the vertex stage, atomics and indirect draws are core WebGPU, but a
    /// downlevel backend can lack the first, and then the CPU path stays.
    fn finish(&self, app: &mut App) {
        let requested = !std::env::var("FL_GPU_SYNC").is_ok_and(|v| v == "0");
        let check = std::env::var("FL_GPU_CHECK").is_ok();
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
        app.insert_resource(GpuSyncConfig {
            enabled: requested && supported,
            check,
        });
    }
}

/// Which bucket entity draws pulled: every unit bucket in GPU mode.
pub fn pull_mesh_for(mesh: &Mesh, bucket: &InstanceBucket, cfg: &GpuSyncConfig) -> Option<PullMesh> {
    cfg.enabled.then(|| PullMesh::from_mesh(mesh, bucket.0)).flatten()
}
