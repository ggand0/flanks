//! Custom instanced rendering for units: one draw call for all 100k cubes.
//! Adapted from bevy 0.19's `shader_advanced/custom_shader_instancing` example.
//! Per-instance data is copied from the `Units` SoA buffers each frame and
//! uploaded as an instance-rate vertex buffer.

use bevy::core_pipeline::core_3d::TransparentSortingInfo3d;
use bevy::pbr::{
    self, MeshInputUniform, MeshPipelineSystems, MeshUniform, SetMeshViewBindingArrayBindGroup,
    ViewKeyCache,
};
use bevy::{
    camera::visibility::NoFrustumCulling,
    core_pipeline::core_3d::Transparent3d,
    ecs::system::{SystemParamItem, lifetimeless::*},
    mesh::{MeshVertexBufferLayoutRef, VertexBufferLayout},
    pbr::{
        MeshPipeline, MeshPipelineKey, RenderMeshInstances, SetMeshBindGroup, SetMeshViewBindGroup,
    },
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
        batching::{NoAutomaticBatching, gpu_preprocessing::BatchedInstanceBuffers},
        mesh::{RenderMesh, RenderMeshBufferInfo, allocator::MeshAllocator},
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        sync_component::{SyncComponent, SyncComponentPlugin},
        sync_world::{MainEntity, RenderEntity},
        view::ExtractedView,
    },
};
use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::camera::primitives::{Frustum, Sphere};
use bevy::math::primitives::ViewFrustum;
use bytemuck::{Pod, Zeroable};

use crate::render_units_gpu::{
    GpuSyncConfig, GpuUnitBuffers, GpuUnitInput, PullMeshGpu, PulledBucketGpu, cpu_sweep,
    pull_mesh_for,
};
use crate::units::Units;

/// Bounding-sphere radius for per-instance frustum culling: cube diagonal
/// plus a generous margin so nothing pops inside the screen edge.
const CULL_RADIUS: f32 = 2.5;

/// Instances drawn this frame after culling (overlay diagnostics).
#[derive(Resource, Default)]
pub struct RenderCounts {
    pub drawn: usize,
    pub total: usize,
    /// Drawn counts per unit kind (summed over detail levels).
    pub bucket_drawn: Vec<usize>,
    /// Drawn counts per detail level (summed over kinds).
    pub lod_drawn: [usize; NUM_LODS],
    /// Fallen soldiers drawn this frame. The counts above are the living.
    pub corpses_drawn: usize,
    /// Cost of sync_instance_data this frame (cull + bucket build).
    pub sync_ms: f32,
    /// FL_GPU_CHECK: what the CPU sweep would have drawn on recent frames,
    /// as (frame, living per bucket, fallen per bucket), for the GPU
    /// readback to compare against.
    pub check: std::collections::VecDeque<(u32, [u32; NUM_BUCKETS], [u32; NUM_BUCKETS])>,
}

impl RenderCounts {
    /// The display counts from per-bucket living and fallen counts.
    /// Buckets are kind-major: per-kind counts sum each run of levels,
    /// per-level counts sum across kinds.
    pub(crate) fn set_from_buckets(
        &mut self,
        living: &[u32; NUM_BUCKETS],
        fallen: &[u32; NUM_BUCKETS],
    ) {
        self.drawn = living.iter().map(|&n| n as usize).sum();
        self.bucket_drawn.clear();
        self.bucket_drawn.extend(
            living
                .chunks(NUM_LODS)
                .map(|levels| levels.iter().map(|&n| n as usize).sum::<usize>()),
        );
        self.lod_drawn = std::array::from_fn(|lod| {
            living.iter().skip(lod).step_by(NUM_LODS).map(|&n| n as usize).sum()
        });
        self.corpses_drawn = fallen.iter().map(|&n| n as usize).sum();
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct InstanceData {
    pub position: Vec3,
    pub scale: f32,
    /// rgb = team color; a = stable per-unit anim seed (NOT opacity).
    pub color: [f32; 4],
    /// x = yaw, y = ground speed m/s, z = lunge 0..1 (attack),
    /// w = fx: [0,1) hit flash, [1,2] death.
    pub anim: [f32; 4],
    /// x = leg length, hip to sole, y = wall 0..1 (shieldwall
    /// shield-front / spearwall leveled spears), z = gait phase in
    /// cycles (gait.rs), w = stagger progress.
    pub anim2: [f32; 4],
}

#[derive(Component, Deref, DerefMut, Default)]
pub struct InstanceMaterialData(pub Vec<InstanceData>);

/// One instanced draw per bucket. Buckets are keyed by unit kind AND
/// detail level (`bucket_of`): one mesh per pair, team identity stays
/// per-instance color.
#[derive(Component)]
pub struct InstanceBucket(pub usize);

/// Detail levels per unit kind: L0 is the full mesh, the last level a
/// couple of blocks for soldiers a few pixels tall.
pub const NUM_LODS: usize = 4;

/// Starting level thresholds: the minimum on-screen soldier height in
/// pixels for L0, L1 and L2. Anything smaller draws the last level.
const LOD_PX_DEFAULT: [f32; NUM_LODS - 1] = [28.0, 12.0, 3.0];
/// Per-soldier threshold spread (+-10% of distance, from the stable anim
/// seed): a regiment changes level as a scattered band, never as a line
/// sweeping across the ranks.
const LOD_JITTER: f32 = 0.2;
/// Hysteresis around each threshold: a soldier jostling right at a
/// boundary keeps his level instead of flickering between two meshes.
const LOD_HYSTERESIS: f32 = 0.04;
/// FL_LOD_DEBUG tints for L1..L3 (L0 stays untinted).
const LOD_DEBUG_TINT: [[f32; 3]; NUM_LODS] =
    [[1.0, 1.0, 1.0], [0.2, 1.0, 0.3], [1.0, 0.9, 0.1], [1.0, 0.15, 0.1]];

/// Level-of-detail settings. A level is picked from the soldier's
/// projected HEIGHT IN PIXELS, not raw distance, so the cost follows the
/// pixels a player actually has: resolution and field of view need no
/// retuning.
#[derive(Resource)]
pub struct LodConfig {
    /// Per kind: minimum on-screen height (px) for L0, L1, L2. A zero
    /// disables that switch. Per kind so an authored mesh set can carry
    /// its own numbers next to a code-built one.
    pub px: [[f32; NUM_LODS - 1]; crate::unit_types::NUM_KINDS],
    /// FL_LOD_DEBUG=1: tint soldiers by level.
    pub debug: bool,
}

impl Default for LodConfig {
    /// FL_LOD=0 draws everything at L0 (the pre-LOD renderer, for A/B
    /// checks). FL_LOD_PX=a,b,c overrides the thresholds for feel passes.
    fn default() -> Self {
        let mut px = LOD_PX_DEFAULT;
        if let Ok(v) = std::env::var("FL_LOD_PX") {
            let parsed: Vec<f32> = v.split(',').filter_map(|t| t.trim().parse().ok()).collect();
            match <[f32; NUM_LODS - 1]>::try_from(parsed) {
                Ok(p) => px = p,
                Err(_) => warn!("FL_LOD_PX wants {} comma-separated numbers", NUM_LODS - 1),
            }
        }
        if std::env::var("FL_LOD").is_ok_and(|v| v == "0") {
            px = [0.0; NUM_LODS - 1];
        }
        Self {
            px: [px; crate::unit_types::NUM_KINDS],
            debug: std::env::var("FL_LOD_DEBUG").is_ok(),
        }
    }
}

/// This frame's level switch distances, squared, per kind. Two extra
/// sets widened and narrowed by the hysteresis bound the level a moving
/// soldier may hold.
pub(crate) struct LodBands {
    pub(crate) plain: [[f32; NUM_LODS - 1]; crate::unit_types::NUM_KINDS],
    pub(crate) fine: [[f32; NUM_LODS - 1]; crate::unit_types::NUM_KINDS],
    pub(crate) coarse: [[f32; NUM_LODS - 1]; crate::unit_types::NUM_KINDS],
}

/// Per-soldier multiplier on the level switch distance, from the stable
/// anim seed in `color.a`.
#[inline]
fn lod_jitter(seed: f32) -> f32 {
    1.0 - 0.5 * LOD_JITTER + LOD_JITTER * seed
}

/// FL_LOD_DEBUG: tint rgb by level. Alpha is the anim seed, not opacity.
#[inline]
fn lod_debug_tint(color: &mut [f32; 4], lod: usize) {
    if lod > 0 {
        for (c, tint) in color.iter_mut().zip(LOD_DEBUG_TINT[lod]) {
            *c = *c * 0.4 + tint * 0.6;
        }
    }
}

impl LodBands {
    /// `px_per_unit` = pixels covered by 1 m at 1 m distance (0 when the
    /// projection has no perspective: everything stays L0).
    pub(crate) fn new(cfg: &LodConfig, px_per_unit: f32) -> Self {
        let mut bands = Self {
            plain: [[f32::INFINITY; NUM_LODS - 1]; crate::unit_types::NUM_KINDS],
            fine: [[f32::INFINITY; NUM_LODS - 1]; crate::unit_types::NUM_KINDS],
            coarse: [[f32::INFINITY; NUM_LODS - 1]; crate::unit_types::NUM_KINDS],
        };
        for (kind, px) in cfg.px.iter().enumerate() {
            let height = 2.0 * crate::unit_types::half_height(kind);
            for (j, px) in px.iter().enumerate() {
                if px_per_unit > 0.0 && *px > 0.0 {
                    // A soldier is `px` tall at this distance.
                    let d = height * px_per_unit / px;
                    bands.plain[kind][j] = d * d;
                    bands.fine[kind][j] = (d * (1.0 + LOD_HYSTERESIS)).powi(2);
                    bands.coarse[kind][j] = (d * (1.0 - LOD_HYSTERESIS)).powi(2);
                }
            }
        }
        bands
    }

    /// Level for a squared distance: the farthest threshold passed wins.
    #[inline]
    fn level(thresholds: &[f32; NUM_LODS - 1], d2: f32) -> u8 {
        let mut lod = 0;
        for (j, t2) in thresholds.iter().enumerate() {
            if d2 > *t2 {
                lod = j as u8 + 1;
            }
        }
        lod
    }
}

/// Per-kind corpse cap (ring-buffered: oldest bodies fade from the field).
pub const CORPSE_CAP: usize = 25_000;

/// Fallen soldiers left where they died: their final topple pose, frozen.
/// Fed by the death sweep, never simulated: the battlefield keeps the
/// story of where the lines stood. A body is an ordinary instance with
/// its death played out, so each frame the instance sync culls the
/// fallen, picks their detail level and draws them from the same
/// kind-by-level buckets as the living.
#[derive(Resource, Default)]
pub struct Corpses {
    data: [Vec<InstanceData>; crate::unit_types::NUM_KINDS],
    cursor: [usize; crate::unit_types::NUM_KINDS],
    /// GPU path: bodies added since the last extract, as the slot they
    /// took in the GPU corpse region (`kind * CORPSE_CAP + ring index`)
    /// and the frozen record. Drained every frame.
    pub(crate) pending: Vec<(u32, InstanceData)>,
}

impl Corpses {
    pub fn push(&mut self, kind: usize, inst: InstanceData) {
        let v = &mut self.data[kind];
        let index = if v.len() < CORPSE_CAP {
            v.push(inst);
            v.len() - 1
        } else {
            let index = self.cursor[kind];
            v[index] = inst;
            self.cursor[kind] = (index + 1) % CORPSE_CAP;
            index
        };
        self.pending.push(((kind * CORPSE_CAP + index) as u32, inst));
    }

    pub fn clear(&mut self) {
        for v in &mut self.data {
            v.clear();
        }
        self.cursor = [0; crate::unit_types::NUM_KINDS];
        self.pending.clear();
    }

    /// Bodies of a kind on the field, at most `CORPSE_CAP`.
    pub(crate) fn len(&self, kind: usize) -> usize {
        self.data[kind].len()
    }
}

/// Number of instance buckets (== instance entities == draw calls).
pub const NUM_BUCKETS: usize = crate::unit_types::NUM_KINDS * NUM_LODS;

/// Which bucket a soldier of `kind` renders in at detail level `lod`.
#[inline]
fn bucket_of(kind: usize, lod: usize) -> usize {
    kind * NUM_LODS + lod
}

impl SyncComponent for InstanceMaterialData {
    type Target = Self;
}

/// Render-world copy of the instance data. Persistent component: the Vec's
/// allocation is reused every frame (extraction copies into it, no clone).
#[derive(Component, Default)]
pub(crate) struct ExtractedInstances(Vec<InstanceData>);

/// Last frame's cost of the two instance data copies, in microseconds,
/// for the overlay's frame breakdown. Extract runs on the main thread at
/// the render sync point, prepare on the render thread.
pub static EXTRACT_US: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
pub static PREPARE_US: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Wall time of the render thread's frame, first render system to last,
/// in microseconds. Under pipelined rendering the main thread waits for
/// this at the end of its own frame, so it is the other half of the
/// `extract+wait render` leg.
pub static RENDER_US: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

#[derive(Resource, Default)]
struct RenderFrameClock(Option<std::time::Instant>);

fn render_frame_begin(mut clock: ResMut<RenderFrameClock>) {
    clock.0 = Some(std::time::Instant::now());
}

fn render_frame_end(clock: Res<RenderFrameClock>) {
    if let Some(t0) = clock.0 {
        RENDER_US.store(
            t0.elapsed().as_micros() as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

fn extract_instance_data(
    main_entities: Extract<Query<(&RenderEntity, &InstanceMaterialData)>>,
    mut extracted: Query<&mut ExtractedInstances>,
    mut commands: Commands,
) {
    let t0 = std::time::Instant::now();
    for (render_entity, data) in &main_entities {
        let e = render_entity.id();
        if let Ok(mut ex) = extracted.get_mut(e) {
            ex.0.clear();
            ex.0.extend_from_slice(&data.0);
        } else {
            commands.entity(e).insert(ExtractedInstances(data.0.clone()));
        }
    }
    EXTRACT_US.store(
        t0.elapsed().as_micros() as u32,
        std::sync::atomic::Ordering::Relaxed,
    );
}

pub struct UnitRenderPlugin;

impl Plugin for UnitRenderPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/unit_instancing.wgsl");
        // Registers the SyncToRenderWorld requirement so instance entities
        // get a render-world twin (ExtractComponentPlugin used to do this).
        app.add_plugins(SyncComponentPlugin::<InstanceMaterialData>::default())
            .init_resource::<RenderCounts>()
            .init_resource::<LodConfig>()
            .init_resource::<Corpses>()
            .init_resource::<crate::gait::Legs>()
            .add_plugins(crate::render_units_gpu::GpuUnitRenderPlugin)
            .add_systems(Startup, setup_unit_mesh)
            // Must run after the camera moves: culling builds a FRESH
            // frustum from this frame's camera transform (the Frustum
            // component is one frame stale — visible pop while panning).
            .add_systems(
                Update,
                sync_instance_data
                    .after(crate::camera::apply_camera_transform)
                    .run_if(cpu_sweep),
            );
        app.sub_app_mut(RenderApp)
            .init_resource::<RenderFrameClock>()
            .add_systems(ExtractSchedule, extract_instance_data)
            .add_systems(
                Render,
                (
                    render_frame_begin.in_set(RenderSystems::ExtractCommands),
                    render_frame_end.in_set(RenderSystems::PostCleanup),
                ),
            )
            .add_render_command::<Transparent3d, DrawCustom>()
            .init_resource::<SpecializedMeshPipelines<CustomPipeline>>()
            .init_resource::<SpecializedRenderPipelines<CustomPipeline>>()
            .add_systems(
                RenderStartup,
                init_custom_pipeline.after(MeshPipelineSystems),
            )
            .add_systems(
                Render,
                (
                    queue_custom.in_set(RenderSystems::QueueMeshes),
                    prepare_instance_buffers.in_set(RenderSystems::PrepareResources),
                ),
            );
    }
}

fn setup_unit_mesh(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    gpu: Res<GpuSyncConfig>,
    mut legs: ResMut<crate::gait::Legs>,
) {
    // One instance entity per unit kind AND detail level, each with its
    // own code-built mesh. Instance positions are not the entity's transform;
    // built-in frustum culling would cull all instances at once, so it
    // stays disabled and we cull per-instance in sync_instance_data.
    //
    // NoAutomaticBatching is REQUIRED, not an optimization toggle: these
    // entities compare batch-equal (same pipeline, draw function, material
    // slot), so bevy's sorted-phase batcher merges their phase items and
    // `SortedRenderPhase::render_range` skips every item after the first —
    // its draw function never runs and that bucket's units silently vanish
    // (the "LOD far bucket invisible" bug, devlog 0013).
    for kind in 0..crate::unit_types::NUM_KINDS {
        let lods = crate::unit_glb::kind_lods(kind);
        let tris: Vec<usize> =
            lods.iter().map(|m| m.indices().map_or(0, |i| i.len() / 3)).collect();
        // The gait pose is built on the leg this mesh has.
        if let Some(leg) = crate::gait::measure(&lods[0]) {
            legs.0[kind] = leg;
        }
        info!(
            "unit meshes: kind {kind} tris per level {tris:?}, leg {:.2} m",
            legs.0[kind]
        );
        // One bucket per detail level, shared by the living and the fallen.
        for (lod, mesh) in lods.into_iter().enumerate() {
            let bucket = InstanceBucket(bucket_of(kind, lod));
            // GPU mode: the bucket also carries its expanded mesh for the
            // pulled draw. Its instance data stays, and stays empty.
            let pulled = pull_mesh_for(&mesh, &bucket, &gpu);
            let mut entity = commands.spawn((
                Mesh3d(meshes.add(mesh)),
                InstanceMaterialData::default(),
                bucket,
                NoFrustumCulling,
                NoAutomaticBatching,
            ));
            if let Some(pulled) = pulled {
                entity.insert(pulled);
            }
        }
    }
}

/// Copy the SoA sim state into the instance buffer (main world side):
/// interpolate between fixed ticks, frustum-cull per instance, tint the
/// selection. Culling is strictly visibility: a unit is skipped only when
/// its bounding sphere is outside the camera frustum.
/// Units per parallel sync chunk.
pub(crate) const SYNC_CHUNK: usize = 16_384;

/// Anim z-channel encoding — the ONE authoritative map (keep in sync
/// with unit_instancing.wgsl, which decodes it):
///   z > 0, < CELEBRATE_BASE: attack = style * 2 + wind-up progress
///     (style 0 stab, 1 classic swing, 2 slash/benched);
///   z >= CELEBRATE_BASE: victory cheer, fraction = progress 0..1;
///   z < 0: stance-band magnitude — tiers 0.25 enemy-near, 0.5 fighting
///     wavering, 0.65 fighting confident, 1.0 charging — smoothed per
///     unit (~0.35 s) before emission so poses never snap.
pub(crate) const CELEBRATE_BASE: f32 = 6.0;

/// Battle stance tier of a regiment, the negative anim z band: 0.25 =
/// enemy in watch range (standing units brace), 0.5 = fighting but
/// wavering (morale low, braces, no taunts), 0.65 = fighting confident,
/// 1.0 = charging (sprint lean and stride). Plain moves carry lowered.
/// Both render paths read these helpers, so they can never disagree.
pub(crate) fn stance_tier(g: &crate::orders::GroupData) -> f32 {
    if g.charging {
        1.0
    } else if g.engaged || matches!(g.order, Some(crate::orders::Order::Attack(_))) {
        if crate::morale::band(g) == crate::morale::Band::Steady { 0.65 } else { 0.5 }
    } else if g.enemy_near {
        0.25
    } else {
        0.0
    }
}

/// Victory cheer progress 0..1, negative when the regiment is not
/// celebrating. Rides the positive band as CELEBRATE_BASE + progress so
/// the shader can ease in and out.
pub(crate) fn celebrate_progress(g: &crate::orders::GroupData) -> f32 {
    if g.celebrate > 0 {
        1.0 - g.celebrate as f32 / crate::frontline::CELEBRATE_TICKS as f32
    } else {
        -1.0
    }
}

/// Wall stance (shieldwall or spearwall by kind, the bucket knows which).
pub(crate) fn wall_signal(g: &crate::orders::GroupData) -> f32 {
    if g.spacing == crate::formation::FormSpacing::Wall && !g.state.is_broken() {
        1.0
    } else {
        0.0
    }
}

/// Per-soldier render state carried between frames, index-aligned with
/// `Units`. The death sweep swap-removes soldiers, and the one it moves
/// into a freed slot takes over that slot's state. Same fields in the
/// same order as `Smooth` in unit_build.wgsl, the GPU path's copy.
#[derive(Clone, Copy, Default)]
struct Smooth {
    /// Smoothed ground speed, m/s.
    walk: f32,
    band: f32,
    wall: f32,
    /// Gait phase in cycles (gait.rs).
    gait: f32,
    /// Detail level last frame (hysteresis).
    lod: u8,
}

#[allow(clippy::too_many_arguments)] // bevy system params
fn sync_instance_data(
    units: Res<Units>,
    selection: Res<crate::orders::Selection>,
    hover: Res<crate::orders::Hover>,
    groups: Res<crate::orders::Groups>,
    time: Res<Time>,
    fixed_time: Res<Time<Fixed>>,
    lod_cfg: Res<LodConfig>,
    corpses: Res<Corpses>,
    camera: Query<(&Camera, &Projection, &Transform), With<Camera3d>>,
    mut query: Query<(&InstanceBucket, &mut InstanceMaterialData)>,
    mut counts: ResMut<RenderCounts>,
    mut no_cull: Local<Option<bool>>,
    mut scratch: Local<Vec<[Vec<InstanceData>; NUM_BUCKETS]>>,
    mut corpse_scratch: Local<Vec<[Vec<InstanceData>; NUM_LODS]>>,
    mut smooth: Local<Vec<Smooth>>,
    (gpu, frame, legs): (
        Res<GpuSyncConfig>,
        Res<bevy::diagnostic::FrameCount>,
        Res<crate::gait::Legs>,
    ),
) {
    let _span = info_span!("sync_instances").entered();
    let t0 = std::time::Instant::now();
    let Ok((cam, projection, cam_tf)) = camera.single() else {
        return;
    };
    // Bucket id -> instance vec, indexable during the unit sweep.
    let mut buckets: Vec<(usize, Mut<InstanceMaterialData>)> = query
        .iter_mut()
        .map(|(bucket, data)| (bucket.0, data))
        .collect();
    if buckets.len() != NUM_BUCKETS {
        return; // instance entities not spawned yet
    }
    buckets.sort_unstable_by_key(|(id, _)| *id);
    debug_assert!(buckets.iter().enumerate().all(|(i, (id, _))| i == *id));
    // Fresh frustum from THIS frame's camera state (camera has no parent,
    // so Transform is authoritative).
    let clip_from_world = projection.get_clip_from_view() * cam_tf.to_matrix().inverse();
    let frustum = Frustum(ViewFrustum::from_clip_from_world(&clip_from_world));
    let cull = !*no_cull.get_or_insert_with(|| std::env::var("FL_NO_CULL").is_ok());
    let alpha = fixed_time.overstep_fraction();
    // Detail levels: a soldier of height H at distance d covers
    // H * px_per_unit / d pixels, from the live field of view and
    // viewport height. Euclidean distance to the camera, not view depth:
    // turning the camera in place then never changes a level.
    let px_per_unit = match (projection, cam.physical_viewport_size()) {
        (Projection::Perspective(p), Some(size)) => {
            size.y as f32 / (2.0 * (p.fov * 0.5).tan())
        }
        _ => 0.0,
    };
    let bands = LodBands::new(&lod_cfg, px_per_unit);
    let bands = &bands;
    let lod_debug = lod_cfg.debug;
    let cam_pos = cam_tf.translation;

    const HIGHLIGHT: [f32; 4] = [1.0, 1.0, 0.55, 1.0];
    // Attack-preview tint: the enemy regiment a right-click would target.
    const HOSTILE: [f32; 4] = [1.0, 0.30, 0.22, 1.0];
    let has_sel = selection.regiments.iter().any(|s| *s);
    let hover_enemy = hover.enemy;
    // Broken regiments render desaturated (no extra instance data needed).
    let broken: Vec<bool> = groups.list.iter().map(|g| g.state.is_broken()).collect();
    let broken = &broken[..];
    // One value per regiment: stance tier, cheer progress, wall signal.
    // Shared with the GPU path.
    let stance: Vec<f32> = groups.list.iter().map(stance_tier).collect();
    let stance = &stance[..];
    let celebrating: Vec<f32> = groups.list.iter().map(celebrate_progress).collect();
    let celebrating = &celebrating[..];
    let walled: Vec<f32> = groups.list.iter().map(wall_signal).collect();
    let walled = &walled[..];
    // Copied out of the resource so the parallel chunks can read it.
    let legs = legs.0;

    // Parallel cull + bucket build into per-chunk scratch, then one memcpy
    // concat per bucket. The scratch vecs keep their allocations across
    // frames (Local).
    let n_chunks = units.len().div_ceil(SYNC_CHUNK);
    if scratch.len() < n_chunks {
        scratch.resize_with(n_chunks, Default::default);
    }
    let units = &*units;
    let selection = &*selection;
    let frustum = &frustum;
    let inv_dt = 1.0 / fixed_time.timestep().as_secs_f32().max(1e-6);
    // Per-unit walk-signal smoothing (~0.25 s): positional-correction
    // shoves arrive in single-tick bursts; unsmoothed they strobe the
    // walk cycle on/off, which reads as sliding with a twitch. Indices
    // shuffle on death-sweep swap-removes — a one-frame inherited value
    // is invisible.
    let smooth = &mut *smooth;
    smooth.resize(units.len(), Smooth::default());
    // The fallen: one job per SYNC_CHUNK of each kind's list, run next to
    // the unit chunks. Their instance data is frozen, so a job is only a
    // cull, a level pick and a copy. No hysteresis: bodies do not move.
    let corpse_jobs: Vec<(usize, &[InstanceData])> = corpses
        .data
        .iter()
        .enumerate()
        .flat_map(|(kind, bodies)| bodies.chunks(SYNC_CHUNK).map(move |c| (kind, c)))
        .collect();
    if corpse_scratch.len() < corpse_jobs.len() {
        corpse_scratch.resize_with(corpse_jobs.len(), Default::default);
    }
    let dt = time.delta_secs();
    let ema_k = (dt / 0.25).min(1.0);
    // Stance tiers are per-regiment and snap; the POSE blends (~0.35 s).
    let band_k = (dt / 0.35).min(1.0);
    // The wall signal blends a touch slower: forming a wall is a
    // deliberate act, not a snap.
    let wall_k = (dt / 0.5).min(1.0);
    bevy::tasks::ComputeTaskPool::get().scope(|scope| {
        for (ci, (chunk_scratch, smooth_chunk)) in scratch
            .iter_mut()
            .zip(smooth.chunks_mut(SYNC_CHUNK))
            .enumerate()
            .take(n_chunks)
        {
            scope.spawn(async move {
                for vec in chunk_scratch.iter_mut() {
                    vec.clear();
                }
                let start = ci * SYNC_CHUNK;
                let end = (start + SYNC_CHUNK).min(units.len());
                for i in start..end {
                    // Walk signal: smoothed ACTUAL per-tick displacement.
                    // Velocity misses positional-correction shoves (which
                    // also kill vel), raw displacement strobes on bursty
                    // corr chains. Updated before the cull so units
                    // re-entering the frustum have a live value.
                    let step = units.pos[i] - units.pos_prev[i];
                    let disp = Vec2::new(step.x, step.z).length() * inv_dt;
                    let gi = units.group[i] as usize;
                    let kind = units.kind[i] as usize;
                    let sm = &mut smooth_chunk[i - start];
                    sm.walk += (disp - sm.walk) * ema_k;
                    let tier = stance.get(gi).copied().unwrap_or(0.0);
                    sm.band += (tier - sm.band) * band_k;
                    sm.wall += (walled.get(gi).copied().unwrap_or(0.0) - sm.wall) * wall_k;
                    sm.gait = crate::gait::advance(sm.gait, crate::gait::rate(sm.walk), dt);

                    let position = units.pos_prev[i].lerp(units.pos[i], alpha);
                    let sphere = Sphere {
                        center: position.into(),
                        radius: CULL_RADIUS,
                    };
                    // intersect_far = false: skip the far-plane test so
                    // distant vistas keep their units.
                    if cull && !frustum.intersects_sphere(&sphere, false) {
                        continue;
                    }
                    // Detail level: jittered distance against this
                    // frame's thresholds, held inside the hysteresis
                    // bounds. Indices shuffle on death-sweep swap-removes
                    // and culled soldiers skip this. A stale previous
                    // level is pulled back inside the bounds at once.
                    let jitter = lod_jitter(units.color[i][3]);
                    let d2 = position.distance_squared(cam_pos) * jitter * jitter;
                    sm.lod = sm.lod.clamp(
                        LodBands::level(&bands.fine[kind], d2),
                        LodBands::level(&bands.coarse[kind], d2),
                    );
                    let lod = sm.lod as usize;
                    let mut color = units.color[i];
                    if broken.get(gi).copied().unwrap_or(false) {
                        let gray =
                            0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2];
                        for c in color.iter_mut().take(3) {
                            *c = *c * 0.55 + gray * 0.45;
                        }
                    } else if has_sel
                        && selection.regiments.get(gi).copied().unwrap_or(false)
                    {
                        for c in 0..3 {
                            color[c] = color[c] * 0.35 + HIGHLIGHT[c] * 0.65;
                        }
                    } else if hover_enemy == Some(gi as u32) {
                        for c in 0..3 {
                            color[c] = color[c] * 0.45 + HOSTILE[c] * 0.55;
                        }
                    }
                    if lod_debug {
                        lod_debug_tint(&mut color, lod);
                    }
                    // Facing interpolates like position (wrap-aware), so
                    // per-tick yaw updates don't snap at render rates.
                    let dy = (units.yaw[i] - units.yaw_prev[i] + std::f32::consts::PI)
                        .rem_euclid(std::f32::consts::TAU)
                        - std::f32::consts::PI;
                    let yaw = units.yaw_prev[i] + dy * alpha;
                    // Attack lunge ramps up quadratically over the wind-up
                    // and snaps back on the strike (chunky, readable).
                    // Charging blows lunge harder (arm angles saturate in
                    // the shader; the extra goes into body lean).
                    let sw = units.swing[i];
                    let lunge = if sw & crate::units::SWING_STATE_MASK
                        == crate::units::SWING_WINDUP
                    {
                        // A bow draw runs on the missile draw time, not
                        // the melee wind-up (the same 0..1 progress then
                        // drives the bow-arm raise + string pull).
                        let w = if sw & crate::units::SWING_RANGED != 0 {
                            crate::unit_types::missile::DRAW_TICKS as f32
                        } else {
                            crate::unit_types::TYPES[units.kind[i] as usize].windup_ticks as f32
                        };
                        // Clamped: the draw-start jitter can put swing_t
                        // above the nominal draw time.
                        let t = ((w - units.swing_t[i] as f32) / w.max(1.0)).max(0.0);
                        let charge = sw & crate::units::SWING_CHARGE != 0;
                        let amp = if charge { 1.35 } else { 1.0 };
                        // A charging swing raises from the leveled run-in
                        // point instead of dipping the blade first (0.18
                        // puts the raise curve at the charge point angle).
                        let lunge =
                            if charge { (t * t * amp).max(0.18) } else { t * t * amp };
                        // Style (stab/slash, picked at wind-up start)
                        // rides the 2s digit of the positive band.
                        let style = ((sw & crate::units::SWING_STYLE_MASK)
                            >> crate::units::SWING_STYLE_SHIFT)
                            as f32;
                        style * 2.0 + lunge
                    } else if units.death_t[i] == 0
                        && celebrating.get(gi).copied().unwrap_or(-1.0) >= 0.0
                    {
                        CELEBRATE_BASE + celebrating[gi]
                    } else if units.death_t[i] == 0 {
                        // Negative lunge = SMOOTHED battle stance (the
                        // regiment tier snaps; a pose must not —
                        // one-frame stance changes aren't immersive).
                        -sm.band
                    } else {
                        0.0
                    };
                    // fx: [0,1) hit flash, [1,2] death progress.
                    let fx = if units.death_t[i] > 0 {
                        2.0 - units.death_t[i] as f32 / crate::movement::DEATH_TICKS as f32
                    } else {
                        units.flash[i] as f32 * 0.25
                    };
                    // Stagger progress (1 at the blow, 0 recovered): the
                    // rocked-back pose. Death pose owns dying men.
                    let stagger = if units.death_t[i] == 0
                        && sw & crate::units::SWING_STAGGERED != 0
                    {
                        // Normalized by the STUMBLE length: an ordinary
                        // stumble plays the full rock for its 0.5 s; a
                        // charge impact or impalement holds it ~1 s.
                        (units.swing_t[i] as f32
                            / crate::movement::HIT_STAGGER_TICKS as f32)
                            .min(1.0)
                    } else {
                        0.0
                    };
                    chunk_scratch[bucket_of(kind, lod)].push(InstanceData {
                        position,
                        scale: 1.0,
                        color,
                        anim: [yaw, sm.walk, lunge, fx],
                        anim2: [legs[kind], sm.wall, sm.gait, stagger],
                    });
                }
            });
        }
        for ((kind, bodies), out) in corpse_jobs.iter().copied().zip(corpse_scratch.iter_mut()) {
            scope.spawn(async move {
                for vec in out.iter_mut() {
                    vec.clear();
                }
                for body in bodies {
                    let sphere = Sphere {
                        center: body.position.into(),
                        radius: CULL_RADIUS,
                    };
                    if cull && !frustum.intersects_sphere(&sphere, false) {
                        continue;
                    }
                    let jitter = lod_jitter(body.color[3]);
                    let d2 = body.position.distance_squared(cam_pos) * jitter * jitter;
                    let lod = LodBands::level(&bands.plain[kind], d2) as usize;
                    let mut body = *body;
                    if lod_debug {
                        lod_debug_tint(&mut body.color, lod);
                    }
                    out[lod].push(body);
                }
            });
        }
    });
    for (b, (_, data)) in buckets.iter_mut().enumerate() {
        data.clear();
        for chunk_scratch in scratch.iter().take(n_chunks) {
            data.extend_from_slice(&chunk_scratch[b]);
        }
    }
    // Per-bucket counts: the living, read before the fallen join the
    // buckets, and the fallen as they join.
    let living: [u32; NUM_BUCKETS] = std::array::from_fn(|b| buckets[b].1.len() as u32);
    let mut fallen = [0u32; NUM_BUCKETS];
    for ((kind, _), out) in corpse_jobs.iter().zip(corpse_scratch.iter()) {
        for (lod, bodies) in out.iter().enumerate() {
            buckets[bucket_of(*kind, lod)].1.extend_from_slice(bodies);
            fallen[bucket_of(*kind, lod)] += bodies.len() as u32;
        }
    }
    counts.total = units.len();
    if gpu.enabled && gpu.check {
        // FL_GPU_CHECK: the GPU path owns the display. This sweep only
        // records what it would have drawn, for the readback to compare.
        counts.check.push_back((frame.0, living, fallen));
        while counts.check.len() > 8 {
            counts.check.pop_front();
        }
    } else {
        counts.set_from_buckets(&living, &fallen);
    }
    counts.sync_ms = t0.elapsed().as_secs_f32() * 1000.0;
}

#[allow(clippy::too_many_arguments)] // bevy system params
fn queue_custom(
    transparent_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Res<CustomPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<CustomPipeline>>,
    mut pull_pipelines: ResMut<SpecializedRenderPipelines<CustomPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    maybe_batched_instance_buffers: Option<
        Res<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    >,
    material_meshes: Query<(Entity, &MainEntity, Option<&PullMeshGpu>), With<ExtractedInstances>>,
    gpu_input: Option<Res<GpuUnitInput>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<&ExtractedView>,
    view_key_cache: Res<ViewKeyCache>,
) {
    let draw_custom = transparent_3d_draw_functions.read().id::<DrawCustom>();
    let lod_debug = gpu_input.is_some_and(|g| g.lod_debug);

    for view in &views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };

        let Some(&view_key) = view_key_cache.get(&view.retained_view_entity) else {
            continue;
        };

        for (entity, main_entity, pull_mesh) in &material_meshes {
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(*main_entity)
            else {
                continue;
            };
            let Some(mesh) = meshes.get(mesh_instance.mesh_asset_id()) else {
                continue;
            };
            let key = view_key
                | MeshPipelineKey::from_primitive_topology_and_strip_index(
                    mesh.primitive_topology(),
                    mesh.index_format(),
                );
            let pipeline = match pull_mesh {
                // GPU mode: the pulled variant, no vertex buffers.
                Some(pull_mesh) => pull_pipelines.specialize(
                    &pipeline_cache,
                    &custom_pipeline,
                    PullPipelineKey {
                        mesh: key,
                        layout: mesh.layout.clone(),
                        verts: pull_mesh.count,
                        bucket: pull_mesh.bucket as u32,
                        lod_debug,
                    },
                ),
                None => pipelines
                    .specialize(&pipeline_cache, &custom_pipeline, key, &mesh.layout)
                    .unwrap(),
            };
            transparent_phase.add_retained(Transparent3d {
                sorting_info: TransparentSortingInfo3d::Sorted {
                    mesh_center: pbr::get_mesh_instance_world_from_local(
                        *main_entity,
                        mesh_instance.current_uniform_index,
                        &render_mesh_instances,
                        maybe_batched_instance_buffers.as_deref(),
                    )
                    .transform_point3(
                        meshes
                            .get(mesh_instance.mesh_asset_id())
                            .unwrap()
                            .aabb_center,
                    ),
                    depth_bias: 0.0,
                },
                entity: (entity, *main_entity),
                pipeline,
                draw_function: draw_custom,
                distance: 0.0,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: true,
            });
        }
    }
}

#[derive(Component)]
pub(crate) struct InstanceBuffer {
    buffer: Buffer,
    length: usize,
    capacity: usize,
}

/// Persistent GPU buffer per instance entity: written in place each frame,
/// reallocated (with slack) only on growth past capacity.
pub(crate) fn prepare_instance_buffers(
    mut commands: Commands,
    mut query: Query<(Entity, &ExtractedInstances, Option<&mut InstanceBuffer>)>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let t0 = std::time::Instant::now();
    for (entity, instances, existing) in &mut query {
        let n = instances.0.len();
        match existing {
            Some(mut buf) if buf.capacity >= n => {
                if n > 0 {
                    queue.write_buffer(&buf.buffer, 0, bytemuck::cast_slice(&instances.0));
                }
                buf.length = n;
            }
            _ => {
                let capacity = (n + n / 2).max(1024);
                let buffer = render_device.create_buffer(&BufferDescriptor {
                    label: Some("unit instance buffer"),
                    size: (capacity * size_of::<InstanceData>()) as u64,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                if n > 0 {
                    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&instances.0));
                }
                commands.entity(entity).insert(InstanceBuffer {
                    buffer,
                    length: n,
                    capacity,
                });
            }
        }
    }
    PREPARE_US.store(
        t0.elapsed().as_micros() as u32,
        std::sync::atomic::Ordering::Relaxed,
    );
}

#[derive(Resource)]
pub(crate) struct CustomPipeline {
    shader: Handle<Shader>,
    mesh_pipeline: MeshPipeline,
    /// Group 3 of a pulled bucket: the instance records, the index list,
    /// the bucket's mesh corners, the bucket table.
    pub(crate) pull_layout: BindGroupLayoutDescriptor,
}

fn init_custom_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mesh_pipeline: Res<MeshPipeline>,
) {
    commands.insert_resource(CustomPipeline {
        shader: load_embedded_asset!(asset_server.as_ref(), "shaders/unit_instancing.wgsl"),
        mesh_pipeline: mesh_pipeline.clone(),
        pull_layout: BindGroupLayoutDescriptor::new(
            "unit pull layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (
                    binding_types::storage_buffer_read_only_sized(false, None),
                    binding_types::storage_buffer_read_only_sized(false, None),
                    binding_types::storage_buffer_read_only_sized(false, None),
                    binding_types::storage_buffer_read_only_sized(false, None),
                ),
            ),
        ),
    });
}

/// Pipeline variant of a pulled bucket. It has NO vertex buffers, and
/// `SpecializedMeshPipelines` caches by vertex buffer 0, so it goes
/// through the plain render pipeline specializer with the mesh layout
/// riding in the key.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct PullPipelineKey {
    mesh: MeshPipelineKey,
    layout: MeshVertexBufferLayoutRef,
    /// Corners per soldier. A shader constant, so the soldier lookup
    /// divides by a literal.
    verts: u32,
    /// The bucket this pipeline draws, also a shader constant.
    bucket: u32,
    /// FL_LOD_DEBUG: tint by level in the vertex shader.
    lod_debug: bool,
}

impl SpecializedRenderPipeline for CustomPipeline {
    type Key = PullPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        // Same targets, depth and view bindings as the instanced variant.
        let mut descriptor = self
            .mesh_pipeline
            .specialize(key.mesh, &key.layout)
            .expect("unit meshes carry every attribute the mesh pipeline asks for");

        descriptor.vertex.shader = self.shader.clone();
        descriptor.fragment.as_mut().unwrap().shader = self.shader.clone();
        // The shader pulls mesh and soldier from group 3 by vertex index.
        descriptor.vertex.buffers.clear();
        descriptor.vertex.entry_point = Some("vertex_pull".into());
        let defs = &mut descriptor.vertex.shader_defs;
        defs.push("VERTEX_PULL".into());
        defs.push(bevy::shader::ShaderDefVal::UInt("PULL_VERTS".into(), key.verts));
        defs.push(bevy::shader::ShaderDefVal::UInt("PULL_BUCKET".into(), key.bucket));
        if key.lod_debug {
            defs.push("LOD_DEBUG".into());
        }
        descriptor.set_layout(3, self.pull_layout.clone());
        descriptor
    }
}

impl SpecializedMeshPipeline for CustomPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key, layout)?;

        descriptor.vertex.shader = self.shader.clone();
        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: size_of::<InstanceData>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                // Locations 8-10: clear of bevy's mesh attributes
                // (0 position, 1 normal, 2 uv, 5 vertex color, 6-7 joints).
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 8,
                },
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: VertexFormat::Float32x4.size(),
                    shader_location: 9,
                },
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: VertexFormat::Float32x4.size() * 2,
                    shader_location: 10,
                },
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: VertexFormat::Float32x4.size() * 3,
                    shader_location: 11,
                },
            ],
        });
        descriptor.fragment.as_mut().unwrap().shader = self.shader.clone();
        Ok(descriptor)
    }
}

type DrawCustom = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    SetMeshBindGroup<2>,
    DrawMeshInstanced,
);

struct DrawMeshInstanced;

impl<P: PhaseItem> RenderCommand<P> for DrawMeshInstanced {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderMeshInstances>,
        SRes<MeshAllocator>,
        SRes<GpuUnitBuffers>,
    );
    type ViewQuery = ();
    type ItemQuery = (Option<Read<InstanceBuffer>>, Option<Read<PulledBucketGpu>>);

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        bucket: Option<(Option<&'w InstanceBuffer>, Option<&'w PulledBucketGpu>)>,
        (meshes, render_mesh_instances, mesh_allocator, gpu): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        // A borrow check workaround.
        let mesh_allocator = mesh_allocator.into_inner();

        let Some((instance_buffer, pulled)) = bucket else {
            return RenderCommandResult::Skip;
        };
        // GPU mode: ONE plain draw over every corner of every soldier in
        // the bucket. The count comes from the indirect buffer the compute
        // pass wrote, no vertex buffers, no instances.
        if let Some(pulled) = pulled {
            let Some(alloc) = &gpu.into_inner().alloc else {
                return RenderCommandResult::Skip;
            };
            pass.set_bind_group(3, &pulled.bind_group, &[]);
            pass.draw_indirect(&alloc.args, pulled.bucket as u64 * 16);
            return RenderCommandResult::Success;
        }

        let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(item.main_entity())
        else {
            return RenderCommandResult::Skip;
        };
        let Some(gpu_mesh) = meshes.into_inner().get(mesh_instance.mesh_asset_id()) else {
            return RenderCommandResult::Skip;
        };
        let Some(instance_buffer) = instance_buffer else {
            return RenderCommandResult::Skip;
        };
        // Most kind-by-level buckets are empty in any one view.
        if instance_buffer.length == 0 {
            return RenderCommandResult::Skip;
        }
        let Some(vertex_buffer_slice) =
            mesh_allocator.mesh_vertex_slice(&mesh_instance.mesh_asset_id())
        else {
            return RenderCommandResult::Skip;
        };

        pass.set_vertex_buffer(0, vertex_buffer_slice.buffer.slice(..));
        pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));

        match &gpu_mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                index_format,
                count,
            } => {
                let Some(index_buffer_slice) =
                    mesh_allocator.mesh_index_slice(&mesh_instance.mesh_asset_id())
                else {
                    return RenderCommandResult::Skip;
                };

                pass.set_index_buffer(index_buffer_slice.buffer.slice(..), *index_format);
                pass.draw_indexed(
                    index_buffer_slice.range.start..(index_buffer_slice.range.start + count),
                    vertex_buffer_slice.range.start as i32,
                    0..instance_buffer.length as u32,
                );
            }
            RenderMeshBufferInfo::NonIndexed => {
                pass.draw(vertex_buffer_slice.range, 0..instance_buffer.length as u32);
            }
        }
        RenderCommandResult::Success
    }
}
