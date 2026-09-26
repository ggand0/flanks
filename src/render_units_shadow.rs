//! Soldiers cast sun shadows (docs/plans/012-graphics-roadmap.md item 3).
//!
//! The unit buckets draw into the sun's cascaded shadow maps through
//! Bevy's own `Shadow` phase, as non-mesh items beside the meshes Bevy
//! queues there. Each item is a depth-only variant of the unit pipeline
//! with the same vertex shader and pose code.
//!
//! GPU path: the build pass (unit_build.wgsl) puts every near soldier
//! whose sphere overlaps a cascade's box on that cascade's caster list for
//! his kind, off screen or not, so a cascade draws only what can shade it.
//! A cascade draws a kind with one level for all its casters: the level
//! the camera picks for a soldier as many pixels tall as his filtered
//! shadow shows detail (`render_units::shadow_level`). No per-soldier CPU
//! work.
//!
//! CPU path (`FL_GPU_SYNC=0`): every cascade draws the camera's near
//! buckets from their instance buffers, the fallback's simple form.
//!
//! Only the near levels cast (`CAST_LODS`). Farther soldiers are a few
//! pixels tall and their shadows would be a few pixels.
//!
//! The whole shadow pass is timed on the GPU as
//! `render/sun_shadows/elapsed_gpu` (`ShadowPassTimer`).

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};

use wgpu::{QuerySet, QuerySetDescriptor, QueryType};

use bevy::{
    core_pipeline::{Core3d, Core3dSystems, core_3d::CORE_3D_DEPTH_FORMAT},
    diagnostic::{Diagnostic, DiagnosticPath, Diagnostics, RegisterDiagnostic},
    ecs::system::{SystemParamItem, lifetimeless::*},
    mesh::MeshVertexBufferLayoutRef,
    pbr::{
        EARLY_SHADOW_PASS, LightEntity, Shadow, ShadowBatchSetKey, ShadowBinKey,
        per_view_shadow_pass, queue_shadows,
    },
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        globals::{GlobalsBuffer, GlobalsUniform},
        mesh::{RenderMesh, allocator::MeshAllocator},
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, InputUniformIndex, PhaseItem,
            RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass,
            ViewBinnedRenderPhases,
        },
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        sync_world::MainEntity,
        view::{ExtractedView, ViewUniform, ViewUniformOffset, ViewUniforms},
    },
};

use crate::render_units::{
    CustomPipeline, DrawMeshInstanced, ExtractedBucket, NUM_BUCKETS, NUM_LODS, PullPipelineKey,
    UnitMeshKey,
};
use crate::render_units_gpu::{
    GpuUnitBuffers, GpuUnitInput, MAX_CASCADES, PullMeshGpu, PulledBucketGpu,
};
use crate::unit_types::NUM_KINDS;

/// Detail levels that cast: soldiers the camera would draw at L0 or L1.
pub(crate) const CAST_LODS: usize = 2;

/// `FL_UNIT_SHADOWS=0`: soldiers cast nothing while the rest of the scene
/// keeps its shadows, the A/B for their share of the shadow pass.
pub(crate) fn unit_shadows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::util::env_or("FL_UNIT_SHADOWS", 1_u32) != 0)
}

/// The depth-only unit pipeline, specialized per bucket like the main one.
#[derive(Resource)]
pub(crate) struct UnitShadowPipeline {
    custom: CustomPipeline,
    /// Group 0: the light view's uniform and the globals, at the bindings
    /// `bevy_pbr::mesh_view_bindings` declares them (0 and 11), so the unit
    /// shader's imports resolve unchanged in the shadow pass.
    view_layout: BindGroupLayoutDescriptor,
    /// Groups 1 and 2: the unit shader uses neither.
    empty_layout: BindGroupLayoutDescriptor,
    /// The device clamps depth instead of clipping it. Without that, the
    /// vertex shader clamps (`UNIT_SHADOW_DEPTH_CLAMP`): a soldier between
    /// the sun and a cascade's near plane must still cast.
    unclipped_depth: bool,
}

impl UnitShadowPipeline {
    /// The unit pipeline turned depth-only for a cascade: no fragment
    /// stage, the shadow map's depth format, one sample, and the shadow
    /// view bindings in groups 0 to 2. Group 3, the bucket's, stays.
    fn shadow_descriptor(&self, mut d: RenderPipelineDescriptor) -> RenderPipelineDescriptor {
        d.label = Some("unit shadow pipeline".into());
        let bucket_layout = d.layout[3].clone();
        d.layout = vec![
            self.view_layout.clone(),
            self.empty_layout.clone(),
            self.empty_layout.clone(),
            bucket_layout,
        ];
        d.fragment = None;
        d.primitive.unclipped_depth = self.unclipped_depth;
        if !self.unclipped_depth {
            d.vertex.shader_defs.push("UNIT_SHADOW_DEPTH_CLAMP".into());
        }
        d.depth_stencil = Some(DepthStencilState {
            format: CORE_3D_DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::GreaterEqual),
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        });
        d.multisample = MultisampleState::default();
        d
    }
}

impl SpecializedRenderPipeline for UnitShadowPipeline {
    type Key = PullPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        self.shadow_descriptor(SpecializedRenderPipeline::specialize(&self.custom, key))
    }
}

impl SpecializedMeshPipeline for UnitShadowPipeline {
    type Key = UnitMeshKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        Ok(self.shadow_descriptor(SpecializedMeshPipeline::specialize(
            &self.custom,
            key,
            layout,
        )?))
    }
}

fn init_unit_shadow_pipeline(
    mut commands: Commands,
    custom: Res<CustomPipeline>,
    device: Res<RenderDevice>,
) {
    commands.insert_resource(UnitShadowPipeline {
        custom: custom.clone(),
        view_layout: BindGroupLayoutDescriptor::new(
            "unit shadow view layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::VERTEX,
                (
                    (0, binding_types::uniform_buffer::<ViewUniform>(true)),
                    (11, binding_types::uniform_buffer::<GlobalsUniform>(false)),
                ),
            ),
        ),
        empty_layout: BindGroupLayoutDescriptor::new("unit shadow empty layout", &[]),
        unclipped_depth: device.features().contains(WgpuFeatures::DEPTH_CLIP_CONTROL),
    });
}

/// The shadow pass bind groups, rebuilt every frame: the view uniform
/// buffer can move between frames.
#[derive(Resource, Default)]
struct UnitShadowBindGroups {
    view: Option<BindGroup>,
    empty: Option<BindGroup>,
}

fn prepare_unit_shadow_bind_groups(
    pipeline: Res<UnitShadowPipeline>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    view_uniforms: Res<ViewUniforms>,
    globals: Res<GlobalsBuffer>,
    mut groups: ResMut<UnitShadowBindGroups>,
) {
    groups.view = match (view_uniforms.uniforms.binding(), globals.buffer.binding()) {
        (Some(view), Some(globals)) => Some(device.create_bind_group(
            "unit shadow view bind group",
            &pipeline_cache.get_bind_group_layout(&pipeline.view_layout),
            &BindGroupEntries::with_indices(((0, view), (11, globals))),
        )),
        _ => None,
    };
    if groups.empty.is_none() {
        groups.empty = Some(device.create_bind_group(
            "unit shadow empty bind group",
            &pipeline_cache.get_bind_group_layout(&pipeline.empty_layout),
            &[],
        ));
    }
}

/// Put the unit casters into every sun cascade's shadow phase. The phase
/// is retained and Bevy's own queue removes what it did not add, so each
/// item is taken out and put back every frame: a handful of hash map
/// operations, no per-soldier work.
#[allow(clippy::too_many_arguments, clippy::type_complexity)] // bevy system params
fn queue_unit_shadows(
    draw_functions: Res<DrawFunctions<Shadow>>,
    pipeline: Res<UnitShadowPipeline>,
    mut pull_pipelines: ResMut<SpecializedRenderPipelines<UnitShadowPipeline>>,
    mut mesh_pipelines: ResMut<SpecializedMeshPipelines<UnitShadowPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    mesh_allocator: Res<MeshAllocator>,
    render_mesh_instances: Res<bevy::pbr::RenderMeshInstances>,
    gpu_input: Res<GpuUnitInput>,
    buckets: Query<(Entity, &MainEntity, &ExtractedBucket, Option<&PullMeshGpu>)>,
    light_views: Query<(&LightEntity, &ExtractedView)>,
    mut phases: ResMut<ViewBinnedRenderPhases<Shadow>>,
) {
    let draw_pulled = draw_functions.read().id::<DrawUnitShadowPulled>();
    let draw_instanced = draw_functions.read().id::<DrawUnitShadowInstanced>();
    for (light, view) in &light_views {
        let &LightEntity::Directional { cascade_index, .. } = light else {
            continue;
        };
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        for (entity, main_entity, bucket, pull_mesh) in &buckets {
            phase.remove(*main_entity);
            if !unit_shadows() || cascade_index >= MAX_CASCADES {
                continue;
            }
            let (kind, lod) = (bucket.0 / NUM_LODS, bucket.0 % NUM_LODS);
            // GPU path: the one level this cascade draws the kind with.
            // CPU path: the near levels, as the camera drew them.
            let casts = match pull_mesh {
                Some(_) => lod == gpu_input.shadow_levels[cascade_index][kind] as usize,
                None => lod < CAST_LODS,
            };
            if !casts {
                continue;
            }
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(*main_entity)
            else {
                continue;
            };
            let mesh_id = mesh_instance.mesh_asset_id();
            let (Some(mesh), Some(slabs)) = (meshes.get(mesh_id), mesh_allocator.mesh_slabs(&mesh_id))
            else {
                continue;
            };
            let mesh_key = bevy::pbr::MeshPipelineKey::from_msaa_samples(1)
                | bevy::pbr::MeshPipelineKey::VIEW_PROJECTION_ORTHOGRAPHIC
                | bevy::pbr::MeshPipelineKey::from_primitive_topology_and_strip_index(
                    mesh.primitive_topology(),
                    mesh.index_format(),
                );
            let (pipeline_id, draw_function) = match pull_mesh {
                Some(pull_mesh) => (
                    pull_pipelines.specialize(
                        &pipeline_cache,
                        &pipeline,
                        PullPipelineKey::shadow(mesh_key, mesh.layout.clone(), pull_mesh),
                    ),
                    draw_pulled,
                ),
                None => match mesh_pipelines.specialize(
                    &pipeline_cache,
                    &pipeline,
                    UnitMeshKey::shadow(mesh_key),
                    &mesh.layout,
                ) {
                    Ok(id) => (id, draw_instanced),
                    Err(err) => {
                        error!("unit shadow pipeline: {err}");
                        continue;
                    }
                },
            };
            phase.add(
                ShadowBatchSetKey {
                    pipeline: pipeline_id,
                    draw_function,
                    material_bind_group_index: None,
                    slabs,
                },
                ShadowBinKey {
                    asset_id: mesh_id.untyped(),
                },
                (entity, *main_entity),
                InputUniformIndex::default(),
                BinnedRenderPhaseType::NonMesh,
            );
        }
    }
}

type DrawUnitShadowPulled = (SetItemPipeline, SetUnitShadowViewGroups, DrawCasterList);
type DrawUnitShadowInstanced = (SetItemPipeline, SetUnitShadowViewGroups, DrawMeshInstanced);

/// A pulled bucket drawn into one cascade: its group 3 with the cascade's
/// set of the bucket table, and the indirect arguments of the cascade's
/// caster list for the bucket's kind (unit_build.wgsl `finalize`).
struct DrawCasterList;

impl<P: PhaseItem> RenderCommand<P> for DrawCasterList {
    type Param = SRes<GpuUnitBuffers>;
    type ViewQuery = Read<LightEntity>;
    type ItemQuery = Read<PulledBucketGpu>;

    #[inline]
    fn render<'w>(
        _item: &P,
        light: &'w LightEntity,
        pulled: Option<&'w PulledBucketGpu>,
        gpu: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let &LightEntity::Directional { cascade_index, .. } = light else {
            return RenderCommandResult::Skip;
        };
        let (Some(pulled), Some(alloc)) = (pulled, &gpu.into_inner().alloc) else {
            return RenderCommandResult::Skip;
        };
        let Some(group) = pulled.shadow.get(cascade_index) else {
            return RenderCommandResult::Skip;
        };
        let kind = pulled.bucket as usize / NUM_LODS;
        let list = NUM_BUCKETS + cascade_index * NUM_KINDS + kind;
        pass.set_bind_group(3, group, &[]);
        pass.draw_indirect(&alloc.args, list as u64 * 16);
        RenderCommandResult::Success
    }
}

/// Groups 0 to 2 of the shadow pipeline: the light view at its uniform
/// offset, then the two empty groups.
struct SetUnitShadowViewGroups;

impl<P: PhaseItem> RenderCommand<P> for SetUnitShadowViewGroups {
    type Param = SRes<UnitShadowBindGroups>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        view_offset: &'w ViewUniformOffset,
        _entity: Option<()>,
        groups: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let groups = groups.into_inner();
        let (Some(view), Some(empty)) = (&groups.view, &groups.empty) else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(0, view, &[view_offset.offset]);
        pass.set_bind_group(1, empty, &[]);
        pass.set_bind_group(2, empty, &[]);
        RenderCommandResult::Success
    }
}

pub struct UnitShadowPlugin;

impl Plugin for UnitShadowPlugin {
    fn build(&self, app: &mut App) {
        app.register_diagnostic(Diagnostic::new(SHADOW_PASS_GPU).with_suffix(" ms"))
            .add_systems(Update, publish_shadow_pass_time);
        app.sub_app_mut(RenderApp)
            .init_resource::<UnitShadowBindGroups>()
            .init_resource::<SpecializedRenderPipelines<UnitShadowPipeline>>()
            .init_resource::<SpecializedMeshPipelines<UnitShadowPipeline>>()
            .add_render_command::<Shadow, DrawUnitShadowPulled>()
            .add_render_command::<Shadow, DrawUnitShadowInstanced>()
            .add_systems(
                RenderStartup,
                (
                    init_unit_shadow_pipeline.after(crate::render_units::init_custom_pipeline),
                    init_shadow_pass_timer,
                ),
            )
            .add_systems(
                Render,
                (
                    queue_unit_shadows
                        .in_set(RenderSystems::QueueMeshes)
                        .after(queue_shadows),
                    prepare_unit_shadow_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                    shadow_timer_map.in_set(RenderSystems::Cleanup),
                ),
            )
            .add_systems(
                Core3d,
                (
                    shadow_timer_begin
                        .after(crate::render_units_gpu::run_unit_build_pass)
                        .before(per_view_shadow_pass::<EARLY_SHADOW_PASS>),
                    shadow_timer_end
                        .after(per_view_shadow_pass::<EARLY_SHADOW_PASS>)
                        .before(Core3dSystems::MainPass),
                ),
            );
    }
}

/// GPU time of the sun's shadow pass: every cascade, Bevy's meshes and the
/// soldiers together. Bevy times its main passes but not this one, and its
/// span recorder cannot open a span in one render system and close it in
/// another, so two timestamps bracket the pass here. The value joins the
/// `render/.../elapsed_gpu` lines of the periodic log.
pub const SHADOW_PASS_GPU: DiagnosticPath =
    DiagnosticPath::const_new("render/sun_shadows/elapsed_gpu");

/// Last read shadow pass time, milliseconds as f32 bits; `u32::MAX` until
/// the first readback.
static SHADOW_PASS_MS: AtomicU32 = AtomicU32::new(u32::MAX);

const TIMER_IDLE: u8 = 0;
/// The resolve and copy are recorded in this frame's commands.
const TIMER_COPIED: u8 = 1;
/// Waiting for the map. The readback buffer must not be copied into.
const TIMER_MAPPING: u8 = 2;

#[derive(Resource)]
struct ShadowPassTimer {
    queries: QuerySet,
    resolve: Buffer,
    readback: Buffer,
    /// Nanoseconds per timestamp tick.
    period: f32,
    state: Arc<AtomicU8>,
}

fn init_shadow_pass_timer(mut commands: Commands, device: Res<RenderDevice>, queue: Res<RenderQueue>) {
    let needed = WgpuFeatures::TIMESTAMP_QUERY | WgpuFeatures::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    if !device.features().contains(needed) {
        return;
    }
    let buffer = |label, usage| {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: 16,
            usage,
            mapped_at_creation: false,
        })
    };
    commands.insert_resource(ShadowPassTimer {
        queries: device.wgpu_device().create_query_set(&QuerySetDescriptor {
            label: Some("sun shadow pass timestamps"),
            ty: QueryType::Timestamp,
            count: 2,
        }),
        resolve: buffer(
            "sun shadow pass timestamps resolve",
            BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
        ),
        readback: buffer(
            "sun shadow pass timestamps readback",
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        ),
        period: queue.get_timestamp_period(),
        state: Arc::new(AtomicU8::new(TIMER_IDLE)),
    });
}

fn shadow_timer_begin(timer: Option<Res<ShadowPassTimer>>, mut ctx: RenderContext) {
    if let Some(timer) = timer {
        ctx.command_encoder().write_timestamp(&timer.queries, 0);
    }
}

fn shadow_timer_end(timer: Option<Res<ShadowPassTimer>>, mut ctx: RenderContext) {
    let Some(timer) = timer else {
        return;
    };
    let encoder = ctx.command_encoder();
    encoder.write_timestamp(&timer.queries, 1);
    if timer.state.load(Ordering::Acquire) == TIMER_IDLE {
        encoder.resolve_query_set(&timer.queries, 0..2, &timer.resolve, 0);
        encoder.copy_buffer_to_buffer(&timer.resolve, 0, &timer.readback, 0, 16);
        timer.state.store(TIMER_COPIED, Ordering::Release);
    }
}

/// After the frame's submit, as Bevy's own readback does: map, read the two
/// stamps, unmap.
fn shadow_timer_map(timer: Option<Res<ShadowPassTimer>>) {
    let Some(timer) = timer else {
        return;
    };
    if timer.state.load(Ordering::Acquire) != TIMER_COPIED {
        return;
    }
    timer.state.store(TIMER_MAPPING, Ordering::Release);
    let buffer = timer.readback.clone();
    let state = timer.state.clone();
    let period = timer.period;
    timer.readback.slice(..).map_async(MapMode::Read, move |res| {
        if res.is_ok() {
            let stamps: [u64; 2] = bytemuck::pod_read_unaligned(&buffer.slice(..).get_mapped_range());
            buffer.unmap();
            let ms = stamps[1].wrapping_sub(stamps[0]) as f64 * period as f64 * 1e-6;
            SHADOW_PASS_MS.store((ms as f32).to_bits(), Ordering::Relaxed);
        }
        state.store(TIMER_IDLE, Ordering::Release);
    });
}

fn publish_shadow_pass_time(mut diagnostics: Diagnostics) {
    let bits = SHADOW_PASS_MS.load(Ordering::Relaxed);
    if bits != u32::MAX {
        diagnostics.add_measurement(&SHADOW_PASS_GPU, || f32::from_bits(bits) as f64);
    }
}
