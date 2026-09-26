//! Soldiers cast sun shadows (docs/plans/012-graphics-roadmap.md item 3).
//!
//! The unit buckets draw into the sun's cascaded shadow maps through
//! Bevy's own `Shadow` phase, as non-mesh items beside the meshes Bevy
//! queues there. Each item is a depth-only variant of the unit pipeline:
//! the same vertex shader and pose code, the same bucket bind group and
//! the same draw (`DrawMeshInstanced`), so on the GPU path a cascade
//! redraws the index lists and indirect arguments the build pass wrote
//! for the camera, with no per-soldier CPU work.
//!
//! Only the near levels cast (`CAST_LODS`). Farther soldiers are a few
//! pixels tall and their shadows would be smaller than a shadow texel.
//!
//! Known limitation: the lists are culled by the camera frustum, not the
//! light's, so a soldier just off the screen edge toward the sun casts
//! nothing. A light-frustum test in the build compute is the later fix.

use bevy::{
    core_pipeline::core_3d::CORE_3D_DEPTH_FORMAT,
    ecs::system::{SystemParamItem, lifetimeless::*},
    mesh::MeshVertexBufferLayoutRef,
    pbr::{LightEntity, Shadow, ShadowBatchSetKey, ShadowBinKey, queue_shadows},
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
        renderer::RenderDevice,
        sync_world::MainEntity,
        view::{ExtractedView, ViewUniform, ViewUniformOffset, ViewUniforms},
    },
};

use crate::render_units::{
    CustomPipeline, DrawMeshInstanced, ExtractedBucket, NUM_LODS, PullPipelineKey, UnitMeshKey,
};
use crate::render_units_gpu::PullMeshGpu;

/// Detail levels that cast: L0 and L1.
const CAST_LODS: usize = 2;

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

/// Put the near-level buckets into every sun cascade's shadow phase.
/// The phase is retained and Bevy's own queue removes what it did not
/// add, so each item is taken out and put back every frame: a handful
/// of hash map operations, no per-soldier work.
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
    buckets: Query<(Entity, &MainEntity, &ExtractedBucket, Option<&PullMeshGpu>)>,
    light_views: Query<(&LightEntity, &ExtractedView)>,
    mut phases: ResMut<ViewBinnedRenderPhases<Shadow>>,
) {
    let draw = draw_functions.read().id::<DrawUnitShadow>();
    for (light, view) in &light_views {
        if !matches!(light, LightEntity::Directional { .. }) {
            continue;
        }
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        for (entity, main_entity, bucket, pull_mesh) in &buckets {
            phase.remove(*main_entity);
            if bucket.0 % NUM_LODS >= CAST_LODS {
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
            let pipeline_id = match pull_mesh {
                Some(pull_mesh) => pull_pipelines.specialize(
                    &pipeline_cache,
                    &pipeline,
                    PullPipelineKey::shadow(mesh_key, mesh.layout.clone(), pull_mesh),
                ),
                None => match mesh_pipelines.specialize(
                    &pipeline_cache,
                    &pipeline,
                    UnitMeshKey::shadow(mesh_key),
                    &mesh.layout,
                ) {
                    Ok(id) => id,
                    Err(err) => {
                        error!("unit shadow pipeline: {err}");
                        continue;
                    }
                },
            };
            phase.add(
                ShadowBatchSetKey {
                    pipeline: pipeline_id,
                    draw_function: draw,
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

type DrawUnitShadow = (SetItemPipeline, SetUnitShadowViewGroups, DrawMeshInstanced);

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
        app.sub_app_mut(RenderApp)
            .init_resource::<UnitShadowBindGroups>()
            .init_resource::<SpecializedRenderPipelines<UnitShadowPipeline>>()
            .init_resource::<SpecializedMeshPipelines<UnitShadowPipeline>>()
            .add_render_command::<Shadow, DrawUnitShadow>()
            .add_systems(
                RenderStartup,
                init_unit_shadow_pipeline.after(crate::render_units::init_custom_pipeline),
            )
            .add_systems(
                Render,
                (
                    queue_unit_shadows
                        .in_set(RenderSystems::QueueMeshes)
                        .after(queue_shadows),
                    prepare_unit_shadow_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            );
    }
}
