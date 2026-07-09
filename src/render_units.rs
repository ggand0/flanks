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
        batching::gpu_preprocessing::BatchedInstanceBuffers,
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
use bytemuck::{Pod, Zeroable};

use crate::units::Units;

/// Instances farther than this from the camera go in the far-LOD bucket.
const LOD_DISTANCE: f32 = 300.0;
/// Bounding-sphere radius for per-instance frustum culling (covers the
/// cube diagonal plus interpolation slop).
const CULL_RADIUS: f32 = 1.3;

/// Which LOD bucket an instance entity renders (0 = near, 1 = far).
/// Far bucket currently uses the same cube; the mesh slot is the point —
/// cheaper LOD meshes and per-type meshes drop in here later.
#[derive(Component)]
pub struct UnitLod(pub u8);

/// Instances drawn this frame after culling (overlay diagnostics).
#[derive(Resource, Default)]
pub struct RenderCounts {
    pub near: usize,
    pub far: usize,
    pub total: usize,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct InstanceData {
    pub position: Vec3,
    pub scale: f32,
    pub color: [f32; 4],
}

#[derive(Component, Deref, DerefMut, Default)]
pub struct InstanceMaterialData(pub Vec<InstanceData>);

impl SyncComponent for InstanceMaterialData {
    type Target = Self;
}

/// Render-world copy of the instance data. Persistent component: the Vec's
/// allocation is reused every frame (extraction copies into it, no clone).
#[derive(Component, Default)]
struct ExtractedInstances(Vec<InstanceData>);

fn extract_instance_data(
    main_entities: Extract<Query<(&RenderEntity, &InstanceMaterialData)>>,
    mut extracted: Query<&mut ExtractedInstances>,
    mut commands: Commands,
) {
    for (render_entity, data) in &main_entities {
        let e = render_entity.id();
        if let Ok(mut ex) = extracted.get_mut(e) {
            ex.0.clear();
            ex.0.extend_from_slice(&data.0);
        } else {
            commands.entity(e).insert(ExtractedInstances(data.0.clone()));
        }
    }
}

pub struct UnitRenderPlugin;

impl Plugin for UnitRenderPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/unit_instancing.wgsl");
        // Registers the SyncToRenderWorld requirement so instance entities
        // get a render-world twin (ExtractComponentPlugin used to do this).
        app.add_plugins(SyncComponentPlugin::<InstanceMaterialData>::default())
            .init_resource::<RenderCounts>()
            .add_systems(Startup, setup_unit_mesh)
            .add_systems(Update, sync_instance_data);
        app.sub_app_mut(RenderApp)
            .add_systems(ExtractSchedule, extract_instance_data)
            .add_render_command::<Transparent3d, DrawCustom>()
            .init_resource::<SpecializedMeshPipelines<CustomPipeline>>()
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

fn setup_unit_mesh(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    // One instance entity per LOD bucket. Slightly taller than wide: reads
    // as a soldier, not a bead. Far mesh is identical for now — the slot
    // exists so cheaper LOD / per-type meshes drop in without infra work.
    // Instance positions are not the entity's transform; built-in frustum
    // culling would cull all instances at once, so it stays disabled and
    // we cull per-instance in sync_instance_data.
    for lod in 0..2u8 {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(0.62, 0.9, 0.62))),
            InstanceMaterialData::default(),
            UnitLod(lod),
            NoFrustumCulling,
        ));
    }
}

/// Copy the SoA sim state into the per-instance buffers (main world side):
/// interpolate between fixed ticks, frustum-cull per instance, split into
/// near/far LOD buckets, tint the selection.
fn sync_instance_data(
    units: Res<Units>,
    selection: Res<crate::orders::Selection>,
    fixed_time: Res<Time<Fixed>>,
    camera: Query<(&Frustum, &GlobalTransform), With<Camera3d>>,
    mut query: Query<(&mut InstanceMaterialData, &UnitLod)>,
    mut counts: ResMut<RenderCounts>,
) {
    let _span = info_span!("sync_instances").entered();
    let Ok((frustum, cam_tf)) = camera.single() else {
        return;
    };
    let cam_pos = cam_tf.translation();
    let alpha = fixed_time.overstep_fraction();

    let mut near = None;
    let mut far = None;
    for (data, lod) in &mut query {
        if lod.0 == 0 {
            near = Some(data);
        } else {
            far = Some(data);
        }
    }
    let (Some(mut near), Some(mut far)) = (near, far) else {
        return;
    };
    near.clear();
    far.clear();

    const HIGHLIGHT: [f32; 4] = [1.0, 1.0, 0.55, 1.0];
    let has_sel = selection.mask.len() == units.len();
    let lod_d2 = LOD_DISTANCE * LOD_DISTANCE;
    for i in 0..units.len() {
        let position = units.pos_prev[i].lerp(units.pos[i], alpha);
        let sphere = Sphere {
            center: position.into(),
            radius: CULL_RADIUS,
        };
        // intersect_far = false: skip the far-plane test so distant vistas
        // keep their units.
        if !frustum.intersects_sphere(&sphere, false) {
            continue;
        }
        let mut color = units.color[i];
        if has_sel && selection.mask[i] {
            for c in 0..3 {
                color[c] = color[c] * 0.35 + HIGHLIGHT[c] * 0.65;
            }
        }
        let instance = InstanceData {
            position,
            scale: 1.0,
            color,
        };
        if position.distance_squared(cam_pos) < lod_d2 {
            near.push(instance);
        } else {
            far.push(instance);
        }
    }
    counts.near = near.len();
    counts.far = far.len();
    counts.total = units.len();
}

fn queue_custom(
    transparent_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Res<CustomPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<CustomPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    maybe_batched_instance_buffers: Option<
        Res<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    >,
    material_meshes: Query<(Entity, &MainEntity), With<ExtractedInstances>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<&ExtractedView>,
    view_key_cache: Res<ViewKeyCache>,
) {
    let draw_custom = transparent_3d_draw_functions.read().id::<DrawCustom>();

    for view in &views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };

        let Some(&view_key) = view_key_cache.get(&view.retained_view_entity) else {
            continue;
        };

        for (entity, main_entity) in &material_meshes {
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
            let pipeline = pipelines
                .specialize(&pipeline_cache, &custom_pipeline, key, &mesh.layout)
                .unwrap();
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
struct InstanceBuffer {
    buffer: Buffer,
    length: usize,
    capacity: usize,
}

/// Persistent GPU buffer per instance entity: written in place each frame,
/// reallocated (with slack) only on growth past capacity.
fn prepare_instance_buffers(
    mut commands: Commands,
    mut query: Query<(Entity, &ExtractedInstances, Option<&mut InstanceBuffer>)>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
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
}

#[derive(Resource)]
struct CustomPipeline {
    shader: Handle<Shader>,
    mesh_pipeline: MeshPipeline,
}

fn init_custom_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mesh_pipeline: Res<MeshPipeline>,
) {
    commands.insert_resource(CustomPipeline {
        shader: load_embedded_asset!(asset_server.as_ref(), "shaders/unit_instancing.wgsl"),
        mesh_pipeline: mesh_pipeline.clone(),
    });
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
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 3, // 0-2 are Position, Normal, UV
                },
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: VertexFormat::Float32x4.size(),
                    shader_location: 4,
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
    );
    type ViewQuery = ();
    type ItemQuery = Read<InstanceBuffer>;

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        instance_buffer: Option<&'w InstanceBuffer>,
        (meshes, render_mesh_instances, mesh_allocator): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        // A borrow check workaround.
        let mesh_allocator = mesh_allocator.into_inner();

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
