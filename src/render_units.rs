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

use crate::units::Units;

/// Bounding-sphere radius for per-instance frustum culling: cube diagonal
/// plus a generous margin so nothing pops inside the screen edge.
const CULL_RADIUS: f32 = 2.5;

/// Instances drawn this frame after culling (overlay diagnostics).
#[derive(Resource, Default)]
pub struct RenderCounts {
    pub drawn: usize,
    pub total: usize,
    /// Per-bucket drawn counts (bucket = team today, unit type later).
    pub bucket_drawn: Vec<usize>,
    /// Cost of sync_instance_data this frame (cull + bucket build).
    pub sync_ms: f32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct InstanceData {
    pub position: Vec3,
    pub scale: f32,
    /// rgb = team color; a = stable per-unit anim seed (NOT opacity).
    pub color: [f32; 4],
    /// x = yaw, y = move amount 0..1 (walk bob/lean),
    /// z = lunge 0..1 (attack), w = fx: [0,1) hit flash, [1,2] death.
    pub anim: [f32; 4],
}

#[derive(Component, Deref, DerefMut, Default)]
pub struct InstanceMaterialData(pub Vec<InstanceData>);

/// One instanced draw per bucket. Buckets are keyed by unit kind: one
/// low-poly mesh per kind, team identity stays per-instance color.
#[derive(Component)]
pub struct InstanceBucket(pub usize);

/// Number of instance buckets (== instance entities == draw calls).
pub const NUM_BUCKETS: usize = crate::unit_types::NUM_KINDS;

/// Which bucket a unit renders in.
#[inline]
fn bucket_of(units: &Units, i: usize) -> usize {
    units.kind[i] as usize
}

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
            // Must run after the camera moves: culling builds a FRESH
            // frustum from this frame's camera transform (the Frustum
            // component is one frame stale — visible pop while panning).
            .add_systems(
                Update,
                sync_instance_data.after(crate::camera::apply_camera_transform),
            );
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
    // One instance entity per unit kind, each with its own code-built
    // low-poly mesh. Instance positions are not the entity's transform;
    // built-in frustum culling would cull all instances at once, so it
    // stays disabled and we cull per-instance in sync_instance_data.
    //
    // NoAutomaticBatching is REQUIRED, not an optimization toggle: these
    // entities compare batch-equal (same pipeline, draw function, material
    // slot), so bevy's sorted-phase batcher merges their phase items and
    // `SortedRenderPhase::render_range` skips every item after the first —
    // its draw function never runs and that bucket's units silently vanish
    // (the "LOD far bucket invisible" bug, devlog 0013).
    let kind_meshes = [
        crate::unit_meshes::build_knight(),
        crate::unit_meshes::build_man_at_arms(),
    ];
    for (bucket, mesh) in kind_meshes.into_iter().enumerate() {
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            InstanceMaterialData::default(),
            InstanceBucket(bucket),
            NoFrustumCulling,
            NoAutomaticBatching,
        ));
    }
}

/// Copy the SoA sim state into the instance buffer (main world side):
/// interpolate between fixed ticks, frustum-cull per instance, tint the
/// selection. Culling is strictly visibility: a unit is skipped only when
/// its bounding sphere is outside the camera frustum.
/// Units per parallel sync chunk.
const SYNC_CHUNK: usize = 16_384;

#[allow(clippy::too_many_arguments)] // bevy system params
fn sync_instance_data(
    units: Res<Units>,
    selection: Res<crate::orders::Selection>,
    groups: Res<crate::orders::Groups>,
    fixed_time: Res<Time<Fixed>>,
    camera: Query<(&Projection, &Transform), With<Camera3d>>,
    mut query: Query<(&InstanceBucket, &mut InstanceMaterialData)>,
    mut counts: ResMut<RenderCounts>,
    mut no_cull: Local<Option<bool>>,
    mut scratch: Local<Vec<[Vec<InstanceData>; NUM_BUCKETS]>>,
) {
    let _span = info_span!("sync_instances").entered();
    let t0 = std::time::Instant::now();
    let Ok((projection, cam_tf)) = camera.single() else {
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

    const HIGHLIGHT: [f32; 4] = [1.0, 1.0, 0.55, 1.0];
    let has_sel = selection.regiments.iter().any(|s| *s);
    // Broken regiments render desaturated (no extra instance data needed).
    let broken: Vec<bool> = groups.list.iter().map(|g| g.state.is_broken()).collect();
    let broken = &broken[..];

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
    bevy::tasks::ComputeTaskPool::get().scope(|scope| {
        for (ci, chunk_scratch) in scratch.iter_mut().enumerate().take(n_chunks) {
            scope.spawn(async move {
                for vec in chunk_scratch.iter_mut() {
                    vec.clear();
                }
                let start = ci * SYNC_CHUNK;
                let end = (start + SYNC_CHUNK).min(units.len());
                for i in start..end {
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
                    let mut color = units.color[i];
                    if broken.get(units.group[i] as usize).copied().unwrap_or(false) {
                        let gray =
                            0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2];
                        for c in color.iter_mut().take(3) {
                            *c = *c * 0.55 + gray * 0.45;
                        }
                    } else if has_sel
                        && selection
                            .regiments
                            .get(units.group[i] as usize)
                            .copied()
                            .unwrap_or(false)
                    {
                        for c in 0..3 {
                            color[c] = color[c] * 0.35 + HIGHLIGHT[c] * 0.65;
                        }
                    }
                    // Deadband + smoothstep: crowd-jitter velocities must
                    // not flicker the walk cycle on and off every frame.
                    let speed_ratio =
                        (units.vel[i].length() / units.speed[i].max(0.01)).clamp(0.0, 1.0);
                    let t = ((speed_ratio - 0.12) / (0.45 - 0.12)).clamp(0.0, 1.0);
                    let move_amount = t * t * (3.0 - 2.0 * t);
                    // Facing interpolates like position (wrap-aware), so
                    // per-tick yaw updates don't snap at render rates.
                    let dy = (units.yaw[i] - units.yaw_prev[i] + std::f32::consts::PI)
                        .rem_euclid(std::f32::consts::TAU)
                        - std::f32::consts::PI;
                    let yaw = units.yaw_prev[i] + dy * alpha;
                    // Attack lunge ramps up quadratically over the wind-up
                    // and snaps back on the strike (chunky, readable).
                    let lunge = if units.swing[i] == crate::units::SWING_WINDUP {
                        let w = crate::unit_types::TYPES[units.kind[i] as usize].windup_ticks
                            as f32;
                        let t = (w - units.swing_t[i] as f32) / w.max(1.0);
                        t * t
                    } else {
                        0.0
                    };
                    // fx: [0,1) hit flash, [1,2] death progress.
                    let fx = if units.death_t[i] > 0 {
                        2.0 - units.death_t[i] as f32 / crate::movement::DEATH_TICKS as f32
                    } else {
                        units.flash[i] as f32 * 0.25
                    };
                    chunk_scratch[bucket_of(units, i)].push(InstanceData {
                        position,
                        scale: 1.0,
                        color,
                        anim: [yaw, move_amount, lunge, fx],
                    });
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
    counts.drawn = buckets.iter().map(|(_, d)| d.len()).sum();
    counts.total = units.len();
    counts.bucket_drawn.clear();
    counts
        .bucket_drawn
        .extend(buckets.iter().map(|(_, d)| d.len()));
    counts.sync_ms = t0.elapsed().as_secs_f32() * 1000.0;
}

#[allow(clippy::too_many_arguments)] // bevy system params
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
