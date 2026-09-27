//! Selection rings: a circle on the ground under every soldier of a
//! selected or hovered regiment, M2TW's selection marker. An evenly filled
//! see-through disc with a point at the front where the soldier faces,
//! ETW's teardrop. Selected regiments are a muted green, the player's
//! regiment under the cursor (or its card) a fainter green, the enemy
//! under the cursor a muted red.
//!
//! GPU path: the build pass (unit_build.wgsl `append_ring`) puts each
//! visible soldier of a flagged regiment on the ring list in the tail of
//! the index list, and one indirect draw reads his interpolated record.
//! CPU path (`FL_GPU_SYNC=0`): the CPU sweep collects the same entries
//! with their records, uploaded here and drawn directly. One shader
//! (shaders/unit_rings.wgsl) serves both.
//!
//! Each ring corner sits on the terrain: the shader samples an uploaded
//! copy of the height field, re-sent when the terrain changes (craters)
//! while rings are shown. On the bridge deck, where the soldier stands
//! above the field, the ring lies flat at his feet.
//!
//! The draw joins the Transparent3d phase only on frames with something
//! to ring, so an empty selection costs nothing.

use std::sync::Arc;

use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    core_pipeline::core_3d::{Transparent3d, TransparentSortingInfo3d},
    ecs::system::{SystemParamItem, lifetimeless::*},
    mesh::MeshVertexBufferLayoutRef,
    pbr::{
        MeshPipeline, MeshPipelineKey, RenderMeshInstances, SetMeshViewBindGroup,
        SetMeshViewBindingArrayBindGroup, ViewKeyCache,
    },
    prelude::*,
    render::{
        ExtractSchedule, MainWorld, Render, RenderApp, RenderStartup, RenderSystems,
        mesh::RenderMesh,
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        sync_world::MainEntity,
        view::ExtractedView,
    },
};

use crate::render_units::{ExtractedBucket, InstanceData};
use crate::render_units_gpu::{GpuSyncConfig, GpuUnitBuffers, RING_ARG};
use crate::unit_types::NUM_KINDS;

/// Ring radius in metres: the move preview's slot circles
/// (orders.rs `draw_order_preview`), so a slot of the preview and the
/// soldier who takes it read the same.
const RADIUS: f32 = 0.42;
/// The teardrop point on the front of the ring, where the soldier faces.
const FACING_POINT: bool = true;

/// Ring styles, in the order unit_rings.wgsl indexes its colours.
const STYLE_SELECTED: u32 = 0;
const STYLE_HOVER_OWN: u32 = 1;
const STYLE_HOVER_ENEMY: u32 = 2;

/// A regiment's ring style: the enemy under the cursor over the
/// selection, the selection over the player's regiment under the cursor.
/// The GPU build picks the same way (unit_build.wgsl `append_ring`).
pub fn ring_style(selected: bool, hover_enemy: bool, hover_own: bool) -> Option<u32> {
    if hover_enemy {
        Some(STYLE_HOVER_ENEMY)
    } else if selected {
        Some(STYLE_SELECTED)
    } else if hover_own {
        Some(STYLE_HOVER_OWN)
    } else {
        None
    }
}

/// The CPU path's rings for this frame: the ringed soldiers' records, and
/// one entry per ring (record index | style << 28 | kind << 30), filled by
/// render_units.rs `sync_instance_data`.
#[derive(Resource, Default)]
pub struct CpuRings {
    pub records: Vec<InstanceData>,
    pub entries: Vec<u32>,
}

/// Whether any regiment is ringed this frame.
#[derive(Resource, Default)]
struct RingsActive(bool);

/// The height field as the ring shader samples it. Copied from `Terrain`
/// when it changed and rings are shown; the generation tells the render
/// world to upload.
#[derive(Resource, Default)]
struct RingTerrain {
    heights: Arc<Vec<f32>>,
    generation: u32,
    /// The field changed while no rings were shown: copy at the next ring.
    stale: bool,
    origin: Vec2,
    verts: UVec2,
}

fn update_rings_active(
    state: Res<State<crate::game_state::GameState>>,
    selection: Res<crate::orders::Selection>,
    hover: Res<crate::orders::Hover>,
    groups: Res<crate::orders::Groups>,
    mut active: ResMut<RingsActive>,
) {
    let selected = selection
        .regiments
        .iter()
        .zip(&groups.list)
        .any(|(s, gd)| *s && !gd.state.is_broken());
    let on = *state.get() == crate::game_state::GameState::Battle
        && (selected || hover.enemy.is_some() || hover.own.is_some());
    if active.0 != on {
        active.0 = on;
    }
}

fn update_ring_terrain(
    terrain: Res<crate::terrain::Terrain>,
    active: Res<RingsActive>,
    mut ring_terrain: ResMut<RingTerrain>,
) {
    let first = ring_terrain.heights.is_empty();
    if terrain.is_changed() && !first {
        ring_terrain.stale = true;
    }
    if !(first || ring_terrain.stale && active.0) {
        return;
    }
    let (vx, vz) = crate::terrain::grid_verts();
    ring_terrain.heights = Arc::new(terrain.heights().to_vec());
    ring_terrain.origin = terrain.origin;
    ring_terrain.verts = UVec2::new(vx as u32, vz as u32);
    ring_terrain.generation = ring_terrain.generation.wrapping_add(1);
    ring_terrain.stale = false;
}

/// The ring shader's uniform (`RingParams` in unit_rings.wgsl).
#[derive(ShaderType, Clone, Copy, Default)]
struct RingParams {
    /// Linear rgb and an alpha scale, per style.
    colors: [Vec4; 3],
    /// Per kind: how far a soldier's position stands above his feet.
    half_heights: Vec4,
    /// xy = the height field's origin, z = its cell, w = the ring radius.
    terrain: Vec4,
    /// xy = the height field's vertex counts, z = 1 with the facing point,
    /// w = the first ring entry in the entry buffer.
    grid: UVec4,
}

/// The extracted frame, render world side.
#[derive(Resource, Default)]
struct RingInput {
    active: bool,
    gpu: bool,
    cpu: CpuRings,
    heights: Arc<Vec<f32>>,
    heights_generation: u32,
    params: RingParams,
}

fn extract_rings(mut main_world: ResMut<MainWorld>, mut input: ResMut<RingInput>) {
    input.active = main_world.resource::<RingsActive>().0;
    input.gpu = main_world.resource::<GpuSyncConfig>().enabled;
    if !input.active {
        return;
    }
    if !input.gpu {
        let mut cpu = main_world.resource_mut::<CpuRings>();
        std::mem::swap(&mut cpu.records, &mut input.cpu.records);
        std::mem::swap(&mut cpu.entries, &mut input.cpu.entries);
    }
    let terrain = main_world.resource::<RingTerrain>();
    if terrain.generation != input.heights_generation {
        input.heights = terrain.heights.clone();
        input.heights_generation = terrain.generation;
    }
    let linear = |c: Color, a: f32| {
        let l = c.to_linear();
        Vec4::new(l.red, l.green, l.blue, a)
    };
    // Muted colours: a dense formation covers the ground with them, and a
    // bright one glares.
    let green = Color::srgb(0.42, 0.74, 0.46);
    input.params = RingParams {
        colors: [
            linear(green, 1.0),
            linear(green, 0.55),
            linear(Color::srgb(0.80, 0.30, 0.24), 1.0),
        ],
        half_heights: Vec4::from_array(std::array::from_fn(crate::unit_types::half_height)),
        terrain: Vec4::new(terrain.origin.x, terrain.origin.y, crate::terrain::CELL, RADIUS),
        grid: UVec4::new(terrain.verts.x, terrain.verts.y, FACING_POINT as u32, 0),
    };
}

#[derive(Resource)]
struct RingPipeline {
    mesh_pipeline: MeshPipeline,
    shader: Handle<Shader>,
    /// Group 2: the records, the ring entries, the height field, the
    /// uniform.
    layout: BindGroupLayoutDescriptor,
}

fn init_ring_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mesh_pipeline: Res<MeshPipeline>,
) {
    commands.insert_resource(RingPipeline {
        mesh_pipeline: mesh_pipeline.clone(),
        shader: load_embedded_asset!(asset_server.as_ref(), "shaders/unit_rings.wgsl"),
        layout: BindGroupLayoutDescriptor::new(
            "selection ring layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (
                    binding_types::storage_buffer_read_only_sized(false, None),
                    binding_types::storage_buffer_read_only_sized(false, None),
                    binding_types::storage_buffer_read_only_sized(false, None),
                    binding_types::uniform_buffer::<RingParams>(false)
                        .visibility(ShaderStages::VERTEX_FRAGMENT),
                ),
            ),
        ),
    });
}

/// The view's mesh pipeline key and any unit mesh's layout: the ring
/// pipeline starts from the mesh pipeline for the view's targets, depth
/// and view bindings, as the pulled unit pipeline does.
#[derive(Clone, PartialEq, Eq, Hash)]
struct RingPipelineKey {
    mesh: MeshPipelineKey,
    layout: MeshVertexBufferLayoutRef,
}

impl SpecializedRenderPipeline for RingPipeline {
    type Key = RingPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut d = self
            .mesh_pipeline
            .specialize(key.mesh, &key.layout)
            .expect("unit meshes carry every attribute the mesh pipeline asks for");
        d.label = Some("selection ring pipeline".into());
        d.vertex.shader = self.shader.clone();
        d.vertex.entry_point = Some("vertex".into());
        d.vertex.buffers.clear();
        let fragment = d.fragment.as_mut().expect("the mesh pipeline has a fragment stage");
        fragment.shader = self.shader.clone();
        fragment.entry_point = Some("fragment".into());
        for target in fragment.targets.iter_mut().flatten() {
            target.blend = Some(BlendState::ALPHA_BLENDING);
        }
        // Tested against the ground and the soldiers, never written: a
        // soldier's feet stand over his ring.
        if let Some(depth) = d.depth_stencil.as_mut() {
            depth.depth_write_enabled = Some(false);
        }
        d.primitive.cull_mode = None;
        // Groups 0 and 1 stay the view's; group 2 is the rings'.
        d.layout.truncate(2);
        d.layout.push(self.layout.clone());
        d
    }
}

/// What the ring draw binds and draws this frame.
enum RingDraw {
    None,
    /// GPU path: the build pass's ring count, from the draw arguments.
    Indirect,
    /// CPU path: this many rings.
    Direct(u32),
}

#[derive(Resource)]
struct RingGpu {
    heights: Option<Buffer>,
    heights_generation: u32,
    params: UniformBuffer<RingParams>,
    cpu_records: Option<(Buffer, usize)>,
    cpu_entries: Option<(Buffer, usize)>,
    bind_group: Option<BindGroup>,
    draw: RingDraw,
    /// The render entity the ring phase item stands for.
    entity: Entity,
}

fn init_ring_gpu(mut commands: Commands) {
    let entity = commands.spawn_empty().id();
    commands.insert_resource(RingGpu {
        heights: None,
        heights_generation: 0,
        params: UniformBuffer::default(),
        cpu_records: None,
        cpu_entries: None,
        bind_group: None,
        draw: RingDraw::None,
        entity,
    });
}

/// Write `data` into the buffer in `slot`, growing it with slack first.
fn upload<T: bytemuck::Pod>(
    device: &RenderDevice,
    queue: &RenderQueue,
    slot: &mut Option<(Buffer, usize)>,
    label: &'static str,
    data: &[T],
) {
    let bytes = size_of_val(data).max(16);
    if slot.as_ref().is_none_or(|(_, cap)| *cap < bytes) {
        let cap = (bytes + bytes / 2).max(4096);
        *slot = Some((
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: cap as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            cap,
        ));
    }
    if !data.is_empty() {
        queue.write_buffer(&slot.as_ref().expect("made above").0, 0, bytemuck::cast_slice(data));
    }
}

#[allow(clippy::too_many_arguments)] // bevy system params
fn prepare_rings(
    input: Res<RingInput>,
    mut rings: ResMut<RingGpu>,
    gpu_units: Res<GpuUnitBuffers>,
    pipeline: Res<RingPipeline>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let rings = &mut *rings;
    rings.draw = RingDraw::None;
    rings.bind_group = None;
    if !input.active || input.heights.is_empty() {
        return;
    }
    if rings.heights.is_none() || rings.heights_generation != input.heights_generation {
        rings.heights = Some(device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("selection ring heights"),
            contents: bytemuck::cast_slice(&input.heights),
            usage: BufferUsages::STORAGE,
        }));
        rings.heights_generation = input.heights_generation;
    }
    let mut params = input.params;
    let (records, entries, draw) = if input.gpu {
        let Some(alloc) = &gpu_units.alloc else { return };
        params.grid.w = alloc.ring_base;
        (alloc.records.as_entire_binding(), alloc.index_list.as_entire_binding(), RingDraw::Indirect)
    } else {
        let n = input.cpu.entries.len() as u32;
        if n == 0 {
            return;
        }
        upload(&device, &queue, &mut rings.cpu_records, "selection ring records", &input.cpu.records);
        upload(&device, &queue, &mut rings.cpu_entries, "selection ring entries", &input.cpu.entries);
        let (Some((records, _)), Some((entries, _))) = (&rings.cpu_records, &rings.cpu_entries)
        else {
            return;
        };
        (records.as_entire_binding(), entries.as_entire_binding(), RingDraw::Direct(n))
    };
    rings.params.set(params);
    rings.params.write_buffer(&device, &queue);
    let (Some(heights), Some(uniform)) = (&rings.heights, rings.params.binding()) else {
        return;
    };
    // Remade every frame the rings show: the unit buffers it binds can be
    // reallocated, and one bind group is cheap.
    rings.bind_group = Some(device.create_bind_group(
        "selection ring bind group",
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((records, entries, heights.as_entire_binding(), uniform)),
    ));
    rings.draw = draw;
}

/// Add the ring draw to the camera's transparent phase for this frame.
#[allow(clippy::too_many_arguments)] // bevy system params
fn queue_rings(
    input: Res<RingInput>,
    rings: Res<RingGpu>,
    draw_functions: Res<DrawFunctions<Transparent3d>>,
    pipeline: Res<RingPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<RingPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    buckets: Query<&MainEntity, With<ExtractedBucket>>,
    views: Query<&ExtractedView>,
    view_key_cache: Res<ViewKeyCache>,
    mut phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
) {
    if !input.active {
        return;
    }
    // Any unit mesh's layout: the ring draws no vertex buffers, but the
    // mesh pipeline specializes from one.
    let Some(mesh) = buckets.iter().find_map(|main_entity| {
        let instance = render_mesh_instances.render_mesh_queue_data(*main_entity)?;
        meshes.get(instance.mesh_asset_id())
    }) else {
        return;
    };
    let draw_rings = draw_functions.read().id::<DrawRings>();
    for view in &views {
        let (Some(phase), Some(&view_key)) = (
            phases.get_mut(&view.retained_view_entity),
            view_key_cache.get(&view.retained_view_entity),
        ) else {
            continue;
        };
        let key = view_key
            | MeshPipelineKey::from_primitive_topology_and_strip_index(
                PrimitiveTopology::TriangleList,
                None,
            );
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            RingPipelineKey { mesh: key, layout: mesh.layout.clone() },
        );
        phase.add_transient(Transparent3d {
            sorting_info: TransparentSortingInfo3d::Sorted {
                mesh_center: Vec3::ZERO,
                depth_bias: 0.0,
            },
            entity: (rings.entity, MainEntity::from(Entity::PLACEHOLDER)),
            pipeline: pipeline_id,
            draw_function: draw_rings,
            distance: 0.0,
            batch_range: 0..1,
            extra_index: PhaseItemExtraIndex::None,
            indexed: false,
        });
    }
}

type DrawRings = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    DrawRingList,
);

struct DrawRingList;

impl<P: PhaseItem> RenderCommand<P> for DrawRingList {
    type Param = (SRes<RingGpu>, SRes<GpuUnitBuffers>);
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        _entity: Option<()>,
        (rings, gpu): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let rings = rings.into_inner();
        let Some(bind_group) = &rings.bind_group else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(2, bind_group, &[]);
        match rings.draw {
            RingDraw::None => return RenderCommandResult::Skip,
            RingDraw::Indirect => {
                let Some(alloc) = &gpu.into_inner().alloc else {
                    return RenderCommandResult::Skip;
                };
                pass.draw_indirect(&alloc.args, RING_ARG as u64 * 16);
            }
            RingDraw::Direct(n) => pass.draw(0..n * 6, 0..1),
        }
        RenderCommandResult::Success
    }
}

pub struct SelectionRingsPlugin;

impl Plugin for SelectionRingsPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/unit_rings.wgsl");
        app.init_resource::<CpuRings>()
            .init_resource::<RingsActive>()
            .init_resource::<RingTerrain>()
            .add_systems(PostUpdate, (update_rings_active, update_ring_terrain).chain());
        app.sub_app_mut(RenderApp)
            .init_resource::<RingInput>()
            .init_resource::<SpecializedRenderPipelines<RingPipeline>>()
            .add_render_command::<Transparent3d, DrawRings>()
            .add_systems(RenderStartup, (init_ring_pipeline, init_ring_gpu))
            .add_systems(ExtractSchedule, extract_rings)
            .add_systems(
                Render,
                (
                    queue_rings.in_set(RenderSystems::QueueMeshes),
                    prepare_rings.in_set(RenderSystems::PrepareBindGroups),
                ),
            );
    }
}

const _: () = assert!(NUM_KINDS == 4, "RingParams packs per-kind half heights in a vec4");
