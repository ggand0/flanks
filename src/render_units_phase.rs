//! The soldiers' own render phase, drawn before the terrain.
//!
//! Soldiers are opaque. Drawn after the ground, as Bevy's transparent phase
//! would, every ground pixel behind them is shaded first and then covered:
//! in a view along a battle line most of the screen is soldiers. Drawn
//! first, they fill the depth buffer and the ground's fragments behind them
//! fail the depth test before they are shaded. The phase runs in its own
//! pass ahead of Bevy's opaque pass, into the same colour and depth targets,
//! and whichever pass comes first clears them.
//!
//! Within the phase the near levels draw first, so near soldiers also hide
//! the far ones behind them. `FL_UNITS_FIRST=0` queues the soldiers into
//! the transparent phase instead, for A/B runs.

use std::ops::Range;

use bevy::{
    camera::{MainPassResolutionOverride, Viewport},
    core_pipeline::{Core3d, Core3dSystems, core_3d::main_opaque_pass_3d},
    ecs::entity::EntityHash,
    pbr::MeshPipeline,
    platform::collections::HashSet,
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderDebugFlags, RenderSystems,
        camera::ExtractedCamera,
        diagnostic::RecordDiagnostics,
        render_phase::{
            CachedRenderPipelinePhaseItem, DrawFunctionId, DrawFunctions, PhaseItem,
            PhaseItemExtraIndex, SortedPhaseItem, SortedRenderPhasePlugin, ViewSortedRenderPhases,
            sort_phase_system,
        },
        render_resource::{CachedRenderPipelineId, RenderPassDescriptor, StoreOp},
        renderer::{RenderContext, ViewQuery},
        sync_world::MainEntity,
        view::{ExtractedView, RetainedViewEntity, ViewDepthTexture, ViewTarget},
    },
};
use indexmap::IndexMap;

/// The soldiers draw in `Units3d` rather than Bevy's transparent phase.
/// Read once: `FL_UNITS_FIRST=0` turns it off.
pub(crate) fn units_first() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::util::env_or("FL_UNITS_FIRST", 1_u32) != 0)
}

/// One unit bucket's draw (render_units.rs `DrawCustom`).
pub(crate) struct Units3d {
    pub pipeline: CachedRenderPipelineId,
    pub entity: (Entity, MainEntity),
    pub draw_function: DrawFunctionId,
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
    pub indexed: bool,
    /// Draw order, lowest first: level-major, so the nearest soldiers
    /// fill the depth buffer before the farther ones draw.
    pub order: u32,
}

impl PhaseItem for Units3d {
    #[inline]
    fn entity(&self) -> Entity {
        self.entity.0
    }

    fn main_entity(&self) -> MainEntity {
        self.entity.1
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index.clone()
    }

    #[inline]
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl SortedPhaseItem for Units3d {
    type SortKey = u32;

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        self.order
    }

    /// The order is fixed per bucket.
    fn recalculate_sort_keys(
        _items: &mut IndexMap<(Entity, MainEntity), Self, EntityHash>,
        _view: &ExtractedView,
    ) {
    }

    #[inline]
    fn indexed(&self) -> bool {
        self.indexed
    }
}

impl CachedRenderPipelinePhaseItem for Units3d {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

/// A phase for every active 3D camera, as Bevy keeps its own main phases.
fn extract_unit_phases(
    mut phases: ResMut<ViewSortedRenderPhases<Units3d>>,
    cameras: Extract<Query<(Entity, &Camera), With<Camera3d>>>,
    mut live: Local<HashSet<RetainedViewEntity>>,
) {
    live.clear();
    for (main_entity, camera) in &cameras {
        if !camera.is_active {
            continue;
        }
        let view = RetainedViewEntity::new(main_entity.into(), None, 0);
        phases.prepare_for_new_frame(view);
        live.insert(view);
    }
    phases.retain(|view, _| live.contains(view));
}

/// Draw the soldiers into the view's colour and depth targets, before
/// Bevy's opaque pass. Timed as `render/main_unit_pass/elapsed_gpu`.
fn main_unit_pass(
    world: &World,
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewTarget,
        &ViewDepthTexture,
        Option<&MainPassResolutionOverride>,
    )>,
    phases: Res<ViewSortedRenderPhases<Units3d>>,
    mut ctx: RenderContext,
) {
    let view_entity = view.entity();
    let (camera, extracted_view, target, depth, resolution_override) = view.into_inner();
    let Some(phase) = phases.get(&extracted_view.retained_view_entity) else {
        return;
    };
    if phase.items.is_empty() {
        return;
    }
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("main_unit_pass"),
        color_attachments: &[Some(target.get_color_attachment())],
        depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let span = diagnostics.pass_span(&mut pass, "main_unit_pass");
    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
    {
        pass.set_camera_viewport(&viewport);
    }
    if let Err(err) = phase.render(&mut pass, world, view_entity) {
        error!("the unit phase failed to render: {err:?}");
    }
    span.end(&mut pass);
}

pub struct UnitPhasePlugin;

impl Plugin for UnitPhasePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(SortedRenderPhasePlugin::<Units3d, MeshPipeline>::new(
            RenderDebugFlags::default(),
        ));
        app.sub_app_mut(RenderApp)
            .init_resource::<DrawFunctions<Units3d>>()
            .add_systems(ExtractSchedule, extract_unit_phases)
            .add_systems(Render, sort_phase_system::<Units3d>.in_set(RenderSystems::PhaseSort))
            .add_systems(
                Core3d,
                main_unit_pass
                    .in_set(Core3dSystems::MainPass)
                    .before(main_opaque_pass_3d),
            );
    }
}
