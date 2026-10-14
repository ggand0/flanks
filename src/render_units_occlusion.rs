//! Occlusion culling of soldiers.
//!
//! After Bevy's opaque pass, which draws the soldiers, the terrain and the
//! trees, a compute pass (`shaders/unit_depth_pyramid.wgsl`) takes the
//! view's depth buffer down to a pyramid of farthest depths that persists
//! into the next frame. That frame's build pass (unit_build.wgsl
//! `occluded`) projects each soldier's box with the matrix the depth was
//! drawn with and reads the pyramid over it: a soldier whose nearest point
//! is behind everything drawn over him last frame is not drawn, as a
//! soldier out of the frustum is not. His shadow and his ring stay.
//!
//! The test lags a frame: a soldier who comes out from behind a nearer man
//! is tested against where that man stood a frame ago. His box is wider
//! than he is, so the nearer man must move by that margin before he shows.
//! A camera cut (render_units_gpu.rs `occlusion_params`), a resize or a
//! frame without a pyramid skips the test.
//!
//! While the test hides little, as in a view from above where the mass
//! never hides itself, the pyramid is built only every few frames as a
//! probe (render_units_gpu.rs `OcclusionProbe`).
//!
//! `FL_OCCLUDE`: 0 off, 1 cull (the default), 2 count (the test runs and
//! counts, nothing is culled). `FL_GPU_CHECK` turns cull into count, since
//! the CPU sweep it compares against has no test. `FL_OCCLUDE_CHECK=1`
//! measures how far each drawn corner reaches from his soldier, against the
//! boxes the test uses (`BOUNDS`), and in count mode draws the soldiers the
//! test hid flat magenta, so any magenta on screen is a hidden soldier who
//! shows (unit_instancing.wgsl). `=2` marks every soldier the test reads.

use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::core_pipeline::core_3d::{main_opaque_pass_3d, main_transparent_pass_3d};
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::prelude::*;
use bevy::render::{
    Render, RenderApp, RenderStartup, RenderSystems,
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    render_resource::*,
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::{ExtractedView, ViewDepthTexture},
};

use crate::unit_types::NUM_KINDS;

/// What the test does. Read once from `FL_OCCLUDE`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OccludeMode {
    Off = 0,
    /// Hidden soldiers are not drawn.
    Cull = 1,
    /// The test runs and counts; every soldier draws as with it off.
    Count = 2,
}

pub fn mode() -> OccludeMode {
    static MODE: std::sync::OnceLock<OccludeMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match crate::util::env_or("FL_OCCLUDE", 1_u32) {
            1 => OccludeMode::Cull,
            2 => OccludeMode::Count,
            _ => OccludeMode::Off,
        };
        if mode == OccludeMode::Cull && std::env::var("FL_GPU_CHECK").is_ok() {
            return OccludeMode::Count;
        }
        mode
    })
}

/// `FL_OCCLUDE_CHECK=1`: the unit draws measure each corner's reach, and
/// in count mode draw the soldiers the test hid flat magenta. `=2` marks
/// every soldier the test reads, hidden or not, to show the mark works.
pub fn check_level() -> u32 {
    static LEVEL: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| crate::util::env_or("FL_OCCLUDE_CHECK", 0_u32).min(2))
}

pub fn bound_check() -> bool {
    check_level() > 0
}

/// The box the test projects, per kind, in the soldier's facing frame
/// from his record position (half his height above his feet): forward,
/// back, sideways either way, up, down, in metres. One for the standing,
/// one for the dying and the fallen.
#[derive(Clone, Copy, Debug)]
pub struct Bound {
    pub fwd: f32,
    pub back: f32,
    pub side: f32,
    pub up: f32,
    pub down: f32,
}

/// The farthest any drawn corner reached under `FL_OCCLUDE_CHECK`, plus
/// 15 cm, rounded up to 5 cm. Measured 2026-10-13 over three battles in the
/// low flank view: 40k for four minutes with the clips and again with
/// `FL_CLIPS=0`, and 8k for five and a half minutes. The man-at-arms plays
/// the knight's clips and takes the knight's box. The spearman's spear
/// reaches 3 m ahead, and a fallen one's spear into the ground. The
/// standing reach down passes the feet: a crouch lowers the whole figure.
pub const BOUNDS: [[Bound; 2]; NUM_KINDS] = {
    const KNIGHT: [Bound; 2] = [
        Bound { fwd: 1.85, back: 2.0, side: 1.55, up: 1.85, down: 1.45 },
        Bound { fwd: 2.05, back: 1.85, side: 2.25, up: 1.75, down: 2.2 },
    ];
    [
        KNIGHT,
        KNIGHT,
        [
            Bound { fwd: 3.2, back: 1.6, side: 0.8, up: 2.25, down: 1.45 },
            Bound { fwd: 3.25, back: 2.8, side: 2.95, up: 2.1, down: 3.5 },
        ],
        [
            Bound { fwd: 1.65, back: 1.5, side: 0.8, up: 1.7, down: 1.45 },
            Bound { fwd: 1.85, back: 1.8, side: 1.9, up: 1.65, down: 1.8 },
        ],
    ]
};

/// The most levels the pyramid keeps: level k's texel covers 2^(k+1)
/// pixels, 4096 at the last.
const MAX_LEVELS: u32 = 12;

/// The pyramid of the view the soldiers draw in, render world side.
#[derive(Resource, Default)]
pub struct UnitDepthPyramid {
    texture: Option<Texture>,
    /// Every level, for the build pass.
    pub all: Option<TextureView>,
    /// One view per level, for the passes that write them.
    levels: Vec<TextureView>,
    /// The depth texture's size the pyramid was made for.
    size: UVec2,
    /// The view whose depth it takes, and the depth texture view the bind
    /// groups hold.
    view: Option<Entity>,
    depth_id: Option<TextureViewId>,
    /// `first`, then one `down` per later level.
    groups: Vec<BindGroup>,
    /// What this frame's pass draws the pyramid from: the camera's
    /// `clip_from_world` and its viewport in pixels. Set when the pass
    /// can run.
    this: Option<(Mat4, Vec4)>,
    /// The same for the pyramid the build pass reads this frame, last
    /// frame's.
    pub last: Option<(Mat4, Vec4)>,
    /// A 1x1 texture the build pass binds without a pyramid.
    pub dummy: Option<TextureView>,
}

impl UnitDepthPyramid {
    /// Levels the build pass can read this frame, 0 without a pyramid.
    pub fn levels(&self) -> u32 {
        self.levels.len() as u32
    }
}

#[derive(Resource)]
struct PyramidPipelines {
    first_layout: BindGroupLayoutDescriptor,
    first_ms_layout: BindGroupLayoutDescriptor,
    down_layout: BindGroupLayoutDescriptor,
    first: CachedComputePipelineId,
    first_ms: CachedComputePipelineId,
    down: CachedComputePipelineId,
}

fn init_pyramid_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    mut pyramid: ResMut<UnitDepthPyramid>,
) {
    let level = || binding_types::texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly);
    let first_layout = BindGroupLayoutDescriptor::new(
        "unit depth pyramid first layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            ((0, binding_types::texture_2d(TextureSampleType::Depth)), (1, level())),
        ),
    );
    let first_ms_layout = BindGroupLayoutDescriptor::new(
        "unit depth pyramid first multisampled layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            ((0, binding_types::texture_2d_multisampled(TextureSampleType::Depth)), (1, level())),
        ),
    );
    let down_layout = BindGroupLayoutDescriptor::new(
        "unit depth pyramid down layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (1, level()),
                (2, binding_types::texture_2d(TextureSampleType::Float { filterable: false })),
            ),
        ),
    );
    let shader = load_embedded_asset!(asset_server.as_ref(), "shaders/unit_depth_pyramid.wgsl");
    let pipeline = |label: &'static str, layout: &BindGroupLayoutDescriptor, entry: &'static str, ms: bool| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: shader.clone(),
            shader_defs: if ms { vec!["MULTISAMPLE".into()] } else { vec![] },
            entry_point: Some(entry.into()),
            zero_initialize_workgroup_memory: false,
        })
    };
    commands.insert_resource(PyramidPipelines {
        first: pipeline("unit depth pyramid first", &first_layout, "first", false),
        first_ms: pipeline("unit depth pyramid first multisampled", &first_ms_layout, "first", true),
        down: pipeline("unit depth pyramid down", &down_layout, "down", false),
        first_layout,
        first_ms_layout,
        down_layout,
    });
    let dummy = device.create_texture(&TextureDescriptor {
        label: Some("unit depth pyramid dummy"),
        size: Extent3d::default(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    pyramid.dummy = Some(dummy.create_view(&TextureViewDescriptor::default()));
}

/// Level sizes for a depth texture of `size`: half of it rounded up, then
/// the mip levels of that, each half the last rounded down, down to one
/// texel on the longer side or `MAX_LEVELS`.
fn level_sizes(size: UVec2) -> Vec<UVec2> {
    let base = (size + 1) / 2;
    let count = (32 - base.max_element().leading_zeros()).min(MAX_LEVELS);
    (0..count).map(|k| (base >> k).max(UVec2::ONE)).collect()
}

/// Before the build pass binds it: hand last frame's pyramid to this
/// frame's test, remake the pyramid when the view's size changed, and the
/// bind groups when its depth texture did.
fn prepare_unit_pyramid(
    mut pyramid: ResMut<UnitDepthPyramid>,
    pipelines: Res<PyramidPipelines>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedView, &ExtractedCamera, &ViewDepthTexture), With<Camera3d>>,
    input: Res<crate::render_units_gpu::GpuUnitInput>,
) {
    let pyramid = &mut *pyramid;
    pyramid.last = pyramid.this.take();
    // The CPU path has no test.
    if !input.enabled {
        pyramid.last = None;
        return;
    }
    let mut views = views.iter().filter(|(.., camera, _)| camera.order == 0);
    let Some((entity, view, _, depth)) = views.next() else {
        pyramid.last = None;
        return;
    };
    let size = UVec2::new(depth.texture.width(), depth.texture.height());
    if pyramid.texture.is_none() || pyramid.size != size || pyramid.view != Some(entity) {
        let sizes = level_sizes(size);
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("unit depth pyramid"),
            size: Extent3d { width: sizes[0].x, height: sizes[0].y, depth_or_array_layers: 1 },
            mip_level_count: sizes.len() as u32,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        pyramid.levels = (0..sizes.len() as u32)
            .map(|level| {
                texture.create_view(&TextureViewDescriptor {
                    label: Some("unit depth pyramid level"),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..default()
                })
            })
            .collect();
        pyramid.all = Some(texture.create_view(&TextureViewDescriptor::default()));
        pyramid.texture = Some(texture);
        pyramid.size = size;
        pyramid.view = Some(entity);
        pyramid.depth_id = None;
        // The texture is new: nothing was drawn into it.
        pyramid.last = None;
        info!(
            "unit depth pyramid: {} levels over {}x{}",
            sizes.len(),
            size.x,
            size.y
        );
    }
    let multisampled = depth.texture.sample_count() > 1;
    if pyramid.depth_id != Some(depth.view().id()) {
        let first_layout = if multisampled { &pipelines.first_ms_layout } else { &pipelines.first_layout };
        let mut groups = vec![device.create_bind_group(
            "unit depth pyramid first",
            &pipeline_cache.get_bind_group_layout(first_layout),
            &BindGroupEntries::with_indices(((0, depth.view()), (1, &pyramid.levels[0]))),
        )];
        let down = pipeline_cache.get_bind_group_layout(&pipelines.down_layout);
        for level in 1..pyramid.levels.len() {
            groups.push(device.create_bind_group(
                "unit depth pyramid down",
                &down,
                &BindGroupEntries::with_indices((
                    (1, &pyramid.levels[level]),
                    (2, &pyramid.levels[level - 1]),
                )),
            ));
        }
        pyramid.groups = groups;
        pyramid.depth_id = Some(depth.view().id());
    }
    let first = if multisampled { pipelines.first_ms } else { pipelines.first };
    // Built only when the main world asks (render_units_gpu.rs
    // `OcclusionProbe`): the next frame tests against it.
    let ready = input.build_pyramid
        && pipeline_cache.get_compute_pipeline(first).is_some()
        && pipeline_cache.get_compute_pipeline(pipelines.down).is_some();
    if ready {
        let clip_from_world = view
            .clip_from_world
            .unwrap_or_else(|| view.clip_from_view * view.world_from_view.to_matrix().inverse());
        pyramid.this = Some((clip_from_world, view.viewport.as_vec4()));
    }
}

/// The pyramid from this frame's depth, for the next frame's test. Timed
/// as `render/unit_depth_pyramid/elapsed_gpu`.
fn run_unit_pyramid_pass(
    view: ViewQuery<&ViewDepthTexture>,
    pyramid: Res<UnitDepthPyramid>,
    pipelines: Res<PyramidPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if pyramid.view != Some(view.entity()) || pyramid.this.is_none() {
        return;
    }
    let first = match view.into_inner().texture.sample_count() > 1 {
        true => pipelines.first_ms,
        false => pipelines.first,
    };
    let (Some(first), Some(down)) = (
        pipeline_cache.get_compute_pipeline(first),
        pipeline_cache.get_compute_pipeline(pipelines.down),
    ) else {
        return;
    };
    let sizes = level_sizes(pyramid.size);
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("unit depth pyramid"),
        timestamp_writes: None,
    });
    let span = diagnostics.pass_span(&mut pass, "unit_depth_pyramid");
    for (level, (group, size)) in pyramid.groups.iter().zip(&sizes).enumerate() {
        pass.set_pipeline(if level == 0 { first } else { down });
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(size.x.div_ceil(8), size.y.div_ceil(8), 1);
    }
    span.end(&mut pass);
}

pub struct UnitOcclusionPlugin;

impl Plugin for UnitOcclusionPlugin {
    fn build(&self, app: &mut App) {
        // The build pass binds the dummy with the test off.
        embedded_asset!(app, "shaders/unit_depth_pyramid.wgsl");
        app.sub_app_mut(RenderApp)
            .init_resource::<UnitDepthPyramid>()
            .add_systems(RenderStartup, init_pyramid_pipelines);
        if mode() == OccludeMode::Off {
            return;
        }
        info!("unit occlusion test on: {:?} (FL_OCCLUDE)", mode());
        app.sub_app_mut(RenderApp)
            .add_systems(
                Render,
                prepare_unit_pyramid
                    .in_set(RenderSystems::PrepareResources)
                    .after(bevy::core_pipeline::core_3d::prepare_core_3d_depth_textures)
                    .before(crate::render_units_gpu::prepare_gpu_units),
            )
            .add_systems(
                Core3d,
                run_unit_pyramid_pass
                    .in_set(Core3dSystems::MainPass)
                    .after(main_opaque_pass_3d)
                    .before(main_transparent_pass_3d),
            );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Level k's texel x covers pixels [x, x + 1) * 2^(k + 1), and the last
    /// texel of a level everything to the edge (unit_depth_pyramid.wgsl):
    /// every pixel lands on a texel of every level, clamped to the last.
    #[test]
    fn levels_cover_every_pixel() {
        for size in [UVec2::new(2494, 1363), UVec2::new(2560, 1360), UVec2::new(1, 1), UVec2::new(3, 1000)] {
            let sizes = level_sizes(size);
            assert!(!sizes.is_empty());
            assert_eq!(sizes[0], (size + 1) / 2);
            for (k, s) in sizes.iter().enumerate() {
                // As wgpu sizes mip k of level 0.
                assert_eq!(*s, (sizes[0] >> k as u32).max(UVec2::ONE));
                for axis in 0..2 {
                    // Pixels each texel reads through the levels below it,
                    // the last texel taking the odd rows and columns.
                    let mut reach: Vec<(u32, u32)> =
                        (0..sizes[0][axis]).map(|x| (2 * x, (2 * x + 1).min(size[axis] - 1))).collect();
                    for next in &sizes[1..=k] {
                        let n = next[axis];
                        let src = std::mem::take(&mut reach);
                        reach = (0..n)
                            .map(|x| {
                                // Two source texels, three for the last.
                                let last = if x == n - 1 { 2 * x + 2 } else { 2 * x + 1 };
                                (src[2 * x as usize].0, src[last.min(src.len() as u32 - 1) as usize].1)
                            })
                            .collect();
                    }
                    for p in 0..size[axis] {
                        let t = (p >> (k + 1)).min(s[axis] - 1);
                        let (lo, hi) = reach[t as usize];
                        assert!(lo <= p && p <= hi, "size {size} level {k} axis {axis} pixel {p} on texel {t} reads {lo}..={hi}");
                    }
                }
            }
        }
    }
}
