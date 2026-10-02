// A Windows GUI program, so opening the game does not also open an empty
// console window. Debug builds keep the console for their log.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ai;
mod arrows;
mod audio;
mod balance;
mod banners;
mod battle_setup;
mod camera;
mod combat;
mod fatigue;
mod formation;
mod gait;
mod frontline;
mod game_files;
mod game_state;
mod gpu_frame_timer;
mod mixer;
mod morale;
mod orders;
mod overlay;
mod picker;
mod regiments;
mod render_units;
mod render_units_gpu;
mod render_units_phase;
mod render_units_shadow;
mod selection;
mod selection_rings;
mod settings;
mod sim;
mod spatial;
mod terrain;
mod unit_cards;
mod unit_glb;
mod unit_meshes;
mod unit_types;
mod units;
mod util;
mod vegetation;
mod water;
mod window_icon;

use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::settings::{InstanceFlags, WgpuSettings};

/// Bevy's defaults minus wgpu's GPU-side validation of indirect draws.
/// Bevy strips that flag only when DX12 is not among the enabled
/// backends, and the default backend set names every backend, so on
/// Linux the check stays on. It injects a validation compute pass with a
/// fresh staging buffer into every render pass that draws indirectly,
/// and on the dev box that pinned the frame to the display refresh:
/// 60 fps with 20k soldiers on the GPU path, 300 fps without the check.
/// The unit draw arguments come from our own compute shader, from
/// counters bounded by the buffer layout. `WGPU_VALIDATION_INDIRECT_CALL=1`
/// turns the check back on for debugging.
fn wgpu_settings() -> WgpuSettings {
    let mut settings = WgpuSettings::default();
    settings.instance_flags.remove(InstanceFlags::VALIDATION_INDIRECT_CALL);
    settings.instance_flags = settings.instance_flags.with_env();
    settings
}

/// Bevy's task pools. The sim's parallel passes run on the async compute
/// pool (util.rs `sim_scope`), which takes half the hardware threads, the
/// file loading pool takes up to two and the compute pool, where every
/// system of the frame runs, the rest: 12, 2 and 10 of 24. Together they
/// hold as many threads as the hardware has. `FL_SIM_ASYNC=0` keeps Bevy's
/// default sizes (16 compute, 4 async compute, 4 file loading of 24).
/// `FL_THREADS` sets the total either way.
fn task_pool_options() -> bevy::app::TaskPoolOptions {
    let mut options = bevy::app::TaskPoolOptions::default();
    let threads = crate::util::env_or("FL_THREADS", 0_usize);
    if threads > 0 {
        options.min_total_threads = threads;
        options.max_total_threads = threads;
    }
    if crate::util::sim_on_async_pool() {
        options.io.max_threads = 2;
        options.io.percent = 0.1;
        options.async_compute.max_threads = usize::MAX;
        options.async_compute.percent = 0.5;
    }
    options
}

/// `FL_WINDOW=2560x1360` opens the window with that client size in
/// physical pixels whatever the display's scale, so runs on differently
/// scaled displays draw the same frame. The interface then draws at scale 1.
/// Without it Bevy opens 1280x720 logical, times the display's scale.
fn window_resolution() -> bevy::window::WindowResolution {
    let size = std::env::var("FL_WINDOW").ok().and_then(|v| {
        let (w, h) = v.split_once('x')?;
        Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
    });
    match size {
        Some((w, h)) => bevy::window::WindowResolution::new(w, h).with_scale_factor_override(1.0),
        None => default(),
    }
}

fn main() {
    // Load before the App so the window opens with the saved video
    // settings instead of switching modes one frame in.
    let user_settings = settings::Settings::load();
    App::new()
        .add_plugins(game_files::GameFilesPlugin)
        .add_plugins(
            DefaultPlugins
                .set(TaskPoolPlugin {
                    task_pool_options: task_pool_options(),
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "flanks".into(),
                        // The app name (X11 WM_CLASS, Wayland app id) a
                        // desktop entry matches with StartupWMClass.
                        name: Some("flanks".into()),
                        // Default vsync off: the FPS overlay should
                        // show real headroom.
                        present_mode: settings::present_mode(&user_settings),
                        mode: settings::window_mode(&user_settings),
                        resolution: window_resolution(),
                        ..default()
                    }),
                    ..default()
                })
                // Resolve assets/ regardless of how the binary is launched
                // (packaged, cargo run or ./target/...). An embed_assets
                // build serves them from memory instead (GameFilesPlugin).
                .set(AssetPlugin {
                    file_path: util::game_root().join("assets").to_string_lossy().into_owned(),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: wgpu_settings().into(),
                    ..default()
                }),
        )
        .insert_resource(user_settings)
        .add_plugins(game_state::GameShellPlugin)
        .add_plugins(settings::SettingsPlugin)
        .add_plugins(battle_setup::BattleSetupPlugin)
        .add_plugins(window_icon::WindowIconPlugin)
        .add_plugins((
            terrain::TerrainPlugin,
            water::WaterPlugin,
            vegetation::VegetationPlugin,
            units::UnitsPlugin,
            regiments::RegimentsPlugin,
            morale::MoralePlugin,
            fatigue::FatiguePlugin,
            ai::AiPlugin,
            banners::BannersPlugin,
            audio::BattleAudioPlugin,
            sim::SimPlugin,
            arrows::ArrowsPlugin,
        ))
        .add_plugins((
            orders::OrdersPlugin,
            selection::SelectionPlugin,
            formation::FormationPlugin,
            frontline::FrontlinePlugin,
            combat::CombatPlugin,
            render_units::UnitRenderPlugin,
            selection_rings::SelectionRingsPlugin,
            camera::RtsCameraPlugin,
            overlay::OverlayPlugin,
            gpu_frame_timer::GpuFrameTimerPlugin,
            unit_cards::UnitCardsPlugin,
            balance::BalancePlugin,
            picker::PickerPlugin,
        ))
        .insert_resource(Time::<Fixed>::from_hz(30.0))
        // A frame that overruns makes the fixed clock owe catch-up sim
        // ticks, which Bevy runs back to back the next frame (default
        // max_delta 250 ms = up to 7 ticks at 30 Hz). Every tick after
        // the first in a frame has to wait for its job in full, so the
        // burst overruns the frame it lands in and one slow frame
        // snowballs into a felt lag spike. Clamping virtual time to
        // FL_CATCHUP tick periods (default 2) caps the burst. Time past
        // the clamp is dropped: a stall plays as a moment of slow motion.
        .insert_resource(Time::<Virtual>::from_max_delta(
            std::time::Duration::from_secs_f64(
                crate::util::env_or("FL_CATCHUP", 2_u32).max(1) as f64 / 30.0,
            ),
        ))
        .insert_resource(ClearColor(Color::srgb(0.62, 0.70, 0.78)))
        // Bevy's default drops an unfocused window to 60 updates a
        // second. That silently caps every fps and frame-time reading
        // the moment the desktop gets a click, so the loop runs
        // continuously either way: a measurement reads frames generated,
        // and a battle keeps going behind another window.
        .insert_resource(bevy::winit::WinitSettings {
            focused_mode: bevy::winit::UpdateMode::Continuous,
            unfocused_mode: bevy::winit::UpdateMode::Continuous,
        })
        .add_systems(Startup, (setup_world, log_task_pools))
        .run();
}

fn log_task_pools() {
    use bevy::tasks::{AsyncComputeTaskPool, ComputeTaskPool, IoTaskPool};
    info!(
        "task pools: compute {}, async compute {} (the sim{}), file loading {}",
        ComputeTaskPool::get().thread_num(),
        AsyncComputeTaskPool::get().thread_num(),
        if crate::util::sim_on_async_pool() { "" } else { " runs on the compute pool" },
        IoTaskPool::get().thread_num(),
    );
}

/// Sun; terrain chunks come from TerrainPlugin.
fn setup_world(mut commands: Commands, settings: Res<settings::Settings>) {
    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            // The Shadows setting, or FL_SHADOWS=0, turns every sun
            // shadow off, units included: without shadow maps the light
            // has no cascade views for the unit draw to cast into, and
            // the unit shader skips the lookup on the light's flag.
            shadow_maps_enabled: settings::shadows_on(&settings),
            ..default()
        },
        // Lowish sun: flat-shaded relief needs directional contrast.
        Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, 0.7, -0.75, 0.0)),
        // Three cascades out to 280 m, split at 40 and 106 m. The first
        // covers the close-up, where a soldier's shadow at his feet needs
        // the finest texels. Soldiers stop casting at about 100 m, where
        // the camera draws them at L2, so the third holds trees and
        // terrain: it gives trees their ground shadow and shaded canopy
        // to mid distance, where without it they read pale. Every cascade
        // is one more view Bevy walks every mesh for each frame, plus a
        // pass, about 0.2 ms of render thread whatever it holds.
        // FL_SHADOW_CASCADES=2 FL_SHADOW_DIST=110 drops the third.
        bevy::light::CascadeShadowConfigBuilder {
            num_cascades: crate::util::env_or("FL_SHADOW_CASCADES", 3_usize).clamp(1, 4),
            first_cascade_far_bound: 40.0,
            maximum_distance: crate::util::env_or("FL_SHADOW_DIST", 280.0_f32).max(41.0),
            ..default()
        }
        .build(),
    ));
}
