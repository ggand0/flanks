mod ai;
mod arrows;
mod audio;
mod banners;
mod camera;
mod combat;
mod fatigue;
mod formation;
mod frontline;
mod game_state;
mod morale;
mod movement;
mod orders;
mod overlay;
mod picker;
mod regiments;
mod render_units;
mod selection;
mod settings;
mod spatial;
mod terrain;
mod unit_cards;
mod unit_meshes;
mod unit_types;
mod units;
mod util;
mod vegetation;
mod water;

use bevy::prelude::*;

fn main() {
    // Load before the App so the window opens with the saved video
    // settings instead of switching modes one frame in.
    let user_settings = settings::Settings::load();
    // FL_THREADS caps the compute task pool (default: all cores). The
    // parallel sim scopes' wall time is gated by their slowest chunk, so
    // on a loaded box a full-width pool oversubscribes and any stolen
    // core spikes the whole tick — leaving headroom for the render
    // thread and whatever else runs trades a little throughput for
    // fewer hitches. Sim-correctness is unaffected (pure data-parallel).
    let threads = crate::util::env_or("FL_THREADS", 0_usize);
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(if threads > 0 {
                    TaskPoolPlugin {
                        task_pool_options: bevy::app::TaskPoolOptions::with_num_threads(threads),
                    }
                } else {
                    TaskPoolPlugin::default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "flanks".into(),
                        // Default vsync off: the FPS overlay should
                        // show real headroom.
                        present_mode: settings::present_mode(&user_settings),
                        mode: settings::window_mode(&user_settings),
                        ..default()
                    }),
                    ..default()
                })
                // Resolve assets/ from the repo regardless of how the
                // binary is launched (cargo run vs ./target/...).
                .set(AssetPlugin {
                    file_path: concat!(env!("CARGO_MANIFEST_DIR"), "/assets").into(),
                    ..default()
                }),
        )
        .insert_resource(user_settings)
        .add_plugins(game_state::GameShellPlugin)
        .add_plugins(settings::SettingsPlugin)
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
            movement::MovementPlugin,
            arrows::ArrowsPlugin,
        ))
        .add_plugins((
            orders::OrdersPlugin,
            selection::SelectionPlugin,
            formation::FormationPlugin,
            frontline::FrontlinePlugin,
            combat::CombatPlugin,
            render_units::UnitRenderPlugin,
            camera::RtsCameraPlugin,
            overlay::OverlayPlugin,
            unit_cards::UnitCardsPlugin,
            picker::PickerPlugin,
        ))
        .insert_resource(Time::<Fixed>::from_hz(30.0))
        .insert_resource(ClearColor(Color::srgb(0.62, 0.70, 0.78)))
        .add_systems(Startup, setup_world)
        .run();
}

/// Sun; terrain chunks come from TerrainPlugin.
fn setup_world(mut commands: Commands) {
    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        // Lowish sun: flat-shaded relief needs directional contrast.
        Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, 0.7, -0.75, 0.0)),
    ));
}
