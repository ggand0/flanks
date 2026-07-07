mod camera;
mod movement;
mod orders;
mod overlay;
mod render_units;
mod spatial;
mod terrain;
mod units;

use bevy::prelude::*;
use bevy::window::PresentMode;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "frontline".into(),
                // Uncapped so the FPS overlay shows real headroom.
                present_mode: PresentMode::AutoNoVsync,
                ..default()
            }),
            ..default()
        }))
        .add_plugins((
            terrain::TerrainPlugin,
            units::UnitsPlugin,
            movement::MovementPlugin,
            orders::OrdersPlugin,
            render_units::UnitRenderPlugin,
            camera::RtsCameraPlugin,
            overlay::OverlayPlugin,
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
