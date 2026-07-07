mod camera;
mod movement;
mod overlay;
mod render_units;
mod spatial;
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
            units::UnitsPlugin,
            movement::MovementPlugin,
            render_units::UnitRenderPlugin,
            camera::RtsCameraPlugin,
            overlay::OverlayPlugin,
        ))
        .insert_resource(Time::<Fixed>::from_hz(30.0))
        .insert_resource(ClearColor(Color::srgb(0.62, 0.70, 0.78)))
        .add_systems(Startup, setup_world)
        .run();
}

/// Ground plane and sun. The plane is a placeholder until M3 terrain.
fn setup_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(3000.0, 3000.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.30, 0.38, 0.26),
            perceptual_roughness: 1.0,
            ..default()
        })),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, 0.6, -1.1, 0.0)),
    ));
}
