//! RTS camera: WASD pan, scroll zoom, middle-drag rotate.

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit};
use bevy::prelude::*;
use bevy::render::view::NoIndirectDrawing;

#[derive(Component)]
pub struct RtsCamera {
    pub focus: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

pub struct RtsCameraPlugin;

impl Plugin for RtsCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera)
            .add_systems(Update, (control_camera, apply_camera_transform).chain());
    }
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        // Required: the unit renderer issues direct (non-indirect) draws.
        NoIndirectDrawing,
        RtsCamera {
            focus: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.9,
            // Overridable for screenshot-based debugging without input injection.
            distance: std::env::var("FL_CAM_DIST")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(280.0),
        },
    ));
}

fn control_camera(
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    time: Res<Time>,
    mut query: Query<&mut RtsCamera>,
) {
    let Ok(mut cam) = query.single_mut() else {
        return;
    };

    // Pan in the camera's yaw frame, speed scales with zoom.
    let mut pan = Vec2::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        pan.y -= 1.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        pan.y += 1.0;
    }
    if keys.pressed(KeyCode::KeyA) {
        pan.x -= 1.0;
    }
    if keys.pressed(KeyCode::KeyD) {
        pan.x += 1.0;
    }
    if pan != Vec2::ZERO {
        let speed = cam.distance * 0.9 * time.delta_secs();
        let (sin_yaw, cos_yaw) = cam.yaw.sin_cos();
        let forward = Vec3::new(-sin_yaw, 0.0, -cos_yaw);
        let right = Vec3::new(cos_yaw, 0.0, -sin_yaw);
        let delta = (right * pan.x + forward * -pan.y) * speed;
        cam.focus += delta;
    }

    let scroll_lines = match scroll.unit {
        MouseScrollUnit::Line => scroll.delta.y,
        // Touchpads/some drivers report pixels; ~50 px per notch.
        MouseScrollUnit::Pixel => scroll.delta.y / 50.0,
    };
    if scroll_lines != 0.0 {
        cam.distance = (cam.distance * 0.9f32.powf(scroll_lines)).clamp(15.0, 900.0);
    }

    if buttons.pressed(MouseButton::Middle) && motion.delta != Vec2::ZERO {
        cam.yaw -= motion.delta.x * 0.005;
        cam.pitch = (cam.pitch + motion.delta.y * 0.005).clamp(0.25, 1.45);
    }
}

pub fn apply_camera_transform(
    mut query: Query<(&mut RtsCamera, &mut Transform)>,
    terrain: Res<crate::terrain::Terrain>,
) {
    let Ok((mut cam, mut transform)) = query.single_mut() else {
        return;
    };
    let min = terrain.min();
    let max = terrain.max();
    cam.focus.x = cam.focus.x.clamp(min.x, max.x);
    cam.focus.z = cam.focus.z.clamp(min.y, max.y);
    cam.focus.y = terrain.height_at(cam.focus.x, cam.focus.z);
    let rot = Quat::from_euler(EulerRot::YXZ, cam.yaw, -cam.pitch, 0.0);
    let offset = rot * Vec3::new(0.0, 0.0, cam.distance);
    *transform = Transform::from_translation(cam.focus + offset).looking_at(cam.focus, Vec3::Y);
}
