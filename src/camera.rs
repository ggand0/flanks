//! RTS camera: WASD + screen-edge pan, smoothed scroll zoom,
//! middle-drag rotate. Keyboard pan takes priority over edge pan (a
//! cursor parked at the bottom edge must not cancel W). Shift pans
//! faster and left Alt slower.

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit};
use bevy::prelude::*;
use bevy::render::view::NoIndirectDrawing;
use bevy::window::PrimaryWindow;

/// Cursor within this many pixels of a window edge pans the camera.
const EDGE_PAN_PX: f32 = 12.0;
/// Pan speed factor while Shift is held.
const PAN_FAST: f32 = 3.0;
/// Pan speed factor while left Alt is held, for slow recording moves.
/// Holding both multiplies the two.
const PAN_SLOW: f32 = 0.25;
/// Seconds the pan takes to ease down to `PAN_SLOW` when left Alt goes
/// down, and back up to full speed when it comes up.
const PAN_SLOW_RAMP: f32 = 1.5;
/// Zoom smoothing time constant (seconds to ~2/3 of the way).
const ZOOM_SMOOTH: f32 = 0.12;

fn max_distance(kind: crate::terrain::MapKind) -> f32 {
    if kind.is_large() { 2800.0 } else { 900.0 }
}

fn far_plane(kind: crate::terrain::MapKind) -> f32 {
    if kind.is_large() { 6000.0 } else { PerspectiveProjection::default().far }
}

#[derive(Component, Clone)]
pub struct RtsCamera {
    pub focus: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    /// Scroll adjusts this; `distance` chases it (smooth zoom).
    pub target_distance: f32,
}

pub struct RtsCameraPlugin;

impl Plugin for RtsCameraPlugin {
    fn build(&self, app: &mut App) {
        // Input only in battle (still active while paused, M2TW style)
        // and never under the settings modal; the transform apply keeps
        // running so the menu/results backdrop stays valid.
        app.add_systems(Startup, spawn_camera).add_systems(
            Update,
            (
                control_camera.run_if(
                    in_state(crate::game_state::GameState::Battle)
                        .and_then(crate::settings::settings_closed),
                ),
                apply_camera_transform,
            )
                .chain(),
        );
    }
}

fn spawn_camera(mut commands: Commands) {
    // The unit occlusion test reads the depth buffer in a compute pass
    // (render_units_occlusion.rs).
    let mut depth_usage = bevy::render::render_resource::TextureUsages::RENDER_ATTACHMENT;
    if crate::render_units_occlusion::mode() != crate::render_units_occlusion::OccludeMode::Off {
        depth_usage |= bevy::render::render_resource::TextureUsages::TEXTURE_BINDING;
    }
    commands.spawn((
        Camera3d {
            depth_texture_usages: depth_usage.into(),
            ..default()
        },
        msaa(),
        // Required: the unit renderer issues direct (non-indirect) draws.
        NoIndirectDrawing,
        RtsCamera {
            focus: Vec3::new(
                crate::util::env_or("FL_CAM_X", 0.0),
                0.0,
                crate::util::env_or("FL_CAM_Z", 0.0),
            ),
            // Overridable for screenshot-based debugging without input
            // injection (pitch in radians, 0.25..1.45).
            yaw: crate::util::env_or("FL_CAM_YAW", 0.0),
            pitch: crate::util::env_or("FL_CAM_PITCH", 0.9),
            distance: crate::util::env_or("FL_CAM_DIST", 280.0),
            target_distance: crate::util::env_or("FL_CAM_DIST", 280.0),
        },
    ));
}

/// Samples per pixel of the 3D view: Bevy's default 4, or `FL_MSAA=1`
/// (off), 2 or 8 for A/B runs.
fn msaa() -> Msaa {
    match crate::util::env_or("FL_MSAA", 4_u32) {
        1 => Msaa::Off,
        2 => Msaa::Sample2,
        8 => Msaa::Sample8,
        _ => Msaa::Sample4,
    }
}

#[allow(clippy::too_many_arguments)] // bevy system params
fn control_camera(
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    window: Query<&Window, With<PrimaryWindow>>,
    time: Res<Time<Real>>,
    settings: Res<crate::settings::Settings>,
    terrain: Res<crate::terrain::Terrain>,
    mut query: Query<&mut RtsCamera>,
    mut slow_ramp: Local<f32>,
) {
    let Ok(mut cam) = query.single_mut() else {
        return;
    };

    // Left Alt eases the pan speed down and back up along a smoothstep
    // (flat at both ends), so a recorded camera move never jerks. The
    // ramp runs whether or not the camera pans.
    let step = time.delta_secs() / PAN_SLOW_RAMP;
    *slow_ramp = if keys.pressed(KeyCode::AltLeft) {
        (*slow_ramp + step).min(1.0)
    } else {
        (*slow_ramp - step).max(0.0)
    };
    let slow = *slow_ramp * *slow_ramp * (3.0 - 2.0 * *slow_ramp);

    // Pan in the camera's yaw frame, speed scales with zoom. Ctrl is a
    // command modifier (Ctrl+A select all etc.): no panning under it.
    let ctrl =
        keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    let mut pan = Vec2::ZERO;
    if !ctrl && keys.pressed(KeyCode::KeyW) {
        pan.y -= 1.0;
    }
    if !ctrl && keys.pressed(KeyCode::KeyS) {
        pan.y += 1.0;
    }
    if !ctrl && keys.pressed(KeyCode::KeyA) {
        pan.x -= 1.0;
    }
    if !ctrl && keys.pressed(KeyCode::KeyD) {
        pan.x += 1.0;
    }
    // Screen-edge pan — only when no key pan is active (opposing inputs
    // must not cancel) and the cursor is inside the window.
    if settings.camera.edge_pan
        && pan == Vec2::ZERO
        && let Ok(win) = window.single()
        && let Some(c) = win.cursor_position()
    {
        let (w, h) = (win.width(), win.height());
        if c.x < EDGE_PAN_PX {
            pan.x -= 1.0;
        } else if c.x > w - EDGE_PAN_PX {
            pan.x += 1.0;
        }
        if c.y < EDGE_PAN_PX {
            pan.y -= 1.0;
        } else if c.y > h - EDGE_PAN_PX {
            pan.y += 1.0;
        }
    }
    if pan != Vec2::ZERO {
        let pan = pan.clamp_length_max(1.0);
        let mut factor = settings.camera.pan_speed;
        if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
            factor *= PAN_FAST;
        }
        factor *= 1.0 + (PAN_SLOW - 1.0) * slow;
        let speed = cam.distance * 0.9 * factor * time.delta_secs();
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
        cam.target_distance =
            (cam.target_distance * 0.9f32.powf(scroll_lines)).clamp(15.0, max_distance(terrain.kind));
    }
    // Smooth zoom: distance chases the scroll target.
    let blend = (time.delta_secs() / ZOOM_SMOOTH).min(1.0);
    cam.distance += (cam.target_distance - cam.distance) * blend;

    if buttons.pressed(MouseButton::Middle) && motion.delta != Vec2::ZERO {
        cam.yaw -= motion.delta.x * 0.005;
        cam.pitch = (cam.pitch + motion.delta.y * 0.005).clamp(0.25, 1.45);
    }
}

pub fn apply_camera_transform(
    mut query: Query<(&mut RtsCamera, &mut Transform, &mut Projection)>,
    terrain: Res<crate::terrain::Terrain>,
    time: Res<Time>,
    state: Res<State<crate::game_state::GameState>>,
    mut path: Local<Option<CamPath>>,
) {
    let Ok((mut cam, mut transform, mut projection)) = query.single_mut() else {
        return;
    };
    let limit = max_distance(terrain.kind);
    // FL_CAM_REC=<file> writes the camera's path while the battle runs;
    // FL_CAM_PLAY=<file> drives the camera along a recorded path, so a
    // measured run repeats the player's own spots and moves.
    if *state.get() == crate::game_state::GameState::Battle {
        let path = path.get_or_insert_with(CamPath::from_env);
        path.tick(&mut cam, time.elapsed_secs());
    }
    if terrain.is_changed() {
        cam.distance = cam.distance.min(limit);
        cam.target_distance = cam.target_distance.min(limit);
        if let Projection::Perspective(p) = &mut *projection {
            p.far = far_plane(terrain.kind);
        }
    }
    // FL_CAM_LOCK: pin the camera to the FL_CAM_* spawn values every
    // frame — screenshot debugging needs a stable frame even with a
    // cursor parked at a screen edge (edge pan) or a stray scroll.
    if std::env::var("FL_CAM_LOCK").is_ok() {
        cam.focus.x = crate::util::env_or("FL_CAM_X", 0.0);
        cam.focus.z = crate::util::env_or("FL_CAM_Z", 0.0);
        cam.yaw = crate::util::env_or("FL_CAM_YAW", 0.0);
        cam.pitch = crate::util::env_or("FL_CAM_PITCH", 0.9);
        cam.distance = crate::util::env_or("FL_CAM_DIST", 280.0);
        // FL_CAM_SWEEP=s: every s seconds the locked camera jumps between
        // the close-up and the map's widest view, the worst case
        // of a sudden zoom, so frame time under camera changes can be
        // measured from a log instead of by hand.
        let sweep = crate::util::env_or("FL_CAM_SWEEP", 0.0_f32);
        if sweep > 0.0 && (time.elapsed_secs() / sweep) as u32 % 2 == 1 {
            cam.distance = limit;
        }
        // A negative period gives a continuous zoom and orbit for checking
        // material transitions in motion, without injecting desktop input.
        if sweep < 0.0 {
            let phase = time.elapsed_secs() / -sweep * std::f32::consts::TAU;
            // The sweep starts at FL_CAM_DIST, so small plants can be seen up close.
            let close_distance = crate::util::env_or("FL_CAM_DIST", 40.0_f32).clamp(1.0, limit);
            cam.distance = close_distance + (limit - close_distance) * (0.5 - 0.5 * phase.cos());
            cam.yaw += phase.sin() * 0.35;
        }
        // FL_CAM_ORBIT=r: the locked camera turns about its focus at r
        // radians a second, for checking what lags a moving camera
        // (render_units_occlusion.rs) without desktop input.
        let orbit = crate::util::env_or("FL_CAM_ORBIT", 0.0_f32);
        cam.yaw += orbit * time.elapsed_secs();
        cam.target_distance = cam.distance;
    }
    let min = terrain.min();
    let max = terrain.max();
    cam.focus.x = cam.focus.x.clamp(min.x, max.x);
    cam.focus.z = cam.focus.z.clamp(min.y, max.y);
    cam.focus.y = terrain.height_at(cam.focus.x, cam.focus.z);
    *transform = Transform::from_translation(camera_eye(&cam, &terrain)).looking_at(cam.focus, Vec3::Y);
}

/// A camera path on disk: one line per sample, `t x z yaw pitch dist`,
/// `t` in seconds from the battle's start. FL_CAM_REC=<file> records one
/// every CAM_REC_STEP seconds; FL_CAM_PLAY=<file> plays one back, the
/// camera set to the sample for the battle's time (the last one held
/// once the path ends), over any input. Both at once record what the
/// playback did, the check of the player. FL_CAM_LOCK, applied after,
/// would override the playback: not together.
pub struct CamPath {
    rec: Option<(std::fs::File, f32)>,
    play: Option<Vec<[f32; 6]>>,
    start: Option<f32>,
}

const CAM_REC_STEP: f32 = 0.25;

impl CamPath {
    fn from_env() -> Self {
        let play = std::env::var("FL_CAM_PLAY").ok().and_then(|p| match std::fs::read_to_string(&p) {
            Ok(text) => {
                let samples: Vec<[f32; 6]> = text
                    .lines()
                    .filter_map(|l| {
                        let v: Vec<f32> = l.split_whitespace().filter_map(|t| t.parse().ok()).collect();
                        <[f32; 6]>::try_from(v).ok()
                    })
                    .collect();
                info!("camera path: playing {} samples from {p}", samples.len());
                Some(samples)
            }
            Err(e) => {
                warn!("camera path: cannot read {p}: {e}");
                None
            }
        });
        let rec = std::env::var("FL_CAM_REC").ok().and_then(|p| match std::fs::File::create(&p) {
            Ok(file) => {
                info!("camera path: recording to {p} every {CAM_REC_STEP} s");
                Some((file, 0.0))
            }
            Err(e) => {
                warn!("camera path: cannot write {p}: {e}");
                None
            }
        });
        Self { rec, play, start: None }
    }

    fn tick(&mut self, cam: &mut RtsCamera, now: f32) {
        if self.rec.is_none() && self.play.is_none() {
            return;
        }
        let t = now - *self.start.get_or_insert(now);
        if let Some(samples) = &self.play
            && let Some(last) = samples.last()
        {
            let i = samples.partition_point(|s| s[0] <= t);
            let s = if i == 0 {
                samples[0]
            } else if i >= samples.len() {
                *last
            } else {
                let (a, b) = (samples[i - 1], samples[i]);
                let k = ((t - a[0]) / (b[0] - a[0]).max(1e-3)).clamp(0.0, 1.0);
                std::array::from_fn(|j| a[j] + (b[j] - a[j]) * k)
            };
            cam.focus.x = s[1];
            cam.focus.z = s[2];
            cam.yaw = s[3];
            cam.pitch = s[4];
            cam.distance = s[5];
            cam.target_distance = s[5];
        }
        if let Some((file, next)) = &mut self.rec
            && t >= *next
        {
            use std::io::Write;
            *next = t + CAM_REC_STEP;
            let _ = writeln!(
                file,
                "{t:.2} {:.2} {:.2} {:.4} {:.4} {:.2}",
                cam.focus.x, cam.focus.z, cam.yaw, cam.pitch, cam.distance
            );
        }
    }
}

fn camera_eye(cam: &RtsCamera, terrain: &crate::terrain::Terrain) -> Vec3 {
    let rot = Quat::from_euler(EulerRot::YXZ, cam.yaw, -cam.pitch, 0.0);
    let mut eye = cam.focus + rot * Vec3::new(0.0, 0.0, cam.distance);
    // The focus can sit downhill from the eye at low pitch.
    eye.y = eye.y.max(terrain.height_at(eye.x, eye.z) + 1.5);
    eye
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::{MapKind, build_terrain};

    #[test]
    fn camera_clears_the_ground_at_minimum_zoom_and_pitch() {
        for kind in MapKind::ALL {
            let terrain = build_terrain(kind);
            let mut lowest = f32::INFINITY;
            for z in (terrain.min().y as i32..=terrain.max().y as i32).step_by(16) {
                for x in (terrain.min().x as i32..=terrain.max().x as i32).step_by(16) {
                    let focus = Vec3::new(x as f32, terrain.height_at(x as f32, z as f32), z as f32);
                    for angle in 0..16 {
                        let yaw = angle as f32 * std::f32::consts::TAU / 16.0;
                        let cam = RtsCamera { focus, yaw, pitch: 0.25, distance: 15.0, target_distance: 15.0 };
                        let eye = camera_eye(&cam, &terrain);
                        lowest = lowest.min(eye.y - terrain.height_at(eye.x, eye.z));
                    }
                }
            }
            println!("{} minimum camera clearance: {lowest:.3} m", kind.label());
            assert!(lowest >= 1.5 - 1e-4, "{} camera clearance {lowest}", kind.label());
        }
    }

    #[test]
    fn map_switches_update_camera_range_and_keep_focus_on_the_field() {
        let mut app = App::new();
        app.init_resource::<Time>()
            .insert_resource(build_terrain(MapKind::BigGrassland))
            .add_systems(Update, apply_camera_transform);
        let camera = app
            .world_mut()
            .spawn((
                RtsCamera {
                    focus: Vec3::new(2000.0, 0.0, 2000.0),
                    yaw: 0.0,
                    pitch: 1.45,
                    distance: 2800.0,
                    target_distance: 2800.0,
                },
                Transform::default(),
                Projection::default(),
            ))
            .id();
        for (kind, limit, far, corner) in [
            (MapKind::BigGrassland, 2800.0, 6000.0, Vec2::splat(1024.0)),
            (MapKind::BigHill, 2800.0, 6000.0, Vec2::new(1024.0, 768.0)),
            (MapKind::Grassland, 900.0, 1000.0, Vec2::new(512.0, 384.0)),
            (MapKind::Hill, 900.0, 1000.0, Vec2::new(512.0, 384.0)),
        ] {
            app.insert_resource(build_terrain(kind));
            app.update();
            let camera = app.world().entity(camera);
            let cam = camera.get::<RtsCamera>().unwrap();
            assert_eq!(cam.distance, limit);
            assert_eq!(cam.target_distance, limit);
            assert_eq!(cam.focus.xz(), corner);
            let Projection::Perspective(p) = camera.get::<Projection>().unwrap() else {
                panic!("expected perspective camera");
            };
            assert_eq!(p.far, far);
        }
    }
}
