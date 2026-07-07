//! Fixed-tick movement sim: goal steering toward a moving target point plus
//! boids-style separation from the spatial grid. Parallelized over SoA chunks
//! with the compute task pool.

use bevy::math::Vec3Swizzles;
use bevy::prelude::*;
use bevy::tasks::ComputeTaskPool;
use std::time::Instant;

use crate::spatial::SpatialGrid;
use crate::units::{UNIT_HALF_HEIGHT, Units};

const SEP_RADIUS: f32 = 1.4;
const SEP_STRENGTH: f32 = 60.0;
/// Cap on summed separation push to avoid explosive forces deep in a crowd.
const SEP_PUSH_MAX: f32 = 2.5;
/// Below this distance repulsion ramps up hard (cubes are 0.62 wide).
const HARD_RADIUS: f32 = 0.9;
const HARD_BOOST: f32 = 4.0;
/// Crowd-density yield: goal drive fades to zero between these two local
/// density values. Scalar density can't cancel out the way opposing push
/// vectors do, so this is what stops a goal-seeking crowd from compressing
/// itself into overlap: packed interior units genuinely stop shoving.
const CROWD_SLOW: f32 = 1.2;
const CROWD_STOP: f32 = 2.5;
const STEER_GAIN: f32 = 3.0;
const MAX_ACCEL: f32 = 50.0;
/// Positional overlap resolution: fraction of pair overlap corrected per
/// tick per unit, and the per-tick cap on total correction distance.
const CORR_GAIN: f32 = 0.5;
const CORR_MAX: f32 = 0.2;
/// Units slow down proportionally inside this distance to the target.
const ARRIVE_RADIUS: f32 = 60.0;
const CHUNK: usize = 2048;

/// Shared goal point, moving so the swarm keeps flowing.
#[derive(Resource, Default)]
pub struct MoveTarget(pub Vec2);

#[derive(Resource, Default)]
pub struct SimStats {
    pub grid_ms: f32,
    pub step_ms: f32,
    /// Smallest nearest-neighbor distance across all units (sampled every
    /// couple of seconds). Cube width is 0.62 — below that means overlap.
    pub nn_min: f32,
    pub nn_avg: f32,
}

/// Debug gizmo master toggle (G).
#[derive(Resource)]
pub struct DebugViz(pub bool);

pub struct MovementPlugin;

impl Plugin for MovementPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MoveTarget>()
            .init_resource::<SimStats>()
            .insert_resource(DebugViz(true))
            .init_resource::<SpatialGrid>()
            .add_systems(FixedUpdate, (update_targets, step_sim).chain())
            .add_systems(Update, (toggle_debug_viz, draw_debug_gizmos));
    }
}

fn update_targets(time: Res<Time>, mut target: ResMut<MoveTarget>) {
    // Slow lissajous sweep, peak speed comparable to unit speed so the
    // swarm can actually catch up and flow around it.
    let t = time.elapsed_secs() * 0.05;
    target.0 = Vec2::new((t * 1.0).cos() * 160.0, (t * 1.7).sin() * 110.0);
}

fn step_sim(
    mut units: ResMut<Units>,
    mut grid: ResMut<SpatialGrid>,
    target: Res<MoveTarget>,
    time: Res<Time>,
    mut stats: ResMut<SimStats>,
    mut tick: Local<u32>,
) {
    let dt = time.delta_secs();
    let Units {
        pos,
        pos_prev,
        vel,
        speed,
        team,
        ..
    } = &mut *units;
    if pos.is_empty() {
        return;
    }

    // pos_prev becomes the state at tick start; pos is fully rewritten below.
    std::mem::swap(pos, pos_prev);

    let t0 = Instant::now();
    grid.rebuild(pos_prev);
    let t1 = Instant::now();
    stats.grid_ms = (t1 - t0).as_secs_f32() * 1000.0;

    let grid = &*grid;
    let pos_prev = &pos_prev[..];
    let speed = &speed[..];
    let _ = team;
    let goal = target.0;

    ComputeTaskPool::get().scope(|scope| {
        for (ci, (p_chunk, v_chunk)) in pos.chunks_mut(CHUNK).zip(vel.chunks_mut(CHUNK)).enumerate()
        {
            let start = ci * CHUNK;
            scope.spawn(async move {
                for j in 0..p_chunk.len() {
                    let i = start + j;
                    let p = pos_prev[i].xz();

                    let to_goal = goal - p;
                    let dist = to_goal.length();
                    let desired_speed = speed[i] * (dist / ARRIVE_RADIUS).min(1.0);
                    let mut desired = if dist > 1e-3 {
                        to_goal * (desired_speed / dist)
                    } else {
                        Vec2::ZERO
                    };

                    let mut push = Vec2::ZERO;
                    let mut corr = Vec2::ZERO;
                    let mut crowd = 0.0f32;
                    grid.for_each_candidate(p, SEP_RADIUS, |o| {
                        let o = o as usize;
                        if o == i {
                            return;
                        }
                        let d = p - pos_prev[o].xz();
                        let d2 = d.length_squared();
                        if d2 < SEP_RADIUS * SEP_RADIUS && d2 > 1e-8 {
                            let len = d2.sqrt();
                            let w = 1.0 - len / SEP_RADIUS;
                            let mut wk = w;
                            if len < HARD_RADIUS {
                                wk += (HARD_RADIUS - len) * HARD_BOOST;
                                // Direct positional resolution of the overlap;
                                // forces alone respond too slowly.
                                corr += d * ((HARD_RADIUS - len) * CORR_GAIN / len);
                            }
                            push += d * (wk / len);
                            crowd += w;
                        }
                    });
                    let corr_len2 = corr.length_squared();
                    if corr_len2 > CORR_MAX * CORR_MAX {
                        corr *= CORR_MAX / corr_len2.sqrt();
                    }
                    let push_len = push.length();
                    if push_len > SEP_PUSH_MAX {
                        push *= SEP_PUSH_MAX / push_len;
                    }
                    // Yield in dense crowds: goal drive fades out entirely so
                    // the mass can't keep compressing itself.
                    desired *= ((CROWD_STOP - crowd) / (CROWD_STOP - CROWD_SLOW)).clamp(0.0, 1.0);

                    let v = v_chunk[j].xz();
                    let mut accel = (desired - v) * STEER_GAIN + push * SEP_STRENGTH;
                    let a2 = accel.length_squared();
                    if a2 > MAX_ACCEL * MAX_ACCEL {
                        accel *= MAX_ACCEL / a2.sqrt();
                    }

                    let mut new_v = v + accel * dt;
                    let vmax = speed[i] * 1.15; // slight overspeed under crowd pressure
                    let v2 = new_v.length_squared();
                    if v2 > vmax * vmax {
                        new_v *= vmax / v2.sqrt();
                    }
                    // After a positional correction, kill the velocity
                    // component still driving into the overlap or it
                    // re-penetrates next tick.
                    if corr_len2 > 1e-12 {
                        let cn = corr.normalize_or_zero();
                        let into = new_v.dot(-cn);
                        if into > 0.0 {
                            new_v += cn * into;
                        }
                    }

                    v_chunk[j] = Vec3::new(new_v.x, 0.0, new_v.y);
                    p_chunk[j] = Vec3::new(
                        pos_prev[i].x + new_v.x * dt + corr.x,
                        UNIT_HALF_HEIGHT,
                        pos_prev[i].z + new_v.y * dt + corr.y,
                    );
                }
            });
        }
    });
    stats.step_ms = t1.elapsed().as_secs_f32() * 1000.0;

    // Overlap audit over the full population, every 60 ticks (~2 s).
    *tick = tick.wrapping_add(1);
    if *tick % 60 == 0 {
        let mut min_d2 = f32::MAX;
        let mut sum_d = 0.0f64;
        let mut counted = 0u64;
        for (i, p) in pos_prev.iter().enumerate() {
            let p2 = p.xz();
            let mut best = f32::MAX;
            grid.for_each_candidate(p2, SEP_RADIUS, |o| {
                let o = o as usize;
                if o != i {
                    best = best.min(p2.distance_squared(pos_prev[o].xz()));
                }
            });
            if best < f32::MAX {
                min_d2 = min_d2.min(best);
                sum_d += best.sqrt() as f64;
                counted += 1;
            }
        }
        stats.nn_min = min_d2.sqrt();
        stats.nn_avg = if counted > 0 {
            (sum_d / counted as f64) as f32
        } else {
            0.0
        };
    }
}

fn toggle_debug_viz(keys: Res<ButtonInput<KeyCode>>, mut viz: ResMut<DebugViz>) {
    if keys.just_pressed(KeyCode::KeyG) {
        viz.0 = !viz.0;
    }
}

fn draw_debug_gizmos(viz: Res<DebugViz>, target: Res<MoveTarget>, mut gizmos: Gizmos) {
    if !viz.0 {
        return;
    }
    let color = Color::srgb(1.0, 0.95, 0.3);
    let center = Vec3::new(target.0.x, 0.5, target.0.y);
    gizmos.circle(
        Isometry3d::new(center, Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
        4.0,
        color,
    );
    gizmos.line(center - Vec3::Y * 0.5, center + Vec3::Y * 12.0, color);
}
