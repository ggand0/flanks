//! Fixed-tick movement sim: goal steering toward a moving target point plus
//! boids-style separation from the spatial grid. Parallelized over SoA chunks
//! with the compute task pool.

use bevy::math::Vec3Swizzles;
use bevy::prelude::*;
use bevy::tasks::ComputeTaskPool;
use std::time::Instant;

use crate::orders::Groups;
use crate::spatial::SpatialGrid;
use crate::terrain::Terrain;
use crate::unit_types::TYPES;
use crate::units::Units;

const SEP_RADIUS: f32 = 1.4;
const SEP_STRENGTH: f32 = 60.0;
/// Neighbor query radius: must cover both separation and the longest
/// melee reach in `unit_types::TYPES`.
const QUERY_RADIUS: f32 = 2.0;
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
/// Facing turn rate (rad/s toward the movement direction).
const YAW_RATE: f32 = 10.0;
/// Positional overlap resolution: fraction of pair overlap corrected per
/// tick per unit, and the per-tick cap on total correction distance.
const CORR_GAIN: f32 = 0.5;
const CORR_MAX: f32 = 0.2;
/// Units slow down proportionally inside this distance to the target.
const ARRIVE_RADIUS: f32 = 35.0;
const CHUNK: usize = 2048;

/// Global damage multiplier. FL_COMBAT_SCALE overrides (fast test battles).
#[derive(Resource)]
pub struct CombatScale(pub f32);

impl Default for CombatScale {
    fn default() -> Self {
        Self(
            std::env::var("FL_COMBAT_SCALE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0),
        )
    }
}

/// A landed swing, resolved after the parallel integrate: chunks emit into
/// their own buffer (no write races), then a serial pass applies damage.
pub struct DamageEvent {
    pub victim: u32,
    pub attacker: u32,
    pub dmg: f32,
}

/// One event buffer per integrate chunk; allocations persist across ticks.
#[derive(Resource, Default)]
pub struct DamageBuffers(pub Vec<Vec<DamageEvent>>);

/// Ticks a corpse persists (death anim) before swap-removal.
pub const DEATH_TICKS: u8 = 18;
/// Ticks of hit flash after taking damage.
const FLASH_TICKS: u8 = 4;

#[derive(Resource, Default)]
pub struct SimStats {
    pub grid_ms: f32,
    pub step_ms: f32,
    /// Density splat + blur + contour (frontline.rs), per tick.
    pub field_ms: f32,
    /// Last nn_audit sweep cost (runs every ~2 s).
    pub audit_ms: f32,
    /// Landed-swing events this tick (damage apply pass).
    pub events: usize,
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
        app.init_resource::<SimStats>()
            .init_resource::<CombatScale>()
            .init_resource::<DamageBuffers>()
            .insert_resource(DebugViz(true))
            .init_resource::<SpatialGrid>()
            .add_systems(FixedUpdate, step_sim)
            .add_systems(Update, toggle_debug_viz);
    }
}

#[allow(clippy::too_many_arguments)] // bevy system params
pub fn step_sim(
    mut units: ResMut<Units>,
    mut grid: ResMut<SpatialGrid>,
    mut damage: ResMut<DamageBuffers>,
    mut cstats: ResMut<crate::combat::CombatStats>,
    groups: Res<Groups>,
    terrain: Res<Terrain>,
    scale: Res<CombatScale>,
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
        kind,
        yaw,
        group,
        hp,
        target,
        swing,
        swing_t,
        flash,
        death_t,
        home,
        ..
    } = &mut *units;
    if pos.is_empty() {
        return;
    }

    // pos_prev becomes the state at tick start; pos is fully rewritten below.
    std::mem::swap(pos, pos_prev);

    let t0 = Instant::now();
    {
        let _span = info_span!("grid_rebuild").entered();
        grid.rebuild(pos_prev, team, kind, death_t);
    }
    let t1 = Instant::now();
    stats.grid_ms = (t1 - t0).as_secs_f32() * 1000.0;

    let grid = &*grid;
    let terrain = &*terrain;
    let pos_prev = &pos_prev[..];
    let speed = &speed[..];
    let team = &team[..];
    let kind = &kind[..];
    let group = &group[..];
    let home = &home[..];
    let orders: Vec<Option<Vec2>> = groups.list.iter().map(|g| g.order).collect();
    let orders = &orders[..];
    let anchors: Vec<Vec2> = groups.list.iter().map(|g| g.anchor).collect();
    let anchors = &anchors[..];
    let bounds_min = terrain.min() + 4.0;
    let bounds_max = terrain.max() - 4.0;

    let combat_scale = scale.0;
    let n_chunks = pos.len().div_ceil(CHUNK);
    if damage.0.len() < n_chunks {
        damage.0.resize_with(n_chunks, Vec::new);
    }
    let tick_seed = tick.wrapping_mul(0x9E37_79B1);
    let integrate_span = info_span!("integrate").entered();
    ComputeTaskPool::get().scope(|scope| {
        for (ci, chunk) in pos
            .chunks_mut(CHUNK)
            .zip(vel.chunks_mut(CHUNK))
            .zip(yaw.chunks_mut(CHUNK))
            .zip(target.chunks_mut(CHUNK))
            .zip(swing.chunks_mut(CHUNK))
            .zip(swing_t.chunks_mut(CHUNK))
            .zip(flash.chunks_mut(CHUNK))
            .zip(death_t.chunks_mut(CHUNK))
            .zip(&mut damage.0)
            .enumerate()
        {
            let ((((((((p_chunk, v_chunk), yaw_chunk), tgt_chunk), sw_chunk), swt_chunk),
                fl_chunk), dt_chunk), events) = chunk;
            let start = ci * CHUNK;
            scope.spawn(async move {
                events.clear();
                for j in 0..p_chunk.len() {
                    let i = start + j;
                    let p = pos_prev[i].xz();
                    let my_kind = kind[i] as usize;
                    let params = &TYPES[my_kind];
                    let dying = dt_chunk[j] > 0;

                    // Hit flash decays here (set by the serial apply pass).
                    fl_chunk[j] = fl_chunk[j].saturating_sub(1);
                    // Corpses play out their death anim: no orders, no
                    // combat; they stay as an obstacle until swept.
                    if dying && dt_chunk[j] > 1 {
                        dt_chunk[j] -= 1;
                    }

                    // Units move ONLY under orders. A regiment order is one
                    // point rigidly translated by each unit's `home` offset
                    // (the block moves; it never converges). No order =
                    // hold at the anchor — a standing order: units drift
                    // back to their slot at reduced gain inside the block.
                    // Enemy contact is pure physics: cross-team separation
                    // blocks, crowd yield stops the shove, combat thins the
                    // block. The "front line" is where that collision is.
                    let gi = group[i] as usize;
                    let mut desired = Vec2::ZERO;
                    if !dying {
                        let holding = orders[gi].is_none();
                        let goal = orders[gi].unwrap_or(anchors[gi]) + home[i];
                        let to_goal = goal - p;
                        let dist = to_goal.length();
                        // Hold deadzone: parked units don't jitter around
                        // their slot point.
                        if !(holding && dist < 1.5) {
                            // Slope penalty: steep ground is slow ground.
                            let slope = terrain.slope_at(p.x, p.y);
                            let slope_mult = 1.0 / (1.0 + 3.0 * slope * slope);
                            let (gain, arrive) = if holding {
                                (0.4, 10.0)
                            } else {
                                (1.0, ARRIVE_RADIUS)
                            };
                            let desired_speed =
                                speed[i] * slope_mult * gain * (dist / arrive).min(1.0);
                            if dist > 1e-3 {
                                desired = to_goal * (desired_speed / dist);
                            }
                        }
                    }

                    // Fused neighbor scan: separation physics + nearest
                    // living enemy in reach (swing targeting). Kept
                    // branch-light and side-effect-free per candidate —
                    // the future SIMD kernel depends on that shape.
                    let mut push = Vec2::ZERO;
                    let mut corr = Vec2::ZERO;
                    let mut crowd = 0.0f32;
                    let my_team_bit = (team[i] as u32) * crate::spatial::META_TEAM;
                    let my_mass = params.mass;
                    let reach2 = params.reach * params.reach;
                    let prev_target = tgt_chunk[j];
                    let mut best_d2 = f32::MAX;
                    let mut best_idx = u32::MAX;
                    let mut sticky = false;
                    grid.for_each_candidate(p, QUERY_RADIUS, |o| {
                        if o.idx as usize == i {
                            return;
                        }
                        let d = p - o.xz();
                        let d2 = d.length_squared();
                        let enemy = (o.meta & crate::spatial::META_TEAM) != my_team_bit
                            && (o.meta & crate::spatial::META_DYING) == 0;
                        if enemy && d2 < reach2 {
                            sticky |= o.idx == prev_target;
                            if d2 < best_d2 {
                                best_d2 = d2;
                                best_idx = o.idx;
                            }
                        }
                        if d2 < SEP_RADIUS * SEP_RADIUS && d2 > 1e-8 {
                            // Mass-weighted: heavies shove lights aside.
                            let o_mass = TYPES[((o.meta & crate::spatial::META_KIND) != 0)
                                as usize]
                                .mass;
                            let mw = 2.0 * o_mass / (my_mass + o_mass);
                            let len = d2.sqrt();
                            let w = 1.0 - len / SEP_RADIUS;
                            let mut wk = w * mw;
                            if len < HARD_RADIUS {
                                wk += (HARD_RADIUS - len) * HARD_BOOST * mw;
                                // Direct positional resolution of the overlap;
                                // forces alone respond too slowly.
                                corr += d * ((HARD_RADIUS - len) * CORR_GAIN * mw / len);
                            }
                            push += d * (wk / len);
                            crowd += w;
                        }
                    });

                    // Swing state machine. All writes are to this unit's own
                    // row; damage goes through the chunk event buffer.
                    let mut face_target = None;
                    if !dying {
                        match sw_chunk[j] {
                            crate::units::SWING_WINDUP => {
                                // Feet planted while winding up.
                                desired *= 0.25;
                                let t = tgt_chunk[j] as usize;
                                if t < pos_prev.len() {
                                    face_target = Some(pos_prev[t].xz());
                                }
                                if swt_chunk[j] == 0 {
                                    // Strike lands; validity (still alive,
                                    // still in reach, still an enemy) is
                                    // checked in the apply pass — a dodged
                                    // or dead target is a whiff.
                                    let jit =
                                        0.85 + 0.3 * crate::units::hash01(tick_seed ^ (i as u32));
                                    events.push(DamageEvent {
                                        victim: tgt_chunk[j],
                                        attacker: i as u32,
                                        dmg: params.damage * combat_scale * jit,
                                    });
                                    sw_chunk[j] = crate::units::SWING_RECOVER;
                                    let cjit = 0.75
                                        + 0.5
                                            * crate::units::hash01(
                                                tick_seed ^ (i as u32).wrapping_mul(0x9E37),
                                            );
                                    swt_chunk[j] = (params.cooldown_ticks as f32 * cjit) as u8;
                                } else {
                                    swt_chunk[j] -= 1;
                                }
                            }
                            crate::units::SWING_RECOVER => {
                                if swt_chunk[j] == 0 {
                                    sw_chunk[j] = crate::units::SWING_READY;
                                } else {
                                    swt_chunk[j] -= 1;
                                }
                            }
                            _ => {
                                // Ready: pick a target from the scan. Stick
                                // with the previous one when still in reach
                                // (duels), else nearest.
                                let chosen = if sticky { prev_target } else { best_idx };
                                if chosen != u32::MAX {
                                    tgt_chunk[j] = chosen;
                                    sw_chunk[j] = crate::units::SWING_WINDUP;
                                    swt_chunk[j] = params.windup_ticks;
                                }
                            }
                        }
                    }
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

                    // Face the combat target when winding up, else the
                    // movement direction, turning at YAW_RATE with wrap.
                    let face_dir = match face_target {
                        Some(t) => t - p,
                        None => new_v,
                    };
                    if !dying && face_dir.length_squared() > 0.25 {
                        let target_yaw = face_dir.x.atan2(face_dir.y);
                        let diff = (target_yaw - yaw_chunk[j] + std::f32::consts::PI)
                            .rem_euclid(std::f32::consts::TAU)
                            - std::f32::consts::PI;
                        yaw_chunk[j] += diff * (YAW_RATE * dt).min(1.0);
                    }

                    v_chunk[j] = Vec3::new(new_v.x, 0.0, new_v.y);
                    let nx = (pos_prev[i].x + new_v.x * dt + corr.x)
                        .clamp(bounds_min.x, bounds_max.x);
                    let nz = (pos_prev[i].z + new_v.y * dt + corr.y)
                        .clamp(bounds_min.y, bounds_max.y);
                    p_chunk[j] = Vec3::new(
                        nx,
                        terrain.height_at(nx, nz) + TYPES[kind[i] as usize].half_height,
                        nz,
                    );
                }
            });
        }
    });
    drop(integrate_span);
    stats.step_ms = t1.elapsed().as_secs_f32() * 1000.0;

    // Serial damage apply: deterministic (chunk order), race-free, and the
    // single place where hp transitions to death. A swing whiffs when its
    // victim died mid-wind-up, changed team slot via swap-remove, or slipped
    // out of reach — checked here, where all columns are whole again.
    {
        let _span = info_span!("damage_apply").entered();
        stats.events = 0;
        for buf in &mut damage.0 {
            stats.events += buf.len();
            for ev in buf.drain(..) {
                let (v, a) = (ev.victim as usize, ev.attacker as usize);
                if v >= hp.len() || a >= hp.len() {
                    continue;
                }
                if hp[v] <= 0.0 || death_t[v] > 0 || team[v] == team[a] {
                    continue;
                }
                let reach = TYPES[kind[a] as usize].reach * 1.15;
                if pos_prev[v].xz().distance_squared(pos_prev[a].xz()) > reach * reach {
                    continue;
                }
                hp[v] -= ev.dmg;
                flash[v] = FLASH_TICKS;
                if hp[v] <= 0.0 {
                    death_t[v] = DEATH_TICKS;
                    // kills[] counts losses OF that team (overlay semantics).
                    cstats.kills[team[v] as usize] += 1;
                }
            }
        }
    }

    // Overlap audit over the full population, every 60 ticks (~2 s).
    // Parallel over chunks; each task returns (min_d2, sum_d, counted).
    *tick = tick.wrapping_add(1);
    if (*tick).is_multiple_of(60) {
        let _span = info_span!("nn_audit").entered();
        let audit_t0 = Instant::now();
        let partials: Vec<(f32, f64, u64)> = ComputeTaskPool::get().scope(|scope| {
            for (ci, chunk) in pos_prev.chunks(CHUNK * 8).enumerate() {
                let start = ci * CHUNK * 8;
                scope.spawn(async move {
                    let mut min_d2 = f32::MAX;
                    let mut sum_d = 0.0f64;
                    let mut counted = 0u64;
                    for (j, p) in chunk.iter().enumerate() {
                        let i = start + j;
                        let p2 = p.xz();
                        let mut best = f32::MAX;
                        grid.for_each_candidate(p2, SEP_RADIUS, |o| {
                            if o.idx as usize != i {
                                best = best.min(p2.distance_squared(o.xz()));
                            }
                        });
                        if best < f32::MAX {
                            min_d2 = min_d2.min(best);
                            sum_d += best.sqrt() as f64;
                            counted += 1;
                        }
                    }
                    (min_d2, sum_d, counted)
                });
            }
        });
        let min_d2 = partials.iter().fold(f32::MAX, |m, p| m.min(p.0));
        let sum_d: f64 = partials.iter().map(|p| p.1).sum();
        let counted: u64 = partials.iter().map(|p| p.2).sum();
        stats.nn_min = min_d2.sqrt();
        stats.nn_avg = if counted > 0 {
            (sum_d / counted as f64) as f32
        } else {
            0.0
        };
        stats.audit_ms = audit_t0.elapsed().as_secs_f32() * 1000.0;
    }
}

fn toggle_debug_viz(keys: Res<ButtonInput<KeyCode>>, mut viz: ResMut<DebugViz>) {
    if keys.just_pressed(KeyCode::KeyG) {
        viz.0 = !viz.0;
    }
}

