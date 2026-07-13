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
/// Base neighbor query radius: covers separation and sword reach. Kinds
/// with longer reach (spears) widen their own scan per unit — the sword
/// majority must not pay for the spear's reach.
const QUERY_RADIUS: f32 = 2.0;
/// Cap on summed separation push to avoid explosive forces deep in a crowd.
const SEP_PUSH_MAX: f32 = 2.5;
/// Below this distance overlap is resolved positionally (bodies are
/// ~0.6 m wide).
const HARD_RADIUS: f32 = 0.9;
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
/// UNDER-relaxed on purpose: resolving overlap over ~3-4 ticks reads as
/// bodies settling; resolving it in 1-2 reads as twitching.
const CORR_GAIN: f32 = 0.3;
const CORR_MAX: f32 = 0.1;
/// Units slow down proportionally inside this distance to the target.
const ARRIVE_RADIUS: f32 = 35.0;
const CHUNK: usize = 2048;

/// Routing units flee at this fraction of their speed: fleeing at
/// exactly max speed made pursuit a zero-kill treadmill.
const ROUT_FLEE_FRAC: f32 = 0.9;
/// Sparse-fight acquisition radius. Two depleted formations interleave
/// with gaps wider than QUERY_RADIUS and the fight stalls (hits/tick 0
/// with both sides standing at "the front"). Pressing units whose normal
/// scan finds no enemy look this far on a staggered 1-in-8-tick cadence
/// and lock a closing target; swings still require reach.
const WIDE_ACQUIRE_R: f32 = 4.0;
/// Speed fraction above which a newly started swing counts as a charge.
const CHARGE_SPEED_FRAC: f32 = 0.6;
/// Damage multiplier for charging hits (momentum bonus).
const CHARGE_MULT: f32 = 1.75;

/// Global damage multiplier. FL_COMBAT_SCALE overrides (fast test battles).
#[derive(Resource)]
pub struct CombatScale(pub f32);

impl Default for CombatScale {
    fn default() -> Self {
        Self(crate::util::env_or("FL_COMBAT_SCALE", 1.0))
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
    /// Mean per-tick displacement in meters (audit cadence). The twitch
    /// metric: a static line should sit near zero, a marching unit ~0.3.
    pub move_avg: f32,
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
    mut groups: ResMut<Groups>,
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
        yaw_prev,
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
    // Orders resolved to this tick's destination (attack orders chase
    // their target regiment's current centroid).
    let orders: Vec<Option<Vec2>> = (0..groups.list.len()).map(|g| groups.goal(g)).collect();
    let orders = &orders[..];
    let anchors: Vec<Vec2> = groups.list.iter().map(|g| g.anchor).collect();
    let anchors = &anchors[..];
    let broken: Vec<bool> = groups.list.iter().map(|g| g.state.is_broken()).collect();
    let broken = &broken[..];
    // Regiments in combat-watch range of an enemy (sparse-fight
    // acquisition): order type is irrelevant — a Move-order fight that
    // went sparse stalls exactly the same way.
    let press: Vec<bool> = groups
        .list
        .iter()
        .map(|g| g.enemy_near || g.engaged)
        .collect();
    let press = &press[..];
    // Direction to the nearest enemy regiment (brace facing).
    let threat: Vec<Vec2> = groups.list.iter().map(|g| g.threat_dir).collect();
    let threat = &threat[..];
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
            .zip(yaw_prev.chunks_mut(CHUNK))
            .zip(target.chunks_mut(CHUNK))
            .zip(swing.chunks_mut(CHUNK))
            .zip(swing_t.chunks_mut(CHUNK))
            .zip(flash.chunks_mut(CHUNK))
            .zip(death_t.chunks_mut(CHUNK))
            .zip(&mut damage.0)
            .enumerate()
        {
            let (((((((((p_chunk, v_chunk), yaw_chunk), yawp_chunk), tgt_chunk), sw_chunk),
                swt_chunk), fl_chunk), dt_chunk), events) = chunk;
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
                    let routed = broken[gi];
                    let mut desired = Vec2::ZERO;
                    if !dying && routed {
                        // Broken: flee toward the own map edge with a
                        // per-unit lateral scatter — slightly SLOWER than
                        // formed pursuers (0.9x): fleeing at exactly max
                        // speed made pursuit a zero-kill treadmill (gap
                        // frozen forever, counts flat for 30-45 s).
                        let flee_z: f32 = if team[i] == 0 { -1.0 } else { 1.0 };
                        let lat = (crate::units::hash01((i as u32).wrapping_mul(17) + 3) - 0.5)
                            * 0.7;
                        let dir = Vec2::new(lat, flee_z).normalize_or_zero();
                        let slope = terrain.slope_at(p.x, p.y);
                        desired = dir * speed[i] * ROUT_FLEE_FRAC / (1.0 + 3.0 * slope * slope);
                    } else if !dying {
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
                    // living enemy in reach (swing targeting). Scalar over
                    // cache-ordered SortedUnits — an 8-wide SIMD variant
                    // measured slower here (devlog 0020): candidate runs
                    // are too short for lane occupancy.
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
                    grid.for_each_candidate(p, QUERY_RADIUS.max(params.reach), |o| {
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
                            let o_mass = TYPES[crate::spatial::meta_kind(o.meta)].mass;
                            let mw = 2.0 * o_mass / (my_mass + o_mass);
                            let len = d2.sqrt();
                            let w = 1.0 - len / SEP_RADIUS;
                            if len < HARD_RADIUS {
                                // Overlap is resolved POSITIONALLY only.
                                // (There used to be a hard force boost here
                                // too — two solvers fighting over the same
                                // overlap made packed crowds oscillate at
                                // the accel cap: the every-frame twitch.)
                                corr += d * ((HARD_RADIUS - len) * CORR_GAIN * mw / len);
                            }
                            push += d * (w * mw / len);
                            crowd += w;
                        }
                    });

                    // Sparse-fight acquisition (see WIDE_ACQUIRE_R): a
                    // pressing unit with an empty scan and open space
                    // around it memoizes a farther enemy in `target` so
                    // the closing drive below can restore contact. Gated
                    // hard (press + no near enemy + low crowd + 1/8
                    // cadence) to stay off the 200k hot path.
                    if best_idx == u32::MAX
                        && !dying
                        && !routed
                        && press[gi]
                        && crowd < CROWD_SLOW
                        && (i as u32).wrapping_add(tick_seed).is_multiple_of(8)
                    {
                        let mut far_d2 = WIDE_ACQUIRE_R * WIDE_ACQUIRE_R;
                        grid.for_each_candidate(p, WIDE_ACQUIRE_R, |o| {
                            let enemy = (o.meta & crate::spatial::META_TEAM) != my_team_bit
                                && (o.meta & crate::spatial::META_DYING) == 0;
                            if enemy {
                                let d2 = (p - o.xz()).length_squared();
                                if d2 < far_d2 {
                                    far_d2 = d2;
                                    tgt_chunk[j] = o.idx;
                                }
                            }
                        });
                    }

                    // Swing state machine. All writes are to this unit's own
                    // row; damage goes through the chunk event buffer.
                    let mut face_target = None;
                    if !dying {
                        match sw_chunk[j] & crate::units::SWING_STATE_MASK {
                            crate::units::SWING_WINDUP => {
                                // Feet planted while winding up — EXCEPT
                                // against a routing target: the cut-down
                                // happens at a run, or the runner is 3 m
                                // gone by the strike tick and every blow
                                // whiffs (the pursuit treadmill).
                                let t = tgt_chunk[j] as usize;
                                let target_routed =
                                    t < pos_prev.len() && broken[group[t] as usize];
                                if !target_routed {
                                    desired *= 0.25;
                                }
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
                                    let charge =
                                        if sw_chunk[j] & crate::units::SWING_CHARGE != 0 {
                                            CHARGE_MULT
                                        } else {
                                            1.0
                                        };
                                    events.push(DamageEvent {
                                        victim: tgt_chunk[j],
                                        attacker: i as u32,
                                        dmg: params.damage * combat_scale * jit * charge,
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
                                // (duels), else nearest. Routing units never
                                // start attacks (they still defend nothing —
                                // pursuit is free hits).
                                let chosen = if sticky { prev_target } else { best_idx };
                                if chosen != u32::MAX && !routed {
                                    tgt_chunk[j] = chosen;
                                    // Arriving at speed = a charging blow:
                                    // momentum converts to damage + a bigger
                                    // lunge (render reads the flag).
                                    let v2 = v_chunk[j].xz().length_squared();
                                    let cs = speed[i] * CHARGE_SPEED_FRAC;
                                    // Attack style for this swing (render
                                    // variety only): 0 = stab, 1 = the
                                    // classic swing. (2 = slash exists in
                                    // the shader, benched by owner.)
                                    // Spears only ever thrust.
                                    let style = if my_kind
                                        == crate::unit_types::KIND_SPEAR as usize
                                    {
                                        0
                                    } else {
                                        (crate::units::hash01(
                                            tick_seed ^ (i as u32).wrapping_mul(0x51ED),
                                        ) * 2.0) as u8
                                    };
                                    let style = style << crate::units::SWING_STYLE_SHIFT;
                                    sw_chunk[j] = if v2 > cs * cs {
                                        crate::units::SWING_WINDUP
                                            | crate::units::SWING_CHARGE
                                            | style
                                    } else {
                                        crate::units::SWING_WINDUP | style
                                    };
                                    swt_chunk[j] = params.windup_ticks;
                                }
                            }
                        }
                    }
                    let mut corr_len2 = corr.length_squared();
                    if corr_len2 < 1e-4 {
                        // Sub-centimeter corrections are settle noise, not
                        // overlap: applying them is pure micro-twitch.
                        corr = Vec2::ZERO;
                        corr_len2 = 0.0;
                    } else if corr_len2 > CORR_MAX * CORR_MAX {
                        corr *= CORR_MAX / corr_len2.sqrt();
                    }
                    let push_len = push.length();
                    if push_len > SEP_PUSH_MAX {
                        push *= SEP_PUSH_MAX / push_len;
                    }
                    // Yield in dense crowds: goal drive fades out entirely
                    // so the mass can't keep compressing itself; `jam` (0 =
                    // free, 1 = packed) also damps the response below.
                    let jam = ((crowd - CROWD_SLOW) / (CROWD_STOP - CROWD_SLOW)).clamp(0.0, 1.0);
                    desired *= 1.0 - jam;
                    // Fighters close the last meter to swing range. Only
                    // active when an enemy is ALREADY in reach — this is
                    // combat execution (like the wind-up foot plant), not
                    // steering; it bypasses the jam yield on purpose so
                    // front lines stay joined instead of settling at the
                    // separation standoff just outside sword range.
                    // Close toward the LOCKED swing target when there is
                    // one — the per-tick nearest enemy flips in a clog and
                    // flip-flopping the close direction reads as twitch.
                    // Falls back to the far-acquisition memo in `target`
                    // when the near scan is empty AND the unit is in open
                    // space — in a dense press the memo would let second
                    // ranks drive through the jam and compress the crowd
                    // (nn regression). The memo may be stale after death
                    // sweeps reindex, so it is validated as "some enemy
                    // within closing range" — a legitimate closing target
                    // regardless of identity.
                    let close_to = if sw_chunk[j] != crate::units::SWING_READY
                        && (tgt_chunk[j] as usize) < pos_prev.len()
                    {
                        tgt_chunk[j]
                    } else if best_idx != u32::MAX {
                        best_idx
                    } else if crowd < CROWD_SLOW {
                        tgt_chunk[j]
                    } else {
                        u32::MAX
                    };
                    if !dying
                        && !routed
                        && (close_to as usize) < pos_prev.len()
                        && team[close_to as usize] != team[i]
                    {
                        let to_enemy = pos_prev[close_to as usize].xz() - p;
                        let dist = to_enemy.length();
                        if dist > 1.2 && dist < WIDE_ACQUIRE_R + 1.0 {
                            let urge = ((dist - 1.2) / 0.8).clamp(0.0, 1.0);
                            desired += to_enemy * (speed[i] * 0.35 * urge / dist);
                        }
                    }

                    let v = v_chunk[j].xz();
                    // Jammed units stop shoving entirely: at full jam the
                    // crowd is quasi-static and overlap resolution is
                    // purely positional — force-based separation in a
                    // wedged mass only produces bang-bang oscillation.
                    let mut accel =
                        (desired - v) * STEER_GAIN + push * (SEP_STRENGTH * (1.0 - jam));
                    let a2 = accel.length_squared();
                    if a2 > MAX_ACCEL * MAX_ACCEL {
                        accel *= MAX_ACCEL / a2.sqrt();
                    }

                    let mut new_v = v + accel * dt;
                    // Viscous damping in the press: bleeds the spring energy
                    // that otherwise ping-pongs between neighbors every tick.
                    new_v *= 1.0 - 0.4 * jam;
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

                    // Facing priority: locked wind-up target > nearest
                    // enemy in reach > movement direction. Fighters keep
                    // eyes on the enemy even while the crowd shoves them;
                    // only routing/unengaged units face their velocity.
                    // yaw_prev snapshots the pre-update angle so the
                    // renderer can interpolate (yaw stepped once per tick
                    // otherwise — visible facing snaps at high fps).
                    yawp_chunk[j] = yaw_chunk[j];
                    let face_dir = match face_target {
                        Some(t) => t - p,
                        None if !routed && best_idx != u32::MAX => {
                            pos_prev[best_idx as usize].xz() - p
                        }
                        // Standing watch: near-stationary units of a
                        // regiment with enemy mass nearby turn toward it
                        // (brace facing) instead of keeping a stale yaw.
                        None if !routed
                            && new_v.length_squared() < 0.25
                            && threat[gi] != Vec2::ZERO =>
                        {
                            threat[gi]
                        }
                        None => new_v,
                    };
                    let min_len2 = if face_target.is_some() || best_idx != u32::MAX {
                        1e-4 // enemies can be close; still face them
                    } else {
                        0.25 // velocity facing ignores micro-drift
                    };
                    if !dying && face_dir.length_squared() > min_len2 {
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
                    groups.list[group[v] as usize].recent_deaths += 1;
                }
            }
        }
    }

    // Overlap + movement audit over the full population, every 60 ticks
    // (~2 s). Parallel over chunks; each task returns
    // (min_d2, sum_d, counted, sum_displacement).
    *tick = tick.wrapping_add(1);
    if (*tick).is_multiple_of(60) {
        let _span = info_span!("nn_audit").entered();
        let audit_t0 = Instant::now();
        let pos_now = &pos[..];
        let partials: Vec<(f32, f64, u64, f64)> = ComputeTaskPool::get().scope(|scope| {
            for (ci, chunk) in pos_prev.chunks(CHUNK * 8).enumerate() {
                let start = ci * CHUNK * 8;
                scope.spawn(async move {
                    let mut min_d2 = f32::MAX;
                    let mut sum_d = 0.0f64;
                    let mut counted = 0u64;
                    let mut sum_disp = 0.0f64;
                    for (j, p) in chunk.iter().enumerate() {
                        let i = start + j;
                        let p2 = p.xz();
                        sum_disp += p2.distance(pos_now[i].xz()) as f64;
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
                    (min_d2, sum_d, counted, sum_disp)
                });
            }
        });
        let min_d2 = partials.iter().fold(f32::MAX, |m, p| m.min(p.0));
        let sum_d: f64 = partials.iter().map(|p| p.1).sum();
        let counted: u64 = partials.iter().map(|p| p.2).sum();
        let sum_disp: f64 = partials.iter().map(|p| p.3).sum();
        stats.nn_min = min_d2.sqrt();
        stats.nn_avg = if counted > 0 {
            (sum_d / counted as f64) as f32
        } else {
            0.0
        };
        stats.move_avg = (sum_disp / pos_prev.len().max(1) as f64) as f32;
        stats.audit_ms = audit_t0.elapsed().as_secs_f32() * 1000.0;
    }
}

fn toggle_debug_viz(keys: Res<ButtonInput<KeyCode>>, mut viz: ResMut<DebugViz>) {
    if keys.just_pressed(KeyCode::KeyG) {
        viz.0 = !viz.0;
    }
}

