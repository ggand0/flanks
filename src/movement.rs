//! Fixed-tick movement sim: goal steering toward a moving target point plus
//! boids-style separation from the spatial grid. Parallelized over SoA chunks
//! with the compute task pool.

use bevy::math::Vec3Swizzles;
use bevy::prelude::*;
use bevy::tasks::ComputeTaskPool;
use std::time::Instant;

use crate::orders::Groups;
use crate::sim::job::{prepare_tick, TickJob, CHUNK};
use crate::spatial::SpatialGrid;
use crate::terrain::Terrain;
use crate::units::Units;

pub use crate::sim::damage::{
    DamageBuffers, DirTestStats, DEATH_TICKS, HIT_STAGGER_TICKS, SECTOR_COS_60,
    STAGGER_P0, STAGGER_P_MIN, STAGGER_P_PER_FACTOR,
};

#[derive(Resource)]
pub struct CombatScale(pub f32);

impl Default for CombatScale {
    fn default() -> Self {
        Self(crate::util::env_or("FL_COMBAT_SCALE", 1.0))
    }
}

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
    /// FL_LOG_STEP accumulators over the current 150-tick window.
    log_step_sum: f64,
    log_grid_sum: f64,
    log_step_n: u32,
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
            .init_resource::<DirTestStats>()
            .insert_resource(DebugViz(true))
            .init_resource::<SpatialGrid>()
            .init_resource::<TickPipeline>()
            .init_resource::<SharedTerrain>()
            .add_systems(
                FixedUpdate,
                (
                    (refresh_shared_terrain, step_sim).chain(),
                    kick_tick.after(crate::orders::clear_arrived_orders),
                )
                    .in_set(crate::game_state::SimSet),
            )
            .add_systems(Update, toggle_debug_viz);
    }
}

/// What an archer regiment's bows shoot at this tick (regiment-level
/// fire solution, resolved in `prepare_tick`).
/// Dedicated OS thread that runs tick jobs. NOT a bevy task: a system
/// that parks waiting for a pooled job can deadlock, because every worker
/// allowed to start that job may itself be parked inside a system task
/// (executor priority inversion, caught live with gdb, devlog 0050). A
/// dedicated thread is always runnable, a panic in the job surfaces as a
/// crash instead of a silent hang, and its scopes never tick the shared
/// executor (`util::sim_scope`), so it cannot steal a parked system and
/// close the cycle from the other side.
struct TickWorker {
    to_worker: std::sync::Mutex<std::sync::mpsc::Sender<Box<TickJob>>>,
    from_worker: std::sync::Mutex<std::sync::mpsc::Receiver<Box<TickJob>>>,
}

impl Default for TickWorker {
    fn default() -> Self {
        let (to_worker, jobs) = std::sync::mpsc::channel::<Box<TickJob>>();
        let (results, from_worker) = std::sync::mpsc::channel::<Box<TickJob>>();
        std::thread::Builder::new()
            .name("sim tick worker".into())
            .spawn(move || {
                crate::util::mark_sim_worker();
                while let Ok(mut job) = jobs.recv() {
                    run_tick_job(&mut job);
                    if results.send(job).is_err() {
                        return;
                    }
                }
            })
            .expect("spawn sim tick worker");
        Self {
            to_worker: std::sync::Mutex::new(to_worker),
            from_worker: std::sync::Mutex::new(from_worker),
        }
    }
}

/// The in-flight tick job, recycled job buffers and the tick counter. The
/// counter lives here, not in a `Local`, because the kick (whose prep
/// seeds the per-tick hashes) and the apply both need it.
#[derive(Resource, Default)]
pub struct TickPipeline {
    worker: Option<TickWorker>,
    in_flight: bool,
    scratch: Option<Box<TickJob>>,
    pub tick: u32,
}

impl TickPipeline {
    /// Block until the in-flight job is done and take it. Normally it
    /// finished long ago: a straggler waits here, inside the fixed tick,
    /// instead of stretching a render frame mid-computation.
    fn take_in_flight(&mut self) -> Option<Box<TickJob>> {
        if !self.in_flight {
            return None;
        }
        // Waiting here ON the worker thread would wait for a job that
        // thread can never finish. Fail loudly, never hang.
        assert!(
            !crate::util::on_sim_worker(),
            "a bevy system is running on the sim tick worker thread: a sim kernel scope bypassed util::sim_scope"
        );
        self.in_flight = false;
        let worker = self.worker.as_ref().expect("worker exists while a job is in flight");
        Some(worker.from_worker.lock().unwrap().recv().expect("sim tick worker alive"))
    }
}

/// FL_PIPELINE=0 computes every tick inline inside FixedUpdate, exactly
/// as the sim ran before the tick left the frame path: the fallback and
/// the A/B knob, bit-identical by FL_HASH.
fn pipeline_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("FL_PIPELINE").is_ok_and(|v| v == "0"))
}

/// `Arc` snapshot of the terrain for the tick job. Refreshed from the ECS
/// resource whenever it changes: terrain is static in battle, so this
/// clones once per map in practice. A job sees a terrain edit one tick
/// late.
#[derive(Resource, Default)]
pub struct SharedTerrain(pub Option<std::sync::Arc<Terrain>>);

pub fn refresh_shared_terrain(terrain: Res<Terrain>, mut shared: ResMut<SharedTerrain>) {
    if terrain.is_changed() || shared.0.is_none() {
        shared.0 = Some(std::sync::Arc::new(terrain.clone()));
    }
}

/// Run one kinematic tick on the job's owned data: grid rebuild, then the
/// parallel integrate. No ECS access, so it can run on any thread. The
/// kernel below is the long-standing integrate loop, verbatim: only the
/// binding preamble changed when the tick became a job. Every scope in
/// here, the grid rebuild's included, goes through `util::sim_scope`.
fn run_tick_job(job: &mut TickJob) {
    let terrain_arc = job.terrain.clone().expect("terrain snapshot set at prep");
    let terrain: &Terrain = &terrain_arc;
    let dt = job.dt;
    let tick_seed = job.tick_seed;
    let tick = job.tick;
    let bounds_min = job.bounds_min;
    let bounds_max = job.bounds_max;
    let faces_spearwall = job.faces_spearwall;
    let TickJob {
        pos_in,
        pos_out,
        vel,
        yaw,
        yaw_prev,
        target,
        swing,
        swing_t,
        flash,
        death_t,
        ammo,
        out_form,
        sight,
        speed,
        team,
        kind,
        group,
        home,
        orders,
        anchors,
        reg_broken,
        press,
        engaged,
        contact,
        fight_point,
        melee_ticks,
        hold,
        threat,
        form_face,
        wall,
        charging,
        fat_speed,
        fat_nocharge,
        shoot_at,
        target_members,
        blocks,
        wall_flags,
        yaw_snapshot,
        grid,
        events,
        arrow_spawns,
        grid_ms,
        step_ms,
        ..
    } = job;

    let t0 = Instant::now();
    {
        let _span = info_span!("grid_rebuild").entered();
        grid.rebuild(pos_in, vel, team, kind, group, death_t, wall_flags);
    }
    let t1 = Instant::now();
    *grid_ms = (t1 - t0).as_secs_f32() * 1000.0;

    let grid = &*grid;
    let pos_prev = &pos_in[..];
    let speed = &speed[..];
    let team = &team[..];
    let kind = &kind[..];
    let group = &group[..];
    let home = &home[..];
    let yaw_snap = &yaw_snapshot[..];
    let orders = &orders[..];
    let anchors = &anchors[..];
    let broken = &reg_broken[..];
    let press = &press[..];
    let engaged = &engaged[..];
    let contact = &contact[..];
    let fight_point = &fight_point[..];
    let melee_ticks = &melee_ticks[..];
    let hold = &hold[..];
    let threat = &threat[..];
    let form_face = &form_face[..];
    let wall = &wall[..];
    let charging = &charging[..];
    let fat_speed = &fat_speed[..];
    let fat_nocharge = &fat_nocharge[..];
    let shoot_at = &shoot_at[..];
    let target_members = &target_members[..];
    let blocks = &*blocks;
    let field = crate::sim::soldier::Field {
        grid,
        terrain,
        pos_prev,
        speed,
        team,
        kind,
        group,
        home,
        yaw_snap,
        orders,
        anchors,
        broken,
        press,
        engaged,
        contact,
        fight_point,
        melee_ticks,
        hold,
        threat,
        form_face,
        wall,
        charging,
        fat_speed,
        fat_nocharge,
        shoot_at,
        target_members,
        blocks,
        faces_spearwall,
        dt,
        tick_seed,
        tick,
        bounds_min,
        bounds_max,
    };
    let f = &field;
    let integrate_span = info_span!("integrate").entered();
    crate::util::sim_scope(|scope| {
        for (ci, chunk) in pos_out
            .chunks_mut(CHUNK)
            .zip(vel.chunks_mut(CHUNK))
            .zip(yaw.chunks_mut(CHUNK))
            .zip(yaw_prev.chunks_mut(CHUNK))
            .zip(target.chunks_mut(CHUNK))
            .zip(swing.chunks_mut(CHUNK))
            .zip(swing_t.chunks_mut(CHUNK))
            .zip(flash.chunks_mut(CHUNK))
            .zip(death_t.chunks_mut(CHUNK))
            .zip(events.iter_mut())
            .zip(ammo.chunks_mut(CHUNK))
            .zip(arrow_spawns.iter_mut())
            .zip(out_form.chunks_mut(CHUNK))
            .zip(sight.chunks_mut(CHUNK))
            .enumerate()
        {
            let (((((((((((((pos, vel), yaw), yaw_prev), target), swing), swing_t), flash),
                death_t), events), ammo), arrows), out_form), sight) = chunk;
            let rows = crate::sim::soldier::Rows {
                start: ci * CHUNK,
                pos,
                vel,
                yaw,
                yaw_prev,
                target,
                swing,
                swing_t,
                flash,
                death_t,
                ammo,
                out_form,
                sight,
            };
            scope.spawn(async move {
                events.clear();
                let mut out = crate::sim::soldier::ChunkOut { events, arrows };
                crate::sim::soldier::tick_chunk(f, rows, &mut out);
            });
        }
    });
    drop(integrate_span);
    *step_ms = t1.elapsed().as_secs_f32() * 1000.0;
}

#[allow(clippy::too_many_arguments)] // bevy system params
pub fn step_sim(
    mut units: ResMut<Units>,
    mut grid: ResMut<SpatialGrid>,
    mut damage: ResMut<DamageBuffers>,
    (mut arrow_spawns, tracks): (
        ResMut<crate::arrows::ArrowSpawns>,
        Res<crate::arrows::RegTracks>,
    ),
    mut cstats: ResMut<crate::combat::CombatStats>,
    mut groups: ResMut<Groups>,
    (terrain, shared_terrain): (Res<Terrain>, Res<SharedTerrain>),
    scale: Res<CombatScale>,
    time: Res<Time>,
    mut stats: ResMut<SimStats>,
    mut pipeline: ResMut<TickPipeline>,
    mut dir_stats: ResMut<DirTestStats>,
) {
    // Take the finished background job. A job computed from an older
    // world (a new battle started while it ran) carries indices that
    // mean nothing here: recycle its buffers and drop the result.
    let finished = match pipeline.take_in_flight() {
        Some(job) if job.generation == units.generation && !units.pos.is_empty() => Some(job),
        Some(stale) => {
            pipeline.scratch = Some(stale);
            None
        }
        None => None,
    };
    if units.pos.is_empty() {
        return;
    }
    // No job waiting (the first tick of a battle, FL_PIPELINE=0, or a
    // stale one dropped above): compute this tick inline.
    let mut job = match finished {
        Some(job) => job,
        None => {
            let Some(terrain_arc) = shared_terrain.0.as_ref() else {
                return;
            };
            let mut job = pipeline.scratch.take().unwrap_or_default();
            prepare_tick(
                &mut job,
                &units,
                &mut groups,
                &tracks,
                terrain_arc,
                time.delta_secs(),
                scale.0,
                pipeline.tick,
            );
            run_tick_job(&mut job);
            job
        }
    };

    // INSTALL: the completed tick becomes the live state. pos_prev <-
    // the state at tick start, pos <- the new kinematics, the
    // read-modify-write columns swap in, and the job keeps last tick's
    // buffers for recycling at the next prep.
    {
        let u = &mut *units;
        std::mem::swap(&mut u.pos, &mut u.pos_prev);
        std::mem::swap(&mut u.pos, &mut job.pos_out);
        std::mem::swap(&mut u.vel, &mut job.vel);
        std::mem::swap(&mut u.yaw, &mut job.yaw);
        std::mem::swap(&mut u.yaw_prev, &mut job.yaw_prev);
        std::mem::swap(&mut u.target, &mut job.target);
        std::mem::swap(&mut u.swing, &mut job.swing);
        std::mem::swap(&mut u.swing_t, &mut job.swing_t);
        std::mem::swap(&mut u.flash, &mut job.flash);
        std::mem::swap(&mut u.death_t, &mut job.death_t);
        std::mem::swap(&mut u.ammo, &mut job.ammo);
        std::mem::swap(&mut u.out_form, &mut job.out_form);
        std::mem::swap(&mut u.sight, &mut job.sight);
    }
    std::mem::swap(&mut *grid, &mut job.grid);
    std::mem::swap(&mut damage.0, &mut job.events);
    std::mem::swap(&mut arrow_spawns.0, &mut job.arrow_spawns);
    stats.grid_ms = job.grid_ms;
    stats.step_ms = job.step_ms;
    // FL_LOG_STEP=1: mean kinematic step and grid time over each 150
    // ticks (5 s), for cost comparisons between builds and knobs.
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ON.get_or_init(|| std::env::var("FL_LOG_STEP").is_ok()) {
            stats.log_step_sum += job.step_ms as f64;
            stats.log_grid_sum += job.grid_ms as f64;
            stats.log_step_n += 1;
            if stats.log_step_n == 150 {
                info!(
                    "[step] mean step {:.2} ms, grid {:.2} ms, {} units",
                    stats.log_step_sum / 150.0,
                    stats.log_grid_sum / 150.0,
                    units.pos.len()
                );
                stats.log_step_sum = 0.0;
                stats.log_grid_sum = 0.0;
                stats.log_step_n = 0;
            }
        }
    }
    stats.events = crate::sim::damage::apply_damage(
        &mut damage.0,
        &mut units,
        &mut groups,
        &crate::sim::damage::ApplyContext {
            wall: &job.wall,
            fat_nocharge: &job.fat_nocharge,
            bounds_min: job.bounds_min,
            bounds_max: job.bounds_max,
            combat_scale: job.combat_scale,
            tick_seed: job.tick_seed,
            terrain: &terrain,
        },
        &mut cstats,
        &mut dir_stats,
    );

    let Units {
        pos,
        pos_prev,
        yaw,
        hp,
        swing,
        swing_t,
        death_t,
        ammo,
        ..
    } = &mut *units;
    let grid = &*grid;
    let tick = &mut pipeline.tick;

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
                        grid.for_each_candidate(p2, crate::sim::soldier::SEP_RADIUS, |o| {
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

    // FL_HASH: FNV-1a fingerprint of the sim state on a fixed tick
    // cadence (every 150 ticks, or every FL_HASH=n ticks) — the
    // bit-identity gate for refactors and optimizations. Log samplers
    // ride wall time, so the same binary logs different casualty digits
    // run to run and log diffs prove nothing. Equal hashes at equal
    // ticks do.
    static HASH_EVERY: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    let hash_every = *HASH_EVERY.get_or_init(|| {
        let n: u32 = std::env::var("FL_HASH").ok()?.parse().unwrap_or(0);
        Some(if n > 1 { n } else { 150 })
    });
    if let Some(every) = hash_every
        && (*tick).is_multiple_of(every)
    {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for i in 0..pos.len() {
            for v in [
                pos[i].x.to_bits(),
                pos[i].z.to_bits(),
                yaw[i].to_bits(),
                hp[i].to_bits(),
                u32::from_le_bytes([death_t[i], swing[i], swing_t[i], ammo[i]]),
            ] {
                h ^= v as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        info!("[hash] tick {} n {} state {:016x}", *tick, pos.len(), h);
    }

    // Spike attribution: name any tick that blows past the norm, with
    // component costs inline — lag hunting works on facts, and the
    // audit above runs every 60 ticks inside this same system, so an
    // audit-tick spike shows its fresh cost here.
    let audit_now = (*tick).is_multiple_of(60);
    if stats.step_ms + stats.grid_ms + if audit_now { stats.audit_ms } else { 0.0 } > 14.0 {
        info!(
            "[spike] step {:.1} + grid {:.1}{} ms ({} damage events, {} units)",
            stats.step_ms,
            stats.grid_ms,
            if audit_now {
                format!(" + AUDIT {:.1}", stats.audit_ms)
            } else {
                String::new()
            },
            stats.events,
            pos.len(),
        );
    }

    pipeline.scratch = Some(job);
}

/// Kick the next tick's job once this tick is completely over: after the
/// arrows, the deaths (whose sweep reindexes every column), fatigue,
/// morale and order clearing. The job then owns a consistent copy of the
/// world, and nothing writes the soldier columns again before the next
/// `step_sim` installs the result. It computes while render frames go by,
/// so a slow tick delays its own completion inside the 33 ms budget
/// instead of stretching a frame. Regiment commands are read here, one
/// tick ahead of the install: orders quantize to the tick, as in a
/// lockstep sim.
pub fn kick_tick(
    units: Res<Units>,
    mut groups: ResMut<Groups>,
    tracks: Res<crate::arrows::RegTracks>,
    shared_terrain: Res<SharedTerrain>,
    scale: Res<CombatScale>,
    time: Res<Time>,
    mut pipeline: ResMut<TickPipeline>,
) {
    if !pipeline_enabled() || units.pos.is_empty() || pipeline.in_flight {
        return;
    }
    let Some(terrain_arc) = shared_terrain.0.as_ref() else {
        return;
    };
    let mut job = pipeline.scratch.take().unwrap_or_default();
    prepare_tick(
        &mut job,
        &units,
        &mut groups,
        &tracks,
        terrain_arc,
        time.delta_secs(),
        scale.0,
        pipeline.tick,
    );
    let worker = pipeline.worker.get_or_insert_with(TickWorker::default);
    worker.to_worker.lock().unwrap().send(job).expect("sim tick worker alive");
    pipeline.in_flight = true;
}

fn toggle_debug_viz(keys: Res<ButtonInput<KeyCode>>, mut viz: ResMut<DebugViz>) {
    if keys.just_pressed(KeyCode::KeyG) {
        viz.0 = !viz.0;
    }
}

