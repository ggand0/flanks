//! The battle simulation's fixed tick. Soldiers are rows in the `Units`
//! columns, never entities. Each tick is a self-contained job (job.rs):
//! the main thread shares the world's columns into it and kicks it, a
//! dedicated worker thread rebuilds the collision grid and runs every
//! soldier through his stages (soldier.rs) in parallel chunks, and the
//! next fixed tick takes the result, installs its columns and applies
//! the landed swings in order (damage.rs). Behavior lives in the soldier
//! stages; this file is the plumbing between the frame and the job.
//!
//! The rule of the fixed tick on the main thread: it does per-regiment
//! work and swaps, never a loop over the army. Whatever the frame needs
//! per soldier (the density field, the regiment runs, this tick's dead)
//! the job computes on the pool and hands over as a product.

use bevy::prelude::*;
use std::time::Instant;

use crate::frontline::InfluenceField;
use crate::orders::Groups;
use crate::spatial::SpatialGrid;
use crate::terrain::Terrain;
use crate::units::Units;

pub mod contact_viz;
pub mod damage;
mod diag;
pub mod job;
pub mod route;
pub mod soldier;

use damage::{DamageBuffers, DirTestStats};
use job::{prepare_tick, regiment_runs, TickJob, CHUNK};

/// Each regiment's men as index ranges in index order, valid for the
/// column layout the fixed tick starts with (before this tick's death
/// sweep). Built by the job from its shared regiment column, or on the
/// main thread when no job was taken. Consumers walk a regiment's runs
/// instead of scanning the army.
#[derive(Resource, Default)]
pub struct RegimentRuns {
    pub runs: Vec<Vec<(u32, u32)>>,
    /// The taken job's sums per regiment (frontline.rs `update_groups`),
    /// the inputs it took them with, and whether they hold for this tick:
    /// a job was taken and no write copied a column they read.
    pub(crate) job_sums: Vec<crate::frontline::RegimentSums>,
    pub(crate) job_inputs: crate::frontline::SumInputs,
    pub(crate) job_sums_valid: bool,
}

impl RegimentRuns {
    /// The index ranges of regiment `g` (empty for an unknown regiment).
    pub fn of(&self, g: usize) -> &[(u32, u32)] {
        self.runs.get(g).map_or(&[], |r| &r[..])
    }
}

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
    /// Last neighbour audit's cost in the tick job (runs every 2 s).
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

pub struct SimPlugin;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimStats>()
            .init_resource::<CombatScale>()
            .init_resource::<DamageBuffers>()
            .init_resource::<DirTestStats>()
            .insert_resource(DebugViz(true))
            .init_resource::<SpatialGrid>()
            .init_resource::<TickPipeline>()
            .init_resource::<SharedTerrain>()
            .init_resource::<RegimentRuns>()
            .init_resource::<route::Routes>()
            .add_systems(
                FixedUpdate,
                (
                    (refresh_shared_terrain, take_tick, step_sim).chain(),
                    route::update_routes.after(crate::orders::clear_arrived_orders),
                    kick_tick.after(route::update_routes),
                )
                    .in_set(crate::game_state::SimSet),
            )
            .init_resource::<contact_viz::ContactViz>()
            .add_systems(
                Update,
                (
                    toggle_debug_viz,
                    route::draw_routes,
                    contact_viz::toggle_contact_viz
                        .run_if(in_state(crate::game_state::GameState::Battle)),
                    contact_viz::draw_contact_viz,
                ),
            );
    }
}

/// Dedicated OS thread that runs tick jobs. NOT a bevy task: a system
/// that parks waiting for a pooled job can deadlock, because every worker
/// allowed to start that job may itself be parked inside a system task
/// (executor priority inversion, caught live with gdb). A
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
                    let start = crate::frame_log::job_start();
                    run_tick_job(&mut job);
                    crate::frame_log::job_end(start);
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

/// The in-flight tick job, the taken one, recycled job buffers and the
/// tick counter. The counter lives here, not in a `Local`, because the
/// kick (whose prep seeds the per-tick hashes) and the apply both need
/// it.
#[derive(Resource, Default)]
pub struct TickPipeline {
    worker: Option<TickWorker>,
    in_flight: bool,
    /// The finished job of this tick, taken at the head of the fixed
    /// tick (`take_tick`) so the systems before the install can read
    /// its products; `step_sim` installs it.
    taken: Option<Box<TickJob>>,
    scratch: Option<Box<TickJob>>,
    pub tick: u32,
    /// The density field resource holds the taken job's field for this
    /// tick (else `update_field` rebuilds it inline).
    pub field_from_job: bool,
    /// The death sweep's candidates for this tick, ascending: the men
    /// whose death countdown ran out in the installed tick and the
    /// living men near their own map edge. Every other man is neither
    /// dead nor fled, so the sweep visits only these.
    pub sweep_candidates: Vec<u32>,
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
        let tick_log_wait = crate::frame_log::tick::time(crate::frame_log::tick::Part::TakeWait);
        let start = crate::frame_log::take_start();
        let job = worker.from_worker.lock().unwrap().recv().expect("sim tick worker alive");
        crate::frame_log::take_end(start);
        drop(tick_log_wait);
        Some(job)
    }
}

/// The head of the fixed tick: take the finished job and hand its
/// products to the systems that run before the install. The regiment
/// runs and the density field are computed from the columns as they
/// were at the kick, which is exactly what those systems read now,
/// because nothing writes the columns while a job is in flight. With no
/// job to take (the first tick of a battle, FL_PIPELINE=0, a stale job)
/// the runs are built here and the field inline in `update_field`.
pub fn take_tick(
    mut pipeline: ResMut<TickPipeline>,
    units: Res<Units>,
    groups: Res<Groups>,
    mut runs: ResMut<RegimentRuns>,
    field: Option<ResMut<InfluenceField>>,
    mut stats: ResMut<SimStats>,
) {
    let _tick_log = crate::frame_log::tick::time(crate::frame_log::tick::Part::Take);
    // A job computed from an older world (a new battle started while it
    // ran) carries indices that mean nothing here: recycle its buffers
    // and drop the result.
    let taken = match pipeline.take_in_flight() {
        Some(job) if job.generation == units.generation && !units.pos.is_empty() => Some(job),
        Some(stale) => {
            pipeline.scratch = Some(stale);
            None
        }
        None => None,
    };
    pipeline.field_from_job = false;
    match taken {
        Some(mut job) => {
            // The job's regiment sums hold when it took them for every
            // regiment and every column they read is still the buffer it
            // read: a write while the job held a column copied it.
            runs.job_sums_valid = job.sums_wanted
                && job.sum_inputs.len() == groups.list.len()
                && job.group_sums.len() == groups.list.len()
                && job.shares_sum_columns(&units);
            // The run is over: let go of the shared columns now, so the
            // systems between here and the install (a regiment closing
            // ranks writes `home`) write in place instead of copying.
            job.release_inputs();
            std::mem::swap(&mut runs.runs, &mut job.reg_runs);
            std::mem::swap(&mut runs.job_sums, &mut job.group_sums);
            std::mem::swap(&mut runs.job_inputs, &mut job.sum_inputs);
            if let Some(mut field) = field {
                std::mem::swap(&mut *field, &mut job.field);
                pipeline.field_from_job = true;
                stats.field_ms = job.field_ms;
            }
            pipeline.taken = Some(job);
        }
        None => {
            regiment_runs(&units.group, groups.list.len(), &mut runs.runs);
            runs.job_sums_valid = false;
        }
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
    let _tick_log = crate::frame_log::tick::time(crate::frame_log::tick::Part::Terrain);
    if terrain.is_changed() || shared.0.is_none() {
        shared.0 = Some(std::sync::Arc::new(terrain.clone()));
    }
}

/// Run one kinematic tick on the job's data: the regiment runs and the
/// member lists of the regiments under fire, the grid rebuild, the
/// density field, then the parallel integrate. No ECS access, so it can
/// run on any thread. The kernel below is the long-standing integrate
/// loop, verbatim: only the binding preamble changed when the tick
/// became a job. Every scope in here, the grid rebuild's and the field's
/// included, goes through `util::sim_scope`.
fn run_tick_job(job: &mut TickJob) {
    let (generation, hit_flash) = (job.generation, job.hit_flash);
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
        speed,
        team,
        kind,
        group,
        home,
        vel_in,
        yaw_in,
        yaw_prev_in,
        target_in,
        swing_in,
        swing_t_in,
        flash_in,
        death_t_in,
        ammo_in,
        out_form_in,
        sight_in,
        chain_in,
        way_in,
        pose_in,
        stuck_in,
        aim_in,
        imposed_in,
        quarry_in,
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
        chain,
        way,
        pose,
        stuck,
        aim,
        imposed,
        quarry,
        orders,
        anchors,
        reg_broken,
        press,
        moving,
        passing,
        stance,
        focus,
        engaged,
        contact,
        fight_point,
        melee_ticks,
        hold,
        threat,
        form_face,
        wall,
        group_wall,
        charging,
        fat_speed,
        fat_nocharge,
        shoot_at,
        targeted,
        blocks,
        reg_runs,
        target_members,
        searched,
        search_blocks,
        sidearm,
        clear_members,
        clear_blocks,
        field,
        field_wanted,
        dead,
        at_edge,
        grid,
        events,
        arrow_spawns,
        grid_ms,
        step_ms,
        field_ms,
        plan_requests,
        plans,
        routes,
        nav,
        audit,
        pack,
        pos_prev_in,
        color_in,
        group_sums,
        sum_inputs,
        sums_wanted,
        ..
    } = job;

    let pos_prev = &pos_in[..];
    let speed = &speed[..];
    let team = &team[..];
    let kind = &kind[..];
    let group = &group[..];
    let home = &home[..];
    let yaw_snap = &yaw_in[..];
    let death_t_in = &death_t_in[..];
    let n_groups = orders.len();

    // The grid rebuild, and beside it, as tasks of the same pool that fill
    // the gaps between the rebuild's passes: the render snapshot (first,
    // the frame asks for it soonest), the regiment runs and members, and
    // the density field. None reads what another writes, and the kernel
    // below needs all four done. The rebuild and the field are timed on
    // their own: each its own work from its start to its end, while the
    // others run beside it.
    let field_wanted = *field_wanted;
    let (targeted, searched, clear_blocks) = (&targeted[..], &searched[..], &clear_blocks[..]);
    let sight_snap = &sight_in[..];
    let slot = pack.as_ref();
    let (runs_out, target_out, clear_out, field_out) = (&mut *reg_runs, &mut *target_members, &mut *clear_members, &mut *field);
    // Each regiment's sums for `update_groups` (frontline.rs), from the
    // columns as kicked: as soon as the runs exist, a task per few
    // regiments joins the pool's other work beside the rebuild.
    let sum_cols = crate::frontline::SumColumns {
        pos: pos_prev,
        home,
        death_t: death_t_in,
        vel: &vel_in[..],
        swing: &swing_in[..],
        target: &target_in[..],
        group,
    };
    let sum_inputs = &*sum_inputs;
    // Prep stored the inputs of every regiment it snapshotted; anything
    // else leaves no sums, and update_groups sums every regiment itself.
    *sums_wanted &= sum_inputs.len() == n_groups;
    if *sums_wanted {
        group_sums.resize_with(n_groups, Default::default);
    }
    let sums_out: Vec<&mut [crate::frontline::RegimentSums]> = if *sums_wanted {
        group_sums.chunks_mut(crate::frontline::REGIMENTS_PER_TASK).collect()
    } else {
        Vec::new()
    };
    let field_took: Vec<f32> = crate::util::sim_scope(|scope| {
        if let Some(slot) = slot {
            let columns = crate::render_units_gpu::PackColumns {
                pos: pos_prev,
                pos_prev: &pos_prev_in[..],
                yaw: yaw_snap,
                yaw_prev: &yaw_prev_in[..],
                kind,
                swing: &swing_in[..],
                swing_t: &swing_t_in[..],
                flash: &flash_in[..],
                group,
                death_t: death_t_in,
                sight: sight_snap,
                color: &color_in[..],
            };
            scope.spawn(async move {
                slot.pack((generation, tick), &columns, hit_flash);
                -1.0
            });
        }
        let (reg_runs, target_members, clear_members) = (runs_out, target_out, clear_out);
        scope.spawn(async move {
            // The regiment runs, and from them the living members of every
            // regiment under fire, in index order.
            regiment_runs(group, n_groups, reg_runs);
            let reg_runs = shared(reg_runs);
            for (k, out) in sums_out.into_iter().enumerate() {
                scope.spawn(async move {
                    let first = k * crate::frontline::REGIMENTS_PER_TASK;
                    for (j, sums) in out.iter_mut().enumerate() {
                        *sums = sum_inputs.sums(&sum_cols, &reg_runs[first + j], first + j);
                    }
                    -1.0
                });
            }
            target_members.resize_with(n_groups, Vec::new);
            for (g, members) in target_members.iter_mut().enumerate() {
                members.clear();
                if targeted[g] {
                    for &(s, e) in &reg_runs[g] {
                        members.extend((s..e).filter(|&i| death_t_in[i as usize] == 0));
                    }
                }
            }
            // And of every regiment under search, its clear men: no enemy in
            // his reach (soldier.rs SIGHT_FOE, as of the last tick) and farther
            // than a shaft's reach from every block of the shooters' side that
            // stands near his (job.rs `clear_blocks`). Ascending, so a kept pick
            // is found by a binary search (soldier.rs `quarry_ok`).
            let clear_r = job::clear_reach();
            clear_members.resize_with(n_groups, Vec::new);
            for (g, members) in clear_members.iter_mut().enumerate() {
                members.clear();
                if searched[g] {
                    let near = &clear_blocks[g];
                    for &(s, e) in &reg_runs[g] {
                        members.extend((s..e).filter(|&i| {
                            let i = i as usize;
                            if death_t_in[i] != 0 || sight_snap[i] & soldier::SIGHT_FOE != 0 {
                                return false;
                            }
                            let p = Vec2::new(pos_prev[i].x, pos_prev[i].z);
                            near.iter().all(|b| b.distance(p) > clear_r)
                        }));
                    }
                }
            }
            -1.0
        });
        if field_wanted {
            let field = field_out;
            scope.spawn(async move {
                let _span = info_span!("density_field").entered();
                let t = Instant::now();
                field.rebuild(pos_prev, team);
                t.elapsed().as_secs_f32() * 1000.0
            });
        }
        let _span = info_span!("grid_rebuild").entered();
        let t0 = Instant::now();
        grid.rebuild(pos_prev, &vel_in[..], team, kind, group, death_t_in, &pose_in[..], group_wall);
        *grid_ms = t0.elapsed().as_secs_f32() * 1000.0;
    });
    let t2 = Instant::now();
    *field_ms = field_took.iter().copied().fold(0.0, f32::max);

    let grid = &*grid;
    let orders = &orders[..];
    let anchors = &anchors[..];
    let broken = &reg_broken[..];
    let press = &press[..];
    let moving = &moving[..];
    let passing = &passing[..];
    let stance = &stance[..];
    let focus = &focus[..];
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
    let clear_members = &clear_members[..];
    let search_blocks = &search_blocks[..];
    let sidearm = &sidearm[..];
    let blocks = &*blocks;
    let field = soldier::Field {
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
        moving,
        passing,
        stance,
        focus,
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
        clear_members,
        search_blocks,
        sidearm,
        death_snap: death_t_in,
        blocks,
        faces_spearwall,
        dt,
        tick_seed,
        tick,
        bounds_min,
        bounds_max,
        paths: route::KernelRoutes { routes: &routes[..], nav: nav.as_deref() },
    };
    let f = &field;
    // The shared read-modify-write columns, for the copy-in per chunk.
    let vel_in = &vel_in[..];
    let yaw_in = &yaw_in[..];
    let yaw_prev_in = &yaw_prev_in[..];
    let target_in = &target_in[..];
    let swing_in = &swing_in[..];
    let swing_t_in = &swing_t_in[..];
    let flash_in = &flash_in[..];
    let ammo_in = &ammo_in[..];
    let out_form_in = &out_form_in[..];
    let sight_in = &sight_in[..];
    let chain_in = &chain_in[..];
    let way_in = &way_in[..];
    let pose_in = &pose_in[..];
    let stuck_in = &stuck_in[..];
    let aim_in = &aim_in[..];
    let imposed_in = &imposed_in[..];
    let quarry_in = &quarry_in[..];
    // The map edges a router leaves the field at (combat.rs), widened by
    // the charge knockback the apply pass can still add to a position:
    // the sweep's candidate list must hold every man it might remove.
    let edge_lo = terrain.min().y + 8.0 + damage::KNOCKBACK_MARGIN;
    let edge_hi = terrain.max().y - 8.0 - damage::KNOCKBACK_MARGIN;
    let integrate_span = info_span!("integrate").entered();
    let lists: Vec<(Vec<u32>, Vec<u32>)> = crate::util::sim_scope(|scope| {
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
            .zip(way.chunks_mut(CHUNK))
            .zip(chain.chunks_mut(CHUNK))
            .zip(pose.chunks_mut(CHUNK))
            .zip(quarry.chunks_mut(CHUNK))
            .zip(stuck.chunks_mut(CHUNK))
            .zip(aim.chunks_mut(CHUNK))
            .zip(imposed.chunks_mut(CHUNK))
            .enumerate()
        {
            let ((((((((((((((((((((pos, vel), yaw), yaw_prev), target), swing), swing_t), flash),
                death_t), events), ammo), arrows), out_form), sight), way), chain), pose), quarry), stuck), aim),
                imposed) = chunk;
            let start = ci * CHUNK;
            let end = start + pos.len();
            scope.spawn(async move {
                // The chunk's rows, copied in from the shared columns.
                vel.copy_from_slice(&vel_in[start..end]);
                yaw.copy_from_slice(&yaw_in[start..end]);
                yaw_prev.copy_from_slice(&yaw_prev_in[start..end]);
                target.copy_from_slice(&target_in[start..end]);
                swing.copy_from_slice(&swing_in[start..end]);
                swing_t.copy_from_slice(&swing_t_in[start..end]);
                flash.copy_from_slice(&flash_in[start..end]);
                death_t.copy_from_slice(&death_t_in[start..end]);
                ammo.copy_from_slice(&ammo_in[start..end]);
                out_form.copy_from_slice(&out_form_in[start..end]);
                sight.copy_from_slice(&sight_in[start..end]);
                way.copy_from_slice(&way_in[start..end]);
                chain.copy_from_slice(&chain_in[start..end]);
                pose.copy_from_slice(&pose_in[start..end]);
                quarry.copy_from_slice(&quarry_in[start..end]);
                stuck.copy_from_slice(&stuck_in[start..end]);
                aim.copy_from_slice(&aim_in[start..end]);
                imposed.copy_from_slice(&imposed_in[start..end]);
                let rows = soldier::Rows {
                    start,
                    pos: &mut *pos,
                    vel,
                    yaw,
                    yaw_prev,
                    target,
                    swing,
                    swing_t,
                    flash,
                    death_t: &mut *death_t,
                    ammo,
                    out_form,
                    sight,
                    way,
                    chain,
                    pose,
                    quarry,
                    stuck,
                    aim,
                    imposed,
                };
                events.clear();
                let mut out = soldier::ChunkOut { events, arrows };
                soldier::tick_chunk(f, rows, &mut out);
                // The sweep's candidates from this chunk: countdowns that
                // just ran out, and living men near their own map edge.
                let mut dead = Vec::new();
                let mut at_edge = Vec::new();
                for j in 0..pos.len() {
                    let i = start + j;
                    if death_t[j] == 1 {
                        dead.push(i as u32);
                    } else if death_t[j] == 0 {
                        let z = pos[j].z;
                        let out = if team[i] == 0 { z < edge_lo } else { z > edge_hi };
                        if out {
                            at_edge.push(i as u32);
                        }
                    }
                }
                (dead, at_edge)
            });
        }
    });
    drop(integrate_span);
    dead.clear();
    at_edge.clear();
    for (d, e) in lists {
        dead.extend(d);
        at_edge.extend(e);
    }
    *step_ms = t2.elapsed().as_secs_f32() * 1000.0;
    // The routes asked for at the kick, planned now that the soldiers are
    // done; the install hands them back.
    match nav.as_deref() {
        Some(nav) => {
            let _span = info_span!("plan_routes").entered();
            route::plan_all(nav, plan_requests, plans);
        }
        None => plans.clear(),
    }
    // The neighbour audit's sums on its ticks, from this tick's grid and
    // start positions; the tick count goes up by one at the install.
    *audit = diag::audit_tick(tick.wrapping_add(1)).then(|| diag::neighbour_sums(grid, pos_prev));
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
    mut sounds: Option<ResMut<crate::audio::SoundEvents>>,
    mut routes: ResMut<route::Routes>,
) {
    let _frame_log = crate::frame_log::time_step();
    let _tick_log = crate::frame_log::tick::time(crate::frame_log::tick::Part::Step);
    let finished = pipeline.taken.take();
    if units.pos.is_empty() {
        return;
    }
    // No job waiting (the first tick of a battle, FL_PIPELINE=0, or a
    // stale one dropped at the take): compute this tick inline.
    let mut job = match finished {
        Some(job) => job,
        None => {
            let _tick_log_inline = crate::frame_log::tick::time(crate::frame_log::tick::Part::StepInline);
            let Some(terrain_arc) = shared_terrain.0.as_ref() else {
                return;
            };
            let mut job = pipeline.scratch.take().unwrap_or_default();
            prepare_tick(
                &mut job,
                &units,
                &mut groups,
                &tracks,
                &mut routes,
                terrain_arc,
                time.delta_secs(),
                scale.0,
                pipeline.tick,
            );
            // update_field rebuilt the resource inline this tick.
            job.field_wanted = false;
            // update_groups summed this tick itself.
            job.sums_wanted = false;
            run_tick_job(&mut job);
            job
        }
    };

    // INSTALL: the completed tick becomes the live state. The job's
    // handles on the shared columns are released first (already done at
    // the take on the threaded path), so the columns the install
    // replaces are unique and come back as the next tick's output
    // buffers: no copy, no allocation. pos_prev <- the state at tick
    // start (the job's shared position column), pos <- the new
    // kinematics.
    let tick_log_install = crate::frame_log::tick::time(crate::frame_log::tick::Part::StepInstall);
    job.release_inputs();
    {
        let u = &mut *units;
        macro_rules! install {
            ($($col:ident <- $out:ident),*) => {$(
                let released = u.$col.install(std::mem::take(&mut job.$out));
                job.$out = std::sync::Arc::try_unwrap(released).unwrap_or_default();
            )*};
        }
        let tick_start = u.pos.install(std::mem::take(&mut job.pos_out));
        let released = u.pos_prev.install_shared(tick_start);
        job.pos_out = std::sync::Arc::try_unwrap(released).unwrap_or_default();
        install!(
            vel <- vel, yaw <- yaw, yaw_prev <- yaw_prev, target <- target, swing <- swing,
            swing_t <- swing_t, flash <- flash, death_t <- death_t, ammo <- ammo,
            out_form <- out_form, sight <- sight, way <- way, chain <- chain,
            pose <- pose, quarry <- quarry, stuck <- stuck, aim <- aim,
            imposed <- imposed
        );
    }
    // The sweep's candidates (combat.rs), ascending.
    pipeline.sweep_candidates.clear();
    pipeline.sweep_candidates.extend_from_slice(&job.dead);
    pipeline.sweep_candidates.extend_from_slice(&job.at_edge);
    pipeline.sweep_candidates.sort_unstable();
    pipeline.sweep_candidates.dedup();
    std::mem::swap(&mut *grid, &mut job.grid);
    routes.hand_back(&mut job.plans, &mut job.plan_requests);
    std::mem::swap(&mut damage.0, &mut job.events);
    std::mem::swap(&mut arrow_spawns.0, &mut job.arrow_spawns);
    stats.grid_ms = job.grid_ms;
    stats.step_ms = job.step_ms;
    diag::log_step(&mut stats, job.step_ms, job.grid_ms, units.pos.len());
    drop(tick_log_install);
    let tick_log_damage = crate::frame_log::tick::time(crate::frame_log::tick::Part::StepDamage);
    stats.events = damage::apply_damage(
        &mut damage.0,
        &mut units,
        &mut groups,
        &damage::ApplyContext {
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
        sounds.as_deref_mut(),
    );
    drop(tick_log_damage);

    pipeline.tick = pipeline.tick.wrapping_add(1);
    let tick = pipeline.tick;
    let audit = job.audit.take();
    if diag::audit_tick(tick) {
        let _frame_log = crate::frame_log::time_audit();
        let _tick_log_audit = crate::frame_log::tick::time(crate::frame_log::tick::Part::StepAudit);
        // The job's sums. When it left none (prepared under another tick
        // count), or a write between the kick and the install made the
        // start positions a copy, they are taken here from the same grid
        // and these positions.
        let sums = audit
            .filter(|sums| sums.of(&units.pos_prev[..]))
            .unwrap_or_else(|| diag::neighbour_sums(&grid, &units.pos_prev[..]));
        diag::audit_stats(&mut stats, &sums, &units.pos[..], &units.pos_prev[..]);
    }
    diag::fingerprint(tick, &units);
    diag::spike_line(&stats, tick, units.pos.len());

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
#[allow(clippy::too_many_arguments)] // bevy system params
pub fn kick_tick(
    units: Res<Units>,
    mut groups: ResMut<Groups>,
    tracks: Res<crate::arrows::RegTracks>,
    shared_terrain: Res<SharedTerrain>,
    scale: Res<CombatScale>,
    time: Res<Time>,
    mut pipeline: ResMut<TickPipeline>,
    (handoff, settings): (
        Option<ResMut<crate::render_units_gpu::SnapshotHandoff>>,
        Res<crate::settings::Settings>,
    ),
    mut routes: ResMut<route::Routes>,
) {
    let _frame_log = crate::frame_log::time_kick();
    let _tick_log = crate::frame_log::tick::time(crate::frame_log::tick::Part::Kick);
    if !pipeline_enabled() || units.pos.is_empty() || pipeline.in_flight {
        return;
    }
    let Some(terrain_arc) = shared_terrain.0.as_ref() else {
        return;
    };
    let tick_log_prep = crate::frame_log::tick::time(crate::frame_log::tick::Part::KickPrep);
    let mut job = pipeline.scratch.take().unwrap_or_default();
    prepare_tick(
        &mut job,
        &units,
        &mut groups,
        &tracks,
        &mut routes,
        terrain_arc,
        time.delta_secs(),
        scale.0,
        pipeline.tick,
    );
    job.field_wanted = true;
    // What the regiment sums read of each regiment, as the job sees it.
    job.sum_inputs = crate::frontline::sum_inputs(&groups, Some(&routes));
    job.sums_wanted = true;
    drop(tick_log_prep);
    let _tick_log_send = crate::frame_log::tick::time(crate::frame_log::tick::Part::KickSend);
    // The render snapshot of the columns as kicked: the job packs it on
    // the sim's pool while the frame runs, instead of the frame packing
    // the same columns after Update (render_units_gpu.rs).
    if let Some(mut handoff) = handoff.filter(|_| crate::render_units_gpu::job_packs_snapshot()) {
        job.pack = Some(handoff.expect((units.generation, pipeline.tick), units.last_changed()));
        job.hit_flash = settings.interface.hit_flash;
    }
    let worker = pipeline.worker.get_or_insert_with(TickWorker::default);
    worker.to_worker.lock().unwrap().send(job).expect("sim tick worker alive");
    pipeline.in_flight = true;
}

/// A finished exclusive borrow, read from now on by tasks beside each other.
fn shared<T>(r: &mut T) -> &T {
    r
}

fn toggle_debug_viz(keys: Res<ButtonInput<KeyCode>>, mut viz: ResMut<DebugViz>) {
    if keys.just_pressed(KeyCode::KeyG) {
        viz.0 = !viz.0;
    }
}

