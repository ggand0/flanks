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
use crate::unit_types::TYPES;
use crate::units::Units;

pub use crate::sim::damage::{
    DamageBuffers, DamageEvent, DirTestStats, DEATH_TICKS, HIT_STAGGER_TICKS, SECTOR_COS_60,
    STAGGER_P0, STAGGER_P_MIN, STAGGER_P_PER_FACTOR,
};

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
/// Facing turn smoothing (fraction/s of the remaining angle) — the
/// settle behavior for SMALL corrections.
const YAW_RATE: f32 = 10.0;
/// A blooded soldier (one whose combat memo holds a living enemy) keeps
/// facing the fight while that enemy stands within this range, instead
/// of dressing back to the ordered facing between swing cycles. Fresh
/// men still hold the line — this is memory of a fight, not proximity
/// awareness, so rear ambushes keep their first-blood advantage.
const KEEP_FACING_R: f32 = 6.0;
/// Hard angular speed cap (rad/s). The smoothing above is exponential,
/// so before this cap a 180° about-face finished in ~0.2 s — rear-
/// attacked soldiers whipped around between two swings and rear
/// attacks played out as frontal fights. A burdened man in ranks
/// turns ~180° in about a second: pi rad/s. Small angles never hit
/// the cap, so marching wheels and duel tracking feel unchanged.
const TURN_SPEED_MAX: f32 = std::f32::consts::PI;
/// Positional overlap resolution: fraction of pair overlap corrected per
/// tick per unit, and the per-tick cap on total correction distance.
/// UNDER-relaxed on purpose: resolving overlap over ~3-4 ticks reads as
/// bodies settling; resolving it in 1-2 reads as twitching.
const CORR_GAIN: f32 = 0.3;
const CORR_MAX: f32 = 0.1;
/// Units slow down proportionally inside this distance to the target.
const ARRIVE_RADIUS: f32 = 35.0;
/// A holding soldier who leaves his mark by more than the hold deadzone
/// walks back at this pace at least, and keeps going until he is within
/// STEP_STOP of it: a real step, never a creep. M2TW's slowest
/// locomotion is its shuffle, 0.75 to 0.9 m/s (devlog 0120).
const STEP_PACE: f32 = 0.8;
const STEP_STOP: f32 = 0.35;
/// Below this ground speed a standing formation's soldier keeps his
/// facing while he moves: he steps sideways or back, he does not turn
/// to walk (M2TW's shuffle keeps facing).
const STEP_FACE_SPEED: f32 = 1.05;
/// Going to an enemy he can see: M2TW's ready-stance `advance`
/// (1.06 m/s) for the last few meters or with a comrade close ahead,
/// its `combat_jog` (2.87 m/s) from further out over open ground
/// (devlog 0123).
const ADVANCE_PACE: f32 = 1.06;
const COMBAT_JOG_PACE: f32 = 2.87;
const JOG_BEYOND: f32 = 3.0;
/// Each man stops short of the enemy he goes to at his own distance,
/// 1.2 m plus up to this much: ranks are men, not a machine.
const SEEK_STOP_SPREAD: f32 = 0.4;
/// A man blocked by a comrade on the way to his enemy sidesteps toward
/// an open lane in some one-second windows (M2TW's isSideStepping),
/// and waits in the others.
const SIDESTEP_WINDOW: u32 = 30;

/// Share of those windows in which a blocked man sidesteps
/// (FL_SIDESTEP, 0 to 1). A play-testing knob.
fn sidestep_chance() -> f32 {
    static P: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *P.get_or_init(|| crate::util::env_or("FL_SIDESTEP", 0.2))
}
/// A standing man plants his feet: a push must exceed this (m/s^2 of
/// separation push) before it moves him. About a 10 cm squeeze from one
/// neighbor. Stepping and running men, and charge knockback, are
/// unaffected; true overlap is still corrected.
const STAND_GRIP: f32 = 6.0;

/// A man of a regiment in melee who sees no enemy joins the fight when
/// he sees the comrade beside him run toward it: in each half-second
/// window he reacts with this chance (FL_JOIN_REACT overrides), so the
/// line rolls up from the contact outward, a man at a time, as in
/// Gota's M2TW test (devlog 0123). Once going he keeps going.
const JOIN_WINDOW: u32 = 15;
fn join_react_chance() -> f32 {
    static P: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *P.get_or_init(|| crate::util::env_or("FL_JOIN_REACT", 0.2_f32).clamp(0.0, 1.0))
}
/// Ground speed toward the fight that reads as "going to it".
const GOING_SPEED: f32 = 1.0;
/// How far a man notices a comrade of his own regiment running to the
/// fight (on his far look, below).
const JOIN_SEE_R: f32 = 6.0;
/// A man does not re-read the comrades around him 30 times a second.
/// He looks at the comrades beside him and in his way every LOOK_TICKS
/// (about a quarter second, a human reaction time), in his own rhythm,
/// and acts on what he last saw (the `sight` column) in between. His
/// far look (enemies in sight, comrades running to the fight) keeps the
/// acquisition's own 1-in-8 rhythm.
const LOOK_TICKS: u32 = 8;
/// `sight` bits: a comrade blocks his way, his left, his right; a
/// comrade ahead; a comrade between him and his mark; a comrade close
/// by running to the fight; one further out (far look).
const SIGHT_WAY: u8 = 1;
const SIGHT_LEFT: u8 = 1 << 1;
const SIGHT_RIGHT: u8 = 1 << 2;
const SIGHT_AHEAD: u8 = 1 << 3;
const SIGHT_MARK: u8 = 1 << 4;
const SIGHT_GO: u8 = 1 << 5;
const SIGHT_GO_FAR: u8 = 1 << 6;
/// Even with nobody going in sight, a man joins once his regiment has
/// been in melee longer than his own patience, spread between these
/// (seconds; FL_JOIN_PATIENCE sets the upper end): the fight's noise and
/// the officers carry further than sight. Far men join late, never not.
const JOIN_PATIENCE_MIN: f32 = 25.0;
fn join_patience_max() -> f32 {
    static D: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *D.get_or_init(|| crate::util::env_or("FL_JOIN_PATIENCE", 50.0_f32).max(JOIN_PATIENCE_MIN))
}

/// Sight radius for a man of a fighting regiment looking for an enemy
/// to go to (FL_SEEK_R, meters). A play-testing knob.
fn seek_radius() -> f32 {
    static R: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *R.get_or_init(|| crate::util::env_or("FL_SEEK_R", 15.0))
}

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
/// The momentum bonus (attacker's charge_bonus attack points) resolves in
/// the damage pass — a braced SPEARWALL nullifies it (points beat momentum).
const CHARGE_SPEED_FRAC: f32 = 0.6;
/// A regiment in its charge phase runs the last stretch home.
const CHARGE_SPEED_BOOST: f32 = 1.15;
/// Walls advance at a deliberate pace — breaking into a run breaks the wall.
const WALL_SPEED_FRAC: f32 = 0.72;
/// Separation rest distance between two SAME-TEAM units who are both in a
/// wall stance: shoulder to shoulder. Without this the physics pushes a
/// wall back out to normal spacing and the tight slots never happen.
const WALL_SEP_RADIUS: f32 = 1.05;
/// The braced spear is a PHYSICAL hazard, not a stance rule: a
/// spearwall soldier's leveled point covers a line along his facing,
/// from POINT_MIN (get inside it and the spear is useless) out to his
/// reach, this wide. A body crossing that line at charge speed runs
/// onto the point — a damage event from the spearman powered by the
/// CHARGER's own momentum, and a stop (stagger). A charger who slips
/// between the lines (rank edge, dead wielder, hole in the wall)
/// reaches the spearman and swings with his full charge bonus: gaps
/// are real, coverage at 1.05 m wall pitch is ~2/3 of the frontage.
const SPEAR_POINT_MIN: f32 = 0.9;
const SPEAR_LINE_HALF_W: f32 = 0.35;

/// Global damage multiplier. FL_COMBAT_SCALE overrides (fast test battles).
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
    let pos = pos_out;
    let integrate_span = info_span!("integrate").entered();
    crate::util::sim_scope(|scope| {
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
            .zip(events.iter_mut())
            .zip(ammo.chunks_mut(CHUNK))
            .zip(arrow_spawns.iter_mut())
            .zip(out_form.chunks_mut(CHUNK))
            .zip(sight.chunks_mut(CHUNK))
            .enumerate()
        {
            let (((((((((((((p_chunk, v_chunk), yaw_chunk), yawp_chunk), tgt_chunk), sw_chunk),
                swt_chunk), fl_chunk), dt_chunk), events), ammo_chunk), arrow_out),
                of_chunk), si_chunk) = chunk;
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
                        desired = dir
                            * speed[i]
                            * ROUT_FLEE_FRAC
                            * terrain.wade_mult(p.x, p.y)
                            / (1.0 + 3.0 * slope * slope);
                    } else if !dying {
                        let holding = orders[gi].is_none();
                        let goal = orders[gi].unwrap_or(anchors[gi]) + home[i];
                        let to_goal = goal - p;
                        let dist = to_goal.length();
                        // Hold deadzone: parked units don't jitter around
                        // their slot point. 0.7 m (was 1.5 when homes were
                        // jittered spawn offsets): rigid slots sit exactly
                        // at the separation rest distance, so a dressed
                        // rank is force-free and can afford tight tolerance
                        // — with 1.5 the ranks never finished dressing.
                        // A soldier already stepping finishes the step
                        // (STEP_STOP) instead of stalling at the
                        // deadzone's edge.
                        let stepping = v_chunk[j].xz().length_squared()
                            > (0.5 * STEP_PACE) * (0.5 * STEP_PACE);
                        let deadzone = if stepping { STEP_STOP } else { 0.7 };
                        if !(holding && dist < deadzone) {
                            // Slope penalty: steep ground is slow ground.
                            // Wading the river is slow too (the bridge
                            // deck is dry: full speed).
                            let slope = terrain.slope_at(p.x, p.y);
                            let slope_mult = 1.0 / (1.0 + 3.0 * slope * slope);
                            let (gain, arrive) = if holding {
                                (0.6, 10.0)
                            } else {
                                (1.0, ARRIVE_RADIUS)
                            };
                            let mut desired_speed = speed[i]
                                * slope_mult
                                * terrain.wade_mult(p.x, p.y)
                                * gain
                                * (dist / arrive).min(1.0);
                            if holding {
                                desired_speed = desired_speed
                                    .max(STEP_PACE * slope_mult * terrain.wade_mult(p.x, p.y));
                            }
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
                    let my_wall = wall[gi] != 0;
                    let my_mass = params.mass;
                    let reach2 = params.reach * params.reach;
                    let prev_target = tgt_chunk[j];
                    let mut best_d2 = f32::MAX;
                    let mut best_idx = u32::MAX;
                    let mut sticky = false;
                    // Moving at charge speed with unspent momentum: braced
                    // enemy spears in the path are a collision hazard, and
                    // the scan must see out to SPEAR reach, not just mine.
                    // A man already run through (flash) or reeling is not
                    // re-impaled every tick — a spear is a point, not an
                    // aura.
                    let spear_reach = TYPES[crate::unit_types::KIND_SPEAR as usize].reach;
                    let cs = speed[i] * CHARGE_SPEED_FRAC;
                    let at_charge_speed = faces_spearwall[team[i] as usize]
                        && !dying
                        && fl_chunk[j] == 0
                        && sw_chunk[j] & crate::units::SWING_STAGGERED == 0
                        && v_chunk[j].xz().length_squared() > cs * cs;
                    let mut impale_idx = u32::MAX;
                    let mut impale_d = f32::MAX;
                    // In melee: his regiment has a fight. He is either
                    // still in formation or out of it (out_form: he left
                    // his slot to fight, a state that sticks until the
                    // melee ends; M2TW's isInFormation). Fighting takes
                    // him out of formation.
                    let in_melee = fight_point[gi].is_some() && !routed && !dying;
                    if !in_melee {
                        of_chunk[j] = false;
                        si_chunk[j] &= !(SIGHT_GO | SIGHT_GO_FAR);
                    } else if sw_chunk[j] & crate::units::SWING_STATE_MASK
                        != crate::units::SWING_READY
                        && sw_chunk[j] & crate::units::SWING_RANGED == 0
                    {
                        of_chunk[j] = true;
                    }
                    let committed = in_melee && of_chunk[j];
                    let join_fp = if in_melee { fight_point[gi] } else { None };
                    let scan_r = if at_charge_speed {
                        QUERY_RADIUS.max(params.reach).max(spear_reach)
                    } else {
                        QUERY_RADIUS.max(params.reach)
                    };
                    grid.for_each_candidate(p, scan_r, |o| {
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
                        // Spear-line collision: he is a braced enemy
                        // spearman, and my body is crossing his leveled
                        // point while I close at speed.
                        if at_charge_speed
                            && enemy
                            && (o.meta & crate::spatial::META_WALL) != 0
                            && crate::spatial::meta_kind(o.meta)
                                == crate::unit_types::KIND_SPEAR as usize
                        {
                            let ys = yaw_snap[o.idx as usize];
                            let f = Vec2::new(ys.sin(), ys.cos());
                            let rel = d; // spearman -> me (d = p - o.xz())
                            let fwd_d = rel.dot(f);
                            let lat = (rel.x * f.y - rel.y * f.x).abs();
                            let closing = v_chunk[j].xz().dot(d) < 0.0;
                            if fwd_d > SPEAR_POINT_MIN
                                && fwd_d < spear_reach
                                && lat < SPEAR_LINE_HALF_W
                                && closing
                                && fwd_d < impale_d
                            {
                                impale_d = fwd_d;
                                impale_idx = o.idx;
                            }
                        }
                        // Two same-team units both in a wall STANCE rest
                        // shoulder to shoulder (symmetric predicate: both
                        // sides compute the same radius). Ordinary ranks
                        // keep parade spacing even while fighting — the
                        // seams between files are what enemy bodies flow
                        // into.
                        let cross = (o.meta & crate::spatial::META_TEAM) != my_team_bit;
                        // No packing rule for fighting regiments: M2TW
                        // keeps its formation grid during melee, and
                        // observably loosens it. The fighting crowd's
                        // spacing is slots plus body collision, nothing
                        // else. A shoulder-to-shoulder press rest for
                        // fighting pairs sealed the very seams the
                        // intermix needs: a pressed front's gaps shrank
                        // to about 1.05 m against 0.95 m bodies and
                        // symmetric fights collapsed to a two-rank duel
                        // line. It must not come back.
                        let sep_r = if !cross && my_wall && (o.meta & crate::spatial::META_WALL) != 0 {
                            WALL_SEP_RADIUS
                        } else {
                            SEP_RADIUS
                        };
                        if d2 < sep_r * sep_r && d2 > 1e-8 {
                            // Mass-weighted: heavies shove lights aside.
                            let o_mass = TYPES[crate::spatial::meta_kind(o.meta)].mass;
                            let mw = 2.0 * o_mass / (my_mass + o_mass);
                            let len = d2.sqrt();
                            let w = 1.0 - len / sep_r;
                            if len < HARD_RADIUS {
                                // Overlap is resolved POSITIONALLY only.
                                // (There used to be a hard force boost here
                                // too — two solvers fighting over the same
                                // overlap made packed crowds oscillate at
                                // the accel cap: the every-frame twitch.)
                                corr += d * ((HARD_RADIUS - len) * CORR_GAIN * mw / len);
                            }
                            push += d * (w * mw / len);
                            // Jam density: COMPRESSED pairs only — a
                            // neighbor resting AT his pair's rest
                            // distance contests nothing, so a formed
                            // man's slot-keeping is never faded by the
                            // settled enemies one stride away (the
                            // yield-and-stay-ragged defect).
                            crowd += w;
                        }
                    });

                    // The enemy he remembers, if still in sight: he tracks
                    // him while he has no enemy in reach, and seeing him
                    // is what makes a man still in formation react. A man
                    // fighting someone in reach out of formation needs
                    // neither, and skips the lookup.
                    let memo = prev_target as usize;
                    let memo_valid = ((best_idx == u32::MAX && !dying && !routed)
                        || (in_melee && !committed))
                        && memo < pos_prev.len()
                        && team[memo] != team[i]
                        && pos_prev[memo].xz().distance_squared(p)
                            < (seek_radius() + 1.0) * (seek_radius() + 1.0);
                    // Which way he is going: to that enemy, else to his
                    // regiment's fight.
                    let memo_dir = if memo_valid {
                        (pos_prev[memo].xz() - p).normalize_or_zero()
                    } else if let Some(fp) = join_fp {
                        (fp - p).normalize_or_zero()
                    } else {
                        Vec2::ZERO
                    };
                    // The blocking test toward a holding man's own mark.
                    let slot_dir = if orders[gi].is_none() && engaged[gi] && !routed {
                        desired.normalize_or_zero()
                    } else {
                        Vec2::ZERO
                    };
                    // Open lanes to either side of his way.
                    let side_dir = Vec2::new(-memo_dir.y, memo_dir.x);
                    // His look at the comrades around him on his way (to
                    // his enemy, to the fight, or back to his mark), in
                    // his own rhythm: a separate pass over the neighbors
                    // of the separation scan, run only by men who are
                    // going somewhere. He looks for a lane only while he
                    // has no enemy in reach, and watches his comrades go
                    // only while he sees no enemy of his own.
                    let look_lanes = memo_dir != Vec2::ZERO && best_idx == u32::MAX && !dying && !routed;
                    let watch_go = in_melee && !committed && !memo_valid;
                    let slot_on = slot_dir != Vec2::ZERO;
                    if (i as u32).wrapping_add(tick).is_multiple_of(LOOK_TICKS) {
                        let mut seen = 0u8;
                        if look_lanes || watch_go || slot_on {
                            let (mut way, mut left, mut right, mut ahead_any, mut mark, mut go) =
                                (false, false, false, false, false, false);
                            // Branch-free: neighbors of both teams interleave
                            // in a melee, and branching on each one
                            // mispredicts.
                            grid.for_each_candidate_vel(p, scan_r, |o, ov| {
                                let comrade = (o.idx as usize != i)
                                    & ((o.meta & crate::spatial::META_TEAM) == my_team_bit);
                                let d = p - o.xz();
                                let d2 = d.length_squared();
                                let reach = 0.707 * d2.sqrt();
                                let near = d2 < SEP_RADIUS * SEP_RADIUS;
                                let going = ov.dot(memo_dir);
                                // A comrade ahead: he walks up to him
                                // instead of jogging, and at arm's length
                                // the man blocks his way. One already
                                // walking away the same way is neither: men
                                // heading for the same fight move together
                                // instead of each waiting for the next.
                                let away = going > 0.5;
                                let ahead = comrade & look_lanes & ((-d).dot(memo_dir) > reach) & !away;
                                ahead_any |= ahead;
                                way |= ahead & near;
                                // A comrade of his own regiment close by,
                                // running to the fight: the sight that
                                // makes him follow.
                                go |= comrade
                                    & watch_go
                                    & (crate::spatial::meta_group(o.meta) == gi)
                                    & (going > GOING_SPEED);
                                let lateral = (-d).dot(side_dir);
                                let beside = comrade & look_lanes & near;
                                left |= beside & (lateral > reach);
                                right |= beside & (lateral < -reach);
                                mark |= comrade & slot_on & near & ((-d).dot(slot_dir) > reach);
                            });
                            seen = (way as u8 * SIGHT_WAY)
                                | (left as u8 * SIGHT_LEFT)
                                | (right as u8 * SIGHT_RIGHT)
                                | (ahead_any as u8 * SIGHT_AHEAD)
                                | (mark as u8 * SIGHT_MARK)
                                | (go as u8 * SIGHT_GO);
                        }
                        si_chunk[j] = (si_chunk[j] & SIGHT_GO_FAR) | seen;
                    }
                    let way_blocked = si_chunk[j] & SIGHT_WAY != 0;
                    let left_blocked = si_chunk[j] & SIGHT_LEFT != 0;
                    let right_blocked = si_chunk[j] & SIGHT_RIGHT != 0;
                    let comrade_ahead = si_chunk[j] & SIGHT_AHEAD != 0;
                    let slot_blocked = si_chunk[j] & SIGHT_MARK != 0;
                    // Ran onto a braced spear: the collision is a damage
                    // event from the SPEARMAN, resolved with everything
                    // else in the serial apply (which also stops the
                    // runner). One point, one wound — the nearest line
                    // crossed this tick.
                    if impale_idx != u32::MAX {
                        let jit = 0.85
                            + 0.3
                                * crate::units::hash01(
                                    tick_seed ^ (i as u32).wrapping_mul(0x7A31),
                                );
                        events.push(DamageEvent {
                            victim: i as u32,
                            attacker: impale_idx,
                            jit,
                            charge: false,
                            impale: true,
                        });
                    }

                    // A regiment fighting on its contact frame never
                    // steps backward, away from its fight, to dress its
                    // ranks: a man ahead of his mark (bunched up behind a
                    // stopped front, or shoved forward) stays; he only
                    // steps forward or sideways to it. Without this a
                    // charging block walked backward to reopen its ranks
                    // the moment it made contact. A regiment struck from
                    // behind has its fight at its back, so it keeps
                    // stepping back to hold its ground.
                    if contact[gi] && !routed && form_face[gi] != Vec2::ZERO {
                        let back = desired.dot(form_face[gi]);
                        if back < 0.0 {
                            desired -= form_face[gi] * back;
                        }
                    }
                    // In melee, holding men step back to their marks only
                    // through open ground: shoved off his mark with a comrade
                    // between him and it, a man stands where he is until
                    // the way clears (the comrade dies, steps up or moves
                    // back to his own mark) instead of pressing into
                    // the man's back.
                    if slot_blocked {
                        desired = Vec2::ZERO;
                    }
                    // Sparse-fight acquisition (see WIDE_ACQUIRE_R): a
                    // pressing unit with an empty scan and open space
                    // around it memoizes a farther enemy in `target` so
                    // the closing drive below can restore contact. Gated
                    // hard (press + no near enemy + low crowd + 1/8
                    // cadence) to stay off the 200k hot path.
                    let acquire_crowd_lim = CROWD_SLOW;
                    // A man of a fighting regiment looks as far as he can
                    // see (seek_radius) on his far look; on the approach
                    // the old short scan stands.
                    let far_look = (i as u32).wrapping_add(tick_seed).is_multiple_of(8);
                    if far_look {
                        si_chunk[j] &= !SIGHT_GO_FAR;
                    }
                    if best_idx == u32::MAX
                        && !dying
                        && !routed
                        && press[gi]
                        && crowd < acquire_crowd_lim
                        && far_look
                    {
                        let look = if engaged[gi] { seek_radius() } else { WIDE_ACQUIRE_R };
                        let mut far_d2 = look * look;
                        grid.for_each_candidate(p, look, |o| {
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
                        // Further than the neighbor scan, within JOIN_SEE_R:
                        // a comrade of his own regiment running to the fight.
                        if watch_go
                            && grid.any_candidate_vel(p, look.min(JOIN_SEE_R), |o, ov| {
                                o.idx as usize != i
                                    && crate::spatial::meta_group(o.meta) == gi
                                    && (p - o.xz()).length_squared() < JOIN_SEE_R * JOIN_SEE_R
                                    && ov.dot(memo_dir) > GOING_SPEED
                            })
                        {
                            si_chunk[j] |= SIGHT_GO_FAR;
                        }
                    }


                    // Leaving formation to fight, in his own time: he
                    // reacts to an enemy in sight or a comrade seen running
                    // to the fight (a chance per half-second window), or
                    // his patience with the fight's noise runs out. Near
                    // men go first and the line rolls up outward.
                    let mut committed = committed;
                    if in_melee && !committed {
                        let patience = (JOIN_PATIENCE_MIN
                            + (join_patience_max() - JOIN_PATIENCE_MIN)
                                * crate::units::hash01((i as u32).wrapping_mul(0x3C6E) ^ 0xF372))
                            * 30.0;
                        let saw_comrade_go = si_chunk[j] & (SIGHT_GO | SIGHT_GO_FAR) != 0;
                        let react = (memo_valid || saw_comrade_go) && {
                            let window = (tick.wrapping_add((i as u32).wrapping_mul(11)) / JOIN_WINDOW)
                                .wrapping_mul(0x9E37_79B1);
                            crate::units::hash01(window ^ (i as u32).wrapping_mul(0x6A09))
                                < join_react_chance()
                        };
                        if react || melee_ticks[gi] as f32 > patience {
                            committed = true;
                            of_chunk[j] = true;
                        }
                    }
                    // Out of formation he never dresses on his slot until
                    // the melee ends and the regiment re-forms: he fights,
                    // goes for an enemy he sees, heads for the fight, or
                    // stands where he is when blocked. Inferring this from
                    // his speed made a man who slowed down run back to his
                    // slot and come out again, over and over.
                    if committed {
                        desired = Vec2::ZERO;
                    }

                    // Swing state machine. All writes are to this unit's own
                    // row; damage goes through the chunk event buffer.
                    let mut face_target = None;
                    if !dying {
                        match sw_chunk[j] & crate::units::SWING_STATE_MASK {
                            crate::units::SWING_WINDUP
                                if sw_chunk[j] & crate::units::SWING_RANGED != 0 =>
                            {
                                // Drawing the bow: feet planted, eyes on
                                // the target block.
                                desired = Vec2::ZERO;
                                if let Some(s) = &shoot_at[gi] {
                                    face_target = Some(s.c);
                                    if swt_chunk[j] == 0 {
                                        let h = |k: u32| {
                                            crate::units::hash01(
                                                tick_seed
                                                    ^ (i as u32).wrapping_mul(k).wrapping_add(k),
                                            )
                                        };
                                        // Aim: an actual soldier of the
                                        // target regiment (M2TW keeps a
                                        // per-soldier aim target, devlog
                                        // 0060), the M2TW range-
                                        // INDEPENDENT landing scatter,
                                        // and a lead for the block's
                                        // drift over the flight. The
                                        // footprint-disc spot stands in
                                        // only if every member died
                                        // this tick.
                                        let members = &target_members[s.t];
                                        let base = if members.is_empty() {
                                            let ang =
                                                h(0x1F3B) * std::f32::consts::TAU;
                                            let rad = s.r * 0.85 * h(0x2E5D).sqrt();
                                            s.c + Vec2::new(ang.cos(), ang.sin()) * rad
                                        } else {
                                            let m = members[(h(0x1F3B)
                                                * members.len() as f32)
                                                as usize
                                                % members.len()]
                                                as usize;
                                            pos_prev[m].xz()
                                        };
                                        let sigma =
                                            crate::unit_types::missile::SCATTER_SIGMA * 1.75;
                                        let mut aim = base
                                            + Vec2::new(
                                                (h(0x3B7F) + h(0x45A3) - 1.0) * sigma,
                                                (h(0x5DC1) + h(0x6B8D) - 1.0) * sigma,
                                            );
                                        aim += s.vel * (aim.distance(p) / 30.0);
                                        // Launch ABOVE the body-hit band
                                        // (arrows.rs tops out at ground
                                        // + 1.15): a shaft leaving at
                                        // head height is inside its own
                                        // shooter's hit cylinder at
                                        // flight-time zero and kills him.
                                        let from =
                                            Vec3::new(p.x, pos_prev[i].y + 0.75, p.y);
                                        let to = Vec3::new(
                                            aim.x,
                                            terrain.height_at(aim.x, aim.y) + 0.7,
                                            aim.y,
                                        );
                                        arrow_out.push(crate::arrows::ArrowSpawn {
                                            pos: from,
                                            vel: crate::arrows::solve_launch(
                                                from,
                                                to,
                                                &blocks[team[i] as usize],
                                                terrain,
                                            ),
                                            team: team[i],
                                            group: group[i],
                                        });
                                        ammo_chunk[j] = ammo_chunk[j].saturating_sub(1);
                                        sw_chunk[j] = crate::units::SWING_RECOVER;
                                        // Reload: the M2TW volley cycle is
                                        // animation-bound at ~10 s; the
                                        // jitter keeps later volleys ragged.
                                        swt_chunk[j] =
                                            (crate::unit_types::missile::RELOAD_TICKS as f32
                                                * (0.8 + 0.25 * h(0x77F1)))
                                                .min(255.0)
                                                as u8;
                                    } else {
                                        swt_chunk[j] -= 1;
                                    }
                                } else {
                                    // Target gone mid-draw: ease off and
                                    // reassess shortly.
                                    sw_chunk[j] = crate::units::SWING_RECOVER;
                                    swt_chunk[j] = crate::unit_types::missile::CANCEL_TICKS;
                                }
                            }
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
                                    events.push(DamageEvent {
                                        victim: tgt_chunk[j],
                                        attacker: i as u32,
                                        jit,
                                        charge: sw_chunk[j] & crate::units::SWING_CHARGE != 0,
                                        impale: false,
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
                                    // A stagger that just wore off leaves
                                    // one free pass against the next one
                                    // (anti-stunlock); a plain recovery
                                    // carries an unspent pass forward.
                                    let immune = if sw_chunk[j]
                                        & crate::units::SWING_STAGGERED
                                        != 0
                                    {
                                        crate::units::SWING_STAGGER_IMMUNE
                                    } else {
                                        sw_chunk[j] & crate::units::SWING_STAGGER_IMMUNE
                                    };
                                    sw_chunk[j] = crate::units::SWING_READY | immune;
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
                                // No eyes in the back of his head: a man
                                // only opens on a target in his forward
                                // half-plane. Being struck tells him where
                                // to turn (the flash facing below), but it
                                // does NOT let him swing backward over his
                                // shoulder — he attacks once he has turned
                                // far enough, at the human turn-speed cap.
                                // Without this gate every rear-approached
                                // victim counter-wound-up on proximity and
                                // was face-on before the first blow landed;
                                // the rear sector never fired in practice.
                                let aware = chosen != u32::MAX && {
                                    let t = chosen as usize;
                                    t < pos_prev.len() && {
                                        let to_t = pos_prev[t].xz() - p;
                                        let fwd = Vec2::new(
                                            yaw_chunk[j].sin(),
                                            yaw_chunk[j].cos(),
                                        );
                                        fwd.dot(to_t) >= 0.0
                                    }
                                };
                                if chosen != u32::MAX && !routed && aware {
                                    tgt_chunk[j] = chosen;
                                    // Arriving at speed = a charging blow:
                                    // momentum converts to damage + a bigger
                                    // lunge (render reads the flag).
                                    let v2 = v_chunk[j].xz().length_squared();
                                    let cs = speed[i] * CHARGE_SPEED_FRAC;
                                    // Attack style for this swing (render
                                    // variety only): 0 = stab, 1 = the
                                    // classic swing. (2 = slash exists in
                                    // the shader but benched.)
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
                                } else if my_kind == crate::unit_types::KIND_ARCHER as usize
                                    && !routed
                                    && ammo_chunk[j] > 0
                                    && v_chunk[j].xz().length_squared() < 4.0
                                    && let Some(s) = &shoot_at[gi]
                                    && p.distance_squared(s.c)
                                        < crate::unit_types::missile::RANGE
                                            * crate::unit_types::missile::RANGE
                                {
                                    // Nock and draw (foot archers shoot
                                    // standing only; the walk gate keeps a
                                    // marching or skirmishing man's bow on
                                    // his back). Style bits stay 0: the
                                    // stab pull-back IS the string draw.
                                    sw_chunk[j] = crate::units::SWING_WINDUP
                                        | crate::units::SWING_RANGED;
                                    swt_chunk[j] = crate::unit_types::missile::DRAW_TICKS
                                        + (crate::units::hash01(
                                            tick_seed ^ (i as u32).wrapping_mul(0x2C9F),
                                        ) * 20.0) as u8;
                                }
                            }
                        }
                        // A reloading archer contacted in melee drops the
                        // reload: he defends at knife tempo instead of
                        // standing through the 8 s bow cycle.
                        if my_kind == crate::unit_types::KIND_ARCHER as usize
                            && best_idx != u32::MAX
                            && sw_chunk[j] & crate::units::SWING_STATE_MASK
                                == crate::units::SWING_RECOVER
                            && swt_chunk[j] > params.cooldown_ticks
                        {
                            swt_chunk[j] = params.cooldown_ticks;
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
                    let mut d_surge = Vec2::ZERO;
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
                    let (close_to, memo_close) = if sw_chunk[j] & crate::units::SWING_STATE_MASK
                        != crate::units::SWING_READY
                        && (tgt_chunk[j] as usize) < pos_prev.len()
                    {
                        (tgt_chunk[j], false)
                    } else if best_idx != u32::MAX {
                        (best_idx, false)
                    } else if crowd < acquire_crowd_lim && (!in_melee || committed) {
                        (tgt_chunk[j], true)
                    } else {
                        (u32::MAX, false)
                    };
                    if !dying
                        && !routed
                        && (close_to as usize) < pos_prev.len()
                        && team[close_to as usize] != team[i]
                    {
                        let to_enemy = pos_prev[close_to as usize].xz() - p;
                        let dist = to_enemy.length();
                        // Held regiments fight at arm's length only.
                        let look = if engaged[gi] { seek_radius() } else { WIDE_ACQUIRE_R };
                        let max_close = if hold[gi] { 2.2 } else { look + 1.0 };
                        // Going to a seen enemy, each man stops at his own
                        // distance; closing the last meter to a man already
                        // in reach stays as it was.
                        let stop = if memo_close {
                            1.2 + SEEK_STOP_SPREAD
                                * crate::units::hash01((i as u32).wrapping_mul(0x3C1B) ^ 0x51F7)
                        } else {
                            1.2
                        };
                        if dist > stop && dist < max_close {
                            let mut urge = ((dist - stop) / 0.8).clamp(0.0, 1.0);
                            // The surge toward a REMEMBERED enemy (no
                            // one in reach yet) is steering, not combat
                            // execution: it yields to the jam like all
                            // steering, so the press brakes on genuine
                            // body-pack. Ungated this factor is always
                            // 1 (the memo gate above already required
                            // crowd < CROWD_SLOW, i.e. jam == 0).
                            if memo_close {
                                urge *= 1.0 - jam;
                                // He steps toward a remembered enemy only
                                // through open ground. With a comrade in
                                // the way he stands and waits for room
                                // instead of leaning on the man's back
                                // (M2TW's crowded soldier, devlog 0121).
                                if way_blocked {
                                    urge = 0.0;
                                    // Waiting; in some windows he sidesteps
                                    // toward whichever side is open.
                                    let window = (tick.wrapping_add((i as u32).wrapping_mul(7))
                                        / SIDESTEP_WINDOW)
                                        .wrapping_mul(0x2545_F491);
                                    let roll = crate::units::hash01(window ^ (i as u32).wrapping_mul(0x9E37));
                                    let chance = sidestep_chance();
                                    if roll < chance {
                                        let side = match (left_blocked, right_blocked) {
                                            (false, true) => side_dir,
                                            (true, false) => -side_dir,
                                            (false, false) if roll < 0.5 * chance => side_dir,
                                            (false, false) => -side_dir,
                                            (true, true) => Vec2::ZERO,
                                        };
                                        desired += side * STEP_PACE;
                                    }
                                }
                            }
                            let pace = if !memo_close {
                                speed[i] * 0.35
                            } else if dist > JOG_BEYOND && !comrade_ahead {
                                COMBAT_JOG_PACE
                            } else {
                                ADVANCE_PACE
                            };
                            desired += to_enemy * (pace * urge / dist);
                            d_surge = to_enemy * (pace * urge / dist);
                        }
                    }
                    // Joining: no enemy to close on, so he heads for the
                    // enemy unit his regiment fights, jogging over open
                    // ground, walking with a comrade close ahead, waiting
                    // or sidestepping when blocked. He picks a soldier to
                    // fight once one is in sight (the acquisition above).
                    if committed
                        && !memo_valid
                        && let Some(fp) = join_fp
                        && best_idx == u32::MAX
                        && d_surge == Vec2::ZERO
                        && sw_chunk[j] & crate::units::SWING_STATE_MASK == crate::units::SWING_READY
                    {
                        let to_fp = fp - p;
                        let dist = to_fp.length();
                        if dist > 1.0 {
                            let dir = to_fp / dist;
                            desired = if !way_blocked {
                                let pace = if comrade_ahead { ADVANCE_PACE } else { COMBAT_JOG_PACE };
                                dir * pace * (1.0 - jam)
                            } else {
                                let window = (tick.wrapping_add((i as u32).wrapping_mul(7))
                                    / SIDESTEP_WINDOW)
                                    .wrapping_mul(0x2545_F491);
                                let roll = crate::units::hash01(window ^ (i as u32).wrapping_mul(0x9E37));
                                let chance = sidestep_chance();
                                let side = match (left_blocked, right_blocked) {
                                    _ if roll >= chance => Vec2::ZERO,
                                    (false, true) => side_dir,
                                    (true, false) => -side_dir,
                                    (false, false) if roll < 0.5 * chance => side_dir,
                                    (false, false) => -side_dir,
                                    (true, true) => Vec2::ZERO,
                                };
                                side * STEP_PACE
                            };
                        }
                    }
                    // Formation pace: walls advance deliberately (running
                    // breaks a wall), the charge phase runs the last
                    // stretch home. Broken/dying already excluded from
                    // both states by construction.
                    if wall[gi] != 0 {
                        desired *= WALL_SPEED_FRAC;
                    } else if charging[gi] && !dying && !routed && !fat_nocharge[gi] {
                        desired *= CHARGE_SPEED_BOOST;
                    }
                    desired *= fat_speed[gi];
                    // A staggered man reels where the blow left him: no
                    // steering, no closing, until the stun runs out. The
                    // shove that staggered him still resolves through
                    // separation — he is a body, not an actor.
                    if sw_chunk[j] & crate::units::SWING_STAGGERED != 0 {
                        desired = Vec2::ZERO;
                    }

                    let v = v_chunk[j].xz();
                    // Jammed units stop shoving entirely: at full jam the
                    // crowd is quasi-static and overlap resolution is
                    // purely positional — force-based separation in a
                    // wedged mass only produces bang-bang oscillation.
                    let mut push_a = push * (SEP_STRENGTH * (1.0 - jam));
                    // A standing man plants his feet: small pushes do not
                    // move him, real shoves do (less the grip). Without it
                    // any squeeze turned straight into sliding, and a
                    // fight's jostle rippled back through packed ranks.
                    if desired.length_squared() < 1e-6 {
                        let pl = push_a.length();
                        push_a *= if pl > STAND_GRIP { (pl - STAND_GRIP) / pl } else { 0.0 };
                    }
                    let mut accel = (desired - v) * STEER_GAIN + push_a;
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
                        // A man IN his swing cycle faces the fight (a
                        // formed one too — he is the fighting rim), and a
                        // man JUST STRUCK turns toward the blow (flash).
                        // A man merely NEAR an enemy does not: turning on
                        // proximity raced the attacker's wind-up and had
                        // every rear-approached victim frontal by first
                        // blood — the whole point of facing, gone. So the
                        // first hit lands in the back, spins its victim,
                        // and THEN he answers. Unformed units (Blob, no
                        // facing claim) still turn on proximity.
                        None if !routed
                            && best_idx != u32::MAX
                            && ((sw_chunk[j] & crate::units::SWING_STATE_MASK)
                                != crate::units::SWING_READY
                                || fl_chunk[j] > 0
                                || form_face[gi] == Vec2::ZERO) =>
                        {
                            pos_prev[best_idx as usize].xz() - p
                        }
                        // Out of formation with no enemy in reach: he faces
                        // where he is going, the enemy he remembers or the
                        // fight he heads for, standing or walking. Without
                        // this a blocked joiner fell through to the formed
                        // man's rule below and stood facing the line's front
                        // with the fight beside him.
                        None if !routed && committed && memo_dir != Vec2::ZERO => memo_dir,
                        // Blooded and the enemy still close: a man who has
                        // traded blows keeps facing the fight while his
                        // last foe (combat memo, validated by team and
                        // distance like the closing drive) stands within
                        // KEEP_FACING_R — no parade dressing with a sword
                        // a few strides away. A player reform takes hold
                        // once the ground near him clears. Fresh men fall
                        // through and hold the ordered line.
                        None if !routed
                            && new_v.length_squared() < STEP_FACE_SPEED * STEP_FACE_SPEED
                            && form_face[gi] != Vec2::ZERO
                            && (tgt_chunk[j] as usize) < pos_prev.len()
                            && team[tgt_chunk[j] as usize] != team[i]
                            && pos_prev[tgt_chunk[j] as usize]
                                .xz()
                                .distance_squared(p)
                                < KEEP_FACING_R * KEEP_FACING_R =>
                        {
                            pos_prev[tgt_chunk[j] as usize].xz() - p
                        }
                        // Standing in formation: HOLD the ordered facing.
                        // M2TW rule — a formed unit never rotates itself
                        // toward a threat (it goes "ready" in place; the
                        // render brace pose keys off enemy_near, not yaw);
                        // facing is the player's job, and leaving a flank
                        // open is supposed to cost.
                        None if !routed
                            && new_v.length_squared() < STEP_FACE_SPEED * STEP_FACE_SPEED
                            && form_face[gi] != Vec2::ZERO =>
                        {
                            form_face[gi]
                        }
                        // Standing watch WITHOUT a formation claim (Blob
                        // mobs, rallied remnants): face the enemy mass
                        // instead of keeping a stale yaw.
                        None if !routed
                            && new_v.length_squared() < STEP_FACE_SPEED * STEP_FACE_SPEED
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
                    // A staggered man cannot even turn — the stun freezes
                    // his facing, so a charge's second blow finds the same
                    // back the first one hit.
                    if !dying
                        && sw_chunk[j] & crate::units::SWING_STAGGERED == 0
                        && face_dir.length_squared() > min_len2
                    {
                        let target_yaw = face_dir.x.atan2(face_dir.y);
                        let diff = (target_yaw - yaw_chunk[j] + std::f32::consts::PI)
                            .rem_euclid(std::f32::consts::TAU)
                            - std::f32::consts::PI;
                        let step = diff * (YAW_RATE * dt).min(1.0);
                        yaw_chunk[j] += step.clamp(-TURN_SPEED_MAX * dt, TURN_SPEED_MAX * dt);
                    }

                    v_chunk[j] = Vec3::new(new_v.x, 0.0, new_v.y);
                    let mut nx = (pos_prev[i].x + new_v.x * dt + corr.x)
                        .clamp(bounds_min.x, bounds_max.x);
                    let mut nz = (pos_prev[i].z + new_v.y * dt + corr.y)
                        .clamp(bounds_min.y, bounds_max.y);
                    // Impassable ground (terrace risers, gorge walls,
                    // crater lips): wall-slide — keep the axis that
                    // stays on walkable ground, drop the one that
                    // doesn't, so crowds flow along the obstacle.
                    if terrain.blocked_at(nx, nz) {
                        let (px, pz) = (pos_prev[i].x, pos_prev[i].z);
                        if !terrain.blocked_at(nx, pz) {
                            nz = pz;
                        } else if !terrain.blocked_at(px, nz) {
                            nx = px;
                        } else {
                            nx = px;
                            nz = pz;
                        }
                    }
                    p_chunk[j] = Vec3::new(
                        nx,
                        terrain.height_at(nx, nz) + crate::unit_types::half_height(kind[i] as usize),
                        nz,
                    );
                }
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

