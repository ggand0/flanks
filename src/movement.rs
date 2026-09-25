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
use crate::unit_types::{BASE_DMG, FACTOR_CLAMP, FACTOR_MULT, TYPES};
use crate::units::Units;

const SEP_RADIUS: f32 = 1.4;
/// FL_RECTFIGHT: rest distance for CROSS-TEAM pairs. Currently equal
/// to SEP_RADIUS (same spacing as same-team). Lower values let enemy
/// bodies enter the formation grid's gaps and produce a multi-rank
/// fighting band; play-testing 0.95–1.25 found the visual clipping
/// unacceptable with the current model scale, so it sits at parade
/// spacing for now. Tuning target for when model proportions are
/// revisited.
const ENEMY_SEP_RADIUS: f32 = 1.4;
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
    *P.get_or_init(|| crate::util::env_or("FL_JOIN_REACT", 0.35_f32).clamp(0.0, 1.0))
}
/// Ground speed toward the fight that reads as "going to it".
const GOING_SPEED: f32 = 1.0;

/// Sight radius for a man of a fighting regiment looking for an enemy
/// to go to (FL_SEEK_R, meters). A play-testing knob.
fn seek_radius() -> f32 {
    static R: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *R.get_or_init(|| crate::util::env_or("FL_SEEK_R", 15.0))
}
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
/// Shieldwall: locked shields count double where the shield covers
/// (front + left — a shieldwall still has no back), at some offense.
const SHIELDWALL_SHIELD_PTS: f32 = 2.0;
/// Shieldwall trades attack points for the cover (arms stay behind shields).
const SHIELDWALL_ATK_PTS: f32 = -1.0;
/// Spearwall: set spears strike harder (the wall of points does the work).
const SPEARWALL_ATK_PTS: f32 = 2.0;
/// Sector test at damage apply: cos(60 deg). Front = attack direction
/// within 60 deg of the victim's facing; rear = beyond 120 deg; else side.
/// Shared with the arrow-impact resolution (arrows.rs).
pub const SECTOR_COS_60: f32 = 0.5;
/// Charge impact (phase B). A charge-flagged hit shoves its victim along
/// the blow: displacement = KNOCKBACK * m_a/(m_a+m_v), applied straight
/// to position (next tick's separation resolves the pile — same pipe as
/// all overlap). Heavy into light ~0.55 m: the line visibly DENTS.
const CHARGE_KNOCKBACK: f32 = 0.9;
/// Braced walls barely budge (0.25x knockback) and never stagger.
const WALL_KNOCKBACK_RESIST: f32 = 0.25;
/// Stagger: a charge-flagged hit cancels the victim's swing and locks
/// him (no acting, steering, or turning) for ~1 s — the M2TW hit-stun
/// that lets a charge land a second blow before the answer. Pub: the
/// render sync derives the rocked-back pose progress from swing_t/this.
pub const STAGGER_TICKS: u8 = 30;
/// Per-hit stagger (M2TW: a hit that connects, is not blocked, and does
/// not kill staggers its victim — and blocking IS the defence stats
/// working). The chance rides the combat factor, the continuous "did
/// the defence stop it": even matchup ~0.65, rear hits ~1.0 (nobody
/// parries what he cannot see — skill and shield are already zero from
/// behind), elite shieldwall frontal ~0.3 (they block and parry a lot),
/// and AP-vs-armour staggers more purely through the halved armour.
/// Charge impacts and impalements stay certain (physics, not a parry
/// contest). A regular hit is a brief stumble; impacts stun full-length
/// (the render pose scales with remaining ticks, so stumbles read
/// lighter than impacts for free).
// Shared with the arrow-impact resolution (arrows.rs): a nonfatal hit
// rolls the same stumble regardless of what delivered it.
pub const STAGGER_P0: f32 = 0.65;
pub const STAGGER_P_PER_FACTOR: f32 = 0.05;
pub const STAGGER_P_MIN: f32 = 0.05;
/// Pub: the render normalizes the pose by the STUMBLE length, so a
/// stumble plays the full-strength rock for 0.5 s and an impact holds
/// it ~1 s — hierarchy by duration, not amplitude.
pub const HIT_STAGGER_TICKS: u8 = 15;

/// FL_DEBUG_STAGGER=1: every landed hit staggers its victim (animation
/// debugging — normally the roll above decides).
fn debug_stagger() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FL_DEBUG_STAGGER").is_ok())
}

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

/// A landed swing, resolved after the parallel integrate: chunks emit into
/// their own buffer (no write races), then a serial pass applies damage.
pub struct DamageEvent {
    pub victim: u32,
    pub attacker: u32,
    /// Per-swing damage jitter (0.85-1.15); everything else — stats,
    /// attack sector, wall stances, the charge bonus — resolves in the
    /// apply pass, where both sides' state is known (a braced spearwall
    /// cancels the charge; the victim's yaw decides front/side/rear).
    pub jit: f32,
    /// Swing started at charging speed (momentum bonus).
    pub charge: bool,
    /// Not a swing at all: the victim ran onto a braced spear at charge
    /// speed (attacker is the spearman). His own charge bonus counts as
    /// attack points against him, and the point stops him (stagger, no
    /// knockback — momentum went INTO the spear).
    pub impale: bool,
}

/// FL_DIAG_REAR: what moved each soldier this tick, as velocity-
/// equivalent terms (devlog 0119).
#[derive(Clone, Copy, Default)]
pub struct RearDiagRow {
    pub i: u32,
    /// bit0 idle (Ready, no enemy in reach), bit1 memo surge, bit2 near
    /// surge, bit3 holding (no order), bit4 a friend's body is in the
    /// way he is driving.
    pub flags: u8,
    pub slot: Vec2,
    pub surge: Vec2,
    pub push: Vec2,
    pub corr: Vec2,
    pub goal_d: f32,
}

pub fn diag_rear() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FL_DIAG_REAR").is_ok())
}

/// FL_DIAG_REAR state: the rows the last tick emitted, each soldier's
/// previous displacement (for direction reversals) and the per-bucket
/// sums of the current log window.
#[derive(Resource, Default)]
pub struct RearDiag {
    pub rows: Vec<Vec<RearDiagRow>>,
    prev_disp: Vec<Vec2>,
    acc: std::collections::BTreeMap<(u8, usize), [f64; 19]>,
    next_log: u32,
    step_sum: f64,
    grid_sum: f64,
    step_n: u32,
}

/// One event buffer per integrate chunk; allocations persist across ticks.
#[derive(Resource, Default)]
pub struct DamageBuffers(pub Vec<Vec<DamageEvent>>);

/// FL_TEST_DIR / FL_ARENA bookkeeping: every hit bucketed by victim team
/// and attack sector, filled by the damage apply pass, logged from
/// regiments.rs (the dir test reads blue victims, the arena reads orange
/// victims — the player's kills).
#[derive(Resource)]
pub struct DirTestStats {
    /// Indexed [victim team][front / side / rear].
    pub kills: [[u64; 3]; 2],
    pub hits: [[u64; 3]; 2],
    pub dmg: [[f64; 3]; 2],
    pub enabled: bool,
}

impl Default for DirTestStats {
    fn default() -> Self {
        Self {
            kills: [[0; 3]; 2],
            hits: [[0; 3]; 2],
            dmg: [[0.0; 3]; 2],
            enabled: std::env::var("FL_TEST_DIR").is_ok() || std::env::var("FL_ARENA").is_ok(),
        }
    }
}

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
            .init_resource::<DirTestStats>()
            .init_resource::<RearDiag>()
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
struct ShootAt {
    c: Vec2,
    vel: Vec2,
    r: f32,
    /// Target regiment (index into `Groups::list`): the per-shot
    /// aim picks one of its living soldiers.
    t: usize,
}

/// Everything one kinematic tick owns: copies of the input state, the
/// per-regiment command snapshot, and the output buffers. Fully
/// self-contained, so the tick can run on a worker thread while the main
/// world keeps rendering the last completed tick — nothing borrows ECS
/// data across frames. Per-soldier buffers are recycled tick to tick
/// (swapped at install, clear + extend at prep).
#[derive(Default)]
pub struct TickJob {
    // Tick-start positions (the kernel's `pos_prev`) and the new ones.
    pos_in: Vec<Vec3>,
    pos_out: Vec<Vec3>,
    // Read-only column copies (the death sweep swap-removes the live
    // columns between ticks, so the job cannot share them).
    speed: Vec<f32>,
    team: Vec<u8>,
    kind: Vec<u8>,
    group: Vec<u32>,
    home: Vec<Vec2>,
    // Read-modify-write columns: copied in at prep, swapped out at install.
    vel: Vec<Vec3>,
    yaw: Vec<f32>,
    yaw_prev: Vec<f32>,
    target: Vec<u32>,
    swing: Vec<u8>,
    swing_t: Vec<u8>,
    flash: Vec<u8>,
    death_t: Vec<u8>,
    ammo: Vec<u8>,
    // Per-regiment command snapshot, taken at prep.
    orders: Vec<Option<Vec2>>,
    anchors: Vec<Vec2>,
    reg_broken: Vec<bool>,
    reg_mover: Vec<bool>,
    press: Vec<bool>,
    engaged: Vec<bool>,
    contact: Vec<bool>,
    fight_point: Vec<Option<Vec2>>,
    melee_ticks: Vec<u32>,
    hold: Vec<bool>,
    threat: Vec<Vec2>,
    form_face: Vec<Vec2>,
    wall: Vec<u8>,
    charging: Vec<bool>,
    fat_speed: Vec<f32>,
    fat_nocharge: Vec<bool>,
    shoot_at: Vec<Option<ShootAt>>,
    target_members: Vec<Vec<u32>>,
    blocks: [Vec<(Vec2, f32, f32)>; 2],
    faces_spearwall: [bool; 2],
    // Per-soldier prep products.
    wall_flags: Vec<bool>,
    broken_flags: Vec<bool>,
    mover_flags: Vec<bool>,
    yaw_snapshot: Vec<f32>,
    /// Tick-start ground velocities, copied only while some regiment is
    /// in melee: a comrade walking away the same way is no obstacle.
    vel_snapshot: Vec<Vec2>,
    // Outputs beyond the columns: the grid the tick built, landed swings
    // and loosed arrows per chunk. All swapped into their resources at
    // install.
    grid: SpatialGrid,
    events: Vec<Vec<DamageEvent>>,
    arrow_spawns: Vec<Vec<crate::arrows::ArrowSpawn>>,
    diag: Vec<Vec<RearDiagRow>>,
    terrain: Option<std::sync::Arc<Terrain>>,
    dt: f32,
    combat_scale: f32,
    tick_seed: u32,
    tick: u32,
    rf: bool,
    bounds_min: Vec2,
    bounds_max: Vec2,
    /// `Units::generation` at prep: a job from a dead world is dropped.
    generation: u64,
    grid_ms: f32,
    step_ms: f32,
}

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

/// Fill `job` from the live world: column copies, the per-regiment
/// command snapshot, the archers' fire solutions and scalars. This is
/// the serial prep that used to open `step_sim`. It also makes the two
/// regiment writes that prep always made: the stand-off anchor snap and
/// the `firing` flag.
#[allow(clippy::too_many_arguments)]
fn prepare_tick(
    job: &mut TickJob,
    units: &Units,
    groups: &mut Groups,
    tracks: &crate::arrows::RegTracks,
    terrain_arc: &std::sync::Arc<Terrain>,
    dt: f32,
    combat_scale: f32,
    tick: u32,
) {
    job.generation = units.generation;
    job.dt = dt;
    job.combat_scale = combat_scale;
    job.terrain = Some(terrain_arc.clone());
    let terrain: &Terrain = terrain_arc;

    macro_rules! copy_col {
        ($($dst:ident <- $src:ident),*) => {$(
            job.$dst.clear();
            job.$dst.extend_from_slice(&units.$src);
        )*};
    }
    // pos_in is the state at tick start; pos_out is fully rewritten by
    // the integrate.
    copy_col!(
        pos_in <- pos, speed <- speed, team <- team, kind <- kind, group <- group,
        home <- home, vel <- vel, yaw <- yaw, yaw_prev <- yaw_prev, target <- target,
        swing <- swing, swing_t <- swing_t, flash <- flash, death_t <- death_t, ammo <- ammo
    );
    job.pos_out.clear();
    job.pos_out.resize(units.pos.len(), Vec3::ZERO);
    let pos_prev = &units.pos[..];
    let group = &units.group[..];
    let death_t = &units.death_t[..];

    // Per-unit wall flag for the grid meta (same-team wall pairs pack
    // tighter in the separation below).
    let group_wall: Vec<bool> = groups
        .list
        .iter()
        .map(|g| crate::formation::wall_kind(g) != 0)
        .collect();
    job.wall_flags.clear();
    job.wall_flags.extend(group.iter().map(|&g| group_wall[g as usize]));
    // Pass-through flags for the grid meta (FL_RECTFIGHT): a BROKEN
    // man is a fleeing body, and a man of a MOVE-ordered regiment is a
    // body deliberately passing through (the engine's explicit
    // formationMovingThrough state) — both collide at body scale
    // instead of commanding the 1.4 m rank-dressing courtesy, so a
    // rout or an ordered withdrawal slips THROUGH a formed line's
    // seams instead of excavating a corridor through the formation.
    let group_broken: Vec<bool> = groups.list.iter().map(|g| g.state.is_broken()).collect();
    let group_mover: Vec<bool> = groups
        .list
        .iter()
        .map(|g| !g.state.is_broken() && matches!(g.order, Some(crate::orders::Order::Move(_))))
        .collect();
    job.broken_flags.clear();
    job.broken_flags.extend(group.iter().map(|&g| group_broken[g as usize]));
    job.mover_flags.clear();
    job.mover_flags.extend(group.iter().map(|&g| group_mover[g as usize]));
    // No packing rule for fighting regiments: vanilla M2TW keeps its
    // formation grid (and observably LOOSENS it) during melee — the
    // fighting crowd's spacing is slots + body collision, nothing
    // else. A shoulder-to-shoulder press rest was tried here (bit 5,
    // META_PRESS) and it sealed the very seams the intermix needs:
    // a pressed front's gaps shrank to ~1.05 m against 0.95 m bodies
    // and symmetric fights collapsed to a two-rank duel line.
    let rf = crate::formation::rectfight();

    // ---- Archer fire solutions (regiment level). An attack order for a
    // ranged regiment is a FIRE order, not a melee charge: the regiment
    // halts once the target is inside range (the stand-off below feeds
    // the order resolution) and volleys from where it stands; the march
    // resumes if the target slips back out of range.
    let n_groups = groups.list.len();
    let standoff: Vec<bool> = groups
        .list
        .iter()
        .map(|gd| {
            gd.kind == crate::unit_types::KIND_ARCHER
                && gd.count > 0
                && !gd.state.is_broken()
                && matches!(gd.order, Some(crate::orders::Order::Attack(t))
                    if groups.list[t as usize].count > 0
                        && groups.list[t as usize].centroid.distance(gd.centroid)
                            < crate::unit_types::missile::RANGE * 0.85)
        })
        .collect();
    for (g, &so) in standoff.iter().enumerate() {
        // Entering the stand-off freezes the frame here (the same anchor
        // snap engagement does): holding units dress where they stopped
        // instead of walking back to a stale anchor.
        if so && groups.list[g].anchor.distance(groups.list[g].centroid) > 2.0 {
            let c = groups.list[g].centroid;
            groups.list[g].anchor = c;
        }
    }
    // What each archer regiment's bows shoot at this tick: the ordered
    // target once standing off, else fire-at-will's nearest live enemy
    // regiment in range. Regiments in melee or on the march don't volley
    // (M2TW: foot archers halt to shoot; fire-at-will pauses on the move).
    let shoot_at: Vec<Option<ShootAt>> = (0..n_groups)
        .map(|g| {
            let gd = &groups.list[g];
            if gd.kind != crate::unit_types::KIND_ARCHER
                || gd.count == 0
                || gd.state.is_broken()
                || gd.engaged
            {
                return None;
            }
            let t = match gd.order {
                Some(crate::orders::Order::Move(_)) => return None,
                Some(crate::orders::Order::Attack(t)) => {
                    // An explicit target overrides fire-at-will: no
                    // shots until the march brings it into range.
                    if !standoff[g] || groups.list[t as usize].count == 0 {
                        return None;
                    }
                    t as usize
                }
                None => {
                    // Fire-at-will: nearest live, unbroken enemy block
                    // that is NOT locked in melee. In M2TW fire-at-will
                    // holds instead of volleying into a fight with
                    // friends standing in it; an explicit target order
                    // (the Attack arm above) still forces the shot.
                    // The flight stays honest either way: no team check
                    // on what the shaft actually crosses.
                    if !gd.fire_at_will {
                        return None;
                    }
                    let mut best: Option<(f32, usize)> = None;
                    for (e, eg) in groups.list.iter().enumerate() {
                        if eg.team == gd.team
                            || eg.count == 0
                            || eg.state.is_broken()
                            || eg.engaged
                        {
                            continue;
                        }
                        let d2 = eg.centroid.distance_squared(gd.centroid);
                        if best.is_none_or(|(bd, _)| d2 < bd) {
                            best = Some((d2, e));
                        }
                    }
                    let (d2, e) = best?;
                    let range = crate::unit_types::missile::RANGE;
                    if d2 > range * range {
                        return None;
                    }
                    e
                }
            };
            let tg = &groups.list[t];
            Some(ShootAt {
                c: tg.centroid,
                vel: tracks.vel.get(t).copied().unwrap_or(Vec2::ZERO),
                r: tg.radius.clamp(3.0, 25.0),
                t,
            })
        })
        .collect();
    // The card's firing indicator: a fire solution exists and there are
    // arrows to spend on it.
    for (g, s) in shoot_at.iter().enumerate() {
        groups.list[g].firing = s.is_some() && groups.list[g].ammo_left > 0;
    }
    // Living members of every regiment under fire this tick: each shot
    // aims at an actual soldier (M2TW's per-soldier aim targets,
    // devlog 0060), not at a spot on the block's footprint. Into a
    // locked melee the shafts head for enemy bodies; friends die only
    // to genuine misses and interceptions.
    let mut targeted = vec![false; n_groups];
    for s in shoot_at.iter().flatten() {
        targeted[s.t] = true;
    }
    let mut target_members: Vec<Vec<u32>> = vec![Vec::new(); n_groups];
    if targeted.iter().any(|t| *t) {
        for i in 0..pos_prev.len() {
            if death_t[i] == 0 && targeted[group[i] as usize] {
                target_members[group[i] as usize].push(i as u32);
            }
        }
    }
    // Friendly blocks per team for the loft-over-friendlies check: every
    // live formed regiment as a disc with a clearance ceiling. A
    // shooter's own regiment is in here too — that is what makes rear
    // ranks loft over their own front rank.
    let blocks: [Vec<(Vec2, f32, f32)>; 2] = [0u8, 1u8].map(|tm| {
        groups
            .list
            .iter()
            .filter(|g| g.team == tm && g.count > 0 && !g.state.is_broken())
            .map(|g| {
                (
                    g.centroid,
                    g.radius.max(3.0),
                    terrain.height_at(g.centroid.x, g.centroid.y) + 2.4,
                )
            })
            .collect()
    });

    // Orders resolved to this tick's destination (attack orders chase
    // their target regiment's current centroid). FL_RECTFIGHT: an
    // ENGAGED attack order stops chasing — the frame froze where
    // contact happened (anchor snap, update_groups) and the men fight
    // individually from there; dragging the slot grid onward through
    // a moving enemy centroid is what smeared blocks into blobs. A
    // BROKEN target keeps the chase alive: pursuit is a hunt, not a
    // fight. Move orders never freeze — pulling a regiment out of
    // melee is the player's call.
    let orders: Vec<Option<Vec2>> = (0..groups.list.len())
        .map(|g| {
            let gd = &groups.list[g];
            // Archer stand-off: in range of the ordered target — stand
            // and shoot (the order survives; the chase resumes if the
            // target moves out of range).
            if standoff[g] {
                return None;
            }
            // The freeze keys on engagement WITH the ordered target
            // (count-gated, frontline.rs) — incidental duels against
            // an overflow trickle never halt the march.
            if rf
                && gd.engaged_with_target
                && let Some(crate::orders::Order::Attack(t)) = gd.order
                && !groups.list[t as usize].state.is_broken()
            {
                return None;
            }
            // Contact frame (frontline.rs): the regiment holds its slots
            // around the frame instead of chasing the target's center.
            if gd.contact {
                return None;
            }
            let mut goal = groups.goal(g);
            // Friendly formed blocks are SOLID to a marching attack
            // (engine: isBlocked/blockedBy — a unit's path routes
            // AROUND a friendly, never through his ranks). The
            // coarsest faithful version: when the straight approach
            // crosses an ally's footprint, aim at a waypoint off that
            // ally's flank; recomputed from live positions each tick,
            // the march swings smoothly around and re-aims at the
            // target beyond.
            if rf
                && !gd.state.is_broken()
                && matches!(gd.order, Some(crate::orders::Order::Attack(_)))
                && let Some(gpos) = goal
            {
                let a = gd.centroid;
                let ab = gpos - a;
                let len2 = ab.length_squared();
                if len2 > 1.0 {
                    let mut hit: Option<(f32, usize, f32)> = None; // (s, ally, radius)
                    for (u, ud) in groups.list.iter().enumerate() {
                        if u == g
                            || ud.team != gd.team
                            || ud.count == 0
                            || ud.shape != crate::formation::FormShape::Rect
                            || ud.state.is_broken()
                        {
                            continue;
                        }
                        let r = (ud.count as f32).sqrt() * crate::formation::BASE_SPACING * 0.55
                            + 1.5;
                        let s = ((ud.centroid - a).dot(ab) / len2).clamp(0.0, 1.0);
                        if s <= 0.02 || s >= 0.98 {
                            continue;
                        }
                        if (a + ab * s).distance_squared(ud.centroid) < r * r
                            && hit.is_none_or(|(hs, ..)| s < hs)
                        {
                            hit = Some((s, u, r));
                        }
                    }
                    if let Some((s, u, r)) = hit {
                        let c = groups.list[u].centroid;
                        let mut side = (a + ab * s - c).normalize_or_zero();
                        if side == Vec2::ZERO {
                            // Path through the ally's center: pick a
                            // flank deterministically per regiment.
                            let d = ab / len2.sqrt();
                            side = Vec2::new(-d.y, d.x) * if g % 2 == 0 { 1.0 } else { -1.0 };
                        }
                        goal = Some(c + side * (r + 2.0));
                    }
                }
            }
            goal
        })
        .collect();
    let anchors: Vec<Vec2> = groups.list.iter().map(|g| g.anchor).collect();
    // Regiments in combat-watch range of an enemy (sparse-fight
    // acquisition): order type is irrelevant — a Move-order fight that
    // went sparse stalls exactly the same way. HOLD regiments never
    // acquire wide: they stand their ground and take what comes.
    let press: Vec<bool> = groups
        .list
        .iter()
        .map(|g| (g.enemy_near || g.engaged) && !g.hold)
        .collect();
    // Regiments in melee: their shoved men wait for room instead of
    // threading back to their marks (a dressing block still threads).
    let engaged: Vec<bool> = groups.list.iter().map(|g| g.engaged).collect();
    // Regiments holding a contact frame (frontline.rs): their fight is
    // ahead of them, and their men never step back to dress.
    let contact: Vec<bool> = groups.list.iter().map(|g| g.contact).collect();
    // Where each regiment in melee has its fight, and for how long.
    let fight_point: Vec<Option<Vec2>> = groups.list.iter().map(|g| g.fight_point).collect();
    let melee_ticks: Vec<u32> = groups.list.iter().map(|g| g.melee_ticks).collect();

    // Hold-position leash: units of a held regiment close only the last
    // step to a swing (no chasing across open ground).
    let hold: Vec<bool> = groups.list.iter().map(|g| g.hold).collect();
    // Direction to the nearest enemy regiment (brace facing).
    let threat: Vec<Vec2> = groups.list.iter().map(|g| g.threat_dir).collect();
    // Formation facing for standing units (ZERO = no claim): a formed
    // regiment DRESSES to its ordered facing when nothing is nearer to
    // worry about — without this, units kept the yaw of their last
    // march step and a drag-ordered line stood looking sideways.
    let form_face: Vec<Vec2> = groups
        .list
        .iter()
        .map(|g| {
            if g.shape == crate::formation::FormShape::Rect && !g.state.is_broken() {
                crate::formation::facing_dir(g.facing)
            } else {
                Vec2::ZERO
            }
        })
        .collect();
    // Wall stance per regiment (0 none / 1 shieldwall / 2 spearwall):
    // slower advance, damage model tweaks in the events + apply pass.
    let wall: Vec<u8> = groups.list.iter().map(crate::formation::wall_kind).collect();
    // Charge phase: the run home (speed boost feeds the per-unit
    // SWING_CHARGE predicate too — momentum the sim can see).
    let charging: Vec<bool> = groups.list.iter().map(|g| g.charging).collect();
    // Fatigue locomotion: tired legs are slow legs, and exhausted
    // regiments cannot sprint the charge home (MTW1 "cannot run or
    // charge"; fleeing men tire too — pursuit catches them).
    let fat_speed: Vec<f32> = groups
        .list
        .iter()
        .map(|g| crate::fatigue::speed_mult(g.fatigue))
        .collect();
    let fat_nocharge: Vec<bool> = groups
        .list
        .iter()
        .map(|g| crate::fatigue::cannot_charge(g.fatigue))
        .collect();
    let bounds_min = terrain.min() + 4.0;
    let bounds_max = terrain.max() - 4.0;

    // Spear-line hazard global gate: marching speed exceeds the charge
    // threshold, so without this every mover in a 200k battle would pay
    // the wider scan hunting for spearwalls that do not exist. Per team:
    // does the ENEMY side field a braced spearwall at all this tick?
    let faces_spearwall: [bool; 2] = [0u8, 1u8].map(|t| {
        groups
            .list
            .iter()
            .any(|g| g.team != t && g.count > 0 && crate::formation::wall_kind(g) == 2)
    });
    // Read-only yaw snapshot (tick-start, like pos_prev) for cross-unit
    // reads inside the parallel integrate — the live yaw column is split
    // into &mut chunks there. The spear-line hazard reads the SPEARMAN's
    // facing from the charger's side of the scan; no spearwalls anywhere,
    // no copy.
    job.vel_snapshot.clear();
    if groups.list.iter().any(|g| g.fight_point.is_some()) {
        job.vel_snapshot.extend(units.vel.iter().map(|v| v.xz()));
    }
    job.yaw_snapshot.clear();
    if faces_spearwall[0] || faces_spearwall[1] {
        job.yaw_snapshot.extend_from_slice(&units.yaw);
    }

    let n_chunks = units.pos.len().div_ceil(CHUNK);
    if job.events.len() < n_chunks {
        job.events.resize_with(n_chunks, Vec::new);
    }
    if job.arrow_spawns.len() < n_chunks {
        job.arrow_spawns.resize_with(n_chunks, Vec::new);
    }
    if job.diag.len() < n_chunks {
        job.diag.resize_with(n_chunks, Vec::new);
    }
    // Spawn buffers come back drained from arrows.rs. A job dropped as
    // stale never got that far.
    for buf in &mut job.arrow_spawns {
        buf.clear();
    }
    job.tick_seed = tick.wrapping_mul(0x9E37_79B1);
    job.tick = tick;
    job.rf = rf;
    job.bounds_min = bounds_min;
    job.bounds_max = bounds_max;
    job.faces_spearwall = faces_spearwall;
    job.orders = orders;
    job.anchors = anchors;
    job.reg_broken = group_broken;
    job.reg_mover = group_mover;
    job.press = press;
    job.engaged = engaged;
    job.contact = contact;
    job.fight_point = fight_point;
    job.melee_ticks = melee_ticks;
    job.hold = hold;
    job.threat = threat;
    job.form_face = form_face;
    job.wall = wall;
    job.charging = charging;
    job.fat_speed = fat_speed;
    job.fat_nocharge = fat_nocharge;
    job.shoot_at = shoot_at;
    job.target_members = target_members;
    job.blocks = blocks;
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
    let rf = job.rf;
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
        speed,
        team,
        kind,
        group,
        home,
        orders,
        anchors,
        reg_broken,
        reg_mover,
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
        broken_flags,
        mover_flags,
        yaw_snapshot,
        vel_snapshot,
        grid,
        events,
        arrow_spawns,
        diag,
        grid_ms,
        step_ms,
        ..
    } = job;

    let t0 = Instant::now();
    {
        let _span = info_span!("grid_rebuild").entered();
        grid.rebuild(pos_in, team, kind, death_t, wall_flags, broken_flags, mover_flags);
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
    let vel_snap = &vel_snapshot[..];
    let orders = &orders[..];
    let anchors = &anchors[..];
    let broken = &reg_broken[..];
    let group_mover = &reg_mover[..];
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
            .zip(diag.iter_mut())
            .enumerate()
        {
            let ((((((((((((p_chunk, v_chunk), yaw_chunk), yawp_chunk), tgt_chunk), sw_chunk),
                swt_chunk), fl_chunk), dt_chunk), events), ammo_chunk), arrow_out), diag_out) =
                chunk;
            let start = ci * CHUNK;
            scope.spawn(async move {
                events.clear();
                diag_out.clear();
                let diag_on = diag_rear();
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
                    let mut d_goal = 0.0f32;
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
                        d_goal = dist;
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
                    // Direction to the enemy remembered last tick: a
                    // comrade's body standing in that direction blocks the
                    // surge toward him (below).
                    let memo = prev_target as usize;
                    let memo_valid = memo < pos_prev.len()
                        && team[memo] != team[i]
                        && pos_prev[memo].xz().distance_squared(p)
                            < (seek_radius() + 1.0) * (seek_radius() + 1.0);
                    // Joining the fight: his regiment has been in melee
                    // longer than his own delay, and he sees no enemy.
                    // Joining the fight: his regiment is in melee and he
                    // sees no enemy. He goes if he is already on his way,
                    // or (after the scan) when he sees a comrade beside him
                    // running to the fight.
                    let join_fp = match fight_point[gi] {
                        Some(fp) if !memo_valid && !routed && !dying && melee_ticks[gi] > 0 => {
                            Some(fp)
                        }
                        _ => None,
                    };
                    let already_going = join_fp.is_some_and(|fp| {
                        v_chunk[j].xz().dot((fp - p).normalize_or_zero()) > GOING_SPEED
                    });
                    let mut join_to = if already_going { join_fp } else { None };
                    let memo_dir = if memo_valid {
                        (pos_prev[memo].xz() - p).normalize_or_zero()
                    } else if let Some(fp) = join_fp {
                        (fp - p).normalize_or_zero()
                    } else {
                        Vec2::ZERO
                    };
                    let mut way_blocked = false;
                    // The same test toward a holding man's own mark.
                    let slot_dir = if orders[gi].is_none() && engaged[gi] && !routed {
                        desired.normalize_or_zero()
                    } else {
                        Vec2::ZERO
                    };
                    let mut slot_blocked = false;
                    // Open lanes to either side of the way to that enemy,
                    // for a sidestep when the way itself is blocked.
                    let side_dir = Vec2::new(-memo_dir.y, memo_dir.x);
                    let mut left_blocked = false;
                    let mut right_blocked = false;
                    // A comrade anywhere ahead within the scan: he walks
                    // up to him instead of jogging.
                    let mut comrade_ahead = false;
                    // A comrade of his own regiment close by, running to
                    // the fight: the sight that makes him follow.
                    let mut saw_comrade_go = false;
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
                        if !cross
                            && memo_dir != Vec2::ZERO
                            && d2 < SEP_RADIUS * SEP_RADIUS
                            && (-d).dot(memo_dir) > 0.707 * d2.sqrt()
                            && !((o.idx as usize) < vel_snap.len()
                                && vel_snap[o.idx as usize].dot(memo_dir) > 0.5)
                        {
                            way_blocked = true;
                        }
                        if join_fp.is_some()
                            && !already_going
                            && !cross
                            && group[o.idx as usize] as usize == gi
                            && (o.idx as usize) < vel_snap.len()
                            && vel_snap[o.idx as usize].dot(memo_dir) > GOING_SPEED
                        {
                            saw_comrade_go = true;
                        }
                        // A comrade already walking away the same way is
                        // no obstacle: men heading for the same fight move
                        // together instead of each waiting for the next.
                        let walking_away = !cross
                            && memo_dir != Vec2::ZERO
                            && (o.idx as usize) < vel_snap.len()
                            && vel_snap[o.idx as usize].dot(memo_dir) > 0.5;
                        if !cross
                            && !walking_away
                            && memo_dir != Vec2::ZERO
                            && (-d).dot(memo_dir) > 0.707 * d2.sqrt()
                        {
                            comrade_ahead = true;
                        }
                        if !cross && memo_dir != Vec2::ZERO && d2 < SEP_RADIUS * SEP_RADIUS {
                            let lateral = (-d).dot(side_dir);
                            if lateral > 0.707 * d2.sqrt() {
                                left_blocked = true;
                            } else if lateral < -0.707 * d2.sqrt() {
                                right_blocked = true;
                            }
                        }
                        if !cross
                            && slot_dir != Vec2::ZERO
                            && d2 < SEP_RADIUS * SEP_RADIUS
                            && (-d).dot(slot_dir) > 0.707 * d2.sqrt()
                        {
                            slot_blocked = true;
                        }
                        // Pass-through pairs: a fleeing body, or a
                        // Move-ordered regiment walking through the
                        // lines (the engine's explicit
                        // formationMovingThrough state), collide at
                        // body scale — the passer slips through a
                        // formed line's seams; the formation is not
                        // excavated into a 1.4 m corridor.
                        let pass_pair = rf
                            && !cross
                            && (routed
                                || group_mover[gi]
                                || (o.meta
                                    & (crate::spatial::META_BROKEN
                                        | crate::spatial::META_MOVER))
                                    != 0);
                        let sep_r = if cross {
                            // Enemy bodies rest at body contact (see
                            // ENEMY_SEP_RADIUS): contact distance is a
                            // property of bodies, not of allegiance —
                            // the wide courtesy gap belongs to ranks
                            // dressing on the same side only.
                            if rf { ENEMY_SEP_RADIUS } else { SEP_RADIUS }
                        } else if pass_pair {
                            ENEMY_SEP_RADIUS
                        } else if my_wall && (o.meta & crate::spatial::META_WALL) != 0 {
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
                            // yield-and-stay-ragged defect). Gated, the
                            // weight reads on the one body scale
                            // regardless of the pair's rest radius —
                            // per-rest weights sum too small at tight
                            // radii to ever brake (the twitch). EXCEPT
                            // pass-through pairs: a body threading the
                            // seams at its intended body distance is
                            // not a wedged crowd — weighing it on the
                            // 1.4 scale made a passer crawl at a
                            // quarter speed and faded the standing
                            // line's slot-keeping (measured, ROUTPASS).
                            // Ungated all formulas are identical.
                            crowd += if rf && !pass_pair {
                                1.0 - len / SEP_RADIUS
                            } else {
                                w
                            };
                        }
                    });

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
                    if join_to.is_none() && saw_comrade_go {
                        let window = (tick.wrapping_add((i as u32).wrapping_mul(11)) / JOIN_WINDOW)
                            .wrapping_mul(0x9E37_79B1);
                        if crate::units::hash01(window ^ (i as u32).wrapping_mul(0x6A09)) < join_react_chance() {
                            join_to = join_fp;
                        }
                    }
                    // A man committed to the fight (an enemy he is going
                    // for, or joining) no longer dresses on his slot until
                    // the fight ends and the regiment re-forms (M2TW keeps
                    // the slot but the man fights out of formation).
                    // Blocked, he stands; he never walks back to his old
                    // place. Keeping the slot pull under the seek made him
                    // loop between the fight and his slot.
                    if fight_point[gi].is_some() && (memo_valid || join_to.is_some()) {
                        desired = Vec2::ZERO;
                    }

                    // Sparse-fight acquisition (see WIDE_ACQUIRE_R): a
                    // pressing unit with an empty scan and open space
                    // around it memoizes a farther enemy in `target` so
                    // the closing drive below can restore contact. Gated
                    // hard (press + no near enemy + low crowd + 1/8
                    // cadence) to stay off the 200k hot path.
                    // FL_RECTFIGHT: acquisition stays open until the
                    // crowd genuinely jams (CROWD_STOP) instead of the
                    // first hint of density — a rank-2 man shoulder to
                    // shoulder in the press must still want in; the
                    // (1 - jam) brake on the surge below is what stops
                    // him, continuously, when the pack is real.
                    let acquire_crowd_lim = if rf { CROWD_STOP } else { CROWD_SLOW };
                    if best_idx == u32::MAX
                        && !dying
                        && !routed
                        && press[gi]
                        && crowd < acquire_crowd_lim
                        && (i as u32).wrapping_add(tick_seed).is_multiple_of(8)
                    {
                        // A man of a fighting regiment looks as far as he
                        // can see (seek_radius); on the approach the old
                        // short scan stands.
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
                    let d_slot = desired;
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
                    } else if crowd < acquire_crowd_lim {
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
                    if let Some(fp) = join_to
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
                            d_surge = desired;
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

                    if diag_on && !dying && !routed {
                        let mult = if wall[gi] != 0 {
                            WALL_SPEED_FRAC
                        } else if charging[gi] && !fat_nocharge[gi] {
                            CHARGE_SPEED_BOOST
                        } else {
                            1.0
                        } * fat_speed[gi];
                        let idle = best_idx == u32::MAX
                            && sw_chunk[j] & crate::units::SWING_STATE_MASK
                                == crate::units::SWING_READY;
                        // Is a friend's body in the way he is trying to go?
                        let want = d_slot + d_surge;
                        let mut blocked = false;
                        if want.length_squared() > 1e-6 {
                            let wd = want.normalize();
                            grid.for_each_candidate(p, SEP_RADIUS, |o| {
                                if o.idx as usize == i
                                    || (o.meta & crate::spatial::META_TEAM) != my_team_bit
                                {
                                    return;
                                }
                                let d = o.xz() - p;
                                let l = d.length();
                                if l < SEP_RADIUS && l > 1e-4 && d.dot(wd) > 0.707 * l {
                                    blocked = true;
                                }
                            });
                        }
                        let flags = idle as u8
                            | (((d_surge != Vec2::ZERO && memo_close) as u8) << 1)
                            | (((d_surge != Vec2::ZERO && !memo_close) as u8) << 2)
                            | ((orders[gi].is_none() as u8) << 3)
                            | ((blocked as u8) << 4);
                        diag_out.push(RearDiagRow {
                            i: i as u32,
                            flags,
                            slot: d_slot * mult,
                            surge: d_surge * mult,
                            push: push * (SEP_STRENGTH * (1.0 - jam) / STEER_GAIN),
                            corr: corr / dt,
                            goal_d: d_goal,
                        });
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
    mut rear: ResMut<RearDiag>,
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
            rear.step_sum += job.step_ms as f64;
            rear.grid_sum += job.grid_ms as f64;
            rear.step_n += 1;
            if rear.step_n == 150 {
                info!(
                    "[step] mean step {:.2} ms, grid {:.2} ms, {} units",
                    rear.step_sum / 150.0,
                    rear.grid_sum / 150.0,
                    units.pos.len()
                );
                rear.step_sum = 0.0;
                rear.grid_sum = 0.0;
                rear.step_n = 0;
            }
        }
    }
    if diag_rear() {
        std::mem::swap(&mut rear.rows, &mut job.diag);
        rear_diag_aggregate(&mut rear, &units, &groups, job.dt, cstats.kills, pipeline.tick);
    }

    let Units {
        pos,
        pos_prev,
        team,
        kind,
        yaw,
        group,
        hp,
        swing,
        swing_t,
        flash,
        death_t,
        ammo,
        ..
    } = &mut *units;
    let team = &team[..];
    let kind = &kind[..];
    let group = &group[..];
    let wall = &job.wall[..];
    let fat_nocharge = &job.fat_nocharge[..];
    let bounds_min = job.bounds_min;
    let bounds_max = job.bounds_max;
    let combat_scale = job.combat_scale;
    let tick_seed = job.tick_seed;
    let terrain = &*terrain;
    let grid = &*grid;
    let tick = &mut pipeline.tick;

    // Serial damage apply: deterministic (chunk order), race-free, and the
    // single place where hp transitions to death. A swing whiffs when its
    // victim died mid-wind-up, changed team slot via swap-remove, or slipped
    // out of reach — checked here, where all columns are whole again.
    {
        let _span = info_span!("damage_apply").entered();
        stats.events = 0;
        // Fatigue stat effects (MTW1 table): weary arms strike with
        // fewer attack points; exhausted men get no charge bonus at all.
        let fat_atk: Vec<f32> = groups
            .list
            .iter()
            .map(|g| crate::fatigue::attack_penalty(g.fatigue))
            .collect();
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
                let pa = &TYPES[kind[a] as usize];
                let reach = pa.reach * 1.15;
                if pos_prev[v].xz().distance_squared(pos_prev[a].xz()) > reach * reach {
                    continue;
                }
                // M2TW directional resolution. Sector of the blow against
                // the victim's facing: defence skill counts vs front and
                // side but NOT rear; the shield covers front + LEFT side
                // only (shield arm); armour counts from everywhere, halved
                // by armour-piercing weapons. A soldier struck from behind
                // loses skill+shield entirely — facing is the defensive
                // resource, and rear charges kill before morale even moves.
                let pv = &TYPES[kind[v] as usize];
                let fwd = Vec2::new(yaw[v].sin(), yaw[v].cos());
                let dir = (pos_prev[a].xz() - pos_prev[v].xz()).normalize_or_zero();
                let along = fwd.dot(dir);
                let rear = along < -SECTOR_COS_60;
                let front = along >= SECTOR_COS_60;
                // Victim's left in the xz plane (facing +Z -> left = +X).
                let left_side = !front && !rear && dir.dot(Vec2::new(fwd.y, -fwd.x)) > 0.0;

                // Wall stances are stat points now: a shieldwall doubles
                // down on cover (only where the shield counts — it still
                // has no back), a braced spearwall strikes harder, and a
                // braced spearwall VICTIM nullifies the attacker's charge
                // momentum (points beat momentum).
                let a_wall = wall[group[a] as usize];
                let v_wall = wall[group[v] as usize];
                let mut attack = pa.attack
                    + fat_atk[group[a] as usize]
                    + match a_wall {
                        1 => SHIELDWALL_ATK_PTS,
                        2 => SPEARWALL_ATK_PTS,
                        _ => 0.0,
                    };
                if ev.charge && !fat_nocharge[group[a] as usize] {
                    attack += pa.charge_bonus;
                }
                // Impalement: he ran onto this spearman's braced point at
                // speed (integrate's spear-line collision) — his OWN
                // momentum is the attack bonus, spent against himself.
                // No stance rules anywhere: a charger who slipped between
                // the spear lines and reached the wielder gets his full
                // charge bonus above like anyone else.
                if ev.impale {
                    attack += pv.charge_bonus;
                }
                // A staggered man cannot parry (M2TW: no blocking while
                // staggering — the outnumbered-men snowball): his defence
                // SKILL is gone until he recovers; the shield still
                // passively covers and armour always counts.
                let v_staggered = swing[v] & crate::units::SWING_STAGGERED != 0;
                let skill = if rear || v_staggered {
                    0.0
                } else {
                    pv.defence_skill
                };
                let shield = if front || left_side {
                    pv.shield + if v_wall == 1 { SHIELDWALL_SHIELD_PTS } else { 0.0 }
                } else {
                    0.0
                };
                let armour = if pa.ap { pv.armour * 0.5 } else { pv.armour };
                let factor =
                    (attack - (skill + armour + shield)).clamp(-FACTOR_CLAMP, FACTOR_CLAMP);
                let dmg = BASE_DMG[pa.weapon as usize]
                    * FACTOR_MULT.powf(factor)
                    * ev.jit
                    * combat_scale;
                hp[v] -= dmg;
                flash[v] = FLASH_TICKS;
                let died = hp[v] <= 0.0;
                if died {
                    death_t[v] = DEATH_TICKS;
                    // kills[] counts losses OF that team (overlay semantics).
                    cstats.kills[team[v] as usize] += 1;
                    groups.list[group[v] as usize].recent_deaths += 1;
                    // The attacker's regiment is winning its exchange.
                    groups.list[group[a] as usize].recent_kills += 1;
                }
                // Charge impact (phase B): momentum becomes a shove and a
                // stun. Walls barely budge and never stagger; a braced
                // SPEARWALL additionally reflects the charge bonus onto
                // the charger (points punish momentum) — the M2TW rule.
                if !died {
                    // Impacts always rock their man; an ordinary blow
                    // staggers when it got through cleanly — the roll
                    // rides the same factor the damage did.
                    let certain = ev.charge || ev.impale || debug_stagger();
                    let p = if certain {
                        1.0
                    } else {
                        (STAGGER_P0 + STAGGER_P_PER_FACTOR * factor)
                            .clamp(STAGGER_P_MIN, 1.0)
                    };
                    let roll = crate::units::hash01(
                        tick_seed
                            ^ (ev.victim.wrapping_mul(0x9E37))
                            ^ (ev.attacker.wrapping_mul(0x85EB)),
                    );
                    // Bracing is posture physics: a wall hit FRONTALLY has
                    // its weight planted — quarter knockback, no stagger;
                    // from the flank or rear it is bodies like any others.
                    let braced = v_wall != 0 && front;
                    if ev.charge {
                        let m_a = pa.mass;
                        let m_v = pv.mass;
                        let resist = if braced { WALL_KNOCKBACK_RESIST } else { 1.0 };
                        let shove = -dir * CHARGE_KNOCKBACK * (m_a / (m_a + m_v)) * resist;
                        let mut nx = (pos[v].x + shove.x).clamp(bounds_min.x, bounds_max.x);
                        let mut nz = (pos[v].z + shove.y).clamp(bounds_min.y, bounds_max.y);
                        // Charges can't knock a man into the river or
                        // through a terrace riser — the shove just dies.
                        if terrain.blocked_at(nx, nz) {
                            nx = pos[v].x;
                            nz = pos[v].z;
                        }
                        pos[v] = Vec3::new(
                            nx,
                            terrain.height_at(nx, nz)
                                + crate::unit_types::half_height(kind[v] as usize),
                            nz,
                        );
                    }
                    // An impaled runner is STOPPED, not thrown — his
                    // momentum went into the point; the stagger is the
                    // stop.
                    if !braced && roll < p {
                        if swing[v] & crate::units::SWING_STAGGER_IMMUNE != 0 {
                            // The free pass earned by the last stagger is
                            // spent absorbing this one.
                            swing[v] &= !crate::units::SWING_STAGGER_IMMUNE;
                        } else {
                            let stun = if certain {
                                STAGGER_TICKS
                            } else {
                                HIT_STAGGER_TICKS
                            };
                            swing[v] = crate::units::SWING_RECOVER
                                | crate::units::SWING_STAGGERED;
                            swing_t[v] = swing_t[v].max(stun);
                        }
                    }
                }
                if dir_stats.enabled {
                    let vt = team[v] as usize;
                    let s = if front { 0 } else if !rear { 1 } else { 2 };
                    dir_stats.hits[vt][s] += 1;
                    dir_stats.dmg[vt][s] += dmg as f64;
                    if died {
                        dir_stats.kills[vt][s] += 1;
                    }
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

/// FL_DIAG_REAR: idle soldiers (Ready, no enemy in reach) of engaged
/// formed regiments, bucketed by order class and rank: how many move,
/// how fast, which term drives them, and how many push into a friend's
/// back. Logs every 60 ticks.
fn rear_diag_aggregate(
    rear: &mut RearDiag,
    units: &Units,
    groups: &Groups,
    dt: f32,
    lost: [u64; 2],
    tick: u32,
) {
    let n = units.pos.len();
    rear.prev_disp.resize(n, Vec2::ZERO);
    let ng = groups.list.len();
    let mut max_fwd = vec![f32::MIN; ng];
    for i in 0..n {
        let g = units.group[i] as usize;
        let fwd = crate::formation::facing_dir(groups.list[g].facing);
        max_fwd[g] = max_fwd[g].max(units.home[i].dot(fwd));
    }
    let rows = std::mem::take(&mut rear.rows);
    for buf in &rows {
        for r in buf {
            let i = r.i as usize;
            if i >= n {
                continue;
            }
            let disp = (units.pos[i] - units.pos_prev[i]).xz();
            let prev = rear.prev_disp[i];
            rear.prev_disp[i] = disp;
            let g = units.group[i] as usize;
            let gd = &groups.list[g];
            if !gd.engaged
                || gd.state.is_broken()
                || gd.shape != crate::formation::FormShape::Rect
                || r.flags & 1 == 0
            {
                continue;
            }
            let fwd = crate::formation::facing_dir(gd.facing);
            let pitch = gd.spacing.pitch().y;
            let rank = (((max_fwd[g] - units.home[i].dot(fwd)) / pitch).round().max(0.0) as usize)
                .min(5);
            let class = match gd.order {
                Some(crate::orders::Order::Attack(_)) => 0u8,
                None => 1,
                Some(crate::orders::Order::Move(_)) => 2,
            };
            let sp = disp.length() / dt;
            let a = rear.acc.entry((class, rank)).or_insert([0.0; 19]);
            // Walking backward: faster than 0.3 m/s, within 45 degrees of
            // straight back from the regiment's facing.
            if disp.length() / dt > 0.3 && disp.dot(fwd) < -0.707 * disp.length() {
                a[18] += 1.0;
            }
            let blocked = r.flags & 16 != 0;
            if blocked {
                a[14] += 1.0;
            }
            a[0] += 1.0;
            a[1] += sp as f64;
            let (ls, lu, lp, lc) =
                (r.slot.length(), r.surge.length(), r.push.length(), r.corr.length());
            a[4] += ls as f64;
            a[5] += lu as f64;
            a[6] += lp as f64;
            a[7] += lc as f64;
            a[13] += r.goal_d as f64;
            if r.flags & 2 != 0 {
                a[12] += 1.0;
            }
            if sp > 0.06 {
                a[2] += 1.0;
                if blocked {
                    a[15] += 1.0;
                    if sp < 1.2 {
                        a[16] += 1.0;
                    }
                }
                if sp < 1.2 {
                    a[17] += 1.0;
                }
                if sp > 0.3 {
                    a[3] += 1.0;
                }
                let pc = lp + lc;
                if ls >= lu && ls >= pc {
                    a[8] += 1.0;
                } else if lu >= pc {
                    a[9] += 1.0;
                } else {
                    a[10] += 1.0;
                }
                if prev.length() / dt > 0.06 && prev.dot(disp) < 0.0 {
                    a[11] += 1.0;
                }
            }
        }
    }
    rear.rows = rows;
    rear.next_log += 1;
    if rear.next_log >= 60 {
        rear.next_log = 0;
        // Depth profile of engaged formed regiments: how full each
        // rank-deep band behind the front is, in men per file. A closed
        // block reads about 1.0 band after band; a hollow behind the
        // front shows as low early bands.
        const BANDS: usize = 8;
        let mut occ = [0.0f32; BANDS];
        let mut regs = 0usize;
        let mut depths: Vec<Vec<f32>> = vec![Vec::new(); ng];
        for i in 0..n {
            let g = units.group[i] as usize;
            let gd = &groups.list[g];
            if gd.engaged
                && !gd.state.is_broken()
                && gd.shape == crate::formation::FormShape::Rect
                && units.death_t[i] == 0
            {
                let f = crate::formation::facing_dir(gd.facing);
                depths[g].push(units.pos[i].xz().dot(f));
            }
        }
        for (g, d) in depths.iter_mut().enumerate() {
            if d.len() < 20 {
                continue;
            }
            d.sort_by(|a, b| b.total_cmp(a));
            let front = d[d.len() / 20];
            let pitch = groups.list[g].spacing.pitch().y;
            let files = groups.list[g].files.max(1) as f32;
            for &x in d.iter() {
                let b = ((front - x) / pitch).max(0.0) as usize;
                if b < BANDS {
                    occ[b] += 1.0 / files;
                }
            }
            regs += 1;
        }
        let occ_s: String = occ
            .iter()
            .map(|o| format!(" {:.2}", o / regs.max(1) as f32))
            .collect();
        let mut s = format!("\n  depth bands (men per file, front first):{occ_s}");
        for ((class, rank), a) in &rear.acc {
            let c = ["atk", "none", "move"][*class as usize];
            let k = a[0].max(1.0);
            let mv = a[2].max(1.0);
            s.push_str(&format!(
                "\n  {c} r{rank}: n{:.0} spd {:.2} mov {:.0}% walk {:.0}% | slot {:.2} surge {:.2} push {:.2} corr {:.2} | dom slot/surge/body {:.0}/{:.0}/{:.0}% rev {:.0}% memo {:.0}% goal_d {:.1} | blocked {:.0}% of idle, {:.0}% of movers; creepers(<1.2) {:.0}% of movers, blocked {:.0}% of creepers, backward {:.1}%",
                a[0] / 60.0,
                a[1] / k,
                100.0 * a[2] / k,
                100.0 * a[3] / k,
                a[4] / k,
                a[5] / k,
                a[6] / k,
                a[7] / k,
                100.0 * a[8] / mv,
                100.0 * a[9] / mv,
                100.0 * a[10] / mv,
                100.0 * a[11] / mv,
                100.0 * a[12] / k,
                a[13] / k,
                100.0 * a[14] / k,
                100.0 * a[15] / mv,
                100.0 * a[17] / mv,
                100.0 * a[16] / a[17].max(1.0),
                100.0 * a[18] / k,
            ));
        }
        info!(
            "[rear-diag] tick {tick} lost blue {} orange {}; idle men of engaged regiments:{s}",
            lost[0], lost[1]
        );
        rear.acc.clear();
    }
}

fn toggle_debug_viz(keys: Res<ButtonInput<KeyCode>>, mut viz: ResMut<DebugViz>) {
    if keys.just_pressed(KeyCode::KeyG) {
        viz.0 = !viz.0;
    }
}

