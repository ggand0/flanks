//! One kinematic tick's job: the shared columns it reads, the
//! per-regiment snapshot, and the buffers it fills. `prepare_tick`
//! builds it on the main thread; the kernel runs it on the worker.

use bevy::prelude::*;
use std::sync::Arc;

use crate::orders::{GroupData, Groups};
use crate::sim::damage::DamageEvent;
use crate::spatial::SpatialGrid;
use crate::terrain::Terrain;
use crate::units::Units;

/// Soldiers per parallel kernel task.
pub(crate) const CHUNK: usize = 2048;

/// What an archer regiment's bows shoot at this tick (regiment-level
/// fire solution, resolved in `prepare_tick`).
pub(crate) struct ShootAt {
    pub(crate) c: Vec2,
    pub(crate) vel: Vec2,
    pub(crate) r: f32,
    /// Target regiment (index into `Groups::list`): the per-shot
    /// aim picks one of its living soldiers.
    pub(crate) t: usize,
    /// The men shoot the clear men of the regiment's `search_blocks`
    /// (sim/mod.rs `clear_members`, sim/soldier.rs `pick_quarry`): the
    /// target is in melee, or coming for the regiment.
    pub(crate) search: bool,
}

/// Ticks between two looks of an archer regiment at the blocks about
/// it: the skirmish triggers, the fire solutions' scan and the sidearm
/// reasons, each a pass over every block on the field, staggered by
/// regiment so a quarter of the archer regiments look each tick. Seven
/// and a half looks a second; a block moves at most a metre between
/// two. What a look found is kept until the next.
pub(crate) const ARCHER_LOOK: u32 = 8;

/// What an archer regiment's scan of the field found, kept between
/// looks (`archer_fire_solutions`): the block coming for it, the
/// nearest block in range that is not in melee, and the blocks in melee
/// in range, each with its edge gap. `tick` is the look's; u32::MAX
/// marks a regiment that has not looked.
#[derive(Clone)]
pub(crate) struct ArcherScan {
    pub(crate) tick: u32,
    pub(crate) danger: Option<(f32, usize)>,
    pub(crate) best_free: Option<(f32, usize)>,
    pub(crate) busy: Vec<(f32, usize)>,
}

impl Default for ArcherScan {
    fn default() -> Self {
        Self { tick: u32::MAX, danger: None, best_free: None, busy: Vec::new() }
    }
}

/// Why an archer regiment's men take their knives out this tick
/// (sim/soldier.rs SIGHT_KNIFE). Empty for every other regiment.
#[derive(Clone, Default)]
pub(crate) struct Sidearm {
    /// Its swords-out order: every man's knife is out.
    pub(crate) swords: bool,
    /// An enemy block is within its bow range: a man with an empty
    /// quiver takes his knife out.
    pub(crate) foe_in_range: bool,
    /// The enemy blocks whose edges may come within a man's knife
    /// distance of him (soldier.rs KNIFE_EDGE): each man measures his
    /// own distance to them.
    pub(crate) blocks: Vec<crate::formation::BlockRect>,
}

/// Everything one kinematic tick owns: the shared columns it reads, the
/// per-regiment command snapshot, and the buffers it fills. Fully
/// self-contained, so the tick can run on a worker thread while the main
/// world keeps rendering the last completed tick.
///
/// The soldier columns are not copied. The job holds `Arc` clones of
/// the live columns (`Units::Column`), taken at the kick. The main world
/// writes its columns only between the install and the kick, when the
/// job holds nothing, so the clones are never stale and the writes never
/// copy. The read-modify-write columns are read from the clones and
/// written to the job's own output buffers, which the install swaps in;
/// the buffers the install releases become the next tick's outputs.
#[derive(Default)]
pub struct TickJob {
    // Shared inputs: the live columns at the kick. `pos_in` is the
    // kernel's `pos_prev`.
    pub(crate) pos_in: Arc<Vec<Vec3>>,
    pub(crate) speed: Arc<Vec<f32>>,
    pub(crate) team: Arc<Vec<u8>>,
    pub(crate) kind: Arc<Vec<u8>>,
    pub(crate) group: Arc<Vec<u32>>,
    pub(crate) home: Arc<Vec<Vec2>>,
    pub(crate) vel_in: Arc<Vec<Vec3>>,
    pub(crate) yaw_in: Arc<Vec<f32>>,
    pub(crate) yaw_prev_in: Arc<Vec<f32>>,
    pub(crate) target_in: Arc<Vec<u32>>,
    pub(crate) swing_in: Arc<Vec<u8>>,
    pub(crate) swing_t_in: Arc<Vec<u8>>,
    pub(crate) flash_in: Arc<Vec<u8>>,
    pub(crate) death_t_in: Arc<Vec<u8>>,
    pub(crate) ammo_in: Arc<Vec<u8>>,
    pub(crate) out_form_in: Arc<Vec<bool>>,
    pub(crate) sight_in: Arc<Vec<u16>>,
    pub(crate) chain_in: Arc<Vec<u8>>,
    pub(crate) way_in: Arc<Vec<Vec2>>,
    pub(crate) pose_in: Arc<Vec<u8>>,
    pub(crate) stuck_in: Arc<Vec<u16>>,
    pub(crate) quarry_in: Arc<Vec<u32>>,
    // Read only to pack the render snapshot.
    pub(crate) pos_prev_in: Arc<Vec<Vec3>>,
    pub(crate) color_in: Arc<Vec<[f32; 4]>>,
    /// Where the job packs the render snapshot of the columns it was
    /// kicked with, first thing (render_units_gpu.rs `SnapshotSlot`). None
    /// on the CPU render path and for an inline tick.
    pub(crate) pack: Option<Arc<crate::render_units_gpu::SnapshotSlot>>,
    /// The Hit flash setting the snapshot packs with.
    pub(crate) hit_flash: bool,
    // Outputs: the columns after the tick. Each kernel task copies its
    // chunk in from the shared column, then runs the stages on it.
    pub(crate) pos_out: Vec<Vec3>,
    pub(crate) vel: Vec<Vec3>,
    pub(crate) yaw: Vec<f32>,
    pub(crate) yaw_prev: Vec<f32>,
    pub(crate) target: Vec<u32>,
    pub(crate) swing: Vec<u8>,
    pub(crate) swing_t: Vec<u8>,
    pub(crate) flash: Vec<u8>,
    pub(crate) death_t: Vec<u8>,
    pub(crate) ammo: Vec<u8>,
    pub(crate) out_form: Vec<bool>,
    pub(crate) sight: Vec<u16>,
    pub(crate) chain: Vec<u8>,
    pub(crate) way: Vec<Vec2>,
    pub(crate) pose: Vec<u8>,
    pub(crate) stuck: Vec<u16>,
    pub(crate) quarry: Vec<u32>,
    // Per-regiment command snapshot, taken at prep.
    pub(crate) orders: Vec<Option<Vec2>>,
    pub(crate) anchors: Vec<Vec2>,
    pub(crate) reg_broken: Vec<bool>,
    pub(crate) press: Vec<bool>,
    pub(crate) moving: Vec<bool>,
    /// Per regiment: the stance its men are drawn in, which sets their
    /// width (sim/soldier.rs STANCE_*).
    pub(crate) stance: Vec<u8>,
    pub(crate) focus: Vec<Option<u32>>,
    pub(crate) engaged: Vec<bool>,
    pub(crate) contact: Vec<bool>,
    pub(crate) fight_point: Vec<Option<Vec2>>,
    pub(crate) melee_ticks: Vec<u32>,
    pub(crate) hold: Vec<bool>,
    pub(crate) threat: Vec<Vec2>,
    pub(crate) form_face: Vec<Vec2>,
    pub(crate) wall: Vec<u8>,
    /// Per regiment: in a wall stance (the grid's META_WALL bit).
    pub(crate) group_wall: Vec<bool>,
    pub(crate) charging: Vec<bool>,
    pub(crate) fat_speed: Vec<f32>,
    pub(crate) fat_nocharge: Vec<bool>,
    pub(crate) shoot_at: Vec<Option<ShootAt>>,
    /// Per regiment: under fire this tick (some regiment's bows aim at it).
    pub(crate) targeted: Vec<bool>,
    pub(crate) blocks: [Vec<(Vec2, f32, f32)>; 2],
    pub(crate) faces_spearwall: [bool; 2],
    // Products of the run beyond the columns, all swapped into their
    // resources when the job is taken or installed.
    /// Each regiment's men as index ranges, in index order (see
    /// `regiment_runs`).
    pub(crate) reg_runs: Vec<Vec<(u32, u32)>>,
    /// Living members of every regiment under fire, per regiment, in
    /// index order.
    pub(crate) target_members: Vec<Vec<u32>>,
    /// Per regiment: some archer regiment's men search in it this tick
    /// (`ShootAt::search`), so the job lists its clear men.
    pub(crate) searched: Vec<bool>,
    /// Per regiment under search: the blocks of the shooters' side
    /// within a shaft's reach of it (`clear_reach`), as rectangles; the
    /// clear men stand farther than that from every one of them.
    pub(crate) clear_blocks: Vec<Vec<crate::formation::BlockRect>>,
    /// Per archer regiment: the enemy blocks its men search in, nearest
    /// first; empty when its bows have a plain target or none.
    pub(crate) search_blocks: Vec<Vec<u32>>,
    /// Per regiment: why its men take their knives out (archers only).
    pub(crate) sidearm: Vec<Sidearm>,
    /// Per regiment: what its last look at the field found (archers
    /// only; prep's own, between looks).
    pub(crate) archer_scans: Vec<ArcherScan>,
    /// The clear men of every regiment under search, per regiment, in
    /// index order: no enemy in their reach (sim/soldier.rs SIGHT_FOE,
    /// as of the last tick) and farther than a shaft's reach from every
    /// block in `clear_blocks`. Prep reads the last run's lists for the
    /// card.
    pub(crate) clear_members: Vec<Vec<u32>>,
    /// The density field of the tick's start positions (frontline.rs),
    /// built when a taker wants it (not on the inline path, where
    /// `update_field` already rebuilt the resource this tick).
    pub(crate) field: crate::frontline::InfluenceField,
    pub(crate) field_wanted: bool,
    /// Men whose death countdown reached its last tick, ascending.
    pub(crate) dead: Vec<u32>,
    /// Living men standing near or beyond their own map edge, ascending:
    /// the sweep's candidates for a router who has left the field.
    pub(crate) at_edge: Vec<u32>,
    pub(crate) grid: SpatialGrid,
    pub(crate) events: Vec<Vec<DamageEvent>>,
    pub(crate) arrow_spawns: Vec<Vec<crate::arrows::ArrowSpawn>>,
    pub(crate) terrain: Option<std::sync::Arc<Terrain>>,
    pub(crate) dt: f32,
    pub(crate) combat_scale: f32,
    pub(crate) tick_seed: u32,
    pub(crate) tick: u32,
    pub(crate) bounds_min: Vec2,
    pub(crate) bounds_max: Vec2,
    /// `Units::generation` at prep: a job from a dead world is dropped.
    pub(crate) generation: u64,
    pub(crate) grid_ms: f32,
    pub(crate) step_ms: f32,
    pub(crate) field_ms: f32,
    // Routes (sim/route.rs): the plans asked for at prep, planned after
    // the kernel, and each regiment's route with the grid its men read.
    pub(crate) plan_requests: Vec<crate::sim::route::PlanJob>,
    pub(crate) plans: Vec<crate::nav::Plan>,
    pub(crate) routes: Vec<Option<crate::sim::route::RouteView>>,
    pub(crate) nav: Option<Arc<crate::nav::NavGrid>>,
}

impl TickJob {
    /// Release the job's handles on the shared columns. Called the moment
    /// a finished job is taken (and again, harmlessly, at the install):
    /// nothing reads them once the run is over, and a column nobody else
    /// holds is written in place by the main thread's fixed tick and
    /// recycled by the install without a copy.
    pub(crate) fn release_inputs(&mut self) {
        self.pos_in = Default::default();
        self.speed = Default::default();
        self.team = Default::default();
        self.kind = Default::default();
        self.group = Default::default();
        self.home = Default::default();
        self.vel_in = Default::default();
        self.yaw_in = Default::default();
        self.yaw_prev_in = Default::default();
        self.target_in = Default::default();
        self.swing_in = Default::default();
        self.swing_t_in = Default::default();
        self.flash_in = Default::default();
        self.death_t_in = Default::default();
        self.ammo_in = Default::default();
        self.out_form_in = Default::default();
        self.sight_in = Default::default();
        self.chain_in = Default::default();
        self.way_in = Default::default();
        self.pose_in = Default::default();
        self.stuck_in = Default::default();
        self.quarry_in = Default::default();
        self.pos_prev_in = Default::default();
        self.color_in = Default::default();
    }
}

/// The index ranges of every regiment's men, in index order. Regiments
/// are contiguous at spawn; only the death sweep's swap-removes scatter
/// men, so a regiment is a few runs. Walking a regiment's runs visits
/// its men in ascending index, the order every serial loop over the
/// army used, so sums taken this way are the same bits.
pub(crate) fn regiment_runs(group: &[u32], n_groups: usize, out: &mut Vec<Vec<(u32, u32)>>) {
    out.resize_with(n_groups, Vec::new);
    for runs in out.iter_mut() {
        runs.clear();
    }
    let mut i = 0usize;
    let n = group.len();
    while i < n {
        let g = group[i];
        let start = i;
        i += 1;
        while i < n && group[i] == g {
            i += 1;
        }
        if (g as usize) < n_groups {
            out[g as usize].push((start as u32, i as u32));
        }
    }
}

/// The archers' fire solutions of a tick.
struct FireSolutions {
    shoot_at: Vec<Option<ShootAt>>,
    /// Per regiment: under fire this tick.
    targeted: Vec<bool>,
    /// Per regiment: under search this tick.
    searched: Vec<bool>,
    /// Per archer regiment: the blocks its men search in.
    search_blocks: Vec<Vec<u32>>,
    /// Per regiment under search: the shooters' blocks near it.
    clear_blocks: Vec<Vec<crate::formation::BlockRect>>,
    /// Friendly formed blocks per team as discs with a clearance ceiling,
    /// for the loft-over-friendlies check.
    blocks: [Vec<(Vec2, f32, f32)>; 2],
}

/// Fill `job` from the live world: the shared columns, the per-regiment
/// command snapshot, the archers' fire solutions and scalars. The
/// serial prep of a tick, on the main thread, per regiment only: nothing
/// here loops over the army. It also makes the two regiment writes prep
/// makes: the stand-off anchor snap and the `firing` flag.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_tick(
    job: &mut TickJob,
    units: &Units,
    groups: &mut Groups,
    tracks: &crate::arrows::RegTracks,
    routes: &mut crate::sim::route::Routes,
    terrain_arc: &std::sync::Arc<Terrain>,
    dt: f32,
    combat_scale: f32,
    tick: u32,
) {
    job.generation = units.generation;
    job.pack = None;
    job.dt = dt;
    job.combat_scale = combat_scale;
    job.terrain = Some(terrain_arc.clone());
    let terrain: &Terrain = terrain_arc;

    let t0 = std::time::Instant::now();
    share_columns(job, units);
    let t1 = std::time::Instant::now();
    skirmish_reactions(groups, tracks, terrain, tick);
    let t2 = std::time::Instant::now();
    open_for_passers(groups, tracks, tick);
    let t2b = std::time::Instant::now();
    let kept_clear_blocks = std::mem::take(&mut job.clear_blocks);
    let fire = archer_fire_solutions(
        groups,
        tracks,
        terrain,
        &job.clear_members,
        &mut job.archer_scans,
        kept_clear_blocks,
        tick,
    );
    let t3 = std::time::Instant::now();
    job.orders = resolve_orders(groups, routes);
    routes.fill_job(&mut job.plan_requests, &mut job.routes, &mut job.nav, groups.list.len());
    snapshot_regiments(job, groups, terrain);
    let t4 = std::time::Instant::now();

    let n_chunks = units.pos.len().div_ceil(CHUNK);
    if job.events.len() < n_chunks {
        job.events.resize_with(n_chunks, Vec::new);
    }
    if job.arrow_spawns.len() < n_chunks {
        job.arrow_spawns.resize_with(n_chunks, Vec::new);
    }
    // Spawn buffers come back drained from arrows.rs. A job dropped as
    // stale never got that far.
    for buf in &mut job.arrow_spawns {
        buf.clear();
    }
    job.tick_seed = tick.wrapping_mul(0x9E37_79B1);
    job.tick = tick;
    job.shoot_at = fire.shoot_at;
    job.targeted = fire.targeted;
    job.searched = fire.searched;
    job.search_blocks = fire.search_blocks;
    job.clear_blocks = fire.clear_blocks;
    job.blocks = fire.blocks;
    let last = std::mem::take(&mut job.sidearm);
    job.sidearm = sidearm_reasons(groups, tracks, terrain, &last, tick);
    job.field.size_to(terrain.min(), terrain.max());
    let t5 = std::time::Instant::now();
    log_prep([t1 - t0, t2 - t1, t2b - t2, t3 - t2b, t4 - t3, t5 - t4, t5 - t0]);
}

/// FL_LOG_STEP=1: the mean time of each part of the prep over 150 ticks
/// (5 s), beside the `[step]` line (sim/diag.rs): the column share, the
/// skirmish reactions, the opening for passers, the fire solutions, the
/// orders and the regiment snapshot, the sidearm reasons, and the whole
/// prep.
fn log_prep(parts: [std::time::Duration; 7]) {
    use std::sync::{Mutex, OnceLock};
    static ON: OnceLock<bool> = OnceLock::new();
    static ACC: Mutex<([f64; 7], u32)> = Mutex::new(([0.0; 7], 0));
    if !*ON.get_or_init(|| std::env::var("FL_LOG_STEP").is_ok()) {
        return;
    }
    let mut acc = ACC.lock().unwrap_or_else(|e| e.into_inner());
    for (a, p) in acc.0.iter_mut().zip(parts) {
        *a += p.as_secs_f64() * 1000.0;
    }
    acc.1 += 1;
    if acc.1 == 150 {
        let m = acc.0.map(|v| v / 150.0);
        info!(
            "[prep] share {:.3}, skirmish {:.3}, make space {:.4}, fire {:.3}, orders and snapshot {:.3}, \
             sidearm {:.3}, whole prep {:.3} ms",
            m[0], m[1], m[2], m[3], m[4], m[5], m[6]
        );
        *acc = ([0.0; 7], 0);
    }
}

/// The soldier columns, shared into the job: `Arc` clones of the live
/// columns, no copy. The output buffers are sized to the army; their
/// contents are overwritten by the kernel (each task copies its chunk in
/// from the shared column first, and the position output is written for
/// every man), so they are never cleared.
fn share_columns(job: &mut TickJob, units: &Units) {
    job.pos_in = units.pos.share();
    job.speed = units.speed.share();
    job.team = units.team.share();
    job.kind = units.kind.share();
    job.group = units.group.share();
    job.home = units.home.share();
    job.vel_in = units.vel.share();
    job.yaw_in = units.yaw.share();
    job.yaw_prev_in = units.yaw_prev.share();
    job.target_in = units.target.share();
    job.swing_in = units.swing.share();
    job.swing_t_in = units.swing_t.share();
    job.flash_in = units.flash.share();
    job.death_t_in = units.death_t.share();
    job.ammo_in = units.ammo.share();
    job.out_form_in = units.out_form.share();
    job.sight_in = units.sight.share();
    job.chain_in = units.chain.share();
    job.way_in = units.way.share();
    job.pose_in = units.pose.share();
    job.stuck_in = units.stuck.share();
    job.quarry_in = units.quarry.share();
    job.pos_prev_in = units.pos_prev.share();
    job.color_in = units.color.share();
    let n = units.pos.len();
    fn fit<T: Clone + Default>(v: &mut Vec<T>, n: usize) {
        if v.len() != n {
            v.resize(n, T::default());
        }
    }
    fit(&mut job.pos_out, n);
    fit(&mut job.vel, n);
    fit(&mut job.yaw, n);
    fit(&mut job.yaw_prev, n);
    fit(&mut job.target, n);
    fit(&mut job.swing, n);
    fit(&mut job.swing_t, n);
    fit(&mut job.flash, n);
    fit(&mut job.death_t, n);
    fit(&mut job.ammo, n);
    fit(&mut job.out_form, n);
    fit(&mut job.sight, n);
    fit(&mut job.chain, n);
    fit(&mut job.way, n);
    fit(&mut job.pose, n);
    fit(&mut job.stuck, n);
    fit(&mut job.quarry, n);
}

/// How far an archer regiment marches at a target it cannot reach: its
/// front rank stops this share of its range short of the target's near
/// edge (`firing_place`), then it stands and shoots. It shoots on until
/// the gap passes the range; the band between is the hysteresis that
/// keeps a block from marching and halting by turns.
const CLOSE_IN: f32 = 0.8;
/// The time an enemy block takes to reach a regiment that counts as
/// coming for it (seconds): the skirmish trigger and, with skirmish off,
/// the moment the men turn their bows on it.
pub(crate) const APPROACH_S: f32 = 8.0;
/// Enemy blocks a regiment's search lists at most, nearest first.
const SEARCH_BLOCKS: usize = 4;

/// The gap between two regiments' block edges along the line between
/// their centres (metres, negative when the blocks overlap). The edges
/// are the blocks' rectangles (formation.rs `block_reach_along`), so a
/// wide line is not credited with a reach it has only sideways.
pub(crate) fn edge_gap(a: &GroupData, b: &GroupData) -> f32 {
    rect_gap(&crate::formation::block_rect(a), &crate::formation::block_rect(b))
}

/// `edge_gap` when it can be within `limit`, else None without the
/// rectangles: the centres' distance less both blocks' bounding radii
/// (`formation::block_bound`) is never more than the gap, so a pair that
/// fails it is beyond `limit` for certain. The per-regiment loops over
/// every block pass most pairs through here.
fn gap_within(a: &GroupData, b: &GroupData, limit: f32) -> Option<f32> {
    let d = a.centroid.distance(b.centroid);
    if d - crate::formation::block_bound(a) - crate::formation::block_bound(b) > limit {
        return None;
    }
    Some(edge_gap(a, b))
}

/// `edge_gap` between two blocks given as rectangles.
fn rect_gap(a: &crate::formation::BlockRect, b: &crate::formation::BlockRect) -> f32 {
    let d = b.centre - a.centre;
    let dist = d.length();
    if dist < 1e-3 {
        return -(a.half_w.max(a.half_d) + b.half_w.max(b.half_d));
    }
    let u = d / dist;
    dist - a.reach_along(u) - b.reach_along(-u)
}

/// How far `g`'s bows reach toward `t` from `from` (metres) with `t`
/// standing: the bows' range plus the height `from` stands above `t`
/// (the high ground's reach). The ground the firing place is laid on.
fn bow_reach_standing(g: &GroupData, from: Vec2, t: &GroupData, terrain: &Terrain) -> f32 {
    let height = terrain.height_at(from.x, from.y) - terrain.height_at(t.centroid.x, t.centroid.y);
    crate::unit_types::bow_range(g.kind) + height
}

/// How far `g`'s bows reach toward `t` from `from` (metres): the
/// standing reach plus three seconds of `t`'s speed toward `from` (the
/// shafts land where the target will be). The range test's reach; the
/// firing place does not follow the target's speed, or the regiment
/// would walk back and forth with every change of it.
fn bow_reach_toward(
    g: &GroupData,
    from: Vec2,
    t: &GroupData,
    t_vel: Vec2,
    terrain: &Terrain,
) -> f32 {
    let to_g = (from - t.centroid).normalize_or_zero();
    bow_reach_standing(g, from, t, terrain) + 3.0 * t_vel.dot(to_g)
}

/// Whether `g` can shoot at `t` from where it stands: the edge gap
/// within `bow_reach_toward`.
fn in_bow_range(g: &GroupData, t: &GroupData, t_vel: Vec2, terrain: &Terrain) -> bool {
    edge_gap(g, t) <= bow_reach_toward(g, g.centroid, t, t_vel, terrain)
}

/// Whether `g`'s frame about `centre` is in range of `t`: the block as
/// it dresses there, not the men's average, which a straggler caught in
/// a friendly block drags off it. The test of a fire order's stand-off.
fn frame_in_bow_range(
    g: &GroupData,
    centre: Vec2,
    t: &GroupData,
    t_vel: Vec2,
    terrain: &Terrain,
) -> bool {
    let here = crate::formation::block_rect_at(g, centre);
    let gap = rect_gap(&here, &crate::formation::block_rect(t));
    gap <= bow_reach_toward(g, centre, t, t_vel, terrain)
}

/// Where an archer regiment's attack order marches it: the frame centre
/// on the line to the target from which its front rank stands CLOSE_IN
/// of its reach short of the target's near edge, kept inside the map.
/// Every man walks to his slot about it, so no slot lies in the enemy
/// and a column's head stops at the place like everyone else.
fn firing_place(g: &GroupData, t: &GroupData, terrain: &Terrain) -> Vec2 {
    let to_t = t.centroid - g.centroid;
    let dist = to_t.length();
    if dist < 1e-3 {
        return g.centroid;
    }
    let u = to_t / dist;
    let reach = bow_reach_standing(g, g.centroid, t, terrain);
    let short = crate::formation::block_reach_along(t, -u)
        + (CLOSE_IN * reach).max(0.0)
        + crate::formation::block_rect(g).half_d;
    let (mn, mx) = (terrain.min() + 12.0, terrain.max() - 12.0);
    (t.centroid - u * short).clamp(mn, mx)
}

/// Where along the line from `origin` (`dir`, `len` metres) the block
/// `b`'s centre is nearest, when the line passes within `b`'s reach of
/// it there, plus `width`.
fn line_hit(
    groups: &Groups,
    origin: Vec2,
    dir: Vec2,
    len: f32,
    width: f32,
    b: usize,
) -> Option<f32> {
    let bd = &groups.list[b];
    let rel = bd.centroid - origin;
    let s = rel.dot(dir).clamp(0.0, len);
    let off = rel - dir * s;
    if off.length() > crate::formation::block_bound(bd) + width {
        return None;
    }
    let n = off.normalize_or_zero();
    let reach = if n == Vec2::ZERO {
        f32::MAX
    } else {
        crate::formation::block_reach_along(bd, -n) + width
    };
    (off.length() <= reach).then_some(s)
}

/// Where along `mover`'s path ahead (`dir`, `len` metres from its
/// centre) the block `b`'s centre is nearest, when the path, swept by
/// the mover's own width, comes within `b`'s reach of it there.
fn path_hit(groups: &Groups, mover: usize, dir: Vec2, len: f32, b: usize) -> Option<f32> {
    let md = &groups.list[mover];
    let rel = groups.list[b].centroid - md.centroid;
    let s = rel.dot(dir).clamp(0.0, len);
    let n = (rel - dir * s).normalize_or_zero();
    let width = crate::formation::block_reach_along(md, n);
    line_hit(groups, md.centroid, dir, len, width, b)
}

/// The way a regiment is going: toward its order's goal when it has one
/// further than a metre off and is not standing off its target, else
/// along `vel` when that is a walk. A regiment holding before a block in
/// its path still heads for its goal: the hold asks whether walking on
/// would run into the block.
fn heading_of(groups: &Groups, g: usize, vel: Vec2) -> Option<Vec2> {
    let gd = &groups.list[g];
    let c = gd.centroid;
    match groups.goal(g) {
        Some(goal) if !gd.standoff && goal.distance_squared(c) > 1.0 => {
            Some((goal - c).normalize())
        }
        _ => (vel.length() >= 0.3).then(|| vel.normalize()),
    }
}

/// A regiment stopping on the march pins its frame where its leading
/// men are: the front rank on them, the rest closing up behind, and the
/// slots dealt in the men's present order (the reform), so nobody walks
/// back. The frame at the centroid would send a column's head walking
/// back; the frame left at the old anchor would send everyone back.
fn pin_frame_at_head(gd: &mut GroupData) {
    let f = crate::formation::facing_dir(gd.facing);
    gd.anchor = gd.centroid + f * (gd.lead - crate::formation::front_offset(gd));
    gd.reform = true;
}

/// Whether a marching regiment's leading men have reached the front
/// rank of its frame at `place`: the foremost man along the facing
/// stands within the Move order's arrival distance of that line, or
/// past it. The first men there are the arrival, as the leading man's
/// is for the reference's formed move; the centroid would wait for
/// every straggler caught in a friendly block. The rest walk on to
/// their slots as a holding regiment dresses.
fn head_arrived(gd: &GroupData, place: Vec2) -> bool {
    let f = crate::formation::facing_dir(gd.facing);
    let head = gd.centroid + f * gd.lead;
    let front = place + f * crate::formation::front_offset(gd);
    (head - front).dot(f) >= -crate::orders::ARRIVE_CLEAR
}

/// Whether the enemy block `e`, going its way, reaches `g` within
/// `horizon` seconds: its path ahead for min(its speed * horizon, `cap`,
/// the distance), swept by its width, comes within `g`'s reach of `g`'s
/// centre, and no friendly block of `g`'s that is not itself running
/// screens `g` on the way: a friend screens when its block covers the
/// path's centre line nearer to `e`, or both of the path's sides (one
/// side only is passed by, as the attacker's men stream around it).
pub(crate) fn approach_hits(
    groups: &Groups,
    g: usize,
    e: usize,
    e_vel: Vec2,
    horizon: f32,
    cap: f32,
) -> bool {
    let gd = &groups.list[g];
    let ed = &groups.list[e];
    let speed = e_vel.length();
    if speed < 0.3 {
        return false;
    }
    let Some(dir) = heading_of(groups, e, e_vel) else {
        return false;
    };
    let len = (speed * horizon).min(cap).min(gd.centroid.distance(ed.centroid));
    let Some(s_g) = path_hit(groups, e, dir, len, g) else {
        return false;
    };
    let perp = Vec2::new(-dir.y, dir.x);
    let side = perp * crate::formation::block_reach_along(ed, perp);
    let before = |origin: Vec2, f: usize| {
        line_hit(groups, origin, dir, len, 0.0, f).is_some_and(|s_f| s_f < s_g)
    };
    !groups.list.iter().enumerate().any(|(f, fd)| {
        f != g
            && fd.team == gd.team
            && fd.count > 0
            && !fd.state.is_broken()
            && !fd.skirmishing
            && (before(ed.centroid, f)
                || (before(ed.centroid + side, f) && before(ed.centroid - side, f)))
    })
}

/// A holding regiment opens this far ahead of a friend coming through it,
/// beyond the ground its opening grows toward him (metres).
const OPEN_MARGIN: f32 = 5.0;
/// ...and never less than this much of the friend's walk ahead (seconds).
const OPEN_LEAD_S: f32 = 8.0;
/// A friend's block this close to an open regiment's keeps it open
/// (metres).
const OPEN_KEEP: f32 = 3.0;

/// A regiment's men standing still are drawn in their ready stance
/// (render_units.rs `stance_tier` above zero, the guard of the pose
/// pass), or a wall: they stand at their combat width.
pub(crate) fn ready_stance(g: &GroupData) -> bool {
    crate::render_units::stance_tier(g) > 0.0 || crate::render_units::wall_signal(g) > 0.0
}

/// The stance a regiment's men are drawn in (sim/soldier.rs STANCE_*):
/// fighting or in a wall, in the ready stance whatever they do; with an
/// enemy near, attacking or charging, in it while they stand; else at
/// ease.
fn stance_of(g: &GroupData) -> u8 {
    if g.engaged || crate::render_units::wall_signal(g) > 0.0 {
        crate::sim::soldier::STANCE_FIGHT
    } else if ready_stance(g) {
        crate::sim::soldier::STANCE_READY
    } else {
        crate::sim::soldier::STANCE_EASE
    }
}

/// The free ground before a holding regiment's front and behind its rear
/// (metres): of every other regiment with men whose block reaches across
/// its frontage, the nearest block's gap on each side, OPEN_ROOM_MAX
/// when none is nearer. What `formation::choose_open` grows into.
fn room_around(groups: &Groups, h: usize) -> (f32, f32) {
    let rect = crate::formation::block_rect(&groups.list[h]);
    let right = Vec2::new(rect.fwd.y, -rect.fwd.x);
    let (mut front, mut back) = (crate::formation::OPEN_ROOM_MAX, crate::formation::OPEN_ROOM_MAX);
    for (o, od) in groups.list.iter().enumerate() {
        if o == h || od.count == 0 {
            continue;
        }
        let other = crate::formation::block_rect(od);
        let d = other.centre - rect.centre;
        if d.dot(right).abs() - other.reach_along(right) >= rect.half_w {
            continue;
        }
        let along = d.dot(rect.fwd);
        let gap = (along.abs() - other.reach_along(rect.fwd) - rect.half_d).max(0.0);
        if along >= 0.0 {
            front = front.min(gap);
        } else {
            back = back.min(gap);
        }
    }
    (front, back)
}

/// A regiment walking somewhere: a goal further than a metre off, not
/// standing off its target, holding before a block or fighting on a
/// contact frame. The friends a holding regiment opens for.
fn goes_to(groups: &Groups, m: usize) -> Option<Vec2> {
    let md = &groups.list[m];
    if md.count == 0 || md.state.is_broken() || md.standoff || md.skirmish_hold.is_some() || md.contact {
        return None;
    }
    groups.goal(m).filter(|goal| goal.distance_squared(md.centroid) > 1.0)
}

/// Allies make space: every ARCHER_LOOK ticks, staggered, a regiment
/// holding its ground (`formation::holds_ground`) looks at the friendly
/// regiments walking somewhere. It opens (FormSpacing::Open, the layout
/// `formation::choose_open` picks for the free ground around it,
/// `room_around`) when one's path to its goal, swept by its width,
/// crosses its block within the walk the opening needs (the ground the
/// block grows toward him plus OPEN_MARGIN at his speed, never less than
/// OPEN_LEAD_S of it), or when one's block already overlaps its own. It
/// closes when no friend's path crosses it, no walking friend's block is
/// within OPEN_KEEP of it and no standing friend stands on its closed
/// ground. An order, a fight or a rout closes it at once (formation.rs
/// `apply_reforms`). A walking regiment whose block is within OPEN_KEEP
/// of an open friend's is passing it (`GroupData::passing`), on its own
/// look. A passing regiment counts its men inside each open friend's
/// block (`GroupData::inside`, frontline.rs): the friend stays open while
/// any of them is inside, and the passer stays passing until none is.
fn open_for_passers(groups: &mut Groups, tracks: &crate::arrows::RegTracks, tick: u32) {
    let n = groups.list.len();
    for m in 0..n {
        if !tick.wrapping_add(m as u32).is_multiple_of(ARCHER_LOOK) {
            continue;
        }
        let gl: &Groups = &*groups;
        let md = &gl.list[m];
        // Walking, it passes an open friend within OPEN_KEEP of it. It
        // stays passing while any of its men stands inside one, its order
        // done or not: its tail is still in the friend when its block
        // has walked on or its head arrives.
        let near = goes_to(gl, m).is_some() && {
            let bound = crate::formation::block_bound(md);
            let rect = crate::formation::block_rect(md);
            gl.list.iter().enumerate().any(|(h, hd)| {
                h != m
                    && hd.team == md.team
                    && hd.spacing == crate::formation::FormSpacing::Open
                    && hd.centroid.distance(md.centroid) - bound - crate::formation::block_bound(hd)
                        <= OPEN_KEEP
                    && rect_gap(&rect, &crate::formation::block_rect(hd)) <= OPEN_KEEP
            })
        };
        let passing = near || md.passing && md.inside.iter().any(|&(_, c)| c > 0);
        groups.list[m].passing = passing;
    }
    for h in 0..n {
        if !tick.wrapping_add(h as u32).is_multiple_of(ARCHER_LOOK) {
            continue;
        }
        let gl: &Groups = &*groups;
        let hd = &gl.list[h];
        if !crate::formation::holds_ground(hd)
            || hd.shape != crate::formation::FormShape::Rect
            || hd.files == 0
        {
            continue;
        }
        let open = hd.spacing == crate::formation::FormSpacing::Open;
        let rect = crate::formation::block_rect(hd);
        let bound = crate::formation::block_bound(hd);
        let fwd = crate::formation::facing_dir(hd.facing);
        // The open order it stands in, or the one it would take now,
        // looked at only once a friend comes near enough.
        let mut plan = open.then_some(hd.open);
        let mut opener: Option<usize> = None;
        let mut keep = false;
        for m in 0..n {
            let md = &gl.list[m];
            if m == h || md.team != hd.team || md.count == 0 || md.state.is_broken() {
                continue;
            }
            // A friend passing keeps it open while one of its men is
            // inside.
            keep |= md.inside.iter().any(|&(o, c)| o as usize == h && c > 0);
            let Some(goal) = goes_to(gl, m) else {
                // A standing friend on its closed ground keeps it open:
                // one ordered onto that ground stands there.
                if open && !md.passing && md.order.is_none() {
                    let ground = crate::formation::closed_rect_at(hd, crate::formation::closed_centre(hd));
                    keep |= rect_gap(&ground, &crate::formation::block_rect(md)) < 0.0;
                }
                continue;
            };
            let track = tracks.vel.get(m).copied().unwrap_or(Vec2::ZERO);
            let speed = if track.length() >= 0.3 {
                track.length()
            } else {
                crate::unit_types::TYPES[md.kind as usize].speed
            };
            let to_goal = goal.distance(md.centroid);
            let apart = hd.centroid.distance(md.centroid) - bound - crate::formation::block_bound(md);
            let most = ((crate::formation::OPEN_ROOM_MAX + OPEN_MARGIN) / speed).max(OPEN_LEAD_S) * speed;
            if apart > to_goal.min(most).max(OPEN_KEEP) {
                continue;
            }
            // The ground the block grows toward him. The lanes run his
            // way: along the facing, or across it for a friend coming
            // from the side.
            let dir = (goal - md.centroid) / to_goal;
            let o = *plan.get_or_insert_with(|| {
                let across = dir.dot(fwd).abs() < dir.dot(Vec2::new(fwd.y, -fwd.x)).abs();
                let (front, back) = room_around(gl, h);
                crate::formation::choose_open(hd, ready_stance(hd), across, front, back)
            });
            // A friend from the side meets the block's flank, which does
            // not move.
            let growth = if o.across {
                0.0
            } else if (md.centroid - hd.centroid).dot(fwd) > 0.0 {
                o.front
            } else {
                o.back
            };
            let ahead = ((growth.max(0.0) + OPEN_MARGIN) / speed).max(OPEN_LEAD_S) * speed;
            let len = to_goal.min(ahead);
            if apart > len.max(OPEN_KEEP) {
                continue;
            }
            let gap = rect_gap(&rect, &crate::formation::block_rect(md));
            let crosses = path_hit(gl, m, dir, len, h).is_some();
            if crosses || gap < 0.0 {
                opener.get_or_insert(m);
            }
            keep |= crosses || gap <= OPEN_KEEP;
        }
        let gd = &mut groups.list[h];
        if !open && let (Some(m), Some(o)) = (opener, plan) {
            crate::formation::open_order(gd, o);
            if o.across {
                info!(
                    "regiment {h} opens for regiment {m} from the side: {} ranks to a group, {:.1} m ahead and {:.1} m behind",
                    o.files, o.front, o.back
                );
            } else {
                info!(
                    "regiment {h} opens for regiment {m}: {} files to a group, {:.1} m ahead and {:.1} m behind",
                    o.files, o.front, o.back
                );
            }
        } else if open && !keep {
            crate::formation::close_order(gd);
            info!("regiment {h} closes");
        }
    }
}

/// An enemy block's edge this close to the regiment's: run (the first
/// skirmish trigger).
const SKIRMISH_NEAR: f32 = 15.0;
/// Added to the approach horizon while the regiment already runs: the
/// run's hysteresis (8 + 12 = 20 s).
const SKIRMISH_RETREAT_S: f32 = 12.0;
/// How far ahead the regiment's own path is tested for an enemy block
/// it would walk into (the third trigger: stand and shoot instead).
const SKIRMISH_WALK_AHEAD: f32 = 50.0;
/// Enemy blocks further than this many ranges off are no trigger.
const SKIRMISH_SCOPE: f32 = 1.5;
/// Headings tried for the retreat point.
const RETREAT_HEADINGS: u32 = 32;
/// Ticks between two recomputations of a running regiment's point, a
/// multiple of ARCHER_LOOK so the look falls on a recomputation.
const RETREAT_REFRESH: u32 = 16;

/// Where a regiment runs to: a bow's range from where it stands, times
/// (1 + the nearest enemy's speed over its own, at most 1), in the one of
/// RETREAT_HEADINGS headings that lies most nearly away from every
/// unbroken enemy block on the field (the least sum of squared angular
/// differences to their bearings), kept inside the map. None when the
/// map's edge leaves no room: the regiment stands and shoots instead.
fn retreat_point(
    groups: &Groups,
    g: usize,
    tracks: &crate::arrows::RegTracks,
    terrain: &Terrain,
) -> Option<Vec2> {
    use std::f32::consts::{PI, TAU};
    let gd = &groups.list[g];
    let mut bearings: Vec<f32> = Vec::new();
    let mut nearest: Option<(f32, usize)> = None;
    for (e, ed) in groups.list.iter().enumerate() {
        if ed.team == gd.team || ed.count == 0 || ed.state.is_broken() {
            continue;
        }
        let d = gd.centroid - ed.centroid;
        let d2 = d.length_squared();
        if d2 < 1e-3 {
            continue;
        }
        bearings.push(d.y.atan2(d.x));
        if nearest.is_none_or(|(bd, _)| d2 < bd) {
            nearest = Some((d2, e));
        }
    }
    let (_, nearest) = nearest?;
    let mut best = (f32::MAX, 0.0f32);
    for k in 0..RETREAT_HEADINGS {
        let h = k as f32 / RETREAT_HEADINGS as f32 * TAU;
        let cost: f32 = bearings
            .iter()
            .map(|&b| {
                let d = (h - b).rem_euclid(TAU);
                let d = if d > PI { TAU - d } else { d };
                d * d
            })
            .sum();
        if cost < best.0 {
            best = (cost, h);
        }
    }
    let heading = Vec2::new(best.1.cos(), best.1.sin());
    let their = tracks.vel.get(nearest).copied().unwrap_or(Vec2::ZERO).length();
    let ours = crate::unit_types::TYPES[gd.kind as usize].speed;
    let ratio = if their > 0.1 { (their / ours).min(1.0) } else { 0.0 };
    let dist = crate::unit_types::bow_range(gd.kind) * (1.0 + ratio);
    let (mn, mx) = (terrain.min() + 12.0, terrain.max() - 12.0);
    let dest = (gd.centroid + heading * dist).clamp(mn, mx);
    // A point the regiment has as good as reached (orders.rs arrival)
    // is no run.
    let arrive = crate::orders::ARRIVE_CLEAR;
    (dest.distance_squared(gd.centroid) > arrive * arrive).then_some(dest)
}

/// The run ends where the regiment stands: its own order comes back, or
/// it dresses where it stopped. A regiment that only held (no run, so
/// no order parked) keeps the order it has.
pub(crate) fn end_skirmish(gd: &mut GroupData) {
    gd.skirmish_hold = None;
    gd.skirmish_to = None;
    if !gd.skirmishing {
        return;
    }
    gd.skirmishing = false;
    gd.order = gd.skirmish_saved.take();
    if gd.order.is_none() {
        gd.anchor = gd.centroid;
        gd.reform = true;
    }
}

/// The skirmish reaction: an archer regiment in skirmish mode runs from
/// an enemy block whose edge is on it (SKIRMISH_NEAR) or that is coming
/// for it (`approach_hits` within the approach horizon, longer while it
/// already runs), to `retreat_point`, recomputed as it goes. The run is
/// a Move order; the regiment's own order, and any order given during
/// the run, waits in `skirmish_saved` and comes back when the trigger
/// lapses. A regiment walking into an enemy block on its path within
/// SKIRMISH_WALK_AHEAD of its leading men stands and shoots that block
/// instead (`skirmish_hold`); an attack order's close-in march holds so
/// with skirmish off too, since that march is the game's own, not a
/// point the player chose. Enemy men already in reach of the regiment's
/// own men, with no real melee begun (the melee clock), are the block's
/// leading men on it: a run. Hold position, skirmish off and a regiment
/// in a real melee: no run.
fn skirmish_reactions(
    groups: &mut Groups,
    tracks: &crate::arrows::RegTracks,
    terrain: &Terrain,
    tick: u32,
) {
    let n = groups.list.len();
    for g in 0..n {
        let gd = &groups.list[g];
        let range = crate::unit_types::bow_range(gd.kind);
        let bows = gd.kind == crate::unit_types::KIND_ARCHER
            && !gd.melee
            && gd.count > 0
            && !gd.state.is_broken()
            && !gd.hold
            && gd.melee_ticks == 0;
        let able = bows && gd.skirmish;
        let closing = bows
            && !gd.skirmishing
            && !gd.standoff
            && matches!(gd.order, Some(crate::orders::Order::Attack(_)));
        if !able && !closing {
            if gd.skirmishing || gd.skirmish_hold.is_some() {
                end_skirmish(&mut groups.list[g]);
            }
            continue;
        }
        // An order anyone else wrote during the run (the player, the AI,
        // the run's own move arriving) is the one the run parks; the
        // run's move is the auto Move order below.
        let runs_move = gd.auto_order && matches!(gd.order, Some(crate::orders::Order::Move(_)));
        if gd.skirmishing && !runs_move {
            let new = gd.order;
            if new.is_some() {
                groups.list[g].skirmish_saved = new;
            }
        }
        // The regiment's look at the field, every ARCHER_LOOK ticks;
        // between looks it keeps on as it is.
        if !tick.wrapping_add(g as u32).is_multiple_of(ARCHER_LOOK) {
            continue;
        }
        let gl: &Groups = &*groups;
        let gd = &gl.list[g];
        let vel = |t: usize| tracks.vel.get(t).copied().unwrap_or(Vec2::ZERO);
        let horizon = if gd.skirmishing { APPROACH_S + SKIRMISH_RETREAT_S } else { APPROACH_S };
        let standing = gd.order.is_none() || gd.standoff || gd.skirmish_hold.is_some();
        let cap = if standing { 0.8 * range } else { 1.1 * range };
        // A running regiment's path is the run's; a standing one, or one
        // without an order, has none: none of them holds.
        let own_dir = if gd.skirmishing || gd.standoff || gd.order.is_none() {
            None
        } else {
            heading_of(gl, g, vel(g))
        };
        // The path ahead starts at the leading men: a column strung out
        // round a friendly block is seen by its head. A regiment already
        // holding looks SKIRMISH_NEAR further, the hysteresis: it walks on
        // once the block has fallen that much further off than where it
        // stopped the regiment.
        let ahead = SKIRMISH_WALK_AHEAD.min(range - 3.0)
            + gd.lead.max(0.0)
            + if gd.skirmish_hold.is_some() { SKIRMISH_NEAR } else { 0.0 };
        let mut run = able && gd.engaged;
        // The block the run is from (the nearest enemy block when the
        // regiment is engaged): what a cornered regiment shoots. `why`
        // names the trigger for the log.
        let mut run_from: Option<(f32, u32)> = None;
        let mut why = "engaged";
        let mut hold: Option<(f32, u32)> = None;
        for e in 0..n {
            let ed = &gl.list[e];
            if ed.team == gd.team || ed.count == 0 || ed.state.is_broken() {
                continue;
            }
            let Some(gap) = gap_within(gd, ed, SKIRMISH_SCOPE * range) else {
                continue;
            };
            if run {
                if run_from.is_none_or(|(rg, _)| gap < rg) {
                    run_from = Some((gap, e as u32));
                }
                continue;
            }
            if able && gap <= SKIRMISH_NEAR {
                (run, why) = (true, "near");
                run_from = Some((gap, e as u32));
                continue;
            }
            if able && approach_hits(gl, g, e, vel(e), horizon, cap) {
                (run, why) = (true, "coming");
                run_from = Some((gap, e as u32));
                continue;
            }
            if let Some(dir) = own_dir
                && let Some(s) = path_hit(gl, g, dir, ahead, e)
                && hold.is_none_or(|(hs, _)| s < hs)
            {
                hold = Some((s, e as u32));
            }
        }
        let hold = hold.map(|(_, e)| e);
        // A running or cornered regiment looks for its point every
        // RETREAT_REFRESH ticks; one about to run looks now.
        let due = if gd.skirmishing || gd.skirmish_hold.is_some() {
            tick.wrapping_add(g as u32).is_multiple_of(RETREAT_REFRESH)
        } else {
            true
        };
        let point = if run && due { retreat_point(gl, g, tracks, terrain) } else { None };
        let gd = &mut groups.list[g];
        if run {
            if !due {
                // On its way, or cornered and standing, until the next look.
            } else if let Some(p) = point {
                if !gd.skirmishing {
                    gd.skirmish_saved = gd.order;
                    gd.skirmishing = true;
                    match run_from {
                        Some((gap, e)) => {
                            info!("regiment {g} runs (skirmish): block {e} {why}, gap {gap:.1} m")
                        }
                        None => info!("regiment {g} runs (skirmish): {why}"),
                    }
                }
                gd.skirmish_hold = None;
                gd.order = Some(crate::orders::Order::Move(p));
                gd.auto_order = true;
                gd.skirmish_to = Some(p);
            } else {
                // Cornered: no room to run. The regiment stands and shoots
                // until room appears or the trigger lapses.
                if gd.skirmishing {
                    end_skirmish(gd);
                }
                let at = run_from.map(|(_, e)| e);
                if gd.skirmish_hold.is_none() && at.is_some() {
                    info!("regiment {g} is cornered, stands and shoots");
                    pin_frame_at_head(gd);
                }
                gd.skirmish_hold = at;
            }
        } else {
            if gd.skirmishing {
                end_skirmish(gd);
                info!("regiment {g} ends its run (skirmish)");
            } else if hold != gd.skirmish_hold {
                match hold {
                    Some(e) => info!("regiment {g} stands and shoots at block {e} in its path"),
                    None => info!("regiment {g} walks on"),
                }
                if gd.skirmish_hold.is_none() {
                    pin_frame_at_head(gd);
                }
            }
            gd.skirmish_hold = hold;
        }
    }
}

/// An attack order for a ranged regiment is a fire order, not a melee
/// charge: in range of its target the regiment stands and shoots where
/// it is, turning nothing; out of range it marches, facing the way it
/// goes, to the firing place (`firing_place`, the march's goal through
/// `Groups::goal`), stands when its leading men arrive there
/// (`head_arrived`) and shoots; it marches again only when the frame it
/// holds is out of range. Fire at will takes the nearest enemy block in range that is
/// not in melee; when every block in range is in melee, or one is
/// coming for the regiment, its men shoot the clear men of those blocks
/// (`ShootAt::search`; the job lists them from `clear_blocks`, sim/mod.rs
/// `clear_members`). Makes the regiment writes prep
/// makes: the stand-off state with its anchor and march facing, the
/// firing place, and the `firing` flag.
fn archer_fire_solutions(
    groups: &mut Groups,
    tracks: &crate::arrows::RegTracks,
    terrain: &Terrain,
    clear_members: &[Vec<u32>],
    scans: &mut Vec<ArcherScan>,
    kept_clear_blocks: Vec<Vec<crate::formation::BlockRect>>,
    tick: u32,
) -> FireSolutions {
    let n_groups = groups.list.len();
    let vel = |t: usize| tracks.vel.get(t).copied().unwrap_or(Vec2::ZERO);
    scans.resize_with(n_groups, Default::default);

    // The stand-off of every archer regiment under an attack order.
    for g in 0..n_groups {
        let gd = &groups.list[g];
        // With swords out the order is a charge like any infantry's.
        let archer = gd.kind == crate::unit_types::KIND_ARCHER
            && !gd.melee
            && gd.count > 0
            && !gd.state.is_broken();
        let target = match gd.order {
            Some(crate::orders::Order::Attack(t)) if archer => t,
            _ => {
                let gd = &mut groups.list[g];
                gd.standoff = false;
                gd.standoff_target = None;
                gd.fire_place = None;
                continue;
            }
        };
        let td = &groups.list[target as usize];
        if td.count == 0 {
            continue;
        }
        let t_vel = vel(target as usize);
        let place = firing_place(gd, td, terrain);
        let fresh = gd.standoff_target != Some(target);
        let was = gd.standoff && !fresh;
        // A fresh order shoots where it stands when it can; a standing
        // regiment stands until the frame it holds is out of range; a
        // marching one halts where its leading men arrive at the firing
        // place, which is in range by its construction.
        let now = if fresh {
            in_bow_range(gd, td, t_vel, terrain)
        } else if was {
            frame_in_bow_range(gd, gd.anchor, td, t_vel, terrain)
        } else {
            head_arrived(gd, place) && frame_in_bow_range(gd, place, td, t_vel, terrain)
        };
        let to_target = td.centroid - gd.centroid;
        if now != gd.standoff || fresh {
            let gap = edge_gap(gd, td);
            let reach = bow_reach_toward(gd, gd.centroid, td, t_vel, terrain);
            info!(
                "regiment {g} {} its target {target} (gap {gap:.1} m, reach {reach:.1} m, \
                 place {:.1} m off, lead {:.1} m{})",
                if now { "stands off" } else { "marches at" },
                gd.centroid.distance(place),
                gd.lead,
                if fresh { ", fresh order" } else { "" },
            );
        }
        let gd = &mut groups.list[g];
        gd.standoff_target = Some(target);
        gd.standoff = now;
        gd.fire_place = Some(place);
        if now && !was {
            // Entering the stand-off pins the frame: a march pins it to
            // the firing place its men already walk to, with the slots
            // dealt in the men's present order, so nobody walks back; a
            // fresh order in range freezes it where the regiment stands
            // (the anchor snap engagement does), so the men dress where
            // they stopped instead of walking back to a stale anchor.
            if !fresh {
                gd.anchor = place;
                gd.reform = true;
            } else if gd.anchor.distance(gd.centroid) > 2.0 {
                gd.anchor = gd.centroid;
            }
        }
        if !now
            && (fresh || was)
            && gd.shape == crate::formation::FormShape::Rect
            && to_target.length() > 1.0
        {
            // The march begins: face the way we go and dress for it. In
            // range the order turns nothing.
            gd.facing = crate::formation::facing_of(to_target);
            gd.reform = true;
        }
    }

    // What each archer regiment's bows shoot at this tick, and the blocks
    // its men search in. Regiments in a real melee (the melee clock; a
    // few enemy men in reach leave the rest of the bows up) or on the
    // march don't volley (foot archers halt to shoot; fire at will pauses
    // on the move).
    let gl: &Groups = &*groups;
    let mut search_blocks: Vec<Vec<u32>> = vec![Vec::new(); n_groups];
    let shoot_at: Vec<Option<ShootAt>> = (0..n_groups)
        .map(|g| {
            let gd = &gl.list[g];
            if gd.kind != crate::unit_types::KIND_ARCHER
                || gd.melee
                || gd.count == 0
                || gd.state.is_broken()
                || gd.melee_ticks > 0
            {
                return None;
            }
            let live = |e: usize| {
                let eg = &gl.list[e];
                eg.team != gd.team && eg.count > 0 && !eg.state.is_broken()
            };
            // With fire at will on, the enemy blocks about: one coming for
            // the regiment within the approach horizon (`danger`), the
            // nearest in range that is not in melee, and those in melee.
            // The look is made every ARCHER_LOOK ticks and kept between
            // looks; a block that died since is dropped.
            let range = crate::unit_types::bow_range(gd.kind);
            let scan = &mut scans[g];
            if !gd.fire_at_will {
                *scan = ArcherScan::default();
            } else if scan.tick == u32::MAX || tick.wrapping_add(g as u32).is_multiple_of(ARCHER_LOOK)
            {
                let mut danger: Option<(f32, usize)> = None;
                let mut best_free: Option<(f32, usize)> = None;
                let mut busy: Vec<(f32, usize)> = Vec::new();
                for e in 0..n_groups {
                    if !live(e) {
                        continue;
                    }
                    let eg = &gl.list[e];
                    let Some(gap) = gap_within(gd, eg, range * 1.5) else {
                        continue;
                    };
                    if danger.is_none_or(|(bg, _)| gap < bg)
                        && approach_hits(gl, g, e, vel(e), APPROACH_S, range * 0.8)
                    {
                        danger = Some((gap, e));
                    }
                    if !in_bow_range(gd, eg, vel(e), terrain) {
                        continue;
                    }
                    if eg.engaged {
                        busy.push((gap, e));
                    } else if best_free.is_none_or(|(bg, _)| gap < bg) {
                        best_free = Some((gap, e));
                    }
                }
                *scan = ArcherScan { tick, danger, best_free, busy };
            }
            let danger = scan.danger.filter(|&(_, e)| live(e));
            let best_free = scan.best_free.filter(|&(_, e)| live(e));
            let mut busy: Vec<(f32, usize)> =
                scan.busy.iter().copied().filter(|&(_, e)| live(e)).collect();
            // A regiment standing instead of walking into an enemy block
            // shoots that block, whatever its order and fire at will say,
            // unless a block is coming for it.
            let held = gd.skirmish_hold.map(|h| h as usize).filter(|&h| live(h));
            let (t, search) = match gd.order {
                _ if danger.is_none() && held.is_some() => {
                    let h = held.unwrap_or_default();
                    (h, gl.list[h].engaged)
                }
                // On the march the bows stay down; a cornered regiment
                // shoots at will.
                Some(crate::orders::Order::Move(_)) if gd.skirmish_hold.is_none() => return None,
                // The ordered target, once standing off, unless a block is
                // coming for the regiment: the men turn their bows on that
                // one first, whatever the order. At a block in melee the
                // men shoot its clear members only.
                Some(crate::orders::Order::Attack(t))
                    if gd.skirmish_hold.is_none() && danger.is_none() =>
                {
                    let t = t as usize;
                    if !gd.standoff || !live(t) {
                        return None;
                    }
                    (t, gl.list[t].engaged)
                }
                // Fire at will: the block coming for the regiment, every
                // man on it (with skirmish on, the run takes the regiment
                // away before this); else the nearest block in range that
                // is not in melee; else the men search the blocks in
                // melee, nearest first. Fire at will off: nothing.
                _ => {
                    if let Some((_, e)) = danger {
                        (e, true)
                    } else if let Some((_, e)) = best_free {
                        (e, false)
                    } else {
                        busy.sort_by(|a, b| a.0.total_cmp(&b.0));
                        let (_, e) = *busy.first()?;
                        search_blocks[g] =
                            busy.iter().take(SEARCH_BLOCKS).map(|&(_, e)| e as u32).collect();
                        (e, true)
                    }
                }
            };
            if search && search_blocks[g].is_empty() {
                search_blocks[g].push(t as u32);
            }
            let tg = &gl.list[t];
            Some(ShootAt {
                c: tg.centroid,
                vel: vel(t),
                r: tg.radius.clamp(3.0, 25.0),
                t,
                search,
            })
        })
        .collect();
    // The card's firing mark: a fire solution, arrows to spend on it and,
    // in a search, clear men to shoot (the lists of the last tick run).
    for (g, s) in shoot_at.iter().enumerate() {
        let has_men = s.as_ref().is_some_and(|s| {
            !s.search
                || search_blocks[g]
                    .iter()
                    .any(|&b| clear_members.get(b as usize).is_some_and(|m| !m.is_empty()))
        });
        groups.list[g].firing = has_men && groups.list[g].ammo_left > 0;
    }
    // Regiments under fire this tick, and those under search. The job
    // lists their living members (`target_members`, built on the worker
    // from its shared columns) and the searched ones' clear members
    // (`clear_members`): each shot aims at an actual soldier, not at a
    // spot on the block's footprint.
    let mut targeted = vec![false; n_groups];
    let mut searched = vec![false; n_groups];
    for (g, s) in shoot_at.iter().enumerate() {
        let Some(s) = s else { continue };
        targeted[s.t] = true;
        if s.search {
            for &b in &search_blocks[g] {
                targeted[b as usize] = true;
                searched[b as usize] = true;
            }
        }
    }
    // The blocks of the shooters' side within a shaft's reach of each
    // searched block, by the gap between the edges, as rectangles: a man
    // of the searched block farther than that reach from every one of
    // them, with no enemy in his own reach, is clear (sim/mod.rs
    // `clear_members`). A block with a wider gap has no point of its
    // footprint within the reach of any man inside the searched block.
    // Broken blocks count too: the rectangle sits on the men's centroid.
    // Listed every ARCHER_LOOK ticks per searched block, staggered, and
    // kept between (the last tick's list, or none for a block newly
    // under search).
    let clear_r = clear_reach();
    let mut kept = kept_clear_blocks;
    kept.resize_with(n_groups, Vec::new);
    let clear_blocks: Vec<Vec<crate::formation::BlockRect>> = kept
        .into_iter()
        .enumerate()
        .map(|(e, old)| {
            if !searched[e] {
                return Vec::new();
            }
            if !old.is_empty() && !tick.wrapping_add(e as u32).is_multiple_of(ARCHER_LOOK) {
                return old;
            }
            let eg = &groups.list[e];
            groups
                .list
                .iter()
                .filter(|fg| {
                    fg.team != eg.team
                        && fg.count > 0
                        && gap_within(fg, eg, clear_r).is_some_and(|gap| gap <= clear_r)
                })
                .map(crate::formation::block_rect)
                .collect()
        })
        .collect();
    // Friendly blocks per team for the loft-over-friendlies check: every
    // live formed regiment as a disc with a clearance ceiling. A
    // shooter's own regiment is in here too — that is what makes rear
    // ranks loft over their own front rank. The ceiling is 1.3 m over the
    // heads at the disc's centre.
    let ceiling = crate::unit_types::tallest() + 1.3;
    let blocks: [Vec<(Vec2, f32, f32)>; 2] = [0u8, 1u8].map(|tm| {
        groups
            .list
            .iter()
            .filter(|g| g.team == tm && g.count > 0 && !g.state.is_broken())
            .map(|g| {
                (
                    g.centroid,
                    g.radius.max(3.0),
                    terrain.height_at(g.centroid.x, g.centroid.y) + ceiling,
                )
            })
            .collect()
    });

    FireSolutions {
        shoot_at,
        targeted,
        searched,
        search_blocks,
        clear_blocks,
        blocks,
    }
}

/// How far a shaft aimed at a man can come down from him: the corner of
/// the aim's scatter box, plus the ground a shaft falling at 45 degrees
/// covers while it crosses the strike band (arrows.rs). A man with no
/// man of the shooters' side within it is one a shaft can be aimed at
/// without landing on a friend of the shooter.
pub(crate) fn clear_reach() -> f32 {
    crate::unit_types::missile::AIM_SPREAD * std::f32::consts::SQRT_2
        + crate::unit_types::tallest()
        + crate::arrows::STRIKE_MARGIN
}

/// Why each archer regiment's men take their knives out this tick
/// (`Sidearm`): its swords-out order, an enemy block in bow range for the
/// men with empty quivers, and the unbroken enemy blocks within the knife
/// distance of its own block. The gap between two blocks along the line
/// between their centres is never more than the distance between any man
/// of one and any point of the other, so no block left out is within the
/// knife distance of a man inside his block. A man outside it is
/// fighting, which his own reach covers, or dressing back into it.
fn sidearm_reasons(
    groups: &Groups,
    tracks: &crate::arrows::RegTracks,
    terrain: &Terrain,
    last: &[Sidearm],
    tick: u32,
) -> Vec<Sidearm> {
    let vel = |t: usize| tracks.vel.get(t).copied().unwrap_or(Vec2::ZERO);
    groups
        .list
        .iter()
        .enumerate()
        .map(|(g, gd)| {
            let mut arm = Sidearm::default();
            if gd.kind != crate::unit_types::KIND_ARCHER || gd.count == 0 {
                return arm;
            }
            // The look at the blocks about, every ARCHER_LOOK ticks; the
            // swords-out order is read every tick.
            if let Some(kept) = last.get(g)
                && !tick.wrapping_add(g as u32).is_multiple_of(ARCHER_LOOK)
            {
                arm = kept.clone();
                arm.swords = gd.melee;
                return arm;
            }
            arm.swords = gd.melee;
            // Blocks beyond the fire solutions' scope (1.5 ranges) are
            // out of the knife's reach and never shot at.
            let scope = (1.5 * crate::unit_types::bow_range(gd.kind)).max(crate::sim::soldier::KNIFE_EDGE);
            for (e, eg) in groups.list.iter().enumerate() {
                if eg.team == gd.team || eg.count == 0 || eg.state.is_broken() {
                    continue;
                }
                let Some(gap) = gap_within(gd, eg, scope) else {
                    continue;
                };
                if gap <= crate::sim::soldier::KNIFE_EDGE {
                    arm.blocks.push(crate::formation::block_rect(eg));
                }
                arm.foe_in_range = arm.foe_in_range
                    || gap <= bow_reach_toward(gd, gd.centroid, eg, vel(e), terrain);
            }
            arm
        })
        .collect()
}

/// Each regiment's order resolved to this tick's destination, or none
/// for a regiment that holds where it is. A regiment on a route heads
/// for its current waypoint (sim/route.rs).
fn resolve_orders(groups: &Groups, routes: &crate::sim::route::Routes) -> Vec<Option<Vec2>> {
    // Orders resolved to this tick's destination (attack orders chase
    // their target regiment's current centroid). A regiment holding a
    // contact frame stops chasing: dragging the slot grid onward
    // through a moving enemy centroid is what smeared blocks into
    // blobs. Move orders never freeze — pulling a regiment out of
    // melee is the player's call.
    let orders: Vec<Option<Vec2>> = (0..groups.list.len())
        .map(|g| {
            let gd = &groups.list[g];
            // Archer stand-off: in range of the ordered target — stand
            // and shoot (the order survives; the chase resumes if the
            // target moves out of range). A regiment that would walk into
            // an enemy block stands and shoots as well (`skirmish_hold`).
            if gd.standoff || gd.skirmish_hold.is_some() {
                return None;
            }
            // Contact frame (frontline.rs): the regiment holds its slots
            // around the frame instead of chasing the target's center.
            if gd.contact {
                return None;
            }
            groups.goal(g).map(|goal| routes.frame(g, goal))
        })
        .collect();
    orders
}

/// The per-regiment state the kernel reads, snapshotted into the job,
/// plus the per-soldier flags derived from it.
fn snapshot_regiments(job: &mut TickJob, groups: &Groups, terrain: &Terrain) {
    // Per-regiment wall flag for the grid meta (same-team wall pairs pack
    // tighter in the separation); the rebuild reads it through the
    // regiment index it already carries per man.
    let group_wall: Vec<bool> = groups
        .list
        .iter()
        .map(|g| crate::formation::wall_kind(g) != 0)
        .collect();
    let group_broken: Vec<bool> = groups.list.iter().map(|g| g.state.is_broken()).collect();
    let anchors: Vec<Vec2> = groups.list.iter().map(|g| g.anchor).collect();
    // Regiments under a Move order: the order takes them out of any fight
    // (M2TW's WITHDRAW, devlog 0120). Their men walk where they were sent,
    // pushing through what blocks them and striking only whoever stands in
    // front of them; they close on no enemy and go after none they see. A
    // regiment breaking off for a new attack target does the same on its
    // way there (orders.rs `retarget`), until its melee with the target
    // starts.
    let moving: Vec<bool> = groups
        .list
        .iter()
        .map(|g| {
            matches!(g.order, Some(crate::orders::Order::Move(_)))
                || (g.retarget && g.melee_ticks == 0)
        })
        .collect();
    // Once that melee starts, its men fight the target alone: they strike,
    // close on and go after only its men, and leave the old enemy be.
    let focus: Vec<Option<u32>> = groups
        .list
        .iter()
        .map(|g| match g.order {
            Some(crate::orders::Order::Attack(t)) if g.retarget && g.melee_ticks > 0 => Some(t),
            _ => None,
        })
        .collect();
    // Regiments in combat-watch range of an enemy (sparse-fight
    // acquisition). HOLD regiments never acquire wide: they stand their
    // ground and take what comes; moving ones go where they were sent.
    let press: Vec<bool> = groups
        .list
        .iter()
        .zip(&moving)
        .map(|(g, &m)| (g.enemy_near || g.engaged) && !g.hold && !m)
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
    // The crash of a charge keeps the charge pace (frontline.rs).
    let charging: Vec<bool> = groups.list.iter().map(|g| g.charging || g.crashing).collect();
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
    // The spear-line hazard reads the SPEARMAN's facing from the
    // charger's side of the scan: the shared yaw column at the tick's
    // start serves as that read-only snapshot.
    job.bounds_min = bounds_min;
    job.bounds_max = bounds_max;
    job.faces_spearwall = faces_spearwall;
    job.anchors = anchors;
    job.reg_broken = group_broken;
    job.press = press;
    job.moving = moving;
    job.stance = groups.list.iter().map(stance_of).collect();
    job.focus = focus;
    job.engaged = engaged;
    job.contact = contact;
    job.fight_point = fight_point;
    job.melee_ticks = melee_ticks;
    job.hold = hold;
    job.threat = threat;
    job.form_face = form_face;
    job.wall = wall;
    job.group_wall = group_wall;
    job.charging = charging;
    job.fat_speed = fat_speed;
    job.fat_nocharge = fat_nocharge;
}
