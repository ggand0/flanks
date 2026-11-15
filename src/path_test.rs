//! FL_TEST_PATH: the pathfinding scenes. Each places regiments, orders
//! them through `order_regiments` as a click would, on a fixed sim tick,
//! and logs every two seconds how they get there.
//!
//! - `FL_TEST_PATH=<case>` with `FL_MAP=obstacles`: one regiment of 200
//!   men for that case of the obstacle map (obstacle_map.rs `CASES`:
//!   cube, wide, narrow, pocket, zigzag, box, pillars, shared; the shared
//!   gap gets a second regiment coming the other way). `all` runs every
//!   case at once.
//! - `FL_TEST_PATH=many` with `FL_MAP=obstacles`: 200 regiments of 40
//!   men south of the long wall, all ordered north through its gaps in
//!   the same tick; logs the plans and the longest frame each second.
//! - `FL_TEST_PATH=river` with `FL_MAP=river`: one regiment ordered
//!   across near the bridge and one far from it; logs whether each
//!   route takes the bridge or the ford.
//! - `FL_TEST_PATH=walls` with `FL_MAP=walled_town` or `FL_MAP=wall_debug`:
//!   a regiment of 40 in the plaza given a line along the walk west of the
//!   gatehouse, as a player mans a wall, which it can reach only up a
//!   stair; after that verdict a line along the walk east of the tower at
//!   x = 80, so it walks over the gatehouse's bridge and through a tower;
//!   and a regiment of 200 on the plain ordered to the plaza, which the
//!   closed gate leaves only the way around an open end of the wall. Logs
//!   which stair, which end.
//! - `FL_TEST_PATH=walls_archers` with `FL_MAP=walled_town` or
//!   `FL_MAP=wall_debug`: a regiment of 60 archers in the plaza given a
//!   line along the walk east of the gate, and 300 enemy men on the slope
//!   below it, in bow range of the walk, ordered up to the wall's foot: the
//!   archers shoot down from the wall, and the enemy's front rank stops
//!   where the blocked ground begins. The archers' skirmish is off, as a
//!   player sets it to hold a wall: the run measures the enemy's distance
//!   straight through the wall.
//! - `FL_TEST_PATH=walls_place` with `FL_MAP=walled_town` or
//!   `FL_MAP=wall_debug`: placing units on the walk from the plateau behind
//!   the wall (through the town's streets on the walled town). 200 archers
//!   given a 40 m line along the walk east of the gate; 200 men the same
//!   line, who must stand beside the archers; 200 more clicked onto the
//!   walk west of the gate. Logs each one's places: how many files, and how
//!   many places on ground a man cannot stand on.
//! - `FL_TEST_PATH=walls_under` with `FL_MAP=walled_town` or
//!   `FL_MAP=wall_debug`: 200 archers at the wall's inner foot east of a
//!   stair, where the defenders deploy, given a line along the walk above
//!   and west of them, so the whole regiment has to walk back along the
//!   foot to the stair.
//! - `FL_TEST_PATH=walls_cross` with `FL_MAP=walled_town` or
//!   `FL_MAP=wall_debug`: two regiments of 200 archers at the wall's inner
//!   foot on either side of the stair east of the gate, each given a line
//!   along the walk on the far side of the stair from where it stands, so
//!   both climb the same stair and cross on it and on the walk. Its log
//!   adds, per regiment, the men who have never stood on the walk and the
//!   waypoint its route has reached, so a route stalled at the stair's
//!   foot, men who never got up and men up but not at their places read
//!   apart. It has 300 s to arrive.
//! - `FL_TEST_PATH=walls_long` with `FL_MAP=walled_town` or
//!   `FL_MAP=wall_debug`: one regiment of 500 archers deployed as one long
//!   line at the wall's inner foot, across both stairs east of the gate,
//!   given a short line on the walk above it, so its men come to one stair
//!   from both sides and fill a run of the walk on both sides of that
//!   stair's head. It has 300 s to arrive.
//! - `FL_TEST_PATH=walls_two_long` with `FL_MAP=walled_town` or
//!   `FL_MAP=wall_debug`: the long scene's regiment, and a second of 500
//!   archers 50 m behind the wall, given a short line on the walk west of
//!   the gate right after the first. A regiment the walk has no room for
//!   gets no order, and the log states it.
//!
//! Per regiment the log has the share of men within 5 m of their slots,
//! the men stuck (slower than 0.1 m/s for 10 s while more than 5 m from
//! their slot), the route's length against the straight line, the
//! waypoint the route has reached, and a verdict once 99% have arrived:
//! the time taken against twice the time the route's length takes at the
//! men's pace. A regiment fails when it has not arrived by its scene's
//! timeout (`timeout_s`) or three times the route's length at its pace,
//! whichever is longer. On the wall scenes the log also counts the men on
//! the walk and the men who have never stood on it, and every ten seconds,
//! after the verdict too, per regiment:
//!
//! - where its men stand (on the walk, a stair, a stair's foot, the wall's
//!   inner face) and how many moved under a metre in those ten seconds;
//! - where its route's frame waits, and the two tests that would move it
//!   on: the centre within the regiment's half width of the waypoint, or
//!   the straight way to the next point clear at that width;
//! - its order, anchor, facing and spacing, the point the sim lays its
//!   slots round, and how many of the places round it are off the walk;
//! - the men out, each with the slot and the aim the sim gives him.
//!
//! `FL_TEST_DUMP=<file>` writes every watched man's position, height
//! included, with his place, the ground under him, his velocity and his
//! facing, every five seconds (`dump_men`).
//!
//! `FL_LOG_ROUTE=1` adds the route events from sim/route.rs: an order's end
//! and whether its route stays, a holding regiment's route dropped, and
//! each move of a frame to a later waypoint.

use bevy::prelude::*;

use crate::obstacle_map::CASES;
use crate::orders::{Groups, Order};
use crate::terrain::Terrain;
use crate::town_layout::spots;
use crate::unit_types::{KIND_ARCHER, KIND_HEAVY};
use crate::units::Units;

/// The sim tick the orders go out on (3 s in).
const ORDER_TICK: u32 = 90;
/// Men per regiment in the case scenes, and in the 200-regiment scene.
const CASE_MEN: usize = 200;
const MANY_MEN: usize = 40;
const MANY_REGS: usize = 200;
const WALK_MEN: usize = 40;
/// Where the archers' target stands: on the slope 70 m out from the wall.
const ARCHERS_TARGET: Vec2 = Vec2::new(60.0, crate::town_layout::WALL_Z + 70.0);
/// Seconds after the order a regiment has to arrive before it fails,
/// on every scene but the crossing (`timeout_s`).
const TIMEOUT_S: f32 = 180.0;
/// The crossing scene's timeout: a pass takes about two minutes, and a
/// regiment stalled on the stair is watched for a while longer.
const CROSS_TIMEOUT_S: f32 = 300.0;
/// The crossing scene's starts, by the stair east of the gate (its foot
/// at x = 36): 40 m west of its foot and 14 m behind the wall's line,
/// (-4, -144); east of its end and 18 m behind the line, (68, -148).
const CROSS_WEST_START: Vec2 = Vec2::new(spots::STAIR_FOOT_EAST.x - 40.0, crate::town_layout::WALL_Z - 14.0);
const CROSS_EAST_START: Vec2 = Vec2::new(spots::STAIR_FOOT_EAST.x + 32.0, crate::town_layout::WALL_Z - 18.0);
/// Its lines along the walk, each on the far side of the stair from its
/// regiment: the west one's from x = 62 to 102, the east one's from
/// x = -14 to 26.
const CROSS_EAST_LINE: (f32, f32) = (spots::STAIR_FOOT_EAST.x + 26.0, spots::STAIR_FOOT_EAST.x + 66.0);
const CROSS_WEST_LINE: (f32, f32) = (spots::STAIR_FOOT_EAST.x - 50.0, spots::STAIR_FOOT_EAST.x - 10.0);
/// The long scene: 500 men in 125 files, the line's middle 13 m behind
/// the wall's line and 19 m east of the stair's foot, and its order, a
/// drag of 8 m on the walk near the stair east of that one.
const LONG_MEN: usize = 500;
const LONG_FILES: u32 = 125;
const LONG_START: Vec2 = Vec2::new(spots::STAIR_FOOT_EAST.x + 19.0, crate::town_layout::WALL_Z - 13.1);
const LONG_LINE: (f32, f32) = (spots::STAIR_FOOT_EAST.x + 64.0, spots::STAIR_FOOT_EAST.x + 72.0);
/// The second regiment of the two long ones: 50 m east of the first's
/// stair and 51.5 m behind the wall's line, and its order, a drag of 38 m
/// westward on the walk from 39.3 m west of the gate.
const LONG2_START: Vec2 = Vec2::new(spots::STAIR_FOOT_EAST.x + 50.4, crate::town_layout::WALL_Z - 51.5);
const LONG2_LINE: (f32, f32) = (-39.3, -77.3);
/// Within this of his slot a man has arrived; beyond it, slower than
/// `STUCK_SPEED` for `STUCK_S`, he is stuck.
const ARRIVED_M: f32 = 5.0;
const STUCK_SPEED: f32 = 0.1;
const STUCK_S: f32 = 10.0;

/// The scene asked for, if any.
pub fn scene() -> Option<String> {
    std::env::var("FL_TEST_PATH").ok()
}

/// One regiment under watch.
struct Subject {
    g: usize,
    /// The sim tick of its order.
    tick: u32,
    case: &'static str,
    goal: Vec2,
    straight: f32,
    route_len: Option<f32>,
    route_note: String,
    done_s: Option<f32>,
    peak_stuck: usize,
    /// The time it has to arrive (s): the scene's timeout until the route
    /// is known, then that or three times the route at its pace.
    limit: f32,
    reported: bool,
    /// Where its centre stood at the verdict, to see that it then stays.
    settled_at: Option<Vec2>,
}

#[derive(Resource, Default)]
struct PathTest {
    subjects: Vec<Subject>,
    /// Per unit, ticks spent stuck so far.
    stuck_ticks: Vec<u32>,
    /// Per unit, whether he has stood on the wall's walk (wall scenes).
    been_on_walk: Vec<bool>,
    /// The file of `FL_TEST_DUMP`, once opened, and whether it was tried.
    dump: Option<std::fs::File>,
    dump_tried: bool,
    /// Per unit, where he stood at the last ten-second line (wall scenes),
    /// to count the men who really moved: a man pressing on a wall keeps
    /// a velocity while he stays where he is.
    pos_10s: Vec<Vec2>,
    ordered: bool,
    summary_done: bool,
    /// The 200-regiment scene: when every regiment had its route.
    many_routed_tick: Option<u32>,
    /// The walls scene's second order, given once its regiment's first
    /// has its verdict: the regiment, the case and the line.
    then: Option<(usize, &'static str, (Vec2, Vec2))>,
}

/// The longest frame in the current second and since the orders.
#[derive(Resource, Default)]
struct FrameWatch {
    window_max: f32,
    since_order_max: f32,
}

pub struct PathTestPlugin;

impl Plugin for PathTestPlugin {
    fn build(&self, app: &mut App) {
        if scene().is_none() {
            return;
        }
        app.init_resource::<PathTest>()
            .init_resource::<FrameWatch>()
            // The orders go out between the install and the next job,
            // where the pipelined tick's kick and the inline tick's next
            // prep both read them.
            .add_systems(
                FixedUpdate,
                (give_orders, log_progress)
                    .chain()
                    .after(crate::orders::clear_arrived_orders)
                    .before(crate::sim::route::update_routes)
                    .in_set(crate::game_state::SimSet),
            )
            .add_systems(Update, watch_frames);
    }
}

/// Place the scene's regiments. A small enemy regiment holds the far
/// north-east corner so the battle is not won at once.
pub fn spawn(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let scene = scene().unwrap_or_default();
    let mut list = Vec::new();
    // The wall scenes run on either map with the wall.
    let wanted_map = match scene.as_str() {
        "river" => Some(crate::terrain::MapKind::River),
        "walls" | "walls_archers" | "walls_place" | "walls_under" | "walls_cross" | "walls_long" | "walls_two_long" => None,
        _ => Some(crate::terrain::MapKind::Obstacles),
    };
    match wanted_map {
        Some(wanted) if terrain.kind != wanted => {
            warn!("[path-test] scene {scene} expects FL_MAP={:?}, the map is {:?}", wanted, terrain.kind);
        }
        None if !terrain.kind.has_wall() => {
            warn!("[path-test] scene {scene} expects FL_MAP=walled_town or wall_debug, the map is {:?}", terrain.kind);
        }
        _ => {}
    }
    let mut add = |list: &mut Vec<_>, at: Vec2, men: usize| {
        crate::regiments::spawn_regiment(units, terrain, list, 0, KIND_HEAVY, at, men, -1.0);
    };
    match scene.as_str() {
        "many" => {
            // Four rows of fifty, 20 m apart along x, in the south strip.
            for r in 0..MANY_REGS {
                let row = r / 50;
                let col = r % 50;
                let at = Vec2::new(-490.0 + col as f32 * 20.0, -370.0 + row as f32 * 16.0);
                add(&mut list, at, MANY_MEN);
            }
        }
        "river" => {
            let (near, far) = river_starts();
            add(&mut list, near, CASE_MEN);
            add(&mut list, far, CASE_MEN);
        }
        "walls" => {
            add(&mut list, spots::PLAZA - Vec2::new(0.0, 10.0), WALK_MEN);
            add(&mut list, spots::PLAIN, CASE_MEN);
        }
        "walls_place" => {
            // On the open plateau behind the town, so all three march
            // through its streets to the wall.
            crate::regiments::spawn_regiment(units, terrain, &mut list, 0, KIND_ARCHER, spots::BEHIND_TOWN, CASE_MEN, -1.0);
            for x in [90.0, -90.0] {
                let at = spots::BEHIND_TOWN + Vec2::new(x, 0.0);
                crate::regiments::spawn_regiment(units, terrain, &mut list, 0, KIND_HEAVY, at, CASE_MEN, -1.0);
            }
            list[0].skirmish = false;
        }
        "walls_under" => {
            let at = Vec2::new(spots::STAIR_FOOT_EAST.x + 36.0, crate::town_layout::WALL_Z - 12.0);
            crate::regiments::spawn_regiment(units, terrain, &mut list, 0, KIND_ARCHER, at, CASE_MEN, -1.0);
            list[0].skirmish = false;
        }
        "walls_cross" => {
            for at in [CROSS_WEST_START, CROSS_EAST_START] {
                crate::regiments::spawn_regiment(units, terrain, &mut list, 0, KIND_ARCHER, at, CASE_MEN, -1.0);
            }
            list[0].skirmish = false;
            list[1].skirmish = false;
        }
        "walls_long" | "walls_two_long" => {
            crate::regiments::spawn_regiment(units, terrain, &mut list, 0, KIND_ARCHER, LONG_START, LONG_MEN, -1.0);
            list[0].files = LONG_FILES;
            crate::formation::snap_to_slots(units, terrain, 0, &mut list[0]);
            list[0].skirmish = false;
            if scene == "walls_two_long" {
                crate::regiments::spawn_regiment(units, terrain, &mut list, 0, KIND_ARCHER, LONG2_START, LONG_MEN, -1.0);
                list[1].skirmish = false;
            }
        }
        "walls_archers" => {
            crate::regiments::spawn_regiment(units, terrain, &mut list, 0, KIND_ARCHER, spots::PLAZA, 60, -1.0);
            crate::regiments::spawn_regiment(units, terrain, &mut list, 1, KIND_HEAVY, ARCHERS_TARGET, 300, 1.0);
            list[0].skirmish = false;
            list[1].hold = true;
        }
        _ => {
            for case in CASES.iter().filter(|c| scene == "all" || c.name == scene) {
                add(&mut list, case.start, CASE_MEN);
                if case.name == "shared" {
                    add(&mut list, case.goal, CASE_MEN);
                }
            }
        }
    }
    let foe = list.len();
    crate::regiments::spawn_regiment(units, terrain, &mut list, 1, KIND_HEAVY, Vec2::new(450.0, 340.0), 20, 1.0);
    list[foe].hold = true;
    info!("[path-test] scene {scene}: {} regiments", list.len() - 1);
    groups.list = list;
}

/// The river scene's starts: west of the river, one 15 m south of the
/// bridge's line, one far to the south. Each is sent straight east across,
/// 120 m.
fn river_starts() -> (Vec2, Vec2) {
    let near_z = crate::terrain::BRIDGE_Z - 15.0;
    let far_z = -150.0;
    (
        Vec2::new(crate::terrain::river_center_x(near_z) - 60.0, near_z),
        Vec2::new(crate::terrain::river_center_x(far_z) - 60.0, far_z),
    )
}

/// Give every regiment its order on `ORDER_TICK`.
fn give_orders(
    pipeline: Res<crate::sim::TickPipeline>,
    mut groups: ResMut<Groups>,
    mut units: ResMut<Units>,
    mut test: ResMut<PathTest>,
    terrain: Res<Terrain>,
) {
    // The second order of a regiment whose first has its verdict.
    if let Some((g, case, line)) = test.then
        && test.subjects.iter().any(|s| s.g == g && s.reported)
    {
        test.then = None;
        info!("[path-test] tick {}: {case} ordered", pipeline.tick);
        crate::orders::line_order(&mut groups, &terrain, &[g], line.0, line.1);
        order(&mut groups, &mut units, &terrain, &mut test, pipeline.tick, g, case, None);
    }
    if test.ordered || pipeline.tick < ORDER_TICK || groups.list.is_empty() {
        return;
    }
    test.ordered = true;
    let scene = scene().unwrap_or_default();
    let mut orders: Vec<(usize, &'static str, Vec2)> = Vec::new();
    match scene.as_str() {
        "many" => {
            for g in 0..MANY_REGS.min(groups.list.len() - 1) {
                let c = groups.list[g].centroid;
                orders.push((g, "many", c + Vec2::new(0.0, 150.0)));
            }
        }
        "river" => {
            orders.push((0, "river-near", groups.list[0].centroid + Vec2::new(120.0, 0.0)));
            orders.push((1, "river-far", groups.list[1].centroid + Vec2::new(120.0, 0.0)));
        }
        "walls_place" => {
            let line = (spots::WALK_EAST - Vec2::new(110.0, 0.0), spots::WALK_EAST - Vec2::new(70.0, 0.0));
            for (g, case) in [(0, "place-archers"), (1, "place-beside")] {
                crate::orders::line_order(&mut groups, &terrain, &[g], line.0, line.1);
                order(&mut groups, &mut units, &terrain, &mut test, pipeline.tick, g, case, None);
            }
            orders.push((2, "place-click", spots::WALK_WEST - Vec2::new(30.0, 0.0)));
        }
        "walls_under" => {
            let c = Vec2::new(spots::STAIR_FOOT_EAST.x - 30.0, crate::town_layout::WALL_Z);
            crate::orders::line_order(&mut groups, &terrain, &[0], c - Vec2::new(20.0, 0.0), c + Vec2::new(20.0, 0.0));
            order(&mut groups, &mut units, &terrain, &mut test, pipeline.tick, 0, "walls-under", None);
        }
        "walls_cross" => {
            let z = crate::town_layout::WALL_Z;
            for (g, case, (a, b)) in [(0, "cross-east", CROSS_EAST_LINE), (1, "cross-west", CROSS_WEST_LINE)] {
                crate::orders::line_order(&mut groups, &terrain, &[g], Vec2::new(a, z), Vec2::new(b, z));
                order(&mut groups, &mut units, &terrain, &mut test, pipeline.tick, g, case, None);
            }
        }
        "walls_long" | "walls_two_long" => {
            let z = crate::town_layout::WALL_Z;
            let mut lines = vec![(0, "walls-long", LONG_LINE)];
            if scene == "walls_two_long" {
                lines.push((1, "walls-long-second", LONG2_LINE));
            }
            for (g, case, (a, b)) in lines {
                crate::orders::line_order(&mut groups, &terrain, &[g], Vec2::new(a, z), Vec2::new(b, z));
                if groups.list[g].order.is_some() {
                    order(&mut groups, &mut units, &terrain, &mut test, pipeline.tick, g, case, None);
                } else {
                    info!("[path-test] {case}: no order given, the walk has no room for regiment {g}");
                }
            }
        }
        "walls_archers" => {
            // 50 m: at the archers' spacing, three ranks, which the walk
            // holds.
            let c = spots::WALK_EAST - Vec2::new(80.0, 0.0);
            crate::orders::line_order(&mut groups, &terrain, &[0], c - Vec2::new(25.0, 0.0), c + Vec2::new(25.0, 0.0));
            order(&mut groups, &mut units, &terrain, &mut test, pipeline.tick, 0, "walls-archers", None);
            orders.push((1, "walls-foot", ARCHERS_TARGET.with_y(crate::town_layout::WALL_Z + 12.0)));
        }
        "walls" => {
            // A drag along the walk: the walk's own places, facing out.
            let line = |c: Vec2| (c + Vec2::new(-15.0, 0.0), c + Vec2::new(15.0, 0.0));
            let (a, b) = line(spots::WALK_WEST);
            crate::orders::line_order(&mut groups, &terrain, &[0], a, b);
            order(&mut groups, &mut units, &terrain, &mut test, pipeline.tick, 0, "walls-climb", None);
            orders.push((1, "walls-round", spots::PLAZA));
            test.then = Some((0, "walls-walk", line(spots::WALK_EAST)));
        }
        _ => {
            let mut g = 0;
            for case in CASES.iter().filter(|c| scene == "all" || c.name == scene) {
                orders.push((g, case.name, case.goal));
                g += 1;
                if case.name == "shared" {
                    orders.push((g, "shared-back", case.start));
                    g += 1;
                }
            }
        }
    }
    for &(g, case, goal) in &orders {
        order(&mut groups, &mut units, &terrain, &mut test, pipeline.tick, g, case, Some(goal));
    }
    info!("[path-test] tick {}: {} regiments ordered", pipeline.tick, test.subjects.len());
}

/// Order regiment `g` to `goal` as a click would and watch it as `case`;
/// with no goal, watch the order it was just given.
#[allow(clippy::too_many_arguments)]
fn order(
    groups: &mut Groups,
    units: &mut Units,
    terrain: &Terrain,
    test: &mut PathTest,
    tick: u32,
    g: usize,
    case: &'static str,
    goal: Option<Vec2>,
) {
    if let Some(goal) = goal {
        crate::orders::click_order(groups, terrain, &[g], goal);
    }
    let goal = match groups.list[g].order {
        Some(Order::Move(p)) => p,
        _ => groups.list[g].centroid,
    };
    // The new facing's slots, written now rather than at the next
    // tick's reform pass, so both kinds of tick read them in the same
    // job.
    if groups.list[g].reform {
        crate::formation::assign_slots(units, g as u32, &mut groups.list[g]);
        let gd = &groups.list[g];
        let (mut off, mut lo, mut hi) = (0, Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
        for i in (0..units.len()).filter(|&i| units.group[i] == g as u32 && units.death_t[i] == 0) {
            let p = goal + units.home[i];
            off += usize::from(terrain.blocked_at(p.x, p.y));
            lo = lo.min(p);
            hi = hi.max(p);
        }
        info!(
            "[path-test] {case}: {} men in {} files, {off} places on blocked ground, places x {:.1} to {:.1}, z {:.1} to {:.1}",
            gd.count, gd.files, lo.x, hi.x, lo.y, hi.y
        );
    }
    let start = groups.list[g].centroid;
    test.subjects.push(Subject {
        g,
        tick,
        case,
        goal,
        straight: start.distance(goal),
        route_len: None,
        route_note: String::new(),
        done_s: None,
        peak_stuck: 0,
        limit: timeout_s(&scene().unwrap_or_default()),
        reported: false,
        settled_at: None,
    });
}

/// Every two seconds: per regiment, the share arrived and the men stuck;
/// a verdict line when it has arrived or timed out, and a summary.
fn log_progress(
    pipeline: Res<crate::sim::TickPipeline>,
    groups: Res<Groups>,
    units: Res<Units>,
    routes: Res<crate::sim::route::Routes>,
    terrain: Res<Terrain>,
    mut test: ResMut<PathTest>,
    mut frames: ResMut<FrameWatch>,
) {
    if !test.ordered {
        return;
    }
    let tick = pipeline.tick;
    let secs = (tick.saturating_sub(ORDER_TICK)) as f32 / 30.0;
    let test = &mut *test;
    if test.stuck_ticks.len() != units.len() {
        test.stuck_ticks = vec![0; units.len()];
    }
    if test.been_on_walk.len() != units.len() {
        test.been_on_walk = vec![false; units.len()];
    }
    let flat = |i: usize| Vec2::new(units.pos[i].x, units.pos[i].z);
    if test.pos_10s.len() != units.len() {
        test.pos_10s = (0..units.len()).map(flat).collect();
    }
    let scene = scene().unwrap_or_default();
    let walls = scene.starts_with("walls");
    let walk_lane = walls.then(|| crate::town_layout::walk_lane(terrain.body_radius()));
    let ten_s = tick.is_multiple_of(300);
    // On the wall scenes the ten-second lines go on after the verdict, for
    // the latest subject of each regiment only (the walls scene orders its
    // first regiment twice).
    let latest: Vec<bool> = (0..test.subjects.len())
        .map(|j| !test.subjects[j + 1..].iter().any(|t| t.g == test.subjects[j].g))
        .collect();
    // The slot each man is going to: his regiment's ordered point (or the
    // anchor once it arrived) plus his offset.
    let target = |g: usize| match groups.list[g].order {
        Some(Order::Move(p)) => p,
        _ => groups.list[g].anchor,
    };
    let mut men = vec![Tally::default(); groups.list.len()];
    for i in 0..units.len() {
        if units.death_t[i] != 0 {
            continue;
        }
        let g = units.group[i] as usize;
        let p = Vec2::new(units.pos[i].x, units.pos[i].z);
        let d = p.distance(target(g) + units.home[i]);
        let v = Vec2::new(units.vel[i].x, units.vel[i].z).length();
        if d > ARRIVED_M && v < STUCK_SPEED {
            test.stuck_ticks[i] += 1;
        } else {
            test.stuck_ticks[i] = 0;
        }
        let e = &mut men[g];
        e.alive += 1;
        e.arrived += (d <= ARRIVED_M) as usize;
        e.stuck += (test.stuck_ticks[i] as f32 >= STUCK_S * 30.0) as usize;
        if let Some(lane) = &walk_lane {
            let on = lane.contains(p);
            test.been_on_walk[i] |= on;
            e.walk += usize::from(on);
            e.never += usize::from(!test.been_on_walk[i]);
        }
    }
    if scene == "many" {
        // From the tick after the orders, once the route pass has seen them.
        if test.many_routed_tick.is_none() && tick > ORDER_TICK {
            let regs = MANY_REGS.min(groups.list.len() - 1);
            let waiting = (0..regs).filter(|&g| routes.pending(g)).count();
            if waiting == 0 {
                test.many_routed_tick = Some(tick);
                let routed = (0..regs).filter(|&g| routes.route_of(g).is_some()).count();
                info!(
                    "[path-test] many: no regiment waits for its plan {:.2} s after the order; {routed} have routes, {} walk straight",
                    secs,
                    regs - routed
                );
            }
        }
        if tick.is_multiple_of(30) {
            info!(
                "[path-test] many t={secs:.0}s: longest frame this second {:.1} ms, since the order {:.1} ms",
                frames.window_max, frames.since_order_max
            );
            frames.window_max = 0.0;
        }
        return;
    }
    for (j, s) in test.subjects.iter_mut().enumerate() {
        let secs = (tick.saturating_sub(s.tick)) as f32 / 30.0;
        let watch = !s.reported || (walls && latest[j]);
        let gd = &groups.list[s.g];
        // From the tick after the order, once the route pass has seen it.
        if s.route_len.is_none() && tick > s.tick && !routes.pending(s.g) {
            if let Some((points, _)) = routes.route_of(s.g) {
                let len: f32 = points.windows(2).map(|w| w[0].distance(w[1])).sum();
                s.route_len = Some(len);
                s.route_note = route_note(points, &scene);
            } else if secs > 1.0 {
                s.route_len = Some(s.straight);
                s.route_note = "no route, the straight way is clear".into();
            }
            if let Some(len) = s.route_len {
                s.limit = timeout_s(&scene).max(3.0 * len / regiment_pace(&units, s.g).max(0.1));
            }
        }
        let Tally { alive, arrived, stuck, walk, never } = men[s.g];
        let g = s.g;
        let frame = || frame_text(&routes, g);
        s.peak_stuck = s.peak_stuck.max(stuck);
        let share = arrived as f32 / alive.max(1) as f32;
        if s.done_s.is_none() && share >= 0.99 {
            s.done_s = Some(secs);
        }
        if tick.is_multiple_of(60) && !s.reported {
            info!(
                "[path-test] {} reg {} t={secs:.0}s: arrived {:.1}% ({arrived}/{alive}), stuck {stuck}, centroid ({:.0}, {:.0}), order {}, {}",
                s.case,
                s.g,
                share * 100.0,
                gd.centroid.x,
                gd.centroid.y,
                if gd.order.is_some() { "on" } else { "done" },
                frame()
            );
        }
        // The point the sim lays this regiment's slots round now.
        let slots_at = sim_frame(&groups, &routes, s.g);
        // On the wall scenes, where the men are on the way up: on the
        // walk, on a stair (flight or platform), on the ground within
        // 6 m of a stair's foot or within 4 m of the wall's inner face, and
        // within a metre of their place; then the frame and the slots.
        if let Some(lane) = &walk_lane
            && ten_s
            && watch
        {
            use crate::town_layout::{STAIRS_X, STAIR_W, WALL_Z};
            let (mut walk, mut stair, mut foot, mut face, mut placed) = (0, 0, 0, 0, 0);
            // Men who moved under a metre since the last line, and of them
            // the ones more than `ARRIVED_M` from their places.
            let (mut still, mut still_out) = (0, 0);
            // The places round the sim's point, and how many are off the walk.
            let (mut lo, mut hi, mut off_walk) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY), 0);
            // Where the men on a flight stand across it: their mean offset
            // from the tread's middle toward the wall (m), and the widest.
            let tread_mid = WALL_Z - 4.0 - (STAIR_W - 0.5) * 0.5;
            let (mut across, mut across_max) = (0.0f32, f32::NEG_INFINITY);
            for i in (0..units.len()).filter(|&i| units.group[i] as usize == s.g && units.death_t[i] == 0) {
                let p = flat(i);
                if lane.contains(p) {
                    walk += 1;
                } else if STAIRS_X.iter().any(|&x| p.x >= x && p.x <= x + 12.0 && p.y >= WALL_Z - 4.0 - STAIR_W && p.y <= WALL_Z - 4.0) {
                    stair += 1;
                    across += p.y - tread_mid;
                    across_max = across_max.max(p.y - tread_mid);
                } else if STAIRS_X.iter().any(|&x| p.distance(Vec2::new(x, WALL_Z - 4.0 - STAIR_W * 0.5)) <= 6.0) {
                    foot += 1;
                } else if p.y >= WALL_Z - 8.0 && p.y <= WALL_Z - 4.0 {
                    face += 1;
                }
                let d = p.distance(target(s.g) + units.home[i]);
                placed += usize::from(d <= 1.0);
                if p.distance(test.pos_10s[i]) < 1.0 {
                    still += 1;
                    still_out += usize::from(d > ARRIVED_M);
                }
                let place = slots_at + units.home[i];
                lo = lo.min(place);
                hi = hi.max(place);
                off_walk += usize::from(!lane.contains(place));
            }
            let pending = match &gd.lane_fill {
                Some(crate::formation::LaneFill::Pending(v)) => v.len(),
                Some(_) => alive,
                None => 0,
            };
            info!(
                "[path-test] {} reg {} t={secs:.0}s: on the walk {walk}, on a stair {stair} (toward the wall {:+.2} m mean, {:+.2} m most), at a foot {foot}, at the wall's inner face {face}, in place {placed}, places not yet taken {pending}, never on the walk {never}, moved under 1 m in 10 s: {still} ({still_out} of them over 5 m from their places)",
                s.case,
                s.g,
                if stair > 0 { across / stair as f32 } else { 0.0 },
                if stair > 0 { across_max } else { 0.0 }
            );
            info!("[path-test] {} reg {} t={secs:.0}s: {}", s.case, s.g, frame_wait_text(&groups, &routes, s.g));
            info!(
                "[path-test] {} reg {} t={secs:.0}s: order {}, anchor ({:.1}, {:.1}), facing {:.2} rad, spacing {:?}, slots laid round ({:.1}, {:.1}); places x {:.1} to {:.1}, z {:.1} to {:.1}, {off_walk} of {alive} places off the walk",
                s.case,
                s.g,
                if gd.order.is_some() { "on" } else { "done" },
                gd.anchor.x,
                gd.anchor.y,
                gd.facing,
                gd.spacing,
                slots_at.x,
                slots_at.y,
                lo.x,
                hi.x,
                lo.y,
                hi.y
            );
        }
        // Where the men still out stand, every ten seconds once most are
        // in (on the wall scenes, from the start). Each man's slot and aim
        // are the sim's: round the point it lays the slots round now.
        if ((0.9..0.99).contains(&share) || walls)
            && ten_s
            && watch
        {
            let mut out: Vec<(f32, Vec2, usize)> = (0..units.len())
                .filter(|&i| units.group[i] as usize == s.g && units.death_t[i] == 0)
                .map(|i| {
                    let p = Vec2::new(units.pos[i].x, units.pos[i].z);
                    (p.distance(target(s.g) + units.home[i]), p, i)
                })
                .filter(|&(d, _, _)| d > ARRIVED_M)
                .collect();
            out.sort_by(|a, b| b.0.total_cmp(&a.0));
            let view = routes.view(s.g);
            let list: Vec<String> = out
                .iter()
                .take(6)
                .map(|&(d, p, i)| {
                    let v = Vec2::new(units.vel[i].x, units.vel[i].z);
                    let slot = slots_at + units.home[i];
                    let aim = match (&view, routes.nav()) {
                        (Some(r), Some(nav)) => {
                            let (t, through) =
                                crate::sim::route::man_target(nav, &terrain, r, slots_at, p, slot, i);
                            format!("aims ({:.1}, {:.1}){}", t.x, t.y, if through { " fast" } else { "" })
                        }
                        _ => "no route".into(),
                    };
                    format!(
                        "({:.1}, {:.1}) {d:.0} m out, v ({:.2}, {:.2}), slot ({:.1}, {:.1}), {aim}",
                        p.x, p.y, v.x, v.y, slot.x, slot.y
                    )
                })
                .collect();
            info!("[path-test] {} reg {} t={secs:.0}s: men out: {}", s.case, s.g, list.join(", "));
        }
        // After the verdict, every ten seconds: how far the centre has moved
        // since (a regiment that arrived stays where it stopped).
        if let Some(at) = s.settled_at
            && tick.is_multiple_of(300)
        {
            let on_wall =
                if walls { format!(", on the walk {walk}, never on the walk {never}") } else { String::new() };
            info!(
                "[path-test] {} reg {} t={secs:.0}s: holding at ({:.0}, {:.0}), {:.1} m from where it stood at the verdict, arrived {:.1}% ({arrived}/{alive}){on_wall}",
                s.case,
                s.g,
                gd.centroid.x,
                gd.centroid.y,
                gd.centroid.distance(at),
                share * 100.0
            );
        }
        let timed_out = secs >= s.limit;
        if !s.reported && (s.done_s.is_some() || timed_out) {
            s.reported = true;
            s.settled_at = Some(gd.centroid);
            let pace = regiment_pace(&units, s.g);
            let len = s.route_len.unwrap_or(s.straight);
            let bound = 2.0 * len / pace.max(0.1);
            let end = target(s.g);
            let verdict = match s.done_s {
                Some(t) if t <= bound => format!("OK in {t:.1} s (bound {bound:.1} s)"),
                Some(t) => format!("SLOW: {t:.1} s against a bound of {bound:.1} s"),
                None => format!("FAIL: {:.1}% arrived after {:.0} s", share * 100.0, s.limit),
            };
            let on_wall = if walls {
                format!("; on the walk {walk}, never on the walk {never}, {}", frame())
            } else {
                String::new()
            };
            info!(
                "[path-test] VERDICT {} reg {}: {verdict}; route {len:.0} m against {:.0} m straight ({}); pace {pace:.2} m/s; most stuck at once {}; ended at ({:.0}, {:.0}), ordered to ({:.0}, {:.0}){on_wall}",
                s.case, s.g, s.straight, s.route_note, s.peak_stuck, end.x, end.y, s.goal.x, s.goal.y
            );
        }
    }
    if walls && ten_s {
        for (i, at) in test.pos_10s.iter_mut().enumerate() {
            *at = flat(i);
        }
    }
    if tick.is_multiple_of(150) {
        dump_men(test, &units, &groups, &terrain, &routes, tick);
    }
    if !test.summary_done && !test.subjects.is_empty() && test.subjects.iter().all(|s| s.reported) {
        test.summary_done = true;
        let ok = test.subjects.iter().filter(|s| s.done_s.is_some()).count();
        info!("[path-test] SUMMARY {ok}/{} regiments arrived", test.subjects.len());
    }
    if tick.is_multiple_of(30) {
        frames.window_max = 0.0;
    }
}

/// One regiment's men in a tick of `log_progress`: the living, those
/// within `ARRIVED_M` of their places, those stuck, and on the wall scenes
/// those on the walk now and those never on it yet.
#[derive(Clone, Copy, Default)]
struct Tally {
    alive: usize,
    arrived: usize,
    stuck: usize,
    walk: usize,
    never: usize,
}

/// The seconds a regiment of `scene` has to arrive, unless its route is
/// so long that three times its length at the men's pace is longer.
fn timeout_s(scene: &str) -> f32 {
    if matches!(scene, "walls_cross" | "walls_long" | "walls_two_long") { CROSS_TIMEOUT_S } else { TIMEOUT_S }
}

/// Where regiment `g`'s route frame is, for the log: the waypoint it is
/// heading for out of the route's last, or no route.
fn frame_text(routes: &crate::sim::route::Routes, g: usize) -> String {
    match routes.route_of(g) {
        Some((points, k)) => format!("frame at waypoint {k} of {}", points.len().saturating_sub(1)),
        None => "no route".into(),
    }
}

/// `FL_TEST_DUMP=<file>`: every five seconds, every living man of the
/// regiments under watch, one line each, for checking each man against
/// where he should stand, away from the game: the seconds since the first
/// order, his regiment and his index, his position (x, height, z), his
/// place at the order's end (x, z), the slot the sim gives him now
/// (x, z), the sim's ground height under him and the drawn ground's,
/// whether the ground under him is blocked, his velocity (x, z), and his
/// facing (radians from +z toward +x: 0 looks along +z, half pi along +x).
fn dump_men(
    test: &mut PathTest,
    units: &Units,
    groups: &Groups,
    terrain: &Terrain,
    routes: &crate::sim::route::Routes,
    tick: u32,
) {
    use std::io::Write;
    if !test.dump_tried {
        test.dump_tried = true;
        let Ok(path) = std::env::var("FL_TEST_DUMP") else { return };
        match std::fs::File::create(&path) {
            Ok(mut file) => {
                let _ = writeln!(file, "t g i x y z place_x place_z slot_x slot_z sim_h ground_h blocked vx vz yaw");
                info!("[path-test] every man's position goes to {path} every 5 s");
                test.dump = Some(file);
            }
            Err(e) => warn!("[path-test] FL_TEST_DUMP: cannot write {path}: {e}"),
        }
    }
    let Some(file) = &mut test.dump else { return };
    let secs = tick.saturating_sub(ORDER_TICK) as f32 / 30.0;
    let watched: Vec<usize> = {
        let mut g: Vec<usize> = test.subjects.iter().map(|s| s.g).collect();
        g.sort_unstable();
        g.dedup();
        g
    };
    for &g in &watched {
        let gd = &groups.list[g];
        let place_at = match gd.order {
            Some(Order::Move(p)) => p,
            _ => gd.anchor,
        };
        let slot_at = sim_frame(groups, routes, g);
        for i in (0..units.len()).filter(|&i| units.group[i] as usize == g && units.death_t[i] == 0) {
            let p = units.pos[i];
            let (place, slot) = (place_at + units.home[i], slot_at + units.home[i]);
            let _ = writeln!(
                file,
                "{secs:.1} {g} {i} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {} {:.2} {:.2} {:.3}",
                p.x,
                p.y,
                p.z,
                place.x,
                place.y,
                slot.x,
                slot.y,
                terrain.height_at(p.x, p.z),
                terrain.ground_at(p.x, p.z),
                u8::from(terrain.blocked_at(p.x, p.z)),
                units.vel[i].x,
                units.vel[i].z,
                units.yaw[i]
            );
        }
    }
}

/// The point the sim lays regiment `g`'s slots round this tick, as
/// sim/job.rs `resolve_orders` hands it to the soldiers' kernel: none for
/// a regiment standing to shoot or holding a contact frame, else the
/// route's frame for its goal; the anchor where there is none
/// (sim/soldier.rs).
fn sim_frame(groups: &Groups, routes: &crate::sim::route::Routes, g: usize) -> Vec2 {
    let gd = &groups.list[g];
    let order = if gd.standoff || gd.skirmish_hold.is_some() || gd.contact {
        None
    } else {
        groups.goal(g).map(|goal| routes.frame(g, goal))
    };
    order.unwrap_or(gd.anchor)
}

/// Where regiment `g`'s frame waits and what would move it on, from the
/// route's advance tests (sim/route.rs `frame_test`).
fn frame_wait_text(groups: &Groups, routes: &crate::sim::route::Routes, g: usize) -> String {
    let Some((points, k)) = routes.route_of(g) else {
        return "no route: the men walk straight at their places".into();
    };
    let last = points.len() - 1;
    if k >= last {
        return "frame at its last waypoint".into();
    }
    let here = points[k];
    let (Some(goal), Some(nav)) = (groups.goal(g), routes.nav()) else {
        return format!("frame waits at waypoint {k} of {last} ({:.1}, {:.1}), with no goal", here.x, here.y);
    };
    let Some(t) = crate::sim::route::frame_test(nav, &groups.list[g], points, k, goal) else {
        return "frame at its last waypoint".into();
    };
    format!(
        "frame waits at waypoint {k} of {last} ({:.1}, {:.1}), next ({:.1}, {:.1}): centre {:.1} m from it, moves on within {:.1} m, or when the way to the next is clear at that half width: clear {}",
        t.here.x,
        t.here.y,
        t.next.x,
        t.next.y,
        t.dist,
        t.w,
        if t.clear { "yes" } else { "no" }
    )
}

/// The men's mean speed (m/s), the pace the bound is taken at.
fn regiment_pace(units: &Units, g: usize) -> f32 {
    let (mut sum, mut n) = (0.0, 0);
    for i in 0..units.len() {
        if units.group[i] as usize == g && units.death_t[i] == 0 {
            sum += units.speed[i];
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f32 }
}

/// What the route does, for the log: on the river map whether it crosses
/// on the bridge or wades the ford; on the wall scenes which stairs it
/// climbs, where it crosses the wall's line and whether it goes over the
/// bridge.
fn route_note(points: &[Vec2], scene: &str) -> String {
    let n = points.len().saturating_sub(1);
    if scene.starts_with("walls") {
        use crate::town_layout::{STAIRS_X, WALL_Z};
        let along = |f: &dyn Fn(Vec2) -> bool| {
            points.windows(2).any(|w| (0..=80).any(|k| f(w[0].lerp(w[1], k as f32 / 80.0))))
        };
        let stairs: Vec<String> = STAIRS_X
            .iter()
            .filter(|&&x| along(&|q| q.x > x + 2.0 && q.x < x + 10.0 && (q.y - (WALL_Z - 6.0)).abs() < 2.0))
            .map(|x| format!("{x:.0}"))
            .collect();
        let crossings: Vec<String> = points
            .windows(2)
            .filter(|w| (w[0].y - WALL_Z) * (w[1].y - WALL_Z) < 0.0)
            .map(|w| format!("{:.0}", w[0].x + (w[1].x - w[0].x) * (WALL_Z - w[0].y) / (w[1].y - w[0].y)))
            .collect();
        let bridge = along(&|q| q.x.abs() < 10.0 && (q.y - WALL_Z).abs() < 3.0);
        return format!(
            "{n} waypoints, stairs at x [{}], crosses the wall's line at x [{}], over the bridge: {bridge}",
            stairs.join(", "),
            crossings.join(", ")
        );
    }
    if scene != "river" {
        return format!("{n} waypoints");
    }
    let on_bridge = points.windows(2).any(|w| {
        (0..=40).any(|k| {
            let q = w[0].lerp(w[1], k as f32 / 40.0);
            crate::terrain::bridge_deck_contains(q.x, q.y)
        })
    });
    format!("{n} waypoints, crosses by the {}", if on_bridge { "bridge" } else { "ford" })
}

fn watch_frames(time: Res<Time<Real>>, test: Res<PathTest>, mut frames: ResMut<FrameWatch>) {
    let dt = time.delta_secs() * 1000.0;
    frames.window_max = frames.window_max.max(dt);
    if test.ordered {
        frames.since_order_max = frames.since_order_max.max(dt);
    }
}
