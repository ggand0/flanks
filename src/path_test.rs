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
//!
//! Per regiment the log has the share of men within 5 m of their slots,
//! the men stuck (slower than 0.1 m/s for 10 s while more than 5 m from
//! their slot), the route's length against the straight line, and a
//! verdict once 99% have arrived: the time taken against twice the time
//! the route's length takes at the men's pace.

use bevy::prelude::*;

use crate::obstacle_map::CASES;
use crate::orders::{Groups, Order};
use crate::terrain::Terrain;
use crate::unit_types::KIND_HEAVY;
use crate::units::Units;

/// The sim tick the orders go out on (3 s in).
const ORDER_TICK: u32 = 90;
/// Men per regiment in the case scenes, and in the 200-regiment scene.
const CASE_MEN: usize = 200;
const MANY_MEN: usize = 40;
const MANY_REGS: usize = 200;
/// Seconds after the order a regiment has to arrive before it fails.
const TIMEOUT_S: f32 = 180.0;
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
    case: &'static str,
    goal: Vec2,
    straight: f32,
    route_len: Option<f32>,
    route_note: String,
    done_s: Option<f32>,
    peak_stuck: usize,
    reported: bool,
}

#[derive(Resource, Default)]
struct PathTest {
    subjects: Vec<Subject>,
    /// Per unit, ticks spent stuck so far.
    stuck_ticks: Vec<u32>,
    ordered: bool,
    summary_done: bool,
    /// The 200-regiment scene: when every regiment had its route.
    many_routed_tick: Option<u32>,
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
    let wanted_map = if scene == "river" {
        crate::terrain::MapKind::River
    } else {
        crate::terrain::MapKind::Obstacles
    };
    if terrain.kind != wanted_map {
        warn!("[path-test] scene {scene} expects FL_MAP={:?}, the map is {:?}", wanted_map, terrain.kind);
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
    mut test: ResMut<PathTest>,
) {
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
        crate::orders::order_regiments(&mut groups, &[g], goal);
        let start = groups.list[g].centroid;
        test.subjects.push(Subject {
            g,
            case,
            goal,
            straight: start.distance(goal),
            route_len: None,
            route_note: String::new(),
            done_s: None,
            peak_stuck: 0,
            reported: false,
        });
    }
    info!("[path-test] tick {}: {} regiments ordered", pipeline.tick, orders.len());
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
    // The slot each man is going to: his regiment's ordered point (or the
    // anchor once it arrived) plus his offset.
    let target = |g: usize| match groups.list[g].order {
        Some(Order::Move(p)) => p,
        _ => groups.list[g].anchor,
    };
    let mut men = vec![(0usize, 0usize, 0usize); groups.list.len()];
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
        e.0 += 1;
        e.1 += (d <= ARRIVED_M) as usize;
        e.2 += (test.stuck_ticks[i] as f32 >= STUCK_S * 30.0) as usize;
    }
    let scene = scene().unwrap_or_default();
    if scene == "many" {
        if test.many_routed_tick.is_none() {
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
    for s in test.subjects.iter_mut() {
        let gd = &groups.list[s.g];
        if s.route_len.is_none() {
            if let Some((points, _)) = routes.route_of(s.g) {
                let len: f32 = points.windows(2).map(|w| w[0].distance(w[1])).sum();
                s.route_len = Some(len);
                s.route_note = route_note(points, &scene);
            } else if secs > 1.0 {
                s.route_len = Some(s.straight);
                s.route_note = "no route, the straight way is clear".into();
            }
        }
        let (alive, arrived, stuck) = men[s.g];
        s.peak_stuck = s.peak_stuck.max(stuck);
        let share = arrived as f32 / alive.max(1) as f32;
        if s.done_s.is_none() && share >= 0.99 {
            s.done_s = Some(secs);
        }
        if tick.is_multiple_of(60) && !s.reported {
            info!(
                "[path-test] {} reg {} t={secs:.0}s: arrived {:.1}% ({arrived}/{alive}), stuck {stuck}, centroid ({:.0}, {:.0}), order {}",
                s.case,
                s.g,
                share * 100.0,
                gd.centroid.x,
                gd.centroid.y,
                if gd.order.is_some() { "on" } else { "done" }
            );
        }
        // Where the men still out stand, every ten seconds once most are in.
        if (0.9..0.99).contains(&share) && tick.is_multiple_of(300) && !s.reported {
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
                    let aim = match (&view, routes.nav()) {
                        (Some(r), Some(nav)) => {
                            let frame = target(s.g);
                            let (t, through) = crate::sim::route::man_target(
                                nav, &terrain, r, frame, p, frame + units.home[i], i,
                            );
                            format!(" aims ({:.1}, {:.1}){}", t.x, t.y, if through { " fast" } else { "" })
                        }
                        _ => " no route".into(),
                    };
                    format!("({:.1}, {:.1}) {d:.0} m out, v ({:.2}, {:.2}){aim}", p.x, p.y, v.x, v.y)
                })
                .collect();
            info!("[path-test] {} reg {} t={secs:.0}s: men out: {}", s.case, s.g, list.join(", "));
        }
        let timed_out = secs >= TIMEOUT_S.max(3.0 * s.route_len.unwrap_or(0.0) / regiment_pace(&units, s.g).max(0.1));
        if !s.reported && (s.done_s.is_some() || timed_out) {
            s.reported = true;
            let pace = regiment_pace(&units, s.g);
            let len = s.route_len.unwrap_or(s.straight);
            let bound = 2.0 * len / pace.max(0.1);
            let end = target(s.g);
            let verdict = match s.done_s {
                Some(t) if t <= bound => format!("OK in {t:.1} s (bound {bound:.1} s)"),
                Some(t) => format!("SLOW: {t:.1} s against a bound of {bound:.1} s"),
                None => format!("FAIL: {:.1}% arrived after {TIMEOUT_S:.0} s", share * 100.0),
            };
            info!(
                "[path-test] VERDICT {} reg {}: {verdict}; route {len:.0} m against {:.0} m straight ({}); pace {pace:.2} m/s; most stuck at once {}; ended at ({:.0}, {:.0}), ordered to ({:.0}, {:.0})",
                s.case, s.g, s.straight, s.route_note, s.peak_stuck, end.x, end.y, s.goal.x, s.goal.y
            );
        }
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
/// on the bridge or wades the ford.
fn route_note(points: &[Vec2], scene: &str) -> String {
    let n = points.len().saturating_sub(1);
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
