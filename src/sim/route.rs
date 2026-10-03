//! Regiment routes: when an order's straight way is blocked, the regiment
//! is given a route (nav.rs) and follows it waypoint by waypoint.
//!
//! - A new order is checked on the main thread first: if a block of the
//!   regiment's width can walk straight to the goal without meeting
//!   blocked ground or water, no route is made and the order behaves
//!   exactly as it always did. Maps with no blocked ground and no water
//!   never get this far (there is no grid).
//! - Otherwise a plan is requested. Requests ride in the tick job, a
//!   fixed number per tick, and are planned on the pool after the
//!   soldiers' kernel; the job's install hands the plans back at the next
//!   tick. The schedule is counted in ticks, never in wall time, so a run
//!   repeats bit for bit. While a regiment waits for its plan (normally
//!   one tick) it walks straight, as before.
//! - The regiment's frame (the point `resolve_orders` hands its men) is
//!   the route's current waypoint; the last waypoint is the live goal. It
//!   moves on to the next waypoint once the way there is clear and dry for
//!   the block's whole width, or once its centre reaches the current one.
//! - Each man keeps walking to his own slot around the frame while his
//!   straight way there is open. When it is not, he follows the route:
//!   the farthest route point ahead of his place on it that he can walk
//!   to straight, never past his regiment's frame; failing any, he steps
//!   round what stands in front of him. He knows his slot, his
//!   regiment's route and the ground ahead of him, nothing else.
//! - A regiment that has arrived keeps its route while it holds at the
//!   route's end, so a man still behind an obstacle finds his way to his
//!   slot after the order is done. The next order replaces it.
//! - An attack order plans again when its target has moved more than the
//!   regiment's half width from where the plan was made to. A goal that
//!   cannot be reached becomes a move to the nearest reachable ground
//!   with room for the regiment, and the log states it.
//!
//! `FL_PATH=0` turns routes off.

use bevy::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::nav::{NavGrid, Plan, PlanRequest, Scratch};
use crate::orders::{GroupData, Groups, Order};
use crate::terrain::Terrain;

/// Plans asked of one tick job at most. They run in parallel on the pool,
/// so a tick costs about as much as its slowest plan.
const PLANS_PER_TICK: usize = 16;
/// An attack order checks this often (ticks) whether its target has left
/// the end of its route.
const ATTACK_RECHECK: u32 = 15;
/// A man looking along his route for a point he can walk to checks every
/// this many metres of it, and looks back along it at most `FOLLOW_BACK`.
const FOLLOW_STEP: f32 = 2.0;
const FOLLOW_BACK: f32 = 30.0;

/// One regiment's route as the kernel reads it: the points from the
/// route's start to its current waypoint (`points[k]`; the regiment's
/// frame stands in for it).
#[derive(Clone)]
pub struct RouteView {
    pub points: Arc<[Vec2]>,
    pub k: u32,
    /// The frame is on the last leg: men slow down near their slots.
    pub last_leg: bool,
}

/// The kernel's part of the routes: per regiment, its route if any, and
/// the grid the men read the ground ahead from.
#[derive(Clone, Copy)]
pub(crate) struct KernelRoutes<'a> {
    pub routes: &'a [Option<RouteView>],
    pub nav: Option<&'a NavGrid>,
}

#[derive(Clone, Default)]
enum State {
    /// No route: the order walks straight (or there is no order).
    #[default]
    None,
    /// A plan is on its way; the regiment walks straight meanwhile, or
    /// keeps the route it had.
    Pending { serial: u32, old: Option<Arc<Route>> },
    Ready(Arc<Route>),
}

/// A planned route and the regiment's progress along it.
struct Route {
    points: Arc<[Vec2]>,
    /// The current waypoint (index into `points`, at least 1).
    k: usize,
}

#[derive(Clone, Default)]
struct RegRoute {
    /// The order this entry was made for: a different one is new.
    order: Option<Order>,
    /// The goal the current plan was made to.
    planned_goal: Vec2,
    state: State,
}

/// Every regiment's route, the plans waiting to go out, and the grid.
#[derive(Resource, Default)]
pub struct Routes {
    nav: Option<Arc<NavGrid>>,
    per: Vec<RegRoute>,
    queue: VecDeque<PlanRequest>,
    /// The requests the next tick job takes.
    outgoing: Vec<PlanRequest>,
    /// Plans handed back by the last install.
    incoming: Vec<Plan>,
    serial: u32,
    generation: u64,
    built: bool,
}

/// `FL_PATH=0` turns routes off.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("FL_PATH").is_ok_and(|v| v == "0"))
}

impl Routes {
    /// The grid, for the job and the debug view.
    pub fn nav(&self) -> Option<&Arc<NavGrid>> {
        self.nav.as_ref()
    }

    /// Fill a tick job's route inputs: the requests waiting to go out,
    /// each regiment's route, and the grid.
    pub(crate) fn fill_job(
        &mut self,
        requests: &mut Vec<PlanRequest>,
        views: &mut Vec<Option<RouteView>>,
        nav: &mut Option<Arc<NavGrid>>,
        n_groups: usize,
    ) {
        requests.clear();
        requests.append(&mut self.outgoing);
        views.clear();
        views.resize(n_groups, None);
        for (g, r) in self.per.iter().enumerate().take(n_groups) {
            let route = match &r.state {
                State::Ready(route) => Some(route),
                State::Pending { old: Some(route), .. } => Some(route),
                _ => None,
            };
            views[g] = route.map(|route| RouteView {
                points: route.points.clone(),
                k: route.k as u32,
                last_leg: route.k + 1 >= route.points.len(),
            });
        }
        *nav = if views.iter().any(Option::is_some) || !requests.is_empty() {
            self.nav.clone()
        } else {
            None
        };
    }

    /// Take the plans a tick job made.
    pub(crate) fn hand_back(&mut self, plans: &mut Vec<Plan>) {
        self.incoming.append(plans);
    }

    /// Each regiment's resolved frame this tick: its route's current
    /// waypoint, or `goal` (the order's own point) on the last leg or
    /// without a route.
    pub(crate) fn frame(&self, g: usize, goal: Vec2) -> Vec2 {
        let route = match self.per.get(g).map(|r| &r.state) {
            Some(State::Ready(route)) => route,
            Some(State::Pending { old: Some(route), .. }) => route,
            _ => return goal,
        };
        if route.k + 1 >= route.points.len() {
            goal
        } else {
            route.points[route.k]
        }
    }

    /// Regiment `g` waits for a plan.
    pub fn pending(&self, g: usize) -> bool {
        matches!(self.per.get(g).map(|r| &r.state), Some(State::Pending { .. }))
    }

    /// Regiment `g`'s route as the kernel sees it.
    pub(crate) fn view(&self, g: usize) -> Option<RouteView> {
        match &self.per.get(g)?.state {
            State::Ready(r) | State::Pending { old: Some(r), .. } => Some(RouteView {
                points: r.points.clone(),
                k: r.k as u32,
                last_leg: r.k + 1 >= r.points.len(),
            }),
            _ => None,
        }
    }

    /// The route points a regiment follows, for the debug view and the
    /// scripted scenes: its whole route and its current waypoint.
    pub fn route_of(&self, g: usize) -> Option<(&[Vec2], usize)> {
        match &self.per.get(g)?.state {
            State::Ready(r) | State::Pending { old: Some(r), .. } => Some((&r.points, r.k)),
            _ => None,
        }
    }
}

/// Half a regiment's extent across the way `dir` (m): what must fit
/// between obstacles for the block to walk on unbroken.
pub fn half_width(gd: &GroupData, dir: Vec2) -> f32 {
    let across = Vec2::new(-dir.y, dir.x).normalize_or(Vec2::X);
    if gd.shape == crate::formation::FormShape::Blob {
        return crate::formation::blob_radius(gd.count);
    }
    let files = (gd.files.max(1) as usize).min(gd.count.max(1));
    let ranks = gd.count.max(1).div_ceil(files);
    let pitch = gd.spacing.pitch();
    let half_w = (files - 1) as f32 * 0.5 * pitch.x + 1.0;
    let half_d = (ranks - 1) as f32 * 0.5 * pitch.y + 1.0;
    let fwd = crate::formation::facing_dir(gd.facing);
    let right = Vec2::new(fwd.y, -fwd.x);
    right.dot(across).abs() * half_w + fwd.dot(across).abs() * half_d
}

/// The fixed tick's route work, per regiment only: keep the grid in step
/// with the terrain, take the plans the last install handed back, notice
/// new orders and moving attack targets, move each frame along its route,
/// and pick the requests the next job carries. Runs after the orders are
/// cleared and before the next job is prepared.
pub fn update_routes(
    mut routes: ResMut<Routes>,
    mut groups: ResMut<Groups>,
    terrain: Res<Terrain>,
    units: Res<crate::units::Units>,
    pipeline: Res<crate::sim::TickPipeline>,
) {
    if !enabled() {
        return;
    }
    let routes = &mut *routes;
    if !routes.built || terrain.is_changed() {
        let fresh = match routes.nav.as_mut() {
            Some(nav) if routes.built => Arc::make_mut(nav).refresh(&terrain),
            _ => {
                routes.nav = NavGrid::build(&terrain).map(Arc::new);
                true
            }
        };
        routes.built = true;
        if fresh {
            if routes.nav.is_some() {
                info!("[path] walkability grid ready for the {} map", terrain.kind.label());
            }
            // Routes made on the old ground are planned again.
            for r in routes.per.iter_mut() {
                r.order = None;
            }
        }
    }
    if units.generation != routes.generation {
        routes.generation = units.generation;
        routes.per.clear();
        routes.queue.clear();
        routes.outgoing.clear();
        routes.incoming.clear();
    }
    let n = groups.list.len();
    routes.per.resize_with(n, RegRoute::default);
    let Some(nav) = routes.nav.clone() else {
        return;
    };
    let tick = pipeline.tick;

    // The plans the last install handed back.
    for plan in std::mem::take(&mut routes.incoming) {
        adopt(routes, &mut groups, plan);
    }

    for g in 0..n {
        let gd = &groups.list[g];
        if gd.order != routes.per[g].order {
            routes.per[g].order = gd.order;
            if gd.order.is_none() {
                // Arrived: the route stays while the regiment holds at its
                // end.
                let kept = match &routes.per[g].state {
                    State::Ready(r) | State::Pending { old: Some(r), .. } => {
                        let last = r.points.len() - 1;
                        (r.points[last].distance(gd.anchor) < 1.0)
                            .then(|| Arc::new(Route { points: r.points.clone(), k: last }))
                    }
                    _ => None,
                };
                // Coming round obstacles has mixed up the ranks: the men
                // take the nearest places in them.
                if kept.is_some() {
                    groups.list[g].reform = true;
                }
                routes.per[g].state = kept.map_or(State::None, State::Ready);
                continue;
            }
            match groups.goal(g) {
                Some(goal) if gd.count > 0 => {
                    let start = gd.centroid;
                    let w = half_width(gd, goal - start);
                    if nav.straight_ok(start, goal, w) {
                        routes.per[g].state = State::None;
                    } else {
                        request(routes, g, start, goal, w);
                    }
                }
                _ => routes.per[g].state = State::None,
            }
            continue;
        }
        // An attack order whose target has gone behind something, or has
        // moved off the end of its route.
        if let Some(Order::Attack(_)) = gd.order
            && (tick + g as u32).is_multiple_of(ATTACK_RECHECK)
            && let Some(goal) = groups.goal(g)
        {
            let start = gd.centroid;
            let w = half_width(gd, goal - start);
            let r = &routes.per[g];
            let replan = match r.state {
                State::None => !nav.straight_ok(start, goal, w),
                State::Ready(_) => goal.distance(r.planned_goal) > w.max(crate::terrain::CELL * 4.0),
                State::Pending { .. } => false,
            };
            if replan {
                request(routes, g, start, goal, w);
            }
        }
    }

    // Each frame along its route; a holding regiment's route goes once it
    // holds elsewhere.
    for g in 0..n {
        let gd = &groups.list[g];
        let Some(goal) = groups.goal(g) else {
            if let State::Ready(r) = &routes.per[g].state
                && r.points[r.points.len() - 1].distance(gd.anchor) >= 1.0
            {
                routes.per[g].state = State::None;
            }
            continue;
        };
        let State::Ready(route) = &routes.per[g].state else { continue };
        let mut k = route.k;
        let last = route.points.len() - 1;
        let w = half_width(gd, goal - gd.centroid);
        while k < last {
            let next = if k + 1 == last { goal } else { route.points[k + 1] };
            let here = route.points[k];
            if gd.centroid.distance(here) <= w || nav.straight_ok(gd.centroid, next, w) {
                k += 1;
            } else {
                break;
            }
        }
        if k != route.k {
            let points = route.points.clone();
            routes.per[g].state = State::Ready(Arc::new(Route { points, k }));
        }
    }

    // The requests the next job carries, if the last ones went out.
    if routes.outgoing.is_empty() {
        while routes.outgoing.len() < PLANS_PER_TICK {
            let Some(req) = routes.queue.pop_front() else { break };
            routes.outgoing.push(req);
        }
    }
}

/// Queue a plan for regiment `g`, superseding any it waits for.
fn request(routes: &mut Routes, g: usize, start: Vec2, goal: Vec2, half_w: f32) {
    routes.serial = routes.serial.wrapping_add(1);
    let serial = routes.serial;
    routes.queue.retain(|q| q.g as usize != g);
    routes.queue.push_back(PlanRequest { g: g as u32, serial, start, goal, half_w });
    let r = &mut routes.per[g];
    let old = match std::mem::take(&mut r.state) {
        State::Ready(route) => Some(route),
        State::Pending { old, .. } => old,
        State::None => None,
    };
    r.state = State::Pending { serial, old };
    r.planned_goal = goal;
}

/// Take a plan for its regiment, if it is still the one asked for.
fn adopt(routes: &mut Routes, groups: &mut Groups, plan: Plan) {
    let g = plan.req.g as usize;
    let Some(r) = routes.per.get_mut(g) else { return };
    match r.state {
        State::Pending { serial, .. } if serial == plan.req.serial => {}
        _ => return,
    }
    let straight = plan.req.start.distance(plan.req.goal);
    info!(
        "[path] regiment {g}: planned in {:.2} ms over {} cells, route {:.0} m against {:.0} m straight, {} waypoints{}",
        plan.ms,
        plan.cells,
        plan.length(),
        straight,
        plan.points.len().saturating_sub(1),
        if plan.reached { "" } else { ", goal out of reach" }
    );
    if plan.points.len() < 2 {
        r.state = State::None;
        return;
    }
    let end = *plan.points.last().unwrap();
    r.state = State::Ready(Arc::new(Route { points: plan.points.into(), k: 1 }));
    if !plan.reached {
        let gd = &mut groups.list[g];
        if let Some(Order::Move(_)) = gd.order {
            // The goal cannot be reached: the order becomes a move to the
            // nearest ground the regiment can reach with room for it.
            gd.order = Some(Order::Move(end));
            r.order = gd.order;
            info!(
                "[path] regiment {g}: the goal ({:.0}, {:.0}) cannot be reached; going to ({:.0}, {:.0}) instead",
                plan.req.goal.x, plan.req.goal.y, end.x, end.y
            );
        }
    }
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

/// Plan a tick job's requests, in parallel on the pool, results in request
/// order. Runs on the sim worker after the kernel.
pub(crate) fn plan_all(nav: &NavGrid, requests: &[PlanRequest], out: &mut Vec<Plan>) {
    out.clear();
    if requests.is_empty() {
        return;
    }
    let t0 = std::time::Instant::now();
    nav.prepare_costs(requests);
    let plans = crate::util::sim_scope(|scope| {
        for req in requests {
            scope.spawn(async move { SCRATCH.with(|s| nav.plan(req, &mut s.borrow_mut())) });
        }
    });
    out.extend(plans);
    info!(
        "[path] {} plans in the tick job: {:.2} ms",
        out.len(),
        t0.elapsed().as_secs_f32() * 1000.0
    );
}

/// Where a man of a regiment on a route walks this tick, and whether he
/// keeps his full pace (he is not yet near the end of the way).
///
/// `frame` is his regiment's frame point (its current waypoint, or the
/// goal on the last leg) and `slot` his own slot around it. With his
/// straight way to his slot open, he walks to it. Otherwise he follows the
/// route: from his place on it (the nearest point of its legs up to the
/// frame), the farthest point ahead he can walk to straight. Failing any,
/// the nearest point behind his place he can walk to. Failing that too, he
/// steps round what stands in front of him, to the side his place on the
/// route lies.
#[inline]
pub(crate) fn man_target(
    nav: &NavGrid,
    terrain: &Terrain,
    route: &RouteView,
    frame: Vec2,
    p: Vec2,
    slot: Vec2,
    i: usize,
) -> (Vec2, bool) {
    if nav.walk_line(p, slot) {
        return (slot, !route.last_leg);
    }
    let k = (route.k as usize).min(route.points.len() - 1);
    let point = |j: usize| if j == k { frame } else { route.points[j] };
    // His place on the route.
    let (mut leg, mut at, mut best) = (0, point(0), f32::MAX);
    for j in 0..k {
        let (a, b) = (point(j), point(j + 1));
        let ab = b - a;
        let t = ((p - a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
        let q = a + ab * t;
        let d = p.distance_squared(q);
        if d < best {
            (leg, at, best) = (j, q, d);
        }
    }
    // The route's waypoints ahead, the farthest first.
    for j in (leg + 1..=k).rev() {
        if nav.walk_line(p, point(j)) {
            return (point(j), true);
        }
    }
    // Along his leg, from his place to its end: the farthest he can reach.
    let end = point(leg + 1);
    let len = at.distance(end);
    let steps = (len / FOLLOW_STEP).ceil() as usize;
    let mut reach = None;
    for s in (0..steps).rev() {
        let q = at.lerp(end, s as f32 * FOLLOW_STEP / len.max(1e-6));
        if nav.walk_line(p, q) {
            reach = Some(q);
            break;
        }
    }
    if let Some(q) = reach {
        return (q, true);
    }
    // Back along the route from his place.
    let start = point(leg);
    let back = at.distance(start).min(FOLLOW_BACK);
    let mut s = FOLLOW_STEP;
    while s <= back {
        let q = at.lerp(start, s / at.distance(start).max(1e-6));
        if nav.walk_line(p, q) {
            return (q, true);
        }
        s += FOLLOW_STEP;
    }
    // Nothing on the route in reach: step round what is in front of him,
    // turning first to the side his place on the route lies.
    let dir = (end - p).normalize_or_zero();
    if dir == Vec2::ZERO {
        return (end, true);
    }
    let open = |d: Vec2| {
        let a = p + d * 1.25;
        let b = p + d * 2.5;
        !terrain.blocked_at(a.x, a.y) && !terrain.blocked_at(b.x, b.y)
    };
    if open(dir) {
        return (end, true);
    }
    let side = dir.perp_dot(at - p);
    let first = if side.abs() > 0.5 {
        side.signum()
    } else if crate::units::hash01(i as u32 ^ 0x51ED_270B) < 0.5 {
        1.0
    } else {
        -1.0
    };
    for step in 1..=5 {
        let a = step as f32 * std::f32::consts::FRAC_PI_6;
        for sign in [first, -first] {
            let d = Vec2::from_angle(a * sign).rotate(dir);
            if open(d) {
                return (p + d * 4.0, true);
            }
        }
    }
    (end, true)
}

/// `FL_NAV_VIZ=1`: draw the blocked cells' edges near the camera and every
/// regiment's route (the legs still ahead in yellow, the current waypoint
/// circled), under the debug overlay switch (G).
pub fn draw_routes(
    routes: Res<Routes>,
    groups: Res<Groups>,
    terrain: Res<Terrain>,
    viz: Res<crate::sim::DebugViz>,
    camera: Query<&crate::camera::RtsCamera>,
    mut gizmos: Gizmos,
) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("FL_NAV_VIZ").is_ok()) || !viz.0 {
        return;
    }
    let Some(nav) = routes.nav() else { return };
    let lift = |p: Vec2, h: f32| Vec3::new(p.x, terrain.height_at(p.x, p.y) + h, p.y);
    let focus = camera.iter().next().map_or(Vec2::ZERO, |c| Vec2::new(c.focus.x, c.focus.z));
    let radius = camera.iter().next().map_or(300.0, |c| (c.distance * 1.5).clamp(120.0, 600.0));
    let half = crate::terrain::CELL * 0.5;
    for p in nav.edge_cells(focus, radius) {
        let y = terrain.height_at(p.x, p.y) + 0.4;
        let c = Color::srgb(0.9, 0.2, 0.15);
        let corners = [
            Vec3::new(p.x - half, y, p.y - half),
            Vec3::new(p.x + half, y, p.y - half),
            Vec3::new(p.x + half, y, p.y + half),
            Vec3::new(p.x - half, y, p.y + half),
        ];
        gizmos.linestrip([corners[0], corners[1], corners[2], corners[3], corners[0]], c);
    }
    for (g, gd) in groups.list.iter().enumerate() {
        let Some((points, k)) = routes.route_of(g) else { continue };
        if gd.count == 0 {
            continue;
        }
        let ahead = Color::srgb(1.0, 0.85, 0.1);
        let behind = Color::srgba(1.0, 0.85, 0.1, 0.35);
        for (j, w) in points.windows(2).enumerate() {
            let c = if j + 1 >= k { ahead } else { behind };
            gizmos.line(lift(w[0], 1.0), lift(w[1], 1.0), c);
        }
        gizmos.line(lift(gd.centroid, 1.0), lift(points[k.min(points.len() - 1)], 1.0), Color::srgb(0.3, 0.9, 1.0));
        let wp = lift(points[k.min(points.len() - 1)], 1.0);
        gizmos.circle(Isometry3d::new(wp, Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)), 3.0, ahead);
    }
}
