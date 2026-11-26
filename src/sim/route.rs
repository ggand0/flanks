//! Regiment routes: when an order's straight way is blocked, the regiment
//! is given a route (nav.rs) and follows it waypoint by waypoint.
//!
//! - A new order is checked on the main thread first: if a block of the
//!   regiment's width can walk straight to the goal without meeting
//!   blocked ground or water, no route is made and the order behaves
//!   exactly as it always did. Maps with no blocked ground and no water
//!   never get this far (there is no grid).
//! - Otherwise a plan is requested. Requests ride in the tick job, a
//!   fixed number at a time, and are searched on the pool after the
//!   soldiers' kernel, each for a budget of cells per tick (`PLAN_BUDGET`,
//!   about a millisecond); a search that is not over comes back and goes
//!   out again with the next job, so no tick waits for a long route. The
//!   job's install hands the finished plans back. A tick that must first
//!   build the cost table for a new regiment width builds only that. The
//!   schedule is counted in ticks, never in wall time, so a run repeats
//!   bit for bit. While a regiment waits for its plan (one or two ticks
//!   for a short route, a second or two for a long one through a town) it
//!   walks straight, as before.
//! - The regiment's frame (the point `resolve_orders` hands its men) is
//!   the route's current waypoint; the last waypoint is the live goal. It
//!   moves on to the next waypoint once the way there is clear and dry for
//!   the block's whole width, or once its centre reaches the current one.
//! - Each man keeps walking to his own slot around the frame while his
//!   straight way there is open. When it is not, he follows the route
//!   from his own place on it, never past his regiment's frame: the
//!   farthest point along it he can walk to straight, or else the nearest
//!   point behind. A man who can walk straight to no point of it is cut
//!   off from it (his regiment went round a wall he is still outside of,
//!   and planned again from inside): he walks a way of his own back to
//!   it, planned for him on the grid. While he waits for it, or where none
//!   is found, he steps around what stands in front of him. He knows his
//!   slot, his regiment's route, the ways planned for its cut-off men and
//!   the ground ahead of him, nothing else.
//! - A way of one man's own: a cut-off man asks for one (`WayAsk`), the
//!   asks go back with the tick job, and the planner searches from where he
//!   stands to his regiment's frame over the same grid and budget as the
//!   regiments' routes (`nav.rs` `PlanRequest::man`), a few searches a
//!   tick (`WAYS_PER_TICK`), each bounded (`nav::MAN_CELLS`). The ways are
//!   kept per regiment until its route changes, and every cut-off man of
//!   it walks the one whose point he can walk to straight that is nearest
//!   the way's end; an ask within `ASK_NEAR` of an earlier one for the same
//!   route starts no search.
//! - A regiment that has arrived keeps its route while it holds at the
//!   route's end, so a man still behind an obstacle finds his way to his
//!   slot after the order is done, and its slots are re-sorted by where
//!   its men stand. The next order replaces the route.
//! - An attack order plans again when its target has gone behind
//!   something, or has moved more than the regiment's half width from
//!   where the plan was made to. A goal that
//!   cannot be reached becomes a move to the nearest reachable ground
//!   with room for the regiment, and the log states it.
//!
//! `FL_PATH=0` turns routes off. `FL_FRAME_SHARE` sets the share of a
//! regiment's men past a waypoint that moves its frame on (`frame_share`;
//! 0 turns it off). `FL_LOG_ROUTE=1` logs each route event
//! (`log_route`): an order's end and whether its route stays, a holding
//! regiment's route dropped, and each move of a frame to a later waypoint.

use bevy::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::nav::{NavGrid, Plan, PlanRequest, Search};
use crate::orders::{GroupData, Groups, Order};
use crate::terrain::Terrain;

/// Searches one tick job carries at most. They run in parallel on the
/// pool, each for `PLAN_BUDGET` cells, so a tick's planning costs about
/// one budget's worth of wall time.
const PLANS_PER_TICK: usize = 16;
/// Cells one search closes per tick at most: about a millisecond, and
/// enough for most routes in one tick (the obstacle map's zig-zag takes
/// twenty).
const PLAN_BUDGET: u32 = 4000;

/// A route search on its way through the tick jobs: the request, and the
/// search once it has begun.
pub struct PlanJob {
    pub req: PlanRequest,
    pub search: Option<Search>,
}
/// An attack order checks this often (ticks) whether its target has left
/// the end of its route.
const ATTACK_RECHECK: u32 = 15;
/// Where a man whose way to his slot is blocked looks for a point of his
/// route he can walk to: this far ahead of his place on it, the farthest
/// first, then this far back. Each is one line check (nav.rs `SIGHT`
/// bounds it), so a man's tick costs at most these plus his route's
/// waypoints.
const FOLLOW_AHEAD: [f32; 5] = [48.0, 24.0, 12.0, 6.0, 3.0];
const FOLLOW_BACK: [f32; 2] = [6.0, 20.0];
/// Searches for cut-off men's ways one tick job carries at most, beside
/// the regiments' `PLANS_PER_TICK`.
const WAYS_PER_TICK: usize = 4;
/// An ask this near (m) to one already made for the same route starts no
/// search: two cells, so one search serves a cluster of cut-off men.
const ASK_NEAR: f32 = 4.0;
/// Searches one route's cut-off men may start at most, and ways kept for
/// it at most.
const ASKS_PER_ROUTE: usize = 32;
const WAYS_PER_ROUTE: usize = 16;

/// A cut-off man's ask for a way of his own (`man_target`): his regiment,
/// where he stands, and the point the way leads to, his regiment's frame.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WayAsk {
    pub g: u32,
    pub p: Vec2,
    pub goal: Vec2,
}

/// One regiment's route as the kernel reads it: the points from the
/// route's start to its current waypoint (`points[k]`; the regiment's
/// frame stands in for it).
#[derive(Clone)]
pub struct RouteView {
    pub points: Arc<[Vec2]>,
    pub k: u32,
    /// The frame is on the last leg: men slow down near their slots.
    pub last_leg: bool,
    /// The ways planned for its cut-off men back to the route, each from a
    /// cell near where one stood to the frame; None when there are none.
    pub ways: Option<Arc<[Arc<[Vec2]>]>>,
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

/// The ways of a regiment's cut-off men (`WayAsk`), for one route.
#[derive(Clone, Default)]
struct Ways {
    /// The route's points they were planned for: another route drops them.
    route: Option<Arc<[Vec2]>>,
    /// Counts the routes; a search for an older one is dropped.
    epoch: u32,
    list: Vec<Arc<[Vec2]>>,
    /// The kernel's view of `list`.
    view: Option<Arc<[Arc<[Vec2]>]>>,
    /// Where the searches for this route started.
    asked: Vec<Vec2>,
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
    /// Per regiment, its cut-off men's ways; the searches for them waiting
    /// to go out; the asks the last install handed back.
    ways: Vec<Ways>,
    way_queue: VecDeque<PlanRequest>,
    asks: Vec<WayAsk>,
    /// The searches the next tick job takes.
    outgoing: Vec<PlanJob>,
    /// Plans handed back by the last install, and the searches it did not
    /// finish.
    incoming: Vec<Plan>,
    unfinished: Vec<PlanJob>,
    serial: u32,
    generation: u64,
    built: bool,
}

/// `FL_PATH=0` turns routes off.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("FL_PATH").is_ok_and(|v| v == "0"))
}

/// `FL_LOG_ROUTE=1`: log the route events, each a `[route]` line with the
/// sim tick. Off by default; it changes no decision, only what is logged.
fn log_route() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FL_LOG_ROUTE").is_ok())
}

/// The share of a regiment's living men that moves its frame on once
/// they are past the waypoint it waits at: a fifth. With a third, the men
/// already through are sent back and fill the passage's mouth before
/// enough are past; with fewer, a few men ahead of the rest move the whole
/// regiment's slots.
const FRAME_SHARE: f32 = 0.2;

/// The share of a regiment's living men past the waypoint its frame
/// waits at that moves the frame on (`update_routes`). The frame's other
/// two tests read the regiment's centre, and a regiment filing through a
/// passage narrower than its block stands strung along its route, its
/// centre never reaches the passage's far end; its head is through.
/// `FL_FRAME_SHARE` sets it; 0 turns it off, and only the two tests on
/// the centre move the frame.
pub fn frame_share() -> f32 {
    static S: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *S.get_or_init(|| crate::util::env_or("FL_FRAME_SHARE", FRAME_SHARE))
}

/// Whether the count `past` (waypoint, men past it, living men; frontline.rs
/// fills it at the tick's start) moves a frame waiting at waypoint `k` on:
/// it was taken at that waypoint, and the men past are `share` of the
/// living or more.
fn share_passed(past: Option<(u32, u32, u32)>, k: usize, share: f32) -> bool {
    matches!(past, Some((at, n, of)) if share > 0.0 && at as usize == k && of > 0 && n as f32 >= share * of as f32)
}

impl Routes {
    /// The grid, for the job and the debug view.
    pub fn nav(&self) -> Option<&Arc<NavGrid>> {
        self.nav.as_ref()
    }

    /// The grid as the placers read it.
    pub fn grid(&self) -> Option<&NavGrid> {
        self.nav.as_deref()
    }

    /// Bring the grid in step with `terrain`: built at the battle's setup
    /// and after a map change, refreshed after a crater. Returns whether
    /// it changed, in which case every route is stale.
    pub fn prepare(&mut self, terrain: &Terrain) -> bool {
        let t0 = std::time::Instant::now();
        let fresh = match self.nav.as_mut() {
            Some(nav) if self.built && nav.kind == terrain.kind => Arc::make_mut(nav).refresh(terrain),
            _ => {
                self.nav = NavGrid::build(terrain).map(Arc::new);
                true
            }
        };
        self.built = true;
        if fresh {
            if self.nav.is_some() {
                info!(
                    "[path] walkability grid ready for the {} map in {:.1} ms",
                    terrain.kind.label(),
                    t0.elapsed().as_secs_f32() * 1000.0
                );
            }
            // Routes made on the old ground are planned again.
            for r in self.per.iter_mut() {
                r.order = None;
            }
        }
        fresh
    }

    /// Fill a tick job's route inputs: the requests waiting to go out,
    /// each regiment's route, and the grid.
    pub(crate) fn fill_job(
        &mut self,
        requests: &mut Vec<PlanJob>,
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
            let ways = self.ways.get(g).filter(|w| same_route(w, route)).and_then(|w| w.view.clone());
            views[g] = route.map(|route| RouteView {
                points: route.points.clone(),
                k: route.k as u32,
                last_leg: route.k + 1 >= route.points.len(),
                ways,
            });
        }
        *nav = self.nav.clone();
    }

    /// Take the plans a tick job made, the searches it did not finish and
    /// its cut-off men's asks, chunk by chunk.
    pub(crate) fn hand_back(&mut self, plans: &mut Vec<Plan>, unfinished: &mut Vec<PlanJob>, asks: &mut [Vec<WayAsk>]) {
        self.incoming.append(plans);
        self.unfinished.append(unfinished);
        for chunk in asks {
            self.asks.append(chunk);
        }
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
                ways: self.ways.get(g).filter(|w| same_route(w, Some(r))).and_then(|w| w.view.clone()),
            }),
            _ => None,
        }
    }

    /// The line regiment `g`'s waiting frame is passed at: the waypoint's
    /// index, the waypoint, and the unit direction of the leg after it
    /// (toward `goal` on the last leg). A man is past the waypoint when
    /// he stands beyond the line through it across that leg. None without
    /// a ready route, at its last waypoint, or with `frame_share` off.
    pub(crate) fn frame_line(&self, g: usize, goal: Vec2) -> Option<(u32, Vec2, Vec2)> {
        if frame_share() <= 0.0 {
            return None;
        }
        let State::Ready(route) = &self.per.get(g)?.state else { return None };
        if route.k + 1 >= route.points.len() {
            return None;
        }
        let here = route.points[route.k];
        let dir = (next_point(&route.points, route.k, goal) - here).normalize_or_zero();
        (dir != Vec2::ZERO).then_some((route.k as u32, here, dir))
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

/// The widest `half_width` a regiment has across any way (m): the
/// block's half diagonal.
pub fn half_width_max(gd: &GroupData) -> f32 {
    half_width(gd, Vec2::Y).hypot(half_width(gd, Vec2::X))
}

/// Half a regiment's extent across the way `dir` (m): what must fit
/// between obstacles for the block to walk on unbroken.
pub fn half_width(gd: &GroupData, dir: Vec2) -> f32 {
    let across = Vec2::new(-dir.y, dir.x).normalize_or(Vec2::X);
    if gd.shape == crate::formation::FormShape::Blob {
        return crate::formation::blob_radius(gd.count);
    }
    let (files, ranks, pitch) = crate::formation::grid(gd);
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
    mut units: ResMut<crate::units::Units>,
    pipeline: Res<crate::sim::TickPipeline>,
) {
    let _tick_log = crate::frame_log::tick::time(crate::frame_log::tick::Part::Routes);
    if !enabled() {
        return;
    }
    let routes = &mut *routes;
    if !routes.built || terrain.is_changed() {
        routes.prepare(&terrain);
    }
    if units.generation != routes.generation {
        routes.generation = units.generation;
        routes.per.clear();
        routes.queue.clear();
        routes.outgoing.clear();
        routes.incoming.clear();
        routes.unfinished.clear();
        routes.ways.clear();
        routes.way_queue.clear();
        routes.asks.clear();
    }
    let n = groups.list.len();
    routes.per.resize_with(n, RegRoute::default);
    routes.ways.resize_with(n, Ways::default);
    let Some(nav) = routes.nav.clone() else {
        return;
    };
    let tick = pipeline.tick;

    // The cut-off men's asks, against the routes the last job saw.
    take_asks(routes);

    // The plans the last install handed back, and the searches it did
    // not finish, which go out again first unless a newer request for
    // their regiment has replaced them.
    for plan in std::mem::take(&mut routes.incoming) {
        if plan.req.man {
            adopt_way(routes, plan);
        } else {
            adopt(routes, &mut groups, plan);
        }
    }
    let mut unfinished = std::mem::take(&mut routes.unfinished);
    unfinished.retain(|job| {
        if job.req.man {
            return routes.ways.get(job.req.g as usize).is_some_and(|w| w.epoch == job.req.serial);
        }
        matches!(routes.per.get(job.req.g as usize).map(|r| &r.state),
            Some(State::Pending { serial, .. }) if *serial == job.req.serial)
    });
    routes.outgoing.append(&mut unfinished);

    for g in 0..n {
        notice_order(routes, &mut groups, &mut units, &nav, g, tick);
    }


    // Each frame along its route: on to the next waypoint once the
    // regiment's centre is within its half width of this one, or the
    // straight way to the next is clear at that width, or a share of its
    // men are past this one (`frame_share`). A holding regiment's route
    // goes once it holds elsewhere; opening its ranks is not that
    // (`formation::hold_point`).
    for g in 0..n {
        advance_frame(routes, &groups, &nav, g, tick);
    }
    // A regiment whose route changed drops its cut-off men's ways.
    for g in 0..n {
        sync_ways(routes, g);
    }

    // The searches the next job carries: the unfinished first, then new
    // requests up to the limit, and the cut-off men's up to theirs.
    let regiments = routes.outgoing.iter().filter(|j| !j.req.man).count();
    let mut men = routes.outgoing.len() - regiments;
    for _ in regiments..PLANS_PER_TICK {
        let Some(req) = routes.queue.pop_front() else { break };
        routes.outgoing.push(PlanJob { req, search: None });
    }
    while men < WAYS_PER_TICK {
        let Some(req) = routes.way_queue.pop_front() else { break };
        if routes.ways.get(req.g as usize).is_some_and(|w| w.epoch == req.serial) {
            routes.outgoing.push(PlanJob { req, search: None });
            men += 1;
        }
    }
}

/// The ways `w` were planned for `route`'s points.
#[inline]
fn same_route(w: &Ways, route: Option<&Arc<Route>>) -> bool {
    matches!((&w.route, route), (Some(a), Some(r)) if Arc::ptr_eq(a, &r.points))
}

/// Regiment `g`'s ways kept while its route's points are the ones they
/// were planned for; another route, or none, drops them and the asks.
#[inline]
fn sync_ways(routes: &mut Routes, g: usize) {
    let route = match &routes.per[g].state {
        State::Ready(r) | State::Pending { old: Some(r), .. } => Some(r),
        _ => None,
    };
    let w = &mut routes.ways[g];
    if same_route(w, route) || (w.route.is_none() && route.is_none()) {
        return;
    }
    w.route = route.map(|r| r.points.clone());
    w.epoch = w.epoch.wrapping_add(1);
    w.list.clear();
    w.view = None;
    w.asked.clear();
}

/// Queue a search for each cut-off man's ask, unless one was made near
/// where he stands for the same route, or the route has had its share.
fn take_asks(routes: &mut Routes) {
    for ask in std::mem::take(&mut routes.asks) {
        let Some(w) = routes.ways.get_mut(ask.g as usize) else { continue };
        if w.route.is_none()
            || w.asked.len() >= ASKS_PER_ROUTE
            || w.asked.iter().any(|q| q.distance_squared(ask.p) < ASK_NEAR * ASK_NEAR)
        {
            continue;
        }
        w.asked.push(ask.p);
        routes.way_queue.push_back(PlanRequest {
            g: ask.g,
            serial: w.epoch,
            start: ask.p,
            goal: ask.goal,
            half_w: crate::terrain::CELL,
            man: true,
        });
    }
}

/// Keep a cut-off man's way for his regiment, if it is for its route and
/// reaches the frame.
fn adopt_way(routes: &mut Routes, plan: Plan) {
    let g = plan.req.g as usize;
    let Some(w) = routes.ways.get_mut(g) else { return };
    if w.epoch != plan.req.serial {
        return;
    }
    info!(
        "[path] way for a cut-off man of regiment {g} from ({:.1}, {:.1}): planned in {:.2} ms over {} cells in {} ticks, {:.0} m, {} points{}",
        plan.req.start.x,
        plan.req.start.y,
        plan.ms,
        plan.cells,
        plan.ticks,
        plan.length(),
        plan.points.len(),
        if plan.reached { "" } else { ", none found" }
    );
    if !plan.reached || plan.points.len() < 2 || w.list.len() >= WAYS_PER_ROUTE {
        return;
    }
    w.list.push(plan.points.into());
    w.view = Some(w.list.clone().into());
}

/// A new order, an arrival or a moving attack target: the route kept,
/// dropped, asked for, or asked for again.
#[inline]
fn notice_order(
    routes: &mut Routes,
    groups: &mut Groups,
    units: &mut crate::units::Units,
    nav: &crate::nav::NavGrid,
    g: usize,
    tick: u32,
) {
    let gd = &groups.list[g];
    if gd.order != routes.per[g].order {
        routes.per[g].order = gd.order;
        if gd.order.is_none() {
            // Arrived: the route stays while the regiment holds at its
            // end.
            let kept = match &routes.per[g].state {
                State::Ready(r) | State::Pending { old: Some(r), .. } => {
                    let last = r.points.len() - 1;
                    (r.points[last].distance(crate::formation::hold_point(gd)) < 1.0)
                        .then(|| Arc::new(Route { points: r.points.clone(), k: last }))
                }
                _ => None,
            };
            if log_route() {
                let a = crate::formation::hold_point(gd);
                let what = match &routes.per[g].state {
                    State::Ready(r) | State::Pending { old: Some(r), .. } => {
                        let end = r.points[r.points.len() - 1];
                        if kept.is_some() {
                            format!("route kept at its end ({:.1}, {:.1})", end.x, end.y)
                        } else {
                            format!(
                                "route dropped: its end ({:.1}, {:.1}) is {:.2} m from where it holds",
                                end.x,
                                end.y,
                                end.distance(a)
                            )
                        }
                    }
                    _ => "it had no route".to_string(),
                };
                info!("[route] tick {tick} regiment {g}: order ended, it holds at ({:.1}, {:.1}); {what}", a.x, a.y);
            }
            // Coming round obstacles has mixed up the ranks: the men
            // take the nearest places in them. Done here, between the
            // install and the next job, so the next job reads the new
            // slots whether the tick is pipelined or inline.
            // Not while its men are still filing onto a lane: their
            // places are handed out as they arrive (formation.rs
            // `fill_lane_slots`).
            if kept.is_some()
                && groups.list[g].count > 0
                && !groups.list[g].state.is_broken()
                && groups.list[g].lane_fill.is_none()
            {
                crate::formation::assign_slots(units, g as u32, &mut groups.list[g]);
            }
            routes.per[g].state = kept.map_or(State::None, State::Ready);
            return;
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
        return;
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

/// The regiment's frame along its route, on to the next waypoint when it
/// is passed; a holding regiment's route dropped once it holds elsewhere.
#[inline]
fn advance_frame(routes: &mut Routes, groups: &Groups, nav: &crate::nav::NavGrid, g: usize, tick: u32) {
    let gd = &groups.list[g];
    let Some(goal) = groups.goal(g) else {
        if let State::Ready(r) = &routes.per[g].state
            && r.points[r.points.len() - 1].distance(crate::formation::hold_point(gd)) >= 1.0
        {
            if log_route() {
                let (a, end) = (crate::formation::hold_point(gd), r.points[r.points.len() - 1]);
                info!(
                    "[route] tick {tick} regiment {g}: it holds at ({:.1}, {:.1}), {:.2} m from its route's end ({:.1}, {:.1}): route dropped",
                    a.x,
                    a.y,
                    end.distance(a),
                    end.x,
                    end.y
                );
            }
            routes.per[g].state = State::None;
        }
        return;
    };
    let State::Ready(route) = &routes.per[g].state else { return };
    let mut k = route.k;
    let last = route.points.len() - 1;
    let w = half_width(gd, goal - gd.centroid);
    // How the last step was passed, for the log.
    let mut passed = "";
    while k < last {
        let next = next_point(&route.points, k, goal);
        let here = route.points[k];
        let near = gd.centroid.distance(here) <= w;
        if near || nav.straight_ok(gd.centroid, next, w) {
            passed = if near { "near" } else { "clear" };
            k += 1;
        } else if share_passed(gd.frame_past, k, frame_share()) {
            // The regiment's centre is short of the waypoint and the
            // way on is too narrow for its block, but enough of its
            // men are past it: it is filing through.
            passed = "share";
            k += 1;
        } else {
            break;
        }
    }
    if k != route.k {
        if log_route() {
            let c = gd.centroid;
            let men = match gd.frame_past {
                Some((at, n, of)) if at as usize == route.k => format!("; {n} of {of} men past waypoint {at}"),
                _ => String::new(),
            };
            info!(
                "[route] tick {tick} regiment {g}: frame from waypoint {} to {k} of {last}; centre ({:.1}, {:.1}), half width {w:.2} m; passed by: {passed}{men}",
                route.k,
                c.x,
                c.y
            );
        }
        let points = route.points.clone();
        routes.per[g].state = State::Ready(Arc::new(Route { points, k }));
    }
}

/// The point after waypoint `k` the frame moves on toward: the next
/// waypoint, or the live goal on the last leg.
#[inline]
fn next_point(points: &[Vec2], k: usize, goal: Vec2) -> Vec2 {
    if k + 2 == points.len() { goal } else { points[k + 1] }
}

/// What the frame's advance loop in `update_routes` tests right now, for
/// a regiment at waypoint `k` of `points` going to `goal`.
pub(crate) struct FrameTest {
    /// The waypoint the frame waits at, and the point after it.
    pub here: Vec2,
    pub next: Vec2,
    /// The regiment's centre's distance to `here` (m).
    pub dist: f32,
    /// The half width the two tests on the centre use (m): the frame moves
    /// on when the centre is within it of `here`, or when the straight way
    /// to `next` is clear for it.
    pub w: f32,
    pub clear: bool,
}

/// The advance loop's two tests on the centre for the log, mirrored: the
/// same `here`, `next` and half width, and the straight check from the
/// centre. The third, the share of men past the waypoint, is not in it.
/// None once the frame is at the route's last waypoint.
pub(crate) fn frame_test(nav: &NavGrid, gd: &GroupData, points: &[Vec2], k: usize, goal: Vec2) -> Option<FrameTest> {
    if k + 1 >= points.len() {
        return None;
    }
    let (here, next) = (points[k], next_point(points, k, goal));
    let w = half_width(gd, goal - gd.centroid);
    Some(FrameTest { here, next, dist: gd.centroid.distance(here), w, clear: nav.straight_ok(gd.centroid, next, w) })
}

/// Queue a plan for regiment `g`, superseding any it waits for.
fn request(routes: &mut Routes, g: usize, start: Vec2, goal: Vec2, half_w: f32) {
    routes.serial = routes.serial.wrapping_add(1);
    let serial = routes.serial;
    routes.queue.retain(|q| q.g as usize != g);
    routes.queue.push_back(PlanRequest { g: g as u32, serial, start, goal, half_w, man: false });
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
        "[path] regiment {g}: planned in {:.2} ms over {} cells in {} ticks, route {:.0} m against {:.0} m straight, {} waypoints{}",
        plan.ms,
        plan.cells,
        plan.ticks,
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

/// Advance a tick job's searches, in parallel on the pool, each by one
/// budget: the finished ones become plans (in request order), the rest
/// go back for the next job. Runs on the sim worker after the kernel.
pub(crate) fn plan_all(nav: &NavGrid, jobs: &mut Vec<PlanJob>, out: &mut Vec<Plan>) {
    out.clear();
    if jobs.is_empty() {
        return;
    }
    let t0 = std::time::Instant::now();
    // A tick that has to build a cost table does only that; the searches
    // begin with the next job, so no tick pays for both.
    let reqs: Vec<PlanRequest> = jobs.iter().map(|j| j.req).collect();
    if nav.prepare_costs(&reqs) {
        info!(
            "[path] cost table built in the tick job: {:.2} ms; {} searches wait for the next",
            t0.elapsed().as_secs_f32() * 1000.0,
            jobs.len()
        );
        return;
    }
    let taken = std::mem::take(jobs);
    let results: Vec<(PlanJob, Option<Plan>)> = crate::util::sim_scope(|scope| {
        for mut job in taken {
            scope.spawn(async move {
                let mut search = job.search.take().unwrap_or_else(|| nav.begin(&job.req));
                let plan = nav.advance(&mut search, PLAN_BUDGET);
                if plan.is_none() {
                    job.search = Some(search);
                }
                (job, plan)
            });
        }
    });
    let n = results.len();
    for (job, plan) in results {
        match plan {
            Some(plan) => out.push(plan),
            None => jobs.push(job),
        }
    }
    info!(
        "[path] {} searches in the tick job: {:.2} ms, {} finished",
        n,
        t0.elapsed().as_secs_f32() * 1000.0,
        out.len()
    );
}

/// Where a man of a regiment on a route walks this tick (`at`), whether
/// he keeps his full pace (`through`: he is not yet near the end of the
/// way), and whether he is cut off from the route with no way of his own
/// in sight (`ask`: he asks for one, `WayAsk`).
pub(crate) struct ManTarget {
    pub at: Vec2,
    pub through: bool,
    pub ask: bool,
}

/// Where a man of a regiment on a route walks this tick (`ManTarget`).
///
/// `frame` is his regiment's frame point (its current waypoint, or the
/// goal on the last leg), `slot` his own slot around it and `kept` the
/// point he chose at his last read. With his straight way to his slot open
/// as far as he sees, he walks to it.
/// Otherwise he follows the route: from his place on it (the nearest
/// point of its legs up to the frame), the farthest of a few points ahead
/// he can walk to straight. Failing any, a point behind his place he can
/// walk to, or else the nearest waypoint behind it he can. Failing that
/// too he is cut off: he walks one of the ways planned for his regiment's
/// cut-off men (`own_way`). With none in sight, he asks for one and steps
/// around what stands in front of him meanwhile, to the side his place on
/// the route lies, where the ground along the step is open.
///
/// A man on such a way (`kept` is one of its points) leaves it only for a
/// slot or a point of the route he sees the whole way to, within
/// `nav::SIGHT`: a line open only as far as he looks is no way back from
/// outside a wall, and walking it would take him back to where he was cut
/// off.
#[inline]
pub(crate) fn man_target(
    nav: &NavGrid,
    route: &RouteView,
    frame: Vec2,
    p: Vec2,
    slot: Vec2,
    kept: Option<Vec2>,
    i: usize,
) -> ManTarget {
    let go = |at: Vec2, through: bool| ManTarget { at, through, ask: false };
    let on_way = kept.is_some_and(|q| route.ways.as_deref().is_some_and(|ways| ways.iter().any(|w| w.contains(&q))));
    let sees = |q: Vec2| nav.walk_line(p, q) && (!on_way || p.distance_squared(q) <= crate::nav::SIGHT * crate::nav::SIGHT);
    if sees(slot) {
        return go(slot, !route.last_leg);
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
        if sees(point(j)) {
            return go(point(j), true);
        }
    }
    // Along his leg from his place toward its end, the farthest first.
    let end = point(leg + 1);
    let len = at.distance(end);
    for &ahead in &FOLLOW_AHEAD {
        if ahead >= len {
            continue;
        }
        let q = at.lerp(end, ahead / len);
        if sees(q) {
            return go(q, true);
        }
    }
    // Back along the route from his place, then the waypoints behind it,
    // the nearest first. His place is the nearest point seen from above,
    // which on a wall is the walk above a man at its foot: the stair's
    // foot, a waypoint behind, is where he can go.
    let start = point(leg);
    let back = at.distance(start);
    for &b in &FOLLOW_BACK {
        if b >= back {
            break;
        }
        let q = at.lerp(start, b / back);
        if sees(q) {
            return go(q, true);
        }
    }
    for j in (0..=leg).rev() {
        if sees(point(j)) {
            return go(point(j), true);
        }
    }
    // Nothing on the route in reach: he is cut off from it. A way planned
    // for his regiment's cut-off men that he can walk onto leads him back.
    if let Some(q) = route.ways.as_deref().and_then(|ways| own_way(nav, ways, p)) {
        return go(q, true);
    }
    // None in sight: he asks for one, and steps round what is in front of
    // him meanwhile, turning first to the side his place on the route
    // lies, where the ground along the step is open.
    let ask = |at: Vec2| ManTarget { at, through: true, ask: true };
    let dir = (end - p).normalize_or_zero();
    if dir == Vec2::ZERO {
        return ask(end);
    }
    let open = |d: Vec2| nav.walk_line(p, p + d * 2.5);
    if open(dir) {
        return ask(end);
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
                return ask(p + d * 4.0);
            }
        }
    }
    ask(end)
}

/// The point a cut-off man at `p` walks to on his regiment's cut-off men's
/// `ways`: of the points he can walk to straight, the one with the least
/// way left after it to its end. A way's end he stands at (within
/// `END_NEAR`) leads him nowhere more: the route has moved on since it was
/// planned. A line check per point, from each way's end back, stopping
/// where a way has more left than the best found.
#[inline]
fn own_way(nav: &NavGrid, ways: &[Arc<[Vec2]>], p: Vec2) -> Option<Vec2> {
    const END_NEAR: f32 = 2.0;
    let mut best: Option<(f32, Vec2)> = None;
    for way in ways {
        let mut left = 0.0;
        for j in (0..way.len()).rev() {
            if j + 1 < way.len() {
                left += way[j].distance(way[j + 1]);
            } else if p.distance_squared(way[j]) < END_NEAR * END_NEAR {
                break;
            }
            if best.is_some_and(|(b, _)| left >= b) {
                break;
            }
            if nav.walk_line(p, way[j]) {
                best = Some((left, way[j]));
                break;
            }
        }
    }
    best.map(|(_, q)| q)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn routes_at(points: &[Vec2], k: usize) -> Routes {
        let route = Route { points: points.into(), k };
        Routes { per: vec![RegRoute { state: State::Ready(Arc::new(route)), ..RegRoute::default() }], ..Routes::default() }
    }

    /// The line a waiting frame is passed at lies through its waypoint,
    /// across the leg after it; on the last leg that leg runs to the goal.
    #[test]
    fn a_waiting_frame_is_passed_across_the_leg_after_its_waypoint() {
        let points = [Vec2::ZERO, Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0), Vec2::new(20.0, 10.0)];
        let goal = Vec2::new(10.0, 30.0);
        assert_eq!(routes_at(&points, 1).frame_line(0, goal), Some((1, points[1], Vec2::Y)));
        // The last leg: from the waypoint before the end toward the goal,
        // not toward the route's own last point.
        assert_eq!(routes_at(&points, 2).frame_line(0, goal), Some((2, points[2], Vec2::Y)));
        // At its last waypoint the frame waits for nothing.
        assert_eq!(routes_at(&points, 3).frame_line(0, goal), None);
        // No route, no regiment, and a route still being planned: no line.
        assert_eq!(Routes::default().frame_line(0, goal), None);
        let mut pending = routes_at(&points, 1);
        pending.per[0].state = State::Pending { serial: 1, old: None };
        assert_eq!(pending.frame_line(0, goal), None);
    }

    /// The soldiers get the grid every tick on a map that has one, with
    /// no route and no search out: their own walk tests read it while
    /// every regiment stands.
    #[test]
    fn the_job_gets_the_grid_with_no_route_and_no_request() {
        let terrain = crate::terrain::build_terrain(crate::terrain::MapKind::WallDebug);
        let mut routes = Routes::default();
        routes.prepare(&terrain);
        let (mut requests, mut views, mut nav) = (Vec::new(), Vec::new(), None);
        routes.fill_job(&mut requests, &mut views, &mut nav, 2);
        assert!(requests.is_empty() && views.iter().all(Option::is_none));
        assert!(nav.is_some());
    }

    /// The frame moves on by its men once a fifth of the living are past
    /// the waypoint it waits at, and only on a count taken at that waypoint.
    #[test]
    fn a_fifth_of_the_living_men_past_its_waypoint_moves_a_frame_on() {
        assert!(share_passed(Some((4, 40, 200)), 4, FRAME_SHARE));
        assert!(!share_passed(Some((4, 39, 200)), 4, FRAME_SHARE));
        // A count taken at an earlier waypoint does not move this one.
        assert!(!share_passed(Some((3, 200, 200)), 4, FRAME_SHARE));
        assert!(!share_passed(Some((4, 0, 0)), 4, FRAME_SHARE));
        assert!(!share_passed(None, 4, FRAME_SHARE));
        // Switched off, no count moves it.
        assert!(!share_passed(Some((4, 200, 200)), 4, 0.0));
    }

    /// The route a regiment inside Wall Debug's town planned from its
    /// centre there, and the frame at its last point.
    fn route_inside() -> RouteView {
        let points = [Vec2::new(88.0, -141.0), Vec2::new(84.0, -152.0), Vec2::new(46.0, -152.0), Vec2::new(44.0, -150.0)];
        RouteView { points: points.as_slice().into(), k: 3, last_leg: true, ways: None }
    }

    /// A man at the wall's outer face, his regiment's route all inside the
    /// town, sees no point of it: he asks for a way. The way planned for him
    /// starts where he can walk to straight, every leg of it is one he
    /// walks straight, it goes round the wall's east end (the gate is
    /// closed), and it ends where the route is in sight. Walking it a metre
    /// at a time, he never asks again and comes to a point of the route.
    #[test]
    fn a_man_cut_off_outside_the_wall_walks_his_own_way_round_its_end() {
        use crate::town_layout::{TOWERS_X, WALL_Z};
        let terrain = crate::terrain::build_terrain(crate::terrain::MapKind::WallDebug);
        let nav = NavGrid::build(&terrain).expect("Wall Debug has a grid");
        let mut route = route_inside();
        let frame = route.points[3];
        let p = Vec2::new(16.0, -125.4);
        let cut = man_target(&nav, &route, frame, p, frame, None, 0);
        assert!(cut.ask, "the man at {p} sees the route: aims at {}", cut.at);
        let req = PlanRequest { g: 0, serial: 0, start: p, goal: frame, half_w: crate::terrain::CELL, man: true };
        let plan = nav.plan(&req);
        assert!(plan.reached && plan.points.len() >= 2, "no way: {:?}", plan.points);
        assert!(plan.cells <= crate::nav::MAN_CELLS);
        for w in plan.points.windows(2) {
            assert!(nav.walk_line(w[0], w[1]), "the leg {} -> {} is not walked straight", w[0], w[1]);
        }
        let crossing = plan.points.windows(2).find_map(|w| {
            let (a, b) = (w[0], w[1]);
            ((a.y - WALL_Z) * (b.y - WALL_Z) <= 0.0 && a.y != b.y).then(|| a.x + (b.x - a.x) * (WALL_Z - a.y) / (b.y - a.y))
        });
        let x = crossing.expect("the way crosses the wall's line");
        assert!(x > TOWERS_X[3], "the way crosses the wall's line at x {x}, not round its east end");
        let end = *plan.points.last().unwrap();
        assert!(route.points.iter().any(|&q| nav.walk_line(end, q)), "the way's end {end} sees no point of the route");
        route.ways = Some(vec![Arc::<[Vec2]>::from(plan.points.clone())].into());
        let first = man_target(&nav, &route, frame, p, frame, None, 0);
        assert!(!first.ask && nav.walk_line(p, first.at), "his first target {} is not in his reach", first.at);
        let mut q = p;
        let mut kept = Some(first.at);
        let mut on_route = false;
        for _ in 0..800 {
            let aim = man_target(&nav, &route, frame, q, frame, kept, 0);
            kept = Some(aim.at);
            assert!(!aim.ask, "he asks again at {q}");
            if route.points.contains(&aim.at) {
                on_route = true;
                break;
            }
            let next = q + (aim.at - q).clamp_length_max(1.0);
            assert!(nav.walk_line(q, next), "a step from {q} to {next} crosses blocked ground");
            q = next;
        }
        assert!(on_route, "he walked to {q} and never reached the route");
    }

    /// A man who can walk straight to a point of his regiment's route
    /// follows it, with no way of his own and no ask.
    #[test]
    fn a_man_who_sees_his_route_asks_for_no_way() {
        let terrain = crate::terrain::build_terrain(crate::terrain::MapKind::WallDebug);
        let nav = NavGrid::build(&terrain).expect("Wall Debug has a grid");
        let route = route_inside();
        let frame = route.points[3];
        let p = Vec2::new(70.0, -160.0);
        let aim = man_target(&nav, &route, frame, p, frame, None, 0);
        assert!(!aim.ask, "the man at {p} asks for a way");
        assert!(nav.walk_line(p, aim.at));
    }
}
