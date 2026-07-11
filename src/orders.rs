//! Regiments (groups), selection, and player orders.
//!
//! A group IS a regiment: a permanent block of units sharing an order,
//! kind, and (later) morale. Selection: hold LMB and draw a line/loop over
//! the field; regiments with enough units near the stroke are selected.
//! Right-click: attack-move — each selected regiment's ORDER is the target
//! translated by its offset from the selection centroid, so a group move
//! preserves the army's arrangement. Units translate the regiment order by
//! their own `home` offset (block moves, never converges).

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::movement::DebugViz;
use crate::terrain::Terrain;
use crate::units::Units;

pub const PLAYER_TEAM: u8 = 0;
const SELECT_RADIUS: f32 = 14.0;
/// Groups whose centroid gets this close to their order target go idle.
const ARRIVE_CLEAR: f32 = 18.0;
/// Right-clicking within this range of an enemy regiment's centroid is an
/// attack order on that regiment (else it's a move order to the point).
const ATTACK_PICK_RADIUS: f32 = 20.0;

/// A regiment order, TW-style: move orders target ground and end on
/// arrival; attack orders hunt a specific enemy regiment (the destination
/// re-resolves to its centroid every tick) and end when it's wiped.
#[derive(Clone, Copy, PartialEq)]
pub enum Order {
    Move(Vec2),
    /// Index into `Groups::list`.
    Attack(u32),
}

/// Regiment morale state.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RegState {
    Steady,
    /// Broken: uncontrollable, flees toward its own map edge. `since` is
    /// the morale-system tick the break happened on (rally timing).
    Routing { since: u32 },
    /// Too depleted to ever rally; flees until despawn.
    Shattered,
}

impl RegState {
    #[inline]
    pub fn is_broken(&self) -> bool {
        !matches!(self, RegState::Steady)
    }
}

pub struct GroupData {
    pub team: u8,
    /// Regiments are homogeneous; index into `unit_types::TYPES`.
    pub kind: u8,
    /// Current order; None = hold at anchor.
    pub order: Option<Order>,
    /// Regiment reference point: units hold at `anchor + home`. Updated to
    /// the order target on arrival.
    pub anchor: Vec2,
    pub count: usize,
    /// Strength at spawn (casualty fraction for morale).
    pub initial_count: usize,
    // --- refreshed every fixed tick by the frontline pass ---
    pub centroid: Vec2,
    /// TW rule: one soldier of the regiment fighting = the whole regiment
    /// is engaged (ground truth from per-unit melee state, with a short
    /// hold to bridge swing-cycle gaps).
    pub engaged: bool,
    /// Ticks of `engaged` left since the last soldier fought.
    pub engage_hold: u8,
    /// In the charge phase: attack order, inside charge range of the
    /// target, not yet in contact. Drives the war cry + sprint pose.
    pub charging: bool,
    // --- morale (regiments.rs updates per tick) ---
    pub morale: f32,
    pub state: RegState,
    /// Deaths since the last morale tick (tallied by the damage apply pass).
    pub recent_deaths: u32,
}

impl GroupData {
    pub fn new(team: u8, kind: u8, anchor: Vec2, count: usize) -> Self {
        Self {
            team,
            kind,
            order: None,
            anchor,
            count,
            initial_count: count,
            centroid: anchor,
            engaged: false,
            engage_hold: 0,
            charging: false,
            morale: 100.0,
            state: RegState::Steady,
            recent_deaths: 0,
        }
    }
}

#[derive(Resource, Default)]
pub struct Groups {
    pub list: Vec<GroupData>,
}

impl Groups {
    /// A group's order destination this tick: Move goes to the point,
    /// Attack chases the target regiment's current centroid.
    pub fn goal(&self, g: usize) -> Option<Vec2> {
        match self.list[g].order? {
            Order::Move(p) => Some(p),
            Order::Attack(t) => {
                let t = &self.list[t as usize];
                (t.count > 0).then_some(t.centroid)
            }
        }
    }
}

/// Selected regiments (player team only) + cached unit total.
#[derive(Resource, Default)]
pub struct Selection {
    pub regiments: Vec<bool>,
    pub count_units: usize,
}

/// In-progress drag: projected ground points of the selection line.
#[derive(Resource, Default)]
struct DragLine {
    points: Vec<Vec2>,
    active: bool,
}

pub struct OrdersPlugin;

impl Plugin for OrdersPlugin {
    fn build(&self, app: &mut App) {
        // Groups start empty; the regiment spawn (regiments.rs) fills the
        // list once at startup and it stays FIXED for the whole battle —
        // stable indices are what make `units.group` a permanent regiment id.
        app.init_resource::<Groups>()
            .init_resource::<Selection>()
            .init_resource::<DragLine>()
            .init_resource::<Hover>()
            .add_systems(
                Update,
                (
                    drag_select,
                    update_hover,
                    issue_order,
                    test_orders_script,
                    draw_order_gizmos,
                ),
            )
            .add_systems(FixedUpdate, clear_arrived_orders.after(crate::movement::step_sim));
    }
}

fn cursor_ground_point(
    window: &Window,
    camera: &Camera,
    cam_tf: &GlobalTransform,
    terrain: &Terrain,
) -> Option<Vec2> {
    let cursor = window.cursor_position()?;
    let ray = camera.viewport_to_world(cam_tf, cursor).ok()?;
    terrain.raycast(ray).map(|hit| Vec2::new(hit.x, hit.z))
}

#[allow(clippy::too_many_arguments)] // bevy system params
fn drag_select(
    buttons: Res<ButtonInput<MouseButton>>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform)>,
    terrain: Res<Terrain>,
    units: Res<Units>,
    groups: Res<Groups>,
    mut drag: ResMut<DragLine>,
    mut selection: ResMut<Selection>,
) {
    let Ok(window) = window.single() else { return };
    let Ok((camera, cam_tf)) = camera.single() else {
        return;
    };

    if buttons.just_pressed(MouseButton::Left) {
        drag.points.clear();
        drag.active = true;
    }
    if drag.active
        && buttons.pressed(MouseButton::Left)
        && let Some(p) = cursor_ground_point(window, camera, cam_tf, &terrain)
        && drag.points.last().is_none_or(|l| l.distance(p) > 3.0)
    {
        drag.points.push(p);
    }

    if drag.active && buttons.just_released(MouseButton::Left) {
        drag.active = false;
        if drag.points.is_empty() {
            return;
        }
        select_along_line(&drag.points, &units, &groups, &mut selection);
        info!(
            "selected {} regiments ({} units)",
            selection.regiments.iter().filter(|s| **s).count(),
            selection.count_units
        );
    }
}

/// A stroke whose endpoints nearly meet is treated as a closed loop.
fn stroke_is_closed(line: &[Vec2]) -> bool {
    if line.len() < 3 {
        return false;
    }
    let path_len: f32 = line.windows(2).map(|w| w[0].distance(w[1])).sum();
    line[0].distance(*line.last().unwrap()) < (path_len * 0.15).clamp(10.0, 40.0)
}

/// Even-odd crossing test against the stroke polygon (closing edge implied).
fn point_in_poly(p: Vec2, poly: &[Vec2]) -> bool {
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[j]);
        if (a.y > p.y) != (b.y > p.y) {
            let x = a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x);
            if p.x < x {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Per-unit stroke hits, promoted to whole regiments: a regiment is
/// selected when enough of its units are near the stroke (a sloppy lasso
/// edge shouldn't grab a neighboring regiment).
pub fn select_along_line(
    line: &[Vec2],
    units: &Units,
    groups: &Groups,
    selection: &mut Selection,
) {
    selection.regiments.clear();
    selection.regiments.resize(groups.list.len(), false);
    selection.count_units = 0;
    let mut hits = vec![0usize; groups.list.len()];

    // Bounding box early-out around the whole stroke.
    let mut lo = Vec2::splat(f32::MAX);
    let mut hi = Vec2::splat(f32::MIN);
    for p in line {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    lo -= SELECT_RADIUS;
    hi += SELECT_RADIUS;

    // War-of-Dots style: an enclosing stroke selects everything inside the
    // loop, in addition to units near the stroke itself.
    let closed = stroke_is_closed(line);
    let r2 = SELECT_RADIUS * SELECT_RADIUS;
    for i in 0..units.len() {
        if units.team[i] != PLAYER_TEAM || units.death_t[i] > 0 {
            continue;
        }
        let p = Vec2::new(units.pos[i].x, units.pos[i].z);
        if p.x < lo.x || p.x > hi.x || p.y < lo.y || p.y > hi.y {
            continue;
        }
        let mut hit = closed && point_in_poly(p, line);
        if !hit {
            if line.len() == 1 {
                hit = p.distance_squared(line[0]) < r2;
            } else {
                for w in line.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    let ab = b - a;
                    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
                    if p.distance_squared(a + ab * t) < r2 {
                        hit = true;
                        break;
                    }
                }
            }
        }
        if hit {
            hits[units.group[i] as usize] += 1;
        }
    }

    // Promote unit hits to whole regiments.
    for (g, group) in groups.list.iter().enumerate() {
        if group.team != PLAYER_TEAM || group.count == 0 {
            continue;
        }
        let threshold = 8.max(group.count * 3 / 100);
        if hits[g] >= threshold.min(group.count) {
            selection.regiments[g] = true;
            selection.count_units += group.count;
        }
    }
}

/// Attack-move `selected` regiments so the formation ARRANGEMENT is
/// preserved: each regiment's order is the target translated by its offset
/// from the selection centroid. Shared by the RMB handler, test scripts,
/// and (later) the AI.
pub fn order_regiments(groups: &mut Groups, selected: &[usize], target: Vec2) {
    // Broken regiments are uncontrollable.
    let selected: Vec<usize> = selected
        .iter()
        .copied()
        .filter(|&g| !groups.list[g].state.is_broken())
        .collect();
    if selected.is_empty() {
        return;
    }
    let centroid: Vec2 =
        selected.iter().map(|&g| groups.list[g].centroid).sum::<Vec2>() / selected.len() as f32;
    for &g in &selected {
        let group = &mut groups.list[g];
        group.order = Some(Order::Move(target + (group.centroid - centroid)));
    }
}

/// Attack-order `selected` regiments at one enemy regiment: everyone hunts
/// the same target (TW behavior — no arrangement translation).
pub fn attack_regiments(groups: &mut Groups, selected: &[usize], target: u32) {
    for &g in selected {
        if !groups.list[g].state.is_broken() {
            groups.list[g].order = Some(Order::Attack(target));
        }
    }
}

/// Regiment under a ground point, if any: nearest live regiment of the
/// wanted side whose centroid is within the pick radius.
fn regiment_at(groups: &Groups, p: Vec2, enemy: bool) -> Option<u32> {
    groups
        .list
        .iter()
        .enumerate()
        .filter(|(_, g)| (g.team != PLAYER_TEAM) == enemy && g.count > 0)
        .map(|(g, gd)| (g, gd.centroid.distance(p)))
        .filter(|(_, d)| *d < ATTACK_PICK_RADIUS)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(g, _)| g as u32)
}

fn enemy_regiment_at(groups: &Groups, p: Vec2) -> Option<u32> {
    regiment_at(groups, p, true)
}

/// Regiments under the cursor this frame. `enemy` doubles as the
/// attack-order preview: it is what a right-click would target (same
/// pick logic), so the render tint and the inspect panel key off it.
#[derive(Resource, Default)]
pub struct Hover {
    pub enemy: Option<u32>,
    pub own: Option<u32>,
}

fn update_hover(
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform)>,
    terrain: Res<Terrain>,
    groups: Res<Groups>,
    mut hover: ResMut<Hover>,
) {
    hover.enemy = None;
    hover.own = None;
    let Ok(window) = window.single() else { return };
    let Ok((camera, cam_tf)) = camera.single() else {
        return;
    };
    let Some(p) = cursor_ground_point(window, camera, cam_tf, &terrain) else {
        return;
    };
    hover.enemy = regiment_at(&groups, p, true);
    hover.own = regiment_at(&groups, p, false);
}

fn issue_order(
    buttons: Res<ButtonInput<MouseButton>>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform)>,
    terrain: Res<Terrain>,
    mut groups: ResMut<Groups>,
    selection: Res<Selection>,
) {
    if !buttons.just_pressed(MouseButton::Right) || selection.count_units == 0 {
        return;
    }
    let Ok(window) = window.single() else { return };
    let Ok((camera, cam_tf)) = camera.single() else {
        return;
    };
    let Some(target) = cursor_ground_point(window, camera, cam_tf, &terrain) else {
        return;
    };
    let selected: Vec<usize> = selection
        .regiments
        .iter()
        .enumerate()
        .filter(|(_, s)| **s)
        .map(|(g, _)| g)
        .collect();
    if let Some(t) = enemy_regiment_at(&groups, target) {
        attack_regiments(&mut groups, &selected, t);
        info!(
            "{} regiments ({} units) ATTACK regiment {t}",
            selected.len(),
            selection.count_units,
        );
    } else {
        order_regiments(&mut groups, &selected, target);
        info!(
            "{} regiments ({} units) move to ({:.0}, {:.0})",
            selected.len(),
            selection.count_units,
            target.x,
            target.y
        );
    }
}

/// Order lifecycle. Move: ends on arrival — the order point becomes the
/// new anchor (holding is a standing order). Attack: never "arrives", it
/// ends when the target regiment is wiped (hold in place there). Uses the
/// centroid refreshed by the frontline pass each tick.
pub fn clear_arrived_orders(mut groups: ResMut<Groups>) {
    let counts: Vec<usize> = groups.list.iter().map(|g| g.count).collect();
    for (g, group) in groups.list.iter_mut().enumerate() {
        match group.order {
            Some(Order::Move(t)) => {
                if group.count > 0
                    && !group.engaged
                    && group.centroid.distance(t) < ARRIVE_CLEAR
                {
                    group.anchor = t;
                    group.order = None;
                    info!("regiment {g} arrived, holding");
                }
            }
            Some(Order::Attack(t)) if counts[t as usize] == 0 => {
                group.anchor = group.centroid;
                group.order = None;
                info!("regiment {g} attack target {t} destroyed, holding");
            }
            _ => {}
        }
    }
}

fn draw_order_gizmos(
    viz: Res<DebugViz>,
    drag: Res<DragLine>,
    groups: Res<Groups>,
    terrain: Res<Terrain>,
    mut gizmos: Gizmos,
) {
    // Attack indicators are player-facing UI (not debug viz): a red marker
    // hangs over every regiment the player is attacking.
    let mut marked = vec![false; groups.list.len()];
    for group in &groups.list {
        if group.team == PLAYER_TEAM
            && let Some(Order::Attack(t)) = group.order
        {
            marked[t as usize] = true;
        }
    }
    for (t, m) in marked.iter().enumerate() {
        if !*m || groups.list[t].count == 0 {
            continue;
        }
        let c = groups.list[t].centroid;
        let top = Vec3::new(c.x, terrain.height_at(c.x, c.y) + 15.0, c.y);
        gizmos.arrow(top, top - Vec3::Y * 4.5, Color::srgb(1.0, 0.25, 0.2));
    }

    if !viz.0 {
        return;
    }
    // Selection line being drawn; when the stroke would close into a loop,
    // show the implied closing edge.
    if drag.active && drag.points.len() > 1 {
        let lift = |p: &Vec2| Vec3::new(p.x, terrain.height_at(p.x, p.y) + 1.0, p.y);
        let pts: Vec<Vec3> = drag.points.iter().map(lift).collect();
        gizmos.linestrip(pts, Color::srgb(0.95, 0.95, 0.4));
        if stroke_is_closed(&drag.points) {
            gizmos.line(
                lift(drag.points.last().unwrap()),
                lift(&drag.points[0]),
                Color::srgb(0.5, 1.0, 0.5),
            );
        }
    }
    // Move-order targets (attack orders have the marker above).
    for group in &groups.list {
        if let Some(Order::Move(t)) = group.order {
            let base = Vec3::new(t.x, terrain.height_at(t.x, t.y) + 0.5, t.y);
            let color = if group.team == 0 {
                Color::srgb(0.4, 0.8, 1.0)
            } else {
                Color::srgb(1.0, 0.6, 0.2)
            };
            gizmos.circle(
                Isometry3d::new(base, Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
                3.0,
                color,
            );
            gizmos.line(base, base + Vec3::Y * 10.0, color);
        }
    }
}

/// RMS spread of a regiment's living units around its centroid — the
/// block-coherence metric for the orders test.
fn regiment_spread(units: &Units, g: u32) -> f32 {
    let (mut sum, mut c, mut centroid) = (0.0f32, 0usize, Vec2::ZERO);
    for i in 0..units.len() {
        if units.group[i] == g && units.death_t[i] == 0 {
            centroid += Vec2::new(units.pos[i].x, units.pos[i].z);
            c += 1;
        }
    }
    if c == 0 {
        return 0.0;
    }
    centroid /= c as f32;
    for i in 0..units.len() {
        if units.group[i] == g && units.death_t[i] == 0 {
            sum += Vec2::new(units.pos[i].x, units.pos[i].z).distance_squared(centroid);
        }
    }
    (sum / c as f32).sqrt()
}

/// FL_TEST_ORDERS=1: regiment cohesion + arrangement-preserving group
/// moves. Stage 0: enclosure-select regiments (exercises the promote
/// path), long-march ONE regiment 350 m and compare its RMS spread on
/// arrival (acceptance: < 1.5x). Stage 1: group-move several regiments
/// and log how well their relative arrangement survived.
fn test_orders_script(
    time: Res<Time>,
    units: Res<Units>,
    mut groups: ResMut<Groups>,
    mut selection: ResMut<Selection>,
    mut stage: Local<u32>,
    mut watch: Local<Option<(u32, f32)>>,
    mut arrangement: Local<Vec<(usize, Vec2)>>,
) {
    if std::env::var("FL_TEST_ORDERS").is_err() {
        return;
    }
    let t = time.elapsed_secs();
    match *stage {
        0 if t > 5.0 => {
            // Enclosure stroke around a spot inside the blue army: must
            // promote to whole regiments.
            let center = Vec2::new(-150.0, -90.0);
            let stroke: Vec<Vec2> = (0..28)
                .map(|k| {
                    let a = k as f32 / 28.0 * std::f32::consts::TAU;
                    center + Vec2::new(a.cos(), a.sin()) * 45.0
                })
                .collect();
            select_along_line(&stroke, &units, &groups, &mut selection);
            info!(
                "[test] enclosure selected {} regiments ({} units)",
                selection.regiments.iter().filter(|s| **s).count(),
                selection.count_units
            );
            // Long-march the blue regiment nearest the enclosure center.
            let g = groups
                .list
                .iter()
                .enumerate()
                .filter(|(_, gr)| gr.team == PLAYER_TEAM && gr.count > 0)
                .min_by(|a, b| {
                    a.1.centroid
                        .distance_squared(center)
                        .total_cmp(&b.1.centroid.distance_squared(center))
                })
                .map(|(g, _)| g as u32)
                .unwrap();
            let spread = regiment_spread(&units, g);
            groups.list[g as usize].order =
                Some(Order::Move(groups.list[g as usize].centroid + Vec2::new(350.0, 60.0)));
            *watch = Some((g, spread));
            info!("[test] regiment {g} long-march ordered, spread at departure {spread:.1} m");
            selection.regiments.fill(false);
            selection.count_units = 0;
            *stage = 1;
        }
        1 => {
            if let Some((g, spread0)) = *watch
                && groups.list[g as usize].order.is_none()
            {
                let spread = regiment_spread(&units, g);
                info!(
                    "[test] regiment {g} arrived: spread {spread:.1} m vs {spread0:.1} m at departure (ratio {:.2})",
                    spread / spread0.max(0.01)
                );
                *stage = 2;
            }
        }
        2 if t > 45.0 => {
            // Group-move: order several blue regiments as one formation.
            let selected: Vec<usize> = groups
                .list
                .iter()
                .enumerate()
                .filter(|(_, gr)| gr.team == PLAYER_TEAM && gr.count > 0 && gr.order.is_none())
                .take(10)
                .map(|(g, _)| g)
                .collect();
            let centroid: Vec2 = selected.iter().map(|&g| groups.list[g].centroid).sum::<Vec2>()
                / selected.len().max(1) as f32;
            *arrangement = selected
                .iter()
                .map(|&g| (g, groups.list[g].centroid - centroid))
                .collect();
            order_regiments(&mut groups, &selected, centroid + Vec2::new(-120.0, -100.0));
            info!("[test] group-move: {} regiments as one formation", selected.len());
            *stage = 3;
        }
        3 if !arrangement.is_empty()
            && arrangement.iter().all(|(g, _)| groups.list[*g].order.is_none()) =>
        {
            let centroid: Vec2 = arrangement
                .iter()
                .map(|(g, _)| groups.list[*g].centroid)
                .sum::<Vec2>()
                / arrangement.len() as f32;
            let err: f32 = arrangement
                .iter()
                .map(|(g, off)| (groups.list[*g].centroid - centroid).distance(*off))
                .sum::<f32>()
                / arrangement.len() as f32;
            info!("[test] group-move arrived: mean arrangement error {err:.1} m");
            *stage = 4;
        }
        _ => {}
    }
}
