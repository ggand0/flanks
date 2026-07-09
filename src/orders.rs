//! Groups, selection, and player orders.
//!
//! Selection: hold LMB and draw a line over the field; friendly (blue) units
//! within SELECT_RADIUS of the projected polyline become the selection.
//! Right-click: attack-move the selection (a strict subset of a group is
//! split off into its own group for now — M5 replaces this with attached
//! salients). C: cut the selection into a new group without ordering it.

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::movement::DebugViz;
use crate::terrain::Terrain;
use crate::units::Units;

pub const PLAYER_TEAM: u8 = 0;
const SELECT_RADIUS: f32 = 14.0;
/// Groups whose centroid gets this close to their order target go idle.
const ARRIVE_CLEAR: f32 = 18.0;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stance {
    /// Wide and shallow.
    Hold,
    /// Narrow and deep.
    Column,
}

pub struct GroupData {
    pub team: u8,
    /// Attack-move target on the ground plane; None = hold.
    pub order: Option<Vec2>,
    pub count: usize,
    /// Currently inert; formation shapes return with the TW-style branch.
    pub stance: Stance,
    // --- refreshed every fixed tick by the frontline pass ---
    pub centroid: Vec2,
    pub engaged: bool,
}

impl GroupData {
    pub fn new(team: u8, count: usize) -> Self {
        Self {
            team,
            order: None,
            count,
            stance: Stance::Hold,
            centroid: Vec2::ZERO,
            engaged: false,
        }
    }
}

#[derive(Resource, Default)]
pub struct Groups {
    pub list: Vec<GroupData>,
}

/// Per-unit selection mask (player team only) + cached count.
#[derive(Resource, Default)]
pub struct Selection {
    pub mask: Vec<bool>,
    pub count: usize,
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
        app.insert_resource(Groups {
            list: vec![
                GroupData::new(0, crate::units::units_per_team()),
                GroupData::new(1, crate::units::units_per_team()),
            ],
        })
        .init_resource::<Selection>()
        .init_resource::<DragLine>()
        .add_systems(
            Update,
            (
                drag_select,
                issue_order,
                cut_key,
                stance_key,
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

fn drag_select(
    buttons: Res<ButtonInput<MouseButton>>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform)>,
    terrain: Res<Terrain>,
    units: Res<Units>,
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
        select_along_line(&drag.points, &units, &mut selection);
        info!("selected {} units", selection.count);
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

fn select_along_line(line: &[Vec2], units: &Units, selection: &mut Selection) {
    selection.mask.clear();
    selection.mask.resize(units.len(), false);
    selection.count = 0;

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
        if units.team[i] != PLAYER_TEAM {
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
            selection.mask[i] = true;
            selection.count += 1;
        }
    }
}

/// Move the selection's units into a fresh group. Returns its index, or the
/// existing group index if the selection is exactly one whole group.
pub fn split_selection(
    units: &mut Units,
    groups: &mut Groups,
    selection: &Selection,
) -> Option<u32> {
    if selection.count == 0 {
        return None;
    }
    // Selection == entire single group? Then no split needed.
    let first_group = units
        .group
        .iter()
        .zip(&selection.mask)
        .find(|(_, s)| **s)
        .map(|(g, _)| *g)?;
    let whole = selection.count == groups.list[first_group as usize].count
        && units
            .group
            .iter()
            .zip(&selection.mask)
            .all(|(g, s)| !*s || *g == first_group);
    if whole {
        return Some(first_group);
    }

    let new_idx = groups.list.len() as u32;
    groups.list.push(GroupData::new(PLAYER_TEAM, selection.count));
    for i in 0..units.len() {
        if selection.mask[i] {
            groups.list[units.group[i] as usize].count -= 1;
            units.group[i] = new_idx;
        }
    }
    info!("split {} units into group {}", selection.count, new_idx);
    Some(new_idx)
}

fn issue_order(
    buttons: Res<ButtonInput<MouseButton>>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform)>,
    terrain: Res<Terrain>,
    mut units: ResMut<Units>,
    mut groups: ResMut<Groups>,
    selection: Res<Selection>,
) {
    if !buttons.just_pressed(MouseButton::Right) || selection.count == 0 {
        return;
    }
    let Ok(window) = window.single() else { return };
    let Ok((camera, cam_tf)) = camera.single() else {
        return;
    };
    let Some(target) = cursor_ground_point(window, camera, cam_tf, &terrain) else {
        return;
    };
    if let Some(g) = split_selection(&mut units, &mut groups, &selection) {
        groups.list[g as usize].order = Some(target);
        info!(
            "group {g} ({} units) attack-move to ({:.0}, {:.0})",
            groups.list[g as usize].count, target.x, target.y
        );
    }
}

fn cut_key(
    keys: Res<ButtonInput<KeyCode>>,
    mut units: ResMut<Units>,
    mut groups: ResMut<Groups>,
    selection: Res<Selection>,
) {
    if keys.just_pressed(KeyCode::KeyC) {
        split_selection(&mut units, &mut groups, &selection);
    }
}

/// Groups whose centroid reached their target go back to hold. Uses the
/// centroid refreshed by the frontline pass each tick.
pub fn clear_arrived_orders(mut groups: ResMut<Groups>) {
    for (g, group) in groups.list.iter_mut().enumerate() {
        if let Some(t) = group.order
            && group.count > 0
            && !group.engaged
            && group.centroid.distance(t) < ARRIVE_CLEAR
        {
            group.order = None;
            info!("group {g} arrived, holding");
        }
    }
}

/// F: toggle stance of every group that has selected units.
fn stance_key(
    keys: Res<ButtonInput<KeyCode>>,
    units: Res<Units>,
    mut groups: ResMut<Groups>,
    selection: Res<Selection>,
) {
    if !keys.just_pressed(KeyCode::KeyF) || selection.count == 0 {
        return;
    }
    let mut touched = vec![false; groups.list.len()];
    for i in 0..units.len() {
        if selection.mask.get(i).copied().unwrap_or(false) {
            touched[units.group[i] as usize] = true;
        }
    }
    for (g, t) in touched.iter().enumerate() {
        if *t {
            let group = &mut groups.list[g];
            group.stance = match group.stance {
                Stance::Hold => Stance::Column,
                Stance::Column => Stance::Hold,
            };
            info!(
                "group {g} stance -> {}",
                if group.stance == Stance::Hold { "hold line" } else { "column" }
            );
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
    // Order targets.
    for group in &groups.list {
        if let Some(t) = group.order {
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

/// FL_TEST_ORDERS=1: scripted carve-out — select a disc of blue units, cut,
/// send them across the map; order the rest of the army elsewhere later.
fn test_orders_script(
    time: Res<Time>,
    mut units: ResMut<Units>,
    mut groups: ResMut<Groups>,
    mut selection: ResMut<Selection>,
    mut stage: Local<u32>,
) {
    if std::env::var("FL_TEST_ORDERS").is_err() {
        return;
    }
    let t = time.elapsed_secs();
    match *stage {
        0 if t > 5.0 => {
            // Enclosure selection: a circular stroke inside the blue army
            // (exercises the closed-loop polygon path end to end).
            let center = Vec2::new(-150.0, -90.0);
            let stroke: Vec<Vec2> = (0..28)
                .map(|k| {
                    let a = k as f32 / 28.0 * std::f32::consts::TAU;
                    center + Vec2::new(a.cos(), a.sin()) * 45.0
                })
                .collect();
            select_along_line(&stroke, &units, &mut selection);
            let brute = (0..units.len())
                .filter(|&i| {
                    units.team[i] == PLAYER_TEAM
                        && Vec2::new(units.pos[i].x, units.pos[i].z).distance(center) < 45.0
                })
                .count();
            info!(
                "[test] enclosure selected {} units (>= {brute} strictly inside)",
                selection.count
            );
            if let Some(g) = split_selection(&mut units, &mut groups, &selection) {
                groups.list[g as usize].order = Some(Vec2::new(220.0, 160.0));
                info!("[test] cut group {g}, ordered across the map");
            }
            *stage = 1;
        }
        1 if t > 20.0 => {
            // Send the rest of the blue army west, staying on its own side.
            selection.mask.clear();
            selection.mask.resize(units.len(), false);
            selection.count = 0;
            for i in 0..units.len() {
                if units.group[i] == 0 {
                    selection.mask[i] = true;
                    selection.count += 1;
                }
            }
            if let Some(g) = split_selection(&mut units, &mut groups, &selection) {
                groups.list[g as usize].order = Some(Vec2::new(-330.0, -120.0));
                info!("[test] main army (group {g}) ordered to the west hills");
            }
            // Deselect so screenshots show team colors again.
            selection.mask.fill(false);
            selection.count = 0;
            *stage = 2;
        }
        _ => {}
    }
}
