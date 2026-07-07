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

pub struct GroupData {
    pub team: u8,
    /// Attack-move target on the ground plane; None = hold.
    pub order: Option<Vec2>,
    pub count: usize,
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
                GroupData {
                    team: 0,
                    order: None,
                    count: crate::units::UNITS_PER_TEAM,
                },
                GroupData {
                    team: 1,
                    order: None,
                    count: crate::units::UNITS_PER_TEAM,
                },
            ],
        })
        .init_resource::<Selection>()
        .init_resource::<DragLine>()
        .add_systems(
            Update,
            (drag_select, issue_order, cut_key, test_orders_script, draw_order_gizmos),
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

fn select_along_line(line: &[Vec2], units: &Units, selection: &mut Selection) {
    selection.mask.clear();
    selection.mask.resize(units.len(), false);
    selection.count = 0;

    // Bounding box early-out around the whole line.
    let mut lo = Vec2::splat(f32::MAX);
    let mut hi = Vec2::splat(f32::MIN);
    for p in line {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    lo -= SELECT_RADIUS;
    hi += SELECT_RADIUS;

    let r2 = SELECT_RADIUS * SELECT_RADIUS;
    for i in 0..units.len() {
        if units.team[i] != PLAYER_TEAM {
            continue;
        }
        let p = Vec2::new(units.pos[i].x, units.pos[i].z);
        if p.x < lo.x || p.x > hi.x || p.y < lo.y || p.y > hi.y {
            continue;
        }
        let mut best = f32::MAX;
        if line.len() == 1 {
            best = p.distance_squared(line[0]);
        } else {
            for w in line.windows(2) {
                let (a, b) = (w[0], w[1]);
                let ab = b - a;
                let t = ((p - a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
                best = best.min(p.distance_squared(a + ab * t));
                if best < r2 {
                    break;
                }
            }
        }
        if best < r2 {
            selection.mask[i] = true;
            selection.count += 1;
        }
    }
}

/// Move the selection's units into a fresh group. Returns its index, or the
/// existing group index if the selection is exactly one whole group.
fn split_selection(units: &mut Units, groups: &mut Groups, selection: &Selection) -> Option<u32> {
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
    groups.list.push(GroupData {
        team: PLAYER_TEAM,
        order: None,
        count: selection.count,
    });
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

/// Groups whose centroid reached their target go back to hold.
pub fn clear_arrived_orders(units: Res<Units>, mut groups: ResMut<Groups>) {
    let n_groups = groups.list.len();
    let mut sums = vec![Vec2::ZERO; n_groups];
    let mut counts = vec![0u32; n_groups];
    for i in 0..units.len() {
        let g = units.group[i] as usize;
        sums[g] += Vec2::new(units.pos[i].x, units.pos[i].z);
        counts[g] += 1;
    }
    for (g, group) in groups.list.iter_mut().enumerate() {
        if let Some(t) = group.order
            && counts[g] > 0
            && (sums[g] / counts[g] as f32).distance(t) < ARRIVE_CLEAR
        {
            group.order = None;
            info!("group {g} arrived, holding");
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
    // Selection line being drawn.
    if drag.active && drag.points.len() > 1 {
        let pts: Vec<Vec3> = drag
            .points
            .iter()
            .map(|p| Vec3::new(p.x, terrain.height_at(p.x, p.y) + 1.0, p.y))
            .collect();
        gizmos.linestrip(pts, Color::srgb(0.95, 0.95, 0.4));
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
            // Disc selection inside the blue army.
            selection.mask.clear();
            selection.mask.resize(units.len(), false);
            selection.count = 0;
            let center = Vec2::new(-150.0, -90.0);
            for i in 0..units.len() {
                if units.team[i] == PLAYER_TEAM
                    && Vec2::new(units.pos[i].x, units.pos[i].z).distance(center) < 45.0
                {
                    selection.mask[i] = true;
                    selection.count += 1;
                }
            }
            info!("[test] selected {} units", selection.count);
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
