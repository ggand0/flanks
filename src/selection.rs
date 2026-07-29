//! Regiment selection: the LMB lasso stroke (line or enclosing loop),
//! hover picking, and control groups. Split out of orders.rs (which keeps
//! order state + issuing) — the 0029 refactor note, forced by the drag-
//! order machinery landing there.

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::movement::DebugViz;
use crate::orders::{Groups, PLAYER_TEAM};
use crate::terrain::Terrain;
use crate::units::Units;

const SELECT_RADIUS: f32 = 14.0;

/// Selected regiments (player team only) + cached unit total.
#[derive(Resource, Default)]
pub struct Selection {
    pub regiments: Vec<bool>,
    pub count_units: usize,
}

impl Selection {
    /// Recompute `count_units` from the mask (player team only). Every
    /// mask mutation site calls this instead of maintaining the
    /// invariant by hand.
    pub fn recount(&mut self, groups: &Groups) {
        self.count_units = self
            .regiments
            .iter()
            .enumerate()
            .filter(|(g, s)| **s && groups.list[*g].team == PLAYER_TEAM)
            .map(|(g, _)| groups.list[g].count)
            .sum();
    }

    /// Selected regiments that are alive and controllable — the set
    /// every selection command acts on (and the HUD previews).
    pub fn picked_controllable<'a>(
        &'a self,
        groups: &'a Groups,
    ) -> impl Iterator<Item = usize> + 'a {
        self.regiments.iter().enumerate().filter_map(move |(g, s)| {
            (*s && {
                let gd = &groups.list[g];
                gd.count > 0 && !gd.state.is_broken()
            })
            .then_some(g)
        })
    }
}

/// In-progress drag: projected ground points of the selection line.
#[derive(Resource, Default)]
pub struct DragLine {
    points: Vec<Vec2>,
    active: bool,
}

/// Ctrl+1..9 stores the current selection mask; 1..9 recalls it.
#[derive(Resource, Default)]
pub struct ControlGroups(pub Vec<Vec<bool>>);

/// Regiments under the cursor this frame. `enemy` doubles as the
/// attack-order preview: it is what a right-click would target (same
/// pick logic), so the render tint and the inspect panel key off it.
#[derive(Resource, Default)]
pub struct Hover {
    pub enemy: Option<u32>,
    pub own: Option<u32>,
}

pub struct SelectionPlugin;

impl Plugin for SelectionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Selection>()
            .init_resource::<DragLine>()
            .init_resource::<Hover>()
            .init_resource::<ControlGroups>()
            .add_systems(
                Update,
                (
                    (drag_select, update_hover, control_group_keys)
                        .in_set(crate::game_state::MapInputSet),
                    draw_selection_gizmos,
                ),
            );
    }
}

pub fn cursor_ground_point(
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
    mut cues: MessageWriter<crate::audio::UiCue>,
    ui: Query<&Interaction>,
) {
    let Ok(window) = window.single() else { return };
    let Ok((camera, cam_tf)) = camera.single() else {
        return;
    };

    // A stroke must start on the map: clicking a unit card or a control
    // button never doubles as a (selection-clearing) lasso on the
    // terrain behind the HUD.
    if buttons.just_pressed(MouseButton::Left)
        && !crate::unit_cards::pointer_over_ui(ui.iter())
    {
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
        if selection.count_units > 0 {
            cues.write(crate::audio::UiCue::Select);
        }
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

/// Regiment under a ground point, if any: nearest live regiment of the
/// wanted side whose centroid is within the pick radius.
pub fn regiment_at(groups: &Groups, p: Vec2, enemy: bool) -> Option<u32> {
    groups
        .list
        .iter()
        .enumerate()
        .filter(|(_, g)| (g.team != PLAYER_TEAM) == enemy && g.count > 0)
        .map(|(g, gd)| (g, gd.centroid.distance(p)))
        .filter(|(_, d)| *d < crate::orders::ATTACK_PICK_RADIUS)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(g, _)| g as u32)
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

/// Ctrl+digit assigns the current selection to a slot; a bare digit
/// recalls it (dead regiments drop out naturally via count checks).
fn control_group_keys(
    keys: Res<ButtonInput<KeyCode>>,
    groups: Res<Groups>,
    mut cg: ResMut<ControlGroups>,
    mut selection: ResMut<Selection>,
    mut cues: MessageWriter<crate::audio::UiCue>,
) {
    const DIGITS: [KeyCode; 9] = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
        KeyCode::Digit9,
    ];
    let Some(slot) = DIGITS.iter().position(|k| keys.just_pressed(*k)) else {
        return;
    };
    cg.0.resize(9, Vec::new());
    let ctrl =
        keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    if ctrl {
        if selection.count_units > 0 {
            cg.0[slot] = selection.regiments.clone();
            info!("control group {} assigned", slot + 1);
        }
    } else if !cg.0[slot].is_empty() {
        selection.regiments = cg.0[slot].clone();
        selection.regiments.resize(groups.list.len(), false);
        selection.recount(&groups);
        if selection.count_units > 0 {
            cues.write(crate::audio::UiCue::Select);
        }
        info!(
            "control group {} recalled ({} units)",
            slot + 1,
            selection.count_units
        );
    }
}

/// Selection stroke feedback (debug-viz gated like the other order gizmos).
fn draw_selection_gizmos(
    viz: Res<DebugViz>,
    drag: Res<DragLine>,
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
}
