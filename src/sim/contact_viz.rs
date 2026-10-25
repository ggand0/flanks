//! The contact overlay: what the contact rules see, drawn on the ground.
//! Under the full debug overlay (F3) or FL_CONTACT_VIZ=1, with the
//! gizmos on (G), for every regiment that is selected, open or passing a
//! friend: each man's body distance and spring rest at his present pose
//! as two circles, a tick from him to his slot with the hold deadzone
//! round the slot of a holding man, his pace, and the lane look's verdict
//! (a man who saw a friend's man in his way); each regiment's block as it
//! stands, its closed ground when open, its anchor with its facing, and
//! the lanes of its open order. The selection rings hide while it shows
//! (selection_rings.rs). The numbers behind the circles are on the
//! overlay text (overlay.rs `contact_text`).

use bevy::prelude::*;

use crate::orders::{Groups, Order, Selection};
use crate::terrain::Terrain;
use crate::units::Units;

use super::soldier::{pose_level, widths, HOLD_DEADZONE, SIGHT_LEFT, SIGHT_PASS};

/// Whether the contact overlay shows: the full debug overlay, or
/// FL_CONTACT_VIZ=1 for a scene run without input.
pub fn contact_viz_on(settings: &crate::settings::Settings) -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    settings.interface.debug_overlay()
        || *ENV.get_or_init(|| std::env::var("FL_CONTACT_VIZ").is_ok())
}

/// The regiments the overlay draws: selected, open, or passing a friend.
pub fn shown(selection: &Selection, groups: &Groups) -> Vec<bool> {
    groups
        .list
        .iter()
        .enumerate()
        .map(|(g, gd)| {
            gd.count > 0
                && (selection.regiments.get(g).copied().unwrap_or(false)
                    || gd.spacing == crate::formation::FormSpacing::Open
                    || gd.passing)
        })
        .collect()
}

/// The pose colour: green at ease, orange in the ready stance.
fn pose_colour(level: usize, alpha: f32) -> Color {
    let t = level as f32 / 3.0;
    Color::srgba(0.3 + 0.7 * t, 0.9 - 0.45 * t, 0.3 - 0.2 * t, alpha)
}

pub fn draw_contact_viz(
    settings: Res<crate::settings::Settings>,
    viz: Res<super::DebugViz>,
    selection: Res<Selection>,
    groups: Res<Groups>,
    units: Res<Units>,
    terrain: Res<Terrain>,
    mut gizmos: Gizmos,
) {
    if !viz.0 || !contact_viz_on(&settings) {
        return;
    }
    let shown = shown(&selection, &groups);
    if !shown.iter().any(|&s| s) {
        return;
    }
    let flat = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    let lift = |p: Vec2, up: f32| Vec3::new(p.x, terrain.height_at(p.x, p.y) + up, p.y);
    const SPRING: Color = Color::srgba(0.45, 0.65, 1.0, 0.55);
    const SLOT: Color = Color::srgb(1.0, 0.9, 0.2);
    const DEADZONE: Color = Color::srgba(1.0, 0.9, 0.2, 0.35);
    const PACE: Color = Color::srgb(1.0, 1.0, 1.0);
    const LANE_LOOK: Color = Color::srgb(1.0, 0.2, 0.9);
    let w = widths();
    for i in 0..units.len() {
        let g = units.group[i] as usize;
        if !shown.get(g).copied().unwrap_or(false) || units.death_t[i] != 0 {
            continue;
        }
        let gd = &groups.list[g];
        let p = units.pos[i];
        let p2 = Vec2::new(p.x, p.z);
        let at = lift(p2, 0.12);
        let level = pose_level(units.pose[i]);
        let width = &w[level];
        gizmos.circle(Isometry3d::new(at, flat), 0.5 * width.body, pose_colour(level, 1.0));
        gizmos.circle(Isometry3d::new(at, flat), 0.5 * width.sep, SPRING).resolution(16);
        // His slot: where his regiment holds, or where its Move order
        // takes it. A tick to it when near, a 1 m pointer when far.
        let frame = match gd.order {
            Some(Order::Move(goal)) => goal,
            _ => gd.anchor,
        };
        let slot = frame + units.home[i];
        let to_slot = slot - p2;
        let dist = to_slot.length();
        if dist > 0.05 {
            let end = if dist <= 4.0 { slot } else { p2 + to_slot / dist };
            gizmos.line(at, lift(end, 0.12), SLOT);
        }
        if gd.order.is_none() && dist <= 4.0 {
            gizmos
                .circle(Isometry3d::new(lift(slot, 0.1), flat), HOLD_DEADZONE, DEADZONE)
                .resolution(10);
        }
        // His pace: a third of a second of his velocity.
        let v = units.vel[i];
        let v2 = Vec2::new(v.x, v.z);
        if v2.length_squared() > 0.01 {
            gizmos.line(lift(p2, 0.4), lift(p2 + v2 * 0.33, 0.4), PACE);
        }
        // The lane look: he saw a friend's man in his way and steps to the
        // side the man is not on.
        let sight = units.sight[i];
        if sight & SIGHT_PASS != 0 {
            gizmos.circle(Isometry3d::new(lift(p2, 0.5), flat), 0.2, LANE_LOOK).resolution(8);
            let dir = v2.normalize_or_zero();
            if dir != Vec2::ZERO {
                let left = Vec2::new(-dir.y, dir.x);
                let side = if sight & SIGHT_LEFT != 0 { -left } else { left };
                gizmos.line(lift(p2, 0.5), lift(p2 + side * 0.8, 0.5), LANE_LOOK);
            }
        }
    }
    const BLOCK: Color = Color::srgb(0.2, 0.9, 1.0);
    const CLOSED: Color = Color::srgba(0.2, 0.9, 1.0, 0.4);
    const LANE: Color = Color::srgb(0.4, 1.0, 0.4);
    const ANCHOR: Color = Color::srgb(1.0, 0.6, 0.1);
    let rect_lines = |gizmos: &mut Gizmos, r: &crate::formation::BlockRect, up: f32, col: Color| {
        let right = Vec2::new(r.fwd.y, -r.fwd.x);
        let c = [
            r.centre + r.fwd * r.half_d + right * r.half_w,
            r.centre + r.fwd * r.half_d - right * r.half_w,
            r.centre - r.fwd * r.half_d - right * r.half_w,
            r.centre - r.fwd * r.half_d + right * r.half_w,
        ];
        for k in 0..4 {
            gizmos.line(lift(c[k], up), lift(c[(k + 1) % 4], up), col);
        }
    };
    for (g, gd) in groups.list.iter().enumerate() {
        if !shown[g] {
            continue;
        }
        let rect = crate::formation::block_rect(gd);
        rect_lines(&mut gizmos, &rect, 0.3, BLOCK);
        let fwd = crate::formation::facing_dir(gd.facing);
        let right = Vec2::new(fwd.y, -fwd.x);
        gizmos.circle(Isometry3d::new(lift(gd.anchor, 0.3), flat), 0.6, ANCHOR);
        gizmos.arrow(lift(gd.anchor, 0.6), lift(gd.anchor + fwd * 3.0, 0.6), ANCHOR);
        if gd.spacing != crate::formation::FormSpacing::Open || gd.files == 0 {
            continue;
        }
        let closed_centre = gd.centroid - fwd * gd.open.shift();
        let closed = crate::formation::closed_rect_at(gd, closed_centre);
        rect_lines(&mut gizmos, &closed, 0.2, CLOSED);
        // The lanes: between the last file of a group and the first of the
        // next, down the block's depth about the anchor.
        let n = gd.count.max(1);
        let files = (gd.files.max(1) as usize).min(n);
        let a = gd.open.files.max(1) as usize;
        let groups_n = crate::formation::open_groups(files, n, gd.close_to, gd.kind, &gd.open);
        let x = crate::formation::open_file_x(groups_n, &gd.open);
        let half_d = crate::formation::front_offset(gd) + 1.0;
        for k in 0..groups_n.saturating_sub(1) {
            let (last, first) = ((k + 1) * a - 1, (k + 1) * a);
            if first >= x.len() {
                break;
            }
            let lane_x = 0.5 * (x[last] + x[first]);
            let mid = gd.anchor + right * lane_x;
            gizmos.line(lift(mid - fwd * half_d, 0.25), lift(mid + fwd * half_d, 0.25), LANE);
        }
    }
}
