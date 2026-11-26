//! FL_TEST_ORDERS: an enclosure select, a long march and a group move,
//! scripted.

use bevy::prelude::*;

use crate::game_state::{BattleClock, PerBattle};
use crate::orders::{Groups, Order, PLAYER_TEAM, Selection, order_regiments, select_along_line};
use crate::units::Units;

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

/// The orders test's stage, the long-marched regiment with its spread at
/// departure, and the group move's arrangement.
type OrdersTest = (u32, Option<(u32, f32)>, Vec<(usize, Vec2)>);

/// FL_TEST_ORDERS=1: regiment cohesion + arrangement-preserving group
/// moves. Stage 0: enclosure-select regiments (exercises the promote
/// path), long-march ONE regiment 350 m and compare its RMS spread on
/// arrival (acceptance: < 1.5x). Stage 1: group-move several regiments
/// and log how well their relative arrangement survived.
pub(super) fn test_orders_script(
    clock: Res<BattleClock>,
    units: Res<Units>,
    mut groups: ResMut<Groups>,
    mut selection: ResMut<Selection>,
    mut state: Local<PerBattle<OrdersTest>>,
) {
    if std::env::var("FL_TEST_ORDERS").is_err() {
        return;
    }
    let (stage, watch, arrangement) = state.of(&clock);
    let t = clock.secs();
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
            select_along_line(&stroke, false, &units, &groups, &mut selection);
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
