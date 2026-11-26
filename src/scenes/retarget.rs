//! FL_TEST_RETARGET: a new attack target mid-fight, the scene and the
//! script that gives the order.

use bevy::prelude::*;

use crate::game_state::{BattleClock, PerBattle};
use crate::orders::{Groups, Order, PLAYER_TEAM};
use crate::regiments::spawn_regiment;
use crate::regiments::{EDGE_MARGIN, SIDE_MARGIN};
use crate::terrain::Terrain;
use crate::unit_types::KIND_LIGHT;
use crate::units::Units;
use super::SPACING;

/// FL_TEST_RETARGET=1: a new attack target mid-fight. Blue R attacks
/// orange A head-on; orange B stands under no order one block width
/// plus 12 m to the side of A (FL_RT_B_DIST, 60 for the 500-man blocks:
/// the near flanks are 12 m apart, adjacent regiments in a line). At
/// 20 s (FL_RETARGET_AT) the retarget hook (`test_retarget_script`
/// below) orders R onto B, as a player would, and logs
/// every 5 s whom R's men fight. A attacks R at spawn, a player order
/// that persists, so A follows R when R leaves: the pursuit case.
/// FL_RT_A_HOLD=1 keeps A in hold instead: the standing case. FL_RT_N
/// lays that many triplets on a grid, each in its own cell, and
/// FL_RT_SIZE sets the regiment size (500): FL_RT_N=48 FL_RT_SIZE=1000
/// is the cost case, 144k men and every third regiment retargeted at
/// once, as many as the 1024 by 768 m field holds.
/// Acceptance: R's block marches on B and fights it; its men answer A's
/// men who come at them and walk back to none who do not.
pub(crate) fn spawn_retarget_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let size: usize = crate::util::env_or("FL_RT_SIZE", 500);
    let n: usize = crate::util::env_or("FL_RT_N", 1).max(1);
    let cols = ((size as f32 * 2.2).sqrt().ceil() as usize).max(1);
    let block_w = cols as f32 * SPACING;
    let block_d = size.div_ceil(cols) as f32 * SPACING;
    let b_dist: f32 = crate::util::env_or("FL_RT_B_DIST", (block_w + 12.0).round());
    let a_hold = std::env::var("FL_RT_A_HOLD").is_ok_and(|v| v == "1");
    // The triplets on a grid: each cell holds R below A and B side by
    // side, with 10 m of clear ground to the next cell (the regiment gap
    // of a normal battle), as many per row as the field is wide.
    let cell_w = b_dist + block_w + 10.0;
    let cell_d = 50.0 + block_d + 10.0;
    let usable_w = (terrain.max().x - terrain.min().x) - 2.0 * SIDE_MARGIN;
    let per_row = ((usable_w / cell_w).floor() as usize).clamp(1, n);
    let n_rows = n.div_ceil(per_row);
    if n_rows as f32 * cell_d > terrain.max().y - terrain.min().y - 2.0 * EDGE_MARGIN {
        warn!("retarget layout: {n_rows} rows of {cell_d:.0} m do not fit the field");
    }
    let x0 = -(per_row as f32 - 1.0) * cell_w / 2.0;
    let z0 = -(n_rows as f32 - 1.0) * cell_d / 2.0;
    let mut list = Vec::new();
    for k in 0..n {
        let o = Vec2::new(x0 + (k % per_row) as f32 * cell_w, z0 + (k / per_row) as f32 * cell_d);
        let r = list.len();
        spawn_regiment(units, terrain, &mut list, 0, KIND_LIGHT, o + Vec2::new(0.0, -40.0), size, -1.0);
        list[r].order = Some(crate::orders::Order::Attack(r as u32 + 1));
        spawn_regiment(units, terrain, &mut list, 1, KIND_LIGHT, o + Vec2::new(0.0, 10.0), size, 1.0);
        if a_hold {
            list[r + 1].hold = true;
        } else {
            list[r + 1].order = Some(crate::orders::Order::Attack(r as u32));
        }
        spawn_regiment(units, terrain, &mut list, 1, KIND_LIGHT, o + Vec2::new(b_dist, 10.0), size, 1.0);
    }
    groups.list = list;
    info!(
        "[retarget-test] {n} triplet(s) of {size}, {per_row} per row: R attacks A; B stands {b_dist:.0} m to the side; A {}",
        if a_hold { "holds" } else { "attacks R" }
    );
}

/// When the retarget test fired (the tick), and each retargeted regiment
/// with its old and new target.
type RetargetTest = Option<(u32, Vec<(usize, usize, usize)>)>;

/// The mid-fight retarget test. In the Retarget scenario (FL_TEST_RETARGET,
/// `spawn_retarget_test` above), or in any battle with
/// FL_RETARGET_AT set: that many seconds in (20 in the scenario), every
/// engaged player regiment is ordered onto the nearest other unbroken
/// enemy regiment, as a player retargeting mid-fight would. Every 5 s
/// after, each one's men are counted: striking a man of the old target,
/// of the new one, and still within 2.5 m of an old-target man, with the
/// regiment's distance to both. It runs on the sim tick and times itself
/// by the battle clock, so two runs of one binary retarget on the same tick.
pub(super) fn test_retarget_script(
    units: Res<Units>,
    clock: Res<BattleClock>,
    mut groups: ResMut<Groups>,
    mut state: Local<PerBattle<RetargetTest>>,
) {
    let scenario = std::env::var("FL_TEST_RETARGET").is_ok();
    let at_s = std::env::var("FL_RETARGET_AT").ok().and_then(|v| v.parse::<f32>().ok());
    let Some(at) = at_s.or(scenario.then_some(20.0)) else {
        return;
    };
    let state = state.of(&clock);
    let tick = clock.tick;
    let secs = clock.secs();
    if state.is_none() {
        if secs < at {
            return;
        }
        let mut picks = Vec::new();
        for g in 0..groups.list.len() {
            let gd = &groups.list[g];
            if gd.team != PLAYER_TEAM || gd.count == 0 || !gd.engaged || gd.state.is_broken() {
                continue;
            }
            // The enemy it fights: its target, else the nearest enemy.
            let near = |skip: Option<usize>| {
                (0..groups.list.len())
                    .filter(|&e| {
                        let eg = &groups.list[e];
                        eg.team != gd.team && eg.count > 0 && !eg.state.is_broken() && Some(e) != skip
                    })
                    .min_by(|&a, &b| {
                        let da = groups.list[a].centroid.distance_squared(gd.centroid);
                        let db = groups.list[b].centroid.distance_squared(gd.centroid);
                        da.total_cmp(&db)
                    })
            };
            let old = match gd.order {
                Some(Order::Attack(t)) => Some(t as usize),
                _ => near(None),
            };
            let Some(old) = old else { continue };
            let Some(new) = near(Some(old)) else { continue };
            picks.push((g, old, new));
        }
        for &(g, _, new) in &picks {
            groups.list[g].order = Some(Order::Attack(new as u32));
            groups.list[g].auto_order = false;
        }
        info!(
            "[retarget-test] tick {tick} ({secs:.1} s): {} engaged regiments retargeted",
            picks.len()
        );
        *state = Some((tick, picks));
        return;
    }
    let Some((last, picks)) = state.as_mut() else { return };
    if tick < *last + 150 {
        return;
    }
    *last = tick;
    // Who each man fights: the man he winds up at, by regiment, and
    // whether an old-target man stands within 2.5 m of him.
    let mut near_old = vec![false; units.len()];
    for &(g, old, _) in picks.iter() {
        let olds: Vec<Vec2> = (0..units.len())
            .filter(|&j| units.group[j] as usize == old && units.death_t[j] == 0)
            .map(|j| Vec2::new(units.pos[j].x, units.pos[j].z))
            .collect();
        for (i, near) in near_old.iter_mut().enumerate() {
            if units.group[i] as usize == g && units.death_t[i] == 0 {
                let p = Vec2::new(units.pos[i].x, units.pos[i].z);
                *near = olds.iter().any(|q| q.distance_squared(p) < 2.5 * 2.5);
            }
        }
    }
    for &(g, old, new) in picks.iter().take(8) {
        let (mut alive, mut at_old, mut at_new, mut touching_old) = (0, 0, 0, 0);
        for (i, &near) in near_old.iter().enumerate() {
            if units.group[i] as usize != g || units.death_t[i] != 0 {
                continue;
            }
            alive += 1;
            touching_old += near as usize;
            if units.swing[i] & crate::units::SWING_STATE_MASK == crate::units::SWING_WINDUP {
                let tg = units.target[i] as usize;
                if tg < units.len() {
                    at_old += (units.group[tg] as usize == old) as usize;
                    at_new += (units.group[tg] as usize == new) as usize;
                }
            }
        }
        let gd = &groups.list[g];
        info!(
            "[retarget-test] t={secs:.0}s reg {g} ({alive} men): striking old {old} {at_old}, new {new} {at_new}; touching old {touching_old}; to old {:.0} m, to new {:.0} m; retarget {} melee {}; old alive {} new alive {}",
            gd.centroid.distance(groups.list[old].centroid),
            gd.centroid.distance(groups.list[new].centroid),
            gd.retarget,
            gd.melee_ticks,
            groups.list[old].count,
            groups.list[new].count
        );
    }
}
