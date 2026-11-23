//! FL_TEST_JOIN: a regiment ordered into a fight from behind a friendly
//! line, and its log.

use bevy::prelude::*;

use crate::orders::Groups;
use crate::regiments::spawn_regiment;
use crate::terrain::Terrain;
use crate::unit_types::KIND_LIGHT;
use crate::units::Units;

/// FL_TEST_JOIN=1: the join-the-fight order (regression repro). Orange
/// attacks blue A's line head-on; blue B stands BEHIND A, ordered onto
/// the same orange regiment at spawn. Acceptance: B's march never
/// halts for the overflow trickle that pokes it, B routes AROUND A's
/// ranks (friendly blocks are solid to a march), reaches the orange
/// mass, and its fighter count ramps into the dozens.
pub(crate) fn spawn_join_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let mut list = Vec::new();
    spawn_regiment(units, terrain, &mut list, 1, KIND_LIGHT, Vec2::new(0.0, 50.0), 500, 1.0);
    list[0].order = Some(crate::orders::Order::Attack(1));
    list[0].auto_order = true;
    // A and B carry PLAYER-issued orders (auto_order stays false): the
    // at-ease AI stands auto-engaged regiments down when their target
    // breaks — correct doctrine, but this repro is about
    // player-ordered attacks, which persist through the rout.
    spawn_regiment(units, terrain, &mut list, 0, KIND_LIGHT, Vec2::new(0.0, 10.0), 500, -1.0);
    list[1].order = Some(crate::orders::Order::Attack(0));
    spawn_regiment(units, terrain, &mut list, 0, KIND_LIGHT, Vec2::new(0.0, -30.0), 500, -1.0);
    list[2].order = Some(crate::orders::Order::Attack(0));
    groups.list = list;
    info!("[join-test] orange attacks A's line; B behind A ordered onto the same orange");
}

/// FL_TEST_JOIN bookkeeping every 4 s: does B actually get into the
/// ordered fight instead of stalling behind A on overflow duels?
pub(super) fn join_test_log(
    units: Res<Units>,
    groups: Res<crate::orders::Groups>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    if std::env::var("FL_TEST_JOIN").is_err() || groups.list.len() < 3 {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 4.0;
    let mut b_windups = 0;
    for i in 0..units.len() {
        if units.group[i] == 2
            && units.death_t[i] == 0
            && units.swing[i] & crate::units::SWING_STATE_MASK == crate::units::SWING_WINDUP
        {
            b_windups += 1;
        }
    }
    info!(
        "[join-test] t={t:.0}s orange {} / A {} / B {} alive; B->orange {:.1} m, B windups {b_windups}",
        groups.list[0].count,
        groups.list[1].count,
        groups.list[2].count,
        groups.list[2].centroid.distance(groups.list[0].centroid),
    );
}
