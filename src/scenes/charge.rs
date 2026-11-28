//! FL_TEST_CHARGE: heavies charging spears in a wall and in normal
//! order, and its log.

use bevy::prelude::*;

use crate::game_state::{BattleClock, PerBattle};
use crate::orders::Groups;
use crate::regiments::spawn_regiment;
use crate::terrain::Terrain;
use crate::unit_types::{KIND_HEAVY, KIND_SPEAR};
use crate::units::Units;

/// FL_TEST_CHARGE=1: the phase-B acceptance. Two lanes of blue spears on
/// HOLD facing the enemy; the west lane stands in SPEARWALL, the east
/// lane in normal order. An orange heavy regiment charges each head-on.
/// Acceptance (bands from measurement): the wall holds its ground
/// (centroid displacement well under the unbraced lane's), the wall lane
/// kills chargers faster (reflected charge bonus), and the unbraced lane
/// still fights back (staggers don't stunlock — attacker losses > 0).
pub(crate) fn spawn_charge_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let mut list = Vec::new();
    for (cx, wall) in [(-150.0_f32, true), (150.0, false)] {
        spawn_regiment(units, terrain, &mut list, 0, KIND_SPEAR, Vec2::new(cx, 0.0), 500, -1.0);
        let v = list.len() - 1;
        list[v].hold = true;
        if wall {
            list[v].spacing = crate::formation::FormSpacing::Wall;
            list[v].reform = true;
        }
        spawn_regiment(
            units,
            terrain,
            &mut list,
            1,
            KIND_HEAVY,
            Vec2::new(cx, 70.0),
            500,
            1.0,
        );
        let g = list.len() - 1;
        list[g].order = Some(crate::orders::Order::Attack(v as u32));
        list[g].auto_order = true;
    }
    groups.list = list;
    info!("[charge-test] west: heavies charge a SPEARWALL; east: heavies charge unbraced spears");
}

/// FL_TEST_CHARGE bookkeeping every 2 s: victim-centroid displacement
/// from spawn (does the wall hold ground?), strengths per side, and the
/// staggered headcount (stunlock watch).
pub(super) fn charge_test_log(
    groups: Res<Groups>,
    units: Res<Units>,
    clock: Res<BattleClock>,
    mut next: Local<PerBattle<f32>>,
) {
    if std::env::var("FL_TEST_CHARGE").is_err() || groups.list.len() < 4 {
        return;
    }
    let next = next.of(&clock);
    let t = clock.secs();
    if t < *next {
        return;
    }
    *next = t + 2.0;
    let mut staggered = [0usize; 2]; // [wall lane, open lane]
    for i in 0..units.len() {
        if units.death_t[i] == 0
            && units.swing[i] & crate::units::SWING_STAGGERED != 0
        {
            staggered[usize::from(units.pos[i].x > 0.0)] += 1;
        }
    }
    let dz = |g: usize| groups.list[g].centroid.y - 0.0;
    info!(
        "[charge-test] t={t:.0}s WALL lane: spears {} (dz {:+.1} m) vs heavies {} | \
         OPEN lane: spears {} (dz {:+.1} m) vs heavies {} | staggered {}/{}",
        groups.list[0].count,
        dz(0),
        groups.list[1].count,
        groups.list[2].count,
        dz(2),
        groups.list[3].count,
        staggered[0],
        staggered[1],
    );
}
