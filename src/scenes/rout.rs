//! FL_TEST_ROUT: one regiment against three, and the morale log it
//! shares with FL_TEST_FRONT.

use bevy::prelude::*;

use crate::orders::{Groups, RegState};
use crate::regiments::spawn_regiment;
use crate::terrain::Terrain;
use crate::unit_types::KIND_LIGHT;
use crate::units::Units;

/// FL_TEST_ROUT: one blue regiment vs three converging orange regiments.
/// Acceptance: blue BREAKS well before annihilation, flees toward its own
/// edge, and despawns there (fled counter), instead of fighting to the
/// last man.
pub(crate) fn spawn_rout_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let mut list = Vec::new();
    let blue = Vec2::new(0.0, -60.0);
    spawn_regiment(units, terrain, &mut list, 0, KIND_LIGHT, blue, 1000, -1.0);
    for (x, z, tx) in [(-90.0, 40.0, -18.0), (0.0, 60.0, 0.0), (90.0, 40.0, 18.0)] {
        let anchor = Vec2::new(x, z);
        spawn_regiment(units, terrain, &mut list, 1, KIND_LIGHT, anchor, 1000, 1.0);
        let g = list.len() - 1;
        list[g].order = Some(crate::orders::Order::Move(Vec2::new(tx, blue.y)));
    }
    groups.list = list;
    info!("[rout-test] 1 blue vs 3 converging orange regiments");
}

/// FL_TEST_ROUT bookkeeping: blue regiment morale/state timeline.
pub fn rout_test_log(
    groups: Res<Groups>,
    stats: Res<crate::combat::CombatStats>,
    readout: Res<crate::morale::MoraleReadout>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    // Regiment-0 morale/fatigue timeline: the rout acceptance AND a
    // line-fight probe for the front battle (flank ring must read ~0 in
    // a frontal press).
    if (std::env::var("FL_TEST_ROUT").is_err() && std::env::var("FL_TEST_FRONT").is_err())
        || groups.list.is_empty()
    {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 2.0;
    let g = &groups.list[0];
    let state = match g.state {
        RegState::Steady => "steady",
        RegState::Routing { .. } => "ROUTING",
        RegState::Shattered => "SHATTERED",
    };
    let f = readout.0.first().copied().unwrap_or_default();
    info!(
        "[rout-test] t={t:.0}s blue: {} alive, morale {:.0}, {}, fled {} | cas {:.1} xchg {:.1} flank {:.1} ({:.0}%) noen {:.1} fat {:.1} r {:.0}m z {:.0} en {}",
        g.count,
        g.morale,
        state,
        stats.fled[0],
        f.casualties,
        f.exchange,
        f.flanked,
        f.flanked01 * 100.0,
        f.no_enemy,
        g.fatigue,
        g.radius,
        g.centroid.y,
        g.enemy_near,
    );
}
