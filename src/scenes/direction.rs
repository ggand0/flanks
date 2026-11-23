//! FL_TEST_DIR: the directional defence scene and its log.

use bevy::prelude::*;

use crate::orders::Groups;
use crate::regiments::spawn_regiment;
use crate::terrain::Terrain;
use crate::unit_types::KIND_LIGHT;
use crate::units::Units;

/// FL_TEST_DIR: the directional-defense acceptance. Two hammer-and-anvil
/// pairs of light infantry, victims on HOLD facing +Z. Each victim is
/// pinned frontally by a near attacker; a farther second attacker arrives
/// ~2.5 s later — from the LEFT side in the control pair (the shield arm:
/// factor-identical to frontal) and from the REAR in the test pair (skill
/// and shield gone). The damage pass buckets every hit on a blue victim by
/// its actual sector at hit time (sim/damage.rs DirTestStats). Acceptance:
/// rear-sector kills (one feeding regiment) >= front-sector kills (TWO
/// feeding regiments), i.e. per-attacker rear kill rate >= 2x frontal.
pub(crate) fn spawn_dir_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let mut list = Vec::new();
    // Attacker spawn offsets per pair: control = front pin + LEFT side
    // (shield arm covers: factor-identical to frontal), test = front pin
    // + REAR, lone = a single unassisted rear charge on a holding block
    // (the M2TW no-auto-face rule: the victims must NOT pirouette to
    // meet it — their facing holds until the melee rim turns to fight).
    let pairs: [(f32, &[Vec2]); 3] = [
        (-250.0, &[Vec2::new(0.0, 35.0), Vec2::new(55.0, 0.0)]),
        (250.0, &[Vec2::new(0.0, 35.0), Vec2::new(0.0, -55.0)]),
        (0.0, &[Vec2::new(0.0, -55.0)]),
    ];
    for (cx, offsets) in pairs {
        spawn_regiment(units, terrain, &mut list, 0, KIND_LIGHT, Vec2::new(cx, 0.0), 500, -1.0);
        let v = list.len() - 1;
        list[v].hold = true;
        for off in offsets {
            let anchor = Vec2::new(cx, 0.0) + *off;
            spawn_regiment(units, terrain, &mut list, 1, KIND_LIGHT, anchor, 500, 1.0);
            let g = list.len() - 1;
            list[g].order = Some(crate::orders::Order::Move(Vec2::new(cx, 0.0)));
            list[g].auto_order = true;
        }
    }
    groups.list = list;
    info!(
        "[dir-test] control pair (front+left) x=-250, test pair (front+rear) x=+250, \
         lone rear charge x=0"
    );
}

/// FL_TEST_DIR bookkeeping: sector-bucketed kills/damage + per-pair victim
/// survival, every 2 s.
#[allow(clippy::too_many_arguments)] // bevy system params
pub(super) fn dir_test_log(
    dir_stats: Res<crate::sim::damage::DirTestStats>,
    units: Res<Units>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    if !dir_stats.enabled || units.len() == 0 {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 2.0;
    let (mut ctl, mut test, mut lone) = (0usize, 0usize, 0usize);
    // The lone-pair victims' facing: mean |yaw| from the ordered facing
    // (0 = +Z). Before the M2TW no-auto-face rule this snapped to ~pi
    // (the whole block turned to meet the rear approach) long before
    // contact; now only soldiers actually fighting may deviate.
    let (mut yaw_dev, mut yaw_n) = (0.0f32, 0usize);
    for i in 0..units.len() {
        if units.team[i] == 0 && units.death_t[i] == 0 {
            if units.pos[i].x < -100.0 {
                ctl += 1;
            } else if units.pos[i].x > 100.0 {
                test += 1;
            } else {
                lone += 1;
                yaw_dev += units.yaw[i].abs();
                yaw_n += 1;
            }
        }
    }
    let k = &dir_stats.kills[0];
    let per_hit = |s: usize| dir_stats.dmg[0][s] / dir_stats.hits[0][s].max(1) as f64;
    info!(
        "[dir-test] t={t:.0}s kills by sector: front {} side {} rear {} \
         (blows {}/{}/{}, dmg/blow {:.1}/{:.1}/{:.1}); victims alive: control {ctl}/500, \
         test {test}/500, lone {lone}/500 (yaw dev {:.2} rad); outcomes {}; \
         parries by depth {:?}",
        k[0],
        k[1],
        k[2],
        dir_stats.hits[0][0],
        dir_stats.hits[0][1],
        dir_stats.hits[0][2],
        per_hit(0),
        per_hit(1),
        per_hit(2),
        yaw_dev / yaw_n.max(1) as f32,
        outcome_line(&dir_stats.outcomes[0]),
        dir_stats.chain,
    );
}

/// The outcome counts of the blows on one team, named, for the logs.
pub(super) fn outcome_line(o: &[u64; crate::sim::damage::NUM_OUTCOMES]) -> String {
    crate::sim::damage::OUTCOME_NAMES
        .iter()
        .zip(o)
        .map(|(n, c)| format!("{n} {c}"))
        .collect::<Vec<_>>()
        .join(", ")
}
