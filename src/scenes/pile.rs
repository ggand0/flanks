//! FL_TEST_PILE: several regiments attacking one, and its log.

use bevy::prelude::*;

use crate::orders::Groups;
use crate::regiments::spawn_regiment;
use crate::terrain::Terrain;
use crate::unit_types::KIND_LIGHT;
use crate::units::Units;

/// The pile-on test's shape. The FL_PILE_* envs override each field,
/// so the scripts that launch it with FL_TEST_PILE keep their setups.
#[derive(Clone, Copy)]
pub struct PileSetup {
    /// Blue regiments attacking the one orange (1 to 6).
    attackers: usize,
    /// Files of each attacker, 0 for the default block.
    attacker_files: u32,
    /// Files of the victim, 0 for the default block.
    victim_files: u32,
    /// The victim stands at ease instead of in hold, so it answers with
    /// its own attack order like a player's regiment.
    victim_at_ease: bool,
}

impl PileSetup {
    /// Six attackers on one holding regiment: the blob repro.
    pub const SIX: Self =
        Self { attackers: 6, attacker_files: 0, victim_files: 0, victim_at_ease: false };
    /// A 100-file line at ease hit by one 12-file block: the M2TW test
    /// of a deep unit hitting a wide one, which wraps it (devlog 0123).
    pub const WIDE: Self =
        Self { attackers: 1, attacker_files: 12, victim_files: 100, victim_at_ease: true };
    /// Two regiments engage one at once (devlog 0123).
    pub const TWO: Self =
        Self { attackers: 2, attacker_files: 0, victim_files: 0, victim_at_ease: false };
}

/// FL_TEST_PILE=1: the pile-on order: blue regiments in a 3x2 block,
/// all attack-ordered at one orange regiment (the blob repro with six).
/// Acceptance: the fight crowds the victim's perimeter and the second
/// wave stands pressed against the fighting mass (not parked at parade
/// pitch, not smeared into one ball); the victim collapses; blues
/// re-dress rectangles afterward.
pub(crate) fn spawn_pile_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups, setup: PileSetup) {
    let files_env = |key: &str, default: u32| {
        std::env::var(key).ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(default)
    };
    let mut list = Vec::new();
    spawn_regiment(units, terrain, &mut list, 1, KIND_LIGHT, Vec2::new(0.0, 60.0), 500, 1.0);
    // FL_PILE_ATEASE=1: the victim at ease.
    list[0].hold = !(setup.victim_at_ease || std::env::var("FL_PILE_ATEASE").is_ok());
    // FL_PILE_VICTIM_FILES: stretch the victim into a wide line.
    let victim_files = files_env("FL_PILE_VICTIM_FILES", setup.victim_files);
    if victim_files > 0 {
        list[0].files = victim_files;
        crate::formation::assign_slots(units, 0, &mut list[0]);
    }
    // FL_PILE_N: how many attackers.
    let n_attackers = crate::util::env_or("FL_PILE_N", setup.attackers).clamp(1, 6);
    // FL_PILE_FILES: stretch the attackers into a wide line (the
    // runaway-flank repro: files with no enemy in front of them).
    let attacker_files = files_env("FL_PILE_FILES", setup.attacker_files);
    for row in 0..2 {
        for col in 0..3 {
            if row * 3 + col >= n_attackers {
                continue;
            }
            let anchor =
                Vec2::new((col as f32 - 1.0) * 55.0, -40.0 - row as f32 * 35.0);
            spawn_regiment(units, terrain, &mut list, 0, KIND_LIGHT, anchor, 500, -1.0);
            let g = list.len() - 1;
            if attacker_files > 0 {
                list[g].files = attacker_files;
                crate::formation::assign_slots(units, g as u32, &mut list[g]);
            }
            list[g].order = Some(crate::orders::Order::Attack(0));
            list[g].auto_order = true;
        }
    }
    groups.list = list;
    info!("[pile-test] {n_attackers} blue regiments attack ONE orange regiment");
}

/// FL_TEST_PILE bookkeeping every 4 s: victim strength + how many of
/// the six attackers are actually fighting.
pub(super) fn pile_test_log(groups: Res<Groups>, units: Res<Units>, time: Res<Time>, mut next: Local<f32>) {
    if std::env::var("FL_TEST_PILE").is_err() || groups.list.len() < 2 {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 1.0;
    let n = groups.list.len() - 1;
    let engaged = groups.list[1..].iter().filter(|g| g.engaged).count();
    let charging = groups.list[1..].iter().filter(|g| g.charging).count();
    let v = &groups.list[0];
    // How far the victim block has moved along its own facing since
    // spawn (z = 60, facing -z): negative = pushed back.
    let fwd = crate::formation::facing_dir(v.facing);
    let moved = (v.centroid - Vec2::new(0.0, 60.0)).dot(fwd);
    // Its front and rear edges (95th and 5th percentile of depth along
    // its facing, relative to the spawn center): the center alone moves
    // back as front-rank men die even if nobody steps.
    let mut d: Vec<f32> = (0..units.len())
        .filter(|&i| units.group[i] == 0 && units.death_t[i] == 0)
        .map(|i| (Vec2::new(units.pos[i].x, units.pos[i].z) - Vec2::new(0.0, 60.0)).dot(fwd))
        .collect();
    d.sort_by(|a, b| a.total_cmp(b));
    let (back, front) = if d.is_empty() {
        (0.0, 0.0)
    } else {
        (d[d.len() / 20], d[d.len() * 19 / 20])
    };
    // Attackers standing beyond the victim's far side (z > 72): a line
    // whose flank files walk on past the fight shows up here.
    let past = (0..units.len())
        .filter(|&i| units.group[i] != 0 && units.death_t[i] == 0 && units.pos[i].z > 72.0)
        .count();
    let (mut jog, mut walk, mut ready, mut run_back, mut out_form) = (0, 0, 0, 0, 0);
    for i in 0..units.len() {
        if units.group[i] == 0 && units.death_t[i] == 0 {
            if units.out_form[i] {
                out_form += 1;
            }
            let vxz = Vec2::new(units.vel[i].x, units.vel[i].z);
            let sp = vxz.length();
            // Running away from his own regiment's fight while it is in
            // melee: the run-back defect.
            if let Some(fp) = v.fight_point
                && units.swing[i] & crate::units::SWING_STAGGERED == 0
                && vxz.dot((fp - Vec2::new(units.pos[i].x, units.pos[i].z)).normalize_or_zero()) < -1.5
            {
                run_back += 1;
            }
            if sp > 1.5 {
                jog += 1;
            } else if sp > 0.3 {
                walk += 1;
            }
            if units.swing[i] & crate::units::SWING_STATE_MASK == crate::units::SWING_READY {
                ready += 1;
            }
        }
    }
    info!("[pile-test] orange moving: jog {jog} walk {walk}, ready {ready}, out of formation {out_form}, running back from the fight {run_back}");
    // Victim men not at the fight: nearest living attacker more than
    // 8 m away (test-only brute force).
    let attackers: Vec<Vec2> = (0..units.len())
        .filter(|&i| units.group[i] != 0 && units.death_t[i] == 0)
        .map(|i| Vec2::new(units.pos[i].x, units.pos[i].z))
        .collect();
    let out = (0..units.len())
        .filter(|&i| {
            units.group[i] == 0 && units.death_t[i] == 0 && {
                let p = Vec2::new(units.pos[i].x, units.pos[i].z);
                attackers.iter().all(|a| a.distance_squared(p) > 64.0)
            }
        })
        .count();
    info!(
        "[pile-test] t={t:.0}s orange {} alive, center {moved:+.2} front {front:+.2} rear {back:+.2} m along its facing, contact {}, engaged {} | blues engaged {engaged}/{n}, charging {charging}, past its far side {past}, orange away from the fight (>8 m) {out}, melee {} fp {:?}",
        v.count, v.contact, v.engaged, v.melee_ticks, v.fight_point
    );
}
