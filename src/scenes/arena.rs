//! FL_ARENA: the hand-testing range for directional damage, and its
//! log.

use bevy::prelude::*;

use crate::orders::Groups;
use crate::regiments::spawn_regiment;
use crate::terrain::Terrain;
use crate::unit_types::KIND_HEAVY;
use crate::units::Units;
use super::direction::outcome_line;

/// FL_ARENA=1: hand-testing range for directional damage. Two mirrored
/// duel lanes of identical heavy regiments; both enemies HOLD facing
/// south (toward the player side) with the AI stood down. Lane FRONT
/// (west): your regiment south of the enemy — attack it head-on. Lane
/// REAR (east): your regiment already parked BEHIND the enemy — attack
/// south into its back. Same kind, same size, same distance; the only
/// variable is the sector, and the [arena] log prints your kills and
/// damage per hit by sector so the comparison is numbers, not vibes.
pub(crate) fn spawn_arena(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let size = crate::util::env_or("FL_REG_SIZE", 500_usize).max(50);
    let mut list = Vec::new();
    for (lane_x, player_z) in [(-80.0_f32, -85.0_f32), (80.0, 85.0)] {
        spawn_regiment(
            units,
            terrain,
            &mut list,
            0,
            KIND_HEAVY,
            Vec2::new(lane_x, player_z),
            size,
            if player_z < 0.0 { -1.0 } else { 1.0 },
        );
        let g = list.len() - 1;
        if player_z > 0.0 {
            // The rear-lane regiment starts behind the enemy: face back
            // south toward it (slots rebake to the new facing).
            list[g].facing = std::f32::consts::PI;
            crate::formation::assign_slots(units, g as u32, &mut list[g]);
        }
        spawn_regiment(units, terrain, &mut list, 1, KIND_HEAVY, Vec2::new(lane_x, 0.0), size, 1.0);
        let e = list.len() - 1;
        list[e].hold = true; // faces south by default (team-1 facing)
        // FL_ARENA_AUTO=1: fire both attacks immediately through the
        // same Order::Attack the RMB click writes — the whole arena
        // becomes a scripted acceptance run for the sector numbers.
        if std::env::var("FL_ARENA_AUTO").is_ok() {
            list[g].order = Some(crate::orders::Order::Attack(e as u32));
        }
    }
    groups.list = list;
    info!(
        "[arena] FRONT lane west: attack the enemy head-on. REAR lane east: \
         attack the enemy from behind. Identical heavies, enemies hold, no AI."
    );
}

/// FL_ARENA bookkeeping: per-lane strengths + the player's kills and
/// damage per hit by sector, every 4 s once fighting starts.
pub(super) fn arena_log(
    dir_stats: Res<crate::sim::damage::DirTestStats>,
    units: Res<Units>,
    groups: Res<Groups>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    if std::env::var("FL_ARENA").is_err() || units.len() == 0 || groups.list.len() < 4 {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 4.0;
    // Lane split at x = 0: west = frontal duel, east = rear duel.
    let mut alive = [[0usize; 2]; 2]; // [lane][team]
    for i in 0..units.len() {
        if units.death_t[i] == 0 {
            let lane = usize::from(units.pos[i].x > 0.0);
            alive[lane][units.team[i] as usize] += 1;
        }
    }
    // Your kills = hits on team-1 victims.
    let k = &dir_stats.kills[1];
    if k.iter().sum::<u64>() == 0 && dir_stats.kills[0].iter().sum::<u64>() == 0 {
        return; // nothing has happened yet — keep the log quiet
    }
    let per_hit = |s: usize| dir_stats.dmg[1][s] / dir_stats.hits[1][s].max(1) as f64;
    info!(
        "[arena] t={t:.0}s FRONT lane you {} vs {} (morale {:.0}) | \
         REAR lane you {} vs {} (morale {:.0}) | \
         your kills: front {} ({:.1}/blow), side {} ({:.1}/blow), rear {} ({:.1}/blow) | \
         your blows: {} | theirs: {} | parries by depth {:?}",
        alive[0][0],
        alive[0][1],
        groups.list[1].morale,
        alive[1][0],
        alive[1][1],
        groups.list[3].morale,
        k[0],
        per_hit(0),
        k[1],
        per_hit(1),
        k[2],
        per_hit(2),
        outcome_line(&dir_stats.outcomes[1]),
        outcome_line(&dir_stats.outcomes[0]),
        dir_stats.chain,
    );
}
