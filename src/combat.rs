//! Death processing: swap-remove dead units from every SoA column (keeping
//! the selection mask in sync), count kills, and occasionally carve a small
//! crater where someone died.

use bevy::prelude::*;

use crate::orders::Selection;
use crate::terrain::Terrain;
use crate::units::{Units, hash01};

/// Fraction of kills that leave a crater. Disabled (owner call) — deaths
/// don't blow holes in the ground; craters return with explosives/artillery.
const CRATER_CHANCE: f32 = 0.0;
/// Remesh guard: max craters carved per tick.
const CRATERS_PER_TICK: usize = 2;

#[derive(Resource, Default)]
pub struct CombatStats {
    pub kills: [u64; 2],
    pub alive: [usize; 2],
}

pub struct CombatPlugin;

impl Plugin for CombatPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CombatStats>().add_systems(
            FixedUpdate,
            process_deaths
                .after(crate::movement::step_sim)
                .before(crate::orders::clear_arrived_orders),
        );
    }
}

fn process_deaths(
    mut units: ResMut<Units>,
    mut selection: ResMut<Selection>,
    mut terrain: ResMut<Terrain>,
    mut stats: ResMut<CombatStats>,
) {
    let _span = info_span!("process_deaths").entered();
    if selection.mask.len() != units.len() {
        selection.mask.clear();
        selection.count = 0;
    }
    let mut craters: Vec<(Vec2, f32)> = Vec::new();
    let mut i = 0;
    while i < units.len() {
        if units.hp[i] > 0.0 {
            i += 1;
            continue;
        }
        let team = units.team[i] as usize;
        stats.kills[team] += 1;
        let seed = stats.kills[team] as u32 ^ ((team as u32) << 30);
        if craters.len() < CRATERS_PER_TICK && hash01(seed) < CRATER_CHANCE {
            craters.push((
                Vec2::new(units.pos[i].x, units.pos[i].z),
                4.5 + 4.0 * hash01(seed ^ 0x5bd1_e995),
            ));
        }
        units.pos.swap_remove(i);
        units.pos_prev.swap_remove(i);
        units.vel.swap_remove(i);
        units.speed.swap_remove(i);
        units.team.swap_remove(i);
        units.group.swap_remove(i);
        units.color.swap_remove(i);
        units.hp.swap_remove(i);
        if !selection.mask.is_empty() && selection.mask.swap_remove(i) {
            selection.count -= 1;
        }
    }
    for (c, r) in craters {
        terrain.carve_crater(c, r, r * 0.4);
    }

    stats.alive = [0, 0];
    for &t in &units.team {
        stats.alive[t as usize] += 1;
    }
}
