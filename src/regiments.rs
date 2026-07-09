//! Regiment battle setup: both armies spawned as a fixed list of regiments
//! (permanent groups) laid out in ranks, heavies in front. The `Groups`
//! list never changes size after this — stable indices make `units.group`
//! a permanent regiment id.

use bevy::prelude::*;

use crate::orders::{GroupData, Groups};
use crate::terrain::Terrain;
use crate::unit_types::{KIND_HEAVY, KIND_LIGHT};
use crate::units::{Units, hash01, push_unit, units_per_team};

/// Unit spacing inside a regiment block.
const SPACING: f32 = 1.4;
/// Gap between regiment blocks.
const REG_GAP: f32 = 10.0;
/// No-man's land between the two armies.
const ARMY_GAP: f32 = 60.0;

pub struct RegimentsPlugin;

impl Plugin for RegimentsPlugin {
    fn build(&self, app: &mut App) {
        // Terrain resource is created in PreStartup (generate_terrain).
        app.add_systems(Startup, spawn_battle);
    }
}

fn reg_size() -> usize {
    std::env::var("FL_REG_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000)
        .max(50)
}

/// Fraction of each army's regiments that are heavy infantry (front ranks).
fn heavy_frac() -> f32 {
    std::env::var("FL_HEAVY_FRAC")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.4)
}

fn spawn_battle(mut units: ResMut<Units>, terrain: Res<Terrain>, mut groups: ResMut<Groups>) {
    if std::env::var("FL_TEST_SURROUND").is_ok() {
        crate::units::spawn_surround_test(&mut units, &terrain, &mut groups);
        return;
    }

    let per_team = units_per_team();
    let size = reg_size();
    let n_regs = (per_team / size).max(1);
    let n_heavy = (n_regs as f32 * heavy_frac()).round() as usize;

    // Block geometry: wider than deep (~2.2:1).
    let cols = ((size as f32 * 2.2).sqrt().ceil() as usize).max(1);
    let rows = size.div_ceil(cols);
    let block_w = cols as f32 * SPACING;
    let block_d = rows as f32 * SPACING;

    // Regiments per rank: prefer filling the map width, but never spawn
    // ranks past the terrain edge (the sim clamps positions to the bounds
    // and stacked rows would squash onto the boundary line). If the army
    // needs more ranks than fit, widen the ranks and shrink the x pitch.
    let usable_w = (terrain.max().x - terrain.min().x) - 60.0;
    let usable_d = terrain.max().y - 8.0 - ARMY_GAP / 2.0;
    let max_ranks = (((usable_d - block_d) / (block_d + REG_GAP)).floor() as usize + 1).max(1);
    let per_rank = ((usable_w / (block_w + REG_GAP)).floor() as usize)
        .max(n_regs.div_ceil(max_ranks))
        .max(1);
    let pitch_x = (usable_w / per_rank as f32).min(block_w + REG_GAP);
    if pitch_x < block_w + 1.0 {
        warn!(
            "regiment layout tight: pitch {pitch_x:.1} m vs block {block_w:.1} m — \
             reduce FL_REG_SIZE or FL_UNITS"
        );
    }

    units.pos.reserve(per_team * 2);
    let mut list: Vec<GroupData> = Vec::with_capacity(n_regs * 2);

    for team in 0..2u8 {
        let dir: f32 = if team == 0 { -1.0 } else { 1.0 };
        for r in 0..n_regs {
            let rank = r / per_rank;
            let file = r % per_rank;
            // This rank's regiment count (last rank may be partial) — center it.
            let in_rank = per_rank.min(n_regs - rank * per_rank);
            let x0 = (file as f32 - (in_rank - 1) as f32 / 2.0) * pitch_x;
            let z0 = dir * (ARMY_GAP / 2.0 + block_d / 2.0 + rank as f32 * (block_d + REG_GAP));
            let anchor = Vec2::new(x0, z0);
            let kind = if r < n_heavy { KIND_HEAVY } else { KIND_LIGHT };

            let g = list.len() as u32;
            list.push(GroupData::new(team, kind, anchor, size));
            for k in 0..size {
                let row = k / cols;
                let col = k % cols;
                let seed = (team as u32) << 30 | (r as u32) << 16 | k as u32;
                let jx = hash01(seed.wrapping_mul(3) + 1) - 0.5;
                let jz = hash01(seed.wrapping_mul(3) + 2) - 0.5;
                let x = x0 + (col as f32 - (cols - 1) as f32 / 2.0) * SPACING + jx * 0.5;
                // Front rows of each block face the enemy.
                let z = z0 + dir * ((row as f32 - (rows - 1) as f32 / 2.0) * SPACING) + jz * 0.5;
                push_unit(&mut units, &terrain, seed, team, kind, g, x, z, anchor);
            }
        }
    }
    let heavies = list.iter().filter(|g| g.kind == KIND_HEAVY).count();
    info!(
        "spawned {} regiments ({} heavy) x {} units per team ({} total units)",
        n_regs,
        heavies / 2,
        size,
        units.len()
    );
    groups.list = list;
}
