//! Regiment battle setup: both armies spawned as a fixed list of regiments
//! (permanent groups) laid out in ranks, heavies in front. The `Groups`
//! list never changes size after this — stable indices make `units.group`
//! a permanent regiment id.

use bevy::prelude::*;

use crate::orders::{GroupData, Groups, RegState};
use crate::terrain::Terrain;
use crate::unit_types::{KIND_HEAVY, KIND_LIGHT, KIND_SPEAR};
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
        // Morale lives in crate::morale (MoralePlugin).
        app.add_systems(Startup, spawn_battle)
            .add_systems(Update, (rout_test_log, restart_key));
    }
}

fn reg_size() -> usize {
    crate::util::env_or("FL_REG_SIZE", 1000_usize).max(50)
}

/// Fraction of each army's regiments that are heavy infantry (front ranks).
fn heavy_frac() -> f32 {
    crate::util::env_or("FL_HEAVY_FRAC", 0.4)
}

/// Fraction of each army's regiments that are spear infantry (behind the
/// heavies, ahead of the lights).
fn spear_frac() -> f32 {
    crate::util::env_or("FL_SPEAR_FRAC", 0.25)
}

/// Spawn one regiment block (units + GroupData). `dir` faces the enemy
/// (+1 toward -Z spawns rows so the block front is enemy-side).
#[allow(clippy::too_many_arguments)] // spawn-time plumbing, all scalars
fn spawn_regiment(
    units: &mut Units,
    terrain: &Terrain,
    list: &mut Vec<GroupData>,
    team: u8,
    kind: u8,
    anchor: Vec2,
    size: usize,
    dir: f32,
) {
    let cols = ((size as f32 * 2.2).sqrt().ceil() as usize).max(1);
    let rows = size.div_ceil(cols);
    let g = list.len() as u32;
    list.push(GroupData::new(team, kind, anchor, size));
    for k in 0..size {
        let row = k / cols;
        let col = k % cols;
        let seed = (team as u32) << 30 | g << 16 | k as u32;
        let jx = hash01(seed.wrapping_mul(3) + 1) - 0.5;
        let jz = hash01(seed.wrapping_mul(3) + 2) - 0.5;
        let x = anchor.x + (col as f32 - (cols - 1) as f32 / 2.0) * SPACING + jx * 0.5;
        let z = anchor.y + dir * ((row as f32 - (rows - 1) as f32 / 2.0) * SPACING) + jz * 0.5;
        push_unit(units, terrain, seed, team, kind, g, x, z, anchor);
    }
    // Rigid rank-and-file by default: exact slots into `home` (the spawn
    // jitter stays in the positions — inside the hold deadzone, so the
    // line reads organic without anyone shuffling at t=0).
    let gd = &mut list[g as usize];
    gd.shape = crate::formation::FormShape::Rect;
    gd.files = cols as u32;
    crate::formation::assign_slots(units, g, gd);
}

fn spawn_battle(mut units: ResMut<Units>, terrain: Res<Terrain>, mut groups: ResMut<Groups>) {
    do_spawn_battle(&mut units, &terrain, &mut groups);
}

/// R: restart the battle from scratch (fresh armies, cleared stats).
#[allow(clippy::too_many_arguments)] // bevy system params
fn restart_key(
    keys: Res<ButtonInput<KeyCode>>,
    mut units: ResMut<Units>,
    terrain: Res<Terrain>,
    mut groups: ResMut<Groups>,
    mut stats: ResMut<crate::combat::CombatStats>,
    mut selection: ResMut<crate::orders::Selection>,
    mut outcome: ResMut<crate::ai::BattleOutcome>,
    mut corpses: ResMut<crate::render_units::Corpses>,
) {
    if !keys.just_pressed(KeyCode::KeyR) {
        return;
    }
    *units = Units::default();
    *stats = crate::combat::CombatStats::default();
    *selection = crate::orders::Selection::default();
    outcome.0 = None;
    corpses.clear();
    do_spawn_battle(&mut units, &terrain, &mut groups);
    info!("battle restarted");
}

fn do_spawn_battle(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    if std::env::var("FL_TEST_SURROUND").is_ok() {
        crate::units::spawn_surround_test(units, terrain, groups);
        return;
    }
    if std::env::var("FL_TEST_ROUT").is_ok() {
        spawn_rout_test(units, terrain, groups);
        return;
    }

    let per_team = units_per_team();
    let size = reg_size();
    let n_regs = (per_team / size).max(1);

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
        // FL_ENEMY_REGS caps the enemy regiment count (sandbox
        // asymmetry, e.g. 5v4 so one player regiment stays unchased).
        let n_regs = if team == 1 {
            crate::util::env_or("FL_ENEMY_REGS", n_regs).clamp(1, n_regs)
        } else {
            n_regs
        };
        let n_heavy = (n_regs as f32 * heavy_frac()).round() as usize;
        let n_spear = (n_regs as f32 * spear_frac()).round() as usize;
        let dir: f32 = if team == 0 { -1.0 } else { 1.0 };
        for r in 0..n_regs {
            let rank = r / per_rank;
            let file = r % per_rank;
            // This rank's regiment count (last rank may be partial) — center it.
            let in_rank = per_rank.min(n_regs - rank * per_rank);
            let x0 = (file as f32 - (in_rank - 1) as f32 / 2.0) * pitch_x;
            let z0 = dir * (ARMY_GAP / 2.0 + block_d / 2.0 + rank as f32 * (block_d + REG_GAP));
            let anchor = Vec2::new(x0, z0);
            // Heavies lead, spears back them, lights fill the rear ranks.
            let kind = if r < n_heavy {
                KIND_HEAVY
            } else if r < n_heavy + n_spear {
                KIND_SPEAR
            } else {
                KIND_LIGHT
            };
            spawn_regiment(units, terrain, &mut list, team, kind, anchor, size, dir);
        }
    }
    let heavies = list.iter().filter(|g| g.kind == KIND_HEAVY).count();
    let spears = list.iter().filter(|g| g.kind == KIND_SPEAR).count();
    let blue = list.iter().filter(|g| g.team == 0).count();
    info!(
        "spawned {} vs {} regiments ({} heavy, {} spear) x {} units ({} total units)",
        blue,
        list.len() - blue,
        heavies,
        spears,
        size,
        units.len()
    );
    groups.list = list;
}

/// FL_TEST_ROUT: one blue regiment vs three converging orange regiments.
/// Acceptance: blue BREAKS well before annihilation, flees toward its own
/// edge, and despawns there (fled counter), instead of fighting to the
/// last man.
fn spawn_rout_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
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
    time: Res<Time>,
    mut next: Local<f32>,
) {
    if std::env::var("FL_TEST_ROUT").is_err() || groups.list.is_empty() {
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
    info!(
        "[rout-test] t={t:.0}s blue: {} alive, morale {:.0}, {}, fled {}",
        g.count, g.morale, state, stats.fled[0]
    );
}
