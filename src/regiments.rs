//! Regiment battle setup: both armies spawned as a fixed list of regiments
//! (permanent groups) laid out in ranks, heavies in front. The `Groups`
//! list never changes size after this — stable indices make `units.group`
//! a permanent regiment id.

use bevy::prelude::*;

use crate::frontline::InfluenceField;
use crate::orders::{GroupData, Groups, RegState};
use crate::terrain::Terrain;
use crate::unit_types::{KIND_HEAVY, KIND_LIGHT, TYPES};
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
        app.add_systems(Startup, spawn_battle)
            .add_systems(Update, (rout_test_log, restart_key))
            .add_systems(
                FixedUpdate,
                update_morale
                    .after(crate::movement::step_sim)
                    .before(crate::orders::clear_arrived_orders),
            );
    }
}

// --- Morale tuning ---
/// Morale lost per (fraction of initial strength) of fresh casualties:
/// ~35% losses alone break a regiment.
const MORALE_CASUALTY: f32 = 280.0;
/// Drain per second when locally outnumbered >2:1 (density ratio).
const MORALE_OUTNUMBERED: f32 = 3.0;
/// Drain per second per routing friendly regiment within RALLY_R (capped).
const MORALE_ROUT_NEIGHBOR: f32 = 2.5;
const MORALE_ROUT_CAP: f32 = 7.5;
/// Recovery per second when unengaged and undisturbed.
const MORALE_RECOVERY: f32 = 3.0;
/// Routing-neighbor / rally-safety radius.
const NEIGHBOR_R: f32 = 60.0;
/// Broken regiments below this fraction of initial strength shatter.
const SHATTER_FRAC: f32 = 0.15;
/// Seconds routing before a rally roll is allowed.
const RALLY_DELAY: f32 = 8.0;
/// Rally chance per second once allowed and safe.
const RALLY_CHANCE: f32 = 0.02;

/// Per-tick regiment morale update (serial; ~200 rows). Runs after the
/// damage apply pass so `recent_deaths` is this tick's tally.
fn update_morale(
    mut groups: ResMut<Groups>,
    field: Res<InfluenceField>,
    time: Res<Time>,
    mut tick: Local<u32>,
) {
    *tick += 1;
    let dt = time.delta_secs();

    // Snapshot routing centroids for the neighbor drain (state from last
    // tick is fine — morale contagion is not latency-sensitive).
    let routing_centroids: Vec<(u8, Vec2)> = groups
        .list
        .iter()
        .filter(|g| g.state.is_broken() && g.count > 0)
        .map(|g| (g.team, g.centroid))
        .collect();

    for (gi, g) in groups.list.iter_mut().enumerate() {
        if g.count == 0 {
            g.recent_deaths = 0;
            continue;
        }
        let resist = TYPES[g.kind as usize].morale_resist;

        match g.state {
            RegState::Steady => {
                let mut drain = 0.0;
                // Fresh casualties.
                drain +=
                    MORALE_CASUALTY * g.recent_deaths as f32 / g.initial_count as f32 * resist;
                // Psychological pressure scales with DEPLETION: a fresh
                // regiment shrugs off routing neighbors and bad odds; a
                // bleeding one panics. Without this, full-strength
                // regiments cascade-rout off contagion alone.
                let frac = g.count as f32 / g.initial_count as f32;
                let depletion = (1.4 - 1.2 * frac).clamp(0.15, 1.4);
                // Locally outnumbered (blurred density ratio at centroid).
                let own = field.density(g.team, g.centroid);
                let enemy = field.density(1 - g.team, g.centroid);
                let mut pressure = 0.0;
                if enemy / (own + 0.1) > 2.0 {
                    pressure += MORALE_OUTNUMBERED * resist;
                }
                // Routing friendlies nearby shake resolve.
                let rout_drain = routing_centroids
                    .iter()
                    .filter(|(t, c)| *t == g.team && c.distance(g.centroid) < NEIGHBOR_R)
                    .count() as f32
                    * MORALE_ROUT_NEIGHBOR;
                pressure += rout_drain.min(MORALE_ROUT_CAP);
                drain += pressure * depletion * dt;

                if drain > 0.0 {
                    g.morale -= drain;
                } else if !g.engaged {
                    g.morale = (g.morale + MORALE_RECOVERY * dt).min(100.0);
                }
                if g.morale <= 0.0 {
                    g.state = RegState::Routing { since: *tick };
                    g.order = None;
                    info!("regiment {gi} BREAKS ({} of {} left)", g.count, g.initial_count);
                }
            }
            RegState::Routing { since } => {
                if (g.count as f32) < g.initial_count as f32 * SHATTER_FRAC {
                    g.state = RegState::Shattered;
                    info!("regiment {gi} shatters");
                } else if (*tick - since) as f32 * dt > RALLY_DELAY {
                    // Rally only when clear of enemies.
                    let enemy = field.density(1 - g.team, g.centroid);
                    let roll = hash01(tick.wrapping_mul(0x9E37_79B1) ^ (gi as u32) << 8);
                    if enemy < 0.1 && roll < RALLY_CHANCE * dt {
                        g.state = RegState::Steady;
                        g.morale = 40.0;
                        g.anchor = g.centroid;
                        info!("regiment {gi} rallies ({} left)", g.count);
                    }
                }
            }
            RegState::Shattered => {}
        }
        g.recent_deaths = 0;
    }
}

fn reg_size() -> usize {
    crate::util::env_or("FL_REG_SIZE", 1000_usize).max(50)
}

/// Fraction of each army's regiments that are heavy infantry (front ranks).
fn heavy_frac() -> f32 {
    crate::util::env_or("FL_HEAVY_FRAC", 0.4)
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
) {
    if !keys.just_pressed(KeyCode::KeyR) {
        return;
    }
    *units = Units::default();
    *stats = crate::combat::CombatStats::default();
    *selection = crate::orders::Selection::default();
    outcome.0 = None;
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
            spawn_regiment(units, terrain, &mut list, team, kind, anchor, size, dir);
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
        list[g].order = Some(Vec2::new(tx, blue.y));
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
