//! Regiment battle setup: both armies spawned as a fixed list of regiments
//! (permanent groups) laid out in ranks, heavies in front. The `Groups`
//! list never changes size after this — stable indices make `units.group`
//! a permanent regiment id.

use bevy::prelude::*;

use crate::orders::{GroupData, Groups};
use crate::scenes::archery::spawn_archery_test;
use crate::scenes::arena::spawn_arena;
use crate::scenes::charge::spawn_charge_test;
use crate::scenes::direction::spawn_dir_test;
use crate::scenes::join::spawn_join_test;
use crate::scenes::pass::spawn_routpass_test;
use crate::scenes::pile::{PileSetup, spawn_pile_test};
use crate::scenes::push::spawn_push_test;
use crate::scenes::retarget::spawn_retarget_test;
use crate::scenes::rout::spawn_rout_test;
use crate::terrain::Terrain;
use crate::unit_types::{KIND_ARCHER, KIND_HEAVY, KIND_LIGHT, KIND_SPEAR, NUM_KINDS};
use crate::units::{Units, hash01, push_unit};

/// Gap between regiment blocks.
pub(crate) const REG_GAP: f32 = 10.0;
/// Margins the army layout keeps from the terrain edges. Shared with the
/// deployment zone (orders.rs), which must contain the spawn strip by
/// construction.
pub const SIDE_MARGIN: f32 = 30.0;
pub const EDGE_MARGIN: f32 = 8.0;
/// No-man's land between the two armies. FL_ARMY_GAP overrides — small
/// sandbox battles (e.g. FL_UNITS=5000 FL_ENEMY_REGS=4) want 250+ m of
/// maneuvering room for flank and rear charges; the 200k default keeps
/// the short march.
pub fn army_gap() -> f32 {
    crate::util::env_or("FL_ARMY_GAP", 60.0)
}

/// The z the no-man's land is centred on and its depth, on the map
/// `kind`. A map with the stone wall puts the player's army (team 0) on
/// the plateau behind it (in the town, on the walled town), its front
/// 10 m north of the wall's line so the first ranks stand at the wall's
/// inner foot, and the enemy on the plain south of z = 150; FL_ARMY_GAP
/// still sets the depth there, around the same centre. Every other map
/// centres the gap on the field.
pub fn army_lines(kind: crate::terrain::MapKind) -> (f32, f32) {
    if kind.has_wall() {
        let (north, south) = (crate::town_layout::WALL_Z - 10.0, 150.0);
        ((north + south) * 0.5, crate::util::env_or("FL_ARMY_GAP", south - north))
    } else {
        (0.0, army_gap())
    }
}

/// The corners of the field the army layout fills: the terrain's bounds,
/// or with FL_ARMY_FIELD=WxD (metres) a field of that size centred on the
/// map and clipped to it. FL_ARMY_FIELD=1024x768 packs a big map's armies
/// as on Grassland, for benchmarks.
fn army_field(terrain: &Terrain) -> (Vec2, Vec2) {
    let size = std::env::var("FL_ARMY_FIELD").ok().and_then(|v| {
        let (w, d) = v.split_once('x')?;
        Some(Vec2::new(w.trim().parse().ok()?, d.trim().parse().ok()?))
    });
    match size {
        Some(size) => (terrain.min().max(-size / 2.0), terrain.max().min(size / 2.0)),
        None => (terrain.min(), terrain.max()),
    }
}

fn heavy_frac() -> f32 {
    crate::util::env_or("FL_HEAVY_FRAC", 0.4)
}

/// Fraction of each army's regiments that are spear infantry (behind the
/// heavies, ahead of the lights).
fn spear_frac() -> f32 {
    crate::util::env_or("FL_SPEAR_FRAC", 0.25)
}

/// Archer regiments per army: a fixed COUNT by default — archers are
/// force multipliers, and scaling them with army size turned big
/// battles into arrow weather; two are enough for how strong they
/// are. FL_ARCHER_FRAC switches back to a
/// fraction of the army for sandbox play (=1 for all-archer fields).
fn archer_regs(n_regs: usize) -> usize {
    match std::env::var("FL_ARCHER_FRAC")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
    {
        Some(frac) => (n_regs as f32 * frac).round() as usize,
        None => 2.min(n_regs),
    }
}

/// Resolve heavy/spear/archer fractions of the regiment budget into
/// counts, clamped in that order; light infantry absorbs the rest.
fn comp_from_fracs(n_regs: usize, heavy: f32, spear: f32, archer: usize) -> [usize; NUM_KINDS] {
    let n_heavy = ((n_regs as f32 * heavy).round() as usize).min(n_regs);
    let n_spear = ((n_regs as f32 * spear).round() as usize).min(n_regs - n_heavy);
    let n_archer = archer.min(n_regs - n_heavy - n_spear);
    let mut comp = [0; NUM_KINDS];
    comp[KIND_HEAVY as usize] = n_heavy;
    comp[KIND_SPEAR as usize] = n_spear;
    comp[KIND_ARCHER as usize] = n_archer;
    comp[KIND_LIGHT as usize] = n_regs - n_heavy - n_spear - n_archer;
    comp
}

/// The classic default split as explicit per-kind regiment counts
/// (KIND_* indexed), honoring the FL_*_FRAC sandbox envs. The Select
/// Units screen prefills from this; test batteries that script
/// composition through the env fracs spawn through it unchanged.
pub fn frac_comp(n_regs: usize) -> [usize; NUM_KINDS] {
    comp_from_fracs(n_regs, heavy_frac(), spear_frac(), archer_regs(n_regs))
}

/// An enemy army style: a coherent composition that fights its own
/// way. Fracs of the regiment budget for heavy/spear/archer; light
/// fills the rest.
struct Archetype {
    name: &'static str,
    heavy: f32,
    spear: f32,
    archer: f32,
}

const ARCHETYPES: &[Archetype] = &[
    Archetype { name: "Balanced Host", heavy: 0.40, spear: 0.25, archer: 0.02 },
    Archetype { name: "Iron Wall", heavy: 0.62, spear: 0.20, archer: 0.01 },
    Archetype { name: "Spear Hedge", heavy: 0.12, spear: 0.58, archer: 0.02 },
    Archetype { name: "Arrow Storm", heavy: 0.16, spear: 0.20, archer: 0.22 },
    Archetype { name: "Skirmish Horde", heavy: 0.05, spear: 0.12, archer: 0.06 },
];

pub fn archetype_count() -> usize {
    ARCHETYPES.len()
}

pub fn archetype_name(idx: usize) -> &'static str {
    ARCHETYPES[idx].name
}

/// The un-jittered representative composition of a style at the given
/// budget: what the picker shows when the style is selected.
pub fn archetype_preview(idx: usize, n_regs: usize) -> [usize; NUM_KINDS] {
    let a = &ARCHETYPES[idx];
    comp_from_fracs(
        n_regs,
        a.heavy,
        a.spear,
        (n_regs as f32 * a.archer).round() as usize,
    )
}

/// A battle-start draw of a style: each fraction jittered by up to a
/// fifth so two draws of the same style differ.
fn archetype_jittered(idx: usize, n_regs: usize, seed: u32) -> [usize; NUM_KINDS] {
    let a = &ARCHETYPES[idx];
    let jit = |i: u32| 0.8 + 0.4 * hash01(seed.wrapping_mul(31).wrapping_add(i));
    comp_from_fracs(
        n_regs,
        a.heavy * jit(1),
        a.spear * jit(2),
        (n_regs as f32 * a.archer * jit(3)).round() as usize,
    )
}

fn time_seed() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(7)
}

/// Spawn one regiment block (units + GroupData). `dir` faces the enemy
/// (+1 toward -Z spawns rows so the block front is enemy-side).
#[allow(clippy::too_many_arguments)] // spawn-time plumbing, all scalars
pub(crate) fn spawn_regiment(
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
        let seed = crate::units::spawn_seed((team as u32) << 30 | g << 16 | k as u32);
        let jx = hash01(seed.wrapping_mul(3) + 1) - 0.5;
        let jz = hash01(seed.wrapping_mul(3) + 2) - 0.5;
        let gap = crate::formation::gap(kind);
        let x = anchor.x + (col as f32 - (cols - 1) as f32 / 2.0) * gap + jx * 0.5;
        let z = anchor.y + dir * ((row as f32 - (rows - 1) as f32 / 2.0) * gap) + jz * 0.5;
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

/// The centre x of each block in a rank laid with no gap between its
/// blocks, given their widths left to right: each block starts where the
/// one before it ends and the whole run is centred on x = 0. The outer
/// files of two neighbours then stand half of each one's pitch from their
/// shared edge, as in the Ctrl drag line (orders.rs `line_layout`).
fn gapless_rank_xs(widths: &[f32]) -> Vec<f32> {
    let total: f32 = widths.iter().sum();
    let mut run = 0.0;
    widths
        .iter()
        .map(|&w| {
            let x = run + w * 0.5 - total * 0.5;
            run += w;
            x
        })
        .collect()
}

/// `nav` is the map's walkability grid when it has one: the layout then
/// keeps every regiment's footprint on open ground.
pub fn do_spawn_battle(
    units: &mut Units,
    terrain: &Terrain,
    groups: &mut Groups,
    config: &crate::game_state::BattleConfig,
    nav: Option<&crate::nav::NavGrid>,
) {
    if spawn_scene(units, terrain, groups, config.scenario) {
        return;
    }

    let per_team = config.units_per_team;
    let size = config.reg_size;
    let n_regs = (per_team / size).max(1);

    // Block geometry: wider than deep (~2.2:1), on the ground the kind
    // with the widest gap needs.
    let cols = ((size as f32 * 2.2).sqrt().ceil() as usize).max(1);
    let rows = size.div_ceil(cols);
    let gap = crate::formation::widest_gap();
    let block_w = cols as f32 * gap;
    let block_d = rows as f32 * gap;

    // Regiments per rank: prefer filling the field's width, but never spawn
    // ranks past its edge (at the terrain's edge the sim clamps positions
    // and stacked rows would squash onto the boundary line). If the army
    // needs more ranks than fit, widen the ranks and shrink the x pitch.
    let (field_min, field_max) = army_field(terrain);
    let usable_w = (field_max.x - field_min.x) - 2.0 * SIDE_MARGIN;
    let (centre, army_gap) = army_lines(terrain.kind);
    // The shallower side's depth behind its front.
    let usable_d = (field_max.y - EDGE_MARGIN - army_gap / 2.0 - centre)
        .min(-field_min.y - EDGE_MARGIN - army_gap / 2.0 + centre);
    let max_ranks = (((usable_d - block_d) / (block_d + REG_GAP)).floor() as usize + 1).max(1);
    let per_rank = ((usable_w / (block_w + REG_GAP)).floor() as usize)
        .max(n_regs.div_ceil(max_ranks))
        .max(1);
    let pitch_x = (usable_w / per_rank as f32).min(block_w + REG_GAP);
    // On Line Debug each rank stands with no gap between its regiments
    // (`gapless_rank_xs`): it is tight only when the rank of the widest
    // blocks would not fit the usable width.
    let gapless = terrain.kind == crate::terrain::MapKind::LineDebug;
    let tight = if gapless { per_rank as f32 * block_w > usable_w } else { pitch_x < block_w + 1.0 };
    if tight {
        warn!(
            "regiment layout tight: pitch {pitch_x:.1} m vs block {block_w:.1} m — \
             reduce FL_REG_SIZE or FL_UNITS"
        );
    }

    units.pos.reserve(per_team * 2);
    let mut list: Vec<GroupData> = Vec::with_capacity(n_regs * 2);
    // A block's footprint, for the ground check.
    let half_ext = Vec2::new(block_w * 0.5 + 1.0, block_d * 0.5 + 1.0);
    let fits = |a: Vec2| nav.is_none_or(|n| n.footprint_clear(a, half_ext));
    let mut moved = 0usize;

    // The FL_*_FRAC envs trump the picker compositions: test batteries
    // and sandbox runs script composition through them and never see
    // the Select Units screen. Scripted runs (FL_TEST_FRONT and kin
    // spawn through this path too) also stay on the classic split so
    // no battery ever meets a random enemy.
    let env_fracs = ["FL_HEAVY_FRAC", "FL_SPEAR_FRAC", "FL_ARCHER_FRAC"]
        .iter()
        .any(|v| std::env::var(v).is_ok())
        || crate::game_state::scripts_active();

    for team in 0..2u8 {
        let comp = army_comp(team, config, n_regs, env_fracs);
        let kinds = ladder(&comp);
        let n_regs = kinds.len();
        // Line Debug: each rank's x positions, the blocks side by side.
        let gapless_xs: Vec<f32> = if gapless {
            kinds
                .chunks(per_rank)
                .flat_map(|rank| {
                    let widths: Vec<f32> = rank
                        .iter()
                        .map(|&k| cols as f32 * crate::formation::FormSpacing::Normal.pitch(k).x)
                        .collect();
                    gapless_rank_xs(&widths)
                })
                .collect()
        } else {
            Vec::new()
        };
        let dir: f32 = if team == 0 { -1.0 } else { 1.0 };
        let rank_z = |rank: usize| centre + dir * (army_gap / 2.0 + block_d / 2.0 + rank as f32 * (block_d + REG_GAP));
        // A regiment whose planned slot stands on blocked ground (a wall,
        // a house, a roof) takes the next free slot in the spare ranks
        // behind the army instead: full-width ranks, walked file by file,
        // out to the field's edge.
        let planned_ranks = n_regs.div_ceil(per_rank);
        let mut spare = planned_ranks * per_rank;
        let past_edge =
            |z: f32| z < field_min.y + EDGE_MARGIN + block_d / 2.0 || z > field_max.y - EDGE_MARGIN - block_d / 2.0;
        for (r, &kind) in kinds.iter().enumerate() {
            let rank = r / per_rank;
            let file = r % per_rank;
            // This rank's regiment count (last rank may be partial) — center it.
            let in_rank = per_rank.min(n_regs - rank * per_rank);
            let x0 = if gapless { gapless_xs[r] } else { (file as f32 - (in_rank - 1) as f32 / 2.0) * pitch_x };
            let mut anchor = Vec2::new(x0, rank_z(rank));
            if !fits(anchor) {
                loop {
                    let (rank, file) = (spare / per_rank, spare % per_rank);
                    spare += 1;
                    let z = rank_z(rank);
                    if past_edge(z) {
                        warn!("regiment layout: no open ground left for a regiment; placed on its planned slot");
                        break;
                    }
                    let candidate = Vec2::new((file as f32 - (per_rank - 1) as f32 / 2.0) * pitch_x, z);
                    if fits(candidate) {
                        anchor = candidate;
                        moved += 1;
                        break;
                    }
                }
            }
            spawn_regiment(units, terrain, &mut list, team, kind, anchor, size, dir);
        }
    }
    if moved > 0 {
        info!("regiment layout: {moved} regiments moved off blocked ground to the spare ranks");
    }
    if let Some(nav) = nav {
        prepare_route_costs(nav, &list);
    }
    // FL_ENEMY_STATIC=1: the enemy army holds position (never auto-
    // engages, never chases — hold's existing semantics) and the AI
    // stands down (ai.rs). A standing target army for charge practice;
    // units still defend themselves in melee and still rout.
    if std::env::var("FL_ENEMY_STATIC").is_ok() {
        for gd in list.iter_mut().filter(|g| g.team == 1) {
            gd.hold = true;
        }
        info!("enemy army is STATIC (hold position, no AI)");
    }
    let heavies = list.iter().filter(|g| g.kind == KIND_HEAVY).count();
    let spears = list.iter().filter(|g| g.kind == KIND_SPEAR).count();
    let archers = list.iter().filter(|g| g.kind == KIND_ARCHER).count();
    let blue = list.iter().filter(|g| g.team == 0).count();
    info!(
        "spawned {} vs {} regiments ({} heavy, {} spear, {} archer) x {} units ({} total units)",
        blue,
        list.len() - blue,
        heavies,
        spears,
        archers,
        size,
        units.len()
    );
    groups.list = list;
}

/// The scripted scenes and the empty scene, which spawn their own
/// regiments or none: whether `scenario` is one of them (or a path
/// scene is set) and was spawned.
#[inline]
fn spawn_scene(
    units: &mut Units,
    terrain: &Terrain,
    groups: &mut Groups,
    scenario: crate::game_state::Scenario,
) -> bool {
    use crate::game_state::Scenario;
    if crate::scenes::path::scene().is_some() {
        crate::scenes::path::spawn(units, terrain, groups);
        return true;
    }
    match scenario {
        Scenario::Surround => {
            crate::scenes::surround::spawn_surround_test(units, terrain, groups);
            true
        }
        Scenario::Rout => { spawn_rout_test(units, terrain, groups); true }
        Scenario::Dir => { spawn_dir_test(units, terrain, groups); true }
        Scenario::Arena => { spawn_arena(units, terrain, groups); true }
        Scenario::Charge => { spawn_charge_test(units, terrain, groups); true }
        Scenario::Pile => {
            spawn_pile_test(units, terrain, groups, PileSetup::SIX);
            true
        }
        Scenario::PileWide => {
            spawn_pile_test(units, terrain, groups, PileSetup::WIDE);
            true
        }
        Scenario::PileTwo => {
            spawn_pile_test(units, terrain, groups, PileSetup::TWO);
            true
        }
        Scenario::Join => { spawn_join_test(units, terrain, groups); true }
        Scenario::Routpass => { spawn_routpass_test(units, terrain, groups); true }
        Scenario::Push => { spawn_push_test(units, terrain, groups); true }
        // The scene: no armies. The victory check stays quiet on an empty
        // regiment list and the camera is free in the Battle state.
        Scenario::Scene => { groups.list.clear(); true }
        Scenario::Archery => { spawn_archery_test(units, terrain, groups); true }
        Scenario::Retarget => { spawn_retarget_test(units, terrain, groups); true }
        Scenario::Normal | Scenario::Demo => false,
    }
}

/// Team `team`'s regiments per kind: the picker's, the enemy's style or
/// the classic split.
#[inline]
fn army_comp(team: u8, config: &crate::game_state::BattleConfig, n_regs: usize, env_fracs: bool) -> [usize; NUM_KINDS] {
    let comp = if team == 0 {
        if env_fracs { frac_comp(n_regs) } else { config.player_regs }
    } else {
        // FL_ENEMY_REGS caps the enemy regiment count (sandbox
        // asymmetry, e.g. 5v4 so one player regiment stays
        // unchased); it also falls back to the classic split.
        let capped = crate::util::env_or("FL_ENEMY_REGS", n_regs).clamp(1, n_regs);
        if env_fracs || capped != n_regs {
            frac_comp(capped)
        } else {
            use crate::game_state::EnemyComp;
            match config.enemy {
                EnemyComp::Manual(comp) => comp,
                EnemyComp::Style(idx) => {
                    let idx = idx.min(ARCHETYPES.len() - 1);
                    info!("enemy army style: {}", archetype_name(idx));
                    archetype_jittered(idx, n_regs, time_seed())
                }
                EnemyComp::Random => {
                    let seed = time_seed();
                    let idx =
                        (hash01(seed) * ARCHETYPES.len() as f32) as usize % ARCHETYPES.len();
                    info!("enemy army style: {}", archetype_name(idx));
                    archetype_jittered(idx, n_regs, seed)
                }
            }
        }
    };
    // An empty army cannot fight; fall back to the classic split.
    if comp.iter().sum::<usize>() == 0 { frac_comp(n_regs) } else { comp }
}

/// The army's regiment kinds in ladder order.
#[inline]
fn ladder(comp: &[usize; NUM_KINDS]) -> Vec<u8> {
    // Ladder order: heavies lead, spears back them, lights fill the
    // middle, archers take the rear ranks (they shoot over everyone).
    let mut kinds: Vec<u8> = Vec::new();
    for k in [KIND_HEAVY, KIND_SPEAR, KIND_LIGHT, KIND_ARCHER] {
        kinds.extend(std::iter::repeat_n(k, comp[k as usize]));
    }
    // FL_ARCHER_SCATTER=1: the archer regiments stand spread through
    // the army, one at the middle of each equal stretch of the
    // ladder, instead of together in the rear ranks. The scene of the
    // ten-archer measurement: every archer regiment has its own
    // neighbours and its own enemy blocks in range.
    let n_archer = comp[KIND_ARCHER as usize];
    if std::env::var("FL_ARCHER_SCATTER").is_ok() && n_archer > 0 {
        let n = kinds.len();
        let slots: Vec<usize> = (0..n_archer).map(|k| (2 * k + 1) * n / (2 * n_archer)).collect();
        let mut others = kinds.iter().copied().filter(|&k| k != KIND_ARCHER);
        kinds = (0..n)
            .map(|r| if slots.contains(&r) { KIND_ARCHER } else { others.next().unwrap_or(KIND_LIGHT) })
            .collect();
    }
    kinds
}

/// The cost tables the routes will need, one per width class a regiment
/// spans across any way it may take, built now rather than in a tick.
#[inline]
fn prepare_route_costs(nav: &crate::nav::NavGrid, list: &[GroupData]) {
    let mut widths: Vec<crate::nav::PlanRequest> = Vec::new();
    for gd in list {
        for half_w in [
            crate::sim::route::half_width(gd, Vec2::Y),
            crate::sim::route::half_width(gd, Vec2::X),
            crate::sim::route::half_width_max(gd),
        ] {
            widths.push(crate::nav::PlanRequest { g: 0, serial: 0, start: Vec2::ZERO, goal: Vec2::ZERO, half_w });
        }
    }
    nav.prepare_costs(&widths);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game_state::{BattleConfig, EnemyComp, Scenario};
    use crate::terrain::{MapKind, build_terrain};

    fn spawn(map: MapKind, per_team: usize, size: usize, player: [usize; NUM_KINDS], enemy: [usize; NUM_KINDS]) -> (Terrain, Units, Groups) {
        let terrain = build_terrain(map);
        let config = BattleConfig {
            units_per_team: per_team,
            reg_size: size,
            map,
            scenario: Scenario::Normal,
            player_regs: player,
            enemy: EnemyComp::Manual(enemy),
            ..default()
        };
        let (mut units, mut groups) = (Units::default(), Groups::default());
        do_spawn_battle(&mut units, &terrain, &mut groups, &config, None);
        (terrain, units, groups)
    }

    /// Each team's ranks, front to back, each its regiment indices left
    /// to right.
    fn ranks(groups: &Groups, team: u8) -> Vec<Vec<usize>> {
        let mut out: Vec<(f32, Vec<usize>)> = Vec::new();
        for (g, gd) in groups.list.iter().enumerate().filter(|(_, gd)| gd.team == team) {
            match out.iter_mut().find(|(z, _)| *z == gd.anchor.y) {
                Some((_, rank)) => rank.push(g),
                None => out.push((gd.anchor.y, vec![g])),
            }
        }
        out.sort_by(|a, b| a.0.abs().total_cmp(&b.0.abs()));
        for (_, rank) in &mut out {
            rank.sort_by(|&a, &b| groups.list[a].anchor.x.total_cmp(&groups.list[b].anchor.x));
        }
        out.into_iter().map(|(_, rank)| rank).collect()
    }

    /// The leftmost and rightmost slot x of regiment `g`.
    fn slot_span(units: &Units, groups: &Groups, g: usize) -> (f32, f32) {
        let anchor = groups.list[g].anchor.x;
        (0..units.len()).filter(|&i| units.group[i] == g as u32).fold((f32::MAX, f32::MIN), |(lo, hi), i| {
            let x = anchor + units.home[i].x;
            (lo.min(x), hi.max(x))
        })
    }

    #[test]
    fn gapless_rank_xs_abut_blocks_of_unequal_pitch() {
        // 47 files at three made-up pitches: the outer files of two
        // neighbours stand the mean of their pitches apart.
        let pitches = [1.4, 2.0, 1.1];
        let widths: Vec<f32> = pitches.iter().map(|p| 47.0 * p).collect();
        let xs = gapless_rank_xs(&widths);
        let total: f32 = widths.iter().sum();
        assert!((xs[0] - widths[0] * 0.5 + total * 0.5).abs() < 1e-3);
        assert!((xs[2] + widths[2] * 0.5 - total * 0.5).abs() < 1e-3);
        for k in 0..2 {
            let right_file = xs[k] + 23.0 * pitches[k];
            let left_file = xs[k + 1] - 23.0 * pitches[k + 1];
            let seam = left_file - right_file;
            assert!((seam - 0.5 * (pitches[k] + pitches[k + 1])).abs() < 1e-3, "seam {seam}");
        }
    }

    /// Line Debug lays every rank of both armies with no gap: neighbours'
    /// outer files the mean of their file pitches apart, every man on the
    /// field.
    #[test]
    fn line_debug_ranks_have_no_gaps() {
        let setups = [
            (100_000, 1000, frac_comp(100), [25, 25, 25, 25]),
            (10_000, 500, [6, 5, 5, 4], [3, 7, 4, 6]),
        ];
        for (per_team, size, player, enemy) in setups {
            let (terrain, units, groups) = spawn(MapKind::LineDebug, per_team, size, player, enemy);
            let (min, max) = (terrain.min(), terrain.max());
            for i in 0..units.len() {
                let p = units.pos[i];
                assert!(p.x > min.x && p.x < max.x && p.z > min.y && p.z < max.y, "man {i} off the field at {p}");
            }
            for team in 0..2 {
                let ranks = ranks(&groups, team);
                for rank in &ranks {
                    for pair in rank.windows(2) {
                        let (a, b) = (&groups.list[pair[0]], &groups.list[pair[1]]);
                        let pitch = 0.5 * (crate::formation::gap(a.kind) + crate::formation::gap(b.kind));
                        let seam = slot_span(&units, &groups, pair[1]).0 - slot_span(&units, &groups, pair[0]).1;
                        assert!((seam - pitch).abs() < 1e-3, "team {team}: seam {seam} m, the pitch {pitch} m");
                    }
                    let mixed = rank.iter().any(|&g| groups.list[g].kind != groups.list[rank[0]].kind);
                    let (lo, hi) = (slot_span(&units, &groups, rank[0]).0, slot_span(&units, &groups, *rank.last().unwrap()).1);
                    assert!(lo > min.x + SIDE_MARGIN - 1.0 && hi < max.x - SIDE_MARGIN + 1.0);
                    println!(
                        "line debug {per_team}/{size} team {team}: {} regiments, line {:.1} m{}",
                        rank.len(),
                        hi - lo,
                        if mixed { ", mixed kinds" } else { "" }
                    );
                }
            }
        }
    }

    /// Off Line Debug the deployment lays its blocks at an even pitch
    /// along each rank: every anchor matches that formula, bit for bit.
    #[test]
    fn grassland_anchors_keep_the_even_pitch_layout() {
        let setups = [
            (100_000, 1000, frac_comp(100), [25, 25, 25, 25]),
            (10_000, 500, [6, 5, 5, 4], [3, 7, 4, 6]),
        ];
        for (per_team, size, player, enemy) in setups {
            let (terrain, _, groups) = spawn(MapKind::Grassland, per_team, size, player, enemy);
            let n_regs = per_team / size;
            let cols = ((size as f32 * 2.2).sqrt().ceil() as usize).max(1);
            let rows = size.div_ceil(cols);
            let gap = crate::formation::widest_gap();
            let (block_w, block_d) = (cols as f32 * gap, rows as f32 * gap);
            let (field_min, field_max) = army_field(&terrain);
            let usable_w = (field_max.x - field_min.x) - 2.0 * SIDE_MARGIN;
            let (centre, army_gap) = army_lines(terrain.kind);
            let usable_d = (field_max.y - EDGE_MARGIN - army_gap / 2.0 - centre)
                .min(-field_min.y - EDGE_MARGIN - army_gap / 2.0 + centre);
            let max_ranks = (((usable_d - block_d) / (block_d + REG_GAP)).floor() as usize + 1).max(1);
            let per_rank = ((usable_w / (block_w + REG_GAP)).floor() as usize).max(n_regs.div_ceil(max_ranks)).max(1);
            let pitch_x = (usable_w / per_rank as f32).min(block_w + REG_GAP);
            let mut g = 0;
            for (team, comp) in [(0u8, player), (1, enemy)] {
                let n: usize = comp.iter().sum();
                let dir: f32 = if team == 0 { -1.0 } else { 1.0 };
                for r in 0..n {
                    let (rank, file) = (r / per_rank, r % per_rank);
                    let in_rank = per_rank.min(n - rank * per_rank);
                    let x0 = (file as f32 - (in_rank - 1) as f32 / 2.0) * pitch_x;
                    let z = centre + dir * (army_gap / 2.0 + block_d / 2.0 + rank as f32 * (block_d + REG_GAP));
                    assert_eq!(groups.list[g].anchor, Vec2::new(x0, z), "regiment {g}");
                    g += 1;
                }
            }
            assert_eq!(g, groups.list.len());
            let rank = ranks(&groups, 0).remove(0);
            let lo = groups.list[rank[0]].anchor.x - block_w * 0.5;
            let hi = groups.list[*rank.last().unwrap()].anchor.x + block_w * 0.5;
            println!("grassland {per_team}/{size}: {} regiments a rank, line {:.1} m, gap {:.1} m", rank.len(), hi - lo, pitch_x - block_w);
        }
    }
}
