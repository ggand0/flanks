//! A battle's starting setup as a file: the map, the army and unit sizes,
//! and where every unit stands. Begin Battle writes the setup it releases
//! to `last_setup.yaml` in the config folder. `FL_SETUP=<file>` starts
//! battles from one: the map and armies come from the file, the unit
//! picker is skipped, and every unit stands as saved while the deployment
//! waits for Begin Battle. `FL_DEMO=1` does the same with a built-in
//! setup (`demo_setup`) for the player's side only.

use std::path::PathBuf;
use std::sync::OnceLock;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::formation::{FormShape, FormSpacing};
use crate::game_state::{BattleConfig, EnemyComp};
use crate::orders::{Groups, PLAYER_TEAM};
use crate::terrain::{MapKind, Terrain};
use crate::unit_types::NUM_KINDS;
use crate::units::Units;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct BattleSetup {
    pub map: MapKind,
    /// Soldiers per side.
    pub units_per_team: usize,
    /// Soldiers per unit.
    pub reg_size: usize,
    pub ai_enabled: bool,
    pub units: Vec<Placement>,
}

/// One unit as it stood when the battle began.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Placement {
    pub team: u8,
    /// Index into `unit_types::TYPES`.
    pub kind: u8,
    /// The formation's anchor on the ground, metres.
    pub x: f32,
    pub z: f32,
    /// Radians, the unit yaw convention (0 faces +Z).
    pub facing: f32,
    /// Soldiers per rank.
    pub files: u32,
    pub shape: FormShape,
    pub spacing: FormSpacing,
    pub hold: bool,
    pub fire_at_will: bool,
    pub skirmish: bool,
}

impl BattleSetup {
    pub fn capture(config: &BattleConfig, groups: &Groups) -> Self {
        Self {
            map: config.map,
            units_per_team: config.units_per_team,
            reg_size: config.reg_size,
            ai_enabled: config.ai_enabled,
            units: groups
                .list
                .iter()
                .map(|g| Placement {
                    team: g.team,
                    kind: g.kind,
                    x: g.anchor.x,
                    z: g.anchor.y,
                    facing: g.facing,
                    files: g.files,
                    shape: g.shape,
                    spacing: g.spacing,
                    hold: g.hold,
                    fire_at_will: g.fire_at_will,
                    skirmish: g.skirmish,
                })
                .collect(),
        }
    }

    /// Units of `team` per kind.
    fn counts(&self, team: u8) -> [usize; NUM_KINDS] {
        let mut counts = [0; NUM_KINDS];
        for p in self.units.iter().filter(|p| p.team == team) {
            counts[p.kind as usize] += 1;
        }
        counts
    }

    fn lists_team(&self, team: u8) -> bool {
        self.units.iter().any(|p| p.team == team)
    }

    /// The battle options this setup was saved with. A listed enemy army
    /// becomes hand-picked counts, so the spawner builds exactly the saved
    /// units instead of rolling a style; an unlisted one keeps the menu's.
    pub fn apply_config(&self, config: &mut BattleConfig) {
        config.map = self.map;
        config.units_per_team = self.units_per_team;
        config.reg_size = self.reg_size;
        config.ai_enabled = self.ai_enabled;
        config.player_regs = self.counts(PLAYER_TEAM);
        if self.lists_team(1 - PLAYER_TEAM) {
            config.enemy = EnemyComp::Manual(self.counts(1 - PLAYER_TEAM));
        }
    }

    /// Stand every spawned unit of the listed teams where the setup has
    /// it: the k-th spawned unit of a team and kind takes the k-th saved
    /// unit of that team and kind. A team the setup does not list stays as
    /// spawned. A unit whose formation already matches (never moved in
    /// deployment) keeps its spawn positions, scatter included, so the
    /// battle starts exactly as saved. Places nothing and returns false
    /// when the spawned armies are not the saved ones (the army was
    /// changed in the menu after loading).
    pub fn place(&self, groups: &mut Groups, units: &mut Units, terrain: &Terrain) -> bool {
        let listed = |team: u8| self.lists_team(team);
        if groups.list.iter().filter(|g| listed(g.team)).count() != self.units.len() {
            return false;
        }
        let mut taken = vec![false; self.units.len()];
        let mut pairs = Vec::with_capacity(self.units.len());
        for (g, gd) in groups.list.iter().enumerate().filter(|(_, gd)| listed(gd.team)) {
            let Some(s) = (0..self.units.len())
                .find(|&s| !taken[s] && self.units[s].team == gd.team && self.units[s].kind == gd.kind)
            else {
                return false;
            };
            taken[s] = true;
            pairs.push((g, s));
        }
        for (g, s) in pairs {
            let p = &self.units[s];
            let gd = &mut groups.list[g];
            gd.hold = p.hold;
            gd.fire_at_will = p.fire_at_will;
            gd.skirmish = p.skirmish;
            let anchor = Vec2::new(p.x, p.z);
            if (gd.anchor, gd.facing, gd.files, gd.shape, gd.spacing)
                != (anchor, p.facing, p.files, p.shape, p.spacing)
            {
                gd.anchor = anchor;
                gd.facing = p.facing;
                gd.files = p.files;
                gd.shape = p.shape;
                gd.spacing = p.spacing;
                gd.order = None;
                gd.auto_order = false;
                crate::formation::snap_to_slots(units, terrain, g as u32, gd);
            }
        }
        true
    }
}

/// The setup `FL_SETUP` names, else the demo setup when `FL_DEMO=1`, read
/// once. A file that cannot be read or parsed is logged and counts as none.
pub fn from_env() -> Option<&'static BattleSetup> {
    static SETUP: OnceLock<Option<BattleSetup>> = OnceLock::new();
    SETUP
        .get_or_init(|| {
            let Ok(path) = std::env::var("FL_SETUP") else {
                return (crate::util::env_or("FL_DEMO", 0_u32) != 0).then(|| {
                    info!("FL_DEMO: battles start from the demo setup");
                    demo_setup()
                });
            };
            let read = std::fs::read_to_string(&path).map_err(|e| e.to_string());
            match read.and_then(|text| serde_yaml_ng::from_str(&text).map_err(|e| e.to_string())) {
                Ok(setup) => {
                    info!("FL_SETUP: battles start from {path}");
                    Some(setup)
                }
                Err(e) => {
                    error!("FL_SETUP={path}: {e}");
                    None
                }
            }
        })
        .as_ref()
}

/// Soldiers per unit in the demo battle.
const DEMO_UNIT: usize = 1000;

/// The demo battle: 200k on the grassland with the AI on, the enemy as the
/// spawner deploys it, and the player's 100 units in a defence after the
/// Flemish at Courtrai (1302), whose second line was there to plug breaks
/// in the first:
///
/// - Line 1: 4 Men-at-Arms, 32 Spearmen, 4 Men-at-Arms, touching and on
///   hold, so the front has no gap for the enemy to push through and
///   never chases forward to open one.
/// - Line 2: 31 Knights 20 m behind, each centred on a join of line 1, to
///   meet whatever gets through.
/// - Line 3: 17 Men-at-Arms in reserve behind the centre.
/// - Flanks: 4 Men-at-Arms each side beside line 2, facing outward.
/// - Archers: 2 per wing beside line 1's ends, 10 m ahead of it, in
///   skirmish mode (they fall back when the enemy closes).
///
/// Normal spacing throughout. The files per unit are chosen so line 1
/// and the archers fill the deployment zone's width, and the three lines
/// fit its depth.
fn demo_setup() -> BattleSetup {
    use crate::formation::BASE_SPACING as P;
    use crate::regiments::{EDGE_MARGIN, SIDE_MARGIN, army_gap};
    use crate::unit_types::{KIND_ARCHER, KIND_HEAVY, KIND_LIGHT, KIND_SPEAR};
    use std::f32::consts::FRAC_PI_2;

    let half = crate::terrain::HALF_EXTENTS;
    let x_max = half.x - SIDE_MARGIN;
    let z_front = -army_gap() * 0.5;
    // Depth of a unit's block, front rank to back rank.
    let depth = |files: u32| (DEMO_UNIT.div_ceil(files as usize) - 1) as f32 * P;

    // Line 1's 40 units and the 4 archers across the zone's width. Units
    // `files * P` apart touch with no gap.
    let files = (2.0 * x_max / (44.0 * P)).floor() as u32;
    let w = files as f32 * P;
    let d = depth(files);

    let mut units = Vec::with_capacity(100);
    let mut push = |kind: u8, x: f32, z: f32, facing: f32, files: u32, hold: bool| {
        units.push(Placement {
            team: PLAYER_TEAM,
            kind,
            x,
            z,
            facing,
            files,
            shape: FormShape::Rect,
            spacing: FormSpacing::Normal,
            hold,
            fire_at_will: true,
            skirmish: kind == KIND_ARCHER,
        });
    };

    // Line 1: centres at (i - 19.5) w, so its joins fall on whole
    // multiples of w.
    let z1 = z_front - 12.0 - d * 0.5;
    for i in 0..40 {
        let kind = if (4..36).contains(&i) { KIND_SPEAR } else { KIND_LIGHT };
        push(kind, (i as f32 - 19.5) * w, z1, 0.0, files, true);
    }
    // Line 2: on the joins from -15 w to 15 w.
    let z2 = z1 - d - 20.0;
    for j in -15..=15 {
        push(KIND_HEAVY, j as f32 * w, z2, 0.0, files, false);
    }
    // Line 3, whose back rank is the deepest in the zone.
    let z3 = z2 - d - 20.0;
    debug_assert!(z3 - d * 0.5 >= -half.y + EDGE_MARGIN, "the demo lines overrun the zone's depth");
    for k in -8..=8 {
        push(KIND_LIGHT, k as f32 * w, z3, 0.0, files, false);
    }
    // Flanks: a column of 4 beside each end of line 2 (15.5 w out), facing
    // outward, so the block's depth runs along x.
    let x_flank = 15.5 * w + 10.0 + d * 0.5;
    for side in [-1.0_f32, 1.0] {
        for k in 0..4 {
            let z = z2 + d * 0.5 - (k as f32 + 0.5) * w;
            push(KIND_LIGHT, side * x_flank, z, side * FRAC_PI_2, files, false);
        }
    }
    // Archers: two across the strip between line 1's end (20 w out) and
    // the zone's side.
    let a_files = ((x_max - 20.0 * w) / (2.0 * P)).floor() as u32;
    let aw = a_files as f32 * P;
    let za = z_front - 2.0 - depth(a_files) * 0.5;
    for side in [-1.0_f32, 1.0] {
        for k in 0..2 {
            push(KIND_ARCHER, side * (20.0 * w + (k as f32 + 0.5) * aw), za, 0.0, a_files, false);
        }
    }

    BattleSetup {
        map: MapKind::Grassland,
        units_per_team: 100 * DEMO_UNIT,
        reg_size: DEMO_UNIT,
        ai_enabled: true,
        units,
    }
}

/// Where Begin Battle writes the setup it releases, next to the settings.
fn last_path() -> Option<PathBuf> {
    Some(dirs::config_dir()?.join("flanks").join("last_setup.yaml"))
}

pub fn save_last(setup: &BattleSetup) {
    let Some(path) = last_path() else {
        return;
    };
    let written = serde_yaml_ng::to_string(setup).map_err(|e| e.to_string()).and_then(|text| {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        std::fs::write(&path, text).map_err(|e| e.to_string())
    });
    match written {
        Ok(()) => info!("setup saved to {}", path.display()),
        Err(e) => warn!("setup not saved to {}: {e}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game_state::Scenario;
    use crate::terrain::build_terrain;

    /// A normal 10k battle on the grassland, freshly spawned.
    fn spawn(config: &BattleConfig, terrain: &Terrain) -> (Units, Groups) {
        let (mut units, mut groups) = (Units::default(), Groups::default());
        crate::regiments::do_spawn_battle(&mut units, terrain, &mut groups, config);
        (units, groups)
    }

    /// Rearrange a spawned army, save it through YAML, spawn a fresh
    /// battle from the file's options and place it: every soldier must
    /// stand exactly where he stood when the setup was saved.
    #[test]
    fn saved_setup_reloads_to_the_same_positions() {
        let terrain = build_terrain(MapKind::Grassland);
        let mut config = BattleConfig {
            units_per_team: 5_000,
            reg_size: 200,
            map: MapKind::Grassland,
            scenario: Scenario::Normal,
            enemy: EnemyComp::Style(1),
            ..default()
        };
        config.player_regs = crate::regiments::frac_comp(config.n_slots());
        let (mut units, mut groups) = spawn(&config, &terrain);

        // A deployment: shift, turn, widen and re-dress some units.
        for (g, gd) in groups.list.iter_mut().enumerate().filter(|(g, _)| g % 3 == 0) {
            gd.anchor += Vec2::new(12.0, -7.5);
            gd.facing += 0.3;
            gd.files = 30;
            gd.spacing = FormSpacing::Wall;
            gd.hold = true;
            crate::formation::snap_to_slots(&mut units, &terrain, g as u32, gd);
        }
        let saved = BattleSetup::capture(&config, &groups);
        let text = serde_yaml_ng::to_string(&saved).unwrap();
        let loaded: BattleSetup = serde_yaml_ng::from_str(&text).unwrap();
        assert_eq!(loaded, saved);

        let mut fresh = BattleConfig { enemy: EnemyComp::Random, ..default() };
        loaded.apply_config(&mut fresh);
        let (mut units2, mut groups2) = spawn(&fresh, &terrain);
        assert!(loaded.place(&mut groups2, &mut units2, &terrain));

        assert_eq!(units2.len(), units.len());
        for i in 0..units.len() {
            assert_eq!(units2.group[i], units.group[i], "soldier {i} unit");
            assert_eq!(units2.pos[i], units.pos[i], "soldier {i} position");
            assert_eq!(units2.yaw[i], units.yaw[i], "soldier {i} facing");
        }
        for (a, b) in groups.list.iter().zip(&groups2.list) {
            assert_eq!((a.anchor, a.facing, a.files), (b.anchor, b.facing, b.files));
            assert_eq!((a.spacing, a.hold), (b.spacing, b.hold));
        }
    }

    /// A setup for other armies places nothing.
    #[test]
    fn a_setup_for_other_armies_places_nothing() {
        let terrain = build_terrain(MapKind::Grassland);
        let mut config = BattleConfig { units_per_team: 5_000, reg_size: 200, ..default() };
        config.player_regs = crate::regiments::frac_comp(config.n_slots());
        config.enemy = EnemyComp::Style(1);
        let (_, groups) = spawn(&config, &terrain);
        let saved = BattleSetup::capture(&config, &groups);

        let mut other = BattleConfig { units_per_team: 10_000, reg_size: 500, ..default() };
        other.player_regs = crate::regiments::frac_comp(other.n_slots());
        let (mut units2, mut groups2) = spawn(&other, &terrain);
        let before = units2.pos.to_vec();
        assert!(!saved.place(&mut groups2, &mut units2, &terrain));
        assert_eq!(units2.pos.to_vec(), before);
    }

    /// The demo army: 100 units of the planned kinds, every block inside the
    /// player's deployment zone, and no two blocks overlapping.
    #[test]
    fn demo_setup_fits_the_deployment_zone() {
        use crate::unit_types::{KIND_ARCHER, KIND_HEAVY, KIND_LIGHT, KIND_SPEAR};
        let setup = demo_setup();
        assert_eq!(setup.units.len(), 100);
        let mut kinds = [0; NUM_KINDS];
        for p in &setup.units {
            kinds[p.kind as usize] += 1;
        }
        assert_eq!((kinds[KIND_HEAVY as usize], kinds[KIND_LIGHT as usize]), (31, 33));
        assert_eq!((kinds[KIND_SPEAR as usize], kinds[KIND_ARCHER as usize]), (32, 4));

        // Soldier-centre footprint of each block: files across, ranks deep,
        // turned a quarter for the outward-facing flanks.
        let p = crate::formation::BASE_SPACING;
        let boxes: Vec<(Vec2, Vec2)> = setup
            .units
            .iter()
            .map(|u| {
                let ranks = DEMO_UNIT.div_ceil(u.files as usize);
                let mut half = Vec2::new((u.files - 1) as f32, (ranks - 1) as f32) * p * 0.5;
                if u.facing.abs() > 1.0 {
                    half = Vec2::new(half.y, half.x);
                }
                let c = Vec2::new(u.x, u.z);
                (c - half, c + half)
            })
            .collect();
        let zone = crate::orders::deploy_zone_for(&build_terrain(MapKind::Grassland), crate::regiments::army_gap());
        for (i, (lo, hi)) in boxes.iter().enumerate() {
            assert!(lo.cmpge(zone.0).all() && hi.cmple(zone.1).all(), "unit {i} leaves the zone: {lo} {hi}");
        }
        for (i, a) in boxes.iter().enumerate() {
            for (j, b) in boxes.iter().enumerate().skip(i + 1) {
                let apart = a.1.x < b.0.x || b.1.x < a.0.x || a.1.y < b.0.y || b.1.y < a.0.y;
                assert!(apart, "units {i} and {j} overlap");
            }
        }
    }

    /// The demo setup lands on a spawned 200k battle: the player's side is
    /// placed, the enemy stays exactly as spawned.
    #[test]
    fn demo_setup_places_the_player_side_only() {
        let terrain = build_terrain(MapKind::Grassland);
        let setup = demo_setup();
        let mut config = BattleConfig { enemy: EnemyComp::Style(0), scenario: Scenario::Normal, ..default() };
        setup.apply_config(&mut config);
        assert!(matches!(config.enemy, EnemyComp::Style(0)));
        let (mut units, mut groups) = spawn(&config, &terrain);
        let enemy = |units: &Units| -> Vec<Vec3> {
            (0..units.len()).filter(|&i| units.team[i] != PLAYER_TEAM).map(|i| units.pos[i]).collect()
        };
        let before = enemy(&units);
        assert!(setup.place(&mut groups, &mut units, &terrain));
        assert_eq!(enemy(&units), before);
        // Placement pairs units by kind, not by list order: compare the
        // player's units and the setup's as sorted (kind, anchor) lists.
        let sorted = |mut v: Vec<(u8, f32, f32)>| {
            v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.total_cmp(&b.2)));
            v
        };
        let placed = groups.list.iter().filter(|g| g.team == PLAYER_TEAM).map(|g| (g.kind, g.anchor.x, g.anchor.y));
        let saved = setup.units.iter().map(|p| (p.kind, p.x, p.z));
        assert_eq!(sorted(placed.collect()), sorted(saved.collect()));
    }
}
