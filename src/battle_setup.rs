//! A battle's starting setup as a file: the map, the army and unit sizes,
//! and where every unit stands. Begin Battle writes the setup it releases
//! to `last_setup.yaml` in the config folder. `FL_SETUP=<file>` starts
//! battles from one: the map and armies come from the file, the unit
//! picker is skipped, and every unit stands as saved while the deployment
//! waits for Begin Battle.

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

    /// The battle options this setup was saved with. The enemy's army
    /// becomes hand-picked counts, so the spawner builds exactly the saved
    /// units instead of rolling a style.
    pub fn apply_config(&self, config: &mut BattleConfig) {
        config.map = self.map;
        config.units_per_team = self.units_per_team;
        config.reg_size = self.reg_size;
        config.ai_enabled = self.ai_enabled;
        config.player_regs = self.counts(PLAYER_TEAM);
        config.enemy = EnemyComp::Manual(self.counts(1 - PLAYER_TEAM));
    }

    /// Stand every spawned unit where the setup has it: the k-th spawned
    /// unit of a team and kind takes the k-th saved unit of that team and
    /// kind. A unit whose formation already matches (never moved in
    /// deployment) keeps its spawn positions, scatter included, so the
    /// battle starts exactly as saved. Places nothing and returns false
    /// when the spawned armies are not the saved ones (the army was
    /// changed in the menu after loading).
    pub fn place(&self, groups: &mut Groups, units: &mut Units, terrain: &Terrain) -> bool {
        if groups.list.len() != self.units.len() {
            return false;
        }
        let mut taken = vec![false; self.units.len()];
        let mut pairs = Vec::with_capacity(self.units.len());
        for (g, gd) in groups.list.iter().enumerate() {
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

/// The setup `FL_SETUP` names, read once. A file that cannot be read or
/// parsed is logged and counts as none.
pub fn from_env() -> Option<&'static BattleSetup> {
    static SETUP: OnceLock<Option<BattleSetup>> = OnceLock::new();
    SETUP
        .get_or_init(|| {
            let path = std::env::var("FL_SETUP").ok()?;
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
}
