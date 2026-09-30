//! A battle's starting setup as a file: the map, the army and unit sizes,
//! and where every unit stands. Begin Battle writes the setup it releases
//! to `last_setup.yaml` in the config folder. `FL_SETUP=<file>` starts
//! battles from one: the map and armies come from the file, the unit
//! picker is skipped, and every unit stands as saved while the deployment
//! waits for Begin Battle. The Demo scenario (`FL_DEMO=1`) does the same
//! with a built-in setup (`demo_setup`) for the player's side only. A unit
//! may carry one scripted order, which `run_setup_script` gives once the
//! battle has begun and its cue comes.

use std::path::PathBuf;
use std::sync::OnceLock;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::formation::{FormShape, FormSpacing};
use crate::game_state::{BattleConfig, Deployment, EnemyComp, GameState, Scenario};
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
    /// The player's deployment zone reaches the map's sides instead of
    /// stopping at the side margins.
    #[serde(default)]
    pub open_sides: bool,
    pub units: Vec<Placement>,
}

/// Metres between the map's side and the deployment zone of a setup with
/// open sides: the soldiers at the end of a line stand just inside the
/// map, with no corridor past them.
pub const OPEN_SIDE_MARGIN: f32 = 2.0;

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
    /// An order the unit gives itself once the battle has begun; none for
    /// a unit that waits for the player's orders.
    #[serde(default)]
    pub script: Option<Scripted>,
}

/// A scripted order: when it is given, and what it is.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct Scripted {
    pub when: When,
    pub then: Then,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum When {
    /// Seconds after the battle begins, on the game clock.
    After(f32),
    /// When the nearest enemy unit's centre comes within this many metres
    /// of the middle of the unit's front rank.
    EnemyWithin(f32),
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Then {
    /// Attack the nearest steady enemy unit.
    Charge,
    /// March straight ahead this many metres.
    Advance(f32),
}

impl BattleSetup {
    pub fn capture(config: &BattleConfig, groups: &Groups) -> Self {
        Self {
            map: config.map,
            units_per_team: config.units_per_team,
            reg_size: config.reg_size,
            ai_enabled: config.ai_enabled,
            open_sides: from_env().is_some_and(|s| s.open_sides),
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
                    script: None,
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
    /// battle starts exactly as saved. Returns, for each saved unit, the
    /// spawned unit it went to; places nothing and returns none when the
    /// spawned armies are not the saved ones (the army was changed in the
    /// menu after loading).
    pub fn place(&self, groups: &mut Groups, units: &mut Units, terrain: &Terrain) -> Option<Vec<usize>> {
        let listed = |team: u8| self.lists_team(team);
        if groups.list.iter().filter(|g| listed(g.team)).count() != self.units.len() {
            return None;
        }
        let mut taken = vec![false; self.units.len()];
        let mut pairs = Vec::with_capacity(self.units.len());
        for (g, gd) in groups.list.iter().enumerate().filter(|(_, gd)| listed(gd.team)) {
            let s = (0..self.units.len())
                .find(|&s| !taken[s] && self.units[s].team == gd.team && self.units[s].kind == gd.kind)?;
            taken[s] = true;
            pairs.push((g, s));
        }
        let mut placed = vec![0; self.units.len()];
        for (g, s) in pairs {
            placed[s] = g;
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
        Some(placed)
    }
}

/// The setup a battle of `scenario` starts from: a normal battle's is the
/// `FL_SETUP` file, if any; the Demo scenario's is the demo setup.
pub fn for_battle(scenario: Scenario) -> Option<&'static BattleSetup> {
    match scenario {
        Scenario::Normal => file_setup(),
        Scenario::Demo => Some(demo()),
        _ => None,
    }
}

/// The setup the launch environment asks for (`FL_SETUP` or `FL_DEMO`):
/// the map built at launch and the menu's first options come from it.
pub fn from_env() -> Option<&'static BattleSetup> {
    for_battle(Scenario::from_env())
}

fn demo() -> &'static BattleSetup {
    static DEMO: OnceLock<BattleSetup> = OnceLock::new();
    DEMO.get_or_init(demo_setup)
}

/// The setup `FL_SETUP` names, read once. A file that cannot be read or
/// parsed is logged and counts as none.
fn file_setup() -> Option<&'static BattleSetup> {
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

/// Soldiers per unit in the demo battle.
const DEMO_UNIT: usize = 1000;
/// Seconds after the demo battle begins when line 2 charges and the reserve
/// moves up.
const DEMO_ORDERS_AFTER: f32 = 20.0;
/// Line 1 counter-charges when an enemy unit's centre comes this close to
/// its front: the enemy's front rank is then about 25 m off, the last
/// seconds before contact.
const DEMO_COUNTER_WITHIN: f32 = 40.0;

/// The demo battle: 200k on the grassland with the AI on, the enemy as the
/// spawner deploys it, and the player's 100 units in a defence after the
/// Flemish at Courtrai (1302), whose second line was there to plug breaks
/// in the first. Units are 21 files wide and 48 ranks deep, except the
/// archer columns, which keep a shallow block so the archers behind them
/// stay in bow range:
///
/// - Line 1: 26 touching units on hold, the front rank on the deployment
///   zone's front edge and the end files at the map's sides (the demo
///   opens the zone's sides), so the front has no gap to push through and
///   no corridor round its ends. From each end: 2 shallow Knights (an
///   archer column), 2 Knights, 8 Spearmen; in the centre 2 shallow
///   Spearmen (the centre archer column). Each counter-charges the
///   nearest enemy in the last seconds before contact.
/// - A Knight line as wide as the centre column, just in front of it and
///   out past the zone's edge, so the centre's Spearmen hold. It
///   counter-charges with line 1.
/// - Archers: 2 behind each of the three archer columns, 5 m back, which
///   take the enemy's first blows for them. Their front rank stands within
///   bow range of the enemy's starting line, and they loft over line 1.
///   The centre pair shoots the volleys a camera following the front from
///   the right flank sees in the middle of the field.
/// - Line 2: 33 units 10 m behind line 1: 11 Knights on each outer side,
///   11 Men-at-Arms in the centre. They charge the nearest enemy 20 s after
///   the battle begins.
/// - Reserve: a row of 3 Men-at-Arms, 12 Spearmen, 3 Men-at-Arms and a row
///   of 16 Men-at-Arms. As line 2 charges they march up to 30 m behind
///   line 1's back rank.
///
/// Normal spacing throughout.
fn demo_setup() -> BattleSetup {
    use crate::formation::BASE_SPACING as P;
    use crate::regiments::{EDGE_MARGIN, army_gap};
    use crate::unit_types::{KIND_ARCHER, KIND_HEAVY, KIND_LIGHT, KIND_SPEAR};

    /// Files of every deep unit: 48 ranks at 1000 men, 70% of the 67 the
    /// first demo layout used.
    const DEEP_FILES: u32 = 21;
    /// Deep units in line 1 on each side of the centre column.
    const DEEP_PER_SIDE: usize = 10;
    /// Line 2's units.
    const LINE2: usize = 33;
    /// Metres between one row's back rank and the next row's front rank.
    const GAP: f32 = 10.0;
    /// Metres from line 1's back rank to the reserve's front rank once it
    /// has moved up.
    const RESERVE_BEHIND: f32 = 30.0;

    let half = crate::terrain::HALF_EXTENTS;
    let x_max = half.x - OPEN_SIDE_MARGIN;
    let front = -army_gap() * 0.5 - 1.0;
    // Depth of a unit's block, front rank to back rank. Units `files * P`
    // apart touch with no gap.
    let depth = |files: u32| (DEMO_UNIT.div_ceil(files as usize) - 1) as f32 * P;
    let w = DEEP_FILES as f32 * P;
    let d = depth(DEEP_FILES);
    // The six archer-column units (two per wing, two in the centre) share
    // the width the deep units leave.
    let col_files = ((2.0 * x_max - 2.0 * DEEP_PER_SIDE as f32 * w) / (6.0 * P)).floor() as u32;
    let cw = col_files as f32 * P;
    let cd = depth(col_files);

    let mut units = Vec::with_capacity(100);
    let mut push = |kind: u8, x: f32, z: f32, files: u32, hold: bool, script: Option<Scripted>| {
        units.push(Placement {
            team: PLAYER_TEAM,
            kind,
            x,
            z,
            facing: 0.0,
            files,
            shape: FormShape::Rect,
            spacing: FormSpacing::Normal,
            hold,
            fire_at_will: true,
            skirmish: false,
            script,
        });
    };
    let counter = Some(Scripted { when: When::EnemyWithin(DEMO_COUNTER_WITHIN), then: Then::Charge });

    // Line 1, from the centre out on each side: a centre-column unit, the
    // deep units, two wing-column units. The front rank is 1 m inside the
    // zone's front edge.
    let deep_x = |i: usize| cw + (i as f32 + 0.5) * w;
    let wing_x = |k: usize| cw + DEEP_PER_SIDE as f32 * w + (k as f32 + 0.5) * cw;
    let columns = [0.5 * cw, wing_x(0), wing_x(1)];
    for side in [-1.0_f32, 1.0] {
        let centre_kind = [KIND_SPEAR, KIND_HEAVY, KIND_HEAVY];
        for (&x, kind) in columns.iter().zip(centre_kind) {
            push(kind, side * x, front - cd * 0.5, col_files, true, counter);
        }
        for i in 0..DEEP_PER_SIDE {
            let kind = if i >= DEEP_PER_SIDE - 2 { KIND_HEAVY } else { KIND_SPEAR };
            push(kind, side * deep_x(i), front - d * 0.5, DEEP_FILES, true, counter);
        }
    }
    // The Knight line in front of the centre column: as wide as it, and
    // thin, its back rank one pitch ahead of line 1's front rank.
    let line_files = (2.0 * cw / P).floor() as u32;
    push(KIND_HEAVY, 0.0, front + P + depth(line_files) * 0.5, line_files, true, counter);
    // Archers behind the three archer columns.
    for side in [-1.0_f32, 1.0] {
        for &x in &columns {
            push(KIND_ARCHER, side * x, front - cd * 1.5 - 5.0, col_files, false, None);
        }
    }
    // Line 2, clear of the archers: it starts deeper than their back rank.
    let z2 = front - d - GAP - d * 0.5;
    let x_of = |i: usize, n: usize| (i as f32 - (n - 1) as f32 * 0.5) * w;
    let charge = Some(Scripted { when: When::After(DEMO_ORDERS_AFTER), then: Then::Charge });
    for j in 0..LINE2 {
        let kind = if j.min(LINE2 - 1 - j) < 11 { KIND_HEAVY } else { KIND_LIGHT };
        push(kind, x_of(j, LINE2), z2, DEEP_FILES, false, charge);
    }
    // The reserve rows, which move up together as line 2 charges.
    let z3 = z2 - d - GAP;
    let z4 = z3 - d - GAP;
    let advance = (front - d - RESERVE_BEHIND) - (z3 + d * 0.5);
    let move_up = Some(Scripted { when: When::After(DEMO_ORDERS_AFTER), then: Then::Advance(advance) });
    for i in 0..18 {
        let kind = if i.min(17 - i) < 3 { KIND_LIGHT } else { KIND_SPEAR };
        push(kind, x_of(i, 18), z3, DEEP_FILES, false, move_up);
    }
    for i in 0..16 {
        push(KIND_LIGHT, x_of(i, 16), z4, DEEP_FILES, false, move_up);
    }
    debug_assert!(z4 - d * 0.5 >= -half.y + EDGE_MARGIN, "the demo rows overrun the zone's depth");

    BattleSetup {
        map: MapKind::Grassland,
        units_per_team: 100 * DEMO_UNIT,
        reg_size: DEMO_UNIT,
        ai_enabled: true,
        open_sides: true,
        units,
    }
}

/// The scripted orders of the battle's setup still to give, by unit
/// index, and the time the battle began on the game clock.
#[derive(Resource, Default)]
pub struct SetupScript {
    pending: Vec<(usize, Scripted)>,
    begun: Option<f32>,
}

impl SetupScript {
    /// The scripted orders of `setup`'s units, placed as `place` returned.
    pub fn new(setup: &BattleSetup, placed: &[usize]) -> Self {
        let pending = setup.units.iter().zip(placed).filter_map(|(p, &g)| Some((g, p.script?))).collect();
        Self { pending, begun: None }
    }
}

/// Give each scripted order once its cue comes. The clock starts when the
/// deployment ends and stops while the game is paused. A unit that has
/// died or broken drops its order.
fn run_setup_script(
    time: Res<Time>,
    deploy: Res<Deployment>,
    mut script: ResMut<SetupScript>,
    mut groups: ResMut<Groups>,
) {
    if deploy.active || script.pending.is_empty() {
        return;
    }
    let now = time.elapsed_secs();
    let since = now - *script.begun.get_or_insert(now);
    let mut due = Vec::new();
    {
        let groups = &*groups;
        script.pending.retain(|&(g, order)| {
            let gd = &groups.list[g];
            if gd.count == 0 || gd.state.is_broken() {
                return false;
            }
            let ready = match order.when {
                When::After(t) => since >= t,
                When::EnemyWithin(r) => {
                    nearest_enemy(groups, gd.team, front_of(gd)).is_some_and(|(_, d2)| d2 <= r * r)
                }
            };
            if ready {
                due.push((g, order.then));
            }
            !ready
        });
    }
    for (g, then) in due {
        let gd = &groups.list[g];
        match then {
            Then::Charge => {
                if let Some((target, _)) = nearest_enemy(&groups, gd.team, front_of(gd)) {
                    crate::orders::attack_regiments(&mut groups, &[g], target);
                }
            }
            Then::Advance(metres) => {
                let dest = gd.centroid + crate::formation::facing_dir(gd.facing) * metres;
                crate::orders::order_regiments(&mut groups, &[g], dest);
            }
        }
    }
}

/// The middle of a unit's front rank.
fn front_of(gd: &crate::orders::GroupData) -> Vec2 {
    let files = gd.files.max(1) as usize;
    let ranks = gd.count.max(1).div_ceil(files);
    let half_depth = (ranks - 1) as f32 * gd.spacing.pitch().y * 0.5;
    gd.centroid + crate::formation::facing_dir(gd.facing) * half_depth
}

/// The steady enemy unit of `team`'s foes whose centre is nearest `at`,
/// with its squared distance.
fn nearest_enemy(groups: &Groups, team: u8, at: Vec2) -> Option<(u32, f32)> {
    groups
        .list
        .iter()
        .enumerate()
        .filter(|(_, e)| e.team != team && e.count > 0 && !e.state.is_broken())
        .map(|(i, e)| (i as u32, e.centroid.distance_squared(at)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

pub struct BattleSetupPlugin;

impl Plugin for BattleSetupPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SetupScript>()
            .add_systems(Update, run_setup_script.run_if(in_state(GameState::Battle)));
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
        assert!(loaded.place(&mut groups2, &mut units2, &terrain).is_some());

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
        assert!(saved.place(&mut groups2, &mut units2, &terrain).is_none());
        assert_eq!(units2.pos.to_vec(), before);
    }

    /// The demo army: 100 units of the planned kinds, every block inside the
    /// player's deployment zone with its sides opened to the map's, and no
    /// two blocks overlapping.
    #[test]
    fn demo_setup_fits_the_deployment_zone() {
        use crate::unit_types::{KIND_ARCHER, KIND_HEAVY, KIND_LIGHT, KIND_SPEAR};
        let setup = demo_setup();
        assert_eq!(setup.units.len(), 100);
        let cued = |when: When, then: fn(Then) -> bool| {
            setup.units.iter().filter(|u| u.script.is_some_and(|s| s.when == when && then(s.then))).count()
        };
        assert_eq!(cued(When::After(DEMO_ORDERS_AFTER), |t| t == Then::Charge), 33, "line 2 charges");
        assert_eq!(cued(When::After(DEMO_ORDERS_AFTER), |t| matches!(t, Then::Advance(_))), 34, "the reserve moves up");
        assert_eq!(cued(When::EnemyWithin(DEMO_COUNTER_WITHIN), |t| t == Then::Charge), 27, "line 1 counter-charges");
        let mut kinds = [0; NUM_KINDS];
        for p in &setup.units {
            kinds[p.kind as usize] += 1;
        }
        assert_eq!((kinds[KIND_HEAVY as usize], kinds[KIND_LIGHT as usize]), (31, 33));
        assert_eq!((kinds[KIND_SPEAR as usize], kinds[KIND_ARCHER as usize]), (30, 6));

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
        assert!(setup.open_sides);
        let mut zone = crate::orders::deploy_zone_for(&build_terrain(MapKind::Grassland), crate::regiments::army_gap());
        zone.0.x = -crate::terrain::HALF_EXTENTS.x + OPEN_SIDE_MARGIN;
        zone.1.x = crate::terrain::HALF_EXTENTS.x - OPEN_SIDE_MARGIN;
        // Only the Knight line in front of the centre may stand out past the
        // zone's front edge, and only a little.
        let mut past_front = 0;
        for (i, (lo, hi)) in boxes.iter().enumerate() {
            if hi.y > zone.1.y {
                past_front += 1;
                assert!(hi.y - zone.1.y < 15.0, "unit {i} stands {} m past the front edge", hi.y - zone.1.y);
            }
            let hi = Vec2::new(hi.x, hi.y.min(zone.1.y));
            assert!(lo.cmpge(zone.0).all() && hi.cmple(zone.1).all(), "unit {i} leaves the zone: {lo} {hi}");
        }
        assert_eq!(past_front, 1);
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
        assert!(setup.place(&mut groups, &mut units, &terrain).is_some());
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

    /// A world with the script system: one unit of ours at (0, -50) facing
    /// +Z, enemies at the given points, and `order` scripted for our unit.
    fn scripted_world(enemies: &[Vec2], order: Scripted) -> (World, impl FnMut(&mut World, f32) + use<>) {
        use crate::orders::GroupData;
        let mut world = World::new();
        let mut groups = Groups::default();
        let unit = |team, at: Vec2| GroupData::new(team, crate::unit_types::KIND_HEAVY, at, 100);
        groups.list.push(unit(0, Vec2::new(0.0, -50.0)));
        for &e in enemies {
            groups.list.push(unit(1, e));
        }
        world.insert_resource(groups);
        world.insert_resource(Deployment { active: false });
        world.insert_resource(SetupScript { pending: vec![(0, order)], begun: None });
        world.insert_resource(Time::<()>::default());
        let mut system = IntoSystem::into_system(run_setup_script);
        system.initialize(&mut world);
        let run = move |world: &mut World, secs: f32| {
            world.resource_mut::<Time>().advance_by(std::time::Duration::from_secs_f32(secs));
            system.run((), world).unwrap();
        };
        (world, run)
    }

    fn order_of(world: &World) -> Option<crate::orders::Order> {
        world.resource::<Groups>().list[0].order
    }

    /// A timed charge waits out the deployment and its time on the game
    /// clock, then attacks the nearest steady enemy unit.
    #[test]
    fn a_timed_charge_attacks_the_nearest_enemy_on_time() {
        use crate::orders::Order;
        let charge = Scripted { when: When::After(20.0), then: Then::Charge };
        let (mut world, mut run) = scripted_world(&[Vec2::new(300.0, 50.0), Vec2::new(20.0, 40.0)], charge);
        world.resource_mut::<Deployment>().active = true;
        run(&mut world, 30.0);
        assert!(order_of(&world).is_none(), "no order while deploying");
        world.resource_mut::<Deployment>().active = false;
        run(&mut world, 1.0);
        run(&mut world, 19.0);
        assert!(order_of(&world).is_none(), "19 s into the battle");
        run(&mut world, 1.5);
        assert!(matches!(order_of(&world), Some(Order::Attack(2))), "the nearer enemy, 20.5 s in");
    }

    /// A counter-charge waits until an enemy unit's centre comes within its
    /// distance of the unit's front rank.
    #[test]
    fn a_counter_charge_waits_for_the_enemy_to_close() {
        use crate::orders::Order;
        let counter = Scripted { when: When::EnemyWithin(40.0), then: Then::Charge };
        let (mut world, mut run) = scripted_world(&[Vec2::new(0.0, 100.0)], counter);
        run(&mut world, 1.0);
        assert!(order_of(&world).is_none(), "the enemy is far off");
        let front = front_of(&world.resource::<Groups>().list[0]);
        world.resource_mut::<Groups>().list[1].centroid = front + Vec2::new(0.0, 39.0);
        run(&mut world, 0.1);
        assert!(matches!(order_of(&world), Some(Order::Attack(1))), "the enemy is 39 m off");
    }

    /// An advance marches the unit straight ahead by its distance.
    #[test]
    fn an_advance_marches_straight_ahead() {
        use crate::orders::Order;
        let advance = Scripted { when: When::After(0.0), then: Then::Advance(55.0) };
        let (mut world, mut run) = scripted_world(&[Vec2::new(0.0, 300.0)], advance);
        run(&mut world, 0.1);
        let Some(Order::Move(dest)) = order_of(&world) else { panic!("no move order") };
        assert!(dest.distance(Vec2::new(0.0, 5.0)) < 1e-3, "moved to {dest}");
    }
}
