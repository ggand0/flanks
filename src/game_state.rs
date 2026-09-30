use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::prelude::*;
use bevy::time::common_conditions::paused as time_paused;

use crate::ai::BattleOutcome;
use crate::combat::CombatStats;
use crate::sim::damage::DirTestStats;
use crate::orders::{Groups, Selection};
use crate::render_units::Corpses;
use crate::terrain::{MapChanged, MapKind, Terrain};
use crate::units::Units;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Scenario {
    #[default]
    Normal,
    Surround,
    Rout,
    Dir,
    Arena,
    Charge,
    Pile,
    /// The pile-on as a wide line at ease hit by a narrow block
    /// (work/scripts/pile-wide.sh).
    PileWide,
    /// The pile-on with two attackers (work/scripts/pile-two.sh).
    PileTwo,
    Join,
    Routpass,
    Archery,
    /// A new attack target mid-fight (regiments.rs `spawn_retarget_test`).
    Retarget,
    /// An empty battlefield with a free camera and no armies, for looking
    /// at the map, the scenery and the light (`FL_SCENE=1`).
    Scene,
}

/// The env var of every scripted scenario, as the scripts launch them.
const SCENARIO_ENVS: [&str; 11] = [
    "FL_TEST_SURROUND",
    "FL_TEST_ROUT",
    "FL_TEST_DIR",
    "FL_ARENA",
    "FL_TEST_CHARGE",
    "FL_TEST_PILE",
    "FL_TEST_JOIN",
    "FL_TEST_ROUTPASS",
    "FL_TEST_ARCHERY",
    "FL_TEST_RETARGET",
    "FL_SCENE",
];

impl Scenario {
    const ALL: &[Scenario] = &[
        Self::Normal,
        Self::Surround,
        Self::Rout,
        Self::Dir,
        Self::Arena,
        Self::Charge,
        Self::Pile,
        Self::PileWide,
        Self::PileTwo,
        Self::Join,
        Self::Routpass,
        Self::Archery,
        Self::Retarget,
        Self::Scene,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Surround => "Surround",
            Self::Rout => "Rout",
            Self::Dir => "Dir Defense",
            Self::Arena => "Arena",
            Self::Charge => "Charge",
            Self::Pile => "Pile-on",
            Self::PileWide => "Wide vs Narrow",
            Self::PileTwo => "Two on One",
            Self::Join => "Join Fight",
            Self::Routpass => "Rout Pass",
            Self::Archery => "Archery",
            Self::Retarget => "Retarget",
            Self::Scene => "Scene",
        }
    }

    /// The env var this scenario sets: its test logging and the systems
    /// that stand down for scripts key off it. The pile-on variants share
    /// FL_TEST_PILE and differ in their `PileSetup`.
    fn env_key(self) -> Option<&'static str> {
        match self {
            Self::Normal => None,
            Self::Surround => Some("FL_TEST_SURROUND"),
            Self::Rout => Some("FL_TEST_ROUT"),
            Self::Dir => Some("FL_TEST_DIR"),
            Self::Arena => Some("FL_ARENA"),
            Self::Charge => Some("FL_TEST_CHARGE"),
            Self::Pile | Self::PileWide | Self::PileTwo => Some("FL_TEST_PILE"),
            Self::Join => Some("FL_TEST_JOIN"),
            Self::Routpass => Some("FL_TEST_ROUTPASS"),
            Self::Archery => Some("FL_TEST_ARCHERY"),
            Self::Retarget => Some("FL_TEST_RETARGET"),
            Self::Scene => Some("FL_SCENE"),
        }
    }

    fn from_env() -> Self {
        Self::ALL
            .iter()
            .copied()
            .find(|s| s.env_key().is_some_and(|k| std::env::var(k).is_ok()))
            .unwrap_or(Self::Normal)
    }
}

/// Enemy composition source, set on the Select Units screen: a random
/// army style rolled at each battle start, a chosen style (still
/// jittered a little per battle), or hand-picked counts.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EnemyComp {
    Random,
    Style(usize),
    Manual([usize; crate::unit_types::NUM_KINDS]),
}

#[derive(Resource)]
pub struct BattleConfig {
    pub units_per_team: usize,
    pub reg_size: usize,
    pub ai_enabled: bool,
    pub map: MapKind,
    pub scenario: Scenario,
    /// Player composition from the Select Units screen: regiments per
    /// kind (KIND_* indexed), summing to at most `n_slots()`.
    pub player_regs: [usize; crate::unit_types::NUM_KINDS],
    pub enemy: EnemyComp,
}

impl BattleConfig {
    /// The regiment budget: how many regiment slots the chosen army
    /// size buys at the current regiment size.
    pub fn n_slots(&self) -> usize {
        (self.units_per_team / self.reg_size).max(1)
    }
}

impl Default for BattleConfig {
    fn default() -> Self {
        let units_per_team = crate::util::env_or("FL_UNITS", 100_000);
        let reg_size = crate::util::env_or("FL_REG_SIZE", 1000_usize).max(50);
        Self {
            player_regs: crate::regiments::frac_comp((units_per_team / reg_size).max(1)),
            enemy: EnemyComp::Random,
            units_per_team,
            reg_size,
            ai_enabled: !std::env::var("FL_AI").is_ok_and(|v| v == "0"),
            map: MapKind::from_env(),
            scenario: Scenario::from_env(),
        }
    }
}

#[derive(States, Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum GameState {
    #[default]
    Menu,
    UnitSelect,
    Battle,
    Results,
}

/// TW-style pre-battle deployment: the battle opens frozen while the
/// player places regiments inside their zone, then a Begin Battle press
/// releases the sim. A plain resource, not a state: `setup_battle`
/// decides it and the flip must gate `SimSet` the same frame, with no
/// transition-schedule lag for a stale phase to slip a tick through.
#[derive(Resource, Default)]
pub struct Deployment {
    pub active: bool,
}

pub fn deploying(d: Res<Deployment>) -> bool {
    d.active
}

/// Any scripted scenario or test battery owns the battle: automatic
/// order sources (ai.rs) stand down. Deliberately NOT cached — menu
/// scenario buttons change the env between battles (sync_scenario_env).
pub fn scripts_active() -> bool {
    Scenario::from_env() != Scenario::Normal
        || std::env::var("FL_TEST_FRONT").is_ok()
        || std::env::var("FL_TEST_ORDERS").is_ok()
        || std::env::var("FL_TEST_FORM").is_ok()
}

#[derive(SystemSet, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SimSet;

#[derive(SystemSet, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BattleInputSet;

/// Map pointer input (lasso, hover pick), inside `BattleInputSet`.
#[derive(SystemSet, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MapInputSet;

/// HUD input (unit cards, control buttons), after `MapInputSet`: the
/// HUD's over-UI guards read hover state that lags synthetic same-frame
/// move+click input by one frame, and if a click ever reaches both
/// paths the HUD must win.
#[derive(SystemSet, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct HudInputSet;

/// Buttons that paint their own state-dependent BackgroundColor: the
/// global hover styler below leaves them alone.
#[derive(Component)]
pub struct CustomStyled;

#[derive(Component)]
struct MenuRoot;

#[derive(Component)]
enum MenuButton {
    StartBattle,
    TestBattles,
    Quit,
}

/// The Test Battles panel over the menu. While it is open the menu's
/// keys (Enter/Space start a battle) stand down.
#[derive(Component)]
struct TestBattlesRoot;

#[derive(Component)]
struct TestBattlesBack;

fn test_battles_closed(open: Query<(), With<TestBattlesRoot>>) -> bool {
    open.is_empty()
}

/// One button of a segmented menu row, with its index into that row's
/// values (`ARMY_SIZES`, `MapKind::ALL`, `AI_VALUES`).
#[derive(Component, Clone, Copy)]
enum Segment {
    Army(usize),
    Map(usize),
    Ai(usize),
}

/// The AI row: index 0 turns the enemy AI on.
const AI_VALUES: [&str; 2] = ["On", "Off"];

#[derive(Component)]
struct DebugButton(Scenario);

#[derive(Component)]
struct PauseRoot;

#[derive(Component)]
enum PauseButton {
    Resume,
    QuitToMenu,
}

#[derive(Component)]
struct DeployRoot;

#[derive(Component)]
struct BeginBattleButton;

#[derive(Component)]
struct ResultsRoot;

#[derive(Component)]
enum ResultsButton {
    PlayAgain,
    MainMenu,
}

pub struct GameShellPlugin;

impl Plugin for GameShellPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BattleConfig>()
            .init_resource::<Deployment>()
            .init_state::<GameState>()
            .configure_sets(
                FixedUpdate,
                SimSet.run_if(
                    in_state(GameState::Battle).and_then(not(deploying)),
                ),
            )
            .configure_sets(
                Update,
                BattleInputSet.run_if(
                    in_state(GameState::Battle).and_then(not(time_paused)),
                ),
            )
            .configure_sets(
                Update,
                (MapInputSet, HudInputSet).chain().in_set(BattleInputSet),
            )
            .add_systems(OnEnter(GameState::Menu), spawn_menu)
            .add_systems(OnEnter(GameState::Battle), setup_battle)
            .add_systems(OnEnter(GameState::Results), spawn_results)
            .add_systems(
                Update,
                (
                    // Menu/pause input yields to the settings modal:
                    // its backdrop blocks picking, and the gate keeps
                    // the keyboard shortcuts (Enter/Space/ESC) from
                    // acting behind it.
                    (menu_buttons, menu_segments, segment_style)
                        .run_if(
                            in_state(GameState::Menu)
                                .and_then(crate::settings::settings_closed)
                                .and_then(test_battles_closed),
                        ),
                    (debug_scenario_buttons, close_test_battles).run_if(in_state(GameState::Menu)),
                    (toggle_pause, pause_buttons).run_if(
                        in_state(GameState::Battle)
                            .and_then(crate::settings::settings_closed),
                    ),
                    begin_battle.run_if(
                        in_state(GameState::Battle)
                            .and_then(deploying)
                            .and_then(not(time_paused))
                            .and_then(crate::settings::settings_closed),
                    ),
                    transition_to_results.run_if(in_state(GameState::Battle)),
                    results_buttons.run_if(in_state(GameState::Results)),
                    button_hover_style,
                ),
            );
    }
}

pub const TEXT_COLOR: Color = Color::srgb(0.92, 0.92, 0.85);
pub const DIM_TEXT_COLOR: Color = Color::srgb(0.55, 0.55, 0.50);
pub const PANEL_BG: Color = Color::srgba(0.05, 0.06, 0.08, 0.92);
pub const BTN_NORMAL: Color = Color::srgba(0.15, 0.16, 0.20, 0.92);
pub const BTN_HOVER: Color = Color::srgba(0.25, 0.27, 0.32, 0.95);
pub const BTN_PRESSED: Color = Color::srgba(0.10, 0.11, 0.14, 0.95);
/// The current value of a segmented row or chip group.
pub const BTN_ACTIVE: Color = Color::srgba(0.22, 0.38, 0.62, 0.95);

pub fn fullscreen_overlay() -> Node {
    Node {
        position_type: PositionType::Absolute,
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        flex_direction: FlexDirection::Column,
        ..default()
    }
}

fn button_node() -> Node {
    Node {
        padding: UiRect::axes(Val::Px(32.0), Val::Px(12.0)),
        margin: UiRect::all(Val::Px(8.0)),
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        ..default()
    }
}

/// The standard big menu button (menu, pause, results screens).
pub fn spawn_text_button(p: &mut ChildSpawnerCommands, label: &str, marker: impl Bundle) {
    p.spawn((Button, button_node(), BackgroundColor(BTN_NORMAL), marker))
        .with_children(|b| {
            b.spawn((
                Text::new(label),
                TextFont {
                    font_size: FontSize::Px(20.0),
                    ..default()
                },
                TextColor(TEXT_COLOR),
            ));
        });
}

// ── Menu ──

const ARMY_SIZES: &[(usize, &str)] = &[
    (5_000, "10k"),
    (10_000, "20k"),
    (25_000, "50k"),
    (50_000, "100k"),
    (100_000, "200k"),
];

/// Set the army size per team. The regiment size and the slot budget
/// follow it, so both compositions reset (stale counts could overflow
/// the new slot total); picking the current size keeps them.
fn set_army_size(config: &mut BattleConfig, per_team: usize) {
    if config.units_per_team == per_team {
        return;
    }
    config.units_per_team = per_team;
    // Smaller armies field smaller units, so a 10k battle still has 25
    // units a side to maneuver.
    config.reg_size = if per_team <= 5_000 {
        200
    } else if per_team <= 10_000 {
        500
    } else {
        1000
    };
    config.player_regs = crate::regiments::frac_comp(config.n_slots());
    config.enemy = EnemyComp::Random;
}

fn spawn_menu(mut commands: Commands) {
    commands
        .spawn((
            fullscreen_overlay(),
            BackgroundColor(PANEL_BG),
            GlobalZIndex(10),
            DespawnOnExit(GameState::Menu),
            MenuRoot,
        ))
        .with_children(|p| {
            p.spawn((
                Text::new("FLANKS"),
                TextFont {
                    font_size: FontSize::Px(56.0),
                    ..default()
                },
                TextColor(TEXT_COLOR),
                Node {
                    margin: UiRect::bottom(Val::Px(12.0)),
                    ..default()
                },
            ));
            p.spawn((
                Text::new("Hold the line. Turn the flank."),
                TextFont {
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(DIM_TEXT_COLOR),
                Node {
                    margin: UiRect::bottom(Val::Px(28.0)),
                    ..default()
                },
            ));

            // Options panel
            p.spawn(Node {
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(16.0)),
                margin: UiRect::bottom(Val::Px(20.0)),
                ..default()
            })
            .with_children(|opts| {
                spawn_segment_row(
                    opts,
                    "Army",
                    ARMY_SIZES.iter().map(|(_, label)| *label),
                    Segment::Army,
                );
                spawn_segment_row(opts, "Map", MapKind::ALL.iter().map(|m| m.label()), Segment::Map);
                spawn_segment_row(opts, "AI", AI_VALUES.into_iter(), Segment::Ai);
            });

            spawn_text_button(p, "Start Battle", MenuButton::StartBattle);
            spawn_text_button(p, "Settings", crate::settings::OpenSettingsButton);
            spawn_text_button(p, "Quit", MenuButton::Quit);
            // The test scenarios sit behind one small button.
            p.spawn((
                Button,
                Node {
                    padding: UiRect::axes(Val::Px(14.0), Val::Px(5.0)),
                    margin: UiRect::top(Val::Px(18.0)),
                    ..default()
                },
                BackgroundColor(BTN_NORMAL),
                MenuButton::TestBattles,
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new("Test Battles"),
                    TextFont {
                        font_size: FontSize::Px(13.0),
                        ..default()
                    },
                    TextColor(DIM_TEXT_COLOR),
                ));
            });

            p.spawn((
                Text::new(concat!("v", env!("CARGO_PKG_VERSION"))),
                TextFont {
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(DIM_TEXT_COLOR),
                Node {
                    margin: UiRect::top(Val::Px(20.0)),
                    ..default()
                },
            ));
        });
}

/// A menu option shown as a row of buttons, one per value, the current
/// one lit (`segment_style`).
fn spawn_segment_row<'a>(
    p: &mut ChildSpawnerCommands,
    label: &str,
    values: impl Iterator<Item = &'a str>,
    segment: fn(usize) -> Segment,
) {
    p.spawn(option_row_node()).with_children(|row| {
        spawn_option_label(row, label);
        row.spawn(Node {
            column_gap: Val::Px(2.0),
            ..default()
        })
        .with_children(|segments| {
            for (i, value) in values.enumerate() {
                segments
                    .spawn((
                        Button,
                        Node {
                            padding: UiRect::axes(Val::Px(12.0), Val::Px(6.0)),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        BackgroundColor(BTN_NORMAL),
                        CustomStyled,
                        segment(i),
                    ))
                    .with_children(|b| {
                        spawn_option_text(b, value);
                    });
            }
        });
    });
}

fn option_row_node() -> Node {
    Node {
        flex_direction: FlexDirection::Row,
        align_items: AlignItems::Center,
        margin: UiRect::vertical(Val::Px(4.0)),
        ..default()
    }
}

fn spawn_option_label(row: &mut ChildSpawnerCommands, label: &str) {
    row.spawn((
        Text::new(format!("{label}:")),
        TextFont {
            font_size: FontSize::Px(15.0),
            ..default()
        },
        TextColor(DIM_TEXT_COLOR),
        Node {
            width: Val::Px(80.0),
            ..default()
        },
    ));
}

fn spawn_option_text(button: &mut ChildSpawnerCommands, value: &str) {
    button.spawn((
        Text::new(value.to_string()),
        TextFont {
            font_size: FontSize::Px(15.0),
            ..default()
        },
        TextColor(TEXT_COLOR),
    ));
}

/// Where Start Battle goes: normal battles pass through the Select
/// Units screen; scripted scenarios and FL_DEPLOY=0 skip deployment and
/// the picker both and drop straight into the fight.
fn start_target() -> GameState {
    if scripts_active() || crate::util::env_or("FL_DEPLOY", 1_u32) == 0 {
        GameState::Battle
    } else {
        GameState::UnitSelect
    }
}

/// Start a normal battle from the menu. A previous debug-scenario
/// battle leaves its env var set until setup_battle syncs, so clear
/// the scenario envs now: the picker decision must see the chosen
/// scenario, not the stale one. Launch-only battery envs
/// (FL_TEST_FRONT and kin) are not scenario envs and still skip the
/// picker on purpose.
fn start_normal_battle(config: &mut BattleConfig, next: &mut NextState<GameState>) {
    config.scenario = Scenario::Normal;
    sync_scenario_env(Scenario::Normal);
    next.set(start_target());
}

fn menu_buttons(
    mut commands: Commands,
    query: Query<(&Interaction, &MenuButton), Changed<Interaction>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut config: ResMut<BattleConfig>,
    mut next: ResMut<NextState<GameState>>,
    mut exit: MessageWriter<AppExit>,
    mut auto: Local<bool>,
) {
    if !*auto && scripts_active() {
        *auto = true;
        next.set(GameState::Battle);
        return;
    }
    // FL_AUTOSTART=1: start a normal battle without a key press (with
    // FL_DEPLOY=0 it skips the picker and deployment too), for measured
    // runs of the real game with the AI on.
    if !*auto && std::env::var("FL_AUTOSTART").is_ok() {
        *auto = true;
        start_normal_battle(&mut config, &mut next);
        return;
    }
    if keys.just_pressed(KeyCode::Enter) || keys.just_pressed(KeyCode::Space) {
        start_normal_battle(&mut config, &mut next);
        return;
    }
    for (interaction, btn) in &query {
        if *interaction == Interaction::Pressed {
            match btn {
                MenuButton::StartBattle => {
                    start_normal_battle(&mut config, &mut next);
                }
                MenuButton::TestBattles => spawn_test_battles(&mut commands),
                MenuButton::Quit => {
                    exit.write(AppExit::Success);
                }
            }
        }
    }
}

/// A segment pressed: its row takes that value. A new map rebuilds the
/// terrain behind the menu.
fn menu_segments(
    query: Query<(&Interaction, &Segment), Changed<Interaction>>,
    mut config: ResMut<BattleConfig>,
    mut maps: MessageWriter<MapChanged>,
) {
    for (interaction, segment) in &query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match *segment {
            Segment::Army(i) => set_army_size(&mut config, ARMY_SIZES[i].0),
            Segment::Map(i) => {
                let map = MapKind::ALL[i];
                if config.map != map {
                    config.map = map;
                    maps.write(MapChanged(map));
                }
            }
            Segment::Ai(i) => config.ai_enabled = i == 0,
        }
    }
}

/// Segment colours: the current value lit, the others with the usual
/// hover and press shades.
fn segment_style(
    mut query: Query<(&Interaction, &Segment, &mut BackgroundColor)>,
    config: Res<BattleConfig>,
) {
    for (interaction, segment, mut bg) in &mut query {
        let current = match *segment {
            Segment::Army(i) => ARMY_SIZES[i].0 == config.units_per_team,
            Segment::Map(i) => MapKind::ALL[i] == config.map,
            Segment::Ai(i) => (i == 0) == config.ai_enabled,
        };
        bg.0 = match interaction {
            _ if current => BTN_ACTIVE,
            Interaction::Pressed => BTN_PRESSED,
            Interaction::Hovered => BTN_HOVER,
            Interaction::None => BTN_NORMAL,
        };
    }
}

/// The test scenarios in a panel over the menu, one button each.
fn spawn_test_battles(commands: &mut Commands) {
    commands
        .spawn((
            fullscreen_overlay(),
            BackgroundColor(Color::srgba(0.01, 0.02, 0.03, 0.6)),
            // Swallow clicks so the menu behind never reacts.
            bevy::ui::FocusPolicy::Block,
            Interaction::None,
            GlobalZIndex(30),
            DespawnOnExit(GameState::Menu),
            TestBattlesRoot,
        ))
        .with_children(|p| {
            p.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    padding: UiRect::axes(Val::Px(28.0), Val::Px(20.0)),
                    ..default()
                },
                BackgroundColor(PANEL_BG),
            ))
            .with_children(|panel| {
                panel.spawn((
                    Text::new("Test Battles"),
                    TextFont {
                        font_size: FontSize::Px(24.0),
                        ..default()
                    },
                    TextColor(TEXT_COLOR),
                    Node {
                        margin: UiRect::bottom(Val::Px(14.0)),
                        ..default()
                    },
                ));
                panel
                    .spawn(Node {
                        flex_direction: FlexDirection::Row,
                        flex_wrap: FlexWrap::Wrap,
                        justify_content: JustifyContent::Center,
                        max_width: Val::Px(480.0),
                        ..default()
                    })
                    .with_children(|grid| {
                        for &scenario in &Scenario::ALL[1..] {
                            grid.spawn((
                                Button,
                                Node {
                                    width: Val::Px(146.0),
                                    padding: UiRect::vertical(Val::Px(7.0)),
                                    margin: UiRect::all(Val::Px(4.0)),
                                    justify_content: JustifyContent::Center,
                                    ..default()
                                },
                                BackgroundColor(BTN_NORMAL),
                                DebugButton(scenario),
                            ))
                            .with_children(|b| {
                                b.spawn((
                                    Text::new(scenario.label()),
                                    TextFont {
                                        font_size: FontSize::Px(14.0),
                                        ..default()
                                    },
                                    TextColor(TEXT_COLOR),
                                ));
                            });
                        }
                    });
                panel
                    .spawn((
                        Button,
                        Node {
                            padding: UiRect::axes(Val::Px(28.0), Val::Px(8.0)),
                            margin: UiRect::top(Val::Px(16.0)),
                            ..default()
                        },
                        BackgroundColor(BTN_NORMAL),
                        TestBattlesBack,
                    ))
                    .with_children(|b| {
                        b.spawn((
                            Text::new("Back"),
                            TextFont {
                                font_size: FontSize::Px(16.0),
                                ..default()
                            },
                            TextColor(TEXT_COLOR),
                        ));
                    });
            });
        });
}

/// Back or ESC closes the Test Battles panel.
fn close_test_battles(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    back: Query<&Interaction, (Changed<Interaction>, With<TestBattlesBack>)>,
    open: Query<Entity, With<TestBattlesRoot>>,
) {
    if open.is_empty() {
        return;
    }
    if keys.just_pressed(KeyCode::Escape) || back.iter().any(|i| *i == Interaction::Pressed) {
        for e in &open {
            commands.entity(e).despawn();
        }
    }
}

fn debug_scenario_buttons(
    query: Query<(&Interaction, &DebugButton), Changed<Interaction>>,
    mut config: ResMut<BattleConfig>,
    mut next: ResMut<NextState<GameState>>,
) {
    for (interaction, btn) in &query {
        if *interaction == Interaction::Pressed {
            config.scenario = btn.0;
            next.set(GameState::Battle);
        }
    }
}

fn sync_scenario_env(scenario: Scenario) {
    for key in SCENARIO_ENVS {
        unsafe {
            if scenario.env_key() == Some(key) {
                std::env::set_var(key, "1");
            } else {
                std::env::remove_var(key);
            }
        }
    }
}

// ── Battle lifecycle ──

#[allow(clippy::too_many_arguments)]
pub fn setup_battle(
    mut commands: Commands,
    mut units: ResMut<Units>,
    terrain: Res<Terrain>,
    mut groups: ResMut<Groups>,
    mut stats: ResMut<CombatStats>,
    mut selection: ResMut<Selection>,
    mut outcome: ResMut<BattleOutcome>,
    mut corpses: ResMut<Corpses>,
    mut dir_stats: ResMut<DirTestStats>,
    mut virt_time: ResMut<Time<Virtual>>,
    mut deploy: ResMut<Deployment>,
    config: Res<BattleConfig>,
    (mut arrows, mut arrow_spawns, mut stuck): (
        ResMut<crate::arrows::Arrows>,
        ResMut<crate::arrows::ArrowSpawns>,
        ResMut<crate::arrows::StuckArrows>,
    ),
) {
    // A new world: any sim tick job still computing on the old one is
    // recognized as stale by the generation and dropped.
    let generation = units.generation.wrapping_add(1);
    *units = Units::default();
    units.generation = generation;
    *stats = CombatStats::default();
    *dir_stats = DirTestStats::default();
    *selection = Selection::default();
    outcome.0 = None;
    corpses.clear();
    crate::arrows::reset(&mut arrows, &mut arrow_spawns, &mut stuck);
    virt_time.unpause();
    sync_scenario_env(config.scenario);
    crate::regiments::do_spawn_battle(&mut units, &terrain, &mut groups, &config);
    // Scripted scenarios and test batteries start fighting immediately;
    // a normal battle opens in deployment. FL_DEPLOY=0 skips it.
    deploy.active = config.scenario == Scenario::Normal
        && !scripts_active()
        && crate::util::env_or("FL_DEPLOY", 1_u32) != 0;
    if deploy.active {
        spawn_deploy_ui(&mut commands);
        info!("deployment phase: place your regiments, then begin the battle");
    }
    info!(
        "battle started: {} per team, map {}, scenario {}, AI {}",
        config.units_per_team,
        config.map.label(),
        config.scenario.label(),
        if config.ai_enabled { "on" } else { "off" },
    );
}

fn spawn_deploy_ui(commands: &mut Commands) {
    // One transparent full-screen column: banner pinned up top, Begin
    // Battle pinned above the card bar. Plain nodes carry no
    // Interaction, so map picking under it is untouched.
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                justify_content: JustifyContent::SpaceBetween,
                // The banner starts under the balance of power bar.
                padding: UiRect::top(Val::Px(36.0)).with_bottom(Val::Px(150.0)),
                ..default()
            },
            GlobalZIndex(5),
            DespawnOnExit(GameState::Battle),
            DeployRoot,
        ))
        .with_children(|p| {
            p.spawn(Node {
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                ..default()
            })
            .with_children(|banner| {
                banner.spawn((
                    Text::new("Deployment"),
                    TextFont {
                        font_size: FontSize::Px(24.0),
                        ..default()
                    },
                    TextColor(TEXT_COLOR),
                ));
                banner.spawn((
                    Text::new("Place your regiments inside the gold zone"),
                    TextFont {
                        font_size: FontSize::Px(13.0),
                        ..default()
                    },
                    TextColor(DIM_TEXT_COLOR),
                ));
            });
            spawn_text_button(p, "Begin Battle (Enter)", BeginBattleButton);
        });
}

/// Begin Battle button or Enter: release the sim and drop the deploy UI.
fn begin_battle(
    mut commands: Commands,
    mut deploy: ResMut<Deployment>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Query<&Interaction, (Changed<Interaction>, With<BeginBattleButton>)>,
    roots: Query<Entity, With<DeployRoot>>,
) {
    let clicked = buttons.iter().any(|i| *i == Interaction::Pressed);
    if !clicked && !keys.just_pressed(KeyCode::Enter) {
        return;
    }
    deploy.active = false;
    for e in &roots {
        commands.entity(e).despawn();
    }
    info!("deployment done: battle begins");
}

fn transition_to_results(
    outcome: Res<BattleOutcome>,
    time: Res<Time<Real>>,
    mut next: ResMut<NextState<GameState>>,
    mut delay: Local<Option<f32>>,
) {
    if outcome.0.is_some() {
        let elapsed = delay.get_or_insert(0.0);
        *elapsed += time.delta_secs();
        if *elapsed >= 3.0 {
            *delay = None;
            next.set(GameState::Results);
        }
    } else {
        *delay = None;
    }
}

// ── Pause ──

fn toggle_pause(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mut virt_time: ResMut<Time<Virtual>>,
    overlay: Query<Entity, With<PauseRoot>>,
) {
    if !keys.just_pressed(KeyCode::Escape) {
        return;
    }
    if virt_time.is_paused() {
        virt_time.unpause();
        for e in &overlay {
            commands.entity(e).despawn();
        }
    } else {
        virt_time.pause();
        spawn_pause_overlay(&mut commands);
    }
}

fn spawn_pause_overlay(commands: &mut Commands) {
    commands
        .spawn((
            fullscreen_overlay(),
            BackgroundColor(Color::srgba(0.02, 0.03, 0.05, 0.75)),
            GlobalZIndex(20),
            PauseRoot,
        ))
        .with_children(|p| {
            p.spawn((
                Text::new("PAUSED"),
                TextFont {
                    font_size: FontSize::Px(48.0),
                    ..default()
                },
                TextColor(TEXT_COLOR),
                Node {
                    margin: UiRect::bottom(Val::Px(32.0)),
                    ..default()
                },
            ));
            spawn_text_button(p, "Resume", PauseButton::Resume);
            spawn_text_button(p, "Settings", crate::settings::OpenSettingsButton);
            spawn_text_button(p, "Quit to Menu", PauseButton::QuitToMenu);
        });
}

fn pause_buttons(
    mut commands: Commands,
    query: Query<(&Interaction, &PauseButton), Changed<Interaction>>,
    mut virt_time: ResMut<Time<Virtual>>,
    overlay: Query<Entity, With<PauseRoot>>,
    mut next: ResMut<NextState<GameState>>,
) {
    for (interaction, btn) in &query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        virt_time.unpause();
        for e in &overlay {
            commands.entity(e).despawn();
        }
        if matches!(btn, PauseButton::QuitToMenu) {
            next.set(GameState::Menu);
        }
    }
}

// ── Results ──

fn spawn_results(
    mut commands: Commands,
    outcome: Res<BattleOutcome>,
    stats: Res<CombatStats>,
) {
    let title = match outcome.0 {
        Some(0) => "VICTORY",
        Some(1) => "DEFEAT",
        _ => "MUTUAL DESTRUCTION",
    };
    let title_color = match outcome.0 {
        Some(0) => Color::srgb(0.4, 0.85, 0.5),
        Some(1) => Color::srgb(0.9, 0.3, 0.25),
        _ => Color::srgb(0.85, 0.75, 0.3),
    };

    commands
        .spawn((
            fullscreen_overlay(),
            BackgroundColor(PANEL_BG),
            GlobalZIndex(10),
            DespawnOnExit(GameState::Results),
            ResultsRoot,
        ))
        .with_children(|p| {
            p.spawn((
                Text::new(title),
                TextFont {
                    font_size: FontSize::Px(52.0),
                    ..default()
                },
                TextColor(title_color),
                Node {
                    margin: UiRect::bottom(Val::Px(32.0)),
                    ..default()
                },
            ));
            let summary = format!(
                "Blue:    {} alive    {} killed    {} fled\n\
                 Orange:  {} alive    {} killed    {} fled",
                stats.alive[0], stats.kills[0], stats.fled[0],
                stats.alive[1], stats.kills[1], stats.fled[1],
            );
            p.spawn((
                Text::new(summary),
                TextFont {
                    font_size: FontSize::Px(16.0),
                    ..default()
                },
                TextColor(TEXT_COLOR),
                Node {
                    margin: UiRect::bottom(Val::Px(32.0)),
                    ..default()
                },
            ));
            spawn_text_button(p, "Play Again", ResultsButton::PlayAgain);
            spawn_text_button(p, "Main Menu", ResultsButton::MainMenu);
        });
}

fn results_buttons(
    query: Query<(&Interaction, &ResultsButton), Changed<Interaction>>,
    mut next: ResMut<NextState<GameState>>,
) {
    for (interaction, btn) in &query {
        if *interaction == Interaction::Pressed {
            match btn {
                ResultsButton::PlayAgain => next.set(GameState::Battle),
                ResultsButton::MainMenu => next.set(GameState::Menu),
            }
        }
    }
}

// ── Shared button hover style ──

#[allow(clippy::type_complexity)]
fn button_hover_style(
    mut query: Query<
        (&Interaction, &mut BackgroundColor),
        (
            Changed<Interaction>,
            With<Button>,
            Without<CustomStyled>,
        ),
    >,
) {
    for (interaction, mut bg) in &mut query {
        *bg = match interaction {
            Interaction::Pressed => BackgroundColor(BTN_PRESSED),
            Interaction::Hovered => BackgroundColor(BTN_HOVER),
            Interaction::None => BackgroundColor(BTN_NORMAL),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every menu size splits into whole units, 10 to 100 a side, and the
    /// player's starting composition fills exactly those slots.
    #[test]
    fn army_sizes_split_into_whole_units() {
        for &(per_team, label) in ARMY_SIZES {
            let mut config = BattleConfig { units_per_team: 0, ..default() };
            set_army_size(&mut config, per_team);
            assert_eq!(per_team % config.reg_size, 0, "{label}: {per_team} by {}", config.reg_size);
            let n = config.n_slots();
            assert!((10..=100).contains(&n), "{label}: {n} units a side");
            assert_eq!(config.player_regs.iter().sum::<usize>(), n, "{label}: composition");
        }
    }

    #[test]
    fn unit_size_follows_army_size() {
        let expect = [(5_000, 200), (10_000, 500), (25_000, 1000), (50_000, 1000), (100_000, 1000)];
        assert_eq!(expect.len(), ARMY_SIZES.len());
        for (per_team, reg_size) in expect {
            let mut config = BattleConfig { units_per_team: 0, ..default() };
            set_army_size(&mut config, per_team);
            assert_eq!(config.reg_size, reg_size, "{per_team} a side");
        }
    }

    #[test]
    fn picking_the_current_size_keeps_the_picks() {
        let mut config = BattleConfig { units_per_team: 0, ..default() };
        set_army_size(&mut config, 5_000);
        config.player_regs = [25, 0, 0, 0];
        config.enemy = EnemyComp::Style(1);
        set_army_size(&mut config, 5_000);
        assert_eq!(config.player_regs, [25, 0, 0, 0]);
        assert!(matches!(config.enemy, EnemyComp::Style(1)));
        set_army_size(&mut config, 10_000);
        assert_eq!(config.player_regs.iter().sum::<usize>(), 20);
        assert!(matches!(config.enemy, EnemyComp::Random));
    }
}
