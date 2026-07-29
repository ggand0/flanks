use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::prelude::*;
use bevy::time::common_conditions::paused as time_paused;

use crate::ai::BattleOutcome;
use crate::combat::CombatStats;
use crate::movement::DirTestStats;
use crate::orders::{Groups, Selection};
use crate::render_units::Corpses;
use crate::terrain::Terrain;
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
    Join,
    Routpass,
    Archery,
}

impl Scenario {
    const ALL: &[Scenario] = &[
        Self::Normal,
        Self::Surround,
        Self::Rout,
        Self::Dir,
        Self::Arena,
        Self::Charge,
        Self::Pile,
        Self::Join,
        Self::Routpass,
        Self::Archery,
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
            Self::Join => "Join Fight",
            Self::Routpass => "Rout Pass",
            Self::Archery => "Archery",
        }
    }

    fn from_env() -> Self {
        if std::env::var("FL_TEST_SURROUND").is_ok() { return Self::Surround; }
        if std::env::var("FL_TEST_ROUT").is_ok() { return Self::Rout; }
        if std::env::var("FL_TEST_DIR").is_ok() { return Self::Dir; }
        if std::env::var("FL_ARENA").is_ok() { return Self::Arena; }
        if std::env::var("FL_TEST_CHARGE").is_ok() { return Self::Charge; }
        if std::env::var("FL_TEST_PILE").is_ok() { return Self::Pile; }
        if std::env::var("FL_TEST_JOIN").is_ok() { return Self::Join; }
        if std::env::var("FL_TEST_ROUTPASS").is_ok() { return Self::Routpass; }
        if std::env::var("FL_TEST_ARCHERY").is_ok() { return Self::Archery; }
        Self::Normal
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
    Quit,
}

#[derive(Component)]
enum OptionButton {
    ArmySize,
    Ai,
}

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
                    (menu_buttons, menu_option_buttons, debug_scenario_buttons)
                        .run_if(
                            in_state(GameState::Menu)
                                .and_then(crate::settings::settings_closed),
                        ),
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
    (10_000, "20k"),
    (25_000, "50k"),
    (50_000, "100k"),
    (100_000, "200k"),
];

fn army_size_label(per_team: usize) -> &'static str {
    ARMY_SIZES
        .iter()
        .find(|(n, _)| *n == per_team)
        .map(|(_, s)| *s)
        .unwrap_or("200k")
}

fn spawn_menu(mut commands: Commands, config: Res<BattleConfig>) {
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
                Text::new("Massed medieval battles, every soldier simulated"),
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
                spawn_option_row(
                    opts,
                    "Army",
                    army_size_label(config.units_per_team),
                    OptionButton::ArmySize,
                );
                spawn_option_row(
                    opts,
                    "AI",
                    if config.ai_enabled { "On" } else { "Off" },
                    OptionButton::Ai,
                );
            });

            spawn_text_button(p, "Start Battle", MenuButton::StartBattle);
            spawn_text_button(p, "Settings", crate::settings::OpenSettingsButton);
            spawn_text_button(p, "Quit", MenuButton::Quit);
            // Debug scenarios
            p.spawn((
                Text::new("Debug Scenarios"),
                TextFont {
                    font_size: FontSize::Px(13.0),
                    ..default()
                },
                TextColor(DIM_TEXT_COLOR),
                Node {
                    margin: UiRect::new(Val::Px(0.0), Val::Px(0.0), Val::Px(24.0), Val::Px(8.0)),
                    ..default()
                },
            ));
            p.spawn(Node {
                flex_direction: FlexDirection::Row,
                flex_wrap: FlexWrap::Wrap,
                justify_content: JustifyContent::Center,
                max_width: Val::Px(520.0),
                ..default()
            })
            .with_children(|row| {
                for &scenario in &Scenario::ALL[1..] {
                    row.spawn((
                        Button,
                        Node {
                            padding: UiRect::axes(Val::Px(12.0), Val::Px(5.0)),
                            margin: UiRect::all(Val::Px(3.0)),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        BackgroundColor(BTN_NORMAL),
                        DebugButton(scenario),
                    ))
                    .with_children(|b| {
                        b.spawn((
                            Text::new(scenario.label()),
                            TextFont {
                                font_size: FontSize::Px(12.0),
                                ..default()
                            },
                            TextColor(DIM_TEXT_COLOR),
                        ));
                    });
                }
            });

            p.spawn((
                Text::new("v0.1.0"),
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

fn spawn_option_row(p: &mut ChildSpawnerCommands, label: &str, value: &str, btn: OptionButton) {
    p.spawn(Node {
        flex_direction: FlexDirection::Row,
        align_items: AlignItems::Center,
        margin: UiRect::vertical(Val::Px(4.0)),
        ..default()
    })
    .with_children(|row| {
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
        row.spawn((
            Button,
            Node {
                padding: UiRect::axes(Val::Px(16.0), Val::Px(6.0)),
                min_width: Val::Px(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(BTN_NORMAL),
            btn,
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(value.to_string()),
                TextFont {
                    font_size: FontSize::Px(15.0),
                    ..default()
                },
                TextColor(TEXT_COLOR),
            ));
        });
    });
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

fn menu_buttons(
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
    if keys.just_pressed(KeyCode::Enter) || keys.just_pressed(KeyCode::Space) {
        config.scenario = Scenario::Normal;
        next.set(start_target());
        return;
    }
    for (interaction, btn) in &query {
        if *interaction == Interaction::Pressed {
            match btn {
                MenuButton::StartBattle => {
                    config.scenario = Scenario::Normal;
                    next.set(start_target());
                }
                MenuButton::Quit => {
                    exit.write(AppExit::Success);
                }
            }
        }
    }
}

fn menu_option_buttons(
    query: Query<(&Interaction, &OptionButton, &Children), Changed<Interaction>>,
    mut texts: Query<&mut Text>,
    mut config: ResMut<BattleConfig>,
) {
    for (interaction, opt, children) in &query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let new_label = match opt {
            OptionButton::ArmySize => {
                let idx = ARMY_SIZES
                    .iter()
                    .position(|(n, _)| *n == config.units_per_team)
                    .map(|i| (i + 1) % ARMY_SIZES.len())
                    .unwrap_or(0);
                config.units_per_team = ARMY_SIZES[idx].0;
                config.reg_size = if config.units_per_team <= 10_000 { 500 } else { 1000 };
                // The regiment budget changed: reset both compositions
                // (stale counts could overflow the new slot total).
                config.player_regs = crate::regiments::frac_comp(config.n_slots());
                config.enemy = EnemyComp::Random;
                ARMY_SIZES[idx].1
            }
            OptionButton::Ai => {
                config.ai_enabled = !config.ai_enabled;
                if config.ai_enabled { "On" } else { "Off" }
            }
        };
        for child in children.iter() {
            if let Ok(mut text) = texts.get_mut(child) {
                text.0 = new_label.to_string();
            }
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
    let vars = [
        ("FL_TEST_SURROUND", Scenario::Surround),
        ("FL_TEST_ROUT", Scenario::Rout),
        ("FL_TEST_DIR", Scenario::Dir),
        ("FL_ARENA", Scenario::Arena),
        ("FL_TEST_CHARGE", Scenario::Charge),
        ("FL_TEST_PILE", Scenario::Pile),
        ("FL_TEST_JOIN", Scenario::Join),
        ("FL_TEST_ROUTPASS", Scenario::Routpass),
        ("FL_TEST_ARCHERY", Scenario::Archery),
    ];
    for (key, s) in vars {
        unsafe {
            if scenario == s {
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
    *units = Units::default();
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
        "battle started: {} per team, scenario {}, AI {}",
        config.units_per_team,
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
                padding: UiRect::top(Val::Px(10.0)).with_bottom(Val::Px(150.0)),
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
