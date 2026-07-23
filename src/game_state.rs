use bevy::prelude::*;
use bevy::time::common_conditions::paused as time_paused;

use crate::ai::BattleOutcome;
use crate::combat::CombatStats;
use crate::movement::DirTestStats;
use crate::orders::{Groups, Selection};
use crate::render_units::Corpses;
use crate::terrain::Terrain;
use crate::units::Units;

#[derive(States, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GameState {
    Menu,
    Battle,
    Results,
}

impl Default for GameState {
    fn default() -> Self {
        GameState::Menu
    }
}

fn scripts_active() -> bool {
    [
        "FL_TEST_FRONT",
        "FL_TEST_ORDERS",
        "FL_TEST_SURROUND",
        "FL_TEST_ROUT",
        "FL_TEST_FORM",
        "FL_TEST_DIR",
        "FL_TEST_CHARGE",
        "FL_TEST_PILE",
        "FL_TEST_JOIN",
        "FL_TEST_ROUTPASS",
        "FL_ARENA",
    ]
    .iter()
    .any(|k| std::env::var(k).is_ok())
}

#[derive(SystemSet, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SimSet;

#[derive(SystemSet, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BattleInputSet;

#[derive(Component)]
struct MenuRoot;

#[derive(Component)]
enum MenuButton {
    StartBattle,
}

#[derive(Component)]
struct PauseRoot;

#[derive(Component)]
enum PauseButton {
    Resume,
    QuitToMenu,
}

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
        app.init_state::<GameState>()
            .configure_sets(
                FixedUpdate,
                SimSet.run_if(in_state(GameState::Battle)),
            )
            .configure_sets(
                Update,
                BattleInputSet.run_if(
                    in_state(GameState::Battle).and_then(not(time_paused)),
                ),
            )
            .add_systems(OnEnter(GameState::Menu), spawn_menu)
            .add_systems(OnEnter(GameState::Battle), setup_battle)
            .add_systems(OnEnter(GameState::Results), spawn_results)
            .add_systems(
                Update,
                (
                    menu_buttons.run_if(in_state(GameState::Menu)),
                    (toggle_pause, pause_buttons, transition_to_results)
                        .run_if(in_state(GameState::Battle)),
                    results_buttons.run_if(in_state(GameState::Results)),
                    button_hover_style,
                ),
            );
    }
}

const TEXT_COLOR: Color = Color::srgb(0.92, 0.92, 0.85);
const DIM_TEXT_COLOR: Color = Color::srgb(0.55, 0.55, 0.50);
const PANEL_BG: Color = Color::srgba(0.05, 0.06, 0.08, 0.92);
const BTN_NORMAL: Color = Color::srgba(0.15, 0.16, 0.20, 0.92);
const BTN_HOVER: Color = Color::srgba(0.25, 0.27, 0.32, 0.95);
const BTN_PRESSED: Color = Color::srgba(0.10, 0.11, 0.14, 0.95);

fn fullscreen_overlay() -> Node {
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

// ── Menu ──

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
                Text::new("FRONTLINE"),
                TextFont {
                    font_size: FontSize::Px(56.0),
                    ..default()
                },
                TextColor(TEXT_COLOR),
                Node {
                    margin: UiRect::bottom(Val::Px(40.0)),
                    ..default()
                },
            ));
            p.spawn((
                Text::new("A Medieval Total War style mass battle prototype"),
                TextFont {
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(DIM_TEXT_COLOR),
                Node {
                    margin: UiRect::bottom(Val::Px(40.0)),
                    ..default()
                },
            ));
            p.spawn((
                Button,
                button_node(),
                BackgroundColor(BTN_NORMAL),
                MenuButton::StartBattle,
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new("Start Battle"),
                    TextFont {
                        font_size: FontSize::Px(20.0),
                        ..default()
                    },
                    TextColor(TEXT_COLOR),
                ));
            });
            p.spawn((
                Text::new("v0.1.0"),
                TextFont {
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(DIM_TEXT_COLOR),
                Node {
                    margin: UiRect::top(Val::Px(40.0)),
                    ..default()
                },
            ));
        });
}

fn menu_buttons(
    query: Query<(&Interaction, &MenuButton), Changed<Interaction>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut next: ResMut<NextState<GameState>>,
    mut auto: Local<bool>,
) {
    if !*auto && scripts_active() {
        *auto = true;
        next.set(GameState::Battle);
        return;
    }
    if keys.just_pressed(KeyCode::Enter) || keys.just_pressed(KeyCode::Space) {
        next.set(GameState::Battle);
        return;
    }
    for (interaction, btn) in &query {
        if *interaction == Interaction::Pressed {
            match btn {
                MenuButton::StartBattle => next.set(GameState::Battle),
            }
        }
    }
}

// ── Battle lifecycle ──

pub fn setup_battle(
    mut units: ResMut<Units>,
    terrain: Res<Terrain>,
    mut groups: ResMut<Groups>,
    mut stats: ResMut<CombatStats>,
    mut selection: ResMut<Selection>,
    mut outcome: ResMut<BattleOutcome>,
    mut corpses: ResMut<Corpses>,
    mut dir_stats: ResMut<DirTestStats>,
    mut virt_time: ResMut<Time<Virtual>>,
) {
    *units = Units::default();
    *stats = CombatStats::default();
    *dir_stats = DirTestStats::default();
    *selection = Selection::default();
    outcome.0 = None;
    corpses.clear();
    virt_time.unpause();
    crate::regiments::do_spawn_battle(&mut units, &terrain, &mut groups);
    info!("battle started");
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
            p.spawn((
                Button,
                button_node(),
                BackgroundColor(BTN_NORMAL),
                PauseButton::Resume,
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new("Resume"),
                    TextFont {
                        font_size: FontSize::Px(20.0),
                        ..default()
                    },
                    TextColor(TEXT_COLOR),
                ));
            });
            p.spawn((
                Button,
                button_node(),
                BackgroundColor(BTN_NORMAL),
                PauseButton::QuitToMenu,
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new("Quit to Menu"),
                    TextFont {
                        font_size: FontSize::Px(20.0),
                        ..default()
                    },
                    TextColor(TEXT_COLOR),
                ));
            });
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
        match btn {
            PauseButton::Resume => {
                virt_time.unpause();
                for e in &overlay {
                    commands.entity(e).despawn();
                }
            }
            PauseButton::QuitToMenu => {
                virt_time.unpause();
                for e in &overlay {
                    commands.entity(e).despawn();
                }
                next.set(GameState::Menu);
            }
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
            p.spawn((
                Button,
                button_node(),
                BackgroundColor(BTN_NORMAL),
                ResultsButton::PlayAgain,
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new("Play Again"),
                    TextFont {
                        font_size: FontSize::Px(20.0),
                        ..default()
                    },
                    TextColor(TEXT_COLOR),
                ));
            });
            p.spawn((
                Button,
                button_node(),
                BackgroundColor(BTN_NORMAL),
                ResultsButton::MainMenu,
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new("Main Menu"),
                    TextFont {
                        font_size: FontSize::Px(20.0),
                        ..default()
                    },
                    TextColor(TEXT_COLOR),
                ));
            });
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

fn button_hover_style(
    mut query: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<Button>)>,
) {
    for (interaction, mut bg) in &mut query {
        *bg = match interaction {
            Interaction::Pressed => BackgroundColor(BTN_PRESSED),
            Interaction::Hovered => BackgroundColor(BTN_HOVER),
            Interaction::None => BackgroundColor(BTN_NORMAL),
        };
    }
}
