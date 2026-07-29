//! The Select Units screen: an M2TW-style pre-battle army picker.
//! Regiment slots are the budget (army size / regiment size); drag a
//! card from the unit roster into the army list, or click it, to add a
//! regiment, and click a placed card to remove it. Team arrows flip
//! between your army and the enemy's, which can also stay on Random: a
//! coherent army style rolled fresh at every battle start.

use bevy::picking::Pickable;
use bevy::picking::pointer::PointerButton;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::game_state::{
    BTN_NORMAL, BattleConfig, DIM_TEXT_COLOR, GameState, PANEL_BG, TEXT_COLOR, fullscreen_overlay,
    spawn_text_button,
};
use crate::unit_cards::KIND_FILL;
use crate::unit_types::{KIND_ARCHER, KIND_HEAVY, KIND_LIGHT, KIND_SPEAR, NUM_KINDS};

const GRID_BG: Color = Color::srgba(0.03, 0.04, 0.05, 0.85);
const CELL_EMPTY: Color = Color::srgba(0.09, 0.10, 0.13, 0.90);
const CARD_BG: Color = Color::srgba(0.10, 0.11, 0.14, 0.92);
const CARD_BG_HOVER: Color = Color::srgba(0.22, 0.24, 0.30, 0.95);

const CELL_W: f32 = 34.0;
const CELL_H: f32 = 42.0;
/// Army list pane width: 12 cells per row plus gaps and padding.
const GRID_W: f32 = 12.0 * (CELL_W + 3.0) + 12.0;

/// Roster display order matches the battle line front to rear.
const ROSTER_ORDER: [u8; NUM_KINDS] = [KIND_HEAVY, KIND_SPEAR, KIND_LIGHT, KIND_ARCHER];

fn kind_name(kind: u8) -> &'static str {
    match kind {
        KIND_HEAVY => "Knights",
        KIND_SPEAR => "Spearmen",
        KIND_ARCHER => "Bowmen",
        _ => "Men-at-Arms",
    }
}

fn kind_desc(kind: u8) -> &'static str {
    match kind {
        KIND_HEAVY => "Knights: slow, armored line breakers. Strongest holding the front rank.",
        KIND_SPEAR => "Spearmen: chainmail line troops. Hold ground and blunt cavalry-less charges.",
        KIND_ARCHER => "Bowmen: volleys over the friendly line. Force multipliers; a few go a long way.",
        _ => "Men-at-Arms: fast and lightly armored. Flankers, screens, and the reserve line.",
    }
}

const HINT: &str = "Drag a card into the army list or click it to add a regiment. \
                    Click a placed card to remove it. Hold Shift for 10 at a time.";

pub struct PickerPlugin;

impl Plugin for PickerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(GameState::UnitSelect), spawn_picker)
            .add_systems(
                Update,
                (picker_buttons, picker_keys, move_ghost, refresh_picker)
                    .run_if(in_state(GameState::UnitSelect)),
            );
    }
}

/// Which side the screen is editing, plus the last hand-built enemy
/// composition (kept across Random/Manual toggles).
#[derive(Resource)]
struct PickerState {
    team: u8,
    enemy_manual: [usize; NUM_KINDS],
}

/// Live drag: the kind being dragged and the ghost card following the
/// pointer. `kind` doubles as the guard that keeps the click ending a
/// drag from also counting as a click-to-add (Click fires before
/// DragEnd on release).
#[derive(Resource, Default)]
struct PickerDrag {
    kind: Option<u8>,
    ghost: Option<Entity>,
}

#[derive(Resource)]
struct PickerIcons([Handle<Image>; NUM_KINDS]);

#[derive(Component)]
struct GridRoot;

#[derive(Component)]
struct RosterCard(u8);

/// A filled army-list cell showing one regiment of this kind.
#[derive(Component)]
struct SlotCell(u8);

/// Panes the refresh toggles between player/enemy and Random/Manual.
#[derive(Component, PartialEq)]
enum PickerPane {
    Grid,
    RandomNote,
    ModeRow,
}

/// Every dynamic text on the screen, refreshed together.
#[derive(Component, PartialEq)]
enum PickerText {
    Team,
    Mode,
    Slots,
    Soldiers,
    Desc,
    Count(u8),
}

#[derive(Component)]
enum PickerButton {
    TeamFlip,
    Mode,
    Back,
    Start,
}

fn spawn_picker(
    mut commands: Commands,
    config: Res<BattleConfig>,
    mut images: ResMut<Assets<Image>>,
) {
    let icons = PickerIcons(std::array::from_fn(|k| {
        images.add(crate::unit_cards::kind_icon_image(k as u8))
    }));
    let enemy_manual = config
        .enemy_regs
        .unwrap_or_else(|| crate::regiments::frac_comp(config.n_slots()));
    commands.insert_resource(PickerState { team: 0, enemy_manual });
    commands.insert_resource(PickerDrag::default());

    commands
        .spawn((
            fullscreen_overlay(),
            BackgroundColor(PANEL_BG),
            GlobalZIndex(10),
            DespawnOnExit(GameState::UnitSelect),
        ))
        .with_children(|p| {
            p.spawn((
                Text::new("Select Units"),
                TextFont { font_size: FontSize::Px(40.0), ..default() },
                TextColor(TEXT_COLOR),
                Node { margin: UiRect::bottom(Val::Px(18.0)), ..default() },
            ));

            // Info column | army list | unit roster.
            p.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::FlexStart,
                column_gap: Val::Px(24.0),
                ..default()
            })
            .with_children(|row| {
                spawn_info_column(row, &config);
                spawn_army_pane(row);
                spawn_roster_pane(row, &icons);
            });

            // Hover description strip (fixed height so nothing jumps).
            p.spawn((
                Text::new(HINT),
                TextFont { font_size: FontSize::Px(13.0), ..default() },
                TextColor(DIM_TEXT_COLOR),
                Node {
                    max_width: Val::Px(860.0),
                    height: Val::Px(36.0),
                    margin: UiRect::top(Val::Px(14.0)),
                    ..default()
                },
                PickerText::Desc,
            ));

            p.spawn(Node {
                flex_direction: FlexDirection::Row,
                margin: UiRect::top(Val::Px(6.0)),
                ..default()
            })
            .with_children(|btns| {
                spawn_text_button(btns, "Back", PickerButton::Back);
                spawn_text_button(btns, "Start Battle", PickerButton::Start);
            });
        });
    commands.insert_resource(icons);
}

fn spawn_info_column(row: &mut ChildSpawnerCommands, config: &BattleConfig) {
    row.spawn(Node {
        flex_direction: FlexDirection::Column,
        row_gap: Val::Px(6.0),
        min_width: Val::Px(150.0),
        padding: UiRect::top(Val::Px(4.0)),
        ..default()
    })
    .with_children(|col| {
        let dim_line = |col: &mut ChildSpawnerCommands, s: String| {
            col.spawn((
                Text::new(s),
                TextFont { font_size: FontSize::Px(13.0), ..default() },
                TextColor(DIM_TEXT_COLOR),
            ));
        };
        let total = 2 * config.units_per_team;
        let army = if total.is_multiple_of(1000) {
            format!("Army: {}k", total / 1000)
        } else {
            format!("Army: {total}")
        };
        dim_line(col, army);
        dim_line(col, format!("Regiment: {} men", config.reg_size));
        col.spawn((
            Text::new(""),
            TextFont { font_size: FontSize::Px(14.0), ..default() },
            TextColor(TEXT_COLOR),
            Node { margin: UiRect::top(Val::Px(10.0)), ..default() },
            PickerText::Slots,
        ));
        col.spawn((
            Text::new(""),
            TextFont { font_size: FontSize::Px(14.0), ..default() },
            TextColor(TEXT_COLOR),
            PickerText::Soldiers,
        ));
    });
}

fn spawn_army_pane(row: &mut ChildSpawnerCommands) {
    row.spawn(Node {
        flex_direction: FlexDirection::Column,
        align_items: AlignItems::Center,
        row_gap: Val::Px(8.0),
        ..default()
    })
    .with_children(|pane| {
        pane.spawn(Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|h| {
            spawn_arrow(h, "<");
            h.spawn((
                Text::new(""),
                TextFont { font_size: FontSize::Px(18.0), ..default() },
                TextColor(TEXT_COLOR),
                TextLayout::justify(Justify::Center),
                Node { width: Val::Px(230.0), ..default() },
                PickerText::Team,
            ));
            spawn_arrow(h, ">");
        });

        // Enemy only: the Random/Manual composition toggle.
        pane.spawn((
            Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(8.0),
                display: Display::None,
                ..default()
            },
            PickerPane::ModeRow,
        ))
        .with_children(|m| {
            m.spawn((
                Text::new("Composition:"),
                TextFont { font_size: FontSize::Px(13.0), ..default() },
                TextColor(DIM_TEXT_COLOR),
            ));
            m.spawn((
                Button,
                Node {
                    padding: UiRect::axes(Val::Px(14.0), Val::Px(4.0)),
                    min_width: Val::Px(90.0),
                    justify_content: JustifyContent::Center,
                    ..default()
                },
                BackgroundColor(BTN_NORMAL),
                PickerButton::Mode,
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new(""),
                    TextFont { font_size: FontSize::Px(13.0), ..default() },
                    TextColor(TEXT_COLOR),
                    PickerText::Mode,
                ));
            });
        });

        pane.spawn((
            Node {
                width: Val::Px(GRID_W),
                min_height: Val::Px(4.0 * (CELL_H + 3.0)),
                flex_direction: FlexDirection::Row,
                flex_wrap: FlexWrap::Wrap,
                align_content: AlignContent::FlexStart,
                column_gap: Val::Px(3.0),
                row_gap: Val::Px(3.0),
                padding: UiRect::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(GRID_BG),
            GridRoot,
            PickerPane::Grid,
        ))
        .observe(grid_drop);

        pane.spawn((
            Node {
                width: Val::Px(GRID_W),
                padding: UiRect::all(Val::Px(20.0)),
                justify_content: JustifyContent::Center,
                display: Display::None,
                ..default()
            },
            PickerPane::RandomNote,
        ))
        .with_children(|n| {
            n.spawn((
                Text::new("A different army style takes the field every battle."),
                TextFont { font_size: FontSize::Px(14.0), ..default() },
                TextColor(DIM_TEXT_COLOR),
            ));
        });
    });
}

fn spawn_arrow(h: &mut ChildSpawnerCommands, label: &str) {
    h.spawn((
        Button,
        Node {
            padding: UiRect::axes(Val::Px(12.0), Val::Px(4.0)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(BTN_NORMAL),
        PickerButton::TeamFlip,
    ))
    .with_children(|b| {
        b.spawn((
            Text::new(label),
            TextFont { font_size: FontSize::Px(16.0), ..default() },
            TextColor(TEXT_COLOR),
        ));
    });
}

fn spawn_roster_pane(row: &mut ChildSpawnerCommands, icons: &PickerIcons) {
    row.spawn(Node {
        flex_direction: FlexDirection::Column,
        row_gap: Val::Px(6.0),
        ..default()
    })
    .with_children(|pane| {
        pane.spawn((
            Text::new("Unit Roster"),
            TextFont { font_size: FontSize::Px(13.0), ..default() },
            TextColor(DIM_TEXT_COLOR),
        ));
        for kind in ROSTER_ORDER {
            pane.spawn((
                Node {
                    width: Val::Px(200.0),
                    height: Val::Px(52.0),
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(10.0),
                    padding: UiRect::right(Val::Px(8.0)),
                    ..default()
                },
                BackgroundColor(CARD_BG),
                RosterCard(kind),
            ))
            .observe(roster_click)
            .observe(roster_drag_start)
            .observe(roster_drag_end)
            .observe(roster_over)
            .observe(roster_out)
            .with_children(|card| {
                card.spawn((
                    Node {
                        width: Val::Px(6.0),
                        height: Val::Percent(100.0),
                        ..default()
                    },
                    BackgroundColor(KIND_FILL[kind as usize]),
                ));
                card.spawn((
                    ImageNode::new(icons.0[kind as usize].clone()),
                    Node {
                        width: Val::Px(20.0),
                        height: Val::Px(24.0),
                        ..default()
                    },
                ));
                card.spawn((
                    Text::new(kind_name(kind)),
                    TextFont { font_size: FontSize::Px(14.0), ..default() },
                    TextColor(TEXT_COLOR),
                    Node { flex_grow: 1.0, ..default() },
                ));
                card.spawn((
                    Text::new(""),
                    TextFont { font_size: FontSize::Px(13.0), ..default() },
                    TextColor(DIM_TEXT_COLOR),
                    TextLayout::no_wrap(),
                    PickerText::Count(kind),
                ));
            });
        }
    });
}

// ── Composition editing ──

fn shift_step(keys: &ButtonInput<KeyCode>) -> usize {
    if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) { 10 } else { 1 }
}

fn add_regs(config: &mut BattleConfig, state: &mut PickerState, kind: u8, n: usize) {
    let slots = config.n_slots();
    if state.team == 0 {
        let free = slots.saturating_sub(config.player_regs.iter().sum::<usize>());
        config.player_regs[kind as usize] += n.min(free);
    } else if config.enemy_regs.is_some() {
        let free = slots.saturating_sub(state.enemy_manual.iter().sum::<usize>());
        state.enemy_manual[kind as usize] += n.min(free);
        config.enemy_regs = Some(state.enemy_manual);
    }
}

fn remove_regs(config: &mut BattleConfig, state: &mut PickerState, kind: u8, n: usize) {
    if state.team == 0 {
        let c = &mut config.player_regs[kind as usize];
        *c = c.saturating_sub(n);
    } else if config.enemy_regs.is_some() {
        let c = &mut state.enemy_manual[kind as usize];
        *c = c.saturating_sub(n);
        config.enemy_regs = Some(state.enemy_manual);
    }
}

/// The composition the screen currently displays; None while the enemy
/// page sits on Random.
fn shown_comp(config: &BattleConfig, state: &PickerState) -> Option<[usize; NUM_KINDS]> {
    if state.team == 0 { Some(config.player_regs) } else { config.enemy_regs }
}

// ── Drag and drop ──

fn roster_click(
    ev: On<Pointer<Click>>,
    cards: Query<&RosterCard>,
    keys: Res<ButtonInput<KeyCode>>,
    drag: Res<PickerDrag>,
    mut config: ResMut<BattleConfig>,
    mut state: ResMut<PickerState>,
) {
    if drag.kind.is_some() || ev.event.button != PointerButton::Primary {
        return;
    }
    let Ok(&RosterCard(kind)) = cards.get(ev.entity) else { return };
    add_regs(&mut config, &mut state, kind, shift_step(&keys));
}

fn roster_drag_start(
    ev: On<Pointer<DragStart>>,
    cards: Query<&RosterCard>,
    icons: Res<PickerIcons>,
    mut drag: ResMut<PickerDrag>,
    mut commands: Commands,
) {
    if ev.event.button != PointerButton::Primary {
        return;
    }
    let Ok(&RosterCard(kind)) = cards.get(ev.entity) else { return };
    let pos = ev.pointer_location.position;
    // The ghost must not occlude picking or the drop target under the
    // pointer would always be the ghost itself.
    let ghost = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(pos.x - CELL_W / 2.0),
                top: Val::Px(pos.y - CELL_H / 2.0),
                width: Val::Px(CELL_W),
                height: Val::Px(CELL_H),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(KIND_FILL[kind as usize].with_alpha(0.85)),
            GlobalZIndex(30),
            Pickable::IGNORE,
            DespawnOnExit(GameState::UnitSelect),
        ))
        .with_children(|g| {
            g.spawn((
                ImageNode::new(icons.0[kind as usize].clone()),
                Node {
                    width: Val::Px(20.0),
                    height: Val::Px(24.0),
                    ..default()
                },
                Pickable::IGNORE,
            ));
        })
        .id();
    drag.kind = Some(kind);
    drag.ghost = Some(ghost);
}

fn move_ghost(
    drag: Res<PickerDrag>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut nodes: Query<&mut Node>,
) {
    let Some(ghost) = drag.ghost else { return };
    let Ok(window) = windows.single() else { return };
    let Some(pos) = window.cursor_position() else { return };
    if let Ok(mut node) = nodes.get_mut(ghost) {
        node.left = Val::Px(pos.x - CELL_W / 2.0);
        node.top = Val::Px(pos.y - CELL_H / 2.0);
    }
}

fn grid_drop(
    ev: On<Pointer<DragDrop>>,
    cards: Query<&RosterCard>,
    keys: Res<ButtonInput<KeyCode>>,
    mut config: ResMut<BattleConfig>,
    mut state: ResMut<PickerState>,
) {
    let Ok(&RosterCard(kind)) = cards.get(ev.event.dropped) else { return };
    add_regs(&mut config, &mut state, kind, shift_step(&keys));
}

fn roster_drag_end(
    _ev: On<Pointer<DragEnd>>,
    mut drag: ResMut<PickerDrag>,
    mut commands: Commands,
) {
    if let Some(ghost) = drag.ghost.take() {
        commands.entity(ghost).despawn();
    }
    drag.kind = None;
}

fn roster_over(
    ev: On<Pointer<Over>>,
    cards: Query<&RosterCard>,
    mut colors: Query<&mut BackgroundColor>,
    mut texts: Query<(&mut Text, &PickerText)>,
) {
    let Ok(&RosterCard(kind)) = cards.get(ev.entity) else { return };
    if let Ok(mut bg) = colors.get_mut(ev.entity) {
        bg.0 = CARD_BG_HOVER;
    }
    for (mut text, tag) in &mut texts {
        if *tag == PickerText::Desc {
            text.0 = kind_desc(kind).to_string();
        }
    }
}

fn roster_out(
    ev: On<Pointer<Out>>,
    cards: Query<&RosterCard>,
    mut colors: Query<&mut BackgroundColor>,
    mut texts: Query<(&mut Text, &PickerText)>,
) {
    if cards.get(ev.entity).is_err() {
        return;
    }
    if let Ok(mut bg) = colors.get_mut(ev.entity) {
        bg.0 = CARD_BG;
    }
    for (mut text, tag) in &mut texts {
        if *tag == PickerText::Desc {
            text.0 = HINT.to_string();
        }
    }
}

fn cell_click(
    ev: On<Pointer<Click>>,
    cells: Query<&SlotCell>,
    keys: Res<ButtonInput<KeyCode>>,
    mut config: ResMut<BattleConfig>,
    mut state: ResMut<PickerState>,
) {
    if ev.event.button != PointerButton::Primary {
        return;
    }
    let Ok(&SlotCell(kind)) = cells.get(ev.entity) else { return };
    remove_regs(&mut config, &mut state, kind, shift_step(&keys));
}

// ── Buttons and refresh ──

fn army_valid(config: &BattleConfig) -> bool {
    config.player_regs.iter().sum::<usize>() > 0
        && config.enemy_regs.is_none_or(|c| c.iter().sum::<usize>() > 0)
}

fn picker_buttons(
    query: Query<(&Interaction, &PickerButton), Changed<Interaction>>,
    mut config: ResMut<BattleConfig>,
    mut state: ResMut<PickerState>,
    mut next: ResMut<NextState<GameState>>,
) {
    for (interaction, btn) in &query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match btn {
            PickerButton::TeamFlip => state.team ^= 1,
            PickerButton::Mode => {
                config.enemy_regs = match config.enemy_regs {
                    Some(_) => None,
                    None => Some(state.enemy_manual),
                };
            }
            PickerButton::Back => next.set(GameState::Menu),
            PickerButton::Start => {
                if army_valid(&config) {
                    next.set(GameState::Battle);
                }
            }
        }
    }
}

fn picker_keys(
    keys: Res<ButtonInput<KeyCode>>,
    config: Res<BattleConfig>,
    mut next: ResMut<NextState<GameState>>,
) {
    if (keys.just_pressed(KeyCode::Enter) || keys.just_pressed(KeyCode::Space))
        && army_valid(&config)
    {
        next.set(GameState::Battle);
    }
    if keys.just_pressed(KeyCode::Escape) {
        next.set(GameState::Menu);
    }
}

#[allow(clippy::too_many_arguments)]
fn refresh_picker(
    config: Res<BattleConfig>,
    state: Res<PickerState>,
    icons: Res<PickerIcons>,
    mut commands: Commands,
    grid: Query<Entity, With<GridRoot>>,
    mut panes: Query<(&mut Node, &PickerPane)>,
    mut texts: Query<(&mut Text, &PickerText)>,
) {
    if !config.is_changed() && !state.is_changed() {
        return;
    }
    let Ok(grid_e) = grid.single() else { return };
    let slots = config.n_slots();
    let comp = shown_comp(&config, &state);
    let used = comp.map(|c| c.iter().sum::<usize>());

    // Pane visibility: the enemy page shows the mode row, and Random
    // mode swaps the army list for a note.
    for (mut node, pane) in &mut panes {
        node.display = match pane {
            PickerPane::ModeRow if state.team == 1 => Display::Flex,
            PickerPane::Grid if comp.is_some() => Display::Flex,
            PickerPane::RandomNote if comp.is_none() => Display::Flex,
            _ => Display::None,
        };
    }

    for (mut text, tag) in &mut texts {
        match tag {
            PickerText::Team => {
                text.0 = if state.team == 0 {
                    "Team 1  -  Your Army".into()
                } else {
                    "Team 2  -  Enemy Army".into()
                };
            }
            PickerText::Mode => {
                text.0 = if config.enemy_regs.is_some() { "Manual".into() } else { "Random".into() };
            }
            PickerText::Slots => {
                text.0 = match used {
                    Some(u) => format!("Regiments: {u} / {slots}"),
                    None => format!("Regiments: ? / {slots}"),
                };
            }
            PickerText::Soldiers => {
                text.0 = match used {
                    Some(u) => format!("Soldiers: {}", u * config.reg_size),
                    None => "Soldiers: ?".into(),
                };
            }
            PickerText::Count(kind) => {
                text.0 = match comp {
                    Some(c) => format!("x {}", c[*kind as usize]),
                    None => "x ?".into(),
                };
            }
            PickerText::Desc => {}
        }
    }

    // Rebuild the army list: filled cells in ladder order, then the
    // remaining empty slots.
    commands.entity(grid_e).despawn_related::<Children>();
    let Some(comp) = comp else { return };
    commands.entity(grid_e).with_children(|g| {
        let cell = || Node {
            width: Val::Px(CELL_W),
            height: Val::Px(CELL_H),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        };
        for kind in [KIND_HEAVY, KIND_SPEAR, KIND_LIGHT, KIND_ARCHER] {
            for _ in 0..comp[kind as usize] {
                g.spawn((cell(), BackgroundColor(KIND_FILL[kind as usize]), SlotCell(kind)))
                    .observe(cell_click)
                    .with_children(|c| {
                        c.spawn((
                            ImageNode::new(icons.0[kind as usize].clone()),
                            Node {
                                width: Val::Px(20.0),
                                height: Val::Px(24.0),
                                ..default()
                            },
                        ));
                    });
            }
        }
        for _ in used.unwrap_or(0)..slots {
            g.spawn((cell(), BackgroundColor(CELL_EMPTY)));
        }
    });
}
