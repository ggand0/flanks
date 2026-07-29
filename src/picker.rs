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
    BTN_NORMAL, BattleConfig, CustomStyled, DIM_TEXT_COLOR, EnemyComp, GameState, PANEL_BG,
    TEXT_COLOR, fullscreen_overlay, spawn_text_button,
};
use crate::unit_cards::KIND_FILL;
use crate::unit_types::{KIND_ARCHER, KIND_HEAVY, KIND_LIGHT, KIND_SPEAR, NUM_KINDS};

const GRID_BG: Color = Color::srgba(0.03, 0.04, 0.05, 0.85);
const CELL_EMPTY: Color = Color::srgba(0.09, 0.10, 0.13, 0.90);
const CARD_BG: Color = Color::srgba(0.10, 0.11, 0.14, 0.92);
const CARD_BG_HOVER: Color = Color::srgba(0.22, 0.24, 0.30, 0.95);
/// The selected enemy-composition chip (same accent as HUD buttons).
const CHIP_ACTIVE: Color = Color::srgba(0.22, 0.38, 0.62, 0.95);

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
                    Right-click it or click a placed card to remove one. Hold Shift for 10 at a time.";

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

/// Panes the refresh toggles between player/enemy and the enemy modes.
#[derive(Component, PartialEq)]
enum PickerPane {
    Grid,
    Note,
    ModeRow,
    DefaultBtn,
}

/// Every dynamic text on the screen, refreshed together.
#[derive(Component, PartialEq)]
enum PickerText {
    Team,
    Note,
    Slots,
    Soldiers,
    Desc,
    Count(u8),
}

#[derive(Component)]
enum PickerButton {
    TeamFlip,
    Default,
    Back,
    Start,
}

/// One enemy-composition chip: Random, a named style, or Manual.
#[derive(Component, Clone, Copy, PartialEq)]
enum ModeChip {
    Random,
    Style(usize),
    Manual,
}

fn spawn_picker(
    mut commands: Commands,
    config: Res<BattleConfig>,
    mut images: ResMut<Assets<Image>>,
) {
    let icons = PickerIcons(std::array::from_fn(|k| {
        images.add(crate::unit_cards::kind_icon_image(k as u8))
    }));
    let enemy_manual = match config.enemy {
        EnemyComp::Manual(comp) => comp,
        _ => crate::regiments::frac_comp(config.n_slots()),
    };
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
            // Restores the classic split for whichever editable
            // composition is on screen; hidden on Random/style pages.
            h.spawn((
                Button,
                Node {
                    padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
                    margin: UiRect::left(Val::Px(10.0)),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(BTN_NORMAL),
                PickerButton::Default,
                PickerPane::DefaultBtn,
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new("Default"),
                    TextFont { font_size: FontSize::Px(13.0), ..default() },
                    TextColor(TEXT_COLOR),
                ));
            });
        });

        // Enemy only: composition chips — Random, each style, Manual.
        pane.spawn((
            Node {
                width: Val::Px(GRID_W),
                flex_direction: FlexDirection::Row,
                flex_wrap: FlexWrap::Wrap,
                justify_content: JustifyContent::Center,
                column_gap: Val::Px(4.0),
                row_gap: Val::Px(4.0),
                display: Display::None,
                ..default()
            },
            PickerPane::ModeRow,
        ))
        .with_children(|m| {
            spawn_chip(m, "Random", ModeChip::Random);
            for idx in 0..crate::regiments::archetype_count() {
                spawn_chip(m, crate::regiments::archetype_name(idx), ModeChip::Style(idx));
            }
            spawn_chip(m, "Manual", ModeChip::Manual);
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
            PickerPane::Note,
        ))
        .with_children(|n| {
            n.spawn((
                Text::new(""),
                TextFont { font_size: FontSize::Px(14.0), ..default() },
                TextColor(DIM_TEXT_COLOR),
                TextLayout::justify(Justify::Center),
                PickerText::Note,
            ));
        });
    });
}

fn spawn_chip(m: &mut ChildSpawnerCommands, label: &str, chip: ModeChip) {
    m.spawn((
        Button,
        Node {
            padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(BTN_NORMAL),
        CustomStyled,
        chip,
    ))
    .with_children(|b| {
        b.spawn((
            Text::new(label),
            TextFont { font_size: FontSize::Px(12.0), ..default() },
            TextColor(TEXT_COLOR),
        ));
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

fn enemy_manual_mode(config: &BattleConfig) -> bool {
    matches!(config.enemy, EnemyComp::Manual(_))
}

fn add_regs(config: &mut BattleConfig, state: &mut PickerState, kind: u8, n: usize) {
    let slots = config.n_slots();
    if state.team == 0 {
        let free = slots.saturating_sub(config.player_regs.iter().sum::<usize>());
        config.player_regs[kind as usize] += n.min(free);
    } else if enemy_manual_mode(config) {
        let free = slots.saturating_sub(state.enemy_manual.iter().sum::<usize>());
        state.enemy_manual[kind as usize] += n.min(free);
        config.enemy = EnemyComp::Manual(state.enemy_manual);
    }
}

fn remove_regs(config: &mut BattleConfig, state: &mut PickerState, kind: u8, n: usize) {
    if state.team == 0 {
        let c = &mut config.player_regs[kind as usize];
        *c = c.saturating_sub(n);
    } else if enemy_manual_mode(config) {
        let c = &mut state.enemy_manual[kind as usize];
        *c = c.saturating_sub(n);
        config.enemy = EnemyComp::Manual(state.enemy_manual);
    }
}

/// The composition the screen displays: real for the player page and
/// Manual, the representative preview for a chosen style, unknown for
/// Random (the only mode that hides the army list).
fn shown_counts(config: &BattleConfig, state: &PickerState) -> Option<[usize; NUM_KINDS]> {
    if state.team == 0 {
        return Some(config.player_regs);
    }
    match config.enemy {
        EnemyComp::Manual(comp) => Some(comp),
        EnemyComp::Style(idx) => {
            Some(crate::regiments::archetype_preview(idx, config.n_slots()))
        }
        EnemyComp::Random => None,
    }
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
    if drag.kind.is_some() {
        return;
    }
    let Ok(&RosterCard(kind)) = cards.get(ev.entity) else { return };
    match ev.event.button {
        PointerButton::Primary => add_regs(&mut config, &mut state, kind, shift_step(&keys)),
        // M2TW muscle memory: right-clicking the roster card removes.
        PointerButton::Secondary => remove_regs(&mut config, &mut state, kind, shift_step(&keys)),
        PointerButton::Middle => {}
    }
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
    if ev.event.button == PointerButton::Middle {
        return;
    }
    let Ok(&SlotCell(kind)) = cells.get(ev.entity) else { return };
    remove_regs(&mut config, &mut state, kind, shift_step(&keys));
}

// ── Buttons and refresh ──

fn army_valid(config: &BattleConfig) -> bool {
    let enemy_ok = match config.enemy {
        EnemyComp::Manual(comp) => comp.iter().sum::<usize>() > 0,
        _ => true,
    };
    config.player_regs.iter().sum::<usize>() > 0 && enemy_ok
}

fn picker_buttons(
    query: Query<(&Interaction, &PickerButton), Changed<Interaction>>,
    chips: Query<(&Interaction, &ModeChip), Changed<Interaction>>,
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
            PickerButton::Default => {
                let classic = crate::regiments::frac_comp(config.n_slots());
                if state.team == 0 {
                    config.player_regs = classic;
                } else if enemy_manual_mode(&config) {
                    state.enemy_manual = classic;
                    config.enemy = EnemyComp::Manual(classic);
                }
            }
            PickerButton::Back => next.set(GameState::Menu),
            PickerButton::Start => {
                if army_valid(&config) {
                    next.set(GameState::Battle);
                }
            }
        }
    }
    for (interaction, chip) in &chips {
        if *interaction != Interaction::Pressed {
            continue;
        }
        config.enemy = match chip {
            ModeChip::Random => EnemyComp::Random,
            ModeChip::Style(idx) => EnemyComp::Style(*idx),
            ModeChip::Manual => EnemyComp::Manual(state.enemy_manual),
        };
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
    mut chip_colors: Query<(&mut BackgroundColor, &ModeChip)>,
    mut texts: Query<(&mut Text, &PickerText)>,
) {
    if !config.is_changed() && !state.is_changed() {
        return;
    }
    let Ok(grid_e) = grid.single() else { return };
    let slots = config.n_slots();
    let counts = shown_counts(&config, &state);
    let used = counts.map(|c| c.iter().sum::<usize>());
    let editable = state.team == 0 || enemy_manual_mode(&config);

    // Pane visibility: the enemy page shows the mode chips; Random
    // hides the army list behind the note, a chosen style shows its
    // preview cards with the jitter caption underneath.
    for (mut node, pane) in &mut panes {
        node.display = match pane {
            PickerPane::ModeRow if state.team == 1 => Display::Flex,
            PickerPane::Grid if counts.is_some() => Display::Flex,
            PickerPane::Note if state.team == 1 && !enemy_manual_mode(&config) => Display::Flex,
            PickerPane::DefaultBtn if editable => Display::Flex,
            _ => Display::None,
        };
    }

    for (mut bg, chip) in &mut chip_colors {
        let active = matches!(
            (chip, config.enemy),
            (ModeChip::Random, EnemyComp::Random) | (ModeChip::Manual, EnemyComp::Manual(_))
        ) || matches!((chip, config.enemy), (ModeChip::Style(i), EnemyComp::Style(j)) if *i == j);
        bg.0 = if active { CHIP_ACTIVE } else { BTN_NORMAL };
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
            PickerText::Note => {
                text.0 = match config.enemy {
                    EnemyComp::Random => {
                        "A different army style takes the field every battle.".into()
                    }
                    EnemyComp::Style(_) => "Jittered a little every battle.".into(),
                    EnemyComp::Manual(_) => String::new(),
                };
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
                text.0 = match counts {
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
    let Some(comp) = counts else { return };
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
