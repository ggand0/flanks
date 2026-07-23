//! TW-style battle HUD: the bottom unit card bar (one card per player
//! regiment) and the minimal control panel (Halt + formation buttons).
//!
//! Cards flex-shrink to fit the window: at ~20 regiments they read as
//! proper cards (kind letter, strength fill, morale strip), at 100+ they
//! compress toward slivers where the kind color + strength fill is the
//! whole message. Everything here is presentation: it reads `Groups` /
//! `Selection` and issues the same commands as the existing hotkeys.

use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::prelude::*;

use crate::formation::{FormCmd, FormShape, FormSpacing, apply_formation_cmd};
use crate::game_state::{BattleInputSet, GameState};
use crate::orders::{Groups, Hover, PLAYER_TEAM, RegState, Selection, halt_selected};
use crate::unit_types::{KIND_HEAVY, KIND_SPEAR};

/// Marker: this Button paints its own BackgroundColor — the global hover
/// styler in game_state.rs must leave it alone.
#[derive(Component)]
pub struct CustomStyled;

/// Card button; the payload is the regiment's index into `Groups::list`.
#[derive(Component)]
struct UnitCard(usize);

/// Strength fill child (bottom-anchored, height = living fraction).
#[derive(Component)]
struct CardFill(usize);

/// Morale strip child (top edge, colored by morale/state).
#[derive(Component)]
struct CardMorale(usize);

#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum ControlButton {
    Halt,
    Wall,
    Loose,
    Blob,
    Hold,
}

const BAR_BG: Color = Color::srgba(0.05, 0.06, 0.08, 0.80);
const CARD_BG: Color = Color::srgba(0.10, 0.11, 0.14, 0.92);
const CARD_BG_HOVER: Color = Color::srgba(0.22, 0.24, 0.30, 0.95);
const CARD_BG_DEAD: Color = Color::srgba(0.04, 0.04, 0.05, 0.85);
/// Strength fill per kind, player-blue family (team color is blue on the
/// field; the kind letter + shade carries the rest).
const KIND_FILL: [Color; 3] = [
    Color::srgb(0.22, 0.42, 0.85), // heavy: deep blue
    Color::srgb(0.42, 0.62, 0.95), // light: pale blue
    Color::srgb(0.30, 0.55, 0.70), // spear: teal blue
];
/// Broken regiments drain to the broken-flag gray (banners.rs).
const FILL_BROKEN: Color = Color::srgb(0.45, 0.50, 0.58);
const SEL_OUTLINE: Color = Color::srgb(0.95, 0.95, 0.90);

const BTN_NORMAL: Color = Color::srgba(0.15, 0.16, 0.20, 0.92);
const BTN_HOVER: Color = Color::srgba(0.25, 0.27, 0.32, 0.95);
const BTN_PRESSED: Color = Color::srgba(0.10, 0.11, 0.14, 0.95);
const BTN_ACTIVE: Color = Color::srgba(0.22, 0.38, 0.62, 0.95);
const BTN_ACTIVE_HOVER: Color = Color::srgba(0.30, 0.48, 0.74, 0.95);
const BTN_DISABLED: Color = Color::srgba(0.09, 0.10, 0.12, 0.80);

/// True when the pointer sits on any interactive UI node: map input
/// (lasso start, RMB order start) must not fire through the HUD.
pub fn pointer_over_ui<'a>(mut interactions: impl Iterator<Item = &'a Interaction>) -> bool {
    interactions.any(|i| *i != Interaction::None)
}

pub struct UnitCardsPlugin;

impl Plugin for UnitCardsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            OnEnter(GameState::Battle),
            spawn_card_bar.after(crate::game_state::setup_battle),
        )
        .add_systems(
            Update,
            (
                (
                    // After the lasso: the over-UI guard in drag_select
                    // reads hover state that lags synthetic same-frame
                    // move+click input by one frame, and if a click ever
                    // slips through both paths the card must win.
                    card_clicks.after(crate::selection::drag_select),
                    card_hover.after(crate::selection::update_hover),
                    control_buttons,
                )
                    .in_set(BattleInputSet),
                // Visual refresh keeps running while paused: the HUD stays
                // readable, only input is gated.
                (refresh_cards, refresh_control_buttons)
                    .run_if(in_state(GameState::Battle)),
            ),
        );
    }
}

fn kind_letter(kind: u8) -> &'static str {
    match kind {
        KIND_HEAVY => "H",
        KIND_SPEAR => "S",
        _ => "L",
    }
}

fn morale_color(gd: &crate::orders::GroupData) -> Color {
    match gd.state {
        RegState::Routing { .. } => Color::srgb(0.95, 0.20, 0.15),
        RegState::Shattered => Color::srgb(0.35, 0.12, 0.10),
        RegState::Steady => {
            // Green (100) -> yellow (50) -> red (0), two-segment lerp.
            let t = (gd.morale / 100.0).clamp(0.0, 1.0);
            if t > 0.5 {
                let k = (t - 0.5) * 2.0;
                Color::srgb(0.85 - 0.55 * k, 0.80, 0.25)
            } else {
                let k = t * 2.0;
                Color::srgb(0.90, 0.25 + 0.55 * k, 0.18)
            }
        }
    }
}

// ── Spawn ──

fn spawn_card_bar(mut commands: Commands, groups: Res<Groups>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                bottom: Val::Px(0.0),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::FlexEnd,
                padding: UiRect::axes(Val::Px(8.0), Val::Px(5.0)),
                column_gap: Val::Px(10.0),
                ..default()
            },
            BackgroundColor(BAR_BG),
            GlobalZIndex(5),
            // Blocks map input in the gaps between cards too.
            Interaction::default(),
            DespawnOnExit(GameState::Battle),
        ))
        .with_children(|bar| {
            spawn_control_panel(bar);
            bar.spawn(Node {
                flex_grow: 1.0,
                min_width: Val::Px(0.0),
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::FlexEnd,
                column_gap: Val::Px(1.0),
                ..default()
            })
            .with_children(|strip| {
                for (g, gd) in groups.list.iter().enumerate() {
                    if gd.team == PLAYER_TEAM {
                        spawn_card(strip, g, gd.kind);
                    }
                }
            });
        });
}

fn spawn_card(strip: &mut ChildSpawnerCommands, g: usize, kind: u8) {
    strip
        .spawn((
            Button,
            Node {
                flex_basis: Val::Px(44.0),
                flex_shrink: 1.0,
                min_width: Val::Px(3.0),
                height: Val::Px(56.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                overflow: Overflow::clip(),
                ..default()
            },
            BackgroundColor(CARD_BG),
            Outline {
                width: Val::Px(2.0),
                offset: Val::Px(0.0),
                color: Color::NONE,
            },
            UnitCard(g),
            CustomStyled,
        ))
        .with_children(|card| {
            card.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(0.0),
                    right: Val::Px(0.0),
                    bottom: Val::Px(0.0),
                    height: Val::Percent(100.0),
                    ..default()
                },
                BackgroundColor(KIND_FILL[kind as usize]),
                CardFill(g),
            ));
            card.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(0.0),
                    right: Val::Px(0.0),
                    top: Val::Px(0.0),
                    height: Val::Px(4.0),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.30, 0.80, 0.25)),
                CardMorale(g),
            ));
            card.spawn((
                Text::new(kind_letter(kind)),
                TextFont {
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(Color::srgba(0.95, 0.95, 0.90, 0.75)),
            ));
        });
}

fn spawn_control_panel(bar: &mut ChildSpawnerCommands) {
    const BUTTONS: [(ControlButton, &str); 5] = [
        (ControlButton::Halt, "Halt"),
        (ControlButton::Wall, "Wall"),
        (ControlButton::Loose, "Loose"),
        (ControlButton::Blob, "Blob"),
        (ControlButton::Hold, "Hold"),
    ];
    bar.spawn(Node {
        width: Val::Px(118.0),
        flex_shrink: 0.0,
        flex_direction: FlexDirection::Row,
        flex_wrap: FlexWrap::Wrap,
        align_content: AlignContent::FlexEnd,
        row_gap: Val::Px(2.0),
        column_gap: Val::Px(2.0),
        ..default()
    })
    .with_children(|panel| {
        for (btn, label) in BUTTONS {
            panel
                .spawn((
                    Button,
                    Node {
                        width: Val::Px(56.0),
                        height: Val::Px(17.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(BTN_DISABLED),
                    btn,
                    CustomStyled,
                ))
                .with_children(|b| {
                    b.spawn((
                        Text::new(label),
                        TextFont {
                            font_size: FontSize::Px(10.0),
                            ..default()
                        },
                        TextColor(Color::srgb(0.80, 0.80, 0.75)),
                    ));
                });
        }
    });
}

// ── Card interaction ──

/// Click selects the regiment; shift/ctrl-click toggles it in the mask.
fn card_clicks(
    cards: Query<(&Interaction, &UnitCard), Changed<Interaction>>,
    keys: Res<ButtonInput<KeyCode>>,
    groups: Res<Groups>,
    mut selection: ResMut<Selection>,
) {
    for (interaction, card) in &cards {
        if *interaction != Interaction::Pressed || groups.list[card.0].count == 0 {
            continue;
        }
        let additive = keys.pressed(KeyCode::ShiftLeft)
            || keys.pressed(KeyCode::ShiftRight)
            || keys.pressed(KeyCode::ControlLeft)
            || keys.pressed(KeyCode::ControlRight);
        if !additive {
            selection.regiments.clear();
        }
        selection.regiments.resize(groups.list.len(), false);
        selection.regiments[card.0] = !additive || !selection.regiments[card.0];
        selection.count_units = selection
            .regiments
            .iter()
            .enumerate()
            .filter(|(g, s)| **s && groups.list[*g].team == PLAYER_TEAM)
            .map(|(g, _)| groups.list[g].count)
            .sum();
    }
}

/// A hovered card feeds the same `Hover.own` the world pick writes, so
/// the inspect panel plaques the regiment. Runs after `update_hover` and
/// overrides its ground raycast (which lands on terrain behind the bar).
fn card_hover(cards: Query<(&Interaction, &UnitCard)>, mut hover: ResMut<Hover>) {
    for (interaction, card) in &cards {
        if *interaction != Interaction::None {
            hover.own = Some(card.0 as u32);
            hover.enemy = None;
            return;
        }
    }
}

// ── Card refresh ──

#[allow(clippy::type_complexity)] // disjoint BackgroundColor access
fn refresh_cards(
    groups: Res<Groups>,
    selection: Res<Selection>,
    mut cards: Query<(&UnitCard, &Interaction, &mut BackgroundColor, &mut Outline)>,
    mut fills: Query<
        (&CardFill, &mut Node, &mut BackgroundColor),
        (Without<UnitCard>, Without<CardMorale>),
    >,
    mut strips: Query<
        (&CardMorale, &mut BackgroundColor),
        (Without<UnitCard>, Without<CardFill>),
    >,
) {
    let set = |bg: &mut BackgroundColor, c: Color| {
        if bg.0 != c {
            bg.0 = c;
        }
    };

    for (card, interaction, mut bg, mut outline) in &mut cards {
        let gd = &groups.list[card.0];
        let want_bg = if gd.count == 0 {
            CARD_BG_DEAD
        } else if *interaction != Interaction::None {
            CARD_BG_HOVER
        } else {
            CARD_BG
        };
        set(&mut bg, want_bg);
        let selected = selection.regiments.get(card.0).copied().unwrap_or(false);
        let want_outline = if selected { SEL_OUTLINE } else { Color::NONE };
        if outline.color != want_outline {
            outline.color = want_outline;
        }
    }

    for (fill, mut node, mut bg) in &mut fills {
        let gd = &groups.list[fill.0];
        // Whole-percent steps: ~10 deaths per write at reg_size 1000,
        // keeps relayout off the per-frame path.
        let pct = (gd.count as f32 / gd.initial_count.max(1) as f32 * 100.0).round();
        let want = Val::Percent(pct);
        if node.height != want {
            node.height = want;
        }
        let color = if gd.state.is_broken() {
            FILL_BROKEN
        } else {
            KIND_FILL[gd.kind as usize]
        };
        set(&mut bg, color);
    }

    for (strip, mut bg) in &mut strips {
        let gd = &groups.list[strip.0];
        let color = if gd.count == 0 {
            Color::NONE
        } else {
            morale_color(gd)
        };
        set(&mut bg, color);
    }
}

// ── Control panel ──

/// Buttons drive the exact hotkey code paths (Backspace/F/L/B/H).
fn control_buttons(
    query: Query<(&Interaction, &ControlButton), Changed<Interaction>>,
    selection: Res<Selection>,
    mut groups: ResMut<Groups>,
) {
    for (interaction, btn) in &query {
        if *interaction != Interaction::Pressed || selection.count_units == 0 {
            continue;
        }
        match btn {
            ControlButton::Halt => halt_selected(&selection, &mut groups),
            ControlButton::Wall => apply_formation_cmd(FormCmd::Wall, &selection, &mut groups),
            ControlButton::Loose => apply_formation_cmd(FormCmd::Loose, &selection, &mut groups),
            ControlButton::Blob => apply_formation_cmd(FormCmd::Blob, &selection, &mut groups),
            ControlButton::Hold => apply_formation_cmd(FormCmd::Hold, &selection, &mut groups),
        }
    }
}

/// Toggle buttons light up when the WHOLE controllable selection is in
/// the mode (matching the hotkeys' toggle-as-a-set semantics: lit means
/// the next press turns it off).
fn refresh_control_buttons(
    groups: Res<Groups>,
    selection: Res<Selection>,
    mut query: Query<(&ControlButton, &Interaction, &mut BackgroundColor)>,
) {
    let picked: Vec<usize> = selection
        .regiments
        .iter()
        .enumerate()
        .filter(|(g, s)| {
            **s && {
                let gd = &groups.list[*g];
                gd.count > 0 && !gd.state.is_broken()
            }
        })
        .map(|(g, _)| g)
        .collect();

    for (btn, interaction, mut bg) in &mut query {
        let want = if picked.is_empty() {
            BTN_DISABLED
        } else {
            let active = match btn {
                ControlButton::Halt => false,
                ControlButton::Wall => {
                    picked.iter().all(|&g| groups.list[g].spacing == FormSpacing::Wall)
                }
                ControlButton::Loose => {
                    picked.iter().all(|&g| groups.list[g].spacing == FormSpacing::Loose)
                }
                ControlButton::Blob => {
                    picked.iter().all(|&g| groups.list[g].shape == FormShape::Blob)
                }
                ControlButton::Hold => picked.iter().all(|&g| groups.list[g].hold),
            };
            match (active, interaction) {
                (_, Interaction::Pressed) => BTN_PRESSED,
                (true, Interaction::Hovered) => BTN_ACTIVE_HOVER,
                (true, Interaction::None) => BTN_ACTIVE,
                (false, Interaction::Hovered) => BTN_HOVER,
                (false, Interaction::None) => BTN_NORMAL,
            }
        };
        if bg.0 != want {
            bg.0 = want;
        }
    }
}
