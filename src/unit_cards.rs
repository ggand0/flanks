//! TW-style battle HUD: the bottom unit card bar (one card per player
//! regiment) and the minimal control panel (Halt + formation buttons).
//!
//! Cards flex-shrink to fit the window: at ~20 regiments they read as
//! proper cards (kind letter, strength fill, morale strip), at 100+ they
//! compress toward slivers where the kind color + strength fill is the
//! whole message. Everything here is presentation: it reads `Groups` /
//! `Selection` and issues the same commands as the existing hotkeys.

use bevy::asset::RenderAssetUsages;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

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

/// Kind letter fallback, shown when the card is too narrow for the icon.
#[derive(Component)]
struct CardGlyph(usize);

/// Node-drawn kind icon (helm / spear / sword), shown on wide cards.
#[derive(Component)]
struct CardIcon(usize);

/// The Loose button's icon reflects what pressing it yields: spread
/// squares while the selection is in close order, tight squares once
/// everyone is loose (M2TW-style state icon).
#[derive(Component)]
struct LooseIconVariant {
    tight: bool,
}

/// Hover tooltip of a control button (name + hotkey).
#[derive(Component)]
struct BtnTooltip(ControlButton);

/// One rasterized icon texture per unit kind, shared by every card so
/// all icons of a kind are pixel-identical. (Node-composed icons round
/// each rect to the pixel grid separately, and under a fractional
/// display scale neighboring cards land on different subpixel phases,
/// visibly warping each copy differently.)
#[derive(Resource)]
struct KindIcons([Handle<Image>; crate::unit_types::NUM_KINDS]);

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

/// M2TW-ish parchment pictograms on the dark circles.
const ICON_COLOR: Color = Color::srgb(0.90, 0.87, 0.76);
const ICON_DETAIL: Color = Color::srgb(0.10, 0.11, 0.13);
/// Cards at least this wide (logical px) swap the kind letter for the icon.
const CARD_ICON_MIN_W: f32 = 30.0;

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

// ── Icon drawing ──
//
// Pictograms are composed from plain UI nodes (rects + border-radius
// rounding), matching the game's flat look with no image assets.

/// Absolute-positioned rectangle inside an icon canvas.
fn shape(
    p: &mut ChildSpawnerCommands,
    (x, y, w, h): (f32, f32, f32, f32),
    color: Color,
    radius: BorderRadius,
) {
    p.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(x),
            top: Val::Px(y),
            width: Val::Px(w),
            height: Val::Px(h),
            border_radius: radius,
            ..default()
        },
        BackgroundColor(color),
    ));
}

/// Fixed-size relative canvas the shapes position inside.
fn icon_canvas(w: f32, h: f32) -> Node {
    Node {
        width: Val::Px(w),
        height: Val::Px(h),
        ..default()
    }
}

// ── Kind icon rasterizer ──
//
// Card kind icons are rendered ONCE into a small texture per kind and
// shared by every card via ImageNode, so all copies are identical.
// Authored at 2x the 20x22 logical canvas for crisp downsampling.

const ICON_TEX_W: u32 = 40;
const ICON_TEX_H: u32 = 44;
const ICON_RGB: [u8; 3] = [230, 222, 194];
const ICON_RGB_DETAIL: [u8; 3] = [26, 28, 33];

/// A rounded rectangle in texture pixels: rect + per-corner radii
/// (tl, tr, br, bl) + straight-alpha color.
struct IconShape {
    rect: (f32, f32, f32, f32),
    radius: [f32; 4],
    rgb: [u8; 3],
}

fn kind_icon_shapes(kind: u8) -> Vec<IconShape> {
    let s = |rect, radius, rgb| IconShape { rect, radius, rgb };
    match kind {
        // Knight's helm: dome, dark eye slit, nose bar.
        KIND_HEAVY => vec![
            s((6.0, 4.0, 28.0, 36.0), [14.0, 14.0, 2.0, 2.0], ICON_RGB),
            s((8.0, 18.0, 24.0, 4.0), [0.0; 4], ICON_RGB_DETAIL),
            s((18.0, 18.0, 4.0, 10.0), [0.0; 4], ICON_RGB),
        ],
        // Spear: leaf head on a long shaft.
        KIND_SPEAR => vec![
            s((14.0, 0.0, 12.0, 18.0), [6.0; 4], ICON_RGB),
            s((18.0, 16.0, 4.0, 28.0), [0.0; 4], ICON_RGB),
        ],
        // Arming sword: blade, crossguard, grip, pommel.
        _ => vec![
            s((17.0, 2.0, 6.0, 24.0), [0.0; 4], ICON_RGB),
            s((10.0, 26.0, 20.0, 4.0), [1.0; 4], ICON_RGB),
            s((18.0, 30.0, 4.0, 10.0), [0.0; 4], ICON_RGB),
            s((16.0, 40.0, 8.0, 4.0), [2.0; 4], ICON_RGB),
        ],
    }
}

/// Signed distance to a rounded rect (y-down; radii tl, tr, br, bl).
fn sd_round_rect(px: f32, py: f32, shape: &IconShape) -> f32 {
    let (x, y, w, h) = shape.rect;
    let (hw, hh) = (w * 0.5, h * 0.5);
    let (qx, qy) = (px - (x + hw), py - (y + hh));
    let r = match (qx > 0.0, qy > 0.0) {
        (false, false) => shape.radius[0],
        (true, false) => shape.radius[1],
        (true, true) => shape.radius[2],
        (false, true) => shape.radius[3],
    }
    .min(hw)
    .min(hh);
    let ax = qx.abs() - (hw - r);
    let ay = qy.abs() - (hh - r);
    let (dx, dy) = (ax.max(0.0), ay.max(0.0));
    (dx * dx + dy * dy).sqrt() + ax.max(ay).min(0.0) - r
}

/// Software rasterization with 1px edge AA, straight-alpha src-over.
fn rasterize_kind_icon(kind: u8) -> Image {
    let (w, h) = (ICON_TEX_W as usize, ICON_TEX_H as usize);
    let mut buf = vec![0.0f32; w * h * 4];
    for shape in kind_icon_shapes(kind) {
        let src: [f32; 3] = shape.rgb.map(|c| c as f32 / 255.0);
        for py in 0..h {
            for px in 0..w {
                let d = sd_round_rect(px as f32 + 0.5, py as f32 + 0.5, &shape);
                let sa = (0.5 - d).clamp(0.0, 1.0);
                if sa <= 0.0 {
                    continue;
                }
                let p = &mut buf[(py * w + px) * 4..(py * w + px) * 4 + 4];
                let da = p[3] * (1.0 - sa);
                let a = sa + da;
                for c in 0..3 {
                    p[c] = (src[c] * sa + p[c] * da) / a;
                }
                p[3] = a;
            }
        }
    }
    let data = buf
        .iter()
        .map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();
    Image::new(
        Extent3d {
            width: ICON_TEX_W,
            height: ICON_TEX_H,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

/// Halt: stop square.
fn draw_halt(p: &mut ChildSpawnerCommands) {
    shape(
        p,
        (6.0, 6.0, 10.0, 10.0),
        ICON_COLOR,
        BorderRadius::all(Val::Px(2.0)),
    );
}

/// Wall: three shields shoulder to shoulder.
fn draw_wall(p: &mut ChildSpawnerCommands) {
    let shield = BorderRadius {
        top_left: Val::Px(1.0),
        top_right: Val::Px(1.0),
        bottom_left: Val::Px(3.0),
        bottom_right: Val::Px(3.0),
    };
    for x in [2.0, 8.0, 14.0] {
        shape(p, (x, 5.0, 5.0, 12.0), ICON_COLOR, shield);
    }
}

/// Loose: 2x3 squares; both spacing variants exist and refresh toggles
/// which one is visible (the icon shows what pressing yields).
fn draw_loose_variant(p: &mut ChildSpawnerCommands, tight: bool) {
    let (xs, ys): (&[f32], &[f32]) = if tight {
        (&[4.0, 9.0, 14.0], &[6.0, 11.0])
    } else {
        (&[1.0, 9.0, 17.0], &[3.0, 13.0])
    };
    for &y in ys {
        for &x in xs {
            shape(p, (x, y, 4.0, 4.0), ICON_COLOR, BorderRadius::ZERO);
        }
    }
}

/// Blob: an undressed scatter of men.
fn draw_blob(p: &mut ChildSpawnerCommands) {
    for (x, y) in [(3.0, 4.0), (12.0, 2.0), (16.0, 10.0), (5.0, 12.0), (11.0, 16.0)] {
        shape(p, (x, y, 4.0, 4.0), ICON_COLOR, BorderRadius::MAX);
    }
}

/// Hold: a planted shield with a boss.
fn draw_hold(p: &mut ChildSpawnerCommands) {
    let shield = BorderRadius {
        top_left: Val::Px(3.0),
        top_right: Val::Px(3.0),
        bottom_left: Val::Px(9.0),
        bottom_right: Val::Px(9.0),
    };
    shape(p, (4.0, 3.0, 14.0, 16.0), ICON_COLOR, shield);
    shape(p, (9.0, 8.0, 4.0, 4.0), ICON_DETAIL, BorderRadius::MAX);
}

// ── Spawn ──

fn spawn_card_bar(
    mut commands: Commands,
    groups: Res<Groups>,
    mut images: ResMut<Assets<Image>>,
    icons: Option<Res<KindIcons>>,
) {
    let icons = match icons {
        Some(r) => r.0.clone(),
        None => {
            let handles =
                [0u8, 1, 2].map(|kind| images.add(rasterize_kind_icon(kind)));
            commands.insert_resource(KindIcons(handles.clone()));
            handles
        }
    };
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
                        spawn_card(strip, g, gd.kind, icons[gd.kind as usize].clone());
                    }
                }
            });
            spawn_control_panel(bar);
        });
}

fn spawn_card(strip: &mut ChildSpawnerCommands, g: usize, kind: u8, icon: Handle<Image>) {
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
                CardGlyph(g),
            ));
            card.spawn((
                Node {
                    width: Val::Px(20.0),
                    height: Val::Px(22.0),
                    ..default()
                },
                ImageNode::new(icon),
                CardIcon(g),
            ));
        });
}

/// M2TW-style round icon buttons, right end of the bar.
fn spawn_control_panel(bar: &mut ChildSpawnerCommands) {
    const BUTTONS: [(ControlButton, &str); 5] = [
        (ControlButton::Halt, "Halt (Backspace)"),
        (ControlButton::Wall, "Wall (F)"),
        (ControlButton::Loose, "Loose (L)"),
        (ControlButton::Blob, "Blob (B)"),
        (ControlButton::Hold, "Hold (H)"),
    ];
    bar.spawn(Node {
        flex_shrink: 0.0,
        flex_direction: FlexDirection::Row,
        align_items: AlignItems::Center,
        align_self: AlignSelf::Center,
        column_gap: Val::Px(6.0),
        ..default()
    })
    .with_children(|panel| {
        for (btn, label) in BUTTONS {
            panel
                .spawn((
                    Button,
                    Node {
                        width: Val::Px(36.0),
                        height: Val::Px(36.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        border_radius: BorderRadius::MAX,
                        ..default()
                    },
                    BackgroundColor(BTN_DISABLED),
                    btn,
                    CustomStyled,
                ))
                .with_children(|b| {
                    // Hover tooltip, floating above the button. Anchored
                    // to its right edge so it never leaves the screen
                    // (the panel hugs the window's right side).
                    b.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            bottom: Val::Px(42.0),
                            right: Val::Px(-2.0),
                            padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                            border_radius: BorderRadius::all(Val::Px(4.0)),
                            ..default()
                        },
                        BackgroundColor(Color::srgba(0.04, 0.05, 0.07, 0.95)),
                        Visibility::Hidden,
                        BtnTooltip(btn),
                    ))
                    .with_children(|t| {
                        t.spawn((
                            Text::new(label),
                            TextFont {
                                font_size: FontSize::Px(11.0),
                                ..default()
                            },
                            TextColor(Color::srgb(0.92, 0.92, 0.85)),
                        ));
                    });
                    match btn {
                        ControlButton::Loose => {
                            // Both spacing variants stacked on the same
                            // canvas; refresh shows one.
                            b.spawn(icon_canvas(22.0, 22.0)).with_children(|c| {
                                for tight in [false, true] {
                                    c.spawn((
                                        Node {
                                            position_type: PositionType::Absolute,
                                            left: Val::Px(0.0),
                                            top: Val::Px(0.0),
                                            width: Val::Px(22.0),
                                            height: Val::Px(22.0),
                                            ..default()
                                        },
                                        if tight {
                                            Visibility::Hidden
                                        } else {
                                            Visibility::Inherited
                                        },
                                        LooseIconVariant { tight },
                                    ))
                                    .with_children(|v| draw_loose_variant(v, tight));
                                }
                            });
                        }
                        _ => {
                            b.spawn(icon_canvas(22.0, 22.0)).with_children(|c| {
                                match btn {
                                    ControlButton::Halt => draw_halt(c),
                                    ControlButton::Wall => draw_wall(c),
                                    ControlButton::Blob => draw_blob(c),
                                    ControlButton::Hold => draw_hold(c),
                                    ControlButton::Loose => unreachable!(),
                                }
                            });
                        }
                    }
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

#[allow(clippy::type_complexity)] // disjoint BackgroundColor/Node access
fn refresh_cards(
    groups: Res<Groups>,
    selection: Res<Selection>,
    mut cards: Query<(
        &UnitCard,
        &ComputedNode,
        &Interaction,
        &mut BackgroundColor,
        &mut Outline,
    )>,
    mut fills: Query<
        (&CardFill, &mut Node, &mut BackgroundColor),
        (Without<UnitCard>, Without<CardMorale>, Without<CardGlyph>, Without<CardIcon>),
    >,
    mut strips: Query<
        (&CardMorale, &mut BackgroundColor),
        (Without<UnitCard>, Without<CardFill>),
    >,
    mut glyphs: Query<
        (&CardGlyph, &mut Node),
        (Without<UnitCard>, Without<CardFill>, Without<CardIcon>),
    >,
    mut icons: Query<
        (&CardIcon, &mut Node),
        (Without<UnitCard>, Without<CardFill>, Without<CardGlyph>),
    >,
) {
    let set = |bg: &mut BackgroundColor, c: Color| {
        if bg.0 != c {
            bg.0 = c;
        }
    };

    // Wide-enough cards show the kind icon, narrow ones the letter
    // (which the sliver clip then eats). Indexed by regiment.
    let mut wide = vec![false; groups.list.len()];

    for (card, computed, interaction, mut bg, mut outline) in &mut cards {
        wide[card.0] =
            computed.size().x * computed.inverse_scale_factor() >= CARD_ICON_MIN_W;
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

    // Display (not Visibility) so the hidden one leaves the flex layout;
    // it only flips when a card crosses the width threshold.
    for (glyph, mut node) in &mut glyphs {
        let want = if wide[glyph.0] { Display::None } else { Display::Flex };
        if node.display != want {
            node.display = want;
        }
    }
    for (icon, mut node) in &mut icons {
        let want = if wide[icon.0] { Display::Flex } else { Display::None };
        if node.display != want {
            node.display = want;
        }
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
    mut loose_icons: Query<(&LooseIconVariant, &mut Visibility)>,
    mut tooltips: Query<
        (&BtnTooltip, &mut Visibility),
        Without<LooseIconVariant>,
    >,
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

    let mut loose_active = false;
    let mut hovered: Option<ControlButton> = None;
    for (btn, interaction, mut bg) in &mut query {
        if matches!(interaction, Interaction::Hovered | Interaction::Pressed) {
            hovered = Some(*btn);
        }
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
            if *btn == ControlButton::Loose {
                loose_active = active;
            }
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

    // The Loose icon previews the press result: spread squares while in
    // close order, tight squares once the whole selection is loose.
    for (variant, mut vis) in &mut loose_icons {
        let want = if variant.tight == loose_active {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *vis != want {
            *vis = want;
        }
    }

    for (tip, mut vis) in &mut tooltips {
        let want = if hovered == Some(tip.0) {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *vis != want {
            *vis = want;
        }
    }
}
