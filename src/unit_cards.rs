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

use crate::formation::{
    FormCmd, FormShape, FormSpacing, apply_formation_cmd, is_spearwall_kind,
};
use crate::game_state::{
    BTN_HOVER, BTN_NORMAL, BTN_PRESSED, CustomStyled, GameState, HudInputSet,
    TEXT_COLOR,
};
use crate::orders::{Groups, Hover, PLAYER_TEAM, RegState, Selection, halt_selected};
use crate::unit_types::{KIND_ARCHER, KIND_HEAVY, KIND_SPEAR, NUM_KINDS};

/// Card button; the payload is the regiment's index into `Groups::list`.
#[derive(Component)]
struct UnitCard(usize);

/// Strength fill child (bottom-anchored, height = living fraction).
#[derive(Component)]
struct CardFill(usize);

/// Morale strip child (top edge, colored by morale/state).
#[derive(Component)]
struct CardMorale(usize);

/// Fatigue strip child (under the morale strip): remaining stamina as a
/// left-anchored fill, tinted by the M2TW fatigue state.
#[derive(Component)]
struct CardFatigue(usize);

/// Ammo strip child (archer cards only, under the fatigue strip):
/// arrows left as a left-anchored fill.
#[derive(Component)]
struct CardAmmo(usize);

/// Firing indicator (archer cards only): a tinted mini bow icon in the
/// card's top-right corner while the regiment has a live fire solution
/// (GroupData::firing) — the M2TW "this unit is shooting" read.
#[derive(Component)]
struct CardFiring(usize);

/// The card's kind art: the rasterized icon (`icon: true`, shown on
/// wide cards) or the letter fallback (`icon: false`, narrow cards).
#[derive(Component)]
struct CardArt {
    g: usize,
    icon: bool,
}

/// One variant of a stateful button icon (M2TW-style: the icon previews
/// what pressing yields). `alt: false` is the default art, `alt: true`
/// the alternate; refresh shows the variant matching the selection.
#[derive(Component)]
struct StatefulIcon {
    btn: ControlButton,
    alt: bool,
}

/// Hover tooltip of a control button (name + hotkey).
#[derive(Component)]
struct BtnTooltip(ControlButton);

/// The Wall tooltip's Text: its label tracks the wall icon variant.
#[derive(Component)]
struct WallTooltipText;

#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum ControlButton {
    Halt,
    Form(FormCmd),
}

const BAR_BG: Color = Color::srgba(0.05, 0.06, 0.08, 0.80);
const CARD_BG: Color = Color::srgba(0.10, 0.11, 0.14, 0.92);
const CARD_BG_HOVER: Color = Color::srgba(0.22, 0.24, 0.30, 0.95);
const CARD_BG_DEAD: Color = Color::srgba(0.04, 0.04, 0.05, 0.85);
/// Strength fill per kind, player-blue family (team color is blue on the
/// field; the kind letter + shade carries the rest).
pub const KIND_FILL: [Color; NUM_KINDS] = [
    Color::srgb(0.22, 0.42, 0.85), // heavy: deep blue
    Color::srgb(0.42, 0.62, 0.95), // light: pale blue
    Color::srgb(0.30, 0.55, 0.70), // spear: teal blue
    Color::srgb(0.38, 0.50, 0.62), // archer: slate blue
];
/// Broken regiments drain to the broken-flag gray.
const FILL_BROKEN: Color = crate::banners::FLAG_BROKEN[0];
/// Ammo strip: pale straw (arrow shafts).
const AMMO_COLOR: Color = Color::srgb(0.85, 0.78, 0.50);
/// Firing indicator tint: hot amber over the bow pictogram.
const FIRING_COLOR: Color = Color::srgb(1.0, 0.62, 0.22);
const SEL_OUTLINE: Color = Color::srgb(0.95, 0.95, 0.90);

const BTN_ACTIVE: Color = Color::srgba(0.22, 0.38, 0.62, 0.95);
const BTN_ACTIVE_HOVER: Color = Color::srgba(0.30, 0.48, 0.74, 0.95);
const BTN_DISABLED: Color = Color::srgba(0.09, 0.10, 0.12, 0.80);

/// Cards at least this wide (logical px) swap the kind letter for the icon.
const CARD_ICON_MIN_W: f32 = 30.0;
/// Total bar height (56px card + 2x5px padding); overlay.rs derives the
/// inspect plaque's offset from it.
pub const BAR_HEIGHT: f32 = 66.0;

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
                (card_clicks, card_hover, control_buttons).in_set(HudInputSet),
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
        KIND_ARCHER => "A",
        _ => "L",
    }
}

/// Stamina tint per fatigue band: cool while fresh, hot when spent.
fn fatigue_color(fatigue: f32) -> Color {
    use crate::fatigue::FatigueState as F;
    match crate::fatigue::state(fatigue) {
        F::Fresh | F::WarmedUp => Color::srgb(0.30, 0.62, 0.68),
        F::Winded => Color::srgb(0.72, 0.68, 0.30),
        F::Tired => Color::srgb(0.82, 0.55, 0.20),
        F::VeryTired => Color::srgb(0.85, 0.38, 0.15),
        F::Exhausted => Color::srgb(0.70, 0.20, 0.12),
    }
}

fn morale_color(gd: &crate::orders::GroupData) -> Color {
    match gd.state {
        RegState::Routing { .. } => Color::srgb(0.95, 0.20, 0.15),
        RegState::Shattered => Color::srgb(0.35, 0.12, 0.10),
        RegState::Steady => {
            // Green (100) -> yellow (50) -> red (0), two-segment lerp.
            let t = crate::morale::morale01(gd);
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

// ── Icon rasterizer ──
//
// Every HUD pictogram (card kind icons AND button icons) is rendered
// once into a small texture and displayed via ImageNode. Node-composed
// icons round each rect to the pixel grid separately and turn to mush
// at 22px; a supersampled SDF rasterization stays crisp and every
// consumer of a texture is pixel-identical. Authored at 2x the logical
// canvas. Card icons are 20x24 logical (integer physical at
// 125/150/200% display scales, so the quad never rounds to a different
// size per card).

const ICON_TEX_W: u32 = 40;
const ICON_TEX_H: u32 = 48;
/// Button icon texture: 2x the 22x22 logical button canvas.
const BTN_TEX: u32 = 44;
const ICON_RGB: [u8; 3] = [230, 222, 194];
const ICON_RGB_DETAIL: [u8; 3] = [26, 28, 33];

/// A shape in texture pixels with a straight-alpha color.
enum IconShape {
    /// Rounded rect: rect + per-corner radii (tl, tr, br, bl).
    Rect {
        rect: (f32, f32, f32, f32),
        radius: [f32; 4],
        rgb: [u8; 3],
    },
    Tri {
        a: Vec2,
        b: Vec2,
        c: Vec2,
        rgb: [u8; 3],
    },
    /// Capsule: a line segment with thickness (angled shafts).
    Seg {
        a: Vec2,
        b: Vec2,
        r: f32,
        rgb: [u8; 3],
    },
}

fn kind_icon_shapes(kind: u8) -> Vec<IconShape> {
    let s = |rect, radius, rgb| IconShape::Rect { rect, radius, rgb };
    let t = |a: (f32, f32), b: (f32, f32), c: (f32, f32)| IconShape::Tri {
        a: Vec2::new(a.0, a.1),
        b: Vec2::new(b.0, b.1),
        c: Vec2::new(c.0, c.1),
        rgb: ICON_RGB,
    };
    match kind {
        // Knight's helm: dome, dark eye slit, nose bar.
        KIND_HEAVY => vec![
            s((6.0, 6.0, 28.0, 36.0), [14.0, 14.0, 2.0, 2.0], ICON_RGB),
            s((8.0, 20.0, 24.0, 4.0), [0.0; 4], ICON_RGB_DETAIL),
            s((18.0, 20.0, 4.0, 10.0), [0.0; 4], ICON_RGB),
        ],
        // Spear: pointed leaf head on a long thin shaft.
        KIND_SPEAR => vec![
            t((20.0, 0.0), (14.5, 14.0), (25.5, 14.0)),
            s((18.5, 13.0, 3.0, 34.0), [0.0; 4], ICON_RGB),
        ],
        // A longbow at full draw, arrow pointing left: the stave is an
        // arc of segments bulging left, the string a V pulled to the
        // nock, the arrow crossing the middle with head and fletching.
        KIND_ARCHER => {
            let seg = |a: (f32, f32), b: (f32, f32), r: f32| IconShape::Seg {
                a: Vec2::new(a.0, a.1),
                b: Vec2::new(b.0, b.1),
                r,
                rgb: ICON_RGB,
            };
            vec![
                // stave arc, tip to tip
                seg((26.0, 5.0), (19.0, 10.0), 2.0),
                seg((19.0, 10.0), (15.5, 17.0), 2.0),
                seg((15.5, 17.0), (14.5, 24.0), 2.0),
                seg((14.5, 24.0), (15.5, 31.0), 2.0),
                seg((15.5, 31.0), (19.0, 38.0), 2.0),
                seg((19.0, 38.0), (26.0, 43.0), 2.0),
                // string drawn back to the nock
                seg((26.0, 5.0), (33.0, 24.0), 0.9),
                seg((33.0, 24.0), (26.0, 43.0), 0.9),
                // the arrow: shaft, broadhead, fletching
                seg((3.0, 24.0), (33.0, 24.0), 1.3),
                t((0.5, 24.0), (8.0, 20.0), (8.0, 28.0)),
                seg((30.0, 21.0), (34.5, 18.0), 1.2),
                seg((30.0, 27.0), (34.5, 30.0), 1.2),
            ]
        }
        // Longsword, point up: tapered tip, blade with a dark fuller,
        // narrow crossguard, grip, pommel.
        _ => vec![
            t((20.0, 2.0), (16.5, 12.0), (23.5, 12.0)),
            s((16.5, 11.0, 7.0, 18.0), [0.0; 4], ICON_RGB),
            s((19.0, 13.0, 2.0, 14.0), [0.0; 4], ICON_RGB_DETAIL),
            s((11.0, 29.0, 18.0, 3.0), [1.0; 4], ICON_RGB),
            s((18.5, 32.0, 3.0, 9.0), [0.0; 4], ICON_RGB),
            s((17.0, 41.0, 6.0, 5.0), [2.5; 4], ICON_RGB),
        ],
    }
}

/// Signed distance to a rounded rect (y-down; radii tl, tr, br, bl).
fn sd_round_rect(
    px: f32,
    py: f32,
    rect: (f32, f32, f32, f32),
    radius: [f32; 4],
) -> f32 {
    let (x, y, w, h) = rect;
    let (hw, hh) = (w * 0.5, h * 0.5);
    let (qx, qy) = (px - (x + hw), py - (y + hh));
    let r = match (qx > 0.0, qy > 0.0) {
        (false, false) => radius[0],
        (true, false) => radius[1],
        (true, true) => radius[2],
        (false, true) => radius[3],
    }
    .min(hw)
    .min(hh);
    let ax = qx.abs() - (hw - r);
    let ay = qy.abs() - (hh - r);
    let (dx, dy) = (ax.max(0.0), ay.max(0.0));
    (dx * dx + dy * dy).sqrt() + ax.max(ay).min(0.0) - r
}

/// Signed distance to a triangle (Inigo Quilez's formula).
fn sd_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> f32 {
    let (e0, e1, e2) = (b - a, c - b, a - c);
    let (v0, v1, v2) = (p - a, p - b, p - c);
    let pq0 = v0 - e0 * (v0.dot(e0) / e0.length_squared()).clamp(0.0, 1.0);
    let pq1 = v1 - e1 * (v1.dot(e1) / e1.length_squared()).clamp(0.0, 1.0);
    let pq2 = v2 - e2 * (v2.dot(e2) / e2.length_squared()).clamp(0.0, 1.0);
    let s = (e0.x * e2.y - e0.y * e2.x).signum();
    let d = (pq0.length_squared(), s * (v0.x * e0.y - v0.y * e0.x));
    let d = (
        d.0.min(pq1.length_squared()),
        d.1.min(s * (v1.x * e1.y - v1.y * e1.x)),
    );
    let d = (
        d.0.min(pq2.length_squared()),
        d.1.min(s * (v2.x * e2.y - v2.y * e2.x)),
    );
    -d.0.sqrt() * d.1.signum()
}

/// Distance from a point to a line segment.
fn sd_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let (pa, ba) = (p - a, b - a);
    let h = (pa.dot(ba) / ba.length_squared()).clamp(0.0, 1.0);
    (pa - ba * h).length()
}

impl IconShape {
    fn sd(&self, px: f32, py: f32) -> f32 {
        match self {
            IconShape::Rect { rect, radius, .. } => {
                sd_round_rect(px, py, *rect, *radius)
            }
            IconShape::Tri { a, b, c, .. } => {
                sd_triangle(Vec2::new(px, py), *a, *b, *c)
            }
            IconShape::Seg { a, b, r, .. } => {
                sd_segment(Vec2::new(px, py), *a, *b) - r
            }
        }
    }

    fn rgb(&self) -> [u8; 3] {
        match self {
            IconShape::Rect { rgb, .. }
            | IconShape::Tri { rgb, .. }
            | IconShape::Seg { rgb, .. } => *rgb,
        }
    }
}

/// Software rasterization with 1px edge AA, straight-alpha src-over.
fn rasterize_icon(shapes: Vec<IconShape>, tex_w: u32, tex_h: u32) -> Image {
    let (w, h) = (tex_w as usize, tex_h as usize);
    let mut buf = vec![0.0f32; w * h * 4];
    for shape in shapes {
        let src: [f32; 3] = shape.rgb().map(|c| c as f32 / 255.0);
        for py in 0..h {
            for px in 0..w {
                let d = shape.sd(px as f32 + 0.5, py as f32 + 0.5);
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
            width: tex_w,
            height: tex_h,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

fn rasterize_kind_icon(kind: u8) -> Image {
    rasterize_icon(kind_icon_shapes(kind), ICON_TEX_W, ICON_TEX_H)
}

/// A fresh rasterization of a kind pictogram for consumers outside the
/// battle HUD (the Select Units screen shares the exact textures).
pub fn kind_icon_image(kind: u8) -> Image {
    rasterize_kind_icon(kind)
}

// ── Button icon shapes (44x44 texture = 2x the 22px button canvas) ──

/// Heater shield: rounded-top body, sides tapering to a bottom point,
/// with a dark boss.
fn heater_shield(out: &mut Vec<IconShape>, x: f32, y: f32, w: f32, h: f32) {
    let body_h = h * 0.60;
    out.push(IconShape::Rect {
        rect: (x, y, w, body_h),
        radius: [w * 0.22, w * 0.22, 0.0, 0.0],
        rgb: ICON_RGB,
    });
    out.push(IconShape::Tri {
        a: Vec2::new(x, y + body_h - 0.5),
        b: Vec2::new(x + w, y + body_h - 0.5),
        c: Vec2::new(x + w * 0.5, y + h),
        rgb: ICON_RGB,
    });
    let r = w * 0.16;
    out.push(IconShape::Rect {
        rect: (x + w * 0.5 - r, y + h * 0.35 - r, r * 2.0, r * 2.0),
        radius: [r; 4],
        rgb: ICON_RGB_DETAIL,
    });
}

/// Spear with a leaf head, from butt to tip.
fn spear(out: &mut Vec<IconShape>, base: Vec2, tip: Vec2, shaft_r: f32) {
    let d = (tip - base).normalize();
    let perp = Vec2::new(-d.y, d.x);
    let head_base = tip - d * 11.0;
    out.push(IconShape::Seg {
        a: base,
        b: head_base + d,
        r: shaft_r,
        rgb: ICON_RGB,
    });
    out.push(IconShape::Tri {
        a: tip,
        b: head_base + perp * 4.0,
        c: head_base - perp * 4.0,
        rgb: ICON_RGB,
    });
}

fn halt_shapes() -> Vec<IconShape> {
    vec![IconShape::Rect {
        rect: (12.0, 12.0, 20.0, 20.0),
        radius: [4.0; 4],
        rgb: ICON_RGB,
    }]
}

/// Shield wall: three heater shields shoulder to shoulder.
fn wall_shield_shapes() -> Vec<IconShape> {
    let mut out = Vec::new();
    for x in [2.0, 15.5, 29.0] {
        heater_shield(&mut out, x, 8.0, 13.0, 28.0);
    }
    out
}

/// Spear wall: a dense rank of five braced spears over a low line.
/// Content bbox spans x 1..42, y 7..37: centered on the 44x44 canvas.
fn wall_spear_shapes() -> Vec<IconShape> {
    let mut out = Vec::new();
    for bx in [3.0, 10.0, 17.0, 24.0, 31.0] {
        spear(
            &mut out,
            Vec2::new(bx, 35.0),
            Vec2::new(bx + 11.0, 7.0),
            1.8,
        );
    }
    out.push(IconShape::Rect {
        rect: (3.0, 32.0, 36.0, 5.0),
        radius: [2.0; 4],
        rgb: ICON_RGB,
    });
    out
}

/// Loose: 2x3 squares; the two spacing variants toggle at runtime.
fn loose_shapes(tight: bool) -> Vec<IconShape> {
    let (xs, ys): (&[f32], &[f32]) = if tight {
        (&[8.0, 18.0, 28.0], &[13.0, 23.0])
    } else {
        (&[2.0, 18.0, 34.0], &[8.0, 28.0])
    };
    let mut out = Vec::new();
    for &y in ys {
        for &x in xs {
            out.push(IconShape::Rect {
                rect: (x, y, 8.0, 8.0),
                radius: [0.0; 4],
                rgb: ICON_RGB,
            });
        }
    }
    out
}

/// Blob: a loose cluster of men — quincunx, symmetric.
fn blob_shapes() -> Vec<IconShape> {
    [(6.0, 6.0), (30.0, 6.0), (18.0, 18.0), (6.0, 30.0), (30.0, 30.0)]
        .into_iter()
        .map(|(x, y)| IconShape::Rect {
            rect: (x, y, 8.0, 8.0),
            radius: [4.0; 4],
            rgb: ICON_RGB,
        })
        .collect()
}

/// Hold: one planted heater shield.
fn hold_shapes() -> Vec<IconShape> {
    let mut out = Vec::new();
    heater_shield(&mut out, 12.0, 5.0, 20.0, 33.0);
    out
}

/// Fire-at-will: a loosed arrow climbing, fletched tail barbs behind.
fn fire_at_will_shapes() -> Vec<IconShape> {
    vec![
        IconShape::Seg {
            a: Vec2::new(8.0, 36.0),
            b: Vec2::new(32.0, 12.0),
            r: 1.8,
            rgb: ICON_RGB,
        },
        IconShape::Tri {
            a: Vec2::new(38.0, 6.0),
            b: Vec2::new(27.0, 10.0),
            c: Vec2::new(34.0, 17.0),
            rgb: ICON_RGB,
        },
        IconShape::Seg {
            a: Vec2::new(8.0, 36.0),
            b: Vec2::new(8.0, 27.0),
            r: 1.5,
            rgb: ICON_RGB,
        },
        IconShape::Seg {
            a: Vec2::new(8.0, 36.0),
            b: Vec2::new(17.0, 36.0),
            r: 1.5,
            rgb: ICON_RGB,
        },
    ]
}

/// Skirmish: back-step chevrons opening the gap from a contact line.
fn skirmish_shapes() -> Vec<IconShape> {
    vec![
        IconShape::Tri {
            a: Vec2::new(28.0, 8.0),
            b: Vec2::new(28.0, 36.0),
            c: Vec2::new(14.0, 22.0),
            rgb: ICON_RGB,
        },
        IconShape::Tri {
            a: Vec2::new(17.0, 8.0),
            b: Vec2::new(17.0, 36.0),
            c: Vec2::new(3.0, 22.0),
            rgb: ICON_RGB,
        },
        IconShape::Rect {
            rect: (34.0, 8.0, 5.0, 28.0),
            radius: [1.5; 4],
            rgb: ICON_RGB,
        },
    ]
}

/// Rasterized button icon textures. Rebuilt on every battle entry: the
/// whole set is ~10 tiny images and sub-millisecond to draw, and the
/// old handles free themselves when the previous bar despawns.
struct ButtonIcons {
    halt: Handle<Image>,
    wall_shield: Handle<Image>,
    wall_spear: Handle<Image>,
    loose_spread: Handle<Image>,
    loose_tight: Handle<Image>,
    blob: Handle<Image>,
    hold: Handle<Image>,
    fire_at_will: Handle<Image>,
    skirmish: Handle<Image>,
}

impl ButtonIcons {
    fn build(images: &mut Assets<Image>) -> Self {
        let mut add = |shapes| images.add(rasterize_icon(shapes, BTN_TEX, BTN_TEX));
        Self {
            halt: add(halt_shapes()),
            wall_shield: add(wall_shield_shapes()),
            wall_spear: add(wall_spear_shapes()),
            loose_spread: add(loose_shapes(false)),
            loose_tight: add(loose_shapes(true)),
            blob: add(blob_shapes()),
            hold: add(hold_shapes()),
            fire_at_will: add(fire_at_will_shapes()),
            skirmish: add(skirmish_shapes()),
        }
    }
}

// ── Spawn ──

fn spawn_card_bar(
    mut commands: Commands,
    groups: Res<Groups>,
    mut images: ResMut<Assets<Image>>,
) {
    let icons: [Handle<Image>; NUM_KINDS] =
        std::array::from_fn(|kind| images.add(rasterize_kind_icon(kind as u8)));
    let btn_icons = ButtonIcons::build(&mut images);
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
            spawn_control_panel(bar, &btn_icons);
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
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(0.0),
                    top: Val::Px(5.0),
                    width: Val::Percent(100.0),
                    height: Val::Px(3.0),
                    ..default()
                },
                BackgroundColor(fatigue_color(0.0)),
                CardFatigue(g),
            ));
            if kind == KIND_ARCHER {
                card.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Px(0.0),
                        top: Val::Px(9.0),
                        width: Val::Percent(100.0),
                        height: Val::Px(2.0),
                        ..default()
                    },
                    BackgroundColor(AMMO_COLOR),
                    CardAmmo(g),
                ));
                // Firing indicator: the card's own bow icon, mini and
                // amber, tucked under the status strips.
                card.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        right: Val::Px(1.0),
                        top: Val::Px(13.0),
                        width: Val::Px(10.0),
                        height: Val::Px(12.0),
                        ..default()
                    },
                    ImageNode {
                        color: FIRING_COLOR,
                        ..ImageNode::new(icon.clone())
                    },
                    Visibility::Hidden,
                    CardFiring(g),
                ));
            }
            card.spawn((
                Text::new(kind_letter(kind)),
                TextFont {
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(Color::srgba(0.95, 0.95, 0.90, 0.75)),
                CardArt { g, icon: false },
            ));
            card.spawn((
                Node {
                    width: Val::Px(20.0),
                    height: Val::Px(24.0),
                    ..default()
                },
                ImageNode::new(icon),
                CardArt { g, icon: true },
            ));
        });
}

/// M2TW-style round icon buttons, right end of the bar.
fn spawn_control_panel(bar: &mut ChildSpawnerCommands, icons: &ButtonIcons) {
    const BUTTONS: [(ControlButton, &str); 7] = [
        (ControlButton::Halt, "Halt (Backspace)"),
        (ControlButton::Form(FormCmd::Wall), "Shield Wall (F)"),
        (ControlButton::Form(FormCmd::Loose), "Loose (L)"),
        (ControlButton::Form(FormCmd::Blob), "Blob (B)"),
        (ControlButton::Form(FormCmd::Hold), "Hold (H)"),
        (ControlButton::Form(FormCmd::FireAtWill), "Fire at Will (T)"),
        (ControlButton::Form(FormCmd::Skirmish), "Skirmish (K)"),
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
                        let mut text = t.spawn((
                            Text::new(label),
                            TextFont {
                                font_size: FontSize::Px(11.0),
                                ..default()
                            },
                            TextColor(TEXT_COLOR),
                        ));
                        // The Wall label tracks the icon variant
                        // (Shield Wall / Spear Wall).
                        if btn == ControlButton::Form(FormCmd::Wall) {
                            text.insert(WallTooltipText);
                        }
                    });
                    // Icon: a 22x22 quad over the rasterized texture.
                    // Stateful buttons stack both variants; refresh
                    // shows the one matching the selection.
                    let canvas = Node {
                        width: Val::Px(22.0),
                        height: Val::Px(22.0),
                        ..default()
                    };
                    let variants: Option<[&Handle<Image>; 2]> = match btn {
                        ControlButton::Form(FormCmd::Wall) => {
                            Some([&icons.wall_shield, &icons.wall_spear])
                        }
                        ControlButton::Form(FormCmd::Loose) => {
                            Some([&icons.loose_spread, &icons.loose_tight])
                        }
                        _ => None,
                    };
                    if let Some(pair) = variants {
                        b.spawn(canvas).with_children(|c| {
                            for (alt, handle) in
                                [(false, pair[0]), (true, pair[1])]
                            {
                                c.spawn((
                                    Node {
                                        position_type: PositionType::Absolute,
                                        left: Val::Px(0.0),
                                        top: Val::Px(0.0),
                                        width: Val::Px(22.0),
                                        height: Val::Px(22.0),
                                        ..default()
                                    },
                                    ImageNode::new(handle.clone()),
                                    if alt {
                                        Visibility::Hidden
                                    } else {
                                        Visibility::Inherited
                                    },
                                    StatefulIcon { btn, alt },
                                ));
                            }
                        });
                    } else {
                        b.spawn((
                            canvas,
                            ImageNode::new(match btn {
                                ControlButton::Halt => icons.halt.clone(),
                                ControlButton::Form(FormCmd::Blob) => {
                                    icons.blob.clone()
                                }
                                ControlButton::Form(FormCmd::Hold) => {
                                    icons.hold.clone()
                                }
                                ControlButton::Form(FormCmd::FireAtWill) => {
                                    icons.fire_at_will.clone()
                                }
                                ControlButton::Form(FormCmd::Skirmish) => {
                                    icons.skirmish.clone()
                                }
                                _ => unreachable!(),
                            }),
                        ));
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
    mut cues: MessageWriter<crate::audio::UiCue>,
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
        selection.recount(&groups);
        cues.write(crate::audio::UiCue::Select);
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

#[allow(clippy::type_complexity, clippy::too_many_arguments)] // bevy system params
fn refresh_cards(
    groups: Res<Groups>,
    selection: Res<Selection>,
    // Wide-enough cards show the kind icon, narrow ones the letter
    // (which the sliver clip then eats). Indexed by regiment; kept
    // across frames to stay allocation-free.
    mut wide: Local<Vec<bool>>,
    mut cards: Query<(
        &UnitCard,
        &ComputedNode,
        &Interaction,
        &mut BackgroundColor,
        &mut Outline,
    )>,
    mut fills: Query<
        (&CardFill, &mut Node, &mut BackgroundColor),
        (Without<UnitCard>, Without<CardMorale>, Without<CardArt>),
    >,
    mut strips: Query<
        (&CardMorale, &mut BackgroundColor),
        (Without<UnitCard>, Without<CardFill>, Without<CardFatigue>),
    >,
    mut stamina: Query<
        (&CardFatigue, &mut Node, &mut BackgroundColor),
        (Without<UnitCard>, Without<CardFill>, Without<CardMorale>, Without<CardArt>),
    >,
    mut arts: Query<
        (&CardArt, &mut Node),
        (Without<UnitCard>, Without<CardFill>, Without<CardFatigue>),
    >,
    mut ammo_strips: Query<
        (&CardAmmo, &mut Node, &mut BackgroundColor),
        (
            Without<UnitCard>,
            Without<CardFill>,
            Without<CardMorale>,
            Without<CardFatigue>,
            Without<CardArt>,
        ),
    >,
    mut firing_icons: Query<(&CardFiring, &mut Visibility)>,
) {
    wide.clear();
    wide.resize(groups.list.len(), false);

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
        bg.set_if_neq(BackgroundColor(want_bg));
        let selected = selection.regiments.get(card.0).copied().unwrap_or(false);
        outline.set_if_neq(Outline {
            width: Val::Px(2.0),
            offset: Val::Px(0.0),
            color: if selected { SEL_OUTLINE } else { Color::NONE },
        });
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
        bg.set_if_neq(BackgroundColor(if gd.state.is_broken() {
            FILL_BROKEN
        } else {
            KIND_FILL[gd.kind as usize]
        }));
    }

    for (strip, mut bg) in &mut strips {
        let gd = &groups.list[strip.0];
        bg.set_if_neq(BackgroundColor(if gd.count == 0 {
            Color::NONE
        } else {
            morale_color(gd)
        }));
    }

    for (fat, mut node, mut bg) in &mut stamina {
        let gd = &groups.list[fat.0];
        if gd.count == 0 {
            bg.set_if_neq(BackgroundColor(Color::NONE));
            continue;
        }
        // Whole-percent steps keep relayout off the per-frame path
        // (fatigue moves ~0.5/s at most).
        let pct = (100.0 - gd.fatigue).clamp(0.0, 100.0).round();
        let want = Val::Percent(pct);
        if node.width != want {
            node.width = want;
        }
        bg.set_if_neq(BackgroundColor(fatigue_color(gd.fatigue)));
    }

    for (ammo, mut node, mut bg) in &mut ammo_strips {
        let gd = &groups.list[ammo.0];
        if gd.count == 0 {
            bg.set_if_neq(BackgroundColor(Color::NONE));
            continue;
        }
        let full = gd.initial_count as u32 * crate::unit_types::missile::AMMO as u32;
        let pct = (gd.ammo_left as f32 / full.max(1) as f32 * 100.0)
            .clamp(0.0, 100.0)
            .round();
        let want = Val::Percent(pct);
        if node.width != want {
            node.width = want;
        }
        bg.set_if_neq(BackgroundColor(AMMO_COLOR));
    }

    for (firing, mut vis) in &mut firing_icons {
        let gd = &groups.list[firing.0];
        vis.set_if_neq(if gd.firing && gd.count > 0 && !gd.state.is_broken() {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        });
    }

    // Display (not Visibility) so the hidden one leaves the flex layout;
    // it only flips when a card crosses the width threshold.
    for (art, mut node) in &mut arts {
        let want = if wide[art.g] == art.icon {
            Display::Flex
        } else {
            Display::None
        };
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
            ControlButton::Form(cmd) => {
                apply_formation_cmd(*cmd, &selection, &mut groups)
            }
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
    mut icons: Query<(&StatefulIcon, &mut Visibility)>,
    mut tooltips: Query<(&BtnTooltip, &mut Visibility), Without<StatefulIcon>>,
    mut wall_tip_text: Query<&mut Text, With<WallTooltipText>>,
) {
    // One pass over the controllable selection accumulates every
    // all-in-mode flag the buttons need (no per-frame Vec).
    let (mut any, mut all_wall, mut all_loose, mut all_blob, mut all_hold) =
        (false, true, true, true, true);
    let mut all_spear = true;
    let (mut any_archer, mut all_faw, mut all_skirm) = (false, true, true);
    for g in selection.picked_controllable(&groups) {
        let gd = &groups.list[g];
        any = true;
        all_wall &= gd.spacing == FormSpacing::Wall;
        all_loose &= gd.spacing == FormSpacing::Loose;
        all_blob &= gd.shape == FormShape::Blob;
        all_hold &= gd.hold;
        all_spear &= is_spearwall_kind(gd.kind);
        if gd.kind == KIND_ARCHER {
            any_archer = true;
            all_faw &= gd.fire_at_will;
            all_skirm &= gd.skirmish;
        }
    }
    let archer_btn = |btn: ControlButton| {
        matches!(
            btn,
            ControlButton::Form(FormCmd::FireAtWill) | ControlButton::Form(FormCmd::Skirmish)
        )
    };
    let active_of = |btn: ControlButton| match btn {
        ControlButton::Halt => false,
        ControlButton::Form(FormCmd::Wall) => all_wall,
        ControlButton::Form(FormCmd::Loose) => all_loose,
        ControlButton::Form(FormCmd::Blob) => all_blob,
        ControlButton::Form(FormCmd::Hold) => all_hold,
        ControlButton::Form(FormCmd::FireAtWill) => any_archer && all_faw,
        ControlButton::Form(FormCmd::Skirmish) => any_archer && all_skirm,
    };

    let mut hovered: Option<ControlButton> = None;
    for (btn, interaction, mut bg) in &mut query {
        if matches!(interaction, Interaction::Hovered | Interaction::Pressed) {
            hovered = Some(*btn);
        }
        let want = if !any || (archer_btn(*btn) && !any_archer) {
            BTN_DISABLED
        } else {
            match (active_of(*btn), interaction) {
                (_, Interaction::Pressed) => BTN_PRESSED,
                (true, Interaction::Hovered) => BTN_ACTIVE_HOVER,
                (true, Interaction::None) => BTN_ACTIVE,
                (false, Interaction::Hovered) => BTN_HOVER,
                (false, Interaction::None) => BTN_NORMAL,
            }
        };
        bg.set_if_neq(BackgroundColor(want));
    }

    // Stateful icons preview what pressing yields: the Loose icon shows
    // the target spacing, the Wall icon the wall the selection's KIND
    // would form (spearwall when everything picked is spears).
    let wall_spear = any && all_spear;
    for (icon, mut vis) in &mut icons {
        let alt_wanted = match icon.btn {
            ControlButton::Form(FormCmd::Loose) => any && all_loose,
            ControlButton::Form(FormCmd::Wall) => wall_spear,
            _ => false,
        };
        vis.set_if_neq(if icon.alt == alt_wanted {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        });
    }
    let wall_label = if wall_spear {
        "Spear Wall (F)"
    } else {
        "Shield Wall (F)"
    };
    for mut text in &mut wall_tip_text {
        if text.0 != wall_label {
            text.0 = wall_label.to_string();
        }
    }

    for (tip, mut vis) in &mut tooltips {
        vis.set_if_neq(if hovered == Some(tip.0) {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        });
    }
}
