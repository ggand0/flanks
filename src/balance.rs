//! The balance of power bar (top center), after M2TW's: the player's
//! share of the men still fighting on the left, the enemy's on the
//! right. Hovering it shows each side's losses and M2TW's verdict line.
//! The Battle HUD setting (F1) hides it with the card bar.

use bevy::prelude::*;

use crate::combat::CombatStats;
use crate::game_state::{GameState, TEXT_COLOR};
use crate::orders::{Groups, PLAYER_TEAM};

const BAR_WIDTH: f32 = 320.0;
const BAR_HEIGHT: f32 = 8.0;
const FRAME_PAD: f32 = 3.0;
/// The unit panel's dark glass.
const FRAME_BG: Color = Color::srgba(0.07, 0.08, 0.10, 0.82);
/// The two armies' colours, muted to mid tones: readable at a glance on
/// the dark frame without the saturated team colours shouting.
const PLAYER_FILL: Color = Color::srgb(0.30, 0.46, 0.70);
const ENEMY_FILL: Color = Color::srgb(0.78, 0.48, 0.24);
const TICK: Color = Color::srgba(0.92, 0.92, 0.85, 0.55);

/// M2TW's verdicts (battle.txt), best first. The bar's split picks one
/// by which seventh of the bar it falls in.
const VERDICTS: [&str; 7] = [
    "Victory seems certain. Only a fool could lose this battle",
    "Victory is almost a certainty",
    "Victory is a distinct possibility",
    "The balance of forces is evenly matched",
    "Defeat is a distinct possibility",
    "Defeat is almost a certainty",
    "Defeat seems certain. Only a military genius could win this battle",
];

#[derive(Component)]
struct BalanceRoot;

/// The framed bar: its hover shows the tooltip.
#[derive(Component)]
struct BalanceBar;

#[derive(Component)]
struct PlayerFill;

#[derive(Component)]
struct BalanceTip;

#[derive(Component)]
struct BalanceTipText;

pub struct BalancePlugin;

impl Plugin for BalancePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            OnEnter(GameState::Battle),
            spawn_balance.after(crate::game_state::setup_battle),
        )
        .add_systems(Update, update_balance.run_if(in_state(GameState::Battle)));
    }
}

fn spawn_balance(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(8.0),
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                ..default()
            },
            GlobalZIndex(5),
            DespawnOnExit(GameState::Battle),
            BalanceRoot,
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    width: Val::Px(BAR_WIDTH + 2.0 * FRAME_PAD),
                    height: Val::Px(BAR_HEIGHT + 2.0 * FRAME_PAD),
                    padding: UiRect::all(Val::Px(FRAME_PAD)),
                    ..default()
                },
                BackgroundColor(FRAME_BG),
                // Hover target for the tooltip; also keeps a click on the
                // bar from reaching the map, like the rest of the HUD.
                Interaction::default(),
                BalanceBar,
            ))
            .with_children(|frame| {
                // The enemy's colour is the track, the player's share
                // fills it from the left.
                frame
                    .spawn((
                        Node {
                            width: Val::Percent(100.0),
                            height: Val::Percent(100.0),
                            ..default()
                        },
                        BackgroundColor(ENEMY_FILL),
                    ))
                    .with_children(|track| {
                        track.spawn((
                            Node {
                                width: Val::Percent(50.0),
                                height: Val::Percent(100.0),
                                ..default()
                            },
                            BackgroundColor(PLAYER_FILL),
                            PlayerFill,
                        ));
                    });
                // Even split mark.
                frame.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Percent(50.0),
                        top: Val::Px(1.0),
                        bottom: Val::Px(1.0),
                        width: Val::Px(1.0),
                        ..default()
                    },
                    BackgroundColor(TICK),
                ));
            });
            root.spawn((
                Node {
                    margin: UiRect::top(Val::Px(4.0)),
                    padding: UiRect::axes(Val::Px(10.0), Val::Px(5.0)),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(FRAME_BG),
                Visibility::Hidden,
                BalanceTip,
            ))
            .with_children(|tip| {
                tip.spawn((
                    Text::new(""),
                    TextFont {
                        font_size: FontSize::Px(13.0),
                        ..default()
                    },
                    TextColor(TEXT_COLOR),
                    TextLayout::justify(Justify::Center),
                    BalanceTipText,
                ));
            });
        });
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)] // bevy system params
fn update_balance(
    groups: Res<Groups>,
    stats: Res<CombatStats>,
    settings: Res<crate::settings::Settings>,
    mut root: Query<&mut Visibility, (With<BalanceRoot>, Without<BalanceTip>)>,
    bar: Query<&Interaction, With<BalanceBar>>,
    mut fill: Query<&mut Node, With<PlayerFill>>,
    mut tip: Query<&mut Visibility, (With<BalanceTip>, Without<BalanceRoot>)>,
    mut tip_text: Query<&mut Text, With<BalanceTipText>>,
) {
    let want = if settings.interface.hud { Visibility::Inherited } else { Visibility::Hidden };
    for mut vis in &mut root {
        vis.set_if_neq(want);
    }

    // Men still fighting: alive in a regiment that has not broken. Men
    // at the start: every regiment's spawn strength.
    let mut fighting = [0usize; 2];
    let mut initial = [0usize; 2];
    for gd in &groups.list {
        let t = (gd.team as usize).min(1);
        initial[t] += gd.initial_count;
        if !gd.state.is_broken() {
            fighting[t] += gd.count;
        }
    }
    let p = PLAYER_TEAM as usize;
    let e = 1 - p;
    let total = fighting[p] + fighting[e];
    let share = if total > 0 { fighting[p] as f32 / total as f32 } else { 0.5 };
    let width = Val::Percent(share * 100.0);
    for mut node in &mut fill {
        if node.width != width {
            node.width = width;
        }
    }

    let hovered = bar.iter().any(|i| *i != Interaction::None);
    let tip_vis = if hovered { Visibility::Inherited } else { Visibility::Hidden };
    for mut vis in &mut tip {
        vis.set_if_neq(tip_vis);
    }
    if !hovered {
        return;
    }
    let killed = |t: usize| {
        if initial[t] > 0 { stats.kills[t] as f32 / initial[t] as f32 * 100.0 } else { 0.0 }
    };
    let verdict = VERDICTS[(((1.0 - share) * 7.0) as usize).min(6)];
    let s = format!(
        "Allies killed: {:.0}%    Enemies killed: {:.0}%\n{verdict}",
        killed(p),
        killed(e),
    );
    for mut text in &mut tip_text {
        if text.0 != s {
            text.0 = s.clone();
        }
    }
}
