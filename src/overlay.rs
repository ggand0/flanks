//! FPS + unit count overlay (top-left), plus a periodic FPS log line.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;

use crate::combat::CombatStats;
use crate::movement::SimStats;
use crate::orders::{Groups, RegState, Selection};
use crate::regiments::MoraleReadout;
use crate::render_units::RenderCounts;
use crate::units::Units;

#[derive(Component)]
struct OverlayText;

/// TW-style regiment plaque (bottom-right): name, strength, morale, and
/// the live morale factor breakdown from `MoraleReadout`.
#[derive(Component)]
struct InspectPanel;
#[derive(Component)]
struct InspectText;

pub struct OverlayPlugin;

impl Plugin for OverlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            RenderDiagnosticsPlugin,
        ))
            .add_systems(Startup, (spawn_overlay, spawn_inspect_panel))
            .add_systems(Update, (update_overlay, update_inspect_panel));
    }
}

fn spawn_overlay(mut commands: Commands) {
    commands.spawn((
        Text::new("..."),
        TextFont {
            font_size: FontSize::Px(16.0),
            ..default()
        },
        TextColor(Color::srgb(0.9, 0.9, 0.7)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            left: Val::Px(8.0),
            ..default()
        },
        OverlayText,
    ));
}

fn spawn_inspect_panel(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(10.0),
                bottom: Val::Px(10.0),
                padding: UiRect::all(Val::Px(10.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.07, 0.08, 0.10, 0.88)),
            Visibility::Hidden,
            InspectPanel,
        ))
        .with_children(|p| {
            p.spawn((
                Text::new(""),
                TextFont {
                    font_size: FontSize::Px(15.0),
                    ..default()
                },
                TextColor(Color::srgb(0.92, 0.92, 0.85)),
                InspectText,
            ));
        });
}

/// Show the hovered regiment (enemy first — it doubles as the attack
/// preview), else a lone selected regiment, else hide.
fn update_inspect_panel(
    hover: Res<crate::orders::Hover>,
    selection: Res<Selection>,
    groups: Res<Groups>,
    readout: Res<MoraleReadout>,
    mut panel: Query<&mut Visibility, With<InspectPanel>>,
    mut text: Query<&mut Text, With<InspectText>>,
) {
    let Ok(mut vis) = panel.single_mut() else { return };
    let Ok(mut text) = text.single_mut() else { return };
    let single_sel = (selection.regiments.iter().filter(|s| **s).count() == 1)
        .then(|| selection.regiments.iter().position(|s| *s).unwrap() as u32);
    let Some(g) = hover.enemy.or(hover.own).or(single_sel) else {
        *vis = Visibility::Hidden;
        return;
    };
    let g = g as usize;
    let Some(gd) = groups.list.get(g).filter(|gd| gd.count > 0) else {
        *vis = Visibility::Hidden;
        return;
    };
    *vis = Visibility::Visible;

    let kind = if gd.kind == crate::unit_types::KIND_HEAVY {
        "Heavy Knights"
    } else {
        "Men-at-Arms"
    };
    let team = if gd.team == 0 { "blue" } else { "orange" };
    let state = match gd.state {
        RegState::Steady if gd.charging => "STEADY - CHARGING",
        RegState::Steady if gd.engaged => "STEADY - engaged",
        RegState::Steady => "STEADY",
        RegState::Routing { .. } => "ROUTING",
        RegState::Shattered => "SHATTERED",
    };
    let mut s = format!(
        "{kind} {g} ({team})\n{}/{} men    morale {:>3.0}    {state}\n",
        gd.count,
        gd.initial_count,
        gd.morale.clamp(0.0, 100.0),
    );
    if matches!(gd.state, RegState::Steady) {
        let f = readout.0.get(g).copied().unwrap_or_default();
        s += &format!(
            "casualties      -{:.1}/s\nflanked {:>3.0}%    -{:.1}/s\noutnumbered     -{:.1}/s\nrout nearby     -{:.1}/s\nallies x{}   psych x{:.2}   depletion x{:.2}",
            f.casualties,
            f.flanked01 * 100.0,
            f.flanked,
            f.outnumbered,
            f.contagion,
            f.friends,
            f.psych_mult,
            f.depletion,
        );
        if f.recovering {
            s += "\nrecovering      +3.0/s";
        }
    }
    text.0 = s;
}

#[allow(clippy::too_many_arguments)] // bevy system params
fn update_overlay(
    diagnostics: Res<DiagnosticsStore>,
    units: Res<Units>,
    stats: Res<SimStats>,
    groups: Res<Groups>,
    selection: Res<Selection>,
    combat: Res<CombatStats>,
    render_counts: Res<RenderCounts>,
    outcome: Res<crate::ai::BattleOutcome>,
    mut query: Query<&mut Text, With<OverlayText>>,
    time: Res<Time>,
    mut log_timer: Local<f32>,
) {
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    let frame_ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);

    let banner = match outcome.0 {
        Some(0) => "\n=== VICTORY: the enemy army is broken ===",
        Some(1) => "\n=== DEFEAT: your army is broken ===",
        Some(2) => "\n=== MUTUAL DESTRUCTION ===",
        _ => "",
    };
    for mut text in &mut query {
        text.0 = format!(
            "{fps:>5.0} fps  {frame_ms:.2} ms\n{} units, drawn {} [{}] (frustum culled)\nsim tick: grid {:.2} ms, step {:.2} ms, field {:.2} ms, audit {:.2} ms | sync {:.2} ms\n{} groups ({} engaged, {} broken), {} selected\nblue {} ({} lost, {} fled)  orange {} ({} lost, {} fled){banner}",
            units.len(),
            render_counts.drawn,
            render_counts
                .bucket_drawn
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join("/"),
            stats.grid_ms,
            stats.step_ms,
            stats.field_ms,
            stats.audit_ms,
            render_counts.sync_ms,
            groups.list.len(),
            groups.list.iter().filter(|g| g.engaged).count(),
            groups.list.iter().filter(|g| g.state.is_broken()).count(),
            selection.count_units,
            combat.alive[0],
            combat.kills[0],
            combat.fled[0],
            combat.alive[1],
            combat.kills[1],
            combat.fled[1],
        );
    }

    // Periodic log so FPS is verifiable from a headless-ish run. GPU pass
    // timings are the real cost signal; present rate can be vsync-clamped.
    *log_timer += time.delta_secs();
    if *log_timer >= 2.0 {
        *log_timer = 0.0;
        info!(
            "fps: {fps:.0} ({frame_ms:.2} ms), units: {} (blue {} / orange {}), sim: grid {:.2} step {:.2} field {:.2} audit {:.2} sync {:.2}, hits/tick: {}, drawn: {} [{}], nn min/avg: {:.2}/{:.2}, move avg: {:.3} m/tick",
            units.len(),
            combat.alive[0],
            combat.alive[1],
            stats.grid_ms,
            stats.step_ms,
            stats.field_ms,
            stats.audit_ms,
            render_counts.sync_ms,
            stats.events,
            render_counts.drawn,
            render_counts
                .bucket_drawn
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join("/"),
            stats.nn_min,
            stats.nn_avg,
            stats.move_avg
        );
        for diag in diagnostics.iter() {
            let path = diag.path().as_str();
            if path.starts_with("render/")
                && path.ends_with("elapsed_gpu")
                && let Some(v) = diag.smoothed()
            {
                info!("  gpu {path}: {v:.2} ms");
            }
        }
    }
}
