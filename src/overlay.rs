//! FPS + unit count overlay (top-left), plus a periodic FPS log line.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;

use crate::movement::SimStats;
use crate::orders::{Groups, Selection};
use crate::units::Units;

#[derive(Component)]
struct OverlayText;

pub struct OverlayPlugin;

impl Plugin for OverlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            RenderDiagnosticsPlugin,
        ))
            .add_systems(Startup, spawn_overlay)
            .add_systems(Update, update_overlay);
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

fn update_overlay(
    diagnostics: Res<DiagnosticsStore>,
    units: Res<Units>,
    stats: Res<SimStats>,
    groups: Res<Groups>,
    selection: Res<Selection>,
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

    for mut text in &mut query {
        text.0 = format!(
            "{fps:>5.0} fps  {frame_ms:.2} ms\n{} units, 1 unit draw call\nsim tick: grid {:.2} ms, step {:.2} ms\n{} groups, {} selected",
            units.len(),
            stats.grid_ms,
            stats.step_ms,
            groups.list.len(),
            selection.count,
        );
    }

    // Periodic log so FPS is verifiable from a headless-ish run. GPU pass
    // timings are the real cost signal; present rate can be vsync-clamped.
    *log_timer += time.delta_secs();
    if *log_timer >= 2.0 {
        *log_timer = 0.0;
        info!(
            "fps: {fps:.0} ({frame_ms:.2} ms), units: {}, nn min/avg: {:.2}/{:.2}",
            units.len(),
            stats.nn_min,
            stats.nn_avg
        );
        for diag in diagnostics.iter() {
            let path = diag.path().as_str();
            if path.starts_with("render/") && path.ends_with("elapsed_gpu") {
                if let Some(v) = diag.smoothed() {
                    info!("  gpu {path}: {v:.2} ms");
                }
            }
        }
    }
}
