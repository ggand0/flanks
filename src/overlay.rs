//! FPS + unit count overlay (top-left), plus a periodic FPS log line.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;

use crate::combat::CombatStats;
use crate::movement::SimStats;
use crate::orders::{Groups, RegState, Selection};
use crate::morale::MoraleReadout;
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

/// Sim ticks run since the last render frame. More than one means the
/// fixed clock is catching up after an overrun frame. The FL_CATCHUP
/// clamp (main.rs) bounds the burst, so a played session should never
/// log more than the clamp allows.
#[derive(Resource, Default)]
struct TicksThisFrame(u32);

fn count_sim_tick(mut ticks: ResMut<TicksThisFrame>) {
    ticks.0 += 1;
}

/// Frame pacing over one log period: frame intervals split by whether
/// the frame carried a sim tick, and the time the fixed tick itself held
/// the frame. The fps average cannot show a pattern that alternates
/// between short and long frames. This does.
#[derive(Resource, Default)]
struct FramePacing {
    plain_ms: Vec<f32>,
    tick_ms: Vec<f32>,
    /// Wall time between FixedFirst and FixedLast, summed per frame.
    fixed_ms: Vec<f32>,
    fixed_start: Option<std::time::Instant>,
    fixed_this_frame: f32,
    last_frame: Option<std::time::Instant>,
}

fn fixed_begin(mut pacing: ResMut<FramePacing>) {
    pacing.fixed_start = Some(std::time::Instant::now());
}

fn fixed_end(mut pacing: ResMut<FramePacing>) {
    if let Some(t0) = pacing.fixed_start.take() {
        pacing.fixed_this_frame += t0.elapsed().as_secs_f32() * 1000.0;
    }
}

/// Runs once per frame in Update, after this frame's fixed ticks: the
/// interval since the last call therefore contains this frame's
/// FixedUpdate, and is filed under this frame's tick count.
fn report_catchup(mut ticks: ResMut<TicksThisFrame>, mut pacing: ResMut<FramePacing>) {
    if ticks.0 > 1 {
        info!("[catchup] {} sim ticks in one frame", ticks.0);
    }
    let now = std::time::Instant::now();
    if let Some(last) = pacing.last_frame.replace(now) {
        let ms = (now - last).as_secs_f32() * 1000.0;
        if ticks.0 > 0 {
            pacing.tick_ms.push(ms);
            let fixed = pacing.fixed_this_frame;
            pacing.fixed_ms.push(fixed);
        } else {
            pacing.plain_ms.push(ms);
        }
    }
    pacing.fixed_this_frame = 0.0;
    ticks.0 = 0;
}

/// Main-thread time per frame in three legs, plus the two copies the
/// render side makes of the instance data. Answers where the CPU frame
/// goes at a given army size, without a tracing build.
#[derive(Resource, Default)]
struct FramePhases {
    first: Option<std::time::Instant>,
    after_fixed: Option<std::time::Instant>,
    last: Option<std::time::Instant>,
    /// First to the end of the fixed loop: input plus every sim tick
    /// this frame ran.
    fixed_leg: Vec<f32>,
    /// End of the fixed loop to Last: Update and PostUpdate (instance
    /// sync, UI, transforms).
    update_leg: Vec<f32>,
    /// Last to the next First: extract into the render world plus the
    /// wait for the render thread to release the previous frame.
    gap_leg: Vec<f32>,
    /// The extract memcpy of the instance buckets (main thread).
    extract: Vec<f32>,
    /// prepare_instance_buffers on the render thread (write_buffer).
    prepare: Vec<f32>,
}

fn phase_first(mut ph: ResMut<FramePhases>) {
    let now = std::time::Instant::now();
    if let Some(last) = ph.last {
        ph.gap_leg.push((now - last).as_secs_f32() * 1000.0);
    }
    ph.first = Some(now);
}

fn phase_after_fixed(mut ph: ResMut<FramePhases>) {
    let now = std::time::Instant::now();
    if let Some(first) = ph.first {
        ph.fixed_leg.push((now - first).as_secs_f32() * 1000.0);
    }
    ph.after_fixed = Some(now);
}

fn phase_last(mut ph: ResMut<FramePhases>) {
    let now = std::time::Instant::now();
    if let Some(after) = ph.after_fixed {
        ph.update_leg.push((now - after).as_secs_f32() * 1000.0);
    }
    ph.last = Some(now);
    let us = |a: &std::sync::atomic::AtomicU32| {
        a.load(std::sync::atomic::Ordering::Relaxed) as f32 / 1000.0
    };
    let e = us(&crate::render_units::EXTRACT_US);
    let p = us(&crate::render_units::PREPARE_US);
    ph.extract.push(e);
    ph.prepare.push(p);
}

/// The fixed-tick clock runs in every state and the phase marks keep
/// their last stamps across the menu, so without a reset the first
/// samples of a battle would carry menu time.
fn reset_frame_stats(mut pacing: ResMut<FramePacing>, mut phases: ResMut<FramePhases>) {
    *pacing = FramePacing::default();
    *phases = FramePhases::default();
}

pub struct OverlayPlugin;

impl Plugin for OverlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            RenderDiagnosticsPlugin,
        ))
            .add_systems(Startup, (spawn_overlay, spawn_inspect_panel))
            .add_systems(
                OnEnter(crate::game_state::GameState::Battle),
                (show_overlay, reset_frame_stats),
            )
            .init_resource::<TicksThisFrame>()
            .init_resource::<FramePacing>()
            .init_resource::<FramePhases>()
            .add_systems(
                First,
                phase_first.run_if(in_state(crate::game_state::GameState::Battle)),
            )
            .add_systems(
                RunFixedMainLoop,
                phase_after_fixed
                    .in_set(bevy::app::RunFixedMainLoopSystems::AfterFixedMainLoop)
                    .run_if(in_state(crate::game_state::GameState::Battle)),
            )
            .add_systems(
                Last,
                phase_last.run_if(in_state(crate::game_state::GameState::Battle)),
            )
            .add_systems(
                FixedUpdate,
                count_sim_tick.in_set(crate::game_state::SimSet),
            )
            .add_systems(FixedFirst, fixed_begin)
            .add_systems(FixedLast, fixed_end)
            .add_systems(
                Update,
                (update_overlay, update_inspect_panel, report_catchup)
                    .run_if(in_state(crate::game_state::GameState::Battle)),
            )
            .add_systems(
                OnEnter(crate::game_state::GameState::Menu),
                hide_overlay,
            )
            .add_systems(
                OnEnter(crate::game_state::GameState::Results),
                hide_overlay,
            );
    }
}

fn show_overlay(mut overlay: Query<&mut Visibility, With<OverlayText>>) {
    for mut vis in &mut overlay {
        *vis = Visibility::Visible;
    }
}

fn hide_overlay(
    mut overlay: Query<&mut Visibility, With<OverlayText>>,
    mut panel: Query<&mut Visibility, (With<InspectPanel>, Without<OverlayText>)>,
) {
    for mut vis in &mut overlay {
        *vis = Visibility::Hidden;
    }
    for mut vis in &mut panel {
        *vis = Visibility::Hidden;
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
        Visibility::Hidden,
        OverlayText,
    ));
}

fn spawn_inspect_panel(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(10.0),
                // Sits above the unit card bar.
                bottom: Val::Px(crate::unit_cards::BAR_HEIGHT + 10.0),
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

    let kind = match gd.kind {
        crate::unit_types::KIND_HEAVY => "Heavy Knights",
        crate::unit_types::KIND_SPEAR => "Spearmen",
        crate::unit_types::KIND_ARCHER => "Archers",
        _ => "Men-at-Arms",
    };
    let team = if gd.team == 0 { "blue" } else { "orange" };
    let state = match gd.state {
        RegState::Steady if gd.charging => "STEADY - CHARGING",
        RegState::Steady if crate::morale::band(gd) == crate::morale::Band::Wavering => {
            "WAVERING"
        }
        RegState::Steady if crate::morale::band(gd) == crate::morale::Band::Shaken => "SHAKEN",
        RegState::Steady if gd.engaged => "STEADY - engaged",
        RegState::Steady => "STEADY",
        RegState::Routing { .. } => "ROUTING",
        RegState::Shattered => "SHATTERED",
    };
    // Formation line: shape/files, spacing mode, engagement stance.
    let formation = match gd.shape {
        crate::formation::FormShape::Blob => "mob".to_string(),
        crate::formation::FormShape::Rect => format!("{} files", gd.files),
    };
    let mode = match crate::formation::wall_kind(gd) {
        1 => "  SHIELDWALL",
        2 => "  SPEARWALL",
        _ if gd.spacing == crate::formation::FormSpacing::Loose => "  loose order",
        _ => "",
    };
    let stance = if gd.hold { "  HOLD POSITION" } else { "" };
    let fat = crate::fatigue::state_name(crate::fatigue::state(gd.fatigue));
    let mut s = format!(
        "{kind} {g} ({team})\n{}/{} men    morale {:+.1}    {state}\n{formation}{mode}{stance}    {fat}\n",
        gd.count, gd.initial_count, gd.morale,
    );
    if gd.count > 0 {
        // Morale is a level now: base + the signed modifier sum. Show
        // every nonzero factor so the player can SEE the state of mind.
        let f = readout.0.get(g).copied().unwrap_or_default();
        s += &format!("base            {:+.1}", f.base);
        for (label, v) in [
            ("casualties", f.casualties),
            ("melee exchange", f.exchange),
            ("flanked", f.flanked),
            ("outnumbered", f.outnumbered),
            ("routing allies", f.contagion),
            ("routing enemies", f.rout_enemies),
            ("disorder", f.disorder),
            ("fatigue", f.fatigue),
            ("allies close", f.support),
            ("no enemy near", f.no_enemy),
            ("braced wall", f.wall),
            ("commander", f.leader),
            ("rout lock", f.rout_lock),
        ] {
            if v.abs() >= 0.05 {
                s += &format!("\n{label:<15} {v:+.1}");
            }
        }
        if f.flanked01 > 0.0 {
            s += &format!("\n(surrounded {:.0}%)", f.flanked01 * 100.0);
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
    mut pacing: ResMut<FramePacing>,
    mut phases: ResMut<FramePhases>,
) {
    // Mean frame time over the diagnostic's history (about two seconds),
    // and the rate that mean implies. The smoothed values chase the
    // latest frame: frames that carry a sim tick are longer than the
    // ones between them, so the readout flickered between two rates.
    let frame_ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.average())
        .unwrap_or(0.0);
    let fps = if frame_ms > 0.0 { 1000.0 / frame_ms } else { 0.0 };

    let banner = match outcome.0 {
        Some(0) => "\n=== VICTORY: the enemy army is broken ===",
        Some(1) => "\n=== DEFEAT: your army is broken ===",
        Some(2) => "\n=== MUTUAL DESTRUCTION ===",
        _ => "",
    };
    for mut text in &mut query {
        text.0 = format!(
            "{fps:>5.0} fps  {frame_ms:.2} ms\n{} units, drawn {} [{}] lod {:?} + {} fallen (frustum culled)\nsim tick: grid {:.2} ms, step {:.2} ms, field {:.2} ms, audit {:.2} ms | sync {:.2} ms\n{} groups ({} engaged, {} broken), {} selected\nblue {} ({} lost, {} fled)  orange {} ({} lost, {} fled){banner}",
            units.len(),
            render_counts.drawn,
            render_counts
                .bucket_drawn
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join("/"),
            render_counts.lod_drawn,
            render_counts.corpses_drawn,
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
            "fps: {fps:.0} ({frame_ms:.2} ms), units: {} (blue {} / orange {}), sim: grid {:.2} step {:.2} field {:.2} audit {:.2} sync {:.2}, hits/tick: {}, drawn: {} [{}], nn min/avg: {:.2}/{:.2}, move avg: {:.3} m/tick, lod: {:?}, fallen drawn: {}",
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
            stats.move_avg,
            render_counts.lod_drawn,
            render_counts.corpses_drawn
        );
        // Frame pacing over this log period, split by whether the frame
        // carried a sim tick. The average above hides a pattern that
        // alternates between short and long frames; this shows it.
        let summary = |ms: &mut Vec<f32>| {
            ms.sort_by(|a, b| a.total_cmp(b));
            let at = |q: f32| ms.get(((ms.len().max(1) - 1) as f32 * q) as usize).copied();
            format!(
                "p50 {:.1} p90 {:.1} max {:.1} (n {})",
                at(0.5).unwrap_or(0.0),
                at(0.9).unwrap_or(0.0),
                at(1.0).unwrap_or(0.0),
                ms.len()
            )
        };
        let pacing = &mut *pacing;
        info!(
            "  frame ms without a tick: {} | with a tick: {} | inside the fixed tick: {}",
            summary(&mut pacing.plain_ms),
            summary(&mut pacing.tick_ms),
            summary(&mut pacing.fixed_ms)
        );
        pacing.plain_ms.clear();
        pacing.tick_ms.clear();
        pacing.fixed_ms.clear();
        let ph = &mut *phases;
        info!(
            "  main thread ms: fixed loop {} | update+post {} | extract+wait render {} || instance copies: extract {} | write_buffer (render thread) {}",
            summary(&mut ph.fixed_leg),
            summary(&mut ph.update_leg),
            summary(&mut ph.gap_leg),
            summary(&mut ph.extract),
            summary(&mut ph.prepare)
        );
        ph.fixed_leg.clear();
        ph.update_leg.clear();
        ph.gap_leg.clear();
        ph.extract.clear();
        ph.prepare.clear();
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
