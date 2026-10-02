//! The debug overlay (top-left, F3) with the periodic FPS log line,
//! and the unit panel (bottom-right, F2).

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;

use crate::combat::CombatStats;
use crate::sim::SimStats;
use crate::orders::{Groups, Selection};
use crate::morale::MoraleReadout;
use crate::render_units::RenderCounts;
use crate::units::Units;

#[derive(Component)]
struct OverlayText;

/// The one-line stats readout, F3's middle state.
#[derive(Component)]
struct StatsLine;

/// A number in the stats line.
#[derive(Component)]
enum StatsValue {
    Fps,
    Soldiers,
    Sim,
}

/// The unit panel (bottom-right), after M2TW's: army, name [class]
/// (men), what the regiment is doing, its morale and fatigue words. The
/// debug overlay adds the morale level, the formation and the live
/// factor breakdown from `MoraleReadout`.
#[derive(Component)]
struct InspectPanel;
#[derive(Component)]
struct InspectTeam;
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
    /// The render thread's frame, first render system to last.
    render: Vec<f32>,
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
    let r = us(&crate::render_units::RENDER_US);
    ph.extract.push(e);
    ph.prepare.push(p);
    ph.render.push(r);
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
            .add_systems(Startup, (spawn_overlay, spawn_stats_line, spawn_inspect_panel))
            .add_systems(
                OnEnter(crate::game_state::GameState::Battle),
                reset_frame_stats,
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
                (show_overlay, update_overlay, update_stats_line, update_inspect_panel, report_catchup)
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

/// The F3 readouts follow the Overlay setting. Only their visibility:
/// `update_overlay` keeps writing the periodic log.
fn show_overlay(
    settings: Res<crate::settings::Settings>,
    mut overlay: Query<&mut Visibility, With<OverlayText>>,
    mut stats_line: Query<&mut Visibility, (With<StatsLine>, Without<OverlayText>)>,
) {
    use crate::settings::Overlay;
    let mode = settings.interface.overlay;
    let shown = |on: bool| if on { Visibility::Visible } else { Visibility::Hidden };
    for mut vis in &mut overlay {
        vis.set_if_neq(shown(mode == Overlay::Full));
    }
    for mut vis in &mut stats_line {
        vis.set_if_neq(shown(mode == Overlay::Stats));
    }
}

#[allow(clippy::type_complexity)]
fn hide_overlay(
    mut overlay: Query<&mut Visibility, With<OverlayText>>,
    mut panel: Query<&mut Visibility, (With<InspectPanel>, Without<OverlayText>)>,
    mut stats_line: Query<
        &mut Visibility,
        (With<StatsLine>, Without<OverlayText>, Without<InspectPanel>),
    >,
) {
    for mut vis in overlay.iter_mut().chain(panel.iter_mut()).chain(stats_line.iter_mut()) {
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

/// The stats line: bright numbers and dim units on a dark pill, so it
/// reads over bright ground and sky in captures.
fn spawn_stats_line(mut commands: Commands) {
    use crate::game_state::TEXT_COLOR;
    // Lighter than the menu's dim grey, which is lost on the pill.
    const UNIT_COLOR: Color = Color::srgb(0.74, 0.74, 0.69);
    let font = TextFont {
        font_size: FontSize::Px(16.0),
        ..default()
    };
    let number = |value| (TextSpan::default(), font.clone(), TextColor(TEXT_COLOR), value);
    let unit = |text: &str| (TextSpan::new(text), font.clone(), TextColor(UNIT_COLOR));
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(8.0),
                left: Val::Px(8.0),
                padding: UiRect::axes(Val::Px(12.0), Val::Px(5.0)),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
            Visibility::Hidden,
            StatsLine,
        ))
        .with_children(|pill| {
            pill.spawn((Text::default(), font.clone(), TextColor(TEXT_COLOR)))
                .with_children(|t| {
                    t.spawn(number(StatsValue::Fps));
                    t.spawn(unit(" fps | "));
                    t.spawn(number(StatsValue::Soldiers));
                    t.spawn(unit(" soldiers | sim "));
                    t.spawn(number(StatsValue::Sim));
                    t.spawn(unit(" ms"));
                });
        });
}

/// The stats line's numbers: the full overlay's fps, soldiers alive on
/// both sides, and the sim tick (its grid, step and field phases).
/// Soldiers add up the units' living counts: those start at full
/// strength, so the number is right in deployment, before the sim has
/// filled `CombatStats`. The font is monospaced, so fps and sim are
/// right-aligned in a fixed width: a value crossing 10 or 100 no longer
/// resizes the pill every update. The soldier count loses a digit at most
/// twice a battle.
fn update_stats_line(
    settings: Res<crate::settings::Settings>,
    diagnostics: Res<DiagnosticsStore>,
    groups: Res<Groups>,
    stats: Res<SimStats>,
    mut spans: Query<(&mut TextSpan, &StatsValue)>,
) {
    if settings.interface.overlay != crate::settings::Overlay::Stats {
        return;
    }
    let (_, fps) = frame_rate(&diagnostics);
    for (mut span, value) in &mut spans {
        let text = match value {
            StatsValue::Fps => format!("{fps:>3.0}"),
            StatsValue::Soldiers => thousands(groups.list.iter().map(|g| g.count).sum()),
            StatsValue::Sim => format!("{:>4.1}", stats.grid_ms + stats.step_ms + stats.field_ms),
        };
        if span.0 != text {
            span.0 = text;
        }
    }
}

/// 199640 as "199,640".
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Mean frame time in ms over the diagnostic's history (Bevy's default,
/// the last 120 frames: under a second at 144 fps), and the rate that mean
/// implies, frames over elapsed time. The smoothed values chase the latest
/// frame: frames that carry a sim tick are longer than the ones between
/// them, so a smoothed readout flickered between two rates.
fn frame_rate(diagnostics: &DiagnosticsStore) -> (f64, f64) {
    let frame_ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.average())
        .unwrap_or(0.0);
    (frame_ms, if frame_ms > 0.0 { 1000.0 / frame_ms } else { 0.0 })
}

/// The panel's army line takes the team's colour, muted to sit on the
/// dark panel.
const TEAM_TEXT: [Color; 2] = [Color::srgb(0.55, 0.72, 0.95), Color::srgb(0.95, 0.66, 0.40)];

fn spawn_inspect_panel(mut commands: Commands) {
    let font = || TextFont {
        font_size: FontSize::Px(15.0),
        ..default()
    };
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(10.0),
                bottom: Val::Px(crate::unit_cards::BAR_HEIGHT + 10.0),
                padding: UiRect::axes(Val::Px(12.0), Val::Px(8.0)),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(Color::srgba(0.07, 0.08, 0.10, 0.82)),
            Visibility::Hidden,
            InspectPanel,
        ))
        .with_children(|p| {
            p.spawn((Text::new(""), font(), TextColor(TEAM_TEXT[0]), InspectTeam));
            p.spawn((
                Text::new(""),
                font(),
                TextColor(Color::srgb(0.92, 0.92, 0.85)),
                InspectText,
            ));
        });
}

/// What the regiment is doing, in M2TW's words (battle.txt).
fn action_word(gd: &crate::orders::GroupData) -> &'static str {
    if gd.state.is_broken() {
        "Routing"
    } else if gd.charging {
        "Charging"
    } else if gd.engaged {
        "Fighting"
    } else if gd.firing {
        "Firing missiles"
    } else if gd.order.is_some() {
        "Marching"
    } else if gd.celebrate > 0 {
        "Taunting"
    } else {
        "Idle"
    }
}

/// Show the hovered regiment (enemy first: it doubles as the attack
/// preview), else a lone selected regiment, else hide. The Unit panel
/// setting (F2) hides it; without the battle HUD (F1) it drops to the
/// screen's edge.
#[allow(clippy::too_many_arguments, clippy::type_complexity)] // bevy system params
fn update_inspect_panel(
    hover: Res<crate::orders::Hover>,
    selection: Res<Selection>,
    groups: Res<Groups>,
    readout: Res<MoraleReadout>,
    settings: Res<crate::settings::Settings>,
    mut panel: Query<(&mut Visibility, &mut Node), With<InspectPanel>>,
    mut team: Query<(&mut Text, &mut TextColor), (With<InspectTeam>, Without<InspectText>)>,
    mut text: Query<&mut Text, (With<InspectText>, Without<InspectTeam>)>,
) {
    let Ok((mut vis, mut node)) = panel.single_mut() else { return };
    let Ok((mut team_text, mut team_color)) = team.single_mut() else { return };
    let Ok(mut text) = text.single_mut() else { return };
    let ui = &settings.interface;
    let bottom = Val::Px(if ui.hud { crate::unit_cards::BAR_HEIGHT + 10.0 } else { 10.0 });
    if node.bottom != bottom {
        node.bottom = bottom;
    }
    let single_sel = (selection.regiments.iter().filter(|s| **s).count() == 1)
        .then(|| selection.regiments.iter().position(|s| *s).unwrap() as u32);
    let shown = ui
        .unit_panel
        .then(|| hover.enemy.or(hover.own).or(single_sel))
        .flatten()
        .map(|g| g as usize)
        .filter(|&g| groups.list.get(g).is_some_and(|gd| gd.count > 0));
    let Some(g) = shown else {
        vis.set_if_neq(Visibility::Hidden);
        return;
    };
    vis.set_if_neq(Visibility::Visible);
    let gd = &groups.list[g];

    let t = (gd.team as usize).min(1);
    let army = if t == 0 { "Blue" } else { "Orange" };
    if team_text.0 != army {
        team_text.0 = army.to_string();
    }
    team_color.set_if_neq(TextColor(TEAM_TEXT[t]));
    let fat = crate::fatigue::state_name(crate::fatigue::state(gd.fatigue));
    let mut s = format!(
        "{} [{}] ({})\n{}\n{}\n{fat}",
        crate::unit_types::kind_name(gd.kind),
        crate::unit_types::kind_class(gd.kind),
        gd.count,
        action_word(gd),
        crate::morale::state_word(gd),
    );
    if ui.debug_overlay() {
        // Formation line: shape/files, spacing mode, engagement stance.
        let formation = match gd.shape {
            crate::formation::FormShape::Blob => "mob".to_string(),
            crate::formation::FormShape::Rect => format!("{} files", gd.files),
        };
        let mode = match crate::formation::wall_kind(gd) {
            1 => "  shieldwall",
            2 => "  spearwall",
            _ if gd.spacing == crate::formation::FormSpacing::Loose => "  loose order",
            _ => "",
        };
        let stance = if gd.hold { "  hold position" } else { "" };
        s += &format!(
            "\n\nregiment {g}  {}/{} men  morale {:+.1}\n{formation}{mode}{stance}",
            gd.count, gd.initial_count, gd.morale,
        );
        // Morale is a level: base + the signed modifier sum. Every
        // nonzero factor, so the state of mind can be read.
        let f = readout.0.get(g).copied().unwrap_or_default();
        s += &format!("\nbase            {:+.1}", f.base);
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
    if text.0 != s {
        text.0 = s;
    }
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
    let (frame_ms, fps) = frame_rate(&diagnostics);

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
            "  main thread ms: fixed loop {} | update+post {} | extract+wait render {} || instance copies: extract {} | write_buffer (render thread) {} || render thread frame {}",
            summary(&mut ph.fixed_leg),
            summary(&mut ph.update_leg),
            summary(&mut ph.gap_leg),
            summary(&mut ph.extract),
            summary(&mut ph.prepare),
            summary(&mut ph.render)
        );
        ph.fixed_leg.clear();
        ph.update_leg.clear();
        ph.gap_leg.clear();
        ph.extract.clear();
        ph.prepare.clear();
        ph.render.clear();
        for diag in diagnostics.iter() {
            let path = diag.path().as_str();
            if !path.starts_with("render/") {
                continue;
            }
            let Some(v) = diag.smoothed() else {
                continue;
            };
            // Per pass, from pipeline statistics queries on devices that have
            // them: vertices shaded, triangles set up, fragments shaded.
            let counted = ["vertex_shader_invocations", "clipper_invocations", "fragment_shader_invocations"]
                .iter()
                .any(|c| path.ends_with(c));
            if path.ends_with("elapsed_gpu") {
                info!("  gpu {path}: {v:.2} ms");
            } else if counted && v >= 1e4 {
                info!("  gpu {path}: {:.2} M", v / 1e6);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::thousands;

    #[test]
    fn thousands_groups_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1000), "1,000");
        assert_eq!(thousands(199_640), "199,640");
        assert_eq!(thousands(1_000_000), "1,000,000");
    }
}
