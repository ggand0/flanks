//! The two shells around one sim. `SimPlugins` holds everything the sim
//! tick needs and registers every sim system; `ViewPlugins` holds the
//! window's side: the drawing, the sound, the camera, the menus and the
//! player's input. The windowed game runs both on bevy's default plugins;
//! the headless run (`--headless`) runs the sim group alone on bevy's
//! minimal plugins, one sim tick per app update, and with `--shots` runs
//! both groups with no window (`headless_render`). The shells differ only
//! in the groups they load, in how time advances and in how the battle
//! starts: headless opens no menu and no deployment.

use bevy::app::PluginGroupBuilder;
use bevy::prelude::*;

/// Everything the sim tick needs, with no window.
pub struct SimPlugins;

impl PluginGroup for SimPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(crate::game_state::GameSimPlugin)
            .add(crate::battle_setup::BattleSetupPlugin)
            .add(crate::terrain::TerrainDataPlugin)
            .add(crate::units::UnitsPlugin)
            .add(crate::scenes::ScenesPlugin)
            .add(crate::morale::MoralePlugin)
            .add(crate::fatigue::FatiguePlugin)
            .add(crate::ai::AiPlugin)
            .add(crate::sim::SimPlugin)
            .add(crate::arrows::ArrowsPlugin)
            .add(crate::orders::OrdersPlugin)
            .add(crate::formation::FormationPlugin)
            .add(crate::frontline::FrontlinePlugin)
            .add(crate::combat::CombatPlugin)
            .add(crate::cli::CliPlugin)
    }
}

/// The window's side of the game: what is drawn, heard and clicked.
pub struct ViewPlugins;

impl PluginGroup for ViewPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(crate::game_state::GameShellPlugin)
            .add(crate::settings::SettingsPlugin)
            .add(crate::window_icon::WindowIconPlugin)
            .add(crate::town::TownPlugin)
            .add(crate::terrain::TerrainPlugin)
            .add(crate::water::WaterPlugin)
            .add(crate::vegetation::VegetationPlugin)
            .add(crate::scenes::path::PathFramesPlugin)
            .add(crate::banners::BannersPlugin)
            .add(crate::audio::BattleAudioPlugin)
            .add(crate::sim::SimViewPlugin)
            .add(crate::arrows::ArrowsViewPlugin)
            .add(crate::orders::OrdersViewPlugin)
            .add(crate::selection::SelectionPlugin)
            .add(crate::formation::FormationViewPlugin)
            .add(crate::frontline::FrontlineViewPlugin)
            .add(crate::render_units::UnitRenderPlugin)
            .add(crate::selection_rings::SelectionRingsPlugin)
            .add(crate::camera::RtsCameraPlugin)
            .add(crate::overlay::OverlayPlugin)
            .add(crate::unit_cards::UnitCardsPlugin)
            .add(crate::balance::BalancePlugin)
            .add(crate::picker::PickerPlugin)
            .add(crate::frame_log::FrameLogPlugin)
            .add(crate::sprite_bake::SpriteBakePlugin)
            .add(crate::shots::ShotsPlugin)
    }
}

/// Every shell's status file: `${XDG_RUNTIME_DIR:-/tmp}/flanks/<pid>`
/// holds the battle's soldier count, written when a battle is set up and
/// removed when the app quits (main.rs, `remove_status`). A script that
/// starts runs beside this process can size them by it.
pub struct StatusFilePlugin;

impl Plugin for StatusFilePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            OnEnter(crate::game_state::GameState::Battle),
            write_status.after(crate::game_state::setup_battle),
        );
    }
}

fn status_path() -> std::path::PathBuf {
    let dir = std::env::var_os("XDG_RUNTIME_DIR").map_or_else(|| "/tmp".into(), std::path::PathBuf::from);
    dir.join("flanks").join(std::process::id().to_string())
}

fn write_status(units: Res<crate::units::Units>) {
    let path = status_path();
    let written = path
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| std::fs::write(&path, format!("{}\n", units.len())));
    if let Err(e) = written {
        warn!("cannot write the status file {}: {e}", path.display());
    }
}

/// The status file goes once the app has quit; there may be none.
pub fn remove_status() {
    let _ = std::fs::remove_file(status_path());
}

/// The sim tick's length: the fixed clock runs at `SIM_HZ`.
pub fn fixed_time() -> Time<Fixed> {
    Time::<Fixed>::from_hz(crate::game_state::SIM_HZ)
}

/// The headless run: the sim group on bevy's minimal plugins. The app
/// loop runs with no wait, and every update moves time on by exactly one
/// tick, so each update runs one sim tick and no wall clock reaches the
/// sim. The battle starts on the launch's first update, with no
/// deployment.
pub fn headless(app: &mut App, task_pools: bevy::app::TaskPoolOptions) {
    let tick = fixed_time().timestep();
    app.add_plugins(
        MinimalPlugins
            .set(TaskPoolPlugin { task_pool_options: task_pools })
            .set(bevy::app::ScheduleRunnerPlugin::run_loop(std::time::Duration::ZERO)),
    )
    .add_plugins((bevy::log::LogPlugin::default(), bevy::state::app::StatesPlugin))
    .insert_resource(crate::game_state::LaunchPlan { autostart: true, deployment: false })
    .add_plugins((SimPlugins, StatusFilePlugin))
    .insert_resource(fixed_time())
    .insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(tick))
    .add_systems(Last, one_tick_per_update)
    .add_systems(
        OnEnter(crate::game_state::GameState::Battle),
        refuse_over_max_men.after(crate::game_state::setup_battle),
    );
}

/// `--max-men N`: a battle of more soldiers than N ends the run before its
/// first tick with exit code 75 (EX_TEMPFAIL in sysexits.h), so a script
/// that starts runs beside a game can tell a refused run from a failed one.
fn refuse_over_max_men(
    units: Res<crate::units::Units>,
    mut run: ResMut<crate::cli::RunState>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(limit) = crate::cli::args().max_men else { return };
    let men = units.len();
    if men > limit {
        info!("[run] refused: {men} men, over the limit of {limit} beside a running game");
        run.end();
        exit.write(AppExit::from_code(75));
    }
}

/// The headless renderer (`--headless --shots N`): the windowed game's
/// sim and view groups on bevy's default plugins with no window and no
/// window event loop, the camera drawing into an image. Time moves as in
/// `headless`, one tick per update, and the battle starts once the
/// ground is loaded, with no deployment.
pub fn headless_render(app: &mut App) {
    let tick = fixed_time().timestep();
    app.insert_resource(crate::game_state::LaunchPlan { autostart: true, deployment: false })
        .insert_resource(crate::game_state::StartHold)
        .add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(std::time::Duration::ZERO))
        .insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(tick))
        .add_systems(
            Update,
            crate::terrain::release_start_when_ground_loaded
                .run_if(resource_exists::<crate::game_state::StartHold>),
        )
        .add_systems(Last, one_tick_per_update);
}

/// The headless loop's promise: once a battle runs, each app update adds
/// exactly one to its clock. A run that breaks it stops with the ticks it
/// saw, since its hashes would no longer stand for the same ticks as the
/// windowed game's.
fn one_tick_per_update(
    clock: Res<crate::game_state::BattleClock>,
    state: Res<State<crate::game_state::GameState>>,
    run: Res<crate::cli::RunState>,
    mut last: Local<Option<(u32, u32)>>,
) {
    // After the last tick the app only waits for its pictures.
    if run.ended() {
        return;
    }
    if *state.get() != crate::game_state::GameState::Battle {
        *last = None;
        return;
    }
    if let Some((battle, tick)) = *last
        && battle == clock.battle
    {
        assert_eq!(
            clock.tick,
            tick + 1,
            "a headless update ran {} sim ticks, not one",
            clock.tick as i64 - tick as i64
        );
    }
    *last = Some((clock.battle, clock.tick));
}
