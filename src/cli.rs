//! The command line, read once at the top of `main`.
//!
//!   --ticks N   run the battle for N sim ticks, print the state's hash at
//!               tick N and the run's speed, then quit with success; a
//!               battle decided sooner ends the run one tick after
//!   --seed N    fix every launch-time random choice: the spawn offsets
//!               (what FL_SEED=N does) and the enemy army's roll
//!   --headless  run the sim with no window, as fast as it goes (needs
//!               --ticks); without --seed it takes the seed from FL_SEED
//!   --speed X   the windowed game's time runs X times as fast
//!   --shots N   save a picture every N sim ticks and at the run's last
//!               tick (shots.rs); with --headless the run carries the
//!               renderer, with no window
//!   --shots-dir DIR  the pictures' folder
//!   --max-men N a headless run refuses a battle of more than N soldiers
//!               (exit 75), for runs beside a running game
//!
//! Every FL_ switch keeps working beside these. An unknown flag prints the
//! usage and quits with exit code 2.

use bevy::prelude::*;

use crate::game_state::{BattleClock, SimSet};

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Args {
    /// Quit once the battle clock reaches this many ticks.
    pub ticks: Option<u32>,
    /// The launch seed; without it the launch-time choices read the clock.
    pub seed: Option<u32>,
    /// The sim with no window (shells.rs `headless`).
    pub headless: bool,
    /// The windowed game's time speed.
    pub speed: Option<f64>,
    /// A picture every this many sim ticks.
    pub shots: Option<u32>,
    /// The pictures' folder.
    pub shots_dir: Option<std::path::PathBuf>,
    /// The most soldiers a headless run takes on.
    pub max_men: Option<usize>,
}

const USAGE: &str = "usage: flanks [--ticks N] [--seed N] [--headless [--max-men N] | --speed X] [--shots N [--shots-dir DIR]]
  --ticks N   run N sim ticks of the battle, print its hash at tick N, then quit
  --seed N    fix the spawn offsets (as FL_SEED=N) and the enemy army's roll
  --headless  run the sim with no window as fast as it goes; needs --ticks,
              and without --seed takes its seed from FL_SEED (default 0)
  --speed X   run the windowed game's time X times as fast (X > 0)
  --shots N   save a picture every N sim ticks and at the run's last tick, as
              tick-000150.png and so on; --headless then carries the renderer
  --shots-dir DIR  the pictures' folder (default tmp/runs/shots/<start time>/)
  --max-men N with --headless and no --shots: quit with exit code 75 before the
              first tick when the battle holds more than N soldiers";

static ARGS: std::sync::OnceLock<Args> = std::sync::OnceLock::new();

/// Parse the process's arguments and keep them for `args()`. A bad command
/// line prints the usage to stderr and exits with code 2.
pub fn init() {
    match parse(std::env::args().skip(1)) {
        Ok(mut args) => {
            // A headless run always repeats: its seed is FL_SEED's when no
            // --seed is given, the army rolls included.
            if args.headless && args.seed.is_none() {
                args.seed = Some(crate::util::env_or("FL_SEED", 0u32));
            }
            ARGS.set(args).expect("the command line is read once");
        }
        Err(e) => {
            eprintln!("flanks: {e}\n{USAGE}");
            std::process::exit(2);
        }
    }
}

/// The command line `init` read; the defaults when it has not run (tests).
pub fn args() -> &'static Args {
    ARGS.get_or_init(Args::default)
}

fn parse(mut it: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut args = Args::default();
    while let Some(flag) = it.next() {
        let mut number = |name: &str| -> Result<u32, String> {
            let v = it.next().ok_or_else(|| format!("{name} needs a number"))?;
            v.parse().map_err(|_| format!("{name} needs a whole number, got {v:?}"))
        };
        match flag.as_str() {
            "--ticks" => args.ticks = Some(number("--ticks")?.max(1)),
            "--seed" => args.seed = Some(number("--seed")?),
            "--headless" => args.headless = true,
            "--shots" => args.shots = Some(number("--shots")?.max(1)),
            "--max-men" => args.max_men = Some(number("--max-men")? as usize),
            "--shots-dir" => args.shots_dir = Some(it.next().ok_or("--shots-dir needs a folder")?.into()),
            "--speed" => {
                let v = it.next().ok_or("--speed needs a number")?;
                match v.parse::<f64>() {
                    Ok(x) if x > 0.0 && x.is_finite() => args.speed = Some(x),
                    _ => return Err(format!("--speed needs a number above 0, got {v:?}")),
                }
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument {flag:?}")),
        }
    }
    if args.headless && args.ticks.is_none() {
        return Err("--headless needs --ticks".into());
    }
    if args.max_men.is_some() && (!args.headless || args.shots.is_some()) {
        return Err("--max-men is for a headless run without --shots".into());
    }
    if args.shots_dir.is_some() && args.shots.is_none() {
        return Err("--shots-dir needs --shots".into());
    }
    if args.headless && args.speed.is_some() {
        return Err("--speed is for the windowed game; --headless runs as fast as it goes".into());
    }
    Ok(args)
}

/// The run condition of `SimSet`: false once a `--ticks` run has ended, so
/// a frame that owes more ticks runs none past its end.
pub fn under_tick_limit(clock: Res<BattleClock>, run: Res<RunState>) -> bool {
    !run.ended && args().ticks.is_none_or(|n| clock.tick < n)
}

/// True on the tick that ends a `--ticks` run: the tick whose install
/// brings the clock to the limit, or the tick after the battle was
/// decided.
pub fn last_tick(clock: &BattleClock, outcome: &crate::ai::BattleOutcome) -> bool {
    match args().ticks {
        Some(n) => clock.tick + 1 == n || outcome.0.is_some(),
        None => false,
    }
}

/// A `--ticks` run's wall clock, from the end of its first tick, whether
/// it has ended, and the work it waits for before the app quits (a
/// picture still being written).
#[derive(Resource, Default)]
pub struct RunState {
    started: Option<std::time::Instant>,
    ended: bool,
    pub holds: u32,
}

impl RunState {
    /// The run's last tick has run.
    pub fn ended(&self) -> bool {
        self.ended
    }

    /// The run ends now: no tick runs after this.
    pub fn end(&mut self) {
        self.ended = true;
    }
}

/// Logs the launch's seeds and ends a `--ticks` run with the run's
/// numbers.
pub struct CliPlugin;

impl Plugin for CliPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RunState>().add_systems(Startup, log_seed);
        if args().ticks.is_none() {
            return;
        }
        app.add_systems(
            FixedUpdate,
            end_at_tick_limit
                .after(crate::game_state::advance_battle_clock)
                .in_set(SimSet),
        )
        .add_systems(Last, quit_when_done);
    }
}

/// The spawn seed in the log, so a run can be repeated with `--seed`. The
/// enemy army's rolls are logged where they are drawn (regiments.rs).
fn log_seed() {
    match args().seed {
        Some(seed) => info!("[seed] spawn seed {seed}, enemy army rolls from it (--seed, or FL_SEED headless)"),
        None => info!(
            "[seed] spawn seed {} from FL_SEED, enemy army rolls from the clock",
            crate::util::env_or("FL_SEED", 0u32)
        ),
    }
}

/// Ends a `--ticks` run at its limit, or on the tick after the battle was
/// decided (the victory check runs after the install, so the decided
/// state's hash is the next tick's), with the run's numbers.
pub fn end_at_tick_limit(
    clock: Res<BattleClock>,
    outcome_now: Res<crate::ai::BattleOutcome>,
    mut run: ResMut<RunState>,
    mut decided: Local<crate::game_state::PerBattle<bool>>,
) {
    let Some(limit) = args().ticks else { return };
    let started = *run.started.get_or_insert_with(std::time::Instant::now);
    let decided = decided.of(&clock);
    let over = clock.tick >= limit || *decided;
    *decided = outcome_now.0.is_some();
    if !over {
        return;
    }
    run.ended = true;
    let wall = started.elapsed().as_secs_f64();
    // Timed from the end of the first tick, so over the ticks after it.
    let timed = clock.tick.saturating_sub(1);
    let result = match outcome_now.0 {
        None => "undecided",
        Some(0) => "victory",
        Some(1) => "defeat",
        Some(_) => "mutual destruction",
    };
    info!(
        "[run] ticks {} wall {wall:.2} s ({} ticks timed) {:.1} ticks/s result {result}",
        clock.tick,
        timed,
        if wall > 0.0 { timed as f64 / wall } else { 0.0 },
    );
}

/// Quits once the run has ended and nothing it waits for is pending.
fn quit_when_done(run: Res<RunState>, mut exit: MessageWriter<AppExit>) {
    if run.ended && run.holds == 0 {
        exit.write(AppExit::Success);
    }
}

/// The windowed game's time speed from `--speed`. The fixed tick keeps its
/// length; more of them run per second of the wall clock.
pub fn apply_speed(mut time: ResMut<Time<Virtual>>) {
    if let Some(speed) = args().speed {
        time.set_relative_speed_f64(speed);
        info!("time runs {speed}x (--speed)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(line: &str) -> Result<Args, String> {
        parse(line.split_whitespace().map(String::from))
    }

    #[test]
    fn flags_parse_and_unknown_ones_fail() {
        assert_eq!(parsed(""), Ok(Args::default()));
        assert_eq!(parsed("--ticks 900 --seed 1"), Ok(Args { ticks: Some(900), seed: Some(1), ..default() }));
        assert_eq!(parsed("--headless --ticks 9"), Ok(Args { ticks: Some(9), headless: true, ..default() }));
        assert_eq!(parsed("--speed 4"), Ok(Args { speed: Some(4.0), ..default() }));
        assert!(parsed("--ticks").is_err());
        assert!(parsed("--ticks x").is_err());
        assert!(parsed("--headless").is_err(), "headless needs --ticks");
        assert!(parsed("--headless --ticks 9 --speed 2").is_err(), "speed is windowed only");
        assert!(parsed("--speed 0").is_err());
        assert!(parsed("--speed -1").is_err());
        assert!(parsed("--bogus").is_err());
        assert_eq!(
            parsed("--headless --ticks 9 --shots 3 --shots-dir x"),
            Ok(Args { ticks: Some(9), headless: true, shots: Some(3), shots_dir: Some("x".into()), ..default() })
        );
        assert!(parsed("--shots").is_err());
        assert!(parsed("--shots x").is_err());
        assert!(parsed("--shots-dir x").is_err(), "a folder needs --shots");
        assert_eq!(
            parsed("--headless --ticks 9 --max-men 500"),
            Ok(Args { ticks: Some(9), headless: true, max_men: Some(500), ..default() })
        );
        assert!(parsed("--max-men 500").is_err(), "the limit is for headless runs");
        assert!(parsed("--headless --ticks 9 --shots 3 --max-men 500").is_err(), "not with the renderer");
        assert!(parsed("--headless --ticks 9 --max-men x").is_err());
    }
}
