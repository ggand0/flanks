//! The command line, read once at the top of `main`.
//!
//!   --ticks N   run the battle for N sim ticks, print the state's hash at
//!               tick N and the run's speed, then quit with success
//!   --seed N    fix every launch-time random choice: the spawn offsets
//!               (what FL_SEED=N does) and the enemy army's roll
//!
//! Every FL_ switch keeps working beside these. An unknown flag prints the
//! usage and quits with exit code 2.

use bevy::prelude::*;

use crate::game_state::{BattleClock, SimSet};

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Args {
    /// Quit once the battle clock reaches this many ticks.
    pub ticks: Option<u32>,
    /// The launch seed; without it the launch-time choices read the clock.
    pub seed: Option<u32>,
}

const USAGE: &str = "usage: flanks [--ticks N] [--seed N]
  --ticks N   run N sim ticks of the battle, print its hash at tick N, then quit
  --seed N    fix the spawn offsets (as FL_SEED=N) and the enemy army's roll";

static ARGS: std::sync::OnceLock<Args> = std::sync::OnceLock::new();

/// Parse the process's arguments and keep them for `args()`. A bad command
/// line prints the usage to stderr and exits with code 2.
pub fn init() {
    match parse(std::env::args().skip(1)) {
        Ok(args) => {
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
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument {flag:?}")),
        }
    }
    Ok(args)
}

/// The run condition of `SimSet`: false once a `--ticks` run has run its
/// ticks, so a frame that owes more ticks runs none past the limit.
pub fn under_tick_limit(clock: Res<BattleClock>) -> bool {
    args().ticks.is_none_or(|n| clock.tick < n)
}

/// True on the tick that ends a `--ticks` run: the tick whose install
/// brings the clock to the limit.
pub fn last_tick(clock: &BattleClock) -> bool {
    args().ticks == Some(clock.tick + 1)
}

/// The wall clock of a `--ticks` run, from the end of its first tick.
#[derive(Resource, Default)]
struct RunTimer(Option<std::time::Instant>);

/// Logs the launch's seeds and quits the app at the tick limit with the
/// run's numbers.
pub struct CliPlugin;

impl Plugin for CliPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, log_seed);
        if args().ticks.is_none() {
            return;
        }
        app.init_resource::<RunTimer>().add_systems(
            FixedUpdate,
            end_at_tick_limit
                .after(crate::game_state::advance_battle_clock)
                .in_set(SimSet),
        );
    }
}

/// The spawn seed in the log, so a run can be repeated with `--seed`. The
/// enemy army's rolls are logged where they are drawn (regiments.rs).
fn log_seed() {
    match args().seed {
        Some(seed) => info!("[seed] spawn seed {seed} from --seed"),
        None => info!(
            "[seed] spawn seed {} from FL_SEED, enemy army rolls from the clock",
            crate::util::env_or("FL_SEED", 0u32)
        ),
    }
}

fn end_at_tick_limit(clock: Res<BattleClock>, mut timer: ResMut<RunTimer>, mut exit: MessageWriter<AppExit>) {
    let Some(limit) = args().ticks else { return };
    let started = *timer.0.get_or_insert_with(std::time::Instant::now);
    if clock.tick < limit {
        return;
    }
    let wall = started.elapsed().as_secs_f64();
    // Timed from the end of the first tick, so over the ticks after it.
    let timed = clock.tick.saturating_sub(1);
    info!(
        "[run] ticks {} wall {wall:.2} s ({} ticks timed) {:.1} ticks/s",
        clock.tick,
        timed,
        if wall > 0.0 { timed as f64 / wall } else { 0.0 },
    );
    exit.write(AppExit::Success);
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
        assert_eq!(parsed("--ticks 900 --seed 1"), Ok(Args { ticks: Some(900), seed: Some(1) }));
        assert!(parsed("--ticks").is_err());
        assert!(parsed("--ticks x").is_err());
        assert!(parsed("--headless").is_err());
    }
}
