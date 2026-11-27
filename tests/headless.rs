//! The headless sim run as the built binary: a run repeats itself, the
//! thread count does not change it, and the command line ends it as told.
//! Each test starts the binary with small armies and at most 600 ticks,
//! one process at a time.

use std::process::Command;
use std::sync::Mutex;

/// One flanks process at a time: they share the machine with the game.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

struct Run {
    code: Option<i32>,
    out: String,
}

impl Run {
    /// The `[hash] tick T n N state H` lines, from `[hash]` on.
    fn hashes(&self) -> Vec<&str> {
        self.out.lines().filter_map(|l| l.find("[hash] tick").map(|i| &l[i..])).collect()
    }
}

fn flanks(envs: &[(&str, &str)], args: &[&str]) -> Run {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let out = Command::new(env!("CARGO_BIN_EXE_flanks"))
        .args(args)
        .envs(envs.iter().copied())
        .env("NO_COLOR", "1")
        .output()
        .expect("the flanks binary starts");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    Run { code: out.status.code(), out: text }
}

/// Long enough that both small battles are fighting: the front scene's
/// first men fall before tick 150, the AI battle's near tick 400. Every
/// run also holds the headless shell's own check that each update runs
/// one sim tick (shells.rs `one_tick_per_update`).
const TICKS: &str = "600";

fn headless(envs: &[(&str, &str)]) -> Run {
    let mut all = vec![("FL_HASH", "50")];
    all.extend_from_slice(envs);
    let run = flanks(&all, &["--headless", "--ticks", TICKS, "--seed", "1"]);
    assert_eq!(run.code, Some(0), "the run fails:\n{}", run.out);
    assert!(!run.hashes().is_empty(), "the run prints no hashes:\n{}", run.out);
    run
}

const PATH: &[(&str, &str)] = &[("FL_TEST_PATH", "walls"), ("FL_MAP", "walled_town")];
const AI: &[(&str, &str)] = &[("FL_UNITS", "2000"), ("FL_REG_SIZE", "200")];
const FRONT: &[(&str, &str)] = &[("FL_TEST_FRONT", "1"), ("FL_UNITS", "2000"), ("FL_REG_SIZE", "200")];

fn repeats(envs: &[(&str, &str)]) {
    let (a, b) = (headless(envs), headless(envs));
    assert_eq!(a.hashes(), b.hashes(), "two runs of one battle differ");
}

#[test]
fn a_path_scene_repeats() {
    repeats(PATH);
}

#[test]
fn an_ai_battle_repeats() {
    repeats(AI);
}

#[test]
fn a_scripted_scene_repeats() {
    repeats(FRONT);
}

#[test]
fn the_thread_count_changes_nothing() {
    let default = headless(AI);
    let six = headless(&[AI, &[("FL_THREADS", "6")]].concat());
    assert_eq!(default.hashes(), six.hashes(), "FL_THREADS=6 changes the battle");
}

#[test]
fn the_command_line_ends_the_run_as_told() {
    let run = flanks(AI, &["--headless", "--ticks", "40"]);
    assert_eq!(run.code, Some(0), "{}", run.out);
    assert!(
        run.hashes().last().is_some_and(|h| h.starts_with("[hash] tick 40 ")),
        "no hash for tick 40 without FL_HASH:\n{}",
        run.out
    );
    assert!(run.out.contains("[run] ticks 40 "), "no [run] line:\n{}", run.out);
    assert_eq!(flanks(&[], &["--headless"]).code, Some(2), "--headless without --ticks");
    assert_eq!(flanks(&[], &["--bogus"]).code, Some(2), "an unknown flag");
    assert_eq!(flanks(&[], &["--headless", "--ticks", "9", "--speed", "2"]).code, Some(2), "--speed with --headless");
}

#[test]
fn a_battle_over_the_men_limit_is_refused() {
    let args = ["--headless", "--ticks", "100", "--seed", "1", "--max-men"];
    let refused = flanks(PATH, &[&args[..], &["10"]].concat());
    assert_eq!(refused.code, Some(75), "{}", refused.out);
    assert!(refused.hashes().is_empty(), "a refused run ran ticks:\n{}", refused.out);
    assert!(refused.out.contains("[run] refused: 260 men, over the limit of 10"), "{}", refused.out);
    let under = flanks(PATH, &[&args[..], &["100000"]].concat());
    let plain = flanks(PATH, &args[..5]);
    assert_eq!(under.code, Some(0), "{}", under.out);
    assert_eq!(under.hashes(), plain.hashes(), "the limit changes the run");
}
