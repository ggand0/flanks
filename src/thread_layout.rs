//! Where the game's threads run. The sim's threads (the async compute
//! pool, util.rs `sim_scope`, and the tick worker, sim/mod.rs) and the
//! frame's threads (main, render, the compute pool) share the machine's
//! CPUs; these switches place them apart or lower the sim's priority.
//! None of them changes what the sim computes, only where and when.
//!
//! - `FL_MAIN_CPUS=<list>` pins the main thread at the start of `main`,
//!   before Bevy builds its pools, so every thread spawned after it (the
//!   render thread, the compute and file loading pools, the sim's
//!   threads) starts on those CPUs; the sim's leave them only with
//!   `FL_SIM_CPUS`. The pools keep their sizes (`machine_threads`).
//! - `FL_SIM_CPUS=<list>` pins each sim thread to those CPUs as it starts.
//! - `FL_SIM_NICE=<n>` sets each sim thread's nice to n as it starts.
//!
//! A list is CPU numbers and ranges, `0,1,12-13`. Unset, nothing moves.
//! Linux only: elsewhere the switches are read and do nothing.

use std::sync::OnceLock;

/// The CPUs a list names, ascending; None for an unset, empty or
/// unreadable list.
fn cpu_list(text: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b): (usize, usize) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
                cpus.extend(a..=b);
            }
            None => cpus.push(part.parse().ok()?),
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    (!cpus.is_empty()).then_some(cpus)
}

fn env_cpus(key: &str) -> Option<Vec<usize>> {
    cpu_list(&std::env::var(key).ok()?)
}

/// The sim threads' placement, read once.
struct SimPlacement {
    cpus: Option<Vec<usize>>,
    nice: Option<i32>,
}

fn sim_placement() -> &'static SimPlacement {
    static P: OnceLock<SimPlacement> = OnceLock::new();
    P.get_or_init(|| SimPlacement {
        cpus: env_cpus("FL_SIM_CPUS"),
        nice: std::env::var("FL_SIM_NICE").ok().and_then(|v| v.trim().parse().ok()),
    })
}

/// What the main thread was pinned to, for the startup log.
static MAIN_CPUS: OnceLock<Option<Vec<usize>>> = OnceLock::new();

/// The hardware threads the game may use, read before the pin.
static MACHINE_THREADS: OnceLock<usize> = OnceLock::new();

/// Pin the calling thread, the main thread, to `FL_MAIN_CPUS`. Called
/// first thing in `main`.
pub fn pin_main_thread() {
    let _ = MACHINE_THREADS.set(std::thread::available_parallelism().map_or(1, |n| n.get()));
    let cpus = env_cpus("FL_MAIN_CPUS").filter(|c| sys::pin(c));
    let _ = MAIN_CPUS.set(cpus);
}

/// The hardware threads as the game started, before `FL_MAIN_CPUS`: Bevy
/// sizes its pools from the calling thread's CPUs, which the pin narrows,
/// so the pools take their sizes from this instead and the pin moves
/// threads without changing how many there are.
pub fn machine_threads() -> usize {
    *MACHINE_THREADS.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// Place the calling sim thread: `FL_SIM_CPUS`, then `FL_SIM_NICE`.
/// Every sim thread calls it once as it starts.
pub fn sim_thread_start() {
    let p = sim_placement();
    if let Some(cpus) = &p.cpus {
        sys::pin(cpus);
    }
    if let Some(n) = p.nice {
        sys::nice(n);
    }
}

/// The layout in one line for the startup log.
pub fn describe() -> String {
    let list = |c: &Option<Vec<usize>>| match c {
        Some(c) => format!("{c:?}"),
        None => "unpinned".into(),
    };
    let p = sim_placement();
    format!(
        "main cpus {}, sim cpus {}, sim nice {}",
        list(MAIN_CPUS.get().unwrap_or(&None)),
        list(&p.cpus),
        p.nice.map_or("unchanged".into(), |n| n.to_string()),
    )
}

#[cfg(target_os = "linux")]
mod sys {
    // glibc's own declarations, so the game needs no crate for two calls.
    // On Linux a pid or who of 0 names the calling thread, not the process.
    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
        fn setpriority(which: i32, who: u32, prio: i32) -> i32;
    }

    /// glibc's cpu_set_t: 1024 CPUs as bits.
    const WORDS: usize = 1024 / 64;

    /// Pin the calling thread to the CPUs; false when the kernel refuses
    /// (no CPU of the list online, or a number past 1023).
    pub fn pin(cpus: &[usize]) -> bool {
        let mut mask = [0_u64; WORDS];
        for &c in cpus {
            if c >= WORDS * 64 {
                return false;
            }
            mask[c / 64] |= 1 << (c % 64);
        }
        // SAFETY: the mask is a live cpu_set_t-sized buffer for the call.
        unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) == 0 }
    }

    /// Set the calling thread's nice (PRIO_PROCESS = 0).
    pub fn nice(n: i32) -> bool {
        // SAFETY: plain syscall wrapper, no pointers.
        unsafe { setpriority(0, 0, n) == 0 }
    }
}

#[cfg(not(target_os = "linux"))]
mod sys {
    pub fn pin(_cpus: &[usize]) -> bool {
        false
    }

    pub fn nice(_n: i32) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::cpu_list;

    #[test]
    fn cpu_lists_read_numbers_and_ranges() {
        assert_eq!(cpu_list("0,1,12,13"), Some(vec![0, 1, 12, 13]));
        assert_eq!(cpu_list("2-4, 14-16,3"), Some(vec![2, 3, 4, 14, 15, 16]));
        assert_eq!(cpu_list(""), None);
        assert_eq!(cpu_list("a"), None);
    }
}
