//! Small shared helpers.

/// Parse an env-var override, falling back to `default`. The FL_* knobs
/// (unit counts, combat scale, camera pose, ...) all go through here.
pub fn env_or<T: std::str::FromStr + Copy>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

thread_local! {
    static SIM_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Mark the calling thread as the sim tick worker (movement.rs). Called
/// once, when that thread starts.
pub fn mark_sim_worker() {
    SIM_WORKER.with(|w| w.set(true));
}

/// Is this the sim tick worker thread? No bevy system may ever run there.
pub fn on_sim_worker() -> bool {
    SIM_WORKER.with(|w| w.get())
}

/// The task pool scope for every sim kernel a tick job can reach (grid
/// rebuild, integrate). A plain `scope` lets the waiting thread tick the
/// shared executor, which means it can pick up ANY queued task, bevy
/// system tasks included. On the tick worker thread that is fatal:
/// `step_sim` stolen this way parks waiting for the very job its thread
/// is computing (a startup hang about one launch in two, caught with
/// gdb). On that thread the scope waits without ticking the shared
/// executor. Chunk tasks still fan out across every pool worker.
/// Everywhere else this is exactly `ComputeTaskPool::get().scope`.
pub fn sim_scope<'env, F, T>(f: F) -> Vec<T>
where
    F: for<'scope> FnOnce(&'scope bevy::tasks::Scope<'scope, 'env, T>),
    T: Send + 'static,
{
    bevy::tasks::ComputeTaskPool::get().scope_with_executor(!on_sim_worker(), None, f)
}
