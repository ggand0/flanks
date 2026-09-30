//! Small shared helpers.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The folder that holds `assets/`. A packaged build has it next to the
/// executable (a zip or tarball), in `Contents/Resources` of a macOS app
/// bundle, or at the root of a running AppImage (`APPDIR`, set by the
/// AppImage runtime). A run from `target/` falls back to the repository
/// the binary was built from. Never the working directory, so any launch
/// path works.
pub fn game_root() -> &'static Path {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let exe_dir = std::env::current_exe().ok().and_then(|exe| exe.parent().map(Path::to_path_buf));
        packaged_root(exe_dir, std::env::var_os("APPDIR").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
    })
}

/// The first of the packaged layouts that has an `assets` folder: next to
/// the executable, a macOS bundle's `Contents/Resources` (the executable
/// is in `Contents/MacOS`), then the AppImage root.
fn packaged_root(exe_dir: Option<PathBuf>, appimage_root: Option<PathBuf>) -> Option<PathBuf> {
    let bundle_resources = exe_dir.as_ref().map(|dir| dir.join("../Resources"));
    [exe_dir, bundle_resources, appimage_root]
        .into_iter()
        .flatten()
        .find(|dir| dir.join("assets").is_dir())
}

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

/// Mark the calling thread as the sim tick worker (sim/mod.rs). Called
/// once, when that thread starts.
pub fn mark_sim_worker() {
    SIM_WORKER.with(|w| w.set(true));
}

/// Is this the sim tick worker thread? No bevy system may ever run there.
pub fn on_sim_worker() -> bool {
    SIM_WORKER.with(|w| w.get())
}

/// The sim's parallel passes run on Bevy's async compute pool, sized to
/// half the machine (main.rs `task_pool_options`). Read once:
/// `FL_SIM_ASYNC=0` runs them on the compute pool with Bevy's default pool
/// sizes, as before, for A/B runs.
pub fn sim_on_async_pool() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_or("FL_SIM_ASYNC", 1_u32) != 0)
}

/// The task pool scope for every sim kernel a tick job can reach (grid
/// rebuild, density field, integrate).
///
/// The scope runs on Bevy's async compute pool, never on the compute pool.
/// Bevy runs every system of the frame as a task on the compute pool, and
/// any thread waiting inside a pool scope (a parallel query, a system's own
/// scope) runs other queued tasks of that pool while it waits, up to a few
/// hundred before it looks at its own work again. Queued there, the hundred
/// 1 ms pieces of a 200k integrate would run inside the frame's systems,
/// and a frame that overlaps a tick would take up to 15 ms. The async pool
/// is an executor with threads of its own, which the frame's tasks never
/// wait on.
///
/// A plain `scope` lets the waiting thread tick the pool's executor, which
/// means it can pick up any queued task of that pool. On the tick worker
/// thread the scope waits without ticking it, so that thread runs nothing
/// but its job: a bevy system run there could block on the very job its
/// thread computes (a startup hang about one launch in two, caught with gdb).
pub fn sim_scope<'env, F, T>(f: F) -> Vec<T>
where
    F: for<'scope> FnOnce(&'scope bevy::tasks::Scope<'scope, 'env, T>),
    T: Send + 'static,
{
    let pool: &bevy::tasks::TaskPool = if sim_on_async_pool() {
        bevy::tasks::AsyncComputeTaskPool::get()
    } else {
        bevy::tasks::ComputeTaskPool::get()
    };
    pool.scope_with_executor(!on_sim_worker(), None, f)
}

#[cfg(test)]
mod tests {
    use super::packaged_root;
    use std::path::{Path, PathBuf};

    /// A fresh folder under the system temp folder with the given
    /// subfolders created.
    fn layout(name: &str, dirs: &[&str]) -> PathBuf {
        let root = std::env::temp_dir().join(format!("flanks-root-test-{name}-{}", std::process::id()));
        for dir in dirs {
            std::fs::create_dir_all(root.join(dir)).unwrap();
        }
        root
    }

    fn assets_of(found: Option<PathBuf>) -> Option<PathBuf> {
        found.map(|dir| dir.join("assets").canonicalize().unwrap())
    }

    fn canonical(path: &Path) -> Option<PathBuf> {
        Some(path.canonicalize().unwrap())
    }

    /// Each packaged layout finds its own `assets`, and a binary with none
    /// nearby finds nothing, so it falls back to the repository.
    #[test]
    fn packaged_builds_find_their_assets() {
        let zip = layout("zip", &["flanks/assets"]);
        assert_eq!(assets_of(packaged_root(Some(zip.join("flanks")), None)), canonical(&zip.join("flanks/assets")));

        let app = layout("app", &["FLANKS.app/Contents/MacOS", "FLANKS.app/Contents/Resources/assets"]);
        let macos = app.join("FLANKS.app/Contents/MacOS");
        let resources = app.join("FLANKS.app/Contents/Resources/assets");
        assert_eq!(assets_of(packaged_root(Some(macos), None)), canonical(&resources));

        let appdir = layout("appimage", &["mount/usr/bin", "mount/assets"]);
        let found = packaged_root(Some(appdir.join("mount/usr/bin")), Some(appdir.join("mount")));
        assert_eq!(assets_of(found), canonical(&appdir.join("mount/assets")));

        let bare = layout("bare", &["target/release"]);
        assert_eq!(packaged_root(Some(bare.join("target/release")), None), None);

        for dir in [zip, app, appdir, bare] {
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
}
