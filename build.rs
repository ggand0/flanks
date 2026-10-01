//! Embeds the FLANKS icon in the Windows executable, the icon Explorer and
//! the taskbar show for the file. Other targets skip it. The one-line
//! resource script is written into OUT_DIR, so the repository holds none.
//!
//! With the `embed_assets` feature it also writes the table of every file
//! under `assets/` that src/game_files.rs compiles into the executable.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/flanks_icon.ico");
    let dir = |var: &str| PathBuf::from(env::var(var).expect("set by cargo"));
    if env::var_os("CARGO_FEATURE_EMBED_ASSETS").is_some() {
        embed_assets(&dir("CARGO_MANIFEST_DIR").join("assets"), &dir("OUT_DIR").join("embedded_assets.rs"));
    }
    if env::var_os("CARGO_CFG_WINDOWS").is_none() {
        return;
    }
    // An absolute path with forward slashes: rc.exe and windres both take
    // it, and a backslash would start an escape in the script's string.
    let icon = dir("CARGO_MANIFEST_DIR").join("assets/flanks_icon.ico");
    let icon = icon.to_string_lossy().replace('\\', "/");
    let script = dir("OUT_DIR").join("flanks.rc");
    std::fs::write(&script, format!("1 ICON \"{icon}\"\n")).expect("write flanks.rc");
    embed_resource::compile(&script, embed_resource::NONE)
        .manifest_optional()
        .expect("compile flanks.rc");
}

/// Writes `out`: `FILES`, every file under `assets` as its path relative
/// to `assets` with `/` separators and an `include_bytes!` of it, sorted
/// by path. Refuses a Git LFS pointer, which a checkout without
/// `git lfs pull` holds in place of a model or texture.
fn embed_assets(assets: &Path, out: &Path) {
    // A directory makes cargo rescan everything under it, so an added or
    // removed file reruns this; include_bytes! tracks each file's contents.
    println!("cargo:rerun-if-changed={}", assets.display());
    let mut files = Vec::new();
    collect(assets, assets, &mut files);
    files.sort();
    let mut table = String::from("pub static FILES: &[(&str, &[u8])] = &[\n");
    for (rel, path) in &files {
        let len = std::fs::metadata(path).expect("stat an asset").len();
        if len < 1024 && std::fs::read(path).expect("read an asset").starts_with(b"version https://git-lfs") {
            panic!("assets/{rel} is a Git LFS pointer: run `git lfs pull` before building with embed_assets");
        }
        table.push_str(&format!("    ({rel:?}, include_bytes!({:?})),\n", path.to_string_lossy()));
    }
    table.push_str("];\n");
    std::fs::write(out, table).expect("write embedded_assets.rs");
}

/// Every file under `dir`, skipping dotfiles, with its path relative to
/// `root`.
fn collect(root: &Path, dir: &Path, files: &mut Vec<(String, PathBuf)>) {
    for entry in std::fs::read_dir(dir).expect("list an assets folder") {
        let path = entry.expect("list an assets folder").path();
        if path.file_name().is_some_and(|name| name.to_string_lossy().starts_with('.')) {
            continue;
        }
        if path.is_dir() {
            collect(root, &path, files);
        } else {
            let rel = path.strip_prefix(root).expect("under assets");
            let rel: Vec<_> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
            files.push((rel.join("/"), path));
        }
    }
}
