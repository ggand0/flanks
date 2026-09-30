//! Embeds the FLANKS icon in the Windows executable, the icon Explorer and
//! the taskbar show for the file. Other targets skip it. The one-line
//! resource script is written into OUT_DIR, so the repository holds none.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/flanks_icon.ico");
    if env::var_os("CARGO_CFG_WINDOWS").is_none() {
        return;
    }
    let dir = |var: &str| PathBuf::from(env::var(var).expect("set by cargo"));
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
