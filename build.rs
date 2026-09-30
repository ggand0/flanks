//! Embeds the FLANKS icon (flanks.rc) in the Windows executable, the icon
//! Explorer and the taskbar show for the file. Other targets skip it.

fn main() {
    println!("cargo:rerun-if-changed=flanks.rc");
    println!("cargo:rerun-if-changed=assets/flanks_icon.ico");
    if std::env::var_os("CARGO_CFG_WINDOWS").is_some() {
        embed_resource::compile("flanks.rc", embed_resource::NONE)
            .manifest_optional()
            .expect("compile flanks.rc");
    }
}
