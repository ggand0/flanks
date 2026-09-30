//! Reading the game's own files. Daily builds read `assets/` from disk,
//! under `util::game_root`. With the `embed_assets` feature, build.rs
//! compiles every file of `assets/` into the executable: Bevy's default
//! asset source serves them from memory, and `read` and `is_file` answer
//! any path inside `assets/` from that copy and never from disk, so a
//! shipped executable needs nothing next to it. Paths outside `assets/`
//! (an `FL_GLB_*` override, the `assets_dev` working copies) always come
//! from disk.

use std::borrow::Cow;
use std::io;
use std::path::Path;

use bevy::prelude::*;

#[cfg(feature = "embed_assets")]
mod embedded {
    include!(concat!(env!("OUT_DIR"), "/embedded_assets.rs"));
}

/// Serves the embedded assets as Bevy's default asset source. Add it
/// before `DefaultPlugins`: a source must be registered before
/// `AssetPlugin` builds, or `AssetPlugin` installs the disk source.
pub struct GameFilesPlugin;

impl Plugin for GameFilesPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(feature = "embed_assets")]
        {
            use bevy::asset::AssetApp;
            use bevy::asset::io::memory::{Dir, MemoryAssetReader};
            use bevy::asset::io::{AssetSourceBuilder, AssetSourceId};
            let dir = Dir::new(std::path::PathBuf::new());
            for (path, bytes) in embedded::FILES {
                dir.insert_asset(Path::new(path), *bytes);
            }
            app.register_asset_source(
                AssetSourceId::Default,
                AssetSourceBuilder::new(move || Box::new(MemoryAssetReader { root: dir.clone() })),
            );
        }
        #[cfg(not(feature = "embed_assets"))]
        let _ = app;
    }
}

/// A file of the game, by its path.
pub fn read(path: &Path) -> io::Result<Cow<'static, [u8]>> {
    #[cfg(feature = "embed_assets")]
    if let Some(rel) = inside_assets(path) {
        return embedded_file(&rel).map(Cow::Borrowed).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("assets/{rel} is not among the embedded assets"))
        });
    }
    std::fs::read(path).map(Cow::Owned)
}

/// A text file of the game, by its path.
pub fn read_to_string(path: &Path) -> io::Result<String> {
    String::from_utf8(read(path)?.into_owned()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Whether the game has a file at `path`.
pub fn is_file(path: &Path) -> bool {
    #[cfg(feature = "embed_assets")]
    if let Some(rel) = inside_assets(path) {
        return embedded_file(&rel).is_some();
    }
    path.is_file()
}

#[cfg(feature = "embed_assets")]
fn embedded_file(rel: &str) -> Option<&'static [u8]> {
    let i = embedded::FILES.binary_search_by(|(path, _)| (*path).cmp(rel)).ok()?;
    Some(embedded::FILES[i].1)
}

#[cfg(feature = "embed_assets")]
fn inside_assets(path: &Path) -> Option<String> {
    relative_to(path, &crate::util::game_root().join("assets"))
}

/// `path` relative to `root` as build.rs keys the embedded files: `/`
/// separators, `.` and `..` resolved. None when it lies outside `root`.
#[cfg(any(feature = "embed_assets", test))]
fn relative_to(path: &Path, root: &Path) -> Option<String> {
    use std::path::Component;
    let mut parts = Vec::new();
    for part in path.strip_prefix(root).ok()?.components() {
        match part {
            Component::Normal(name) => parts.push(name.to_str()?),
            Component::CurDir => {}
            Component::ParentDir => {
                parts.pop()?;
            }
            Component::Prefix(_) | Component::RootDir => return None,
        }
    }
    Some(parts.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_key_like_build_rs() {
        let root = Path::new("game").join("assets");
        let key = |p: &str| relative_to(&root.join(p), &root);
        assert_eq!(key("units/knight.glb").as_deref(), Some("units/knight.glb"));
        assert_eq!(key("units/./spearman.stab.json").as_deref(), Some("units/spearman.stab.json"));
        assert_eq!(key("vegetation/a/../oak_pale.glb").as_deref(), Some("vegetation/oak_pale.glb"));
        assert_eq!(key("../assets_dev/knight/knight.glb"), None);
        assert_eq!(relative_to(Path::new("elsewhere/knight.glb"), &root), None);
    }

    /// Every file under `assets/` is in the executable, byte for byte, and
    /// `read` serves it from there.
    #[cfg(feature = "embed_assets")]
    #[test]
    fn every_asset_is_embedded() {
        fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.file_name().unwrap().to_string_lossy().starts_with('.') {
                    continue;
                }
                if path.is_dir() {
                    walk(&path, out);
                } else {
                    out.push(path);
                }
            }
        }
        let assets = crate::util::game_root().join("assets");
        let mut files = Vec::new();
        walk(&assets, &mut files);
        assert_eq!(files.len(), embedded::FILES.len());
        for file in &files {
            let rel = inside_assets(file).unwrap();
            assert!(matches!(read(file).unwrap(), Cow::Borrowed(_)), "{rel} came from disk");
            assert!(read(file).unwrap() == std::fs::read(file).unwrap(), "{rel} differs from disk");
        }
        assert!(!is_file(&assets.join("units/not_a_model.glb")));
    }
}
