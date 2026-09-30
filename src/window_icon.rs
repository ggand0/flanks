//! The game window's icon, which the Windows taskbar and X11 desktops that
//! read a window's own icon (KDE, Xfce) show while the game runs. GNOME
//! ignores it: its dock takes a running app's icon only from the installed
//! desktop entry whose StartupWMClass matches the window's app name
//! (`flanks`, see cargo-appimage.desktop). macOS takes the app bundle's.

use bevy::asset::RenderAssetUsages;
use bevy::ecs::system::NonSendMarker;
use bevy::image::{CompressedImageFormats, ImageSampler, ImageType};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy::winit::WINIT_WINDOWS;

pub struct WindowIconPlugin;

impl Plugin for WindowIconPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, set_window_icon);
    }
}

/// Frames to wait for the winit window before showing the game window
/// without its icon.
const WAIT_FRAMES: u32 = 30;

/// Set the icon once the primary window's winit window exists, a frame or
/// so after startup, then show the window, which starts hidden so the dock
/// sees the icon when it appears. `NonSendMarker` keeps the system on the
/// main thread, where winit requires window calls.
fn set_window_icon(
    mut window: Query<(Entity, &mut Window), With<PrimaryWindow>>,
    mut frames: Local<u32>,
    mut done: Local<bool>,
    _main_thread: NonSendMarker,
) {
    if *done {
        return;
    }
    let Ok((entity, mut window)) = window.single_mut() else {
        return;
    };
    *frames += 1;
    let set = WINIT_WINDOWS.with_borrow(|windows| {
        let winit_window = windows.get_window(entity)?;
        match icon() {
            Some(icon) => winit_window.set_window_icon(Some(icon)),
            None => warn!("window icon: could not decode assets/flanks_icon_rounded_256.png"),
        }
        Some(())
    });
    if set.is_some() || *frames >= WAIT_FRAMES {
        window.visible = true;
        *done = true;
    }
}

/// The 256 px app icon (rounded, with a margin), built into the binary so
/// it needs no asset path.
fn icon() -> Option<winit::window::Icon> {
    let image = Image::from_buffer(
        include_bytes!("../assets/flanks_icon_rounded_256.png"),
        ImageType::Extension("png"),
        CompressedImageFormats::NONE,
        true,
        ImageSampler::Default,
        RenderAssetUsages::MAIN_WORLD,
    )
    .ok()?;
    let (width, height) = (image.width(), image.height());
    winit::window::Icon::from_rgba(image.data?, width, height).ok()
}
