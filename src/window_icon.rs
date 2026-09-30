//! The game window's icon, which the Windows taskbar and X11 desktops that
//! read a window's own icon (KDE, Xfce) show while the game runs. GNOME
//! ignores it: its dock takes a running app's icon only from the installed
//! desktop entry whose StartupWMClass matches the window's app name
//! (`flanks`, see resources/linux/flanks.desktop). macOS takes the app bundle's.

use bevy::asset::RenderAssetUsages;
use bevy::ecs::system::NonSendMarker;
use bevy::image::{CompressedImageFormats, ImageSampler, ImageType};
use bevy::prelude::*;
use bevy::window::WindowCreated;
use bevy::winit::WINIT_WINDOWS;

pub struct WindowIconPlugin;

impl Plugin for WindowIconPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, set_window_icon.run_if(on_message::<WindowCreated>));
    }
}

/// Set the icon on each window bevy has just created; bevy sends
/// `WindowCreated` once the winit window exists, so this runs once at
/// startup. `NonSendMarker` keeps the system on the main thread, where winit
/// requires window calls.
fn set_window_icon(mut created: MessageReader<WindowCreated>, _main_thread: NonSendMarker) {
    WINIT_WINDOWS.with_borrow(|windows| {
        for created in created.read() {
            let Some(winit_window) = windows.get_window(created.window) else {
                continue;
            };
            match icon() {
                Some(icon) => winit_window.set_window_icon(Some(icon)),
                None => warn!("window icon: could not decode assets/flanks_icon_rounded_256.png"),
            }
        }
    });
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
