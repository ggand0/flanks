//! Persistent user settings + the settings modal.
//!
//! Settings live in a plain `Settings` resource, saved as YAML to the
//! platform config dir (Linux: ~/.config/frontline/settings.yaml).
//! Saving is debounced off resource change detection, so every edit
//! path (sliders, toggles, future ones) persists without remembering
//! to call save. Consumers (audio, camera, window) read the resource
//! live; nothing else touches the file.
//!
//! The modal opens from the main menu and from the ESC pause overlay
//! (any button tagged `OpenSettingsButton`). While it is open the
//! shell input systems are gated off via `settings_closed`, so ESC
//! closes the modal instead of toggling pause and clicks cannot fall
//! through to the screen behind.

use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::prelude::*;
use bevy::ui::FocusPolicy;
use bevy::window::{MonitorSelection, PresentMode, PrimaryWindow, WindowMode};
use serde::{Deserialize, Serialize};

use crate::game_state::{BTN_NORMAL, DIM_TEXT_COLOR, GameState, TEXT_COLOR};

// ── Data + persistence ──

#[derive(Resource, Serialize, Deserialize, Clone, PartialEq)]
#[serde(default)]
pub struct Settings {
    pub audio: AudioSettings,
    pub camera: CameraSettings,
    pub video: VideoSettings,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
#[serde(default)]
pub struct AudioSettings {
    /// All linear 0..1. Master scales everything; battle covers beds,
    /// combat one-shots, vox and horns; ui covers the HUD clicks.
    pub master: f32,
    pub battle: f32,
    pub ui: f32,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
#[serde(default)]
pub struct CameraSettings {
    /// Multiplier on the pan speed (keys and screen edge), 0.3..2.0.
    pub pan_speed: f32,
    pub edge_pan: bool,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
#[serde(default)]
pub struct VideoSettings {
    /// Default off: the FPS overlay should show real headroom.
    pub vsync: bool,
    /// Borderless fullscreen on the current monitor.
    pub fullscreen: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            audio: AudioSettings { master: 1.0, battle: 1.0, ui: 1.0 },
            camera: CameraSettings { pan_speed: 1.0, edge_pan: true },
            video: VideoSettings { vsync: false, fullscreen: false },
        }
    }
}

impl Default for AudioSettings {
    fn default() -> Self { Settings::default().audio }
}
impl Default for CameraSettings {
    fn default() -> Self { Settings::default().camera }
}
impl Default for VideoSettings {
    fn default() -> Self { Settings::default().video }
}

fn settings_path() -> Option<std::path::PathBuf> {
    Some(dirs::config_dir()?.join("frontline").join("settings.yaml"))
}

impl Settings {
    /// Load from disk; any failure (missing file, bad YAML) falls back
    /// to defaults. Unknown fields are ignored, missing ones default,
    /// so the file survives schema growth in both directions.
    pub fn load() -> Self {
        let Some(path) = settings_path() else { return Self::default() };
        match std::fs::read_to_string(&path) {
            Ok(text) => match serde_yaml_ng::from_str::<Self>(&text) {
                Ok(mut s) => {
                    s.sanitize();
                    s
                }
                Err(e) => {
                    warn!("settings: failed to parse {}: {e}", path.display());
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }

    fn save(&self) {
        let Some(path) = settings_path() else {
            warn!("settings: no config dir on this platform, not saving");
            return;
        };
        let res = (|| -> std::io::Result<()> {
            std::fs::create_dir_all(path.parent().unwrap())?;
            let yaml = serde_yaml_ng::to_string(self)
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            // Write-then-rename so a crash mid-write can't truncate the
            // existing file.
            let tmp = path.with_extension("yaml.tmp");
            std::fs::write(&tmp, yaml)?;
            std::fs::rename(&tmp, &path)
        })();
        match res {
            Ok(()) => info!("settings: saved {}", path.display()),
            Err(e) => warn!("settings: failed to save {}: {e}", path.display()),
        }
    }

    /// Clamp file-loaded values into sane ranges.
    fn sanitize(&mut self) {
        self.audio.master = self.audio.master.clamp(0.0, 1.0);
        self.audio.battle = self.audio.battle.clamp(0.0, 1.0);
        self.audio.ui = self.audio.ui.clamp(0.0, 1.0);
        self.camera.pan_speed = self.camera.pan_speed.clamp(PAN_SPEED_MIN, PAN_SPEED_MAX);
    }
}

const PAN_SPEED_MIN: f32 = 0.3;
const PAN_SPEED_MAX: f32 = 2.0;

// ── Plugin ──

/// Tag for any button (menu, pause overlay) that opens the modal.
#[derive(Component)]
pub struct OpenSettingsButton;

/// Run condition for shell input systems that must yield to the modal.
pub fn settings_closed(open: Query<(), With<SettingsRoot>>) -> bool {
    open.is_empty()
}

#[derive(Component)]
pub struct SettingsRoot;

#[derive(Component, Clone, Copy, PartialEq)]
enum Slider {
    Master,
    Battle,
    Ui,
    PanSpeed,
}

#[derive(Component, Clone, Copy)]
enum Toggle {
    EdgePan,
    VSync,
    Fullscreen,
}

/// On the slider track button; fill bar and value text are looked up
/// by their own `Slider`-carrying marker components.
#[derive(Component)]
struct SliderTrack;

#[derive(Component)]
struct SliderFill;

#[derive(Component)]
struct SliderValue;

/// Track entity being dragged; cleared on mouse release. Kept as a
/// resource so the drag survives the cursor leaving the track node.
#[derive(Resource, Default)]
struct SliderDrag(Option<(Entity, Slider)>);

/// Volume-slider release feedback: play the UI click at the new volume.
#[derive(Resource)]
struct ClickSound(Handle<AudioSource>);

pub struct SettingsPlugin;

impl Plugin for SettingsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SliderDrag>()
            .add_systems(Startup, load_click_sound)
            .add_systems(
                Update,
                (
                    open_buttons,
                    close_input,
                    slider_drag,
                    toggle_buttons,
                    sync_widgets,
                    apply_video,
                    save_debounced,
                ),
            )
            .add_systems(OnExit(GameState::Menu), close_on_state_change)
            .add_systems(OnExit(GameState::Battle), close_on_state_change);
    }
}

fn load_click_sound(mut commands: Commands, assets: Res<AssetServer>) {
    commands.insert_resource(ClickSound(assets.load("sfx_new/ui_select1.mp3")));
}

// ── Modal UI ──

const PANEL_BG: Color = Color::srgba(0.07, 0.08, 0.10, 0.97);
const BACKDROP: Color = Color::srgba(0.01, 0.02, 0.03, 0.6);
const TRACK_BG: Color = Color::srgb(0.20, 0.21, 0.25);
const FILL_BG: Color = Color::srgb(0.55, 0.58, 0.66);
const TRACK_WIDTH: f32 = 220.0;
const LABEL_WIDTH: f32 = 110.0;
const VALUE_WIDTH: f32 = 52.0;

impl Slider {
    /// Current position as a 0..1 fraction of the track.
    fn frac(self, s: &Settings) -> f32 {
        match self {
            Self::Master => s.audio.master,
            Self::Battle => s.audio.battle,
            Self::Ui => s.audio.ui,
            Self::PanSpeed => {
                (s.camera.pan_speed - PAN_SPEED_MIN) / (PAN_SPEED_MAX - PAN_SPEED_MIN)
            }
        }
    }

    fn set_frac(self, s: &mut Settings, f: f32) {
        let f = f.clamp(0.0, 1.0);
        match self {
            Self::Master => s.audio.master = f,
            Self::Battle => s.audio.battle = f,
            Self::Ui => s.audio.ui = f,
            Self::PanSpeed => {
                // Snap to 0.05 steps so the label reads clean.
                let v = PAN_SPEED_MIN + f * (PAN_SPEED_MAX - PAN_SPEED_MIN);
                s.camera.pan_speed = (v / 0.05).round() * 0.05;
            }
        }
    }

    fn value_label(self, s: &Settings) -> String {
        match self {
            Self::Master => format!("{:.0}%", s.audio.master * 100.0),
            Self::Battle => format!("{:.0}%", s.audio.battle * 100.0),
            Self::Ui => format!("{:.0}%", s.audio.ui * 100.0),
            Self::PanSpeed => format!("{:.2}x", s.camera.pan_speed),
        }
    }
}

impl Toggle {
    fn get(self, s: &Settings) -> bool {
        match self {
            Self::EdgePan => s.camera.edge_pan,
            Self::VSync => s.video.vsync,
            Self::Fullscreen => s.video.fullscreen,
        }
    }

    fn flip(self, s: &mut Settings) {
        match self {
            Self::EdgePan => s.camera.edge_pan = !s.camera.edge_pan,
            Self::VSync => s.video.vsync = !s.video.vsync,
            Self::Fullscreen => s.video.fullscreen = !s.video.fullscreen,
        }
    }

    fn label(self, s: &Settings) -> &'static str {
        let on = self.get(s);
        match self {
            Self::Fullscreen => {
                if on { "Borderless" } else { "Windowed" }
            }
            _ => {
                if on { "On" } else { "Off" }
            }
        }
    }
}

fn section_header(p: &mut ChildSpawnerCommands, label: &str) {
    p.spawn((
        Text::new(label),
        TextFont { font_size: FontSize::Px(13.0), ..default() },
        TextColor(DIM_TEXT_COLOR),
        Node {
            margin: UiRect::new(Val::Px(0.0), Val::Px(0.0), Val::Px(14.0), Val::Px(6.0)),
            ..default()
        },
    ));
}

fn row_label(row: &mut ChildSpawnerCommands, label: &str) {
    row.spawn((
        Text::new(label),
        TextFont { font_size: FontSize::Px(15.0), ..default() },
        TextColor(TEXT_COLOR),
        Node { width: Val::Px(LABEL_WIDTH), ..default() },
    ));
}

fn slider_row(p: &mut ChildSpawnerCommands, label: &str, slider: Slider, s: &Settings) {
    p.spawn(Node {
        flex_direction: FlexDirection::Row,
        align_items: AlignItems::Center,
        margin: UiRect::vertical(Val::Px(5.0)),
        ..default()
    })
    .with_children(|row| {
        row_label(row, label);
        // The track button is taller than the visible bar for a
        // forgiving hit area; the bar and fill are non-interactive
        // children.
        row.spawn((
            Button,
            Node {
                width: Val::Px(TRACK_WIDTH),
                height: Val::Px(22.0),
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::NONE),
            crate::game_state::CustomStyled,
            SliderTrack,
            slider,
        ))
        .with_children(|track| {
            track.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(6.0),
                    ..default()
                },
                BackgroundColor(TRACK_BG),
            ))
            .with_children(|bar| {
                bar.spawn((
                    Node {
                        width: Val::Percent(slider.frac(s) * 100.0),
                        height: Val::Percent(100.0),
                        ..default()
                    },
                    BackgroundColor(FILL_BG),
                    SliderFill,
                    slider,
                ));
            });
        });
        row.spawn((
            Text::new(slider.value_label(s)),
            TextFont { font_size: FontSize::Px(14.0), ..default() },
            TextColor(TEXT_COLOR),
            Node {
                width: Val::Px(VALUE_WIDTH),
                margin: UiRect::left(Val::Px(12.0)),
                ..default()
            },
            SliderValue,
            slider,
        ));
    });
}

fn toggle_row(p: &mut ChildSpawnerCommands, label: &str, toggle: Toggle, s: &Settings) {
    p.spawn(Node {
        flex_direction: FlexDirection::Row,
        align_items: AlignItems::Center,
        margin: UiRect::vertical(Val::Px(5.0)),
        ..default()
    })
    .with_children(|row| {
        row_label(row, label);
        row.spawn((
            Button,
            Node {
                padding: UiRect::axes(Val::Px(16.0), Val::Px(5.0)),
                min_width: Val::Px(110.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(BTN_NORMAL),
            toggle,
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(toggle.label(s)),
                TextFont { font_size: FontSize::Px(14.0), ..default() },
                TextColor(TEXT_COLOR),
            ));
        });
    });
}

fn spawn_modal(commands: &mut Commands, s: &Settings) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(BACKDROP),
            // Swallow clicks so nothing behind the modal reacts.
            FocusPolicy::Block,
            Interaction::None,
            GlobalZIndex(40),
            SettingsRoot,
        ))
        .with_children(|p| {
            p.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::axes(Val::Px(36.0), Val::Px(24.0)),
                    ..default()
                },
                BackgroundColor(PANEL_BG),
            ))
            .with_children(|panel| {
                panel.spawn((
                    Text::new("SETTINGS"),
                    TextFont { font_size: FontSize::Px(30.0), ..default() },
                    TextColor(TEXT_COLOR),
                    Node {
                        margin: UiRect::bottom(Val::Px(4.0)),
                        align_self: AlignSelf::Center,
                        ..default()
                    },
                ));

                section_header(panel, "Audio");
                slider_row(panel, "Master", Slider::Master, s);
                slider_row(panel, "Battle", Slider::Battle, s);
                slider_row(panel, "Interface", Slider::Ui, s);

                section_header(panel, "Camera");
                slider_row(panel, "Pan speed", Slider::PanSpeed, s);
                toggle_row(panel, "Edge pan", Toggle::EdgePan, s);

                section_header(panel, "Video");
                toggle_row(panel, "Window", Toggle::Fullscreen, s);
                toggle_row(panel, "VSync", Toggle::VSync, s);

                panel
                    .spawn((
                        Button,
                        Node {
                            padding: UiRect::axes(Val::Px(32.0), Val::Px(10.0)),
                            margin: UiRect::top(Val::Px(22.0)),
                            align_self: AlignSelf::Center,
                            ..default()
                        },
                        BackgroundColor(BTN_NORMAL),
                        CloseButton,
                    ))
                    .with_children(|b| {
                        b.spawn((
                            Text::new("Back"),
                            TextFont { font_size: FontSize::Px(18.0), ..default() },
                            TextColor(TEXT_COLOR),
                        ));
                    });
            });
        });
}

#[derive(Component)]
struct CloseButton;

// ── Systems ──

fn open_buttons(
    mut commands: Commands,
    query: Query<&Interaction, (Changed<Interaction>, With<OpenSettingsButton>)>,
    open: Query<(), With<SettingsRoot>>,
    settings: Res<Settings>,
) {
    for interaction in &query {
        if *interaction == Interaction::Pressed && open.is_empty() {
            spawn_modal(&mut commands, &settings);
        }
    }
}

/// ESC or the Back button closes the modal. `toggle_pause` is gated on
/// `settings_closed`, and the despawn command lands after this frame's
/// condition checks, so the same ESC press never also toggles pause.
fn close_input(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    back: Query<&Interaction, (Changed<Interaction>, With<CloseButton>)>,
    open: Query<Entity, With<SettingsRoot>>,
    settings: Res<Settings>,
    mut drag: ResMut<SliderDrag>,
) {
    if open.is_empty() {
        return;
    }
    let clicked = back.iter().any(|i| *i == Interaction::Pressed);
    if clicked || keys.just_pressed(KeyCode::Escape) {
        for e in &open {
            commands.entity(e).despawn();
        }
        drag.0 = None;
        // Immediate save on close; the debounce also covers this, but
        // closing is the natural commit point.
        settings.save();
    }
}

fn close_on_state_change(
    mut commands: Commands,
    open: Query<Entity, With<SettingsRoot>>,
    mut drag: ResMut<SliderDrag>,
) {
    for e in &open {
        commands.entity(e).despawn();
    }
    drag.0 = None;
}

/// Press on a track starts a drag; the value tracks the cursor's
/// fraction along the track until the button is released, even if the
/// cursor leaves the node. Geometry comes from the track's computed
/// layout (physical px) against the physical cursor position.
#[allow(clippy::too_many_arguments, clippy::type_complexity)] // bevy system params
fn slider_drag(
    mut commands: Commands,
    buttons: Res<ButtonInput<MouseButton>>,
    started: Query<(Entity, &Interaction, &Slider), (Changed<Interaction>, With<SliderTrack>)>,
    tracks: Query<(&ComputedNode, &UiGlobalTransform), With<SliderTrack>>,
    window: Query<&Window, With<PrimaryWindow>>,
    mut drag: ResMut<SliderDrag>,
    mut settings: ResMut<Settings>,
    click: Option<Res<ClickSound>>,
) {
    for (entity, interaction, slider) in &started {
        if *interaction == Interaction::Pressed {
            drag.0 = Some((entity, *slider));
        }
    }
    let Some((entity, slider)) = drag.0 else { return };

    if !buttons.pressed(MouseButton::Left) {
        drag.0 = None;
        // Release feedback for the volume sliders: hear the new level.
        if slider != Slider::PanSpeed
            && let Some(click) = click
        {
            let vol = settings.audio.master
                * settings.audio.ui
                * crate::audio::env_master();
            commands.spawn((
                AudioPlayer::new(click.0.clone()),
                PlaybackSettings {
                    volume: bevy::audio::Volume::Linear(0.5 * vol),
                    ..PlaybackSettings::DESPAWN
                },
            ));
        }
        return;
    }

    let Ok((node, transform)) = tracks.get(entity) else {
        drag.0 = None;
        return;
    };
    let Ok(win) = window.single() else { return };
    let Some(cursor) = win.cursor_position() else { return };
    let cursor_px = cursor.x * win.scale_factor();
    let width = node.size().x;
    let left = transform.translation.x - width / 2.0;
    let frac = ((cursor_px - left) / width).clamp(0.0, 1.0);
    if (frac - slider.frac(&settings)).abs() > f32::EPSILON {
        slider.set_frac(&mut settings, frac);
    }
}

fn toggle_buttons(
    query: Query<(&Interaction, &Toggle), Changed<Interaction>>,
    mut settings: ResMut<Settings>,
) {
    for (interaction, toggle) in &query {
        if *interaction == Interaction::Pressed {
            toggle.flip(&mut settings);
        }
    }
}

/// Keep fills, value labels and toggle captions in sync with the
/// resource (covers drags, toggles, and any future programmatic edit).
#[allow(clippy::type_complexity)]
fn sync_widgets(
    settings: Res<Settings>,
    mut fills: Query<(&mut Node, &Slider), With<SliderFill>>,
    mut values: Query<(&mut Text, &Slider), (With<SliderValue>, Without<Toggle>)>,
    toggles: Query<(&Children, &Toggle)>,
    mut texts: Query<&mut Text, Without<SliderValue>>,
) {
    if !settings.is_changed() {
        return;
    }
    for (mut node, slider) in &mut fills {
        node.width = Val::Percent(slider.frac(&settings) * 100.0);
    }
    for (mut text, slider) in &mut values {
        text.0 = slider.value_label(&settings);
    }
    for (children, toggle) in &toggles {
        for child in children.iter() {
            if let Ok(mut text) = texts.get_mut(child) {
                text.0 = toggle.label(&settings).to_string();
            }
        }
    }
}

/// Push video settings into the window whenever they change (also fires
/// once on startup insert, which is harmless: it writes what main()
/// already configured).
fn apply_video(
    settings: Res<Settings>,
    mut window: Query<&mut Window, With<PrimaryWindow>>,
) {
    if !settings.is_changed() {
        return;
    }
    let Ok(mut win) = window.single_mut() else { return };
    let mode = window_mode(&settings);
    let present = present_mode(&settings);
    // Window is Changed-detected wholesale; only touch it on real edits.
    if win.mode != mode {
        win.mode = mode;
    }
    if win.present_mode != present {
        win.present_mode = present;
    }
}

pub fn window_mode(s: &Settings) -> WindowMode {
    if s.video.fullscreen {
        WindowMode::BorderlessFullscreen(MonitorSelection::Current)
    } else {
        WindowMode::Windowed
    }
}

pub fn present_mode(s: &Settings) -> PresentMode {
    if s.video.vsync {
        PresentMode::AutoVsync
    } else {
        PresentMode::AutoNoVsync
    }
}

/// Debounced autosave: write 0.8 s after the last edit. Mid-drag frames
/// keep pushing the deadline, so a drag saves once, on release.
fn save_debounced(
    settings: Res<Settings>,
    time: Res<Time<Real>>,
    mut pending: Local<Option<f32>>,
    mut seen_insert: Local<bool>,
) {
    if settings.is_changed() {
        // Skip the initial insert; only user edits schedule a save.
        if !*seen_insert {
            *seen_insert = true;
        } else {
            *pending = Some(0.8);
        }
    }
    if let Some(t) = pending.as_mut() {
        *t -= time.delta_secs();
        if *t <= 0.0 {
            *pending = None;
            settings.save();
        }
    }
}
