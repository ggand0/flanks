//! `--shots N`: a picture of the battle every N sim ticks and at the run's
//! last tick, counted on the battle clock, so the windowed game and the
//! headless renderer save the same moments. A picture named tick T shows
//! tick T: once tick T is due, no later tick runs before the frame that
//! draws it, and that frame draws the soldiers where tick T put them, not
//! partway from the tick before. The run quits only once every picture
//! is written.

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk};

use crate::game_state::{BattleClock, SimSet};

/// The pictures' folder and the tick whose picture this frame draws.
#[derive(Resource)]
pub struct Shots {
    every: u32,
    dir: std::path::PathBuf,
    due: Option<u32>,
    last: Option<u32>,
}

impl Shots {
    /// How far between the tick before and this tick the frame draws the
    /// soldiers: the fixed clock's overstep, or the whole way on a
    /// picture's frame, where they stand as the tick left them.
    pub fn render_alpha(shots: Option<&Self>, fixed: &Time<Fixed>) -> f32 {
        match shots {
            Some(s) if s.due.is_some() => 1.0,
            _ => fixed.overstep_fraction(),
        }
    }
}

/// The image the camera draws into when there is no window: the headless
/// renderer's picture.
#[derive(Resource)]
pub struct OffscreenTarget(pub Handle<Image>);

pub struct ShotsPlugin;

impl Plugin for ShotsPlugin {
    fn build(&self, app: &mut App) {
        let args = crate::cli::args();
        let Some(every) = args.shots else { return };
        let dir = args.shots_dir.clone().unwrap_or_else(|| {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            crate::util::game_root().join("tmp/runs/shots").join(stamp.to_string())
        });
        if let Err(e) = std::fs::create_dir_all(&dir) {
            error!("--shots: cannot make {}: {e}", dir.display());
        }
        info!("--shots: a picture every {every} ticks into {}", dir.display());
        app.insert_resource(Shots { every, dir, due: None, last: None })
            .configure_sets(FixedUpdate, SimSet.run_if(no_picture_waiting))
            .add_systems(
                FixedUpdate,
                mark_due
                    .after(crate::game_state::advance_battle_clock)
                    .after(crate::cli::end_at_tick_limit)
                    .in_set(SimSet),
            )
            .add_systems(PostUpdate, take_picture)
            .add_systems(Last, picture_taken);
    }
}

/// No tick runs while a due picture waits for its frame.
fn no_picture_waiting(shots: Res<Shots>) -> bool {
    shots.due.is_none()
}

/// After each tick: the tick is due on the cadence and at the run's end.
fn mark_due(clock: Res<BattleClock>, run: Res<crate::cli::RunState>, mut shots: ResMut<Shots>) {
    let tick = clock.tick;
    if (tick.is_multiple_of(shots.every) || run.ended()) && shots.last != Some(tick) {
        shots.due = Some(tick);
    }
}

/// The due tick's picture of this frame: the primary window's, or the
/// offscreen image's when there is no window. The run waits for it to be
/// written.
fn take_picture(
    mut commands: Commands,
    mut shots: ResMut<Shots>,
    mut run: ResMut<crate::cli::RunState>,
    window: Query<(), With<bevy::window::PrimaryWindow>>,
    offscreen: Option<Res<OffscreenTarget>>,
) {
    let Some(tick) = shots.due else { return };
    let shot = if !window.is_empty() {
        Screenshot::primary_window()
    } else if let Some(target) = offscreen {
        Screenshot::image(target.0.clone())
    } else {
        warn!("--shots: no window and no offscreen image, tick {tick} not saved");
        return;
    };
    shots.last = Some(tick);
    let path = shots.dir.join(format!("tick-{tick:06}.png"));
    let mut save = save_to_disk(path);
    run.holds += 1;
    commands.spawn(shot).observe(move |captured: On<ScreenshotCaptured>, mut run: ResMut<crate::cli::RunState>| {
        save(captured);
        run.holds -= 1;
    });
}

/// The picture's frame is over: the ticks go on.
fn picture_taken(mut shots: ResMut<Shots>) {
    shots.due = None;
}

/// The offscreen image's size: FL_WINDOW's, else 1600x900.
pub fn offscreen_size() -> UVec2 {
    std::env::var("FL_WINDOW")
        .ok()
        .and_then(|v| {
            let (w, h) = v.split_once('x')?;
            Some(UVec2::new(w.trim().parse().ok()?, h.trim().parse().ok()?))
        })
        .unwrap_or(UVec2::new(1600, 900))
}
