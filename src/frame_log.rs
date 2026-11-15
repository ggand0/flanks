//! `FL_FRAME_LOG=<file>`: one text line per main-thread frame in battle
//! (`M`) and one per render-thread frame extracted from one (`R`), so a
//! felt hitch can be tied to the exact frame and to the piece of the frame
//! that took the time. The fps average over two seconds hides single long
//! frames; this record keeps every one.
//!
//! Without the variable every hook below is one branch on a value read
//! once, no system or resource is added, and nothing else changes. With
//! it, the pieces are stamped into statics where they happen (the tick
//! worker, the fixed tick, PostUpdate, the render thread), the main
//! thread's `Last` writes the M line and resets them, and the render
//! thread writes its R line at the end of its frame. All times are
//! microseconds from the moment the log opened. The file is flushed about
//! once a second from the main thread, so the last second of a run that is
//! killed is lost.

use std::io::Write;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use bevy::diagnostic::{DiagnosticPath, DiagnosticsStore};
use bevy::prelude::*;
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};

use crate::game_state::GameState;

/// The open log: the shared time base and the writer both threads use.
struct Log {
    base: Instant,
    out: std::sync::Mutex<std::io::BufWriter<std::fs::File>>,
}

impl Log {
    /// Microseconds from the log's start to `t`.
    fn at(&self, t: Instant) -> u64 {
        t.saturating_duration_since(self.base).as_micros() as u64
    }

    fn now(&self) -> u64 {
        self.at(Instant::now())
    }

    fn write(&self, line: std::fmt::Arguments) {
        let mut out = self.out.lock().unwrap();
        let _ = out.write_fmt(line);
    }
}

const M_FIELDS: &str = "M frame_id t_first_us interval_us ticks fixed_loop_us update_post_us gap_prev_us take_wait_us step_us audit_us kick_us pack_wait_us job_us job_slack_us real_delta_us alpha cam_x cam_z cam_yaw cam_pitch cam_dist cut pyramid gpu_unit_us units";
const R_FIELDS: &str = "R frame_id t_begin_us t_end_us prepare_us fresh";

/// The log, opened on the first call when `FL_FRAME_LOG` names a file.
fn log() -> Option<&'static Log> {
    static LOG: std::sync::OnceLock<Option<Log>> = std::sync::OnceLock::new();
    LOG.get_or_init(|| {
        let path = std::env::var("FL_FRAME_LOG").ok().filter(|p| !p.is_empty())?;
        let file = match std::fs::File::create(&path) {
            Ok(file) => file,
            Err(e) => {
                warn!("FL_FRAME_LOG: cannot create {path}: {e}");
                return None;
            }
        };
        let log = Log {
            base: Instant::now(),
            out: std::sync::Mutex::new(std::io::BufWriter::with_capacity(1 << 20, file)),
        };
        log.write(format_args!("# {M_FIELDS}\n# {R_FIELDS}\n"));
        info!("FL_FRAME_LOG: one line per frame into {path}");
        Some(log)
    })
    .as_ref()
}

/// The record is on.
pub fn on() -> bool {
    log().is_some()
}

fn us(d: std::time::Duration) -> u64 {
    d.as_micros() as u64
}

// Main thread, summed over the frame and reset by the M line.
static TICKS: AtomicU64 = AtomicU64::new(0);
static TAKE_WAIT: AtomicU64 = AtomicU64::new(0);
static STEP: AtomicU64 = AtomicU64::new(0);
static AUDIT: AtomicU64 = AtomicU64::new(0);
static KICK: AtomicU64 = AtomicU64::new(0);
static PACK: AtomicU64 = AtomicU64::new(0);
static JOB: AtomicU64 = AtomicU64::new(0);
/// The smallest slack of the jobs taken this frame, `i64::MAX` for none.
static SLACK: AtomicI64 = AtomicI64::new(i64::MAX);
static ALPHA: AtomicU32 = AtomicU32::new(0);
static CUT: AtomicU32 = AtomicU32::new(0);
static PYRAMID: AtomicU32 = AtomicU32::new(0);

// The tick worker: the last job's run time and when it finished, stored
// before the job is sent back, read right after it is received.
static JOB_RUN: AtomicU64 = AtomicU64::new(0);
static JOB_DONE_AT: AtomicU64 = AtomicU64::new(0);

// Render thread, reset by the R line.
static R_BEGIN: AtomicU64 = AtomicU64::new(0);
static R_END: AtomicU64 = AtomicU64::new(0);
static R_PREPARE: AtomicU64 = AtomicU64::new(0);
static R_FRESH: AtomicU32 = AtomicU32::new(0);

/// Adds the wall time from its creation to its drop into one slot. Holds
/// nothing when the record is off.
pub struct Span(Option<(Instant, &'static AtomicU64)>);

impl Drop for Span {
    fn drop(&mut self) {
        if let Some((t0, slot)) = self.0 {
            slot.fetch_add(us(t0.elapsed()), Relaxed);
        }
    }
}

fn span(slot: &'static AtomicU64) -> Span {
    Span(on().then(|| (Instant::now(), slot)))
}

/// Times one `step_sim` (the install, the damage, the audit when it runs,
/// and the inline tick when no job was waiting), and counts it as a tick.
pub fn time_step() -> Span {
    if on() {
        TICKS.fetch_add(1, Relaxed);
    }
    span(&STEP)
}

/// Times one `neighbour_audit`.
pub fn time_audit() -> Span {
    span(&AUDIT)
}

/// Times one `kick_tick`.
pub fn time_kick() -> Span {
    span(&KICK)
}

/// Times one `pack_soldier_snapshot` that had a change to pack.
pub fn time_pack() -> Span {
    span(&PACK)
}

/// Times one `prepare_gpu_units` on the render thread; `fresh` is whether
/// it uploads a new snapshot.
pub fn time_prepare(fresh: bool) -> Span {
    if on() {
        R_FRESH.store(fresh as u32, Relaxed);
    }
    span(&R_PREPARE)
}

/// The start of a tick job on the worker thread.
pub fn job_start() -> Option<Instant> {
    on().then(Instant::now)
}

/// The end of a tick job on the worker thread, before it is sent back.
pub fn job_end(start: Option<Instant>) {
    if let (Some(t0), Some(log)) = (start, log()) {
        let end = Instant::now();
        JOB_RUN.store(us(end - t0), Relaxed);
        JOB_DONE_AT.store(log.at(end), Relaxed);
    }
}

/// The main thread enters the take of the in-flight job.
pub fn take_start() -> Option<Instant> {
    on().then(Instant::now)
}

/// The main thread received the job: the wait, the job's run time and its
/// slack (take entered minus job finished, negative when the main thread
/// had to wait).
pub fn take_end(start: Option<Instant>) {
    if let (Some(t0), Some(log)) = (start, log()) {
        TAKE_WAIT.fetch_add(us(t0.elapsed()), Relaxed);
        JOB.fetch_add(JOB_RUN.load(Relaxed), Relaxed);
        let slack = log.at(t0) as i64 - JOB_DONE_AT.load(Relaxed) as i64;
        SLACK.fetch_min(slack, Relaxed);
    }
}

/// What `build_frame_params` sent the unit pass this frame: the
/// interpolation factor, the camera cut, and whether the occlusion depth
/// pyramid is built after this frame's opaque pass.
pub fn set_frame_params(alpha: f32, cut: bool, pyramid: bool) {
    if on() {
        ALPHA.store(alpha.to_bits(), Relaxed);
        CUT.store(cut as u32, Relaxed);
        PYRAMID.store(pyramid as u32, Relaxed);
    }
}

/// `render_frame_begin` ran.
pub fn render_begin() {
    if let Some(log) = log() {
        R_BEGIN.store(log.now(), Relaxed);
    }
}

/// `render_frame_end` ran.
pub fn render_end() {
    if let Some(log) = log() {
        R_END.store(log.now(), Relaxed);
    }
}

/// The main side's frame marks, in microseconds from the log's start.
#[derive(Resource, Default)]
pub struct FrameLogState {
    /// The current main frame in battle; counts across battles.
    id: u64,
    /// This frame's `First`, when it ran in battle.
    first: Option<u64>,
    prev_first: Option<u64>,
    after_fixed: Option<u64>,
    prev_last: Option<u64>,
    /// The frame whose M line was written this frame, for the extract.
    written: Option<u64>,
    last_flush: u64,
}

/// The main frame the render world's current frame was extracted from,
/// `None` when it wrote no M line.
#[derive(Resource, Default)]
struct RenderFrameId(Option<u64>);

fn reset_sums() {
    for slot in [&TICKS, &TAKE_WAIT, &STEP, &AUDIT, &KICK, &PACK, &JOB] {
        slot.store(0, Relaxed);
    }
    SLACK.store(i64::MAX, Relaxed);
    for slot in [&ALPHA, &CUT, &PYRAMID] {
        slot.store(0, Relaxed);
    }
}

/// The marks carry the previous battle's or the menu's times, so a new
/// battle starts them over. The frame counter goes on.
fn reset_state(mut state: ResMut<FrameLogState>) {
    let id = state.id;
    *state = FrameLogState { id, ..default() };
    reset_sums();
}

fn frame_first(game: Res<State<GameState>>, mut state: ResMut<FrameLogState>) {
    state.written = None;
    if *game.get() != GameState::Battle {
        return;
    }
    let Some(log) = log() else { return };
    state.id += 1;
    state.first = Some(log.now());
    state.after_fixed = None;
}

fn frame_after_fixed(mut state: ResMut<FrameLogState>) {
    if let Some(log) = log() {
        state.after_fixed = Some(log.now());
    }
}

/// The diagnostic of the GPU unit pass, in ms.
const GPU_UNIT: DiagnosticPath = DiagnosticPath::const_new("render/main_unit_pass/elapsed_gpu");

fn frame_last(
    mut state: ResMut<FrameLogState>,
    real: Res<Time<Real>>,
    camera: Query<&crate::camera::RtsCamera>,
    diagnostics: Res<DiagnosticsStore>,
    units: Res<crate::units::Units>,
) {
    let Some(log) = log() else { return };
    let now = log.now();
    let prev_last = state.prev_last.replace(now);
    // The frame that entered battle ran its First in the menu.
    let Some(first) = state.first.take() else {
        reset_sums();
        return;
    };
    let interval = state.prev_first.replace(first).map_or(0, |p| first.saturating_sub(p));
    let gap = prev_last.map_or(0, |p| first.saturating_sub(p));
    let after = state.after_fixed.unwrap_or(first);
    let take = |slot: &AtomicU64| slot.swap(0, Relaxed);
    let slack = SLACK.swap(i64::MAX, Relaxed);
    let slack = if slack == i64::MAX { 0 } else { slack };
    let cam = camera.single().ok();
    let (cx, cz, yaw, pitch, dist) =
        cam.map_or((0.0, 0.0, 0.0, 0.0, 0.0), |c| (c.focus.x, c.focus.z, c.yaw, c.pitch, c.distance));
    let gpu_us = diagnostics.get(&GPU_UNIT).and_then(|d| d.value()).unwrap_or(0.0) * 1000.0;
    log.write(format_args!(
        "M {} {first} {interval} {} {} {} {gap} {} {} {} {} {} {} {slack} {} {:.4} {cx:.3} {cz:.3} {yaw:.4} {pitch:.4} {dist:.3} {} {} {gpu_us:.0} {}\n",
        state.id,
        take(&TICKS),
        after.saturating_sub(first),
        now.saturating_sub(after),
        take(&TAKE_WAIT),
        take(&STEP),
        take(&AUDIT),
        take(&KICK),
        take(&PACK),
        take(&JOB),
        us(real.delta()),
        f32::from_bits(ALPHA.swap(0, Relaxed)),
        CUT.swap(0, Relaxed),
        PYRAMID.swap(0, Relaxed),
        units.pos.len(),
    ));
    state.written = Some(state.id);
    if now.saturating_sub(state.last_flush) >= 1_000_000 {
        state.last_flush = now;
        let _ = log.out.lock().unwrap().flush();
    }
}

fn extract_frame_id(main: Extract<Res<FrameLogState>>, mut id: ResMut<RenderFrameId>) {
    id.0 = main.written;
}

fn render_line(id: Res<RenderFrameId>) {
    let Some(log) = log() else { return };
    let prepare = R_PREPARE.swap(0, Relaxed);
    let fresh = R_FRESH.swap(0, Relaxed);
    let Some(id) = id.0 else { return };
    log.write(format_args!(
        "R {id} {} {} {prepare} {fresh}\n",
        R_BEGIN.load(Relaxed),
        R_END.load(Relaxed),
    ));
}

pub struct FrameLogPlugin;

impl Plugin for FrameLogPlugin {
    fn build(&self, app: &mut App) {
        if !on() {
            return;
        }
        app.init_resource::<FrameLogState>()
            .add_systems(OnEnter(GameState::Battle), reset_state)
            .add_systems(First, frame_first)
            .add_systems(
                RunFixedMainLoop,
                frame_after_fixed
                    .in_set(bevy::app::RunFixedMainLoopSystems::AfterFixedMainLoop)
                    .run_if(in_state(GameState::Battle)),
            )
            .add_systems(Last, frame_last.run_if(in_state(GameState::Battle)));
        app.sub_app_mut(RenderApp)
            .init_resource::<RenderFrameId>()
            .add_systems(ExtractSchedule, extract_frame_id)
            .add_systems(
                Render,
                render_line
                    .in_set(RenderSystems::PostCleanup)
                    .after(crate::render_units::render_frame_end),
            );
    }
}
