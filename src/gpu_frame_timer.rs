//! `FL_GPU_FRAME_TIMER=1`: how long the GPU spends on a frame from its first
//! command to its last, and how long it waits between frames. The periodic
//! log's per-pass times add up to less than the frame (devlog 0173: about
//! 0.6 to 0.9 ms a frame); these two say whether the rest is GPU work no
//! pass span covers or the GPU waiting on the CPU.
//!
//! Two timestamps bracket the frame's command buffers: one in a buffer of
//! its own ahead of every render graph system, one after the camera's
//! schedule. They are written as two consecutive frames run, so the second
//! frame's first stamp less the first frame's last is the time between them:
//! wgpu's staging copies of the frame's buffer writes, which run ahead of
//! its first command buffer, and any idle time. A pair of frames is sampled
//! whenever the last readback is in, a few frames apart.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use bevy::core_pipeline::schedule::camera_driver;
use bevy::diagnostic::{DiagnosticPath, Diagnostics, RegisterDiagnostic};
use bevy::prelude::*;
use bevy::render::{
    Render, RenderApp, RenderStartup, RenderSystems,
    render_resource::*,
    renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
};
use wgpu::{QuerySet, QuerySetDescriptor, QueryType};

/// From the frame's first command to its last, GPU time.
pub const FRAME_SPAN_GPU: DiagnosticPath = DiagnosticPath::const_new("render/frame_span/elapsed_gpu");
/// From one frame's last command to the next frame's first.
pub const FRAME_GAP_GPU: DiagnosticPath = DiagnosticPath::const_new("render/frame_gap/elapsed_gpu");

/// Stamps: 0, 1 the first frame's begin and end, 2, 3 the second's.
const STAMPS: u32 = 4;

const IDLE: u8 = 0;
/// The first frame's stamps are recorded.
const FIRST: u8 = 1;
/// Both frames' stamps, the resolve and the copy are recorded.
const COPIED: u8 = 2;
/// Waiting for the map.
const MAPPING: u8 = 3;

#[derive(Resource)]
struct FrameTimer {
    queries: QuerySet,
    resolve: Buffer,
    readback: Buffer,
    /// Nanoseconds per timestamp tick.
    period: f32,
    state: Arc<AtomicU8>,
}

/// The last sample's span and gap, nanoseconds; u64::MAX before the first.
static SPAN_NS: AtomicU64 = AtomicU64::new(u64::MAX);
static GAP_NS: AtomicU64 = AtomicU64::new(u64::MAX);

fn enabled() -> bool {
    std::env::var("FL_GPU_FRAME_TIMER").is_ok_and(|v| v == "1")
}

pub struct GpuFrameTimerPlugin;

impl Plugin for GpuFrameTimerPlugin {
    fn build(&self, app: &mut App) {
        if !enabled() {
            return;
        }
        app.register_diagnostic(bevy::diagnostic::Diagnostic::new(FRAME_SPAN_GPU).with_suffix(" ms"))
            .register_diagnostic(bevy::diagnostic::Diagnostic::new(FRAME_GAP_GPU).with_suffix(" ms"))
            .add_systems(Update, publish);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_timer)
            .add_systems(
                RenderGraph,
                (
                    stamp_begin.in_set(RenderGraphSystems::Begin),
                    stamp_end.in_set(RenderGraphSystems::Render).after(camera_driver),
                ),
            )
            .add_systems(Render, map_readback.in_set(RenderSystems::Cleanup));
    }
}

fn init_timer(mut commands: Commands, device: Res<RenderDevice>, queue: Res<RenderQueue>) {
    let needed = WgpuFeatures::TIMESTAMP_QUERY | WgpuFeatures::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    if !device.features().contains(needed) {
        warn!("FL_GPU_FRAME_TIMER: this device has no timestamps inside command encoders");
        return;
    }
    let buffer = |label, usage| {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: STAMPS as u64 * 8,
            usage,
            mapped_at_creation: false,
        })
    };
    commands.insert_resource(FrameTimer {
        queries: device.wgpu_device().create_query_set(&QuerySetDescriptor {
            label: Some("frame timestamps"),
            ty: QueryType::Timestamp,
            count: STAMPS,
        }),
        resolve: buffer("frame timestamps resolve", BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC),
        readback: buffer("frame timestamps readback", BufferUsages::MAP_READ | BufferUsages::COPY_DST),
        period: queue.get_timestamp_period(),
        state: Arc::new(AtomicU8::new(IDLE)),
    });
}

fn stamp_begin(timer: Option<Res<FrameTimer>>, mut ctx: RenderContext) {
    let Some(timer) = timer else {
        return;
    };
    match timer.state.load(Ordering::Acquire) {
        IDLE => ctx.command_encoder().write_timestamp(&timer.queries, 0),
        FIRST => ctx.command_encoder().write_timestamp(&timer.queries, 2),
        _ => {}
    }
}

fn stamp_end(timer: Option<Res<FrameTimer>>, mut ctx: RenderContext) {
    let Some(timer) = timer else {
        return;
    };
    match timer.state.load(Ordering::Acquire) {
        IDLE => {
            ctx.command_encoder().write_timestamp(&timer.queries, 1);
            timer.state.store(FIRST, Ordering::Release);
        }
        FIRST => {
            let encoder = ctx.command_encoder();
            encoder.write_timestamp(&timer.queries, 3);
            encoder.resolve_query_set(&timer.queries, 0..STAMPS, &timer.resolve, 0);
            encoder.copy_buffer_to_buffer(&timer.resolve, 0, &timer.readback, 0, STAMPS as u64 * 8);
            timer.state.store(COPIED, Ordering::Release);
        }
        _ => {}
    }
}

/// After the frame's submit: map, read the four stamps, unmap.
fn map_readback(timer: Option<Res<FrameTimer>>) {
    let Some(timer) = timer else {
        return;
    };
    if timer.state.load(Ordering::Acquire) != COPIED {
        return;
    }
    timer.state.store(MAPPING, Ordering::Release);
    let buffer = timer.readback.clone();
    let state = timer.state.clone();
    let period = timer.period as f64;
    timer.readback.slice(..).map_async(MapMode::Read, move |res| {
        if res.is_ok() {
            let t: [u64; 4] = bytemuck::pod_read_unaligned(&buffer.slice(..).get_mapped_range());
            buffer.unmap();
            let ns = |a: u64, b: u64| (b.wrapping_sub(a) as f64 * period) as u64;
            SPAN_NS.store((ns(t[0], t[1]) + ns(t[2], t[3])) / 2, Ordering::Relaxed);
            GAP_NS.store(ns(t[1], t[2]), Ordering::Relaxed);
        }
        state.store(IDLE, Ordering::Release);
    });
}

fn publish(mut diagnostics: Diagnostics) {
    for (path, value) in [(&FRAME_SPAN_GPU, &SPAN_NS), (&FRAME_GAP_GPU, &GAP_NS)] {
        let ns = value.load(Ordering::Relaxed);
        if ns != u64::MAX {
            diagnostics.add_measurement(path, || ns as f64 * 1e-6);
        }
    }
}
