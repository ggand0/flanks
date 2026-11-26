//! `FL_SPRITE_BAKE=<dir>`: poses each kind's far mesh for the far soldiers'
//! sprite sheet (tools/sprite_sheet.py), with the draw's own pose code, and
//! exits. The tool writes `<dir>/bake_in.bin`, the records to pose: per
//! record five vec4s of f32, (kind, time, 0, 0) then the instance record
//! (render_units.rs InstanceData). The game poses every corner of the
//! kind's level `BAKE_LOD` for each record (shaders/sprite_bake.wgsl) and
//! writes `<dir>/bake_out.bin`: per kind its corner count as u32, its
//! record count as u32, then per record and corner two vec4s of f32, the
//! placed position in metres from the record's position with the atlas
//! coordinate's bits (unorm16 u, v) in w, and the placed normal with the
//! part id in w. The corners come three to a triangle. Nothing is written
//! when a step fails; the game ends either way.

use std::path::PathBuf;

use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::prelude::*;
use bevy::render::{
    Render, RenderApp, RenderStartup, RenderSystems,
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
};

use crate::render_units::{ExtractedBucket, NUM_LODS, RigBuffer};
use crate::render_units_gpu::PullMeshGpu;
use crate::unit_types::NUM_KINDS;

/// The level the sheet is drawn from: the nearest level the far soldiers
/// are drawn with by their parts (render_units_gpu.rs SKEL_LODS).
pub const BAKE_LOD: usize = crate::render_units_gpu::SKEL_LODS;

/// The bake's folder, when `FL_SPRITE_BAKE` names one. Read once.
pub fn bake_dir() -> Option<PathBuf> {
    static DIR: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| std::env::var("FL_SPRITE_BAKE").ok().filter(|v| !v.is_empty()).map(PathBuf::from))
        .clone()
}

#[derive(Resource)]
struct BakePipeline {
    pipeline: CachedComputePipelineId,
    empty: BindGroupLayoutDescriptor,
    layout: BindGroupLayoutDescriptor,
}

fn init_bake_pipeline(mut commands: Commands, asset_server: Res<AssetServer>, pipeline_cache: Res<PipelineCache>) {
    let empty = BindGroupLayoutDescriptor::new("sprite bake empty layout", &[]);
    let layout = BindGroupLayoutDescriptor::new(
        "sprite bake layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (0, binding_types::storage_buffer_read_only_sized(false, None)),
                (1, binding_types::storage_buffer_read_only_sized(false, None)),
                (2, binding_types::storage_buffer_sized(false, None)),
                (6, binding_types::uniform_buffer_sized(false, None)),
                (7, binding_types::storage_buffer_read_only_sized(false, None)),
            ),
        ),
    );
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("sprite bake".into()),
        layout: vec![empty.clone(), empty.clone(), empty.clone(), layout.clone()],
        immediate_size: 0,
        shader: load_embedded_asset!(asset_server.as_ref(), "shaders/sprite_bake.wgsl"),
        shader_defs: vec![],
        entry_point: Some("bake".into()),
        zero_initialize_workgroup_memory: false,
    });
    commands.insert_resource(BakePipeline { pipeline, empty, layout });
}

/// Once every kind's level `BAKE_LOD` and rig are on the GPU and the
/// pipeline is built: pose, read back and write the file; the main world then ends the game.
#[allow(clippy::too_many_arguments)] // bevy system params
fn run_bake(
    buckets: Query<(&ExtractedBucket, &PullMeshGpu, &RigBuffer)>,
    bake: Res<BakePipeline>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(bake.pipeline) else {
        return;
    };
    let mut kinds: [Option<(&PullMeshGpu, &RigBuffer)>; NUM_KINDS] = [None; NUM_KINDS];
    for (bucket, mesh, rig) in &buckets {
        if bucket.0 % NUM_LODS == BAKE_LOD {
            kinds[bucket.0 / NUM_LODS] = Some((mesh, rig));
        }
    }
    if kinds.iter().any(Option::is_none) {
        return;
    }
    *done = true;
    let dir = bake_dir().expect("the bake runs only with FL_SPRITE_BAKE");
    let input = match std::fs::read(dir.join("bake_in.bin")) {
        Ok(input) => input,
        Err(e) => {
            error!("FL_SPRITE_BAKE: cannot read {}: {e}", dir.join("bake_in.bin").display());
            BAKE_DONE.store(true, std::sync::atomic::Ordering::Release);
            return;
        }
    };
    let records: &[[f32; 20]] = bytemuck::cast_slice(&input);
    let mut out: Vec<u8> = Vec::new();
    let layout = pipeline_cache.get_bind_group_layout(&bake.layout);
    let empty = device.create_bind_group("sprite bake empty", &pipeline_cache.get_bind_group_layout(&bake.empty), &[]);
    for (kind, entry) in kinds.iter().enumerate() {
        let (mesh, rig) = entry.expect("checked above");
        let mine: Vec<[f32; 20]> = records.iter().filter(|r| r[0] as usize == kind).copied().collect();
        // Per record: (time, 0, 0, 0), then the record's four vec4s.
        let samples: Vec<[f32; 20]> = mine
            .iter()
            .map(|r| {
                let mut s = *r;
                s[0] = r[1];
                s[1] = 0.0;
                s
            })
            .collect();
        let corners = mesh.count as usize;
        out.extend_from_slice(&(corners as u32).to_le_bytes());
        out.extend_from_slice(&(samples.len() as u32).to_le_bytes());
        if samples.is_empty() {
            continue;
        }
        let size = (samples.len() * corners * 32) as u64;
        let samples_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("sprite bake records"),
            contents: bytemuck::cast_slice(&samples),
            usage: BufferUsages::STORAGE,
        });
        let placed = device.create_buffer(&BufferDescriptor {
            label: Some("sprite bake placed"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("sprite bake staging"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(
            "sprite bake",
            &layout,
            &BindGroupEntries::with_indices((
                (0, samples_buffer.as_entire_binding()),
                (1, mesh.vertices.as_entire_binding()),
                (2, placed.as_entire_binding()),
                (6, rig.rig.as_entire_binding()),
                (7, rig.clips.as_entire_binding()),
            )),
        );
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: Some("sprite bake") });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("sprite bake"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            for set in 0..3 {
                pass.set_bind_group(set, &empty, &[]);
            }
            pass.set_bind_group(3, &group, &[]);
            pass.dispatch_workgroups(((samples.len() * corners) as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&placed, 0, &staging, 0, size);
        queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        device.map_buffer(&slice, MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        if let Err(e) = device.poll(wgpu::PollType::wait_indefinitely()) {
            error!("FL_SPRITE_BAKE: the GPU wait failed: {e:?}");
            BAKE_DONE.store(true, std::sync::atomic::Ordering::Release);
            return;
        }
        if !matches!(rx.recv(), Ok(Ok(()))) {
            error!("FL_SPRITE_BAKE: cannot read the placed corners back");
            BAKE_DONE.store(true, std::sync::atomic::Ordering::Release);
            return;
        }
        out.extend_from_slice(&slice.get_mapped_range());
        info!("FL_SPRITE_BAKE: kind {kind}, {} records of {corners} corners", samples.len());
    }
    let path = dir.join("bake_out.bin");
    match std::fs::write(&path, &out) {
        Ok(()) => info!("FL_SPRITE_BAKE: wrote {} ({} bytes)", path.display(), out.len()),
        Err(e) => error!("FL_SPRITE_BAKE: cannot write {}: {e}", path.display()),
    }
    BAKE_DONE.store(true, std::sync::atomic::Ordering::Release);
}

/// Set by the render world once the file is written (or failed to be).
static BAKE_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Ends the game once the bake is done.
fn exit_after_bake(mut exit: MessageWriter<AppExit>) {
    if BAKE_DONE.load(std::sync::atomic::Ordering::Acquire) {
        exit.write(AppExit::Success);
    }
}

/// With `FL_SPRITE_BAKE` set: the bake pass, which ends the game once it
/// has written its file. Without it, nothing.
pub struct SpriteBakePlugin;

impl Plugin for SpriteBakePlugin {
    fn build(&self, app: &mut App) {
        if bake_dir().is_none() {
            return;
        }
        embedded_asset!(app, "shaders/sprite_bake.wgsl");
        app.add_systems(Update, exit_after_bake);
        app.sub_app_mut(RenderApp)
            .add_systems(RenderStartup, init_bake_pipeline)
            .add_systems(Render, run_bake.in_set(RenderSystems::PrepareBindGroups));
    }
}
