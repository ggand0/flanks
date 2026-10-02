// The pose pass (render_units_gpu.rs): after the build pass, one thread per
// drawn soldier of one kind poses him once into his kind's pose buffer, and
// every corner of his mesh reads it back (unit_instancing.wgsl). One
// dispatch per kind, with the kind's rig and pose buffer bound at group 3,
// its size from the build pass's count of the kind's pose slots.
#import bevy_render::globals::Globals
#import flanks::unit_pose::pose_write

// The instance record (render_units.rs InstanceData).
struct Record {
    pos_scale: vec4<f32>,
    color: vec4<f32>,
    anim: vec4<f32>,
    anim2: vec4<f32>,
};

struct Params {
    // Per kind: the first entry of its region in `pose_src`, and pose slots
    // per soldier in its pose buffer.
    src_base: vec4<u32>,
    stride: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> records: array<Record>;
// Per pose slot, the record it poses (unit_build.wgsl `list_entry`).
@group(0) @binding(2) var<storage, read> pose_src: array<u32>;
// The build pass's counters (unit_build.wgsl `counts`): 49..53 the pose
// slots taken per kind.
@group(0) @binding(3) var<storage, read> counts: array<u32>;
@group(0) @binding(4) var<uniform> globals: Globals;
// x = the kind this dispatch poses.
@group(3) @binding(10) var<uniform> kind: vec4<u32>;

const POSE_COUNTER: u32 = 49u;

@compute @workgroup_size(64)
fn pose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = kind.x;
    let slot = gid.x;
    if slot >= counts[POSE_COUNTER + k] {
        return;
    }
    // The top bit: the soldier is drawn at a body-only level
    // (unit_build.wgsl `list_entry`).
    let src = pose_src[params.src_base[k] + slot];
    let r = records[src & 0x7fffffffu];
    pose_write(slot * params.stride[k], r.pos_scale, r.color, r.anim, r.anim2, globals.time, (src >> 31u) != 0u);
}
