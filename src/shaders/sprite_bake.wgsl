// FL_SPRITE_BAKE (sprite_bake.rs): poses one kind's mesh for a list of
// instance records with the draw's own pose code, for the sprite sheet
// (tools/sprite_sheet.py). One thread per record and corner: the corner
// placed as the vertex shader places it (unit_pose.wgsl `pose_begin` and
// `place`), by the soldier's parts, as his far levels are drawn.
#import flanks::unit_pose::{pose_begin, place}

// One record to pose: x = its time in seconds, then the instance record
// (render_units.rs InstanceData).
struct Sample {
    time: vec4<f32>,
    pos_scale: vec4<f32>,
    color: vec4<f32>,
    anim: vec4<f32>,
    anim2: vec4<f32>,
};

// A corner of the level mesh (render_units_gpu.rs PullVertex).
struct Corner {
    position: vec3<f32>,
    part: f32,
    normal: vec3<f32>,
    pivot: f32,
    color: u32,
    atlas_uv: u32,
    skin: u32,
    weight: f32,
};

@group(3) @binding(0) var<storage, read> samples: array<Sample>;
@group(3) @binding(1) var<storage, read> corners: array<Corner>;
// Per record and corner, two vec4s: the placed position and the atlas
// coordinate's bits, then the placed normal and the part id.
@group(3) @binding(2) var<storage, read_write> placed: array<vec4<f32>>;

@compute @workgroup_size(64)
fn bake(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = arrayLength(&corners);
    let total = n * arrayLength(&samples);
    if gid.x >= total {
        return;
    }
    let s = samples[gid.x / n];
    let c = corners[gid.x % n];
    pose_begin(s.pos_scale, s.color, s.anim, s.anim2, s.time.x, false);
    let p = place(c.position, c.normal, c.part, c.pivot, c.skin, c.weight);
    placed[2u * gid.x] = vec4<f32>(p.position, bitcast<f32>(c.atlas_uv));
    placed[2u * gid.x + 1u] = vec4<f32>(p.normal, c.part);
}
