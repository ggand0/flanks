// Selection rings (selection_rings.rs): one quad on the ground per ringed
// soldier, drawn as a see-through disc with a brighter rim and a notch at
// the front where he faces. Each corner sits on the terrain.

#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::view_transformations::position_world_to_clip

// The instance record (render_units.rs InstanceData): xyz of pos_scale is
// the soldier's interpolated position, anim.x his interpolated yaw.
struct Record {
    pos_scale: vec4<f32>,
    color: vec4<f32>,
    anim: vec4<f32>,
    anim2: vec4<f32>,
};

struct RingParams {
    // Linear rgb and an alpha scale, per style: selected, own regiment
    // under the cursor, enemy under the cursor.
    colors: array<vec4<f32>, 3>,
    // Per kind: how far a soldier's position stands above his feet.
    half_heights: vec4<f32>,
    // xy = the height field's origin, z = its cell, w = the ring radius.
    terrain: vec4<f32>,
    // xy = the height field's vertex counts, z = 1 with the facing notch,
    // w = the first ring entry in `entries`.
    grid: vec4<u32>,
};

@group(2) @binding(0) var<storage, read> records: array<Record>;
// Record index | style << 28 | kind << 30, one per ring.
@group(2) @binding(1) var<storage, read> entries: array<u32>;
// The terrain's vertex heights, row-major [z][x] (terrain.rs).
@group(2) @binding(2) var<storage, read> heights: array<f32>;
@group(2) @binding(3) var<uniform> ring: RingParams;

// The quad's half size in ring radii: room for the notch.
const EXTENT: f32 = 1.35;
// Fill and rim opacity, before the style's scale.
const FILL: f32 = 0.30;
const RIM: f32 = 0.90;
// Rim width in ring radii, at least 1.5 pixels.
const RIM_WIDTH: f32 = 0.16;
// The notch: a triangle from the rim out to its apex, in ring radii.
const NOTCH_BASE: f32 = 0.94;
const NOTCH_APEX: f32 = 1.30;
const NOTCH_HALF_WIDTH: f32 = 0.24;
// Above the ground, growing with distance so depth precision never lets
// the terrain through.
const LIFT: f32 = 0.03;
const LIFT_PER_M: f32 = 0.0003;
// Farther than this between his feet and the field under him, he stands on
// something else (the bridge deck) and the ring lies flat at his feet.
const OFF_FIELD: f32 = 0.3;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Ring space in radii: x to his right, y where he faces.
    @location(0) local: vec2<f32>,
    @location(1) @interpolate(flat) style: u32,
};

// Bilinear height of the field, as terrain.rs `height_at` samples it.
fn ground(x: f32, z: f32) -> f32 {
    let nx = ring.grid.x;
    let nz = ring.grid.y;
    let g = (vec2<f32>(x, z) - ring.terrain.xy) / ring.terrain.z;
    let gx = clamp(g.x, 0.0, f32(nx - 2u));
    let gz = clamp(g.y, 0.0, f32(nz - 2u));
    let x0 = u32(gx);
    let z0 = u32(gz);
    let fx = gx - f32(x0);
    let fz = gz - f32(z0);
    let i = z0 * nx + x0;
    let near = mix(heights[i], heights[i + 1u], fx);
    let far = mix(heights[i + nx], heights[i + nx + 1u], fx);
    return mix(near, far, fz);
}

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let entry = entries[ring.grid.w + vertex_index / 6u];
    let rec = records[entry & 0x0fffffffu];
    let style = (entry >> 28u) & 3u;
    let kind = entry >> 30u;

    // Two triangles over [-1, 1]^2.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    let local = corners[vertex_index % 6u] * EXTENT;

    // Yaw 0 faces +z (formation.rs `facing_dir`).
    let yaw = rec.anim.x;
    let fwd = vec2<f32>(sin(yaw), cos(yaw));
    let right = vec2<f32>(fwd.y, -fwd.x);
    let centre = rec.pos_scale.xyz;
    let xz = centre.xz + (right * local.x + fwd * local.y) * ring.terrain.w;

    let feet = centre.y - ring.half_heights[kind];
    var y = feet;
    if abs(feet - ground(centre.x, centre.z)) < OFF_FIELD {
        y = ground(xz.x, xz.y);
    }
    let world = vec3<f32>(xz.x, y, xz.y);
    let lift = LIFT + LIFT_PER_M * distance(world, view.world_position);

    var out: VertexOutput;
    out.clip_position = position_world_to_clip(world + vec3<f32>(0.0, lift, 0.0));
    out.local = local;
    out.style = style;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = length(in.local);
    let aa = max(fwidth(d), 1e-4);
    let inside = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, d);
    let rim_width = max(RIM_WIDTH, 1.5 * aa);
    let rim = inside * smoothstep(1.0 - rim_width - aa, 1.0 - rim_width + aa, d);
    var alpha = inside * FILL + rim * (RIM - FILL);

    if ring.grid.z != 0u {
        // A triangle pointing out of the front of the rim: its half width
        // narrows from the base to the apex.
        let t = (NOTCH_APEX - in.local.y) / (NOTCH_APEX - NOTCH_BASE);
        let edge = t * NOTCH_HALF_WIDTH - abs(in.local.x);
        let notch_aa = max(fwidth(edge), 1e-4);
        let notch = smoothstep(-notch_aa, notch_aa, edge)
            * step(NOTCH_BASE - 0.1, in.local.y)
            * (1.0 - smoothstep(NOTCH_APEX - aa, NOTCH_APEX + aa, in.local.y));
        alpha = max(alpha, notch * RIM);
    }

    let colour = ring.colors[in.style];
    alpha *= colour.a;
    if alpha < 0.004 {
        discard;
    }
    return vec4<f32>(colour.rgb, alpha);
}
