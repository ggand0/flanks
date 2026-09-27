// Selection rings (selection_rings.rs): one quad on the ground per ringed
// soldier, drawn as an evenly filled see-through disc with a point at the
// front where he faces, ETW's teardrop. Each corner sits on the terrain.

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
    // xy = the height field's vertex counts, z = 1 with the facing point,
    // w = the first ring entry in `entries`.
    grid: vec4<u32>,
};

@group(2) @binding(0) var<storage, read> records: array<Record>;
// Record index | style << 28 | kind << 30, one per ring.
@group(2) @binding(1) var<storage, read> entries: array<u32>;
// The terrain's vertex heights, row-major [z][x] (terrain.rs).
@group(2) @binding(2) var<storage, read> heights: array<f32>;
@group(2) @binding(3) var<uniform> ring: RingParams;

// The quad's half size in ring radii: room for the point.
const EXTENT: f32 = 1.5;
// Opacity of the whole shape, before the style's scale. One even fill: a
// brighter rim would stack the colour up where formations are dense.
const FILL: f32 = 0.45;
// The point's tip, in ring radii from the centre. The two lines from the
// tip that touch the circle close the teardrop.
const TIP: f32 = 1.45;
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
    var shape = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, d);

    if ring.grid.z != 0u {
        // The tangent lines from the tip touch the circle at height 1/TIP.
        // Past that height the teardrop is the wedge between them; below
        // it the disc already covers the wedge.
        let touch = 1.0 / TIP;
        let slope = sqrt(1.0 - touch * touch) / (TIP - touch);
        let edge = (TIP - in.local.y) * slope - abs(in.local.x);
        let edge_aa = max(fwidth(edge), 1e-4);
        let wedge = smoothstep(-edge_aa, edge_aa, edge) * step(touch, in.local.y);
        shape = max(shape, wedge);
    }

    let colour = ring.colors[in.style];
    let alpha = shape * FILL * colour.a;
    if alpha < 0.004 {
        discard;
    }
    return vec4<f32>(colour.rgb, alpha);
}
