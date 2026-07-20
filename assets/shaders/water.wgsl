// River water: chunky quantized vertex waves, flat derivative normals,
// depth-banded color with a wobbling foam band at the shoreline.
// Mesh: position @location(0), uv @location(2) where uv.x = shore
// factor (0 center, 1 shoreline), uv.y = bed depth fraction.

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}
#import bevy_pbr::mesh_view_bindings::globals

struct WaterUniforms {
    // xyz = direction toward the sun, w spare.
    sun_dir: vec4<f32>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> water: WaterUniforms;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) crest: f32,
};

fn hash2(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

@vertex
fn vertex(v: Vertex) -> VertexOutput {
    var out: VertexOutput;
    var pos = v.position;
    let time = globals.time;
    // One wave phase per 3 m cell: shared cell corners move together
    // (watertight), neighbors twinkle in blocks — faceted, not rolling.
    let cell = floor(pos.xz / 3.0);
    let ph = hash2(cell) * 6.2831;
    let w1 = sin(time * 1.6 + ph);
    let w2 = sin(time * 2.7 + ph * 1.7 + pos.x * 0.35);
    let wave = w1 * 0.10 + w2 * 0.05;
    pos.y += wave;
    out.crest = wave * 6.0;
    let world_from_local = get_world_from_local(v.instance_index);
    out.world = (world_from_local * vec4<f32>(pos, 1.0)).xyz;
    out.clip = mesh_position_local_to_clip(world_from_local, vec4<f32>(pos, 1.0));
    out.uv = v.uv;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Flat per-face normal from screen-space derivatives.
    let n = normalize(cross(dpdx(in.world), dpdy(in.world)));
    let ndl = max(dot(n, water.sun_dir.xyz), 0.0);

    // Deep/shallow banding by bed depth.
    let shallow = vec3<f32>(0.28, 0.52, 0.55);
    let deep = vec3<f32>(0.06, 0.19, 0.33);
    var col = mix(shallow, deep, clamp(in.uv.y, 0.0, 1.0));

    // Foam band creeping at the shoreline.
    let wob = sin(globals.time * 1.3 + in.world.x * 0.5 + in.world.z * 0.8) * 0.04;
    let foam = smoothstep(0.87, 0.99, abs(in.uv.x) + wob);
    col = mix(col, vec3<f32>(0.82, 0.88, 0.86), foam);

    // Crest highlight.
    col += vec3<f32>(0.10, 0.10, 0.10) * clamp(in.crest, 0.0, 1.0);

    let light = 0.45 + 0.55 * ndl;
    return vec4<f32>(col * light, 0.88);
}
