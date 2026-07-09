#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}
#import bevy_pbr::mesh_view_bindings::globals

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,

    // Per-instance: xyz = world position, w = uniform scale.
    @location(3) i_pos_scale: vec4<f32>,
    // rgb = team color, a = stable per-unit anim seed (not opacity).
    @location(4) i_color: vec4<f32>,
    // x = yaw, y = move amount 0..1, z = lunge 0..1,
    // w = fx: [0,1) hit-flash intensity, [1,2] = 1 + death progress.
    @location(5) i_anim: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

fn rot_y(p: vec3<f32>, c: f32, s: f32) -> vec3<f32> {
    return vec3<f32>(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let yaw = vertex.i_anim.x;
    let moving = vertex.i_anim.y;
    let lunge = vertex.i_anim.z;
    let fx = vertex.i_anim.w;
    let seed = vertex.i_color.a;

    var local = vertex.position;
    var normal = vertex.normal;

    // Walk cycle: bob + slight forward lean, phase-offset per unit.
    let phase = globals.time * 9.0 + seed * 6.2831853;
    let bob = 0.06 * moving * sin(phase);
    let lean = (0.10 * moving + 0.35 * lunge) * clamp(local.y + 0.5, 0.0, 1.5);
    local.z += lean * 0.3;

    // Death: topple forward around the feet and sink slightly.
    let death = clamp(fx - 1.0, 0.0, 1.0);
    if death > 0.0 {
        let ang = death * 1.45; // ~83 degrees
        let ca = cos(ang);
        let sa = sin(ang);
        // Rotate around the X axis at foot height (local -0.55..-0.5 ~ -0.5).
        let fy = local.y + 0.5;
        local = vec3<f32>(local.x, fy * ca - local.z * sa - 0.5, fy * sa + local.z * ca);
        normal = vec3<f32>(normal.x, normal.y * ca - normal.z * sa, normal.y * sa + normal.z * ca);
    }

    // Face yaw (0 = +Z): rotate position and normal.
    let c = cos(yaw);
    let s = sin(yaw);
    local = rot_y(local, c, s);
    normal = rot_y(normal, c, s);

    let position = local * vertex.i_pos_scale.w
        + vertex.i_pos_scale.xyz
        + vec3<f32>(0.0, bob - 0.15 * death, 0.0);

    var out: VertexOutput;
    // Instance entity sits at the origin with identity transform, so passing
    // index 0 is fine (same hack as the upstream instancing example).
    out.clip_position = mesh_position_local_to_clip(
        get_world_from_local(0u),
        vec4<f32>(position, 1.0)
    );

    // Flat-shaded lambert: normals are per-face, so per-vertex lighting is
    // exact; normals are rotated with the instance above.
    let n = normalize(normal);
    let sun_dir = normalize(vec3<f32>(0.45, 0.85, 0.3));
    let ndl = max(dot(n, sun_dir), 0.0);
    let sky = 0.5 + 0.5 * n.y; // hemispheric ambient, brighter from above
    let light = 0.30 + 0.20 * sky + 0.65 * ndl;

    // Hit flash lerps toward white; death darkens.
    let flash = clamp(fx, 0.0, 1.0) * step(fx, 1.0);
    var rgb = vertex.i_color.rgb * light;
    rgb = mix(rgb, vec3<f32>(1.0, 1.0, 1.0), flash * 0.8);
    rgb = rgb * (1.0 - 0.45 * death);
    out.color = vec4<f32>(rgb, 1.0);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
