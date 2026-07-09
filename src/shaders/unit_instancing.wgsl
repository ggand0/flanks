#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}
#import bevy_pbr::mesh_view_bindings::globals

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // NOT texture coords: x = body part id (0 body, 1 sword arm,
    // 2 left leg, 3 right leg), y = the part's pivot height.
    @location(2) part_pivot: vec2<f32>,
    // Part material: rgb = fixed color, a = team-color blend amount.
    @location(5) v_color: vec4<f32>,

    // Per-instance: xyz = world position, w = uniform scale.
    @location(8) i_pos_scale: vec4<f32>,
    // rgb = team color, a = stable per-unit anim seed (not opacity).
    @location(9) i_color: vec4<f32>,
    // x = yaw, y = move amount 0..1, z = lunge 0..1 (wind-up progress),
    // w = fx: [0,1) hit-flash intensity, [1,2] = 1 + death progress.
    @location(10) i_anim: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

fn rot_y(p: vec3<f32>, c: f32, s: f32) -> vec3<f32> {
    return vec3<f32>(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

// Pitch (about +X) around a pivot at height py: +angle takes +Z toward +Y.
fn pitch_about(p: vec3<f32>, py: f32, ang: f32) -> vec3<f32> {
    let c = cos(ang);
    let s = sin(ang);
    let y = p.y - py;
    return vec3<f32>(p.x, py + y * c + p.z * s, -y * s + p.z * c);
}

fn pitch_normal(n: vec3<f32>, ang: f32) -> vec3<f32> {
    let c = cos(ang);
    let s = sin(ang);
    return vec3<f32>(n.x, n.y * c + n.z * s, -n.y * s + n.z * c);
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let yaw = vertex.i_anim.x;
    let moving = vertex.i_anim.y;
    let lunge = vertex.i_anim.z;
    let fx = vertex.i_anim.w;
    let seed = vertex.i_color.a;
    let part = vertex.part_pivot.x;
    let pivot = vertex.part_pivot.y;

    var local = vertex.position;
    var normal = vertex.normal;

    let phase = globals.time * 9.0 + seed * 6.2831853;

    // --- Part animation (rotations around the part pivot) ---
    if part > 1.5 {
        // Legs: opposite-phase walk swing.
        let side = select(1.0, -1.0, part > 2.5);
        let ang = 0.55 * moving * sin(phase) * side;
        local = pitch_about(local, pivot, ang);
        normal = pitch_normal(normal, ang);
    } else if part > 0.5 {
        // Sword arm: raise the blade up/back through the wind-up, then a
        // fast chop over the last stretch so the blade lands exactly when
        // the damage event fires (lunge hits 1.0 at the strike tick).
        let raise = smoothstep(0.0, 0.8, lunge);
        let chop = smoothstep(0.85, 1.0, lunge);
        // Idle/walk sway when not attacking.
        let sway = 0.18 * moving * sin(phase + 3.1415) * (1.0 - raise);
        let ang = 1.9 * raise - 2.5 * chop + sway;
        local = pitch_about(local, pivot, ang);
        normal = pitch_normal(normal, ang);
    }

    // Walk bob + slight forward lean; lunge adds body punch.
    let bob = 0.05 * moving * sin(phase * 2.0);
    let lean = (0.10 * moving + 0.30 * lunge) * clamp(local.y + 0.5, 0.0, 1.5);
    local.z += lean * 0.3;

    // Death: topple forward around the feet and sink slightly.
    let death = clamp(fx - 1.0, 0.0, 1.0);
    if death > 0.0 {
        let ang = death * 1.45; // ~83 degrees
        let ca = cos(ang);
        let sa = sin(ang);
        // Rotate around the X axis at foot height (local ~ -0.5).
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

    // Part material blended with the team color (a = team amount), then
    // hit flash lerps toward white and death darkens.
    let base = mix(vertex.v_color.rgb, vertex.i_color.rgb, vertex.v_color.a);
    let flash = clamp(fx, 0.0, 1.0) * step(fx, 1.0);
    var rgb = base * light;
    rgb = mix(rgb, vec3<f32>(1.0, 1.0, 1.0), flash * 0.8);
    rgb = rgb * (1.0 - 0.45 * death);
    out.color = vec4<f32>(rgb, 1.0);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
