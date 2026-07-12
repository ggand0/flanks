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
    // x = yaw, y = move amount 0..1, z = lunge: 0..1 wind-up progress,
    // negative = battle-stance amount (-0.5 blade leveled, -1 charging),
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
    let lunge = max(vertex.i_anim.z, 0.0);
    // Negative lunge band: 0..0.5 ramps into battle stance (blade
    // leveled), 0.5..1 adds the charge sprint (lean, stride, bob).
    let band = max(-vertex.i_anim.z, 0.0);
    let stance = smoothstep(0.0, 0.5, band);
    let sprint = smoothstep(0.5, 1.0, band);
    let fx = vertex.i_anim.w;
    let seed = vertex.i_color.a;
    let part = vertex.part_pivot.x;
    let pivot = vertex.part_pivot.y;

    var local = vertex.position;
    var normal = vertex.normal;

    let phase = globals.time * 9.0 + seed * 6.2831853;

    // --- Part animation (rotations around the part pivot) ---
    if part > 1.5 {
        // Legs: opposite-phase walk swing, harder stride on the charge.
        let side = select(1.0, -1.0, part > 2.5);
        let ang = 0.55 * (1.0 + 0.35 * sprint) * moving * sin(phase) * side;
        local = pitch_about(local, pivot, ang);
        normal = pitch_normal(normal, ang);
    } else if part > 0.5 {
        // Sword arm. Three per-unit attack styles (stable seed pick), all
        // timed so the blow lands exactly when the damage event fires
        // (lunge hits 1.0 at the strike tick): overhead chop, forward
        // stab, horizontal slash.
        let raise = smoothstep(0.0, 0.8, lunge);
        let chop = smoothstep(0.85, 1.0, lunge);
        // Idle/walk sway when not attacking.
        let sway = 0.18 * moving * sin(phase + 3.1415) * (1.0 - raise);
        // Ordinary moves carry the blade lowered at the side; battle
        // stance levels it at the enemy (slightly above horizontal).
        let carry = mix(-0.55, 0.25, stance) * moving * (1.0 - raise);
        // Taunt: STANDING units of a fighting regiment pump the blade
        // skyward for ~1.5 s every ~7 s, staggered per unit by seed —
        // the rear ranks jeer while the front works.
        let tc = fract(globals.time / 7.3 + seed * 5.13);
        let tpulse = smoothstep(0.02, 0.12, tc) * (1.0 - smoothstep(0.24, 0.34, tc));
        let taunt = tpulse * stance * (1.0 - moving) * (1.0 - smoothstep(0.0, 0.05, lunge));
        let ang_taunt = taunt * (1.7 + 0.22 * sin(globals.time * 16.0));

        let style = fract(seed * 7.31);
        var ang = 0.0;
        if style < 0.40 {
            // Overhead chop (the original swing).
            ang = 1.9 * raise - 2.5 * chop;
        } else if style < 0.75 {
            // Stab: draw the arm back, then thrust the blade forward
            // near-level. Translation happens in local space (pre-yaw).
            ang = 0.55 * raise - 0.45 * chop;
            local.z += -0.30 * raise + 1.05 * chop;
        } else {
            // Slash: horizontal sweep around the body axis — wind back,
            // cut across.
            let yawoff = -1.1 * raise + 2.3 * chop;
            let yc = cos(yawoff);
            let ys = sin(yawoff);
            local = rot_y(local, yc, ys);
            normal = rot_y(normal, yc, ys);
            ang = 0.5 * raise - 0.3 * chop;
        }
        ang += sway + carry + ang_taunt;
        local = pitch_about(local, pivot, ang);
        normal = pitch_normal(normal, ang);
    }

    // Walk bob + slight forward lean; lunge adds body punch, charging
    // adds a sprint lean and a heavier bob. Sprint lean is walk-gated
    // (the stance band is no longer walk-scaled) so jammed stragglers
    // don't posture.
    let bob = 0.05 * (1.0 + 0.4 * sprint) * moving * sin(phase * 2.0);
    let lean = (0.10 * moving + 0.30 * lunge + 0.24 * sprint * (0.25 + 0.75 * moving))
        * clamp(local.y + 0.5, 0.0, 1.5);
    local.z += lean * 0.3;

    // Death: topple around the feet and sink slightly. Fall direction
    // varies per unit (seed): forward, backward, or to either side —
    // corpses keep their seed, so the pose persists on the ground.
    let death = clamp(fx - 1.0, 0.0, 1.0);
    if death > 0.0 {
        let dvar = fract(seed * 13.73);
        let fy = local.y + 0.5;
        if dvar < 0.62 {
            // Forward (most common) or backward topple about X.
            let dir = select(1.0, -0.9, dvar > 0.45);
            let ang = dir * death * 1.45;
            let ca = cos(ang);
            let sa = sin(ang);
            local = vec3<f32>(local.x, fy * ca - local.z * sa - 0.5, fy * sa + local.z * ca);
            normal =
                vec3<f32>(normal.x, normal.y * ca - normal.z * sa, normal.y * sa + normal.z * ca);
        } else {
            // Sideways collapse about Z (either side).
            let dir = select(1.0, -1.0, dvar < 0.81);
            let ang = dir * death * 1.4;
            let ca = cos(ang);
            let sa = sin(ang);
            local = vec3<f32>(local.x * ca + fy * sa, fy * ca - local.x * sa - 0.5, local.z);
            normal =
                vec3<f32>(normal.x * ca + normal.y * sa, normal.y * ca - normal.x * sa, normal.z);
        }
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
