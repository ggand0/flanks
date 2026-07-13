#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}
#import bevy_pbr::mesh_view_bindings::globals

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // NOT texture coords: x = body part id (0 body, 1 sword arm,
    // 2 left leg, 3 right leg, 4 spear arm, 5 shield arm), y = the
    // part's pivot height.
    @location(2) part_pivot: vec2<f32>,
    // Part material: rgb = fixed color, a = team-color blend amount.
    @location(5) v_color: vec4<f32>,

    // Per-instance: xyz = world position, w = uniform scale.
    @location(8) i_pos_scale: vec4<f32>,
    // rgb = team color, a = stable per-unit anim seed (not opacity).
    @location(9) i_color: vec4<f32>,
    // x = yaw, y = move amount 0..1. z positive = attack: style*2 +
    // wind-up progress (style 0 stab, 1 slash); z negative = stance
    // band (-0.25 enemy near, -0.5 blade leveled, -1 charging).
    // w = fx: [0,1) hit-flash intensity, [1,2] = 1 + death progress.
    @location(10) i_anim: vec4<f32>,
    // Regiment pose signals (smoothed per unit CPU-side): x = march-in-
    // step 0..1, y = wall 0..1 (shieldwall/spearwall by bucket),
    // z = regiment walk-phase offset, w = spare.
    @location(11) i_anim2: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// Standing brace pose (split legs, crouch, raised guard): benched by
// the owner ("weird") but kept — set to 1.0 to re-enable. Standing
// units near an enemy hold the plain forward point instead.
const BRACE_ON: f32 = 0.0;
// Rear-rank taunt: benched pending an owner rework ("something is
// making me uncomfortable") — set to 1.0 to re-enable.
const TAUNT_ON: f32 = 0.0;

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
    // Positive z: attack progress. The 2s digit is the swing STYLE
    // (0 = stab, 1 = classic swing, 2 = slash), remainder = lunge
    // 0..~1.35. z >= 6 = victory cheer (no one left to swing at).
    let zpos = max(vertex.i_anim.z, 0.0);
    let style = floor(zpos * 0.5 + 0.001);
    let lunge = zpos - style * 2.0;
    // Cheer: z = 6 + progress. Ease in over the first ~8% and back out
    // over the last ~10% — poses must never snap in one frame.
    let cele_t = clamp(zpos - 6.0, 0.0, 1.0);
    let celebrate = step(5.0, zpos)
        * smoothstep(0.0, 0.08, cele_t)
        * (1.0 - smoothstep(0.90, 1.0, cele_t));
    // Negative z band: 0.25 = enemy in watch range (brace when
    // standing), 0.5 = fighting but wavering (brace, no taunt),
    // 0.65 = fighting confident (taunts), 1 = charging (sprint).
    let band = max(-vertex.i_anim.z, 0.0);
    let ready = smoothstep(0.05, 0.25, band);
    let stance = smoothstep(0.25, 0.5, band);
    let confident = smoothstep(0.55, 0.65, band);
    let sprint = smoothstep(0.7, 1.0, band);
    let fx = vertex.i_anim.w;
    let seed = vertex.i_color.a;
    let part = vertex.part_pivot.x;
    let pivot = vertex.part_pivot.y;
    // Brace: standing, enemy near/engaged, not attacking — a planted
    // fight stance (split legs, crouch, blade at ready guard). Only
    // ~half the line braces (per-unit pick); the rest keep the plain
    // standing point, so a waiting line mixes both poses.
    let bracer = step(0.45, fract(seed * 3.77));
    let brace = BRACE_ON
        * bracer
        * ready
        * (1.0 - moving)
        * (1.0 - smoothstep(0.0, 0.05, lunge))
        * (1.0 - celebrate);

    var local = vertex.position;
    var normal = vertex.normal;

    let march = vertex.i_anim2.x;
    let wall = vertex.i_anim2.y;
    let phase = globals.time * 9.0 + seed * 6.2831853;
    // March-in-step: the whole regiment walks on ONE phase (offset per
    // regiment so neighboring blocks aren't synced with each other). The
    // walk oscillators mix toward it; everything else stays per-unit.
    let phase_reg = globals.time * 9.0 + vertex.i_anim2.z;
    let walk_s = mix(sin(phase), sin(phase_reg), march);
    let walk_s2 = mix(sin(phase * 2.0), sin(phase_reg * 2.0), march);

    // --- Part animation (rotations around the part pivot) ---
    if part > 4.5 {
        // Shield arm: carried at the side; the wall signal swings it
        // around the body to FACE THE FRONT and lifts it into a guard —
        // a shieldwall is a wall of team color from the enemy's side.
        // (Spear bucket: same fronting reads as the spearwall's off-hand
        // cover behind the leveled spears.)
        if wall > 0.001 {
            let ang = 1.05 * wall;
            let c2 = cos(ang);
            let s2 = sin(ang);
            local = rot_y(local, c2, s2);
            normal = rot_y(normal, c2, s2);
            local.y += 0.10 * wall;
        }
    } else if part > 3.5 {
        // Spear arm: the shaft is carried VERTICAL. Battle stance (or a
        // watch-range advance) levels the point at the enemy — a line of
        // spears coming down IS the brace — and the stab thrusts the
        // leveled shaft forward. Charging carries it leveled too.
        let raise = smoothstep(0.0, 0.8, lunge);
        let chop = smoothstep(0.85, 1.0, lunge);
        // Spearwall: points come down and STAY down, even standing idle.
        let level = max(max(stance, ready * 0.75), max(max(sprint, raise), wall));
        // Slight walk sway while the spear is upright; vertical pump on
        // a victory cheer.
        var ang = -1.42 * level * (1.0 - celebrate)
            + 0.05 * moving * sin(phase + 3.1415) * (1.0 - level)
            + 0.10 * celebrate * sin(globals.time * 9.0 + seed * 6.2831853);
        local = pitch_about(local, pivot, ang);
        normal = pitch_normal(normal, ang);
        // Draw back, then punch the point home (the damage tick lands at
        // lunge 1.0, same timing as every other weapon).
        local.z += -0.30 * raise + 1.15 * chop;
    } else if part > 1.5 {
        // Legs: opposite-phase walk swing, harder stride on the charge;
        // bracing or a wall stance splits the legs (one foot planted).
        let side = select(1.0, -1.0, part > 2.5);
        let ang = 0.55 * (1.0 + 0.35 * sprint) * moving * walk_s * side
            + 0.32 * brace * side
            + 0.22 * wall * (1.0 - moving) * side;
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
        // stance levels it at the enemy (slightly above horizontal),
        // and even a watch-range advance (`ready`) brings it most of
        // the way up — the braced walk.
        let carry = mix(-0.55, 0.25, max(stance, ready * 0.75)) * moving * (1.0 - raise);
        // Taunt: STANDING units of a CONFIDENT fighting regiment pump
        // the blade skyward for ~1.5 s every ~7 s, staggered per unit —
        // the rear ranks jeer while the front works. Wavering regiments
        // (morale low) stop jeering and just hold the brace.
        let tc = fract(globals.time / 7.3 + seed * 5.13);
        let tpulse = smoothstep(0.02, 0.12, tc) * (1.0 - smoothstep(0.24, 0.34, tc));
        let taunt =
            TAUNT_ON * tpulse * confident * (1.0 - moving) * (1.0 - smoothstep(0.0, 0.05, lunge));
        let ang_taunt = taunt * (1.7 + 0.22 * sin(globals.time * 16.0));

        // Style picked per SWING by the sim (swing bits): 0 = stab,
        // 1 = classic swing, 2 = slash (benched).
        var ang = 0.0;
        if celebrate > 0.001 {
            // Victory cheer: blade pumped skyward, bouncing with the hop.
            ang = celebrate * (1.75 + 0.35 * sin(globals.time * 9.0 + seed * 6.2831853));
        } else if style < 0.5 {
            // Stab: draw the arm back, then thrust the blade forward
            // near-level. Translation happens in local space (pre-yaw).
            ang = 0.55 * raise - 0.45 * chop;
            local.z += -0.30 * raise + 1.05 * chop;
        } else if style < 1.5 {
            // The classic swing: raise up/back, fast chop.
            ang = 1.9 * raise - 2.5 * chop;
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
        // Braced units hold a proper ready guard; the non-bracers of
        // the line keep the plain forward point (ang 0 standing).
        ang += sway + carry + ang_taunt + 0.6 * brace;
        local = pitch_about(local, pivot, ang);
        normal = pitch_normal(normal, ang);
    }

    // Walk bob + slight forward lean; lunge adds body punch, charging
    // adds a sprint lean and a heavier bob. Sprint lean is walk-gated
    // (the stance band is no longer walk-scaled) so jammed stragglers
    // don't posture.
    let bob = 0.05 * (1.0 + 0.4 * sprint) * moving * walk_s2;
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

    // Cheer hop: celebrating units bounce; braced-walk adds a slight
    // crouch on the advance too.
    let hop = 0.06 * celebrate * max(sin(globals.time * 9.0 + seed * 6.2831853), 0.0);
    // Wall stance carries a slight crouch (the planted, braced line).
    let position = local * vertex.i_pos_scale.w
        + vertex.i_pos_scale.xyz
        + vec3<f32>(
            0.0,
            bob - 0.15 * death - 0.07 * brace - 0.03 * ready * moving - 0.05 * wall + hop,
            0.0,
        );

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
