#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}
#import bevy_pbr::mesh_view_bindings::globals

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // NOT texture coords: x = body part id (0 body, 1 sword arm,
    // 2 left leg, 3 right leg, 4 spear arm, 5 shield arm, 6 bow arm),
    // y = the part's pivot height.
    @location(2) part_pivot: vec2<f32>,
    // Part material: rgb = fixed color, a = team-color blend amount.
    @location(5) v_color: vec4<f32>,

    // Per-instance: xyz = world position, w = uniform scale.
    @location(8) i_pos_scale: vec4<f32>,
    // rgb = team color, a = stable per-unit anim seed (not opacity).
    @location(9) i_color: vec4<f32>,
    // x = yaw, y = ground speed in m/s. z positive = attack: style*2 +
    // wind-up progress (style 0 stab, 1 slash); z negative = stance
    // band (-0.25 enemy near, -0.5 blade leveled, -1 charging).
    // w = fx: [0,1) hit-flash intensity, [1,2] = 1 + death progress.
    @location(10) i_anim: vec4<f32>,
    // x = leg length, hip to sole (0 on corpses), y = wall 0..1
    // (shieldwall/spearwall by bucket), z = gait phase in cycles,
    // w = stagger.
    @location(11) i_anim2: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// Standing brace pose (split legs, crouch, raised guard): read as
// weird in play-testing, benched but kept — set to 1.0 to re-enable.
// Standing units near an enemy hold the plain forward point instead.
const BRACE_ON: f32 = 0.0;
// Rear-rank taunt: never looked right in play-testing; benched
// pending a rework — set to 1.0 to re-enable.
const TAUNT_ON: f32 = 0.0;

const TAU: f32 = 6.2831853;
const PI: f32 = 3.14159265;

// How much a soldier is moving, 0..1, from his smoothed ground speed.
// The deadband sits over crowd jitter (about 0.03 m/s, devlog 0021) and
// it saturates by 1.2 m/s, so a slow shove still reads.
fn walk_gate(speed: f32) -> f32 {
    return smoothstep(0.06, 1.2, speed);
}

// Gait cycles per second at this ground speed (gait.rs `rate`).
fn gait_rate(speed: f32) -> f32 {
    return 1.0 + 0.16 * speed;
}

// Hip swing half angle at full stride.
const GAIT_SWING: f32 = 0.60;
// Where a foot lands ahead of the hip, as a share of how far behind the
// hip it leaves the ground.
const GAIT_FRONT: f32 = 0.5;
// The longest share of the cycle a foot stays down, a walk's.
const GAIT_DUTY_MAX: f32 = 0.62;

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

// The same rotation about a joint that has moved off the body axis, for
// the knee once the thigh has swung.
fn pitch_about_at(p: vec3<f32>, cy: f32, cz: f32, ang: f32) -> vec3<f32> {
    let c = cos(ang);
    let s = sin(ang);
    let y = p.y - cy;
    let z = p.z - cz;
    return vec3<f32>(p.x, cy + y * c + z * s, cz - y * s + z * c);
}

fn pitch_normal(n: vec3<f32>, ang: f32) -> vec3<f32> {
    let c = cos(ang);
    let s = sin(ang);
    return vec3<f32>(n.x, n.y * c + n.z * s, -n.y * s + n.z * c);
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    return unit_vertex(vertex);
}

#ifdef VERTEX_PULL
// GPU-built path (render_units_gpu.rs): ONE plain draw of soldiers *
// PULL_VERTS vertices per bucket, no vertex buffers, no instances. The
// soldier's record comes through the bucket's index list, the mesh corner
// from the expanded level mesh. Same pose code as the instanced entry, so
// the two paths cannot look different.
struct PullInstance {
    pos_scale: vec4<f32>,
    color: vec4<f32>,
    anim: vec4<f32>,
    anim2: vec4<f32>,
};

// The level mesh with its index list expanded (render_units_gpu.rs PullVertex).
struct PullVertex {
    position: vec3<f32>,
    part: f32,
    normal: vec3<f32>,
    pivot: f32,
    color: vec4<f32>,
};

@group(3) @binding(0) var<storage, read> pull_records: array<PullInstance>;
@group(3) @binding(1) var<storage, read> pull_index: array<u32>;
@group(3) @binding(2) var<storage, read> pull_vertices: array<PullVertex>;
// Per bucket: x = first index slot, y = corners per soldier.
@group(3) @binding(3) var<storage, read> pull_buckets: array<vec4<u32>>;

@vertex
fn vertex_pull(@builtin(vertex_index) index: u32) -> VertexOutput {
    let soldier = index / #{PULL_VERTS}u;
    let corner = index - soldier * #{PULL_VERTS}u;
    let entry = pull_index[pull_buckets[#{PULL_BUCKET}u].x + soldier];
    var inst = pull_records[entry & 0x3fffffffu];
#ifdef LOD_DEBUG
    // FL_LOD_DEBUG: tint by level (L1 green, L2 yellow, L3 red), the
    // level rides the top two bits of the index entry.
    let lod = entry >> 30u;
    if lod > 0u {
        var tint = vec3<f32>(1.0, 0.15, 0.1);
        if lod == 1u {
            tint = vec3<f32>(0.2, 1.0, 0.3);
        } else if lod == 2u {
            tint = vec3<f32>(1.0, 0.9, 0.1);
        }
        inst.color = vec4<f32>(inst.color.rgb * 0.4 + tint * 0.6, inst.color.a);
    }
#endif
    let v = pull_vertices[corner];
    return unit_vertex(Vertex(
        v.position,
        v.normal,
        vec2<f32>(v.part, v.pivot),
        v.color,
        inst.pos_scale,
        inst.color,
        inst.anim,
        inst.anim2,
    ));
}
#endif

fn unit_vertex(vertex: Vertex) -> VertexOutput {
    let yaw = vertex.i_anim.x;
    let speed = vertex.i_anim.y;
    let moving = walk_gate(speed);
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

    let wall = vertex.i_anim2.y;
    // Gait. The phase counts cycles of two steps, offset per soldier so
    // a block does not move in lockstep. A planted foot travels back
    // under the hip for `duty` of the cycle and has to cover the ground
    // the soldier covers meanwhile, `speed * duty / rate`. At speed that
    // is the full stride and the foot is down briefly, which is a run.
    // Slower, the foot stays down longer, up to a walk's share, and
    // below that the stride shortens. Either way the foot holds the
    // ground.
    let leg = vertex.i_anim2.x;
    let gait = fract(vertex.i_anim2.z + seed);
    let rate = gait_rate(speed);
    let full = (1.0 + GAIT_FRONT) * leg * sin(GAIT_SWING);
    let duty = min(full * rate / max(speed, 1e-3), GAIT_DUTY_MAX);
    let sweep = speed * duty / rate;
    // 0 standing, 1 at full stride.
    let stride = sweep / max(full, 1e-4);
    // How far behind the hip the planted foot leaves the ground.
    let reach = sweep / (1.0 + GAIT_FRONT);
    // 1 when both feet leave the ground between steps, 0 for a walk.
    let run = 1.0 - smoothstep(0.25, 0.55, duty);
    // The step wave the arms counter, and the idle wobble, which is the
    // one oscillator that is not locomotion.
    let limb = cos(TAU * gait);
    let wobble = globals.time * 9.0 + seed * TAU;
    // Hip drop. A walk vaults over the planted leg, lowest when the feet
    // are furthest apart. A run is lowest at mid-stance, where the knee
    // takes the landing, and highest in the air between steps. q is the
    // step phase, 0 at touchdown.
    let q = fract(2.0 * gait);
    let walk_dip =
        (leg - sqrt(max(leg * leg - reach * reach, 0.0))) * (0.5 + 0.5 * cos(TAU * q));
    let run_dip = stride * leg * (0.03 + 0.07 * (0.5 + 0.5 * cos(TAU * (q - duty))));
    let dip = mix(walk_dip, run_dip, run);

    // --- Part animation (rotations around the part pivot) ---
    if part > 6.5 {
        // Arrow projectile (arrows.rs buckets): rigid mesh, flight
        // pitch rides anim2.z (a dead channel for these instances —
        // march is always 0 here); yaw is the shared rotation below.
        let ang = vertex.i_anim2.z;
        local = pitch_about(local, 0.0, ang);
        normal = pitch_normal(normal, ang);
    } else if part > 5.5 {
        // Bow arm: stave carried vertical at the side. The draw tilts
        // arm and bow up toward the loft angle (the whole part pitches,
        // so the stave cants back over the shoulder — an archer aiming
        // high); the loose settles it, recover eases back to carry.
        // The draw hand is plain PART_ARM running the stab style: its
        // pull-back-then-snap IS the string draw and release.
        let raise = smoothstep(0.0, 0.8, lunge);
        let chop = smoothstep(0.85, 1.0, lunge);
        var ang = 0.75 * raise - 0.20 * chop
            - 0.06 * moving * limb * (1.0 - raise)
            + celebrate * (1.5 + 0.3 * sin(wobble));
        local = pitch_about(local, pivot, ang);
        normal = pitch_normal(normal, ang);
    } else if part > 4.5 {
        // Shield arm: carried at the side; the wall signal swings it
        // around the body to FACE THE FRONT and lifts it into a guard —
        // a shieldwall is a wall of team color from the enemy's side.
        // (Spear bucket: same fronting reads as the spearwall's off-hand
        // cover behind the leveled spears.) On the move it swings against
        // the weapon arm, except in a wall.
        let sway = -(0.12 + 0.20 * run) * moving * limb * (1.0 - wall);
        local = pitch_about(local, pivot, sway);
        normal = pitch_normal(normal, sway);
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
            + 0.05 * moving * limb * (1.0 - level)
            + 0.10 * celebrate * sin(wobble);
        local = pitch_about(local, pivot, ang);
        normal = pitch_normal(normal, ang);
        // Draw back, then punch the point home (the damage tick lands at
        // lunge 1.0, same timing as every other weapon).
        local.z += -0.30 * raise + 1.15 * chop;
    } else if part > 1.5 {
        // Corpses carry no leg length and keep the legs they fell with.
        if leg > 0.0 {
            // Legs, two bones each, posed from where the foot has to be.
            // Planted, it sweeps from `GAIT_FRONT * reach` ahead of the
            // hip to `reach` behind. Swinging, it lifts and comes forward
            // again. The knee bends by however far the foot sits inside
            // the leg's length.
            let side = select(1.0, -1.0, part > 2.5);
            let p = fract(gait + select(0.0, 0.5, part > 2.5));
            var fz = 0.0;
            var fy = dip - leg;
            if p < duty {
                fz = reach * (GAIT_FRONT - (1.0 + GAIT_FRONT) * p / duty);
            } else {
                let t = (p - duty) / (1.0 - duty);
                fz = reach * mix(-1.0, GAIT_FRONT, smoothstep(0.0, 1.0, t));
                // A runner tucks the heel up under him, a walker barely lifts.
                fy += stride * leg * (0.08 + 0.24 * run) * sin(PI * t);
            }
            // A braced or walled stance splits the feet, one forward one back.
            fz += (0.32 * brace + 0.22 * wall * (1.0 - moving)) * side * leg;
            let far = clamp(length(vec2<f32>(fz, fy)), 0.25 * leg, leg);
            let flex = 2.0 * acos(min(far / leg, 1.0));
            let thigh = atan2(fz, -fy) + 0.5 * flex;
            local = pitch_about(local, pivot, thigh);
            normal = pitch_normal(normal, thigh);
            // Below the knee the leg folds back by the flex, blended over a
            // band around the joint so the mesh bends instead of tearing.
            let down = clamp((pivot - vertex.position.y) / leg, 0.0, 1.0);
            let shin = smoothstep(0.38, 0.62, down);
            if shin > 0.001 {
                let knee = 0.5 * leg;
                local = pitch_about_at(
                    local,
                    pivot - knee * cos(thigh),
                    knee * sin(thigh),
                    -flex * shin,
                );
                normal = pitch_normal(normal, -flex * shin);
            }
        }
    } else if part > 0.5 {
        // Sword arm. Three per-unit attack styles (stable seed pick), all
        // timed so the blow lands exactly when the damage event fires
        // (lunge hits 1.0 at the strike tick): overhead chop, forward
        // stab, horizontal slash.
        let raise = smoothstep(0.0, 0.8, lunge);
        let chop = smoothstep(0.85, 1.0, lunge);
        // The arm counters the legs, and pumps harder at a run.
        let sway = (0.18 + 0.30 * run) * moving * limb * (1.0 - raise);
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
            ang = celebrate * (1.75 + 0.35 * sin(wobble));
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

    // The body rides the hip drop from the gait. A runner leans as far
    // forward as a charger.
    let bob = -dip;
    let lean = (0.10 * moving
        + 0.24 * max(run * moving, sprint * (0.25 + 0.75 * moving))
        + 0.30 * lunge)
        * clamp(local.y + 0.5, 0.0, 1.5);
    local.z += lean * 0.3;

    // Stagger (anim2.w = remaining stun, 1 at the blow -> 0 recovered):
    // rocked back off a charging blow or a spear point. The whole body
    // pitches backward from the feet with a short reel, hardest at the
    // hit and easing out as the man finds his balance; the knee-buckle
    // sink rides the position offset below.
    let stagger = vertex.i_anim2.w;
    if stagger > 0.001 {
        let reel = 1.0 + 0.18 * sin(globals.time * 22.0 + seed * TAU);
        let sang = -0.42 * stagger * stagger * reel;
        let sca = cos(sang);
        let ssa = sin(sang);
        let sfy = local.y + 0.5;
        local = vec3<f32>(local.x, sfy * sca - local.z * ssa - 0.5, sfy * ssa + local.z * sca);
        normal =
            vec3<f32>(normal.x, normal.y * sca - normal.z * ssa, normal.y * ssa + normal.z * sca);
    }

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
    let hop = 0.06 * celebrate * max(sin(wobble), 0.0);
    // Wall stance carries a slight crouch (the planted, braced line).
    let position = local * vertex.i_pos_scale.w
        + vertex.i_pos_scale.xyz
        + vec3<f32>(
            0.0,
            bob - 0.15 * death - 0.07 * brace - 0.03 * ready * moving - 0.05 * wall
                - 0.10 * stagger + hop,
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
