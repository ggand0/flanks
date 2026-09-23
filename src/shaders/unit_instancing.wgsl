#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}
#import bevy_pbr::mesh_view_bindings::globals

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // NOT texture coords: x = body part id (0 body, 1 sword arm,
    // 2 left leg, 3 right leg, 4 spear arm, 5 shield arm, 6 bow arm),
    // y = the part's pivot height.
    @location(2) part_pivot: vec2<f32>,
    // Where the vertex samples the kind's atlas (zero without one).
    @location(3) atlas_uv: vec2<f32>,
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
#ifdef UNIT_ATLAS
    // rgb = lit vertex colour, a = hit flash.
    @location(0) color: vec4<f32>,
    // rgb = team colour for the atlas mask, a = death darkening.
    @location(1) team: vec4<f32>,
    @location(2) atlas_uv: vec2<f32>,
#else
    @location(0) color: vec4<f32>,
#endif
};

#ifdef UNIT_ATLAS
// The kind's atlas: rgb colour, alpha the team tint mask. Only buckets
// with an atlas compile this path (render_units.rs `atlas_defs`).
@group(3) @binding(4) var unit_atlas: texture_2d<f32>;
@group(3) @binding(5) var unit_atlas_sampler: sampler;
#endif

// The weapon arm of this bucket's kind (unit_meshes.rs `Rig`): joints in
// the soldier's pitch plane, as (y, z) of local space in the rest pose.
struct Rig {
    shoulder: vec2<f32>,
    elbow: vec2<f32>,
    grip: vec2<f32>,
    // Unit direction from the grip toward the weapon's point.
    tip: vec2<f32>,
    // How far the weapon reaches behind the grip, to butt or pommel.
    rear: f32,
    // The weapon arm's part id, 0 when the arm has no rig.
    arm: f32,
    // 1 sword, 2 spear, 3 the archer's draw hand.
    hold: f32,
    pad: f32,
};
@group(3) @binding(6) var<uniform> rig: Rig;

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

// Ground a planted foot sweeps at full stride, in legs, and where it
// lands ahead of the hip as a share of how far behind it leaves.
const GAIT_SWEEP: f32 = 0.85;
const GAIT_FRONT: f32 = 0.6;
// The longest share of the cycle a foot stays down, a walk's.
const GAIT_DUTY_MAX: f32 = 0.62;
// Knee and ankle as a share of the leg down from the hip, and the ball
// of the foot ahead of the ankle, measured on the knight's leg.
const KNEE: f32 = 0.50;
const ANKLE: f32 = 0.90;
const BALL: f32 = 0.13;

// One soldier's gait this frame, the same for all his vertices.
struct Gait {
    leg: f32,
    // Share of the cycle a foot is down.
    duty: f32,
    // 0 standing, 1 at full stride.
    stride: f32,
    // Ground a planted foot sweeps, and how far behind the hip it leaves.
    sweep: f32,
    reach: f32,
    // 1 when both feet leave the ground between steps, 0 for a walk.
    run: f32,
};

// A planted foot travels back under the hip for `duty` of the cycle and
// covers the ground the soldier covers meanwhile, `speed * duty / rate`.
// At speed that is the full sweep and the foot is down briefly, a run.
// Slower, the foot stays down longer, up to a walk's share, and below
// that the stride shortens. Either way the foot holds the ground.
fn gait_at(speed: f32, leg: f32) -> Gait {
    let rate = gait_rate(speed);
    let full = GAIT_SWEEP * leg;
    let duty = min(full * rate / max(speed, 1e-3), GAIT_DUTY_MAX);
    let sweep = speed * duty / rate;
    return Gait(
        leg,
        duty,
        sweep / max(full, 1e-4),
        sweep,
        sweep / (1.0 + GAIT_FRONT),
        1.0 - smoothstep(0.25, 0.55, duty),
    );
}

// Hip drop at step phase q, 0 at touchdown. A walk vaults over the
// planted leg and is lowest when the front foot is furthest ahead. A run
// is lowest at mid-stance, where the knee takes the landing, and highest
// in the air between steps.
fn gait_dip(q: f32, g: Gait) -> f32 {
    let chain = ANKLE * g.leg;
    let front = GAIT_FRONT * g.reach;
    let walk = (chain - sqrt(max(chain * chain - front * front, 0.0))) * (0.5 + 0.5 * cos(TAU * q))
        + 0.015 * g.stride * g.leg;
    let run = g.stride * g.leg * (0.05 + 0.025 * (0.5 + 0.5 * cos(TAU * (q - g.duty))));
    return mix(walk, run, g.run);
}

// A (z, y) offset turned by a pitch angle, + taking +z toward +y.
fn pitch2(v: vec2<f32>, ang: f32) -> vec2<f32> {
    let c = cos(ang);
    let s = sin(ang);
    return vec2<f32>(v.x * c - v.y * s, v.y * c + v.x * s);
}

fn wrap_pi(x: f32) -> f32 {
    return x - TAU * floor((x + PI) / TAU);
}

// The smallest heel lift, zero or negative, that brings the ankle within
// r of the hip while the foot pivots on its ball at (bz, by). The ankle's
// squared distance from the hip is a constant plus
// `ka * cos(lift) + kb * sin(lift)`, and `kc` is what that sum may reach.
fn heel_lift(bz: f32, by: f32, a: f32, b: f32, r: f32) -> f32 {
    let ka = 2.0 * (a * by - b * bz);
    let kb = -2.0 * (a * bz + b * by);
    let kc = r * r - (bz * bz + by * by + a * a + b * b);
    if ka <= kc {
        return 0.0;
    }
    let n = length(vec2<f32>(ka, kb));
    if abs(kc) > n {
        return -1.2;
    }
    let delta = atan2(kb, ka);
    let w = acos(kc / n);
    // Of the two roots, the heel-up one nearest a flat foot.
    let r1 = wrap_pi(delta - w);
    let r2 = wrap_pi(delta + w);
    var lift = -1.2;
    if r1 <= 0.0 {
        lift = max(lift, r1);
    }
    if r2 <= 0.0 {
        lift = max(lift, r2);
    }
    return lift;
}

// Ankle (z, y) from the hip, and the foot's pitch, at share s of the
// stance. The ball of the foot is planted and sweeps back with the
// ground. The heel rises once a flat foot would pull the knee straight,
// and a runner also points the foot to push off.
fn stance_ankle(s: f32, g: Gait, dip: f32) -> vec3<f32> {
    let a = (1.0 - ANKLE) * g.leg;
    let b = BALL * g.leg;
    let ground = dip - g.leg;
    let ball = b + g.reach * (GAIT_FRONT - (1.0 + GAIT_FRONT) * s);
    // Only a foot behind the hip rises onto its ball, easing in as it
    // passes under.
    let need = heel_lift(ball, ground, a, b, 0.985 * ANKLE * g.leg)
        * (1.0 - smoothstep(-0.08 * g.leg, 0.0, ball));
    let push = -0.75 * g.run * g.stride * smoothstep(0.3, 1.0, s);
    let pitch = min(need, push);
    return vec3<f32>(vec2<f32>(ball, ground) + pitch2(vec2<f32>(-b, a), pitch), pitch);
}

fn hermite(p0: f32, m0: f32, p1: f32, m1: f32, u: f32) -> f32 {
    let u2 = u * u;
    let u3 = u2 * u;
    return (2.0 * u3 - 3.0 * u2 + 1.0) * p0 + (u3 - 2.0 * u2 + u) * m0
        + (3.0 * u2 - 2.0 * u3) * p1 + (u3 - u2) * m1;
}

// Thigh angle (+ forward) and knee flex that put the ankle at (z, y)
// from the hip.
fn leg_ik(ankle: vec2<f32>, leg: f32) -> vec2<f32> {
    let l1 = KNEE * leg;
    let l2 = (ANKLE - KNEE) * leg;
    let d = clamp(length(ankle), abs(l1 - l2) + 1e-4, l1 + l2 - 1e-6);
    let inner = acos(clamp((l1 * l1 + l2 * l2 - d * d) / (2.0 * l1 * l2), -1.0, 1.0));
    let lead = acos(clamp((l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d), -1.0, 1.0));
    return vec2<f32>(atan2(ankle.x, -ankle.y) + lead, PI - inner);
}

// Thigh, knee flex and the foot's pitch for a leg at phase p of its
// cycle, touchdown at 0.
fn leg_pose(p: f32, g: Gait) -> vec3<f32> {
    if p < g.duty {
        let st = stance_ankle(p / g.duty, g, gait_dip(fract(2.0 * p), g));
        return vec3<f32>(leg_ik(st.xy, g.leg), st.z);
    }
    let u = (p - g.duty) / (1.0 - g.duty);
    // Toe-off and touchdown, each at the hip height of its own moment.
    let off = stance_ankle(1.0, g, gait_dip(fract(2.0 * g.duty), g));
    let land = stance_ankle(0.0, g, gait_dip(0.0, g));
    // The foot leaves still travelling back and lands already sweeping
    // back, at a share of the stance speed. Between, it lifts early,
    // heel toward the seat, and reaches forward.
    let m = -g.sweep / g.duty * (1.0 - g.duty);
    let z = hermite(off.x, 0.30 * m, land.x, 0.20 * m, u);
    let lift = g.stride * g.leg * 0.34 * (0.25 + 0.75 * g.run)
        * sin(PI * pow(max(u, 1e-6), 0.7565));
    let y = mix(off.y, land.y, smoothstep(0.0, 1.0, u)) + lift;
    let pose = leg_ik(vec2<f32>(z, y), g.leg);
    // In the air the foot is set against the shin: it leaves pointed,
    // hangs about square to the shin and comes down flat.
    let at_off = leg_ik(off.xy, g.leg);
    let at_land = leg_ik(land.xy, g.leg);
    var rel = mix(off.z - (at_off.x - at_off.y), 0.05, smoothstep(0.0, 0.45, u));
    rel = mix(rel, at_land.y - at_land.x, smoothstep(0.55, 1.0, u));
    return vec3<f32>(pose, pose.x - pose.y + rel);
}

// Angle of a (y, z) direction: 0 straight down, PI/2 ahead, PI up. A
// pitch turn by `a` adds `a` to it.
fn yz_angle(v: vec2<f32>) -> f32 {
    return atan2(v.y, -v.x);
}

fn yz_dir(a: f32) -> vec2<f32> {
    return vec2<f32>(-cos(a), sin(a));
}

// A (y, z) offset turned by a pitch angle, + taking +z toward +y.
fn yz_turn(v: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(v.x * c + v.y * s, -v.x * s + v.y * c);
}

// Shoulder, elbow and wrist turns, from the rest pose, that put the grip
// at `goal` and point the weapon along `aim`. The elbow bends backward,
// as an arm does. Out of reach, the arm straightens toward the goal.
fn arm_pose(goal: vec2<f32>, aim: f32) -> vec3<f32> {
    let upper0 = rig.elbow - rig.shoulder;
    let fore0 = rig.grip - rig.elbow;
    let l1 = length(upper0);
    let l2 = length(fore0);
    let to = goal - rig.shoulder;
    let d = clamp(length(to), abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4);
    let lead = acos(clamp((l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d), -1.0, 1.0));
    let toward = yz_angle(to);
    let upper = toward - lead;
    let fore = yz_angle(d * yz_dir(toward) - l1 * yz_dir(upper));
    let shoulder = wrap_pi(upper - yz_angle(upper0));
    let fore_turn = wrap_pi(fore - yz_angle(fore0));
    return vec3<f32>(
        shoulder,
        wrap_pi(fore_turn - shoulder),
        wrap_pi(aim - yz_angle(rig.tip) - fore_turn),
    );
}

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
    // unorm8 rgba.
    color: u32,
    // unorm16 each.
    atlas_uv: u32,
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
        unpack2x16unorm(v.atlas_uv),
        unpack4x8unorm(v.color),
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
    // a block does not move in lockstep.
    let leg = vertex.i_anim2.x;
    let gait = fract(vertex.i_anim2.z + seed);
    let g = gait_at(speed, leg);
    let run = g.run;
    // The step wave the arms counter, and the idle wobble, which is the
    // one oscillator that is not locomotion.
    let limb = cos(TAU * gait);
    let wobble = globals.time * 9.0 + seed * TAU;
    let dip = gait_dip(fract(2.0 * gait), g);

    // --- Part animation (rotations around the part pivot) ---
    // Attack progress, shared by every arm: raise over the wind-up, chop
    // at the strike.
    let raise = smoothstep(0.0, 0.8, lunge);
    let chop = smoothstep(0.85, 1.0, lunge);
    if rig.arm > 0.5 && (part > 7.5 || abs(part - rig.arm) < 0.5) {
        // The weapon arm and its weapon: shoulder, elbow and wrist. The
        // hand stays on the arm and the arm on the shoulder, and the
        // weapon turns and slides in the hand.
        let reach = length(rig.elbow - rig.shoulder) + length(rig.grip - rig.elbow);
        let rest = yz_angle(rig.tip);
        var goal = rig.grip;
        var aim = rest;
        // Turn of the whole arm about the shoulder, on top of the pose.
        var turn = 0.0;
        var slide = 0.0;
        if rig.hold > 1.5 && rig.hold < 2.5 {
            // Spear. It rides upright at rest and on the march. A watch
            // range advance, a fight, a charge or a wall levels it: the
            // upper arm hangs, the forearm points ahead with the shaft
            // along it, and the grip moves back toward the butt for reach.
            let level = max(max(stance, ready * 0.75), max(max(sprint, raise), wall))
                * (1.0 - celebrate);
            aim = rest + wrap_pi(0.5 * PI + 0.05 - rest) * level;
            let l1 = length(rig.elbow - rig.shoulder);
            let l2 = length(rig.grip - rig.elbow);
            let levelled = rig.shoulder + vec2<f32>(-l1, 0.95 * l2);
            // Draw back, then drive the point home along the shaft. The
            // damage tick lands at lunge 1.0, as for every weapon.
            goal = mix(rig.grip, levelled, level)
                + yz_dir(aim) * reach * (-0.25 * raise + 0.9 * chop);
            slide = 0.58 * rig.rear * level;
            turn = 0.05 * moving * limb * (1.0 - level) + 0.10 * celebrate * sin(wobble);
        } else if rig.hold > 2.5 {
            // The archer's draw hand pulls the string back to the jaw and
            // lets go, lifted with the bow toward the loft angle.
            let jaw = rig.shoulder + vec2<f32>(0.2, 0.15) * reach;
            goal = mix(rig.grip, jaw, raise * (1.0 - chop));
            turn = 0.75 * raise - 0.20 * chop
                - 0.06 * moving * limb * (1.0 - raise)
                + celebrate * (1.5 + 0.3 * sin(wobble));
        } else {
            // Sword. The arm counters the legs, carries the blade lowered
            // on the move and levels it in battle stance.
            let sway = (0.18 - 0.06 * run) * moving * limb * (1.0 - raise);
            let carry = mix(-0.55, 0.25, max(stance, ready * 0.75)) * moving * (1.0 - raise);
            let tc = fract(globals.time / 7.3 + seed * 5.13);
            let tpulse = smoothstep(0.02, 0.12, tc) * (1.0 - smoothstep(0.24, 0.34, tc));
            let taunt = TAUNT_ON * tpulse * confident * (1.0 - moving)
                * (1.0 - smoothstep(0.0, 0.05, lunge));
            if celebrate > 0.001 {
                // Victory cheer: blade pumped skyward.
                turn = celebrate * (1.75 + 0.35 * sin(wobble));
            } else if style < 0.5 {
                // Stab: the hand pulls back, then thrusts along the blade,
                // the wrist keeping the point near level.
                goal = rig.grip + yz_dir(rest) * reach * (-0.3 * raise + 0.7 * chop);
                aim = rest + 0.15 * raise - 0.1 * chop;
            } else if style < 1.5 {
                // The classic swing: raise up and back, fast chop.
                turn = 1.9 * raise - 2.5 * chop;
            } else {
                // Slash: a sweep around the body axis, wind back, cut across.
                let yawoff = -1.1 * raise + 2.3 * chop;
                local = rot_y(local, cos(yawoff), sin(yawoff));
                normal = rot_y(normal, cos(yawoff), sin(yawoff));
                turn = 0.5 * raise - 0.3 * chop;
            }
            turn += sway + carry + taunt * (1.7 + 0.22 * sin(globals.time * 16.0)) + 0.6 * brace;
        }
        let pose = arm_pose(goal, aim);
        let shoulder = pose.x + turn;
        // Past the elbow the forearm turns, blended over a band so the
        // sleeve bends. The weapon turns with the hand.
        let upper = rig.elbow - rig.shoulder;
        let past = dot(vertex.position.yz - rig.elbow, upper) / dot(upper, upper);
        let weapon = part > 7.5;
        let fore = select(smoothstep(-0.15, 0.15, past), 1.0, weapon);
        local = pitch_about_at(local, rig.shoulder.x, rig.shoulder.y, shoulder);
        let elbow = rig.shoulder + yz_turn(upper, shoulder);
        local = pitch_about_at(local, elbow.x, elbow.y, pose.y * fore);
        var total = shoulder + pose.y * fore;
        if weapon {
            let grip = elbow + yz_turn(rig.grip - rig.elbow, shoulder + pose.y);
            local = pitch_about_at(local, grip.x, grip.y, pose.z);
            let way = yz_dir(aim + turn) * slide;
            local.y += way.x;
            local.z += way.y;
            total += pose.z;
        }
        normal = pitch_normal(normal, total);
    } else if part > 6.5 && part < 7.5 {
        // Arrow projectile (arrows.rs buckets): rigid mesh, flight
        // pitch rides anim2.z (a dead channel for these instances —
        // march is always 0 here); yaw is the shared rotation below.
        let ang = vertex.i_anim2.z;
        local = pitch_about(local, 0.0, ang);
        normal = pitch_normal(normal, ang);
    } else if part > 5.5 && part < 6.5 {
        // Bow arm: stave carried vertical at the side. The draw tilts
        // arm and bow up toward the loft angle (the whole part pitches,
        // so the stave cants back over the shoulder — an archer aiming
        // high); the loose settles it, recover eases back to carry.
        // The draw hand is plain PART_ARM running the stab style: its
        // pull-back-then-snap IS the string draw and release.
        var ang = 0.75 * raise - 0.20 * chop
            - 0.06 * moving * limb * (1.0 - raise)
            + celebrate * (1.5 + 0.3 * sin(wobble));
        local = pitch_about(local, pivot, ang);
        normal = pitch_normal(normal, ang);
    } else if part > 4.5 && part < 5.5 {
        // Shield arm: carried at the side; the wall signal swings it
        // around the body to FACE THE FRONT and lifts it into a guard —
        // a shieldwall is a wall of team color from the enemy's side.
        // (Spear bucket: same fronting reads as the spearwall's off-hand
        // cover behind the leveled spears.) On the move it swings against
        // the weapon arm, except in a wall.
        let sway = -0.06 * moving * limb * (1.0 - wall);
        local = pitch_about(local, pivot, sway);
        normal = pitch_normal(normal, sway);
        if wall > 0.001 {
            // The shield is on +X, the left hand: turning it forward is a
            // negative turn about Y.
            let ang = -1.05 * wall;
            let c2 = cos(ang);
            let s2 = sin(ang);
            local = rot_y(local, c2, s2);
            normal = rot_y(normal, c2, s2);
            local.y += 0.10 * wall;
        }
    } else if part > 3.5 && part < 4.5 {
        // Spear arm without a rig: arm and shaft turn as one about the
        // shoulder. Battle stance, a watch-range advance, a charge or a
        // wall levels the point, and a stab tips it forward.
        let level = max(max(stance, ready * 0.75), max(max(sprint, raise), wall));
        let ang = -1.42 * level * (1.0 - celebrate)
            - 0.15 * chop
            + 0.05 * moving * limb * (1.0 - level)
            + 0.10 * celebrate * sin(wobble);
        local = pitch_about(local, pivot, ang);
        normal = pitch_normal(normal, ang);
    } else if part > 1.5 && part < 3.5 {
        // Corpses carry no leg length and keep the legs they fell with.
        if leg > 0.0 {
            // Hip, knee and ankle, posed by `leg_pose`. Knee and ankle
            // bend over a band around each joint, so the mesh bends
            // instead of tearing.
            let side = select(1.0, -1.0, part > 2.5);
            let pose = leg_pose(fract(gait + select(0.0, 0.5, part > 2.5)), g);
            // A braced or walled stance splits the feet, one forward one back.
            let thigh = pose.x + (0.32 * brace + 0.22 * wall * (1.0 - moving)) * side;
            let flex = pose.y;
            local = pitch_about(local, pivot, thigh);
            normal = pitch_normal(normal, thigh);
            let down = clamp((pivot - vertex.position.y) / leg, 0.0, 1.0);
            let knee = KNEE * leg;
            let kz = knee * sin(thigh);
            let ky = pivot - knee * cos(thigh);
            let bend = -flex * smoothstep(KNEE - 0.1, KNEE + 0.1, down);
            local = pitch_about_at(local, ky, kz, bend);
            normal = pitch_normal(normal, bend);
            // The foot turns about the ankle to the pitch the pose asks for.
            let shin = thigh - flex;
            let shank = (ANKLE - KNEE) * leg;
            let foot = (pose.z - shin) * smoothstep(ANKLE - 0.06, ANKLE + 0.02, down);
            local = pitch_about_at(local, ky - shank * cos(shin), kz + shank * sin(shin), foot);
            normal = pitch_normal(normal, foot);
        }
    } else if part > 0.5 {
        // Sword arm without a rig, and its weapon: they turn as one about
        // the shoulder. Three per-unit attack styles (stable seed pick), all
        // timed so the blow lands exactly when the damage event fires
        // (lunge hits 1.0 at the strike tick): overhead chop, forward
        // stab, horizontal slash.
        // The arm counters the legs. A runner holds the blade steadier.
        let sway = (0.18 - 0.06 * run) * moving * limb * (1.0 - raise);
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
            // Stab: draw the arm up and back, then drop the point forward.
            ang = 0.55 * raise - 0.45 * chop;
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

    // The shoulders turn against the legs and the upper body shifts over
    // the planted foot, easing in up the torso so the hips stay square.
    if part < 1.5 || (part > 3.5 && part < 6.5) || part > 7.5 {
        let w = select(1.0, smoothstep(0.0, 0.25, vertex.position.y), part < 0.5);
        let turn = 0.10 * g.run * g.stride * limb * w;
        local = rot_y(local, cos(turn), sin(turn));
        normal = rot_y(normal, cos(turn), sin(turn));
        local.x += 0.02 * leg * g.run * g.stride * cos(TAU * (gait - 0.5 * g.duty)) * w;
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
    // hit flash lerps toward white and death darkens. With an atlas the
    // fragment does the last two after sampling it.
    let base = mix(vertex.v_color.rgb, vertex.i_color.rgb, vertex.v_color.a);
    let flash = clamp(fx, 0.0, 1.0) * step(fx, 1.0);
#ifdef UNIT_ATLAS
    out.color = vec4<f32>(base * light, flash);
    out.team = vec4<f32>(vertex.i_color.rgb, death);
    out.atlas_uv = vertex.atlas_uv;
#else
    var rgb = base * light;
    rgb = mix(rgb, vec3<f32>(1.0, 1.0, 1.0), flash * 0.8);
    rgb = rgb * (1.0 - 0.45 * death);
    out.color = vec4<f32>(rgb, 1.0);
#endif
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
#ifdef UNIT_ATLAS
    // The atlas mask tints rather than replaces, so cloth keeps its weave
    // under the team colour.
    let texel = textureSample(unit_atlas, unit_atlas_sampler, in.atlas_uv);
    let tint = mix(vec3<f32>(1.0), in.team.rgb, texel.a);
    var rgb = in.color.rgb * texel.rgb * tint;
    rgb = mix(rgb, vec3<f32>(1.0, 1.0, 1.0), in.color.a * 0.8);
    rgb = rgb * (1.0 - 0.45 * in.team.a);
    return vec4<f32>(rgb, 1.0);
#else
    return in.color;
#endif
}
