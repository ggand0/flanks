#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}
#import bevy_pbr::mesh_view_bindings::globals

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // NOT texture coords: x = body part id (0 body, 1 sword arm,
    // 2 left leg, 3 right leg, 4 spear arm, 5 shield arm, 6 bow arm,
    // 7 arrow, 8 weapon, 9 and 10 a jointed weapon arm's forearm and
    // hand, and an archer's bow rig: 11 and 12 the bow forearm and hand,
    // 13 and 17 the string halves, 14 and 15 the limbs, 16 the held
    // arrow, 18 head, 19 torso), y = the part's pivot height.
    @location(2) part_pivot: vec2<f32>,
    // Where the vertex samples the kind's atlas (zero without one).
    @location(3) atlas_uv: vec2<f32>,
    // Part material: rgb = fixed color, a = team-color blend amount.
    @location(5) v_color: vec4<f32>,

    // Per-instance: xyz = world position. w = an arrow's uniform scale,
    // or how far a soldier's bow is up from the carry toward the drawn
    // ready pose, 0 to 1 (render_units.rs InstanceData).
    @location(8) i_pos_scale: vec4<f32>,
    // rgb = team color, a = stable per-unit anim seed (not opacity).
    @location(9) i_color: vec4<f32>,
    // x = yaw, y = ground speed in m/s. z = attack or cheer
    // (render_units.rs CELEBRATE_BASE), 0 for neither. w = fx: [0,1)
    // hit-flash intensity, [1,2] = 1 + death progress, 2 on a corpse.
    @location(10) i_anim: vec4<f32>,
    // x = stance band (0.25 enemy near, 0.5 fighting, 1 charging),
    // y = wall 0..1 (shieldwall/spearwall by bucket), z = gait phase in
    // cycles, w = stagger.
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

// This bucket's kind (unit_meshes.rs `Rig`): its leg, and its weapon
// arm's joints in the soldier's pitch plane, as (y, z) of local space in
// the rest pose.
struct Rig {
    shoulder: vec2<f32>,
    elbow: vec2<f32>,
    wrist: vec2<f32>,
    grip: vec2<f32>,
    // Unit direction from the grip toward the weapon's point.
    tip: vec2<f32>,
    // How far the weapon reaches behind the grip, to butt or pommel.
    rear: f32,
    // The weapon arm's part id, 0 when the arm has no rig.
    arm: f32,
    // 1 sword, 2 spear, 3 the archer's draw hand.
    hold: f32,
    // How far the weapon slides through the hand once levelled.
    slide: f32,
    // Hip to sole.
    leg: f32,
    // 1 for a jointed arm: upper arm, forearm (9) and hand (10).
    chain: f32,
    // A jointed arm's attack, even samples over the wind-up and over the
    // follow-through: shoulder, elbow and wrist turns from the rest pose,
    // and how far the weapon is levelled. windup[0] is the guard.
    windup: array<vec4<f32>, 33>,
    recover: array<vec4<f32>, 33>,
    bow: Bow,
};

// An archer's jointed arms and strung bow (unit_meshes.rs `Bow`), local
// space.
struct Bow {
    // The drawing arm's shoulder, elbow and wrist.
    draw: array<vec4<f32>, 3>,
    // The bow arm's shoulder, elbow and wrist, and the bow's grip.
    hold: array<vec4<f32>, 4>,
    // Where the upper and lower limbs bend.
    limbs: array<vec4<f32>, 2>,
    // The string's ends at rest, upper then lower.
    tips: array<vec4<f32>, 2>,
    // The nock at rest, where the string halves and the arrow turn.
    nock: vec4<f32>,
    neck: vec4<f32>,
    waist: vec4<f32>,
    // x = each string half's length, y = how far the arrow passes beside
    // the grip, z = share of the reload before the next arrow shows,
    // w = 1 when the kind has this rig.
    params: vec4<f32>,
    // Start (in vec4s) and sample count of each table in `clips`:
    // (raise, release) then (reload, the reload's free-arrow rows).
    clips: array<vec4<u32>, 2>,
};
@group(3) @binding(6) var<uniform> rig: Rig;
// The shot tables of this bucket's kind (unit_glb.rs `Shots::buffer`).
// A pose is 8 vec4s: six joint turns as xyzw quaternions (the drawing
// arm's shoulder, elbow and wrist, then the bow arm's), (bow yaw, limb
// bend, arrow shown, body yaw), (torso pitch, bow pitch). A free-arrow
// row is 2: (nock position, how far it has settled onto the string),
// then its turn.
@group(3) @binding(7) var<storage, read> clips: array<vec4<f32>>;

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

// A jointed arm's attack pose at progress x of the wind-up, or of the
// follow-through, straight between the two nearest samples.
fn attack_pose(follow: bool, x: f32) -> vec4<f32> {
    let t = clamp(x, 0.0, 1.0) * 32.0;
    let i = min(u32(t), 31u);
    var a = rig.windup[i];
    var b = rig.windup[i + 1u];
    if follow {
        a = rig.recover[i];
        b = rig.recover[i + 1u];
    }
    return mix(a, b, t - f32(i));
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

const QI: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 1.0);

fn qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Turns about +X and +Y, right-handed.
fn qx(a: f32) -> vec4<f32> {
    return vec4<f32>(sin(0.5 * a), 0.0, 0.0, cos(0.5 * a));
}

fn qy(a: f32) -> vec4<f32> {
    return vec4<f32>(0.0, sin(0.5 * a), 0.0, cos(0.5 * a));
}

// The shortest turn taking unit direction a onto b.
fn qalign(a: vec3<f32>, b: vec3<f32>) -> vec4<f32> {
    return normalize(vec4<f32>(cross(a, b), 1.0 + dot(a, b)));
}

// Where a shot is in its table: the first of the two poses it lies
// between, how far toward the second, and where the table starts.
struct ShotAt {
    at: u32,
    i: u32,
    f: f32,
};

fn shot_at(clip: u32, x: f32) -> ShotAt {
    var span = rig.bow.clips[0].xy;
    if clip == 1u {
        span = rig.bow.clips[0].zw;
    } else if clip == 2u {
        span = rig.bow.clips[1].xy;
    }
    let t = clamp(x, 0.0, 1.0) * f32(span.y - 1u);
    let i = min(u32(t), span.y - 2u);
    return ShotAt(span.x, i, t - f32(i));
}

// Joint turn j of a shot, blended from the carry by w.
fn shot_turn(s: ShotAt, j: u32, w: f32) -> vec4<f32> {
    let a = clips[s.at + s.i * 8u + j];
    var b = clips[s.at + (s.i + 1u) * 8u + j];
    b = select(b, -b, dot(a, b) < 0.0);
    var q = normalize(mix(a, b, s.f));
    q = select(q, -q, q.w < 0.0);
    return normalize(mix(QI, q, w));
}

// Scalar row k of a shot (0: bow yaw, limb bend, arrow shown, body yaw;
// 1: torso pitch, bow pitch).
fn shot_scalars(s: ShotAt, k: u32) -> vec4<f32> {
    return mix(clips[s.at + s.i * 8u + 6u + k], clips[s.at + (s.i + 1u) * 8u + 6u + k], s.f);
}

struct Posed {
    pos: vec3<f32>,
    // The part's turn, for its normals.
    turn: vec4<f32>,
};

// One vertex of an archer's bow rig, posed by shot `clip` at progress x
// and blended from the carry by w: the arm chains, bow, string and arrow
// as tools/blender/archer/motion.py `transforms` builds them, then the
// body's yaw and the torso's pitch about the waist. The rigid march and
// cheer swings of the old arms, draw_ang and bow_ang, fade out as the
// bow comes up.
fn bow_pose(part: f32, p: vec3<f32>, clip: u32, x: f32, w: f32, draw_ang: f32, bow_ang: f32) -> Posed {
    let s = shot_at(clip, x);
    let lead = shot_scalars(s, 0u) * w;
    let tail = shot_scalars(s, 1u) * w;
    let body = qy(lead.w);
    let pitch = qx(-tail.x);
    let waist = rig.bow.waist.xyz;
    if abs(part - 18.0) < 0.5 {
        // The head rides the torso but keeps facing the shot.
        let neck = rig.bow.neck.xyz;
        let moved = waist + qrot(pitch, qrot(body, neck) - waist);
        return Posed(moved + qrot(pitch, p - neck), pitch);
    }
    var at = p;
    var turn = QI;
    let drawing = abs(part - 1.0) < 0.5 || abs(part - 9.0) < 0.5 || abs(part - 10.0) < 0.5;
    if drawing || (part > 5.5 && part < 17.5 && !(part > 6.5 && part < 7.5)) {
        // Shoulder, elbow and wrist, each carrying the next.
        var first = 0u;
        var joints = array<vec3<f32>, 3>(rig.bow.draw[0].xyz, rig.bow.draw[1].xyz, rig.bow.draw[2].xyz);
        if !drawing {
            first = 3u;
            joints = array<vec3<f32>, 3>(rig.bow.hold[0].xyz, rig.bow.hold[1].xyz, rig.bow.hold[2].xyz);
        }
        let q1 = shot_turn(s, first, w);
        let q2 = qmul(q1, shot_turn(s, first + 1u, w));
        let q3 = qmul(q2, shot_turn(s, first + 2u, w));
        let elbow = joints[0] + qrot(q1, joints[1] - joints[0]);
        let wrist = elbow + qrot(q2, joints[2] - joints[1]);
        if abs(part - 1.0) < 0.5 || abs(part - 6.0) < 0.5 {
            at = joints[0] + qrot(q1, p - joints[0]);
            turn = q1;
        } else if abs(part - 9.0) < 0.5 || abs(part - 11.0) < 0.5 {
            at = elbow + qrot(q2, p - joints[1]);
            turn = q2;
        } else if abs(part - 10.0) < 0.5 || abs(part - 12.0) < 0.5 {
            at = wrist + qrot(q3, p - joints[2]);
            turn = q3;
        } else {
            // The bow in the bow hand, with its own yaw and pitch.
            let g = rig.bow.hold[3].xyz;
            let grip = wrist + qrot(q3, g - joints[2]);
            let held = qmul(qmul(vec4<f32>(-body.xyz, body.w), qx(-tail.y)), qy(lead.x));
            if abs(part - 8.0) < 0.5 {
                at = grip + qrot(held, p - g);
                turn = held;
            } else {
                // The limbs bend opposite ways about their own ends of
                // the grip, and each string half keeps its length from
                // its tip to where the two meet.
                let upper = qmul(held, qx(-lead.y));
                let lower = qmul(held, qx(lead.y));
                let l0 = rig.bow.limbs[0].xyz;
                let l1 = rig.bow.limbs[1].xyz;
                let top = grip + qrot(held, l0 - g) + qrot(upper, rig.bow.tips[0].xyz - l0);
                let bottom = grip + qrot(held, l1 - g) + qrot(lower, rig.bow.tips[1].xyz - l1);
                let half = 0.5 * length(top - bottom);
                let sag = sqrt(max(rig.bow.params.x * rig.bow.params.x - half * half, 0.0));
                let nock = 0.5 * (top + bottom) - qrot(held, vec3<f32>(0.0, 0.0, sag));
                let rest = rig.bow.nock.xyz;
                if abs(part - 14.0) < 0.5 {
                    at = grip + qrot(held, l0 - g) + qrot(upper, p - l0);
                    turn = upper;
                } else if abs(part - 15.0) < 0.5 {
                    at = grip + qrot(held, l1 - g) + qrot(lower, p - l1);
                    turn = lower;
                } else if abs(part - 13.0) < 0.5 {
                    turn = qalign(normalize(rig.bow.tips[0].xyz - rest), normalize(top - nock));
                    at = nock + qrot(turn, p - rest);
                } else if abs(part - 17.0) < 0.5 {
                    turn = qalign(normalize(rig.bow.tips[1].xyz - rest), normalize(bottom - nock));
                    at = nock + qrot(turn, p - rest);
                } else {
                    // The held arrow lies from the nock to beside the grip.
                    // In the reload it comes out of the quiver free and
                    // settles onto the string.
                    turn = qalign(
                        vec3<f32>(0.0, 0.0, 1.0),
                        normalize(grip + qrot(held, vec3<f32>(rig.bow.params.y, 0.0, 0.0)) - nock),
                    );
                    var origin = nock;
                    var shown = clips[s.at + s.i * 8u + 6u].z > 0.5;
                    if clip == 2u {
                        shown = x >= rig.bow.params.z;
                        let row = rig.bow.clips[1].z + s.i * 2u;
                        let a = clips[row];
                        let b = clips[row + 2u];
                        let settled = mix(a.w, b.w, s.f);
                        let qa = clips[row + 1u];
                        var qb = clips[row + 3u];
                        qb = select(qb, -qb, dot(qa, qb) < 0.0);
                        var free = normalize(mix(qa, qb, s.f));
                        free = select(free, -free, dot(free, turn) < 0.0);
                        turn = normalize(mix(free, turn, settled));
                        origin = mix(mix(a.xyz, b.xyz, s.f), nock, settled);
                        // A free arrow follows the reload's hand, which is only
                        // there with the bow fully up.
                        shown = shown && (settled > 0.999 || w > 0.999);
                    }
                    at = select(nock, origin + qrot(turn, p - rest), shown && w > 0.5);
                }
            }
        }
        // The old rigid swings, about the shoulder, fading as the bow
        // comes up.
        var ang = bow_ang;
        if drawing {
            ang = draw_ang;
        }
        let swing = qx(-ang * (1.0 - w));
        at = joints[0] + qrot(swing, at - joints[0]);
        turn = qmul(swing, turn);
    }
    // The upper body turns side-on about the vertical axis and leans at
    // the waist.
    return Posed(waist + qrot(pitch, qrot(body, at) - waist), qmul(pitch, qmul(body, turn)));
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
    // A bow shot from 16: clip * 2 + progress, clip 0 the raise, 1 the
    // release, 2 the reload (render_units.rs RANGED_BASE). A kind with a
    // bow rig plays it from its tables. Without one the draw arms read it
    // as a stab: the raise is the wind-up and the release the
    // follow-through.
    let bow_rig = rig.bow.params.w > 0.5;
    let zraw = max(vertex.i_anim.z, 0.0);
    let shooting = zraw > 15.5;
    let shot_clip = floor((zraw - 16.0) * 0.5 + 0.001);
    let shot_x = clamp(zraw - 16.0 - shot_clip * 2.0, 0.0, 1.0);
    var zpos = zraw;
    if shooting {
        zpos = 0.0;
        if !bow_rig && shot_clip < 0.5 {
            zpos = shot_x;
        } else if !bow_rig && shot_clip < 1.5 {
            zpos = 1.4 + 0.59 * shot_x;
        }
    }
    // How far the bow is up from the carry toward the drawn ready pose,
    // and the clip it poses: between shots, the raise at its start.
    let bow_w = select(0.0, smoothstep(0.0, 1.0, vertex.i_pos_scale.w), bow_rig);
    var clip = 0u;
    var clip_x = 0.0;
    if shooting {
        clip = u32(clamp(shot_clip, 0.0, 2.0));
        clip_x = shot_x;
    }
    // Attack: the 2s digit is the swing style (0 stab, 1 classic swing,
    // 2 slash), plus 3 on a charging blow. The remainder is the wind-up,
    // 0..1 linear in time, or from 1.4 the follow-through after the blow
    // landed (render_units.rs FOLLOW_BASE). From 12 the victory cheer.
    let cheering = zpos > 11.5;
    let digit = floor(zpos * 0.5 + 0.001);
    let charge = digit > 2.5 && !cheering;
    let style = digit - select(0.0, 3.0, charge);
    let within = zpos - digit * 2.0;
    let following = within > 1.395 && !cheering;
    let windup = select(select(within, 0.0, following), 0.0, cheering);
    let follow = clamp((within - 1.4) / 0.59, 0.0, 1.0);
    // The lunge the code-posed arms and the body lean follow: quadratic
    // over the wind-up and harder on a charge, where it raises from the
    // levelled run-in point instead of dipping the blade first. After the
    // blow it holds a moment and eases back to 0.
    let amp = select(1.0, 1.35, charge);
    var wound = windup * windup * amp;
    if charge {
        wound = max(wound, 0.18);
    }
    let settle = 1.0 - smoothstep(0.25, 1.0, follow);
    let lunge = select(wound, settle * amp, following);
    // Attack progress for every arm: `raise` draws back over the wind-up,
    // `chop` drives the blow home over its last stretch so it lands on the
    // strike tick, and both ease back together after it.
    let raise = select(smoothstep(0.0, 0.55, wound), settle, following);
    let chop = select(smoothstep(0.55, 1.0, wound), settle, following);
    // Cheer: z = 12 + progress. Ease in over the first ~8% and back out
    // over the last ~10% — poses must never snap in one frame.
    let cele_t = clamp(zpos - 12.0, 0.0, 1.0);
    let celebrate = select(0.0, 1.0, cheering)
        * smoothstep(0.0, 0.08, cele_t)
        * (1.0 - smoothstep(0.90, 1.0, cele_t));
    // Stance band: 0.25 = enemy in watch range (brace when standing),
    // 0.5 = fighting but wavering (brace, no taunt), 0.65 = fighting
    // confident (taunts), 1 = charging (sprint).
    let band = max(vertex.i_anim2.x, 0.0);
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
    // a block does not move in lockstep. The fallen keep the legs they
    // fell with.
    let leg = select(0.0, rig.leg, fx < 1.99);
    let gait = fract(vertex.i_anim2.z + seed);
    let g = gait_at(speed, leg);
    let run = g.run;
    // The step wave the arms counter, and the idle wobble, which is the
    // one oscillator that is not locomotion.
    let limb = cos(TAU * gait);
    let wobble = globals.time * 9.0 + seed * TAU;
    let dip = gait_dip(fract(2.0 * gait), g);

    // --- Part animation (rotations around the part pivot) ---
    if bow_rig && (abs(part - 1.0) < 0.5 || abs(part - 6.0) < 0.5 || (part > 7.5 && part < 19.5)) {
        // An archer's bow rig. A shot plays its clip, and between shots
        // the drawn ready pose (the raise at its start) eases in and out
        // with the bow. The drawing arm keeps the old march, cheer and
        // melee swings and the bow arm its march and cheer swings, rigid
        // about each shoulder, while the bow is down.
        let sway = (0.18 - 0.06 * run) * moving * limb * (1.0 - raise);
        let carry = mix(-0.55, 0.25, max(stance, ready * 0.75)) * moving * (1.0 - raise);
        var draw_ang = select(1.9 * raise - 2.5 * chop, 0.55 * raise - 0.45 * chop, style < 0.5);
        draw_ang = select(draw_ang, celebrate * (1.75 + 0.35 * sin(wobble)), celebrate > 0.001);
        draw_ang += sway + carry + 0.6 * brace;
        let bow_ang = -0.06 * moving * limb + celebrate * (1.5 + 0.3 * sin(wobble));
        let posed = bow_pose(part, local, clip, clip_x, bow_w, draw_ang, bow_ang);
        local = posed.pos;
        normal = qrot(posed.turn, normal);
    } else if rig.chain > 0.5 && (abs(part - rig.arm) < 0.5 || (part > 7.5 && part < 10.5)) {
        // A jointed weapon arm: upper arm, forearm and hand turn at the
        // shoulder, elbow and wrist, each carrying the next, and the
        // weapon turns and slides through the hand. The rest pose is the
        // carry. Readiness levels the weapon into the guard, and an attack
        // plays the kind's tables from the guard and back to it.
        let ready_level = max(max(stance, ready * 0.75), max(sprint, wall)) * (1.0 - celebrate);
        // Carry to the guard is a straight blend of the joint turns, and
        // so is carry to any attack pose. A wind-up that starts short of
        // the guard reaches it in its first third, and the follow-through
        // ends in the guard and settles to readiness as it runs out.
        var pose = rig.windup[0] * ready_level;
        if following {
            pose = attack_pose(true, follow) * max(ready_level, 1.0 - smoothstep(0.92, 1.0, follow));
        } else if windup > 0.0 {
            pose = attack_pose(false, windup) * max(ready_level, smoothstep(0.0, 0.35, windup));
        }
        let upper = pose.x;
        let fore = upper + pose.y;
        let hand = fore + pose.z;
        let elbow = rig.shoulder + yz_turn(rig.elbow - rig.shoulder, upper);
        let wrist = elbow + yz_turn(rig.wrist - rig.elbow, fore);
        let grip = wrist + yz_turn(rig.grip - rig.wrist, hand);
        var ang = upper;
        var joint = rig.shoulder;
        var moved = rig.shoulder;
        var shift = vec2<f32>(0.0);
        if abs(part - 9.0) < 0.5 {
            ang = fore;
            joint = rig.elbow;
            moved = elbow;
        } else if abs(part - 10.0) < 0.5 {
            ang = hand;
            joint = rig.wrist;
            moved = wrist;
        } else if abs(part - 8.0) < 0.5 {
            // The weapon keeps its own pitch, not the hand's. Levelling
            // turns it from upright to straight ahead and slides it up
            // through the fist toward the butt.
            ang = -0.5 * PI * pose.w;
            joint = rig.grip;
            moved = grip;
            shift = vec2<f32>(rig.slide * pose.w, 0.0);
        }
        let yz = moved + yz_turn(local.yz - joint + shift, ang);
        local = vec3<f32>(local.x, yz.x, yz.y);
        // The whole arm swings a little on the march and in the cheer.
        let turn = 0.05 * moving * limb * (1.0 - pose.w) + 0.10 * celebrate * sin(wobble);
        local = pitch_about_at(local, rig.shoulder.x, rig.shoulder.y, turn);
        normal = pitch_normal(normal, ang + turn);
    } else if rig.arm > 0.5 && (abs(part - 8.0) < 0.5 || abs(part - rig.arm) < 0.5) {
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
            // Draw back to the hip, then drive the point home along the
            // shaft.
            goal = mix(rig.grip, levelled, level)
                + yz_dir(aim) * reach * (-0.45 * raise * (1.0 - chop) + 0.9 * chop);
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
                goal = rig.grip + yz_dir(rest) * reach * (-0.4 * raise * (1.0 - chop) + 0.7 * chop);
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
        let weapon = abs(part - 8.0) < 0.5;
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

    // An archer's hips and legs turn side-on with his body as his bow
    // comes up. The rest of his body turned in `bow_pose`.
    if bow_rig && bow_w > 0.0 && (part < 0.5 || (part > 1.5 && part < 3.5)) {
        let body = qy(shot_scalars(shot_at(clip, clip_x), 0u).w * bow_w);
        local = qrot(body, local);
        normal = qrot(body, normal);
    }

    // The shoulders turn against the legs and the upper body shifts over
    // the planted foot, easing in up the torso so the hips stay square.
    if part < 1.5 || (part > 3.5 && part < 6.5) || (part > 7.5 && part < 19.5) {
        let w = select(
            1.0,
            smoothstep(0.0, 0.25, vertex.position.y),
            part < 0.5 || abs(part - 19.0) < 0.5,
        );
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
        + 0.30 * lunge
        + 0.25 * chop)
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
    let position = local * select(1.0, vertex.i_pos_scale.w, part > 6.5 && part < 7.5)
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
