// The soldier's pose: his kind's rig, the gait, the arms, the bow and the
// fall, shared by the unit draw (unit_instancing.wgsl) and the pose pass
// (unit_pose_pass.wgsl, render_units_gpu.rs). Three builds of it:
// UNIT_POSE_WRITE is the pass, which poses each drawn soldier once into
// the pose buffer; UNIT_POSE_READ is the pulled draw reading that buffer;
// without either, each corner works out the part of the pose it needs.
#define_import_path flanks::unit_pose

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
// The deadband sits over crowd jitter (about 0.03 m/s) and
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
    // 1, or 0 for an arrow that is not shown: every corner lands on the
    // nock.
    scale: f32,
};

// One vertex of an archer's bow rig, posed by shot `clip` at progress x
// and blended from the carry by w: the arm chains, bow, string and arrow
// as tools/blender/archer/motion.py `transforms` builds them, then the
// body's yaw and the torso's pitch about the waist. The rigid march and
// cheer swings of unjointed arms, draw_ang and bow_ang, fade out as the
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
        return Posed(moved + qrot(pitch, p - neck), pitch, 1.0);
    }
    var at = p;
    var turn = QI;
    var scale = 1.0;
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
                    let show = shown && w > 0.5;
                    at = select(nock, origin + qrot(turn, p - rest), show);
                    scale = select(0.0, 1.0, show);
                }
            }
        }
        // The rigid swings about the shoulder, fading as the bow comes
        // up.
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
    return Posed(waist + qrot(pitch, qrot(body, at) - waist), qmul(pitch, qmul(body, turn)), scale);
}

// One soldier's pose for this frame: everything about his figure that does
// not depend on the corner being placed. With the pose pass on, the compute
// shader (unit_pose_pass.wgsl) writes it once per drawn soldier and every
// corner of his mesh reads it back (`place`). Without it, and on the
// instanced path, each corner works out the parts of it that it needs.
// Slots of four floats:
// World position; the whole figure's vertical offset.
const P_POS: u32 = 0u;
// Facing cos, sin; the lean; the shoulder counter-turn at full weight.
const P_YAW: u32 = 1u;
// That turn's cos, sin; the upper body's sideways shift at full weight; hip
// to sole, 0 on the fallen.
const P_TURN: u32 = 2u;
// An archer's hip turn cos, sin; the stagger's cos, sin.
const P_HIP: u32 = 3u;
// The death topple's cos, sin; its axis (0 none, 1 about x, 2 about z);
// death progress.
const P_DEATH: u32 = 4u;
// Team colour rgb; the hit flash.
const P_TEAM: u32 = 5u;
// Two slots, left then right leg: thigh cos, sin; knee flex; the foot's
// turn below the ankle band.
const P_LEG: u32 = 6u;
// Left shin cos, sin; right shin cos, sin.
const P_SHIN: u32 = 8u;
// Shield sway cos, sin; the wall's turn cos, sin.
const P_SHIELD: u32 = 9u;
// The wall's lift; an arrow's pitch cos, sin; an arrow's scale.
const P_MISC: u32 = 10u;
// Pitch cos, sin of an arm without a rig (part 1) and of a spear arm
// without a rig (part 4).
const P_ARMS: u32 = 11u;
// Pitch cos, sin of a bow arm without a bow rig (part 6); the slash's yaw
// cos, sin.
const P_ARMS2: u32 = 12u;
// Four slots for a weapon arm with a rig: one per part of a jointed arm
// (`put_chain`), or the posed joints of an arm without joints
// (`put_rig_arm`).
const P_RIG: u32 = 13u;
// Two slots per bow rig part (`bow_part`): the rotation, then the offset
// and the scale.
const P_BOW: u32 = 17u;
// Slots per soldier: without a bow rig, and with one (render_units_gpu.rs
// POSE_SLOTS).
const POSE_SLOTS: u32 = 17u;
const POSE_SLOTS_BOW: u32 = 45u;
// Parts of the bow rig.
const BOW_PARTS: u32 = 14u;

// The instance record the pose is built from (render_units.rs InstanceData).
struct Inst {
    pos_scale: vec4<f32>,
    color: vec4<f32>,
    anim: vec4<f32>,
    anim2: vec4<f32>,
};

// One corner of a soldier's mesh, at rest in his local space.
struct Corner {
    position: vec3<f32>,
    normal: vec3<f32>,
    // Body part id (unit_instancing.wgsl `Vertex`) and its pivot height.
    part: f32,
    pivot: f32,
};

struct Placed {
    position: vec3<f32>,
    normal: vec3<f32>,
};

#ifdef UNIT_POSE_READ
@group(3) @binding(9) var<storage, read> poses: array<vec4<f32>>;
#endif
#ifdef UNIT_POSE_WRITE
@group(3) @binding(9) var<storage, read_write> poses: array<vec4<f32>>;
#endif

#ifdef UNIT_POSE_READ
// First slot of the soldier whose corner is being placed.
var<private> pose_base: u32;

fn pose_at(k: u32) -> vec4<f32> {
    return poses[pose_base + k];
}

// The corner's pose from the buffer, `base` its first slot.
fn pose_from(base: u32) {
    pose_base = base;
}
#else
#ifdef UNIT_POSE_WRITE
var<private> pose_base: u32;

fn pose_at(k: u32) -> vec4<f32> {
    return poses[pose_base + k];
}

fn pose_put(k: u32, v: vec4<f32>) {
    poses[pose_base + k] = v;
}
#else
// Worked out per corner: only the slots its part reads are filled, the
// bow rig part and the jointed arm part at their first slot.
var<private> pose_inline: array<vec4<f32>, 19>;

fn pose_at(k: u32) -> vec4<f32> {
    return pose_inline[k];
}

fn pose_put(k: u32, v: vec4<f32>) {
    pose_inline[k] = v;
}
#endif

// The soldier's animation state for this frame, from his record.
struct Pre {
    pos: vec3<f32>,
    // An arrow's uniform scale.
    scale: f32,
    team: vec3<f32>,
    seed: f32,
    yaw: f32,
    moving: f32,
    fx: f32,
    // An arrow's flight pitch.
    pitch: f32,
    bow_rig: bool,
    bow_w: f32,
    clip: u32,
    clip_x: f32,
    style: f32,
    following: bool,
    windup: f32,
    follow: f32,
    lunge: f32,
    raise: f32,
    chop: f32,
    celebrate: f32,
    ready: f32,
    stance: f32,
    confident: f32,
    sprint: f32,
    brace: f32,
    wall: f32,
    leg: f32,
    gait: f32,
    g: Gait,
    limb: f32,
    wobble: f32,
    dip: f32,
    stagger: f32,
    time: f32,
};

var<private> pre: Pre;

// Work out the animation state of the soldier with this instance record
// (render_units.rs InstanceData) and fill the slots every corner reads.
fn pose_begin(pos_scale: vec4<f32>, color: vec4<f32>, anim: vec4<f32>, anim2: vec4<f32>, time: f32) {
    let inst = Inst(pos_scale, color, anim, anim2);
    let yaw = inst.anim.x;
    let speed = inst.anim.y;
    let moving = walk_gate(speed);
    // A bow shot from 16: clip * 2 + progress, clip 0 the raise, 1 the
    // release, 2 the reload (render_units.rs RANGED_BASE). A kind with a
    // bow rig plays it from its tables. Without one the draw arms read it
    // as a stab: the raise is the wind-up and the release the
    // follow-through.
    let bow_rig = rig.bow.params.w > 0.5;
    let zraw = max(inst.anim.z, 0.0);
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
    let bow_w = select(0.0, smoothstep(0.0, 1.0, inst.pos_scale.w), bow_rig);
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
    // over the last ~10%, so a pose never snaps in one frame.
    let cele_t = clamp(zpos - 12.0, 0.0, 1.0);
    let celebrate = select(0.0, 1.0, cheering)
        * smoothstep(0.0, 0.08, cele_t)
        * (1.0 - smoothstep(0.90, 1.0, cele_t));
    // Stance band: 0.25 = enemy in watch range (brace when standing),
    // 0.5 = fighting but wavering (brace, no taunt), 0.65 = fighting
    // confident (taunts), 1 = charging (sprint).
    let band = max(inst.anim2.x, 0.0);
    let ready = smoothstep(0.05, 0.25, band);
    let stance = smoothstep(0.25, 0.5, band);
    let confident = smoothstep(0.55, 0.65, band);
    let sprint = smoothstep(0.7, 1.0, band);
    let fx = inst.anim.w;
    let seed = inst.color.a;
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
    let wall = inst.anim2.y;
    // Gait. The phase counts cycles of two steps, offset per soldier so
    // a block does not move in lockstep. The fallen keep the legs they
    // fell with.
    let leg = select(0.0, rig.leg, fx < 1.99);
    let gait = fract(inst.anim2.z + seed);
    let g = gait_at(speed, leg);
    // The step wave the arms counter, and the idle wobble, which is the
    // one oscillator that is not locomotion.
    let limb = cos(TAU * gait);
    let wobble = time * 9.0 + seed * TAU;
    let dip = gait_dip(fract(2.0 * gait), g);

    pre = Pre(
        inst.pos_scale.xyz,
        inst.pos_scale.w,
        inst.color.rgb,
        seed,
        yaw,
        moving,
        fx,
        inst.anim2.z,
        bow_rig,
        bow_w,
        clip,
        clip_x,
        style,
        following,
        windup,
        follow,
        lunge,
        raise,
        chop,
        celebrate,
        ready,
        stance,
        confident,
        sprint,
        brace,
        wall,
        leg,
        gait,
        g,
        limb,
        wobble,
        dip,
        inst.anim2.w,
        time,
    );
    put_common();
}

// The slots every corner reads: where he stands and faces, the whole
// body's lean, turn, stagger and fall, his colours.
fn put_common() {
    let g = pre.g;
    // An archer's hips and legs turn side-on with his body as his bow
    // comes up. The rest of his body turns in `bow_pose`.
    var hip = vec2<f32>(1.0, 0.0);
    if pre.bow_rig && pre.bow_w > 0.0 {
        let a = shot_scalars(shot_at(pre.clip, pre.clip_x), 0u).w * pre.bow_w;
        hip = vec2<f32>(cos(a), sin(a));
    }
    // The shoulders turn against the legs and the upper body shifts over
    // the planted foot, both at full weight here: the torso eases them in
    // up its height (`place`).
    let turn = 0.10 * g.run * g.stride * pre.limb;
    let shift = 0.02 * pre.leg * g.run * g.stride * cos(TAU * (pre.gait - 0.5 * g.duty));
    // The body rides the hip drop from the gait. A runner leans as far
    // forward as a charger.
    let lean = 0.10 * pre.moving
        + 0.24 * max(g.run * pre.moving, pre.sprint * (0.25 + 0.75 * pre.moving))
        + 0.30 * pre.lunge
        + 0.25 * pre.chop;
    // Stagger (anim2.w = remaining stun, 1 at the blow -> 0 recovered):
    // rocked back off a charging blow or a spear point. The whole body
    // pitches backward from the feet with a short reel, hardest at the
    // hit and easing out as the man finds his balance; the knee-buckle
    // sink rides the vertical offset.
    var stagger = vec2<f32>(1.0, 0.0);
    if pre.stagger > 0.001 {
        let reel = 1.0 + 0.18 * sin(pre.time * 22.0 + pre.seed * TAU);
        let sang = -0.42 * pre.stagger * pre.stagger * reel;
        stagger = vec2<f32>(cos(sang), sin(sang));
    }
    // Death: topple around the feet and sink slightly. Fall direction
    // varies per unit (seed): forward, backward, or to either side —
    // corpses keep their seed, so the pose persists on the ground.
    let death = clamp(pre.fx - 1.0, 0.0, 1.0);
    var topple = vec3<f32>(1.0, 0.0, 0.0);
    if death > 0.0 {
        let dvar = fract(pre.seed * 13.73);
        if dvar < 0.62 {
            // Forward (most common) or backward topple about X.
            let dir = select(1.0, -0.9, dvar > 0.45);
            let ang = dir * death * 1.45;
            topple = vec3<f32>(cos(ang), sin(ang), 1.0);
        } else {
            // Sideways collapse about Z (either side).
            let dir = select(1.0, -1.0, dvar < 0.81);
            let ang = dir * death * 1.4;
            topple = vec3<f32>(cos(ang), sin(ang), 2.0);
        }
    }
    // Cheer hop: celebrating units bounce. Brace, the braced walk and the
    // wall carry a slight crouch.
    let hop = 0.06 * pre.celebrate * max(sin(pre.wobble), 0.0);
    let bob = -pre.dip;
    let offset = bob - 0.15 * death - 0.07 * pre.brace - 0.03 * pre.ready * pre.moving
        - 0.05 * pre.wall - 0.10 * pre.stagger + hop;
    pose_put(P_POS, vec4<f32>(pre.pos, offset));
    pose_put(P_YAW, vec4<f32>(cos(pre.yaw), sin(pre.yaw), lean, turn));
    pose_put(P_TURN, vec4<f32>(cos(turn), sin(turn), shift, pre.leg));
    pose_put(P_HIP, vec4<f32>(hip, stagger));
    pose_put(P_DEATH, vec4<f32>(topple, death));
    pose_put(
        P_TEAM,
        vec4<f32>(pre.team, clamp(pre.fx, 0.0, 1.0) * step(pre.fx, 1.0)),
    );
}

// The bow rig part in slot `k`: parts 1 and 6, then 8 to 19.
fn bow_part(k: u32) -> f32 {
    if k == 0u {
        return 1.0;
    }
    if k == 1u {
        return 6.0;
    }
    return f32(k + 6u);
}

// An archer's bow rig. A shot plays its clip, and between shots the
// drawn ready pose (the raise at its start) eases in and out with the
// bow. While the bow is down, the drawing arm swings rigidly about its
// shoulder on the march, in the cheer and in a melee blow, and the bow
// arm on the march and in the cheer. Every part is rigid: its rotation,
// and where its local origin lands.
fn put_bow(k: u32, slot: u32) {
    let run = pre.g.run;
    let sway = (0.18 - 0.06 * run) * pre.moving * pre.limb * (1.0 - pre.raise);
    let carry = mix(-0.55, 0.25, max(pre.stance, pre.ready * 0.75)) * pre.moving * (1.0 - pre.raise);
    var draw_ang = select(1.9 * pre.raise - 2.5 * pre.chop, 0.55 * pre.raise - 0.45 * pre.chop, pre.style < 0.5);
    draw_ang = select(draw_ang, pre.celebrate * (1.75 + 0.35 * sin(pre.wobble)), pre.celebrate > 0.001);
    draw_ang += sway + carry + 0.6 * pre.brace;
    let bow_ang = -0.06 * pre.moving * pre.limb + pre.celebrate * (1.5 + 0.3 * sin(pre.wobble));
    let posed = bow_pose(bow_part(k), vec3<f32>(0.0), pre.clip, pre.clip_x, pre.bow_w, draw_ang, bow_ang);
    pose_put(P_BOW + 2u * slot, posed.turn);
    pose_put(P_BOW + 2u * slot + 1u, vec4<f32>(posed.pos, posed.scale));
}

// A jointed weapon arm: upper arm, forearm and hand turn at the shoulder,
// elbow and wrist, each carrying the next, and the weapon turns and slides
// through the hand. The rest pose is the carry. Readiness levels the
// weapon into the guard, and an attack plays the kind's tables from the
// guard and back to it. Part `k` (`chain_slot`) as one turn in the pitch
// plane and where its local origin lands.
fn put_chain(k: u32, slot: u32) {
    let ready_level = max(max(pre.stance, pre.ready * 0.75), max(pre.sprint, pre.wall)) * (1.0 - pre.celebrate);
    // Carry to the guard is a straight blend of the joint turns, and
    // so is carry to any attack pose. A wind-up that starts short of
    // the guard reaches it in its first third, and the follow-through
    // ends in the guard and settles to readiness as it runs out.
    var pose = rig.windup[0] * ready_level;
    if pre.following {
        pose = attack_pose(true, pre.follow) * max(ready_level, 1.0 - smoothstep(0.92, 1.0, pre.follow));
    } else if pre.windup > 0.0 {
        pose = attack_pose(false, pre.windup) * max(ready_level, smoothstep(0.0, 0.35, pre.windup));
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
    if k == 1u {
        ang = fore;
        joint = rig.elbow;
        moved = elbow;
    } else if k == 2u {
        ang = hand;
        joint = rig.wrist;
        moved = wrist;
    } else if k == 3u {
        // The weapon keeps its own pitch, not the hand's. Levelling
        // turns it from upright to straight ahead and slides it up
        // through the fist toward the butt.
        ang = -0.5 * PI * pose.w;
        joint = rig.grip;
        moved = grip;
        shift = vec2<f32>(rig.slide * pose.w, 0.0);
    }
    // The whole arm swings a little on the march and in the cheer.
    let turn = 0.05 * pre.moving * pre.limb * (1.0 - pose.w) + 0.10 * pre.celebrate * sin(pre.wobble);
    let yz = moved + yz_turn(shift - joint, ang);
    let origin = pitch_about_at(vec3<f32>(0.0, yz.x, yz.y), rig.shoulder.x, rig.shoulder.y, turn);
    let a = ang + turn;
    pose_put(P_RIG + slot, vec4<f32>(cos(a), sin(a), origin.y, origin.z));
}

// The weapon arm and its weapon: shoulder, elbow and wrist. The hand
// stays on the arm and the arm on the shoulder, and the weapon turns and
// slides in the hand. The elbow blends over a band, so `place` finishes
// it per corner.
fn put_rig_arm() {
    let reach = length(rig.elbow - rig.shoulder) + length(rig.grip - rig.elbow);
    let rest = yz_angle(rig.tip);
    var goal = rig.grip;
    var aim = rest;
    // Turn of the whole arm about the shoulder, on top of the pose.
    var turn = 0.0;
    var slide = 0.0;
    // The slash's sweep about the body axis, cos and sin.
    var slash = vec2<f32>(1.0, 0.0);
    let moving = pre.moving;
    let limb = pre.limb;
    let raise = pre.raise;
    let chop = pre.chop;
    let celebrate = pre.celebrate;
    let wobble = pre.wobble;
    if rig.hold > 1.5 && rig.hold < 2.5 {
        // Spear. It rides upright at rest and on the march. A watch
        // range advance, a fight, a charge or a wall levels it: the
        // upper arm hangs, the forearm points ahead with the shaft
        // along it, and the grip moves back toward the butt for reach.
        let level = max(max(pre.stance, pre.ready * 0.75), max(max(pre.sprint, raise), pre.wall))
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
        let sway = (0.18 - 0.06 * pre.g.run) * moving * limb * (1.0 - raise);
        let carry = mix(-0.55, 0.25, max(pre.stance, pre.ready * 0.75)) * moving * (1.0 - raise);
        let tc = fract(pre.time / 7.3 + pre.seed * 5.13);
        let tpulse = smoothstep(0.02, 0.12, tc) * (1.0 - smoothstep(0.24, 0.34, tc));
        let taunt = TAUNT_ON * tpulse * pre.confident * (1.0 - moving)
            * (1.0 - smoothstep(0.0, 0.05, pre.lunge));
        if celebrate > 0.001 {
            // Victory cheer: blade pumped skyward.
            turn = celebrate * (1.75 + 0.35 * sin(wobble));
        } else if pre.style < 0.5 {
            // Stab: the hand pulls back, then thrusts along the blade,
            // the wrist keeping the point near level.
            goal = rig.grip + yz_dir(rest) * reach * (-0.4 * raise * (1.0 - chop) + 0.7 * chop);
            aim = rest + 0.15 * raise - 0.1 * chop;
        } else if pre.style < 1.5 {
            // The classic swing: raise up and back, fast chop.
            turn = 1.9 * raise - 2.5 * chop;
        } else {
            // Slash: a sweep around the body axis, wind back, cut across.
            let yawoff = -1.1 * raise + 2.3 * chop;
            slash = vec2<f32>(cos(yawoff), sin(yawoff));
            turn = 0.5 * raise - 0.3 * chop;
        }
        turn += sway + carry + taunt * (1.7 + 0.22 * sin(pre.time * 16.0)) + 0.6 * pre.brace;
    }
    let pose = arm_pose(goal, aim);
    let shoulder = pose.x + turn;
    let elbow = rig.shoulder + yz_turn(rig.elbow - rig.shoulder, shoulder);
    let grip = elbow + yz_turn(rig.grip - rig.elbow, shoulder + pose.y);
    let way = yz_dir(aim + turn) * slide;
    pose_put(P_RIG, vec4<f32>(cos(shoulder), sin(shoulder), elbow));
    pose_put(P_RIG + 1u, vec4<f32>(pose.y, cos(pose.z), sin(pose.z), shoulder));
    pose_put(P_RIG + 2u, vec4<f32>(grip, way));
    pose_put(P_RIG + 3u, vec4<f32>(slash, pose.z, 0.0));
}

// Arms without a rig, the shield arm and an arrow: each one turn.
fn put_simple() {
    let moving = pre.moving;
    let limb = pre.limb;
    let raise = pre.raise;
    let chop = pre.chop;
    let celebrate = pre.celebrate;
    let wobble = pre.wobble;
    let wall = pre.wall;

    // Bow arm: stave carried vertical at the side. The draw tilts
    // arm and bow up toward the loft angle (the whole part pitches,
    // so the stave cants back over the shoulder — an archer aiming
    // high); the loose settles it, recover eases back to carry.
    // The draw hand is plain PART_ARM running the stab style: its
    // pull-back-then-snap IS the string draw and release.
    let bow_arm = 0.75 * raise - 0.20 * chop
        - 0.06 * moving * limb * (1.0 - raise)
        + celebrate * (1.5 + 0.3 * sin(wobble));

    // Shield arm: carried at the side; the wall signal swings it
    // around the body to FACE THE FRONT and lifts it into a guard —
    // a shieldwall is a wall of team color from the enemy's side.
    // (Spear bucket: same fronting reads as the spearwall's off-hand
    // cover behind the leveled spears.) On the move it swings against
    // the weapon arm, except in a wall. The shield is on +X, the left
    // hand: turning it forward is a negative turn about Y.
    let sway = -0.06 * moving * limb * (1.0 - wall);
    var front = vec2<f32>(1.0, 0.0);
    var lift = 0.0;
    if wall > 0.001 {
        let ang = -1.05 * wall;
        front = vec2<f32>(cos(ang), sin(ang));
        lift = 0.10 * wall;
    }

    // Spear arm without a rig: arm and shaft turn as one about the
    // shoulder. Battle stance, a watch-range advance, a charge or a
    // wall levels the point, and a stab tips it forward.
    let level = max(max(pre.stance, pre.ready * 0.75), max(max(pre.sprint, raise), wall));
    let spear = -1.42 * level * (1.0 - celebrate)
        - 0.15 * chop
        + 0.05 * moving * limb * (1.0 - level)
        + 0.10 * celebrate * sin(wobble);

    // Sword arm without a rig, and its weapon: they turn as one about
    // the shoulder. Three per-unit attack styles (stable seed pick), all
    // timed so the blow lands exactly when the damage event fires
    // (lunge hits 1.0 at the strike tick): overhead chop, forward
    // stab, horizontal slash.
    // The arm counters the legs. A runner holds the blade steadier.
    let arm_sway = (0.18 - 0.06 * pre.g.run) * moving * limb * (1.0 - raise);
    // Ordinary moves carry the blade lowered at the side; battle
    // stance levels it at the enemy (slightly above horizontal),
    // and even a watch-range advance (`ready`) brings it most of
    // the way up — the braced walk.
    let carry = mix(-0.55, 0.25, max(pre.stance, pre.ready * 0.75)) * moving * (1.0 - raise);
    // Taunt: STANDING units of a CONFIDENT fighting regiment pump
    // the blade skyward for ~1.5 s every ~7 s, staggered per unit —
    // the rear ranks jeer while the front works. Wavering regiments
    // (morale low) stop jeering and just hold the brace.
    let tc = fract(pre.time / 7.3 + pre.seed * 5.13);
    let tpulse = smoothstep(0.02, 0.12, tc) * (1.0 - smoothstep(0.24, 0.34, tc));
    let taunt =
        TAUNT_ON * tpulse * pre.confident * (1.0 - moving) * (1.0 - smoothstep(0.0, 0.05, pre.lunge));
    let ang_taunt = taunt * (1.7 + 0.22 * sin(pre.time * 16.0));
    // Style picked per SWING by the sim (swing bits): 0 = stab,
    // 1 = classic swing, 2 = slash (benched).
    var arm = 0.0;
    var slash = vec2<f32>(1.0, 0.0);
    if celebrate > 0.001 {
        // Victory cheer: blade pumped skyward, bouncing with the hop.
        arm = celebrate * (1.75 + 0.35 * sin(wobble));
    } else if pre.style < 0.5 {
        // Stab: draw the arm up and back, then drop the point forward.
        arm = 0.55 * raise - 0.45 * chop;
    } else if pre.style < 1.5 {
        // The classic swing: raise up/back, fast chop.
        arm = 1.9 * raise - 2.5 * chop;
    } else {
        // Slash: horizontal sweep around the body axis — wind back,
        // cut across.
        let yawoff = -1.1 * raise + 2.3 * chop;
        slash = vec2<f32>(cos(yawoff), sin(yawoff));
        arm = 0.5 * raise - 0.3 * chop;
    }
    // Braced units hold a proper ready guard; the non-bracers of
    // the line keep the plain forward point (ang 0 standing).
    arm += arm_sway + carry + ang_taunt + 0.6 * pre.brace;

    pose_put(P_SHIELD, vec4<f32>(cos(sway), sin(sway), front));
    // Arrow projectile (arrows.rs buckets): rigid mesh, flight pitch
    // rides anim2.z (a dead channel for these instances — march is
    // always 0 here); yaw is the shared rotation.
    pose_put(P_MISC, vec4<f32>(lift, cos(pre.pitch), sin(pre.pitch), pre.scale));
    pose_put(P_ARMS, vec4<f32>(cos(arm), sin(arm), cos(spear), sin(spear)));
    pose_put(P_ARMS2, vec4<f32>(cos(bow_arm), sin(bow_arm), slash));
}

// Hip, knee and ankle of one leg, posed by `leg_pose`, into leg slot
// `slot`. Knee and ankle bend over a band around each joint, so the mesh
// bends instead of tearing: `place` finishes them per corner. Returns the
// shin's cos and sin.
fn put_leg(right: bool, slot: u32) -> vec2<f32> {
    let pose = leg_pose(fract(pre.gait + select(0.0, 0.5, right)), pre.g);
    // A braced or walled stance splits the feet, one forward one back.
    let thigh = pose.x
        + (0.32 * pre.brace + 0.22 * pre.wall * (1.0 - pre.moving)) * select(1.0, -1.0, right);
    let flex = pose.y;
    let shin = thigh - flex;
    pose_put(P_LEG + slot, vec4<f32>(cos(thigh), sin(thigh), flex, pose.z - shin));
    return vec2<f32>(cos(shin), sin(shin));
}
#endif

// The slot of a jointed arm's part: the upper arm, forearm, hand, weapon.
fn chain_slot(part: f32) -> u32 {
    if abs(part - 9.0) < 0.5 {
        return 1u;
    }
    if abs(part - 10.0) < 0.5 {
        return 2u;
    }
    if abs(part - 8.0) < 0.5 {
        return 3u;
    }
    return 0u;
}

#ifdef UNIT_POSE_WRITE
// The whole pose of the soldier with this instance record, into the slots
// from `base` on.
fn pose_write(
    base: u32,
    pos_scale: vec4<f32>,
    color: vec4<f32>,
    anim: vec4<f32>,
    anim2: vec4<f32>,
    time: f32,
) {
    pose_base = base;
    pose_begin(pos_scale, color, anim, anim2, time);
    put_simple();
    // The fallen keep the legs they fell with (`place`).
    if pre.leg > 0.0 {
        let left = put_leg(false, 0u);
        let right = put_leg(true, 1u);
        pose_put(P_SHIN, vec4<f32>(left, right));
    }
    if rig.bow.params.w > 0.5 {
        for (var k = 0u; k < BOW_PARTS; k++) {
            put_bow(k, k);
        }
    }
    if rig.chain > 0.5 {
        for (var k = 0u; k < 4u; k++) {
            put_chain(k, k);
        }
    } else if rig.arm > 0.5 {
        put_rig_arm();
    }
}
#else

// A pitch (about +X) by an angle given as its cos and sin, around a pivot
// at height py, as `pitch_about`.
fn pitch_about_cs(p: vec3<f32>, py: f32, c: f32, s: f32) -> vec3<f32> {
    let y = p.y - py;
    return vec3<f32>(p.x, py + y * c + p.z * s, -y * s + p.z * c);
}

fn pitch_about_at_cs(p: vec3<f32>, cy: f32, cz: f32, c: f32, s: f32) -> vec3<f32> {
    let y = p.y - cy;
    let z = p.z - cz;
    return vec3<f32>(p.x, cy + y * c + z * s, cz - y * s + z * c);
}

fn pitch_normal_cs(n: vec3<f32>, c: f32, s: f32) -> vec3<f32> {
    return vec3<f32>(n.x, n.y * c + n.z * s, -n.y * s + n.z * c);
}

// Where a corner of the mesh lands in the world, posed: its rest position
// and normal in the soldier's local space, its body part and the part's
// pivot height.
fn place(rest: vec3<f32>, rest_normal: vec3<f32>, part_id: f32, pivot_height: f32) -> Placed {
    let corner = Corner(rest, rest_normal, part_id, pivot_height);
    let part = corner.part;
    let pivot = corner.pivot;
    var local = corner.position;
    var normal = corner.normal;
    let bow_rig = rig.bow.params.w > 0.5;

    // --- Part animation (rotations around the part pivot) ---
    if bow_rig && (abs(part - 1.0) < 0.5 || abs(part - 6.0) < 0.5 || (part > 7.5 && part < 19.5)) {
#ifdef UNIT_POSE_READ
        var k = u32(part + 0.5) - 6u;
        if part < 1.5 {
            k = 0u;
        } else if part < 6.5 {
            k = 1u;
        }
#else
        var k = 0u;
        if part < 1.5 {
            put_bow(0u, 0u);
        } else if part < 6.5 {
            put_bow(1u, 0u);
        } else {
            put_bow(u32(part + 0.5) - 6u, 0u);
        }
#endif
        let turn = pose_at(P_BOW + 2u * k);
        let at = pose_at(P_BOW + 2u * k + 1u);
        local = at.xyz + qrot(turn, local) * at.w;
        normal = qrot(turn, normal);
    } else if rig.chain > 0.5 && (abs(part - rig.arm) < 0.5 || (part > 7.5 && part < 10.5)) {
#ifdef UNIT_POSE_READ
        let r = pose_at(P_RIG + chain_slot(part));
#else
        put_chain(chain_slot(part), 0u);
        let r = pose_at(P_RIG);
#endif
        let yz = vec2<f32>(local.y * r.x + local.z * r.y, -local.y * r.y + local.z * r.x) + r.zw;
        local = vec3<f32>(local.x, yz.x, yz.y);
        normal = pitch_normal_cs(normal, r.x, r.y);
    } else if rig.arm > 0.5 && (abs(part - 8.0) < 0.5 || abs(part - rig.arm) < 0.5) {
#ifndef UNIT_POSE_READ
        put_rig_arm();
#endif
        let shoulder = pose_at(P_RIG);
        let joints = pose_at(P_RIG + 1u);
        let grip = pose_at(P_RIG + 2u);
        let slash = pose_at(P_RIG + 3u);
        local = rot_y(local, slash.x, slash.y);
        normal = rot_y(normal, slash.x, slash.y);
        // Past the elbow the forearm turns, blended over a band so the
        // sleeve bends. The weapon turns with the hand.
        let upper = rig.elbow - rig.shoulder;
        let past = dot(corner.position.yz - rig.elbow, upper) / dot(upper, upper);
        let weapon = abs(part - 8.0) < 0.5;
        let fore = select(smoothstep(-0.15, 0.15, past), 1.0, weapon);
        local = pitch_about_at_cs(local, rig.shoulder.x, rig.shoulder.y, shoulder.x, shoulder.y);
        local = pitch_about_at(local, shoulder.z, shoulder.w, joints.x * fore);
        var total = joints.w + joints.x * fore;
        if weapon {
            local = pitch_about_at_cs(local, grip.x, grip.y, joints.y, joints.z);
            local.y += grip.z;
            local.z += grip.w;
            total += slash.z;
        }
        normal = pitch_normal(normal, total);
    } else if part > 6.5 && part < 7.5 {
#ifndef UNIT_POSE_READ
        put_simple();
#endif
        let m = pose_at(P_MISC);
        local = pitch_about_cs(local, 0.0, m.y, m.z);
        normal = pitch_normal_cs(normal, m.y, m.z);
    } else if part > 5.5 && part < 6.5 {
#ifndef UNIT_POSE_READ
        put_simple();
#endif
        let a = pose_at(P_ARMS2);
        local = pitch_about_cs(local, pivot, a.x, a.y);
        normal = pitch_normal_cs(normal, a.x, a.y);
    } else if part > 4.5 && part < 5.5 {
#ifndef UNIT_POSE_READ
        put_simple();
#endif
        let s = pose_at(P_SHIELD);
        local = pitch_about_cs(local, pivot, s.x, s.y);
        normal = pitch_normal_cs(normal, s.x, s.y);
        local = rot_y(local, s.z, s.w);
        normal = rot_y(normal, s.z, s.w);
        local.y += pose_at(P_MISC).x;
    } else if part > 3.5 && part < 4.5 {
#ifndef UNIT_POSE_READ
        put_simple();
#endif
        let a = pose_at(P_ARMS);
        local = pitch_about_cs(local, pivot, a.z, a.w);
        normal = pitch_normal_cs(normal, a.z, a.w);
    } else if part > 1.5 && part < 3.5 {
        // Corpses carry no leg length and keep the legs they fell with.
        let leg = pose_at(P_TURN).w;
        if leg > 0.0 {
            let right = part > 2.5;
#ifdef UNIT_POSE_READ
            let l = pose_at(P_LEG + select(0u, 1u, right));
            let shins = pose_at(P_SHIN);
            let shin = select(shins.xy, shins.zw, right);
#else
            let shin = put_leg(right, 0u);
            let l = pose_at(P_LEG);
#endif
            local = pitch_about_cs(local, pivot, l.x, l.y);
            normal = pitch_normal_cs(normal, l.x, l.y);
            let down = clamp((pivot - corner.position.y) / leg, 0.0, 1.0);
            let knee = KNEE * leg;
            let kz = knee * l.y;
            let ky = pivot - knee * l.x;
            let bend = -l.z * smoothstep(KNEE - 0.1, KNEE + 0.1, down);
            local = pitch_about_at(local, ky, kz, bend);
            normal = pitch_normal(normal, bend);
            // The foot turns about the ankle to the pitch the pose asks for.
            let shank = (ANKLE - KNEE) * leg;
            let foot = l.w * smoothstep(ANKLE - 0.06, ANKLE + 0.02, down);
            local = pitch_about_at(local, ky - shank * shin.x, kz + shank * shin.y, foot);
            normal = pitch_normal(normal, foot);
        }
    } else if part > 0.5 {
#ifndef UNIT_POSE_READ
        put_simple();
#endif
        let a = pose_at(P_ARMS);
        let slash = pose_at(P_ARMS2);
        local = rot_y(local, slash.z, slash.w);
        normal = rot_y(normal, slash.z, slash.w);
        local = pitch_about_cs(local, pivot, a.x, a.y);
        normal = pitch_normal_cs(normal, a.x, a.y);
    }

    // An archer's hips and legs turn side-on with his body as his bow
    // comes up. The rest of his body turned with the bow rig.
    let hip = pose_at(P_HIP);
    if bow_rig && (part < 0.5 || (part > 1.5 && part < 3.5)) {
        local = rot_y(local, hip.x, hip.y);
        normal = rot_y(normal, hip.x, hip.y);
    }

    // The shoulders turn against the legs and the upper body shifts over
    // the planted foot, easing in up the torso so the hips stay square.
    let yaw = pose_at(P_YAW);
    if part < 1.5 || (part > 3.5 && part < 6.5) || (part > 7.5 && part < 19.5) {
        let turn = pose_at(P_TURN);
        let torso = part < 0.5 || abs(part - 19.0) < 0.5;
        let w = select(1.0, smoothstep(0.0, 0.25, corner.position.y), torso);
        var c = turn.x;
        var s = turn.y;
        if torso {
            let ang = yaw.w * w;
            c = cos(ang);
            s = sin(ang);
        }
        local = rot_y(local, c, s);
        normal = rot_y(normal, c, s);
        local.x += turn.z * w;
    }

    let lean = yaw.z * clamp(local.y + 0.5, 0.0, 1.5);
    local.z += lean * 0.3;

    // Stagger: the whole body pitches backward from the feet. The sine is
    // zero only while he is not staggered.
    if hip.w != 0.0 {
        let sca = hip.z;
        let ssa = hip.w;
        let sfy = local.y + 0.5;
        local = vec3<f32>(local.x, sfy * sca - local.z * ssa - 0.5, sfy * ssa + local.z * sca);
        normal =
            vec3<f32>(normal.x, normal.y * sca - normal.z * ssa, normal.y * ssa + normal.z * sca);
    }

    // Death: the topple about the feet.
    let topple = pose_at(P_DEATH);
    if topple.z > 0.5 {
        let ca = topple.x;
        let sa = topple.y;
        let fy = local.y + 0.5;
        if topple.z < 1.5 {
            local = vec3<f32>(local.x, fy * ca - local.z * sa - 0.5, fy * sa + local.z * ca);
            normal =
                vec3<f32>(normal.x, normal.y * ca - normal.z * sa, normal.y * sa + normal.z * ca);
        } else {
            local = vec3<f32>(local.x * ca + fy * sa, fy * ca - local.x * sa - 0.5, local.z);
            normal =
                vec3<f32>(normal.x * ca + normal.y * sa, normal.y * ca - normal.x * sa, normal.z);
        }
    }

    // Face yaw (0 = +Z): rotate position and normal.
    local = rot_y(local, yaw.x, yaw.y);
    normal = rot_y(normal, yaw.x, yaw.y);

    let at = pose_at(P_POS);
    var scale = 1.0;
    if part > 6.5 && part < 7.5 {
        scale = pose_at(P_MISC).w;
    }
    return Placed(local * scale + at.xyz + vec3<f32>(0.0, at.w, 0.0), normal);
}
#endif
