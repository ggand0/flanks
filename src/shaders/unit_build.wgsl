// Builds the unit render data on the GPU: the port of render_units.rs
// `sync_instance_data`, one thread per soldier. Reads the per-tick
// soldier snapshot, the per-frame regiment records and the camera,
// writes the 64 byte instance record the vertex shader consumes, and
// appends the soldier to his kind-by-level index list. A near soldier
// whose shadow can fall in view also goes on the caster list for his kind
// of each sun shadow cascade that serves the depths it falls at, whether
// or not the camera sees him (render_units_shadow.rs). A second entry
// turns the list counts into indirect draw arguments. With the pose pass on,
// each drawn soldier also takes a pose slot in his kind's region, and his
// list entries hold that slot instead of his record (unit_pose_pass.wgsl).
//
// Step order and constants follow the CPU pass line for line (the
// contract table in docs/plans/gpu-render-data-item2.md). Culled soldiers
// still update their smoothers, exactly as the CPU does.

// Per-tick snapshot of one soldier (render_units_gpu.rs GpuSoldier, 56
// bytes, all scalars so the stride stays 56).
struct Soldier {
    px: f32,
    py: f32,
    pz: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    yaw: f32,
    yaw_prev: f32,
    // kind | swing << 8 | swing_t << 16 | flash << 24
    a: u32,
    // group | death_t << 24
    b: u32,
    cr: f32,
    cg: f32,
    cb: f32,
    seed: f32,
};

// The instance record (render_units.rs InstanceData).
struct Record {
    pos_scale: vec4<f32>,
    color: vec4<f32>,
    anim: vec4<f32>,
    anim2: vec4<f32>,
};

// Per-soldier smoothing state, indexed by soldier index. The death sweep
// swap-removes soldiers, and the one it moves into a freed slot takes
// over that slot's state.
struct Smooth {
    // Smoothed ground speed, m/s.
    walk: f32,
    band: f32,
    wall: f32,
    // Gait phase in cycles (gait.rs).
    gait: f32,
    lod: u32,
    // Seconds of follow-through left after a blow.
    follow: f32,
    // Wind-up progress shown last frame, running back to 0 once the
    // wind-up is cut short.
    atk: f32,
    // The swing byte of the current or last attack, and the SM_ bits.
    swing: u32,
    // Seconds left of the release, reload and hold after his last shot.
    shot: f32,
    // How far his bow is up toward the drawn ready pose, 0 to 1.
    bow: f32,
    // Smoothed ground velocity, m/s (x, z): `walk` is its length.
    drift_x: f32,
    drift_z: f32,
    // A skinned soldier's steps, which the CPU path does not play
    // (`steps`): where he is in the step clip, in plays, its slot from
    // the left step, and how far he is in it rather than in his stance or
    // his gait, 0 to 1.
    step: f32,
    step_clip: u32,
    step_w: f32,
};

// Per-regiment pose signals for this frame (render_units_gpu.rs
// RegimentRecord).
struct Regiment {
    stance: f32,
    // Victory cheer progress 0..1, negative when not celebrating.
    celebrate: f32,
    walled: f32,
    flags: u32,
};

struct Params {
    planes: array<vec4<f32>, 6>,
    cam_pos: vec3<f32>,
    alpha: f32,
    k_walk: f32,
    k_band: f32,
    k_wall: f32,
    inv_dt: f32,
    n: u32,
    n_regs: u32,
    cull: u32,
    corpse_base: u32,
    corpse_len: vec4<u32>,
    corpse_cap: u32,
    frame: u32,
    dt: f32,
    // render_units.rs BAND_FIGHTING.
    fighting: f32,
    // [kind * 3 + set]: set 0 fine, 1 coarse, 2 plain. xyz = squared
    // switch distances for L0/L1, L1/L2, L2/L3 (inf = switch disabled).
    bands: array<vec4<f32>, 12>,
    windup: vec4<f32>,
    // Per kind: its cut's wind-up ticks (0 without a cut), and the
    // seconds the cut takes to come back.
    cut_windup: vec4<f32>,
    cut_return: vec4<f32>,
    // Per kind, the step clips left, right, forward and backward: the
    // ground one play covers in metres (0 without the clip), and its
    // length in seconds.
    step_travel: array<vec4<f32>, 4>,
    step_len: array<vec4<f32>, 4>,
    // soldier.rs STEP_FACE_SPEED, render_units.rs STEP_SETTLE_S.
    steps: vec4<f32>,
    // draw_ticks, death_ticks, hit_stagger_ticks, celebrate_base
    consts: vec4<f32>,
    // FOLLOW_S, FOLLOW_BASE, FOLLOW_SPAN, REWIND_S (render_units.rs).
    attack: vec4<f32>,
    // RELEASE_S, RELOAD_S, HOLD_S, BOW_WALK_MS (render_units.rs).
    shot: vec4<f32>,
    // BOW_RISE_S, BOW_FALL_S, CANCEL_TICKS, RANGED_BASE.
    bow: vec4<f32>,
    // The block's raise, hold and whole length in ticks (sim/damage.rs
    // BLOCK_*) and TWITCH_TICKS; the parry's, and in w which kinds have
    // the clips: bit 2k the block of kind k, bit 2k + 1 his parry.
    guard: vec4<f32>,
    parry: vec4<f32>,
    // x = first index slot of the bucket, y = mesh corners per soldier,
    // w = soldiers per instance of its indexed draw (0: expanded).
    buckets: array<vec4<u32>, 16>,
    // The camera's forward axis, for view depth.
    cam_fwd: vec4<f32>,
    // Where a soldier's shadow can fall: xz = the centre of the ground his
    // cull sphere shades, from his position, w = its radius.
    shadow_reach: vec4<f32>,
    // The view depths each cascade is sampled at, blend band included.
    cascade_near: vec4<f32>,
    cascade_far: vec4<f32>,
    // Per cascade, one entry per kind: the first slot of the cascade's
    // caster list, and the corners per soldier of the level it casts with.
    shadow_lists: array<vec4<u32>, 4>,
    shadow_corners: array<vec4<u32>, 4>,
    // Cascades to fill, 0 with shadows off.
    n_cascades: u32,
    // Detail levels that cast: a soldier farther than these casts nothing.
    cast_lods: u32,
    // The first slot of the ring list in `index_list`.
    ring_base: u32,
    // Per kind: the first entry of its region in `pose_src`.
    pose_src_base: vec4<u32>,
    // Per kind: the slots in its region of the pose sources and of its
    // pose buffer.
    pose_cap: vec4<u32>,
    // 1 with the pose pass on.
    pose_pass: u32,
    // Per cascade and kind: soldiers per instance of the level the caster
    // list draws with (0: expanded).
    shadow_groups: array<vec4<u32>, 4>,
    // Depth bins per unit of log2 of the squared camera distance (0: every
    // soldier in bin 0, the camera's lists in the order the build found
    // them).
    order: f32,
    // The occlusion test (render_units_occlusion.rs): x the mode (0 off,
    // 1 cull, 2 count), y the pyramid levels it reads, 0 to skip it this
    // frame, z 1 on a camera cut, w the FL_OCCLUDE_CHECK level (1 marks
    // the hidden in count mode, 2 every soldier the test reads).
    occlude: vec4<u32>,
    // The camera's clip_from_world the pyramid was drawn with, last
    // frame's, and its viewport in pixels: x, y, width, height.
    occlude_clip: mat4x4<f32>,
    occlude_view: vec4<f32>,
    // Per kind, the box the test projects: forward, back, sideways, up,
    // standing and then dying or fallen, and how far down each reaches.
    bound_live: array<vec4<f32>, 4>,
    bound_fall: array<vec4<f32>, 4>,
    bound_down_live: vec4<f32>,
    bound_down_fall: vec4<f32>,
};

struct DrawArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> soldiers: array<Soldier>;
@group(0) @binding(2) var<storage, read> regiments: array<Regiment>;
@group(0) @binding(3) var<storage, read_write> smoothing: array<Smooth>;
@group(0) @binding(4) var<storage, read_write> records: array<Record>;
@group(0) @binding(5) var<storage, read_write> index_list: array<u32>;
// 0..16 soldiers per bucket (living and fallen), 16..32 the fallen alone,
// 32..48 the casters per cascade and kind (cascade * 4 + kind), 48 the
// selection rings, 49..53 the pose slots taken per kind, then per bucket
// and depth bin (`BIN_COUNTER`) the soldiers in it, which `finalize` turns
// into the bin's first slot in the bucket's list, then the far levels'
// pose slots per kind, then per bucket the soldiers the occlusion test hid
// (`HIDDEN_COUNTER`).
@group(0) @binding(6) var<storage, read_write> counts: array<atomic<u32>, 2121>;
// 0..16 the camera's buckets, 16..32 the casters as in `counts`, 32 the
// selection rings, 33..37 the pose pass's dispatch per kind (x, y, z).
@group(0) @binding(7) var<storage, read_write> args: array<DrawArgs, 37>;
// Copied back to the CPU by Bevy's readback plugin: the 32 counts, the
// frame stamp and the soldier count, 1 if the occlusion test ran, one
// spare, the 16 caster counts, the 16 hidden counts.
@group(0) @binding(8) var<storage, read_write> readback: array<u32, 68>;
// Per pose slot, the record it poses, in one region per kind.
@group(0) @binding(9) var<storage, read_write> pose_src: array<u32>;
// Five words per list, 0..16 the camera's buckets and 16..32 the casters as
// in `counts`: the arguments of its draw when its level is indexed.
@group(0) @binding(10) var<storage, read_write> indexed_args: array<u32, 160>;

// Where a build thread's soldier or body goes on the camera's lists, for
// `scatter` to place once `finalize` knows where every bin starts.
struct Draw {
    // bucket * DEPTH_BINS + depth bin, NO_DRAW when the camera does not
    // draw him.
    cell: u32,
    // His slot in the bin.
    slot: u32,
    // His list entry, the level in the top two bits.
    entry: u32,
};

// One per build thread: the living, then the fallen.
@group(0) @binding(11) var<storage, read_write> draws: array<Draw>;

// Last frame's farthest depths over the view (render_units_occlusion.rs):
// level k's texel covers 2^(k + 1) pixels on each axis.
@group(0) @binding(12) var depth_pyramid: texture_2d<f32>;

// An index list slot past the last soldier, up to the end of the last group
// of an indexed draw: the vertex shader draws nothing for it
// (unit_instancing.wgsl `vertex_pull`).
const EMPTY_SLOT: u32 = 0xffffffffu;

const CULL_RADIUS: f32 = 2.5;
const LOD_JITTER: f32 = 0.2;
const NUM_LODS: u32 = 4u;
const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;

// units.rs swing bits.
const SWING_STATE_MASK: u32 = 3u;
const SWING_WINDUP: u32 = 1u;
const SWING_RECOVER: u32 = 2u;
const SWING_DEFEND: u32 = 3u;
const SWING_CHARGE: u32 = 4u;
const SWING_STYLE_SHIFT: u32 = 3u;
const SWING_STYLE_MASK: u32 = 24u;
const SWING_STAGGERED: u32 = 32u;
const SWING_RANGED: u32 = 128u;
// Bits `Smooth::swing` carries above the swing byte: he was winding up
// last frame; his current or last draw started from the held aim; he
// holds the drawn aim (render_units.rs `Smooth::up`, `Smooth::drawn`).
const SM_WINDING: u32 = 256u;
const SM_UP: u32 = 512u;
const SM_DRAWN: u32 = 1024u;
// render_units.rs DRAWN_EARLY_S.
const DRAWN_EARLY_S: f32 = 0.2;

// RegimentRecord flags.
const REG_BROKEN: u32 = 1u;
const REG_SELECTED: u32 = 2u;
const REG_HOVERED: u32 = 4u;
const REG_HOVER_OWN: u32 = 8u;

// Ring styles (selection_rings.rs `ring_style`).
const RING_SELECTED: u32 = 0u;
const RING_HOVER_OWN: u32 = 1u;
const RING_HOVER_ENEMY: u32 = 2u;
// The ring count in `counts`, and the rings' entry in `args`.
const RING_COUNTER: u32 = 48u;
const RING_ARG: u32 = 32u;
// The pose slot counters in `counts`, and the pose dispatches in `args`.
// Per kind, the near levels' slots count up from the front of the kind's
// region and the far levels' down from its back (`list_entry`).
const POSE_COUNTER: u32 = 49u;
const POSE_ARG: u32 = 33u;
// Depth bins per bucket, and where their counters start in `counts`.
const DEPTH_BINS: u32 = 128u;
const BIN_COUNTER: u32 = 53u;
const POSE_FAR_COUNTER: u32 = BIN_COUNTER + 16u * DEPTH_BINS;
// Per bucket, the soldiers in the frustum the occlusion test hid.
const HIDDEN_COUNTER: u32 = POSE_FAR_COUNTER + 4u;
// Texels per axis the occlusion test reads at most: it reads the finest
// level where the box spans this many.
const OCCLUDE_SPAN: u32 = 8u;
// The levels a skinned kind poses by its skeleton (render_units_gpu.rs
// SKEL_LODS, unit_pose.wgsl `pose_write`).
const SKEL_LODS: u32 = 2u;
// Levels drawn near to far: below this one. The near levels' soldiers are
// large on screen, so the depth test spares the shading of most of those
// behind them. A far soldier is a few pixels, and his neighbours in his
// regiment, next to him in the build's order, write the same framebuffer
// tiles: sorted by depth, a level's consecutive soldiers lie along a band
// across the screen instead, and the far levels draw slower.
const ORDERED_LODS: u32 = 2u;
// A thread that put nothing on the camera's lists this frame.
const NO_DRAW: u32 = 0xffffffffu;

// Detail level for a squared distance: the farthest threshold passed wins.
fn level(t: vec4<f32>, d2: f32) -> u32 {
    var lod = 0u;
    if d2 > t.x {
        lod = 1u;
    }
    if d2 > t.y {
        lod = 2u;
    }
    if d2 > t.z {
        lod = 3u;
    }
    return lod;
}

// Frustum::intersects_sphere with intersect_far = false: the first five
// half spaces, out when the sphere lies entirely behind one.
fn culled(p: vec3<f32>) -> bool {
    if params.cull == 0u {
        return false;
    }
    let c = vec4<f32>(p, 1.0);
    for (var i = 0u; i < 5u; i++) {
        if dot(params.planes[i], c) + CULL_RADIUS <= 0.0 {
            return true;
        }
    }
    return false;
}

// Whether the soldier at `p` facing `yaw` was hidden last frame: his box
// projected with last frame's camera, his nearest point behind the farthest
// depth of every pixel the box covers. A box partly off last frame's
// screen, or reaching past the camera's near plane, is never hidden.
fn occluded(p: vec3<f32>, yaw: f32, kind: u32, fallen: bool) -> bool {
    var b = params.bound_live[kind];
    var down = params.bound_down_live[kind];
    if fallen {
        b = params.bound_fall[kind];
        down = params.bound_down_fall[kind];
    }
    // The world directions of his forward (+z) and his right (+x), as
    // unit_pose.wgsl `rot_y` turns them.
    let f = vec2<f32>(sin(yaw), cos(yaw));
    let r = vec2<f32>(f.y, -f.x);
    var lo = vec2<f32>(1e9);
    var hi = vec2<f32>(-1e9);
    var near = 0.0;
    for (var k = 0u; k < 8u; k++) {
        let lz = select(-b.y, b.x, (k & 1u) != 0u);
        let lx = select(-b.z, b.z, (k & 2u) != 0u);
        let ly = select(-down, b.w, (k & 4u) != 0u);
        let w = p + vec3<f32>(r.x * lx + f.x * lz, ly, r.y * lx + f.y * lz);
        let c = params.occlude_clip * vec4<f32>(w, 1.0);
        // Reversed depth: z over w grows toward the camera and passes 1
        // at the near plane.
        if c.w <= 0.0 || c.z >= c.w {
            return false;
        }
        let ndc = c.xyz / c.w;
        lo = min(lo, ndc.xy);
        hi = max(hi, ndc.xy);
        near = max(near, ndc.z);
    }
    // To pixels, y down. A sample of pixel i lies in [i, i + 1).
    let vp = params.occlude_view;
    let px_lo = vp.xy + vec2<f32>(lo.x * 0.5 + 0.5, 0.5 - hi.y * 0.5) * vp.zw;
    let px_hi = vp.xy + vec2<f32>(hi.x * 0.5 + 0.5, 0.5 - lo.y * 0.5) * vp.zw;
    if any(px_lo < vp.xy) || any(px_hi >= vp.xy + vp.zw) {
        return false;
    }
    let a = vec2<u32>(px_lo);
    let z = vec2<u32>(px_hi);
    var level = 0u;
    while level < params.occlude.y && any((z >> vec2<u32>(level + 1u)) - (a >> vec2<u32>(level + 1u)) >= vec2<u32>(OCCLUDE_SPAN)) {
        level++;
    }
    if level >= params.occlude.y {
        return false;
    }
    // The last texel of a level covers everything to the screen's edge
    // (unit_depth_pyramid.wgsl).
    let last = textureDimensions(depth_pyramid, level) - 1u;
    let ta = min(a >> vec2<u32>(level + 1u), last);
    let tz = min(z >> vec2<u32>(level + 1u), last);
    for (var y = ta.y; y <= tz.y; y++) {
        for (var x = ta.x; x <= tz.x; x++) {
            if textureLoad(depth_pyramid, vec2<u32>(x, y), i32(level)).r <= near {
                return false;
            }
        }
    }
    return true;
}

// The occlusion test for a soldier or body in the frustum, counted on his
// bucket: DRAWN, or hidden and left undrawn in cull mode (HIDDEN), or
// hidden and drawn marked in count mode under FL_OCCLUDE_CHECK (MARKED):
// unit_instancing.wgsl shows a marked soldier flat magenta, so any magenta
// on screen is a hidden soldier who shows.
const DRAWN: u32 = 0u;
const HIDDEN: u32 = 1u;
const MARKED: u32 = 2u;
fn occlusion(bucket: u32, p: vec3<f32>, yaw: f32, kind: u32, fallen: bool) -> u32 {
    // FL_OCCLUDE_CHECK=2 marks every soldier the test reads, to show the
    // mark works.
    if params.occlude.w == 2u && params.occlude.y != 0u {
        return MARKED;
    }
    if params.occlude.y == 0u || !occluded(p, yaw, kind, fallen) {
        return DRAWN;
    }
    atomicAdd(&counts[HIDDEN_COUNTER + bucket], 1u);
    if params.occlude.x == 1u {
        return HIDDEN;
    }
    return select(DRAWN, MARKED, params.occlude.w != 0u);
}

// Gait cycles per second at this ground speed (gait.rs `rate`).
fn gait_rate(speed: f32) -> f32 {
    return 1.0 + 0.16 * speed;
}

fn lod_jitter(seed: f32) -> f32 {
    return 1.0 - 0.5 * LOD_JITTER + LOD_JITTER * seed;
}

// Count build thread `t`'s soldier or body, `dist2` squared metres from the
// camera, into his bucket and its depth bin, and keep where he goes:
// `scatter` writes his entry once `finalize` knows where each bin starts.
// The bins run near to far on the levels below ORDERED_LODS; a farther
// level keeps one bin. The entry carries his level in its top two bits and
// the occlusion check's mark in the bit below (`occlusion`).
fn queue_draw(t: u32, bucket: u32, entry: u32, lod: u32, dist2: f32, marked: bool) {
    atomicAdd(&counts[bucket], 1u);
    var bin = 0u;
    if lod < ORDERED_LODS {
        bin = u32(clamp(log2(max(dist2, 1.0)) * params.order, 0.0, f32(DEPTH_BINS - 1u)));
    }
    let cell = bucket * DEPTH_BINS + bin;
    let slot = atomicAdd(&counts[BIN_COUNTER + cell], 1u);
    draws[t] = Draw(cell, slot, entry | (lod << 30u) | select(0u, 1u << 29u, marked));
}

// The list entry of drawn record `r` of a soldier of `kind` at level
// `lod`: the record, or with the pose pass a pose slot of his kind that
// the pass fills from the record. A soldier or body takes at most one
// slot a frame, so a kind's region holds its living and its fallen.
//
// A skinned kind poses by its skeleton at the near levels only
// (unit_pose.wgsl `pose_write`), and the level rides the top two bits of
// the slot's source. The near levels take their slots from the front of
// the region and the far levels from its back: the pose pass runs the
// skeleton for a whole warp when one thread of it needs it, and in a
// region in soldier order nearly every warp of a kind holds one of the
// near soldiers (8% of the knights in the flank view, 93% of its warps).
fn list_entry(kind: u32, r: u32, lod: u32) -> u32 {
    if params.pose_pass == 0u {
        return r;
    }
    var slot: u32;
    if lod < SKEL_LODS {
        slot = atomicAdd(&counts[POSE_COUNTER + kind], 1u);
    } else {
        slot = params.pose_cap[kind] - 1u - atomicAdd(&counts[POSE_FAR_COUNTER + kind], 1u);
    }
    pose_src[params.pose_src_base[kind] + slot] = r | (lod << 30u);
    return slot;
}

// The cascades a soldier at `p` casts into, one bit each: none when the
// ground his shadow can fall on is out of view, else every cascade that
// is sampled at the view depths of that ground. His shadow lands where
// it does whether or not the camera sees him.
fn cascade_mask(p: vec3<f32>) -> u32 {
    if params.n_cascades == 0u {
        return 0u;
    }
    let reach = vec3<f32>(p.x + params.shadow_reach.x, p.y, p.z + params.shadow_reach.z);
    let r = params.shadow_reach.w;
    if params.cull != 0u {
        let c = vec4<f32>(reach, 1.0);
        for (var i = 0u; i < 5u; i++) {
            if dot(params.planes[i], c) + r <= 0.0 {
                return 0u;
            }
        }
    }
    let depth = dot(reach - params.cam_pos, params.cam_fwd.xyz);
    var mask = 0u;
    for (var c = 0u; c < params.n_cascades; c++) {
        if depth + r >= params.cascade_near[c] && depth - r <= params.cascade_far[c] {
            mask |= 1u << c;
        }
    }
    return mask;
}

// Put record `entry` of a soldier of `kind` on the caster list of every
// cascade in `mask`.
fn append_casters(mask: u32, kind: u32, entry: u32) {
    for (var c = 0u; c < params.n_cascades; c++) {
        if (mask & (1u << c)) != 0u {
            let slot = atomicAdd(&counts[32u + c * 4u + kind], 1u);
            index_list[params.shadow_lists[c][kind] + slot] = entry;
        }
    }
}

// Whether a swing byte's blow is this kind's slow cut, as
// unit_types.rs `UnitTypeParams::is_cut`.
fn is_cut(kind: u32, swing: u32) -> bool {
    return params.cut_windup[kind] > 0.0
        && (swing & SWING_RANGED) == 0u
        && (swing & SWING_STYLE_MASK) == (1u << SWING_STYLE_SHIFT);
}

// A skinned soldier's steps. Near the enemy and slower than he turns to
// walk, he steps in the clip nearest his motion relative to his facing,
// one play per the ground it covers, so his feet keep to the ground.
// A play starts and ends in the ready pose, so he changes clip only
// there. When he stops or speeds up into his gait, the foot in the air
// lands at the clip's own pace and he settles into his stance. Returns
// the record's fourth position float: the clip slot, plus 4 times the
// place in the play and 4096 times his weight in it, both in 1023rds.
fn steps(sm: ptr<function, Smooth>, kind: u32, vel: vec2<f32>, yaw: f32, alive: bool) -> f32 {
    let travel = params.step_travel[kind];
    let fwd = vec2<f32>(sin(yaw), cos(yaw));
    let local = vec2<f32>(dot(vel, vec2<f32>(fwd.y, -fwd.x)), dot(vel, fwd));
    var pick = select(1u, 0u, local.x > 0.0);
    if abs(local.y) > abs(local.x) {
        pick = select(3u, 2u, local.y > 0.0);
    }
    let speed = length(vel);
    let want = alive && (*sm).band > 0.15 && (*sm).walk < params.steps.x && speed > 0.06
        && travel[pick] > 0.0;
    if (*sm).step_w <= 0.0 {
        (*sm).step = 0.0;
        (*sm).step_clip = pick;
    }
    let ease = params.dt / params.steps.y;
    var x = (*sm).step;
    if want {
        x += speed * params.dt / max(travel[(*sm).step_clip], 0.01);
        (*sm).step_w = min((*sm).step_w + ease, 1.0);
    } else {
        // A step is half a play: one foot moves. The foot in the air
        // lands, then he settles.
        let half = x * 2.0;
        if half - floor(half) > 0.0 {
            x = min(x + params.dt / max(params.step_len[kind][(*sm).step_clip], 0.01), (floor(half) + 1.0) * 0.5);
        } else {
            (*sm).step_w = max((*sm).step_w - ease, 0.0);
        }
    }
    if x >= 1.0 {
        x -= floor(x);
        if want {
            (*sm).step_clip = pick;
        }
    }
    (*sm).step = x;
    return f32((*sm).step_clip) + 4.0 * round(x * 1023.0) + 4096.0 * round((*sm).step_w * 1023.0);
}

// Seconds a blow of this swing byte takes to come back once it lands, as
// render_units.rs `follow_s`.
fn follow_s(kind: u32, swing: u32) -> f32 {
    if is_cut(kind, swing) {
        return params.cut_return[kind];
    }
    return params.attack.x;
}

fn build_soldier(i: u32) {
    let s = soldiers[i];
    let pos = vec3<f32>(s.px, s.py, s.pz);
    let prev = vec3<f32>(s.qx, s.qy, s.qz);
    let kind = s.a & 0xffu;
    let swing = (s.a >> 8u) & 0xffu;
    let swing_t = f32((s.a >> 16u) & 0xffu);
    let flash = f32((s.a >> 24u) & 0xffu);
    let group = s.b & 0xffffffu;
    let death_t = f32((s.b >> 24u) & 0xffu);

    var reg = Regiment(0.0, -1.0, 0.0, 0u);
    if group < params.n_regs {
        reg = regiments[group];
    }
    var sm = smoothing[i];

    // Walk signal: the ACTUAL per-tick displacement smoothed as a vector,
    // then its length, updated before the cull so units re-entering the
    // frustum have a live value. A man jostled back and forth in a press
    // gets nowhere and does not read as walking.
    let step = pos - prev;
    let drift = mix(vec2<f32>(sm.drift_x, sm.drift_z), step.xz * params.inv_dt, params.k_walk);
    sm.drift_x = drift.x;
    sm.drift_z = drift.y;
    sm.walk = length(drift);

    // The attack on anim z, 0 when he is not attacking, as in
    // render_units.rs `attack_signal`.
    let shot_s = params.shot.x + params.shot.y + params.shot.z;
    let winding = (swing & SWING_STATE_MASK) == SWING_WINDUP && death_t == 0.0;
    var flags = sm.swing & (SM_UP | SM_DRAWN);
    sm.shot = max(sm.shot - params.dt, 0.0);
    if winding {
        if (sm.swing & SM_WINDING) == 0u && (swing & SWING_RANGED) != 0u {
            flags = select(flags & ~SM_UP, flags | SM_UP, (flags & SM_DRAWN) != 0u);
        }
        var w = params.windup[kind];
        if (swing & SWING_RANGED) != 0u {
            w = params.consts.x;
        } else if is_cut(kind, swing) {
            w = params.cut_windup[kind];
        }
        sm.atk = clamp((w - swing_t + params.alpha) / (w + 1.0), 0.0, 1.0);
        sm.swing = swing;
        sm.shot = 0.0;
    } else if (sm.swing & SM_WINDING) != 0u
        && (swing & SWING_STATE_MASK) == SWING_RECOVER
        && (swing & SWING_STAGGERED) == 0u
        && ((sm.swing & SWING_RANGED) == 0u || swing_t > params.bow.z) {
        if (sm.swing & SWING_RANGED) != 0u {
            sm.shot = shot_s;
            flags &= ~SM_DRAWN;
        } else {
            sm.follow = follow_s(kind, sm.swing);
        }
        sm.atk = 0.0;
    } else {
        sm.follow = max(sm.follow - params.dt, 0.0);
        sm.atk = max(sm.atk - params.dt / params.attack.w, 0.0);
    }
    let ranged = (sm.swing & SWING_RANGED) != 0u;
    let shooting = (winding && ranged) || sm.shot > 0.0;
    var up = 0.0;
    if shooting && death_t == 0.0 && sm.walk < params.shot.w {
        up = 1.0;
    }
    sm.bow = clamp(
        sm.bow + clamp(up - sm.bow, -params.dt / params.bow.y, params.dt / params.bow.x),
        0.0,
        1.0,
    );
    // His reload ends at the drawn aim, which a skeleton's draw reaches
    // DRAWN_EARLY_S before the reload's end.
    if sm.shot > 0.0 && sm.shot <= params.shot.z + DRAWN_EARLY_S {
        flags |= SM_DRAWN;
    }
    if sm.bow <= 0.0 {
        flags &= ~SM_DRAWN;
    }
    sm.swing = (sm.swing & 0xffu) | flags | select(0u, SM_WINDING, winding);
    var digit = f32((sm.swing & SWING_STYLE_MASK) >> SWING_STYLE_SHIFT);
    if (sm.swing & SWING_CHARGE) != 0u {
        digit += 3.0;
    }
    var attack = 0.0;
    if sm.shot > 0.0 {
        let t = shot_s - sm.shot;
        let start = select(0.0, 6.0, (flags & SM_UP) != 0u);
        // Holding the drawn aim.
        attack = params.bow.w + 6.0;
        if t < params.shot.x {
            attack = params.bow.w + start + 2.0 + t / params.shot.x;
        } else if t < params.shot.x + params.shot.y {
            attack = params.bow.w + start + 4.0 + (t - params.shot.x) / params.shot.y;
        }
    } else if ranged && (sm.atk > 0.0 || sm.bow > 0.0) {
        // A draw, or one cut short going back, from where it started;
        // else the bow coming down from the aim or from ready.
        let aim = select(SM_DRAWN, SM_UP, winding || sm.atk > 0.0);
        attack = params.bow.w + select(0.0, 6.0, (flags & aim) != 0u) + sm.atk;
    } else if sm.follow > 0.0 {
        attack = digit * 2.0 + params.attack.y + params.attack.z * (1.0 - sm.follow / follow_s(kind, sm.swing));
    } else if sm.atk > 0.0 {
        attack = digit * 2.0 + sm.atk;
    }

    var tier = reg.stance;
    if attack > 0.0 && attack < params.bow.w && (sm.swing & SWING_RANGED) == 0u {
        tier = max(tier, params.fighting);
    }
    sm.band += (tier - sm.band) * params.k_band;
    sm.wall += (reg.walled - sm.wall) * params.k_wall;
    // The step is capped as in gait.rs `advance`.
    let g = sm.gait + min(gait_rate(sm.walk) * params.dt, 0.25);
    sm.gait = g - floor(g);
    // The record's fourth position float: an archer's bow, a skinned
    // kind's steps.
    var fourth = sm.bow;
    if any(params.step_travel[kind] > vec4<f32>(0.0)) {
        fourth = steps(&sm, kind, step.xz * params.inv_dt, s.yaw, death_t == 0.0);
    }

    let position = mix(prev, pos, params.alpha);
    let jitter = lod_jitter(s.seed);
    let d = position - params.cam_pos;
    let d2 = dot(d, d) * jitter * jitter;
    let in_view = !culled(position);
    // Facing interpolates like position, wrap-aware.
    let dyr = s.yaw - s.yaw_prev + PI;
    let dy = dyr - floor(dyr / TAU) * TAU - PI;
    let yaw = s.yaw_prev + dy * params.alpha;
    // Detail level: jittered distance against this frame's thresholds,
    // held inside the hysteresis bounds. A soldier the camera does not see
    // keeps his held level for when he comes back, and casts by the plain
    // thresholds. One hidden behind nearer things keeps his level up to
    // date, so he comes back at it.
    var lod = level(params.bands[kind * 3u + 2u], d2);
    var occl = DRAWN;
    if in_view {
        let fine = level(params.bands[kind * 3u], d2);
        let coarse = level(params.bands[kind * 3u + 1u], d2);
        sm.lod = clamp(sm.lod, fine, coarse);
        lod = sm.lod;
        occl = occlusion(kind * NUM_LODS + lod, position, yaw, kind, death_t > 0.0);
    }
    let visible = in_view && occl != HIDDEN;
    var casts = 0u;
    if lod < params.cast_lods {
        casts = cascade_mask(position);
    }
    if !in_view && casts == 0u {
        smoothing[i] = sm;
        return;
    }

    var rgb = vec3<f32>(s.cr, s.cg, s.cb);
    if (reg.flags & REG_BROKEN) != 0u {
        let gray = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
        rgb = rgb * 0.55 + vec3<f32>(gray) * 0.45;
    }

    // The attack, else the victory cheer, else nothing.
    var lunge = attack;
    if attack <= 0.0 && death_t == 0.0 && reg.celebrate >= 0.0 {
        lunge = params.consts.w + reg.celebrate;
    }

    // fx: [0,1] hit flash, (1,2] death progress. A death's first tick has
    // progress 0, the living pose, and at 1.0 it would read as a full
    // flash, so it keeps the hit flash like a living man.
    // A flash value of 16 or more is a twitch code, not a flash.
    var fx = select(flash * 0.25, 0.0, flash >= 16.0);
    if death_t > 0.0 && death_t < params.consts.y {
        fx = 2.0 - death_t / params.consts.y;
    }

    // The fourth anim2 float: the stagger 0 to 1; a block 1 + how far
    // its clip has run, 0 to 1; a parry 2 + the same; a twitch away from
    // a blow the dice stopped, 3 + the side it came from (0 front, 1 his
    // left, 2 his right, 3 behind) + how much is left (render_units.rs
    // the same). Through the wait before a clip he shows nothing. A kind
    // without the clip of his answer shows the twitch, as a man not
    // answering does.
    var stagger = 0.0;
    let parrying = (swing & SWING_STYLE_MASK) == (1u << SWING_STYLE_SHIFT);
    let has_answer = ((u32(params.parry.w) >> (kind * 2u + select(0u, 1u, parrying))) & 1u) != 0u;
    if death_t == 0.0 && (swing & SWING_STAGGERED) != 0u {
        stagger = min(swing_t / params.consts.z, 1.0);
    } else if death_t == 0.0 && (swing & SWING_STATE_MASK) == SWING_DEFEND && has_answer {
        let len = select(params.guard.z, params.parry.z, parrying);
        let e = len - swing_t + params.alpha;
        if e > 0.0 {
            stagger = select(1.0, 2.0, parrying) + min(e / len, 0.999);
        }
    } else if death_t == 0.0 && flash >= 16.0 {
        let side = floor(flash / 16.0);
        let left = (flash - side * 16.0) / params.guard.w;
        stagger = 2.0 + side + min(left, 0.999);
    }

    records[i] = Record(
        vec4<f32>(position, fourth),
        vec4<f32>(rgb, s.seed),
        vec4<f32>(yaw, sm.walk, lunge, fx),
        vec4<f32>(sm.band, sm.wall, sm.gait, stagger),
    );
    smoothing[i] = sm;
    // A hidden soldier keeps his ring, which reads his record: it lies on
    // the ground around him, past his box.
    if in_view && death_t == 0.0 && (reg.flags & (REG_SELECTED | REG_HOVERED | REG_HOVER_OWN)) != 0u {
        append_ring(i, reg.flags, kind);
    }
    if !visible && casts == 0u {
        return;
    }
    let entry = list_entry(kind, i, lod);
    if visible {
        queue_draw(i, kind * NUM_LODS + lod, entry, lod, dot(d, d), occl == MARKED);
    }
    append_casters(casts, kind, entry);
}

// A living soldier of a selected or hovered regiment gets a ring under
// his feet: his record index, the ring style and his kind in one entry
// (selection_rings.rs, unit_rings.wgsl). The enemy under the cursor wins
// over the selection, the selection over a hovered own regiment.
fn append_ring(i: u32, flags: u32, kind: u32) {
    var style = RING_HOVER_OWN;
    if (flags & REG_HOVERED) != 0u {
        style = RING_HOVER_ENEMY;
    } else if (flags & REG_SELECTED) != 0u {
        style = RING_SELECTED;
    }
    let slot = atomicAdd(&counts[RING_COUNTER], 1u);
    index_list[params.ring_base + slot] = i | (style << 28u) | (kind << 30u);
}

// The fallen: a frozen record in the corpse region of `records`. A cull,
// a level pick from the plain thresholds (no hysteresis, bodies do not
// move), then the camera's list and the caster lists as for the living.
fn build_corpse(j: u32, t: u32) {
    var kind = 0u;
    var start = 0u;
    for (var k = 0u; k < 4u; k++) {
        let len = params.corpse_len[k];
        if j < start + len {
            kind = k;
            break;
        }
        start += len;
    }
    let ridx = params.corpse_base + kind * params.corpse_cap + (j - start);
    let rec = records[ridx];
    let position = rec.pos_scale.xyz;
    let jitter = lod_jitter(rec.color.a);
    let d = position - params.cam_pos;
    let d2 = dot(d, d) * jitter * jitter;
    let lod = level(params.bands[kind * 3u + 2u], d2);
    let in_view = !culled(position);
    var occl = DRAWN;
    if in_view {
        occl = occlusion(kind * NUM_LODS + lod, position, rec.anim.x, kind, true);
    }
    let visible = in_view && occl != HIDDEN;
    var casts = 0u;
    if lod < params.cast_lods {
        casts = cascade_mask(position);
    }
    if !visible && casts == 0u {
        return;
    }
    let entry = list_entry(kind, ridx, lod);
    if visible {
        let bucket = kind * NUM_LODS + lod;
        atomicAdd(&counts[16u + bucket], 1u);
        queue_draw(t, bucket, entry, lod, dot(d, d), occl == MARKED);
    }
    append_casters(casts, kind, entry);
}

@compute @workgroup_size(64)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= build_threads() {
        return;
    }
    draws[i].cell = NO_DRAW;
    if i < params.n {
        build_soldier(i);
    } else {
        build_corpse(i - params.n, i);
    }
}

// The living, then the fallen of every kind.
fn build_threads() -> u32 {
    return params.n + params.corpse_len.x + params.corpse_len.y + params.corpse_len.z + params.corpse_len.w;
}

// After `finalize`: each soldier or body on the camera's lists goes to his
// bin's first slot plus his slot in the bin.
@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if t >= build_threads() {
        return;
    }
    let d = draws[t];
    if d.cell == NO_DRAW {
        return;
    }
    let first = atomicLoad(&counts[BIN_COUNTER + d.cell]);
    index_list[params.buckets[d.cell / DEPTH_BINS].x + first + d.slot] = d.entry;
}

// The arguments of list `b`'s indexed draw: `group` soldiers of `corners`
// indices per instance, as many instances as cover `count` soldiers. The
// slots from the last soldier to the end of the last group get the empty
// mark: the build left whatever an earlier frame wrote there. Every list has
// room for them (render_units_gpu.rs `MAX_GROUP`).
fn finish_indexed(b: u32, first: u32, count: u32, corners: u32, group: u32) {
    if group == 0u {
        return;
    }
    let instances = (count + group - 1u) / group;
    indexed_args[b * 5u] = corners * group;
    indexed_args[b * 5u + 1u] = instances;
    indexed_args[b * 5u + 2u] = 0u;
    indexed_args[b * 5u + 3u] = 0u;
    indexed_args[b * 5u + 4u] = 0u;
    for (var k = count; k < instances * group; k++) {
        index_list[first + k] = EMPTY_SLOT;
    }
}

// One thread per list: the list count becomes the arguments of its draw,
// the corner count of the expanded draw and the groups of the indexed one.
// A camera bucket's thread also turns its depth bin counts into the first
// slot of each bin. `counts` was cleared before `build` ran.
@compute @workgroup_size(32)
fn finalize(@builtin(local_invocation_index) b: u32) {
    if b >= 16u {
        // A caster list: cascade * 4 + kind.
        let s = b - 16u;
        let count = atomicLoad(&counts[32u + s]);
        let corners = params.shadow_corners[s / 4u][s % 4u];
        args[b] = DrawArgs(count * corners, 1u, 0u, 0u);
        finish_indexed(b, params.shadow_lists[s / 4u][s % 4u], count, corners, params.shadow_groups[s / 4u][s % 4u]);
        readback[36u + s] = count;
        return;
    }
    let count = atomicLoad(&counts[b]);
    let fallen = atomicLoad(&counts[16u + b]);
    var first = 0u;
    for (var k = b * DEPTH_BINS; k < (b + 1u) * DEPTH_BINS; k++) {
        let in_bin = atomicLoad(&counts[BIN_COUNTER + k]);
        atomicStore(&counts[BIN_COUNTER + k], first);
        first += in_bin;
    }
    let bucket = params.buckets[b];
    args[b] = DrawArgs(count * bucket.y, 1u, 0u, 0u);
    finish_indexed(b, bucket.x, count, bucket.y, bucket.w);
    readback[b] = count;
    readback[16u + b] = fallen;
    readback[52u + b] = atomicLoad(&counts[HIDDEN_COUNTER + b]);
    if b < 4u {
        let posed = atomicLoad(&counts[POSE_COUNTER + b]) + atomicLoad(&counts[POSE_FAR_COUNTER + b]);
        args[POSE_ARG + b] = DrawArgs((posed + 63u) / 64u, 1u, 1u, 0u);
    }
    if b == 0u {
        readback[32u] = params.frame;
        readback[33u] = params.n;
        // Whether the occlusion test ran this frame.
        readback[34u] = select(0u, 1u, params.occlude.y != 0u);
        // Six corners per ring (two triangles).
        args[RING_ARG] = DrawArgs(atomicLoad(&counts[RING_COUNTER]) * 6u, 1u, 0u, 0u);
    }
}
