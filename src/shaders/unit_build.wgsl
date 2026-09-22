// Builds the unit render data on the GPU: the port of render_units.rs
// `sync_instance_data`, one thread per soldier. Reads the per-tick
// soldier snapshot, the per-frame regiment records and the camera,
// writes the 64 byte instance record the vertex shader consumes, and
// appends the soldier to his kind-by-level index list. A second entry
// turns the list counts into indirect draw arguments.
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

// Per-soldier smoothing state, indexed by soldier index. Indices shuffle
// on death-sweep swap-removes: a one-frame inherited value is invisible.
struct Smooth {
    walk: f32,
    band: f32,
    march: f32,
    wall: f32,
    lod: u32,
};

// Per-regiment pose signals for this frame (render_units_gpu.rs
// RegimentRecord).
struct Regiment {
    stance: f32,
    // Victory cheer progress 0..1, negative when not celebrating.
    celebrate: f32,
    marching: f32,
    walled: f32,
    phase: f32,
    flags: u32,
};

struct Params {
    planes: array<vec4<f32>, 6>,
    cam_pos: vec3<f32>,
    alpha: f32,
    k_walk: f32,
    k_band: f32,
    k_march: f32,
    inv_dt: f32,
    n: u32,
    n_regs: u32,
    cull: u32,
    corpse_base: u32,
    corpse_len: vec4<u32>,
    corpse_cap: u32,
    frame: u32,
    pad0: u32,
    pad1: u32,
    // [kind * 3 + set]: set 0 fine, 1 coarse, 2 plain. xyz = squared
    // switch distances for L0/L1, L1/L2, L2/L3 (inf = switch disabled).
    bands: array<vec4<f32>, 12>,
    windup: vec4<f32>,
    // draw_ticks, death_ticks, hit_stagger_ticks, celebrate_base
    consts: vec4<f32>,
    // x = first index slot of the bucket, y = mesh corners per soldier.
    buckets: array<vec4<u32>, 16>,
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
// 0..16 soldiers per bucket (living and fallen), 16..32 the fallen alone.
@group(0) @binding(6) var<storage, read_write> counts: array<atomic<u32>, 32>;
@group(0) @binding(7) var<storage, read_write> args: array<DrawArgs, 16>;
// Copied back to the CPU by Bevy's readback plugin: the 32 counts, the
// frame stamp and the soldier count.
@group(0) @binding(8) var<storage, read_write> readback: array<u32, 36>;

const CULL_RADIUS: f32 = 2.5;
const LOD_JITTER: f32 = 0.2;
const NUM_LODS: u32 = 4u;
const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;

// units.rs swing bits.
const SWING_STATE_MASK: u32 = 3u;
const SWING_WINDUP: u32 = 1u;
const SWING_CHARGE: u32 = 4u;
const SWING_STYLE_SHIFT: u32 = 3u;
const SWING_STYLE_MASK: u32 = 24u;
const SWING_STAGGERED: u32 = 32u;
const SWING_RANGED: u32 = 128u;

// RegimentRecord flags.
const REG_BROKEN: u32 = 1u;
const REG_SELECTED: u32 = 2u;
const REG_HOVERED: u32 = 4u;

const HIGHLIGHT: vec3<f32> = vec3<f32>(1.0, 1.0, 0.55);
const HOSTILE: vec3<f32> = vec3<f32>(1.0, 0.30, 0.22);

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

fn lod_jitter(seed: f32) -> f32 {
    return 1.0 - 0.5 * LOD_JITTER + LOD_JITTER * seed;
}

fn append(bucket: u32, entry: u32, lod: u32) {
    let slot = atomicAdd(&counts[bucket], 1u);
    index_list[params.buckets[bucket].x + slot] = entry | (lod << 30u);
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

    var reg = Regiment(0.0, -1.0, 0.0, 0.0, 0.0, 0u);
    if group < params.n_regs {
        reg = regiments[group];
    }
    var sm = smoothing[i];

    // Walk signal: smoothed ACTUAL per-tick displacement, updated before
    // the cull so units re-entering the frustum have a live value.
    let step = pos - prev;
    let disp = length(step.xz) * params.inv_dt;
    sm.walk += (disp - sm.walk) * params.k_walk;
    sm.band += (reg.stance - sm.band) * params.k_band;
    sm.march += (reg.marching - sm.march) * params.k_march;
    sm.wall += (reg.walled - sm.wall) * params.k_march;

    let position = mix(prev, pos, params.alpha);
    if culled(position) {
        smoothing[i] = sm;
        return;
    }

    // Detail level: jittered distance against this frame's thresholds,
    // held inside the hysteresis bounds.
    let jitter = lod_jitter(s.seed);
    let d = position - params.cam_pos;
    let d2 = dot(d, d) * jitter * jitter;
    let fine = level(params.bands[kind * 3u], d2);
    let coarse = level(params.bands[kind * 3u + 1u], d2);
    sm.lod = clamp(sm.lod, fine, coarse);
    let lod = sm.lod;

    var rgb = vec3<f32>(s.cr, s.cg, s.cb);
    if (reg.flags & REG_BROKEN) != 0u {
        let gray = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
        rgb = rgb * 0.55 + vec3<f32>(gray) * 0.45;
    } else if (reg.flags & REG_SELECTED) != 0u {
        rgb = rgb * 0.35 + HIGHLIGHT * 0.65;
    } else if (reg.flags & REG_HOVERED) != 0u {
        rgb = rgb * 0.45 + HOSTILE * 0.55;
    }

    // Walk amount: deadband + smoothstep on the smoothed displacement.
    let t = clamp((sm.walk - 0.06) / (1.2 - 0.06), 0.0, 1.0);
    let move_amount = t * t * (3.0 - 2.0 * t);

    // Facing interpolates like position, wrap-aware.
    let dyr = s.yaw - s.yaw_prev + PI;
    let dy = dyr - floor(dyr / TAU) * TAU - PI;
    let yaw = s.yaw_prev + dy * params.alpha;

    // Attack lunge: quadratic wind-up, style in the 2s digit. Else the
    // victory cheer, else the smoothed stance band, else nothing.
    var lunge = 0.0;
    if (swing & SWING_STATE_MASK) == SWING_WINDUP {
        var w = params.windup[kind];
        if (swing & SWING_RANGED) != 0u {
            w = params.consts.x;
        }
        let tw = max((w - swing_t) / max(w, 1.0), 0.0);
        let charge = (swing & SWING_CHARGE) != 0u;
        var l = tw * tw;
        if charge {
            l = max(l * 1.35, 0.18);
        }
        let style = f32((swing & SWING_STYLE_MASK) >> SWING_STYLE_SHIFT);
        lunge = style * 2.0 + l;
    } else if death_t == 0.0 && reg.celebrate >= 0.0 {
        lunge = params.consts.w + reg.celebrate;
    } else if death_t == 0.0 {
        lunge = -sm.band;
    }

    // fx: [0,1) hit flash, [1,2] death progress.
    var fx = flash * 0.25;
    if death_t > 0.0 {
        fx = 2.0 - death_t / params.consts.y;
    }

    var stagger = 0.0;
    if death_t == 0.0 && (swing & SWING_STAGGERED) != 0u {
        stagger = min(swing_t / params.consts.z, 1.0);
    }

    records[i] = Record(
        vec4<f32>(position, 1.0),
        vec4<f32>(rgb, s.seed),
        vec4<f32>(yaw, move_amount, lunge, fx),
        vec4<f32>(sm.march, sm.wall, reg.phase, stagger),
    );
    smoothing[i] = sm;
    append(kind * NUM_LODS + lod, i, lod);
}

// The fallen: a frozen record in the corpse region of `records`. A cull,
// a level pick from the plain thresholds (no hysteresis, bodies do not
// move) and an index append.
fn build_corpse(j: u32) {
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
    if culled(position) {
        return;
    }
    let jitter = lod_jitter(rec.color.a);
    let d = position - params.cam_pos;
    let d2 = dot(d, d) * jitter * jitter;
    let lod = level(params.bands[kind * 3u + 2u], d2);
    let bucket = kind * NUM_LODS + lod;
    atomicAdd(&counts[16u + bucket], 1u);
    append(bucket, ridx, lod);
}

@compute @workgroup_size(64)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < params.n {
        build_soldier(i);
        return;
    }
    let j = i - params.n;
    let total = params.corpse_len.x + params.corpse_len.y + params.corpse_len.z + params.corpse_len.w;
    if j < total {
        build_corpse(j);
    }
}

// One thread per bucket: the list count becomes the pulled draw's vertex
// count. `counts` was cleared before `build` ran.
@compute @workgroup_size(16)
fn finalize(@builtin(local_invocation_index) b: u32) {
    let count = atomicLoad(&counts[b]);
    let fallen = atomicLoad(&counts[16u + b]);
    args[b] = DrawArgs(count * params.buckets[b].y, 1u, 0u, 0u);
    readback[b] = count;
    readback[16u + b] = fallen;
    if b == 0u {
        readback[32u] = params.frame;
        readback[33u] = params.n;
    }
}
