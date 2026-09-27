// Builds the unit render data on the GPU: the port of render_units.rs
// `sync_instance_data`, one thread per soldier. Reads the per-tick
// soldier snapshot, the per-frame regiment records and the camera,
// writes the 64 byte instance record the vertex shader consumes, and
// appends the soldier to his kind-by-level index list. A near soldier
// whose shadow can fall in view also goes on the caster list for his kind
// of each sun shadow cascade that serves the depths it falls at, whether
// or not the camera sees him (render_units_shadow.rs). A second entry
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
    // The swing byte of the current or last attack, and 1 << 8 when he
    // was winding up last frame.
    swing: u32,
    // Seconds left of the release, reload and hold after his last shot.
    shot: f32,
    // How far his bow is up toward the drawn ready pose, 0 to 1.
    bow: f32,
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
    // draw_ticks, death_ticks, hit_stagger_ticks, celebrate_base
    consts: vec4<f32>,
    // FOLLOW_S, FOLLOW_BASE, FOLLOW_SPAN, REWIND_S (render_units.rs).
    attack: vec4<f32>,
    // RELEASE_S, RELOAD_S, HOLD_S, BOW_WALK_MS (render_units.rs).
    shot: vec4<f32>,
    // BOW_RISE_S, BOW_FALL_S, CANCEL_TICKS, RANGED_BASE.
    bow: vec4<f32>,
    // x = first index slot of the bucket, y = mesh corners per soldier.
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
// selection rings.
@group(0) @binding(6) var<storage, read_write> counts: array<atomic<u32>, 49>;
// 0..16 the camera's buckets, 16..32 the casters as in `counts`, 32 the
// selection rings.
@group(0) @binding(7) var<storage, read_write> args: array<DrawArgs, 33>;
// Copied back to the CPU by Bevy's readback plugin: the 32 counts, the
// frame stamp and the soldier count, two spare, the 16 caster counts.
@group(0) @binding(8) var<storage, read_write> readback: array<u32, 52>;

const CULL_RADIUS: f32 = 2.5;
const LOD_JITTER: f32 = 0.2;
const NUM_LODS: u32 = 4u;
const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;

// units.rs swing bits.
const SWING_STATE_MASK: u32 = 3u;
const SWING_WINDUP: u32 = 1u;
const SWING_RECOVER: u32 = 2u;
const SWING_CHARGE: u32 = 4u;
const SWING_STYLE_SHIFT: u32 = 3u;
const SWING_STYLE_MASK: u32 = 24u;
const SWING_STAGGERED: u32 = 32u;
const SWING_RANGED: u32 = 128u;

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

// Gait cycles per second at this ground speed (gait.rs `rate`).
fn gait_rate(speed: f32) -> f32 {
    return 1.0 + 0.16 * speed;
}

fn lod_jitter(seed: f32) -> f32 {
    return 1.0 - 0.5 * LOD_JITTER + LOD_JITTER * seed;
}

fn append(bucket: u32, entry: u32, lod: u32) {
    let slot = atomicAdd(&counts[bucket], 1u);
    index_list[params.buckets[bucket].x + slot] = entry | (lod << 30u);
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

    // Walk signal: smoothed ACTUAL per-tick displacement, updated before
    // the cull so units re-entering the frustum have a live value.
    let step = pos - prev;
    let disp = length(step.xz) * params.inv_dt;
    sm.walk += (disp - sm.walk) * params.k_walk;

    // The attack on anim z, 0 when he is not attacking, as in
    // render_units.rs `attack_signal`.
    let follow_s = params.attack.x;
    let shot_s = params.shot.x + params.shot.y + params.shot.z;
    let winding = (swing & SWING_STATE_MASK) == SWING_WINDUP && death_t == 0.0;
    sm.shot = max(sm.shot - params.dt, 0.0);
    if winding {
        var w = params.windup[kind];
        if (swing & SWING_RANGED) != 0u {
            w = params.consts.x;
        }
        sm.atk = clamp((w - swing_t + params.alpha) / (w + 1.0), 0.0, 1.0);
        sm.swing = swing;
        sm.shot = 0.0;
    } else if (sm.swing & 256u) != 0u
        && (swing & SWING_STATE_MASK) == SWING_RECOVER
        && (swing & SWING_STAGGERED) == 0u
        && ((sm.swing & SWING_RANGED) == 0u || swing_t > params.bow.z) {
        if (sm.swing & SWING_RANGED) != 0u {
            sm.shot = shot_s;
        } else {
            sm.follow = follow_s;
        }
        sm.atk = 0.0;
    } else {
        sm.follow = max(sm.follow - params.dt, 0.0);
        sm.atk = max(sm.atk - params.dt / params.attack.w, 0.0);
    }
    sm.swing = (sm.swing & 0xffu) | select(0u, 256u, winding);
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
    var digit = f32((sm.swing & SWING_STYLE_MASK) >> SWING_STYLE_SHIFT);
    if (sm.swing & SWING_CHARGE) != 0u {
        digit += 3.0;
    }
    var attack = 0.0;
    if sm.shot > 0.0 {
        let t = shot_s - sm.shot;
        attack = params.bow.w;
        if t < params.shot.x {
            attack = params.bow.w + 2.0 + t / params.shot.x;
        } else if t < params.shot.x + params.shot.y {
            attack = params.bow.w + 4.0 + (t - params.shot.x) / params.shot.y;
        }
    } else if ranged && (sm.atk > 0.0 || sm.bow > 0.0) {
        attack = params.bow.w + sm.atk;
    } else if sm.follow > 0.0 {
        attack = digit * 2.0 + params.attack.y + params.attack.z * (1.0 - sm.follow / follow_s);
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

    let position = mix(prev, pos, params.alpha);
    let jitter = lod_jitter(s.seed);
    let d = position - params.cam_pos;
    let d2 = dot(d, d) * jitter * jitter;
    let visible = !culled(position);
    // Detail level: jittered distance against this frame's thresholds,
    // held inside the hysteresis bounds. A soldier the camera does not see
    // keeps his held level for when he comes back, and casts by the plain
    // thresholds.
    var lod = level(params.bands[kind * 3u + 2u], d2);
    if visible {
        let fine = level(params.bands[kind * 3u], d2);
        let coarse = level(params.bands[kind * 3u + 1u], d2);
        sm.lod = clamp(sm.lod, fine, coarse);
        lod = sm.lod;
    }
    var casts = 0u;
    if lod < params.cast_lods {
        casts = cascade_mask(position);
    }
    if !visible && casts == 0u {
        smoothing[i] = sm;
        return;
    }

    var rgb = vec3<f32>(s.cr, s.cg, s.cb);
    if (reg.flags & REG_BROKEN) != 0u {
        let gray = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
        rgb = rgb * 0.55 + vec3<f32>(gray) * 0.45;
    }

    // Facing interpolates like position, wrap-aware.
    let dyr = s.yaw - s.yaw_prev + PI;
    let dy = dyr - floor(dyr / TAU) * TAU - PI;
    let yaw = s.yaw_prev + dy * params.alpha;

    // The attack, else the victory cheer, else nothing.
    var lunge = attack;
    if attack <= 0.0 && death_t == 0.0 && reg.celebrate >= 0.0 {
        lunge = params.consts.w + reg.celebrate;
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
        vec4<f32>(position, sm.bow),
        vec4<f32>(rgb, s.seed),
        vec4<f32>(yaw, sm.walk, lunge, fx),
        vec4<f32>(sm.band, sm.wall, sm.gait, stagger),
    );
    smoothing[i] = sm;
    if visible {
        append(kind * NUM_LODS + lod, i, lod);
        if death_t == 0.0 && (reg.flags & (REG_SELECTED | REG_HOVERED | REG_HOVER_OWN)) != 0u {
            append_ring(i, reg.flags, kind);
        }
    }
    append_casters(casts, kind, i);
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
// move) and an index append, then the caster lists as for the living.
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
    let jitter = lod_jitter(rec.color.a);
    let d = position - params.cam_pos;
    let d2 = dot(d, d) * jitter * jitter;
    let lod = level(params.bands[kind * 3u + 2u], d2);
    if !culled(position) {
        let bucket = kind * NUM_LODS + lod;
        atomicAdd(&counts[16u + bucket], 1u);
        append(bucket, ridx, lod);
    }
    if lod < params.cast_lods {
        append_casters(cascade_mask(position), kind, ridx);
    }
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

// One thread per list: the list count becomes the pulled draw's vertex
// count. `counts` was cleared before `build` ran.
@compute @workgroup_size(32)
fn finalize(@builtin(local_invocation_index) b: u32) {
    if b >= 16u {
        // A caster list: cascade * 4 + kind.
        let s = b - 16u;
        let count = atomicLoad(&counts[32u + s]);
        args[b] = DrawArgs(count * params.shadow_corners[s / 4u][s % 4u], 1u, 0u, 0u);
        readback[36u + s] = count;
        return;
    }
    let count = atomicLoad(&counts[b]);
    let fallen = atomicLoad(&counts[16u + b]);
    args[b] = DrawArgs(count * params.buckets[b].y, 1u, 0u, 0u);
    readback[b] = count;
    readback[16u + b] = fallen;
    if b == 0u {
        readback[32u] = params.frame;
        readback[33u] = params.n;
        // Six corners per ring (two triangles).
        args[RING_ARG] = DrawArgs(atomicLoad(&counts[RING_COUNTER]) * 6u, 1u, 0u, 0u);
    }
}
