//! Code-built low-poly unit meshes: merged cuboids, flat per-face normals,
//! chunky Thronefall / Kingdoms-and-Castles proportions (big head, compact
//! torso, readable weapon). One mesh per unit kind.
//!
//! Readability comes from PER-PART COLOR, not silhouette alone: each vertex
//! carries a color whose alpha says how much the per-instance TEAM color
//! blends in (a=1 pure team cloth, a=0 fixed material like skin or steel).
//! The UV channel carries animation data: uv.x = body part id (PART_*),
//! uv.y = the part's pivot height. The vertex shader rotates parts around
//! their pivot: legs swing with the walk cycle, the sword arm raises during
//! wind-up and chops on the strike.
//!
//! Local convention: origin at mid-body, +Z is forward (yaw 0), feet at
//! y = -half_height (matching `TYPES[kind]`). The soldier's left hand is
//! on +X and holds the shield. The builders below are written as seen
//! from the front, shield on -X, and `build_kind_lods` mirrors every
//! level into place.
//!
//! Every kind builds at `NUM_LODS` detail levels. L0 is the full mesh.
//! L1 merges and drops what is under about a pixel in its band. L2 is
//! six blocks: torso, head, two legs, weapon, shield. L3 is a body block
//! and a head block. The main masses keep their size and position at
//! every level, surviving parts keep their part id and pivot (so poses
//! match across a switch), and merged blocks take the area-weighted
//! material of what they replace (`blend`), so a regiment keeps its hue.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;

use crate::render_units::NUM_LODS;

/// Body part ids (uv.x). Keep in sync with unit_instancing.wgsl.
const PART_BODY: f32 = 0.0;
const PART_ARM: f32 = 1.0; // sword arm, or the archer's draw arm
const PART_LEG_L: f32 = 2.0;
const PART_LEG_R: f32 = 3.0;
/// Spear arm: shaft modeled VERTICAL; the shader levels it at the enemy
/// (battle stance / spearwall) and thrusts it on the stab.
const PART_SPEAR_ARM: f32 = 4.0;
/// Shield arm + shield: static normally, raised/fronted in shieldwall.
const PART_SHIELD: f32 = 5.0;
/// Bow arm + bow (stave modeled VERTICAL in the left hand): the shader
/// tilts arm and bow up to the loft angle during the draw and settles
/// them on the loose. The right (draw) hand is plain PART_ARM — the
/// stab style's pull-back-then-snap reads as the string draw.
const PART_BOW_ARM: f32 = 6.0;
/// Arrow projectile (arrows.rs buckets, not a body part): rigid, with
/// flight pitch riding the anim2.z instance channel.
const PART_ARROW: f32 = 7.0;
/// The weapon in the weapon hand, sword or spear. Its pivot is the grip,
/// and the shader turns it there (`Rig`).
const PART_WEAPON: f32 = 8.0;

/// How a weapon hand holds (`Rig::hold`).
pub const HOLD_SWORD: f32 = 1.0;
pub const HOLD_SPEAR: f32 = 2.0;
/// The archer's draw hand: no weapon, it pulls the string.
pub const HOLD_DRAW: f32 = 3.0;

/// Samples in each attack table of a jointed arm (`Rig::windup`).
pub const ATTACK_SAMPLES: usize = 33;

/// The kind's skeleton for the vertex shader (`Rig` in
/// unit_instancing.wgsl): the leg length the gait is built on, and the
/// weapon arm's joints in the soldier's pitch plane, as (y, z) of local
/// space in the rest pose.
///
/// A bent arm (`chain` 0) is one mesh the shader bends at the elbow, and
/// it turns the held weapon at the grip. A jointed arm (`chain` 1) is
/// upper arm, forearm and hand as rigid parts, posed by the attack tables.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Rig {
    pub shoulder: [f32; 2],
    pub elbow: [f32; 2],
    pub wrist: [f32; 2],
    pub grip: [f32; 2],
    /// Unit direction from the grip toward the weapon's point.
    pub tip: [f32; 2],
    /// How far the weapon reaches behind the grip, to butt or pommel.
    pub rear: f32,
    /// The weapon arm's part id, 0 when the arm has no rig.
    pub arm: f32,
    /// `HOLD_SWORD`, `HOLD_SPEAR` or `HOLD_DRAW`.
    pub hold: f32,
    /// How far the weapon slides through the hand once levelled.
    pub slide: f32,
    /// Hip to sole (gait.rs `measure`).
    pub leg: f32,
    /// 1 for a jointed arm.
    pub chain: f32,
    /// A jointed arm's attack, sampled evenly over the wind-up and over
    /// the follow-through: shoulder, elbow and wrist turns from the rest
    /// pose, and how far the weapon is levelled. Entry 0 of the wind-up
    /// is the guard. The rest pose is the carry, all zero.
    pub windup: [[f32; 4]; ATTACK_SAMPLES],
    pub recover: [[f32; 4]; ATTACK_SAMPLES],
}

impl Default for Rig {
    fn default() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

impl Rig {
    /// The same rig on a mesh scaled by `s` about the local origin.
    pub fn scaled(mut self, s: f32) -> Self {
        for v in [&mut self.shoulder, &mut self.elbow, &mut self.wrist, &mut self.grip] {
            v[0] *= s;
            v[1] *= s;
        }
        self.rear *= s;
        self.slide *= s;
        self.leg *= s;
        self
    }
}

/// The code-built kinds' weapon arms, read off their builders below. Each
/// arm points forward from the shoulder with the hand at its end. The
/// elbow sits a little below the straight line, so the rest pose is a
/// bent arm the shader's arm solver reproduces exactly.
pub fn code_rig(kind: usize) -> Rig {
    let rig = |shoulder: f32, hand: [f32; 2], tip: [f32; 2], rear, arm, hold| Rig {
        shoulder: [shoulder, 0.0],
        elbow: [0.5 * (shoulder + hand[0]) - 0.03, 0.45 * hand[1]],
        wrist: hand,
        grip: hand,
        tip,
        rear,
        arm,
        hold,
        ..Rig::default()
    };
    let forward = [0.0, 1.0];
    let up = [1.0, 0.0];
    let r = match kind as u8 {
        crate::unit_types::KIND_HEAVY => rig(0.16, [0.14, 0.28], forward, 0.12, PART_ARM, HOLD_SWORD),
        crate::unit_types::KIND_LIGHT => rig(0.14, [0.12, 0.235], forward, 0.10, PART_ARM, HOLD_SWORD),
        crate::unit_types::KIND_SPEAR => rig(0.14, [0.12, 0.10], up, 0.478, PART_SPEAR_ARM, HOLD_SPEAR),
        _ => rig(0.14, [0.12, 0.15], forward, 0.0, PART_ARM, HOLD_DRAW),
    };
    r.scaled(crate::unit_types::unit_scale())
}

/// Part palette: rgb = material color, a = team-color blend amount.
/// Team color must stay DOMINANT (Thronefall rule): steel is darker than
/// the cloth and team-tinted so armies read at every zoom.
const TEAM: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const SKIN: [f32; 4] = [0.94, 0.73, 0.55, 0.0];
const STEEL: [f32; 4] = [0.52, 0.55, 0.62, 0.35];
const DARK_STEEL: [f32; 4] = [0.36, 0.38, 0.45, 0.20];
const BLADE: [f32; 4] = [0.88, 0.91, 0.97, 0.0];
const WOOD: [f32; 4] = [0.44, 0.30, 0.18, 0.0];
const PANTS: [f32; 4] = [0.50, 0.46, 0.42, 0.30];
/// Chainmail: duller than plate, a little team dye in the rings.
const CHAIN: [f32; 4] = [0.46, 0.48, 0.53, 0.22];
/// Quiver leather: darker than the bow wood.
const LEATHER: [f32; 4] = [0.30, 0.20, 0.12, 0.0];
/// Arrow fletching: pale goose feather.
const FLETCH: [f32; 4] = [0.88, 0.86, 0.78, 0.0];
/// Bow stave: rich warm yew.
const BOW_WOOD: [f32; 4] = [0.46, 0.29, 0.13, 0.0];
/// Bowstring: pale flax. The string is what makes a bow read as a bow
/// (without it the stave is just a stick), so it stays BRIGHT and thin.
const STRING: [f32; 4] = [0.74, 0.70, 0.60, 0.0];
/// Archer hood: takes the per-team CLOTH instance color (units.rs
/// ARCHER_CLOTH — Lincoln green for blue, russet for orange), darkened
/// a touch by the base so hood and tunic stay two tones of one dye.
/// No other kind wears cloth on the head.
const HOOD: [f32; 4] = [0.20, 0.22, 0.15, 0.80];
/// Archer tunic: same per-team cloth, lighter than the hood.
const TUNIC: [f32; 4] = [0.32, 0.34, 0.24, 0.85];
/// Archer hose: dark brown wool.
const HOSE: [f32; 4] = [0.36, 0.30, 0.22, 0.0];

/// FL_MESH_TESS=n: render perf probe. Splits every cuboid face into an
/// n x n grid: same silhouette, tris x n^2. Prices denser authored
/// meshes on the real pipeline before any asset exists.
fn mesh_tess() -> usize {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| crate::util::env_or("FL_MESH_TESS", 1_usize).clamp(1, 16))
}

/// FL_LOD_WEAPON=f: feel probe. Thickens the L2 weapon block by `f` so
/// blades and shafts stay readable at mid zoom, the way distant M2TW
/// sprites exaggerate them. Default 1 matches the full mesh exactly.
fn weapon_fat() -> f32 {
    static F: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *F.get_or_init(|| crate::util::env_or("FL_LOD_WEAPON", 1.0_f32).clamp(1.0, 4.0))
}

/// Area-weighted material for a far-level block that replaces several
/// surfaces. The shader shows `mix(rgb, team, a)`: the team amount
/// averages directly, and rgb averages by its VISIBLE share (1 - a).
/// Weights are areas as the battle camera sees them, looking down at
/// about 50 degrees: top faces count for more than fronts, and legs
/// are half hidden under the torso.
pub(crate) fn blend(parts: &[([f32; 4], f32)]) -> [f32; 4] {
    let total: f32 = parts.iter().map(|(_, w)| w).sum();
    let mut out = [0.0; 4];
    for (c, w) in parts {
        let w = w / total;
        out[3] += w * c[3];
        for k in 0..3 {
            out[k] += w * c[k] * (1.0 - c[3]);
        }
    }
    let visible = (1.0 - out[3]).max(1e-4);
    for c in out.iter_mut().take(3) {
        *c /= visible;
    }
    out
}

pub(crate) struct MeshBuf {
    /// Face grid size: the FL_MESH_TESS probe on L0, 1 everywhere else.
    tess: usize,
    pos: Vec<[f32; 3]>,
    nrm: Vec<[f32; 3]>,
    uv: Vec<[f32; 2]>,
    col: Vec<[f32; 4]>,
    idx: Vec<u32>,
}

impl MeshBuf {
    /// Buffer for a soldier mesh at detail level `lod`. A dense L0 over
    /// plain far levels is the shape of an authored mesh set: the
    /// FL_MESH_TESS probe prices exactly that.
    fn for_level(lod: usize) -> Self {
        Self {
            tess: if lod == 0 { mesh_tess() } else { 1 },
            ..Self::new()
        }
    }

    pub(crate) fn new() -> Self {
        Self {
            tess: 1,
            pos: Vec::new(),
            nrm: Vec::new(),
            uv: Vec::new(),
            col: Vec::new(),
            idx: Vec::new(),
        }
    }

    /// Axis-aligned cuboid: 24 verts (4 per face, per-face normals),
    /// 12 tris. `part`/`pivot_y` ride in the UV channel for shader anim.
    pub(crate) fn cuboid(
        &mut self,
        center: Vec3,
        half: Vec3,
        part: f32,
        pivot_y: f32,
        col: [f32; 4],
    ) {
        const FACES: [([f32; 3], [usize; 2]); 6] = [
            ([1.0, 0.0, 0.0], [1, 2]),  // +X, spanned by y,z
            ([-1.0, 0.0, 0.0], [1, 2]), // -X
            ([0.0, 1.0, 0.0], [0, 2]),  // +Y, spanned by x,z
            ([0.0, -1.0, 0.0], [0, 2]), // -Y
            ([0.0, 0.0, 1.0], [0, 1]),  // +Z, spanned by x,y
            ([0.0, 0.0, -1.0], [0, 1]), // -Z
        ];
        // Each face is a t x t grid of quads (t = 1 outside the probe).
        let t = self.tess;
        let stride = t as u32 + 1;
        for (n, span) in FACES {
            let base = self.pos.len() as u32;
            let normal = Vec3::from_array(n);
            let face_center = center + normal * (half * normal.abs());
            let mut u_axis = Vec3::ZERO;
            let mut v_axis = Vec3::ZERO;
            u_axis[span[0]] = half[span[0]];
            v_axis[span[1]] = half[span[1]];
            for j in 0..=t {
                for i in 0..=t {
                    let su = -1.0 + 2.0 * i as f32 / t as f32;
                    let sv = -1.0 + 2.0 * j as f32 / t as f32;
                    let p = face_center + u_axis * su + v_axis * sv;
                    self.pos.push(p.to_array());
                    self.nrm.push(n);
                    self.uv.push([part, pivot_y]);
                    self.col.push(col);
                }
            }
            // Winding so the face is CCW seen from outside: flip when the
            // (u, v) basis cross-product points against the face normal.
            let flip = u_axis.cross(v_axis).dot(normal) < 0.0;
            for j in 0..t as u32 {
                for i in 0..t as u32 {
                    let a = base + j * stride + i;
                    let (b, c, d) = (a + 1, a + stride + 1, a + stride);
                    let quad = if flip { [a, c, b, a, d, c] } else { [a, b, c, a, c, d] };
                    self.idx.extend(quad);
                }
            }
        }
    }

    /// Tapered N-gon frustum on the Y axis, flat-shaded per side, with a
    /// top cap when `r1` > 0 (bottom is left open — it sits inside the
    /// head/body). For helmet cones: a real taper instead of the
    /// stacked-box ziggurat. Benched with the archer's cervelliere
    /// (the bare cube face under it read weird) — kept for future
    /// helmeted kinds.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn frustum_y(
        &mut self,
        center: Vec3,
        r0: f32,
        r1: f32,
        h: f32,
        sides: usize,
        part: f32,
        pivot_y: f32,
        col: [f32; 4],
    ) {
        let n = sides.max(3) as f32;
        let slope = (r0 - r1) / h;
        let inv = 1.0 / (1.0 + slope * slope).sqrt();
        for i in 0..(n as usize) {
            let a1 = std::f32::consts::TAU * i as f32 / n;
            let a2 = std::f32::consts::TAU * (i + 1) as f32 / n;
            let am = (a1 + a2) * 0.5;
            let (sm, cm) = am.sin_cos();
            let nrm = [cm * inv, slope * inv, sm * inv];
            let b1 = center + Vec3::new(a1.cos() * r0, -h * 0.5, a1.sin() * r0);
            let b2 = center + Vec3::new(a2.cos() * r0, -h * 0.5, a2.sin() * r0);
            let t1 = center + Vec3::new(a1.cos() * r1, h * 0.5, a1.sin() * r1);
            let t2 = center + Vec3::new(a2.cos() * r1, h * 0.5, a2.sin() * r1);
            let base = self.pos.len() as u32;
            for p in [b1, t1, t2, b2] {
                self.pos.push(p.to_array());
                self.nrm.push(nrm);
                self.uv.push([part, pivot_y]);
                self.col.push(col);
            }
            self.idx
                .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
            if r1 > 0.0 {
                let top = center + Vec3::Y * (h * 0.5);
                let cbase = self.pos.len() as u32;
                for p in [top, t2, t1] {
                    self.pos.push(p.to_array());
                    self.nrm.push([0.0, 1.0, 0.0]);
                    self.uv.push([part, pivot_y]);
                    self.col.push(col);
                }
                self.idx.extend([cbase, cbase + 1, cbase + 2]);
            }
        }
    }

    /// A sword pointing forward (+Z) from the hand at `hand`: wood grip,
    /// dark crossguard, bright blade with a tip block. PART_WEAPON, with
    /// the grip at the hand.
    fn sword(&mut self, hand: Vec3, blade_len: f32, scale: f32) {
        let pivot_y = hand.y;
        let s = scale;
        self.cuboid(
            hand + Vec3::new(0.0, 0.0, -0.05 * s),
            Vec3::new(0.024, 0.024, 0.05) * s,
            PART_WEAPON,
            pivot_y,
            WOOD,
        );
        self.cuboid(
            hand + Vec3::new(0.0, 0.0, 0.02 * s),
            Vec3::new(0.10, 0.02, 0.018) * s,
            PART_WEAPON,
            pivot_y,
            DARK_STEEL,
        );
        self.cuboid(
            hand + Vec3::new(0.0, 0.0, 0.04 * s + blade_len / 2.0),
            Vec3::new(0.042 * s, 0.014 * s, blade_len / 2.0),
            PART_WEAPON,
            pivot_y,
            BLADE,
        );
        self.cuboid(
            hand + Vec3::new(0.0, 0.0, 0.04 * s + blade_len + 0.035 * s),
            Vec3::new(0.02, 0.014, 0.035) * s,
            PART_WEAPON,
            pivot_y,
            BLADE,
        );
    }
}

impl MeshBuf {
    /// Far-level sword: blade and tip as ONE block on the sword arm, no
    /// grip or crossguard. Same reach as `sword`, so the swing reads the
    /// same. `fat` thickens the two thin sides (`weapon_fat`).
    fn blade(&mut self, hand: Vec3, blade_len: f32, scale: f32, fat: f32) {
        let pivot_y = hand.y;
        let s = scale;
        let len = blade_len + 0.07 * s;
        self.cuboid(
            hand + Vec3::new(0.0, 0.0, 0.04 * s + len / 2.0),
            Vec3::new(0.042 * s * fat, 0.014 * s * fat, len / 2.0),
            PART_WEAPON,
            pivot_y,
            BLADE,
        );
    }
}

pub(crate) fn build(m: MeshBuf) -> Mesh {
    // No texture here. The atlas UV channel is still present, so every
    // unit mesh has the same vertex layout (unit_glb.rs fills it).
    let atlas_uv = vec![[0.0f32; 2]; m.pos.len()];
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, atlas_uv)
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, m.pos)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, m.nrm)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, m.uv)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, m.col)
    .with_inserted_indices(Indices::U32(m.idx))
}

/// Heavy knight: broad steel-and-tabard chest over narrow hips, pauldrons,
/// full helm with nose guard, tall team-colored kite shield, arming sword.
/// 17 cuboids, 204 tris at L0 (then 120 / 72 / 24). Height 1.1 m
/// (half_height 0.55).
pub fn build_knight(lod: usize) -> Mesh {
    let mut m = MeshBuf::for_level(lod);
    let hip_pivot = -0.18;
    let shoulder = 0.16;
    // Far levels: head block and flared crown as one steel helm.
    let helm = (Vec3::new(0.0, 0.355, 0.0), Vec3::new(0.115, 0.125, 0.115));
    let shield = (Vec3::new(-0.345, -0.02, 0.09), Vec3::new(0.03, 0.26, 0.17));
    let hand = Vec3::new(0.285, shoulder - 0.02, 0.28);
    if lod >= 3 {
        // L3: legs, tabard, pauldrons and shield as one body block.
        m.cuboid(
            Vec3::new(0.0, -0.155, 0.0),
            Vec3::new(0.214, 0.395, 0.12),
            PART_BODY,
            0.0,
            blend(&[(DARK_STEEL, 0.21), (TEAM, 0.61), (STEEL, 0.18)]),
        );
        m.cuboid(helm.0, helm.1, PART_BODY, 0.0, STEEL);
        return build(m);
    }
    // armored legs (walk-swing parts)
    m.cuboid(
        Vec3::new(-0.115, -0.38, 0.0),
        Vec3::new(0.085, 0.17, 0.10),
        PART_LEG_L,
        hip_pivot,
        DARK_STEEL,
    );
    m.cuboid(
        Vec3::new(0.115, -0.38, 0.0),
        Vec3::new(0.085, 0.17, 0.10),
        PART_LEG_R,
        hip_pivot,
        DARK_STEEL,
    );
    if lod == 2 {
        // L2: hips, chest, pauldrons and arms as one torso.
        m.cuboid(
            Vec3::new(0.0, 0.0125, 0.0),
            Vec3::new(0.25, 0.2275, 0.135),
            PART_BODY,
            0.0,
            blend(&[(TEAM, 0.77), (STEEL, 0.23)]),
        );
        m.cuboid(helm.0, helm.1, PART_BODY, 0.0, STEEL);
        m.cuboid(shield.0, shield.1, PART_SHIELD, shoulder, TEAM);
        m.blade(hand, 0.42, 1.2, weapon_fat());
        return build(m);
    }
    // hips (team tabard) -> broad chest (team tabard over armor)
    m.cuboid(
        Vec3::new(0.0, -0.10, 0.0),
        Vec3::new(0.185, 0.115, 0.12),
        PART_BODY,
        0.0,
        TEAM,
    );
    m.cuboid(
        Vec3::new(0.0, 0.09, 0.0),
        Vec3::new(0.24, 0.115, 0.15),
        PART_BODY,
        0.0,
        TEAM,
    );
    // steel pauldrons capping the shoulders
    m.cuboid(
        Vec3::new(-0.285, 0.185, 0.0),
        Vec3::new(0.075, 0.055, 0.10),
        PART_BODY,
        0.0,
        STEEL,
    );
    m.cuboid(
        Vec3::new(0.285, 0.185, 0.0),
        Vec3::new(0.075, 0.055, 0.10),
        PART_BODY,
        0.0,
        STEEL,
    );
    if lod == 1 {
        // L1: one-block helm, shield without its arm stub, sleeve and
        // vambrace as one arm, plain blade.
        m.cuboid(helm.0, helm.1, PART_BODY, 0.0, STEEL);
        m.cuboid(shield.0, shield.1, PART_SHIELD, shoulder, TEAM);
        m.cuboid(
            Vec3::new(0.285, shoulder - 0.02, 0.105),
            Vec3::new(0.06, 0.06, 0.155),
            PART_ARM,
            shoulder,
            blend(&[(TEAM, 0.6), (DARK_STEEL, 0.4)]),
        );
        m.blade(hand, 0.42, 1.2, 1.0);
        return build(m);
    }
    // full steel helm: head block + flared crown + nose guard
    m.cuboid(
        Vec3::new(0.0, 0.335, 0.0),
        Vec3::new(0.105, 0.105, 0.105),
        PART_BODY,
        0.0,
        STEEL,
    );
    m.cuboid(
        Vec3::new(0.0, 0.445, 0.0),
        Vec3::new(0.125, 0.035, 0.125),
        PART_BODY,
        0.0,
        STEEL,
    );
    m.cuboid(
        Vec3::new(0.0, 0.33, 0.112),
        Vec3::new(0.032, 0.075, 0.014),
        PART_BODY,
        0.0,
        DARK_STEEL,
    );
    // steel shield arm stub + tall team kite shield on the left flank
    m.cuboid(
        Vec3::new(-0.29, 0.05, 0.03),
        Vec3::new(0.06, 0.06, 0.09),
        PART_SHIELD,
        shoulder,
        STEEL,
    );
    m.cuboid(
        Vec3::new(-0.345, -0.02, 0.09),
        Vec3::new(0.03, 0.26, 0.17),
        PART_SHIELD,
        shoulder,
        TEAM,
    );
    // sword arm: tabard-sleeved shoulder + steel vambrace, then the sword
    m.cuboid(
        Vec3::new(0.285, shoulder - 0.02, 0.05),
        Vec3::new(0.065, 0.065, 0.10),
        PART_ARM,
        shoulder,
        TEAM,
    );
    m.cuboid(
        Vec3::new(0.285, shoulder - 0.02, 0.19),
        Vec3::new(0.05, 0.05, 0.07),
        PART_ARM,
        shoulder,
        DARK_STEEL,
    );
    m.sword(Vec3::new(0.285, shoulder - 0.02, 0.28), 0.42, 1.2);
    build(m)
}

/// Light man-at-arms: slim tunic, bare face under a steel kettle hat,
/// wooden buckler, shorter sword. 15 cuboids, 180 tris at L0 (then
/// 120 / 72 / 24). Height 1.0 m (half_height 0.50).
pub fn build_man_at_arms(lod: usize) -> Mesh {
    let mut m = MeshBuf::for_level(lod);
    let hip_pivot = -0.16;
    let shoulder = 0.14;
    // Far levels: bare face and kettle hat as one head block.
    let head = (Vec3::new(0.0, 0.3325, 0.0), Vec3::new(0.11, 0.1275, 0.11));
    let head_col = blend(&[(SKIN, 0.25), (STEEL, 0.75)]);
    let buckler = (Vec3::new(-0.265, 0.04, 0.10), Vec3::new(0.022, 0.11, 0.11));
    let hand = Vec3::new(0.22, shoulder - 0.02, 0.235);
    if lod >= 3 {
        m.cuboid(
            Vec3::new(0.0, -0.155, 0.0),
            Vec3::new(0.152, 0.345, 0.095),
            PART_BODY,
            0.0,
            blend(&[(PANTS, 0.38), (TEAM, 0.57), (WOOD, 0.05)]),
        );
        m.cuboid(head.0, head.1, PART_BODY, 0.0, head_col);
        return build(m);
    }
    // cloth legs
    m.cuboid(
        Vec3::new(-0.09, -0.345, 0.0),
        Vec3::new(0.068, 0.155, 0.082),
        PART_LEG_L,
        hip_pivot,
        PANTS,
    );
    m.cuboid(
        Vec3::new(0.09, -0.345, 0.0),
        Vec3::new(0.068, 0.155, 0.082),
        PART_LEG_R,
        hip_pivot,
        PANTS,
    );
    if lod == 2 {
        m.cuboid(
            Vec3::new(0.0, 0.005, 0.0),
            Vec3::new(0.18, 0.185, 0.10),
            PART_BODY,
            0.0,
            TEAM,
        );
        m.cuboid(head.0, head.1, PART_BODY, 0.0, head_col);
        m.cuboid(buckler.0, buckler.1, PART_SHIELD, shoulder, WOOD);
        m.blade(hand, 0.30, 1.0, weapon_fat());
        return build(m);
    }
    // hips -> tunic chest (team cloth, slimmer than the knight)
    m.cuboid(
        Vec3::new(0.0, -0.08, 0.0),
        Vec3::new(0.14, 0.10, 0.095),
        PART_BODY,
        0.0,
        TEAM,
    );
    m.cuboid(
        Vec3::new(0.0, 0.09, 0.0),
        Vec3::new(0.175, 0.10, 0.11),
        PART_BODY,
        0.0,
        TEAM,
    );
    // bare face + steel kettle-hat brim and crown
    m.cuboid(
        Vec3::new(0.0, 0.30, 0.0),
        Vec3::new(0.095, 0.095, 0.095),
        PART_BODY,
        0.0,
        SKIN,
    );
    m.cuboid(
        Vec3::new(0.0, 0.395, 0.0),
        Vec3::new(0.15, 0.02, 0.15),
        PART_BODY,
        0.0,
        STEEL,
    );
    m.cuboid(
        Vec3::new(0.0, 0.435, 0.0),
        Vec3::new(0.075, 0.025, 0.075),
        PART_BODY,
        0.0,
        STEEL,
    );
    if lod == 1 {
        // L1 keeps the face and the kettle hat (the kind's tell from
        // above): buckler without its stub, sleeve and hand as one arm.
        m.cuboid(buckler.0, buckler.1, PART_SHIELD, shoulder, WOOD);
        m.cuboid(
            Vec3::new(0.22, shoulder - 0.02, 0.09),
            Vec3::new(0.05, 0.05, 0.13),
            PART_ARM,
            shoulder,
            blend(&[(TEAM, 0.65), (SKIN, 0.35)]),
        );
        m.blade(hand, 0.30, 1.0, 1.0);
        return build(m);
    }
    // buckler arm stub (cloth sleeve) + small wooden buckler
    m.cuboid(
        Vec3::new(-0.225, 0.04, 0.03),
        Vec3::new(0.05, 0.05, 0.08),
        PART_SHIELD,
        shoulder,
        TEAM,
    );
    m.cuboid(
        Vec3::new(-0.265, 0.04, 0.10),
        Vec3::new(0.022, 0.11, 0.11),
        PART_SHIELD,
        shoulder,
        WOOD,
    );
    // cloth sword arm + shorter sword
    m.cuboid(
        Vec3::new(0.22, shoulder - 0.02, 0.045),
        Vec3::new(0.055, 0.055, 0.085),
        PART_ARM,
        shoulder,
        TEAM,
    );
    m.cuboid(
        Vec3::new(0.22, shoulder - 0.02, 0.16),
        Vec3::new(0.042, 0.042, 0.06),
        PART_ARM,
        shoulder,
        SKIN,
    );
    m.sword(Vec3::new(0.22, shoulder - 0.02, 0.235), 0.30, 1.0);
    build(m)
}

/// Spear infantry: chainmail hauberk under a team surcoat, bare face in a
/// wide-brim steel kettle hat over a mail coif, round team shield, and a
/// tall spear carried VERTICAL (the shader levels it at the enemy and
/// thrusts it on the stab). 17 cuboids, 204 tris at L0 (then 132 / 72 /
/// 24). Height 1.0 m (half_height 0.50).
pub fn build_spearman(lod: usize) -> Mesh {
    let mut m = MeshBuf::for_level(lod);
    let hip_pivot = -0.16;
    let shoulder = 0.14;
    // Far levels: coif, face and kettle hat as one head block.
    let head = (Vec3::new(0.0, 0.327, 0.0), Vec3::new(0.115, 0.14, 0.115));
    let head_col = blend(&[(SKIN, 0.2), (STEEL, 0.72), (CHAIN, 0.08)]);
    let shield = (Vec3::new(-0.27, 0.04, 0.09), Vec3::new(0.022, 0.13, 0.13));
    if lod >= 3 {
        m.cuboid(
            Vec3::new(0.0, -0.155, 0.0),
            Vec3::new(0.157, 0.345, 0.10),
            PART_BODY,
            0.0,
            blend(&[(PANTS, 0.34), (CHAIN, 0.24), (TEAM, 0.42)]),
        );
        m.cuboid(head.0, head.1, PART_BODY, 0.0, head_col);
        return build(m);
    }
    // cloth legs
    m.cuboid(
        Vec3::new(-0.09, -0.345, 0.0),
        Vec3::new(0.068, 0.155, 0.082),
        PART_LEG_L,
        hip_pivot,
        PANTS,
    );
    m.cuboid(
        Vec3::new(0.09, -0.345, 0.0),
        Vec3::new(0.068, 0.155, 0.082),
        PART_LEG_R,
        hip_pivot,
        PANTS,
    );
    if lod == 2 {
        m.cuboid(
            Vec3::new(0.0, 0.005, 0.0),
            Vec3::new(0.175, 0.185, 0.105),
            PART_BODY,
            0.0,
            blend(&[(CHAIN, 0.45), (TEAM, 0.55)]),
        );
        m.cuboid(head.0, head.1, PART_BODY, 0.0, head_col);
        m.cuboid(shield.0, shield.1, PART_SHIELD, shoulder, TEAM);
        // Shaft, blade and ferrule as one upright block: a leveled line
        // of these still reads as a brace at mid zoom.
        let fat = weapon_fat();
        m.cuboid(
            Vec3::new(0.24, 0.506, 0.10),
            Vec3::new(0.024 * fat, 0.864, 0.024 * fat),
            PART_WEAPON,
            shoulder - 0.02,
            blend(&[(WOOD, 0.87), (BLADE, 0.13)]),
        );
        return build(m);
    }
    // chainmail hauberk hem -> team surcoat chest
    m.cuboid(
        Vec3::new(0.0, -0.08, 0.0),
        Vec3::new(0.15, 0.10, 0.10),
        PART_BODY,
        0.0,
        CHAIN,
    );
    m.cuboid(
        Vec3::new(0.0, 0.09, 0.0),
        Vec3::new(0.18, 0.10, 0.115),
        PART_BODY,
        0.0,
        TEAM,
    );
    // mail coif collar (L0 only) + bare face
    if lod == 0 {
        m.cuboid(
            Vec3::new(0.0, 0.215, 0.0),
            Vec3::new(0.115, 0.03, 0.115),
            PART_BODY,
            0.0,
            CHAIN,
        );
    }
    m.cuboid(
        Vec3::new(0.0, 0.30, 0.0),
        Vec3::new(0.095, 0.095, 0.095),
        PART_BODY,
        0.0,
        SKIN,
    );
    // kettle hat: wide brim + shallow crown
    m.cuboid(
        Vec3::new(0.0, 0.395, 0.0),
        Vec3::new(0.165, 0.02, 0.165),
        PART_BODY,
        0.0,
        STEEL,
    );
    m.cuboid(
        Vec3::new(0.0, 0.44, 0.0),
        Vec3::new(0.085, 0.028, 0.085),
        PART_BODY,
        0.0,
        STEEL,
    );
    if lod == 1 {
        // L1 keeps face and kettle hat: shield without stub and boss,
        // sleeve and hand as one arm, blade and tip as one spearhead.
        m.cuboid(shield.0, shield.1, PART_SHIELD, shoulder, TEAM);
        m.cuboid(
            Vec3::new(0.228, shoulder - 0.02, 0.06),
            Vec3::new(0.055, 0.055, 0.095),
            PART_SPEAR_ARM,
            shoulder,
            blend(&[(CHAIN, 0.7), (SKIN, 0.3)]),
        );
        m.cuboid(
            Vec3::new(0.24, 0.40, 0.10),
            Vec3::new(0.024, 0.75, 0.024),
            PART_WEAPON,
            shoulder - 0.02,
            WOOD,
        );
        m.cuboid(
            Vec3::new(0.24, 1.26, 0.10),
            Vec3::new(0.03, 0.11, 0.013),
            PART_WEAPON,
            shoulder - 0.02,
            BLADE,
        );
        return build(m);
    }
    // shield arm (mail sleeve) + round team shield with a steel boss
    m.cuboid(
        Vec3::new(-0.225, 0.04, 0.03),
        Vec3::new(0.05, 0.05, 0.08),
        PART_SHIELD,
        shoulder,
        CHAIN,
    );
    m.cuboid(
        Vec3::new(-0.27, 0.04, 0.09),
        Vec3::new(0.022, 0.13, 0.13),
        PART_SHIELD,
        shoulder,
        TEAM,
    );
    m.cuboid(
        Vec3::new(-0.295, 0.04, 0.09),
        Vec3::new(0.012, 0.05, 0.05),
        PART_SHIELD,
        shoulder,
        STEEL,
    );
    // spear arm: mail shoulder + bare hand gripping the shaft
    m.cuboid(
        Vec3::new(0.22, shoulder - 0.02, 0.045),
        Vec3::new(0.055, 0.055, 0.085),
        PART_SPEAR_ARM,
        shoulder,
        CHAIN,
    );
    m.cuboid(
        Vec3::new(0.24, shoulder - 0.02, 0.10),
        Vec3::new(0.042, 0.042, 0.05),
        PART_SPEAR_ARM,
        shoulder,
        SKIN,
    );
    // the spear, upright: long ash shaft, leaf blade, butt ferrule
    m.cuboid(
        Vec3::new(0.24, 0.40, 0.10),
        Vec3::new(0.024, 0.75, 0.024),
        PART_WEAPON,
        shoulder - 0.02,
        WOOD,
    );
    m.cuboid(
        Vec3::new(0.24, 1.24, 0.10),
        Vec3::new(0.034, 0.09, 0.014),
        PART_WEAPON,
        shoulder - 0.02,
        BLADE,
    );
    m.cuboid(
        Vec3::new(0.24, 1.335, 0.10),
        Vec3::new(0.016, 0.035, 0.010),
        PART_WEAPON,
        shoulder - 0.02,
        BLADE,
    );
    m.cuboid(
        Vec3::new(0.24, -0.33, 0.10),
        Vec3::new(0.028, 0.028, 0.028),
        PART_WEAPON,
        shoulder - 0.02,
        DARK_STEEL,
    );
    build(m)
}

/// Archer: a levy longbowman. The three long-distance tells, in order:
/// the BOW (a strung longbow with the stave/string gap opened SIDEWAYS
/// — a gap in depth projects onto one line from front and behind,
/// which is what made the old bow read as a stick), the arrow FAN
/// poking above the head from the back quiver, and the HOOD. The hood
/// is historically right for a bowman (longbowmen wore wool hoods or
/// coifs, not helms) and it solves the cube-head problem by WRAPPING
/// the head — crown, back panel, cheek flaps, chin wrap, shoulder
/// mantle, liripipe tail — so the head reads as cloth with a face
/// opening, never as a cube with a hat on. Body after the M2TW
/// Sherwood archers: all-green forester — plain hip-length tunic
/// (green with a whisper of team dye), long sleeves, leather belt,
/// brown hose, low shoes, back quiver with the fletching fan, leather
/// bracer on the bow arm. 42 cuboids, ~500 tris at L0 (then 168 / 72 /
/// 24). Height ~1.0 m (half_height 0.50; the arrow fan overtops it like
/// the spearman's point). The far levels keep the three tells as long
/// as each is more than a speck: bow and string to L1, stave and fan to
/// L2, the hood color to L3.
pub fn build_archer(lod: usize) -> Mesh {
    let mut m = MeshBuf::for_level(lod);
    let hip_pivot = -0.16;
    let shoulder = 0.14;
    if lod >= 1 {
        // Far levels: face and wrapped hood as one head block.
        let head = (Vec3::new(0.0, 0.305, -0.01), Vec3::new(0.11, 0.113, 0.10));
        let head_col = blend(&[(HOOD, 0.8), (SKIN, 0.2)]);
        let stave = (Vec3::new(-0.262, 0.11, 0.16), Vec3::new(0.02, 0.475, 0.02));
        let fan = (Vec3::new(0.155, 0.47, -0.15), Vec3::new(0.062, 0.075, 0.03));
        if lod >= 3 {
            m.cuboid(
                Vec3::new(0.0, -0.145, 0.0),
                Vec3::new(0.159, 0.365, 0.10),
                PART_BODY,
                0.0,
                blend(&[(HOSE, 0.33), (LEATHER, 0.07), (TUNIC, 0.60)]),
            );
            m.cuboid(head.0, head.1, PART_BODY, 0.0, head_col);
            return build(m);
        }
        // hose and shoe as one leg
        let leg_col = blend(&[(HOSE, 0.8), (LEATHER, 0.2)]);
        for (x, part) in [(-0.09, PART_LEG_L), (0.09, PART_LEG_R)] {
            m.cuboid(
                Vec3::new(x, -0.335, 0.004),
                Vec3::new(0.067, 0.175, 0.085),
                part,
                hip_pivot,
                leg_col,
            );
        }
        if lod == 2 {
            m.cuboid(
                Vec3::new(0.0, 0.011, 0.0),
                Vec3::new(0.175, 0.209, 0.105),
                PART_BODY,
                0.0,
                blend(&[(TUNIC, 0.8), (HOOD, 0.2)]),
            );
            m.cuboid(head.0, head.1, PART_BODY, 0.0, head_col);
            let fat = weapon_fat();
            m.cuboid(
                stave.0,
                Vec3::new(stave.1.x * fat, stave.1.y, stave.1.z * fat),
                PART_BOW_ARM,
                shoulder,
                BOW_WOOD,
            );
            m.cuboid(fan.0, fan.1, PART_BODY, 0.0, FLETCH);
            return build(m);
        }
        // L1: tunic chest, skirt with its hem, the hood's shoulder
        // mantle as one slab
        m.cuboid(
            Vec3::new(0.0, 0.09, 0.0),
            Vec3::new(0.165, 0.10, 0.105),
            PART_BODY,
            0.0,
            TUNIC,
        );
        m.cuboid(
            Vec3::new(0.0, -0.1065, 0.0),
            Vec3::new(0.157, 0.0915, 0.106),
            PART_BODY,
            0.0,
            blend(&[(TUNIC, 0.85), (HOOD, 0.15)]),
        );
        m.cuboid(
            Vec3::new(0.0, 0.19, -0.013),
            Vec3::new(0.24, 0.03, 0.11),
            PART_BODY,
            0.0,
            HOOD,
        );
        // open face, the hood as one block set back behind it
        m.cuboid(
            Vec3::new(0.0, 0.30, 0.005),
            Vec3::new(0.09, 0.09, 0.09),
            PART_BODY,
            0.0,
            SKIN,
        );
        m.cuboid(
            Vec3::new(0.0, 0.305, -0.02),
            Vec3::new(0.112, 0.113, 0.095),
            PART_BODY,
            0.0,
            HOOD,
        );
        // quiver, the three shafts as one slat, the fletching fan
        m.cuboid(
            Vec3::new(0.155, 0.03, -0.14),
            Vec3::new(0.045, 0.15, 0.045),
            PART_BODY,
            0.0,
            LEATHER,
        );
        m.cuboid(
            Vec3::new(0.155, 0.30, -0.15),
            Vec3::new(0.045, 0.12, 0.011),
            PART_BODY,
            0.0,
            WOOD,
        );
        m.cuboid(fan.0, fan.1, PART_BODY, 0.0, FLETCH);
        // bow arm as one block, straight stave, the bright string
        m.cuboid(
            Vec3::new(-0.222, shoulder - 0.015, 0.0865),
            Vec3::new(0.05, 0.046, 0.1015),
            PART_BOW_ARM,
            shoulder,
            blend(&[(TUNIC, 0.6), (LEATHER, 0.2), (SKIN, 0.2)]),
        );
        m.cuboid(stave.0, stave.1, PART_BOW_ARM, shoulder, BOW_WOOD);
        m.cuboid(
            Vec3::new(-0.335, 0.11, 0.16),
            Vec3::new(0.007, 0.46, 0.007),
            PART_BOW_ARM,
            shoulder,
            STRING,
        );
        // draw arm as one block
        m.cuboid(
            Vec3::new(0.22, shoulder - 0.015, 0.085),
            Vec3::new(0.05, 0.046, 0.105),
            PART_ARM,
            shoulder,
            blend(&[(TUNIC, 0.75), (SKIN, 0.25)]),
        );
        return build(m);
    }
    // brown hose and low leather shoes
    m.cuboid(
        Vec3::new(-0.09, -0.30, 0.0),
        Vec3::new(0.066, 0.14, 0.08),
        PART_LEG_L,
        hip_pivot,
        HOSE,
    );
    m.cuboid(
        Vec3::new(0.09, -0.30, 0.0),
        Vec3::new(0.066, 0.14, 0.08),
        PART_LEG_R,
        hip_pivot,
        HOSE,
    );
    m.cuboid(
        Vec3::new(-0.09, -0.475, 0.012),
        Vec3::new(0.068, 0.035, 0.095),
        PART_LEG_L,
        hip_pivot,
        LEATHER,
    );
    m.cuboid(
        Vec3::new(0.09, -0.475, 0.012),
        Vec3::new(0.068, 0.035, 0.095),
        PART_LEG_R,
        hip_pivot,
        LEATHER,
    );
    // plain hip-length green tunic (the Sherwood cut): chest, skirt,
    // darker hem band, leather belt
    m.cuboid(
        Vec3::new(0.0, 0.09, 0.0),
        Vec3::new(0.165, 0.10, 0.105),
        PART_BODY,
        0.0,
        TUNIC,
    );
    m.cuboid(
        Vec3::new(0.0, -0.10, 0.0),
        Vec3::new(0.155, 0.085, 0.105),
        PART_BODY,
        0.0,
        TUNIC,
    );
    m.cuboid(
        Vec3::new(0.0, -0.18, 0.0),
        Vec3::new(0.158, 0.018, 0.108),
        PART_BODY,
        0.0,
        HOOD,
    );
    m.cuboid(
        Vec3::new(0.0, -0.005, 0.0),
        Vec3::new(0.152, 0.024, 0.103),
        PART_BODY,
        0.0,
        LEATHER,
    );
    // the hood's shoulder cape covers the arm joints (one garment with
    // the hood — bare team rolls next to the green read as a mismatch)
    m.cuboid(
        Vec3::new(-0.185, 0.185, -0.01),
        Vec3::new(0.055, 0.032, 0.095),
        PART_BODY,
        0.0,
        HOOD,
    );
    m.cuboid(
        Vec3::new(0.185, 0.185, -0.01),
        Vec3::new(0.055, 0.032, 0.095),
        PART_BODY,
        0.0,
        HOOD,
    );
    // open face, then the hood WRAPPED around it: crown slab overhanging
    // the brow, back panel falling to the neck, cheek flaps framing the
    // face opening, chin wrap, shoulder mantle, liripipe tail
    m.cuboid(
        Vec3::new(0.0, 0.30, 0.005),
        Vec3::new(0.09, 0.09, 0.09),
        PART_BODY,
        0.0,
        SKIN,
    );
    m.cuboid(
        Vec3::new(0.0, 0.385, -0.005),
        Vec3::new(0.105, 0.030, 0.105),
        PART_BODY,
        0.0,
        HOOD,
    );
    m.cuboid(
        Vec3::new(0.0, 0.295, -0.095),
        Vec3::new(0.105, 0.120, 0.022),
        PART_BODY,
        0.0,
        HOOD,
    );
    m.cuboid(
        Vec3::new(-0.095, 0.295, -0.015),
        Vec3::new(0.022, 0.100, 0.085),
        PART_BODY,
        0.0,
        HOOD,
    );
    m.cuboid(
        Vec3::new(0.095, 0.295, -0.015),
        Vec3::new(0.022, 0.100, 0.085),
        PART_BODY,
        0.0,
        HOOD,
    );
    m.cuboid(
        Vec3::new(0.0, 0.21, 0.04),
        Vec3::new(0.07, 0.02, 0.04),
        PART_BODY,
        0.0,
        HOOD,
    );
    m.cuboid(
        Vec3::new(0.0, 0.195, -0.015),
        Vec3::new(0.155, 0.026, 0.125),
        PART_BODY,
        0.0,
        HOOD,
    );
    m.cuboid(
        Vec3::new(0.03, 0.13, -0.115),
        Vec3::new(0.016, 0.065, 0.016),
        PART_BODY,
        0.0,
        HOOD,
    );
    // back quiver over the right shoulder, three shafts fanned up with
    // pale fletchings cresting the shoulder line — the archer tell
    m.cuboid(
        Vec3::new(0.155, 0.03, -0.14),
        Vec3::new(0.045, 0.15, 0.045),
        PART_BODY,
        0.0,
        LEATHER,
    );
    m.cuboid(
        Vec3::new(0.12, 0.30, -0.15),
        Vec3::new(0.011, 0.14, 0.011),
        PART_BODY,
        0.0,
        WOOD,
    );
    m.cuboid(
        Vec3::new(0.155, 0.33, -0.155),
        Vec3::new(0.011, 0.17, 0.011),
        PART_BODY,
        0.0,
        WOOD,
    );
    m.cuboid(
        Vec3::new(0.19, 0.30, -0.145),
        Vec3::new(0.011, 0.14, 0.011),
        PART_BODY,
        0.0,
        WOOD,
    );
    m.cuboid(
        Vec3::new(0.12, 0.445, -0.15),
        Vec3::new(0.026, 0.045, 0.026),
        PART_BODY,
        0.0,
        FLETCH,
    );
    m.cuboid(
        Vec3::new(0.155, 0.505, -0.155),
        Vec3::new(0.028, 0.05, 0.028),
        PART_BODY,
        0.0,
        FLETCH,
    );
    m.cuboid(
        Vec3::new(0.19, 0.445, -0.145),
        Vec3::new(0.026, 0.045, 0.026),
        PART_BODY,
        0.0,
        FLETCH,
    );
    // bow arm: long tunic sleeve to the wrist, leather bracer, hand
    m.cuboid(
        Vec3::new(-0.21, shoulder - 0.01, 0.04),
        Vec3::new(0.055, 0.05, 0.055),
        PART_BOW_ARM,
        shoulder,
        TUNIC,
    );
    m.cuboid(
        Vec3::new(-0.225, shoulder - 0.02, 0.10),
        Vec3::new(0.045, 0.042, 0.035),
        PART_BOW_ARM,
        shoulder,
        TUNIC,
    );
    m.cuboid(
        Vec3::new(-0.23, shoulder - 0.02, 0.135),
        Vec3::new(0.046, 0.045, 0.025),
        PART_BOW_ARM,
        shoulder,
        LEATHER,
    );
    m.cuboid(
        Vec3::new(-0.235, shoulder - 0.02, 0.16),
        Vec3::new(0.038, 0.038, 0.028),
        PART_BOW_ARM,
        shoulder,
        SKIN,
    );
    // the longbow, strung and carried vertical in the left hand: grip,
    // limbs stepping OUTWARD as they rise and fall, nocks, and a bright
    // string tip to tip. The stave/string gap is ~10 cm of X so the
    // two lines stay separated from front and behind — that gap, not
    // the stave, is what reads as "bow".
    m.cuboid(
        Vec3::new(-0.24, 0.11, 0.16),
        Vec3::new(0.028, 0.09, 0.026),
        PART_BOW_ARM,
        shoulder,
        BOW_WOOD,
    );
    m.cuboid(
        Vec3::new(-0.26, 0.30, 0.16),
        Vec3::new(0.022, 0.12, 0.022),
        PART_BOW_ARM,
        shoulder,
        BOW_WOOD,
    );
    m.cuboid(
        Vec3::new(-0.26, -0.08, 0.16),
        Vec3::new(0.022, 0.12, 0.022),
        PART_BOW_ARM,
        shoulder,
        BOW_WOOD,
    );
    m.cuboid(
        Vec3::new(-0.285, 0.49, 0.16),
        Vec3::new(0.016, 0.09, 0.018),
        PART_BOW_ARM,
        shoulder,
        BOW_WOOD,
    );
    m.cuboid(
        Vec3::new(-0.285, -0.27, 0.16),
        Vec3::new(0.016, 0.09, 0.018),
        PART_BOW_ARM,
        shoulder,
        BOW_WOOD,
    );
    m.cuboid(
        Vec3::new(-0.305, 0.565, 0.16),
        Vec3::new(0.012, 0.02, 0.014),
        PART_BOW_ARM,
        shoulder,
        BOW_WOOD,
    );
    m.cuboid(
        Vec3::new(-0.305, -0.345, 0.16),
        Vec3::new(0.012, 0.02, 0.014),
        PART_BOW_ARM,
        shoulder,
        BOW_WOOD,
    );
    m.cuboid(
        Vec3::new(-0.335, 0.11, 0.16),
        Vec3::new(0.007, 0.46, 0.007),
        PART_BOW_ARM,
        shoulder,
        STRING,
    );
    // draw arm: long tunic sleeve + hand (no weapon — the stab-style
    // pull-back-and-snap is the string draw; melee is a scrappy bash)
    m.cuboid(
        Vec3::new(0.22, shoulder - 0.01, 0.035),
        Vec3::new(0.055, 0.05, 0.055),
        PART_ARM,
        shoulder,
        TUNIC,
    );
    m.cuboid(
        Vec3::new(0.22, shoulder - 0.02, 0.10),
        Vec3::new(0.045, 0.042, 0.04),
        PART_ARM,
        shoulder,
        TUNIC,
    );
    m.cuboid(
        Vec3::new(0.22, shoulder - 0.02, 0.15),
        Vec3::new(0.04, 0.04, 0.04),
        PART_ARM,
        shoulder,
        SKIN,
    );
    build(m)
}

/// The builders draw the soldier as seen from the front, with his shield
/// on -X. Mirroring puts it in his left hand, where the sim counts it.
fn mirror_x(mesh: &mut Mesh) {
    use bevy::mesh::VertexAttributeValues as V;
    for attribute in [Mesh::ATTRIBUTE_POSITION, Mesh::ATTRIBUTE_NORMAL] {
        if let Some(V::Float32x3(values)) = mesh.attribute_mut(attribute) {
            for v in values.iter_mut() {
                v[0] = -v[0];
            }
        }
    }
    // A mirror turns every triangle inside out, and back faces are culled.
    if let Some(Indices::U32(idx)) = mesh.indices_mut() {
        for t in idx.chunks_mut(3) {
            t.swap(1, 2);
        }
    }
}

/// These meshes are written at the kind's own half height, so the
/// display scale (unit_types::unit_scale) has to be applied to them.
/// An imported mesh is built to the scaled half height already.
fn apply_scale(mesh: &mut Mesh, scale: f32) {
    use bevy::mesh::VertexAttributeValues as V;
    if let Some(V::Float32x3(pos)) = mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION) {
        for p in pos.iter_mut() {
            for c in p.iter_mut() {
                *c *= scale;
            }
        }
    }
    // uv.y is the part's pivot height, which lives in the same space.
    if let Some(V::Float32x2(uv)) = mesh.attribute_mut(Mesh::ATTRIBUTE_UV_0) {
        for uv in uv.iter_mut() {
            uv[1] *= scale;
        }
    }
}

/// All detail levels of one unit kind, L0 first.
pub fn build_kind_lods(kind: usize) -> [Mesh; NUM_LODS] {
    let builder: fn(usize) -> Mesh = match kind as u8 {
        crate::unit_types::KIND_HEAVY => build_knight,
        crate::unit_types::KIND_LIGHT => build_man_at_arms,
        crate::unit_types::KIND_SPEAR => build_spearman,
        _ => build_archer,
    };
    let scale = crate::unit_types::unit_scale();
    std::array::from_fn(|lod| {
        let mut mesh = builder(lod);
        mirror_x(&mut mesh);
        if (scale - 1.0).abs() > 1e-4 {
            apply_scale(&mut mesh, scale);
        }
        mesh
    })
}

/// Arrow projectile: shaft + head + fletching along +Z (flight
/// direction), origin at the shaft center. 3 cuboids, 36 tris. Sized up
/// slightly from a true arrow so a volley reads at gameplay zoom.
pub fn build_arrow() -> Mesh {
    let mut m = MeshBuf::new();
    m.cuboid(
        Vec3::ZERO,
        Vec3::new(0.014, 0.014, 0.36),
        PART_ARROW,
        0.0,
        WOOD,
    );
    m.cuboid(
        Vec3::new(0.0, 0.0, 0.385),
        Vec3::new(0.022, 0.022, 0.035),
        PART_ARROW,
        0.0,
        BLADE,
    );
    m.cuboid(
        Vec3::new(0.0, 0.0, -0.31),
        Vec3::new(0.03, 0.03, 0.06),
        PART_ARROW,
        0.0,
        FLETCH,
    );
    build(m)
}
