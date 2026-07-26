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
//! y = -half_height (matching `TYPES[kind]`).

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;

/// Body part ids (uv.x). Keep in sync with unit_instancing.wgsl.
const PART_BODY: f32 = 0.0;
const PART_ARM: f32 = 1.0; // sword arm + sword: attack swing
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
/// Archer hood: muted Lincoln-green wool — the forester's color, and
/// no other kind wears cloth on the head.
const HOOD: [f32; 4] = [0.30, 0.34, 0.22, 0.0];
/// Archer tunic: forester green, lighter than the hood, with a bit of
/// team dye so the two armies' archers don't wear the exact same
/// cloth (M2TW Sherwood archers: all-green, hood to boots).
const TUNIC: [f32; 4] = [0.34, 0.40, 0.24, 0.25];
/// Archer hose: dark brown wool.
const HOSE: [f32; 4] = [0.36, 0.30, 0.22, 0.0];

struct MeshBuf {
    pos: Vec<[f32; 3]>,
    nrm: Vec<[f32; 3]>,
    uv: Vec<[f32; 2]>,
    col: Vec<[f32; 4]>,
    idx: Vec<u32>,
}

impl MeshBuf {
    fn new() -> Self {
        Self {
            pos: Vec::new(),
            nrm: Vec::new(),
            uv: Vec::new(),
            col: Vec::new(),
            idx: Vec::new(),
        }
    }

    /// Axis-aligned cuboid: 24 verts (4 per face, per-face normals),
    /// 12 tris. `part`/`pivot_y` ride in the UV channel for shader anim.
    fn cuboid(&mut self, center: Vec3, half: Vec3, part: f32, pivot_y: f32, col: [f32; 4]) {
        const FACES: [([f32; 3], [usize; 2]); 6] = [
            ([1.0, 0.0, 0.0], [1, 2]),  // +X, spanned by y,z
            ([-1.0, 0.0, 0.0], [1, 2]), // -X
            ([0.0, 1.0, 0.0], [0, 2]),  // +Y, spanned by x,z
            ([0.0, -1.0, 0.0], [0, 2]), // -Y
            ([0.0, 0.0, 1.0], [0, 1]),  // +Z, spanned by x,y
            ([0.0, 0.0, -1.0], [0, 1]), // -Z
        ];
        for (n, span) in FACES {
            let base = self.pos.len() as u32;
            let normal = Vec3::from_array(n);
            let face_center = center + normal * (half * normal.abs());
            let mut u_axis = Vec3::ZERO;
            let mut v_axis = Vec3::ZERO;
            u_axis[span[0]] = half[span[0]];
            v_axis[span[1]] = half[span[1]];
            for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let p = face_center + u_axis * su + v_axis * sv;
                self.pos.push(p.to_array());
                self.nrm.push(n);
                self.uv.push([part, pivot_y]);
                self.col.push(col);
            }
            // Winding so the face is CCW seen from outside: flip when the
            // (u, v) basis cross-product points against the face normal.
            let flip = u_axis.cross(v_axis).dot(normal) < 0.0;
            let quad = if flip {
                [0, 2, 1, 0, 3, 2]
            } else {
                [0, 1, 2, 0, 2, 3]
            };
            self.idx.extend(quad.map(|k| base + k));
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
    /// dark crossguard, bright blade with a tip block. All PART_ARM.
    fn sword(&mut self, hand: Vec3, blade_len: f32, scale: f32, pivot_y: f32) {
        let s = scale;
        self.cuboid(
            hand + Vec3::new(0.0, 0.0, -0.05 * s),
            Vec3::new(0.024, 0.024, 0.05) * s,
            PART_ARM,
            pivot_y,
            WOOD,
        );
        self.cuboid(
            hand + Vec3::new(0.0, 0.0, 0.02 * s),
            Vec3::new(0.10, 0.02, 0.018) * s,
            PART_ARM,
            pivot_y,
            DARK_STEEL,
        );
        self.cuboid(
            hand + Vec3::new(0.0, 0.0, 0.04 * s + blade_len / 2.0),
            Vec3::new(0.042 * s, 0.014 * s, blade_len / 2.0),
            PART_ARM,
            pivot_y,
            BLADE,
        );
        self.cuboid(
            hand + Vec3::new(0.0, 0.0, 0.04 * s + blade_len + 0.035 * s),
            Vec3::new(0.02, 0.014, 0.035) * s,
            PART_ARM,
            pivot_y,
            BLADE,
        );
    }
}

fn build(m: MeshBuf) -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, m.pos)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, m.nrm)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, m.uv)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, m.col)
    .with_inserted_indices(Indices::U32(m.idx))
}

/// Heavy knight: broad steel-and-tabard chest over narrow hips, pauldrons,
/// full helm with nose guard, tall team-colored kite shield, arming sword.
/// 17 cuboids, 204 tris. Height 1.1 m (half_height 0.55).
pub fn build_knight() -> Mesh {
    let mut m = MeshBuf::new();
    let hip_pivot = -0.18;
    let shoulder = 0.16;
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
    m.sword(Vec3::new(0.285, shoulder - 0.02, 0.28), 0.42, 1.2, shoulder);
    build(m)
}

/// Light man-at-arms: slim tunic, bare face under a steel kettle hat,
/// wooden buckler, shorter sword. 16 cuboids, 192 tris. Height 1.0 m
/// (half_height 0.50).
pub fn build_man_at_arms() -> Mesh {
    let mut m = MeshBuf::new();
    let hip_pivot = -0.16;
    let shoulder = 0.14;
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
    m.sword(Vec3::new(0.22, shoulder - 0.02, 0.235), 0.30, 1.0, shoulder);
    build(m)
}

/// Spear infantry: chainmail hauberk under a team surcoat, bare face in a
/// wide-brim steel kettle hat over a mail coif, round team shield, and a
/// tall spear carried VERTICAL (the shader levels it at the enemy and
/// thrusts it on the stab). 17 cuboids, 204 tris. Height 1.0 m
/// (half_height 0.50).
pub fn build_spearman() -> Mesh {
    let mut m = MeshBuf::new();
    let hip_pivot = -0.16;
    let shoulder = 0.14;
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
    // mail coif collar + bare face
    m.cuboid(
        Vec3::new(0.0, 0.215, 0.0),
        Vec3::new(0.115, 0.03, 0.115),
        PART_BODY,
        0.0,
        CHAIN,
    );
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
        PART_SPEAR_ARM,
        shoulder,
        WOOD,
    );
    m.cuboid(
        Vec3::new(0.24, 1.24, 0.10),
        Vec3::new(0.034, 0.09, 0.014),
        PART_SPEAR_ARM,
        shoulder,
        BLADE,
    );
    m.cuboid(
        Vec3::new(0.24, 1.335, 0.10),
        Vec3::new(0.016, 0.035, 0.010),
        PART_SPEAR_ARM,
        shoulder,
        BLADE,
    );
    m.cuboid(
        Vec3::new(0.24, -0.33, 0.10),
        Vec3::new(0.028, 0.028, 0.028),
        PART_SPEAR_ARM,
        shoulder,
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
/// bracer on the bow arm. 42 cuboids, ~500 tris. Height ~1.0 m
/// (half_height 0.50; the arrow fan overtops it like the spearman's
/// point).
pub fn build_archer() -> Mesh {
    let mut m = MeshBuf::new();
    let hip_pivot = -0.16;
    let shoulder = 0.14;
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
