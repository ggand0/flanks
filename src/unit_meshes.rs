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
