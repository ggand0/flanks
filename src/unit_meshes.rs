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
/// Quilted gambeson: undyed padded linen with a hint of team dye.
const GAMBESON: [f32; 4] = [0.62, 0.52, 0.36, 0.15];
/// Quiver leather: darker than the bow wood.
const LEATHER: [f32; 4] = [0.30, 0.20, 0.12, 0.0];
/// Arrow fletching: pale goose feather.
const FLETCH: [f32; 4] = [0.88, 0.86, 0.78, 0.0];
/// Ranger cloak and hood: team color pulled toward black — reads as the
/// army's color in a darker, weathered cloth than the tabard.
const CLOAK: [f32; 4] = [0.05, 0.06, 0.08, 0.80];
/// The dark under the hood: the cowl swallows the upper face.
const COWL_SHADOW: [f32; 4] = [0.07, 0.055, 0.05, 0.0];
/// Bow stave: dark oiled yew, distinct from the pale arrow shafts.
const BOW_WOOD: [f32; 4] = [0.33, 0.21, 0.11, 0.0];
/// Bowstring: pale linen (light against the dark stave and cloak).
const STRING: [f32; 4] = [0.80, 0.76, 0.66, 0.0];

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

/// Archer: a ranger. Deep pointed hood swept back over the head (team
/// cloth pulled toward black, cowl shadow swallowing the upper face),
/// short shoulder-mantled cloak over a gambeson and team tabard, back
/// quiver with fletched shafts, and a tall strung longbow — grip, two
/// recurved limbs and nocks stepping forward, pale linen string tip to
/// tip. No steel anywhere: the hooded silhouette IS the archer read.
/// 32 cuboids, 384 tris. Height ~1.05 m to the hood point
/// (half_height 0.50).
pub fn build_archer() -> Mesh {
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
    // gambeson hem -> team tabard chest
    m.cuboid(
        Vec3::new(0.0, -0.08, 0.0),
        Vec3::new(0.14, 0.10, 0.095),
        PART_BODY,
        0.0,
        GAMBESON,
    );
    m.cuboid(
        Vec3::new(0.0, 0.09, 0.0),
        Vec3::new(0.165, 0.10, 0.105),
        PART_BODY,
        0.0,
        TEAM,
    );
    // the cloak: a back panel falling from the shoulders to the thighs
    // and a mantle draped over the shoulders
    m.cuboid(
        Vec3::new(0.0, -0.02, -0.128),
        Vec3::new(0.185, 0.27, 0.020),
        PART_BODY,
        0.0,
        CLOAK,
    );
    m.cuboid(
        Vec3::new(0.0, 0.203, -0.02),
        Vec3::new(0.205, 0.042, 0.125),
        PART_BODY,
        0.0,
        CLOAK,
    );
    // the face: chin and mouth in the light, the upper half lost in the
    // cowl's shadow under a jutting brim
    m.cuboid(
        Vec3::new(0.0, 0.295, 0.005),
        Vec3::new(0.088, 0.088, 0.088),
        PART_BODY,
        0.0,
        SKIN,
    );
    m.cuboid(
        Vec3::new(0.0, 0.345, 0.068),
        Vec3::new(0.086, 0.042, 0.028),
        PART_BODY,
        0.0,
        COWL_SHADOW,
    );
    m.cuboid(
        Vec3::new(0.0, 0.383, 0.085),
        Vec3::new(0.102, 0.02, 0.05),
        PART_BODY,
        0.0,
        CLOAK,
    );
    // hood shell: cheek panels and back
    m.cuboid(
        Vec3::new(-0.104, 0.305, -0.02),
        Vec3::new(0.017, 0.10, 0.10),
        PART_BODY,
        0.0,
        CLOAK,
    );
    m.cuboid(
        Vec3::new(0.104, 0.305, -0.02),
        Vec3::new(0.017, 0.10, 0.10),
        PART_BODY,
        0.0,
        CLOAK,
    );
    m.cuboid(
        Vec3::new(0.0, 0.305, -0.115),
        Vec3::new(0.104, 0.10, 0.018),
        PART_BODY,
        0.0,
        CLOAK,
    );
    // the point: crown tiers stepping up and SWEEPING BACK, ending in a
    // drooped tip — the ranger silhouette
    m.cuboid(
        Vec3::new(0.0, 0.418, -0.02),
        Vec3::new(0.104, 0.036, 0.112),
        PART_BODY,
        0.0,
        CLOAK,
    );
    m.cuboid(
        Vec3::new(0.0, 0.468, -0.065),
        Vec3::new(0.073, 0.030, 0.080),
        PART_BODY,
        0.0,
        CLOAK,
    );
    m.cuboid(
        Vec3::new(0.0, 0.503, -0.118),
        Vec3::new(0.047, 0.026, 0.055),
        PART_BODY,
        0.0,
        CLOAK,
    );
    m.cuboid(
        Vec3::new(0.0, 0.522, -0.172),
        Vec3::new(0.026, 0.020, 0.036),
        PART_BODY,
        0.0,
        CLOAK,
    );
    // back quiver over the right shoulder, clear of the cloak panel
    m.cuboid(
        Vec3::new(0.148, 0.05, -0.175),
        Vec3::new(0.052, 0.17, 0.045),
        PART_BODY,
        0.0,
        LEATHER,
    );
    m.cuboid(
        Vec3::new(0.13, 0.27, -0.18),
        Vec3::new(0.013, 0.06, 0.013),
        PART_BODY,
        0.0,
        WOOD,
    );
    m.cuboid(
        Vec3::new(0.168, 0.25, -0.165),
        Vec3::new(0.013, 0.06, 0.013),
        PART_BODY,
        0.0,
        WOOD,
    );
    m.cuboid(
        Vec3::new(0.13, 0.35, -0.18),
        Vec3::new(0.028, 0.04, 0.028),
        PART_BODY,
        0.0,
        FLETCH,
    );
    m.cuboid(
        Vec3::new(0.168, 0.32, -0.165),
        Vec3::new(0.024, 0.035, 0.024),
        PART_BODY,
        0.0,
        FLETCH,
    );
    // bow arm: gambeson sleeve, leather bracer, bare hand
    m.cuboid(
        Vec3::new(-0.21, shoulder - 0.02, 0.04),
        Vec3::new(0.055, 0.055, 0.075),
        PART_BOW_ARM,
        shoulder,
        GAMBESON,
    );
    m.cuboid(
        Vec3::new(-0.23, shoulder - 0.02, 0.115),
        Vec3::new(0.045, 0.045, 0.035),
        PART_BOW_ARM,
        shoulder,
        LEATHER,
    );
    m.cuboid(
        Vec3::new(-0.235, shoulder - 0.02, 0.155),
        Vec3::new(0.038, 0.038, 0.028),
        PART_BOW_ARM,
        shoulder,
        SKIN,
    );
    // the longbow, strung and carried vertical: thick grip, mid limbs,
    // recurved nocks — each tier stepping FORWARD to draw the curve —
    // and the pale string closing tip to tip behind it
    m.cuboid(
        Vec3::new(-0.24, 0.12, 0.16),
        Vec3::new(0.028, 0.105, 0.030),
        PART_BOW_ARM,
        shoulder,
        BOW_WOOD,
    );
    m.cuboid(
        Vec3::new(-0.24, 0.345, 0.181),
        Vec3::new(0.023, 0.125, 0.024),
        PART_BOW_ARM,
        shoulder,
        BOW_WOOD,
    );
    m.cuboid(
        Vec3::new(-0.24, -0.105, 0.181),
        Vec3::new(0.023, 0.125, 0.024),
        PART_BOW_ARM,
        shoulder,
        BOW_WOOD,
    );
    m.cuboid(
        Vec3::new(-0.24, 0.555, 0.211),
        Vec3::new(0.019, 0.095, 0.019),
        PART_BOW_ARM,
        shoulder,
        BOW_WOOD,
    );
    m.cuboid(
        Vec3::new(-0.24, -0.315, 0.211),
        Vec3::new(0.019, 0.095, 0.019),
        PART_BOW_ARM,
        shoulder,
        BOW_WOOD,
    );
    m.cuboid(
        Vec3::new(-0.24, 0.12, 0.242),
        Vec3::new(0.007, 0.53, 0.007),
        PART_BOW_ARM,
        shoulder,
        STRING,
    );
    // draw arm: gambeson sleeve + bare hand (no weapon — the stab-style
    // pull-back-and-snap is the string draw; melee is a scrappy bash)
    m.cuboid(
        Vec3::new(0.22, shoulder - 0.02, 0.045),
        Vec3::new(0.055, 0.055, 0.085),
        PART_ARM,
        shoulder,
        GAMBESON,
    );
    m.cuboid(
        Vec3::new(0.22, shoulder - 0.02, 0.13),
        Vec3::new(0.04, 0.04, 0.045),
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
