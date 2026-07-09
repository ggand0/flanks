//! Code-built low-poly unit meshes: merged cuboids, flat per-face normals,
//! WoD/Minecraft chunky silhouettes. One mesh per unit kind; per-instance
//! team color does the rest. Local convention: origin at mid-body, +Z is
//! forward (yaw 0), feet at y = -half_height (matching `TYPES[kind]`).

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;

struct MeshBuf {
    pos: Vec<[f32; 3]>,
    nrm: Vec<[f32; 3]>,
    uv: Vec<[f32; 2]>,
    idx: Vec<u32>,
}

impl MeshBuf {
    fn new() -> Self {
        Self {
            pos: Vec::new(),
            nrm: Vec::new(),
            uv: Vec::new(),
            idx: Vec::new(),
        }
    }

    /// Axis-aligned cuboid: 24 verts (4 per face, per-face normals), 12 tris.
    fn cuboid(&mut self, center: Vec3, half: Vec3) {
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
                self.uv.push([0.0, 0.0]);
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

    fn build(self) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.pos)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.nrm)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, self.uv)
        .with_inserted_indices(Indices::U32(self.idx))
    }
}

/// Heavy knight: wide armored torso, pauldrons, helmet, sword arm, shield
/// slab on the left. 9 cuboids, 108 tris. Height 1.1 m (half_height 0.55).
pub fn build_knight() -> Mesh {
    let mut m = MeshBuf::new();
    // legs
    m.cuboid(Vec3::new(0.14, -0.35, 0.0), Vec3::new(0.10, 0.20, 0.11));
    m.cuboid(Vec3::new(-0.14, -0.35, 0.0), Vec3::new(0.10, 0.20, 0.11));
    // torso (wide, armored)
    m.cuboid(Vec3::new(0.0, 0.02, 0.0), Vec3::new(0.30, 0.18, 0.16));
    // pauldrons
    m.cuboid(Vec3::new(0.34, 0.16, 0.0), Vec3::new(0.08, 0.07, 0.10));
    m.cuboid(Vec3::new(-0.34, 0.16, 0.0), Vec3::new(0.08, 0.07, 0.10));
    // head + helmet brim
    m.cuboid(Vec3::new(0.0, 0.36, 0.0), Vec3::new(0.11, 0.11, 0.11));
    m.cuboid(Vec3::new(0.0, 0.30, 0.0), Vec3::new(0.15, 0.02, 0.15));
    // sword arm thrust forward-right
    m.cuboid(Vec3::new(0.30, -0.02, 0.16), Vec3::new(0.07, 0.07, 0.18));
    // shield slab on the left flank
    m.cuboid(Vec3::new(-0.36, 0.02, 0.08), Vec3::new(0.03, 0.22, 0.16));
    m.build()
}

/// Light man-at-arms: slim body, vertical spear with a tip above the head.
/// 7 cuboids, 84 tris. Height 1.0 m (half_height 0.50).
pub fn build_man_at_arms() -> Mesh {
    let mut m = MeshBuf::new();
    // legs
    m.cuboid(Vec3::new(0.10, -0.30, 0.0), Vec3::new(0.07, 0.20, 0.08));
    m.cuboid(Vec3::new(-0.10, -0.30, 0.0), Vec3::new(0.07, 0.20, 0.08));
    // torso
    m.cuboid(Vec3::new(0.0, 0.05, 0.0), Vec3::new(0.18, 0.16, 0.11));
    // head
    m.cuboid(Vec3::new(0.0, 0.33, 0.0), Vec3::new(0.10, 0.10, 0.10));
    // spear arm
    m.cuboid(Vec3::new(0.20, 0.02, 0.08), Vec3::new(0.06, 0.06, 0.12));
    // spear shaft (vertical) + tip
    m.cuboid(Vec3::new(0.26, 0.15, 0.0), Vec3::new(0.025, 0.55, 0.025));
    m.cuboid(Vec3::new(0.26, 0.76, 0.0), Vec3::new(0.045, 0.06, 0.045));
    m.build()
}
