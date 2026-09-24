//! Leg cycle timing, shared by both render paths.
//!
//! Each soldier carries a gait phase in cycles, one cycle being two
//! steps. It advances at `rate`, which is a plain function of his speed,
//! so the cadence rises and falls smoothly with it. The vertex shader
//! builds the pose from the phase and sizes the stance so a planted foot
//! travels back exactly as fast as the ground (unit_instancing.wgsl).
//!
//! Ordinary movement uses the same run as a charge. A walk mode with its
//! own pace comes later.
//!
//! `rate` is repeated in unit_build.wgsl and unit_instancing.wgsl
//! (`gait_rate`). Keep the three in sync.

use bevy::mesh::Mesh;

/// Cycles per second at rest, and what each m/s adds. A knight at his
/// 6 m/s march turns over about 3.9 steps a second.
const CADENCE_REST: f32 = 1.0;
const CADENCE_PER_MS: f32 = 0.16;

/// Cycles per second at this ground speed.
pub(crate) fn rate(speed: f32) -> f32 {
    CADENCE_REST + CADENCE_PER_MS * speed
}

/// Advance a phase and wrap it. The step is capped so a long frame
/// cannot jump a leg to a different part of its stride.
pub(crate) fn advance(phase: f32, rate: f32, dt: f32) -> f32 {
    let p = phase + (rate * dt).min(0.25);
    p - p.floor()
}

/// Hip pivot to sole in a mesh: the pivot the leg parts carry, less the
/// lowest point they reach. None when the mesh has no legs.
pub fn measure(mesh: &Mesh) -> Option<f32> {
    use bevy::mesh::VertexAttributeValues as V;
    let V::Float32x3(pos) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)? else {
        return None;
    };
    let V::Float32x2(uv) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)? else {
        return None;
    };
    // Part ids 2 and 3 are the legs (unit_meshes.rs PART_LEG_L, PART_LEG_R).
    let legs = || uv.iter().enumerate().filter(|(_, uv)| (1.5..3.5).contains(&uv[0]));
    let sole = legs().map(|(i, _)| pos[i][1]).fold(f32::MAX, f32::min);
    let pivot = legs().map(|(_, uv)| uv[1]).fold(f32::MIN, f32::max);
    (sole < f32::MAX && pivot > sole).then_some(pivot - sole)
}
