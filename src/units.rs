//! SoA storage for all units. Units are never individual Bevy entities.

use bevy::prelude::*;

/// Units per team. `FL_UNITS` overrides (e.g. FL_UNITS=50000 -> 100k
/// total). Default 100k/team = 200k total — the perf standard. At 500
/// spawn columns the formation depth caps out around 125k/team before
/// rows fall off the terrain edge.
pub fn units_per_team() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("FL_UNITS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100_000)
    })
}

/// All per-unit state, structure-of-arrays.
#[derive(Resource, Default)]
pub struct Units {
    pub pos: Vec<Vec3>,
    /// Position at the previous fixed tick; rendering lerps prev -> pos.
    pub pos_prev: Vec<Vec3>,
    pub vel: Vec<Vec3>,
    /// Per-unit max speed (small variation breaks lockstep patterns).
    pub speed: Vec<f32>,
    pub team: Vec<u8>,
    /// Unit type: index into `unit_types::TYPES` (also the render bucket).
    pub kind: Vec<u8>,
    /// Smoothed facing angle around Y (0 = +Z); sim-owned, render-consumed.
    pub yaw: Vec<f32>,
    /// Index into `Groups::list`.
    pub group: Vec<u32>,
    pub hp: Vec<f32>,
    /// Base render color (team color with per-unit variation baked in).
    /// Alpha carries a stable per-unit anim seed, not opacity.
    pub color: Vec<[f32; 4]>,
}

impl Units {
    pub fn len(&self) -> usize {
        self.pos.len()
    }
}

pub struct UnitsPlugin;

impl Plugin for UnitsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Units>()
            .add_systems(Startup, spawn_armies);
    }
}

/// Cheap deterministic hash -> [0, 1). Used for spawn jitter, color variation,
/// terrain noise.
pub fn hash01(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    (x >> 8) as f32 / 16_777_216.0
}

fn spawn_armies(mut units: ResMut<Units>, terrain: Res<crate::terrain::Terrain>) {
    const COLS: usize = 500;
    const SPACING: f32 = 1.4;
    const GAP: f32 = 60.0; // no-man's land between the two armies

    let team_colors: [Vec3; 2] = [
        Vec3::new(0.20, 0.45, 0.85), // blue
        Vec3::new(0.90, 0.40, 0.15), // orange
    ];

    let per_team = units_per_team();
    let n = per_team * 2;
    units.pos.reserve(n);
    units.team.reserve(n);
    units.color.reserve(n);

    let rows = per_team.div_ceil(COLS);
    let heavy_rows = (rows as f32 * heavy_frac()).round() as usize;
    for team in 0..2u8 {
        let dir = if team == 0 { -1.0 } else { 1.0 };
        for k in 0..per_team {
            let row = k / COLS;
            let col = k % COLS;
            let i = (team as usize * per_team + k) as u32;
            // Heavies hold the front rows (nearest the gap).
            let kind = if row < heavy_rows {
                crate::unit_types::KIND_HEAVY
            } else {
                crate::unit_types::KIND_LIGHT
            };
            let params = &crate::unit_types::TYPES[kind as usize];
            let jx = hash01(i.wrapping_mul(3) + 1) - 0.5;
            let jz = hash01(i.wrapping_mul(3) + 2) - 0.5;
            let x = (col as f32 - (COLS - 1) as f32 / 2.0) * SPACING + jx * 0.6;
            let z = dir * (GAP / 2.0 + row as f32 * SPACING) + jz * 0.6;
            let p = Vec3::new(x, terrain.height_at(x, z) + params.half_height, z);
            units.pos.push(p);
            units.pos_prev.push(p);
            units.vel.push(Vec3::ZERO);
            units
                .speed
                .push(params.speed * (0.9 + 0.2 * hash01(i.wrapping_mul(7) + 5)));
            units.team.push(team);
            units.kind.push(kind);
            // Face the enemy at spawn (0 = +Z).
            units.yaw.push(if team == 0 { 0.0 } else { std::f32::consts::PI });
            units.group.push(team as u32);
            units.hp.push(params.hp);

            // Per-unit tonal variation so a block of 50k doesn't read as a flat texture.
            let tone = 0.85 + 0.3 * hash01(i.wrapping_mul(3));
            let c = team_colors[team as usize] * tone;
            // Alpha = stable anim seed (walk-bob phase offset), not opacity.
            units.color.push([c.x, c.y, c.z, hash01(i.wrapping_mul(11) + 7)]);
        }
    }
}

/// Fraction of each army spawned as heavy infantry (front rows).
/// FL_HEAVY_FRAC overrides; the real per-regiment mix lands with regiments.
fn heavy_frac() -> f32 {
    std::env::var("FL_HEAVY_FRAC")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.4)
}
