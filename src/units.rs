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

/// Half the height of a unit cube; units sit on the ground at this Y.
pub const UNIT_HALF_HEIGHT: f32 = 0.45;

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
    /// Index into `Groups::list`.
    pub group: Vec<u32>,
    pub hp: Vec<f32>,
    /// Base render color (team color with per-unit variation baked in).
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

    for team in 0..2u8 {
        let dir = if team == 0 { -1.0 } else { 1.0 };
        for k in 0..per_team {
            let row = k / COLS;
            let col = k % COLS;
            let i = (team as usize * per_team + k) as u32;
            let jx = hash01(i.wrapping_mul(3) + 1) - 0.5;
            let jz = hash01(i.wrapping_mul(3) + 2) - 0.5;
            let x = (col as f32 - (COLS - 1) as f32 / 2.0) * SPACING + jx * 0.6;
            let z = dir * (GAP / 2.0 + row as f32 * SPACING) + jz * 0.6;
            let p = Vec3::new(x, terrain.height_at(x, z) + UNIT_HALF_HEIGHT, z);
            units.pos.push(p);
            units.pos_prev.push(p);
            units.vel.push(Vec3::ZERO);
            units.speed.push(9.0 * (0.9 + 0.2 * hash01(i.wrapping_mul(7) + 5)));
            units.team.push(team);
            units.group.push(team as u32);
            units.hp.push(100.0);

            // Per-unit tonal variation so a block of 50k doesn't read as a flat texture.
            let tone = 0.85 + 0.3 * hash01(i.wrapping_mul(3));
            let c = team_colors[team as usize] * tone;
            units.color.push([c.x, c.y, c.z, 1.0]);
        }
    }
}
