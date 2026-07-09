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
    /// Current melee target (unit index); `u32::MAX` = none. Only trusted
    /// within a swing cycle — validated through the grid and at hit time.
    pub target: Vec<u32>,
    /// Swing state: 0 = Ready, 1 = WindUp, 2 = Recover.
    pub swing: Vec<u8>,
    /// Ticks left in the current swing phase.
    pub swing_t: Vec<u8>,
    /// Hit-flash ticks remaining (render feedback).
    pub flash: Vec<u8>,
    /// 0 = alive. Set to DEATH_TICKS on death; corpse plays its anim and is
    /// swap-removed when it reaches 1.
    pub death_t: Vec<u8>,
}

/// Swing states (the `swing` column).
pub const SWING_READY: u8 = 0;
pub const SWING_WINDUP: u8 = 1;
pub const SWING_RECOVER: u8 = 2;

impl Units {
    pub fn len(&self) -> usize {
        self.pos.len()
    }
}

pub struct UnitsPlugin;

impl Plugin for UnitsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Units>()
            .add_systems(Startup, spawn_armies)
            .add_systems(Update, surround_test_log);
    }
}

/// FL_TEST_SURROUND bookkeeping: per-capita survival of the encircled
/// pocket (x < 0) vs the line fight (x > 0). Swing combat's acceptance
/// criterion is the pocket dying clearly faster.
fn surround_test_log(units: Res<Units>, time: Res<Time>, mut next: Local<f32>) {
    if std::env::var("FL_TEST_SURROUND").is_err() {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 2.0;
    let mut pocket = 0usize;
    let mut line = 0usize;
    for i in 0..units.len() {
        if units.team[i] == 0 && units.death_t[i] == 0 {
            if units.pos[i].x < 0.0 {
                pocket += 1;
            } else {
                line += 1;
            }
        }
    }
    info!("[surround-test] t={t:.0}s blue alive: pocket {pocket}/500, line {line}/500");
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

/// Append one fully-initialized unit to every SoA column.
#[allow(clippy::too_many_arguments)] // spawn-time plumbing, all scalars
pub fn push_unit(
    units: &mut Units,
    terrain: &crate::terrain::Terrain,
    seed: u32,
    team: u8,
    kind: u8,
    group: u32,
    x: f32,
    z: f32,
) {
    const TEAM_COLORS: [Vec3; 2] = [
        Vec3::new(0.20, 0.45, 0.85), // blue
        Vec3::new(0.90, 0.40, 0.15), // orange
    ];
    let params = &crate::unit_types::TYPES[kind as usize];
    let p = Vec3::new(x, terrain.height_at(x, z) + params.half_height, z);
    units.pos.push(p);
    units.pos_prev.push(p);
    units.vel.push(Vec3::ZERO);
    units
        .speed
        .push(params.speed * (0.9 + 0.2 * hash01(seed.wrapping_mul(7) + 5)));
    units.team.push(team);
    units.kind.push(kind);
    // Face the enemy at spawn (0 = +Z).
    units.yaw.push(if team == 0 { 0.0 } else { std::f32::consts::PI });
    units.group.push(group);
    units.hp.push(params.hp);

    // Per-unit tonal variation so a block of 50k doesn't read as a flat
    // texture; alpha = stable anim seed (walk-bob phase), not opacity.
    let tone = 0.85 + 0.3 * hash01(seed.wrapping_mul(3));
    let c = TEAM_COLORS[team as usize] * tone;
    units.color.push([c.x, c.y, c.z, hash01(seed.wrapping_mul(11) + 7)]);
    units.target.push(u32::MAX);
    units.swing.push(SWING_RECOVER);
    // Stagger first swings across a full cooldown: contact must not
    // produce a synchronized metronome wave.
    units
        .swing_t
        .push((params.cooldown_ticks as f32 * hash01(seed.wrapping_mul(13) + 9)) as u8);
    units.flash.push(0);
    units.death_t.push(0);
}

fn spawn_armies(
    mut units: ResMut<Units>,
    terrain: Res<crate::terrain::Terrain>,
    mut groups: ResMut<crate::orders::Groups>,
) {
    const COLS: usize = 500;
    const SPACING: f32 = 1.4;
    const GAP: f32 = 60.0; // no-man's land between the two armies

    if std::env::var("FL_TEST_SURROUND").is_ok() {
        spawn_surround_test(&mut units, &terrain, &mut groups);
        return;
    }

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
            let jx = hash01(i.wrapping_mul(3) + 1) - 0.5;
            let jz = hash01(i.wrapping_mul(3) + 2) - 0.5;
            let x = (col as f32 - (COLS - 1) as f32 / 2.0) * SPACING + jx * 0.6;
            let z = dir * (GAP / 2.0 + row as f32 * SPACING) + jz * 0.6;
            push_unit(&mut units, &terrain, i, team, kind, team as u32, x, z);
        }
    }
}

/// FL_TEST_SURROUND: two equal blue detachments of light infantry, one
/// fully encircled, one fighting line-against-line, both against the same
/// odds. The surrounded pocket must die per-capita faster — that is the
/// acceptance test for swing combat (the old gather model traded 1:1).
/// Region split for bookkeeping: pocket at x = -200, line at x = +200.
/// Orange groups get press orders (units advance only under orders).
fn spawn_surround_test(
    units: &mut Units,
    terrain: &crate::terrain::Terrain,
    groups: &mut crate::orders::Groups,
) {
    use crate::orders::GroupData;
    use crate::unit_types::KIND_LIGHT;
    let mut seed = 1u32;
    let mut disc = |units: &mut Units,
                    team: u8,
                    group: u32,
                    cx: f32,
                    cz: f32,
                    r0: f32,
                    r1: f32,
                    n: usize| {
        for _ in 0..n {
            seed = seed.wrapping_add(1);
            let a = hash01(seed.wrapping_mul(3) + 1) * std::f32::consts::TAU;
            let r = (r0 * r0 + (r1 * r1 - r0 * r0) * hash01(seed.wrapping_mul(3) + 2)).sqrt();
            let (x, z) = (cx + r * a.cos(), cz + r * a.sin());
            push_unit(units, terrain, seed, team, KIND_LIGHT, group, x, z);
        }
    };
    // Pocket: 500 blue in a disc, 2000 orange in the surrounding annulus.
    disc(units, 0, 0, -200.0, 0.0, 0.0, 13.0, 500);
    disc(units, 1, 1, -200.0, 0.0, 14.0, 27.0, 2000);
    // Line: 500 blue vs 2000 orange, meeting only along one front.
    let mut block =
        |units: &mut Units, team: u8, group: u32, cz: f32, depth: f32, n: usize| {
            for _ in 0..n {
                seed = seed.wrapping_add(1);
                let x = 200.0 + (hash01(seed.wrapping_mul(3) + 1) - 0.5) * 60.0;
                let z = cz + (hash01(seed.wrapping_mul(3) + 2) - 0.5) * depth;
                push_unit(units, terrain, seed, team, KIND_LIGHT, group, x, z);
            }
        };
    block(units, 0, 0, -6.0, 10.0, 500);
    block(units, 1, 2, 21.0, 40.0, 2000);

    // Group 0: all blue, stands fast. Groups 1/2: orange, ordered to press
    // into their blue opponents so melee stays joined as ranks thin.
    let mut pocket_press = GroupData::new(1, 2000);
    pocket_press.order = Some(Vec2::new(-200.0, 0.0));
    let mut line_press = GroupData::new(1, 2000);
    line_press.order = Some(Vec2::new(200.0, -6.0));
    groups.list = vec![GroupData::new(0, 1000), pocket_press, line_press];
}

/// Fraction of each army spawned as heavy infantry (front rows).
/// FL_HEAVY_FRAC overrides; the real per-regiment mix lands with regiments.
fn heavy_frac() -> f32 {
    std::env::var("FL_HEAVY_FRAC")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.4)
}
