//! SoA storage for all units. Units are never individual Bevy entities.

use bevy::prelude::*;

/// Units per team. `FL_UNITS` overrides (e.g. FL_UNITS=50000 -> 100k
/// total). Default 100k/team = 200k total — the perf standard. At 500
/// spawn columns the formation depth caps out around 125k/team before
/// rows fall off the terrain edge.
pub fn units_per_team() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| crate::util::env_or("FL_UNITS", 100_000))
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
    /// Facing at the previous fixed tick; rendering lerps yaw_prev -> yaw.
    pub yaw_prev: Vec<f32>,
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
    /// Offset from the regiment anchor: a regiment order is the SAME point
    /// for the whole block, rigidly translated per unit by this offset.
    /// Captured at spawn (loose); rigid formations will write slot offsets.
    pub home: Vec<Vec2>,
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

/// Append one fully-initialized unit to every SoA column. `anchor` is the
/// unit's regiment anchor; the spawn offset from it becomes `home`.
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
    anchor: Vec2,
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
    let spawn_yaw = if team == 0 { 0.0 } else { std::f32::consts::PI };
    units.yaw.push(spawn_yaw);
    units.yaw_prev.push(spawn_yaw);
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
    units.home.push(Vec2::new(x, z) - anchor);
}

/// FL_TEST_SURROUND: two equal blue detachments of light infantry, one
/// fully encircled, one fighting line-against-line, both against the same
/// odds. The surrounded pocket must die per-capita faster — that is the
/// acceptance test for swing combat (the old gather model traded 1:1).
/// Region split for bookkeeping: pocket at x = -200, line at x = +200.
/// Orange groups get press orders (units advance only under orders). A
/// block order translates a formation (it never converges one), so the
/// encirclement is four arc regiments each ordered inward past the pocket.
pub fn spawn_surround_test(
    units: &mut Units,
    terrain: &crate::terrain::Terrain,
    groups: &mut crate::orders::Groups,
) {
    use crate::orders::GroupData;
    use crate::unit_types::KIND_LIGHT;
    let pocket_c = Vec2::new(-200.0, 0.0);
    let mut seed = 1u32;
    let mut list: Vec<GroupData> = Vec::new();

    // Blue pocket (group 0): disc, holds.
    list.push(GroupData::new(0, KIND_LIGHT, pocket_c, 500));
    for _ in 0..500 {
        seed = seed.wrapping_add(1);
        let a = hash01(seed.wrapping_mul(3) + 1) * std::f32::consts::TAU;
        let r = 13.0 * hash01(seed.wrapping_mul(3) + 2).sqrt();
        let (x, z) = (pocket_c.x + r * a.cos(), pocket_c.y + r * a.sin());
        push_unit(units, terrain, seed, 0, KIND_LIGHT, 0, x, z, pocket_c);
    }
    // Orange annulus: four arc regiments (groups 1-4), each pressing its
    // anchor to the pocket center.
    for q in 0..4u32 {
        let a0 = q as f32 * std::f32::consts::FRAC_PI_2;
        let mid = a0 + std::f32::consts::FRAC_PI_4;
        let anchor = pocket_c + Vec2::new(mid.cos(), mid.sin()) * 20.5;
        let g = list.len() as u32;
        let mut gd = GroupData::new(1, KIND_LIGHT, anchor, 500);
        gd.order = Some(pocket_c);
        list.push(gd);
        for _ in 0..500 {
            seed = seed.wrapping_add(1);
            let a = a0 + hash01(seed.wrapping_mul(3) + 1) * std::f32::consts::FRAC_PI_2;
            let rr = 14.0f32 * 14.0
                + (27.0f32 * 27.0 - 14.0 * 14.0) * hash01(seed.wrapping_mul(3) + 2);
            let r = rr.sqrt();
            let (x, z) = (pocket_c.x + r * a.cos(), pocket_c.y + r * a.sin());
            push_unit(units, terrain, seed, 1, KIND_LIGHT, g, x, z, anchor);
        }
    }
    // Line fight: blue block (group 5) holds; orange block (group 6)
    // presses 27 m south through it.
    let blue_anchor = Vec2::new(200.0, -6.0);
    let orange_anchor = Vec2::new(200.0, 21.0);
    let block = |units: &mut Units,
                 seed: &mut u32,
                     team: u8,
                     group: u32,
                     anchor: Vec2,
                     depth: f32,
                     n: usize| {
        for _ in 0..n {
            *seed = seed.wrapping_add(1);
            let x = 200.0 + (hash01(seed.wrapping_mul(3) + 1) - 0.5) * 60.0;
            let z = anchor.y + (hash01(seed.wrapping_mul(3) + 2) - 0.5) * depth;
            push_unit(units, terrain, *seed, team, KIND_LIGHT, group, x, z, anchor);
        }
    };
    list.push(GroupData::new(0, KIND_LIGHT, blue_anchor, 500));
    block(units, &mut seed, 0, 5, blue_anchor, 10.0, 500);
    let mut press = GroupData::new(1, KIND_LIGHT, orange_anchor, 2000);
    // Press to CONTACT, not through: the fight must stay frontal (the
    // pocket is the surrounded case; this is the control).
    press.order = Some(Vec2::new(orange_anchor.x, 13.0));
    list.push(press);
    block(units, &mut seed, 1, 6, orange_anchor, 40.0, 2000);

    groups.list = list;
}
