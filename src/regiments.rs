//! Regiment battle setup: both armies spawned as a fixed list of regiments
//! (permanent groups) laid out in ranks, heavies in front. The `Groups`
//! list never changes size after this — stable indices make `units.group`
//! a permanent regiment id.

use bevy::prelude::*;

use crate::orders::{GroupData, Groups, RegState};
use crate::terrain::Terrain;
use crate::unit_types::{KIND_HEAVY, KIND_LIGHT, KIND_SPEAR};
use crate::units::{Units, hash01, push_unit, units_per_team};

/// Unit spacing inside a regiment block.
const SPACING: f32 = 1.4;
/// Gap between regiment blocks.
const REG_GAP: f32 = 10.0;
/// No-man's land between the two armies. FL_ARMY_GAP overrides — small
/// sandbox battles (e.g. FL_UNITS=5000 FL_ENEMY_REGS=4) want 250+ m of
/// maneuvering room for flank and rear charges; the 200k default keeps
/// the short march.
fn army_gap() -> f32 {
    crate::util::env_or("FL_ARMY_GAP", 60.0)
}

pub struct RegimentsPlugin;

impl Plugin for RegimentsPlugin {
    fn build(&self, app: &mut App) {
        // Terrain resource is created in PreStartup (generate_terrain).
        // Morale lives in crate::morale (MoralePlugin).
        app.add_systems(Startup, spawn_battle)
            .add_systems(Update, (rout_test_log, dir_test_log, arena_log, restart_key));
    }
}

fn reg_size() -> usize {
    crate::util::env_or("FL_REG_SIZE", 1000_usize).max(50)
}

/// Fraction of each army's regiments that are heavy infantry (front ranks).
fn heavy_frac() -> f32 {
    crate::util::env_or("FL_HEAVY_FRAC", 0.4)
}

/// Fraction of each army's regiments that are spear infantry (behind the
/// heavies, ahead of the lights).
fn spear_frac() -> f32 {
    crate::util::env_or("FL_SPEAR_FRAC", 0.25)
}

/// Spawn one regiment block (units + GroupData). `dir` faces the enemy
/// (+1 toward -Z spawns rows so the block front is enemy-side).
#[allow(clippy::too_many_arguments)] // spawn-time plumbing, all scalars
fn spawn_regiment(
    units: &mut Units,
    terrain: &Terrain,
    list: &mut Vec<GroupData>,
    team: u8,
    kind: u8,
    anchor: Vec2,
    size: usize,
    dir: f32,
) {
    let cols = ((size as f32 * 2.2).sqrt().ceil() as usize).max(1);
    let rows = size.div_ceil(cols);
    let g = list.len() as u32;
    list.push(GroupData::new(team, kind, anchor, size));
    for k in 0..size {
        let row = k / cols;
        let col = k % cols;
        let seed = (team as u32) << 30 | g << 16 | k as u32;
        let jx = hash01(seed.wrapping_mul(3) + 1) - 0.5;
        let jz = hash01(seed.wrapping_mul(3) + 2) - 0.5;
        let x = anchor.x + (col as f32 - (cols - 1) as f32 / 2.0) * SPACING + jx * 0.5;
        let z = anchor.y + dir * ((row as f32 - (rows - 1) as f32 / 2.0) * SPACING) + jz * 0.5;
        push_unit(units, terrain, seed, team, kind, g, x, z, anchor);
    }
    // Rigid rank-and-file by default: exact slots into `home` (the spawn
    // jitter stays in the positions — inside the hold deadzone, so the
    // line reads organic without anyone shuffling at t=0).
    let gd = &mut list[g as usize];
    gd.shape = crate::formation::FormShape::Rect;
    gd.files = cols as u32;
    crate::formation::assign_slots(units, g, gd);
}

fn spawn_battle(mut units: ResMut<Units>, terrain: Res<Terrain>, mut groups: ResMut<Groups>) {
    do_spawn_battle(&mut units, &terrain, &mut groups);
}

/// R: restart the battle from scratch (fresh armies, cleared stats).
#[allow(clippy::too_many_arguments)] // bevy system params
fn restart_key(
    keys: Res<ButtonInput<KeyCode>>,
    mut units: ResMut<Units>,
    terrain: Res<Terrain>,
    mut groups: ResMut<Groups>,
    mut stats: ResMut<crate::combat::CombatStats>,
    mut selection: ResMut<crate::orders::Selection>,
    mut outcome: ResMut<crate::ai::BattleOutcome>,
    mut corpses: ResMut<crate::render_units::Corpses>,
    mut dir_stats: ResMut<crate::movement::DirTestStats>,
) {
    if !keys.just_pressed(KeyCode::KeyR) {
        return;
    }
    *units = Units::default();
    *stats = crate::combat::CombatStats::default();
    *dir_stats = crate::movement::DirTestStats::default();
    *selection = crate::orders::Selection::default();
    outcome.0 = None;
    corpses.clear();
    do_spawn_battle(&mut units, &terrain, &mut groups);
    info!("battle restarted");
}

fn do_spawn_battle(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    if std::env::var("FL_TEST_SURROUND").is_ok() {
        crate::units::spawn_surround_test(units, terrain, groups);
        return;
    }
    if std::env::var("FL_TEST_ROUT").is_ok() {
        spawn_rout_test(units, terrain, groups);
        return;
    }
    if std::env::var("FL_TEST_DIR").is_ok() {
        spawn_dir_test(units, terrain, groups);
        return;
    }
    if std::env::var("FL_ARENA").is_ok() {
        spawn_arena(units, terrain, groups);
        return;
    }

    let per_team = units_per_team();
    let size = reg_size();
    let n_regs = (per_team / size).max(1);

    // Block geometry: wider than deep (~2.2:1).
    let cols = ((size as f32 * 2.2).sqrt().ceil() as usize).max(1);
    let rows = size.div_ceil(cols);
    let block_w = cols as f32 * SPACING;
    let block_d = rows as f32 * SPACING;

    // Regiments per rank: prefer filling the map width, but never spawn
    // ranks past the terrain edge (the sim clamps positions to the bounds
    // and stacked rows would squash onto the boundary line). If the army
    // needs more ranks than fit, widen the ranks and shrink the x pitch.
    let usable_w = (terrain.max().x - terrain.min().x) - 60.0;
    let army_gap = army_gap();
    let usable_d = terrain.max().y - 8.0 - army_gap / 2.0;
    let max_ranks = (((usable_d - block_d) / (block_d + REG_GAP)).floor() as usize + 1).max(1);
    let per_rank = ((usable_w / (block_w + REG_GAP)).floor() as usize)
        .max(n_regs.div_ceil(max_ranks))
        .max(1);
    let pitch_x = (usable_w / per_rank as f32).min(block_w + REG_GAP);
    if pitch_x < block_w + 1.0 {
        warn!(
            "regiment layout tight: pitch {pitch_x:.1} m vs block {block_w:.1} m — \
             reduce FL_REG_SIZE or FL_UNITS"
        );
    }

    units.pos.reserve(per_team * 2);
    let mut list: Vec<GroupData> = Vec::with_capacity(n_regs * 2);

    for team in 0..2u8 {
        // FL_ENEMY_REGS caps the enemy regiment count (sandbox
        // asymmetry, e.g. 5v4 so one player regiment stays unchased).
        let n_regs = if team == 1 {
            crate::util::env_or("FL_ENEMY_REGS", n_regs).clamp(1, n_regs)
        } else {
            n_regs
        };
        let n_heavy = (n_regs as f32 * heavy_frac()).round() as usize;
        let n_spear = (n_regs as f32 * spear_frac()).round() as usize;
        let dir: f32 = if team == 0 { -1.0 } else { 1.0 };
        for r in 0..n_regs {
            let rank = r / per_rank;
            let file = r % per_rank;
            // This rank's regiment count (last rank may be partial) — center it.
            let in_rank = per_rank.min(n_regs - rank * per_rank);
            let x0 = (file as f32 - (in_rank - 1) as f32 / 2.0) * pitch_x;
            let z0 = dir * (army_gap / 2.0 + block_d / 2.0 + rank as f32 * (block_d + REG_GAP));
            let anchor = Vec2::new(x0, z0);
            // Heavies lead, spears back them, lights fill the rear ranks.
            let kind = if r < n_heavy {
                KIND_HEAVY
            } else if r < n_heavy + n_spear {
                KIND_SPEAR
            } else {
                KIND_LIGHT
            };
            spawn_regiment(units, terrain, &mut list, team, kind, anchor, size, dir);
        }
    }
    // FL_ENEMY_STATIC=1: the enemy army holds position (never auto-
    // engages, never chases — hold's existing semantics) and the AI
    // stands down (ai.rs). A standing target army for charge practice;
    // units still defend themselves in melee and still rout.
    if std::env::var("FL_ENEMY_STATIC").is_ok() {
        for gd in list.iter_mut().filter(|g| g.team == 1) {
            gd.hold = true;
        }
        info!("enemy army is STATIC (hold position, no AI)");
    }
    let heavies = list.iter().filter(|g| g.kind == KIND_HEAVY).count();
    let spears = list.iter().filter(|g| g.kind == KIND_SPEAR).count();
    let blue = list.iter().filter(|g| g.team == 0).count();
    info!(
        "spawned {} vs {} regiments ({} heavy, {} spear) x {} units ({} total units)",
        blue,
        list.len() - blue,
        heavies,
        spears,
        size,
        units.len()
    );
    groups.list = list;
}

/// FL_TEST_ROUT: one blue regiment vs three converging orange regiments.
/// Acceptance: blue BREAKS well before annihilation, flees toward its own
/// edge, and despawns there (fled counter), instead of fighting to the
/// last man.
fn spawn_rout_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let mut list = Vec::new();
    let blue = Vec2::new(0.0, -60.0);
    spawn_regiment(units, terrain, &mut list, 0, KIND_LIGHT, blue, 1000, -1.0);
    for (x, z, tx) in [(-90.0, 40.0, -18.0), (0.0, 60.0, 0.0), (90.0, 40.0, 18.0)] {
        let anchor = Vec2::new(x, z);
        spawn_regiment(units, terrain, &mut list, 1, KIND_LIGHT, anchor, 1000, 1.0);
        let g = list.len() - 1;
        list[g].order = Some(crate::orders::Order::Move(Vec2::new(tx, blue.y)));
    }
    groups.list = list;
    info!("[rout-test] 1 blue vs 3 converging orange regiments");
}

/// FL_TEST_DIR: the directional-defense acceptance. Two hammer-and-anvil
/// pairs of light infantry, victims on HOLD facing +Z. Each victim is
/// pinned frontally by a near attacker; a farther second attacker arrives
/// ~2.5 s later — from the LEFT side in the control pair (the shield arm:
/// factor-identical to frontal) and from the REAR in the test pair (skill
/// and shield gone). The damage pass buckets every hit on a blue victim by
/// its actual sector at hit time (movement.rs DirTestStats). Acceptance:
/// rear-sector kills (one feeding regiment) >= front-sector kills (TWO
/// feeding regiments), i.e. per-attacker rear kill rate >= 2x frontal.
fn spawn_dir_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let mut list = Vec::new();
    // Attacker spawn offsets per pair: control = front pin + LEFT side
    // (shield arm covers: factor-identical to frontal), test = front pin
    // + REAR, lone = a single unassisted rear charge on a holding block
    // (the M2TW no-auto-face rule: the victims must NOT pirouette to
    // meet it — their facing holds until the melee rim turns to fight).
    let pairs: [(f32, &[Vec2]); 3] = [
        (-250.0, &[Vec2::new(0.0, 35.0), Vec2::new(55.0, 0.0)]),
        (250.0, &[Vec2::new(0.0, 35.0), Vec2::new(0.0, -55.0)]),
        (0.0, &[Vec2::new(0.0, -55.0)]),
    ];
    for (cx, offsets) in pairs {
        spawn_regiment(units, terrain, &mut list, 0, KIND_LIGHT, Vec2::new(cx, 0.0), 500, -1.0);
        let v = list.len() - 1;
        list[v].hold = true;
        for off in offsets {
            let anchor = Vec2::new(cx, 0.0) + *off;
            spawn_regiment(units, terrain, &mut list, 1, KIND_LIGHT, anchor, 500, 1.0);
            let g = list.len() - 1;
            list[g].order = Some(crate::orders::Order::Move(Vec2::new(cx, 0.0)));
            list[g].auto_order = true;
        }
    }
    groups.list = list;
    info!(
        "[dir-test] control pair (front+left) x=-250, test pair (front+rear) x=+250, \
         lone rear charge x=0"
    );
}

/// FL_TEST_DIR bookkeeping: sector-bucketed kills/damage + per-pair victim
/// survival, every 2 s.
#[allow(clippy::too_many_arguments)] // bevy system params
fn dir_test_log(
    dir_stats: Res<crate::movement::DirTestStats>,
    units: Res<Units>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    if !dir_stats.enabled || units.len() == 0 {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 2.0;
    let (mut ctl, mut test, mut lone) = (0usize, 0usize, 0usize);
    // The lone-pair victims' facing: mean |yaw| from the ordered facing
    // (0 = +Z). Before the M2TW no-auto-face rule this snapped to ~pi
    // (the whole block turned to meet the rear approach) long before
    // contact; now only soldiers actually fighting may deviate.
    let (mut yaw_dev, mut yaw_n) = (0.0f32, 0usize);
    for i in 0..units.len() {
        if units.team[i] == 0 && units.death_t[i] == 0 {
            if units.pos[i].x < -100.0 {
                ctl += 1;
            } else if units.pos[i].x > 100.0 {
                test += 1;
            } else {
                lone += 1;
                yaw_dev += units.yaw[i].abs();
                yaw_n += 1;
            }
        }
    }
    let k = &dir_stats.kills[0];
    let per_hit = |s: usize| dir_stats.dmg[0][s] / dir_stats.hits[0][s].max(1) as f64;
    info!(
        "[dir-test] t={t:.0}s kills by sector: front {} side {} rear {} \
         (dmg/hit {:.1}/{:.1}/{:.1}); victims alive: control {ctl}/500, test {test}/500, \
         lone {lone}/500 (yaw dev {:.2} rad)",
        k[0],
        k[1],
        k[2],
        per_hit(0),
        per_hit(1),
        per_hit(2),
        yaw_dev / yaw_n.max(1) as f32,
    );
}

/// FL_ARENA=1: hand-testing range for directional damage. Two mirrored
/// duel lanes of identical heavy regiments; both enemies HOLD facing
/// south (toward the player side) with the AI stood down. Lane FRONT
/// (west): your regiment south of the enemy — attack it head-on. Lane
/// REAR (east): your regiment already parked BEHIND the enemy — attack
/// south into its back. Same kind, same size, same distance; the only
/// variable is the sector, and the [arena] log prints your kills and
/// damage per hit by sector so the comparison is numbers, not vibes.
fn spawn_arena(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let size = crate::util::env_or("FL_REG_SIZE", 500_usize).max(50);
    let mut list = Vec::new();
    for (lane_x, player_z) in [(-80.0_f32, -85.0_f32), (80.0, 85.0)] {
        spawn_regiment(
            units,
            terrain,
            &mut list,
            0,
            KIND_HEAVY,
            Vec2::new(lane_x, player_z),
            size,
            if player_z < 0.0 { -1.0 } else { 1.0 },
        );
        let g = list.len() - 1;
        if player_z > 0.0 {
            // The rear-lane regiment starts behind the enemy: face back
            // south toward it (slots rebake to the new facing).
            list[g].facing = std::f32::consts::PI;
            crate::formation::assign_slots(units, g as u32, &mut list[g]);
        }
        spawn_regiment(units, terrain, &mut list, 1, KIND_HEAVY, Vec2::new(lane_x, 0.0), size, 1.0);
        let e = list.len() - 1;
        list[e].hold = true; // faces south by default (team-1 facing)
    }
    groups.list = list;
    info!(
        "[arena] FRONT lane west: attack the enemy head-on. REAR lane east: \
         attack the enemy from behind. Identical heavies, enemies hold, no AI."
    );
}

/// FL_ARENA bookkeeping: per-lane strengths + the player's kills and
/// damage per hit by sector, every 4 s once fighting starts.
fn arena_log(
    dir_stats: Res<crate::movement::DirTestStats>,
    units: Res<Units>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    if std::env::var("FL_ARENA").is_err() || units.len() == 0 {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 4.0;
    // Lane split at x = 0: west = frontal duel, east = rear duel.
    let mut alive = [[0usize; 2]; 2]; // [lane][team]
    for i in 0..units.len() {
        if units.death_t[i] == 0 {
            let lane = usize::from(units.pos[i].x > 0.0);
            alive[lane][units.team[i] as usize] += 1;
        }
    }
    // Your kills = hits on team-1 victims.
    let k = &dir_stats.kills[1];
    if k.iter().sum::<u64>() == 0 && dir_stats.kills[0].iter().sum::<u64>() == 0 {
        return; // nothing has happened yet — keep the log quiet
    }
    let per_hit = |s: usize| dir_stats.dmg[1][s] / dir_stats.hits[1][s].max(1) as f64;
    info!(
        "[arena] t={t:.0}s FRONT lane you {} vs {} | REAR lane you {} vs {} | \
         your kills: front {} ({:.1}/hit), side {} ({:.1}/hit), rear {} ({:.1}/hit)",
        alive[0][0],
        alive[0][1],
        alive[1][0],
        alive[1][1],
        k[0],
        per_hit(0),
        k[1],
        per_hit(1),
        k[2],
        per_hit(2),
    );
}

/// FL_TEST_ROUT bookkeeping: blue regiment morale/state timeline.
pub fn rout_test_log(
    groups: Res<Groups>,
    stats: Res<crate::combat::CombatStats>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    if std::env::var("FL_TEST_ROUT").is_err() || groups.list.is_empty() {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 2.0;
    let g = &groups.list[0];
    let state = match g.state {
        RegState::Steady => "steady",
        RegState::Routing { .. } => "ROUTING",
        RegState::Shattered => "SHATTERED",
    };
    info!(
        "[rout-test] t={t:.0}s blue: {} alive, morale {:.0}, {}, fled {}",
        g.count, g.morale, state, stats.fled[0]
    );
}
