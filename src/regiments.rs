//! Regiment battle setup: both armies spawned as a fixed list of regiments
//! (permanent groups) laid out in ranks, heavies in front. The `Groups`
//! list never changes size after this — stable indices make `units.group`
//! a permanent regiment id.

use bevy::prelude::*;

use crate::orders::{GroupData, Groups, RegState};
use crate::terrain::Terrain;
use crate::unit_types::{KIND_ARCHER, KIND_HEAVY, KIND_LIGHT, KIND_SPEAR};
use crate::units::{Units, hash01, push_unit};

/// Unit spacing inside a regiment block.
const SPACING: f32 = 1.4;
/// Gap between regiment blocks.
const REG_GAP: f32 = 10.0;
/// Margins the army layout keeps from the terrain edges. Shared with the
/// deployment zone (orders.rs), which must contain the spawn strip by
/// construction.
pub const SIDE_MARGIN: f32 = 30.0;
pub const EDGE_MARGIN: f32 = 8.0;
/// No-man's land between the two armies. FL_ARMY_GAP overrides — small
/// sandbox battles (e.g. FL_UNITS=5000 FL_ENEMY_REGS=4) want 250+ m of
/// maneuvering room for flank and rear charges; the 200k default keeps
/// the short march.
pub fn army_gap() -> f32 {
    crate::util::env_or("FL_ARMY_GAP", 60.0)
}

pub struct RegimentsPlugin;

impl Plugin for RegimentsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (rout_test_log, dir_test_log, arena_log, charge_test_log, pile_test_log, melee_diag_log, join_test_log, routpass_test_log, archery_log),
        );
    }
}

fn heavy_frac() -> f32 {
    crate::util::env_or("FL_HEAVY_FRAC", 0.4)
}

/// Fraction of each army's regiments that are spear infantry (behind the
/// heavies, ahead of the lights).
fn spear_frac() -> f32 {
    crate::util::env_or("FL_SPEAR_FRAC", 0.25)
}

/// Archer regiments per army: a fixed COUNT by default — archers are
/// force multipliers, and scaling them with army size turned big
/// battles into arrow weather (owner: "two is probably enough
/// considering how OP they are"). FL_ARCHER_FRAC switches back to a
/// fraction of the army for sandbox play (=1 for all-archer fields).
fn archer_regs(n_regs: usize) -> usize {
    match std::env::var("FL_ARCHER_FRAC")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
    {
        Some(frac) => (n_regs as f32 * frac).round() as usize,
        None => 2.min(n_regs),
    }
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
        let seed = crate::units::spawn_seed((team as u32) << 30 | g << 16 | k as u32);
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

pub fn do_spawn_battle(
    units: &mut Units,
    terrain: &Terrain,
    groups: &mut Groups,
    config: &crate::game_state::BattleConfig,
) {
    use crate::game_state::Scenario;
    match config.scenario {
        Scenario::Surround => {
            crate::units::spawn_surround_test(units, terrain, groups);
            return;
        }
        Scenario::Rout => { spawn_rout_test(units, terrain, groups); return; }
        Scenario::Dir => { spawn_dir_test(units, terrain, groups); return; }
        Scenario::Arena => { spawn_arena(units, terrain, groups); return; }
        Scenario::Charge => { spawn_charge_test(units, terrain, groups); return; }
        Scenario::Pile => { spawn_pile_test(units, terrain, groups); return; }
        Scenario::Join => { spawn_join_test(units, terrain, groups); return; }
        Scenario::Routpass => { spawn_routpass_test(units, terrain, groups); return; }
        Scenario::Archery => { spawn_archery_test(units, terrain, groups); return; }
        Scenario::Normal => {}
    }

    let per_team = config.units_per_team;
    let size = config.reg_size;
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
    let usable_w = (terrain.max().x - terrain.min().x) - 2.0 * SIDE_MARGIN;
    let army_gap = army_gap();
    let usable_d = terrain.max().y - EDGE_MARGIN - army_gap / 2.0;
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
        let n_archer = archer_regs(n_regs).min(n_regs.saturating_sub(n_heavy + n_spear));
        let dir: f32 = if team == 0 { -1.0 } else { 1.0 };
        for r in 0..n_regs {
            let rank = r / per_rank;
            let file = r % per_rank;
            // This rank's regiment count (last rank may be partial) — center it.
            let in_rank = per_rank.min(n_regs - rank * per_rank);
            let x0 = (file as f32 - (in_rank - 1) as f32 / 2.0) * pitch_x;
            let z0 = dir * (army_gap / 2.0 + block_d / 2.0 + rank as f32 * (block_d + REG_GAP));
            let anchor = Vec2::new(x0, z0);
            // Heavies lead, spears back them, lights fill the middle,
            // archers take the rear ranks (they shoot over everyone).
            let kind = if r < n_heavy {
                KIND_HEAVY
            } else if r < n_heavy + n_spear {
                KIND_SPEAR
            } else if r >= n_regs - n_archer {
                KIND_ARCHER
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
    let archers = list.iter().filter(|g| g.kind == KIND_ARCHER).count();
    let blue = list.iter().filter(|g| g.team == 0).count();
    info!(
        "spawned {} vs {} regiments ({} heavy, {} spear, {} archer) x {} units ({} total units)",
        blue,
        list.len() - blue,
        heavies,
        spears,
        archers,
        size,
        units.len()
    );
    groups.list = list;
}

/// FL_TEST_ARCHERY: the archer sandbox (also on the menu's debug list).
/// Two blue archer regiments stand behind a friendly light screen. The
/// enemy is fully STATIC: the middle orange block holds INSIDE bow
/// range (fire-at-will opens up within seconds), the flank blocks hold
/// just outside it (attack-order the archers onto one to watch the
/// stand-off halt and the arcs over the screen's heads). Nothing
/// chases. FL_ARCHERY_ATTACK=1 adds the old deep attacker for the
/// fire-at-will / lead / skirmish demo.
fn spawn_archery_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let size = crate::util::env_or("FL_REG_SIZE", 500_usize).max(50);
    let mut list = Vec::new();
    // Blue: light screen up front, two archer blocks behind it.
    spawn_regiment(units, terrain, &mut list, 0, KIND_LIGHT, Vec2::new(0.0, -70.0), size, -1.0);
    list[0].hold = true;
    for x in [-30.0, 30.0] {
        spawn_regiment(units, terrain, &mut list, 0, KIND_ARCHER, Vec2::new(x, -90.0), size, -1.0);
    }
    // Orange: middle block in range (~110 m), flanks just outside.
    // FL_ARCHERY_TARGET=heavy|spear swaps the middle block's kind (the
    // calibration target: kills per volley by armour class).
    let mid_kind = match std::env::var("FL_ARCHERY_TARGET").as_deref() {
        Ok("heavy") => KIND_HEAVY,
        Ok("spear") => KIND_SPEAR,
        _ => KIND_LIGHT,
    };
    for (x, z, kind) in [
        (-60.0, 40.0, KIND_LIGHT),
        (0.0, 20.0, mid_kind),
        (60.0, 40.0, KIND_LIGHT),
    ] {
        spawn_regiment(units, terrain, &mut list, 1, kind, Vec2::new(x, z), size, 1.0);
        let g = list.len() - 1;
        list[g].hold = true;
    }
    // Optional deep attacker, ordered onto the west archer regiment.
    if std::env::var("FL_ARCHERY_ATTACK").is_ok() {
        spawn_regiment(units, terrain, &mut list, 1, KIND_LIGHT, Vec2::new(-30.0, 110.0), size, 1.0);
        let g = list.len() - 1;
        list[g].order = Some(crate::orders::Order::Attack(1));
        list[g].auto_order = true;
    }
    groups.list = list;
    info!(
        "[archery] 2 archer regiments behind a screen; orange holds STATIC — middle block in \
         bow range, flanks beyond it (attack-order for the stand-off); FL_ARCHERY_ATTACK=1 \
         adds a charging attacker"
    );
}

/// FL_TEST_ARCHERY bookkeeping every 4 s: archer strength + ammo, the
/// live arrow pool, and orange's bleed.
fn archery_log(
    groups: Res<Groups>,
    arrows: Res<crate::arrows::Arrows>,
    astats: Res<crate::arrows::ArrowStats>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    if std::env::var("FL_TEST_ARCHERY").is_err() || groups.list.len() < 6 {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 4.0;
    let flag = |gd: &crate::orders::GroupData| {
        format!(
            "{}{}{}{}",
            if gd.engaged { "E" } else { "-" },
            if gd.state.is_broken() { "B" } else { "-" },
            if gd.skirmishing { "S" } else { "-" },
            if gd.order.is_some() { "O" } else { "-" },
        )
    };
    let attacker = groups
        .list
        .get(6)
        .map(|a| {
            format!(
                ", attacker {} ({}) at {:.0} m",
                a.count,
                flag(a),
                a.centroid.distance(groups.list[1].centroid)
            )
        })
        .unwrap_or_default();
    info!(
        "[archery] t={t:.0}s archers {} ({}, ammo {}) / {} ({}, ammo {}) | flight {} \
         (dropped {}) | screen {} | holders {}/{}/{}{attacker}",
        groups.list[1].count,
        flag(&groups.list[1]),
        groups.list[1].ammo_left,
        groups.list[2].count,
        flag(&groups.list[2]),
        groups.list[2].ammo_left,
        arrows.len(),
        astats.dropped,
        groups.list[0].count,
        groups.list[3].count,
        groups.list[4].count,
        groups.list[5].count,
    );
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

/// FL_TEST_CHARGE=1: the phase-B acceptance. Two lanes of blue spears on
/// HOLD facing the enemy; the west lane stands in SPEARWALL, the east
/// lane in normal order. An orange heavy regiment charges each head-on.
/// Acceptance (bands from measurement): the wall holds its ground
/// (centroid displacement well under the unbraced lane's), the wall lane
/// kills chargers faster (reflected charge bonus), and the unbraced lane
/// still fights back (staggers don't stunlock — attacker losses > 0).
fn spawn_charge_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let mut list = Vec::new();
    for (cx, wall) in [(-150.0_f32, true), (150.0, false)] {
        spawn_regiment(units, terrain, &mut list, 0, KIND_SPEAR, Vec2::new(cx, 0.0), 500, -1.0);
        let v = list.len() - 1;
        list[v].hold = true;
        if wall {
            list[v].spacing = crate::formation::FormSpacing::Wall;
            list[v].reform = true;
        }
        spawn_regiment(
            units,
            terrain,
            &mut list,
            1,
            KIND_HEAVY,
            Vec2::new(cx, 70.0),
            500,
            1.0,
        );
        let g = list.len() - 1;
        list[g].order = Some(crate::orders::Order::Attack(v as u32));
        list[g].auto_order = true;
    }
    groups.list = list;
    info!("[charge-test] west: heavies charge a SPEARWALL; east: heavies charge unbraced spears");
}

/// FL_TEST_CHARGE bookkeeping every 2 s: victim-centroid displacement
/// from spawn (does the wall hold ground?), strengths per side, and the
/// staggered headcount (stunlock watch).
fn charge_test_log(
    groups: Res<Groups>,
    units: Res<Units>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    if std::env::var("FL_TEST_CHARGE").is_err() || groups.list.len() < 4 {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 2.0;
    let mut staggered = [0usize; 2]; // [wall lane, open lane]
    for i in 0..units.len() {
        if units.death_t[i] == 0
            && units.swing[i] & crate::units::SWING_STAGGERED != 0
        {
            staggered[usize::from(units.pos[i].x > 0.0)] += 1;
        }
    }
    let dz = |g: usize| groups.list[g].centroid.y - 0.0;
    info!(
        "[charge-test] t={t:.0}s WALL lane: spears {} (dz {:+.1} m) vs heavies {} | \
         OPEN lane: spears {} (dz {:+.1} m) vs heavies {} | staggered {}/{}",
        groups.list[0].count,
        dz(0),
        groups.list[1].count,
        groups.list[2].count,
        dz(2),
        groups.list[3].count,
        staggered[0],
        staggered[1],
    );
}

/// FL_TEST_PILE=1: the pile-on order — six blue regiments in a 3x2
/// block, ALL attack-ordered at one holding orange regiment (the blob
/// repro from the FL_RECTFIGHT saga). Acceptance: the fight crowds the
/// victim's perimeter and the second wave stands PRESSED against the
/// fighting mass (not parked at parade pitch, not smeared into one
/// ball); the victim collapses; blues re-dress rectangles afterward.
fn spawn_pile_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let mut list = Vec::new();
    spawn_regiment(units, terrain, &mut list, 1, KIND_LIGHT, Vec2::new(0.0, 60.0), 500, 1.0);
    list[0].hold = true;
    for row in 0..2 {
        for col in 0..3 {
            let anchor =
                Vec2::new((col as f32 - 1.0) * 55.0, -40.0 - row as f32 * 35.0);
            spawn_regiment(units, terrain, &mut list, 0, KIND_LIGHT, anchor, 500, -1.0);
            let g = list.len() - 1;
            list[g].order = Some(crate::orders::Order::Attack(0));
            list[g].auto_order = true;
        }
    }
    groups.list = list;
    info!("[pile-test] six blue regiments attack ONE holding orange regiment");
}

/// FL_TEST_JOIN=1: the join-the-fight order (regression repro). Orange
/// attacks blue A's line head-on; blue B stands BEHIND A, ordered onto
/// the same orange regiment at spawn. Acceptance: B's march never
/// halts for the overflow trickle that pokes it, B routes AROUND A's
/// ranks (friendly blocks are solid to a march), reaches the orange
/// mass, and its fighter count ramps into the dozens.
fn spawn_join_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let mut list = Vec::new();
    spawn_regiment(units, terrain, &mut list, 1, KIND_LIGHT, Vec2::new(0.0, 50.0), 500, 1.0);
    list[0].order = Some(crate::orders::Order::Attack(1));
    list[0].auto_order = true;
    // A and B carry PLAYER-issued orders (auto_order stays false): the
    // at-ease AI stands auto-engaged regiments down when their target
    // breaks — correct doctrine, but this repro is about
    // player-ordered attacks, which persist through the rout.
    spawn_regiment(units, terrain, &mut list, 0, KIND_LIGHT, Vec2::new(0.0, 10.0), 500, -1.0);
    list[1].order = Some(crate::orders::Order::Attack(0));
    spawn_regiment(units, terrain, &mut list, 0, KIND_LIGHT, Vec2::new(0.0, -30.0), 500, -1.0);
    list[2].order = Some(crate::orders::Order::Attack(0));
    groups.list = list;
    info!("[join-test] orange attacks A's line; B behind A ordered onto the same orange");
}

/// FL_TEST_JOIN bookkeeping every 4 s: does B actually get into the
/// ordered fight instead of stalling behind A on overflow duels?
fn join_test_log(
    units: Res<Units>,
    groups: Res<crate::orders::Groups>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    if std::env::var("FL_TEST_JOIN").is_err() || groups.list.len() < 3 {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 4.0;
    let mut b_windups = 0;
    for i in 0..units.len() {
        if units.group[i] == 2
            && units.death_t[i] == 0
            && units.swing[i] & crate::units::SWING_STATE_MASK == crate::units::SWING_WINDUP
        {
            b_windups += 1;
        }
    }
    info!(
        "[join-test] t={t:.0}s orange {} / A {} / B {} alive; B->orange {:.1} m, B windups {b_windups}, B locked {}",
        groups.list[0].count,
        groups.list[1].count,
        groups.list[2].count,
        groups.list[2].centroid.distance(groups.list[0].centroid),
        groups.list[2].engaged_with_target,
    );
}

/// FL_TEST_ROUTPASS=1: an ordered withdrawal straight through a formed
/// friendly line (the "unit makes stupid space for retreating unit"
/// repro). A holds its line; the passer regiment is Move-ordered
/// through A's ranks. Acceptance: the passer slips through the seams
/// at body scale, A's disorder bumps briefly and re-dresses — no
/// corridor excavated, no lasting raggedness.
fn spawn_routpass_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let mut list = Vec::new();
    spawn_regiment(units, terrain, &mut list, 0, KIND_LIGHT, Vec2::new(0.0, 0.0), 500, 1.0);
    list[0].hold = true;
    spawn_regiment(units, terrain, &mut list, 0, KIND_LIGHT, Vec2::new(0.0, 35.0), 500, -1.0);
    list[1].order = Some(crate::orders::Order::Move(Vec2::new(0.0, -60.0)));
    list[1].auto_order = true;
    groups.list = list;
    info!("[routpass-test] passer Move-ordered straight through A's held line");
}

/// FL_TEST_ROUTPASS bookkeeping every 2 s: A's shape while the passer
/// walks through it, and the overlap floor.
fn routpass_test_log(
    groups: Res<crate::orders::Groups>,
    stats: Res<crate::movement::SimStats>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    if std::env::var("FL_TEST_ROUTPASS").is_err() || groups.list.len() < 2 {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 2.0;
    info!(
        "[routpass-test] t={t:.0}s A disorder {:.2} m, passer z {:.1}, nn min {:.2}",
        groups.list[0].disorder,
        groups.list[1].centroid.y,
        stats.nn_min,
    );
}

/// FL_TEST_PILE bookkeeping every 4 s: victim strength + how many of
/// the six attackers are actually fighting.
fn pile_test_log(groups: Res<Groups>, time: Res<Time>, mut next: Local<f32>) {
    if std::env::var("FL_TEST_PILE").is_err() || groups.list.len() < 7 {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 4.0;
    let engaged = groups.list[1..].iter().filter(|g| g.engaged).count();
    info!(
        "[pile-test] t={t:.0}s orange {} alive, blues engaged {engaged}/6",
        groups.list[0].count
    );
}

/// FL_DIAG_MELEE=1: per-rank ground truth for the first engaged
/// regiment, every 4 s — each rank's headcount, mean distance to the
/// nearest living enemy, the share of it inside its own weapon reach,
/// and the share mid-windup. The log that answers "why does only the
/// first row fight" with data instead of theory.
fn melee_diag_log(
    units: Res<Units>,
    groups: Res<crate::orders::Groups>,
    grid: Res<crate::spatial::SpatialGrid>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    if std::env::var("FL_DIAG_MELEE").is_err() {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 4.0;
    let Some((g, gd)) = groups.list.iter().enumerate().find(|(_, gd)| {
        gd.engaged
            && gd.count > 0
            && !gd.state.is_broken()
            && gd.shape == crate::formation::FormShape::Rect
    }) else {
        return;
    };
    let fwd = crate::formation::facing_dir(gd.facing);
    let pitch = gd.spacing.pitch().y;
    let mut members: Vec<(usize, f32)> = Vec::new();
    let mut max_fwd = f32::MIN;
    for i in 0..units.len() {
        if units.group[i] as usize == g && units.death_t[i] == 0 {
            let f = units.home[i].dot(fwd);
            max_fwd = max_fwd.max(f);
            members.push((i, f));
        }
    }
    if members.is_empty() {
        return;
    }
    const R: usize = 8; // ranks 0..6 individually, 7+ pooled
    let mut cnt = [0u32; R];
    let mut dsum = [0.0f64; R];
    let mut in_reach = [0u32; R];
    let mut windup = [0u32; R];
    for (i, f) in members {
        let rank = ((((max_fwd - f) / pitch).round() as i64).max(0) as usize).min(R - 1);
        let p = Vec2::new(units.pos[i].x, units.pos[i].z);
        let team_bit = (units.team[i] as u32) * crate::spatial::META_TEAM;
        let mut best = f32::MAX;
        grid.for_each_candidate(p, 6.0, |o| {
            if (o.meta & crate::spatial::META_TEAM) != team_bit
                && (o.meta & crate::spatial::META_DYING) == 0
            {
                best = best.min(p.distance_squared(o.xz()));
            }
        });
        cnt[rank] += 1;
        if best < f32::MAX {
            let d = best.sqrt();
            dsum[rank] += d as f64;
            if d <= crate::unit_types::TYPES[units.kind[i] as usize].reach {
                in_reach[rank] += 1;
            }
        } else {
            dsum[rank] += 6.0; // beyond the 6 m probe
        }
        if units.swing[i] & crate::units::SWING_STATE_MASK == crate::units::SWING_WINDUP {
            windup[rank] += 1;
        }
    }
    let mut s = String::new();
    for r in 0..R {
        if cnt[r] == 0 {
            continue;
        }
        s.push_str(&format!(
            " r{r}[n{} d{:.2} reach{}% wind{}%]",
            cnt[r],
            dsum[r] / cnt[r] as f64,
            in_reach[r] * 100 / cnt[r],
            windup[r] * 100 / cnt[r],
        ));
    }
    info!("[melee-diag] t={t:.0}s regiment {g}:{s}");
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
        // FL_ARENA_AUTO=1: fire both attacks immediately through the
        // same Order::Attack the RMB click writes — the whole arena
        // becomes a scripted acceptance run for the sector numbers.
        if std::env::var("FL_ARENA_AUTO").is_ok() {
            list[g].order = Some(crate::orders::Order::Attack(e as u32));
        }
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
    groups: Res<Groups>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    if std::env::var("FL_ARENA").is_err() || units.len() == 0 || groups.list.len() < 4 {
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
        "[arena] t={t:.0}s FRONT lane you {} vs {} (morale {:.0}) | \
         REAR lane you {} vs {} (morale {:.0}) | \
         your kills: front {} ({:.1}/hit), side {} ({:.1}/hit), rear {} ({:.1}/hit)",
        alive[0][0],
        alive[0][1],
        groups.list[1].morale,
        alive[1][0],
        alive[1][1],
        groups.list[3].morale,
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
    readout: Res<crate::morale::MoraleReadout>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    // Regiment-0 morale/fatigue timeline: the rout acceptance AND a
    // line-fight probe for the front battle (flank ring must read ~0 in
    // a frontal press).
    if (std::env::var("FL_TEST_ROUT").is_err() && std::env::var("FL_TEST_FRONT").is_err())
        || groups.list.is_empty()
    {
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
    let f = readout.0.first().copied().unwrap_or_default();
    info!(
        "[rout-test] t={t:.0}s blue: {} alive, morale {:.0}, {}, fled {} | cas {:.1} xchg {:.1} flank {:.1} ({:.0}%) noen {:.1} fat {:.1} r {:.0}m z {:.0} en {}",
        g.count,
        g.morale,
        state,
        stats.fled[0],
        f.casualties,
        f.exchange,
        f.flanked,
        f.flanked01 * 100.0,
        f.no_enemy,
        g.fatigue,
        g.radius,
        g.centroid.y,
        g.enemy_near,
    );
}
