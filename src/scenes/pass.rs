//! FL_TEST_ROUTPASS: a regiment walking through a friendly line, and
//! its log; FL_PASS_LINE: the second line of the normal battle sent
//! through the first.

use bevy::prelude::*;

use crate::orders::Groups;
use crate::regiments::{REG_GAP, spawn_regiment};
use crate::terrain::Terrain;
use crate::unit_types::{KIND_HEAVY, KIND_LIGHT, KIND_SPEAR};
use crate::units::Units;

/// FL_TEST_ROUTPASS=1: an ordered withdrawal straight through a formed
/// friendly line (the "unit makes stupid space for retreating unit"
/// repro). A holds its line; the passer regiment is Move-ordered
/// through A's ranks. Acceptance: the passer slips through the seams
/// at body scale, A's disorder bumps briefly and re-dresses — no
/// corridor excavated, no lasting raggedness.
///
/// Variants: FL_ROUTPASS_SIZE=<men> both regiments that size, at the
/// 15 by 8 aspect of a 120-man regiment (500, the default, at
/// the default layout); FL_ROUTPASS_KINDS=1 knights through spearmen;
/// FL_ROUTPASS_GAP=<m> the passer that far behind A's rear rank instead
/// of 35 m from its centre, room for A to open toward it;
/// FL_ROUTPASS_SPEED=<m/s> the passer's men at about that speed;
/// FL_ROUTPASS_FLANKS=1 two more holders like A 10 m to its sides;
/// FL_ROUTPASS_FRONT=<m> a held enemy like A that far in front of A,
/// the passer ordered to attack it; FL_ROUTPASS_CROSS=<m> a third
/// friendly regiment like the passer that far off A's right flank,
/// walking across A's facing while the passer walks through A.
pub(crate) fn spawn_routpass_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let size: Option<usize> = std::env::var("FL_ROUTPASS_SIZE").ok().and_then(|v| v.parse().ok());
    let n = size.unwrap_or(500).max(1);
    let (holder_kind, passer_kind) = if std::env::var("FL_ROUTPASS_KINDS").is_ok_and(|v| v == "1") {
        (KIND_SPEAR, KIND_HEAVY)
    } else {
        (KIND_LIGHT, KIND_LIGHT)
    };
    let mut list = Vec::new();
    // A faces the way the passer walks, its rear to the passer, as a
    // first line with a second line ordered through it.
    spawn_regiment(units, terrain, &mut list, 0, holder_kind, Vec2::new(0.0, 0.0), n, -1.0);
    list[0].hold = true;
    let gap: Option<f32> = std::env::var("FL_ROUTPASS_GAP").ok().and_then(|v| v.parse().ok());
    // The passer's centre: 35 m from A's, or `gap` behind A's rear rank.
    let spawned_files = list[0].files;
    let files = |n: usize| {
        if size.is_some() {
            ((n as f32 * 15.0 / 8.0).sqrt().round() as u32).max(1)
        } else {
            spawned_files
        }
    };
    let half_depth = |n: usize, kind: u8| {
        (n.div_ceil(files(n) as usize) - 1) as f32 * 0.5 * crate::formation::gap(kind)
    };
    let z = match gap {
        Some(gap) => half_depth(n, holder_kind) + gap + half_depth(n, passer_kind),
        None => 35.0,
    };
    let holder_files = files(n);
    let holder_depth = half_depth(n, holder_kind);
    let width = (holder_files - 1) as f32 * crate::formation::gap(holder_kind);
    spawn_regiment(units, terrain, &mut list, 0, passer_kind, Vec2::new(0.0, z), n, -1.0);
    // FL_ROUTPASS_STOP=1: the passer's destination is A's own ground, so
    // it stops inside A; at 30 s `routpass_test_log` sends it on past A.
    // The men it leaves behind, and the men of A carried along, are the
    // stragglers seen in play.
    // FL_ROUTPASS_SIDE=1: the passer starts to A's right, `gap` (10 m by
    // default) off its flank, facing and walking across A's facing, so A
    // opens its lanes across.
    let side = std::env::var("FL_ROUTPASS_SIDE").is_ok_and(|v| v == "1");
    let goal = if std::env::var("FL_ROUTPASS_STOP").is_ok() {
        Vec2::ZERO
    } else if side {
        Vec2::new(-70.0, 0.0)
    } else {
        Vec2::new(0.0, -60.0)
    };
    if side {
        let x = 0.5 * width + gap.unwrap_or(10.0) + half_depth(n, passer_kind);
        let p = &mut list[1];
        p.anchor = Vec2::new(x, 0.0);
        p.centroid = p.anchor;
    }
    list[1].order = Some(crate::orders::Order::Move(goal));
    list[1].auto_order = true;
    // A small held enemy far off the pass, so the battle does not end
    // with no enemy on the field (ai.rs `check_victory`) and freeze it.
    spawn_regiment(units, terrain, &mut list, 1, KIND_LIGHT, Vec2::new(150.0, 0.0), 16, -1.0);
    list[2].hold = true;
    list[2].steadfast = true;
    // FL_ROUTPASS_FLANKS=1: two more holding regiments like A, 10 m to
    // each side of it, as in the default battle's rows.
    if std::env::var("FL_ROUTPASS_FLANKS").is_ok_and(|v| v == "1") {
        for x in [-1.0f32, 1.0] {
            let at = Vec2::new(x * (width + REG_GAP), 0.0);
            spawn_regiment(units, terrain, &mut list, 0, holder_kind, at, n, -1.0);
            list.last_mut().unwrap().hold = true;
        }
    }
    // FL_ROUTPASS_FRONT=<m>: a held enemy regiment like A that far in
    // front of A's front rank, and the passer ordered through A to
    // attack it, as a second line sent through the first.
    if let Some(d) = std::env::var("FL_ROUTPASS_FRONT").ok().and_then(|v| v.parse::<f32>().ok()) {
        let at = Vec2::new(0.0, -(holder_depth + d + holder_depth));
        spawn_regiment(units, terrain, &mut list, 1, holder_kind, at, n, 1.0);
        let e = list.len() - 1;
        list[e].hold = true;
        list[e].steadfast = true;
        list[1].order = Some(crate::orders::Order::Attack(e as u32));
    }
    // FL_ROUTPASS_FRIEND=<m>: a friendly holder like A that far in front
    // of A's front rank, the second row's geometry of the default
    // battle: with blocks 10 m in front and behind, A opens in pairs of
    // files.
    if let Some(d) = std::env::var("FL_ROUTPASS_FRIEND").ok().and_then(|v| v.parse::<f32>().ok()) {
        let at = Vec2::new(0.0, -(holder_depth + d + holder_depth));
        spawn_regiment(units, terrain, &mut list, 0, holder_kind, at, n, -1.0);
        list.last_mut().unwrap().hold = true;
    }
    // FL_ROUTPASS_CROSS=<m>: a third friendly regiment like the passer,
    // that far off A's right flank, facing and walking across A's
    // facing. It reaches A once the passer has A open along its facing
    // and meets the files from the side, as seen in play: two units
    // through one block at right angles.
    let cross: Option<f32> = std::env::var("FL_ROUTPASS_CROSS").ok().and_then(|v| v.parse().ok());
    if let Some(off) = cross {
        let x = 0.5 * width + off + half_depth(n, passer_kind);
        spawn_regiment(units, terrain, &mut list, 0, passer_kind, Vec2::new(x, 0.0), n, -1.0);
        let c = list.len() - 1;
        // FL_ROUTPASS_CROSS_GOAL=<x>: where across A the crosser is sent,
        // -70 by default. Farther, -150 for example, it does not arrive
        // while its soldiers are still in A.
        let goal_x: f32 = crate::util::env_or("FL_ROUTPASS_CROSS_GOAL", -70.0);
        list[c].order = Some(crate::orders::Order::Move(Vec2::new(goal_x, 0.0)));
        list[c].auto_order = true;
    }
    let cross_g = cross.map(|_| list.len() - 1);
    for (g, gd) in list.iter_mut().enumerate() {
        if g == 2 {
            continue;
        }
        gd.facing = if gd.team == 0 { std::f32::consts::PI } else { 0.0 };
        if (g == 1 && side) || Some(g) == cross_g {
            gd.facing = -std::f32::consts::FRAC_PI_2;
        }
        if size.is_some() {
            gd.files = files(n);
        }
        // FL_ROUTPASS_OPEN=1: A stands in the open order from the start.
        if g == 0 && std::env::var("FL_ROUTPASS_OPEN").is_ok_and(|v| v == "1") {
            let room = crate::formation::OPEN_ROOM_MAX;
            if let Some(o) = crate::formation::choose_open(gd, false, false, room, room) {
                crate::formation::open_order(gd, o);
            }
        }
        for i in crate::formation::assign_slots(units, g as u32, gd) {
            let i = i as usize;
            let at = gd.anchor + units.home[i];
            let y = terrain.height_at(at.x, at.y) + crate::unit_types::half_height(units.kind[i] as usize);
            units.pos[i] = Vec3::new(at.x, y, at.y);
            units.pos_prev[i] = units.pos[i];
        }
    }
    if let Some(v) = std::env::var("FL_ROUTPASS_SPEED").ok().and_then(|v| v.parse::<f32>().ok()) {
        let scale = v / crate::unit_types::TYPES[passer_kind as usize].speed;
        for i in 0..units.len() {
            if units.group[i] == 1 || Some(units.group[i] as usize) == cross_g {
                units.speed[i] *= scale;
            }
        }
    }
    groups.list = list;
    info!(
        "[routpass-test] passer of {n} Move-ordered straight through A's held line of {n}, {} files, from z {z:.1}",
        files(n)
    );
}

/// FL_TEST_ROUTPASS bookkeeping every 2 s: A's shape while the passer
/// walks through it, the overlap floor, the passers inside A's
/// rectangle (its frame about its anchor), and A's men off their file
/// line (the mean lateral distance to their slots). Watched every
/// frame: the time from the first passer in to the last out.
/// FL_PASS_LINE=1: in the normal battle, at 10 s, every player regiment
/// standing behind its army's median line gets a Move order 120 m toward
/// the enemy, through the regiments in front of it: the second line
/// passing the first, at the battle's full size, to measure what
/// letting friends through costs.
pub(super) fn pass_line_script(
    mut groups: ResMut<crate::orders::Groups>,
    time: Res<Time>,
    mut sent: Local<bool>,
) {
    if *sent || std::env::var("FL_PASS_LINE").is_err() || time.elapsed_secs() < 10.0 {
        return;
    }
    *sent = true;
    let enemy: Vec<Vec2> = groups.list.iter().filter(|g| g.team == 1 && g.count > 0).map(|g| g.centroid).collect();
    if enemy.is_empty() {
        return;
    }
    let enemy_mid = enemy.iter().copied().sum::<Vec2>() / enemy.len() as f32;
    let mut dists: Vec<f32> = groups
        .list
        .iter()
        .filter(|g| g.team == 0 && g.count > 0)
        .map(|g| g.centroid.distance(enemy_mid))
        .collect();
    dists.sort_by(|a, b| a.total_cmp(b));
    let Some(&median) = dists.get(dists.len() / 2) else { return };
    let mut n = 0;
    for g in groups.list.iter_mut() {
        if g.team != 0 || g.count == 0 || g.centroid.distance(enemy_mid) <= median {
            continue;
        }
        let dir = (enemy_mid - g.centroid).normalize_or_zero();
        g.order = Some(crate::orders::Order::Move(g.centroid + dir * 120.0));
        g.auto_order = true;
        g.reform = true;
        n += 1;
    }
    info!("[pass-line] {n} rear regiments ordered 120 m through the line in front");
}

pub(super) fn routpass_test_log(
    units: Res<Units>,
    mut groups: ResMut<crate::orders::Groups>,
    stats: Res<crate::sim::SimStats>,
    time: Res<Time>,
    mut next: Local<f32>,
    mut passes: Local<[(Option<f32>, Option<f32>); 2]>,
    mut sent_on: Local<bool>,
) {
    if std::env::var("FL_TEST_ROUTPASS").is_err() || groups.list.len() < 2 {
        return;
    }
    // The crossing regiment (FL_ROUTPASS_CROSS): the last friendly one.
    let cross_g = std::env::var("FL_ROUTPASS_CROSS")
        .ok()
        .and_then(|_| groups.list.iter().rposition(|gd| gd.team == 0))
        .filter(|&g| g > 1);
    let t = time.elapsed_secs();
    if std::env::var("FL_ROUTPASS_STOP").is_ok() && t >= 30.0 && !*sent_on {
        *sent_on = true;
        let p = &mut groups.list[1];
        p.order = Some(crate::orders::Order::Move(Vec2::new(0.0, -60.0)));
        p.auto_order = true;
        p.reform = true;
        info!("[routpass-test] t={t:.0}s the passer is sent on past A");
    }
    let a = &groups.list[0];
    let rect = crate::formation::block_rect_at(a, a.anchor);
    let fwd = crate::formation::facing_dir(a.facing);
    let right = Vec2::new(fwd.y, -fwd.x);
    let (mut inside, mut cross_in, mut off, mut held) = (0usize, 0usize, 0.0f32, 0usize);
    for i in 0..units.len() {
        if units.death_t[i] != 0 {
            continue;
        }
        let p = Vec2::new(units.pos[i].x, units.pos[i].z);
        match units.group[i] as usize {
            0 => {
                off += (p - a.anchor - units.home[i]).dot(right).abs();
                held += 1;
            }
            1 if rect.distance(p) == 0.0 => inside += 1,
            g if Some(g) == cross_g && rect.distance(p) == 0.0 => cross_in += 1,
            _ => {}
        }
    }
    // The time from the first man of a passing regiment inside A to the
    // last one out, for the passer and for the crossing regiment
    // (`passes`, one each).
    let time_the_pass = |pass: &mut (Option<f32>, Option<f32>), inside: usize, name: &str| {
        let (first_in, last_out) = pass;
        if inside > 0 && first_in.is_none() {
            *first_in = Some(t);
        }
        if inside == 0
            && let Some(t0) = *first_in
            && last_out.is_none()
        {
            *last_out = Some(t);
            info!("[routpass-test] {name}: first in t={t0:.1}s, last out t={t:.1}s, {:.1} s", t - t0);
        }
    };
    let [pass, cross_pass] = &mut *passes;
    time_the_pass(pass, inside, "pass");
    if cross_g.is_some() {
        time_the_pass(cross_pass, cross_in, "cross pass");
    }
    if t < *next {
        return;
    }
    *next = t + 2.0;
    info!(
        "[routpass-test] t={t:.0}s A disorder {:.2} m, off file line {:.2} m, passer z {:.1}, {inside} inside, cross {cross_in} inside, nn min {:.2}",
        a.disorder,
        if held > 0 { off / held as f32 } else { 0.0 },
        groups.list[1].centroid.y,
        stats.nn_min,
    );
}
