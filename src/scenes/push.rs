//! FL_TEST_PUSH: a mass sent past a line through its gaps, and its log.

use bevy::prelude::*;

use crate::orders::Groups;
use crate::regiments::spawn_regiment;
use crate::terrain::Terrain;
use crate::unit_types::KIND_LIGHT;
use crate::units::Units;
use super::SPACING;

/// FL_TEST_PUSH=1: a lane behind a gap. Three deep orange regiments of
/// FL_PUSH_LINE_SIZE men (384, 23 ranks) hold in a line in the middle of
/// the field with FL_PUSH_GAP metres of air between them (3), facing a blue mass of
/// FL_PUSH_ACROSS by FL_PUSH_DEEP regiments (4 by 3) of FL_PUSH_SIZE men
/// (128) that stands 30 m in front of them under a Move order to a point
/// 60 m behind them: the mass clogs against the line and presses past it
/// through the gaps while the fight at the rim kills. Every regiment is
/// as wide as a mass regiment, and the line never breaks, so the scene
/// lasts; FL_COMBAT_SCALE slows the killing further. Acceptance: the mass
/// packs against the line, gaps open where men fall, and the lane behind
/// a gap flows through instead of surging and stopping as one body.
pub(crate) fn spawn_push_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let size: usize = crate::util::env_or("FL_PUSH_SIZE", 128).max(1);
    let line_size: usize = crate::util::env_or("FL_PUSH_LINE_SIZE", 3 * size).max(1);
    let gap: f32 = crate::util::env_or("FL_PUSH_GAP", 3.0);
    let across: usize = crate::util::env_or("FL_PUSH_ACROSS", 4).max(1);
    let deep: usize = crate::util::env_or("FL_PUSH_DEEP", 3).max(1);
    let cols = ((size as f32 * 2.2).sqrt().ceil() as usize).max(1);
    let width = cols as f32 * SPACING;
    let depth = size.div_ceil(cols) as f32 * SPACING;
    let mut list = Vec::new();
    // The gap is air between the outer men of two regiments: the edge
    // files stand (cols - 1) pitches apart, and a man is a body distance
    // across.
    let line_pitch = (cols - 1) as f32 * SPACING + crate::sim::soldier::body_distance() + gap;
    for k in 0..3 {
        let x = (k as f32 - 1.0) * line_pitch;
        spawn_regiment(units, terrain, &mut list, 1, KIND_LIGHT, Vec2::new(x, 0.0), line_size, 1.0);
        let g = list.len() - 1;
        list[g].files = cols as u32;
        // Spawned at the wide default layout; set every man on his slot
        // in the deep one, since the mass is on the line in seconds.
        for i in crate::formation::assign_slots(units, g as u32, &mut list[g]) {
            let i = i as usize;
            let at = list[g].anchor + units.home[i];
            let y = terrain.height_at(at.x, at.y) + crate::unit_types::half_height(KIND_LIGHT as usize);
            units.pos[i] = Vec3::new(at.x, y, at.y);
            units.pos_prev[i] = units.pos[i];
        }
        list[g].hold = true;
        list[g].steadfast = true;
    }
    for row in 0..deep {
        for col in 0..across {
            let x = (col as f32 - (across as f32 - 1.0) / 2.0) * (width + 4.0);
            let z = 30.0 + depth / 2.0 + row as f32 * (depth + 4.0);
            spawn_regiment(units, terrain, &mut list, 0, KIND_LIGHT, Vec2::new(x, z), size, -1.0);
            let g = list.len() - 1;
            list[g].order = Some(crate::orders::Order::Move(Vec2::new(x, -60.0)));
            list[g].auto_order = true;
        }
    }
    groups.list = list;
    info!("[push-test] {} blue regiments of {size} sent past 3 orange regiments of {line_size} holding {gap} m apart", across * deep);
}

/// FL_TEST_PUSH bookkeeping every 4 s: the line's men and centre, the
/// mass's men, and how many of them are past the line.
pub(super) fn push_test_log(
    units: Res<Units>,
    groups: Res<crate::orders::Groups>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    if std::env::var("FL_TEST_PUSH").is_err() || groups.list.len() < 4 {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 4.0;
    let line: Vec<_> = groups.list.iter().take(3).collect();
    let line_men: usize = line.iter().map(|g| g.count).sum();
    let line_z = line.iter().map(|g| g.centroid.y).sum::<f32>() / 3.0;
    let mass_men: usize = groups.list.iter().skip(3).map(|g| g.count).sum();
    let past = (0..units.len())
        .filter(|&i| units.team[i] == 0 && units.death_t[i] == 0 && units.pos[i].z < line_z - 6.0)
        .count();
    info!("[push-test] t={t:.0}s line {line_men} men at z {line_z:.1}, mass {mass_men} men, {past} of them past the line");
}
