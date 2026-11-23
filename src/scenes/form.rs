//! FL_TEST_FORM: the drag-order line and the spacing modes, scripted.

use bevy::prelude::*;

use crate::formation::FormSpacing;
use crate::orders::Groups;
use crate::units::Units;

/// Mean living-unit distance to slot (`anchor + home`) — how ON their
/// marks a regiment stands.
fn slot_error(units: &Units, groups: &Groups, g: usize) -> f32 {
    let gd = &groups.list[g];
    let (mut sum, mut n) = (0.0f32, 0usize);
    for i in 0..units.len() {
        if units.group[i] as usize == g && units.death_t[i] == 0 {
            let p = Vec2::new(units.pos[i].x, units.pos[i].z);
            sum += p.distance(gd.anchor + units.home[i]);
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f32 }
}

/// Mean nearest-neighbor distance within a regiment (brute force — test
/// only). The spacing-mode metric: wall < normal < loose.
fn regiment_nn(units: &Units, g: usize) -> f32 {
    let pts: Vec<Vec2> = (0..units.len())
        .filter(|&i| units.group[i] as usize == g && units.death_t[i] == 0)
        .map(|i| Vec2::new(units.pos[i].x, units.pos[i].z))
        .collect();
    if pts.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for (k, p) in pts.iter().enumerate() {
        let mut best = f32::MAX;
        for (j, q) in pts.iter().enumerate() {
            if j != k {
                best = best.min(p.distance_squared(*q));
            }
        }
        sum += best.sqrt();
    }
    sum / pts.len() as f32
}

/// FL_TEST_FORM=1: the drag-order acceptance, driven through the SAME
/// line_order path the mouse uses. Stage 0: draw a 120 m line ahead of
/// four blue regiments. Stage 1 (all arrived): mean slot error must be
/// small (units stand ON their previewed marks) and the block extents
/// must match files x pitch. Then one regiment forms the wall, one goes
/// loose. Stage 2 (+8 s): within-regiment nearest-neighbor spacing must
/// order wall < normal < loose.
pub(super) fn test_form_script(
    time: Res<Time>,
    units: Res<Units>,
    terrain: Res<crate::terrain::Terrain>,
    mut groups: ResMut<Groups>,
    mut stage: Local<u32>,
    mut watch: Local<Vec<usize>>,
    mut settle_at: Local<f32>,
) {
    if std::env::var("FL_TEST_FORM").is_err() {
        return;
    }
    let t = time.elapsed_secs();
    match *stage {
        0 if t > 3.0 => {
            let picked: Vec<usize> = groups
                .list
                .iter()
                .enumerate()
                .filter(|(_, gd)| gd.team == 0 && gd.count > 0)
                .take(4)
                .map(|(g, _)| g)
                .collect();
            if picked.len() < 4 {
                warn!("[form-test] needs >= 4 blue regiments");
                *stage = 99;
                return;
            }
            let mean: Vec2 = picked.iter().map(|&g| groups.list[g].centroid).sum::<Vec2>()
                / picked.len() as f32;
            // A 120 m line 45 m toward the enemy, drawn left to right.
            let a = mean + Vec2::new(-60.0, 45.0);
            let b = mean + Vec2::new(60.0, 45.0);
            // Drag-direction invariance: the same line drawn with the
            // opposite hand must face the same way (facing follows the
            // enemy side of the line, not the drag direction). The
            // reversed order is immediately overwritten by the real one.
            crate::orders::line_order(&mut groups, &terrain, &picked, b, a);
            let rev_facing = groups.list[picked[0]].facing;
            crate::orders::line_order(&mut groups, &terrain, &picked, a, b);
            let fwd_facing = groups.list[picked[0]].facing;
            if (rev_facing - fwd_facing).abs() < 0.01 {
                info!(
                    "[form-test] drag invariance: fwd {fwd_facing:.2} == rev {rev_facing:.2} -> OK"
                );
            } else {
                warn!(
                    "[form-test] drag invariance FAILED: fwd {fwd_facing:.2} vs rev {rev_facing:.2}"
                );
            }
            for &g in &picked {
                info!(
                    "[form-test] regiment {g}: files {} facing {:.2}",
                    groups.list[g].files, groups.list[g].facing
                );
            }
            *watch = picked;
            *stage = 1;
        }
        1 if !watch.is_empty()
            && watch.iter().all(|&g| groups.list[g].order.is_none()) =>
        {
            // "Arrived" is a centroid predicate; the tail of a deep
            // column is still tens of meters out when it fires — give
            // the ranks time to pour in and dress before measuring.
            *settle_at = t + 15.0;
            *stage = 2;
        }
        2 if t > *settle_at => {
            for &g in watch.iter() {
                let gd = &groups.list[g];
                let err = slot_error(&units, &groups, g);
                // Mean angular error of unit yaw vs the ordered facing:
                // a settled formation must LOOK the way it was told to.
                let (mut yerr, mut n) = (0.0f32, 0usize);
                for i in 0..units.len() {
                    if units.group[i] as usize == g && units.death_t[i] == 0 {
                        let d = (units.yaw[i] - gd.facing + std::f32::consts::PI)
                            .rem_euclid(std::f32::consts::TAU)
                            - std::f32::consts::PI;
                        yerr += d.abs();
                        n += 1;
                    }
                }
                let yerr = if n > 0 { yerr / n as f32 } else { 0.0 };
                info!(
                    "[form-test] regiment {g} settled: slot err {err:.2} m, facing err {yerr:.2} rad ({} men, files {}) -> {}",
                    gd.count,
                    gd.files,
                    if err < 1.5 && yerr < 0.3 { "OK" } else { "FAIL" }
                );
            }
            // Spacing modes: wall on the first, loose on the second.
            let (w, l) = (watch[0], watch[1]);
            groups.list[w].spacing = FormSpacing::Wall;
            groups.list[w].reform = true;
            groups.list[l].spacing = FormSpacing::Loose;
            groups.list[l].reform = true;
            info!("[form-test] regiment {w} -> wall, regiment {l} -> loose");
            *settle_at = t + 10.0;
            *stage = 3;
        }
        3 if t > *settle_at => {
            let nn_wall = regiment_nn(&units, watch[0]);
            let nn_loose = regiment_nn(&units, watch[1]);
            let nn_normal = regiment_nn(&units, watch[2]);
            let ordered = nn_wall < nn_normal - 0.1 && nn_normal < nn_loose - 0.2;
            info!(
                "[form-test] spacing nn: wall {nn_wall:.2} < normal {nn_normal:.2} < loose {nn_loose:.2} -> {}",
                if ordered { "OK" } else { "FAIL" }
            );
            *stage = 4;
        }
        _ => {}
    }
}
