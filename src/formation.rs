//! Rigid regiment formations: a slot generator writing into the existing
//! `units.home` column (the seam prepared in devlogs 0003/0018). A regiment
//! order stays ONE point; formations only change the per-unit offset from
//! it — the movement hot path is untouched.
//!
//! Shapes: `Rect` (rank-and-file grid with facing, the default) and `Blob`
//! (the pre-formations look: spawn-captured or regenerated loose cluster —
//! kept for low-morale levies later and for the FL_TEST_* geometry).
//! Spacing modes: Normal, Loose (missile dispersion later), Wall
//! (shieldwall for sword kinds, spearwall for spears — tight files).
//!
//! Slots are recomputed on EVENTS (new order, mode toggle, rally, close
//! ranks), never per tick: `reform` on `GroupData` requests it, and
//! `apply_reforms` runs before the sim step so fresh offsets take effect
//! the tick they were asked for.

use bevy::prelude::*;

use crate::orders::{GroupData, Groups};
use crate::units::{Units, hash01};

/// Base slot pitch inside a regiment block (matches the historical spawn
/// spacing; per-mode multipliers below).
pub const BASE_SPACING: f32 = 1.4;
/// Files-to-ranks bias of the default block (~2.2:1, wider than deep).
const DEFAULT_ASPECT: f32 = 2.2;
/// Close ranks when this fraction of the last-formed strength has fallen
/// (and at least 8 men, so skirmish dribble doesn't churn the grid).
const CLOSE_RANKS_FRAC: f32 = 0.06;
/// Ticks between close-ranks checks per regiment (staggered by index).
const CLOSE_RANKS_PERIOD: u32 = 90;
/// Ticks between close-ranks checks while ENGAGED (FL_RECTFIGHT, ~8 s):
/// casualties empty slots mid-fight and the block re-dresses where it
/// stands. Without the gate nobody dresses in contact (pre-existing
/// behavior, unchanged).
const ENGAGED_CLOSE_RANKS_PERIOD: u32 = 240;

/// FL_RECTFIGHT=1 (owner-gated): the M2TW melee model from the engine
/// research (devlog 0036) — four mechanisms, all men-level or existing
/// machinery. (1) FREEZE: an engaged attack order stops chasing the
/// target centroid; the frame anchors where contact happened instead
/// of dragging the slot grid through the enemy block. (2) PRESS:
/// soldiers of a fighting regiment pack shoulder-to-shoulder toward
/// the fight (META_PRESS pairs rest at the wall radius) — bodies stop
/// them, not a density rule. (3) SLOT MEMORY: unchanged — men always
/// keep their slots. (4) REFORM: when the fight ends the regiment
/// re-dresses where it stands (a discrete event, like the engine's
/// `reforming` state).
pub fn rectfight() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FL_RECTFIGHT").is_ok())
}

/// Formation shape of a regiment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FormShape {
    /// Rank-and-file grid with facing — the default battle formation.
    Rect,
    /// Loose cluster: spawn-captured offsets (test geometry) or a
    /// regenerated jittered disc (B toggle). Facing is meaningless.
    Blob,
}

/// Slot density of a Rect formation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FormSpacing {
    Normal,
    /// Open order: spread against missiles (arrows later), weak in melee.
    Loose,
    /// Shieldwall (sword kinds) / spearwall (spears): tight files, braced.
    Wall,
}

impl FormSpacing {
    /// (lateral, depth) slot pitch in meters. Wall matches the tightened
    /// separation rest distance (movement.rs WALL_SEP_RADIUS) — slots the
    /// physics refuses to hold are lies.
    pub fn pitch(self) -> Vec2 {
        match self {
            FormSpacing::Normal => Vec2::splat(BASE_SPACING),
            FormSpacing::Loose => Vec2::splat(BASE_SPACING * 1.9),
            FormSpacing::Wall => Vec2::new(1.05, 1.15),
        }
    }
}

/// Default file count for a regiment of `n` (the ~2.2:1 block the armies
/// have always spawned with).
pub fn default_files(n: usize) -> u32 {
    ((n as f32 * DEFAULT_ASPECT).sqrt().ceil() as u32).max(1)
}

/// Facing direction (unit yaw convention: 0 = +Z).
#[inline]
pub fn facing_dir(facing: f32) -> Vec2 {
    let (s, c) = facing.sin_cos();
    Vec2::new(s, c)
}

/// Yaw angle of a direction (inverse of `facing_dir`).
#[inline]
pub fn facing_of(dir: Vec2) -> f32 {
    dir.x.atan2(dir.y)
}

/// Row-major LOCAL slot offsets (x = lateral, y = forward) for `n` units
/// in `files` columns: front rank first, left to right, partial last rank
/// centered, block centered on the origin. The one grid used by slot
/// assignment AND the drag-order placement preview — what you see is
/// where they stand.
pub fn slot_offsets(n: usize, files: usize, pitch: Vec2) -> Vec<Vec2> {
    let files = files.clamp(1, n.max(1));
    let ranks = n.div_ceil(files);
    let half_depth = (ranks - 1) as f32 / 2.0;
    let mut out = Vec::with_capacity(n);
    for r in 0..ranks {
        let in_row = files.min(n - r * files);
        let half_w = (in_row - 1) as f32 / 2.0;
        let fwd = (half_depth - r as f32) * pitch.y;
        for c in 0..in_row {
            out.push(Vec2::new((c as f32 - half_w) * pitch.x, fwd));
        }
    }
    out
}

/// Rewrite the `home` slot offsets of regiment `g` for its current shape,
/// files, spacing, and facing. Rect: row-major grid centered on the
/// anchor, front rank toward `facing`, partial last rank centered. Units
/// are assigned to slots front-to-back / left-to-right in their CURRENT
/// relative order, so a reform never marches men through each other.
pub fn assign_slots(units: &mut Units, g: u32, gd: &mut GroupData) {
    // (unit index, forward coord, lateral coord) relative to the anchor.
    let fwd = facing_dir(gd.facing);
    let right = Vec2::new(fwd.y, -fwd.x);
    let mut members: Vec<(u32, f32, f32)> = Vec::new();
    for i in 0..units.len() {
        if units.group[i] == g && units.death_t[i] == 0 {
            let d = Vec2::new(units.pos[i].x, units.pos[i].z) - gd.anchor;
            members.push((i as u32, d.dot(fwd), d.dot(right)));
        }
    }
    let n = members.len();
    if n == 0 {
        return;
    }

    match gd.shape {
        FormShape::Blob => {
            // Jittered disc sized to hold n at roughly normal density.
            let r_max = (n as f32).sqrt() * BASE_SPACING * 0.55;
            for (k, m) in members.iter().enumerate() {
                let seed = g.wrapping_mul(0x9E37_79B1) ^ (k as u32);
                let a = hash01(seed.wrapping_mul(3) + 1) * std::f32::consts::TAU;
                let r = r_max * hash01(seed.wrapping_mul(3) + 2).sqrt();
                units.home[m.0 as usize] = Vec2::new(r * a.cos(), r * a.sin());
            }
        }
        FormShape::Rect => {
            let files = (gd.files.max(1) as usize).min(n);
            let pitch = gd.spacing.pitch();
            let slots = slot_offsets(n, files, pitch);
            // Front-to-back, then left-to-right within each rank: the
            // greedy crossing-minimizer, matching the slots' row-major
            // order.
            members.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            for row in members.chunks_mut(files) {
                row.sort_unstable_by(|a, b| a.2.total_cmp(&b.2));
            }
            for (m, slot) in members.iter().zip(&slots) {
                units.home[m.0 as usize] = right * slot.x + fwd * slot.y;
            }
        }
    }
    gd.count_at_reform = n;
    gd.reform = false;
}

/// Wall flavor of a regiment (spacing == Wall and still fighting):
/// 0 = none, 1 = shieldwall (sword kinds), 2 = spearwall (spears).
/// Shared by the sim damage model and the render wall pose.
#[inline]
pub fn wall_kind(gd: &GroupData) -> u8 {
    if gd.spacing != FormSpacing::Wall || gd.state.is_broken() {
        0
    } else if gd.kind == crate::unit_types::KIND_SPEAR {
        2
    } else {
        1
    }
}

pub struct FormationPlugin;

impl Plugin for FormationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            FixedUpdate,
            apply_reforms.before(crate::movement::step_sim),
        )
        .add_systems(Update, (formation_keys, test_form_script, rectfight_log));
    }
}

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
fn test_form_script(
    time: Res<Time>,
    units: Res<Units>,
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
            crate::orders::line_order(&mut groups, &picked, b, a);
            let rev_facing = groups.list[picked[0]].facing;
            crate::orders::line_order(&mut groups, &picked, a, b);
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

/// FL_RECTFIGHT diagnostic, every 4 s while anyone fights: mean
/// disorder (m off the slots) of engaged regiments — the
/// blocks-stay-blocks number. FL_LOG_DISORDER=1 fires it with the
/// gate off, for the A/B baseline.
fn rectfight_log(groups: Res<Groups>, time: Res<Time>, mut next: Local<f32>) {
    if !rectfight() && std::env::var("FL_LOG_DISORDER").is_err() {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 4.0;
    let (mut disorder, mut regs) = (0.0f32, 0usize);
    for gd in groups.list.iter().filter(|gd| gd.engaged && gd.count > 0) {
        disorder += gd.disorder;
        regs += 1;
    }
    if regs > 0 {
        info!(
            "[rectfight] t={t:.0}s {regs} engaged regiments, mean disorder {:.2} m",
            disorder / regs as f32
        );
    }
}

/// Formation hotkeys for the selection — F: wall on/off (shieldwall for
/// sword regiments, spearwall for spears), L: loose order on/off,
/// B: blob <-> ranks (the pre-formation look, kept for levies).
fn formation_keys(
    keys: Res<ButtonInput<KeyCode>>,
    selection: Res<crate::orders::Selection>,
    mut groups: ResMut<Groups>,
) {
    let wall = keys.just_pressed(KeyCode::KeyF);
    let loose = keys.just_pressed(KeyCode::KeyL);
    let blob = keys.just_pressed(KeyCode::KeyB);
    let hold = keys.just_pressed(KeyCode::KeyH);
    if !(wall || loose || blob || hold) || selection.count_units == 0 {
        return;
    }
    let picked: Vec<usize> = selection
        .regiments
        .iter()
        .enumerate()
        .filter(|(g, s)| {
            **s && {
                let gd = &groups.list[*g];
                gd.count > 0 && !gd.state.is_broken()
            }
        })
        .map(|(g, _)| g)
        .collect();
    if picked.is_empty() {
        return;
    }

    if wall || loose {
        let want = if wall { FormSpacing::Wall } else { FormSpacing::Loose };
        // Toggle as a set: if anyone is not yet in the mode, everyone
        // enters it; if all are, everyone falls back to close order.
        let on = picked.iter().any(|&g| groups.list[g].spacing != want);
        let mut shield = 0;
        let mut spear = 0;
        for &g in &picked {
            let gd = &mut groups.list[g];
            gd.spacing = if on { want } else { FormSpacing::Normal };
            gd.shape = FormShape::Rect;
            if gd.files == 0 {
                gd.files = default_files(gd.count);
            }
            gd.reform = true;
            match wall_kind(gd) {
                1 => shield += 1,
                2 => spear += 1,
                _ => {}
            }
        }
        if !on {
            info!("{} regiments back to close order", picked.len());
        } else if wall {
            info!("wall formed: {shield} shieldwall, {spear} spearwall regiments");
        } else {
            info!("{} regiments to LOOSE order", picked.len());
        }
    }

    if blob {
        let on = picked.iter().any(|&g| groups.list[g].shape != FormShape::Blob);
        for &g in &picked {
            let gd = &mut groups.list[g];
            gd.shape = if on { FormShape::Blob } else { FormShape::Rect };
            if gd.files == 0 {
                gd.files = default_files(gd.count);
            }
            gd.reform = true;
        }
        info!(
            "{} regiments to {}",
            picked.len(),
            if on { "BLOB (mob)" } else { "ranks" }
        );
    }

    if hold {
        // Hold position (defend) vs at ease: held regiments never chase
        // and never engage on their own — they fight what steps into
        // reach and nothing else.
        let on = picked.iter().any(|&g| !groups.list[g].hold);
        for &g in &picked {
            groups.list[g].hold = on;
        }
        info!(
            "{} regiments {}",
            picked.len(),
            if on { "HOLD POSITION" } else { "AT EASE" }
        );
    }
}

/// Consume `reform` requests and run the close-ranks check: a Rect
/// regiment that lost enough men since it last formed compacts its grid
/// (only out of contact — nobody dresses ranks mid-melee).
fn apply_reforms(mut units: ResMut<Units>, mut groups: ResMut<Groups>, mut tick: Local<u32>) {
    *tick = tick.wrapping_add(1);
    for g in 0..groups.list.len() {
        let gd = &mut groups.list[g];
        if gd.count == 0 || gd.state.is_broken() {
            gd.reform = false;
            continue;
        }
        let mut do_reform = gd.reform;
        if !do_reform && gd.shape == FormShape::Rect && (!gd.engaged || rectfight()) {
            let period = if gd.engaged {
                ENGAGED_CLOSE_RANKS_PERIOD
            } else {
                CLOSE_RANKS_PERIOD
            };
            if (*tick).wrapping_add(g as u32).is_multiple_of(period) {
                let lost = gd.count_at_reform.saturating_sub(gd.count);
                if lost >= 8.max((gd.count_at_reform as f32 * CLOSE_RANKS_FRAC) as usize) {
                    do_reform = true;
                    // The anchor stays put on purpose: for an attacker
                    // it is the frozen destination (the press must not
                    // retreat to wherever the block currently stands),
                    // for a defender it is the ground he holds — men
                    // shoved off it lean back toward their line.
                    info!(
                        "regiment {g} closes ranks ({} men{})",
                        gd.count,
                        if gd.engaged { ", engaged" } else { "" }
                    );
                }
            }
        }
        if do_reform {
            let gd = &mut groups.list[g];
            assign_slots(&mut units, g as u32, gd);
        }
    }
}
