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
    /// (lateral, depth) slot pitch in meters.
    pub fn pitch(self) -> Vec2 {
        match self {
            FormSpacing::Normal => Vec2::splat(BASE_SPACING),
            FormSpacing::Loose => Vec2::splat(BASE_SPACING * 1.9),
            FormSpacing::Wall => Vec2::new(1.0, 1.15),
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
            let ranks = n.div_ceil(files);
            let pitch = gd.spacing.pitch();
            // Front-to-back, then left-to-right within each rank: the
            // greedy crossing-minimizer.
            members.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let half_depth = (ranks - 1) as f32 / 2.0;
            for r in 0..ranks {
                let row = &mut members[r * files..((r + 1) * files).min(n)];
                row.sort_unstable_by(|a, b| a.2.total_cmp(&b.2));
                let in_row = row.len();
                let half_w = (in_row - 1) as f32 / 2.0;
                let z = (half_depth - r as f32) * pitch.y;
                for (c, m) in row.iter().enumerate() {
                    let x = (c as f32 - half_w) * pitch.x;
                    units.home[m.0 as usize] = right * x + fwd * z;
                }
            }
        }
    }
    gd.count_at_reform = n;
    gd.reform = false;
}

pub struct FormationPlugin;

impl Plugin for FormationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            FixedUpdate,
            apply_reforms.before(crate::movement::step_sim),
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
        if !do_reform
            && gd.shape == FormShape::Rect
            && !gd.engaged
            && (*tick).wrapping_add(g as u32).is_multiple_of(CLOSE_RANKS_PERIOD)
        {
            let lost = gd.count_at_reform.saturating_sub(gd.count);
            if lost >= 8.max((gd.count_at_reform as f32 * CLOSE_RANKS_FRAC) as usize) {
                do_reform = true;
                info!("regiment {g} closes ranks ({} men)", gd.count);
            }
        }
        if do_reform {
            let gd = &mut groups.list[g];
            assign_slots(&mut units, g as u32, gd);
        }
    }
}
