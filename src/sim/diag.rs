//! The tick's diagnostics: the FL_LOG_STEP cost line, the neighbour
//! audit, the FL_HASH fingerprint that gates refactors, and the spike
//! line that names a slow tick.

use bevy::math::Vec3Swizzles;
use bevy::prelude::*;
use bevy::tasks::ComputeTaskPool;
use std::time::Instant;

use super::job::CHUNK;
use super::SimStats;
use crate::spatial::SpatialGrid;
use crate::units::Units;

/// FL_LOG_STEP=1: the mean kernel and grid time over each 150 ticks
/// (5 s), for cost comparisons between builds and knobs.
pub(super) fn log_step(stats: &mut SimStats, step_ms: f32, grid_ms: f32, units: usize) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var("FL_LOG_STEP").is_ok()) {
        stats.log_step_sum += step_ms as f64;
        stats.log_grid_sum += grid_ms as f64;
        stats.log_step_n += 1;
        if stats.log_step_n == 150 {
            info!(
                "[step] mean step {:.2} ms, grid {:.2} ms, {} units",
                stats.log_step_sum / 150.0,
                stats.log_grid_sum / 150.0,
                units
            );
            stats.log_step_sum = 0.0;
            stats.log_grid_sum = 0.0;
            stats.log_step_n = 0;
        }
    }
}

/// The overlap and movement audit over the full population: the
/// smallest and mean nearest-neighbour distance and the mean per-tick
/// displacement, for the overlay. Parallel over chunks.
pub(super) fn neighbour_audit(stats: &mut SimStats, grid: &SpatialGrid, pos: &[Vec3], pos_prev: &[Vec3]) {
{
    let _span = info_span!("nn_audit").entered();
    let audit_t0 = Instant::now();
    let pos_now = pos;
    let partials: Vec<(f32, f64, u64, f64)> = ComputeTaskPool::get().scope(|scope| {
        for (ci, chunk) in pos_prev.chunks(CHUNK * 8).enumerate() {
            let start = ci * CHUNK * 8;
            scope.spawn(async move {
                let mut min_d2 = f32::MAX;
                let mut sum_d = 0.0f64;
                let mut counted = 0u64;
                let mut sum_disp = 0.0f64;
                for (j, p) in chunk.iter().enumerate() {
                    let i = start + j;
                    let p2 = p.xz();
                    sum_disp += p2.distance(pos_now[i].xz()) as f64;
                    let mut best = f32::MAX;
                    grid.for_each_candidate(p2, crate::sim::soldier::SEP_RADIUS, |o| {
                        if o.idx as usize != i {
                            best = best.min(p2.distance_squared(o.xz()));
                        }
                    });
                    if best < f32::MAX {
                        min_d2 = min_d2.min(best);
                        sum_d += best.sqrt() as f64;
                        counted += 1;
                    }
                }
                (min_d2, sum_d, counted, sum_disp)
            });
        }
    });
    let min_d2 = partials.iter().fold(f32::MAX, |m, p| m.min(p.0));
    let sum_d: f64 = partials.iter().map(|p| p.1).sum();
    let counted: u64 = partials.iter().map(|p| p.2).sum();
    let sum_disp: f64 = partials.iter().map(|p| p.3).sum();
    stats.nn_min = min_d2.sqrt();
    stats.nn_avg = if counted > 0 {
        (sum_d / counted as f64) as f32
    } else {
        0.0
    };
    stats.move_avg = (sum_disp / pos_prev.len().max(1) as f64) as f32;
    stats.audit_ms = audit_t0.elapsed().as_secs_f32() * 1000.0;
}

}

/// FL_HASH: an FNV-1a fingerprint of the sim state on a fixed tick
/// cadence (every 150 ticks, or every FL_HASH=n ticks), the bit-identity
/// gate for refactors and optimisations. Log samplers ride wall time, so
/// the same binary logs different casualty digits run to run and log
/// diffs prove nothing; equal hashes at equal ticks do.
pub(super) fn fingerprint(tick: u32, units: &Units) {
    let Units {
        pos,
        yaw,
        hp,
        swing,
        swing_t,
        death_t,
        ammo,
        ..
    } = units;
static HASH_EVERY: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
let hash_every = *HASH_EVERY.get_or_init(|| {
    let n: u32 = std::env::var("FL_HASH").ok()?.parse().unwrap_or(0);
    Some(if n > 1 { n } else { 150 })
});
if let Some(every) = hash_every
    && tick.is_multiple_of(every)
{
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for i in 0..pos.len() {
        for v in [
            pos[i].x.to_bits(),
            pos[i].z.to_bits(),
            yaw[i].to_bits(),
            hp[i].to_bits(),
            u32::from_le_bytes([death_t[i], swing[i], swing_t[i], ammo[i]]),
        ] {
            h ^= v as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    info!("[hash] tick {} n {} state {:016x}", tick, pos.len(), h);
}

}

/// Name any tick that blows past the norm, with its component costs:
/// lag hunting works on facts. The audit runs every 60 ticks, so an
/// audit tick shows its fresh cost here.
pub(super) fn spike_line(stats: &SimStats, tick: u32, units: usize) {
let audit_now = tick.is_multiple_of(60);
if stats.step_ms + stats.grid_ms + if audit_now { stats.audit_ms } else { 0.0 } > 14.0 {
    info!(
        "[spike] step {:.1} + grid {:.1}{} ms ({} damage events, {} units)",
        stats.step_ms,
        stats.grid_ms,
        if audit_now {
            format!(" + AUDIT {:.1}", stats.audit_ms)
        } else {
            String::new()
        },
        stats.events,
        units,
    );
}

}
