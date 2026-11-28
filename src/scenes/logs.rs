//! The diagnostic logs of any battle: FL_LOG_STRAGGLER, FL_DIAG_MELEE,
//! FL_LOG_ROOM and FL_LOG_DISORDER.

use bevy::prelude::*;

use crate::game_state::{BattleClock, PerBattle};
use crate::orders::Groups;
use crate::units::Units;

/// FL_LOG_STRAGGLER=1: every 2 s, every man standing inside another
/// friendly regiment's held block more than 5 m from his own place, with his
/// state: his regiment's order, the host block open or closed and how
/// many of its men stand within half a metre of their places, his
/// speed, his stuck count, his try, his squeeze, his aim, his pose and
/// turn levels, and his six nearest neighbours with their regiments,
/// distances and how far each stands from his own place. It shows why
/// a man is left behind in play.
pub(super) fn straggler_log(units: Res<Units>, groups: Res<crate::orders::Groups>, clock: Res<BattleClock>, mut next: Local<PerBattle<f32>>) {
    if std::env::var("FL_LOG_STRAGGLER").is_err() {
        return;
    }
    let next = next.of(&clock);
    let t = clock.secs();
    if t < *next {
        return;
    }
    *next = t + 2.0;
    let ring_aims = crate::sim::soldier::RING_AIMS.swap(0, std::sync::atomic::Ordering::Relaxed);
    if ring_aims > 0 {
        info!("[straggler] t={t:.0}s {ring_aims} ring aims in the last 2 s");
    }
    // The held blocks: regiments with no order and living men.
    let mut blocks: Vec<(usize, u8, crate::formation::BlockRect)> = Vec::new();
    for (h, gd) in groups.list.iter().enumerate() {
        if gd.order.is_none() && gd.count > 0 {
            blocks.push((h, gd.team, crate::formation::block_rect_at(gd, gd.anchor)));
        }
    }
    if blocks.is_empty() {
        return;
    }
    let mut found = 0usize;
    for i in 0..units.len() {
        if units.death_t[i] != 0 {
            continue;
        }
        let g = units.group[i] as usize;
        let gd = &groups.list[g];
        let p = Vec2::new(units.pos[i].x, units.pos[i].z);
        let slot = gd.anchor + units.home[i];
        if slot.distance_squared(p) < 25.0 {
            continue;
        }
        let Some(&(h, _, _)) = blocks.iter().find(|(h, team, rect)| *h != g && *team == gd.team && rect.distance(p) == 0.0) else {
            continue;
        };
        found += 1;
        if found > 12 {
            continue;
        }
        // His six nearest neighbours within 2.5 m, by brute force (a
        // handful of men): each one's regiment, whether it holds, his
        // distance, and how far he stands from his own place (a dressed
        // block reads near zero, a heap reads a metre or more). Then the
        // host block's own dressing: the share of its men within half a
        // metre of their places.
        let mut near: Vec<(f32, usize)> = Vec::new();
        for j in 0..units.len() {
            if j == i || units.death_t[j] != 0 {
                continue;
            }
            let q = Vec2::new(units.pos[j].x, units.pos[j].z);
            let d = p.distance(q);
            if d < 2.5 {
                near.push((d, j));
            }
        }
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        let nb: Vec<String> = near
            .iter()
            .take(6)
            .map(|(d, j)| {
                let jg = units.group[*j] as usize;
                let q = Vec2::new(units.pos[*j].x, units.pos[*j].z);
                let off = (groups.list[jg].anchor + units.home[*j]).distance(q);
                format!("g{jg}{}@{d:.2}s{off:.1}", if groups.list[jg].order.is_none() { "h" } else { "" })
            })
            .collect();
        let hd = &groups.list[h];
        let (mut host_n, mut host_near) = (0u32, 0u32);
        for j in 0..units.len() {
            if units.group[j] as usize == h && units.death_t[j] == 0 {
                host_n += 1;
                let q = Vec2::new(units.pos[j].x, units.pos[j].z);
                host_near += u32::from((hd.anchor + units.home[j]).distance(q) < 0.5);
            }
        }
        let v = units.stuck[i];
        let aim = units.aim[i];
        info!(
            "[straggler] t={t:.0}s man {i} of g{g} ({}) inside g{h}'s {} block ({} of {} men within 0.5 m of their places) at ({:.1},{:.1}): slot {:.1} m off, v {:.2} m/s, stuck {} try {} squeeze {} aim {} pose {} turn {} | {}",
            if gd.order.is_some() { "ordered" } else { "arrived" },
            if hd.spacing == crate::formation::FormSpacing::Open { "open" } else { "closed" },
            host_near,
            host_n,
            p.x,
            p.y,
            slot.distance(p),
            units.vel[i].xz().length(),
            crate::sim::soldier::stuck_count(v),
            crate::sim::soldier::stuck_trying(v),
            crate::sim::soldier::squeezing(v),
            if aim.x.is_finite() { "yes" } else { "no" },
            crate::sim::soldier::pose_level(units.pose[i]),
            crate::sim::soldier::turn_level(units.pose[i]),
            nb.join(" ")
        );
    }
    if found > 0 {
        info!("[straggler] t={t:.0}s {found} men inside a friend's held block more than 5 m from their places");
    }
}

/// FL_DIAG_MELEE=1: per-rank ground truth for the first engaged
/// regiment, every 4 s — each rank's headcount, mean distance to the
/// nearest living enemy, the share of it inside its own weapon reach,
/// and the share mid-windup. The log that answers "why does only the
/// first row fight" with data instead of theory.
pub(super) fn melee_diag_log(
    units: Res<Units>,
    groups: Res<crate::orders::Groups>,
    grid: Res<crate::spatial::SpatialGrid>,
    clock: Res<BattleClock>,
    mut next: Local<PerBattle<f32>>,
) {
    if std::env::var("FL_DIAG_MELEE").is_err() {
        return;
    }
    let next = next.of(&clock);
    let t = clock.secs();
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
    let pitch = crate::formation::grid(gd).2.y;
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

/// Regiments alive up to which the room log prints a line for each.
const ROOM_LINES: usize = 8;

/// How far the room log looks for a man's nearest comrade and enemy.
const ROOM_PROBE: f32 = 6.0;

/// FL_LOG_ROOM=1: every 5 s, the room the men of each regiment have.
/// Per regiment: each man's distance to the nearest living comrade of his
/// own regiment (10th percentile, median, 90th), the median distance to
/// the nearest enemy, the share of men with an enemy within 3 m, and the
/// box the men's centres cover along the regiment's facing with its
/// ground per man, and the shares of men out of formation and moving
/// faster than 0.4 m/s. One line for each regiment while few are alive, and
/// always a line over all regiments by state: fighting (engaged), moving
/// (under an order) and standing.
pub(super) fn room_log(
    units: Res<Units>,
    groups: Res<Groups>,
    grid: Res<crate::spatial::SpatialGrid>,
    clock: Res<BattleClock>,
    mut next: Local<PerBattle<f32>>,
) {
    if std::env::var("FL_LOG_ROOM").is_err() {
        return;
    }
    let next = next.of(&clock);
    let t = clock.secs();
    if t < *next {
        return;
    }
    *next = t + 5.0;
    struct Room {
        comrade: Vec<f32>,
        enemy: Vec<f32>,
        near_enemy: u32,
        men: u32,
        out_form: u32,
        walking: u32,
        lo: Vec2,
        hi: Vec2,
    }
    let mut rooms: Vec<Room> = groups
        .list
        .iter()
        .map(|_| Room {
            comrade: Vec::new(),
            enemy: Vec::new(),
            near_enemy: 0,
            men: 0,
            out_form: 0,
            walking: 0,
            lo: Vec2::MAX,
            hi: Vec2::MIN,
        })
        .collect();
    for i in 0..units.len() {
        if units.death_t[i] != 0 {
            continue;
        }
        let g = units.group[i] as usize;
        let gd = &groups.list[g];
        let p = Vec2::new(units.pos[i].x, units.pos[i].z);
        let team_bit = (units.team[i] as u32) * crate::spatial::META_TEAM;
        let (mut comrade, mut enemy) = (f32::MAX, f32::MAX);
        grid.for_each_candidate(p, ROOM_PROBE, |o| {
            if o.idx as usize == i || (o.meta & crate::spatial::META_DYING) != 0 {
                return;
            }
            let d2 = p.distance_squared(o.xz());
            if (o.meta & crate::spatial::META_TEAM) != team_bit {
                enemy = enemy.min(d2);
            } else if crate::spatial::meta_group(o.meta) == g {
                comrade = comrade.min(d2);
            }
        });
        let room = &mut rooms[g];
        room.men += 1;
        room.out_form += units.out_form[i] as u32;
        room.walking += (units.vel[i].length_squared() > 0.16) as u32;
        if comrade < ROOM_PROBE * ROOM_PROBE {
            room.comrade.push(comrade.sqrt());
        }
        if enemy < ROOM_PROBE * ROOM_PROBE {
            room.enemy.push(enemy.sqrt());
            room.near_enemy += (enemy < 9.0) as u32;
        }
        let fwd = crate::formation::facing_dir(gd.facing);
        let along = Vec2::new(p.dot(Vec2::new(fwd.y, -fwd.x)), p.dot(fwd));
        room.lo = room.lo.min(along);
        room.hi = room.hi.max(along);
    }
    // The value at share `q` of a sorted list.
    let at = |v: &[f32], q: f32| v[((v.len() - 1) as f32 * q).round() as usize];
    const STATES: [&str; 3] = ["fighting", "moving", "standing"];
    let mut by_state: [(Vec<f32>, Vec<f32>); 3] = Default::default();
    let alive = rooms.iter().filter(|r| r.men > 0).count();
    for (g, room) in rooms.iter_mut().enumerate() {
        if room.men < 2 || room.comrade.is_empty() {
            continue;
        }
        let gd = &groups.list[g];
        let state = if gd.engaged {
            0
        } else if gd.order.is_some() {
            1
        } else {
            2
        };
        room.comrade.sort_unstable_by(f32::total_cmp);
        room.enemy.sort_unstable_by(f32::total_cmp);
        let median = at(&room.comrade, 0.5);
        let size = room.hi - room.lo;
        let ground = size.x * size.y / room.men as f32;
        by_state[state].0.push(median);
        by_state[state].1.push(ground);
        if alive <= ROOM_LINES {
            let enemy = if room.enemy.is_empty() {
                "none in sight".to_string()
            } else {
                format!(
                    "{:.2} m, {}% within 3 m of one",
                    at(&room.enemy, 0.5),
                    room.near_enemy * 100 / room.men
                )
            };
            info!(
                "[room] t={t:.0}s reg {g} {} {} n={}: comrade {:.2}/{:.2}/{:.2} m (p10/median/p90), enemy {enemy}, box {:.1} x {:.1} m, {ground:.2} m2 a man, {}% out of formation, {}% walking",
                crate::unit_types::kind_name(gd.kind),
                STATES[state],
                room.men,
                at(&room.comrade, 0.1),
                median,
                at(&room.comrade, 0.9),
                size.x,
                size.y,
                room.out_form * 100 / room.men,
                room.walking * 100 / room.men,
            );
        }
    }
    let mut line = String::new();
    for (state, (medians, grounds)) in by_state.iter_mut().enumerate() {
        if medians.is_empty() {
            continue;
        }
        medians.sort_unstable_by(f32::total_cmp);
        grounds.sort_unstable_by(f32::total_cmp);
        line.push_str(&format!(
            " | {} {}: comrade {:.2} m ({:.2} to {:.2}), {:.2} m2 a man",
            STATES[state],
            medians.len(),
            at(medians, 0.5),
            medians[0],
            medians[medians.len() - 1],
            at(grounds, 0.5),
        ));
    }
    if !line.is_empty() {
        info!("[room] t={t:.0}s regiments by state, the median of their medians (lowest to highest){line}");
    }
    let back = crate::sim::soldier::STEP_BACK_TICKS.swap(0, std::sync::atomic::Ordering::Relaxed);
    info!(
        "[room] t={t:.0}s stepping back from an enemy inside the fighting distance: {back} man-ticks in 5 s, {:.1} men a tick",
        back as f32 / 150.0
    );
}

/// FL_LOG_DISORDER=1: every 4 s while anyone fights, the mean
/// disorder (m off the slots) of engaged regiments — the
/// blocks-stay-blocks number.
pub(super) fn disorder_log(groups: Res<Groups>, clock: Res<BattleClock>, mut next: Local<PerBattle<f32>>) {
    if std::env::var("FL_LOG_DISORDER").is_err() {
        return;
    }
    let next = next.of(&clock);
    let t = clock.secs();
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
            "[disorder] t={t:.0}s {regs} engaged regiments, mean disorder {:.2} m",
            disorder / regs as f32
        );
    }
}
