//! FL_LOG_SHAKE: the men who shake in place.

use bevy::prelude::*;

use crate::orders::Groups;
use crate::units::Units;

/// What `shake_log` keeps from tick to tick: each man's last step, and over
/// the report's window his turns and the length of the steps he turned on.
#[derive(Default)]
pub(super) struct Shake {
    tick: u32,
    next: f32,
    last: Vec<Vec2>,
    turns: Vec<u8>,
    step: Vec<f32>,
    /// For the wander count: each man's place when its window began, the
    /// ground he has covered since, and the shake windows reported so far.
    wstart: Vec<Vec2>,
    path: Vec<f32>,
    windows: u32,
}

/// A man wanders when he covers WANDER_PATH metres over WANDER_WINDOWS
/// shake windows and ends within WANDER_NET metres of where he began: the
/// man who goes to an enemy and back to his slot, over and over (devlog
/// 0212), whom the turn count misses because he turns about once a second.
const WANDER_WINDOWS: u32 = 2;
const WANDER_PATH: f32 = 4.0;
const WANDER_NET: f32 = 1.5;

/// A step shorter than this is no step, in metres.
const SHAKE_STEP: f32 = 0.003;
/// Seconds per report, and the turns within it that make a man a shaker
/// and a hard one.
const SHAKE_WINDOW: f32 = 2.0;
const SHAKE_TURNS: u8 = 10;
const SHAKE_HARD: u8 = 30;
/// How many of the hardest shakers a report lists.
const SHAKE_LINES: usize = 6;

/// FL_LOG_SHAKE=1: the men who shake in place. A man turns when his step
/// of one tick points against his step of the tick before, both longer
/// than SHAKE_STEP. Every SHAKE_WINDOW: how many living men turned on at
/// least SHAKE_TURNS ticks of it (and on at least SHAKE_HARD), their mean
/// step, and what they share: their regiment's state, out of formation, an
/// enemy within 3 m, a crowd under the brake's start, and how many bodies
/// touch them. The hardest are listed with their place on the field.
pub(super) fn shake_log(
    units: Res<Units>,
    groups: Res<Groups>,
    grid: Res<crate::spatial::SpatialGrid>,
    pipe: Res<crate::sim::TickPipeline>,
    time: Res<Time>,
    mut sh: Local<Shake>,
) {
    if std::env::var("FL_LOG_SHAKE").is_err() || pipe.tick == sh.tick {
        return;
    }
    sh.tick = pipe.tick;
    let n = units.len();
    // The death sweep moves the last men into the places of the removed:
    // those few carry another man's count for one window, the rest keep
    // theirs.
    sh.last.resize(n, Vec2::ZERO);
    sh.turns.resize(n, 0);
    sh.step.resize(n, 0.0);
    sh.wstart.resize(n, Vec2::ZERO);
    sh.path.resize(n, 0.0);
    for i in 0..n {
        let d = units.pos[i] - units.pos_prev[i];
        let d = Vec2::new(d.x, d.z);
        sh.path[i] += d.length();
        let was = sh.last[i];
        if d.length_squared() > SHAKE_STEP * SHAKE_STEP
            && was.length_squared() > SHAKE_STEP * SHAKE_STEP
            && d.dot(was) < 0.0
        {
            sh.turns[i] = sh.turns[i].saturating_add(1);
            sh.step[i] += d.length();
        }
        sh.last[i] = d;
    }
    let t = time.elapsed_secs();
    if t < sh.next {
        return;
    }
    sh.next = t + SHAKE_WINDOW;
    let (slow, _) = crate::sim::soldier::crowd_brake();
    let body = crate::sim::soldier::ready_body();
    let sep = crate::sim::soldier::SEP_RADIUS;
    let (mut alive, mut hard, mut turns_sum, mut step_sum) = (0u32, 0u32, 0u32, 0.0f32);
    let (mut by_state, mut by_touch) = ([0u32; 3], [0u32; 3]);
    let (mut out_form, mut near_enemy, mut thin) = (0u32, 0u32, 0u32);
    // Men trying a side right now (sim/soldier.rs `try_a_side`).
    let mut trying = 0u32;
    let mut shakers: Vec<(u8, usize, f32, u32)> = Vec::new();
    for i in 0..n {
        if units.death_t[i] != 0 {
            continue;
        }
        alive += 1;
        trying += (crate::sim::soldier::stuck_trying(units.stuck[i]) != 0) as u32;
        let turns = sh.turns[i];
        if turns < SHAKE_TURNS {
            continue;
        }
        let p = Vec2::new(units.pos[i].x, units.pos[i].z);
        let team_bit = (units.team[i] as u32) * crate::spatial::META_TEAM;
        let (mut crowd, mut touching, mut enemy) = (0.0f32, 0u32, f32::MAX);
        grid.for_each_candidate(p, 3.0, |o| {
            if o.idx as usize == i || (o.meta & crate::spatial::META_DYING) != 0 {
                return;
            }
            let d = p.distance(o.xz());
            if d < sep {
                crowd += 1.0 - d / sep;
            }
            touching += (d < body) as u32;
            if (o.meta & crate::spatial::META_TEAM) != team_bit {
                enemy = enemy.min(d);
            }
        });
        let gd = &groups.list[units.group[i] as usize];
        by_state[if gd.engaged { 0 } else if gd.order.is_some() { 1 } else { 2 }] += 1;
        by_touch[match touching { 0 => 0, 1 | 2 => 1, _ => 2 }] += 1;
        hard += (turns >= SHAKE_HARD) as u32;
        turns_sum += turns as u32;
        step_sum += sh.step[i];
        out_form += units.out_form[i] as u32;
        near_enemy += (enemy < 3.0) as u32;
        thin += (crowd < slow) as u32;
        shakers.push((turns, i, crowd, touching));
    }
    shakers.sort_unstable_by_key(|s| std::cmp::Reverse(s.0));
    let mut listed = String::new();
    for &(turns, i, crowd, touching) in shakers.iter().take(SHAKE_LINES) {
        listed.push_str(&format!(
            " | reg {} at ({:.0}, {:.0}): {turns} turns, crowd {crowd:.2}, touching {touching}{}",
            units.group[i],
            units.pos[i].x,
            units.pos[i].z,
            if units.out_form[i] { ", out of formation" } else { "" },
        ));
    }
    info!(
        "[shake] t={t:.0}s {} of {alive} men shake ({hard} hard), mean step {:.1} cm | regiment fighting {} moving {} standing {} | out of formation {out_form} | enemy within 3 m {near_enemy} | crowd under the brake's start {thin} | touching none {} one or two {} more {} | trying a side {trying}{listed}",
        shakers.len(),
        step_sum * 100.0 / turns_sum.max(1) as f32,
        by_state[0],
        by_state[1],
        by_state[2],
        by_touch[0],
        by_touch[1],
        by_touch[2],
    );
    sh.windows += 1;
    if sh.windows.is_multiple_of(WANDER_WINDOWS) {
        let (mut wander, mut w_out, mut w_thin, mut w_shake) = (0u32, 0u32, 0u32, 0u32);
        let mut w_state = [0u32; 3];
        let mut wanderers: Vec<(f32, f32, usize, f32)> = Vec::new();
        for i in 0..n {
            let p = Vec2::new(units.pos[i].x, units.pos[i].z);
            let path = sh.path[i];
            let net = p.distance(sh.wstart[i]);
            sh.wstart[i] = p;
            sh.path[i] = 0.0;
            if units.death_t[i] != 0 || path < WANDER_PATH || net > WANDER_NET {
                continue;
            }
            let mut crowd = 0.0f32;
            grid.for_each_candidate(p, 3.0, |o| {
                if o.idx as usize == i || (o.meta & crate::spatial::META_DYING) != 0 {
                    return;
                }
                let d = p.distance(o.xz());
                if d < sep {
                    crowd += 1.0 - d / sep;
                }
            });
            let gd = &groups.list[units.group[i] as usize];
            w_state[if gd.engaged { 0 } else if gd.order.is_some() { 1 } else { 2 }] += 1;
            wander += 1;
            w_out += units.out_form[i] as u32;
            w_thin += (crowd < slow) as u32;
            w_shake += (sh.turns[i] >= SHAKE_TURNS) as u32;
            wanderers.push((path, net, i, crowd));
        }
        wanderers.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let mut listed = String::new();
        for &(path, net, i, crowd) in wanderers.iter().take(SHAKE_LINES) {
            listed.push_str(&format!(
                " | reg {} at ({:.0}, {:.0}): {path:.1} m walked, {net:.1} m from his start, crowd {crowd:.2}{}",
                units.group[i],
                units.pos[i].x,
                units.pos[i].z,
                if units.out_form[i] { ", out of formation" } else { "" },
            ));
        }
        info!(
            "[wander] t={t:.0}s {wander} of {alive} men walked over {WANDER_PATH} m in {} s and ended within {WANDER_NET} m of their start | shakers among them {w_shake} | regiment fighting {} moving {} standing {} | out of formation {w_out} | crowd under the brake's start {w_thin}{listed}",
            WANDER_WINDOWS as f32 * SHAKE_WINDOW,
            w_state[0],
            w_state[1],
            w_state[2],
        );
    }
    sh.turns.fill(0);
    sh.step.fill(0.0);
}
