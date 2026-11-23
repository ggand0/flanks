//! The fight: the far look for an enemy, leaving formation to join, the
//! swing and the archer's shot, and the footwork about the enemy in reach.

use bevy::math::Vec3Swizzles;
use bevy::prelude::*;

use super::*;
use crate::sim::damage::DamageEvent;
use crate::unit_types::TYPES;

/// A man blocked by a comrade on the way to his enemy sidesteps toward
/// an open lane in some one-second windows (M2TW's isSideStepping),
/// and waits in the others.
const SIDESTEP_WINDOW: u32 = 30;
/// A man fights at his weapon's length. A blade lands with its far end,
/// so he stands where that end meets his enemy: he closes to this share
/// of his reach, and steps back to it from an enemy who has come inside
/// it when the ground behind him is open.
pub(super) const FIGHT_SHARE: f32 = 0.9;
/// He lets an enemy this far inside his fighting distance before he
/// steps back, and then steps all the way out to it: a real step, as on
/// the way back to his mark.
const BACK_SLACK: f32 = 0.3;
/// Man-ticks of that step back since the room log last read them
/// (FL_LOG_ROOM).
pub(crate) static STEP_BACK_TICKS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);
/// Share of those windows in which a blocked man sidesteps
/// (FL_SIDESTEP, 0 to 1). A play-testing knob.
fn sidestep_chance() -> f32 {
    static P: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *P.get_or_init(|| crate::util::env_or("FL_SIDESTEP", 0.2))
}

/// When a soldier still in formation reacts to an enemy he sees, or to a
/// comrade of his regiment going to the fight. Under the rule of joining
/// as a walker (`joins_as_walker`) he leaves at once: the chance is 1, and
/// the only spread left is his look, which comes every few ticks in his
/// own rhythm. Without that rule he reacts with a chance of 0.2 in each
/// half-second window, so the line rolls up from the contact outward, a
/// man at a time. FL_JOIN_REACT=<chance> sets the chance under either.
/// Once going he keeps going.
const JOIN_WINDOW: u32 = 15;
fn join_react_chance() -> f32 {
    static P: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *P.get_or_init(|| {
        let chance = if joins_as_walker() { 1.0_f32 } else { 0.2_f32 };
        crate::util::env_or("FL_JOIN_REACT", chance).clamp(0.0, 1.0)
    })
}
/// Ground speed toward the fight that reads as "going to it". A regiment
/// starts a charge only when its centre closes on its target this fast
/// (frontline.rs).
pub(crate) const GOING_SPEED: f32 = 1.0;
/// How far a man notices a comrade of his own regiment running to the
/// fight (on his far look, below).
const JOIN_SEE_R: f32 = 6.0;
/// Even with nobody going in sight, a man joins once his regiment has
/// been in melee longer than his own patience, spread between these
/// (seconds; FL_JOIN_PATIENCE sets the upper end): the fight's noise and
/// the officers carry further than sight. Far men join late, never not.
const JOIN_PATIENCE_MIN: f32 = 25.0;
fn join_patience_max() -> f32 {
    static D: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *D.get_or_init(|| crate::util::env_or("FL_JOIN_PATIENCE", 50.0_f32).max(JOIN_PATIENCE_MIN))
}

/// Sight radius for a man of a fighting regiment looking for an enemy
/// to go to (FL_SEEK_R, meters). A play-testing knob.
#[inline]
pub(super) fn seek_radius() -> f32 {
    static R: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *R.get_or_init(|| crate::util::env_or("FL_SEEK_R", 15.0))
}

/// How near an enemy of any regiment must be, coming at a man whose
/// regiment fights one ordered target (sim/job.rs `focus`), before he
/// turns to fight him (FL_THREAT_R, meters). Beyond it he goes after the
/// target's men only. A play-testing knob: M2TW's seek radius is not
/// known. At the combat jog a man covers 4 m in 1.4 s, a
/// quarter second of which is his next look and a second a full turn.
#[inline]
pub(super) fn threat_radius() -> f32 {
    static R: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *R.get_or_init(|| crate::util::env_or("FL_THREAT_R", 4.0))
}
/// Sparse-fight acquisition radius. Two depleted formations interleave
/// with gaps wider than QUERY_RADIUS and the fight stalls (hits/tick 0
/// with both sides standing at "the front"). Pressing units whose normal
/// scan finds no enemy look this far on a staggered 1-in-8-tick cadence
/// and lock a closing target; swings still require reach.
pub(super) const WIDE_ACQUIRE_R: f32 = 4.0;
/// Speed fraction above which a newly started swing counts as a charge.
/// The momentum bonus (attacker's charge_bonus attack points) resolves in
/// the damage pass — a braced SPEARWALL nullifies it (points beat momentum).
pub(super) const CHARGE_SPEED_FRAC: f32 = 0.6;

/// The far look, every eighth tick: a pressing man with nobody in reach
/// and open ground around him remembers the nearest enemy within his sight
/// (15 m for a fighting regiment, 4 m on the approach; with his regiment
/// focused on one target, its men, and any man within his threat distance
/// coming at him), and a man still in formation notices a comrade of his
/// regiment running to the fight within 6 m.
#[inline]
pub(super) fn acquire(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { grid, press, engaged, focus, tick_seed, .. } = *f;
    let i = s.i;
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    let best_idx = st.foe.best_idx;
    let crowd = st.scan.crowd;
    let watch_go = st.sight.watch_go;
    let memo_dir = st.sight.memo_dir;
    let my_team_bit = st.scan.my_team_bit;
    // Sparse-fight acquisition (see WIDE_ACQUIRE_R): a pressing unit with an
    // empty scan and open space around it memoizes a farther enemy in
    // `target` so the closing drive below can restore contact. Gated hard
    // (press + no near enemy + low crowd + 1/8 cadence) to stay off the 200k
    // hot path.
    let acquire_crowd_lim = CROWD_SLOW;
    // A man of a fighting regiment looks as far as he can see (seek_radius)
    // on his far look; on the approach the old short scan stands.
    let far_look = (i as u32).wrapping_add(tick_seed).is_multiple_of(8);
    if far_look {
        *s.sight &= !SIGHT_GO_FAR;
    }
    if best_idx == u32::MAX
        && !dying
        && !routed
        && press[gi]
        && crowd < acquire_crowd_lim
        && far_look
    {
        let look = if engaged[gi] { seek_radius() } else { WIDE_ACQUIRE_R };
        let mut far_d2 = look * look;
        if let Some(t) = focus[gi] {
            // His regiment fights one ordered target (sim/job.rs `focus`):
            // he looks for its men anywhere in his sight, and for any
            // enemy within his threat distance who is coming at him,
            // closing faster than GOING_SPEED. A man of another regiment
            // standing off or going elsewhere is not his business. The
            // same cells as the look below, with each man's velocity.
            let threat2 = threat_radius() * threat_radius();
            grid.for_each_candidate_vel(p, look, |o, ov| {
                let enemy = (o.meta & crate::spatial::META_TEAM) != my_team_bit
                    && (o.meta & crate::spatial::META_DYING) == 0;
                if enemy {
                    let d = p - o.xz();
                    let d2 = d.length_squared();
                    let closing = ov.dot(d);
                    let counts = crate::spatial::meta_group(o.meta) as u32 == t
                        || (d2 < threat2
                            && closing > 0.0
                            && closing * closing > GOING_SPEED * GOING_SPEED * d2);
                    if counts && d2 < far_d2 {
                        far_d2 = d2;
                        *s.target = o.idx;
                    }
                }
            });
        } else {
            grid.for_each_candidate(p, look, |o| {
                let enemy = (o.meta & crate::spatial::META_TEAM) != my_team_bit
                    && (o.meta & crate::spatial::META_DYING) == 0;
                if enemy {
                    let d2 = (p - o.xz()).length_squared();
                    if d2 < far_d2 {
                        far_d2 = d2;
                        *s.target = o.idx;
                    }
                }
            });
        }
        // Further than the neighbor scan, within JOIN_SEE_R: a comrade of his
        // own regiment running to the fight.
        if watch_go
            && grid.any_candidate_vel(p, look.min(JOIN_SEE_R), |o, ov| {
                o.idx as usize != i
                    && crate::spatial::meta_group(o.meta) == gi
                    && (p - o.xz()).length_squared() < JOIN_SEE_R * JOIN_SEE_R
                    && ov.dot(memo_dir) > GOING_SPEED
            })
        {
            *s.sight |= SIGHT_GO_FAR;
        }
    }
}

/// Leaving formation to fight, in his own time: a chance per half-second
/// window while he sees an enemy or a comrade going, and patience as the
/// backstop. Once out he never dresses on his slot while he sees an enemy
/// or his regiment's melee runs. A man walking to the fight keeps the
/// wish `drive` gave him until an enemy is in his reach.
#[inline]
pub(super) fn decide_to_join(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { melee_ticks, engaged, tick, .. } = *f;
    let i = s.i;
    let in_melee = st.fight.in_melee;
    let may_leave = st.fight.may_leave;
    let gi = st.gi;
    let memo_valid = st.foe.memo_valid;
    let memo_d2 = st.foe.memo_d2;
    let best_idx = st.foe.best_idx;
    let mut desired = st.desired;
    let mut committed = st.fight.committed;
    // Out of formation with his regiment's melee over, or never begun: he
    // comes back to his slot once he sees no enemy, nobody in reach and no
    // remembered enemy within his closing range (close_in). While his
    // regiment's melee runs he stays out until it ends, heading for the
    // fight when he sees nobody.
    if committed && !in_melee {
        let look = if engaged[gi] { seek_radius() } else { WIDE_ACQUIRE_R };
        let sees = best_idx != u32::MAX
            || (memo_valid && memo_d2 < (look + 1.0) * (look + 1.0));
        if !sees {
            committed = false;
            *s.out_form = false;
        }
    }
    // Leaving formation to fight, in his own time: he reacts to an enemy in
    // sight or a comrade seen running to the fight (a chance per half-second
    // window), or his patience with the fight's noise runs out. Near men go
    // first and the line rolls up outward. With no melee on his regiment's
    // clock (a lone enemy at the flank, a thin contact) he goes only for an
    // enemy within the short look (WIDE_ACQUIRE_R): the men nearest him
    // leave the ranks for him, the rest stay in formation.
    if may_leave && !committed {
        let patience = (JOIN_PATIENCE_MIN
            + (join_patience_max() - JOIN_PATIENCE_MIN)
                * crate::units::hash01((i as u32).wrapping_mul(0x3C6E) ^ 0xF372))
            * 30.0;
        let saw_comrade_go = *s.sight & (SIGHT_GO | SIGHT_GO_FAR) != 0;
        let sees = if in_melee {
            memo_valid || saw_comrade_go
        } else {
            memo_valid && memo_d2 < WIDE_ACQUIRE_R * WIDE_ACQUIRE_R
        };
        let react = sees && {
            let window = (tick.wrapping_add((i as u32).wrapping_mul(11)) / JOIN_WINDOW)
                .wrapping_mul(0x9E37_79B1);
            crate::units::hash01(window ^ (i as u32).wrapping_mul(0x6A09))
                < join_react_chance()
        };
        if react || (in_melee && melee_ticks[gi] as f32 > patience) {
            committed = true;
            *s.out_form = true;
            st.fight.set_off = true;
        }
    }
    // Out of formation he never dresses on his slot: he fights, goes for an
    // enemy he sees, heads for the fight, or stands where he is when
    // blocked. Inferring this from his speed made a man who slowed down run
    // back to his slot and come out again, over and over. A man walking to
    // the fight (`joining`) keeps that walk from `drive` until an enemy is
    // in his reach this tick; then he fights (`close_in`). A man who came
    // back into formation this tick walks to his slot again.
    if committed && (!st.fight.joining || best_idx != u32::MAX) {
        desired = Vec2::ZERO;
        st.fight.joining = false;
    } else if st.fight.joining && !committed {
        desired = st.fight.slot_wish;
        st.fight.joining = false;
    }
    st.fight.committed = committed;
    st.desired = desired;
}

/// The swing state machine: an archer's draw and loose, a wind-up and
/// its strike (a damage event, resolved by the serial apply), the recovery,
/// and a ready man's choice of target in his forward half-plane.
#[inline]
pub(super) fn swing(f: &Field, s: &mut Soldier, st: &mut Step, out: &mut ChunkOut) {
    let params = &TYPES[st.my_kind];
    let mut desired = st.desired;
    // Swing state machine. All writes are to this unit's own row; damage goes
    // through the chunk event buffer.
    let mut face_target = None;
    if !st.dying {
        match *s.swing & crate::units::SWING_STATE_MASK {
            crate::units::SWING_WINDUP
                if *s.swing & crate::units::SWING_RANGED != 0 =>
            {
                face_target = draw(f, s, st, out, &mut desired);
            }
            crate::units::SWING_WINDUP => face_target = wind_up(f, s, st, out, &mut desired),
            crate::units::SWING_DEFEND => face_target = defend(f, s, st, out, &mut desired),
            crate::units::SWING_RECOVER => recover(s),
            _ => start_a_swing(f, s, st, out),
        }
        // A reloading archer contacted in melee drops the reload: he defends
        // at knife tempo instead of standing through the 8 s bow cycle.
        if st.my_kind == crate::unit_types::KIND_ARCHER as usize
            && st.foe.best_idx != u32::MAX
            && *s.swing & crate::units::SWING_STATE_MASK
                == crate::units::SWING_RECOVER
            && *s.swing_t > params.cooldown_ticks
        {
            *s.swing_t = params.cooldown_ticks;
        }
    }
    st.desired = desired;
    st.fight.face_target = face_target;
}

/// Drawing the bow at the man he picked, with his feet planted, and the
/// loose when the draw is done; with the target gone or his knife out he
/// lets the draw go. Returns where he looks.
#[inline]
fn draw(f: &Field, s: &mut Soldier, st: &Step, out: &mut ChunkOut, desired: &mut Vec2) -> Option<Vec2> {
    let Field { pos_prev, shoot_at, .. } = *f;
    let gi = st.gi;
    let mut face_target = None;
    // Drawing the bow: feet planted, eyes on the target block.
    // A man whose knife came out lets the draw go.
    *desired = Vec2::ZERO;
    let knife = *s.sight & SIGHT_KNIFE != 0;
    if let Some(shot) = shoot_at[gi].as_ref().filter(|_| !knife) {
        let q = *s.quarry as usize;
        face_target =
            Some(if q < pos_prev.len() { pos_prev[q].xz() } else { shot.c });
        if *s.swing_t == 0 {
            loose(f, s, st, out, shot);
        } else {
            *s.swing_t -= 1;
        }
    } else {
        // Target gone mid-draw, or the knife out: ease off and
        // reassess shortly.
        *s.swing = crate::units::SWING_RECOVER;
        *s.swing_t = crate::unit_types::missile::CANCEL_TICKS;
    }
    face_target
}

/// The loose: an arrow at the man he drew at, scattered and led for his
/// block's drift, and the reload.
#[inline]
fn loose(f: &Field, s: &mut Soldier, st: &Step, out: &mut ChunkOut, shot: &crate::sim::job::ShootAt) {
    let Field { terrain, pos_prev, team, group, target_members, blocks, tick_seed, .. } = *f;
    let i = s.i;
    let p = st.p;
    let my_kind = st.my_kind;
    let h = |k: u32| {
        crate::units::hash01(
            tick_seed
                ^ (i as u32).wrapping_mul(k).wrapping_add(k),
        )
    };
    // Aim: the man he drew at (every man has his own
    // aim target) while he lives, else another man of
    // the block; the footprint-disc spot stands in only
    // if every member died this tick. Then the range-
    // INDEPENDENT landing scatter and a lead for the
    // block's drift over the flight.
    let members = &target_members[shot.t];
    let q = *s.quarry as usize;
    let base = if q < pos_prev.len()
        && team[q] != team[i]
        && f.death_snap[q] == 0
    {
        pos_prev[q].xz()
    } else if members.is_empty() {
        let ang =
            h(0x1F3B) * std::f32::consts::TAU;
        let rad = shot.r * 0.85 * h(0x2E5D).sqrt();
        shot.c + Vec2::new(ang.cos(), ang.sin()) * rad
    } else {
        let m = members[(h(0x1F3B)
            * members.len() as f32)
            as usize
            % members.len()]
            as usize;
        pos_prev[m].xz()
    };
    let sigma = crate::unit_types::missile::AIM_SPREAD;
    let mut aim = base
        + Vec2::new(
            (h(0x3B7F) + h(0x45A3) - 1.0) * sigma,
            (h(0x5DC1) + h(0x6B8D) - 1.0) * sigma,
        );
    aim += shot.vel * (aim.distance(p) / 30.0);
    // Launch above the body-hit band (arrows.rs): a
    // shaft leaving at head height is inside its own
    // shooter's hit cylinder at flight-time zero and
    // kills him.
    let over = crate::unit_types::half_height(my_kind)
        + crate::arrows::LAUNCH_OVER_HEAD;
    let from = Vec3::new(p.x, pos_prev[i].y + over, p.y);
    let to = Vec3::new(
        aim.x,
        terrain.height_at(aim.x, aim.y)
            + crate::arrows::AIM_HEIGHT * crate::unit_types::tallest(),
        aim.y,
    );
    out.arrows.push(crate::arrows::ArrowSpawn {
        pos: from,
        vel: crate::arrows::solve_launch(
            from,
            to,
            &blocks[team[i] as usize],
            terrain,
        ),
        team: team[i],
        group: group[i],
    });
    *s.ammo = s.ammo.saturating_sub(1);
    *s.swing = crate::units::SWING_RECOVER;
    // Reload: the M2TW volley cycle is animation-bound at
    // ~10 s; the jitter keeps later volleys ragged.
    *s.swing_t =
        (crate::unit_types::missile::RELOAD_TICKS as f32
            * (0.8 + 0.25 * h(0x77F1)))
            .min(255.0)
            as u8;
}

/// A wind-up and its strike: feet planted, eyes on the target, and on the
/// last tick the blow as a damage event. Returns where he looks.
#[inline]
fn wind_up(f: &Field, s: &mut Soldier, st: &Step, out: &mut ChunkOut, desired: &mut Vec2) -> Option<Vec2> {
    let Field { pos_prev, group, broken, tick_seed, .. } = *f;
    let i = s.i;
    let params = &TYPES[st.my_kind];
    let mut face_target = None;
    // Feet planted while winding up — EXCEPT against a routing
    // target: the cut-down happens at a run, or the runner is 3 m
    // gone by the strike tick and every blow whiffs (the pursuit
    // treadmill).
    let t = *s.target as usize;
    let target_routed =
        t < pos_prev.len() && broken[group[t] as usize];
    if !target_routed {
        *desired *= 0.25;
    }
    if t < pos_prev.len() {
        face_target = Some(pos_prev[t].xz());
    }
    if *s.swing_t == 0 {
        // Strike lands; validity (still alive, still in reach,
        // still an enemy) is checked in the apply pass — a dodged
        // or dead target is a whiff. A cut lands harder.
        let cut = params.is_cut(*s.swing);
        let mut jit =
            0.85 + 0.3 * crate::units::hash01(tick_seed ^ (i as u32));
        if cut {
            jit *= params.cut_weight();
        }
        out.events.push(DamageEvent {
            victim: *s.target,
            attacker: i as u32,
            jit,
            charge: *s.swing & crate::units::SWING_CHARGE != 0,
            impale: false,
            warn: 0,
            chain: 0,
        });
        *s.swing = crate::units::SWING_RECOVER;
        let cjit = (1.0 - crate::unit_types::COOLDOWN_JITTER)
            + 2.0
                * crate::unit_types::COOLDOWN_JITTER
                * crate::units::hash01(
                    tick_seed ^ (i as u32).wrapping_mul(0x9E37),
                );
        *s.swing_t = (params.cooldown_ticks as f32 * cjit) as u8;
        if cut {
            *s.swing_t = (*s.swing_t).max(params.cut_return_ticks);
        }
    } else {
        *s.swing_t -= 1;
    }
    face_target
}

/// Answering a swing he saw coming, and the counter of a parry that
/// stopped the blow. Returns where he looks.
#[inline]
fn defend(f: &Field, s: &mut Soldier, st: &Step, out: &mut ChunkOut, desired: &mut Vec2) -> Option<Vec2> {
    let Field { pos_prev, tick_seed, .. } = *f;
    let i = s.i;
    let params = &TYPES[st.my_kind];
    let mut face_target = None;
    // Answering a swing he saw coming (sim/damage.rs): feet
    // planted, eyes on the attacker, the shield or the blade
    // up to meet the blow, then back on guard. The damage
    // apply set this and may end it early: a parry that
    // stopped the blow turns into the counter's wind-up.
    *desired *= 0.25;
    let t = *s.target as usize;
    if t < pos_prev.len() {
        face_target = Some(pos_prev[t].xz());
    }
    // A parry that stopped the blow counters: a stab at
    // the man he parried. In his kind's window he launches
    // it with a chance each tick by his skill, at the
    // window's end for sure.
    let stopped = *s.swing & crate::sim::damage::GUARD_STOPPED != 0;
    let elapsed = crate::sim::damage::PARRY_LEN.saturating_sub(*s.swing_t);
    let (from, by) = crate::sim::damage::COUNTER_WINDOW[st.my_kind];
    let launch = stopped
        && elapsed >= from
        && (elapsed >= by
            || crate::units::hash01(tick_seed ^ (i as u32).wrapping_mul(0x7F4A_7C15))
                < params.defence_skill * crate::sim::damage::counter_per_tick());
    if launch {
        *s.swing = crate::units::SWING_WINDUP
            | (*s.swing & crate::units::SWING_STAGGER_IMMUNE);
        *s.swing_t = ((params.windup_ticks as f32
            * crate::sim::damage::counter_windup())
            .ceil() as u8)
            .clamp(1, params.windup_ticks);
        // The man he parried is warned like any target, one
        // deeper in the exchange: he may answer from the
        // bind, with less chance each time (sim/damage.rs).
        out.events.push(DamageEvent {
            victim: *s.target,
            attacker: i as u32,
            jit: 0.0,
            charge: false,
            impale: false,
            warn: *s.swing_t,
            chain: s.chain.saturating_add(1),
        });
    } else if *s.swing_t == 0 {
        *s.swing = crate::units::SWING_READY
            | (*s.swing & crate::units::SWING_STAGGER_IMMUNE);
    } else {
        *s.swing_t -= 1;
    }
    face_target
}

/// The recovery after a blow, a shot or a stagger, back to ready.
#[inline]
fn recover(s: &mut Soldier) {
    if *s.swing_t == 0 {
        // A stagger that just wore off leaves one free pass
        // against the next one (anti-stunlock); a plain recovery
        // carries an unspent pass forward.
        let immune = if *s.swing
            & crate::units::SWING_STAGGERED
            != 0
        {
            crate::units::SWING_STAGGER_IMMUNE
        } else {
            *s.swing & crate::units::SWING_STAGGER_IMMUNE
        };
        *s.swing = crate::units::SWING_READY | immune;
    } else {
        *s.swing_t -= 1;
    }
}

/// A ready man: a swing at the enemy in reach he faces, or an archer's
/// draw at a man of the block his regiment shoots at.
#[inline]
fn start_a_swing(f: &Field, s: &mut Soldier, st: &Step, out: &mut ChunkOut) {
    let Field { pos_prev, speed, shoot_at, tick_seed, .. } = *f;
    let i = s.i;
    let params = &TYPES[st.my_kind];
    let p = st.p;
    let gi = st.gi;
    let routed = st.routed;
    let sticky = st.foe.sticky;
    let prev_target = st.foe.prev_target;
    let best_idx = st.foe.best_idx;
    let my_kind = st.my_kind;
    // Ready: pick a target from the scan. Stick with the previous
    // one when still in reach (duels), else nearest. Routing
    // units never start attacks (they still defend nothing —
    // pursuit is free hits).
    let chosen = if sticky { prev_target } else { best_idx };
    // No eyes in the back of his head: a man only opens on a
    // target in his forward half-plane. Being struck tells him
    // where to turn (the flash facing in `face`), but it does NOT let
    // him swing backward over his shoulder — he attacks once he
    // has turned far enough, at the human turn-speed cap. Without
    // this gate every rear-approached victim counter-wound-up on
    // proximity and was face-on before the first blow landed; the
    // rear sector never fired in practice.
    let aware = chosen != u32::MAX && {
        let t = chosen as usize;
        t < pos_prev.len() && {
            let to_t = pos_prev[t].xz() - p;
            let fwd = Vec2::new(
                s.yaw.sin(),
                s.yaw.cos(),
            );
            fwd.dot(to_t) >= 0.0
        }
    };
    if chosen != u32::MAX && !routed && aware {
        *s.target = chosen;
        // Arriving at speed = a charging blow: momentum converts
        // to damage + a bigger lunge (render reads the flag).
        let v2 = s.vel.xz().length_squared();
        let cs = speed[i] * CHARGE_SPEED_FRAC;
        // Attack style for this swing: 0 = stab, 1 = the classic
        // overhead swing (2 = slash exists in the shader but
        // benched). A kind with a cut draws 1 as its slow cut;
        // its 2 is the overhead swing at the stab's pace, drawn
        // once the overhead is a clip of its own (the code's
        // swing over the guard turns the blade sideways). Spears
        // only ever thrust, and a charging man with a cut stabs:
        // a slow cut at a run carries him into the man before it
        // lands.
        let charge = v2 > cs * cs;
        let style = if my_kind
            == crate::unit_types::KIND_SPEAR as usize
            || (charge && params.cut_windup_ticks > 0)
        {
            0
        } else {
            (crate::units::hash01(
                tick_seed ^ (i as u32).wrapping_mul(0x51ED),
            ) * 2.0) as u8
        };
        let style = style << crate::units::SWING_STYLE_SHIFT;
        *s.swing = if charge {
            crate::units::SWING_WINDUP
                | crate::units::SWING_CHARGE
                | style
        } else {
            crate::units::SWING_WINDUP | style
        };
        *s.swing_t = params.windup_for(*s.swing);
        // The target is told of the swing now, so that he
        // can answer it with a block or a parry timed to the
        // strike (sim/damage.rs).
        out.events.push(DamageEvent {
            victim: chosen,
            attacker: i as u32,
            jit: 0.0,
            charge: false,
            impale: false,
            warn: *s.swing_t,
            chain: 0,
        });
    } else if my_kind == crate::unit_types::KIND_ARCHER as usize
        && !routed
        && *s.ammo > 0
        && *s.sight & SIGHT_KNIFE == 0
        && s.vel.xz().length_squared() < 4.0
        && let Some(shot) = &shoot_at[gi]
        && let Some(q) = pick_quarry(f, s, p, gi, shot)
    {
        // Nock and draw at his man (foot archers shoot standing
        // only; the speed test keeps a marching or skirmishing
        // man's bow on his back). Style bits stay 0: the stab
        // pull-back IS the string draw.
        *s.quarry = q;
        *s.swing = crate::units::SWING_WINDUP
            | crate::units::SWING_RANGED;
        *s.swing_t = crate::unit_types::missile::DRAW_TICKS
            + (crate::units::hash01(
                tick_seed ^ (i as u32).wrapping_mul(0x2C9F),
            ) * 20.0) as u8;
    }
}

/// The share of the archers who keep shooting at the same man while he
/// lives and stays in reach; the rest pick afresh for every shot, as men
/// in the chaos of a battle do.
const KEEP_QUARRY_SHARE: f32 = 1.0 / 3.0;
/// An archer without a man to shoot looks for one every this many ticks.
const QUARRY_LOOK_TICKS: u32 = 8;
/// Random members he tries in one look.
const QUARRY_TRIES: u32 = 3;

/// Whether `m` is a man this archer may shoot: an enemy of a block his
/// regiment shoots at, alive, within his reach and, in a search, among
/// his block's clear men (sim/mod.rs `clear_members`).
#[inline]
fn quarry_ok(
    f: &Field,
    p: Vec2,
    my_team: u8,
    reach2: f32,
    shot: &crate::sim::job::ShootAt,
    blocks: &[u32],
    m: u32,
) -> bool {
    let Field { pos_prev, team, group, death_snap, clear_members, .. } = *f;
    let mi = m as usize;
    mi < pos_prev.len()
        && team[mi] != my_team
        && death_snap[mi] == 0
        && blocks.contains(&group[mi])
        && pos_prev[mi].xz().distance_squared(p) <= reach2
        && (!shot.search || clear_members[group[mi] as usize].binary_search(&m).is_ok())
}

/// The man an archer draws at: the one he keeps, when he is one of the
/// men who keep theirs and the man still serves; else a new pick from
/// his regiment's blocks (sim/job.rs `shoot_at`, `search_blocks`): a
/// random living member, or in a search a random clear member, the
/// nearest block first, a few tries a look and a look every few ticks.
/// None: he waits with the bow down.
#[inline]
fn pick_quarry(
    f: &Field,
    s: &Soldier,
    p: Vec2,
    gi: usize,
    shot: &crate::sim::job::ShootAt,
) -> Option<u32> {
    let Field { team, kind, target_members, clear_members, search_blocks, tick, tick_seed, .. } =
        *f;
    let i = s.i;
    let my_team = team[i];
    let reach = crate::unit_types::bow_reach(kind[i]);
    let reach2 = reach * reach;
    let single = [shot.t as u32];
    let blocks: &[u32] = if shot.search && !search_blocks[gi].is_empty() {
        &search_blocks[gi]
    } else {
        &single
    };
    let keeps =
        crate::units::hash01((i as u32).wrapping_mul(0x9E37) ^ 0x1B3D) < KEEP_QUARRY_SHARE;
    if keeps && quarry_ok(f, p, my_team, reach2, shot, blocks, *s.quarry) {
        return Some(*s.quarry);
    }
    if !tick.wrapping_add((i as u32).wrapping_mul(7)).is_multiple_of(QUARRY_LOOK_TICKS) {
        return None;
    }
    for k in 0..QUARRY_TRIES {
        let b = blocks[k as usize % blocks.len()] as usize;
        let list: &[u32] = if shot.search { &clear_members[b] } else { &target_members[b] };
        if list.is_empty() {
            continue;
        }
        let h = crate::units::hash01(
            tick_seed
                ^ (i as u32).wrapping_mul(0x7F4A_7C15).wrapping_add(k.wrapping_mul(0x2545_F491)),
        );
        let m = list[((h * list.len() as f32) as usize).min(list.len() - 1)];
        if quarry_ok(f, p, my_team, reach2, shot, blocks, m) {
            return Some(m);
        }
    }
    None
}

/// A waiting man's sidestep: in some one-second windows he steps toward
/// whichever side is open, at the shuffle's pace, and in the others he
/// stands.
#[inline]
fn sidestep(i: usize, tick: u32, left_blocked: bool, right_blocked: bool, side_dir: Vec2) -> Vec2 {
    let window = (tick.wrapping_add((i as u32).wrapping_mul(7)) / SIDESTEP_WINDOW)
        .wrapping_mul(0x2545_F491);
    let roll = crate::units::hash01(window ^ (i as u32).wrapping_mul(0x9E37));
    let chance = sidestep_chance();
    let side = match (left_blocked, right_blocked) {
        _ if roll >= chance => Vec2::ZERO,
        (false, true) => side_dir,
        (true, false) => -side_dir,
        (false, false) if roll < 0.5 * chance => side_dir,
        (false, false) => -side_dir,
        (true, true) => Vec2::ZERO,
    };
    side * STEP_PACE
}

/// His footwork around the enemy in reach: closing to his weapon's
/// length and stepping back to it from an enemy who has come inside it.
/// With nobody in reach, going to the enemy he
/// remembers through open ground (waiting behind a comrade, sidestepping
/// now and then), or, with nobody to close on, joining his regiment's
/// fight at the jog. Under the friction contact's brake a joining
/// soldier's walk comes from `drive` (`join_goal`), and this does neither.
#[inline]
pub(super) fn close_in(f: &Field, s: &mut Soldier, st: &mut Step) {
    let mut desired = st.desired;
    let mut fade = st.walk.fade;
    let (close_to, memo_close) = close_target(f, s, st);
    let d_surge = close_on(f, s, st, close_to, memo_close, &mut desired);
    join_the_fight(f, s, st, d_surge, &mut desired, &mut fade);
    st.desired = desired;
    st.walk.fade = fade;
}

/// Whom he closes on (`close_in`): his locked swing target, the enemy in
/// reach, or out of formation in open space the enemy he remembers; and
/// whether that is the one he remembers.
#[inline]
fn close_target(f: &Field, s: &Soldier, st: &Step) -> (u32, bool) {
    let Field { pos_prev, .. } = *f;
    let committed = st.fight.committed;
    let best_idx = st.foe.best_idx;
    let crowd = st.scan.crowd;
    // Fighters close the last meter to swing range. Only active when an enemy
    // is ALREADY in reach — this is combat execution (like the wind-up foot
    // plant), not steering; it bypasses the jam yield on purpose so front
    // lines stay joined instead of settling at the separation standoff just
    // outside sword range. Close toward the LOCKED swing target when there is
    // one — the per-tick nearest enemy flips in a clog and flip-flopping the
    // close direction reads as twitch. Falls back to the far-acquisition memo
    // in `target` when the near scan is empty AND the unit is in open space —
    // in a dense press the memo would let second ranks drive through the jam
    // and compress the crowd into overlap. The memo may be stale after
    // death sweeps reindex, so it is validated as "some enemy within closing
    // range" — a legitimate closing target regardless of identity. Only a
    // man out of formation goes to a remembered enemy: a formed man seeing
    // one leaves formation first (decide_to_join), so the slot never pulls
    // against the chase. Under the friction contact's brake a joining
    // soldier's walk to the enemy he remembers comes from `drive`
    // (`join_goal`), and this does not run for him.
    if *s.swing & crate::units::SWING_STATE_MASK
        != crate::units::SWING_READY
        && (*s.target as usize) < pos_prev.len()
    {
        (*s.target, false)
    } else if best_idx != u32::MAX {
        (best_idx, false)
    } else if !joins_as_walker() && crowd < st.scan.brake.0 && committed {
        (*s.target, true)
    } else {
        (u32::MAX, false)
    }
}

/// Closing on `close_to` to his weapon's length, or stepping back to it
/// from an enemy who has come inside it (`close_in`). Returns the surge
/// toward him, part of his wish.
#[inline]
fn close_on(f: &Field, s: &mut Soldier, st: &Step, close_to: u32, memo_close: bool, desired: &mut Vec2) -> Vec2 {
    let Field { pos_prev, speed, team, group, broken, wall, engaged, hold, moving, focus, tick, .. } = *f;
    let i = s.i;
    let params = &TYPES[st.my_kind];
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    let committed = st.fight.committed;
    let jam = st.walk.jam;
    let way_blocked = st.sight.way_blocked;
    let left_blocked = st.sight.left_blocked;
    let right_blocked = st.sight.right_blocked;
    let comrade_ahead = st.sight.comrade_ahead;
    let side_dir = st.sight.side_dir;
    let mut d_surge = Vec2::ZERO;
    // A man sent somewhere by a Move order closes on nobody: he walks on,
    // striking only a man in front of him (the swing above). With his
    // regiment focused on one target he closes on a man of another
    // regiment only within his threat distance (look_around).
    if !dying
        && !routed
        && !moving[gi]
        && (close_to as usize) < pos_prev.len()
        && team[close_to as usize] != team[i]
        && focus[gi].is_none_or(|t| {
            group[close_to as usize] == t
                || pos_prev[close_to as usize].xz().distance_squared(p)
                    < threat_radius() * threat_radius()
        })
    {
        let to_enemy = pos_prev[close_to as usize].xz() - p;
        let dist = to_enemy.length();
        // Held regiments fight at arm's length only.
        let look = if engaged[gi] { seek_radius() } else { WIDE_ACQUIRE_R };
        let max_close = if hold[gi] { 2.2 } else { look + 1.0 };
        // He closes on an enemy in reach to his weapon's length. Going to
        // a seen enemy, each man stops at his own distance, between that
        // and just inside his reach.
        let own = crate::units::hash01((i as u32).wrapping_mul(0x3C1B) ^ 0x51F7);
        let fight = FIGHT_SHARE * params.reach;
        let stop = if memo_close { fight + 0.8 * (params.reach - fight) * own } else { fight };
        if dist > stop && dist < max_close {
            let mut urge = ((dist - stop) / 0.8).clamp(0.0, 1.0);
            // The surge toward a REMEMBERED enemy (no one in reach yet) is
            // steering, not combat execution: it yields to the jam like all
            // steering, so the press brakes on genuine body-pack. Ungated
            // this factor is always 1 (the memo gate above already required
            // a crowd under the brake's start, i.e. jam == 0).
            if memo_close {
                urge *= 1.0 - jam;
                // He steps toward a remembered enemy only through open
                // ground. With a comrade in the way he stands and waits for
                // room instead of leaning on the man's back (M2TW's crowded
                // soldier).
                if way_blocked {
                    urge = 0.0;
                    // Waiting; in some windows he sidesteps toward whichever
                    // side is open.
                    *desired += sidestep(i, tick, left_blocked, right_blocked, side_dir);
                }
            }
            let pace = if !memo_close {
                speed[i] * 0.35
            } else if dist > JOG_BEYOND && !comrade_ahead {
                COMBAT_JOG_PACE
            } else {
                ADVANCE_PACE
            };
            *desired += to_enemy * (pace * urge / dist);
            d_surge = to_enemy * (pace * urge / dist);
        } else if !memo_close
            && committed
            && wall[gi] == 0
            && !st.sight.back_blocked
            && *s.swing & crate::units::SWING_STATE_MASK != crate::units::SWING_WINDUP
            && !broken[group[close_to as usize] as usize]
            && dist > 1e-3
        {
            // The enemy has come inside his weapon's length: out of
            // formation, with open ground behind him and no blow under way,
            // he steps back to it. He lets the man well inside first, and a
            // step once begun goes all the way out. A man in formation holds
            // his place, and nobody backs off a broken enemy.
            let stepping = s.vel.xz().dot(to_enemy) < -0.5 * STEP_PACE * dist;
            let inside = if stepping { fight } else { fight - BACK_SLACK };
            if dist < inside {
                *desired -= to_enemy * (STEP_PACE / dist);
                STEP_BACK_TICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    d_surge
}

/// Joining his regiment's fight at the jog when he has no enemy to close
/// on (`close_in`), without the friction contact's brake.
#[inline]
fn join_the_fight(f: &Field, s: &Soldier, st: &Step, d_surge: Vec2, desired: &mut Vec2, fade: &mut Vec2) {
    let Field { tick, .. } = *f;
    let i = s.i;
    let p = st.p;
    let committed = st.fight.committed;
    let memo_valid = st.foe.memo_valid;
    let join_fp = st.fight.join_fp;
    let best_idx = st.foe.best_idx;
    let jam = st.walk.jam;
    let way_blocked = st.sight.way_blocked;
    let left_blocked = st.sight.left_blocked;
    let right_blocked = st.sight.right_blocked;
    let comrade_ahead = st.sight.comrade_ahead;
    let side_dir = st.sight.side_dir;
    // Joining: no enemy to close on, so he heads for the enemy unit his
    // regiment fights, jogging over open ground, walking with a comrade close
    // ahead, waiting or sidestepping when blocked. He picks a soldier to
    // fight once one is in sight (the acquisition above). Under the friction
    // contact's brake this walk comes from `drive` (`join_goal`) instead.
    if !joins_as_walker()
        && committed
        && !memo_valid
        && let Some(fp) = join_fp
        && best_idx == u32::MAX
        && d_surge == Vec2::ZERO
        && *s.swing & crate::units::SWING_STATE_MASK == crate::units::SWING_READY
    {
        let to_fp = fp - p;
        let dist = to_fp.length();
        if dist > 1.0 {
            let dir = to_fp / dist;
            *desired = if !way_blocked {
                let pace = if comrade_ahead { ADVANCE_PACE } else { COMBAT_JOG_PACE };
                *fade = dir * pace * jam;
                dir * pace * (1.0 - jam)
            } else {
                *fade = Vec2::ZERO;
                sidestep(i, tick, left_blocked, right_blocked, side_dir)
            };
        }
    }
}
