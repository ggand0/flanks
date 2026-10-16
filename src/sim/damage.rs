//! The serial damage apply: every swing, arrow impact aside, is resolved
//! here after the parallel kernel, in chunk order, so the sim stays
//! deterministic and hp reaches death in exactly one place. Two kinds of
//! event come through: the warning of a swing that just started (the
//! victim may answer it with a block or a parry) and the blow itself
//! (stopped by a raised guard, or rolled against the stats, and landed).

use bevy::math::Vec3Swizzles;
use bevy::prelude::*;

use crate::orders::Groups;
use crate::terrain::Terrain;
use crate::unit_types::{BASE_DMG, TYPES};
use crate::units::Units;

/// Shieldwall: locked shields count double where the shield covers
/// (front + left — a shieldwall still has no back), at some offense.
const SHIELDWALL_SHIELD_PTS: f32 = 2.0;
/// Shieldwall trades attack points for the cover (arms stay behind shields).
const SHIELDWALL_ATK_PTS: f32 = -1.0;
/// Spearwall: set spears strike harder (the wall of points does the work).
const SPEARWALL_ATK_PTS: f32 = 2.0;
/// Sector test at damage apply: cos(60 deg). Front = attack direction
/// within 60 deg of the victim's facing; rear = beyond 120 deg; else side.
/// Shared with the arrow-impact resolution (arrows.rs).
pub const SECTOR_COS_60: f32 = 0.5;
/// A raised guard meets a blow from within 45 degrees of the man's
/// facing (cos 45): his shield or blade is between him and the attacker.
const GUARD_COS_45: f32 = std::f32::consts::FRAC_1_SQRT_2;
/// Attack points a blow gets against a man's open side (his right, no
/// shield there) and against his back, on top of the skill and shield he
/// cannot bring to bear: he cannot put his weapon between himself and
/// what he does not see.
const OPEN_SIDE_PTS: f32 = 5.0;
const REAR_PTS: f32 = 10.0;
/// The land roll's scale: M2TW's `combat-balancing/melee-hit-rate`, 1.75
/// (a blow at even points lands 35 times in 100). FL_HIT_RATE overrides
/// it for the calibration runs.
pub const HIT_RATE: f32 = 1.75;
/// How often a man's answer to a swing he sees is up in time, per point
/// of defence skill (a knight 6 in 10, a man-at-arms 4, a spearman 3, a
/// bowman 2), less the fatigue penalty of his attack. FL_GUARD overrides
/// it for the calibration runs.
pub const GUARD_PER_SKILL: f32 = 0.10;
/// The answers, in the style bits of a defend swing byte: the shield
/// block, and the weapon parry that counters at once when it is on time.
pub const GUARD_BLOCK: u8 = 0;
pub const GUARD_PARRY: u8 = 1 << crate::units::SWING_STYLE_SHIFT;
/// The block's timeline in ticks, the knight's block clip's
/// (`assets/units/knight.block.json`, 0.80 s): the shield is up at its
/// guard event (0.20 s), the blow recoils off it at 0.40 s, and the
/// rest is the recovery. The guard stops blows from the raise to the
/// recoil.
pub const BLOCK_RAISE: u8 = 6;
pub const BLOCK_UP: u8 = 6;
pub const BLOCK_LEN: u8 = 24;
/// The parry's, the knight's parry clip's (`knight.parry.json`, 0.90 s):
/// the blade is up at its guard event (0.15 s) and deflects at 0.20 s;
/// a parry that stops the blow leaves the clip there for the counter,
/// a late one plays the whole wide follow-through.
pub const PARRY_RAISE: u8 = 5;
pub const PARRY_UP: u8 = 3;
pub const PARRY_LEN: u8 = 27;
/// A parry that stopped the blow counters from the bind: a stab at this
/// share of the kind's wind-up, while the attacker is still recovering
/// from his stopped swing.
pub const COUNTER_WINDUP: f32 = 0.5;
/// The latest the counter starts, ticks into the parry clip (0.37 s:
/// through the deflect at 0.20 s). From the tick after his blade stops
/// the blow the man has `COUNTER_PER_TICK` per point of defence skill
/// each tick to launch it (sim/soldier.rs), so the counters of a
/// skilled man come early and spread, a poor man's mostly at the
/// latest. The stopped blow marks the parry with `GUARD_STOPPED`.
pub const PARRY_COUNTER_AT: u8 = 11;
/// His chance per tick to launch the counter, per point of defence
/// skill (a knight 0.24 a tick, a spearman 0.12). FL_COUNTER_TICK.
pub const COUNTER_PER_TICK: f32 = 0.04;
/// Whether a parry that stopped the blow counters at all: this much,
/// plus `COUNTER_CHANCE_PER_SKILL` per point of defence skill less the
/// fatigue penalty (a knight 0.98, a bowman 0.86). A parry that does
/// not counter plays its whole clip and he is back on guard.
/// FL_COUNTER_BASE and FL_COUNTER_SKILL.
pub const COUNTER_CHANCE_BASE: f32 = 0.8;
pub const COUNTER_CHANCE_PER_SKILL: f32 = 0.03;
/// A counter can be answered too, from the bind: the man it is aimed at
/// has just had his blow stopped, his blade is at the other man's, so
/// he may answer through his recovery. His chance of being on time is
/// his share times this, by how deep the exchange is (the blow 1, its
/// counter, the counter of that parried counter, and so on); beyond the
/// table he answers nothing, so no exchange runs on for ever. A counter
/// is never blocked, only parried back or taken on the body: the parry
/// earned it. FL_CHAIN as a comma list.
pub const CHAIN_SHARE: [f32; 4] = [1.0, 0.6, 0.3, 0.1];
/// On a parrying defend byte: the blow was stopped, the counter follows
/// at `PARRY_COUNTER_AT` (the charge bit, free in the defend state).
pub const GUARD_STOPPED: u8 = crate::units::SWING_CHARGE;
/// A blow the dice stopped leaves a twitch in the hit-flash column for
/// the render: values from 16 up are not a flash but the side the blow
/// came from (1 front, 2 his left, 3 his right, 4 behind) in the high
/// nibble and the ticks left in the low one. The kernel counts the low
/// nibble down and drops the value at zero. The render rocks him away
/// from it; the sim never reads it.
pub const TWITCH_TICKS: u8 = 9;

/// The twitch a flash value carries: the side the blow came from (1 to
/// 4) and how much is left, 1 at the blow; (0, 0) for a plain flash.
pub fn twitch_of(flash: u8) -> (u8, f32) {
    if flash < 16 {
        (0, 0.0)
    } else {
        (flash >> 4, f32::from(flash & 15) / f32::from(TWITCH_TICKS))
    }
}
/// Charge impact. A charge-flagged hit shoves its victim along
/// the blow: displacement = KNOCKBACK * m_a/(m_a+m_v), applied straight
/// to position (next tick's separation resolves the pile — same pipe as
/// all overlap). Heavy into light ~0.55 m: the line visibly DENTS.
const CHARGE_KNOCKBACK: f32 = 0.9;
/// The farthest the apply pass moves a man after the job wrote his
/// position (the charge shove above, at most): the job's list of men
/// near the map edge (sim/mod.rs) reaches this much further in, so the
/// death sweep's candidates hold every man the shove could carry out.
pub const KNOCKBACK_MARGIN: f32 = CHARGE_KNOCKBACK;
/// Braced walls barely budge (0.25x knockback) and never stagger.
const WALL_KNOCKBACK_RESIST: f32 = 0.25;
/// Stagger: a charge-flagged hit cancels the victim's swing and locks
/// him (no acting, steering, or turning) for ~1 s — the M2TW hit-stun
/// that lets a charge land a second blow before the answer. Pub: the
/// render sync derives the rocked-back pose progress from swing_t/this.
pub const STAGGER_TICKS: u8 = 30;
/// Per-hit stagger: a blow that landed and did not kill rocks its man
/// this often (M2TW: a hit that connects and does not kill shakes him).
/// Whether the defence stopped the blow was decided by the land roll
/// before this; nothing of the stats is left in the stumble. Charge
/// impacts and impalements stay certain (physics, not a contest). A
/// regular hit is a brief stumble; impacts stun full-length (the render
/// pose scales with remaining ticks, so stumbles read lighter than
/// impacts for free).
pub const STAGGER_P0: f32 = 0.65;
/// The arrow impact's stumble still rides the missile factor
/// (arrows.rs): the missile curve has no land roll of its own.
pub const STAGGER_P_PER_FACTOR: f32 = 0.05;
pub const STAGGER_P_MIN: f32 = 0.05;
/// Pub: the render normalizes the pose by the STUMBLE length, so a
/// stumble plays the full-strength rock for 0.5 s and an impact holds
/// it ~1 s — hierarchy by duration, not amplitude.
pub const HIT_STAGGER_TICKS: u8 = 15;

/// What became of a blow, the index into `DirTestStats::outcomes`.
pub const OUT_LANDED: usize = 0;
/// Met by a shield raised in time.
pub const OUT_BLOCKED: usize = 1;
/// Met by a blade raised in time (and countered).
pub const OUT_PARRIED: usize = 2;
/// The roll failed and the shield's points took the credit.
pub const OUT_SHIELD: usize = 3;
/// The roll failed and the skill's points took the credit: the blow was
/// turned, with no answer timed to it.
pub const OUT_SKILL: usize = 4;
/// The roll failed and the armour's points took the credit: a glance.
pub const OUT_ARMOUR: usize = 5;
/// The roll failed with no defence to credit: a miss.
pub const OUT_MISS: usize = 6;
pub const NUM_OUTCOMES: usize = 7;
pub const OUTCOME_NAMES: [&str; NUM_OUTCOMES] =
    ["landed", "blocked", "parried", "shield", "skill", "armour", "miss"];

/// FL_DEBUG_STAGGER=1: every landed hit staggers its victim (animation
/// debugging — normally the roll above decides).
fn debug_stagger() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FL_DEBUG_STAGGER").is_ok())
}

fn hit_rate() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::util::env_or("FL_HIT_RATE", HIT_RATE))
}

fn guard_per_skill() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::util::env_or("FL_GUARD", GUARD_PER_SKILL))
}

fn chain_share(depth: u8) -> f32 {
    static V: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    let table = V.get_or_init(|| {
        std::env::var("FL_CHAIN")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .filter(|t: &Vec<f32>| !t.is_empty())
            .unwrap_or_else(|| CHAIN_SHARE.to_vec())
    });
    table.get(usize::from(depth)).copied().unwrap_or(0.0)
}

pub fn counter_per_tick() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::util::env_or("FL_COUNTER_TICK", COUNTER_PER_TICK))
}

fn counter_chance(skill: f32) -> f32 {
    static V: std::sync::OnceLock<(f32, f32)> = std::sync::OnceLock::new();
    let (base, per_skill) = *V.get_or_init(|| {
        (
            crate::util::env_or("FL_COUNTER_BASE", COUNTER_CHANCE_BASE),
            crate::util::env_or("FL_COUNTER_SKILL", COUNTER_CHANCE_PER_SKILL),
        )
    });
    base + per_skill * skill
}

/// The chance a blow lands from its points, attack minus defence: M2TW's
/// hit curve (`Combat_rollHit`, verified in the decompile), a chance in
/// thousandths, piecewise linear with kinks at -6 and -12, times the hit
/// rate. Even points land 35 in 100; each point near parity is worth 3.5
/// in 100, half that from 6 behind, a fifth from 12 behind.
pub fn land_chance(d: f32) -> f32 {
    let per_mille = if d >= -6.0 {
        20.0 * d + 200.0
    } else if d >= -12.0 {
        10.0 * d + 140.0
    } else {
        4.0 * d + 68.0
    };
    (per_mille.clamp(1.0, 990.0) / 1000.0 * hit_rate()).clamp(0.01, 0.99)
}

/// The answer's timeline: ticks to raise the guard, ticks it stays up,
/// the whole clip.
pub fn guard_timeline(parry: bool) -> (u8, u8, u8) {
    if parry { (PARRY_RAISE, PARRY_UP, PARRY_LEN) } else { (BLOCK_RAISE, BLOCK_UP, BLOCK_LEN) }
}

/// Whether a defending man's guard is up at this point of his answer:
/// `swing_t` counts the answer down, and the last `len` ticks are the
/// clip (the ticks before them are the wait for the blow he saw coming).
pub fn guard_up(swing: u8, swing_t: u8) -> bool {
    let parry = swing & crate::units::SWING_STYLE_MASK == GUARD_PARRY;
    let (raise, up, len) = guard_timeline(parry);
    swing_t <= len && {
        let elapsed = len - swing_t;
        elapsed >= raise && elapsed < raise + up
    }
}

/// A swing, emitted by the parallel kernel and resolved here: its
/// warning when the wind-up starts, its blow when the strike tick comes.
pub struct DamageEvent {
    pub victim: u32,
    pub attacker: u32,
    /// Per-swing damage jitter (0.85-1.15); everything else — stats,
    /// attack sector, wall stances, the charge bonus — resolves in the
    /// apply pass, where both sides' state is known (a braced spearwall
    /// cancels the charge; the victim's yaw decides front/side/rear).
    pub jit: f32,
    /// Swing started at charging speed (momentum bonus).
    pub charge: bool,
    /// Not a swing at all: the victim ran onto a braced spear at charge
    /// speed (attacker is the spearman). His own charge bonus counts as
    /// attack points against him, and the point stops him (stagger, no
    /// knockback — momentum went INTO the spear).
    pub impale: bool,
    /// Nonzero: not a blow but its warning, this many ticks before the
    /// strike. The victim who sees it coming answers with a block or a
    /// parry timed to meet it (M2TW tells the target at the swing's
    /// start and schedules his reaction to its strike frame).
    pub warn: u8,
    /// With a warning: how deep in an exchange the swing is. 0 a plain
    /// blow, 1 the counter of a parried blow, 2 the counter of a parried
    /// counter, and so on (`CHAIN_SHARE`).
    pub chain: u8,
}

/// One event buffer per integrate chunk; allocations persist across ticks.
#[derive(Resource, Default)]
pub struct DamageBuffers(pub Vec<Vec<DamageEvent>>);

/// FL_TEST_DIR / FL_ARENA bookkeeping: every blow bucketed by victim
/// team, attack sector and outcome, filled by the damage apply pass,
/// logged from regiments.rs (the dir test reads blue victims, the arena
/// reads orange victims, the player's kills).
#[derive(Resource)]
pub struct DirTestStats {
    /// Indexed [victim team][front / side / rear].
    pub kills: [[u64; 3]; 2],
    /// Blows that reached their man, landed or not.
    pub hits: [[u64; 3]; 2],
    pub dmg: [[f64; 3]; 2],
    /// Indexed [victim team][OUT_*].
    pub outcomes: [[u64; NUM_OUTCOMES]; 2],
    /// Parries that stopped the blow, by how deep in an exchange the
    /// blow was: a plain blow, a counter, a counter's counter, on.
    pub chain: [u64; CHAIN_DEPTHS],
    pub enabled: bool,
}

/// How many depths of an exchange the stats count.
pub const CHAIN_DEPTHS: usize = 5;

impl Default for DirTestStats {
    fn default() -> Self {
        Self {
            kills: [[0; 3]; 2],
            hits: [[0; 3]; 2],
            dmg: [[0.0; 3]; 2],
            outcomes: [[0; NUM_OUTCOMES]; 2],
            chain: [0; CHAIN_DEPTHS],
            enabled: std::env::var("FL_TEST_DIR").is_ok() || std::env::var("FL_ARENA").is_ok(),
        }
    }
}

/// Ticks a corpse persists (death anim) before swap-removal.
pub const DEATH_TICKS: u8 = 18;
/// Ticks of hit flash after taking damage.
const FLASH_TICKS: u8 = 4;

/// What the apply reads beside the soldier columns: the tick's regiment
/// snapshot and scalars, taken from the finished job.
pub struct ApplyContext<'a> {
    /// Wall stance per regiment (0 none, 1 shieldwall, 2 spearwall).
    pub wall: &'a [u8],
    /// Regiments too tired for a charge bonus.
    pub fat_nocharge: &'a [bool],
    pub bounds_min: Vec2,
    pub bounds_max: Vec2,
    pub combat_scale: f32,
    pub tick_seed: u32,
    pub terrain: &'a Terrain,
}

/// The sound of a blow: where it was struck and what it struck. Audio
/// only; the sim never reads it.
fn blow_sound(
    sounds: &mut Option<&mut crate::audio::SoundEvents>,
    at: (Vec3, Vec3, u32, u32),
    material: crate::audio::Material,
    killed: bool,
) {
    if let Some(se) = sounds.as_deref_mut() {
        se.push_blow(crate::audio::Blow {
            victim: at.0,
            attacker: at.1,
            victim_group: at.2,
            attacker_group: at.3,
            material,
            killed,
        });
    }
}

/// Serial damage apply: deterministic (chunk order), race-free, and the
/// single place where hp transitions to death. A swing whiffs when its
/// victim died mid-wind-up, changed team slot via swap-remove, or slipped
/// out of reach — checked here, where all columns are whole again.
/// Returns the number of events applied this tick.
pub fn apply_damage(
    buffers: &mut [Vec<DamageEvent>],
    units: &mut Units,
    groups: &mut Groups,
    ctx: &ApplyContext,
    cstats: &mut crate::combat::CombatStats,
    dir_stats: &mut DirTestStats,
    mut sounds: Option<&mut crate::audio::SoundEvents>,
) -> usize {
    let Units {
        pos,
        pos_prev,
        team,
        kind,
        yaw,
        group,
        hp,
        target,
        swing,
        swing_t,
        flash,
        death_t,
        chain,
        ..
    } = units;
    let team = &team[..];
    let kind = &kind[..];
    let group = &group[..];
    let wall = ctx.wall;
    let fat_nocharge = ctx.fat_nocharge;
    let bounds_min = ctx.bounds_min;
    let bounds_max = ctx.bounds_max;
    let combat_scale = ctx.combat_scale;
    let tick_seed = ctx.tick_seed;
    let terrain = ctx.terrain;
    let mut events = 0usize;
    {
        let _span = info_span!("damage_apply").entered();
        // Fatigue stat effects (MTW1 table): weary arms strike with
        // fewer attack points; exhausted men get no charge bonus at all.
        let fat_atk: Vec<f32> = groups
            .list
            .iter()
            .map(|g| crate::fatigue::attack_penalty(g.fatigue))
            .collect();
        let broken: Vec<bool> = groups.list.iter().map(|g| g.state.is_broken()).collect();
        for buf in buffers.iter_mut() {
            events += buf.len();
            for ev in buf.drain(..) {
                let (v, a) = (ev.victim as usize, ev.attacker as usize);
                if v >= hp.len() || a >= hp.len() {
                    continue;
                }
                if hp[v] <= 0.0 || death_t[v] > 0 || team[v] == team[a] {
                    continue;
                }
                let pa = &TYPES[kind[a] as usize];
                let pv = &TYPES[kind[v] as usize];
                let fwd = Vec2::new(yaw[v].sin(), yaw[v].cos());
                let dir = (pos_prev[a].xz() - pos_prev[v].xz()).normalize_or_zero();
                let along = fwd.dot(dir);
                let rear = along < -SECTOR_COS_60;
                let front = along >= SECTOR_COS_60;
                // Victim's left in the xz plane (facing +Z -> left = +X).
                let left_side = !front && !rear && dir.dot(Vec2::new(fwd.y, -fwd.x)) > 0.0;
                let roll = |k: u32| {
                    crate::units::hash01(
                        tick_seed
                            ^ ev.victim.wrapping_mul(k)
                            ^ ev.attacker.wrapping_mul(0x1656_67B1),
                    )
                };

                if ev.warn > 0 {
                    // The warning of a swing. He answers it if he is free
                    // to (ready, or past the follow-through of his own
                    // last blow, not reeling, not already answering
                    // another) and sees the attacker in front of him
                    // (within 90 degrees). Routed men answer nothing.
                    // The counter of his own parried blow he may answer
                    // from the bind, anywhere in his recovery.
                    let vs = swing[v];
                    let state = vs & crate::units::SWING_STATE_MASK;
                    let free = match state {
                        crate::units::SWING_READY => true,
                        crate::units::SWING_RECOVER => {
                            vs & crate::units::SWING_STAGGERED == 0
                                && (u16::from(swing_t[v]) + u16::from(pv.windup_ticks)
                                    <= u16::from(pv.cooldown_ticks)
                                    || (ev.chain > 0 && target[v] == ev.attacker))
                        }
                        _ => false,
                    };
                    let share = chain_share(ev.chain);
                    if !free || share <= 0.0 || along < 0.0 || broken[group[v] as usize] {
                        continue;
                    }
                    // Which answer: the shield meets a blow from its side,
                    // except when he chooses the parry; a blow from the
                    // open side, or a man without a shield, parries. A
                    // counter is never blocked: the parry that earned it
                    // is answered with the blade or taken on the body.
                    let shield_side = pv.shield > 0.0 && (front || left_side);
                    let parry = ev.chain > 0 || !shield_side || roll(0x27D4_EB2F) < pv.parry_share;
                    // On time: his skill, less what fatigue took from his
                    // arm. A late answer raises the guard as the blow
                    // arrives and still costs him the clip; a late parry
                    // meets the blow with his guard open.
                    let skill = (pv.defence_skill + fat_atk[group[v] as usize]).max(0.0);
                    let on_time = roll(0x9E37_79B9) < skill * guard_per_skill() * share;
                    let (raise, _, len) = guard_timeline(parry);
                    let wait = if on_time && ev.warn >= raise { ev.warn - raise } else { ev.warn };
                    swing[v] = crate::units::SWING_DEFEND
                        | if parry { GUARD_PARRY } else { GUARD_BLOCK }
                        | (vs & crate::units::SWING_STAGGER_IMMUNE);
                    swing_t[v] = wait.saturating_add(len);
                    target[v] = ev.attacker;
                    chain[v] = ev.chain;
                    continue;
                }

                let reach = pa.reach * 1.15;
                if pos_prev[v].xz().distance_squared(pos_prev[a].xz()) > reach * reach {
                    continue;
                }
                let vt = team[v] as usize;
                let sector = if front { 0 } else if !rear { 1 } else { 2 };
                if dir_stats.enabled {
                    dir_stats.hits[vt][sector] += 1;
                }
                let blow_at = (pos[v], pos[a], group[v], group[a]);

                // A raised guard meets the blow: the victim is answering
                // a swing, his shield or blade is up now, and the blow
                // comes from within 45 degrees of his front (the swing
                // he answered, or another that arrived while he was
                // up). Stopped before the stats say anything. A parry
                // that stops the blow counters, most of the time: a
                // stab from the bind at half the wind-up, while the
                // attacker is still recovering. A parry that is not up
                // yet has his guard open: his skill counts for nothing
                // against this blow.
                let v_defending =
                    swing[v] & crate::units::SWING_STATE_MASK == crate::units::SWING_DEFEND;
                let v_parrying =
                    v_defending && swing[v] & crate::units::SWING_STYLE_MASK == GUARD_PARRY;
                let v_guard_up = v_defending && guard_up(swing[v], swing_t[v]);
                if v_guard_up && along >= GUARD_COS_45 && !ev.impale {
                    let outcome = if v_parrying {
                        // Whether the counter follows (the kernel
                        // launches it): his skill, less what fatigue
                        // took from his arm.
                        let skill = (pv.defence_skill + fat_atk[group[v] as usize]).max(0.0);
                        if roll(0x2545_F491) < counter_chance(skill) {
                            swing[v] |= GUARD_STOPPED;
                            target[v] = ev.attacker;
                        }
                        if dir_stats.enabled {
                            dir_stats.chain[usize::from(chain[v]).min(CHAIN_DEPTHS - 1)] += 1;
                        }
                        OUT_PARRIED
                    } else {
                        OUT_BLOCKED
                    };
                    let material = if v_parrying {
                        crate::audio::Material::Steel
                    } else {
                        crate::audio::Material::Wood
                    };
                    blow_sound(&mut sounds, blow_at, material, false);
                    if dir_stats.enabled {
                        dir_stats.outcomes[vt][outcome] += 1;
                    }
                    continue;
                }

                // M2TW directional resolution, the points of the land
                // roll. Sector of the blow against the victim's facing:
                // defence skill counts in full from the front, half from
                // the sides and not from the rear; the shield covers front
                // + LEFT side only (shield arm); armour counts from
                // everywhere, halved by armour-piercing weapons; the open
                // side and the back give the blow points of their own. A
                // soldier struck from behind loses skill+shield entirely:
                // facing is the defensive resource, and rear charges kill
                // before morale even moves.
                // Wall stances are stat points: a shieldwall doubles
                // down on cover (only where the shield counts — it still
                // has no back), a braced spearwall strikes harder, and a
                // braced spearwall VICTIM nullifies the attacker's charge
                // momentum (points beat momentum).
                let a_wall = wall[group[a] as usize];
                let v_wall = wall[group[v] as usize];
                let mut attack = pa.attack
                    + fat_atk[group[a] as usize]
                    + match a_wall {
                        1 => SHIELDWALL_ATK_PTS,
                        2 => SPEARWALL_ATK_PTS,
                        _ => 0.0,
                    };
                if ev.charge && !fat_nocharge[group[a] as usize] {
                    attack += pa.charge_bonus;
                }
                // Impalement: he ran onto this spearman's braced point at
                // speed (integrate's spear-line collision) — his OWN
                // momentum is the attack bonus, spent against himself.
                // No stance rules anywhere: a charger who slipped between
                // the spear lines and reached the wielder gets his full
                // charge bonus above like anyone else.
                if ev.impale {
                    attack += pv.charge_bonus;
                }
                if rear {
                    attack += REAR_PTS;
                } else if !front && !left_side {
                    attack += OPEN_SIDE_PTS;
                }
                // A staggered man cannot parry (M2TW: no blocking while
                // staggering — the outnumbered-men snowball): his defence
                // SKILL is gone until he recovers; the shield still
                // passively covers and armour always counts. So is the
                // skill of a man caught with his blade not yet up.
                let v_staggered = swing[v] & crate::units::SWING_STAGGERED != 0;
                let skill = if rear || v_staggered || v_parrying {
                    0.0
                } else if front {
                    pv.defence_skill
                } else {
                    0.5 * pv.defence_skill
                };
                let shield = if front || left_side {
                    pv.shield + if v_wall == 1 { SHIELDWALL_SHIELD_PTS } else { 0.0 }
                } else {
                    0.0
                };
                let armour = if pa.ap { pv.armour * 0.5 } else { pv.armour };
                let d = attack - (skill + armour + shield);
                if roll(0x85EB_CA6B) >= land_chance(d) {
                    // The stats turned it. Which defence gets the credit
                    // is drawn in proportion to the points each put up;
                    // the sound and the counts read it, the sim never
                    // does.
                    let total = skill + shield + armour;
                    let x = roll(0xC2B2_AE35) * total;
                    let (outcome, material) = if total <= 0.0 {
                        (OUT_MISS, None)
                    } else if x < skill {
                        (OUT_SKILL, Some(crate::audio::Material::Steel))
                    } else if x < skill + shield {
                        (OUT_SHIELD, Some(crate::audio::Material::Wood))
                    } else {
                        (OUT_ARMOUR, Some(crate::audio::armour_material(kind[v])))
                    };
                    if let Some(m) = material {
                        blow_sound(&mut sounds, blow_at, m, false);
                    }
                    // The render's cue: he twitches away from the side
                    // the blow came from, unless he is reeling. A man
                    // whose answer clip is playing shows that instead;
                    // a kind without the clip shows the twitch.
                    if !v_staggered && flash[v] < 16 {
                        let side: u8 = if front { 1 } else if rear { 4 } else if left_side { 2 } else { 3 };
                        flash[v] = (side << 4) | TWITCH_TICKS;
                    }
                    if dir_stats.enabled {
                        dir_stats.outcomes[vt][outcome] += 1;
                    }
                    continue;
                }
                let dmg = BASE_DMG[pa.weapon as usize] * ev.jit * combat_scale;
                hp[v] -= dmg;
                flash[v] = FLASH_TICKS;
                let died = hp[v] <= 0.0;
                if died {
                    death_t[v] = DEATH_TICKS;
                    // kills[] counts losses OF that team (overlay semantics).
                    cstats.kills[team[v] as usize] += 1;
                    groups.list[group[v] as usize].recent_deaths += 1;
                    // The attacker's regiment is winning its exchange.
                    groups.list[group[a] as usize].recent_kills += 1;
                }
                // The blow's sound: what it struck.
                blow_sound(&mut sounds, blow_at, crate::audio::armour_material(kind[v]), died);
                // Charge impact: momentum becomes a shove and a
                // stun. Walls barely budge and never stagger; a braced
                // SPEARWALL additionally reflects the charge bonus onto
                // the charger (points punish momentum) — the M2TW rule.
                if !died {
                    // Impacts always rock their man; an ordinary blow that
                    // landed stumbles him most of the time.
                    let certain = ev.charge || ev.impale || debug_stagger();
                    let p = if certain { 1.0 } else { STAGGER_P0 };
                    let stumble = roll(0x9E37) < p;
                    // Bracing is posture physics: a wall hit FRONTALLY has
                    // its weight planted — quarter knockback, no stagger;
                    // from the flank or rear it is bodies like any others.
                    let braced = v_wall != 0 && front;
                    if ev.charge {
                        let m_a = pa.mass;
                        let m_v = pv.mass;
                        let resist = if braced { WALL_KNOCKBACK_RESIST } else { 1.0 };
                        let shove = -dir * CHARGE_KNOCKBACK * (m_a / (m_a + m_v)) * resist;
                        let mut nx = (pos[v].x + shove.x).clamp(bounds_min.x, bounds_max.x);
                        let mut nz = (pos[v].z + shove.y).clamp(bounds_min.y, bounds_max.y);
                        // Charges can't knock a man into the river or
                        // through a terrace riser — the shove just dies.
                        if terrain.blocked_at(nx, nz) {
                            nx = pos[v].x;
                            nz = pos[v].z;
                        }
                        pos[v] = Vec3::new(
                            nx,
                            terrain.height_at(nx, nz)
                                + crate::unit_types::half_height(kind[v] as usize),
                            nz,
                        );
                    }
                    // An impaled runner is STOPPED, not thrown — his
                    // momentum went into the point; the stagger is the
                    // stop.
                    if !braced && stumble {
                        if swing[v] & crate::units::SWING_STAGGER_IMMUNE != 0 {
                            // The free pass earned by the last stagger is
                            // spent absorbing this one.
                            swing[v] &= !crate::units::SWING_STAGGER_IMMUNE;
                        } else {
                            let stun = if certain {
                                STAGGER_TICKS
                            } else {
                                HIT_STAGGER_TICKS
                            };
                            // A man in the middle of an answer keeps
                            // none of it: the blow that got through
                            // took his guard with it.
                            let held = if v_defending { 0 } else { swing_t[v] };
                            swing[v] = crate::units::SWING_RECOVER
                                | crate::units::SWING_STAGGERED;
                            swing_t[v] = held.max(stun);
                        }
                    }
                }
                if dir_stats.enabled {
                    dir_stats.dmg[vt][sector] += dmg as f64;
                    dir_stats.outcomes[vt][OUT_LANDED] += 1;
                    if died {
                        dir_stats.kills[vt][sector] += 1;
                    }
                }
            }
        }
    }
    events
}
