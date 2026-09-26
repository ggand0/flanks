//! The serial damage apply: every landed swing, arrow impact aside, is
//! resolved here after the parallel kernel, in chunk order, so the sim
//! stays deterministic and hp reaches death in exactly one place.

use bevy::math::Vec3Swizzles;
use bevy::prelude::*;

use crate::orders::Groups;
use crate::terrain::Terrain;
use crate::unit_types::{BASE_DMG, FACTOR_CLAMP, FACTOR_MULT, TYPES};
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
/// Charge impact (phase B). A charge-flagged hit shoves its victim along
/// the blow: displacement = KNOCKBACK * m_a/(m_a+m_v), applied straight
/// to position (next tick's separation resolves the pile — same pipe as
/// all overlap). Heavy into light ~0.55 m: the line visibly DENTS.
const CHARGE_KNOCKBACK: f32 = 0.9;
/// Braced walls barely budge (0.25x knockback) and never stagger.
const WALL_KNOCKBACK_RESIST: f32 = 0.25;
/// Stagger: a charge-flagged hit cancels the victim's swing and locks
/// him (no acting, steering, or turning) for ~1 s — the M2TW hit-stun
/// that lets a charge land a second blow before the answer. Pub: the
/// render sync derives the rocked-back pose progress from swing_t/this.
pub const STAGGER_TICKS: u8 = 30;
/// Per-hit stagger (M2TW: a hit that connects, is not blocked, and does
/// not kill staggers its victim — and blocking IS the defence stats
/// working). The chance rides the combat factor, the continuous "did
/// the defence stop it": even matchup ~0.65, rear hits ~1.0 (nobody
/// parries what he cannot see — skill and shield are already zero from
/// behind), elite shieldwall frontal ~0.3 (they block and parry a lot),
/// and AP-vs-armour staggers more purely through the halved armour.
/// Charge impacts and impalements stay certain (physics, not a parry
/// contest). A regular hit is a brief stumble; impacts stun full-length
/// (the render pose scales with remaining ticks, so stumbles read
/// lighter than impacts for free).
// Shared with the arrow-impact resolution (arrows.rs): a nonfatal hit
// rolls the same stumble regardless of what delivered it.
pub const STAGGER_P0: f32 = 0.65;
pub const STAGGER_P_PER_FACTOR: f32 = 0.05;
pub const STAGGER_P_MIN: f32 = 0.05;
/// Pub: the render normalizes the pose by the STUMBLE length, so a
/// stumble plays the full-strength rock for 0.5 s and an impact holds
/// it ~1 s — hierarchy by duration, not amplitude.
pub const HIT_STAGGER_TICKS: u8 = 15;

/// FL_DEBUG_STAGGER=1: every landed hit staggers its victim (animation
/// debugging — normally the roll above decides).
fn debug_stagger() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FL_DEBUG_STAGGER").is_ok())
}


/// A landed swing, resolved after the parallel integrate: chunks emit into
/// their own buffer (no write races), then a serial pass applies damage.
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
}

/// One event buffer per integrate chunk; allocations persist across ticks.
#[derive(Resource, Default)]
pub struct DamageBuffers(pub Vec<Vec<DamageEvent>>);

/// FL_TEST_DIR / FL_ARENA bookkeeping: every hit bucketed by victim team
/// and attack sector, filled by the damage apply pass, logged from
/// regiments.rs (the dir test reads blue victims, the arena reads orange
/// victims — the player's kills).
#[derive(Resource)]
pub struct DirTestStats {
    /// Indexed [victim team][front / side / rear].
    pub kills: [[u64; 3]; 2],
    pub hits: [[u64; 3]; 2],
    pub dmg: [[f64; 3]; 2],
    pub enabled: bool,
}

impl Default for DirTestStats {
    fn default() -> Self {
        Self {
            kills: [[0; 3]; 2],
            hits: [[0; 3]; 2],
            dmg: [[0.0; 3]; 2],
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
) -> usize {
    let Units {
        pos,
        pos_prev,
        team,
        kind,
        yaw,
        group,
        hp,
        swing,
        swing_t,
        flash,
        death_t,
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
                let reach = pa.reach * 1.15;
                if pos_prev[v].xz().distance_squared(pos_prev[a].xz()) > reach * reach {
                    continue;
                }
                // M2TW directional resolution. Sector of the blow against
                // the victim's facing: defence skill counts vs front and
                // side but NOT rear; the shield covers front + LEFT side
                // only (shield arm); armour counts from everywhere, halved
                // by armour-piercing weapons. A soldier struck from behind
                // loses skill+shield entirely — facing is the defensive
                // resource, and rear charges kill before morale even moves.
                let pv = &TYPES[kind[v] as usize];
                let fwd = Vec2::new(yaw[v].sin(), yaw[v].cos());
                let dir = (pos_prev[a].xz() - pos_prev[v].xz()).normalize_or_zero();
                let along = fwd.dot(dir);
                let rear = along < -SECTOR_COS_60;
                let front = along >= SECTOR_COS_60;
                // Victim's left in the xz plane (facing +Z -> left = +X).
                let left_side = !front && !rear && dir.dot(Vec2::new(fwd.y, -fwd.x)) > 0.0;

                // Wall stances are stat points now: a shieldwall doubles
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
                // A staggered man cannot parry (M2TW: no blocking while
                // staggering — the outnumbered-men snowball): his defence
                // SKILL is gone until he recovers; the shield still
                // passively covers and armour always counts.
                let v_staggered = swing[v] & crate::units::SWING_STAGGERED != 0;
                let skill = if rear || v_staggered {
                    0.0
                } else {
                    pv.defence_skill
                };
                let shield = if front || left_side {
                    pv.shield + if v_wall == 1 { SHIELDWALL_SHIELD_PTS } else { 0.0 }
                } else {
                    0.0
                };
                let armour = if pa.ap { pv.armour * 0.5 } else { pv.armour };
                let factor =
                    (attack - (skill + armour + shield)).clamp(-FACTOR_CLAMP, FACTOR_CLAMP);
                let dmg = BASE_DMG[pa.weapon as usize]
                    * FACTOR_MULT.powf(factor)
                    * ev.jit
                    * combat_scale;
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
                // Charge impact (phase B): momentum becomes a shove and a
                // stun. Walls barely budge and never stagger; a braced
                // SPEARWALL additionally reflects the charge bonus onto
                // the charger (points punish momentum) — the M2TW rule.
                if !died {
                    // Impacts always rock their man; an ordinary blow
                    // staggers when it got through cleanly — the roll
                    // rides the same factor the damage did.
                    let certain = ev.charge || ev.impale || debug_stagger();
                    let p = if certain {
                        1.0
                    } else {
                        (STAGGER_P0 + STAGGER_P_PER_FACTOR * factor)
                            .clamp(STAGGER_P_MIN, 1.0)
                    };
                    let roll = crate::units::hash01(
                        tick_seed
                            ^ (ev.victim.wrapping_mul(0x9E37))
                            ^ (ev.attacker.wrapping_mul(0x85EB)),
                    );
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
                    if !braced && roll < p {
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
                            swing[v] = crate::units::SWING_RECOVER
                                | crate::units::SWING_STAGGERED;
                            swing_t[v] = swing_t[v].max(stun);
                        }
                    }
                }
                if dir_stats.enabled {
                    let vt = team[v] as usize;
                    let s = if front { 0 } else if !rear { 1 } else { 2 };
                    dir_stats.hits[vt][s] += 1;
                    dir_stats.dmg[vt][s] += dmg as f64;
                    if died {
                        dir_stats.kills[vt][s] += 1;
                    }
                }
            }
        }
    }
    events
}
