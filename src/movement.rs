//! Fixed-tick movement sim: goal steering toward a moving target point plus
//! boids-style separation from the spatial grid. Parallelized over SoA chunks
//! with the compute task pool.

use bevy::math::Vec3Swizzles;
use bevy::prelude::*;
use bevy::tasks::ComputeTaskPool;
use std::time::Instant;

use crate::orders::Groups;
use crate::spatial::SpatialGrid;
use crate::terrain::Terrain;
use crate::unit_types::{BASE_DMG, FACTOR_CLAMP, FACTOR_MULT, TYPES};
use crate::units::Units;

const SEP_RADIUS: f32 = 1.4;
/// FL_RECTFIGHT: rest distance for CROSS-TEAM pairs. Currently equal
/// to SEP_RADIUS (same spacing as same-team). Lower values let enemy
/// bodies enter the formation grid's gaps and produce a multi-rank
/// fighting band; play-testing 0.95–1.25 found the visual clipping
/// unacceptable with the current model scale, so it sits at parade
/// spacing for now. Tuning target for when model proportions are
/// revisited.
const ENEMY_SEP_RADIUS: f32 = 1.4;
const SEP_STRENGTH: f32 = 60.0;
/// Base neighbor query radius: covers separation and sword reach. Kinds
/// with longer reach (spears) widen their own scan per unit — the sword
/// majority must not pay for the spear's reach.
const QUERY_RADIUS: f32 = 2.0;
/// Cap on summed separation push to avoid explosive forces deep in a crowd.
const SEP_PUSH_MAX: f32 = 2.5;
/// Below this distance overlap is resolved positionally (bodies are
/// ~0.6 m wide).
const HARD_RADIUS: f32 = 0.9;
/// Crowd-density yield: goal drive fades to zero between these two local
/// density values. Scalar density can't cancel out the way opposing push
/// vectors do, so this is what stops a goal-seeking crowd from compressing
/// itself into overlap: packed interior units genuinely stop shoving.
const CROWD_SLOW: f32 = 1.2;
const CROWD_STOP: f32 = 2.5;
const STEER_GAIN: f32 = 3.0;
const MAX_ACCEL: f32 = 50.0;
/// Facing turn smoothing (fraction/s of the remaining angle) — the
/// settle behavior for SMALL corrections.
const YAW_RATE: f32 = 10.0;
/// A blooded soldier (one whose combat memo holds a living enemy) keeps
/// facing the fight while that enemy stands within this range, instead
/// of dressing back to the ordered facing between swing cycles. Fresh
/// men still hold the line — this is memory of a fight, not proximity
/// awareness, so rear ambushes keep their first-blood advantage.
const KEEP_FACING_R: f32 = 6.0;
/// Hard angular speed cap (rad/s). The smoothing above is exponential,
/// so before this cap a 180° about-face finished in ~0.2 s — rear-
/// attacked soldiers whipped around between two swings and rear
/// attacks played out as frontal fights. A burdened man in ranks
/// turns ~180° in about a second: pi rad/s. Small angles never hit
/// the cap, so marching wheels and duel tracking feel unchanged.
const TURN_SPEED_MAX: f32 = std::f32::consts::PI;
/// Positional overlap resolution: fraction of pair overlap corrected per
/// tick per unit, and the per-tick cap on total correction distance.
/// UNDER-relaxed on purpose: resolving overlap over ~3-4 ticks reads as
/// bodies settling; resolving it in 1-2 reads as twitching.
const CORR_GAIN: f32 = 0.3;
const CORR_MAX: f32 = 0.1;
/// Units slow down proportionally inside this distance to the target.
const ARRIVE_RADIUS: f32 = 35.0;
const CHUNK: usize = 2048;

/// Routing units flee at this fraction of their speed: fleeing at
/// exactly max speed made pursuit a zero-kill treadmill.
const ROUT_FLEE_FRAC: f32 = 0.9;
/// Sparse-fight acquisition radius. Two depleted formations interleave
/// with gaps wider than QUERY_RADIUS and the fight stalls (hits/tick 0
/// with both sides standing at "the front"). Pressing units whose normal
/// scan finds no enemy look this far on a staggered 1-in-8-tick cadence
/// and lock a closing target; swings still require reach.
const WIDE_ACQUIRE_R: f32 = 4.0;
/// Speed fraction above which a newly started swing counts as a charge.
/// The momentum bonus (attacker's charge_bonus attack points) resolves in
/// the damage pass — a braced SPEARWALL nullifies it (points beat momentum).
const CHARGE_SPEED_FRAC: f32 = 0.6;
/// A regiment in its charge phase runs the last stretch home.
const CHARGE_SPEED_BOOST: f32 = 1.15;
/// Walls advance at a deliberate pace — breaking into a run breaks the wall.
const WALL_SPEED_FRAC: f32 = 0.72;
/// Separation rest distance between two SAME-TEAM units who are both in a
/// wall stance: shoulder to shoulder. Without this the physics pushes a
/// wall back out to normal spacing and the tight slots never happen.
const WALL_SEP_RADIUS: f32 = 1.05;
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
const STAGGER_P0: f32 = 0.65;
const STAGGER_P_PER_FACTOR: f32 = 0.05;
const STAGGER_P_MIN: f32 = 0.05;
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

/// The braced spear is a PHYSICAL hazard, not a stance rule: a
/// spearwall soldier's leveled point covers a line along his facing,
/// from POINT_MIN (get inside it and the spear is useless) out to his
/// reach, this wide. A body crossing that line at charge speed runs
/// onto the point — a damage event from the spearman powered by the
/// CHARGER's own momentum, and a stop (stagger). A charger who slips
/// between the lines (rank edge, dead wielder, hole in the wall)
/// reaches the spearman and swings with his full charge bonus: gaps
/// are real, coverage at 1.05 m wall pitch is ~2/3 of the frontage.
const SPEAR_POINT_MIN: f32 = 0.9;
const SPEAR_LINE_HALF_W: f32 = 0.35;

/// Global damage multiplier. FL_COMBAT_SCALE overrides (fast test battles).
#[derive(Resource)]
pub struct CombatScale(pub f32);

impl Default for CombatScale {
    fn default() -> Self {
        Self(crate::util::env_or("FL_COMBAT_SCALE", 1.0))
    }
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

#[derive(Resource, Default)]
pub struct SimStats {
    pub grid_ms: f32,
    pub step_ms: f32,
    /// Density splat + blur + contour (frontline.rs), per tick.
    pub field_ms: f32,
    /// Last nn_audit sweep cost (runs every ~2 s).
    pub audit_ms: f32,
    /// Landed-swing events this tick (damage apply pass).
    pub events: usize,
    /// Mean per-tick displacement in meters (audit cadence). The twitch
    /// metric: a static line should sit near zero, a marching unit ~0.3.
    pub move_avg: f32,
    /// Smallest nearest-neighbor distance across all units (sampled every
    /// couple of seconds). Cube width is 0.62 — below that means overlap.
    pub nn_min: f32,
    pub nn_avg: f32,
}

/// Debug gizmo master toggle (G).
#[derive(Resource)]
pub struct DebugViz(pub bool);

pub struct MovementPlugin;

impl Plugin for MovementPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimStats>()
            .init_resource::<CombatScale>()
            .init_resource::<DamageBuffers>()
            .init_resource::<DirTestStats>()
            .insert_resource(DebugViz(true))
            .init_resource::<SpatialGrid>()
            .add_systems(FixedUpdate, step_sim.in_set(crate::game_state::SimSet))
            .add_systems(Update, toggle_debug_viz);
    }
}

#[allow(clippy::too_many_arguments)] // bevy system params
pub fn step_sim(
    mut units: ResMut<Units>,
    mut grid: ResMut<SpatialGrid>,
    mut damage: ResMut<DamageBuffers>,
    (mut arrow_spawns, tracks): (
        ResMut<crate::arrows::ArrowSpawns>,
        Res<crate::arrows::RegTracks>,
    ),
    mut cstats: ResMut<crate::combat::CombatStats>,
    mut groups: ResMut<Groups>,
    terrain: Res<Terrain>,
    scale: Res<CombatScale>,
    time: Res<Time>,
    mut stats: ResMut<SimStats>,
    mut tick: Local<u32>,
    mut wall_flags: Local<Vec<bool>>,
    mut broken_flags: Local<Vec<bool>>,
    mut mover_flags: Local<Vec<bool>>,
    mut dir_stats: ResMut<DirTestStats>,
    mut yaw_snapshot: Local<Vec<f32>>,
) {
    let dt = time.delta_secs();
    let Units {
        pos,
        pos_prev,
        vel,
        speed,
        team,
        kind,
        yaw,
        yaw_prev,
        group,
        hp,
        target,
        swing,
        swing_t,
        flash,
        death_t,
        home,
        ammo,
        ..
    } = &mut *units;
    if pos.is_empty() {
        return;
    }

    // pos_prev becomes the state at tick start; pos is fully rewritten below.
    std::mem::swap(pos, pos_prev);

    // Per-unit wall flag for the grid meta (same-team wall pairs pack
    // tighter in the separation below).
    let group_wall: Vec<bool> = groups
        .list
        .iter()
        .map(|g| crate::formation::wall_kind(g) != 0)
        .collect();
    wall_flags.clear();
    wall_flags.extend(group.iter().map(|&g| group_wall[g as usize]));
    // Pass-through flags for the grid meta (FL_RECTFIGHT): a BROKEN
    // man is a fleeing body, and a man of a MOVE-ordered regiment is a
    // body deliberately passing through (the engine's explicit
    // formationMovingThrough state) — both collide at body scale
    // instead of commanding the 1.4 m rank-dressing courtesy, so a
    // rout or an ordered withdrawal slips THROUGH a formed line's
    // seams instead of excavating a corridor through the formation.
    let group_broken: Vec<bool> = groups.list.iter().map(|g| g.state.is_broken()).collect();
    let group_mover: Vec<bool> = groups
        .list
        .iter()
        .map(|g| !g.state.is_broken() && matches!(g.order, Some(crate::orders::Order::Move(_))))
        .collect();
    broken_flags.clear();
    broken_flags.extend(group.iter().map(|&g| group_broken[g as usize]));
    mover_flags.clear();
    mover_flags.extend(group.iter().map(|&g| group_mover[g as usize]));
    // No packing rule for fighting regiments: vanilla M2TW keeps its
    // formation grid (and observably LOOSENS it) during melee — the
    // fighting crowd's spacing is slots + body collision, nothing
    // else. A shoulder-to-shoulder press rest was tried here (bit 5,
    // META_PRESS) and it sealed the very seams the intermix needs:
    // a pressed front's gaps shrank to ~1.05 m against 0.95 m bodies
    // and symmetric fights collapsed to a two-rank duel line.
    let rf = crate::formation::rectfight();

    let t0 = Instant::now();
    {
        let _span = info_span!("grid_rebuild").entered();
        grid.rebuild(pos_prev, team, kind, death_t, &wall_flags, &broken_flags, &mover_flags);
    }
    let t1 = Instant::now();
    stats.grid_ms = (t1 - t0).as_secs_f32() * 1000.0;

    // ---- Archer fire solutions (regiment level). An attack order for a
    // ranged regiment is a FIRE order, not a melee charge: the regiment
    // halts once the target is inside range (the stand-off below feeds
    // the order resolution) and volleys from where it stands; the march
    // resumes if the target slips back out of range.
    let n_groups = groups.list.len();
    let standoff: Vec<bool> = groups
        .list
        .iter()
        .map(|gd| {
            gd.kind == crate::unit_types::KIND_ARCHER
                && gd.count > 0
                && !gd.state.is_broken()
                && matches!(gd.order, Some(crate::orders::Order::Attack(t))
                    if groups.list[t as usize].count > 0
                        && groups.list[t as usize].centroid.distance(gd.centroid)
                            < crate::unit_types::missile::RANGE * 0.85)
        })
        .collect();
    for (g, &so) in standoff.iter().enumerate() {
        // Entering the stand-off freezes the frame here (the same anchor
        // snap engagement does): holding units dress where they stopped
        // instead of walking back to a stale anchor.
        if so && groups.list[g].anchor.distance(groups.list[g].centroid) > 2.0 {
            let c = groups.list[g].centroid;
            groups.list[g].anchor = c;
        }
    }
    // What each archer regiment's bows shoot at this tick: the ordered
    // target once standing off, else fire-at-will's nearest live enemy
    // regiment in range. Regiments in melee or on the march don't volley
    // (M2TW: foot archers halt to shoot; fire-at-will pauses on the move).
    struct ShootAt {
        c: Vec2,
        vel: Vec2,
        r: f32,
    }
    let shoot_at: Vec<Option<ShootAt>> = (0..n_groups)
        .map(|g| {
            let gd = &groups.list[g];
            if gd.kind != crate::unit_types::KIND_ARCHER
                || gd.count == 0
                || gd.state.is_broken()
                || gd.engaged
            {
                return None;
            }
            let t = match gd.order {
                Some(crate::orders::Order::Move(_)) => return None,
                Some(crate::orders::Order::Attack(t)) => {
                    // An explicit target overrides fire-at-will: no
                    // shots until the march brings it into range.
                    if !standoff[g] || groups.list[t as usize].count == 0 {
                        return None;
                    }
                    t as usize
                }
                None => {
                    // Fire-at-will: nearest live, unbroken enemy block.
                    if !gd.fire_at_will {
                        return None;
                    }
                    let mut best: Option<(f32, usize)> = None;
                    for (e, eg) in groups.list.iter().enumerate() {
                        if eg.team == gd.team || eg.count == 0 || eg.state.is_broken() {
                            continue;
                        }
                        let d2 = eg.centroid.distance_squared(gd.centroid);
                        if best.is_none_or(|(bd, _)| d2 < bd) {
                            best = Some((d2, e));
                        }
                    }
                    let (d2, e) = best?;
                    let range = crate::unit_types::missile::RANGE;
                    if d2 > range * range {
                        return None;
                    }
                    e
                }
            };
            let tg = &groups.list[t];
            Some(ShootAt {
                c: tg.centroid,
                vel: tracks.vel.get(t).copied().unwrap_or(Vec2::ZERO),
                r: tg.radius.clamp(3.0, 25.0),
            })
        })
        .collect();
    let shoot_at = &shoot_at[..];
    // Friendly blocks per team for the loft-over-friendlies check: every
    // live formed regiment as a disc with a clearance ceiling. A
    // shooter's own regiment is in here too — that is what makes rear
    // ranks loft over their own front rank.
    let blocks: [Vec<(Vec2, f32, f32)>; 2] = [0u8, 1u8].map(|tm| {
        groups
            .list
            .iter()
            .filter(|g| g.team == tm && g.count > 0 && !g.state.is_broken())
            .map(|g| {
                (
                    g.centroid,
                    g.radius.max(3.0),
                    terrain.height_at(g.centroid.x, g.centroid.y) + 2.4,
                )
            })
            .collect()
    });
    let blocks = &blocks;

    let grid = &*grid;
    let terrain = &*terrain;
    let pos_prev = &pos_prev[..];
    let speed = &speed[..];
    let team = &team[..];
    let kind = &kind[..];
    let group = &group[..];
    let home = &home[..];
    // Orders resolved to this tick's destination (attack orders chase
    // their target regiment's current centroid). FL_RECTFIGHT: an
    // ENGAGED attack order stops chasing — the frame froze where
    // contact happened (anchor snap, update_groups) and the men fight
    // individually from there; dragging the slot grid onward through
    // a moving enemy centroid is what smeared blocks into blobs. A
    // BROKEN target keeps the chase alive: pursuit is a hunt, not a
    // fight. Move orders never freeze — pulling a regiment out of
    // melee is the player's call.
    let orders: Vec<Option<Vec2>> = (0..groups.list.len())
        .map(|g| {
            let gd = &groups.list[g];
            // Archer stand-off: in range of the ordered target — stand
            // and shoot (the order survives; the chase resumes if the
            // target moves out of range).
            if standoff[g] {
                return None;
            }
            // The freeze keys on engagement WITH the ordered target
            // (count-gated, frontline.rs) — incidental duels against
            // an overflow trickle never halt the march.
            if rf
                && gd.engaged_with_target
                && let Some(crate::orders::Order::Attack(t)) = gd.order
                && !groups.list[t as usize].state.is_broken()
            {
                return None;
            }
            let mut goal = groups.goal(g);
            // Friendly formed blocks are SOLID to a marching attack
            // (engine: isBlocked/blockedBy — a unit's path routes
            // AROUND a friendly, never through his ranks). The
            // coarsest faithful version: when the straight approach
            // crosses an ally's footprint, aim at a waypoint off that
            // ally's flank; recomputed from live positions each tick,
            // the march swings smoothly around and re-aims at the
            // target beyond.
            if rf
                && !gd.state.is_broken()
                && matches!(gd.order, Some(crate::orders::Order::Attack(_)))
                && let Some(gpos) = goal
            {
                let a = gd.centroid;
                let ab = gpos - a;
                let len2 = ab.length_squared();
                if len2 > 1.0 {
                    let mut hit: Option<(f32, usize, f32)> = None; // (s, ally, radius)
                    for (u, ud) in groups.list.iter().enumerate() {
                        if u == g
                            || ud.team != gd.team
                            || ud.count == 0
                            || ud.shape != crate::formation::FormShape::Rect
                            || ud.state.is_broken()
                        {
                            continue;
                        }
                        let r = (ud.count as f32).sqrt() * crate::formation::BASE_SPACING * 0.55
                            + 1.5;
                        let s = ((ud.centroid - a).dot(ab) / len2).clamp(0.0, 1.0);
                        if s <= 0.02 || s >= 0.98 {
                            continue;
                        }
                        if (a + ab * s).distance_squared(ud.centroid) < r * r
                            && hit.is_none_or(|(hs, ..)| s < hs)
                        {
                            hit = Some((s, u, r));
                        }
                    }
                    if let Some((s, u, r)) = hit {
                        let c = groups.list[u].centroid;
                        let mut side = (a + ab * s - c).normalize_or_zero();
                        if side == Vec2::ZERO {
                            // Path through the ally's center: pick a
                            // flank deterministically per regiment.
                            let d = ab / len2.sqrt();
                            side = Vec2::new(-d.y, d.x) * if g % 2 == 0 { 1.0 } else { -1.0 };
                        }
                        goal = Some(c + side * (r + 2.0));
                    }
                }
            }
            goal
        })
        .collect();
    let orders = &orders[..];
    let anchors: Vec<Vec2> = groups.list.iter().map(|g| g.anchor).collect();
    let anchors = &anchors[..];
    let broken = &group_broken[..];
    let group_mover = &group_mover[..];
    // Regiments in combat-watch range of an enemy (sparse-fight
    // acquisition): order type is irrelevant — a Move-order fight that
    // went sparse stalls exactly the same way. HOLD regiments never
    // acquire wide: they stand their ground and take what comes.
    let press: Vec<bool> = groups
        .list
        .iter()
        .map(|g| (g.enemy_near || g.engaged) && !g.hold)
        .collect();
    let press = &press[..];
    // Hold-position leash: units of a held regiment close only the last
    // step to a swing (no chasing across open ground).
    let hold: Vec<bool> = groups.list.iter().map(|g| g.hold).collect();
    let hold = &hold[..];
    // Direction to the nearest enemy regiment (brace facing).
    let threat: Vec<Vec2> = groups.list.iter().map(|g| g.threat_dir).collect();
    let threat = &threat[..];
    // Formation facing for standing units (ZERO = no claim): a formed
    // regiment DRESSES to its ordered facing when nothing is nearer to
    // worry about — without this, units kept the yaw of their last
    // march step and a drag-ordered line stood looking sideways.
    let form_face: Vec<Vec2> = groups
        .list
        .iter()
        .map(|g| {
            if g.shape == crate::formation::FormShape::Rect && !g.state.is_broken() {
                crate::formation::facing_dir(g.facing)
            } else {
                Vec2::ZERO
            }
        })
        .collect();
    let form_face = &form_face[..];
    // Wall stance per regiment (0 none / 1 shieldwall / 2 spearwall):
    // slower advance, damage model tweaks in the events + apply pass.
    let wall: Vec<u8> = groups.list.iter().map(crate::formation::wall_kind).collect();
    let wall = &wall[..];
    // Charge phase: the run home (speed boost feeds the per-unit
    // SWING_CHARGE predicate too — momentum the sim can see).
    let charging: Vec<bool> = groups.list.iter().map(|g| g.charging).collect();
    let charging = &charging[..];
    // Fatigue locomotion: tired legs are slow legs, and exhausted
    // regiments cannot sprint the charge home (MTW1 "cannot run or
    // charge"; fleeing men tire too — pursuit catches them).
    let fat_speed: Vec<f32> = groups
        .list
        .iter()
        .map(|g| crate::fatigue::speed_mult(g.fatigue))
        .collect();
    let fat_speed = &fat_speed[..];
    let fat_nocharge: Vec<bool> = groups
        .list
        .iter()
        .map(|g| crate::fatigue::cannot_charge(g.fatigue))
        .collect();
    let fat_nocharge = &fat_nocharge[..];
    let bounds_min = terrain.min() + 4.0;
    let bounds_max = terrain.max() - 4.0;

    // Spear-line hazard global gate: marching speed exceeds the charge
    // threshold, so without this every mover in a 200k battle would pay
    // the wider scan hunting for spearwalls that do not exist. Per team:
    // does the ENEMY side field a braced spearwall at all this tick?
    let faces_spearwall: [bool; 2] = [0u8, 1u8].map(|t| {
        groups
            .list
            .iter()
            .any(|g| g.team != t && g.count > 0 && crate::formation::wall_kind(g) == 2)
    });
    // Read-only yaw snapshot (tick-start, like pos_prev) for cross-unit
    // reads inside the parallel integrate — the live yaw column is split
    // into &mut chunks there. The spear-line hazard reads the SPEARMAN's
    // facing from the charger's side of the scan; no spearwalls anywhere,
    // no copy.
    yaw_snapshot.clear();
    if faces_spearwall[0] || faces_spearwall[1] {
        yaw_snapshot.extend_from_slice(yaw);
    }
    let yaw_snap = &yaw_snapshot[..];

    let combat_scale = scale.0;
    let n_chunks = pos.len().div_ceil(CHUNK);
    if damage.0.len() < n_chunks {
        damage.0.resize_with(n_chunks, Vec::new);
    }
    if arrow_spawns.0.len() < n_chunks {
        arrow_spawns.0.resize_with(n_chunks, Vec::new);
    }
    let tick_seed = tick.wrapping_mul(0x9E37_79B1);
    let integrate_span = info_span!("integrate").entered();
    ComputeTaskPool::get().scope(|scope| {
        for (ci, chunk) in pos
            .chunks_mut(CHUNK)
            .zip(vel.chunks_mut(CHUNK))
            .zip(yaw.chunks_mut(CHUNK))
            .zip(yaw_prev.chunks_mut(CHUNK))
            .zip(target.chunks_mut(CHUNK))
            .zip(swing.chunks_mut(CHUNK))
            .zip(swing_t.chunks_mut(CHUNK))
            .zip(flash.chunks_mut(CHUNK))
            .zip(death_t.chunks_mut(CHUNK))
            .zip(&mut damage.0)
            .zip(ammo.chunks_mut(CHUNK))
            .zip(&mut arrow_spawns.0)
            .enumerate()
        {
            let (((((((((((p_chunk, v_chunk), yaw_chunk), yawp_chunk), tgt_chunk), sw_chunk),
                swt_chunk), fl_chunk), dt_chunk), events), ammo_chunk), arrow_out) = chunk;
            let start = ci * CHUNK;
            scope.spawn(async move {
                events.clear();
                for j in 0..p_chunk.len() {
                    let i = start + j;
                    let p = pos_prev[i].xz();
                    let my_kind = kind[i] as usize;
                    let params = &TYPES[my_kind];
                    let dying = dt_chunk[j] > 0;

                    // Hit flash decays here (set by the serial apply pass).
                    fl_chunk[j] = fl_chunk[j].saturating_sub(1);
                    // Corpses play out their death anim: no orders, no
                    // combat; they stay as an obstacle until swept.
                    if dying && dt_chunk[j] > 1 {
                        dt_chunk[j] -= 1;
                    }

                    // Units move ONLY under orders. A regiment order is one
                    // point rigidly translated by each unit's `home` offset
                    // (the block moves; it never converges). No order =
                    // hold at the anchor — a standing order: units drift
                    // back to their slot at reduced gain inside the block.
                    // Enemy contact is pure physics: cross-team separation
                    // blocks, crowd yield stops the shove, combat thins the
                    // block. The "front line" is where that collision is.
                    let gi = group[i] as usize;
                    let routed = broken[gi];
                    let mut desired = Vec2::ZERO;
                    if !dying && routed {
                        // Broken: flee toward the own map edge with a
                        // per-unit lateral scatter — slightly SLOWER than
                        // formed pursuers (0.9x): fleeing at exactly max
                        // speed made pursuit a zero-kill treadmill (gap
                        // frozen forever, counts flat for 30-45 s).
                        let flee_z: f32 = if team[i] == 0 { -1.0 } else { 1.0 };
                        let lat = (crate::units::hash01((i as u32).wrapping_mul(17) + 3) - 0.5)
                            * 0.7;
                        let dir = Vec2::new(lat, flee_z).normalize_or_zero();
                        let slope = terrain.slope_at(p.x, p.y);
                        desired = dir
                            * speed[i]
                            * ROUT_FLEE_FRAC
                            * terrain.wade_mult(p.x, p.y)
                            / (1.0 + 3.0 * slope * slope);
                    } else if !dying {
                        let holding = orders[gi].is_none();
                        let goal = orders[gi].unwrap_or(anchors[gi]) + home[i];
                        let to_goal = goal - p;
                        let dist = to_goal.length();
                        // Hold deadzone: parked units don't jitter around
                        // their slot point. 0.7 m (was 1.5 when homes were
                        // jittered spawn offsets): rigid slots sit exactly
                        // at the separation rest distance, so a dressed
                        // rank is force-free and can afford tight tolerance
                        // — with 1.5 the ranks never finished dressing.
                        if !(holding && dist < 0.7) {
                            // Slope penalty: steep ground is slow ground.
                            // Wading the river is slow too (the bridge
                            // deck is dry: full speed).
                            let slope = terrain.slope_at(p.x, p.y);
                            let slope_mult = 1.0 / (1.0 + 3.0 * slope * slope);
                            let (gain, arrive) = if holding {
                                (0.6, 10.0)
                            } else {
                                (1.0, ARRIVE_RADIUS)
                            };
                            let desired_speed = speed[i]
                                * slope_mult
                                * terrain.wade_mult(p.x, p.y)
                                * gain
                                * (dist / arrive).min(1.0);
                            if dist > 1e-3 {
                                desired = to_goal * (desired_speed / dist);
                            }
                        }
                    }

                    // Fused neighbor scan: separation physics + nearest
                    // living enemy in reach (swing targeting). Scalar over
                    // cache-ordered SortedUnits — an 8-wide SIMD variant
                    // measured slower here (devlog 0020): candidate runs
                    // are too short for lane occupancy.
                    let mut push = Vec2::ZERO;
                    let mut corr = Vec2::ZERO;
                    let mut crowd = 0.0f32;
                    let my_team_bit = (team[i] as u32) * crate::spatial::META_TEAM;
                    let my_wall = wall[gi] != 0;
                    let my_mass = params.mass;
                    let reach2 = params.reach * params.reach;
                    let prev_target = tgt_chunk[j];
                    let mut best_d2 = f32::MAX;
                    let mut best_idx = u32::MAX;
                    let mut sticky = false;
                    // Moving at charge speed with unspent momentum: braced
                    // enemy spears in the path are a collision hazard, and
                    // the scan must see out to SPEAR reach, not just mine.
                    // A man already run through (flash) or reeling is not
                    // re-impaled every tick — a spear is a point, not an
                    // aura.
                    let spear_reach = TYPES[crate::unit_types::KIND_SPEAR as usize].reach;
                    let cs = speed[i] * CHARGE_SPEED_FRAC;
                    let at_charge_speed = faces_spearwall[team[i] as usize]
                        && !dying
                        && fl_chunk[j] == 0
                        && sw_chunk[j] & crate::units::SWING_STAGGERED == 0
                        && v_chunk[j].xz().length_squared() > cs * cs;
                    let mut impale_idx = u32::MAX;
                    let mut impale_d = f32::MAX;
                    let scan_r = if at_charge_speed {
                        QUERY_RADIUS.max(params.reach).max(spear_reach)
                    } else {
                        QUERY_RADIUS.max(params.reach)
                    };
                    grid.for_each_candidate(p, scan_r, |o| {
                        if o.idx as usize == i {
                            return;
                        }
                        let d = p - o.xz();
                        let d2 = d.length_squared();
                        let enemy = (o.meta & crate::spatial::META_TEAM) != my_team_bit
                            && (o.meta & crate::spatial::META_DYING) == 0;
                        if enemy && d2 < reach2 {
                            sticky |= o.idx == prev_target;
                            if d2 < best_d2 {
                                best_d2 = d2;
                                best_idx = o.idx;
                            }
                        }
                        // Spear-line collision: he is a braced enemy
                        // spearman, and my body is crossing his leveled
                        // point while I close at speed.
                        if at_charge_speed
                            && enemy
                            && (o.meta & crate::spatial::META_WALL) != 0
                            && crate::spatial::meta_kind(o.meta)
                                == crate::unit_types::KIND_SPEAR as usize
                        {
                            let ys = yaw_snap[o.idx as usize];
                            let f = Vec2::new(ys.sin(), ys.cos());
                            let rel = d; // spearman -> me (d = p - o.xz())
                            let fwd_d = rel.dot(f);
                            let lat = (rel.x * f.y - rel.y * f.x).abs();
                            let closing = v_chunk[j].xz().dot(d) < 0.0;
                            if fwd_d > SPEAR_POINT_MIN
                                && fwd_d < spear_reach
                                && lat < SPEAR_LINE_HALF_W
                                && closing
                                && fwd_d < impale_d
                            {
                                impale_d = fwd_d;
                                impale_idx = o.idx;
                            }
                        }
                        // Two same-team units both in a wall STANCE rest
                        // shoulder to shoulder (symmetric predicate: both
                        // sides compute the same radius). Ordinary ranks
                        // keep parade spacing even while fighting — the
                        // seams between files are what enemy bodies flow
                        // into.
                        let cross = (o.meta & crate::spatial::META_TEAM) != my_team_bit;
                        // Pass-through pairs: a fleeing body, or a
                        // Move-ordered regiment walking through the
                        // lines (the engine's explicit
                        // formationMovingThrough state), collide at
                        // body scale — the passer slips through a
                        // formed line's seams; the formation is not
                        // excavated into a 1.4 m corridor.
                        let pass_pair = rf
                            && !cross
                            && (routed
                                || group_mover[gi]
                                || (o.meta
                                    & (crate::spatial::META_BROKEN
                                        | crate::spatial::META_MOVER))
                                    != 0);
                        let sep_r = if cross {
                            // Enemy bodies rest at body contact (see
                            // ENEMY_SEP_RADIUS): contact distance is a
                            // property of bodies, not of allegiance —
                            // the wide courtesy gap belongs to ranks
                            // dressing on the same side only.
                            if rf { ENEMY_SEP_RADIUS } else { SEP_RADIUS }
                        } else if pass_pair {
                            ENEMY_SEP_RADIUS
                        } else if my_wall && (o.meta & crate::spatial::META_WALL) != 0 {
                            WALL_SEP_RADIUS
                        } else {
                            SEP_RADIUS
                        };
                        if d2 < sep_r * sep_r && d2 > 1e-8 {
                            // Mass-weighted: heavies shove lights aside.
                            let o_mass = TYPES[crate::spatial::meta_kind(o.meta)].mass;
                            let mw = 2.0 * o_mass / (my_mass + o_mass);
                            let len = d2.sqrt();
                            let w = 1.0 - len / sep_r;
                            if len < HARD_RADIUS {
                                // Overlap is resolved POSITIONALLY only.
                                // (There used to be a hard force boost here
                                // too — two solvers fighting over the same
                                // overlap made packed crowds oscillate at
                                // the accel cap: the every-frame twitch.)
                                corr += d * ((HARD_RADIUS - len) * CORR_GAIN * mw / len);
                            }
                            push += d * (w * mw / len);
                            // Jam density: COMPRESSED pairs only — a
                            // neighbor resting AT his pair's rest
                            // distance contests nothing, so a formed
                            // man's slot-keeping is never faded by the
                            // settled enemies one stride away (the
                            // yield-and-stay-ragged defect). Gated, the
                            // weight reads on the one body scale
                            // regardless of the pair's rest radius —
                            // per-rest weights sum too small at tight
                            // radii to ever brake (the twitch). EXCEPT
                            // pass-through pairs: a body threading the
                            // seams at its intended body distance is
                            // not a wedged crowd — weighing it on the
                            // 1.4 scale made a passer crawl at a
                            // quarter speed and faded the standing
                            // line's slot-keeping (measured, ROUTPASS).
                            // Ungated all formulas are identical.
                            crowd += if rf && !pass_pair {
                                1.0 - len / SEP_RADIUS
                            } else {
                                w
                            };
                        }
                    });

                    // Ran onto a braced spear: the collision is a damage
                    // event from the SPEARMAN, resolved with everything
                    // else in the serial apply (which also stops the
                    // runner). One point, one wound — the nearest line
                    // crossed this tick.
                    if impale_idx != u32::MAX {
                        let jit = 0.85
                            + 0.3
                                * crate::units::hash01(
                                    tick_seed ^ (i as u32).wrapping_mul(0x7A31),
                                );
                        events.push(DamageEvent {
                            victim: i as u32,
                            attacker: impale_idx,
                            jit,
                            charge: false,
                            impale: true,
                        });
                    }

                    // Sparse-fight acquisition (see WIDE_ACQUIRE_R): a
                    // pressing unit with an empty scan and open space
                    // around it memoizes a farther enemy in `target` so
                    // the closing drive below can restore contact. Gated
                    // hard (press + no near enemy + low crowd + 1/8
                    // cadence) to stay off the 200k hot path.
                    // FL_RECTFIGHT: acquisition stays open until the
                    // crowd genuinely jams (CROWD_STOP) instead of the
                    // first hint of density — a rank-2 man shoulder to
                    // shoulder in the press must still want in; the
                    // (1 - jam) brake on the surge below is what stops
                    // him, continuously, when the pack is real.
                    let acquire_crowd_lim = if rf { CROWD_STOP } else { CROWD_SLOW };
                    if best_idx == u32::MAX
                        && !dying
                        && !routed
                        && press[gi]
                        && crowd < acquire_crowd_lim
                        && (i as u32).wrapping_add(tick_seed).is_multiple_of(8)
                    {
                        let mut far_d2 = WIDE_ACQUIRE_R * WIDE_ACQUIRE_R;
                        grid.for_each_candidate(p, WIDE_ACQUIRE_R, |o| {
                            let enemy = (o.meta & crate::spatial::META_TEAM) != my_team_bit
                                && (o.meta & crate::spatial::META_DYING) == 0;
                            if enemy {
                                let d2 = (p - o.xz()).length_squared();
                                if d2 < far_d2 {
                                    far_d2 = d2;
                                    tgt_chunk[j] = o.idx;
                                }
                            }
                        });
                    }

                    // Swing state machine. All writes are to this unit's own
                    // row; damage goes through the chunk event buffer.
                    let mut face_target = None;
                    if !dying {
                        match sw_chunk[j] & crate::units::SWING_STATE_MASK {
                            crate::units::SWING_WINDUP
                                if sw_chunk[j] & crate::units::SWING_RANGED != 0 =>
                            {
                                // Drawing the bow: feet planted, eyes on
                                // the target block.
                                desired = Vec2::ZERO;
                                if let Some(s) = &shoot_at[gi] {
                                    face_target = Some(s.c);
                                    if swt_chunk[j] == 0 {
                                        let h = |k: u32| {
                                            crate::units::hash01(
                                                tick_seed
                                                    ^ (i as u32).wrapping_mul(k).wrapping_add(k),
                                            )
                                        };
                                        // Aim: a spot on the target's
                                        // footprint, the M2TW range-
                                        // INDEPENDENT landing scatter
                                        // (devlog 0060), and a lead for
                                        // the block's drift over the
                                        // flight.
                                        let ang = h(0x1F3B) * std::f32::consts::TAU;
                                        let rad = s.r * 0.85 * h(0x2E5D).sqrt();
                                        let sigma =
                                            crate::unit_types::missile::SCATTER_SIGMA * 1.75;
                                        let mut aim = s.c
                                            + Vec2::new(ang.cos(), ang.sin()) * rad
                                            + Vec2::new(
                                                (h(0x3B7F) + h(0x45A3) - 1.0) * sigma,
                                                (h(0x5DC1) + h(0x6B8D) - 1.0) * sigma,
                                            );
                                        aim += s.vel * (aim.distance(p) / 30.0);
                                        let from =
                                            Vec3::new(p.x, pos_prev[i].y + 0.55, p.y);
                                        let to = Vec3::new(
                                            aim.x,
                                            terrain.height_at(aim.x, aim.y) + 0.7,
                                            aim.y,
                                        );
                                        arrow_out.push(crate::arrows::ArrowSpawn {
                                            pos: from,
                                            vel: crate::arrows::solve_launch(
                                                from,
                                                to,
                                                &blocks[team[i] as usize],
                                            ),
                                            team: team[i],
                                            group: group[i],
                                        });
                                        ammo_chunk[j] = ammo_chunk[j].saturating_sub(1);
                                        sw_chunk[j] = crate::units::SWING_RECOVER;
                                        // Reload: the M2TW volley cycle is
                                        // animation-bound at ~10 s; the
                                        // jitter keeps later volleys ragged.
                                        swt_chunk[j] =
                                            (crate::unit_types::missile::RELOAD_TICKS as f32
                                                * (0.8 + 0.25 * h(0x77F1)))
                                                .min(255.0)
                                                as u8;
                                    } else {
                                        swt_chunk[j] -= 1;
                                    }
                                } else {
                                    // Target gone mid-draw: ease off and
                                    // reassess shortly.
                                    sw_chunk[j] = crate::units::SWING_RECOVER;
                                    swt_chunk[j] = 20;
                                }
                            }
                            crate::units::SWING_WINDUP => {
                                // Feet planted while winding up — EXCEPT
                                // against a routing target: the cut-down
                                // happens at a run, or the runner is 3 m
                                // gone by the strike tick and every blow
                                // whiffs (the pursuit treadmill).
                                let t = tgt_chunk[j] as usize;
                                let target_routed =
                                    t < pos_prev.len() && broken[group[t] as usize];
                                if !target_routed {
                                    desired *= 0.25;
                                }
                                if t < pos_prev.len() {
                                    face_target = Some(pos_prev[t].xz());
                                }
                                if swt_chunk[j] == 0 {
                                    // Strike lands; validity (still alive,
                                    // still in reach, still an enemy) is
                                    // checked in the apply pass — a dodged
                                    // or dead target is a whiff.
                                    let jit =
                                        0.85 + 0.3 * crate::units::hash01(tick_seed ^ (i as u32));
                                    events.push(DamageEvent {
                                        victim: tgt_chunk[j],
                                        attacker: i as u32,
                                        jit,
                                        charge: sw_chunk[j] & crate::units::SWING_CHARGE != 0,
                                        impale: false,
                                    });
                                    sw_chunk[j] = crate::units::SWING_RECOVER;
                                    let cjit = 0.75
                                        + 0.5
                                            * crate::units::hash01(
                                                tick_seed ^ (i as u32).wrapping_mul(0x9E37),
                                            );
                                    swt_chunk[j] = (params.cooldown_ticks as f32 * cjit) as u8;
                                } else {
                                    swt_chunk[j] -= 1;
                                }
                            }
                            crate::units::SWING_RECOVER => {
                                if swt_chunk[j] == 0 {
                                    // A stagger that just wore off leaves
                                    // one free pass against the next one
                                    // (anti-stunlock); a plain recovery
                                    // carries an unspent pass forward.
                                    let immune = if sw_chunk[j]
                                        & crate::units::SWING_STAGGERED
                                        != 0
                                    {
                                        crate::units::SWING_STAGGER_IMMUNE
                                    } else {
                                        sw_chunk[j] & crate::units::SWING_STAGGER_IMMUNE
                                    };
                                    sw_chunk[j] = crate::units::SWING_READY | immune;
                                } else {
                                    swt_chunk[j] -= 1;
                                }
                            }
                            _ => {
                                // Ready: pick a target from the scan. Stick
                                // with the previous one when still in reach
                                // (duels), else nearest. Routing units never
                                // start attacks (they still defend nothing —
                                // pursuit is free hits).
                                let chosen = if sticky { prev_target } else { best_idx };
                                // No eyes in the back of his head: a man
                                // only opens on a target in his forward
                                // half-plane. Being struck tells him where
                                // to turn (the flash facing below), but it
                                // does NOT let him swing backward over his
                                // shoulder — he attacks once he has turned
                                // far enough, at the human turn-speed cap.
                                // Without this gate every rear-approached
                                // victim counter-wound-up on proximity and
                                // was face-on before the first blow landed;
                                // the rear sector never fired in practice.
                                let aware = chosen != u32::MAX && {
                                    let t = chosen as usize;
                                    t < pos_prev.len() && {
                                        let to_t = pos_prev[t].xz() - p;
                                        let fwd = Vec2::new(
                                            yaw_chunk[j].sin(),
                                            yaw_chunk[j].cos(),
                                        );
                                        fwd.dot(to_t) >= 0.0
                                    }
                                };
                                if chosen != u32::MAX && !routed && aware {
                                    tgt_chunk[j] = chosen;
                                    // Arriving at speed = a charging blow:
                                    // momentum converts to damage + a bigger
                                    // lunge (render reads the flag).
                                    let v2 = v_chunk[j].xz().length_squared();
                                    let cs = speed[i] * CHARGE_SPEED_FRAC;
                                    // Attack style for this swing (render
                                    // variety only): 0 = stab, 1 = the
                                    // classic swing. (2 = slash exists in
                                    // the shader but benched.)
                                    // Spears only ever thrust.
                                    let style = if my_kind
                                        == crate::unit_types::KIND_SPEAR as usize
                                    {
                                        0
                                    } else {
                                        (crate::units::hash01(
                                            tick_seed ^ (i as u32).wrapping_mul(0x51ED),
                                        ) * 2.0) as u8
                                    };
                                    let style = style << crate::units::SWING_STYLE_SHIFT;
                                    sw_chunk[j] = if v2 > cs * cs {
                                        crate::units::SWING_WINDUP
                                            | crate::units::SWING_CHARGE
                                            | style
                                    } else {
                                        crate::units::SWING_WINDUP | style
                                    };
                                    swt_chunk[j] = params.windup_ticks;
                                } else if my_kind == crate::unit_types::KIND_ARCHER as usize
                                    && !routed
                                    && ammo_chunk[j] > 0
                                    && v_chunk[j].xz().length_squared() < 4.0
                                    && let Some(s) = &shoot_at[gi]
                                    && p.distance_squared(s.c)
                                        < crate::unit_types::missile::RANGE
                                            * crate::unit_types::missile::RANGE
                                {
                                    // Nock and draw (foot archers shoot
                                    // standing only; the walk gate keeps a
                                    // marching or skirmishing man's bow on
                                    // his back). Style bits stay 0: the
                                    // stab pull-back IS the string draw.
                                    sw_chunk[j] = crate::units::SWING_WINDUP
                                        | crate::units::SWING_RANGED;
                                    swt_chunk[j] = crate::unit_types::missile::DRAW_TICKS
                                        + (crate::units::hash01(
                                            tick_seed ^ (i as u32).wrapping_mul(0x2C9F),
                                        ) * 20.0) as u8;
                                }
                            }
                        }
                        // A reloading archer contacted in melee drops the
                        // reload: he defends at knife tempo instead of
                        // standing through the 8 s bow cycle.
                        if my_kind == crate::unit_types::KIND_ARCHER as usize
                            && best_idx != u32::MAX
                            && sw_chunk[j] & crate::units::SWING_STATE_MASK
                                == crate::units::SWING_RECOVER
                            && swt_chunk[j] > params.cooldown_ticks
                        {
                            swt_chunk[j] = params.cooldown_ticks;
                        }
                    }
                    let mut corr_len2 = corr.length_squared();
                    if corr_len2 < 1e-4 {
                        // Sub-centimeter corrections are settle noise, not
                        // overlap: applying them is pure micro-twitch.
                        corr = Vec2::ZERO;
                        corr_len2 = 0.0;
                    } else if corr_len2 > CORR_MAX * CORR_MAX {
                        corr *= CORR_MAX / corr_len2.sqrt();
                    }
                    let push_len = push.length();
                    if push_len > SEP_PUSH_MAX {
                        push *= SEP_PUSH_MAX / push_len;
                    }
                    // Yield in dense crowds: goal drive fades out entirely
                    // so the mass can't keep compressing itself; `jam` (0 =
                    // free, 1 = packed) also damps the response below.
                    let jam = ((crowd - CROWD_SLOW) / (CROWD_STOP - CROWD_SLOW)).clamp(0.0, 1.0);
                    desired *= 1.0 - jam;
                    // Fighters close the last meter to swing range. Only
                    // active when an enemy is ALREADY in reach — this is
                    // combat execution (like the wind-up foot plant), not
                    // steering; it bypasses the jam yield on purpose so
                    // front lines stay joined instead of settling at the
                    // separation standoff just outside sword range.
                    // Close toward the LOCKED swing target when there is
                    // one — the per-tick nearest enemy flips in a clog and
                    // flip-flopping the close direction reads as twitch.
                    // Falls back to the far-acquisition memo in `target`
                    // when the near scan is empty AND the unit is in open
                    // space — in a dense press the memo would let second
                    // ranks drive through the jam and compress the crowd
                    // (nn regression). The memo may be stale after death
                    // sweeps reindex, so it is validated as "some enemy
                    // within closing range" — a legitimate closing target
                    // regardless of identity.
                    let (close_to, memo_close) = if sw_chunk[j] & crate::units::SWING_STATE_MASK
                        != crate::units::SWING_READY
                        && (tgt_chunk[j] as usize) < pos_prev.len()
                    {
                        (tgt_chunk[j], false)
                    } else if best_idx != u32::MAX {
                        (best_idx, false)
                    } else if crowd < acquire_crowd_lim {
                        (tgt_chunk[j], true)
                    } else {
                        (u32::MAX, false)
                    };
                    if !dying
                        && !routed
                        && (close_to as usize) < pos_prev.len()
                        && team[close_to as usize] != team[i]
                    {
                        let to_enemy = pos_prev[close_to as usize].xz() - p;
                        let dist = to_enemy.length();
                        // Held regiments fight at arm's length only.
                        let max_close = if hold[gi] { 2.2 } else { WIDE_ACQUIRE_R + 1.0 };
                        if dist > 1.2 && dist < max_close {
                            let mut urge = ((dist - 1.2) / 0.8).clamp(0.0, 1.0);
                            // The surge toward a REMEMBERED enemy (no
                            // one in reach yet) is steering, not combat
                            // execution: it yields to the jam like all
                            // steering, so the press brakes on genuine
                            // body-pack. Ungated this factor is always
                            // 1 (the memo gate above already required
                            // crowd < CROWD_SLOW, i.e. jam == 0).
                            if memo_close {
                                urge *= 1.0 - jam;
                            }
                            desired += to_enemy * (speed[i] * 0.35 * urge / dist);
                        }
                    }
                    // Formation pace: walls advance deliberately (running
                    // breaks a wall), the charge phase runs the last
                    // stretch home. Broken/dying already excluded from
                    // both states by construction.
                    if wall[gi] != 0 {
                        desired *= WALL_SPEED_FRAC;
                    } else if charging[gi] && !dying && !routed && !fat_nocharge[gi] {
                        desired *= CHARGE_SPEED_BOOST;
                    }
                    desired *= fat_speed[gi];
                    // A staggered man reels where the blow left him: no
                    // steering, no closing, until the stun runs out. The
                    // shove that staggered him still resolves through
                    // separation — he is a body, not an actor.
                    if sw_chunk[j] & crate::units::SWING_STAGGERED != 0 {
                        desired = Vec2::ZERO;
                    }

                    let v = v_chunk[j].xz();
                    // Jammed units stop shoving entirely: at full jam the
                    // crowd is quasi-static and overlap resolution is
                    // purely positional — force-based separation in a
                    // wedged mass only produces bang-bang oscillation.
                    let mut accel =
                        (desired - v) * STEER_GAIN + push * (SEP_STRENGTH * (1.0 - jam));
                    let a2 = accel.length_squared();
                    if a2 > MAX_ACCEL * MAX_ACCEL {
                        accel *= MAX_ACCEL / a2.sqrt();
                    }

                    let mut new_v = v + accel * dt;
                    // Viscous damping in the press: bleeds the spring energy
                    // that otherwise ping-pongs between neighbors every tick.
                    new_v *= 1.0 - 0.4 * jam;
                    let vmax = speed[i] * 1.15; // slight overspeed under crowd pressure
                    let v2 = new_v.length_squared();
                    if v2 > vmax * vmax {
                        new_v *= vmax / v2.sqrt();
                    }
                    // After a positional correction, kill the velocity
                    // component still driving into the overlap or it
                    // re-penetrates next tick.
                    if corr_len2 > 1e-12 {
                        let cn = corr.normalize_or_zero();
                        let into = new_v.dot(-cn);
                        if into > 0.0 {
                            new_v += cn * into;
                        }
                    }

                    // Facing priority: locked wind-up target > nearest
                    // enemy in reach > movement direction. Fighters keep
                    // eyes on the enemy even while the crowd shoves them;
                    // only routing/unengaged units face their velocity.
                    // yaw_prev snapshots the pre-update angle so the
                    // renderer can interpolate (yaw stepped once per tick
                    // otherwise — visible facing snaps at high fps).
                    yawp_chunk[j] = yaw_chunk[j];
                    let face_dir = match face_target {
                        Some(t) => t - p,
                        // A man IN his swing cycle faces the fight (a
                        // formed one too — he is the fighting rim), and a
                        // man JUST STRUCK turns toward the blow (flash).
                        // A man merely NEAR an enemy does not: turning on
                        // proximity raced the attacker's wind-up and had
                        // every rear-approached victim frontal by first
                        // blood — the whole point of facing, gone. So the
                        // first hit lands in the back, spins its victim,
                        // and THEN he answers. Unformed units (Blob, no
                        // facing claim) still turn on proximity.
                        None if !routed
                            && best_idx != u32::MAX
                            && ((sw_chunk[j] & crate::units::SWING_STATE_MASK)
                                != crate::units::SWING_READY
                                || fl_chunk[j] > 0
                                || form_face[gi] == Vec2::ZERO) =>
                        {
                            pos_prev[best_idx as usize].xz() - p
                        }
                        // Blooded and the enemy still close: a man who has
                        // traded blows keeps facing the fight while his
                        // last foe (combat memo, validated by team and
                        // distance like the closing drive) stands within
                        // KEEP_FACING_R — no parade dressing with a sword
                        // a few strides away. A player reform takes hold
                        // once the ground near him clears. Fresh men fall
                        // through and hold the ordered line.
                        None if !routed
                            && new_v.length_squared() < 0.25
                            && form_face[gi] != Vec2::ZERO
                            && (tgt_chunk[j] as usize) < pos_prev.len()
                            && team[tgt_chunk[j] as usize] != team[i]
                            && pos_prev[tgt_chunk[j] as usize]
                                .xz()
                                .distance_squared(p)
                                < KEEP_FACING_R * KEEP_FACING_R =>
                        {
                            pos_prev[tgt_chunk[j] as usize].xz() - p
                        }
                        // Standing in formation: HOLD the ordered facing.
                        // M2TW rule — a formed unit never rotates itself
                        // toward a threat (it goes "ready" in place; the
                        // render brace pose keys off enemy_near, not yaw);
                        // facing is the player's job, and leaving a flank
                        // open is supposed to cost.
                        None if !routed
                            && new_v.length_squared() < 0.25
                            && form_face[gi] != Vec2::ZERO =>
                        {
                            form_face[gi]
                        }
                        // Standing watch WITHOUT a formation claim (Blob
                        // mobs, rallied remnants): face the enemy mass
                        // instead of keeping a stale yaw.
                        None if !routed
                            && new_v.length_squared() < 0.25
                            && threat[gi] != Vec2::ZERO =>
                        {
                            threat[gi]
                        }
                        None => new_v,
                    };
                    let min_len2 = if face_target.is_some() || best_idx != u32::MAX {
                        1e-4 // enemies can be close; still face them
                    } else {
                        0.25 // velocity facing ignores micro-drift
                    };
                    // A staggered man cannot even turn — the stun freezes
                    // his facing, so a charge's second blow finds the same
                    // back the first one hit.
                    if !dying
                        && sw_chunk[j] & crate::units::SWING_STAGGERED == 0
                        && face_dir.length_squared() > min_len2
                    {
                        let target_yaw = face_dir.x.atan2(face_dir.y);
                        let diff = (target_yaw - yaw_chunk[j] + std::f32::consts::PI)
                            .rem_euclid(std::f32::consts::TAU)
                            - std::f32::consts::PI;
                        let step = diff * (YAW_RATE * dt).min(1.0);
                        yaw_chunk[j] += step.clamp(-TURN_SPEED_MAX * dt, TURN_SPEED_MAX * dt);
                    }

                    v_chunk[j] = Vec3::new(new_v.x, 0.0, new_v.y);
                    let mut nx = (pos_prev[i].x + new_v.x * dt + corr.x)
                        .clamp(bounds_min.x, bounds_max.x);
                    let mut nz = (pos_prev[i].z + new_v.y * dt + corr.y)
                        .clamp(bounds_min.y, bounds_max.y);
                    // Impassable ground (terrace risers, gorge walls,
                    // crater lips): wall-slide — keep the axis that
                    // stays on walkable ground, drop the one that
                    // doesn't, so crowds flow along the obstacle.
                    if terrain.blocked_at(nx, nz) {
                        let (px, pz) = (pos_prev[i].x, pos_prev[i].z);
                        if !terrain.blocked_at(nx, pz) {
                            nz = pz;
                        } else if !terrain.blocked_at(px, nz) {
                            nx = px;
                        } else {
                            nx = px;
                            nz = pz;
                        }
                    }
                    p_chunk[j] = Vec3::new(
                        nx,
                        terrain.height_at(nx, nz) + TYPES[kind[i] as usize].half_height,
                        nz,
                    );
                }
            });
        }
    });
    drop(integrate_span);
    stats.step_ms = t1.elapsed().as_secs_f32() * 1000.0;

    // Serial damage apply: deterministic (chunk order), race-free, and the
    // single place where hp transitions to death. A swing whiffs when its
    // victim died mid-wind-up, changed team slot via swap-remove, or slipped
    // out of reach — checked here, where all columns are whole again.
    {
        let _span = info_span!("damage_apply").entered();
        stats.events = 0;
        // Fatigue stat effects (MTW1 table): weary arms strike with
        // fewer attack points; exhausted men get no charge bonus at all.
        let fat_atk: Vec<f32> = groups
            .list
            .iter()
            .map(|g| crate::fatigue::attack_penalty(g.fatigue))
            .collect();
        for buf in &mut damage.0 {
            stats.events += buf.len();
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
                        pos[v] = Vec3::new(nx, terrain.height_at(nx, nz) + pv.half_height, nz);
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

    // Overlap + movement audit over the full population, every 60 ticks
    // (~2 s). Parallel over chunks; each task returns
    // (min_d2, sum_d, counted, sum_displacement).
    *tick = tick.wrapping_add(1);
    if (*tick).is_multiple_of(60) {
        let _span = info_span!("nn_audit").entered();
        let audit_t0 = Instant::now();
        let pos_now = &pos[..];
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
                        grid.for_each_candidate(p2, SEP_RADIUS, |o| {
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

    // Spike attribution: name any tick that blows past the norm, with
    // component costs inline — lag hunting works on facts, and the
    // audit above runs every 60 ticks inside this same system, so an
    // audit-tick spike shows its fresh cost here.
    let audit_now = (*tick).is_multiple_of(60);
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
            pos.len(),
        );
    }
}

fn toggle_debug_viz(keys: Res<ButtonInput<KeyCode>>, mut viz: ResMut<DebugViz>) {
    if keys.just_pressed(KeyCode::KeyG) {
        viz.0 = !viz.0;
    }
}

