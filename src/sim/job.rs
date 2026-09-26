//! One kinematic tick's job: the shared columns it reads, the
//! per-regiment snapshot, and the buffers it fills. `prepare_tick`
//! builds it on the main thread; the kernel runs it on the worker.

use bevy::prelude::*;
use std::sync::Arc;

use crate::orders::Groups;
use crate::sim::damage::DamageEvent;
use crate::spatial::SpatialGrid;
use crate::terrain::Terrain;
use crate::units::Units;

/// Soldiers per parallel kernel task.
pub(crate) const CHUNK: usize = 2048;

/// What an archer regiment's bows shoot at this tick (regiment-level
/// fire solution, resolved in `prepare_tick`).
pub(crate) struct ShootAt {
    pub(crate) c: Vec2,
    pub(crate) vel: Vec2,
    pub(crate) r: f32,
    /// Target regiment (index into `Groups::list`): the per-shot
    /// aim picks one of its living soldiers.
    pub(crate) t: usize,
}

/// Everything one kinematic tick owns: the shared columns it reads, the
/// per-regiment command snapshot, and the buffers it fills. Fully
/// self-contained, so the tick can run on a worker thread while the main
/// world keeps rendering the last completed tick.
///
/// The soldier columns are not copied. The job holds `Arc` clones of
/// the live columns (`Units::Column`), taken at the kick. The main world
/// writes its columns only between the install and the kick, when the
/// job holds nothing, so the clones are never stale and the writes never
/// copy. The read-modify-write columns are read from the clones and
/// written to the job's own output buffers, which the install swaps in;
/// the buffers the install releases become the next tick's outputs.
#[derive(Default)]
pub struct TickJob {
    // Shared inputs: the live columns at the kick. `pos_in` is the
    // kernel's `pos_prev`.
    pub(crate) pos_in: Arc<Vec<Vec3>>,
    pub(crate) speed: Arc<Vec<f32>>,
    pub(crate) team: Arc<Vec<u8>>,
    pub(crate) kind: Arc<Vec<u8>>,
    pub(crate) group: Arc<Vec<u32>>,
    pub(crate) home: Arc<Vec<Vec2>>,
    pub(crate) vel_in: Arc<Vec<Vec3>>,
    pub(crate) yaw_in: Arc<Vec<f32>>,
    pub(crate) yaw_prev_in: Arc<Vec<f32>>,
    pub(crate) target_in: Arc<Vec<u32>>,
    pub(crate) swing_in: Arc<Vec<u8>>,
    pub(crate) swing_t_in: Arc<Vec<u8>>,
    pub(crate) flash_in: Arc<Vec<u8>>,
    pub(crate) death_t_in: Arc<Vec<u8>>,
    pub(crate) ammo_in: Arc<Vec<u8>>,
    pub(crate) out_form_in: Arc<Vec<bool>>,
    pub(crate) sight_in: Arc<Vec<u8>>,
    // Outputs: the columns after the tick. Each kernel task copies its
    // chunk in from the shared column, then runs the stages on it.
    pub(crate) pos_out: Vec<Vec3>,
    pub(crate) vel: Vec<Vec3>,
    pub(crate) yaw: Vec<f32>,
    pub(crate) yaw_prev: Vec<f32>,
    pub(crate) target: Vec<u32>,
    pub(crate) swing: Vec<u8>,
    pub(crate) swing_t: Vec<u8>,
    pub(crate) flash: Vec<u8>,
    pub(crate) death_t: Vec<u8>,
    pub(crate) ammo: Vec<u8>,
    pub(crate) out_form: Vec<bool>,
    pub(crate) sight: Vec<u8>,
    // Per-regiment command snapshot, taken at prep.
    pub(crate) orders: Vec<Option<Vec2>>,
    pub(crate) anchors: Vec<Vec2>,
    pub(crate) reg_broken: Vec<bool>,
    pub(crate) press: Vec<bool>,
    pub(crate) engaged: Vec<bool>,
    pub(crate) contact: Vec<bool>,
    pub(crate) fight_point: Vec<Option<Vec2>>,
    pub(crate) melee_ticks: Vec<u32>,
    pub(crate) hold: Vec<bool>,
    pub(crate) threat: Vec<Vec2>,
    pub(crate) form_face: Vec<Vec2>,
    pub(crate) wall: Vec<u8>,
    /// Per regiment: in a wall stance (the grid's META_WALL bit).
    pub(crate) group_wall: Vec<bool>,
    pub(crate) charging: Vec<bool>,
    pub(crate) fat_speed: Vec<f32>,
    pub(crate) fat_nocharge: Vec<bool>,
    pub(crate) shoot_at: Vec<Option<ShootAt>>,
    /// Per regiment: under fire this tick (some regiment's bows aim at it).
    pub(crate) targeted: Vec<bool>,
    pub(crate) blocks: [Vec<(Vec2, f32, f32)>; 2],
    pub(crate) faces_spearwall: [bool; 2],
    // Products of the run beyond the columns, all swapped into their
    // resources when the job is taken or installed.
    /// Each regiment's men as index ranges, in index order (see
    /// `regiment_runs`).
    pub(crate) reg_runs: Vec<Vec<(u32, u32)>>,
    /// Living members of every regiment under fire, per regiment, in
    /// index order.
    pub(crate) target_members: Vec<Vec<u32>>,
    /// The density field of the tick's start positions (frontline.rs),
    /// built when a taker wants it (not on the inline path, where
    /// `update_field` already rebuilt the resource this tick).
    pub(crate) field: crate::frontline::InfluenceField,
    pub(crate) field_wanted: bool,
    /// Men whose death countdown reached its last tick, ascending.
    pub(crate) dead: Vec<u32>,
    /// Living men standing near or beyond their own map edge, ascending:
    /// the sweep's candidates for a router who has left the field.
    pub(crate) at_edge: Vec<u32>,
    pub(crate) grid: SpatialGrid,
    pub(crate) events: Vec<Vec<DamageEvent>>,
    pub(crate) arrow_spawns: Vec<Vec<crate::arrows::ArrowSpawn>>,
    pub(crate) terrain: Option<std::sync::Arc<Terrain>>,
    pub(crate) dt: f32,
    pub(crate) combat_scale: f32,
    pub(crate) tick_seed: u32,
    pub(crate) tick: u32,
    pub(crate) bounds_min: Vec2,
    pub(crate) bounds_max: Vec2,
    /// `Units::generation` at prep: a job from a dead world is dropped.
    pub(crate) generation: u64,
    pub(crate) grid_ms: f32,
    pub(crate) step_ms: f32,
    pub(crate) field_ms: f32,
}

/// The index ranges of every regiment's men, in index order. Regiments
/// are contiguous at spawn; only the death sweep's swap-removes scatter
/// men, so a regiment is a few runs. Walking a regiment's runs visits
/// its men in ascending index, the order every serial loop over the
/// army used, so sums taken this way are the same bits.
pub(crate) fn regiment_runs(group: &[u32], n_groups: usize, out: &mut Vec<Vec<(u32, u32)>>) {
    out.resize_with(n_groups, Vec::new);
    for runs in out.iter_mut() {
        runs.clear();
    }
    let mut i = 0usize;
    let n = group.len();
    while i < n {
        let g = group[i];
        let start = i;
        i += 1;
        while i < n && group[i] == g {
            i += 1;
        }
        if (g as usize) < n_groups {
            out[g as usize].push((start as u32, i as u32));
        }
    }
}

/// The archers' fire solutions of a tick.
struct FireSolutions {
    /// Archer regiments standing off their ordered target: in range,
    /// halted, volleying.
    standoff: Vec<bool>,
    shoot_at: Vec<Option<ShootAt>>,
    /// Per regiment: under fire this tick.
    targeted: Vec<bool>,
    /// Friendly formed blocks per team as discs with a clearance ceiling,
    /// for the loft-over-friendlies check.
    blocks: [Vec<(Vec2, f32, f32)>; 2],
}

/// Fill `job` from the live world: the shared columns, the per-regiment
/// command snapshot, the archers' fire solutions and scalars. The
/// serial prep of a tick, on the main thread, per regiment only: nothing
/// here loops over the army. It also makes the two regiment writes prep
/// makes: the stand-off anchor snap and the `firing` flag.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_tick(
    job: &mut TickJob,
    units: &Units,
    groups: &mut Groups,
    tracks: &crate::arrows::RegTracks,
    terrain_arc: &std::sync::Arc<Terrain>,
    dt: f32,
    combat_scale: f32,
    tick: u32,
) {
    job.generation = units.generation;
    job.dt = dt;
    job.combat_scale = combat_scale;
    job.terrain = Some(terrain_arc.clone());
    let terrain: &Terrain = terrain_arc;

    share_columns(job, units);
    let fire = archer_fire_solutions(groups, tracks, terrain);
    job.orders = resolve_orders(groups, &fire.standoff);
    snapshot_regiments(job, groups, terrain);

    let n_chunks = units.pos.len().div_ceil(CHUNK);
    if job.events.len() < n_chunks {
        job.events.resize_with(n_chunks, Vec::new);
    }
    if job.arrow_spawns.len() < n_chunks {
        job.arrow_spawns.resize_with(n_chunks, Vec::new);
    }
    // Spawn buffers come back drained from arrows.rs. A job dropped as
    // stale never got that far.
    for buf in &mut job.arrow_spawns {
        buf.clear();
    }
    job.tick_seed = tick.wrapping_mul(0x9E37_79B1);
    job.tick = tick;
    job.shoot_at = fire.shoot_at;
    job.targeted = fire.targeted;
    job.blocks = fire.blocks;
    job.field.size_to(terrain.min(), terrain.max());
}

/// The soldier columns, shared into the job: `Arc` clones of the live
/// columns, no copy. The output buffers are sized to the army; their
/// contents are overwritten by the kernel (each task copies its chunk in
/// from the shared column first, and the position output is written for
/// every man), so they are never cleared.
fn share_columns(job: &mut TickJob, units: &Units) {
    job.pos_in = units.pos.share();
    job.speed = units.speed.share();
    job.team = units.team.share();
    job.kind = units.kind.share();
    job.group = units.group.share();
    job.home = units.home.share();
    job.vel_in = units.vel.share();
    job.yaw_in = units.yaw.share();
    job.yaw_prev_in = units.yaw_prev.share();
    job.target_in = units.target.share();
    job.swing_in = units.swing.share();
    job.swing_t_in = units.swing_t.share();
    job.flash_in = units.flash.share();
    job.death_t_in = units.death_t.share();
    job.ammo_in = units.ammo.share();
    job.out_form_in = units.out_form.share();
    job.sight_in = units.sight.share();
    let n = units.pos.len();
    fn fit<T: Clone + Default>(v: &mut Vec<T>, n: usize) {
        if v.len() != n {
            v.resize(n, T::default());
        }
    }
    fit(&mut job.pos_out, n);
    fit(&mut job.vel, n);
    fit(&mut job.yaw, n);
    fit(&mut job.yaw_prev, n);
    fit(&mut job.target, n);
    fit(&mut job.swing, n);
    fit(&mut job.swing_t, n);
    fit(&mut job.flash, n);
    fit(&mut job.death_t, n);
    fit(&mut job.ammo, n);
    fit(&mut job.out_form, n);
    fit(&mut job.sight, n);
}

/// An attack order for a ranged regiment is a fire order, not a melee
/// charge: the regiment halts once the target is inside range and
/// volleys from where it stands; the march resumes if the target slips
/// back out of range. Makes the two regiment writes prep always made:
/// the stand-off anchor snap and the `firing` flag.
fn archer_fire_solutions(
    groups: &mut Groups,
    tracks: &crate::arrows::RegTracks,
    terrain: &Terrain,
) -> FireSolutions {
    // An attack order for a
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
                    // Fire-at-will: nearest live, unbroken enemy block
                    // that is NOT locked in melee. In M2TW fire-at-will
                    // holds instead of volleying into a fight with
                    // friends standing in it; an explicit target order
                    // (the Attack arm above) still forces the shot.
                    // The flight stays honest either way: no team check
                    // on what the shaft actually crosses.
                    if !gd.fire_at_will {
                        return None;
                    }
                    let mut best: Option<(f32, usize)> = None;
                    for (e, eg) in groups.list.iter().enumerate() {
                        if eg.team == gd.team
                            || eg.count == 0
                            || eg.state.is_broken()
                            || eg.engaged
                        {
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
                t,
            })
        })
        .collect();
    // The card's firing indicator: a fire solution exists and there are
    // arrows to spend on it.
    for (g, s) in shoot_at.iter().enumerate() {
        groups.list[g].firing = s.is_some() && groups.list[g].ammo_left > 0;
    }
    // Regiments under fire this tick. The job lists their living members
    // (`target_members`, built on the worker from its shared columns):
    // each shot aims at an actual soldier (M2TW's per-soldier aim
    // targets), not at a spot on the block's footprint. Into a locked
    // melee the shafts head for enemy bodies; friends die only to
    // genuine misses and interceptions.
    let mut targeted = vec![false; n_groups];
    for s in shoot_at.iter().flatten() {
        targeted[s.t] = true;
    }
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

    FireSolutions {
        standoff,
        shoot_at,
        targeted,
        blocks,
    }
}

/// Each regiment's order resolved to this tick's destination, or none
/// for a regiment that holds where it is.
fn resolve_orders(groups: &Groups, standoff: &[bool]) -> Vec<Option<Vec2>> {
    // Orders resolved to this tick's destination (attack orders chase
    // their target regiment's current centroid). A regiment holding a
    // contact frame stops chasing: dragging the slot grid onward
    // through a moving enemy centroid is what smeared blocks into
    // blobs. Move orders never freeze — pulling a regiment out of
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
            // Contact frame (frontline.rs): the regiment holds its slots
            // around the frame instead of chasing the target's center.
            if gd.contact {
                return None;
            }
            groups.goal(g)
        })
        .collect();
    orders
}

/// The per-regiment state the kernel reads, snapshotted into the job,
/// plus the per-soldier flags derived from it.
fn snapshot_regiments(job: &mut TickJob, groups: &Groups, terrain: &Terrain) {
    // Per-regiment wall flag for the grid meta (same-team wall pairs pack
    // tighter in the separation); the rebuild reads it through the
    // regiment index it already carries per man.
    let group_wall: Vec<bool> = groups
        .list
        .iter()
        .map(|g| crate::formation::wall_kind(g) != 0)
        .collect();
    let group_broken: Vec<bool> = groups.list.iter().map(|g| g.state.is_broken()).collect();
    let anchors: Vec<Vec2> = groups.list.iter().map(|g| g.anchor).collect();
    // Regiments in combat-watch range of an enemy (sparse-fight
    // acquisition): order type is irrelevant — a Move-order fight that
    // went sparse stalls exactly the same way. HOLD regiments never
    // acquire wide: they stand their ground and take what comes.
    let press: Vec<bool> = groups
        .list
        .iter()
        .map(|g| (g.enemy_near || g.engaged) && !g.hold)
        .collect();
    // Regiments in melee: their shoved men wait for room instead of
    // threading back to their marks (a dressing block still threads).
    let engaged: Vec<bool> = groups.list.iter().map(|g| g.engaged).collect();
    // Regiments holding a contact frame (frontline.rs): their fight is
    // ahead of them, and their men never step back to dress.
    let contact: Vec<bool> = groups.list.iter().map(|g| g.contact).collect();
    // Where each regiment in melee has its fight, and for how long.
    let fight_point: Vec<Option<Vec2>> = groups.list.iter().map(|g| g.fight_point).collect();
    let melee_ticks: Vec<u32> = groups.list.iter().map(|g| g.melee_ticks).collect();

    // Hold-position leash: units of a held regiment close only the last
    // step to a swing (no chasing across open ground).
    let hold: Vec<bool> = groups.list.iter().map(|g| g.hold).collect();
    // Direction to the nearest enemy regiment (brace facing).
    let threat: Vec<Vec2> = groups.list.iter().map(|g| g.threat_dir).collect();
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
    // Wall stance per regiment (0 none / 1 shieldwall / 2 spearwall):
    // slower advance, damage model tweaks in the events + apply pass.
    let wall: Vec<u8> = groups.list.iter().map(crate::formation::wall_kind).collect();
    // Charge phase: the run home (speed boost feeds the per-unit
    // SWING_CHARGE predicate too — momentum the sim can see).
    // The crash of a charge keeps the charge pace (frontline.rs).
    let charging: Vec<bool> = groups.list.iter().map(|g| g.charging || g.crashing).collect();
    // Fatigue locomotion: tired legs are slow legs, and exhausted
    // regiments cannot sprint the charge home (MTW1 "cannot run or
    // charge"; fleeing men tire too — pursuit catches them).
    let fat_speed: Vec<f32> = groups
        .list
        .iter()
        .map(|g| crate::fatigue::speed_mult(g.fatigue))
        .collect();
    let fat_nocharge: Vec<bool> = groups
        .list
        .iter()
        .map(|g| crate::fatigue::cannot_charge(g.fatigue))
        .collect();
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
    // The spear-line hazard reads the SPEARMAN's facing from the
    // charger's side of the scan: the shared yaw column at the tick's
    // start serves as that read-only snapshot.
    job.bounds_min = bounds_min;
    job.bounds_max = bounds_max;
    job.faces_spearwall = faces_spearwall;
    job.anchors = anchors;
    job.reg_broken = group_broken;
    job.press = press;
    job.engaged = engaged;
    job.contact = contact;
    job.fight_point = fight_point;
    job.melee_ticks = melee_ticks;
    job.hold = hold;
    job.threat = threat;
    job.form_face = form_face;
    job.wall = wall;
    job.group_wall = group_wall;
    job.charging = charging;
    job.fat_speed = fat_speed;
    job.fat_nocharge = fat_nocharge;
}
