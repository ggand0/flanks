//! Arrow projectiles: a SoA pool + instanced draws, never entities
//! Arrows are real objects — a solved ballistic launch,
//! 30 Hz flight under gravity, and damage applied to whatever body the
//! shaft actually crosses. No hit rolls at fire time, no accuracy auras:
//! friendly fire, the high-ground range bonus, and shield facing all
//! emerge from the flight itself, as they do in M2TW.
//!
//! Ordering contract: `update_arrows` runs after `step_sim` (fresh grid,
//! index-stable columns, spawn buffers filled) and before
//! `process_deaths` (hp/death_t writes join this tick's death sweep and
//! morale pass).

use bevy::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// Nanoseconds spent in `update_arrows` and `sync_arrow_instances` since
/// the overlay last read them (overlay.rs, the fps line).
pub static ARROW_UPDATE_NS: AtomicU64 = AtomicU64::new(0);
pub static ARROW_PACK_NS: AtomicU64 = AtomicU64::new(0);

/// Adds the time since its making to a counter when dropped.
struct Timed(&'static AtomicU64, std::time::Instant);

impl Drop for Timed {
    fn drop(&mut self) {
        self.0.fetch_add(self.1.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

use bevy::camera::visibility::NoFrustumCulling;
use bevy::render::batching::NoAutomaticBatching;

use crate::render_units::{InstanceData, InstanceMaterialData};
use crate::unit_types::{TYPES, missile};
use crate::units::hash01;

/// The missile factor's clamp: +/-12 caps the damage spread of the
/// missile curve at about 20x either way.
const MISSILE_FACTOR_CLAMP: f32 = 12.0;

/// Live-arrow pool cap. Sized ~4x the steady state of a 200k battle with
/// a full archer rear rank; looses past it are dropped and counted.
pub const ARROW_CAP: usize = 65_536;
/// Arrows stuck in the ground persist up to this many (ring buffer,
/// oldest overwritten). FL_ARROW_LITTER overrides; 0 disables.
const LITTER_CAP_DEFAULT: usize = 50_000;
/// Body-hit lateral radius (m): the shaft passes within this of a
/// soldier's axis to strike him. A man is about 0.5 m across the torso
/// and 0.8 m with his arms.
const HIT_RADIUS: f32 = 0.35;
/// FL_WOUND=<share> overrides `missile::WOUND_SHARE` for calibration.
fn wound_share() -> f32 {
    static W: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *W.get_or_init(|| crate::util::env_or("FL_WOUND", missile::WOUND_SHARE).clamp(0.0, 1.0))
}
/// Collision tests only run below the tallest body plus this.
pub(crate) const STRIKE_MARGIN: f32 = 0.3;
/// How far over his own head an archer's arrow leaves the bow
/// (sim/soldier.rs). Above the body-hit band's top margin below, or a
/// shaft is inside its own shooter at flight-time zero and kills him.
pub const LAUNCH_OVER_HEAD: f32 = 0.25;
/// The point an archer aims at on a man, as a share of the man's height.
pub const AIM_HEIGHT: f32 = 0.7;

/// One loosed arrow, queued by step_sim's integrate chunks (per-chunk
/// buffers, no write races) and drained here the same tick.
pub struct ArrowSpawn {
    pub pos: Vec3,
    pub vel: Vec3,
    pub team: u8,
    /// Shooter regiment (kill attribution feeds morale).
    pub group: u32,
}

#[derive(Resource, Default)]
pub struct ArrowSpawns(pub Vec<Vec<ArrowSpawn>>);

/// All live arrows, structure-of-arrays.
#[derive(Resource, Default)]
pub struct Arrows {
    pub pos: Vec<Vec3>,
    /// Position at the previous fixed tick; rendering lerps prev -> pos.
    pub pos_prev: Vec<Vec3>,
    pub vel: Vec<Vec3>,
    /// The shaft's heading and climb from its velocity, set at the tick
    /// (the frame's instance packing reads them instead of taking two
    /// arctangents per arrow per frame).
    pub yaw: Vec<f32>,
    pub pitch: Vec<f32>,
    pub team: Vec<u8>,
    pub group: Vec<u32>,
    /// Stable per-arrow id (monotonic): swap-removes reindex the pool
    /// every tick, so anything that follows ONE arrow across frames —
    /// the attached fly-loop sounds (audio.rs) — tracks this instead.
    pub id: Vec<u32>,
    next_id: u32,
}

impl Arrows {
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    fn swap_remove(&mut self, i: usize) {
        self.pos.swap_remove(i);
        self.pos_prev.swap_remove(i);
        self.vel.swap_remove(i);
        self.yaw.swap_remove(i);
        self.pitch.swap_remove(i);
        self.team.swap_remove(i);
        self.group.swap_remove(i);
        self.id.swap_remove(i);
    }

    fn clear(&mut self) {
        self.pos.clear();
        self.pos_prev.clear();
        self.vel.clear();
        self.yaw.clear();
        self.pitch.clear();
        self.team.clear();
        self.group.clear();
        self.id.clear();
    }
}

/// Per-regiment velocity (m/s, EMA-smoothed): the volley lead, and the
/// range tests' closing speed. Updated here each tick from the men's
/// mean velocity (`GroupData::vel`), read by step_sim's loose the next
/// tick.
#[derive(Resource, Default)]
pub struct RegTracks {
    pub vel: Vec<Vec2>,
}

/// Per-tick counters: audio budgets and the overlay read these.
#[derive(Resource, Default)]
pub struct ArrowStats {
    /// Arrows loosed this tick.
    pub loosed: usize,
    /// Arrows that ended their flight this tick (body or ground).
    pub landed: usize,
    /// Bodies struck this tick.
    pub hits: usize,
    /// Looses dropped at the pool cap (cumulative; should stay 0).
    pub dropped: u64,
    /// Since the start, for the archery tally: arrows loosed, and by the
    /// victim's team and kind the men struck, wounded and killed.
    pub loosed_total: u64,
    pub struck: [[u32; crate::unit_types::NUM_KINDS]; 2],
    pub wounded: [[u32; crate::unit_types::NUM_KINDS]; 2],
    pub killed: [[u32; crate::unit_types::NUM_KINDS]; 2],
}

/// Arrows standing in the ground where they landed (ring buffer like
/// `Corpses`): the battlefield keeps the story of every volley.
#[derive(Resource)]
pub struct StuckArrows {
    data: Vec<InstanceData>,
    cursor: usize,
    cap: usize,
    dirty: bool,
}

impl Default for StuckArrows {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            cursor: 0,
            cap: crate::util::env_or("FL_ARROW_LITTER", LITTER_CAP_DEFAULT),
            dirty: false,
        }
    }
}

impl StuckArrows {
    fn push(&mut self, inst: InstanceData) {
        if self.cap == 0 {
            return;
        }
        if self.data.len() < self.cap {
            self.data.push(inst);
        } else {
            self.data[self.cursor] = inst;
            self.cursor = (self.cursor + 1) % self.cap;
        }
        self.dirty = true;
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.cursor = 0;
        self.dirty = true;
    }
}

/// Live-arrow instanced draw bucket (refilled every frame).
#[derive(Component)]
struct ArrowBucket;
/// Stuck-arrow instanced draw bucket (refreshed when the ring changes).
#[derive(Component)]
struct StuckBucket;

pub struct ArrowsPlugin;

impl Plugin for ArrowsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Arrows>()
            .init_resource::<ArrowSpawns>()
            .init_resource::<RegTracks>()
            .init_resource::<ArrowStats>()
            .init_resource::<StuckArrows>()
            .add_systems(Startup, setup_arrow_buckets)
            .add_systems(
                FixedUpdate,
                (
                    ammo_tally
                        .after(crate::sim::take_tick)
                        .before(crate::sim::step_sim),
                    update_arrows
                        .after(crate::sim::step_sim)
                        .before(crate::combat::process_deaths),
                )
                    .in_set(crate::game_state::SimSet),
            )
            .add_systems(Update, (sync_arrow_instances, sync_stuck_instances));
    }
}

/// The arrow mesh rides the SAME instanced pipeline as units (any entity
/// with InstanceMaterialData is queued): one bucket for flying arrows,
/// one for the ground litter. Same NoAutomaticBatching requirement as
/// the unit buckets.
fn setup_arrow_buckets(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let mesh = meshes.add(crate::unit_glb::arrow_mesh());
    commands.spawn((
        Mesh3d(mesh.clone()),
        InstanceMaterialData::default(),
        ArrowBucket,
        NoFrustumCulling,
        NoAutomaticBatching,
    ));
    commands.spawn((
        Mesh3d(mesh),
        InstanceMaterialData::default(),
        StuckBucket,
        NoFrustumCulling,
        NoAutomaticBatching,
    ));
}

/// Solve a launch velocity from `from` to `to` (M2TW model: the engine
/// picks a speed in the 20..48 m/s band and an arc that fits). The flat
/// low-root solution at full draw when the path clears every friendly
/// block AND the terrain en route; otherwise the slowest lofted arc
/// that reaches — a high lob just over the crowd (M2TW's
/// loft-over-friendlies; the ~65 degree max angle emerges from the
/// speed floor). `blocks` = friendly regiment discs as
/// (centroid, radius, clearance top y).
pub fn solve_launch(
    from: Vec3,
    to: Vec3,
    blocks: &[(Vec2, f32, f32)],
    terrain: &crate::terrain::Terrain,
) -> Vec3 {
    let flat = to.xz() - from.xz();
    let d = flat.length().max(0.5);
    let dir = flat / d;
    let dy = to.y - from.y;
    let g = missile::GRAVITY;

    let mut vv = missile::SPEED_MAX * missile::SPEED_MAX;
    let disc = vv * vv - g * (g * d * d + 2.0 * dy * vv);
    let mut tan = f32::MAX;
    let mut lofted = disc <= 0.0;
    if !lofted {
        tan = (vv - disc.sqrt()) / (g * d);
        // Does the flat arc clear the friendlies under it? Height of the
        // parabola at forward distance x: x*tan - g*x^2*(1+tan^2)/(2*vv).
        for &(c, r, top) in blocks {
            let s = (c - from.xz()).dot(dir);
            if s < 2.0 || s > d - 2.0 {
                continue;
            }
            let lat = (c - from.xz()).perp_dot(dir).abs();
            if lat > r {
                continue;
            }
            let y = from.y + s * tan - g * s * s * (1.0 + tan * tan) / (2.0 * vv);
            if y < top {
                lofted = true;
                break;
            }
        }
    }
    if !lofted {
        // Terrain line of sight: ground RISING above the shooter's feet
        // blocks the flat shot — firing up a slope, the flat arc threads
        // exactly where the uphill front rank's heads are (rear archers
        // shot their own front rows), or straight into the hillside.
        // Only raised ground counts: flat fields keep their flat
        // volleys, downhill stays a sniping shot, and the last meters
        // are exempt (the arrow has to come down on the target). The
        // arc must clear a head standing on any such rise.
        let from_ground = terrain.height_at(from.x, from.z);
        let head = crate::unit_types::tallest() + 0.2;
        let step = (d / 24.0).max(4.0);
        let mut s = 2.0_f32;
        while s < d - 4.0 {
            let p = from.xz() + dir * s;
            let g_s = terrain.height_at(p.x, p.y);
            if g_s > from_ground + 0.35 {
                let y = from.y + s * tan - g * s * s * (1.0 + tan * tan) / (2.0 * vv);
                if y < g_s + head {
                    lofted = true;
                    break;
                }
            }
            s += step;
        }
    }
    if lofted {
        // Slowest speed that still reaches -> the high root sits near
        // the minimum-energy 45 degrees, +15% keeps it a clean lob
        // instead of a grazing edge case.
        let vmin2 = g * (dy + (dy * dy + d * d).sqrt());
        vv = (vmin2 * 1.15).clamp(
            missile::SPEED_MIN * missile::SPEED_MIN,
            missile::SPEED_MAX * missile::SPEED_MAX,
        );
        let disc = vv * vv - g * (g * d * d + 2.0 * dy * vv);
        tan = (vv + disc.max(0.0).sqrt()) / (g * d);
    }
    let cos = 1.0 / (1.0 + tan * tan).sqrt();
    let speed = vv.sqrt();
    Vec3::new(dir.x * speed * cos, speed * tan * cos, dir.y * speed * cos)
}

#[allow(clippy::too_many_arguments)] // bevy system params
fn update_arrows(
    mut arrows: ResMut<Arrows>,
    mut spawns: ResMut<ArrowSpawns>,
    mut units: ResMut<crate::units::Units>,
    mut groups: ResMut<crate::orders::Groups>,
    grid: Res<crate::spatial::SpatialGrid>,
    terrain: Res<crate::terrain::Terrain>,
    scale: Res<crate::sim::CombatScale>,
    mut cstats: ResMut<crate::combat::CombatStats>,
    mut stats: ResMut<ArrowStats>,
    mut stuck: ResMut<StuckArrows>,
    mut tracks: ResMut<RegTracks>,
    time: Res<Time>,
    mut sounds: Option<ResMut<crate::audio::SoundEvents>>,
    mut tick: Local<u32>,
) {
    let _timed = Timed(&ARROW_UPDATE_NS, std::time::Instant::now());
    let _span = info_span!("update_arrows").entered();
    let dt = time.delta_secs();
    *tick = tick.wrapping_add(1);

    // Regiment velocities (the volley lead, one tick stale by
    // construction: step_sim consumed last tick's values already): the
    // men's mean velocity, not the centroid's motion, which jumps when
    // men fall and read as a speed of tens of m/s for a dying block.
    if tracks.vel.len() != groups.list.len() {
        tracks.vel = vec![Vec2::ZERO; groups.list.len()];
    }
    for (g, gd) in groups.list.iter().enumerate() {
        // EMA over ~0.5 s: per-tick noise must not steer aim.
        tracks.vel[g] = tracks.vel[g] * 0.9 + gd.vel * 0.1;
    }

    // Drain this tick's looses into the pool.
    stats.loosed = 0;
    for buf in &mut spawns.0 {
        for s in buf.drain(..) {
            if arrows.len() >= ARROW_CAP {
                stats.dropped += 1;
                continue;
            }
            stats.loosed += 1;
            stats.loosed_total += 1;
            if let Some(se) = sounds.as_deref_mut() {
                se.push_loose(s.group, s.pos);
            }
            arrows.pos.push(s.pos);
            arrows.pos_prev.push(s.pos);
            arrows.vel.push(s.vel);
            let (yaw, pitch) = flight_angles(s.vel);
            arrows.yaw.push(yaw);
            arrows.pitch.push(pitch);
            arrows.team.push(s.team);
            arrows.group.push(s.group);
            let id = arrows.next_id;
            arrows.id.push(id);
            arrows.next_id = arrows.next_id.wrapping_add(1);
        }
    }

    // Integrate + collide. Hits are collected and applied serially after
    // the sweep (one arrow can kill; the next must see the corpse).
    stats.landed = 0;
    stats.hits = 0;
    struct Hit {
        victim: usize,
        team: u8,
        group: u32,
        /// Direction victim -> arrow origin side (the melee `dir`
        /// convention: where the blow comes FROM).
        from_dir: Vec2,
        jit: f32,
        /// False for a shaft that glanced off: heard, nothing else.
        wound: bool,
    }
    let mut hits: Vec<Hit> = Vec::new();
    let strike_band = crate::unit_types::tallest() + STRIKE_MARGIN;
    let mut i = 0;
    while i < arrows.len() {
        let prev = arrows.pos[i];
        let mut p = prev;
        let mut v = arrows.vel[i];
        v.y -= missile::GRAVITY * dt;
        p += v * dt;
        arrows.pos_prev[i] = prev;
        arrows.pos[i] = p;
        arrows.vel[i] = v;
        (arrows.yaw[i], arrows.pitch[i]) = flight_angles(v);

        let ground = terrain.height_at(p.x, p.z);
        if p.y.min(prev.y) > ground + strike_band {
            i += 1;
            continue;
        }

        // Body sweep: does the segment prev -> p pass through a soldier?
        // Grid holds tick-start candidates; hp/death are re-checked live.
        let a = prev.xz();
        let b = p.xz();
        let ab = b - a;
        let len2 = ab.length_squared().max(1e-6);
        let mid = (a + b) * 0.5;
        let r_query = ab.length() * 0.5 + HIT_RADIUS + 0.4;
        let mut best: Option<(f32, usize)> = None;
        grid.for_each_candidate(mid, r_query, |o| {
            if (o.meta & crate::spatial::META_DYING) != 0 {
                return;
            }
            let u = o.idx as usize;
            let t = ((o.xz() - a).dot(ab) / len2).clamp(0.0, 1.0);
            if o.xz().distance_squared(a + ab * t) > HIT_RADIUS * HIT_RADIUS {
                return;
            }
            let arrow_y = prev.y + (p.y - prev.y) * t;
            let hh = crate::unit_types::half_height(units.kind[u] as usize);
            let uy = units.pos[u].y;
            // The top margin stays below the launch height
            // (LAUNCH_OVER_HEAD) or shooters hit themselves.
            if arrow_y < uy - hh || arrow_y > uy + hh + 0.05 {
                return;
            }
            if best.is_none_or(|(bt, _)| t < bt) {
                best = Some((t, u));
            }
        });
        if let Some((_, u)) = best {
            // Live re-check: the grid snapshot can lag a melee death by
            // one tick.
            if units.death_t[u] == 0 && units.hp[u] > 0.0 {
                let seed = (*tick).wrapping_mul(0x9E37_79B1) ^ (i as u32).wrapping_mul(0x85EB);
                // The shaft that crosses a man strikes him; whether it
                // wounds him is the one roll here. The rest glance off
                // armour, helmet or the shield's rim.
                hits.push(Hit {
                    victim: u,
                    team: arrows.team[i],
                    group: arrows.group[i],
                    from_dir: -v.xz().normalize_or_zero(),
                    jit: 0.85 + 0.3 * hash01(seed),
                    wound: hash01(seed ^ 0x6A09_E667) < wound_share(),
                });
                stats.hits += 1;
                stats.landed += 1;
                arrows.swap_remove(i);
                continue;
            }
        }

        // Ground: plant the shaft where it struck, angled as it flew.
        if p.y <= ground {
            let dir = v.normalize_or_zero();
            let yaw = dir.x.atan2(dir.z);
            let pitch = dir.y.atan2(dir.xz().length());
            // Center pulled back along the flight so the head sits
            // buried and the fletching stands proud.
            let center = Vec3::new(p.x, ground, p.z) - dir * 0.16;
            stuck.push(InstanceData {
                position: center,
                w: 1.0,
                color: [1.0, 1.0, 1.0, 0.0],
                anim: [yaw, 0.0, 0.0, 0.0],
                anim2: [0.0, 0.0, pitch, 0.0],
            });
            stats.landed += 1;
            if let Some(se) = sounds.as_deref_mut() {
                se.push_arrow_ground(Vec3::new(p.x, ground, p.z));
            }
            arrows.swap_remove(i);
            continue;
        }
        i += 1;
    }

    // Serial damage apply, mirroring the melee pass (sim/damage.rs): the
    // M2TW missile resolution is armour everywhere (no AP on plain
    // arrows), shield front + LEFT only, and NO defence skill — a
    // falling shaft cannot be parried. Friendly fire is real: no team
    // check anywhere (M2TW-evidenced), though a friendly kill is not a
    // "winning exchange" for the shooter's morale.
    let combat_scale = scale.0;
    for h in hits {
        let v = h.victim;
        if units.hp[v] <= 0.0 || units.death_t[v] > 0 {
            continue;
        }
        let pv = &TYPES[units.kind[v] as usize];
        let (vt, vk) = (units.team[v] as usize, units.kind[v] as usize);
        stats.struck[vt][vk] += 1;
        let fwd = Vec2::new(units.yaw[v].sin(), units.yaw[v].cos());
        let along = fwd.dot(h.from_dir);
        let front = along >= crate::sim::damage::SECTOR_COS_60;
        let rear = along < -crate::sim::damage::SECTOR_COS_60;
        let left_side = !front && !rear && h.from_dir.dot(Vec2::new(fwd.y, -fwd.x)) > 0.0;
        let shield = if front || left_side { pv.shield } else { 0.0 };
        let factor =
            (missile::ATTACK - (pv.armour + shield)).clamp(-MISSILE_FACTOR_CLAMP, MISSILE_FACTOR_CLAMP);
        // The missile curve is steeper than melee's (missile::FACTOR_MULT):
        // armour classes must separate hard under arrows.
        let dmg = missile::BASE_DMG * missile::FACTOR_MULT.powf(factor) * h.jit * combat_scale;
        if h.wound {
            units.hp[v] -= dmg;
            units.flash[v] = 4;
        }
        if let Some(se) = sounds.as_deref_mut() {
            // A shaft cannot be parried: shield or armour takes it.
            se.push_arrow_hit(crate::audio::ArrowHit {
                pos: units.pos[v],
                group: units.group[v],
                material: crate::audio::struck_material(
                    0.0,
                    shield,
                    pv.armour,
                    units.kind[v],
                    hash01((*tick).wrapping_mul(0x27D4_EB2F) ^ (v as u32).wrapping_mul(0x1656_67B1)),
                ),
                killed: units.hp[v] <= 0.0,
            });
        }
        if !h.wound {
            continue;
        }
        stats.wounded[vt][vk] += 1;
        if units.hp[v] <= 0.0 {
            stats.killed[vt][vk] += 1;
            units.death_t[v] = crate::sim::damage::DEATH_TICKS;
            cstats.kills[units.team[v] as usize] += 1;
            groups.list[units.group[v] as usize].recent_deaths += 1;
            if units.team[v] != h.team {
                groups.list[h.group as usize].recent_kills += 1;
            }
        } else {
            // A nonfatal shaft rocks its man: the SAME stumble roll as a
            // melee blow, on the same factor — no separate mechanism.
            // M2TW has no missile-stagger parameter anywhere; what it does
            // have is the generic
            // hit reaction played on every nonfatal impact, arrows
            // included — which is exactly what the melee stumble models.
            // Braced walls hit frontally stay planted and the
            // anti-stunlock immunity pass applies, like melee.
            let braced = front
                && crate::formation::wall_kind(&groups.list[units.group[v] as usize]) != 0;
            let p = (crate::sim::damage::STAGGER_P0
                + crate::sim::damage::STAGGER_P_PER_FACTOR * factor)
                .clamp(crate::sim::damage::STAGGER_P_MIN, 1.0);
            let roll = hash01(
                (*tick)
                    .wrapping_mul(0x85EB_CA6B)
                    ^ (v as u32).wrapping_mul(0x9E37),
            );
            if !braced && roll < p {
                if units.swing[v] & crate::units::SWING_STAGGER_IMMUNE != 0 {
                    units.swing[v] &= !crate::units::SWING_STAGGER_IMMUNE;
                } else {
                    units.swing[v] =
                        crate::units::SWING_RECOVER | crate::units::SWING_STAGGERED;
                    // Clamp the stun window: inheriting a reloading
                    // archer's 8 s bow cycle as stagger time would
                    // freeze him into a statue.
                    units.swing_t[v] = units.swing_t[v]
                        .clamp(crate::sim::damage::HIT_STAGGER_TICKS, 45);
                }
            }
        }
    }
    if tally_log() && tick.is_multiple_of(120) {
        log_tally(*tick, &stats, arrows.len());
    }
}

/// FL_TEST_ARCHERY logs the calibration tally every 4 s of sim time.
fn tally_log() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("FL_TEST_ARCHERY").is_ok())
}

/// The calibration tally: arrows loosed and ended, and by kind
/// (knights/men-at-arms/spearmen/bowmen) the men struck, wounded and
/// killed on each side.
fn log_tally(tick: u32, stats: &ArrowStats, in_flight: usize) {
    let kinds = |c: &[u32; crate::unit_types::NUM_KINDS]| {
        c.iter().map(|n| n.to_string()).collect::<Vec<_>>().join("/")
    };
    let side = |team: usize| {
        format!(
            "struck {} wounded {} killed {}",
            kinds(&stats.struck[team]),
            kinds(&stats.wounded[team]),
            kinds(&stats.killed[team])
        )
    };
    info!(
        "[archery-tally] tick {tick} loosed {} ended {} | orange {} | blue {}",
        stats.loosed_total,
        stats.loosed_total - in_flight as u64,
        side(1),
        side(0),
    );
}

/// Archer regiment upkeep, before the tick: the regiment ammo pools for
/// the HUD, summed over the archer regiments' own men. The skirmish run
/// is a reaction of the tick's prep (sim/job.rs `skirmish_reactions`).
fn ammo_tally(
    mut groups: ResMut<crate::orders::Groups>,
    units: Res<crate::units::Units>,
    runs: Res<crate::sim::RegimentRuns>,
) {
    for (g, gd) in groups.list.iter_mut().enumerate() {
        if gd.kind != crate::unit_types::KIND_ARCHER {
            continue;
        }
        gd.ammo_left = 0;
        for &(s, e) in runs.of(g) {
            for i in s as usize..e as usize {
                if units.kind[i] == crate::unit_types::KIND_ARCHER && units.death_t[i] == 0 {
                    gd.ammo_left += units.ammo[i] as u32;
                }
            }
        }
    }
}

/// Clear every arrow resource for a fresh battle (setup_battle calls).
pub fn reset(arrows: &mut Arrows, spawns: &mut ArrowSpawns, stuck: &mut StuckArrows) {
    arrows.clear();
    for buf in &mut spawns.0 {
        buf.clear();
    }
    stuck.clear();
}

/// Refill the live-arrow bucket every frame, interpolated between fixed
/// ticks like units; orientation comes from the velocity.
fn sync_arrow_instances(
    arrows: Res<Arrows>,
    fixed: Res<Time<Fixed>>,
    mut query: Query<&mut InstanceMaterialData, With<ArrowBucket>>,
) {
    let _timed = Timed(&ARROW_PACK_NS, std::time::Instant::now());
    let Ok(mut data) = query.single_mut() else {
        return;
    };
    let alpha = fixed.overstep_fraction();
    data.0.clear();
    data.0.reserve(arrows.len());
    for i in 0..arrows.len() {
        let p = arrows.pos_prev[i].lerp(arrows.pos[i], alpha);
        data.0.push(InstanceData {
            position: p,
            // Flying arrows draw a third oversized: a volley must READ
            // at battle zoom (the ground litter stays true-scale).
            w: 1.35,
            color: [1.0, 1.0, 1.0, 0.0],
            anim: [arrows.yaw[i], 0.0, 0.0, 0.0],
            anim2: [0.0, 0.0, arrows.pitch[i], 0.0],
        });
    }
}

/// A shaft's heading and climb (radians) from its velocity.
#[inline]
fn flight_angles(v: Vec3) -> (f32, f32) {
    (v.x.atan2(v.z), v.y.atan2(v.xz().length()))
}

/// Copy the litter ring into its bucket when it changed.
fn sync_stuck_instances(
    mut stuck: ResMut<StuckArrows>,
    mut query: Query<&mut InstanceMaterialData, With<StuckBucket>>,
) {
    if !stuck.dirty {
        return;
    }
    let Ok(mut data) = query.single_mut() else {
        return;
    };
    stuck.dirty = false;
    data.0.clear();
    data.0.extend_from_slice(&stuck.data);
}
