//! Arrow projectiles: a SoA pool + instanced draws, never entities
//! (devlog 0011). Arrows are real objects — a solved ballistic launch,
//! 30 Hz flight under gravity, and damage applied to whatever body the
//! shaft actually crosses. No hit rolls at fire time, no accuracy auras:
//! friendly fire, the high-ground range bonus, and shield facing all
//! emerge from the flight itself (M2TW-evidenced, devlog 0060).
//!
//! Ordering contract: `update_arrows` runs after `step_sim` (fresh grid,
//! index-stable columns, spawn buffers filled) and before
//! `process_deaths` (hp/death_t writes join this tick's death sweep and
//! morale pass).

use bevy::prelude::*;

use bevy::camera::visibility::NoFrustumCulling;
use bevy::render::batching::NoAutomaticBatching;

use crate::render_units::{InstanceData, InstanceMaterialData};
use crate::unit_types::{FACTOR_CLAMP, FACTOR_MULT, TYPES, missile};
use crate::units::hash01;

/// Live-arrow pool cap. Sized ~4x the steady state of a 200k battle with
/// a full archer rear rank; looses past it are dropped and counted.
pub const ARROW_CAP: usize = 65_536;
/// Arrows stuck in the ground persist up to this many (ring buffer,
/// oldest overwritten). FL_ARROW_LITTER overrides; 0 disables.
const LITTER_CAP_DEFAULT: usize = 50_000;
/// Body-hit lateral radius (m): the shaft passes within this of a
/// soldier's axis to strike him. On the body scale (cube ~0.6 m wide).
const HIT_RADIUS: f32 = 0.35;
/// Collision tests only run below ground + this (tallest body ~1.1 m).
const STRIKE_BAND: f32 = 1.4;

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
    pub team: Vec<u8>,
    pub group: Vec<u32>,
}

impl Arrows {
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    fn swap_remove(&mut self, i: usize) {
        self.pos.swap_remove(i);
        self.pos_prev.swap_remove(i);
        self.vel.swap_remove(i);
        self.team.swap_remove(i);
        self.group.swap_remove(i);
    }

    fn clear(&mut self) {
        self.pos.clear();
        self.pos_prev.clear();
        self.vel.clear();
        self.team.clear();
        self.group.clear();
    }
}

/// Per-regiment centroid velocity (m/s, EMA-smoothed): the volley lead.
/// Updated here each tick, read by step_sim's loose the next tick.
#[derive(Resource, Default)]
pub struct RegTracks {
    last: Vec<Vec2>,
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
                    skirmish_and_ammo.before(crate::movement::step_sim),
                    update_arrows
                        .after(crate::movement::step_sim)
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
/// the unit buckets (devlog 0013).
fn setup_arrow_buckets(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let mesh = meshes.add(crate::unit_meshes::build_arrow());
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
/// block en route; otherwise the slowest lofted arc that reaches — a
/// high lob just over the crowd (M2TW's loft-over-friendlies; the ~65
/// degree max angle emerges from the speed floor). `blocks` = friendly
/// regiment discs as (centroid, radius, clearance top y).
pub fn solve_launch(from: Vec3, to: Vec3, blocks: &[(Vec2, f32, f32)]) -> Vec3 {
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
    scale: Res<crate::movement::CombatScale>,
    mut cstats: ResMut<crate::combat::CombatStats>,
    mut stats: ResMut<ArrowStats>,
    mut stuck: ResMut<StuckArrows>,
    mut tracks: ResMut<RegTracks>,
    time: Res<Time>,
    mut tick: Local<u32>,
) {
    let _span = info_span!("update_arrows").entered();
    let dt = time.delta_secs();
    *tick = tick.wrapping_add(1);

    // Regiment centroid velocities (the volley lead, one tick stale by
    // construction — step_sim consumed last tick's values already).
    if tracks.last.len() != groups.list.len() {
        tracks.last = groups.list.iter().map(|g| g.centroid).collect();
        tracks.vel = vec![Vec2::ZERO; groups.list.len()];
    }
    for (g, gd) in groups.list.iter().enumerate() {
        let v = (gd.centroid - tracks.last[g]) / dt.max(1e-3);
        // EMA over ~0.5 s: per-tick centroid noise must not steer aim.
        tracks.vel[g] = tracks.vel[g] * 0.9 + v * 0.1;
        tracks.last[g] = gd.centroid;
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
            arrows.pos.push(s.pos);
            arrows.pos_prev.push(s.pos);
            arrows.vel.push(s.vel);
            arrows.team.push(s.team);
            arrows.group.push(s.group);
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
    }
    let mut hits: Vec<Hit> = Vec::new();
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

        let ground = terrain.height_at(p.x, p.z);
        if p.y.min(prev.y) > ground + STRIKE_BAND {
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
            let hh = TYPES[units.kind[u] as usize].half_height;
            let uy = units.pos[u].y;
            // Top margin stays BELOW the launch height (movement.rs
            // looses at +0.75 over mid-body) or shooters hit themselves.
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
                hits.push(Hit {
                    victim: u,
                    team: arrows.team[i],
                    group: arrows.group[i],
                    from_dir: -v.xz().normalize_or_zero(),
                    jit: 0.85 + 0.3 * hash01(seed),
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
                scale: 1.0,
                color: [1.0, 1.0, 1.0, 0.0],
                anim: [yaw, 0.0, 0.0, 0.0],
                anim2: [0.0, 0.0, pitch, 0.0],
            });
            stats.landed += 1;
            arrows.swap_remove(i);
            continue;
        }
        i += 1;
    }

    // Serial damage apply, mirroring the melee pass (movement.rs): the
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
        let fwd = Vec2::new(units.yaw[v].sin(), units.yaw[v].cos());
        let along = fwd.dot(h.from_dir);
        let front = along >= crate::movement::SECTOR_COS_60;
        let rear = along < -crate::movement::SECTOR_COS_60;
        let left_side = !front && !rear && h.from_dir.dot(Vec2::new(fwd.y, -fwd.x)) > 0.0;
        let shield = if front || left_side { pv.shield } else { 0.0 };
        let factor =
            (missile::ATTACK - (pv.armour + shield)).clamp(-FACTOR_CLAMP, FACTOR_CLAMP);
        let dmg = missile::BASE_DMG * FACTOR_MULT.powf(factor) * h.jit * combat_scale;
        units.hp[v] -= dmg;
        units.flash[v] = 4;
        if units.hp[v] <= 0.0 {
            units.death_t[v] = crate::movement::DEATH_TICKS;
            cstats.kills[units.team[v] as usize] += 1;
            groups.list[units.group[v] as usize].recent_deaths += 1;
            if units.team[v] != h.team {
                groups.list[h.group as usize].recent_kills += 1;
            }
        }
    }
}

/// Skirmish trigger: EDGE gap (centroid distance minus both footprint
/// radii) — centroid rings read absurdly late for 1000-man blocks
/// whose fronts touch at 30+ m of centroid separation. No M2TW value
/// exists (the manual only says "keep a safe distance, usually its
/// missile range") — these are feel knobs sized so the back-step
/// starts with real room to run.
const SKIRMISH_TRIGGER: f32 = 22.0;
/// The withdrawal runs until the edge gap reopens to this.
const SKIRMISH_REOPEN: f32 = 40.0;

/// Archer regiment upkeep, before the tick: the M2TW skirmish rule
/// ("keep a safe distance between itself and the enemy", manual) as an
/// auto Move order — real movement through the real order pipe, cleared
/// when the gap reopens. Player-issued orders always win: skirmish only
/// steers an idle regiment or its own withdrawal. Also tallies the
/// regiment ammo pools for the HUD.
fn skirmish_and_ammo(
    mut groups: ResMut<crate::orders::Groups>,
    units: Res<crate::units::Units>,
    terrain: Res<crate::terrain::Terrain>,
) {
    // Regiment ammo pools (the unit-card ammo bar).
    for gd in groups.list.iter_mut() {
        if gd.kind == crate::unit_types::KIND_ARCHER {
            gd.ammo_left = 0;
        }
    }
    for i in 0..units.len() {
        if units.kind[i] == crate::unit_types::KIND_ARCHER && units.death_t[i] == 0 {
            groups.list[units.group[i] as usize].ammo_left += units.ammo[i] as u32;
        }
    }

    let n = groups.list.len();
    for g in 0..n {
        let gd = &groups.list[g];
        if gd.kind != crate::unit_types::KIND_ARCHER
            || gd.count == 0
            || gd.state.is_broken()
            || !gd.skirmish
            || gd.hold
            || gd.engaged
        {
            continue;
        }
        // Player orders win; only idle regiments and our own withdrawal
        // Move are steered.
        if gd.order.is_some() && !gd.skirmishing {
            continue;
        }
        // Nearest formed enemy block, by EDGE gap.
        let mut nearest: Option<(f32, Vec2)> = None;
        for eg in groups.list.iter() {
            if eg.team == gd.team || eg.count == 0 || eg.state.is_broken() {
                continue;
            }
            let gap = eg.centroid.distance(gd.centroid) - eg.radius - gd.radius;
            if nearest.is_none_or(|(bd, _)| gap < bd) {
                nearest = Some((gap, eg.centroid));
            }
        }
        let centroid = gd.centroid;
        let was_skirmishing = gd.skirmishing;
        match nearest {
            Some((gap, ec)) if gap < SKIRMISH_TRIGGER => {
                // Back-step directly away, far enough to reopen the gap.
                let away = (centroid - ec).normalize_or_zero();
                let away = if away == Vec2::ZERO { Vec2::Y } else { away };
                let mut dest = centroid + away * (SKIRMISH_REOPEN - gap + 5.0);
                let (mn, mx) = (terrain.min() + 12.0, terrain.max() - 12.0);
                dest = dest.clamp(mn, mx);
                let gd = &mut groups.list[g];
                gd.order = Some(crate::orders::Order::Move(dest));
                gd.auto_order = true;
                gd.skirmishing = true;
            }
            Some((gap, _)) if was_skirmishing && gap > SKIRMISH_REOPEN => {
                // Gap reopened: stand, dress where we are, shoot.
                let gd = &mut groups.list[g];
                gd.skirmishing = false;
                gd.order = None;
                gd.anchor = gd.centroid;
                gd.reform = true;
            }
            None if was_skirmishing => {
                let gd = &mut groups.list[g];
                gd.skirmishing = false;
                gd.order = None;
                gd.anchor = gd.centroid;
            }
            _ => {}
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
    let Ok(mut data) = query.single_mut() else {
        return;
    };
    let alpha = fixed.overstep_fraction();
    data.0.clear();
    data.0.reserve(arrows.len());
    for i in 0..arrows.len() {
        let p = arrows.pos_prev[i].lerp(arrows.pos[i], alpha);
        let v = arrows.vel[i];
        let yaw = v.x.atan2(v.z);
        let pitch = v.y.atan2(v.xz().length());
        data.0.push(InstanceData {
            position: p,
            // Flying arrows draw a third oversized: a volley must READ
            // at battle zoom (the ground litter stays true-scale).
            scale: 1.35,
            color: [1.0, 1.0, 1.0, 0.0],
            anim: [yaw, 0.0, 0.0, 0.0],
            anim2: [0.0, 0.0, pitch, 0.0],
        });
    }
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
