//! Frontline VISUALIZATION. The "front line" is not a mechanic: it is a
//! readout of where the two masses physically collide. Units never steer by
//! it — movement is orders + collision (sim/soldier.rs).
//!
//! Per fixed tick, on a coarse 8 m grid: splat + blur per-team density,
//! then marching-squares the phi = 0 contour of phi = blue − orange,
//! restricted to cells where both teams are present. Drawn as gizmos.
//! Morale reads the density for its flank and surround terms. The tick
//! job builds the field on the worker from its shared columns (the
//! positions this tick starts with) and hands it over at the take;
//! `update_field` rebuilds it inline only when no job was taken.

use bevy::prelude::*;

use crate::sim::DebugViz;
use crate::orders::Groups;
use crate::terrain::Terrain;
use crate::units::Units;

pub const FIELD_CELL: f32 = 8.0;
/// Both teams' blurred density must exceed this for a cell to be "contact":
/// the drawn line only exists where masses genuinely collide.
const CONTACT_T: f32 = 0.5;
/// Attack orders enter the charge phase (war cry, sprint pose) inside
/// this distance to the target regiment's centroid. Pub: the audio war
/// cry keys off the same range.
pub const CHARGE_RANGE: f32 = 60.0;
/// Ticks `engaged` stays on after the last soldier's wind-up — bridges
/// the recover/ready gaps between swing cycles (~1.5 s at 30 Hz).
const ENGAGE_HOLD_TICKS: u8 = 45;
/// Enemy regiment centroid within this range flags `enemy_near`: the
/// regiment's units run the sparse-fight wide acquisition.
const ENEMY_NEAR_R: f32 = 60.0;
/// Victory-cheer length (~5 s at 30 Hz) after the last nearby unbroken
/// enemy regiment routs or dies. Pub: render encodes cheer progress.
pub const CELEBRATE_TICKS: u16 = 150;
/// The fixed tick.
const TICK_DT: f32 = 1.0 / 30.0;
/// A crashing block has been stopped when its smoothed forward speed
/// falls under this (m/s). Its crash lasts at most the time its rear
/// needs to arrive at the charge pace, and never longer than the cap.
const CRASH_STALL: f32 = 0.3;
const CRASH_PACE: f32 = 4.0;
const CRASH_MAX_TICKS: u16 = 240;

#[derive(Resource, Default)]
pub struct InfluenceField {
    origin: Vec2,
    w: usize,
    h: usize,
    /// Blurred per-team density, units per cell.
    d: [Vec<f32>; 2],
    scratch: Vec<f32>,
    /// Per-chunk splat accumulators for the parallel density rebuild.
    splat_scratch: Vec<[Vec<f32>; 2]>,
    /// Front contour segments (world space), for gizmos.
    pub segments: Vec<(Vec2, Vec2)>,
}

impl InfluenceField {
    fn new(min: Vec2, max: Vec2) -> Self {
        let w = ((max.x - min.x) / FIELD_CELL).ceil() as usize + 1;
        let h = ((max.y - min.y) / FIELD_CELL).ceil() as usize + 1;
        Self {
            origin: min,
            w,
            h,
            d: [vec![0.0; w * h], vec![0.0; w * h]],
            scratch: vec![0.0; w * h],
            splat_scratch: Vec::new(),
            segments: Vec::new(),
        }
    }

    #[inline]
    fn cell_index(&self, p: Vec2) -> usize {
        let g = (p - self.origin) / FIELD_CELL;
        let x = (g.x.round().max(0.0) as usize).min(self.w - 1);
        let z = (g.y.round().max(0.0) as usize).min(self.h - 1);
        z * self.w + x
    }

    /// Blurred density of a team at a point (nearest cell).
    pub fn density(&self, team: u8, p: Vec2) -> f32 {
        self.d[team as usize][self.cell_index(p)]
    }

    #[inline]
    fn grid_world(&self, x: usize, z: usize) -> Vec2 {
        self.origin + Vec2::new(x as f32, z as f32) * FIELD_CELL
    }

    #[inline]
    fn phi(&self, i: usize) -> f32 {
        self.d[0][i] - self.d[1][i]
    }

    /// Size the field to the battlefield if it is not already: the job's
    /// own field starts empty and follows the terrain of the battle.
    pub(crate) fn size_to(&mut self, min: Vec2, max: Vec2) {
        let w = ((max.x - min.x) / FIELD_CELL).ceil() as usize + 1;
        let h = ((max.y - min.y) / FIELD_CELL).ceil() as usize + 1;
        if self.origin != min || self.w != w || self.h != h {
            *self = Self::new(min, max);
        }
    }

    /// The field of these positions: the density splat and blur per
    /// team, then the contour. Callable from the tick job (every scope
    /// is `util::sim_scope`) and from the main thread alike; the counts
    /// are integers, so the partition into chunks changes no bit.
    pub(crate) fn rebuild(&mut self, pos: &[Vec3], team: &[u8]) {
        if self.w == 0 || self.h == 0 {
            return;
        }
        self.rebuild_density(pos, team);
        self.extract_contour();
    }

    fn rebuild_density(&mut self, pos: &[Vec3], team: &[u8]) {
        // Parallel splat into per-chunk fields, each task clearing its
        // own, then a merge parallel over strips of cells; the blur
        // passes stay serial (the field is only ~12k cells).
        const CHUNK: usize = 16_384;
        let n = pos.len();
        let n_chunks = n.div_ceil(CHUNK);
        let (w, h, origin) = (self.w, self.h, self.origin);
        let cells = w * h;
        self.splat_scratch
            .resize_with(n_chunks.max(self.splat_scratch.len()), Default::default);
        crate::util::sim_scope(|scope| {
            for (ci, chunk_fields) in self.splat_scratch.iter_mut().enumerate().take(n_chunks) {
                scope.spawn(async move {
                    for f in chunk_fields.iter_mut() {
                        f.clear();
                        f.resize(cells, 0.0);
                    }
                    let start = ci * CHUNK;
                    let end = (start + CHUNK).min(n);
                    for i in start..end {
                        let g = (Vec2::new(pos[i].x, pos[i].z) - origin) / FIELD_CELL;
                        let x = (g.x as usize).min(w - 1);
                        let z = (g.y as usize).min(h - 1);
                        chunk_fields[team[i] as usize][z * w + x] += 1.0;
                    }
                });
            }
        });
        // Merge: every task sums the chunk fields over its own strip of
        // cells, for both teams.
        const STRIP: usize = 1024;
        let scratch = &self.splat_scratch[..n_chunks];
        let [d0, d1] = &mut self.d;
        crate::util::sim_scope(|scope| {
            for (si, (s0, s1)) in d0.chunks_mut(STRIP).zip(d1.chunks_mut(STRIP)).enumerate() {
                let off = si * STRIP;
                scope.spawn(async move {
                    let len = s0.len();
                    s0.fill(0.0);
                    s1.fill(0.0);
                    for chunk_fields in scratch {
                        for (dst, src) in s0.iter_mut().zip(&chunk_fields[0][off..off + len]) {
                            *dst += *src;
                        }
                        for (dst, src) in s1.iter_mut().zip(&chunk_fields[1][off..off + len]) {
                            *dst += *src;
                        }
                    }
                });
            }
        });
        for team in 0..2 {
            for _ in 0..3 {
                self.blur_pass(team);
            }
        }
    }

    /// Separable radius-1 box blur (both axes).
    fn blur_pass(&mut self, team: usize) {
        let (w, h) = (self.w, self.h);
        let d = &mut self.d[team];
        let s = &mut self.scratch;
        for z in 0..h {
            let row = z * w;
            for x in 0..w {
                let l = d[row + x.saturating_sub(1)];
                let r = d[row + (x + 1).min(w - 1)];
                s[row + x] = (l + d[row + x] + r) / 3.0;
            }
        }
        for z in 0..h {
            for x in 0..w {
                let u = s[z.saturating_sub(1) * w + x];
                let b = s[(z + 1).min(h - 1) * w + x];
                d[z * w + x] = (u + s[z * w + x] + b) / 3.0;
            }
        }
    }

    /// Marching squares on phi = 0, masked to contact cells.
    fn extract_contour(&mut self) {
        self.segments.clear();
        for z in 0..self.h - 1 {
            for x in 0..self.w - 1 {
                let i00 = z * self.w + x;
                let i10 = i00 + 1;
                let i01 = i00 + self.w;
                let i11 = i01 + 1;
                // Contact: both teams present at any corner of this square.
                let contact = [i00, i10, i01, i11]
                    .iter()
                    .any(|&i| self.d[0][i].min(self.d[1][i]) > CONTACT_T);
                if !contact {
                    continue;
                }
                let f = [self.phi(i00), self.phi(i10), self.phi(i11), self.phi(i01)];
                let mut case = 0usize;
                for (b, v) in f.iter().enumerate() {
                    if *v >= 0.0 {
                        case |= 1 << b;
                    }
                }
                if case == 0 || case == 15 {
                    continue;
                }
                // Corner positions (00,10,11,01 clockwise-ish).
                let p = [
                    self.grid_world(x, z),
                    self.grid_world(x + 1, z),
                    self.grid_world(x + 1, z + 1),
                    self.grid_world(x, z + 1),
                ];
                // Edge interpolators: edge k joins corner k and (k+1)%4.
                let edge = |k: usize| -> Vec2 {
                    let (a, b) = (k, (k + 1) % 4);
                    let t = f[a] / (f[a] - f[b]);
                    p[a].lerp(p[b], t.clamp(0.0, 1.0))
                };
                // Segment table (pairs of edges), ambiguous cases split arbitrarily.
                const TABLE: [&[(usize, usize)]; 16] = [
                    &[],
                    &[(3, 0)],
                    &[(0, 1)],
                    &[(3, 1)],
                    &[(1, 2)],
                    &[(3, 0), (1, 2)],
                    &[(0, 2)],
                    &[(3, 2)],
                    &[(2, 3)],
                    &[(2, 0)],
                    &[(0, 1), (2, 3)],
                    &[(2, 1)],
                    &[(1, 3)],
                    &[(1, 0)],
                    &[(0, 3)],
                    &[],
                ];
                for &(e0, e1) in TABLE[case] {
                    self.segments.push((edge(e0), edge(e1)));
                }
            }
        }
    }

}

pub struct FrontlinePlugin;

impl Plugin for FrontlinePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_field)
            .add_systems(
                FixedUpdate,
                (update_field, update_groups)
                    .chain()
                    .after(crate::sim::take_tick)
                    .before(crate::sim::step_sim)
                    .in_set(crate::game_state::SimSet),
            )
            .add_systems(Update, (draw_front_gizmos, test_front_script));
    }
}

fn init_field(mut commands: Commands, terrain: Res<Terrain>) {
    commands.insert_resource(InfluenceField::new(terrain.min(), terrain.max()));
}

/// The density field of this tick's start positions. The taken job
/// built it on the worker and `take_tick` swapped it in; with no job
/// taken it is rebuilt here, inline, from the same positions.
fn update_field(
    field: Option<ResMut<InfluenceField>>,
    units: Res<Units>,
    pipeline: Res<crate::sim::TickPipeline>,
    mut stats: ResMut<crate::sim::SimStats>,
) {
    if pipeline.field_from_job {
        return;
    }
    let Some(mut field) = field else { return };
    let t0 = std::time::Instant::now();
    {
        let _span = info_span!("density_field").entered();
        field.rebuild(&units.pos[..], &units.team[..]);
    }
    stats.field_ms = t0.elapsed().as_secs_f32() * 1000.0;
}

/// Refresh group centroids, contact flags, and charge state (bookkeeping
/// only — nothing here steers units).
/// Fraction of living strength that must be fighting for the melee
/// clock and the contact frame to start. The
/// engine tracks both engagedSoldiers count and engagedRatio per
/// enemy unit; a percentage scales to remnants (10 absolute was a
/// third of a 30-man remnant but 1% of a full regiment). Floor of 4
/// so a 2-man trickle against a 50-man remnant never locks.
const ENGAGE_LOCK_FRAC: f32 = 0.03;
const ENGAGE_LOCK_FLOOR: u32 = 4;

/// One regiment's sums over its men, in index order.
#[derive(Clone, Copy)]
struct RegimentSums {
    pos: Vec2,
    count: usize,
    home: Vec2,
    r2: f32,
    slot_err: f32,
    front_off: f32,
    back_off: f32,
    line_sum: f32,
    line_n: u32,
    fight_n: u32,
}

fn update_groups(
    units: Res<Units>,
    runs: Res<crate::sim::RegimentRuns>,
    mut groups: ResMut<Groups>,
) {
    let n = groups.list.len();
    // Disorder measures SHAPE coherence, not travel: deviation from the
    // slot relative to the regiment's own centroid (last tick's — 33 ms
    // stale is nothing at 2 s smoothing). A rigid march scores ~0; a
    // block churned up by melee scores meters.
    let prev_cents: Vec<Vec2> = groups.list.iter().map(|g| g.centroid).collect();
    let prev_bias: Vec<Vec2> = groups.list.iter().map(|g| g.home_bias).collect();
    // Contact frame inputs (see the frame below): each regiment's
    // forward vector, its front-most slot, and the depth of the men
    // fighting an enemy ahead of the frame.
    let fwd: Vec<Vec2> = groups
        .list
        .iter()
        .map(|g| crate::formation::facing_dir(g.facing))
        .collect();
    // The sums over the men, one task per regiment walking its runs in
    // index order: the same additions in the same order as one scan of
    // the army, regiment by regiment, so the same bits. Bevy's scope
    // returns the tasks' results in spawn order.
    let units = &*units;
    let runs = &*runs;
    let (prev_cents_r, prev_bias_r, fwd_r) = (&prev_cents, &prev_bias, &fwd);
    let sums: Vec<RegimentSums> = bevy::tasks::ComputeTaskPool::get().scope(|scope| {
        for g in 0..n {
            scope.spawn(async move {
                let (pc, pb, f) = (prev_cents_r[g], prev_bias_r[g], fwd_r[g]);
                let mut a = RegimentSums {
                    pos: Vec2::ZERO,
                    count: 0,
                    home: Vec2::ZERO,
                    r2: 0.0,
                    slot_err: 0.0,
                    front_off: f32::MIN,
                    back_off: f32::MAX,
                    line_sum: 0.0,
                    line_n: 0,
                    fight_n: 0,
                };
                for &(s, e) in runs.of(g) {
                    for i in s as usize..e as usize {
                        let p = Vec2::new(units.pos[i].x, units.pos[i].z);
                        a.front_off = a.front_off.max(units.home[i].dot(f));
                        a.back_off = a.back_off.min(units.home[i].dot(f));
                        a.pos += p;
                        a.count += 1;
                        a.home += units.home[i];
                        a.r2 += (p - pc).length_squared();
                        a.slot_err += (p - pc - units.home[i] + pb).length();
                        // Ground-truth contact: a unit in WIND-UP has an enemy in reach
                        // and is striking. TW rule: one soldier fighting engages the
                        // regiment. (`target` is stale outside a swing cycle and `swing`
                        // spawns in Recover for strike staggering, so neither is usable.)
                        // A bow DRAW is a wind-up too but not melee: counting it made
                        // an archer regiment "engaged" the moment it drew, which
                        // silenced its own fire solution before the first loose.
                        if units.death_t[i] == 0
                            && units.swing[i] & crate::units::SWING_STATE_MASK
                                == crate::units::SWING_WINDUP
                            && units.swing[i] & crate::units::SWING_RANGED == 0
                        {
                            a.fight_n += 1;
                            let ti = units.target[i] as usize;
                            if ti < units.len() {
                                let d = Vec2::new(units.pos[ti].x - p.x, units.pos[ti].z - p.y);
                                if d.dot(f) > 0.5 * d.length() {
                                    a.line_sum += p.dot(f);
                                    a.line_n += 1;
                                }
                            }
                        }
                    }
                }
                a
            });
        }
    });
    let counts: Vec<usize> = sums.iter().map(|a| a.count).collect();
    let fighting: Vec<bool> = sums.iter().map(|a| a.fight_n > 0).collect();
    let cents: Vec<Vec2> = sums
        .iter()
        .map(|a| if a.count > 0 { a.pos / a.count as f32 } else { Vec2::ZERO })
        .collect();
    let teams: Vec<u8> = groups.list.iter().map(|g| g.team).collect();
    let broken: Vec<bool> = groups.list.iter().map(|g| g.state.is_broken()).collect();

    for (g, group) in groups.list.iter_mut().enumerate() {
        group.count = counts[g];
        if counts[g] == 0 {
            group.engaged = false;
            group.engage_hold = 0;
            group.charging = false;
            continue;
        }
        group.centroid = cents[g];
        group.home_bias = sums[g].home / counts[g] as f32;
        // Formation disorder: how far the regiment stands from its slots,
        // smoothed ~2 s. Only Rect makes the discipline claim.
        let err = if group.shape == crate::formation::FormShape::Rect {
            sums[g].slot_err / counts[g] as f32
        } else {
            0.0
        };
        group.disorder += (err - group.disorder) / 60.0;
        // Footprint radius: RMS distance x 1.5 reaches the block edge
        // (uniform disc: RMS = R/sqrt(2)); smoothed like disorder.
        let r = (sums[g].r2 / counts[g] as f32).sqrt() * 1.5;
        group.radius += (r - group.radius) / 60.0;
        let mut nearest_d2 = ENEMY_NEAR_R * ENEMY_NEAR_R;
        let mut threat = Vec2::ZERO;
        let mut hostile = false;
        let mut nearest_formed_d2 = ENEMY_NEAR_R * ENEMY_NEAR_R;
        let mut nearest_formed: Option<Vec2> = None;
        for t in 0..n {
            if t != g && counts[t] > 0 && teams[t] != group.team {
                let d2 = cents[t].distance_squared(group.centroid);
                if d2 < nearest_d2 {
                    nearest_d2 = d2;
                    threat = cents[t] - group.centroid;
                }
                if !broken[t] && d2 < nearest_formed_d2 {
                    nearest_formed_d2 = d2;
                    nearest_formed = Some(cents[t]);
                }
                if d2 < ENEMY_NEAR_R * ENEMY_NEAR_R && !broken[t] {
                    hostile = true;
                }
            }
        }
        group.enemy_near = threat != Vec2::ZERO;
        group.threat_dir = threat.normalize_or_zero();
        // Victory cheer: the last UNBROKEN enemy regiment nearby routed
        // or died — the line roars (render-only, ~5 s).
        if group.hostile_near && !hostile && !group.state.is_broken() {
            group.celebrate = CELEBRATE_TICKS;
            info!("regiment {g} CHEERS");
        }
        if hostile || group.state.is_broken() {
            group.celebrate = 0;
        } else {
            group.celebrate = group.celebrate.saturating_sub(1);
        }
        group.hostile_near = hostile;

        if fighting[g] {
            group.engage_hold = ENGAGE_HOLD_TICKS;
        } else {
            group.engage_hold = group.engage_hold.saturating_sub(1);
        }
        let engaged = group.engage_hold > 0;
        let lock_threshold = ((group.count as f32 * ENGAGE_LOCK_FRAC) as u32).max(ENGAGE_LOCK_FLOOR);
        if engaged != group.engaged {
            info!(
                "regiment {g} {}",
                if engaged { "ENGAGED" } else { "DISENGAGED" }
            );
        }
        // Contact frame. An attacking regiment's slots are laid on its
        // target's live center, which is right for the approach but
        // wrong in melee: every slot sits inside the enemy and moves
        // with it, so rear men lean on the backs ahead and whole blocks
        // slide after a moving center. M2TW stops
        // updating a formation near the end of its path
        // (formation_hold_distance). So once a real share of the
        // regiment is fighting, whoever the enemy is, the frame holds:
        // laterally where the block stood at contact, and in depth with
        // its front slot on the fight line, the mean position of the
        // men striking at an enemy ahead, as it stands at contact. Then
        // the frame stays put for the whole fight, as an M2TW formation
        // does: a file with no enemy in front of it holds its slots, and
        // men who see an enemy go to him themselves (sim/soldier.rs).
        // Built from the regiment's own slot geometry, so any width,
        // depth and spacing works.
        // Melee clock and fight point (sim/soldier.rs joins the men out of
        // sight of an enemy to the fight after their own delay). The
        // clock starts at the count gate, so a stray poke does not pull
        // a whole regiment in; M2TW engages a unit when enough enemy
        // soldiers are in its proximity zone.
        // The crash of a charge: the block keeps coming until the enemy
        // has stopped it; only then does the melee begin.
        if engaged
            && !group.state.is_broken()
            && !group.crashing
            && (group.melee_ticks > 0 || sums[g].fight_n >= lock_threshold)
        {
            group.melee_ticks = group.melee_ticks.saturating_add(1);
        } else {
            // The melee is over: its men come back into formation
            // (sim/soldier.rs clears out_form). A regiment with no attack
            // order re-forms where it stands, M2TW's discrete reforming
            // state; an attacker's order lays its slots again.
            if group.melee_ticks > 0
                && group.shape == crate::formation::FormShape::Rect
                && !group.state.is_broken()
                && !matches!(group.order, Some(crate::orders::Order::Attack(_)))
            {
                group.anchor = group.centroid;
                group.reform = true;
            }
            group.melee_ticks = 0;
        }
        group.fight_point = if group.melee_ticks > 0 && !group.hold {
            match group.order {
                Some(crate::orders::Order::Attack(t)) if counts[t as usize] > 0 && !broken[t as usize] => {
                    Some(cents[t as usize])
                }
                _ => nearest_formed,
            }
        } else {
            None
        };
        let attacking = matches!(group.order, Some(crate::orders::Order::Attack(t))
            if counts[t as usize] > 0 && !broken[t as usize]);
        let formed = group.shape == crate::formation::FormShape::Rect
            && !group.state.is_broken();
        let starts = sums[g].fight_n >= lock_threshold;
        if formed && attacking && engaged && !group.crashing && (group.contact || starts) {
            let f = fwd[g];
            let r = Vec2::new(f.y, -f.x);
            if !group.contact {
                group.contact = true;
                // Sideways the frame stays where the slots already
                // were: an attack lays them around the target's center.
                // Snapping to the men's average instead sent a wide
                // line's flanks, still converging, walking back out.
                group.contact_lateral = match group.order {
                    Some(crate::orders::Order::Attack(t)) => cents[t as usize].dot(r),
                    _ => (group.centroid - group.home_bias).dot(r),
                };
                let mut depth = (group.centroid - group.home_bias).dot(f);
                if sums[g].line_n > 0 {
                    depth = sums[g].line_sum / sums[g].line_n as f32 - sums[g].front_off;
                }
                group.anchor = r * group.contact_lateral + f * depth;
                info!("regiment {g} holds a contact frame");
            }
        } else if group.contact {
            group.contact = false;
            info!("regiment {g} releases its contact frame");
        }
        group.engaged = engaged;

        // Charge phase: explicit attack order, inside charge range of the
        // target, not yet in contact. Pure predicate — no latch, nothing
        // inferred from density or speed.
        let charging = if engaged || group.state.is_broken() {
            false
        } else if let Some(crate::orders::Order::Attack(t)) = group.order {
            // An archer regiment with arrows left is VOLLEYING, not
            // charging — its attack order is a fire order (the
            // stand-off), even at point-blank. Gated on ammo rather
            // than the firing flag, which lags a tick behind a fresh
            // order and let a one-tick charge blip through. Out of
            // ammo, the same order is a real knife charge and flags
            // like one.
            let t = t as usize;
            counts[t] > 0
                && cents[t].distance(group.centroid) < CHARGE_RANGE
                && !(group.kind == crate::unit_types::KIND_ARCHER && group.ammo_left > 0)
        } else {
            false
        };
        if charging != group.charging {
            info!(
                "regiment {g} {}",
                if charging { "CHARGES" } else { "CHARGE ENDS" }
            );
        }
        // The crash: a charging regiment that engages keeps its block
        // moving (M2TW runs the charge task until most of the unit has
        // charged) until the enemy has stopped it, read off its centroid's
        // smoothed forward speed, or until its rear has had time to arrive.
        let v_fwd = (group.centroid - prev_cents[g]).dot(fwd[g]) / TICK_DT;
        group.adv_speed += (v_fwd - group.adv_speed) * 0.1;
        if group.charging && !charging && engaged && group.melee_ticks == 0 && !group.crashing {
            group.crashing = true;
            group.crash_ticks = 0;
            // The rear needs depth / pace to arrive; that long at most.
            let depth = (sums[g].front_off - sums[g].back_off).max(0.0);
            group.crash_cap = ((depth / CRASH_PACE / TICK_DT) as u16).clamp(30, CRASH_MAX_TICKS);
            info!("regiment {g} CRASHES");
        } else if group.crashing {
            group.crash_ticks = group.crash_ticks.saturating_add(1);
            let stalled = group.crash_ticks > 15 && group.adv_speed < CRASH_STALL;
            if stalled || group.crash_ticks > group.crash_cap || !engaged {
                group.crashing = false;
                info!("regiment {g} CRASH ENDS ({} ticks)", group.crash_ticks);
            }
        }
        group.charging = charging;
    }
}

fn draw_front_gizmos(
    viz: Res<DebugViz>,
    field: Option<Res<InfluenceField>>,
    terrain: Res<Terrain>,
    mut gizmos: Gizmos,
) {
    let Some(field) = field else { return };
    if !viz.0 {
        return;
    }
    let lift = |p: Vec2| Vec3::new(p.x, terrain.height_at(p.x, p.y) + 1.5, p.y);
    for (a, b) in &field.segments {
        gizmos.line(lift(*a), lift(*b), Color::srgb(1.0, 0.95, 0.35));
    }
}

/// FL_TEST_FRONT=1: march both armies of regiments into contact to form a
/// battle line, hold one regiment unordered near the front (drift watch),
/// then push it through the line as a salient.
fn test_front_script(
    time: Res<Time>,
    mut groups: ResMut<Groups>,
    mut stage: Local<u32>,
    mut next_reorder: Local<f32>,
    mut watch: Local<Option<(u32, Vec2)>>,
) {
    if std::env::var("FL_TEST_FRONT").is_err() {
        return;
    }
    let t = time.elapsed_secs();

    // Stand-in for player/AI: idle unengaged regiments ATTACK their
    // nearest enemy regiment every 15 s so remnant pockets hunt each
    // other down (also exercises attack orders + charge phase).
    if *stage >= 1 && t > *next_reorder {
        *next_reorder = t + 15.0;
        let snapshot: Vec<(u8, usize, Vec2)> = groups
            .list
            .iter()
            .map(|g| (g.team, g.count, g.centroid))
            .collect();
        for (g, group) in groups.list.iter_mut().enumerate() {
            if watch.is_some_and(|(w, _)| w as usize == g) {
                continue; // drift-watch regiment must stay unordered
            }
            if group.count == 0 || group.engaged || group.order.is_some() {
                continue;
            }
            let nearest = snapshot
                .iter()
                .enumerate()
                .filter(|(_, (team, count, _))| *team != group.team && *count > 0)
                .min_by(|a, b| {
                    a.1 .2
                        .distance_squared(group.centroid)
                        .total_cmp(&b.1 .2.distance_squared(group.centroid))
                })
                .map(|(t, _)| t as u32);
            if let Some(t) = nearest {
                group.order = Some(crate::orders::Order::Attack(t));
            }
        }
    }

    match *stage {
        0 if t > 3.0 => {
            // Every regiment advances straight across the gap; blocks keep
            // their x, fronts collide near z = 0.
            for group in groups.list.iter_mut() {
                let dir: f32 = if group.team == 0 { 1.0 } else { -1.0 };
                group.order = Some(crate::orders::Order::Move(Vec2::new(group.anchor.x, dir * 10.0)));
            }
            info!("[front-test] all regiments ordered into contact");
            *stage = 1;
        }
        1 if t > 35.0 => {
            // Hold the blue regiment nearest a spot behind the front, NO
            // order: it must stand fast (units move only when commanded).
            let spot = Vec2::new(40.0, -60.0);
            if let Some((g, _)) = groups
                .list
                .iter()
                .enumerate()
                .filter(|(_, gr)| gr.team == 0 && gr.count > 0)
                .min_by(|a, b| {
                    a.1.centroid
                        .distance_squared(spot)
                        .total_cmp(&b.1.centroid.distance_squared(spot))
                })
            {
                let group = &mut groups.list[g];
                group.anchor = group.centroid;
                group.order = None;
                *watch = Some((g as u32, group.centroid));
                info!("[front-test] regiment {g} held near the front, NO order — watching drift");
            }
            *stage = 2;
        }
        2 if t > 55.0 => {
            if let Some((g, start)) = *watch {
                let drift = groups.list[g as usize].centroid.distance(start);
                info!("[front-test] held regiment {g} drift over 20s: {drift:.2} m");
                // Now push it through the line as a salient.
                groups.list[g as usize].order =
                    Some(crate::orders::Order::Move(Vec2::new(40.0, 120.0)));
                info!("[front-test] salient regiment {g} ordered through the line");
            }
            *stage = 3;
        }
        _ => {}
    }
}
