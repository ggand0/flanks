//! Frontline VISUALIZATION. The "front line" is not a mechanic: it is a
//! readout of where the two masses physically collide. Units never steer by
//! it — movement is orders + collision (movement.rs).
//!
//! Per fixed tick, on a coarse 8 m grid: splat + blur per-team density,
//! then marching-squares the phi = 0 contour of phi = blue − orange,
//! restricted to cells where both teams are present. Drawn as gizmos.
//! The density field also answers "is this group in contact?" for
//! order-arrival bookkeeping.

use bevy::prelude::*;

use crate::movement::DebugViz;
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

#[derive(Resource)]
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

    fn rebuild_density(&mut self, units: &Units) {
        // Parallel splat into per-chunk fields, then a linear merge; the
        // blur passes stay serial (the field is only ~12k cells).
        const CHUNK: usize = 16_384;
        let n_chunks = units.len().div_ceil(CHUNK);
        let (w, h, origin) = (self.w, self.h, self.origin);
        let cells = w * h;
        self.splat_scratch
            .resize_with(n_chunks.max(self.splat_scratch.len()), Default::default);
        bevy::tasks::ComputeTaskPool::get().scope(|scope| {
            for (ci, chunk_fields) in self.splat_scratch.iter_mut().enumerate().take(n_chunks) {
                scope.spawn(async move {
                    for f in chunk_fields.iter_mut() {
                        f.clear();
                        f.resize(cells, 0.0);
                    }
                    let start = ci * CHUNK;
                    let end = (start + CHUNK).min(units.len());
                    for i in start..end {
                        let g = (Vec2::new(units.pos[i].x, units.pos[i].z) - origin) / FIELD_CELL;
                        let x = (g.x as usize).min(w - 1);
                        let z = (g.y as usize).min(h - 1);
                        chunk_fields[units.team[i] as usize][z * w + x] += 1.0;
                    }
                });
            }
        });
        for team in 0..2 {
            self.d[team].fill(0.0);
            for chunk_fields in self.splat_scratch.iter().take(n_chunks) {
                for (dst, src) in self.d[team].iter_mut().zip(&chunk_fields[team]) {
                    *dst += *src;
                }
            }
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
                    .before(crate::movement::step_sim),
            )
            .add_systems(Update, (draw_front_gizmos, test_front_script));
    }
}

fn init_field(mut commands: Commands, terrain: Res<Terrain>) {
    commands.insert_resource(InfluenceField::new(terrain.min(), terrain.max()));
}

fn update_field(
    mut field: ResMut<InfluenceField>,
    units: Res<Units>,
    mut stats: ResMut<crate::movement::SimStats>,
) {
    let t0 = std::time::Instant::now();
    {
        let _span = info_span!("density_field").entered();
        field.rebuild_density(&units);
    }
    {
        let _span = info_span!("contour").entered();
        field.extract_contour();
    }
    stats.field_ms = t0.elapsed().as_secs_f32() * 1000.0;
}

/// Refresh group centroids, contact flags, and charge state (bookkeeping
/// only — nothing here steers units).
fn update_groups(units: Res<Units>, mut groups: ResMut<Groups>) {
    let n = groups.list.len();
    let mut sums = vec![Vec2::ZERO; n];
    let mut counts = vec![0usize; n];
    let mut fighting = vec![false; n];
    for i in 0..units.len() {
        let g = units.group[i] as usize;
        sums[g] += Vec2::new(units.pos[i].x, units.pos[i].z);
        counts[g] += 1;
        // Ground-truth contact: a unit in WIND-UP has an enemy in reach
        // and is striking. TW rule — one soldier fighting engages the
        // regiment. (`target` is stale outside a swing cycle and `swing`
        // spawns in Recover for strike staggering — neither is usable.)
        if units.death_t[i] == 0
            && units.swing[i] & crate::units::SWING_STATE_MASK == crate::units::SWING_WINDUP
        {
            fighting[g] = true;
        }
    }
    let cents: Vec<Vec2> = sums
        .iter()
        .zip(&counts)
        .map(|(s, c)| if *c > 0 { *s / *c as f32 } else { Vec2::ZERO })
        .collect();
    let teams: Vec<u8> = groups.list.iter().map(|g| g.team).collect();

    for (g, group) in groups.list.iter_mut().enumerate() {
        group.count = counts[g];
        if counts[g] == 0 {
            group.engaged = false;
            group.engage_hold = 0;
            group.charging = false;
            continue;
        }
        group.centroid = cents[g];
        let mut nearest_d2 = ENEMY_NEAR_R * ENEMY_NEAR_R;
        let mut threat = Vec2::ZERO;
        for t in 0..n {
            if t != g && counts[t] > 0 && teams[t] != group.team {
                let d2 = cents[t].distance_squared(group.centroid);
                if d2 < nearest_d2 {
                    nearest_d2 = d2;
                    threat = cents[t] - group.centroid;
                }
            }
        }
        group.enemy_near = threat != Vec2::ZERO;
        group.threat_dir = threat.normalize_or_zero();

        if fighting[g] {
            group.engage_hold = ENGAGE_HOLD_TICKS;
        } else {
            group.engage_hold = group.engage_hold.saturating_sub(1);
        }
        let engaged = group.engage_hold > 0;
        if engaged != group.engaged {
            info!(
                "regiment {g} {}",
                if engaged { "ENGAGED" } else { "DISENGAGED" }
            );
        }
        group.engaged = engaged;

        // Charge phase: explicit attack order, inside charge range of the
        // target, not yet in contact. Pure predicate — no latch, nothing
        // inferred from density or speed.
        let charging = if engaged || group.state.is_broken() {
            false
        } else if let Some(crate::orders::Order::Attack(t)) = group.order {
            let t = t as usize;
            counts[t] > 0 && cents[t].distance(group.centroid) < CHARGE_RANGE
        } else {
            false
        };
        if charging != group.charging {
            info!(
                "regiment {g} {}",
                if charging { "CHARGES" } else { "CHARGE ENDS" }
            );
        }
        group.charging = charging;
    }
}

fn draw_front_gizmos(
    viz: Res<DebugViz>,
    field: Res<InfluenceField>,
    terrain: Res<Terrain>,
    mut gizmos: Gizmos,
) {
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
