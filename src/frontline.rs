//! Frontline solver. Runs on a coarse (8 m) blurred density grid per team,
//! entirely in 2D, once per fixed tick:
//!
//! 1. Splat unit positions into per-team density fields, box-blur them.
//! 2. Per group: centroid, facing (order direction, else enemy-density
//!    gradient), engaged test (enemy density probe ahead).
//! 3. Engaged groups extract a front curve: K rays marched along facing to
//!    the point where enemy influence balances friendly influence.
//! 4. Same-team engaged groups are sorted laterally and their adjacent
//!    curve endpoints are blended together so the combined front connects.
//!
//! Units of engaged groups steer to the curve (movement.rs): each unit
//! projects onto the group's lateral axis, samples the curve there, and
//! holds station `clamp(depth·0.85, 0.7, max_depth)` behind it — the
//! continuous forward pull plus crowd yield produces ranks and reserves.

use bevy::prelude::*;

use crate::movement::DebugViz;
use crate::orders::{Groups, Stance};
use crate::terrain::Terrain;
use crate::units::Units;

pub const FIELD_CELL: f32 = 8.0;
pub const K_SAMPLES: usize = 48;
/// Blurred density (units per cell) above which enemy presence counts.
const ENGAGE_T: f32 = 1.5;
const SLOT_SPACING: f32 = 1.05;
/// Ray march range relative to each curve sample's base point.
const MARCH_BACK: f32 = -24.0;
const MARCH_FWD: f32 = 72.0;
const MARCH_STEP: f32 = 3.0;
/// Curve advance when a ray finds no contact (leading-edge default).
const DEFAULT_ADV: f32 = 14.0;

#[derive(Resource)]
pub struct InfluenceField {
    origin: Vec2,
    w: usize,
    h: usize,
    /// Blurred per-team density, units per cell.
    d: [Vec<f32>; 2],
    scratch: Vec<f32>,
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
        }
    }

    #[inline]
    pub fn sample(&self, team: usize, p: Vec2) -> f32 {
        let g = (p - self.origin) / FIELD_CELL;
        let gx = g.x.clamp(0.0, (self.w - 2) as f32);
        let gz = g.y.clamp(0.0, (self.h - 2) as f32);
        let (x0, z0) = (gx as usize, gz as usize);
        let (fx, fz) = (gx - x0 as f32, gz - z0 as f32);
        let d = &self.d[team];
        let i = z0 * self.w + x0;
        d[i] * (1.0 - fx) * (1.0 - fz)
            + d[i + 1] * fx * (1.0 - fz)
            + d[i + self.w] * (1.0 - fx) * fz
            + d[i + self.w + 1] * fx * fz
    }

    /// Gradient of a team's density (points toward increasing density).
    #[inline]
    pub fn grad(&self, team: usize, p: Vec2) -> Vec2 {
        const E: f32 = FIELD_CELL;
        Vec2::new(
            self.sample(team, p + Vec2::new(E, 0.0)) - self.sample(team, p - Vec2::new(E, 0.0)),
            self.sample(team, p + Vec2::new(0.0, E)) - self.sample(team, p - Vec2::new(0.0, E)),
        ) / (2.0 * E)
    }

    fn rebuild(&mut self, units: &Units) {
        self.d[0].fill(0.0);
        self.d[1].fill(0.0);
        for i in 0..units.len() {
            let g = (Vec2::new(units.pos[i].x, units.pos[i].z) - self.origin) / FIELD_CELL;
            let x = (g.x as usize).min(self.w - 1);
            let z = (g.y as usize).min(self.h - 1);
            self.d[units.team[i] as usize][z * self.w + x] += 1.0;
        }
        for team in 0..2 {
            for _ in 0..2 {
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
}

/// Endpoint links between adjacent group fronts (for gizmos).
#[derive(Resource, Default)]
pub struct FrontLinks(pub Vec<(Vec2, Vec2)>);

pub struct FrontlinePlugin;

impl Plugin for FrontlinePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FrontLinks>()
            .add_systems(Startup, init_field)
            .add_systems(
                FixedUpdate,
                (update_field, update_frontlines)
                    .chain()
                    .before(crate::movement::step_sim),
            )
            .add_systems(Update, (draw_front_gizmos, test_front_script));
    }
}

fn init_field(mut commands: Commands, terrain: Res<Terrain>) {
    commands.insert_resource(InfluenceField::new(terrain.min(), terrain.max()));
}

fn update_field(mut field: ResMut<InfluenceField>, units: Res<Units>) {
    field.rebuild(&units);
}

fn update_frontlines(
    units: Res<Units>,
    mut groups: ResMut<Groups>,
    field: Res<InfluenceField>,
    mut links: ResMut<FrontLinks>,
) {
    // Refresh centroids and counts.
    let n = groups.list.len();
    let mut sums = vec![Vec2::ZERO; n];
    let mut counts = vec![0usize; n];
    for i in 0..units.len() {
        let g = units.group[i] as usize;
        sums[g] += Vec2::new(units.pos[i].x, units.pos[i].z);
        counts[g] += 1;
    }

    // First per-group pass: centroid + facing + axis (needed before we can
    // measure lateral spread).
    for (g, group) in groups.list.iter_mut().enumerate() {
        group.count = counts[g];
        if counts[g] == 0 {
            group.engaged = false;
            group.front.clear();
            continue;
        }
        group.centroid = sums[g] / counts[g] as f32;
        let enemy = 1 - group.team as usize;

        // Facing: order direction wins; otherwise toward enemy mass.
        // Temporally smoothed — the raw density gradient jitters and a
        // twitching facing makes the whole curve swing around.
        let raw = match group.order {
            Some(t) => (t - group.centroid).normalize_or_zero(),
            None => field.grad(enemy, group.centroid).normalize_or_zero(),
        };
        let facing = if raw == Vec2::ZERO {
            group.facing
        } else if group.facing == Vec2::ZERO {
            raw
        } else {
            group.facing.lerp(raw, 0.2).normalize_or_zero()
        };
        if facing == Vec2::ZERO {
            group.engaged = false;
            group.front.clear();
            continue;
        }
        group.facing = facing;
        group.axis = Vec2::new(-facing.y, facing.x);
    }

    // Second unit pass: lateral spread (variance of projection onto each
    // group's axis). The curve must hug the actual mass — a width derived
    // from unit count alone paints phantom front lines over empty ground.
    let mut u2 = vec![0.0f32; n];
    for i in 0..units.len() {
        let g = units.group[i] as usize;
        let group = &groups.list[g];
        if group.count == 0 {
            continue;
        }
        let u = (Vec2::new(units.pos[i].x, units.pos[i].z) - group.centroid).dot(group.axis);
        u2[g] += u * u;
    }

    for (g, group) in groups.list.iter_mut().enumerate() {
        if group.count == 0 {
            continue;
        }
        let enemy = 1 - group.team as usize;
        let facing = group.facing;
        if facing == Vec2::ZERO {
            continue;
        }
        let spread = (u2[g] / group.count as f32).sqrt();
        let rows = match group.stance {
            Stance::Hold => 10.0,
            Stance::Column => 50.0,
        };
        // Stance width is the ceiling the group grows toward; +20 m margin
        // past the current spread lets it actually widen over time.
        let stance_hw = ((group.count as f32 / rows) * SLOT_SPACING * 0.5).max(8.0);
        group.half_width = (2.2 * spread + 20.0).min(stance_hw).clamp(8.0, 400.0);
        group.max_depth = rows * SLOT_SPACING * 1.5;

        // Engaged when enemy influence is present at or ahead of the mass.
        let ahead = group.centroid + facing * 30.0;
        group.engaged = field.sample(enemy, group.centroid).max(field.sample(enemy, ahead))
            > ENGAGE_T;
        if !group.engaged {
            group.front.clear();
            continue;
        }

        // Extract the front curve: per lateral sample, march toward the
        // enemy until their influence balances ours.
        group.front.resize(K_SAMPLES, Vec2::ZERO);
        for k in 0..K_SAMPLES {
            let u = (k as f32 / (K_SAMPLES - 1) as f32 - 0.5) * 2.0 * group.half_width;
            let base = group.centroid + group.axis * u;
            let mut pt = base + facing * DEFAULT_ADV;
            let mut t = MARCH_BACK;
            while t < MARCH_FWD {
                let p = base + facing * t;
                let e = field.sample(enemy, p);
                if e > ENGAGE_T && e >= field.sample(group.team as usize, p) {
                    pt = p;
                    break;
                }
                t += MARCH_STEP;
            }
            group.front[k] = pt;
        }
        // Lateral smoothing.
        for _ in 0..2 {
            for k in 1..K_SAMPLES - 1 {
                group.front[k] =
                    (group.front[k - 1] + group.front[k] * 2.0 + group.front[k + 1]) / 4.0;
            }
        }
    }

    // Neighbor links: per team, sort engaged groups laterally and blend
    // adjacent curve endpoints together so the combined front connects.
    links.0.clear();
    for team in 0..2u8 {
        let mut idx: Vec<usize> = (0..groups.list.len())
            .filter(|&g| groups.list[g].team == team && groups.list[g].engaged)
            .collect();
        if idx.len() < 2 {
            continue;
        }
        // Team lateral axis: perp of average facing.
        let avg_facing: Vec2 = idx
            .iter()
            .map(|&g| groups.list[g].facing)
            .sum::<Vec2>()
            .normalize_or_zero();
        let lateral = Vec2::new(-avg_facing.y, avg_facing.x);
        idx.sort_by(|&a, &b| {
            let pa = groups.list[a].centroid.dot(lateral);
            let pb = groups.list[b].centroid.dot(lateral);
            pa.partial_cmp(&pb).unwrap()
        });
        for w in idx.windows(2) {
            let (a, b) = (w[0], w[1]);
            // Closest endpoint pair between the two curves.
            let ends_a = [0, K_SAMPLES - 1];
            let ends_b = [0, K_SAMPLES - 1];
            let (mut best, mut ea, mut eb) = (f32::MAX, 0, 0);
            for &i in &ends_a {
                for &j in &ends_b {
                    let d2 = groups.list[a].front[i].distance_squared(groups.list[b].front[j]);
                    if d2 < best {
                        best = d2;
                        ea = i;
                        eb = j;
                    }
                }
            }
            // Only stitch fronts that are reasonably close.
            if best > 120.0 * 120.0 {
                continue;
            }
            let m = (groups.list[a].front[ea] + groups.list[b].front[eb]) * 0.5;
            blend_end(&mut groups.list[a].front, ea, m);
            blend_end(&mut groups.list[b].front, eb, m);
            links
                .0
                .push((groups.list[a].front[ea], groups.list[b].front[eb]));
        }
    }
}

/// Pull the outermost samples of a curve end toward `m` with falloff.
fn blend_end(front: &mut [Vec2], end: usize, m: Vec2) {
    const REACH: usize = 6;
    for j in 0..REACH.min(front.len()) {
        let w = (1.0 - j as f32 / REACH as f32) * 0.6;
        let k = if end == 0 { j } else { front.len() - 1 - j };
        front[k] = front[k].lerp(m, w);
    }
}

fn draw_front_gizmos(
    viz: Res<DebugViz>,
    groups: Res<Groups>,
    links: Res<FrontLinks>,
    terrain: Res<Terrain>,
    mut gizmos: Gizmos,
) {
    if !viz.0 {
        return;
    }
    let lift = |p: Vec2| Vec3::new(p.x, terrain.height_at(p.x, p.y) + 1.5, p.y);
    for group in &groups.list {
        if !group.engaged || group.front.is_empty() {
            continue;
        }
        let color = if group.team == 0 {
            Color::srgb(0.2, 0.9, 1.0)
        } else {
            Color::srgb(1.0, 0.5, 0.1)
        };
        gizmos.linestrip(group.front.iter().map(|p| lift(*p)), color);
    }
    for (a, b) in &links.0 {
        gizmos.line(lift(*a), lift(*b), Color::srgb(0.9, 0.9, 0.9));
    }
}

/// FL_TEST_FRONT=1: march the armies into each other to form a battle line,
/// then cut a disc out of blue and push it through as a salient.
fn test_front_script(
    time: Res<Time>,
    mut units: ResMut<Units>,
    mut groups: ResMut<Groups>,
    mut selection: ResMut<crate::orders::Selection>,
    mut stage: Local<u32>,
) {
    if std::env::var("FL_TEST_FRONT").is_err() {
        return;
    }
    let t = time.elapsed_secs();
    match *stage {
        0 if t > 3.0 => {
            groups.list[0].order = Some(Vec2::new(0.0, 40.0));
            groups.list[1].order = Some(Vec2::new(0.0, -40.0));
            info!("[front-test] armies ordered into contact");
            *stage = 1;
        }
        1 if t > 35.0 => {
            // Carve a salient force out of the blue rear.
            selection.mask.clear();
            selection.mask.resize(units.len(), false);
            selection.count = 0;
            let center = Vec2::new(40.0, -60.0);
            for i in 0..units.len() {
                if units.team[i] == 0
                    && Vec2::new(units.pos[i].x, units.pos[i].z).distance(center) < 40.0
                {
                    selection.mask[i] = true;
                    selection.count += 1;
                }
            }
            if let Some(g) = crate::orders::split_selection(&mut units, &mut groups, &selection) {
                groups.list[g as usize].order = Some(Vec2::new(40.0, 120.0));
                info!(
                    "[front-test] salient group {g} ({} units) ordered through the line",
                    groups.list[g as usize].count
                );
            }
            selection.mask.fill(false);
            selection.count = 0;
            *stage = 2;
        }
        _ => {}
    }
}
