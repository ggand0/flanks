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
/// Enemy blurred density at a group's centroid above which it counts as
/// engaged (keeps its order pressing instead of "arriving").
const ENGAGED_T: f32 = 0.8;

#[derive(Resource)]
pub struct InfluenceField {
    origin: Vec2,
    w: usize,
    h: usize,
    /// Blurred per-team density, units per cell.
    d: [Vec<f32>; 2],
    scratch: Vec<f32>,
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
        self.d[0].fill(0.0);
        self.d[1].fill(0.0);
        for i in 0..units.len() {
            let g = (Vec2::new(units.pos[i].x, units.pos[i].z) - self.origin) / FIELD_CELL;
            let x = (g.x as usize).min(self.w - 1);
            let z = (g.y as usize).min(self.h - 1);
            self.d[units.team[i] as usize][z * self.w + x] += 1.0;
        }
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
                    .before(crate::movement::step_sim),
            )
            .add_systems(Update, (draw_front_gizmos, test_front_script));
    }
}

fn init_field(mut commands: Commands, terrain: Res<Terrain>) {
    commands.insert_resource(InfluenceField::new(terrain.min(), terrain.max()));
}

fn update_field(mut field: ResMut<InfluenceField>, units: Res<Units>) {
    field.rebuild_density(&units);
    field.extract_contour();
}

/// Refresh group centroids and contact flags (bookkeeping only).
fn update_groups(units: Res<Units>, mut groups: ResMut<Groups>, field: Res<InfluenceField>) {
    let n = groups.list.len();
    let mut sums = vec![Vec2::ZERO; n];
    let mut counts = vec![0usize; n];
    for i in 0..units.len() {
        let g = units.group[i] as usize;
        sums[g] += Vec2::new(units.pos[i].x, units.pos[i].z);
        counts[g] += 1;
    }
    for (g, group) in groups.list.iter_mut().enumerate() {
        group.count = counts[g];
        if counts[g] == 0 {
            group.engaged = false;
            continue;
        }
        group.centroid = sums[g] / counts[g] as f32;
        group.engaged = field.density(1 - group.team, group.centroid) > ENGAGED_T;
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

/// FL_TEST_FRONT=1: march the armies into each other to form a battle line,
/// then cut a disc out of blue and push it through as a salient.
fn test_front_script(
    time: Res<Time>,
    mut units: ResMut<Units>,
    mut groups: ResMut<Groups>,
    mut selection: ResMut<crate::orders::Selection>,
    mut stage: Local<u32>,
    mut next_reorder: Local<f32>,
    mut watch: Local<Option<(u32, Vec2)>>,
) {
    if std::env::var("FL_TEST_FRONT").is_err() {
        return;
    }
    let t = time.elapsed_secs();

    // Stand-in for player/AI: idle unengaged groups re-target the enemy
    // mass every 15 s so remnant pockets hunt each other down.
    if *stage >= 1 && t > *next_reorder {
        *next_reorder = t + 15.0;
        let mut sums = [Vec2::ZERO; 2];
        let mut counts = [0usize; 2];
        for i in 0..units.len() {
            let tm = units.team[i] as usize;
            sums[tm] += Vec2::new(units.pos[i].x, units.pos[i].z);
            counts[tm] += 1;
        }
        for (g, group) in groups.list.iter_mut().enumerate() {
            if watch.is_some_and(|(w, _)| w as usize == g) {
                continue; // drift-watch group must stay unordered
            }
            let enemy = 1 - group.team as usize;
            if group.count > 0 && !group.engaged && counts[enemy] > 0 {
                group.order = Some(sums[enemy] / counts[enemy] as f32);
            }
        }
    }

    match *stage {
        0 if t > 3.0 => {
            groups.list[0].order = Some(Vec2::new(0.0, 40.0));
            groups.list[1].order = Some(Vec2::new(0.0, -40.0));
            info!("[front-test] armies ordered into contact");
            *stage = 1;
        }
        1 if t > 35.0 => {
            // Cut a group right behind the active front, NO order: it must
            // stand fast (units move only when commanded).
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
                groups.list[g as usize].order = None;
                *watch = Some((g, groups.list[g as usize].centroid));
                info!(
                    "[front-test] cut group {g} ({} units) near the front, NO order — watching drift",
                    groups.list[g as usize].count
                );
            }
            selection.mask.fill(false);
            selection.count = 0;
            *stage = 2;
        }
        2 if t > 55.0 => {
            if let Some((g, start)) = *watch {
                let drift = groups.list[g as usize].centroid.distance(start);
                info!("[front-test] unordered group {g} drift over 20s: {drift:.2} m");
                // Now push it through the line as a salient.
                groups.list[g as usize].order = Some(Vec2::new(40.0, 120.0));
                info!("[front-test] salient group {g} ordered through the line");
            }
            *stage = 3;
        }
        _ => {}
    }
}
