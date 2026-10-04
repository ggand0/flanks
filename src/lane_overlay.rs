//! FL_DEBUG_LANE=1: the lane of each soldier's blow, drawn on the ground
//! under the soldiers near the camera. A soldier with an enemy in reach
//! shows the strip from himself to that enemy (sim/soldier.rs `in_lane`):
//! green when nothing stands in it, red when a comrade does and he does
//! not strike. A soldier with no enemy in reach shows nothing.
//! FL_DEBUG_LANE=<m> sets how far from the camera's focus soldiers are
//! drawn.
//!
//! The overlay runs the sim's test on every frame. The sim acts on a
//! man's last look, up to a quarter second old.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::math::Vec3Swizzles;
use bevy::mesh::PrimitiveTopology;
use bevy::prelude::*;

use crate::sim::soldier::{SPEAR_LINE_HALF_W, in_lane};
use crate::unit_types::TYPES;

/// Soldiers this close to the camera's focus are drawn (metres).
const RADIUS_DEFAULT: f32 = 30.0;
/// Height of the overlay over the ground.
const LIFT: f32 = 0.06;
/// Width of the line along each side of a lane (metres).
const LINE: f32 = 0.035;

/// Fill colours; the line along a lane is the same colour, stronger.
const GREEN: [f32; 4] = [0.15, 0.95, 0.30, 0.30];
const RED: [f32; 4] = [1.00, 0.12, 0.08, 0.42];
const LINE_ALPHA: f32 = 0.9;

#[derive(Resource)]
struct LaneOverlay {
    radius: f32,
}

#[derive(Component)]
struct LaneMesh;

pub struct LaneOverlayPlugin;

impl Plugin for LaneOverlayPlugin {
    fn build(&self, app: &mut App) {
        let Ok(value) = std::env::var("FL_DEBUG_LANE") else {
            return;
        };
        let radius = value.parse::<f32>().ok().filter(|r| *r > 1.0).unwrap_or(RADIUS_DEFAULT);
        app.insert_resource(LaneOverlay { radius })
            .add_systems(Startup, spawn_overlay)
            .add_systems(Update, draw_overlay);
    }
}

fn spawn_overlay(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        // One hidden triangle: the renderer gives an empty mesh no buffer to
        // grow from.
        Mesh3d(meshes.add(overlay_mesh(vec![[0.0; 3]; 3], vec![[0.0; 4]; 3]))),
        MeshMaterial3d(mats.add(StandardMaterial {
            base_color: Color::WHITE,
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            double_sided: true,
            ..default()
        })),
        Visibility::Hidden,
        LaneMesh,
        NoFrustumCulling,
        bevy::light::NotShadowCaster,
    ));
}

/// Flat triangles with a colour per corner.
fn overlay_mesh(positions: Vec<[f32; 3]>, colors: Vec<[f32; 4]>) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    fill_mesh(&mut mesh, positions, colors);
    mesh
}

fn fill_mesh(mesh: &mut Mesh, positions: Vec<[f32; 3]>, colors: Vec<[f32; 4]>) {
    let normals = vec![[0.0, 1.0, 0.0]; positions.len()];
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
}

/// The triangles being drawn this frame, laid on the terrain.
struct Canvas<'a> {
    terrain: &'a crate::terrain::Terrain,
    positions: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
}

impl Canvas<'_> {
    /// A ground point, lifted just over the terrain.
    fn on_ground(&self, at: Vec2) -> Vec3 {
        Vec3::new(at.x, self.terrain.height_at(at.x, at.y) + LIFT, at.y)
    }

    fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, color: [f32; 4]) {
        for at in [a, b, c, a, c, d] {
            self.positions.push(at.to_array());
            self.colors.push(color);
        }
    }

    /// A line from `a` to `b`, LINE wide.
    fn line(&mut self, a: Vec3, b: Vec3, color: [f32; 4]) {
        let along = (b - a).xz().normalize_or_zero();
        let side = Vec3::new(-along.y, 0.0, along.x) * (0.5 * LINE);
        self.quad(a + side, b + side, b - side, a - side, color);
    }
}

#[allow(clippy::too_many_arguments)] // bevy system params
fn draw_overlay(
    overlay: Res<LaneOverlay>,
    units: Res<crate::units::Units>,
    groups: Res<crate::orders::Groups>,
    grid: Res<crate::spatial::SpatialGrid>,
    terrain: Res<crate::terrain::Terrain>,
    fixed_time: Res<Time<Fixed>>,
    cams: Query<&crate::camera::RtsCamera>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut overlays: Query<(&Mesh3d, &mut Visibility), With<LaneMesh>>,
) {
    let (Ok(cam), Ok((mesh, mut visibility))) = (cams.single(), overlays.single_mut()) else {
        return;
    };
    let centre = cam.focus.xz();
    // The living soldiers around the camera's focus, as the grid has them:
    // their index and their position at the tick's start.
    let mut near: Vec<(usize, Vec2)> = Vec::new();
    grid.for_each_candidate(centre, overlay.radius, |o| {
        if (o.meta & crate::spatial::META_DYING) == 0
            && o.xz().distance_squared(centre) < overlay.radius * overlay.radius
        {
            near.push((o.idx as usize, o.xz()));
        }
    });
    let alpha = fixed_time.overstep_fraction();
    let mut canvas = Canvas { terrain: &terrain, positions: Vec::new(), colors: Vec::new() };
    let mut comrades: Vec<Vec2> = Vec::new();
    for (i, at) in near {
        // The grid is a tick old: a death sweep may have moved the rows.
        if i >= units.len() || units.death_t[i] != 0 {
            continue;
        }
        if groups.list[units.group[i] as usize].state.is_broken() {
            continue;
        }
        let reach = TYPES[units.kind[i] as usize].reach;
        let team_bit = (units.team[i] as u32) * crate::spatial::META_TEAM;
        // The nearest enemy in reach and the comrades around him, from the
        // same snapshot the sim's look reads.
        comrades.clear();
        let mut enemy: Option<(f32, Vec2)> = None;
        grid.for_each_candidate(at, reach.max(2.0), |o| {
            if o.idx as usize == i || (o.meta & crate::spatial::META_DYING) != 0 {
                return;
            }
            let to = o.xz() - at;
            if (o.meta & crate::spatial::META_TEAM) == team_bit {
                comrades.push(to);
            } else {
                let d2 = to.length_squared();
                if d2 < reach * reach && enemy.is_none_or(|(best, _)| d2 < best) {
                    enemy = Some((d2, to));
                }
            }
        });
        let Some((d2, to)) = enemy else { continue };
        let enemy_d = d2.sqrt();
        let dir = to / enemy_d.max(1e-3);
        let blocked = comrades.iter().any(|c| in_lane(*c, dir, enemy_d));
        let fill = if blocked { RED } else { GREEN };
        let edge = [fill[0], fill[1], fill[2], LINE_ALPHA];
        // Drawn where the soldier is drawn: between his last two ticks.
        let p = units.pos_prev[i].lerp(units.pos[i], alpha).xz();
        let side = Vec2::new(-dir.y, dir.x) * SPEAR_LINE_HALF_W;
        let far = p + dir * enemy_d;
        let [a, b, c, d] =
            [p + side, far + side, far - side, p - side].map(|at| canvas.on_ground(at));
        canvas.quad(a, b, c, d, fill);
        canvas.line(a, b, edge);
        canvas.line(d, c, edge);
    }
    if canvas.positions.is_empty() {
        *visibility = Visibility::Hidden;
        return;
    }
    *visibility = Visibility::Visible;
    if let Some(mut mesh) = meshes.get_mut(&mesh.0) {
        fill_mesh(&mut mesh, canvas.positions, canvas.colors);
    }
}
