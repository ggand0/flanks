//! FL_DEBUG_ARC=1: the room each soldier near the camera has to strike,
//! drawn on the ground under him. A sword kind shows the arc of his cut
//! (sim/soldier.rs `in_cut_arc`), a spearman the lane of his thrust to
//! the enemy in reach (`in_thrust_lane`). Grey: no enemy in reach. Green:
//! an enemy in reach and nothing in the way. Red: an enemy in reach and a
//! comrade in the arc or lane, so he does not strike. FL_DEBUG_ARC=<m>
//! sets how far from the camera's focus soldiers are drawn.
//!
//! The overlay runs the same tests as the sim on every frame. The sim
//! acts on a man's last look, up to a quarter second old.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::math::Vec3Swizzles;
use bevy::mesh::PrimitiveTopology;
use bevy::prelude::*;

use crate::sim::soldier::{ARC_HALF_COS, ARC_REACH, SPEAR_LINE_HALF_W, in_cut_arc, in_thrust_lane};
use crate::unit_types::{TYPES, Weapon};

/// Soldiers this close to the camera's focus are drawn (metres).
const RADIUS_DEFAULT: f32 = 30.0;
/// Triangles per arc.
const SEGMENTS: usize = 12;
/// Height of the overlay over the ground.
const LIFT: f32 = 0.06;

/// Width of the line around each shape (metres).
const LINE: f32 = 0.035;

/// Fill colours; the line around a shape is the same colour, `LINE_ALPHA`
/// strong.
const GREY: [f32; 4] = [0.90, 0.90, 0.90, 0.05];
const GREEN: [f32; 4] = [0.15, 0.95, 0.30, 0.28];
const RED: [f32; 4] = [1.00, 0.12, 0.08, 0.38];
const LINE_ALPHA: [f32; 3] = [0.30, 0.85, 0.90];

#[derive(Resource)]
struct ArcOverlay {
    radius: f32,
}

#[derive(Component)]
struct ArcMesh;

pub struct ArcOverlayPlugin;

impl Plugin for ArcOverlayPlugin {
    fn build(&self, app: &mut App) {
        let Ok(value) = std::env::var("FL_DEBUG_ARC") else {
            return;
        };
        let radius = value.parse::<f32>().ok().filter(|r| *r > 1.0).unwrap_or(RADIUS_DEFAULT);
        app.insert_resource(ArcOverlay { radius })
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
        ArcMesh,
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

    fn triangle(&mut self, corners: [Vec3; 3], color: [f32; 4]) {
        for at in corners {
            self.positions.push(at.to_array());
            self.colors.push(color);
        }
    }

    fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, color: [f32; 4]) {
        self.triangle([a, b, c], color);
        self.triangle([a, c, d], color);
    }

    /// A line from `a` to `b`, LINE wide.
    fn line(&mut self, a: Vec3, b: Vec3, color: [f32; 4]) {
        let along = (b - a).xz().normalize_or_zero();
        let side = Vec3::new(-along.y, 0.0, along.x) * (0.5 * LINE);
        self.quad(a + side, b + side, b - side, a - side, color);
    }
}

/// The colour of the line around a shape filled with `fill`.
fn line_color(fill: [f32; 4]) -> [f32; 4] {
    let alpha = if fill == GREY {
        LINE_ALPHA[0]
    } else if fill == GREEN {
        LINE_ALPHA[1]
    } else {
        LINE_ALPHA[2]
    };
    [fill[0], fill[1], fill[2], alpha]
}

#[allow(clippy::too_many_arguments)] // bevy system params
fn draw_overlay(
    overlay: Res<ArcOverlay>,
    units: Res<crate::units::Units>,
    groups: Res<crate::orders::Groups>,
    grid: Res<crate::spatial::SpatialGrid>,
    terrain: Res<crate::terrain::Terrain>,
    fixed_time: Res<Time<Fixed>>,
    cams: Query<&crate::camera::RtsCamera>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut overlays: Query<(&Mesh3d, &mut Visibility), With<ArcMesh>>,
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
    let half_angle = ARC_HALF_COS.acos();
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
        let params = &TYPES[units.kind[i] as usize];
        let team_bit = (units.team[i] as u32) * crate::spatial::META_TEAM;
        // The nearest enemy in reach and the comrades around him, from the
        // same snapshot the sim's look reads.
        comrades.clear();
        let mut enemy: Option<(f32, Vec2)> = None;
        grid.for_each_candidate(at, params.reach.max(2.0), |o| {
            if o.idx as usize == i || (o.meta & crate::spatial::META_DYING) != 0 {
                return;
            }
            let to = o.xz() - at;
            if (o.meta & crate::spatial::META_TEAM) == team_bit {
                comrades.push(to);
            } else {
                let d2 = to.length_squared();
                if d2 < params.reach * params.reach && enemy.is_none_or(|(best, _)| d2 < best) {
                    enemy = Some((d2, to));
                }
            }
        });
        // Drawn where the soldier is drawn: between his last two ticks.
        let p = units.pos_prev[i].lerp(units.pos[i], alpha).xz();
        let turn = (units.yaw[i] - units.yaw_prev[i] + std::f32::consts::PI)
            .rem_euclid(std::f32::consts::TAU)
            - std::f32::consts::PI;
        let yaw = units.yaw_prev[i] + turn * alpha;
        if params.weapon == Weapon::Spear {
            let Some((d2, to)) = enemy else { continue };
            let enemy_d = d2.sqrt();
            let dir = to / enemy_d.max(1e-3);
            let blocked = comrades.iter().any(|c| in_thrust_lane(*c, dir, enemy_d));
            let color = if blocked { RED } else { GREEN };
            let side = Vec2::new(-dir.y, dir.x) * SPEAR_LINE_HALF_W;
            let far = p + dir * enemy_d;
            let [a, b, c, d] =
                [p + side, far + side, far - side, p - side].map(|at| canvas.on_ground(at));
            canvas.quad(a, b, c, d, color);
            canvas.line(a, b, line_color(color));
            canvas.line(d, c, line_color(color));
        } else {
            let fwd = Vec2::new(units.yaw[i].sin(), units.yaw[i].cos());
            let blocked = comrades.iter().any(|c| in_cut_arc(*c, fwd));
            let color = match (enemy.is_some(), blocked) {
                (false, _) => GREY,
                (true, false) => GREEN,
                (true, true) => RED,
            };
            let rim: [Vec3; SEGMENTS + 1] = std::array::from_fn(|k| {
                let a = yaw - half_angle + 2.0 * half_angle * k as f32 / SEGMENTS as f32;
                canvas.on_ground(p + Vec2::new(a.sin(), a.cos()) * ARC_REACH)
            });
            let centre = canvas.on_ground(p);
            let edge = line_color(color);
            for k in 0..SEGMENTS {
                canvas.triangle([centre, rim[k], rim[k + 1]], color);
                canvas.line(rim[k], rim[k + 1], edge);
            }
            canvas.line(centre, rim[0], edge);
            canvas.line(centre, rim[SEGMENTS], edge);
        }
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
