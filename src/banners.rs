//! Regiment banners: a TW-style flag over each regiment showing team,
//! kind (flag shape), strength, and morale at a glance. ~200 plain bevy
//! entities — the no-per-unit-entity rule is about units; render plumbing
//! at regiment count is fine.
//!
//! The whole banner billboards toward the camera and scales with camera
//! distance so it stays readable zoomed out. The morale bar drains as a
//! regiment approaches its break point (the invisible stat becomes
//! gameplay); wavering regiments pulse; broken ones fly a gray flag.

use bevy::prelude::*;

use crate::orders::{Groups, Selection};
use crate::terrain::Terrain;

/// Bar fill width in banner-local units.
const BAR_W: f32 = 1.5;
pub struct BannersPlugin;

impl Plugin for BannersPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            OnEnter(crate::game_state::GameState::Battle),
            spawn_banners.after(crate::game_state::setup_battle),
        )
        .add_systems(
            Update,
            update_banners
                .after(crate::camera::apply_camera_transform)
                .run_if(in_state(crate::game_state::GameState::Battle)),
        );
    }
}

/// Root component; child entity ids for the dynamic parts.
/// Desaturated per-team flag colors for broken regiments. The HUD's
/// card strength fill drains to the same gray (unit_cards.rs).
pub const FLAG_BROKEN: [Color; 2] = [
    Color::srgb(0.45, 0.50, 0.58),
    Color::srgb(0.58, 0.50, 0.42),
];

#[derive(Component)]
struct Banner {
    group: usize,
    flag: Entity,
    morale_fill: Entity,
    strength_fill: Entity,
    sel_marker: Entity,
}

#[derive(Resource)]
struct BannerAssets {
    flag_mats: [Handle<StandardMaterial>; 2],
    flag_mats_broken: [Handle<StandardMaterial>; 2],
}

fn spawn_banners(
    mut commands: Commands,
    groups: Res<Groups>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
) {
    let unlit = |c: Color| StandardMaterial {
        base_color: c,
        unlit: true,
        ..default()
    };
    let flag_mats = [
        mats.add(unlit(Color::srgb(0.25, 0.50, 0.95))),
        mats.add(unlit(Color::srgb(0.95, 0.45, 0.12))),
    ];
    let flag_mats_broken = FLAG_BROKEN.map(|c| mats.add(unlit(c)));
    let pole_mat = mats.add(unlit(Color::srgb(0.24, 0.19, 0.14)));
    let back_mat = mats.add(unlit(Color::srgb(0.07, 0.07, 0.07)));
    let morale_mat = mats.add(unlit(Color::srgb(0.90, 0.16, 0.10)));
    let strength_mat = mats.add(unlit(Color::srgb(0.95, 0.85, 0.30)));
    let sel_mat = mats.add(unlit(Color::WHITE));

    let pole_mesh = meshes.add(Cuboid::new(0.09, 4.2, 0.09));
    // Flag shape encodes kind (indexed by `GroupData::kind` — keep in
    // step with unit_types::NUM_KINDS): heavies fly a square standard,
    // lights a long thin pennant, spears a mid-size banderole, archers
    // a small swallowtail-narrow guidon.
    let flag_meshes = [
        meshes.add(Cuboid::new(1.5, 1.0, 0.05)),
        meshes.add(Cuboid::new(1.8, 0.5, 0.05)),
        meshes.add(Cuboid::new(1.2, 0.75, 0.05)),
        meshes.add(Cuboid::new(0.9, 0.55, 0.05)),
    ];
    let unit_cube = meshes.add(Cuboid::new(1.0, 1.0, 1.0));
    let sel_mesh = meshes.add(Cuboid::new(0.4, 0.4, 0.4));

    for (g, gd) in groups.list.iter().enumerate() {
        let flag = commands
            .spawn((
                Mesh3d(flag_meshes[gd.kind as usize].clone()),
                MeshMaterial3d(flag_mats[gd.team as usize].clone()),
                Transform::from_xyz(0.85, 3.55, 0.0),
            ))
            .id();
        let pole = commands
            .spawn((
                Mesh3d(pole_mesh.clone()),
                MeshMaterial3d(pole_mat.clone()),
                Transform::from_xyz(0.0, 2.1, 0.0),
            ))
            .id();
        let mut bar = |y: f32, z: f32, w: f32, h: f32, mat: &Handle<StandardMaterial>| {
            commands
                .spawn((
                    Mesh3d(unit_cube.clone()),
                    MeshMaterial3d(mat.clone()),
                    Transform::from_xyz(0.0, y, z).with_scale(Vec3::new(w, h, 0.04)),
                ))
                .id()
        };
        let morale_back = bar(4.62, -0.02, BAR_W + 0.08, 0.20, &back_mat);
        let morale_fill = bar(4.62, 0.02, BAR_W, 0.14, &morale_mat);
        let strength_back = bar(4.38, -0.02, BAR_W + 0.08, 0.20, &back_mat);
        let strength_fill = bar(4.38, 0.02, BAR_W, 0.14, &strength_mat);
        let sel_marker = commands
            .spawn((
                Mesh3d(sel_mesh.clone()),
                MeshMaterial3d(sel_mat.clone()),
                Transform::from_xyz(0.0, 5.15, 0.0).with_rotation(Quat::from_rotation_z(
                    std::f32::consts::FRAC_PI_4,
                )),
                Visibility::Hidden,
            ))
            .id();
        commands
            .spawn((
                Transform::default(),
                Visibility::default(),
                DespawnOnExit(crate::game_state::GameState::Battle),
                Banner {
                    group: g,
                    flag,
                    morale_fill,
                    strength_fill,
                    sel_marker,
                },
            ))
            .add_children(&[
                pole,
                flag,
                morale_back,
                morale_fill,
                strength_back,
                strength_fill,
                sel_marker,
            ]);
    }
    commands.insert_resource(BannerAssets {
        flag_mats,
        flag_mats_broken,
    });
}

#[allow(clippy::too_many_arguments)] // bevy system params
fn update_banners(
    groups: Res<Groups>,
    selection: Res<Selection>,
    terrain: Res<Terrain>,
    viz: Res<crate::movement::DebugViz>,
    camera: Query<&crate::camera::RtsCamera>,
    time: Res<Time>,
    assets: Option<Res<BannerAssets>>,
    mut roots: Query<(&Banner, &mut Transform, &mut Visibility)>,
    mut parts: Query<(&mut Transform, &mut Visibility), Without<Banner>>,
    mut part_mats: Query<&mut MeshMaterial3d<StandardMaterial>>,
) {
    let Ok(cam) = camera.single() else { return };
    let Some(assets) = assets else { return };
    let billboard = Quat::from_rotation_y(cam.yaw);
    let dist_scale = (cam.distance * 0.013).clamp(0.8, 3.2);

    for (banner, mut tf, mut vis) in &mut roots {
        let Some(gd) = groups.list.get(banner.group) else {
            *vis = Visibility::Hidden;
            continue;
        };
        if gd.count == 0 || !viz.0 {
            *vis = Visibility::Hidden;
            continue;
        }
        *vis = Visibility::Visible;
        let broken = gd.state.is_broken();

        tf.translation = Vec3::new(
            gd.centroid.x,
            terrain.height_at(gd.centroid.x, gd.centroid.y) + 0.6,
            gd.centroid.y,
        );
        tf.rotation = billboard;
        let mut s = dist_scale;
        if !broken && crate::morale::band(gd) == crate::morale::Band::Wavering {
            // About to break: the whole banner trembles.
            s *= 1.0 + 0.07 * (time.elapsed_secs() * 7.0).sin();
        }
        tf.scale = Vec3::splat(s);

        // Left-anchored bar fills.
        let morale = crate::morale::morale01(gd);
        if let Ok((mut t, _)) = parts.get_mut(banner.morale_fill) {
            t.scale.x = (BAR_W * morale).max(0.001);
            t.translation.x = -BAR_W * (1.0 - morale) / 2.0;
        }
        let strength = (gd.count as f32 / gd.initial_count.max(1) as f32).clamp(0.0, 1.0);
        if let Ok((mut t, _)) = parts.get_mut(banner.strength_fill) {
            t.scale.x = (BAR_W * strength).max(0.001);
            t.translation.x = -BAR_W * (1.0 - strength) / 2.0;
        }

        if let Ok((_, mut v)) = parts.get_mut(banner.sel_marker) {
            *v = if selection
                .regiments
                .get(banner.group)
                .copied()
                .unwrap_or(false)
            {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
        }

        // Broken regiments fly a gray flag.
        if let Ok(mut mat) = part_mats.get_mut(banner.flag) {
            let want = if broken {
                &assets.flag_mats_broken[gd.team as usize]
            } else {
                &assets.flag_mats[gd.team as usize]
            };
            if mat.0 != *want {
                mat.0 = want.clone();
            }
        }
    }
}
