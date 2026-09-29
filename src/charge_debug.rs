//! F3 view of the charge state, kept while the charge yells are tuned: a
//! ring on the ground around every regiment in the charge state (amber,
//! red once it has charged STUCK_S without reaching the enemy), a label
//! with its time in the charge and its distance to the target, a panel
//! listing the charges, and a dot on every soldier voicing a charge yell
//! right now.

use bevy::picking::Pickable;
use bevy::prelude::*;

use crate::camera::RtsCamera;
use crate::game_state::GameState;
use crate::orders::{Groups, Order};

pub struct ChargeDebugPlugin;

impl Plugin for ChargeDebugPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_charge_debug)
            .add_systems(Update, update_charge_debug.run_if(in_state(GameState::Battle)))
            .add_systems(OnExit(GameState::Battle), hide_charge_debug);
    }
}

/// Labels and yell dots on screen at most.
const LABELS: usize = 24;
const DOTS: usize = 64;
/// A charge this long without contact counts as stuck (s).
const STUCK_S: f32 = 5.0;

const PANEL_BG: Color = Color::srgba(0.04, 0.05, 0.07, 0.85);
const TEXT: Color = Color::srgb(0.92, 0.9, 0.82);
const CHARGING: Color = Color::srgb(1.0, 0.72, 0.25);
const STUCK: Color = Color::srgb(1.0, 0.3, 0.25);
const DOT: Color = Color::srgba(1.0, 0.85, 0.35, 0.85);

#[derive(Component)]
struct ChargePanel;
#[derive(Component)]
struct ChargePanelText;
#[derive(Component)]
struct ChargeLabel(usize);
#[derive(Component)]
struct YellDot;

type PanelVis<'w, 's> =
    Query<'w, 's, &'static mut Visibility, (With<ChargePanel>, Without<ChargeLabel>, Without<YellDot>)>;
type Labels<'w, 's> = Query<
    'w,
    's,
    (&'static ChargeLabel, &'static mut Text, &'static mut TextColor, &'static mut Node, &'static mut Visibility),
    (Without<ChargePanel>, Without<YellDot>),
>;
type Dots<'w, 's> =
    Query<'w, 's, (&'static mut Node, &'static mut Visibility), (With<YellDot>, Without<ChargePanel>, Without<ChargeLabel>)>;
type AllViews<'w, 's> =
    Query<'w, 's, &'static mut Visibility, Or<(With<ChargePanel>, With<ChargeLabel>, With<YellDot>)>>;

fn spawn_charge_debug(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                // Below the balance bar and the debug text.
                top: Val::Px(140.0),
                right: Val::Px(8.0),
                padding: UiRect::all(Val::Px(8.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                ..default()
            },
            BackgroundColor(PANEL_BG),
            Visibility::Hidden,
            GlobalZIndex(4),
            Pickable::IGNORE,
            ChargePanel,
        ))
        .with_children(|p| {
            p.spawn((
                Text::new(""),
                TextFont {
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(TEXT),
                Pickable::IGNORE,
                ChargePanelText,
            ));
        });
    for i in 0..LABELS {
        commands.spawn((
            Text::new(""),
            TextFont {
                font_size: FontSize::Px(13.0),
                ..default()
            },
            TextColor(CHARGING),
            Node {
                position_type: PositionType::Absolute,
                padding: UiRect::axes(Val::Px(4.0), Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                ..default()
            },
            BackgroundColor(PANEL_BG),
            Visibility::Hidden,
            GlobalZIndex(4),
            Pickable::IGNORE,
            ChargeLabel(i),
        ));
    }
    for _ in 0..DOTS {
        commands.spawn((
            Node {
                position_type: PositionType::Absolute,
                width: Val::Px(7.0),
                height: Val::Px(7.0),
                border_radius: BorderRadius::MAX,
                ..default()
            },
            BackgroundColor(DOT),
            Visibility::Hidden,
            GlobalZIndex(4),
            Pickable::IGNORE,
            YellDot,
        ));
    }
}

fn hide_charge_debug(mut vis: AllViews) {
    for mut v in &mut vis {
        *v = Visibility::Hidden;
    }
}

#[allow(clippy::too_many_arguments)] // bevy system params
fn update_charge_debug(
    settings: Res<crate::settings::Settings>,
    groups: Res<Groups>,
    terrain: Res<crate::terrain::Terrain>,
    mixer: Res<crate::mixer::Mixer>,
    time: Res<Time>,
    camera: Query<(&Camera, &GlobalTransform), With<RtsCamera>>,
    mut gizmos: Gizmos,
    mut panel: PanelVis,
    mut panel_text: Query<&mut Text, (With<ChargePanelText>, Without<ChargeLabel>)>,
    mut labels: Labels,
    mut dots: Dots,
    mut in_charge: Local<Vec<f32>>,
) {
    // Seconds each regiment has been in the charge state (kept even while
    // the view is off, so turning it on shows true durations).
    in_charge.resize(groups.list.len(), 0.0);
    for (g, gd) in groups.list.iter().enumerate() {
        in_charge[g] = if gd.charging && gd.count > 0 { in_charge[g] + time.delta_secs() } else { 0.0 };
    }

    let on = settings.interface.debug_overlay;
    let show = if on { Visibility::Inherited } else { Visibility::Hidden };
    for mut v in &mut panel {
        *v = show;
    }
    let Ok((cam, cam_tf)) = camera.single() else { return };
    if !on {
        for (_, _, _, _, mut v) in &mut labels {
            *v = Visibility::Hidden;
        }
        for (_, mut v) in &mut dots {
            *v = Visibility::Hidden;
        }
        return;
    }

    let ground = |p: Vec2, up: f32| Vec3::new(p.x, terrain.height_at(p.x, p.y) + up, p.y);
    // The charges, longest first.
    let mut charges: Vec<usize> = (0..groups.list.len()).filter(|&g| in_charge[g] > 0.0).collect();
    charges.sort_by(|a, b| in_charge[*b].total_cmp(&in_charge[*a]));

    let target_dist = |g: usize| -> Option<f32> {
        match groups.list[g].order {
            Some(Order::Attack(t)) => groups.list.get(t as usize).map(|tg| tg.centroid.distance(groups.list[g].centroid)),
            _ => None,
        }
    };
    let describe = |g: usize| -> String {
        let gd = &groups.list[g];
        let team = if gd.team == 0 { "own" } else { "enemy" };
        let to = target_dist(g).map_or(String::new(), |d| format!(", {d:.0} m to target"));
        format!("g{g} {team} {}: {:.1} s in charge{to}", crate::unit_types::kind_name(gd.kind), in_charge[g])
    };

    // Rings on the ground, and a line to the target.
    for &g in &charges {
        let gd = &groups.list[g];
        let color = if in_charge[g] >= STUCK_S { STUCK } else { CHARGING };
        let iso = Isometry3d::new(ground(gd.centroid, 0.4), Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2));
        gizmos.circle(iso, gd.radius.max(4.0), color);
        if let Some(Order::Attack(t)) = gd.order
            && let Some(tg) = groups.list.get(t as usize)
        {
            gizmos.line(ground(gd.centroid, 0.4), ground(tg.centroid, 0.4), color.with_alpha(0.5));
        }
    }

    // Labels above the rings, longest charges first.
    let mut shown = 0;
    for (label, mut text, mut color, mut node, mut vis) in &mut labels {
        let Some(&g) = charges.get(label.0) else {
            *vis = Visibility::Hidden;
            continue;
        };
        let at = ground(groups.list[g].centroid, 6.0);
        match cam.world_to_viewport(cam_tf, at) {
            Ok(p) => {
                *text = Text::new(describe(g));
                color.0 = if in_charge[g] >= STUCK_S { STUCK } else { CHARGING };
                node.left = Val::Px(p.x);
                node.top = Val::Px(p.y);
                *vis = Visibility::Inherited;
                shown += 1;
            }
            Err(_) => *vis = Visibility::Hidden,
        }
    }

    // A dot on every soldier voicing a charge yell.
    let yells = mixer.live_positions("charge yell");
    let mut dot_positions = yells.iter().filter_map(|p| cam.world_to_viewport(cam_tf, *p).ok());
    for (mut node, mut vis) in &mut dots {
        match dot_positions.next() {
            Some(p) => {
                node.left = Val::Px(p.x - 3.5);
                node.top = Val::Px(p.y - 3.5);
                *vis = Visibility::Inherited;
            }
            None => *vis = Visibility::Hidden,
        }
    }

    let stuck = charges.iter().filter(|&&g| in_charge[g] >= STUCK_S).count();
    let mut lines = vec![format!(
        "Charges: {} ({stuck} over {STUCK_S:.0} s without contact), {} yells playing, {} labels on screen",
        charges.len(),
        yells.len(),
        shown
    )];
    lines.extend(charges.iter().take(12).map(|&g| describe(g)));
    if let Ok(mut t) = panel_text.single_mut() {
        *t = Text::new(lines.join("\n"));
    }
}
