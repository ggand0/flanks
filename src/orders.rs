//! Regiments (groups) and player orders. Selection/hover/control groups
//! live in selection.rs (re-exported here — consumers keep their paths).
//!
//! A group IS a regiment: a permanent block of units sharing an order,
//! kind, formation, and morale. Right-click CLICK: attack-move — each
//! selected regiment's ORDER is the target translated by its offset from
//! the selection centroid, so a group move preserves the army's
//! arrangement. Right-click DRAG (M2TW): paint the new front line on the
//! ground — regiments divide the drawn width, set their files to fill it,
//! and face perpendicular to it, toward the enemy army's side; soft slot
//! markers preview the placement until release. Units translate the
//! regiment order by their own `home` offset (block moves, never
//! converges).

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::formation::{FormShape, facing_dir, facing_of, slot_offsets};
use crate::selection::cursor_ground_point;
pub use crate::selection::{Hover, Selection, select_along_line};
use crate::terrain::Terrain;
use crate::units::Units;

pub const PLAYER_TEAM: u8 = 0;
/// Groups whose centroid gets this close to their order target go idle.
const ARRIVE_CLEAR: f32 = 18.0;
/// Right-clicking within this range of an enemy regiment's centroid is an
/// attack order on that regiment (else it's a move order to the point).
pub const ATTACK_PICK_RADIUS: f32 = 20.0;
/// RMB travel on the ground beyond this is a formation drag, not a click.
const ORDER_DRAG_MIN: f32 = 5.0;
/// Gap left between regiments sharing a drawn line.
const LINE_REG_GAP: f32 = 3.0;
/// Per-soldier preview circles are drawn up to this many total units;
/// bigger selections preview as block outlines (gizmo budget).
const PREVIEW_CIRCLE_CAP: usize = 3000;

/// A regiment order, TW-style: move orders target ground and end on
/// arrival; attack orders hunt a specific enemy regiment (the destination
/// re-resolves to its centroid every tick) and end when it's wiped.
#[derive(Clone, Copy, PartialEq)]
pub enum Order {
    Move(Vec2),
    /// Index into `Groups::list`.
    Attack(u32),
}

/// Regiment morale state.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RegState {
    Steady,
    /// Broken: uncontrollable, flees toward its own map edge. `since` is
    /// the morale-system tick the break happened on (rally timing).
    Routing { since: u32 },
    /// Too depleted to ever rally; flees until despawn.
    Shattered,
}

impl RegState {
    #[inline]
    pub fn is_broken(&self) -> bool {
        !matches!(self, RegState::Steady)
    }
}

pub struct GroupData {
    pub team: u8,
    /// Regiments are homogeneous; index into `unit_types::TYPES`.
    pub kind: u8,
    /// Current order; None = hold at anchor.
    pub order: Option<Order>,
    /// Set when the current order was issued by at-ease auto-engagement
    /// or a script, not the player: UI click feedback stays silent.
    pub auto_order: bool,
    /// Regiment reference point: units hold at `anchor + home`. Updated to
    /// the order target on arrival.
    pub anchor: Vec2,
    pub count: usize,
    /// Strength at spawn (casualty fraction for morale).
    pub initial_count: usize,
    // --- formation (formation.rs consumes; slots land in `units.home`) ---
    pub shape: crate::formation::FormShape,
    pub spacing: crate::formation::FormSpacing,
    /// Formation facing (unit yaw convention, 0 = +Z). Baked into the slot
    /// offsets on reform; meaningless for Blob.
    pub facing: f32,
    /// Grid width in files (soldiers per rank). 0 = auto on first reform.
    pub files: u32,
    /// Hold position (defend mode): never auto-engage, never chase — at
    /// ease (false, default) idle regiments engage enemies that come near.
    pub hold: bool,
    /// Request: rewrite this regiment's slots before the next sim tick.
    pub reform: bool,
    /// Living strength when slots were last written (close-ranks trigger).
    pub count_at_reform: usize,
    /// Mean distance of living units to their slot (m), smoothed ~2 s
    /// (update_groups). A formation fighting in disarray bleeds morale;
    /// zero for Blob (a mob makes no discipline claim).
    pub disorder: f32,
    /// Mean `home` of living units last tick (update_groups): casualties
    /// skew the living slots' center away from the anchor, and disorder
    /// must not count that skew as chaos.
    pub home_bias: Vec2,
    // --- refreshed every fixed tick by the frontline pass ---
    pub centroid: Vec2,
    /// TW rule: one soldier of the regiment fighting = the whole regiment
    /// is engaged (ground truth from per-unit melee state, with a short
    /// hold to bridge swing-cycle gaps).
    pub engaged: bool,
    /// Ticks of `engaged` left since the last soldier fought.
    pub engage_hold: u8,
    /// FL_RECTFIGHT: enough of this regiment's soldiers are in a swing
    /// cycle against its ORDERED attack target's men (the engine keeps
    /// per-enemy-unit engagedSoldiers counts and gates engagement on
    /// "enough soldiers in the proximity zone"). THIS is what freezes
    /// the attack path — a trickle of overflow duels never halts the
    /// march; the poked men defend individually while the block keeps
    /// walking to its ordered fight.
    pub engaged_with_target: bool,
    /// Ticks of `engaged_with_target` left (same swing-gap bridge).
    pub engage_target_hold: u8,
    /// In the charge phase: attack order, inside charge range of the
    /// target, not yet in contact. Drives the war cry + sprint pose.
    pub charging: bool,
    /// An enemy regiment's centroid is within combat-watch range: units
    /// of this regiment scan wider for adjacent enemies (sparse-fight
    /// acquisition, movement.rs) and brace when standing.
    pub enemy_near: bool,
    /// Normalized direction to the nearest enemy regiment (ZERO when
    /// none in watch range): standing units face it (brace facing).
    pub threat_dir: Vec2,
    /// An UNBROKEN enemy regiment is within watch range. The falling
    /// edge (last nearby foe routed or wiped) triggers the cheer.
    pub hostile_near: bool,
    /// Victory-cheer ticks remaining (render-only celebration).
    pub celebrate: u16,
    // --- morale (morale.rs updates per tick) ---
    pub morale: f32,
    pub state: RegState,
    /// Deaths since the last morale tick (tallied by the damage apply pass).
    pub recent_deaths: u32,
}

impl GroupData {
    pub fn new(team: u8, kind: u8, anchor: Vec2, count: usize) -> Self {
        Self {
            team,
            kind,
            order: None,
            auto_order: false,
            anchor,
            count,
            initial_count: count,
            // Blob preserves hand-built spawn geometry (tests); the battle
            // spawn switches to Rect and forms slots explicitly.
            shape: crate::formation::FormShape::Blob,
            spacing: crate::formation::FormSpacing::Normal,
            facing: if team == 0 { 0.0 } else { std::f32::consts::PI },
            files: 0,
            hold: false,
            reform: false,
            count_at_reform: count,
            disorder: 0.0,
            home_bias: Vec2::ZERO,
            centroid: anchor,
            engaged: false,
            engage_hold: 0,
            engaged_with_target: false,
            engage_target_hold: 0,
            charging: false,
            enemy_near: false,
            threat_dir: Vec2::ZERO,
            hostile_near: false,
            celebrate: 0,
            morale: 100.0,
            state: RegState::Steady,
            recent_deaths: 0,
        }
    }
}

#[derive(Resource, Default)]
pub struct Groups {
    pub list: Vec<GroupData>,
}

impl Groups {
    /// A group's order destination this tick: Move goes to the point,
    /// Attack chases the target regiment's current centroid.
    pub fn goal(&self, g: usize) -> Option<Vec2> {
        match self.list[g].order? {
            Order::Move(p) => Some(p),
            Order::Attack(t) => {
                let t = &self.list[t as usize];
                (t.count > 0).then_some(t.centroid)
            }
        }
    }
}

/// One regiment's placement on a drawn order line.
struct LinePlacement {
    g: usize,
    /// Block CENTER (the Move target; the front rank sits on the line).
    anchor: Vec2,
    facing: f32,
    files: u32,
}

/// RMB press/drag state for order issuing.
#[derive(Resource, Default)]
struct OrderDrag {
    /// Ground point at RMB press (None = no press in flight).
    start: Option<Vec2>,
    cur: Vec2,
    /// Travel exceeded ORDER_DRAG_MIN: this is a formation drag.
    active: bool,
    layout: Vec<LinePlacement>,
}

pub struct OrdersPlugin;

impl Plugin for OrdersPlugin {
    fn build(&self, app: &mut App) {
        // Groups start empty; the regiment spawn (regiments.rs) fills the
        // list once at startup and it stays FIXED for the whole battle —
        // stable indices are what make `units.group` a permanent regiment id.
        app.init_resource::<Groups>()
            .init_resource::<OrderDrag>()
            .add_systems(
                Update,
                (
                    (issue_order, draw_order_preview).chain(),
                    halt_key,
                    test_orders_script,
                    draw_order_gizmos,
                ),
            )
            .add_systems(FixedUpdate, clear_arrived_orders.after(crate::movement::step_sim));
    }
}

/// Attack-move `selected` regiments so the formation ARRANGEMENT is
/// preserved: each regiment's order is the target translated by its offset
/// from the selection centroid. Shared by the RMB click handler, test
/// scripts, and (later) the AI.
pub fn order_regiments(groups: &mut Groups, selected: &[usize], target: Vec2) {
    // Broken regiments are uncontrollable.
    let selected: Vec<usize> = selected
        .iter()
        .copied()
        .filter(|&g| !groups.list[g].state.is_broken())
        .collect();
    if selected.is_empty() {
        return;
    }
    let centroid: Vec2 =
        selected.iter().map(|&g| groups.list[g].centroid).sum::<Vec2>() / selected.len() as f32;
    for &g in &selected {
        let group = &mut groups.list[g];
        let dest = target + (group.centroid - centroid);
        group.order = Some(Order::Move(dest));
        group.auto_order = false;
        // March facing the way we're going: the front rank leads.
        let dir = dest - group.centroid;
        if group.shape == FormShape::Rect && dir.length() > 1.0 {
            group.facing = facing_of(dir);
            group.reform = true;
        }
    }
}

/// Attack-order `selected` regiments at one enemy regiment: everyone hunts
/// the same target (TW behavior — no arrangement translation).
pub fn attack_regiments(groups: &mut Groups, selected: &[usize], target: u32) {
    let target_c = groups.list[target as usize].centroid;
    for &g in selected {
        let group = &mut groups.list[g];
        if group.state.is_broken() {
            continue;
        }
        group.order = Some(Order::Attack(target));
        group.auto_order = false;
        let dir = target_c - group.centroid;
        if group.shape == FormShape::Rect && dir.length() > 1.0 {
            group.facing = facing_of(dir);
            group.reform = true;
        }
    }
}

fn enemy_regiment_at(groups: &Groups, p: Vec2) -> Option<u32> {
    crate::selection::regiment_at(groups, p, true)
}

/// Lay `selected` regiments out along the ground line a -> b: the drawn
/// width is divided by strength, files fill each share, and every block
/// faces perpendicular to the line, toward the ENEMY army's side of it —
/// a drawn battle line faces the enemy no matter which hand drew it.
/// Fallback when there is no enemy signal (none left alive, or the enemy
/// is dead-parallel to the line): face away from where the selection
/// stands (troops walk up to the line and face past it). Regiments keep
/// their left-to-right order along the line to minimize crossing.
fn line_layout(groups: &Groups, selected: &[usize], a: Vec2, b: Vec2) -> Vec<LinePlacement> {
    let picked: Vec<usize> = selected
        .iter()
        .copied()
        .filter(|&g| {
            let gd = &groups.list[g];
            gd.count > 0 && !gd.state.is_broken()
        })
        .collect();
    if picked.is_empty() {
        return Vec::new();
    }
    let len = a.distance(b);
    let dir = (b - a) / len.max(1e-3);
    let mean: Vec2 =
        picked.iter().map(|&g| groups.list[g].centroid).sum::<Vec2>() / picked.len() as f32;
    let perp = Vec2::new(-dir.y, dir.x);
    let mid = (a + b) * 0.5;
    // Which side of the line does the enemy stand on? Strength-weighted
    // enemy centroid so a routed straggler can't flip the line. The old
    // rule (away from the selection) broke on the common gesture of
    // dressing a line in place: with the line ON the selection the side
    // signal was noise, and the tie fell to the raw drag perpendicular —
    // facing followed the drag hand, backward for half of all drags.
    let team = groups.list[picked[0]].team;
    let mut enemy_sum = Vec2::ZERO;
    let mut enemy_weight = 0.0f32;
    for gd in &groups.list {
        if gd.team != team && gd.count > 0 && !gd.state.is_broken() {
            enemy_sum += gd.centroid * gd.count as f32;
            enemy_weight += gd.count as f32;
        }
    }
    let enemy_side = if enemy_weight > 0.0 {
        perp.dot(enemy_sum / enemy_weight - mid)
    } else {
        0.0
    };
    let side = if enemy_side.abs() > 1.0 {
        enemy_side
    } else {
        perp.dot(mid - mean)
    };
    let fwd = if side >= 0.0 { perp } else { -perp };
    let facing = facing_of(fwd);

    // Keep current left-to-right order along the line.
    let mut segs: Vec<(usize, f32)> = picked
        .iter()
        .map(|&g| (g, (groups.list[g].centroid - a).dot(dir)))
        .collect();
    segs.sort_unstable_by(|x, y| x.1.total_cmp(&y.1));

    let total: usize = picked.iter().map(|&g| groups.list[g].count).sum();
    let mut placements = Vec::with_capacity(segs.len());
    let mut off = 0.0;
    for (g, _) in segs {
        let gd = &groups.list[g];
        let share = len * gd.count as f32 / total as f32;
        let pitch = gd.spacing.pitch();
        let files = (((share - LINE_REG_GAP) / pitch.x).floor() as u32)
            .clamp(1, gd.count as u32);
        let ranks = (gd.count as u32).div_ceil(files);
        // Front rank ON the drawn line; the block center sits behind it.
        let line_pos = a + dir * (off + share * 0.5);
        let anchor = line_pos - fwd * ((ranks - 1) as f32 * 0.5 * pitch.y);
        placements.push(LinePlacement { g, anchor, facing, files });
        off += share;
    }
    placements
}

fn apply_line_order(groups: &mut Groups, layout: Vec<LinePlacement>, a: Vec2, b: Vec2) {
    let n = layout.len();
    let facing = layout.first().map(|p| p.facing).unwrap_or(0.0);
    for p in layout {
        let gd = &mut groups.list[p.g];
        gd.facing = p.facing;
        gd.files = p.files;
        gd.shape = FormShape::Rect;
        gd.order = Some(Order::Move(p.anchor));
        gd.auto_order = false;
        gd.reform = true;
    }
    info!(
        "line order: {n} regiments across {:.0} m, facing {facing:.2} rad",
        a.distance(b),
    );
}

/// Drive the full drag-order path programmatically (the FL_TEST_FORM
/// script uses this — one code path for the mouse and the test).
pub fn line_order(groups: &mut Groups, selected: &[usize], a: Vec2, b: Vec2) {
    let layout = line_layout(groups, selected, a, b);
    if !layout.is_empty() {
        apply_line_order(groups, layout, a, b);
    }
}

/// RMB order issuing: press -> (optional drag with live preview) ->
/// release. A short click keeps the classic behavior (attack the regiment
/// under the cursor, else arrangement-preserving group move); a drag
/// beyond ORDER_DRAG_MIN paints the new front line.
#[allow(clippy::too_many_arguments)] // bevy system params
fn issue_order(
    buttons: Res<ButtonInput<MouseButton>>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform)>,
    terrain: Res<Terrain>,
    mut groups: ResMut<Groups>,
    selection: Res<Selection>,
    mut drag: ResMut<OrderDrag>,
) {
    if selection.count_units == 0 {
        drag.start = None;
        drag.active = false;
        drag.layout.clear();
        return;
    }
    let Ok(window) = window.single() else { return };
    let Ok((camera, cam_tf)) = camera.single() else {
        return;
    };
    let ground = cursor_ground_point(window, camera, cam_tf, &terrain);
    let selected: Vec<usize> = selection
        .regiments
        .iter()
        .enumerate()
        .filter(|(_, s)| **s)
        .map(|(g, _)| g)
        .collect();

    if buttons.just_pressed(MouseButton::Right) {
        drag.start = ground;
        drag.active = false;
        drag.layout.clear();
    }
    if buttons.pressed(MouseButton::Right)
        && let (Some(start), Some(cur)) = (drag.start, ground)
    {
        drag.cur = cur;
        if !drag.active && start.distance(cur) >= ORDER_DRAG_MIN {
            drag.active = true;
        }
        if drag.active {
            drag.layout = line_layout(&groups, &selected, start, cur);
        }
    }
    if !buttons.just_released(MouseButton::Right) {
        return;
    }
    let Some(start) = drag.start.take() else { return };

    if drag.active && !drag.layout.is_empty() {
        apply_line_order(&mut groups, std::mem::take(&mut drag.layout), start, drag.cur);
    } else if let Some(t) = enemy_regiment_at(&groups, start) {
        attack_regiments(&mut groups, &selected, t);
        info!(
            "{} regiments ({} units) ATTACK regiment {t}",
            selected.len(),
            selection.count_units,
        );
    } else {
        order_regiments(&mut groups, &selected, start);
        info!(
            "{} regiments ({} units) move to ({:.0}, {:.0})",
            selected.len(),
            selection.count_units,
            start.x,
            start.y
        );
    }
    drag.active = false;
    drag.layout.clear();
}

/// Placement preview while a formation drag is in flight: the drawn line
/// plus a soft green marker per soldier slot (block outlines above the
/// gizmo budget), so the player can tune width and facing before letting
/// go. Player-facing UI — not gated by the debug-viz toggle.
fn draw_order_preview(
    drag: Res<OrderDrag>,
    groups: Res<Groups>,
    terrain: Res<Terrain>,
    mut gizmos: Gizmos,
) {
    if !drag.active || drag.layout.is_empty() {
        return;
    }
    let start = drag.start.unwrap_or(drag.cur);
    let lift = |p: Vec2, up: f32| Vec3::new(p.x, terrain.height_at(p.x, p.y) + up, p.y);
    // The pleasant kind of green.
    let line_col = Color::srgba(0.75, 0.95, 0.6, 0.9);
    let slot_col = Color::srgba(0.55, 0.92, 0.62, 0.75);

    gizmos.line(lift(start, 0.4), lift(drag.cur, 0.4), line_col);

    let total: usize = drag.layout.iter().map(|p| groups.list[p.g].count).sum();
    for p in &drag.layout {
        let gd = &groups.list[p.g];
        let fwd = facing_dir(p.facing);
        let right = Vec2::new(fwd.y, -fwd.x);
        let pitch = gd.spacing.pitch();
        let world = |off: Vec2| p.anchor + right * off.x + fwd * off.y;

        if total <= PREVIEW_CIRCLE_CAP {
            for off in slot_offsets(gd.count, p.files as usize, pitch) {
                gizmos.circle(
                    Isometry3d::new(
                        lift(world(off), 0.25),
                        Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                    ),
                    0.42,
                    slot_col,
                );
            }
        } else {
            // Block outline: cheap stand-in when the selection is huge.
            let ranks = (gd.count as u32).div_ceil(p.files) as f32;
            let hw = p.files as f32 * pitch.x * 0.5;
            let hd = ranks * pitch.y * 0.5;
            let c = [
                world(Vec2::new(-hw, hd)),
                world(Vec2::new(hw, hd)),
                world(Vec2::new(hw, -hd)),
                world(Vec2::new(-hw, -hd)),
            ];
            for k in 0..4 {
                gizmos.line(lift(c[k], 0.3), lift(c[(k + 1) % 4], 0.3), slot_col);
            }
        }
        // Facing arrow off the front rank.
        let ranks = (gd.count as u32).div_ceil(p.files) as f32;
        let front = p.anchor + fwd * ((ranks - 1.0) * 0.5 * pitch.y + 1.2);
        gizmos.arrow(lift(front, 0.6), lift(front + fwd * 3.5, 0.6), line_col);
    }
}

/// Backspace = HALT the selection (TW keybind): drop orders and hold in
/// place (the anchor moves to the current centroid so the block stands
/// where it is). S stays camera pan.
fn halt_key(
    keys: Res<ButtonInput<KeyCode>>,
    selection: Res<Selection>,
    mut groups: ResMut<Groups>,
) {
    if !keys.just_pressed(KeyCode::Backspace) || selection.count_units == 0 {
        return;
    }
    let mut halted = 0;
    for (g, sel) in selection.regiments.iter().enumerate() {
        let group = &mut groups.list[g];
        if *sel && group.count > 0 && !group.state.is_broken() {
            group.anchor = group.centroid;
            group.order = None;
            halted += 1;
        }
    }
    if halted > 0 {
        info!("HALT: {halted} regiments hold");
    }
}

/// Order lifecycle. Move: ends on arrival — the order point becomes the
/// new anchor (holding is a standing order). Attack: never "arrives", it
/// ends when the target regiment is wiped (hold in place there). Uses the
/// centroid refreshed by the frontline pass each tick.
pub fn clear_arrived_orders(mut groups: ResMut<Groups>) {
    let counts: Vec<usize> = groups.list.iter().map(|g| g.count).collect();
    for (g, group) in groups.list.iter_mut().enumerate() {
        match group.order {
            Some(Order::Move(t)) => {
                if group.count > 0
                    && !group.engaged
                    && group.centroid.distance(t) < ARRIVE_CLEAR
                {
                    group.anchor = t;
                    group.order = None;
                    info!("regiment {g} arrived, holding");
                }
            }
            Some(Order::Attack(t)) if counts[t as usize] == 0 => {
                group.anchor = group.centroid;
                group.order = None;
                info!("regiment {g} attack target {t} destroyed, holding");
            }
            _ => {}
        }
    }
}

fn draw_order_gizmos(
    viz: Res<crate::movement::DebugViz>,
    groups: Res<Groups>,
    terrain: Res<Terrain>,
    mut gizmos: Gizmos,
) {
    // Attack indicators are player-facing UI (not debug viz): a red marker
    // hangs over every regiment the player is attacking.
    let mut marked = vec![false; groups.list.len()];
    for group in &groups.list {
        if group.team == PLAYER_TEAM
            && let Some(Order::Attack(t)) = group.order
        {
            marked[t as usize] = true;
        }
    }
    for (t, m) in marked.iter().enumerate() {
        if !*m || groups.list[t].count == 0 {
            continue;
        }
        let c = groups.list[t].centroid;
        let top = Vec3::new(c.x, terrain.height_at(c.x, c.y) + 15.0, c.y);
        gizmos.arrow(top, top - Vec3::Y * 4.5, Color::srgb(1.0, 0.25, 0.2));
    }

    if !viz.0 {
        return;
    }
    // Move-order targets (attack orders have the marker above).
    for group in &groups.list {
        if let Some(Order::Move(t)) = group.order {
            let base = Vec3::new(t.x, terrain.height_at(t.x, t.y) + 0.5, t.y);
            let color = if group.team == 0 {
                Color::srgb(0.4, 0.8, 1.0)
            } else {
                Color::srgb(1.0, 0.6, 0.2)
            };
            gizmos.circle(
                Isometry3d::new(base, Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
                3.0,
                color,
            );
            gizmos.line(base, base + Vec3::Y * 10.0, color);
        }
    }
}

/// RMS spread of a regiment's living units around its centroid — the
/// block-coherence metric for the orders test.
fn regiment_spread(units: &Units, g: u32) -> f32 {
    let (mut sum, mut c, mut centroid) = (0.0f32, 0usize, Vec2::ZERO);
    for i in 0..units.len() {
        if units.group[i] == g && units.death_t[i] == 0 {
            centroid += Vec2::new(units.pos[i].x, units.pos[i].z);
            c += 1;
        }
    }
    if c == 0 {
        return 0.0;
    }
    centroid /= c as f32;
    for i in 0..units.len() {
        if units.group[i] == g && units.death_t[i] == 0 {
            sum += Vec2::new(units.pos[i].x, units.pos[i].z).distance_squared(centroid);
        }
    }
    (sum / c as f32).sqrt()
}

/// FL_TEST_ORDERS=1: regiment cohesion + arrangement-preserving group
/// moves. Stage 0: enclosure-select regiments (exercises the promote
/// path), long-march ONE regiment 350 m and compare its RMS spread on
/// arrival (acceptance: < 1.5x). Stage 1: group-move several regiments
/// and log how well their relative arrangement survived.
fn test_orders_script(
    time: Res<Time>,
    units: Res<Units>,
    mut groups: ResMut<Groups>,
    mut selection: ResMut<Selection>,
    mut stage: Local<u32>,
    mut watch: Local<Option<(u32, f32)>>,
    mut arrangement: Local<Vec<(usize, Vec2)>>,
) {
    if std::env::var("FL_TEST_ORDERS").is_err() {
        return;
    }
    let t = time.elapsed_secs();
    match *stage {
        0 if t > 5.0 => {
            // Enclosure stroke around a spot inside the blue army: must
            // promote to whole regiments.
            let center = Vec2::new(-150.0, -90.0);
            let stroke: Vec<Vec2> = (0..28)
                .map(|k| {
                    let a = k as f32 / 28.0 * std::f32::consts::TAU;
                    center + Vec2::new(a.cos(), a.sin()) * 45.0
                })
                .collect();
            select_along_line(&stroke, &units, &groups, &mut selection);
            info!(
                "[test] enclosure selected {} regiments ({} units)",
                selection.regiments.iter().filter(|s| **s).count(),
                selection.count_units
            );
            // Long-march the blue regiment nearest the enclosure center.
            let g = groups
                .list
                .iter()
                .enumerate()
                .filter(|(_, gr)| gr.team == PLAYER_TEAM && gr.count > 0)
                .min_by(|a, b| {
                    a.1.centroid
                        .distance_squared(center)
                        .total_cmp(&b.1.centroid.distance_squared(center))
                })
                .map(|(g, _)| g as u32)
                .unwrap();
            let spread = regiment_spread(&units, g);
            groups.list[g as usize].order =
                Some(Order::Move(groups.list[g as usize].centroid + Vec2::new(350.0, 60.0)));
            *watch = Some((g, spread));
            info!("[test] regiment {g} long-march ordered, spread at departure {spread:.1} m");
            selection.regiments.fill(false);
            selection.count_units = 0;
            *stage = 1;
        }
        1 => {
            if let Some((g, spread0)) = *watch
                && groups.list[g as usize].order.is_none()
            {
                let spread = regiment_spread(&units, g);
                info!(
                    "[test] regiment {g} arrived: spread {spread:.1} m vs {spread0:.1} m at departure (ratio {:.2})",
                    spread / spread0.max(0.01)
                );
                *stage = 2;
            }
        }
        2 if t > 45.0 => {
            // Group-move: order several blue regiments as one formation.
            let selected: Vec<usize> = groups
                .list
                .iter()
                .enumerate()
                .filter(|(_, gr)| gr.team == PLAYER_TEAM && gr.count > 0 && gr.order.is_none())
                .take(10)
                .map(|(g, _)| g)
                .collect();
            let centroid: Vec2 = selected.iter().map(|&g| groups.list[g].centroid).sum::<Vec2>()
                / selected.len().max(1) as f32;
            *arrangement = selected
                .iter()
                .map(|&g| (g, groups.list[g].centroid - centroid))
                .collect();
            order_regiments(&mut groups, &selected, centroid + Vec2::new(-120.0, -100.0));
            info!("[test] group-move: {} regiments as one formation", selected.len());
            *stage = 3;
        }
        3 if !arrangement.is_empty()
            && arrangement.iter().all(|(g, _)| groups.list[*g].order.is_none()) =>
        {
            let centroid: Vec2 = arrangement
                .iter()
                .map(|(g, _)| groups.list[*g].centroid)
                .sum::<Vec2>()
                / arrangement.len() as f32;
            let err: f32 = arrangement
                .iter()
                .map(|(g, off)| (groups.list[*g].centroid - centroid).distance(*off))
                .sum::<f32>()
                / arrangement.len() as f32;
            info!("[test] group-move arrived: mean arrangement error {err:.1} m");
            *stage = 4;
        }
        _ => {}
    }
}
