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
//!
//! A click or a drag on a lane (the stone wall's walk,
//! town_layout.rs `Lane`) takes the lane's own places instead
//! (`lane_layout`): the ranks are as deep as the lane holds, the files
//! run along it from the drag's start as far as the drag or the men need,
//! the block faces out, and the run slides along the lane past the runs
//! the team's other regiments hold there. A regiment coming onto the lane
//! from off it fills its run from the far end as its men arrive
//! (formation.rs `fill_lane_slots`), so nobody passes anybody on it.

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::formation::{FormShape, LaneFill, facing_dir, facing_of, slot_offsets};
use crate::selection::cursor_ground_point;
pub use crate::selection::{Hover, Selection, select_along_line};
use crate::terrain::Terrain;
use crate::town_layout::Lane;
use crate::units::Units;

pub const PLAYER_TEAM: u8 = 0;
/// Groups whose centroid gets this close to their order target go idle.
pub(crate) const ARRIVE_CLEAR: f32 = 18.0;
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
    /// Set when the current order was issued by the automatic attack
    /// (ai.rs `auto_engage`) or a script, not the player: UI click
    /// feedback stays silent.
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
    /// The spacing an open regiment (FormSpacing::Open) closes back to.
    pub close_to: crate::formation::FormSpacing,
    /// How an open regiment stands (formation.rs `choose_open`).
    pub open: crate::formation::OpenOrder,
    /// Walking through a friendly regiment that opened for it: its block
    /// within OPEN_KEEP of an open friend's, or any of its men still
    /// inside one (sim/job.rs `open_for_passers`). The friend's
    /// bookkeeping: it counts this regiment's men inside it.
    pub passing: bool,
    /// Its living men inside each open friend's block within its reach,
    /// as (friend, men) (frontline.rs), what the friend reads to know
    /// when the last man is out. Empty with no open friend in reach.
    pub inside: Vec<(u32, u32)>,
    /// How far its farthest living man stands from its centroid (metres,
    /// frontline.rs): the stragglers' reach, so an open friend counts this
    /// regiment's men only when one could be inside it.
    pub reach: f32,
    /// Formation facing (unit yaw convention, 0 = +Z). Baked into the slot
    /// offsets on reform; meaningless for Blob.
    pub facing: f32,
    /// Grid width in files (soldiers per rank). 0 = auto on first reform.
    pub files: u32,
    /// Hold position (defend mode): never auto-engage, never chase. Out
    /// of it (false, default), an idle regiment attacks an enemy that comes
    /// near when the battle's automatic attack is on (ai.rs `auto_engage`).
    pub hold: bool,
    /// A scripted regiment that never breaks, whatever its morale: the
    /// line a test scene presses against.
    pub steadfast: bool,
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
    /// Footprint radius estimate (m): 1.5x the RMS distance of living
    /// units to the centroid, smoothed ~2 s. The morale flank ring
    /// probes OUTSIDE this — a probe inside our own mass measures
    /// nothing (a 1000-man block swallows any fixed ring).
    pub radius: f32,
    // --- refreshed every fixed tick by the frontline pass ---
    pub centroid: Vec2,
    /// TW rule: one soldier of the regiment fighting = the whole regiment
    /// is engaged (ground truth from per-unit melee state, with a short
    /// hold to bridge swing-cycle gaps).
    pub engaged: bool,
    /// Ticks of `engaged` left since the last soldier fought.
    pub engage_hold: u8,
    /// Holding a contact frame (frontline.rs): an attacking regiment in
    /// melee stops chasing its target's center and holds its slots with
    /// the front rank on the fight line, the way M2TW stops updating a
    /// formation near the end of its path.
    pub contact: bool,
    /// The contact frame's lateral position along the regiment's right
    /// vector, fixed when contact begins so the block cannot slide
    /// sideways while it fights.
    pub contact_lateral: f32,
    /// Ticks this regiment has been in a real melee (engaged, with at
    /// least the count gate's share of its men fighting once); 0 when
    /// not. Men out of sight of an enemy join the fight after their own
    /// delay counted from here (sim/soldier/).
    pub melee_ticks: u32,
    /// Where the enemy unit this regiment fights stands: its ordered
    /// target if alive and unbroken, else the nearest unbroken enemy
    /// regiment. None when not in melee or in hold (guard) mode.
    pub fight_point: Option<Vec2>,
    /// Where its men fighting an enemy ahead stand along its forward
    /// (frontline.rs; the axis is `formation::facing_dir(facing)`, the
    /// soldiers' `form_face`): set while the regiment's melee runs, moved a
    /// tenth of the way to the live mean each tick, kept when nobody fights
    /// for a tick, NaN otherwise. Under the friction contact a joiner behind
    /// it walks straight ahead up to it (sim/soldier/motion.rs `join_goal`).
    pub fight_line: f32,
    /// In the charge phase: attack order, inside charge range of the
    /// target, not yet in contact, and running at it (frontline.rs).
    /// Drives the charge pace, the sprint pose and the charge cries.
    pub charging: bool,
    /// The crash of a charge (frontline.rs): from the moment a charging
    /// regiment engages until the enemy has stopped its block. The frame
    /// keeps moving, the charge pace stays, the melee clock waits.
    pub crashing: bool,
    pub crash_ticks: u16,
    /// The centroid's forward speed, smoothed (m/s): the crash ends when
    /// it dies (the enemy has stopped the block).
    pub adv_speed: f32,
    /// The centroid's speed toward its attack target, smoothed (m/s): a
    /// charge runs only while the regiment closes on its target.
    pub approach_speed: f32,
    /// Ticks the crash may last at most: the block's depth over the
    /// charge pace, the time its rear needs to arrive.
    pub crash_cap: u16,
    /// The attack target the frontline pass saw last tick, to notice a
    /// new one.
    pub seen_target: Option<u32>,
    /// Breaking off for a new target: the attack order named another
    /// regiment while this one fought. Until enough of its men fight the
    /// new target, or it disengages, it has no fight and no contact frame
    /// and its men go after no enemy, as under a Move order: the block
    /// marches on the new target and its men strike only whoever stands
    /// in front of them.
    pub retarget: bool,
    /// An enemy regiment's centroid is within combat-watch range: units
    /// of this regiment scan wider for adjacent enemies (sparse-fight
    /// acquisition, sim/soldier/fight.rs) and brace when standing.
    pub enemy_near: bool,
    /// Normalized direction to the nearest enemy regiment (ZERO when
    /// none in watch range): standing units face it (brace facing).
    pub threat_dir: Vec2,
    /// An UNBROKEN enemy regiment is within watch range. The falling
    /// edge (last nearby foe routed or wiped) triggers the cheer.
    pub hostile_near: bool,
    /// Victory-cheer ticks remaining (render-only celebration).
    pub celebrate: u16,
    /// The army's command regiment (the captain/general rides here).
    /// M2TW: every army has a leader; his presence is an army-wide
    /// morale term and his fall a shock. Assigned once by morale.rs.
    pub leader: bool,
    // --- morale (morale.rs updates per tick) ---
    /// Effective morale LEVEL (M2TW model): base stat + the signed sum of
    /// situational modifiers, recomputed every tick — NOT an accumulator.
    /// Rout threshold is at -11 (morale.rs); typical range ~-20..+20.
    pub morale: f32,
    /// Ticks the effective level has sat at/below the rout line
    /// (hysteresis: one spiked tick must not break a regiment).
    pub break_ticks: u8,
    // --- fatigue (fatigue.rs updates per tick) ---
    /// 0 (fresh) .. 100 (exhausted cap). Fighting > charging > running
    /// accumulate; standing recovers. Banded into the six M2TW states.
    pub fatigue: f32,
    pub state: RegState,
    /// Deaths since the last morale tick (tallied by the damage apply pass).
    pub recent_deaths: u32,
    /// Kills MADE by this regiment since the last morale tick (damage
    /// apply pass): recent_kills vs recent_deaths is the winning/losing-
    /// melee morale factor.
    pub recent_kills: u32,
    /// How many times this regiment's slots have been dealt, by
    /// `formation::assign_slots` or by `formation::fill_lane_slots` when
    /// places on a lane change hands, which is every write of its men's
    /// `home` after the kick: the tick job's regiment sums (frontline.rs)
    /// hold only while it is the count they were taken with.
    pub slots_dealt: u32,
    // --- ranged (archer regiments only; dead flags for melee kinds) ---
    /// Fire-at-will (M2TW default ON): with no explicit target order,
    /// the regiment's bows engage the nearest enemy block in range.
    pub fire_at_will: bool,
    /// Swords out (archers; the F key, shared with the wall): the
    /// regiment fights with its sidearm and shoots nothing, an attack
    /// order is a charge like any infantry's, the skirmish run is off,
    /// and it engages at ease as infantry does.
    pub melee: bool,
    /// Skirmish mode (on by default for missile troops): the regiment
    /// runs from an enemy block on it or coming for it, a bow's range
    /// away, and shoots again (sim/job.rs `skirmish_reactions`).
    pub skirmish: bool,
    /// Running from an enemy (the skirmish reaction, sim/job.rs
    /// `skirmish_reactions`): the regiment's Move order is the run's,
    /// and its own order waits in `skirmish_saved`.
    pub skirmishing: bool,
    /// The run's own Move order, so an order anyone else writes during
    /// the run is noticed and parked.
    pub skirmish_to: Option<Vec2>,
    /// The order the run parked, handed back when the run ends.
    pub skirmish_saved: Option<Order>,
    /// The enemy block in the regiment's path that it stands and shoots
    /// at instead of walking into (the skirmish trigger's third case;
    /// with skirmish off, the attack order's close-in march alone).
    pub skirmish_hold: Option<u32>,
    /// Arrows left across the regiment (summed per tick, HUD ammo bar).
    pub ammo_left: u32,
    /// The regiment has a live fire solution this tick (archers:
    /// target in range, not marching, not in melee, arrows left) —
    /// drives the unit card's firing indicator, M2TW-style.
    pub firing: bool,
    /// An archer regiment under an attack order standing off its target:
    /// in range, halted, volleying. Kept by sim/job.rs: entered where the
    /// order found the target in range or where the march arrived at the
    /// firing place, left when the target is out of range.
    pub standoff: bool,
    /// The attack target the stand-off was judged for; a new target is
    /// judged afresh.
    pub standoff_target: Option<u32>,
    /// Where an archer regiment's attack order marches it: the frame
    /// centre from which its front rank stands the close-in distance
    /// short of the target's near edge (sim/job.rs `firing_place`). The
    /// march's goal in place of the target's centre, so no man's slot
    /// lies in the enemy. None without a bows-out attack order.
    pub fire_place: Option<Vec2>,
    /// How far the regiment's foremost living man stands ahead of its
    /// centre along its facing (metres; frontline.rs). A column strung
    /// out on the march is seen by its head, not its middle.
    pub lead: f32,
    /// The mean velocity of its living men (m/s; frontline.rs): how the
    /// regiment moves. The centroid's own motion is not that: it jumps
    /// whenever men fall, and a shrinking block's jumps read as speed.
    pub vel: Vec2,
    /// The regiment is coming onto a lane and its men take their places
    /// there as they arrive (formation.rs `fill_lane_slots`).
    pub lane_fill: Option<LaneFill>,
    /// Its block stands, or is ordered to stand, in a lane's rows (the
    /// wall's walk): in normal order its places are then the lane's pitch
    /// apart (`formation::pitch_of`). Set by a lane order and kept in
    /// step with its order and anchor by `formation::fill_lane_slots`.
    pub lane: bool,
    /// The living men past the waypoint its route's frame waits at, as
    /// (waypoint, men past, living men), counted from the tick's start
    /// positions (frontline.rs). The frame moves on once their share
    /// reaches `route::frame_share` (sim/route.rs `update_routes`). None
    /// with no frame short of its route's end.
    pub frame_past: Option<(u32, u32, u32)>,
}

impl GroupData {
    /// The spacing the player or the AI chose: an open regiment's is the
    /// one it closes back to.
    pub fn chosen_spacing(&self) -> crate::formation::FormSpacing {
        if self.spacing == crate::formation::FormSpacing::Open { self.close_to } else { self.spacing }
    }

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
            close_to: crate::formation::FormSpacing::Normal,
            open: crate::formation::OpenOrder::default(),
            passing: false,
            inside: Vec::new(),
            reach: 0.0,
            facing: if team == 0 { 0.0 } else { std::f32::consts::PI },
            files: 0,
            hold: false,
            steadfast: false,
            reform: false,
            count_at_reform: count,
            disorder: 0.0,
            home_bias: Vec2::ZERO,
            radius: 0.0,
            centroid: anchor,
            engaged: false,
            engage_hold: 0,
            contact: false,
            contact_lateral: 0.0,
            melee_ticks: 0,
            fight_point: None,
            fight_line: f32::NAN,
            charging: false,
            crashing: false,
            crash_ticks: 0,
            adv_speed: 0.0,
            approach_speed: 0.0,
            crash_cap: 0,
            seen_target: None,
            retarget: false,
            enemy_near: false,
            threat_dir: Vec2::ZERO,
            hostile_near: false,
            celebrate: 0,
            leader: false,
            // Overwritten by the first morale tick (base + calm bonuses).
            morale: crate::unit_types::TYPES[kind as usize].base_morale,
            break_ticks: 0,
            fatigue: 0.0,
            state: RegState::Steady,
            recent_deaths: 0,
            recent_kills: 0,
            slots_dealt: 0,
            fire_at_will: true,
            melee: false,
            skirmish: kind == crate::unit_types::KIND_ARCHER,
            skirmishing: false,
            skirmish_to: None,
            skirmish_saved: None,
            skirmish_hold: None,
            ammo_left: if kind == crate::unit_types::KIND_ARCHER {
                count as u32 * crate::unit_types::missile::AMMO as u32
            } else {
                0
            },
            firing: false,
            standoff: false,
            standoff_target: None,
            fire_place: None,
            lead: 0.0,
            vel: Vec2::ZERO,
            lane_fill: None,
            lane: false,
            frame_past: None,
        }
    }
}

/// The attack front: an attacker with a living friend between it and its
/// target lays its front rank on the enemy's outline where the line to the
/// target crosses it (`Groups::front_on_outline`), so it stands behind the
/// friend instead of walking into the friend's ranks toward the target's
/// centre. On by default under the friction contact and off without it
/// (sim/soldier/body.rs `friction_contact`); FL_ATTACK_FRONT=3 turns it on and
/// any other value turns it off under either contact. Off, every attacker
/// goes for the target's centre.
pub(crate) fn attack_front() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("FL_ATTACK_FRONT") {
        Ok(v) => v == "3",
        Err(_) => crate::sim::soldier::friction_contact(),
    })
}

#[derive(Resource, Default)]
pub struct Groups {
    pub list: Vec<GroupData>,
}

impl Groups {
    /// A group's order destination this tick: Move goes to the point,
    /// Attack chases the target regiment's current centroid, or for an
    /// archer regiment shooting its target, the firing place short of it
    /// (sim/job.rs `archer_fire_solutions`), so that no man's slot lies
    /// in the enemy and the march ends where the bows reach.
    pub fn goal(&self, g: usize) -> Option<Vec2> {
        let gd = &self.list[g];
        match gd.order? {
            Order::Move(p) => Some(p),
            Order::Attack(t) => {
                let td = &self.list[t as usize];
                if td.count == 0 {
                    return None;
                }
                match gd.fire_place {
                    Some(p) if gd.standoff_target == Some(t) => Some(p),
                    // With a friend between, on the enemy's outline where
                    // the line to the target crosses it. A regiment in its
                    // charge or its crash goes for the target itself, as a
                    // direct attack does, so every one of its soldiers runs
                    // in and the whole regiment piles up on what stops it.
                    // Its places come back to the outline when the charge is
                    // over, and under the friction contact its soldiers do
                    // not step back to them (sim/soldier/passing.rs `hold_the_frame`).
                    _ if attack_front() => {
                        if gd.charging || gd.crashing {
                            Some(td.centroid)
                        } else {
                            Some(self.front_on_outline(g, t as usize).map_or(td.centroid, |(goal, _)| goal))
                        }
                    }
                    _ => Some(td.centroid),
                }
            }
        }
    }

    /// The attack front: for regiment `g` attacking `t`, None when no
    /// living friend stands between them (`outline_contact`), else (the
    /// goal, the contact point).
    pub(crate) fn front_on_outline(&self, g: usize, t: usize) -> Option<(Vec2, Vec2)> {
        self.outline_contact(g, t)
            .and_then(|(goal, point, friend)| friend.then_some((goal, point)))
    }

    /// The attack front: for regiment `g` attacking `t`, (the goal on the
    /// outline, the contact point, whether a living friend stands between
    /// them), whatever the friend test finds; None only when the two
    /// centroids coincide. Only `front_on_outline` sends a regiment to this
    /// goal, and only with a friend between; the F3 view draws it either
    /// way. "Between": the friend's centroid lies in the corridor from
    /// `g`'s centroid to `t`'s, ahead of `g`'s centroid and short of `t`'s,
    /// no further to the side of the line than half `g`'s block width
    /// across it (`sim::route::half_width`). The contact point is where the
    /// line from `g`'s centroid to `t`'s centroid first crosses the outline
    /// of the nearest enemy regiment on the way (`t`, or a living enemy
    /// whose centroid lies in the same corridor; nearest = the smallest
    /// distance along the line to its outline, `outline_entry`). The goal
    /// is on the line, `g`'s own half depth (`formation::front_offset`)
    /// short of the contact point, so `g`'s front rank stands on it.
    pub(crate) fn outline_contact(&self, g: usize, t: usize) -> Option<(Vec2, Vec2, bool)> {
        let gd = &self.list[g];
        let td = &self.list[t];
        let a = gd.centroid;
        let len = (td.centroid - a).length();
        if len < 1e-3 {
            return None;
        }
        let d = (td.centroid - a) / len;
        let across = Vec2::new(-d.y, d.x);
        let half = crate::sim::route::half_width(gd, d);
        let in_corridor = |c: Vec2| {
            let s = (c - a).dot(d);
            s > 0.0 && s < len && (c - a).dot(across).abs() < half
        };
        let friend = self
            .list
            .iter()
            .enumerate()
            .any(|(f, fd)| f != g && fd.team == gd.team && fd.count > 0 && in_corridor(fd.centroid));
        let mut edge = outline_entry(td, a, d);
        for (h, hd) in self.list.iter().enumerate() {
            if h != t && hd.team != gd.team && hd.count > 0 && in_corridor(hd.centroid) {
                edge = edge.min(outline_entry(hd, a, d));
            }
        }
        Some((a + d * (edge - crate::formation::front_offset(gd)), a + d * edge, friend))
    }
}

/// The attack front: how far along the line from `a` in the unit
/// direction `d` it first meets regiment `hd`'s outline. The outline is
/// the formation code's own block (`formation::block_rect`: about the
/// centroid, turned to the facing, files and ranks at the spacing as the
/// block stands, open order included; a blob or a block with no files a
/// disc of its footprint radius) without its metre of margin, so its
/// front edge is the front rank (`formation::front_offset`), its sides
/// the flank files and its back the rear rank. A line that misses the
/// outline (only for an enemy beside the line, never the target, which
/// the line runs to) meets it where the outline's nearest corner stands
/// along the line. Negative when `a` stands inside the outline.
pub(crate) fn outline_entry(hd: &GroupData, a: Vec2, d: Vec2) -> f32 {
    let r = crate::formation::block_rect(hd);
    let p = a - r.centre;
    if r.fwd == Vec2::ZERO {
        let b = p.dot(d);
        let disc = b * b - (p.length_squared() - r.half_w * r.half_w);
        if disc >= 0.0 {
            return -b - disc.sqrt();
        }
        return -b - r.half_w;
    }
    let half_w = (r.half_w - 1.0).max(0.0);
    let half_d = (r.half_d - 1.0).max(0.0);
    let right = Vec2::new(r.fwd.y, -r.fwd.x);
    let (px, pz, dx, dz) = (p.dot(right), p.dot(r.fwd), d.dot(right), d.dot(r.fwd));
    let mut lo = f32::NEG_INFINITY;
    let mut hi = f32::INFINITY;
    for (q, v, h) in [(px, dx, half_w), (pz, dz, half_d)] {
        if v.abs() < 1e-6 {
            if q.abs() > h {
                lo = f32::INFINITY;
            }
            continue;
        }
        let (s1, s2) = ((-h - q) / v, (h - q) / v);
        lo = lo.max(s1.min(s2));
        hi = hi.min(s1.max(s2));
    }
    if lo <= hi && hi >= 0.0 {
        return lo;
    }
    -p.dot(d) - (dx.abs() * half_w + dz.abs() * half_d)
}

/// One regiment's placement on a drawn order line.
struct LinePlacement {
    g: usize,
    /// Block CENTER (the Move target; the front rank sits on the line).
    anchor: Vec2,
    facing: f32,
    files: u32,
    /// The placement is a run on a lane.
    lane: bool,
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
        // The selection is plain data the scripts and the battle setup
        // touch too, so it lives with the regiments.
        app.init_resource::<Groups>()
            .init_resource::<crate::selection::Selection>()
            .add_systems(
                FixedUpdate,
                clear_arrived_orders
                    .after(crate::sim::step_sim)
                    .in_set(crate::game_state::SimSet),
            );
    }
}

/// The player's orders: the mouse and the keys, and their drawing.
pub struct OrdersViewPlugin;

impl Plugin for OrdersViewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<OrderDrag>()
            .add_systems(
                Update,
                (
                    (issue_order, draw_order_preview)
                        .chain()
                        .in_set(crate::game_state::BattleInputSet),
                    halt_key.in_set(crate::game_state::BattleInputSet),
                    draw_deploy_zone.run_if(
                        in_state(crate::game_state::GameState::Battle)
                            .and_then(crate::game_state::deploying),
                    ),
                    draw_order_gizmos,
                ),
            );
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
        group.lane_fill = None;
        group.lane = false;
        // March facing the way we're going: the front rank leads.
        let dir = dest - group.centroid;
        if group.shape == FormShape::Rect && dir.length() > 1.0 {
            group.facing = facing_of(dir);
            group.reform = true;
        }
    }
}

/// A click order on the map: on a lane, the lane's places from the
/// click on (`lane_layout` with a line of no length); anywhere else
/// `order_regiments`. Returns whether any regiment was ordered: a click
/// on a lane with no room for any of them orders nobody, and is never a
/// march to the point on the wall itself.
pub fn click_order(groups: &mut Groups, terrain: &Terrain, selected: &[usize], target: Vec2) -> bool {
    if lane_at(terrain, target).is_some() {
        let layout = line_layout(groups, terrain, selected, target, target, Vec2::ZERO, false);
        if layout.is_empty() {
            return false;
        }
        apply_line_order(groups, layout, target, target);
        return true;
    }
    order_regiments(groups, selected, target);
    true
}

/// The map's lane an order at `p` is meant for, if any: the stone wall's
/// walk, for `p` anywhere on the wall (`Lane::catches`).
fn lane_at(terrain: &Terrain, p: Vec2) -> Option<Lane> {
    if !terrain.kind.has_wall() {
        return None;
    }
    let lane = crate::town_layout::walk_lane(terrain.body_radius());
    lane.catches(p).then_some(lane)
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
        group.lane_fill = None;
        // An archer regiment with its bows turns and dresses only when it
        // has to march (sim/job.rs): in range it shoots where it stands.
        // With swords out its order is a charge like any other.
        if group.kind == crate::unit_types::KIND_ARCHER && !group.melee {
            continue;
        }
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
/// faces perpendicular to the line. Which side the blocks face is the
/// M2TW drag-hand rule when `face_hint` is non-zero (the caller derives
/// it from the camera: a left-to-right drag on screen faces away from
/// the camera, right-to-left toward it — the player picks the facing
/// with the hand that draws). A zero hint (script callers with no
/// camera) falls back to the enemy side of the line, then to facing
/// away from where the selection stands. Regiments keep their
/// left-to-right order along the line to minimize crossing. With `tight`
/// the blocks stand side by side with no gap between them, as one
/// continuous line (the player holds Ctrl); a drag on a lane ignores it.
fn line_layout(
    groups: &Groups,
    terrain: &Terrain,
    selected: &[usize],
    a: Vec2,
    b: Vec2,
    face_hint: Vec2,
    tight: bool,
) -> Vec<LinePlacement> {
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
    // A drag with either end on the wall is a drag along it.
    if let Some(lane) = lane_at(terrain, a).or_else(|| lane_at(terrain, b)) {
        return lane_layout(groups, &lane, &picked, a, b);
    }
    let len = a.distance(b);
    let dir = (b - a) / len.max(1e-3);
    let mean: Vec2 =
        picked.iter().map(|&g| groups.list[g].centroid).sum::<Vec2>() / picked.len() as f32;
    let perp = Vec2::new(-dir.y, dir.x);
    let mid = (a + b) * 0.5;
    let side = if face_hint != Vec2::ZERO {
        perp.dot(face_hint)
    } else {
        // No camera signal: which side of the line does the enemy stand
        // on? Strength-weighted enemy centroid so a routed straggler
        // can't flip the line; the last fallback faces away from where
        // the selection stands.
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
        if enemy_side.abs() > 1.0 {
            enemy_side
        } else {
            perp.dot(mid - mean)
        }
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
    // Each regiment's files and the distance of its centre along the line
    // from `a`.
    let mut blocks: Vec<(usize, u32, f32)> = Vec::with_capacity(segs.len());
    if tight {
        // No gaps: each block is `files` pitches wide and starts where the
        // one before it ends, so the outer files of two neighbours stand
        // half of each one's pitch from their shared edge. The whole run
        // is centred on the drag's midpoint.
        let mut run = 0.0;
        for &(g, _) in &segs {
            let gd = &groups.list[g];
            let share = len * gd.count as f32 / total as f32;
            let pitch = gd.chosen_spacing().pitch(gd.kind);
            let files = ((share / pitch.x).round() as u32).clamp(1, gd.count as u32);
            let width = files as f32 * pitch.x;
            blocks.push((g, files, run + width * 0.5));
            run += width;
        }
        let shift = (len - run) * 0.5;
        for block in &mut blocks {
            block.2 += shift;
        }
    } else {
        // Each block is centred in its share, LINE_REG_GAP and the
        // rounding's leftover between it and the next.
        let mut off = 0.0;
        for &(g, _) in &segs {
            let gd = &groups.list[g];
            let share = len * gd.count as f32 / total as f32;
            let pitch = gd.chosen_spacing().pitch(gd.kind);
            let files = (((share - LINE_REG_GAP) / pitch.x).floor() as u32)
                .clamp(1, gd.count as u32);
            blocks.push((g, files, off + share * 0.5));
            off += share;
        }
    }
    let mut placements = Vec::with_capacity(blocks.len());
    for (g, files, centre) in blocks {
        let gd = &groups.list[g];
        let pitch = gd.chosen_spacing().pitch(gd.kind);
        let ranks = (gd.count as u32).div_ceil(files);
        // Front rank ON the drawn line; the block center sits behind it.
        let line_pos = a + dir * centre;
        let anchor = line_pos - fwd * ((ranks - 1) as f32 * 0.5 * pitch.y);
        placements.push(LinePlacement { g, anchor, facing, files, lane: false });
    }
    placements
}

/// One regiment of a line order: the block re-dressed `files` wide
/// facing `facing`, its centre marching to `anchor`. Shared by the drag
/// order and a setup file's `line` order (battle_setup.rs).
pub fn line_place(gd: &mut GroupData, anchor: Vec2, facing: f32, files: u32) {
    gd.facing = facing;
    gd.files = files;
    gd.shape = FormShape::Rect;
    gd.order = Some(Order::Move(anchor));
    gd.auto_order = false;
    gd.reform = true;
    // A new order ends a fill of the lane; the lane order starts its own
    // after this.
    gd.lane_fill = None;
}

/// One line of the order log: every order the player gives, with the
/// game clock, the regiment and what it was told, in a form that
/// turns back into a setup file's timed orders, so a play can be rerun
/// from its log.
pub fn log_order(t: f32, g: usize, gd: &GroupData, what: &str) {
    info!("[order] t={t:.1}s g{g} ({}, {} men): {what}", crate::unit_types::kind_name(gd.kind), gd.count);
}

/// Lay `picked` regiments along a lane from `a` toward `b` (a click is
/// `a == b`): each block faces out, its places are the lane's pitch apart
/// with every other rank shifted half a file (`formation::lane_pitch`,
/// `lane_stagger`), its ranks are as many as the lane holds at that
/// pitch, its files are the fewest its men need in those ranks and one
/// more for each file's pitch of its share of the drag, and its run
/// slides along the lane past the runs the team's other regiments hold
/// (`free_run`). A run that starts on another regiment's run is laid
/// behind it instead, in the rows that run leaves free (`behind_run`).
/// The regiments keep their order along the drag, each run after the one
/// before it.
///
/// A regiment gets a placement only where its whole run is on the lane
/// and clear of every other run, or behind them: where the drawn width
/// has no such stretch beside the runs it takes the least width its ranks
/// allow, and where that has none either it gets no placement and keeps
/// the order it had. Places are never laid past the lane's ends or over
/// another regiment's.
fn lane_layout(groups: &Groups, lane: &Lane, picked: &[usize], a: Vec2, b: Vec2) -> Vec<LinePlacement> {
    let dir = if b.x >= a.x { 1.0 } else { -1.0 };
    let facing = lane.facing();
    let body = crate::sim::soldier::body_distance();
    let mut segs: Vec<(usize, f32)> =
        picked.iter().map(|&g| (g, groups.list[g].centroid.x * dir)).collect();
    segs.sort_unstable_by(|x, y| x.1.total_cmp(&y.1));
    let len = (b.x - a.x).abs();
    let total: usize = picked.iter().map(|&g| groups.list[g].count).sum();
    let runs = lane_runs(groups, lane, groups.list[picked[0]].team, picked);
    let mut taken: Vec<(f32, f32)> = runs.iter().map(|r| (r.0, r.1)).collect();
    let mut backs: Vec<f32> = runs.iter().map(|r| r.2).collect();
    let mut cursor = a.x;
    let mut placements = Vec::with_capacity(segs.len());
    for (g, _) in segs {
        let gd = &groups.list[g];
        let pitch = crate::formation::lane_pitch(gd.chosen_spacing(), gd.kind);
        let stagger = crate::formation::lane_stagger(gd.chosen_spacing(), gd.kind);
        let share = len * gd.count as f32 / total as f32;
        let n = gd.count as u32;
        // The ranks the lane holds in front of `front` (a man's centre),
        // to its back edge. The margins, a tenth of a millimetre and a
        // ten-thousandth of a rank, keep a rank whose centre falls on the
        // back edge from being rounded away: at FL_BODY=1.4 the walk gives
        // a man's centre 4.6 m, which a shieldwall's five ranks 1.15 m
        // apart fill exactly, and a shieldwall behind one four ranks deep
        // has its front rank on the back edge. At the default body every
        // spacing's depth is a hundredth of a rank or more from a whole
        // number of ranks, and the margins change nothing.
        let rows = |front: f32| {
            if front < lane.across.0 - 1e-4 { 0 } else { ((front - lane.across.0) / pitch.y + 1e-4).floor() as u32 + 1 }
        };
        let width = |files: u32| (files - 1) as f32 * pitch.x + if n > files { stagger } else { 0.0 } + body;
        // The drag widens the block from its deepest at once: each file's
        // pitch of drag adds a file to the fewest the rows allow.
        let extra = (share / pitch.x).floor() as u32;
        let least = n.div_ceil(rows(lane.across.1).max(1)).clamp(1, n);
        // A drag that starts on another regiment's run places this one
        // behind it, in the rows that run leaves free; else beside the
        // runs, at the front.
        let behind = taken.iter().any(|t| cursor >= t.0 && cursor <= t.1);
        let placed = if behind {
            behind_run(lane, &taken, &backs, cursor, dir, n, extra, pitch.y, &rows, &width)
        } else {
            [(least + extra).min(n), least].into_iter().find_map(|files| {
                free_run(lane, &taken, cursor, dir, width(files)).map(|run| (files, run, lane.across.1))
            })
        };
        let Some((files, run, front)) = placed else { continue };
        let ranks = n.div_ceil(files);
        let anchor = Vec2::new((run.0 + run.1) * 0.5, front - (ranks - 1) as f32 * 0.5 * pitch.y);
        cursor = if dir > 0.0 { run.1 + LINE_REG_GAP } else { run.0 - LINE_REG_GAP };
        taken.push(run);
        backs.push(front - (ranks - 1) as f32 * pitch.y);
        placements.push(LinePlacement { g, anchor, facing, files, lane: true });
    }
    placements
}

/// A run on the lane from `start` along `dir` behind the runs it meets
/// (`taken`, with `backs` the centre of each one's places nearest the
/// lane's back edge, `lane_runs`): the rows in front of `front` are
/// theirs, the rows behind are free. `n` men, `extra` files over the
/// fewest the free rows allow, the ranks `pitch` deep. The run is
/// widened until its men fit the rows its whole width leaves free; None
/// where no row is free behind them or the run leaves the lane. Returns
/// the files, the run and the front rank's centre across the lane.
#[allow(clippy::too_many_arguments)]
fn behind_run(
    lane: &Lane,
    taken: &[(f32, f32)],
    backs: &[f32],
    start: f32,
    dir: f32,
    n: u32,
    extra: u32,
    pitch: f32,
    rows: &dyn Fn(f32) -> u32,
    width: &dyn Fn(u32) -> f32,
) -> Option<(u32, (f32, f32), f32)> {
    // Each pass that does not return takes more files than the last, and
    // never more than `n`, which the rows always hold: the loop ends.
    let mut files = 1;
    loop {
        let w = width(files);
        let run = if dir > 0.0 { (start, start + w) } else { (start - w, start) };
        if run.0 < lane.along.0 - 0.01 || run.1 > lane.along.1 + 0.01 {
            return None;
        }
        let front = taken
            .iter()
            .zip(backs)
            .filter(|(t, _)| run.0 < t.1 && run.1 > t.0)
            .map(|(_, &b)| b - pitch)
            .fold(lane.across.1, f32::min);
        let free = rows(front);
        if free == 0 {
            return None;
        }
        let need = (n.div_ceil(free) + extra).min(n);
        if need <= files {
            return Some((files, run, front));
        }
        files = need;
    }
}

/// The stretches of the lane (x ranges) the team's regiments other than
/// `skip` stand on or are going to, each with the centre of its places
/// nearest the lane's back edge, across the lane, its block taken at its
/// facing (a block facing out has its rear rank there, one facing along
/// the lane its outer file on that side): each one whose order's point,
/// or its anchor without an order, is on the lane; and a regiment of the
/// lane that was told to attack, for as long as its men are still on it
/// (archers shooting from the wall keep their places).
fn lane_runs(groups: &Groups, lane: &Lane, team: u8, skip: &[usize]) -> Vec<(f32, f32, f32)> {
    let body = crate::sim::soldier::body_distance();
    groups
        .list
        .iter()
        .enumerate()
        .filter(|(g, gd)| gd.team == team && gd.count > 0 && !skip.contains(g) && gd.shape == FormShape::Rect)
        .filter_map(|(_, gd)| {
            let at = match gd.order {
                Some(Order::Move(p)) => p,
                Some(Order::Attack(_)) if gd.lane && lane.contains(gd.centroid) => gd.anchor,
                Some(Order::Attack(_)) => return None,
                None => gd.anchor,
            };
            if !lane.contains(at) {
                return None;
            }
            let files = if gd.files == 0 { crate::formation::default_files(gd.count) } else { gd.files };
            let pitch = crate::formation::lane_pitch(gd.chosen_spacing(), gd.kind);
            let files = files.min(gd.count as u32);
            let shift = if (gd.count as u32) > files { crate::formation::lane_stagger(gd.chosen_spacing(), gd.kind) } else { 0.0 };
            let half_w = (files - 1) as f32 * 0.5 * pitch.x + 0.5 * shift;
            let half_d = (gd.count as u32).div_ceil(files).saturating_sub(1) as f32 * 0.5 * pitch.y;
            // The block as its slots lay it, at its facing (formation.rs
            // `assign_slots`), on the lane's axes.
            let reach = crate::formation::turned_half_extents(half_w, half_d, gd.facing);
            let half = reach.x + body * 0.5;
            Some((at.x - half, at.x + half, at.y - reach.y))
        })
        .collect()
}

/// The first stretch of `width` on the lane from `start` along `dir`
/// that is clear of `taken`, slid past each run it meets and kept inside
/// the lane's ends; failing that, the first one the other way; failing
/// both, none: the lane has no room for that width.
fn free_run(lane: &Lane, taken: &[(f32, f32)], start: f32, dir: f32, width: f32) -> Option<(f32, f32)> {
    let fits = |run: (f32, f32)| {
        run.0 >= lane.along.0 - 0.01
            && run.1 <= lane.along.1 + 0.01
            && !taken.iter().any(|t| run.0 < t.1 + LINE_REG_GAP && run.1 > t.0 - LINE_REG_GAP)
    };
    let along = |start: f32, dir: f32| {
        let mut run = if dir > 0.0 { (start, start + width) } else { (start - width, start) };
        // Back inside the lane's end, if the start was near it.
        if run.1 > lane.along.1 {
            run = (lane.along.1 - width, lane.along.1);
        }
        if run.0 < lane.along.0 {
            run = (lane.along.0, lane.along.0 + width);
        }
        for _ in 0..=taken.len() {
            if fits(run) {
                return Some(run);
            }
            let hit = taken
                .iter()
                .filter(|t| run.0 < t.1 + LINE_REG_GAP && run.1 > t.0 - LINE_REG_GAP)
                .min_by(|s, t| if dir > 0.0 { s.1.total_cmp(&t.1) } else { t.0.total_cmp(&s.0) })?;
            run = if dir > 0.0 {
                (hit.1 + LINE_REG_GAP, hit.1 + LINE_REG_GAP + width)
            } else {
                (hit.0 - LINE_REG_GAP - width, hit.0 - LINE_REG_GAP)
            };
            if run.0 < lane.along.0 || run.1 > lane.along.1 {
                return None;
            }
        }
        None
    };
    along(start, dir).or_else(|| along(start, -dir))
}

fn apply_line_order(groups: &mut Groups, layout: Vec<LinePlacement>, a: Vec2, b: Vec2) {
    let n = layout.len();
    let facing = layout.first().map(|p| p.facing).unwrap_or(0.0);
    let lane = layout.first().is_some_and(|p| p.lane);
    for p in layout {
        let gd = &mut groups.list[p.g];
        line_place(gd, p.anchor, p.facing, p.files);
        gd.lane = p.lane;
        // Coming onto the lane from off it: the men take their places as
        // they arrive. Already on it, they keep their order as any
        // reform does.
        gd.lane_fill = (p.lane && (gd.centroid.y - p.anchor.y).abs() > 3.0).then_some(LaneFill::Fresh);
    }
    info!(
        "{} order: {n} regiments across {:.0} m, facing {facing:.2} rad",
        if lane { "lane" } else { "line" },
        a.distance(b),
    );
}

/// Drive the full drag-order path programmatically (the FL_TEST_FORM
/// script uses this — one code path for the mouse and the test). No
/// camera here: facing falls back to the enemy-side rule.
pub fn line_order(groups: &mut Groups, terrain: &Terrain, selected: &[usize], a: Vec2, b: Vec2) {
    let layout = line_layout(groups, terrain, selected, a, b, Vec2::ZERO, false);
    if !layout.is_empty() {
        apply_line_order(groups, layout, a, b);
    }
}

// ── Deployment placement ──
//
// Same RMB press/drag/release gestures as battle orders, but while the
// Deployment flag is up the release TELEPORTS regiments instead of
// issuing movement orders (TW deployment is instant placement), and
// everything is clamped into the player's deployment zone.

/// The player's (team 0) deployment strip: their side of the no-man's
/// land, inside the same margins the spawner keeps. Returned as
/// (min, max) corners in ground coordinates.
fn deploy_zone(terrain: &Terrain) -> (Vec2, Vec2) {
    let (centre, gap) = crate::regiments::army_lines(terrain.kind);
    let (mut lo, mut hi) = deploy_zone_for(terrain, gap);
    hi.y += centre;
    // The defender of a wall deploys on it: the zone runs to the wall's
    // outer face, so its walk is in it.
    if terrain.kind.has_wall() {
        hi.y = hi.y.max(crate::town_layout::WALL_Z + crate::town_layout::WALL_HALF);
    }
    // A setup with open sides (the demo battle) deploys out to the map's
    // sides, so a line can close the corridors past its ends.
    if crate::battle_setup::from_env().is_some_and(|s| s.open_sides) {
        lo.x = terrain.min().x + crate::battle_setup::OPEN_SIDE_MARGIN;
        hi.x = terrain.max().x - crate::battle_setup::OPEN_SIDE_MARGIN;
    }
    (lo, hi)
}

/// `deploy_zone` with the army gap given. The enemy deploys in its
/// mirror across z = 0; the grassland plants keep clear of both.
pub(crate) fn deploy_zone_for(terrain: &Terrain, army_gap: f32) -> (Vec2, Vec2) {
    use crate::regiments::{EDGE_MARGIN, SIDE_MARGIN};
    (
        Vec2::new(terrain.min().x + SIDE_MARGIN, terrain.min().y + EDGE_MARGIN),
        Vec2::new(terrain.max().x - SIDE_MARGIN, -army_gap * 0.5),
    )
}

/// Clamp an anchor so the block stays inside the zone; a block deeper or
/// wider than the zone itself sits centered on the tight axis.
/// The anchor a block of half extents `he` may be deployed at nearest
/// `anchor`: inside the deployment zone, and on a map with blocked
/// ground, wholly on open ground (the nearest such spot within 300 m,
/// else the zone clamp alone).
fn clamp_anchor(anchor: Vec2, he: Vec2, terrain: &Terrain, nav: Option<&crate::nav::NavGrid>) -> Vec2 {
    let (lo, hi) = deploy_zone(terrain);
    let axis = |v: f32, lo: f32, hi: f32| {
        if lo > hi { (lo + hi) * 0.5 } else { v.clamp(lo, hi) }
    };
    let in_zone = |a: Vec2| {
        Vec2::new(
            axis(a.x, lo.x + he.x, hi.x - he.x),
            axis(a.y, lo.y + he.y, hi.y - he.y),
        )
    };
    let a = in_zone(anchor);
    match nav {
        Some(nav) if !nav.footprint_clear(a, he) => nav
            .nearest_footprint(a, he, 300.0, |p| in_zone(p) == p)
            .unwrap_or(a),
        _ => a,
    }
}

/// Clamp a drag layout into the zone IN THE PREVIEW, so the green slots
/// always show exactly where the men will stand (TW behavior: a drag
/// outside the zone slides along its edge).
fn clamp_layout_to_zone(
    groups: &Groups,
    layout: &mut [LinePlacement],
    terrain: &Terrain,
    nav: Option<&crate::nav::NavGrid>,
) {
    for p in layout {
        let gd = &groups.list[p.g];
        if p.lane {
            // A lane's run is on the lane's own places already
            // (`lane_layout`), which the route grid's coarse cells do not
            // see: only the zone can move it, and moved it is a block on
            // the ground.
            let he = crate::formation::block_half_extents(gd, p.files, p.facing, true);
            if clamp_anchor(p.anchor, he, terrain, None) == p.anchor {
                continue;
            }
            p.lane = false;
        }
        let he = crate::formation::block_half_extents(gd, p.files, p.facing, false);
        p.anchor = clamp_anchor(p.anchor, he, terrain, nav);
    }
}

/// Deployment drag release, or a click on a lane: place the line by
/// teleport.
fn deploy_place_line(
    groups: &mut Groups,
    units: &mut Units,
    terrain: &Terrain,
    layout: Vec<LinePlacement>,
) {
    let n = layout.len();
    for p in layout {
        let gd = &mut groups.list[p.g];
        gd.anchor = p.anchor;
        gd.facing = p.facing;
        gd.files = p.files;
        gd.shape = FormShape::Rect;
        gd.order = None;
        gd.auto_order = false;
        gd.lane = p.lane;
        crate::formation::snap_to_slots(units, terrain, p.g as u32, gd);
    }
    info!("deployment: placed {n} regiments");
}

/// Deployment click: arrangement-preserving group teleport (the deploy
/// twin of `order_regiments` — same centroid translation, no marching,
/// facing kept). `picked` is the controllable selection
/// (`Selection::picked_controllable`).
fn deploy_move(
    groups: &mut Groups,
    units: &mut Units,
    terrain: &Terrain,
    nav: Option<&crate::nav::NavGrid>,
    picked: &[usize],
    target: Vec2,
) {
    if picked.is_empty() {
        return;
    }
    let centroid: Vec2 =
        picked.iter().map(|&g| groups.list[g].centroid).sum::<Vec2>() / picked.len() as f32;
    for &g in picked {
        let gd = &mut groups.list[g];
        let he = crate::formation::block_half_extents(gd, gd.files, gd.facing, false);
        gd.anchor = clamp_anchor(target + (gd.centroid - centroid), he, terrain, nav);
        gd.order = None;
        gd.auto_order = false;
        gd.lane = false;
        crate::formation::snap_to_slots(units, terrain, g as u32, gd);
    }
    info!("deployment: moved {} regiments", picked.len());
}

/// The deployment zone, drawn as terrain-following gizmo lines: gold
/// boundary toward the enemy, dimmer side and back edges.
fn draw_deploy_zone(terrain: Res<Terrain>, mut gizmos: Gizmos) {
    let (lo, hi) = deploy_zone(&terrain);
    let gold = Color::srgba(0.95, 0.80, 0.30, 0.9);
    let dim = Color::srgba(0.95, 0.80, 0.30, 0.35);
    let lift = |q: Vec2| Vec3::new(q.x, terrain.height_at(q.x, q.y) + 0.5, q.y);
    let mut ground_line = |a: Vec2, b: Vec2, col: Color| {
        let steps = (a.distance(b) / 8.0).ceil().max(1.0) as usize;
        let mut prev = lift(a);
        for s in 1..=steps {
            let p = lift(a.lerp(b, s as f32 / steps as f32));
            gizmos.line(prev, p, col);
            prev = p;
        }
    };
    // Front (facing the enemy), then sides and back.
    ground_line(Vec2::new(lo.x, hi.y), Vec2::new(hi.x, hi.y), gold);
    ground_line(Vec2::new(lo.x, lo.y), Vec2::new(lo.x, hi.y), dim);
    ground_line(Vec2::new(hi.x, lo.y), Vec2::new(hi.x, hi.y), dim);
    ground_line(Vec2::new(lo.x, lo.y), Vec2::new(hi.x, lo.y), dim);
}

/// RMB order issuing: press -> (optional drag with live preview) ->
/// release. A short click keeps the classic behavior (attack the regiment
/// under the cursor, else arrangement-preserving group move); a drag
/// beyond ORDER_DRAG_MIN paints the new front line.
#[allow(clippy::too_many_arguments)] // bevy system params
fn issue_order(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform)>,
    terrain: Res<Terrain>,
    mut groups: ResMut<Groups>,
    mut units: ResMut<Units>,
    selection: Res<Selection>,
    mut drag: ResMut<OrderDrag>,
    deploy: Res<crate::game_state::Deployment>,
    mut cues: MessageWriter<crate::audio::UiCue>,
    ui: Query<&Interaction>,
    routes: Res<crate::sim::route::Routes>,
    time: Res<Time>,
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

    // An order press must start on the map: a right-click on the HUD
    // (card bar, control panel) never falls through to the terrain
    // behind it. A drag that STARTED on the map may pass over the HUD.
    if buttons.just_pressed(MouseButton::Right)
        && !crate::unit_cards::pointer_over_ui(ui.iter())
    {
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
            // M2TW drag-hand facing: project the drag onto the camera's
            // ground axes; a rightward drag on screen faces the camera's
            // forward side (away from the viewer), leftward faces back.
            let cam_fwd = cam_tf.forward();
            let cam_right = cam_tf.right();
            let fwd_g = Vec2::new(cam_fwd.x, cam_fwd.z).normalize_or_zero();
            let right_g = Vec2::new(cam_right.x, cam_right.z).normalize_or_zero();
            let hand = (cur - start).dot(right_g);
            let face_hint = fwd_g * if hand >= 0.0 { 1.0 } else { -1.0 };
            // Ctrl held: the blocks stand with no gap between them. Read
            // every frame, so the preview follows the key and the release
            // applies the layout last shown.
            let tight = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
            drag.layout = line_layout(&groups, &terrain, &selected, start, cur, face_hint, tight);
            if deploy.active {
                clamp_layout_to_zone(&groups, &mut drag.layout, &terrain, routes.grid());
            }
        }
    }
    if !buttons.just_released(MouseButton::Right) {
        return;
    }
    let Some(start) = drag.start.take() else { return };
    let t = time.elapsed_secs();

    if drag.active {
        drag_release(&mut groups, &mut units, &terrain, &mut drag, &selected, deploy.active, start, &mut cues, t);
    } else {
        click_release(&mut groups, &mut units, &terrain, &routes, &selection, &selected, deploy.active, start, &mut cues, t);
    }
    drag.active = false;
    drag.layout.clear();
}

/// A drag let go: its line laid out as the preview showed it.
#[allow(clippy::too_many_arguments)]
#[inline]
fn drag_release(
    groups: &mut Groups,
    units: &mut Units,
    terrain: &Terrain,
    drag: &mut OrderDrag,
    selected: &[usize],
    deploy_active: bool,
    start: Vec2,
    cues: &mut MessageWriter<crate::audio::UiCue>,
    t: f32,
) {
    let layout = std::mem::take(&mut drag.layout);
    // The regiments of the selection the drag placed nowhere: a lane
    // with no room for them. They keep the orders they had.
    let unplaced = selected
        .iter()
        .filter(|&&g| {
            let gd = &groups.list[g];
            gd.count > 0 && !gd.state.is_broken() && !layout.iter().any(|p| p.g == g)
        })
        .count();
    if unplaced > 0 {
        info!("no room on the lane: {unplaced} of {} regiments got no order", selected.len());
    }
    if layout.is_empty() {
        // Nobody placed: no order, and never a click's order instead.
    } else if deploy_active {
        deploy_place_line(groups, units, terrain, layout);
        cues.write(crate::audio::UiCue::Deploy);
    } else {
        for p in &layout {
            let what = format!(
                "line to ({:.1}, {:.1}) facing {:.3} files {}",
                p.anchor.x, p.anchor.y, p.facing, p.files
            );
            log_order(t, p.g, &groups.list[p.g], &what);
        }
        apply_line_order(groups, layout, start, drag.cur);
        cues.write(crate::audio::UiCue::Move);
    }
}

/// A click let go: a placement while deploying, an attack on the enemy
/// regiment under it, or a move.
#[allow(clippy::too_many_arguments)]
#[inline]
fn click_release(
    groups: &mut Groups,
    units: &mut Units,
    terrain: &Terrain,
    routes: &crate::sim::route::Routes,
    selection: &Selection,
    selected: &[usize],
    deploy_active: bool,
    start: Vec2,
    cues: &mut MessageWriter<crate::audio::UiCue>,
    t: f32,
) {
    if deploy_active {
        // No attack targets while deploying: every click is a placement,
        // on a lane the lane's places from the click on, as in battle
        // (`click_order`).
        let picked: Vec<usize> = selection.picked_controllable(groups).collect();
        if lane_at(terrain, start).is_some() {
            let mut layout = line_layout(groups, terrain, &picked, start, start, Vec2::ZERO, false);
            clamp_layout_to_zone(groups, &mut layout, terrain, routes.grid());
            if layout.is_empty() {
                info!("no room on the lane at ({:.0}, {:.0}): {} regiments placed nowhere", start.x, start.y, picked.len());
                return;
            }
            deploy_place_line(groups, units, terrain, layout);
        } else {
            deploy_move(groups, units, terrain, routes.grid(), &picked, start);
        }
        cues.write(crate::audio::UiCue::Deploy);
    } else if let Some(e) = enemy_regiment_at(groups, start) {
        attack_regiments(groups, selected, e);
        cues.write(crate::audio::UiCue::Attack);
        info!(
            "{} regiments ({} units) ATTACK regiment {e}",
            selected.len(),
            selection.count_units,
        );
        for &g in selected {
            let gd = &groups.list[g];
            if gd.order == Some(Order::Attack(e)) && !gd.state.is_broken() {
                log_order(t, g, gd, &format!("attack g{e}"));
            }
        }
    } else if !click_order(groups, terrain, selected, start) {
        info!("no room on the lane at ({:.0}, {:.0}): {} regiments got no order", start.x, start.y, selected.len());
    } else {
        cues.write(crate::audio::UiCue::Move);
        info!(
            "{} regiments ({} units) move to ({:.0}, {:.0})",
            selected.len(),
            selection.count_units,
            start.x,
            start.y
        );
        // Each regiment's own destination (the click translated by its
        // offset from the selection's centre) and the facing it took.
        for &g in selected {
            let gd = &groups.list[g];
            if let Some(Order::Move(d)) = gd.order
                && !gd.state.is_broken()
            {
                log_order(t, g, gd, &format!("move to ({:.1}, {:.1}) facing {:.3}", d.x, d.y, gd.facing));
            }
        }
    }
}

/// Placement preview while a formation drag is in flight: the drawn line
/// plus a soft green marker per soldier slot (block outlines above the
/// gizmo budget), so the player can tune width and facing before letting
/// go. Player-facing UI, drawn with the debug-viz toggle on or off. On a lane
/// (`lane_layout`) the drawn line is left out: the runs are laid in the
/// lane's rows, not along the drag, and only the places show.
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

    if !drag.layout.iter().any(|p| p.lane) {
        gizmos.line(lift(start, 0.4), lift(drag.cur, 0.4), line_col);
    }

    let total: usize = drag.layout.iter().map(|p| groups.list[p.g].count).sum();
    for p in &drag.layout {
        let gd = &groups.list[p.g];
        let fwd = facing_dir(p.facing);
        let right = Vec2::new(fwd.y, -fwd.x);
        let (pitch, stagger) = if p.lane {
            (
                crate::formation::lane_pitch(gd.chosen_spacing(), gd.kind),
                crate::formation::lane_stagger(gd.chosen_spacing(), gd.kind),
            )
        } else {
            (gd.chosen_spacing().pitch(gd.kind), 0.0)
        };
        let world = |off: Vec2| p.anchor + right * off.x + fwd * off.y;

        if total <= PREVIEW_CIRCLE_CAP {
            for off in slot_offsets(gd.count, p.files as usize, pitch, stagger) {
                gizmos.circle(
                    Isometry3d::new(
                        lift(world(off), 0.25),
                        Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                    ),
                    crate::selection_rings::RADIUS,
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
    time: Res<Time>,
) {
    if !keys.just_pressed(KeyCode::Backspace) {
        return;
    }
    halt_selected(&selection, &mut groups, time.elapsed_secs());
}

/// Halt one regiment: its order dropped, its anchor where its block
/// stands. Shared by `halt_selected` and a setup file's `halt` order.
pub fn halt_regiment(gd: &mut GroupData) {
    gd.anchor = gd.centroid;
    gd.order = None;
    gd.lane_fill = None;
}

/// Shared by the Backspace hotkey and the Halt button (unit_cards.rs);
/// `t` is the game clock for the order log.
pub fn halt_selected(selection: &Selection, groups: &mut Groups, t: f32) {
    if selection.count_units == 0 {
        return;
    }
    let mut halted = 0;
    for (g, sel) in selection.regiments.iter().enumerate() {
        let group = &mut groups.list[g];
        if *sel && group.count > 0 && !group.state.is_broken() {
            halt_regiment(group);
            log_order(t, g, group, &format!("halt at ({:.1}, {:.1})", group.anchor.x, group.anchor.y));
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
    let _tick_log = crate::frame_log::tick::time(crate::frame_log::tick::Part::ClearOrders);
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
    viz: Res<crate::sim::DebugViz>,
    groups: Res<Groups>,
    selection: Res<Selection>,
    terrain: Res<Terrain>,
    mut gizmos: Gizmos,
) {
    // Attack indicators are player-facing UI (not debug viz): a red marker
    // hangs over every regiment the selected regiments are attacking.
    let mut marked = vec![false; groups.list.len()];
    for (g, group) in groups.list.iter().enumerate() {
        if group.team == PLAYER_TEAM
            && selection.regiments.get(g).copied().unwrap_or(false)
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
    // Move-order targets of the selected regiments (attack orders have
    // the marker above). Every regiment's would pile up on the map.
    for (g, group) in groups.list.iter().enumerate() {
        if !selection.regiments.get(g).copied().unwrap_or(false) {
            continue;
        }
        if let Some(Order::Move(t)) = group.order {
            let base = Vec3::new(t.x, terrain.height_at(t.x, t.y) + 0.5, t.y);
            let color = Color::srgb(0.4, 0.8, 1.0);
            gizmos.circle(
                Isometry3d::new(base, Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
                3.0,
                color,
            );
            gizmos.line(base, base + Vec3::Y * 10.0, color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wall's walk as the game lays it, and a regiment of archers
    /// behind the wall.
    fn walk() -> Lane {
        crate::town_layout::walk_lane(0.5 * crate::sim::soldier::body_distance())
    }

    fn archers(men: usize, at: Vec2) -> GroupData {
        let mut gd = GroupData::new(0, crate::unit_types::KIND_ARCHER, at, men);
        gd.shape = FormShape::Rect;
        gd.files = crate::formation::default_files(men);
        gd.centroid = at;
        gd
    }

    /// The lane order as the drag gives it: the placements, applied.
    fn lane_order(groups: &mut Groups, lane: &Lane, g: usize, a: f32, b: f32) -> bool {
        let z = crate::town_layout::WALL_Z;
        let layout = lane_layout(groups, lane, &[g], Vec2::new(a, z), Vec2::new(b, z));
        let placed = !layout.is_empty();
        apply_line_order(groups, layout, Vec2::new(a, z), Vec2::new(b, z));
        placed
    }

    /// A regiment's run on the lane: its outer centres, the ranks' shift
    /// included, less and plus half a body.
    fn run_of(gd: &GroupData) -> (f32, f32) {
        let Some(Order::Move(at)) = gd.order else { panic!("no order") };
        let shift = if gd.count > gd.files as usize { crate::formation::stagger_of(gd) } else { 0.0 };
        let half = (gd.files - 1) as f32 * 0.5 * crate::formation::pitch_of(gd).x + 0.5 * shift + 0.5 * crate::sim::soldier::body_distance();
        (at.x - half, at.x + half)
    }

    #[test]
    fn a_run_slides_past_a_taken_one_and_stays_on_the_lane() {
        let lane = walk();
        assert_eq!(free_run(&lane, &[(0.0, 50.0)], 10.0, 1.0, 30.0), Some((53.0, 83.0)));
        // No room on that side of the taken run: the other way.
        let run = free_run(&lane, &[(100.0, 150.0)], 120.0, 1.0, 30.0).expect("room to the west");
        assert_eq!(run, (67.0, 97.0));
        // A start past the lane's end comes back inside it.
        let run = free_run(&lane, &[], 150.0, 1.0, 30.0).expect("an empty lane");
        assert!((run.1 - lane.along.1).abs() < 1e-4 && (run.1 - run.0 - 30.0).abs() < 1e-4);
    }

    #[test]
    fn a_lane_with_no_room_gives_no_run() {
        let lane = walk();
        // One regiment holds from the gate to the east end, the next is
        // dragged westward from west of the gate and needs 174.8 m of the
        // 136 m left.
        assert_eq!(free_run(&lane, &[(-16.4, 158.4)], -39.3, -1.0, 174.8), None);
        // Wider than the lane itself.
        assert_eq!(free_run(&lane, &[], 0.0, 1.0, lane.along.1 - lane.along.0 + 1.0), None);
    }

    /// Three regiments of 500 sent onto the walk one after the other: the
    /// first two fit, each whole on the lane and clear of the other; the
    /// third has no room, gets no order and stays what it was.
    #[test]
    fn a_regiment_the_lane_has_no_room_for_gets_no_order() {
        let lane = walk();
        let behind = Vec2::new(0.0, crate::town_layout::WALL_Z - 50.0);
        let mut groups = Groups { list: (0..3).map(|_| archers(500, behind)).collect() };
        assert!(lane_order(&mut groups, &lane, 0, 100.0, 108.0));
        assert!(lane_order(&mut groups, &lane, 1, -39.3, -77.3));
        let (a, b) = (run_of(&groups.list[0]), run_of(&groups.list[1]));
        for run in [a, b] {
            assert!(run.0 >= lane.along.0 - 0.01 && run.1 <= lane.along.1 + 0.01, "the run {run:?} is on the lane");
        }
        assert!(b.1 + LINE_REG_GAP <= a.0 + 0.01, "the runs {b:?} and {a:?} are clear of each other");
        // The fewest files at the lane's pitch, 84 in six ranks, and one
        // more for each pitch of drag: 7 for the 8 m drag, 34 for the 38 m.
        for (g, grid) in [(0, (91, 6)), (1, (118, 5))] {
            let gd = &groups.list[g];
            assert!(gd.lane, "regiment {g} is a lane's");
            assert_eq!((gd.files, crate::formation::grid(gd).1), grid, "regiment {g}");
        }
        assert!(!lane_order(&mut groups, &lane, 2, 0.0, 8.0));
        let third = &groups.list[2];
        assert!(third.order.is_none() && !third.lane && third.lane_fill.is_none());
        // A click does the same, and is no march to the point on the wall.
        let terrain = crate::terrain::build_terrain(crate::terrain::MapKind::WallDebug);
        assert!(!click_order(&mut groups, &terrain, &[2], Vec2::new(0.0, crate::town_layout::WALL_Z)));
        assert!(groups.list[2].order.is_none());
    }

    /// A drag wider than the room left takes the least width the lane's
    /// ranks allow instead of no place at all.
    #[test]
    fn a_drag_too_wide_for_the_room_takes_the_least_width() {
        let lane = walk();
        let behind = Vec2::new(0.0, crate::town_layout::WALL_Z - 50.0);
        let mut groups = Groups { list: (0..2).map(|_| archers(200, behind)).collect() };
        // The first takes all but 60 m at the west end, in one rank or two.
        assert!(lane_order(&mut groups, &lane, 0, lane.along.0 + 60.0, lane.along.1));
        // The second is dragged 100 m into the 57 m left: 34 files by 6.
        assert!(lane_order(&mut groups, &lane, 1, lane.along.0, lane.along.0 + 100.0));
        let gd = &groups.list[1];
        assert_eq!(gd.files, 34);
        let (a, b) = (run_of(&groups.list[0]), run_of(gd));
        assert!(b.0 >= lane.along.0 - 0.01 && b.1 + LINE_REG_GAP <= a.0 + 0.01, "the runs {b:?} and {a:?}");
    }

    /// A regiment in block order standing at `at`.
    fn regiment(kind: u8, men: usize, at: Vec2, spacing: crate::formation::FormSpacing) -> GroupData {
        let mut gd = GroupData::new(0, kind, at, men);
        gd.shape = FormShape::Rect;
        gd.files = crate::formation::default_files(men);
        gd.centroid = at;
        gd.spacing = spacing;
        gd
    }

    /// A drag along x from -half to half on open ground, with the
    /// regiments' places as (distance along the line from its start,
    /// width of `files` pitches, files).
    fn drag_x(groups: &Groups, half: f32, tight: bool) -> Vec<(f32, f32, u32)> {
        let terrain = crate::terrain::build_terrain(crate::terrain::MapKind::Grassland);
        let picked: Vec<usize> = (0..groups.list.len()).collect();
        let (a, b) = (Vec2::new(-half, 0.0), Vec2::new(half, 0.0));
        let layout = line_layout(groups, &terrain, &picked, a, b, Vec2::ZERO, tight);
        assert_eq!(layout.len(), groups.list.len());
        layout
            .iter()
            .map(|p| {
                let gd = &groups.list[p.g];
                let pitch = gd.chosen_spacing().pitch(gd.kind);
                // The front rank stands on the drawn line.
                let ranks = (gd.count as u32).div_ceil(p.files);
                assert!((p.anchor.y.abs() - (ranks - 1) as f32 * 0.5 * pitch.y).abs() < 1e-3);
                (p.anchor.x + half, p.files as f32 * pitch.x, p.files)
            })
            .collect()
    }

    /// Neighbouring blocks touch: their centres are half the sum of their
    /// widths apart, and the run is centred on the drag's midpoint.
    fn assert_tight(places: &[(f32, f32, u32)], len: f32) {
        for w in places.windows(2) {
            let (gap, need) = (w[1].0 - w[0].0, 0.5 * (w[0].1 + w[1].1));
            assert!((gap - need).abs() < 1e-3, "centres {gap} apart, widths need {need}");
        }
        let first = places.first().unwrap();
        let last = places.last().unwrap();
        let mid = 0.5 * ((first.0 - 0.5 * first.1) + (last.0 + 0.5 * last.1));
        assert!((mid - 0.5 * len).abs() < 1e-3, "the run's middle {mid} is off the drag's {}", 0.5 * len);
    }

    fn three_heavies() -> Groups {
        let normal = crate::formation::FormSpacing::Normal;
        Groups {
            list: [-40.0, 0.0, 40.0]
                .map(|x| regiment(crate::unit_types::KIND_HEAVY, 120, Vec2::new(x, -20.0), normal))
                .into(),
        }
    }

    /// Without Ctrl each regiment's block is centred in its share of the
    /// drag, with LINE_REG_GAP taken off its files.
    #[test]
    fn a_line_without_ctrl_keeps_its_gaps() {
        let groups = three_heavies();
        let pitch = crate::formation::FormSpacing::Normal.pitch(crate::unit_types::KIND_HEAVY).x;
        let files = ((40.0 - LINE_REG_GAP) / pitch).floor() as u32;
        let places = drag_x(&groups, 60.0, false);
        for (place, centre) in places.iter().zip([20.0, 60.0, 100.0]) {
            assert_eq!(place.2, files);
            assert!((place.0 - centre).abs() < 1e-4, "centre {} for {centre}", place.0);
        }
    }

    #[test]
    fn a_line_with_ctrl_has_no_gaps() {
        let groups = three_heavies();
        let places = drag_x(&groups, 60.0, true);
        let pitch = crate::formation::FormSpacing::Normal.pitch(crate::unit_types::KIND_HEAVY).x;
        for place in &places {
            assert_eq!(place.2, (40.0 / pitch).round() as u32);
        }
        assert_tight(&places, 120.0);
    }

    /// Regiments of different pitches: each block is its own files times
    /// its own pitch wide, and still none has a gap to the next.
    #[test]
    fn a_line_with_ctrl_has_no_gaps_across_pitches() {
        use crate::formation::FormSpacing;
        let groups = Groups {
            list: vec![
                regiment(crate::unit_types::KIND_HEAVY, 120, Vec2::new(-40.0, -20.0), FormSpacing::Normal),
                regiment(crate::unit_types::KIND_HEAVY, 80, Vec2::new(0.0, -20.0), FormSpacing::Loose),
                regiment(crate::unit_types::KIND_ARCHER, 100, Vec2::new(40.0, -20.0), FormSpacing::Normal),
            ],
        };
        let pitch = |g: usize| groups.list[g].chosen_spacing().pitch(groups.list[g].kind).x;
        assert!((pitch(0) - pitch(1)).abs() > 0.5, "the two spacings differ in pitch");
        let places = drag_x(&groups, 75.0, true);
        assert_tight(&places, 150.0);
    }

    /// A drag far shorter than the men need: one file each, side by side.
    #[test]
    fn a_short_line_with_ctrl_has_no_gaps() {
        let groups = three_heavies();
        let places = drag_x(&groups, 3.0, true);
        assert!(places.iter().all(|p| p.2 >= 1));
        assert_tight(&places, 6.0);
    }

    /// While deploying, a 500-man regiment dragged along the walk, or
    /// clicked onto it, stands on the walk: every man in the walk's band at
    /// its height, six ranks deep for a short drag, fewer for a long one.
    #[test]
    fn a_regiment_deploys_onto_the_walk() {
        let terrain = crate::terrain::build_terrain(crate::terrain::MapKind::WallDebug);
        let nav = crate::nav::NavGrid::build(&terrain);
        let z = crate::town_layout::WALL_Z;
        let behind = Vec2::new(0.0, z - 40.0);
        for (len, ranks) in [(0.0, 6), (30.0, 5), (150.0, 3)] {
            let mut groups = Groups { list: vec![archers(500, behind)] };
            let mut units = Units::default();
            for k in 0..500 {
                let p = behind + Vec2::new((k % 50) as f32 * 1.4 - 35.0, (k / 50) as f32 * 1.4);
                crate::units::push_unit(&mut units, &terrain, k, 0, crate::unit_types::KIND_ARCHER, 0, p.x, p.y, behind);
            }
            let (a, b) = (Vec2::new(-60.0, z + 1.0), Vec2::new(-60.0 + len, z - 1.0));
            let mut layout = line_layout(&groups, &terrain, &[0], a, b, Vec2::ZERO, false);
            clamp_layout_to_zone(&groups, &mut layout, &terrain, nav.as_ref());
            assert!(layout.len() == 1 && layout[0].lane, "a {len} m drag is a lane's");
            deploy_place_line(&mut groups, &mut units, &terrain, layout);
            let gd = &groups.list[0];
            assert!(gd.lane);
            assert_eq!(crate::formation::grid(gd).1, ranks, "ranks for a {len} m drag");
            for i in 0..units.len() {
                let p = units.pos[i];
                assert!((p.z - z).abs() <= 3.0, "man {i} at {p} is on the walk's band");
                let walk = crate::town_layout::WALK + terrain.ground_at(p.x, p.z);
                assert!(p.y > walk, "man {i} at {p} stands up on the walk ({walk} m)");
            }
        }
    }

    /// The drag widens a block on the walk from its deepest at once: a
    /// click gives six ranks, every pitch of drag a file more.
    #[test]
    fn a_short_drag_on_the_walk_already_widens_the_block() {
        let lane = walk();
        let behind = Vec2::new(0.0, crate::town_layout::WALL_Z - 50.0);
        let pitch = crate::formation::lane_pitch(crate::formation::FormSpacing::Normal, crate::unit_types::KIND_ARCHER).x;
        for (len, files) in [(0.0, 34), (pitch * 3.0 + 0.01, 37), (20.0, 34 + (20.0 / pitch) as u32)] {
            let mut groups = Groups { list: vec![archers(200, behind)] };
            assert!(lane_order(&mut groups, &lane, 0, 0.0, len));
            assert_eq!(groups.list[0].files, files, "a {len} m drag");
        }
    }

    /// A drag that starts on a thin regiment's run on the walk places the
    /// next one behind it, in the rows it leaves free; one that starts
    /// beside it goes beside it.
    #[test]
    fn a_drag_on_a_thin_regiment_places_the_next_behind_it() {
        let lane = walk();
        let behind = Vec2::new(0.0, crate::town_layout::WALL_Z - 50.0);
        let mut groups = Groups { list: (0..3).map(|_| archers(100, behind)).collect() };
        // Two ranks of 50 at the merlons, x 0 to about 55.
        assert!(lane_order(&mut groups, &lane, 0, 0.0, 55.0));
        let first = &groups.list[0];
        assert_eq!(crate::formation::grid(first).1, 2);
        let (run, Some(Order::Move(at))) = (run_of(first), first.order) else { panic!("no order") };
        let pitch = crate::formation::pitch_of(first).y;
        // Started on it: behind it, over the same stretch, its front rank
        // one rank behind the first one's rear rank.
        assert!(lane_order(&mut groups, &lane, 1, 10.0, 10.0));
        let second = &groups.list[1];
        let Some(Order::Move(at2)) = second.order else { panic!("no order") };
        let ranks2 = crate::formation::grid(second).1;
        assert!(ranks2 <= 4, "{ranks2} ranks behind two");
        let first_back = at.y - 0.5 * pitch;
        let second_front = at2.y + (ranks2 - 1) as f32 * 0.5 * pitch;
        assert!((first_back - pitch - second_front).abs() < 1e-3, "front {second_front} against the first's back {first_back}");
        let r2 = run_of(second);
        assert!(r2.0 >= run.0 - 0.01 && r2.0 < run.1, "the run {r2:?} starts over {run:?}");
        // Started beside it: at the front, clear of it.
        assert!(lane_order(&mut groups, &lane, 2, 80.0, 80.0));
        let third = &groups.list[2];
        let Some(Order::Move(at3)) = third.order else { panic!("no order") };
        assert!(at3.y > at2.y, "the third stands at the front");
        assert!(run_of(third).0 >= run.1, "and beside the first");
    }

    /// Ctrl changes nothing on the wall's walk.
    #[test]
    fn a_lane_drag_ignores_ctrl() {
        let terrain = crate::terrain::build_terrain(crate::terrain::MapKind::WallDebug);
        let behind = Vec2::new(0.0, crate::town_layout::WALL_Z - 50.0);
        let groups = Groups { list: vec![archers(200, behind), archers(200, behind + Vec2::X)] };
        let (a, b) = (Vec2::new(0.0, crate::town_layout::WALL_Z), Vec2::new(60.0, crate::town_layout::WALL_Z));
        let key = |p: &LinePlacement| (p.g, p.anchor.to_array(), p.facing, p.files, p.lane);
        let loose: Vec<_> = line_layout(&groups, &terrain, &[0, 1], a, b, Vec2::ZERO, false).iter().map(key).collect();
        let tight: Vec<_> = line_layout(&groups, &terrain, &[0, 1], a, b, Vec2::ZERO, true).iter().map(key).collect();
        assert!(!loose.is_empty() && loose.iter().all(|p| p.4), "the drag is a lane's");
        assert_eq!(loose, tight);
    }

    /// Bowmen told to shoot from the walk keep their run: nobody else is
    /// placed on it while they still stand there.
    #[test]
    fn a_regiment_shooting_from_the_lane_keeps_its_run() {
        let lane = walk();
        let behind = Vec2::new(0.0, crate::town_layout::WALL_Z - 50.0);
        let mut groups = Groups { list: (0..2).map(|_| archers(200, behind)).collect() };
        assert!(lane_order(&mut groups, &lane, 0, 0.0, 8.0));
        let held = run_of(&groups.list[0]);
        let gd = &mut groups.list[0];
        let Some(Order::Move(at)) = gd.order else { panic!("no order") };
        (gd.anchor, gd.centroid) = (at, at);
        gd.order = Some(Order::Attack(1));
        let runs: Vec<(f32, f32)> = lane_runs(&groups, &lane, 0, &[1]).iter().map(|r| (r.0, r.1)).collect();
        assert_eq!(runs, vec![held]);
        // Once its men have left the walk the run is free.
        groups.list[0].centroid = behind;
        assert!(lane_runs(&groups, &lane, 0, &[1]).is_empty());
    }

    /// A regiment holding on the walk turned to face along it: its run is
    /// its two ranks' depth along the walk, its places nearest the walk's
    /// back edge are its outer file on that side, and the next regiment
    /// dragged from beside that run is placed clear of it.
    #[test]
    fn a_block_facing_along_the_walk_holds_its_depth() {
        let lane = walk();
        let at = Vec2::new(0.0, crate::town_layout::WALL_Z);
        let behind = Vec2::new(0.0, crate::town_layout::WALL_Z - 50.0);
        let mut held = archers(100, at);
        held.files = 50;
        held.anchor = at;
        held.facing = crate::formation::facing_of(Vec2::X);
        let mut groups = Groups { list: vec![held, archers(100, behind)] };
        let gd = &groups.list[0];
        let pitch = crate::formation::lane_pitch(gd.chosen_spacing(), gd.kind);
        let stagger = crate::formation::lane_stagger(gd.chosen_spacing(), gd.kind);
        let half = 0.5 * pitch.y + 0.5 * crate::sim::soldier::body_distance();
        let back = at.y - (49.0 * 0.5 * pitch.x + 0.5 * stagger);
        let runs = lane_runs(&groups, &lane, 0, &[1]);
        assert_eq!(runs.len(), 1);
        let (lo, hi, rear) = runs[0];
        assert!((lo - (at.x - half)).abs() < 1e-4 && (hi - (at.x + half)).abs() < 1e-4, "the run ({lo}, {hi}) is the block's depth");
        assert!((rear - back).abs() < 1e-4, "the rear {rear} is the outer file at {back}");
        assert!(lane_order(&mut groups, &lane, 1, lo - 1.0, lo - 1.0));
        let next = run_of(&groups.list[1]);
        assert!(
            next.0 >= hi + LINE_REG_GAP - 0.01 || next.1 <= lo - LINE_REG_GAP + 0.01,
            "the run {next:?} is clear of ({lo}, {hi})"
        );
    }
}
