//! Rigid regiment formations: a slot generator writing into the existing
//! `units.home` column. A regiment
//! order stays ONE point; formations only change the per-unit offset from
//! it — the movement hot path is untouched.
//!
//! Shapes: `Rect` (rank-and-file grid with facing, the default) and `Blob`
//! (the pre-formations look: spawn-captured or regenerated loose cluster —
//! kept for low-morale levies later and for the FL_TEST_* geometry).
//! Spacing modes: Normal, Loose (missile dispersion later), Wall
//! (shieldwall for sword kinds, spearwall for spears: tight files),
//! Open (columns with lanes between them while a friend passes through;
//! the officer's order, never the player's).
//!
//! Slots are recomputed on EVENTS (new order, mode toggle, rally, close
//! ranks), never per tick: `reform` on `GroupData` requests it, and
//! `apply_reforms` runs before the sim step so fresh offsets take effect
//! the tick they were asked for.

use bevy::prelude::*;

use crate::orders::{GroupData, Groups};
use crate::units::{Units, hash01};

/// Slot pitch inside a regiment block in normal order, every kind's
/// default (per-mode multipliers below).
pub const BASE_SPACING: f32 = 1.4;

/// A kind's slot pitch in normal order: how far apart its men stand.
/// FL_GAP=k,m,s,a sets it per kind in metres (knights, men-at-arms,
/// spearmen, bowmen). Never under the separation rest distance at ease
/// (sim/soldier.rs `sep_ease`): the physics pushes closer slots apart.
pub fn gap(kind: u8) -> f32 {
    static GAPS: std::sync::OnceLock<[f32; crate::unit_types::NUM_KINDS]> =
        std::sync::OnceLock::new();
    GAPS.get_or_init(|| {
        let mut gaps = [BASE_SPACING; crate::unit_types::NUM_KINDS];
        if let Ok(v) = std::env::var("FL_GAP") {
            let parsed: Vec<f32> = v.split(',').filter_map(|t| t.trim().parse().ok()).collect();
            match <[f32; crate::unit_types::NUM_KINDS]>::try_from(parsed) {
                Ok(given) => {
                    let floor = crate::sim::soldier::sep_ease();
                    if given.iter().any(|g| *g < floor) {
                        warn!("FL_GAP: a gap under {floor} m cannot be held, raised to it");
                    }
                    gaps = given.map(|g| g.clamp(floor, 6.0));
                }
                Err(_) => warn!(
                    "FL_GAP wants {} comma-separated numbers",
                    crate::unit_types::NUM_KINDS
                ),
            }
        }
        gaps
    })[kind as usize]
}

/// The widest normal-order pitch of any kind: army layouts size every
/// block's ground with it.
pub fn widest_gap() -> f32 {
    (0..crate::unit_types::NUM_KINDS as u8).map(gap).fold(0.0, f32::max)
}
/// Files-to-ranks bias of the default block (~2.2:1, wider than deep).
const DEFAULT_ASPECT: f32 = 2.2;
/// Close ranks when this fraction of the last-formed strength has fallen
/// (and at least 8 men, so skirmish dribble doesn't churn the grid).
const CLOSE_RANKS_FRAC: f32 = 0.06;
/// Ticks between close-ranks checks per regiment (staggered by index).
/// Nobody dresses while engaged: the fight owns the block until the
/// melee ends and the regiment re-forms.
const CLOSE_RANKS_PERIOD: u32 = 90;

/// Formation shape of a regiment.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum FormShape {
    /// Rank-and-file grid with facing — the default battle formation.
    Rect,
    /// Loose cluster: spawn-captured offsets (test geometry) or a
    /// regenerated jittered disc (B toggle). Facing is meaningless.
    Blob,
}

/// Two men on a lane whose places cross by more than this (m) along it
/// trade places (`fill_lane_slots`).
const LANE_CROSS: f32 = 2.0;

/// A regiment coming onto a lane (town_layout.rs `Lane`) fills its run
/// from the far end as its men arrive: `fill_lane_slots`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LaneFill {
    /// Ordered onto the lane; its slots are not written yet.
    Fresh,
    /// The men (unit indices) still on their way to the lane, each holding
    /// a place another arrival may take over.
    Pending(Vec<u32>),
}

/// Slot density of a Rect formation.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum FormSpacing {
    Normal,
    /// Open order: spread against missiles (arrows later), weak in melee.
    Loose,
    /// Shieldwall (sword kinds) / spearwall (spears): tight files, braced.
    Wall,
    /// Groups of files shoulder to shoulder with a lane between groups,
    /// across the block's frontage, the men stacked front to back in each
    /// file, so a friendly regiment can walk through the lanes
    /// (`OpenOrder`, sim/job.rs `open_for_passers`). The spacing to go
    /// back to is `GroupData::close_to`.
    Open,
}

/// The most files a group of the open order stands.
const OPEN_FILES_MAX: u8 = 4;
/// Free ground the open order leaves before a block in front or behind
/// (metres).
const OPEN_ROOM_MARGIN: f32 = 3.0;
/// Ground further than this counts as all the room there is (metres).
pub const OPEN_ROOM_MAX: f32 = 60.0;

/// How an open regiment stands: `files` files to a group, the men of a
/// group `sep` apart (their spring rest), the groups `pitch` apart from
/// the first man of one to the first of the next, the block grown
/// `front` metres ahead of its closed front and `back` behind its closed
/// rear (negative when it stands shorter), and the anchor it had closed.
#[derive(Clone, Copy, Default)]
pub struct OpenOrder {
    pub files: u8,
    pub sep: f32,
    pub pitch: f32,
    /// The lanes run across the facing, not along it: for a friend
    /// crossing the block from the side. The groups are then groups of
    /// ranks, each rank keeps the closed frontage, and the block grows
    /// deeper as it does for lanes along the facing.
    pub across: bool,
    /// Metres grown past the closed front and behind the closed rear,
    /// negative when the block stands shorter.
    pub front: f32,
    pub back: f32,
    pub closed_anchor: Vec2,
}

impl OpenOrder {
    /// How far the anchor stands ahead of the closed one: half the
    /// difference between the growth in front and behind.
    pub fn shift(&self) -> f32 {
        0.5 * (self.front - self.back)
    }
}

impl FormSpacing {
    /// (lateral, depth) slot pitch of a kind in meters. Wall matches the
    /// tightened separation rest distance (sim/soldier.rs WALL_SEP_RADIUS):
    /// slots the physics refuses to hold are lies.
    pub fn pitch(self, kind: u8) -> Vec2 {
        match self {
            FormSpacing::Normal => Vec2::splat(gap(kind)),
            FormSpacing::Loose => Vec2::splat(gap(kind) * 1.9),
            FormSpacing::Wall => Vec2::new(1.05, 1.15),
            FormSpacing::Open => Vec2::splat(gap(kind)),
        }
    }
}

/// The slot pitch of a regiment on a lane (the wall's walk). In normal
/// order it is the rest distance of two men at ease (sim/soldier.rs
/// `sep_ease`), the closest two standing men keep without pushing each
/// other, so the walk holds a rank more than at the field's pitch; the
/// other spacings keep their own.
pub fn lane_pitch(spacing: FormSpacing, kind: u8) -> Vec2 {
    match spacing {
        FormSpacing::Normal => Vec2::splat(crate::sim::soldier::sep_ease()),
        _ => spacing.pitch(kind),
    }
}

/// A regiment's slot pitch as it stands: its spacing's, or the lane's
/// while its block is on a lane (`GroupData::lane`).
pub fn pitch_of(gd: &GroupData) -> Vec2 {
    if gd.lane { lane_pitch(gd.spacing, gd.kind) } else { gd.spacing.pitch(gd.kind) }
}

/// The groups the open order `o` lays: for lanes along the facing, the
/// groups of files that cover the closed frontage of `files` files at
/// `closed` spacing within half a group pitch, never more files than
/// men; for lanes across it, the closed ranks in groups of `o.files`.
pub(crate) fn open_groups(files: usize, n: usize, closed: FormSpacing, kind: u8, o: &OpenOrder) -> usize {
    let a = o.files.max(1) as usize;
    if o.across {
        return n.max(1).div_ceil(files.max(1)).div_ceil(a);
    }
    let width = (files.max(1) - 1) as f32 * closed.pitch(kind).x;
    ((width / o.pitch).round() as usize + 1).clamp(1, n.max(1).div_ceil(a))
}

/// Each line's offset from the block's middle along the open order's
/// span: a file's lateral offset, or a rank's forward offset for lanes
/// across the facing. Lines of a group `sep` apart, groups `pitch`
/// apart; `lines` of them, centred.
pub(crate) fn open_lane_offsets(groups: usize, lines: usize, o: &OpenOrder) -> Vec<f32> {
    let a = o.files.max(1) as usize;
    let lines = lines.clamp(1, groups * a);
    let at = |f: usize| (f / a) as f32 * o.pitch + (f % a) as f32 * o.sep;
    let mid = 0.5 * at(lines - 1);
    (0..lines).map(|f| at(f) - mid).collect()
}

/// The lines of an open order: its files, or its ranks for lanes across
/// the facing (one line per closed rank, each keeping the closed
/// frontage).
pub(crate) fn open_lines(files: usize, n: usize, groups: usize, o: &OpenOrder) -> usize {
    let a = o.files.max(1) as usize;
    let lines = if o.across { n.max(1).div_ceil(files.max(1)) } else { groups * a };
    lines.clamp(1, n.max(1))
}

/// A Rect regiment's grid as it stands: files, ranks and slot pitch. In
/// the open order the lines of every group (files, or ranks for lanes
/// across the facing) and the mean pitch along the span, so the block's
/// extent comes out right. Every reader of the block's extent goes
/// through here.
pub fn grid(gd: &GroupData) -> (usize, usize, Vec2) {
    let n = gd.count.max(1);
    let files = (gd.files.max(1) as usize).min(n);
    if gd.spacing != FormSpacing::Open {
        return (files, n.div_ceil(files), pitch_of(gd));
    }
    let groups = open_groups(files, n, gd.close_to, gd.kind, &gd.open);
    let lines = open_lines(files, n, groups, &gd.open);
    let at = open_lane_offsets(groups, lines, &gd.open);
    let span = at.last().copied().unwrap_or(0.0) - at[0];
    let mean = if lines > 1 { span / (lines - 1) as f32 } else { 0.0 };
    let closed = gd.spacing.pitch(gd.kind);
    if gd.open.across {
        (files, lines, Vec2::new(closed.x, mean))
    } else {
        (lines, n.div_ceil(lines), Vec2::new(mean, closed.y))
    }
}

/// The open order a holding regiment takes to let a friend through
/// (sim/job.rs `open_for_passers`), with `room_front` and `room_back`
/// metres of free ground before its closed front and behind its closed
/// rear. Lanes `across` the facing, for a friend coming from the side,
/// are the ranks in groups at the same pitch, each rank keeping the
/// closed frontage, so the block grows deeper either way. Its men stand at their spring rest at their present width (the
/// ready stance's when `ready`), the lane the air for two men at ease
/// abreast (a lane narrower than two men abreast slows a pass
/// steeply). It takes the fewest files to a group whose
/// added depth fits the room on both sides together, less a margin on
/// each; None when no layout fits, and the regiment stays closed. A
/// layout of a single group leaves no lane, so it is no open order and is
/// never taken. The added depth goes to the two sides in proportion to
/// their room, so the block backs up only into free ground.
pub fn choose_open(gd: &GroupData, ready: bool, across: bool, room_front: f32, room_back: f32) -> Option<OpenOrder> {
    let front = (room_front - OPEN_ROOM_MARGIN).max(0.0);
    let back = (room_back - OPEN_ROOM_MARGIN).max(0.0);
    let layout = open_layouts(gd, ready, across);
    let (mut o, added) = (1..=OPEN_FILES_MAX).filter_map(layout).find(|&(_, added)| added <= front + back)?;
    let share = if added > 0.0 && front + back > 0.0 { front / (front + back) } else { 0.5 };
    o.front = added * share;
    o.back = added - o.front;
    o.closed_anchor = gd.anchor;
    Some(o)
}

/// The least depth any open layout adds to the block (metres): what
/// `choose_open` needs on both sides together, past the margins, to open
/// at all. None when no layout leaves a lane.
pub fn least_open_growth(gd: &GroupData, ready: bool, across: bool) -> Option<f32> {
    let layout = open_layouts(gd, ready, across);
    (1..=OPEN_FILES_MAX).filter_map(layout).map(|(_, added)| added).reduce(f32::min)
}

/// The open layout of `a` files to a group for `gd` (`choose_open`), with
/// the depth it adds to the closed block; its growth on each side and its
/// closed anchor still unset. None when it comes to a single group: the
/// lanes are the gaps between groups, and one group has none. None for a
/// regiment on the wall's walk: the walk's rows are all the ground it has.
fn open_layouts(gd: &GroupData, ready: bool, across: bool) -> impl Fn(u8) -> Option<(OpenOrder, f32)> {
    let on_lane = gd.lane;
    let n = gd.count.max(1);
    let files = (gd.files.max(1) as usize).min(n);
    let closed = gd.chosen_spacing();
    let width = &crate::sim::soldier::widths()[if ready { 3 } else { 0 }];
    let ease = &crate::sim::soldier::widths()[0];
    let shut = (n.div_ceil(files) - 1) as f32 * closed.pitch(gd.kind).y;
    let kind = gd.kind;
    move |a: u8| {
        if on_lane {
            return None;
        }
        let o = OpenOrder {
            files: a,
            sep: width.sep,
            pitch: (a - 1) as f32 * width.sep + width.body + 2.0 * ease.body,
            across,
            ..OpenOrder::default()
        };
        let groups = open_groups(files, n, closed, kind, &o);
        if groups < 2 {
            return None;
        }
        let lines = open_lines(files, n, groups, &o);
        let depth = if across {
            // The ranks at their lane offsets.
            let at = open_lane_offsets(groups, lines, &o);
            at.last().copied().unwrap_or(0.0) - at[0]
        } else {
            (n.div_ceil(lines) - 1) as f32 * gap(kind)
        };
        Some((o, depth - shut))
    }
}

/// Where an open regiment's closed block is centred: its centroid less
/// the anchor's shift along the facing.
pub fn closed_centre(gd: &GroupData) -> Vec2 {
    gd.centroid - facing_dir(gd.facing) * gd.open.shift()
}

/// The block of a regiment about `centre` as it stands closed: an open
/// regiment's ground when it is not letting anyone through, `centre`
/// its closed centre.
pub fn closed_rect_at(gd: &GroupData, centre: Vec2) -> BlockRect {
    if gd.spacing != FormSpacing::Open || gd.shape == FormShape::Blob || gd.files == 0 {
        return block_rect_at(gd, centre);
    }
    let n = gd.count.max(1);
    let files = (gd.files as usize).min(n);
    let pitch = gd.close_to.pitch(gd.kind);
    BlockRect {
        centre,
        fwd: facing_dir(gd.facing),
        half_w: (files - 1) as f32 * 0.5 * pitch.x + 1.0,
        half_d: (n.div_ceil(files) - 1) as f32 * 0.5 * pitch.y + 1.0,
    }
}

/// A regiment holding its ground, the kind that opens for a friend
/// passing through: no order to go anywhere (none, or archers standing
/// off their target or holding before a block), not in a fight and not
/// broken.
pub fn holds_ground(gd: &GroupData) -> bool {
    (gd.order.is_none() || gd.standoff || gd.skirmish_hold.is_some())
        && !gd.engaged
        && gd.melee_ticks == 0
        && !gd.state.is_broken()
        && gd.count > 0
}

/// An open regiment goes back to the spacing it had and reforms. Its
/// anchor goes back to where it stood closed unless something moved it
/// since it opened (an order).
pub fn close_order(gd: &mut GroupData) {
    if gd.spacing == FormSpacing::Open {
        if gd.anchor == open_anchor(gd) {
            gd.anchor = gd.open.closed_anchor;
        }
        gd.spacing = gd.close_to;
        gd.reform = true;
    }
}

/// A holding regiment opens into `o`: its anchor moves along the facing
/// so the block grows `o.front` ahead and `o.back` behind.
pub fn open_order(gd: &mut GroupData, o: OpenOrder) {
    gd.close_to = gd.spacing;
    gd.spacing = FormSpacing::Open;
    gd.open = o;
    gd.anchor = open_anchor(gd);
    gd.reform = true;
}

/// Where an open regiment's anchor stands.
fn open_anchor(gd: &GroupData) -> Vec2 {
    gd.open.closed_anchor + facing_dir(gd.facing) * gd.open.shift()
}

/// Where a regiment holds its ground: its anchor, or while it stands in
/// the open order the anchor it had closed. Opening moves the anchor
/// along the facing so the block grows where it has room; the regiment
/// has not gone anywhere. An order given since it opened sets a new
/// anchor, which `close_order` keeps, and it then holds there.
pub fn hold_point(gd: &GroupData) -> Vec2 {
    if gd.spacing == FormSpacing::Open && gd.anchor == open_anchor(gd) {
        gd.open.closed_anchor
    } else {
        gd.anchor
    }
}

/// Default file count for a regiment of `n` (the ~2.2:1 block the armies
/// have always spawned with).
pub fn default_files(n: usize) -> u32 {
    ((n as f32 * DEFAULT_ASPECT).sqrt().ceil() as u32).max(1)
}

/// Facing direction (unit yaw convention: 0 = +Z).
#[inline]
pub fn facing_dir(facing: f32) -> Vec2 {
    let (s, c) = facing.sin_cos();
    Vec2::new(s, c)
}

/// Yaw angle of a direction (inverse of `facing_dir`).
#[inline]
pub fn facing_of(dir: Vec2) -> f32 {
    dir.x.atan2(dir.y)
}

/// Row-major LOCAL slot offsets (x = lateral, y = forward) for `n` units
/// in `files` columns: front rank first, left to right, partial last rank
/// centered, block centered on the origin. The one grid used by slot
/// assignment AND the drag-order placement preview — what you see is
/// where they stand.
pub fn slot_offsets(n: usize, files: usize, pitch: Vec2) -> Vec<Vec2> {
    let files = files.clamp(1, n.max(1));
    let ranks = n.div_ceil(files);
    let half_depth = (ranks - 1) as f32 / 2.0;
    let mut out = Vec::with_capacity(n);
    for r in 0..ranks {
        let in_row = files.min(n - r * files);
        let half_w = (in_row - 1) as f32 / 2.0;
        let fwd = (half_depth - r as f32) * pitch.y;
        for c in 0..in_row {
            out.push(Vec2::new((c as f32 - half_w) * pitch.x, fwd));
        }
    }
    out
}

/// Radius of the loose Blob disc holding `n` men at roughly normal
/// density — the one blob footprint, shared by slot generation and the
/// deployment zone clamp.
pub fn blob_radius(n: usize) -> f32 {
    (n as f32).sqrt() * BASE_SPACING * 0.55
}

/// Rewrite the `home` slot offsets of regiment `g` for its current shape,
/// files, spacing, and facing. Rect: row-major grid centered on the
/// anchor, front rank toward `facing`, partial last rank centered. Units
/// are assigned to slots front-to-back / left-to-right in their CURRENT
/// relative order, so a reform never marches men through each other.
/// Returns the living member indices so callers (the deployment teleport)
/// don't rescan the unit buffer for the set this already built.
pub fn assign_slots(units: &mut Units, g: u32, gd: &mut GroupData) -> Vec<u32> {
    gd.slots_dealt = gd.slots_dealt.wrapping_add(1);
    // (unit index, forward coord, lateral coord) relative to the anchor.
    let fwd = facing_dir(gd.facing);
    let right = Vec2::new(fwd.y, -fwd.x);
    let mut members: Vec<(u32, f32, f32)> = Vec::new();
    for i in 0..units.len() {
        if units.group[i] == g && units.death_t[i] == 0 {
            let d = Vec2::new(units.pos[i].x, units.pos[i].z) - gd.anchor;
            members.push((i as u32, d.dot(fwd), d.dot(right)));
        }
    }
    let n = members.len();
    if n == 0 {
        return Vec::new();
    }

    match gd.shape {
        FormShape::Blob => {
            // Jittered disc sized to hold n at roughly normal density.
            let r_max = blob_radius(n);
            for (k, m) in members.iter().enumerate() {
                let seed = g.wrapping_mul(0x9E37_79B1) ^ (k as u32);
                let a = hash01(seed.wrapping_mul(3) + 1) * std::f32::consts::TAU;
                let r = r_max * hash01(seed.wrapping_mul(3) + 2).sqrt();
                units.home[m.0 as usize] = Vec2::new(r * a.cos(), r * a.sin());
            }
        }
        FormShape::Rect if gd.spacing == FormSpacing::Open && gd.open.across => {
            // Lanes across the facing: the ranks down the depth in their
            // groups, the men dealt to them evenly by their forward order
            // and laid left to right in each in their lateral order at the
            // closed file pitch, centred, so nobody crosses anybody.
            let files = (gd.files.max(1) as usize).min(n);
            let groups = open_groups(files, n, gd.close_to, gd.kind, &gd.open);
            let rows = open_lines(files, n, groups, &gd.open);
            let ys = open_lane_offsets(groups, rows, &gd.open);
            let pitch = gd.spacing.pitch(gd.kind);
            members.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let mut stacks: Vec<Vec<(u32, f32, f32)>> = vec![Vec::new(); rows];
            for (k, m) in members.iter().enumerate() {
                stacks[k * rows / n].push(*m);
            }
            for (r, row) in stacks.iter_mut().enumerate() {
                row.sort_unstable_by(|a, b| a.2.total_cmp(&b.2));
                let half_w = (row.len().max(1) - 1) as f32 * 0.5;
                for (c, m) in row.iter().enumerate() {
                    let x = (c as f32 - half_w) * pitch.x;
                    // The rows run front to back from the last offset.
                    units.home[m.0 as usize] = right * x + fwd * ys[rows - 1 - r];
                }
            }
        }
        FormShape::Rect if gd.spacing == FormSpacing::Open => {
            // Files across the frontage in their groups, the men dealt to
            // them evenly by their lateral order and stacked front to back
            // in each in their forward order, so nobody crosses anybody.
            // The files stand front-aligned, the deepest one centred on
            // the anchor.
            let files = (gd.files.max(1) as usize).min(n);
            let groups = open_groups(files, n, gd.close_to, gd.kind, &gd.open);
            let cols = open_lines(files, n, groups, &gd.open);
            let xs = open_lane_offsets(groups, cols, &gd.open);
            let pitch = gd.spacing.pitch(gd.kind);
            let half_d = (n.div_ceil(cols) - 1) as f32 * 0.5;
            members.sort_unstable_by(|a, b| a.2.total_cmp(&b.2));
            let mut stacks: Vec<Vec<(u32, f32, f32)>> = vec![Vec::new(); cols];
            for (k, m) in members.iter().enumerate() {
                stacks[k * cols / n].push(*m);
            }
            for (c, column) in stacks.iter_mut().enumerate() {
                column.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                for (r, m) in column.iter().enumerate() {
                    let y = (half_d - r as f32) * pitch.y;
                    units.home[m.0 as usize] = right * xs[c] + fwd * y;
                }
            }
        }
        FormShape::Rect => {
            let files = (gd.files.max(1) as usize).min(n);
            let pitch = pitch_of(gd);
            let slots = slot_offsets(n, files, pitch);
            // Front-to-back, then left-to-right within each rank: the
            // greedy crossing-minimizer, matching the slots' row-major
            // order.
            members.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            for row in members.chunks_mut(files) {
                row.sort_unstable_by(|a, b| a.2.total_cmp(&b.2));
            }
            for (m, slot) in members.iter().zip(&slots) {
                units.home[m.0 as usize] = right * slot.x + fwd * slot.y;
            }
        }
    }
    gd.count_at_reform = n;
    gd.reform = false;
    members.into_iter().map(|m| m.0).collect()
}

/// Axis-aligned half extents of a regiment block at `facing`: the
/// deployment zone clamp keeps the whole block inside, not just its
/// anchor. Lives here with the rest of the block geometry (pitch,
/// `slot_offsets`, `blob_radius`) so a density retune can't silently
/// diverge from the clamp footprint. The block at `files` in the spacing
/// it closes to: a new placement is a new order, and an order closes an
/// open regiment.
pub fn block_half_extents(gd: &GroupData, files: u32, facing: f32) -> Vec2 {
    if gd.shape == FormShape::Blob {
        return Vec2::splat(blob_radius(gd.count));
    }
    let files = (files.max(1) as usize).min(gd.count.max(1));
    let ranks = gd.count.max(1).div_ceil(files);
    let pitch = if gd.lane { lane_pitch(gd.chosen_spacing(), gd.kind) } else { gd.chosen_spacing().pitch(gd.kind) };
    let half_w = (files - 1) as f32 * 0.5 * pitch.x + 1.0;
    let half_d = (ranks - 1) as f32 * 0.5 * pitch.y + 1.0;
    let fwd = facing_dir(facing);
    let right = Vec2::new(fwd.y, -fwd.x);
    Vec2::new(
        (right.x * half_w).abs() + (fwd.x * half_d).abs(),
        (right.y * half_w).abs() + (fwd.y * half_d).abs(),
    )
}

/// How far a regiment's block edge stands from its centre toward the
/// unit direction `dir` (the rectangle's support in that direction):
/// the gap between two blocks is their centre distance less both of
/// these along the line between them (sim/job.rs `edge_gap`). A blob,
/// or a block not yet given its files, answers its footprint radius.
pub fn block_reach_along(gd: &GroupData, dir: Vec2) -> f32 {
    block_rect(gd).reach_along(dir)
}

/// A regiment's block as the gap tests see it: its rectangle (files and
/// ranks at its spacing, a metre of margin around the outer men, turned
/// to its facing) about its centroid, or for a blob a disc.
#[derive(Clone, Copy, Debug)]
pub struct BlockRect {
    pub centre: Vec2,
    /// The facing, zero for a disc.
    pub fwd: Vec2,
    /// Half its width and half its depth; a disc's radius in both.
    pub half_w: f32,
    pub half_d: f32,
}

impl BlockRect {
    /// How far the edge stands from the centre toward the unit direction
    /// `dir`.
    pub fn reach_along(&self, dir: Vec2) -> f32 {
        if self.fwd == Vec2::ZERO {
            return self.half_w;
        }
        let right = Vec2::new(self.fwd.y, -self.fwd.x);
        dir.dot(right).abs() * self.half_w + dir.dot(self.fwd).abs() * self.half_d
    }

    /// The distance from `p` to the nearest point of the block, 0 inside.
    pub fn distance(&self, p: Vec2) -> f32 {
        let d = p - self.centre;
        if self.fwd == Vec2::ZERO {
            return (d.length() - self.half_w).max(0.0);
        }
        let right = Vec2::new(self.fwd.y, -self.fwd.x);
        let out = Vec2::new(
            (d.dot(right).abs() - self.half_w).max(0.0),
            (d.dot(self.fwd).abs() - self.half_d).max(0.0),
        );
        out.length()
    }
}

/// The block of a regiment (`BlockRect`). A blob, or a block not yet
/// given its files, is a disc of its footprint radius.
pub fn block_rect(gd: &GroupData) -> BlockRect {
    block_rect_at(gd, gd.centroid)
}

/// The block of a regiment as it stands about `centre`: its frame when
/// `centre` is its anchor, the dressed formation rather than the men's
/// average, which stragglers drag off it.
pub fn block_rect_at(gd: &GroupData, centre: Vec2) -> BlockRect {
    if gd.shape == FormShape::Blob || gd.files == 0 {
        let r = if gd.radius > 0.0 { gd.radius } else { blob_radius(gd.count) };
        return BlockRect { centre, fwd: Vec2::ZERO, half_w: r, half_d: r };
    }
    let (files, _, pitch) = grid(gd);
    BlockRect {
        centre,
        fwd: facing_dir(gd.facing),
        half_w: (files - 1) as f32 * 0.5 * pitch.x + 1.0,
        half_d: front_offset(gd) + 1.0,
    }
}

/// A radius about the centroid that holds the whole of `block_rect`
/// (the corner's distance), found without the facing: the cheap first
/// test of a gap or a path hit, exact because no `reach_along` exceeds
/// it.
pub fn block_bound(gd: &GroupData) -> f32 {
    if gd.shape == FormShape::Blob || gd.files == 0 {
        return if gd.radius > 0.0 { gd.radius } else { blob_radius(gd.count) };
    }
    let (files, _, pitch) = grid(gd);
    let half_w = (files - 1) as f32 * 0.5 * pitch.x + 1.0;
    let half_d = front_offset(gd) + 1.0;
    (half_w * half_w + half_d * half_d).sqrt()
}

/// How far a block's front rank stands ahead of its frame centre: the
/// slots' own half depth. A blob's is its radius.
pub fn front_offset(gd: &GroupData) -> f32 {
    if gd.shape == FormShape::Blob || gd.files == 0 {
        return if gd.radius > 0.0 { gd.radius } else { blob_radius(gd.count) };
    }
    let (_, ranks, pitch) = grid(gd);
    (ranks - 1) as f32 * 0.5 * pitch.y
}

/// Deployment placement: rebake slots, then TELEPORT every man onto his
/// mark (pos, prev, yaw, zero velocity) — TW deployment moves are
/// instant, and the sim is frozen so nobody could march there anyway.
/// Also pins the (sim-owned, now stale) centroid to the anchor so
/// selection, banners, and follow-up placements read the new spot.
pub fn snap_to_slots(
    units: &mut Units,
    terrain: &crate::terrain::Terrain,
    g: u32,
    gd: &mut GroupData,
) {
    for i in assign_slots(units, g, gd) {
        let i = i as usize;
        // Off blocked ground (a block deployed over a wall's foot): a man
        // whose centre is on a blocked vertex could never move again.
        let p = gd.anchor + units.home[i];
        let p = terrain.nearest_open(p, 12.0).unwrap_or(p);
        let hh = crate::unit_types::half_height(units.kind[i] as usize);
        let pos = Vec3::new(p.x, terrain.height_at(p.x, p.y) + hh, p.y);
        units.pos[i] = pos;
        units.pos_prev[i] = pos;
        units.vel[i] = Vec3::ZERO;
        units.yaw[i] = gd.facing;
        units.yaw_prev[i] = gd.facing;
    }
    gd.centroid = gd.anchor;
}

/// While deploying, formation commands (wall/loose/blob, width) re-dress
/// the block instantly: `apply_reforms` lives in the frozen sim, so the
/// pending `reform` flags are snapped here instead, right after the
/// input sets that raise them.
fn deploy_reform_snap(
    mut units: ResMut<Units>,
    terrain: Res<crate::terrain::Terrain>,
    mut groups: ResMut<Groups>,
) {
    for g in 0..groups.list.len() {
        let gd = &mut groups.list[g];
        if gd.reform && gd.team == crate::orders::PLAYER_TEAM && gd.count > 0 {
            snap_to_slots(&mut units, &terrain, g as u32, gd);
        }
    }
}

/// Wall flavor of a regiment (spacing == Wall and still fighting):
/// 0 = none, 1 = shieldwall (sword kinds), 2 = spearwall (spears).
/// Shared by the sim damage model and the render wall pose.
#[inline]
/// Which wall a unit kind forms when Wall spacing goes on: spear kinds
/// brace a spearwall, everyone else locks shields. The HUD's wall
/// button previews this, so the classification lives in one place.
pub fn is_spearwall_kind(kind: u8) -> bool {
    kind == crate::unit_types::KIND_SPEAR
}

pub fn wall_kind(gd: &GroupData) -> u8 {
    if gd.spacing != FormSpacing::Wall
        || gd.state.is_broken()
        // Archers have no shields to lock: no wall of any flavor.
        || gd.kind == crate::unit_types::KIND_ARCHER
    {
        0
    } else if is_spearwall_kind(gd.kind) {
        2
    } else {
        1
    }
}

pub struct FormationPlugin;

impl Plugin for FormationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            FixedUpdate,
            (apply_reforms, fill_lane_slots)
                .chain()
                .after(crate::sim::take_tick)
                .before(crate::sim::step_sim)
                .in_set(crate::game_state::SimSet),
        )
        .add_systems(
            Update,
            (
                formation_keys.in_set(crate::game_state::BattleInputSet),
                deploy_reform_snap
                    .after(crate::game_state::BattleInputSet)
                    .run_if(
                        in_state(crate::game_state::GameState::Battle)
                            .and_then(crate::game_state::deploying),
                    ),
            ),
        );
    }
}

/// Formation commands for the current selection. One code path for the
/// hotkeys and the control-panel buttons (unit_cards.rs).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FormCmd {
    /// Wall on/off (shieldwall for sword regiments, spearwall for spears).
    Wall,
    /// Loose order on/off.
    Loose,
    /// Hold position <-> at ease.
    Hold,
    /// Fire-at-will on/off (archer regiments in the selection only).
    FireAtWill,
    /// Skirmish mode on/off (archer regiments in the selection only).
    Skirmish,
    /// Swords out on/off (archer regiments in the selection only). The F
    /// key is `Wall`, which does this to the archers it finds.
    Melee,
}

/// Formation hotkeys for the selection: F wall (swords out for archers),
/// L loose, B hold position, T fire-at-will, K skirmish.
fn formation_keys(
    keys: Res<ButtonInput<KeyCode>>,
    selection: Res<crate::orders::Selection>,
    mut groups: ResMut<Groups>,
) {
    const KEYS: [(KeyCode, FormCmd); 5] = [
        (KeyCode::KeyF, FormCmd::Wall),
        (KeyCode::KeyL, FormCmd::Loose),
        (KeyCode::KeyB, FormCmd::Hold),
        (KeyCode::KeyT, FormCmd::FireAtWill),
        (KeyCode::KeyK, FormCmd::Skirmish),
    ];
    for (key, cmd) in KEYS {
        if keys.just_pressed(key) {
            apply_formation_cmd(cmd, &selection, &mut groups);
        }
    }
}

pub fn apply_formation_cmd(
    cmd: FormCmd,
    selection: &crate::orders::Selection,
    groups: &mut Groups,
) {
    if selection.count_units == 0 {
        return;
    }
    let picked: Vec<usize> = selection.picked_controllable(groups).collect();
    if picked.is_empty() {
        return;
    }

    // The F key is one key for the whole selection: the wall on the
    // swordsmen and spearmen, the swords on the archers, who form no wall.
    if cmd == FormCmd::Wall
        && picked.iter().any(|&g| groups.list[g].kind == crate::unit_types::KIND_ARCHER)
    {
        apply_formation_cmd(FormCmd::Melee, selection, groups);
    }
    match cmd {
        FormCmd::Wall | FormCmd::Loose => toggle_spacing(cmd, &picked, groups),
        FormCmd::Hold => toggle_hold(&picked, groups),
        FormCmd::FireAtWill | FormCmd::Skirmish | FormCmd::Melee => toggle_archers(cmd, picked, groups),
    }
}

/// Wall or loose order on or off for the picked regiments.
#[inline]
fn toggle_spacing(cmd: FormCmd, picked: &[usize], groups: &mut Groups) {
    let wall = cmd == FormCmd::Wall;
    let want = if wall { FormSpacing::Wall } else { FormSpacing::Loose };
    let picked: Vec<usize> = picked
        .iter()
        .copied()
        .filter(|&g| !(wall && groups.list[g].kind == crate::unit_types::KIND_ARCHER))
        .collect();
    if picked.is_empty() {
        return;
    }
    // Toggle as a set: if anyone is not yet in the mode, everyone
    // enters it; if all are, everyone falls back to close order.
    // An open regiment's spacing is the one it closes back to.
    let on = picked.iter().any(|&g| groups.list[g].chosen_spacing() != want);
    let mut shield = 0;
    let mut spear = 0;
    for &g in &picked {
        let gd = &mut groups.list[g];
        let spacing = if on { want } else { FormSpacing::Normal };
        if gd.spacing == FormSpacing::Open {
            gd.close_to = spacing;
        } else {
            gd.spacing = spacing;
        }
        gd.shape = FormShape::Rect;
        if gd.files == 0 {
            gd.files = default_files(gd.count);
        }
        gd.reform = true;
        match wall_kind(gd) {
            1 => shield += 1,
            2 => spear += 1,
            _ => {}
        }
    }
    if !on {
        info!("{} regiments back to close order", picked.len());
    } else if wall {
        info!("wall formed: {shield} shieldwall, {spear} spearwall regiments");
    } else {
        info!("{} regiments to LOOSE order", picked.len());
    }
}

/// Hold position on or off for the picked regiments.
#[inline]
fn toggle_hold(picked: &[usize], groups: &mut Groups) {
    // Hold position (defend) vs at ease: held regiments never chase
    // and never engage on their own — they fight what steps into
    // reach and nothing else.
    let on = picked.iter().any(|&g| !groups.list[g].hold);
    for &g in picked {
        groups.list[g].hold = on;
    }
    info!(
        "{} regiments {}",
        picked.len(),
        if on { "HOLD POSITION" } else { "AT EASE" }
    );
}

/// An archer toggle (fire at will, skirmish, swords out) for the picked
/// archer regiments.
#[inline]
fn toggle_archers(cmd: FormCmd, picked: Vec<usize>, groups: &mut Groups) {
    // Archer toggles; melee regiments in the selection ignore them.
    let archers: Vec<usize> = picked
        .into_iter()
        .filter(|&g| groups.list[g].kind == crate::unit_types::KIND_ARCHER)
        .collect();
    if archers.is_empty() {
        return;
    }
    let on = archers.iter().any(|&g| {
        let gd = &groups.list[g];
        match cmd {
            FormCmd::FireAtWill => !gd.fire_at_will,
            FormCmd::Skirmish => !gd.skirmish,
            _ => !gd.melee,
        }
    });
    for &g in &archers {
        let gd = &mut groups.list[g];
        match cmd {
            FormCmd::FireAtWill => gd.fire_at_will = on,
            FormCmd::Skirmish => {
                gd.skirmish = on;
                // Switching skirmish off mid-run: the run ends
                // where the regiment stands and its own order
                // comes back.
                if !on && gd.skirmishing {
                    crate::sim::job::end_skirmish(gd);
                }
            }
            _ => {
                gd.melee = on;
                // Swords out end a run the same way; the bows go
                // down with the toggle (sim/job.rs).
                if on && gd.skirmishing {
                    crate::sim::job::end_skirmish(gd);
                }
            }
        }
    }
    info!(
        "{} archer regiments {} {}",
        archers.len(),
        match cmd {
            FormCmd::FireAtWill => "fire-at-will",
            FormCmd::Skirmish => "skirmish",
            _ => "swords out",
        },
        if on { "ON" } else { "OFF" }
    );
}

/// Consume `reform` requests and run the close-ranks check: a Rect
/// regiment that lost enough men since it last formed compacts its grid
/// (only out of contact — nobody dresses ranks mid-melee).
fn apply_reforms(mut units: ResMut<Units>, mut groups: ResMut<Groups>, mut tick: Local<u32>) {
    let _tick_log = crate::frame_log::tick::time(crate::frame_log::tick::Part::Reforms);
    *tick = tick.wrapping_add(1);
    for g in 0..groups.list.len() {
        let gd = &mut groups.list[g];
        // An open regiment that gets an order, fights or breaks closes at
        // once; its order's own reform deals the closed slots.
        if gd.spacing == FormSpacing::Open && !holds_ground(gd) {
            info!("regiment {g} closes");
            close_order(gd);
        }
        if gd.count == 0 || gd.state.is_broken() {
            gd.reform = false;
            continue;
        }
        let mut do_reform = gd.reform;
        // No dressing while the men are still filing onto a lane: the
        // places are handed out as they arrive.
        if !do_reform
            && gd.shape == FormShape::Rect
            && !gd.engaged
            && gd.lane_fill.is_none()
            && (*tick).wrapping_add(g as u32).is_multiple_of(CLOSE_RANKS_PERIOD)
        {
            let lost = gd.count_at_reform.saturating_sub(gd.count);
            if lost >= 8.max((gd.count_at_reform as f32 * CLOSE_RANKS_FRAC) as usize) {
                do_reform = true;
                // The anchor stays put on purpose: for an attacker it is
                // the frozen destination (the press must not retreat to
                // wherever the block currently stands), for a defender
                // it is the ground he holds — men shoved off it lean
                // back toward their line.
                info!("regiment {g} closes ranks ({} men)", gd.count);
            }
        }
        if do_reform {
            let gd = &mut groups.list[g];
            assign_slots(&mut units, g as u32, gd);
        }
    }
}

/// Whether a regiment's block is a lane's: it is ordered onto the lane,
/// holds on it, or was told to attack from it and its men still stand on
/// it (archers shooting from a wall).
pub(crate) fn stands_on_lane(gd: &GroupData, lane: &crate::town_layout::Lane) -> bool {
    gd.count > 0
        && gd.shape == FormShape::Rect
        && !gd.state.is_broken()
        && match gd.order {
            Some(crate::orders::Order::Move(p)) => lane.contains(p),
            Some(crate::orders::Order::Attack(_)) => gd.lane && lane.contains(gd.centroid),
            None => lane.contains(gd.anchor),
        }
}

/// Keeps each regiment's lane flag in step with its order
/// (`stands_on_lane`), re-dressing it when the flag changes. A regiment
/// coming onto a lane (orders.rs `lane_layout`) hands out its
/// places in arrival order: the man who steps onto the lane takes the
/// farthest place still held by a man behind him, and that man takes the
/// place he had. The men already standing then always hold the far
/// places, so nobody has to pass anybody along a walk too narrow to
/// pass on. The men on the lane also keep their order along it. One test
/// per man still on his way and one pass over the men there per tick,
/// nothing once they are all there; a new order or a fight ends it.
fn fill_lane_slots(mut units: ResMut<Units>, mut groups: ResMut<Groups>, terrain: Res<crate::terrain::Terrain>) {
    if !terrain.kind.has_wall() {
        return;
    }
    let lane = crate::town_layout::walk_lane(terrain.body_radius());
    for g in 0..groups.list.len() {
        let gd = &mut groups.list[g];
        // Every order comes through here, whoever gave it, so a regiment
        // sent off the lane dresses back to its field pitch.
        let on = stands_on_lane(gd, &lane);
        if on != gd.lane {
            gd.lane = on;
            gd.reform = true;
        }
        if gd.lane_fill.is_none() {
            continue;
        }
        if gd.count == 0 || gd.state.is_broken() || gd.engaged || gd.shape != FormShape::Rect {
            gd.lane_fill = None;
            continue;
        }
        if gd.lane_fill == Some(LaneFill::Fresh) {
            if gd.reform {
                continue;
            }
            let members: Vec<u32> = (0..units.len())
                .filter(|&i| units.group[i] == g as u32 && units.death_t[i] == 0)
                .map(|i| i as u32)
                .collect();
            gd.lane_fill = Some(LaneFill::Pending(members));
        }
        let at = match gd.order {
            Some(crate::orders::Order::Move(p)) => p,
            _ => gd.anchor,
        };
        let Some(LaneFill::Pending(pending)) = &mut gd.lane_fill else { continue };
        // Whether a place changed hands this tick: its slots are dealt
        // again (`GroupData::slots_dealt`), as by `assign_slots`.
        let mut dealt = false;
        let mut k = 0;
        while k < pending.len() {
            let i = pending[k] as usize;
            if units.death_t[i] != 0 {
                pending.swap_remove(k);
                continue;
            }
            let p = Vec2::new(units.pos[i].x, units.pos[i].z);
            if !lane.contains(p) {
                k += 1;
                continue;
            }
            // The farthest place a man still on his way holds, his own
            // among them.
            let far = pending
                .iter()
                .map(|&j| j as usize)
                .max_by(|&a, &b| {
                    let d = |j: usize| (at + units.home[j]).distance_squared(p);
                    d(a).total_cmp(&d(b))
                })
                .unwrap_or(i);
            if far != i {
                let (a, b) = (units.home[i], units.home[far]);
                units.home[i] = b;
                units.home[far] = a;
                dealt = true;
            }
            pending.swap_remove(k);
        }
        // The men on the lane keep their order along it: the crowd at a
        // stair's head shoves a man past the head, and the place he took
        // on the far side would have him pass every man standing between.
        // Two neighbours in that order whose places cross by more than
        // `LANE_CROSS` swap them, one pass a tick, so the order settles
        // without two men side by side trading places every tick.
        pending.sort_unstable();
        let mut on: Vec<(f32, usize)> = (0..units.len())
            .filter(|&i| units.group[i] == g as u32 && units.death_t[i] == 0)
            .filter(|&i| pending.binary_search(&(i as u32)).is_err())
            .map(|i| (units.pos[i].x, i))
            .collect();
        on.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        for w in on.windows(2) {
            let (a, b) = (w[0].1, w[1].1);
            if units.home[a].x > units.home[b].x + LANE_CROSS {
                let (ha, hb) = (units.home[a], units.home[b]);
                units.home[a] = hb;
                units.home[b] = ha;
                dealt = true;
            }
        }
        if dealt {
            gd.slots_dealt = gd.slots_dealt.wrapping_add(1);
        }
        if pending.is_empty() {
            info!("regiment {g}: its men have all taken their places on the lane");
            gd.lane_fill = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 200 archers in 50 files by 4 ranks, closed, facing +z.
    fn holder() -> GroupData {
        let mut gd = GroupData::new(0, crate::unit_types::KIND_ARCHER, Vec2::ZERO, 200);
        gd.shape = FormShape::Rect;
        gd.files = 50;
        gd.facing = facing_of(Vec2::Y);
        gd
    }

    /// No room or too little gives no plan; just enough gives the layout
    /// with the least growth, split as the room is.
    #[test]
    fn choose_open_needs_room_for_its_least_growth() {
        let gd = holder();
        let least = least_open_growth(&gd, false, false).expect("a layout with a lane");
        // Three files to a group: five ranks of 1.4 m where four stood.
        assert!((least - 1.4).abs() < 1e-4, "the least growth is {least}");
        let m = OPEN_ROOM_MARGIN;
        assert!(choose_open(&gd, false, false, 0.0, 0.0).is_none());
        assert!(choose_open(&gd, false, false, m + least * 0.75 - 0.01, m + least * 0.25).is_none());
        let (front, back) = (least * 0.75 + 0.01, least * 0.25);
        let o = choose_open(&gd, false, false, m + front, m + back).expect("a plan");
        assert_eq!(o.files, 3);
        assert!((o.front + o.back - least).abs() < 1e-4);
        assert!((o.front - least * front / (front + back)).abs() < 1e-4);
    }

    /// On the wall's walk a regiment in normal order stands at the rest
    /// distance of men at ease, five ranks on the walk; ordered off the
    /// walk it is back at its field pitch.
    #[test]
    fn a_regiment_stands_tighter_on_a_lane_and_only_there() {
        let lane = crate::town_layout::walk_lane(0.6);
        let on_walk = Vec2::new(20.0, crate::town_layout::WALL_Z);
        let mut gd = holder();
        gd.order = Some(crate::orders::Order::Move(on_walk));
        assert!(stands_on_lane(&gd, &lane));
        gd.lane = true;
        let tight = crate::sim::soldier::sep_ease();
        assert_eq!(pitch_of(&gd), Vec2::splat(tight));
        assert_eq!(grid(&gd).2, Vec2::splat(tight));
        assert!(((lane.across.1 - lane.across.0) / tight).floor() as u32 + 1 == 5, "ranks the walk holds");
        // Holding there, and told to shoot from there with its men on it.
        gd.order = None;
        gd.anchor = on_walk;
        assert!(stands_on_lane(&gd, &lane));
        gd.order = Some(crate::orders::Order::Attack(1));
        gd.centroid = on_walk;
        assert!(stands_on_lane(&gd, &lane));
        // Sent to the ground behind the wall.
        gd.order = Some(crate::orders::Order::Move(on_walk - Vec2::new(0.0, 40.0)));
        assert!(!stands_on_lane(&gd, &lane));
        gd.lane = false;
        assert_eq!(pitch_of(&gd), Vec2::splat(gap(gd.kind)));
        // The other spacings keep their own pitch on the lane.
        gd.lane = true;
        gd.spacing = FormSpacing::Wall;
        assert_eq!(pitch_of(&gd), FormSpacing::Wall.pitch(gd.kind));
    }

    /// A block that opens 3.9 m behind and nothing ahead moves its anchor
    /// 1.95 m back, but it still holds where it stood closed: a route that
    /// ended there still ends where the regiment holds.
    #[test]
    fn an_open_regiment_holds_where_it_stood_closed() {
        let mut gd = holder();
        gd.anchor = Vec2::new(20.9, -129.7);
        let route_end = gd.anchor;
        assert_eq!(hold_point(&gd), route_end);
        let o = choose_open(&gd, false, true, OPEN_ROOM_MARGIN, OPEN_ROOM_MAX).expect("a plan across");
        assert!(o.front.abs() < 1e-4 && (o.back - 3.9).abs() < 0.05, "grows {} ahead, {} behind", o.front, o.back);
        open_order(&mut gd, o);
        assert!((route_end.distance(gd.anchor) - 1.95).abs() < 0.03);
        assert_eq!(hold_point(&gd), route_end);
        // An order since it opened sets a new anchor, and it holds there.
        let elsewhere = Vec2::new(60.0, -129.7);
        gd.anchor = elsewhere;
        assert_eq!(hold_point(&gd), elsewhere);
        // Closed again where it opened, it holds at its closed anchor.
        gd.anchor = open_anchor(&gd);
        close_order(&mut gd);
        assert_eq!(gd.anchor, route_end);
        assert_eq!(hold_point(&gd), route_end);
    }

    /// Lanes across the facing of a block four ranks deep: all four ranks
    /// in one group would stand shallower than the closed block and fit
    /// in no room at all, but they leave no lane. The least a layout with
    /// a lane adds is two groups of two ranks.
    #[test]
    fn a_layout_of_one_group_is_no_open_order() {
        let gd = holder();
        let least = least_open_growth(&gd, false, true).expect("a layout with a lane");
        assert!(least > 0.0, "the least growth across is {least}");
        let m = OPEN_ROOM_MARGIN;
        assert!(choose_open(&gd, false, true, 0.0, 0.0).is_none());
        assert!(choose_open(&gd, false, true, m, m + least - 0.01).is_none());
        let o = choose_open(&gd, false, true, m, m + least + 0.01).expect("a plan across");
        assert_eq!(o.files, 2);
        assert!(o.across);
        assert!(o.front.abs() < 1e-4 && (o.back - least).abs() < 1e-4);
        // A block one rank deep has no ranks to part.
        let mut line = holder();
        line.files = 200;
        assert!(least_open_growth(&line, false, true).is_none());
        assert!(choose_open(&line, false, true, OPEN_ROOM_MAX, OPEN_ROOM_MAX).is_none());
    }
}
