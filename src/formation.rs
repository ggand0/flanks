//! Rigid regiment formations: a slot generator writing into the existing
//! `units.home` column. A regiment
//! order stays ONE point; formations only change the per-unit offset from
//! it — the movement hot path is untouched.
//!
//! Shapes: `Rect` (rank-and-file grid with facing, the default) and `Blob`
//! (the pre-formations look: spawn-captured or regenerated loose cluster —
//! kept for low-morale levies later and for the FL_TEST_* geometry).
//! Spacing modes: Normal, Loose (missile dispersion later), Wall
//! (shieldwall for sword kinds, spearwall for spears — tight files),
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

/// The groups the open order `o` lays across the frontage of `files`
/// files at `closed` spacing: they span that width within half a group
/// pitch, never more files than men.
fn open_groups(files: usize, n: usize, closed: FormSpacing, kind: u8, o: &OpenOrder) -> usize {
    let width = (files.max(1) - 1) as f32 * closed.pitch(kind).x;
    let a = o.files.max(1) as usize;
    ((width / o.pitch).round() as usize + 1).clamp(1, n.max(1).div_ceil(a))
}

/// Across the open order: each file's lateral offset from the block's
/// middle, files of a group `sep` apart, groups `pitch` apart.
fn open_file_x(groups: usize, o: &OpenOrder) -> Vec<f32> {
    let a = o.files.max(1) as usize;
    let span = (groups - 1) as f32 * o.pitch + (a - 1) as f32 * o.sep;
    (0..groups * a).map(|f| (f / a) as f32 * o.pitch + (f % a) as f32 * o.sep - 0.5 * span).collect()
}

/// A Rect regiment's grid as it stands: files, ranks and slot pitch. In
/// the open order the files of every group, and the lateral pitch the
/// mean one, so the block's width comes out right. Every reader of the
/// block's extent goes through here.
pub fn grid(gd: &GroupData) -> (usize, usize, Vec2) {
    let n = gd.count.max(1);
    let files = (gd.files.max(1) as usize).min(n);
    if gd.spacing != FormSpacing::Open {
        return (files, n.div_ceil(files), gd.spacing.pitch(gd.kind));
    }
    let groups = open_groups(files, n, gd.close_to, gd.kind, &gd.open);
    let x = open_file_x(groups, &gd.open);
    let open = x.len().min(n);
    let span = x.last().copied().unwrap_or(0.0) - x[0];
    let pitch_x = if open > 1 { span / (open - 1) as f32 } else { 0.0 };
    (open, n.div_ceil(open), Vec2::new(pitch_x, gd.spacing.pitch(gd.kind).y))
}

/// The open order a holding regiment takes to let a friend through
/// (sim/job.rs `open_for_passers`), with `room_front` and `room_back`
/// metres of free ground before its closed front and behind its closed
/// rear. Its men stand at their spring rest at their present width (the
/// ready stance's when `ready`), the lane the air for two men at ease
/// abreast (in the toy port the 500-man pass took 46 s at 2.7 m centre
/// to centre with a 0.9 m body, and climbed steeply under two men
/// abreast, devlog 0311). It takes the fewest files to a group whose
/// added depth fits the room on both sides together, less a margin on
/// each, else the most files. The added depth goes to the two sides in
/// proportion to their room, so the block backs up only into free
/// ground.
pub fn choose_open(gd: &GroupData, ready: bool, room_front: f32, room_back: f32) -> OpenOrder {
    let n = gd.count.max(1);
    let files = (gd.files.max(1) as usize).min(n);
    let closed = gd.chosen_spacing();
    let width = &crate::sim::soldier::widths()[if ready { 3 } else { 0 }];
    let ease = &crate::sim::soldier::widths()[0];
    let shut = (n.div_ceil(files) - 1) as f32 * closed.pitch(gd.kind).y;
    let front = (room_front - OPEN_ROOM_MARGIN).max(0.0);
    let back = (room_back - OPEN_ROOM_MARGIN).max(0.0);
    let layout = |a: u8| {
        let o = OpenOrder {
            files: a,
            sep: width.sep,
            pitch: (a - 1) as f32 * width.sep + width.body + 2.0 * ease.body,
            ..OpenOrder::default()
        };
        let groups = open_groups(files, n, closed, gd.kind, &o);
        let ranks = n.div_ceil((groups * a as usize).min(n));
        (o, (ranks - 1) as f32 * gap(gd.kind) - shut)
    };
    let a = (1..=OPEN_FILES_MAX).find(|&a| layout(a).1 <= front + back).unwrap_or(OPEN_FILES_MAX);
    let (mut o, added) = layout(a);
    let share = if added > 0.0 && front + back > 0.0 { front / (front + back) } else { 0.5 };
    o.front = added * share;
    o.back = added - o.front;
    o.closed_anchor = gd.anchor;
    o
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
        FormShape::Rect if gd.spacing == FormSpacing::Open => {
            // Files across the frontage in their groups, the men dealt to
            // them evenly by their lateral order and stacked front to back
            // in each in their forward order, so nobody crosses anybody.
            // The files stand front-aligned, the deepest one centred on
            // the anchor.
            let files = (gd.files.max(1) as usize).min(n);
            let groups = open_groups(files, n, gd.close_to, gd.kind, &gd.open);
            let xs = open_file_x(groups, &gd.open);
            let cols = xs.len().min(n);
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
            let pitch = gd.spacing.pitch(gd.kind);
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
    let pitch = gd.chosen_spacing().pitch(gd.kind);
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
        let p = gd.anchor + units.home[i];
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
            apply_reforms
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
                test_form_script,
                disorder_log,
            ),
        );
    }
}

/// Mean living-unit distance to slot (`anchor + home`) — how ON their
/// marks a regiment stands.
fn slot_error(units: &Units, groups: &Groups, g: usize) -> f32 {
    let gd = &groups.list[g];
    let (mut sum, mut n) = (0.0f32, 0usize);
    for i in 0..units.len() {
        if units.group[i] as usize == g && units.death_t[i] == 0 {
            let p = Vec2::new(units.pos[i].x, units.pos[i].z);
            sum += p.distance(gd.anchor + units.home[i]);
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f32 }
}

/// Mean nearest-neighbor distance within a regiment (brute force — test
/// only). The spacing-mode metric: wall < normal < loose.
fn regiment_nn(units: &Units, g: usize) -> f32 {
    let pts: Vec<Vec2> = (0..units.len())
        .filter(|&i| units.group[i] as usize == g && units.death_t[i] == 0)
        .map(|i| Vec2::new(units.pos[i].x, units.pos[i].z))
        .collect();
    if pts.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for (k, p) in pts.iter().enumerate() {
        let mut best = f32::MAX;
        for (j, q) in pts.iter().enumerate() {
            if j != k {
                best = best.min(p.distance_squared(*q));
            }
        }
        sum += best.sqrt();
    }
    sum / pts.len() as f32
}

/// FL_TEST_FORM=1: the drag-order acceptance, driven through the SAME
/// line_order path the mouse uses. Stage 0: draw a 120 m line ahead of
/// four blue regiments. Stage 1 (all arrived): mean slot error must be
/// small (units stand ON their previewed marks) and the block extents
/// must match files x pitch. Then one regiment forms the wall, one goes
/// loose. Stage 2 (+8 s): within-regiment nearest-neighbor spacing must
/// order wall < normal < loose.
fn test_form_script(
    time: Res<Time>,
    units: Res<Units>,
    mut groups: ResMut<Groups>,
    mut stage: Local<u32>,
    mut watch: Local<Vec<usize>>,
    mut settle_at: Local<f32>,
) {
    if std::env::var("FL_TEST_FORM").is_err() {
        return;
    }
    let t = time.elapsed_secs();
    match *stage {
        0 if t > 3.0 => {
            let picked: Vec<usize> = groups
                .list
                .iter()
                .enumerate()
                .filter(|(_, gd)| gd.team == 0 && gd.count > 0)
                .take(4)
                .map(|(g, _)| g)
                .collect();
            if picked.len() < 4 {
                warn!("[form-test] needs >= 4 blue regiments");
                *stage = 99;
                return;
            }
            let mean: Vec2 = picked.iter().map(|&g| groups.list[g].centroid).sum::<Vec2>()
                / picked.len() as f32;
            // A 120 m line 45 m toward the enemy, drawn left to right.
            let a = mean + Vec2::new(-60.0, 45.0);
            let b = mean + Vec2::new(60.0, 45.0);
            // Drag-direction invariance: the same line drawn with the
            // opposite hand must face the same way (facing follows the
            // enemy side of the line, not the drag direction). The
            // reversed order is immediately overwritten by the real one.
            crate::orders::line_order(&mut groups, &picked, b, a);
            let rev_facing = groups.list[picked[0]].facing;
            crate::orders::line_order(&mut groups, &picked, a, b);
            let fwd_facing = groups.list[picked[0]].facing;
            if (rev_facing - fwd_facing).abs() < 0.01 {
                info!(
                    "[form-test] drag invariance: fwd {fwd_facing:.2} == rev {rev_facing:.2} -> OK"
                );
            } else {
                warn!(
                    "[form-test] drag invariance FAILED: fwd {fwd_facing:.2} vs rev {rev_facing:.2}"
                );
            }
            for &g in &picked {
                info!(
                    "[form-test] regiment {g}: files {} facing {:.2}",
                    groups.list[g].files, groups.list[g].facing
                );
            }
            *watch = picked;
            *stage = 1;
        }
        1 if !watch.is_empty()
            && watch.iter().all(|&g| groups.list[g].order.is_none()) =>
        {
            // "Arrived" is a centroid predicate; the tail of a deep
            // column is still tens of meters out when it fires — give
            // the ranks time to pour in and dress before measuring.
            *settle_at = t + 15.0;
            *stage = 2;
        }
        2 if t > *settle_at => {
            for &g in watch.iter() {
                let gd = &groups.list[g];
                let err = slot_error(&units, &groups, g);
                // Mean angular error of unit yaw vs the ordered facing:
                // a settled formation must LOOK the way it was told to.
                let (mut yerr, mut n) = (0.0f32, 0usize);
                for i in 0..units.len() {
                    if units.group[i] as usize == g && units.death_t[i] == 0 {
                        let d = (units.yaw[i] - gd.facing + std::f32::consts::PI)
                            .rem_euclid(std::f32::consts::TAU)
                            - std::f32::consts::PI;
                        yerr += d.abs();
                        n += 1;
                    }
                }
                let yerr = if n > 0 { yerr / n as f32 } else { 0.0 };
                info!(
                    "[form-test] regiment {g} settled: slot err {err:.2} m, facing err {yerr:.2} rad ({} men, files {}) -> {}",
                    gd.count,
                    gd.files,
                    if err < 1.5 && yerr < 0.3 { "OK" } else { "FAIL" }
                );
            }
            // Spacing modes: wall on the first, loose on the second.
            let (w, l) = (watch[0], watch[1]);
            groups.list[w].spacing = FormSpacing::Wall;
            groups.list[w].reform = true;
            groups.list[l].spacing = FormSpacing::Loose;
            groups.list[l].reform = true;
            info!("[form-test] regiment {w} -> wall, regiment {l} -> loose");
            *settle_at = t + 10.0;
            *stage = 3;
        }
        3 if t > *settle_at => {
            let nn_wall = regiment_nn(&units, watch[0]);
            let nn_loose = regiment_nn(&units, watch[1]);
            let nn_normal = regiment_nn(&units, watch[2]);
            let ordered = nn_wall < nn_normal - 0.1 && nn_normal < nn_loose - 0.2;
            info!(
                "[form-test] spacing nn: wall {nn_wall:.2} < normal {nn_normal:.2} < loose {nn_loose:.2} -> {}",
                if ordered { "OK" } else { "FAIL" }
            );
            *stage = 4;
        }
        _ => {}
    }
}

/// FL_LOG_DISORDER=1: every 4 s while anyone fights, the mean
/// disorder (m off the slots) of engaged regiments — the
/// blocks-stay-blocks number.
fn disorder_log(groups: Res<Groups>, time: Res<Time>, mut next: Local<f32>) {
    if std::env::var("FL_LOG_DISORDER").is_err() {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 4.0;
    let (mut disorder, mut regs) = (0.0f32, 0usize);
    for gd in groups.list.iter().filter(|gd| gd.engaged && gd.count > 0) {
        disorder += gd.disorder;
        regs += 1;
    }
    if regs > 0 {
        info!(
            "[disorder] t={t:.0}s {regs} engaged regiments, mean disorder {:.2} m",
            disorder / regs as f32
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
        FormCmd::Wall | FormCmd::Loose => {
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

        FormCmd::Hold => {
            // Hold position (defend) vs at ease: held regiments never chase
            // and never engage on their own — they fight what steps into
            // reach and nothing else.
            let on = picked.iter().any(|&g| !groups.list[g].hold);
            for &g in &picked {
                groups.list[g].hold = on;
            }
            info!(
                "{} regiments {}",
                picked.len(),
                if on { "HOLD POSITION" } else { "AT EASE" }
            );
        }

        FormCmd::FireAtWill | FormCmd::Skirmish | FormCmd::Melee => {
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
    }
}

/// Consume `reform` requests and run the close-ranks check: a Rect
/// regiment that lost enough men since it last formed compacts its grid
/// (only out of contact — nobody dresses ranks mid-melee).
fn apply_reforms(mut units: ResMut<Units>, mut groups: ResMut<Groups>, mut tick: Local<u32>) {
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
        if !do_reform
            && gd.shape == FormShape::Rect
            && !gd.engaged
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
