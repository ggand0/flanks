//! SoA storage for all units. Units are never individual Bevy entities.

use bevy::prelude::*;
use std::sync::Arc;

/// A soldier column the tick job shares with the main world. The job
/// takes a clone of the `Arc` at its kick and reads through it; nothing
/// is copied. Writes go through `DerefMut`, which is `Arc::make_mut`: a
/// write while nobody else holds the column (every write of the fixed
/// tick, which happens between the install and the kick) is a plain
/// write, and a write while the job holds a clone (a formation key
/// pressed mid-tick) copies that one column once. Reads and writes at
/// every call site look exactly like a `Vec`.
#[derive(Clone)]
pub struct Column<T>(Arc<Vec<T>>);

impl<T> Default for Column<T> {
    fn default() -> Self {
        Self(Arc::new(Vec::new()))
    }
}

impl<T> Column<T> {
    /// The shared buffer, for a reader on another thread: a pointer copy.
    pub fn share(&self) -> Arc<Vec<T>> {
        self.0.clone()
    }

    /// Install `v` as the column; returns the buffer it replaces.
    pub fn install(&mut self, v: Vec<T>) -> Arc<Vec<T>> {
        std::mem::replace(&mut self.0, Arc::new(v))
    }

    /// Install a buffer another column just released.
    pub fn install_shared(&mut self, a: Arc<Vec<T>>) -> Arc<Vec<T>> {
        std::mem::replace(&mut self.0, a)
    }
}

impl<T> std::ops::Deref for Column<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> {
        &self.0
    }
}

impl<T: Clone> std::ops::DerefMut for Column<T> {
    fn deref_mut(&mut self) -> &mut Vec<T> {
        Arc::make_mut(&mut self.0)
    }
}

/// All per-unit state, structure-of-arrays. The columns the tick job
/// reads are `Column`s (shared with the job); `hp` and `color` are the
/// main world's own.
#[derive(Resource, Default)]
pub struct Units {
    pub pos: Column<Vec3>,
    /// Position at the previous fixed tick; rendering lerps prev -> pos.
    pub pos_prev: Column<Vec3>,
    pub vel: Column<Vec3>,
    /// Per-unit max speed (small variation breaks lockstep patterns).
    pub speed: Column<f32>,
    pub team: Column<u8>,
    /// Unit type: index into `unit_types::TYPES` (also the render bucket).
    pub kind: Column<u8>,
    /// Smoothed facing angle around Y (0 = +Z); sim-owned, render-consumed.
    pub yaw: Column<f32>,
    /// Facing at the previous fixed tick; rendering lerps yaw_prev -> yaw.
    pub yaw_prev: Column<f32>,
    /// Index into `Groups::list`.
    pub group: Column<u32>,
    pub hp: Vec<f32>,
    /// Base render color (team color with per-unit variation baked in).
    /// Alpha carries a stable per-unit anim seed, not opacity.
    pub color: Column<[f32; 4]>,
    /// Combat memo (unit index; `u32::MAX` = none): the melee target
    /// while a swing is in flight, and the sparse-fight closing memo
    /// otherwise (sim/soldier.rs wide acquisition). NEVER cleared, and
    /// death-sweep swap-removes reindex units, so it can point at an
    /// arbitrary unit later — every consumer MUST validate on use
    /// (swings re-check at hit time; the closing drive checks team and
    /// distance). Do not read it as "current enemy" anywhere else.
    pub target: Column<u32>,
    /// The enemy soldier an archer has aimed at (unit index; `u32::MAX`
    /// = none): picked as his draw starts, shot at the loose, kept by
    /// some men between shots (sim/soldier.rs `pick_quarry`). Reindexed
    /// by the death sweep like `target`: validated on use.
    pub quarry: Column<u32>,
    /// Swing state: 0 = Ready, 1 = WindUp, 2 = Recover.
    pub swing: Column<u8>,
    /// Ticks left in the current swing phase.
    pub swing_t: Column<u8>,
    /// Hit-flash ticks remaining (render feedback).
    pub flash: Column<u8>,
    /// 0 = alive. Set to DEATH_TICKS on death; corpse plays its anim and is
    /// swap-removed when it reaches 1.
    pub death_t: Column<u8>,
    /// Offset from the regiment anchor: a regiment order is the SAME point
    /// for the whole block, rigidly translated per unit by this offset.
    /// Captured at spawn (loose); rigid formations will write slot offsets.
    pub home: Column<Vec2>,
    /// Arrows left (archers; 0 for melee kinds). Decremented on loose;
    /// an empty quiver means melee only.
    pub ammo: Column<u8>,
    /// Out of formation: he has left his slot to fight (M2TW keeps this
    /// per soldier, isInFormation). While his regiment is in melee he
    /// does not walk back to his slot; the flag clears when the melee
    /// ends and the regiment re-forms (sim/soldier.rs).
    pub out_form: Column<bool>,
    /// What he last saw of the comrades around him and of comrades
    /// running to the fight (sim/soldier.rs SIGHT_* bits). A man looks
    /// around every few ticks, not every tick, and acts on what he saw.
    pub sight: Column<u16>,
    /// The depth of the exchange he is answering (sim/damage.rs): 0 a
    /// plain blow, 1 the counter of his parried blow, 2 the counter of
    /// a parried counter, and so on. Written when he answers a warning,
    /// read when his own counter warns its target.
    pub chain: Column<u8>,
    /// Where he last chose to walk on his regiment's route (sim/route.rs
    /// `man_target`): NaN when his regiment has no route, infinite when
    /// his way to his slot was clear. He reads his route every few ticks,
    /// not every tick, and walks on what he chose.
    pub way: Column<Vec2>,
    /// How far he is into his ready stance, 0 at ease to POSE_READY (the
    /// low four bits), and how far he has turned his shoulders to squeeze
    /// through a crack, 0 square to POSE_READY (the high four bits;
    /// sim/soldier.rs `pose_level`, `turn_level`): his body is wider in
    /// the stance and narrower turned.
    pub pose: Column<u8>,
    /// Stuck on his way and trying a side (sim/soldier.rs `try_a_side`,
    /// the STUCK_* fields): the ticks he has wanted to walk and not moved
    /// while pressed from ahead, the ticks left of his try, how long he
    /// remembers the side he tried, and that side.
    pub stuck: Column<u16>,
    /// The opening he saw on his last wide look and walks for (sim/soldier.rs
    /// `choose_opening`), a point on the ground; NaN when he has none. He
    /// looks wide only every few ticks and keeps what he found between.
    pub aim: Column<Vec2>,
    /// Under the friction contact (sim/soldier.rs `friction_contact`,
    /// `steer`, `integrate`): the speed the contacts have put into him
    /// that his feet have not yet taken back. Zero otherwise.
    pub imposed: Column<Vec2>,
    /// Under the friction contact: he stood last tick (sim/soldier.rs
    /// `steer`): his own wish to move was under the speed of a step. On the
    /// walk to his place with nothing else to do (`on_his_way`), once he has
    /// set off, he stands again only after he has arrived there
    /// (`still_stands`); walking to the fight, once the soldiers ahead have
    /// stopped him. A standing soldier is braced against any body
    /// (sim/soldier.rs `BRACE`), a walking one only against the other side
    /// (`BRACE_BASIC`).
    pub stood: Column<bool>,
    /// Bumped every time a battle rebuilds this world. A sim tick job
    /// computed from an older world carries indices that mean nothing
    /// here: sim/mod.rs drops it instead of installing it.
    pub generation: u64,
}

/// Swing states (the `swing` column, low bits) + the charge flag bit.
pub const SWING_READY: u8 = 0;
pub const SWING_WINDUP: u8 = 1;
pub const SWING_RECOVER: u8 = 2;
/// Answering a swing he saw coming: the shield block or the weapon parry
/// of sim/damage.rs (`GUARD_*`). `swing_t` counts the answer down, the
/// style bits say which answer, `target` is the attacker he squares up
/// to. Set by the damage apply when the warning of a swing arrives,
/// cleared when the timer runs out or a parry that stopped the blow
/// turns into its counter wind-up.
pub const SWING_DEFEND: u8 = 3;
pub const SWING_STATE_MASK: u8 = 3;
/// Attack style for the CURRENT swing (bits 3-4, set at wind-up start,
/// wiped with the state): 0 = stab, 1 = classic overhead swing, 2 = slash
/// (benched). Only the look, except for a kind with a cut, whose style 1
/// is the slow cut (`UnitTypeParams::is_cut`) and whose style 2 is the
/// overhead swing at the stab's pace. In the defend state the same bits
/// name the answer: 0 the shield block, 1 the weapon parry.
pub const SWING_STYLE_SHIFT: u8 = 3;
pub const SWING_STYLE_MASK: u8 = 0b11 << SWING_STYLE_SHIFT;
/// Set when the swing was started at charging speed: bonus damage and a
/// bigger lunge — momentum matters (cavalry-less charge bonus for now).
pub const SWING_CHARGE: u8 = 4;
/// Staggered by a charging blow (state is Recover): can't act, can't
/// steer, can't even turn until `swing_t` runs out — the M2TW hit-stun.
/// Set by the damage apply pass, cleared when Recover expires.
pub const SWING_STAGGERED: u8 = 1 << 5;
/// One free pass against the NEXT stagger (set when a stagger expires,
/// consumed by the would-be stagger or wiped by the man's next swing):
/// chain-charging cannot stunlock a man who never gets to act.
pub const SWING_STAGGER_IMMUNE: u8 = 1 << 6;
/// The current wind-up is a bow DRAW, not a melee swing (archers): the
/// strike tick looses an arrow instead of a DamageEvent, and the render
/// wind-up progress runs on missile::DRAW_TICKS. Cleared on the loose.
pub const SWING_RANGED: u8 = 1 << 7;

impl Units {
    pub fn len(&self) -> usize {
        self.pos.len()
    }
}

pub struct UnitsPlugin;

impl Plugin for UnitsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Units>();
    }
}

/// Cheap deterministic hash -> [0, 1). Used for spawn jitter, color variation,
/// terrain noise.
pub fn hash01(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    (x >> 8) as f32 / 16_777_216.0
}

/// FL_SEED: offsets every per-unit spawn hash — perturbation runs for
/// battle-outcome sensitivity sweeps (the sim itself is deterministic,
/// so run-to-run variance must be injected at spawn). Apply it where a
/// seed is BUILT, not inside `push_unit`: the positional jitter is
/// computed at the call sites, and a seed offset applied further down
/// would leave every run with identical formation geometry.
pub fn spawn_seed(raw: u32) -> u32 {
    raw.wrapping_add(spawn_seed_offset())
}

fn spawn_seed_offset() -> u32 {
    static S: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *S.get_or_init(|| crate::util::env_or("FL_SEED", 0u32).wrapping_mul(0x9E37_79B1))
}

/// Append one fully-initialized unit to every SoA column. `anchor` is the
/// unit's regiment anchor; the spawn offset from it becomes `home`.
#[allow(clippy::too_many_arguments)] // spawn-time plumbing, all scalars
pub fn push_unit(
    units: &mut Units,
    terrain: &crate::terrain::Terrain,
    seed: u32,
    team: u8,
    kind: u8,
    group: u32,
    x: f32,
    z: f32,
    anchor: Vec2,
) {
    const TEAM_COLORS: [Vec3; 2] = [
        Vec3::new(0.20, 0.45, 0.85), // blue
        Vec3::new(0.90, 0.40, 0.15), // orange
    ];
    /// Archers wear CLOTH palettes instead of the raw team color: the
    /// mesh's team-blend parts (hood, tunic) lerp toward this per-unit
    /// instance color — muted woolen dyes of each army's hue, so
    /// allegiance reads at a glance while banners and cards carry the
    /// exact team color.
    const ARCHER_CLOTH: [Vec3; 2] = [
        Vec3::new(0.20, 0.32, 0.48), // blue army: dyed slate-blue wool
        Vec3::new(0.58, 0.35, 0.16), // orange army: autumn russet
    ];
    let params = &crate::unit_types::TYPES[kind as usize];
    let hh = crate::unit_types::half_height(kind as usize);
    // Off blocked ground (a block spawned over a wall's foot): a man whose
    // centre is on a blocked vertex could never move again.
    let at = terrain.nearest_open(Vec2::new(x, z), 12.0).unwrap_or(Vec2::new(x, z));
    let p = Vec3::new(at.x, terrain.height_at(at.x, at.y) + hh, at.y);
    units.pos.push(p);
    units.pos_prev.push(p);
    units.vel.push(Vec3::ZERO);
    units
        .speed
        .push(params.speed * (0.9 + 0.2 * hash01(seed.wrapping_mul(7) + 5)));
    units.team.push(team);
    units.kind.push(kind);
    // Face the enemy at spawn (0 = +Z).
    let spawn_yaw = if team == 0 { 0.0 } else { std::f32::consts::PI };
    units.yaw.push(spawn_yaw);
    units.yaw_prev.push(spawn_yaw);
    units.group.push(group);
    units.hp.push(params.hp);

    // Per-unit tonal variation so a block of 50k doesn't read as a flat
    // texture; alpha = stable anim seed (walk-bob phase), not opacity.
    let tone = 0.85 + 0.3 * hash01(seed.wrapping_mul(3));
    let c = if kind == crate::unit_types::KIND_ARCHER {
        ARCHER_CLOTH[team as usize] * tone
    } else {
        TEAM_COLORS[team as usize] * tone
    };
    units.color.push([c.x, c.y, c.z, hash01(seed.wrapping_mul(11) + 7)]);
    units.target.push(u32::MAX);
    units.quarry.push(u32::MAX);
    units.swing.push(SWING_RECOVER);
    // Stagger first swings across a full cooldown: contact must not
    // produce a synchronized metronome wave.
    units
        .swing_t
        .push((params.cooldown_ticks as f32 * hash01(seed.wrapping_mul(13) + 9)) as u8);
    units.flash.push(0);
    units.death_t.push(0);
    units.home.push(Vec2::new(x, z) - anchor);
    units.ammo.push(if kind == crate::unit_types::KIND_ARCHER {
        crate::unit_types::missile::AMMO
    } else {
        0
    });
    units.out_form.push(false);
    units.sight.push(0);
    units.chain.push(0);
    units.way.push(Vec2::NAN);
    units.pose.push(0);
    units.stuck.push(0);
    units.aim.push(Vec2::NAN);
    units.imposed.push(Vec2::ZERO);
    units.stood.push(true);
}
