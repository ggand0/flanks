//! Per-type unit parameters. Types are data, not code: sim systems index
//! this table by the `kind` SoA column. Two types for the MVP; archers etc.
//! append rows later.

/// Unit kind ids (the `kind` column value and the render bucket index).
/// NOTE: the spatial grid packs kind into a 2-bit meta field — at most 4
/// kinds without widening it (spatial.rs META_KIND).
pub const KIND_HEAVY: u8 = 0;
pub const KIND_LIGHT: u8 = 1;
pub const KIND_SPEAR: u8 = 2;
pub const NUM_KINDS: usize = 3;

/// Weapon class: indexes `BASE_DMG` and (later) spear/pike special rules.
/// The swing CYCLE is already per-kind via windup/cooldown ticks.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Weapon {
    Sword,
    Spear,
}

/// Damage of a landed hit at combat factor 0, per weapon class. Anchored
/// so the median matchup (light vs light, frontal) deals the same damage
/// per hit as the old flat model: pacing anchors survive the stat swap.
pub const BASE_DMG: [f32; 2] = [
    23.0, // Sword: light-vs-light frontal factor is -2 -> 23 * 1.13^-2 = 18
    32.5, // Spear: spear-vs-light frontal factor is -4 -> 32.5 * 1.13^-4 = 20
];

/// Damage swing per combat factor point (M2TW uses ~1.2 on kill CHANCE;
/// on hp damage 1.13 keeps a +/-6 factor spread inside ~2x swings).
pub const FACTOR_MULT: f32 = 1.13;
/// Factor clamp: +/-12 caps the damage spread at ~4.3x either way.
pub const FACTOR_CLAMP: f32 = 12.0;

pub struct UnitTypeParams {
    pub hp: f32,
    /// M2TW-style combat stats. Damage per hit is
    /// `BASE_DMG[weapon] * FACTOR_MULT^(attack + situational - defence)`,
    /// resolved directionally at damage apply (movement.rs): defence_skill
    /// counts vs front/side attacks but NOT rear; shield covers the front
    /// and LEFT side only (shield arm); armour counts everywhere, halved
    /// by `ap` attackers.
    pub attack: f32,
    /// Extra attack points while the swing carries the charge flag
    /// (momentum). A braced spearwall victim nullifies it.
    pub charge_bonus: f32,
    pub defence_skill: f32,
    pub armour: f32,
    pub shield: f32,
    /// Armour-piercing weapon: halves the victim's armour points.
    pub ap: bool,
    pub weapon: Weapon,
    /// Melee reach in meters (swing target must be inside).
    pub reach: f32,
    /// Fixed ticks (30 Hz) of wind-up before a swing lands.
    pub windup_ticks: u8,
    /// Base ticks between swings (±25% jitter applied per swing).
    pub cooldown_ticks: u8,
    /// Base move speed m/s (±10% per-unit jitter at spawn).
    pub speed: f32,
    /// Relative shove weight in separation (heavier pushes lighter).
    pub mass: f32,
    /// Multiplier on morale damage taken by this unit's regiment.
    pub morale_resist: f32,
    /// Mesh half height; units sit on the terrain at this Y.
    pub half_height: f32,
}

/// Indexed by kind. Baseline feel: the median frontal matchup (light vs
/// light) keeps its ~5-9 s 1v1 kill so battle lines grind instead of
/// evaporating. Stats are scaled from the vanilla M2TW EDU (devlog 0031):
/// heavies ~ Dismounted Feudal Knights, lights ~ Armored Sergeants'
/// sword-and-board cousins, spears ~ upper Spear Militia. Elite frontal
/// fights run LONGER than the old flat model (grindy shield-on-shield,
/// owner-approved); rear/flank hits skip skill+shield and kill 2-4x
/// faster — facing is the defensive resource now.
pub const TYPES: [UnitTypeParams; NUM_KINDS] = [
    // KIND_HEAVY — knights: slow, armored, hard-hitting, shove-heavy.
    UnitTypeParams {
        hp: 160.0,
        attack: 13.0,
        charge_bonus: 5.0,
        defence_skill: 6.0,
        armour: 5.0,
        shield: 5.0,
        ap: false,
        weapon: Weapon::Sword,
        reach: 1.8,
        windup_ticks: 12,
        cooldown_ticks: 48,
        speed: 6.0,
        mass: 2.0,
        morale_resist: 0.6,
        half_height: 0.55,
    },
    // KIND_LIGHT — men-at-arms: fast, fragile, quicker swings, spear reach.
    UnitTypeParams {
        hp: 90.0,
        attack: 9.0,
        charge_bonus: 3.0,
        defence_skill: 4.0,
        armour: 3.0,
        shield: 4.0,
        ap: false,
        weapon: Weapon::Sword,
        reach: 2.0,
        windup_ticks: 9,
        cooldown_ticks: 33,
        speed: 9.5,
        mass: 1.0,
        morale_resist: 1.0,
        half_height: 0.50,
    },
    // KIND_SPEAR — spear infantry: chainmail line troops behind big
    // shields. Longest reach (first strike on contact), steadier than
    // lights, slower swings; low attack is the M2TW spear anti-infantry
    // penalty folded in (the anti-cavalry bonus waits for cavalry). The
    // spearwall (formation.rs) is where they earn their keep.
    UnitTypeParams {
        hp: 100.0,
        attack: 7.0,
        charge_bonus: 2.0,
        defence_skill: 3.0,
        armour: 4.0,
        shield: 6.0,
        ap: false,
        weapon: Weapon::Spear,
        reach: 2.4,
        windup_ticks: 10,
        cooldown_ticks: 38,
        speed: 8.5,
        mass: 1.2,
        morale_resist: 0.85,
        half_height: 0.50,
    },
];
