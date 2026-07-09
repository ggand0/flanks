//! Per-type unit parameters. Types are data, not code: sim systems index
//! this table by the `kind` SoA column. Two types for the MVP; archers etc.
//! append rows later.

/// Unit kind ids (the `kind` column value and the render bucket index).
pub const KIND_HEAVY: u8 = 0;
pub const KIND_LIGHT: u8 = 1;
pub const NUM_KINDS: usize = 2;

pub struct UnitTypeParams {
    pub hp: f32,
    /// Damage per landed swing (FL_COMBAT_SCALE multiplies).
    pub damage: f32,
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

/// Indexed by kind. Baseline feel: a 1v1 kill takes ~5-9 s so battle lines
/// grind instead of evaporating; heavies out-fight lights ~2:1 but move
/// notably slower.
pub const TYPES: [UnitTypeParams; NUM_KINDS] = [
    // KIND_HEAVY — knights: slow, armored, hard-hitting, shove-heavy.
    UnitTypeParams {
        hp: 160.0,
        damage: 34.0,
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
        damage: 18.0,
        reach: 2.0,
        windup_ticks: 9,
        cooldown_ticks: 33,
        speed: 9.5,
        mass: 1.0,
        morale_resist: 1.0,
        half_height: 0.50,
    },
];
