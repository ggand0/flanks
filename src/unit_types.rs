//! Per-type unit parameters. Types are data, not code: sim systems index
//! this table by the `kind` SoA column.

/// Unit kind ids (the `kind` column value and the render bucket index).
/// NOTE: the spatial grid packs kind into a 2-bit meta field — archers
/// take the LAST slot; a 5th kind requires widening it (spatial.rs
/// META_KIND).
pub const KIND_HEAVY: u8 = 0;
pub const KIND_LIGHT: u8 = 1;
pub const KIND_SPEAR: u8 = 2;
pub const KIND_ARCHER: u8 = 3;
pub const NUM_KINDS: usize = 4;

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

/// Ranged stats, archer-only for now (a second ranged kind needs the
/// spatial META widen first, so these stay consts instead of a TYPES
/// column). All values M2TW-evidenced (devlog 0060) unless noted.
pub mod missile {
    /// EDU missile attack: Peasant Archers 5, Longbowmen 6, Yeoman 8.
    /// Plain-archer tier with a bit of punch (owner call: plain first).
    pub const ATTACK: f32 = 6.0;
    /// Meters, the vanilla `arrow` range (bodkin/composite reach 160).
    pub const RANGE: f32 = 120.0;
    /// Arrows per man — 30 for every vanilla foot archer.
    pub const AMMO: u8 = 30;
    /// Draw time before the loose (ticks at 30 Hz): 1.5 s.
    pub const DRAW_TICKS: u8 = 45;
    /// Reload between volleys: 8 s -> a ~9.5 s cycle, matching the
    /// animation-bound ~10 s / "6 volleys a minute" M2TW longbow cycle.
    pub const RELOAD_TICKS: u8 = 240;
    /// Launch speed band the arc solver may use (verbatim `velocity 20 48`).
    pub const SPEED_MIN: f32 = 20.0;
    pub const SPEED_MAX: f32 = 48.0;
    /// M2TW solves arcs in SI units; arrows fall at real gravity.
    pub const GRAVITY: f32 = 9.81;
    /// Landing scatter sigma in meters, RANGE-INDEPENDENT (the measured
    /// M2TW quirk: accuracy does not improve up close).
    pub const SCATTER_SIGMA: f32 = 3.0;
    /// Damage of a landed arrow at combat factor 0. No direct M2TW
    /// transplant exists (their kill rolls differ from our hp model);
    /// anchored so a 4-volley exchange from an equal regiment costs a
    /// light regiment on the order of 5% casualties (plan doc §9.3),
    /// then tuned in playtest.
    pub const BASE_DMG: f32 = 12.0;
}

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
    /// Base morale level (M2TW `stat_mental` first field). The situational
    /// modifier sum in morale.rs rides on top of this; vanilla M2TW scale:
    /// measured vanilla range is 1..11 (peasants 1, militia 3, sergeants
    /// and pikemen 5, knights 9-11) — see devlog 0057.
    pub base_morale: f32,
    /// Multiplier on morale SHOCK modifiers (flanked, rout contagion) —
    /// M2TW discipline: "determines the amount of morale lost when morale
    /// shocks occur". Lower = steadier under shock.
    pub discipline: f32,
    /// Fatigue accumulation multiplier — the M2TW `stat_heat` analog:
    /// heavy armour tires its wearer faster. Recovery is unscaled.
    pub fatigue_rate: f32,
    /// Mesh half height; units sit on the terrain at this Y.
    pub half_height: f32,
}

/// Indexed by kind. Baseline feel: the median frontal matchup (light vs
/// light) keeps its ~5-9 s 1v1 kill so battle lines grind instead of
/// evaporating. Stats are scaled from the vanilla M2TW EDU (devlog 0031):
/// heavies ~ Dismounted Feudal Knights, lights ~ Armored Sergeants'
/// sword-and-board cousins, spears ~ upper Spear Militia. Elite frontal
/// fights run LONGER than the old flat model (grindy shield-on-shield,
/// by design); rear/flank hits skip skill+shield and kill 2-4x
/// faster — facing is the defensive resource now.
pub const TYPES: [UnitTypeParams; NUM_KINDS] = [
    // KIND_HEAVY — knights: slow, armored, hard-hitting, shove-heavy.
    UnitTypeParams {
        hp: 240.0,
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
        // Masses sit on the vanilla EDU scale (militia 0.8 / spears 1.0 /
        // sergeants 1.2 / dismounted knights 1.5) so future cavalry mounts
        // (~3.5) slot in above; mass drives separation shove and charge
        // knockback ratios.
        mass: 1.5,
        // MEASURED vanilla EDU scale (devlog 0057): base morale runs
        // 1..11 across all 413 units — 11 is the CEILING (Dismounted
        // English Knights, Demi Lancers, Norman Knights), not a midpoint.
        // Our heavies map to the dismounted-knight rows of devlog 0031.
        base_morale: 11.0,
        discipline: 0.6,
        fatigue_rate: 1.3,
        half_height: 0.55,
    },
    // KIND_LIGHT — men-at-arms: fast, fragile, quicker swings, spear reach.
    UnitTypeParams {
        hp: 135.0,
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
        mass: 0.9,
        // Armored Sergeants / Billmen row = 5 on the measured scale.
        base_morale: 5.0,
        discipline: 1.0,
        fatigue_rate: 1.0,
        half_height: 0.50,
    },
    // KIND_SPEAR — spear infantry: chainmail line troops behind big
    // shields. Longest reach (first strike on contact), steadier than
    // lights, slower swings; low attack is the M2TW spear anti-infantry
    // penalty folded in (the anti-cavalry bonus waits for cavalry). The
    // spearwall (formation.rs) is where they earn their keep.
    UnitTypeParams {
        hp: 150.0,
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
        mass: 1.0,
        // Pikemen row = 5 (highly_trained); spear militia sit at 3.
        base_morale: 5.0,
        discipline: 0.85,
        fatigue_rate: 1.1,
        half_height: 0.50,
    },
    // KIND_ARCHER — levy bowmen: the ranged kind (missile stats live in
    // `missile`); this row is their WEAK melee fallback. Flesh-tier
    // protection, knife-and-buckler-less scrap, fast on their feet.
    // Scaled from the Peasant Archers / Longbowmen EDU band (devlog 0060):
    // melee attack 2..7, armour 0..1, no shield, morale 3 untrained,
    // mass 0.8.
    UnitTypeParams {
        hp: 120.0,
        attack: 4.0,
        charge_bonus: 1.0,
        defence_skill: 2.0,
        armour: 1.0,
        shield: 0.0,
        ap: false,
        weapon: Weapon::Sword,
        reach: 1.6,
        windup_ticks: 9,
        cooldown_ticks: 33,
        speed: 9.5,
        mass: 0.8,
        base_morale: 3.0,
        discipline: 1.15,
        fatigue_rate: 0.9,
        half_height: 0.50,
    },
];
