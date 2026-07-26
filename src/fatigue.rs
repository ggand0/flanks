//! Regiment fatigue, the M2TW model (devlog 0055 evidence record):
//! an accumulator fed by activity — fighting fastest, then charging and
//! running — that recovers while standing, banded into the six M2TW
//! display states. Effects follow the MTW1 official-guide table (the
//! only numeric table in the engine family; M2TW's own rates were never
//! extracted): attack penalties from winded, morale penalties from
//! tired, exhausted men cannot charge. Speed loss is community-attested
//! qualitatively; the multipliers here are our calibration.
//!
//! Per-REGIMENT, not per-soldier (M2TW stores a per-soldier 4-bit level
//! and quantizes to the unit): one f32 per regiment is the 200k-scale
//! budget and the display granularity is the unit card anyway.

use bevy::prelude::*;

use crate::orders::Groups;
use crate::unit_types::TYPES;

pub struct FatiguePlugin;

impl Plugin for FatiguePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            FixedUpdate,
            update_fatigue
                .after(crate::movement::step_sim)
                .before(crate::morale::update_morale)
                .in_set(crate::game_state::SimSet),
        );
    }
}

/// The six M2TW fatigue states (engine display names).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum FatigueState {
    Fresh = 0,
    WarmedUp,
    Winded,
    Tired,
    VeryTired,
    Exhausted,
}

/// Band boundaries on the 0..100 accumulator.
const BANDS: [f32; 5] = [15.0, 30.0, 50.0, 70.0, 85.0];
const ORDER: [FatigueState; 6] = [
    FatigueState::Fresh,
    FatigueState::WarmedUp,
    FatigueState::Winded,
    FatigueState::Tired,
    FatigueState::VeryTired,
    FatigueState::Exhausted,
];

/// Per-state effects, indexed by `state(f) as usize`: display name,
/// attack points (MTW1 official table; it lists attack only, so defence
/// stays untouched), morale level modifier (same table), and locomotion
/// multiplier (qualitative evidence only — "clear hard penalty at
/// exhausted, cannot chase" — kept gentle pending a feel pass).
const EFFECTS: [(&str, f32, f32, f32); 6] = [
    ("fresh", 0.0, 0.0, 1.0),
    ("warmed up", 0.0, 0.0, 1.0),
    ("winded", -2.0, 0.0, 1.0),
    ("tired", -3.0, -3.0, 0.95),
    ("very tired", -4.0, -6.0, 0.88),
    ("exhausted", -6.0, -8.0, 0.78),
];

#[inline]
pub fn state(fatigue: f32) -> FatigueState {
    ORDER[BANDS.iter().position(|b| fatigue < *b).unwrap_or(5)]
}

pub fn state_name(s: FatigueState) -> &'static str {
    EFFECTS[s as usize].0
}

#[inline]
pub fn attack_penalty(fatigue: f32) -> f32 {
    EFFECTS[state(fatigue) as usize].1
}

#[inline]
pub fn morale_penalty(fatigue: f32) -> f32 {
    EFFECTS[state(fatigue) as usize].2
}

#[inline]
pub fn speed_mult(fatigue: f32) -> f32 {
    EFFECTS[state(fatigue) as usize].3
}

/// MTW1: completely exhausted units "cannot run or charge" — the charge
/// bonus and the charge-phase sprint are denied.
#[inline]
pub fn cannot_charge(fatigue: f32) -> bool {
    state(fatigue) == FatigueState::Exhausted
}

// --- Accumulation rates (per second, before the kind fatigue_rate
// mult). M2TW's exact rates are hardcoded and were never published for
// ANY title in the lineage, so these are knobs — but they now have a
// MEASURED anchor (devlog 0057): reading the engine's own per-unit
// fatigue counter through two live battles, 855 s of hard fighting
// moved units only to 2-5 on the engine's 0..15 scale, i.e. roughly
// warmed up to winded. M2TW tires men SLOWLY; nothing came close to
// exhausted in a quarter-hour battle.
//
// Calibrated to that anchor, then to the feel passes (first
// round: far too fast; second round: "could deplete slightly faster").
// Now: continuous melee reaches Tired at ~4 min and Exhausted at
// ~7 min; standing recovers Very Tired -> Fresh in ~10 min. Ordering
// (fight > charge > move, idle recovers) remains the evidenced part.
const FAT_FIGHT: f32 = 0.20;
const FAT_CHARGE: f32 = 0.28;
/// All ordered movement runs today; when the walk/run toggle lands,
/// walking drops to a token trickle and this becomes the RUN rate.
const FAT_MOVE: f32 = 0.11;
const FAT_RECOVER: f32 = 0.12;

/// Per-tick fatigue update (serial, ~200 rows). Runs after step_sim so
/// `engaged`/`charging` are this tick's truth, before the morale tick
/// that reads the result.
pub fn update_fatigue(mut groups: ResMut<Groups>, time: Res<Time>) {
    let dt = time.delta_secs();
    for g in groups.list.iter_mut() {
        if g.count == 0 {
            continue;
        }
        let moving = g.order.is_some();
        let rate = if g.engaged {
            FAT_FIGHT
        } else if g.charging {
            FAT_CHARGE
        } else if g.state.is_broken() || moving {
            // Broken regiments are running for their lives.
            FAT_MOVE
        } else {
            -FAT_RECOVER
        };
        let mult = if rate > 0.0 {
            TYPES[g.kind as usize].fatigue_rate
        } else {
            1.0
        };
        g.fatigue = (g.fatigue + rate * mult * dt).clamp(0.0, 100.0);
    }
}
