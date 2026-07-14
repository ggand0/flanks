//! Regiment morale: the most-tuned system in the game, extracted from
//! the spawn code so its knobs live in one place.
//!
//! Model (owner-directed, devlog-refined): CASUALTIES are the anchor
//! (~67% losses alone break a light; a heavy cannot be broken by
//! attrition), being locally OUTNUMBERED is deliberately minor (a
//! soldier can't count a battlefield), being OUTFLANKED is the dominant
//! psychological drain, steady ALLIES nearby stiffen the line, and all
//! psychological pressure scales with DEPLETION. Per-regiment factor
//! rates are published to `MoraleReadout` for the inspect panel.

use bevy::prelude::*;

use crate::frontline::InfluenceField;
use crate::orders::{Groups, RegState};
use crate::unit_types::TYPES;
use crate::units::hash01;

pub struct MoralePlugin;

impl Plugin for MoralePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MoraleReadout>().add_systems(
            FixedUpdate,
            update_morale
                .after(crate::movement::step_sim)
                .before(crate::orders::clear_arrived_orders),
        );
    }
}

/// Per-regiment morale drain contributions (morale/s, after all
/// multipliers) from the last tick — feeds the inspect panel so the
/// player can SEE why a regiment is wavering.
#[derive(Clone, Copy, Default)]
pub struct MoraleFactors {
    /// Casualty drain, smoothed over ~1 s (raw per-tick spikes are
    /// unreadable).
    pub casualties: f32,
    pub outnumbered: f32,
    pub flanked: f32,
    /// Raw 0..1 surround fraction behind `flanked`.
    pub flanked01: f32,
    pub contagion: f32,
    /// Fighting in disarray (formation disorder past the free band).
    pub disorder: f32,
    /// Steady friendly regiments counted for support (0..=3).
    pub friends: u32,
    /// Combined resist/support multiplier applied to psychological terms.
    pub psych_mult: f32,
    pub depletion: f32,
    pub recovering: bool,
}

#[derive(Resource, Default)]
pub struct MoraleReadout(pub Vec<MoraleFactors>);

// --- Morale tuning ---
/// Morale lost per (fraction of initial strength) of fresh casualties:
/// ~67% losses alone break a light regiment; a heavy (0.6 resist) can
/// NOT be broken by casualties alone — it holds until flanked or the
/// line collapses around it. Owner pacing rounds: 280 ("10 s routs")
/// -> 200 ("routing at 500 men left is shameful") -> 150.
const MORALE_CASUALTY: f32 = 150.0;
/// Drain per second when locally outnumbered >3:1 (density ratio).
/// Deliberately minor — a soldier can't count the battlefield (owner
/// direction); FLANKS break formations, not headcounts.
const MORALE_OUTNUMBERED: f32 = 1.5;
/// Peak drain per second when fully surrounded. Scales with how much of
/// the compass around the regiment holds enemy mass beyond a normal
/// frontal arc — the dominant psychological factor.
const MORALE_FLANKED: f32 = 6.0;
/// Radius of the 8-probe flank ring around the centroid.
const FLANK_RING_R: f32 = 22.0;
/// Enemy blurred density at a ring probe below which a sector can't be
/// hostile regardless of ratio (noise floor).
const FLANK_T: f32 = 0.6;
/// A sector is hostile only when enemy density exceeds OWN density by
/// this factor there — probes landing inside our own block or the
/// front-line mixing zone must not count (a frontal press ≠ a flank).
const FLANK_DOMINANCE: f32 = 1.2;
/// Psychological-pressure damping per steady friendly regiment within
/// NEIGHBOR_R (up to 3 counted): allies in view stiffen the line.
const MORALE_SUPPORT: f32 = 0.35;
/// Drain per second per routing friendly regiment within RALLY_R (capped).
const MORALE_ROUT_NEIGHBOR: f32 = 2.0;
const MORALE_ROUT_CAP: f32 = 6.0;
/// Peak drain per second for fighting in complete disarray. Disorder is
/// the mean slot deviation (GroupData::disorder); the drain ramps over
/// DISORDER_FREE..DISORDER_FULL meters and only while ENGAGED — a
/// scattered regiment dressing its ranks in peace is fine, the same
/// scatter in melee is how formations die.
const MORALE_DISORDER: f32 = 3.5;
const DISORDER_FREE: f32 = 2.0;
const DISORDER_FULL: f32 = 7.0;
/// Psychological damping while in a wall stance: braced men stand.
const WALL_STEADY: f32 = 0.85;
/// Recovery per second when unengaged and undisturbed.
const MORALE_RECOVERY: f32 = 3.0;
/// Attacked-in-the-rear/flank (phase D of formation combat v2, pulled
/// forward): blades landing in a regiment's back are a STANDING morale
/// drain while the pressure lasts — M2TW's attacked-in-rear modifier;
/// there a rear attack routs the unit through morale, not attrition,
/// and the slaughter is the pursuit. Driven by the per-sector hit
/// tallies the damage apply fills (~1 s memory, saturation-normalized):
/// self-measuring, so a lone backstab is noise and a full-width rear
/// engagement pins the drain at maximum. Rear at saturation matches
/// full encirclement on the density ring (MORALE_FLANKED) — the two
/// stack for the true hammer-and-anvil.
const MORALE_REAR_ATTACK: f32 = 6.0;
const MORALE_FLANK_ATTACK: f32 = 2.5;
/// Sector hits (tally units) that count as full pressure.
const SHOCK_SAT: f32 = 10.0;
/// Per-tick decay of the tallies (~1 s memory).
const SHOCK_DECAY: f32 = 0.92;
/// Log throttle for the rear-attack event line (ticks).
const SHOCK_LOG_TICKS: u16 = 300;
/// Routing-neighbor / rally-safety radius.
const NEIGHBOR_R: f32 = 60.0;
/// Broken regiments below this fraction of initial strength shatter.
const SHATTER_FRAC: f32 = 0.15;
/// Seconds routing before a rally roll is allowed.
const RALLY_DELAY: f32 = 8.0;
/// Rally chance per second once allowed and safe.
const RALLY_CHANCE: f32 = 0.02;

/// Per-tick regiment morale update (serial; ~200 rows). Runs after the
/// damage apply pass so `recent_deaths` is this tick's tally.
fn update_morale(
    mut groups: ResMut<Groups>,
    field: Res<InfluenceField>,
    time: Res<Time>,
    mut readout: ResMut<MoraleReadout>,
    mut tick: Local<u32>,
) {
    *tick += 1;
    let dt = time.delta_secs();
    readout.0.resize(groups.list.len(), MoraleFactors::default());

    // Snapshot routing/steady centroids for the neighbor terms (state
    // from last tick is fine — morale contagion is not latency-sensitive).
    let routing_centroids: Vec<(u8, Vec2)> = groups
        .list
        .iter()
        .filter(|g| g.state.is_broken() && g.count > 0)
        .map(|g| (g.team, g.centroid))
        .collect();
    let steady_centroids: Vec<(usize, u8, Vec2)> = groups
        .list
        .iter()
        .enumerate()
        .filter(|(_, g)| !g.state.is_broken() && g.count > 0)
        .map(|(i, g)| (i, g.team, g.centroid))
        .collect();

    for (gi, g) in groups.list.iter_mut().enumerate() {
        if g.count == 0 {
            g.recent_deaths = 0;
            continue;
        }
        let resist = TYPES[g.kind as usize].morale_resist;

        // Attacked-in-the-rear/flank pressure (0..1) from this tick's
        // sector-hit tallies; the tallies then decay (~1 s memory).
        let p_rear = (g.shock_rear / SHOCK_SAT).min(1.0);
        let p_flank = (g.shock_flank / SHOCK_SAT).min(1.0);
        if g.shock_cd > 0 {
            g.shock_cd -= 1;
        }
        if p_rear >= 0.5 && g.shock_cd == 0 && !g.state.is_broken() {
            g.shock_cd = SHOCK_LOG_TICKS;
            info!("regiment {gi} is taking blades in the REAR");
        }
        g.shock_rear *= SHOCK_DECAY;
        g.shock_flank *= SHOCK_DECAY;

        match g.state {
            RegState::Steady => {
                let mut drain = 0.0;
                // Fresh casualties.
                let casualty_drain =
                    MORALE_CASUALTY * g.recent_deaths as f32 / g.initial_count as f32 * resist;
                drain += casualty_drain;
                // Psychological pressure scales with DEPLETION: a fresh
                // regiment shrugs off routing neighbors and bad odds; a
                // bleeding one panics. Without this, full-strength
                // regiments cascade-rout off contagion alone.
                let frac = g.count as f32 / g.initial_count as f32;
                let depletion = (1.4 - 1.2 * frac).clamp(0.15, 1.4);
                let mut pressure = 0.0;
                // Locally outnumbered (blurred density ratio at centroid):
                // minor and hard to trigger — a clogged line is NOT panic.
                let own = field.density(g.team, g.centroid);
                let enemy = field.density(1 - g.team, g.centroid);
                let outnumbered = enemy / (own + 0.1) > 3.0;
                if outnumbered {
                    pressure += MORALE_OUTNUMBERED;
                }
                // Outflanked — the main killer. 8 probes on a ring around
                // the centroid; a sector is hostile only where enemy mass
                // DOMINATES own mass (see FLANK_DOMINANCE — absolute
                // density alone made every line fight read as encircled).
                // The largest hostile-free arc is the safe side: a full
                // frontal contact (3 adjacent sectors) scores exactly 0,
                // a pincer ~0.4, encirclement 1.
                let mut hostile = [false; 8];
                for (k, h) in hostile.iter_mut().enumerate() {
                    let a = k as f32 / 8.0 * std::f32::consts::TAU;
                    let p = g.centroid + Vec2::new(a.cos(), a.sin()) * FLANK_RING_R;
                    let e = field.density(1 - g.team, p);
                    *h = e > FLANK_T && e > field.density(g.team, p) * FLANK_DOMINANCE;
                }
                let mut safe_run = 0usize;
                let mut run = 0usize;
                for k in 0..16 {
                    if hostile[k % 8] {
                        run = 0;
                    } else {
                        run += 1;
                        safe_run = safe_run.max(run);
                    }
                }
                let safe_arc = safe_run.min(8) as f32 * 45.0;
                let flanked = ((225.0 - safe_arc) / 225.0).clamp(0.0, 1.0);
                pressure += MORALE_FLANKED * flanked;
                // Blades actually landing in the back/side (per-sector
                // hit pressure): the ring above reads enemy MASS around
                // the block; this reads the wounds themselves, so a
                // single regiment carving the rear registers even when
                // the density ring calls the fight one-sided.
                pressure += MORALE_REAR_ATTACK * p_rear + MORALE_FLANK_ATTACK * p_flank;
                // Routing friendlies nearby shake resolve.
                let rout_drain = routing_centroids
                    .iter()
                    .filter(|(t, c)| *t == g.team && c.distance(g.centroid) < NEIGHBOR_R)
                    .count() as f32
                    * MORALE_ROUT_NEIGHBOR;
                pressure += rout_drain.min(MORALE_ROUT_CAP);
                // Fighting in disarray: the formation-shape drain.
                let disorder01 = if g.engaged {
                    ((g.disorder - DISORDER_FREE) / (DISORDER_FULL - DISORDER_FREE))
                        .clamp(0.0, 1.0)
                } else {
                    0.0
                };
                pressure += MORALE_DISORDER * disorder01;
                // Steady friends in view stiffen the line; kind resist
                // (heavies steadier) applies to ALL psychological terms.
                // Casualties stay undamped — dead men are dead men.
                let friends = steady_centroids
                    .iter()
                    .filter(|(i, t, c)| {
                        *i != gi && *t == g.team && c.distance(g.centroid) < NEIGHBOR_R
                    })
                    .count()
                    .min(3) as f32;
                // Braced walls stand steadier under every fear.
                let wall_mult = if crate::formation::wall_kind(g) != 0 {
                    WALL_STEADY
                } else {
                    1.0
                };
                let psych_mult = resist * wall_mult / (1.0 + MORALE_SUPPORT * friends);
                pressure *= psych_mult;
                drain += pressure * depletion * dt;

                let recovering = drain <= 0.0 && !g.engaged;
                if drain > 0.0 {
                    g.morale -= drain;
                } else if !g.engaged {
                    g.morale = (g.morale + MORALE_RECOVERY * dt).min(100.0);
                }

                // Inspect-panel readout: per-second rates after all
                // multipliers; casualty rate smoothed over ~1 s.
                let f = &mut readout.0[gi];
                let k = (dt / 1.0).min(1.0);
                let cas_rate = if dt > 0.0 { casualty_drain / dt } else { 0.0 };
                f.casualties += (cas_rate - f.casualties) * k;
                let m = psych_mult * depletion;
                f.outnumbered = if outnumbered { MORALE_OUTNUMBERED * m } else { 0.0 };
                f.flanked = MORALE_FLANKED * flanked * m;
                f.flanked01 = flanked;
                f.contagion = rout_drain.min(MORALE_ROUT_CAP) * m;
                f.disorder = MORALE_DISORDER * disorder01 * m;
                f.friends = friends as u32;
                f.psych_mult = psych_mult;
                f.depletion = depletion;
                f.recovering = recovering;
                if g.morale <= 0.0 {
                    g.state = RegState::Routing { since: *tick };
                    g.order = None;
                    info!("regiment {gi} BREAKS ({} of {} left)", g.count, g.initial_count);
                }
            }
            RegState::Routing { since } => {
                readout.0[gi] = MoraleFactors::default();
                if (g.count as f32) < g.initial_count as f32 * SHATTER_FRAC {
                    g.state = RegState::Shattered;
                    info!("regiment {gi} shatters");
                } else if (*tick - since) as f32 * dt > RALLY_DELAY {
                    // Rally only when clear of enemies.
                    let enemy = field.density(1 - g.team, g.centroid);
                    let roll = hash01(tick.wrapping_mul(0x9E37_79B1) ^ (gi as u32) << 8);
                    if enemy < 0.1 && roll < RALLY_CHANCE * dt {
                        g.state = RegState::Steady;
                        g.morale = 40.0;
                        g.anchor = g.centroid;
                        // Re-dress the ranks where the rout ended, facing
                        // back the way they fled from.
                        g.facing = if g.team == 0 { 0.0 } else { std::f32::consts::PI };
                        g.reform = true;
                        info!("regiment {gi} rallies ({} left)", g.count);
                    }
                }
            }
            RegState::Shattered => {
                readout.0[gi] = MoraleFactors::default();
            }
        }
        g.recent_deaths = 0;
    }
}
