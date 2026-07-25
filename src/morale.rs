//! Regiment morale: the most-tuned system in the game.
//!
//! Model (owner-directed rework, devlog 0055): morale is a LEVEL, not a
//! draining tank — the M2TW engine keeps a per-unit `moraleLevel` plus a
//! list of concurrent situational effects summed onto the unit's base
//! stat, recomputed continuously (M2TWEOP disassembly). Every tick:
//!
//!   effective = base(kind) + casualties + melee exchange + flanked
//!             + rout contagion + fatigue + disorder + secure flanks
//!             + no-enemy calm + wall stance
//!
//! Steady above 0; WAVERING in the 0..-6 band; at -6 or less the
//! regiment breaks (MTW1 official-guide thresholds — CA's design
//! template; M2TW's own numbers were never extracted). A routing
//! regiment keeps recomputing the same level as it flees: when the
//! situation genuinely improves (clear of enemies, contagion gone) the
//! level climbs back over -5 and it rallies — no dice roll. Factor
//! values follow the MTW1 table where M2TW's are unknown; the flank
//! ring, disorder term, and all radii are ours (flagged in the devlog).
//! Per-regiment factor values are published to `MoraleReadout` for the
//! inspect panel.

use bevy::prelude::*;

use crate::frontline::InfluenceField;
use crate::orders::{Groups, RegState};
use crate::unit_types::TYPES;

pub struct MoralePlugin;

impl Plugin for MoralePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MoraleReadout>().add_systems(
            FixedUpdate,
            update_morale
                .after(crate::movement::step_sim)
                .before(crate::orders::clear_arrived_orders)
                .in_set(crate::game_state::SimSet),
        );
    }
}

/// Per-regiment morale modifier values (LEVEL points, after discipline)
/// from the last tick — feeds the inspect panel so the player can SEE
/// why a regiment is wavering.
#[derive(Clone, Copy, Default)]
pub struct MoraleFactors {
    pub base: f32,
    /// Cumulative-casualty penalty (permanent for the battle).
    pub casualties: f32,
    /// Winning (+) / losing (-) the melee exchange, smoothed ~3 s.
    pub exchange: f32,
    pub outnumbered: f32,
    pub flanked: f32,
    /// Raw 0..1 surround fraction behind `flanked`.
    pub flanked01: f32,
    pub contagion: f32,
    /// Visible routing ENEMIES lift spirits.
    pub rout_enemies: f32,
    /// Fighting in disarray (formation disorder past the free band).
    pub disorder: f32,
    pub fatigue: f32,
    /// Steady friendly regiments covering the flanks.
    pub support: f32,
    /// No enemy anywhere near: the calm bonus (also what lets a fled
    /// regiment rally).
    pub no_enemy: f32,
    pub wall: f32,
    /// Army leadership: +2 while the command regiment stands (+8 more
    /// on the commander's own regiment), the death shock / permanent
    /// loss after it breaks.
    pub leader: f32,
    /// The rout lock while broken (0 when steady): decays away, and
    /// until it does the regiment cannot rally.
    pub rout_lock: f32,
    /// The recomputed level (== GroupData::morale).
    pub effective: f32,
    /// Steady friendly regiments counted for support (0..=3).
    pub friends: u32,
}

#[derive(Resource, Default)]
pub struct MoraleReadout(pub Vec<MoraleFactors>);

// --- Morale tuning ---
// MTW1 official-guide values (the numeric template CA carried forward;
// M2TW's own are hardcoded and unpublished) unless flagged otherwise.
/// State bands, MEASURED from the live engine (devlog 0057: base morale
/// plus the summed effect amounts, per state, across 13950 samples).
/// Medians: high +12, firm +3, shaken -3, wavering -7, routing -11.
/// The bands overlap in the engine (documented anti-thrash hysteresis);
/// we take those medians as edges and get the hysteresis from
/// BREAK_HOLD_S plus the rally gap.
const SHAKEN_AT: f32 = -3.0;
const WAVER_AT: f32 = -7.0;
/// Break at this level or less; rally hysteresis re-enters at RALLY_AT.
/// Confirmed by the capture: a Pikemen unit (base 5) broke holding
/// casualties -12 and a -4 neighbour term = exactly -11.
const ROUT_AT: f32 = -11.0;
const RALLY_AT: f32 = -7.0;
/// The ROUT LOCK. On breaking, M2TW injects a -50 effect that is present
/// in essentially every routing sample and in no other state — it is
/// what keeps routers running instead of recovering the moment the
/// threat passes. The engine holds it flat (its rally is driven by
/// separate counters we do not model), so we DECAY ours: a broken
/// regiment can only rally once it is both safe and has had time to
/// collect itself.
const ROUT_LOCK: f32 = -50.0;
const ROUT_LOCK_DECAY_S: f32 = 25.0;
/// Level must sit at/below ROUT_AT this long before the break — the
/// engine's waveringTimer analog; overlapping state bands are the
/// documented anti-thrash device.
const BREAK_HOLD_S: f32 = 1.0;
/// Casualty ladder, MEASURED from the live engine (devlog 0057, proven
/// by bucketing every sample's effect amount against that unit's actual
/// soldiers/soldiersMax): discrete STEPS, and nothing at all below 10%.
/// The 25% step was previously unknown to the community.
const CASUALTY_STEPS: [(f32, f32); 4] =
    [(0.10, -2.0), (0.25, -4.0), (0.50, -8.0), (0.80, -12.0)];
/// Losing combat scales to -8, winning to +6 (MTW1), by the smoothed
/// kill/death exchange. Saturates when combined losses reach
/// EXCHANGE_FULL_RATE of current strength per second (our calibration).
const EXCHANGE_LOSING: f32 = -8.0;
const EXCHANGE_WINNING: f32 = 6.0;
const EXCHANGE_TAU: f32 = 3.0;
const EXCHANGE_FULL_RATE: f32 = 0.015;
/// Locally outnumbered >3:1 (blurred density ratio at the centroid):
/// deliberately minor — a soldier can't count a battlefield (owner
/// direction); FLANKS break formations, not headcounts.
const MORALE_OUTNUMBERED: f32 = -2.0;
/// Fully surrounded penalty peak; MTW1: one flank -2, both/rear -6.
/// The 8-probe ring maps frontal contact to 0, pincer ~0.4, and
/// encirclement 1.0 of this.
const MORALE_FLANKED: f32 = -6.0;
/// Minimum radius of the 8-probe flank ring around the centroid (ours).
/// The live ring is the regiment's own footprint radius + FLANK_MARGIN:
/// probes must sit OUTSIDE our own mass, or a big block swallows the
/// ring and encirclement reads as nothing (rout-test regression, 0055).
const FLANK_RING_R: f32 = 22.0;
const FLANK_MARGIN: f32 = 10.0;
/// Enemy blurred density at a ring probe below which a sector can't be
/// hostile regardless of ratio (noise floor).
const FLANK_T: f32 = 0.6;
/// A sector is hostile only when enemy density exceeds OWN density by
/// this factor there — probes landing inside our own block or the
/// front-line mixing zone must not count (a frontal press ≠ a flank).
const FLANK_DOMINANCE: f32 = 1.2;
/// Routing friendly regiments within NEIGHBOR_R — the documented MTW
/// curve (primary source, devlog 0055 round 2): -6 per WEIGHTED routing
/// unit, saturating at two units (-12). Routers are weighted by class
/// against observer discipline (rout_weight): drilled troops half-count
/// lesser men streaming past — the built-in anti-cascade anchor.
const CONTAGION_PER: f32 = -6.0;
const CONTAGION_SAT: f32 = 2.0;
/// Routing ENEMY regiments in view: +4 per weighted unit, cap two (+8).
const ROUT_ENEMY_PER: f32 = 4.0;
/// "Flanks secure" +4 (MTW1): earned by steady friendlies nearby,
/// eroded as the flank ring finds hostiles.
const SUPPORT_MAX: f32 = 4.0;
/// "No enemy nearby" +4 (MTW1) — the calm term; doubles as the rally
/// pull once a fled regiment gets clear.
const NO_ENEMY_BONUS: f32 = 4.0;
/// Wall stances: braced men stand steadier — the RTW formation-morale
/// analog (phalanx +2).
const WALL_BONUS: f32 = 2.0;
/// Fighting in complete disarray, peak. NOT M2TW-evidenced: carried
/// over from the previous owner-approved model (formations die when
/// they lose shape in contact), demoted to a modest level term.
const MORALE_DISORDER: f32 = -3.0;
const DISORDER_FREE: f32 = 2.0;
const DISORDER_FULL: f32 = 7.0;
/// Routing-neighbor / support / rout-enemy radius (ours; the M2TW
/// radii were never extracted).
const NEIGHBOR_R: f32 = 60.0;
/// Broken regiments below this fraction of initial strength shatter
/// (never rally, flee until despawn). OURS, not M2TW: the engine keeps
/// routing units routing — the capture caught a Pikemen unit fleeing
/// with 1 man of 120 left (devlog 0057). With the measured rout line at
/// -11, regiments break so late that a 15% floor made every break
/// shatter instantly and the rout phase disappeared; 3% restores it.
const SHATTER_FRAC: f32 = 0.03;
/// Seconds routing before rally is possible (engine minRoutDelay analog).
const RALLY_DELAY: f32 = 8.0;
/// Leadership. M2TW: every army is LED — a captain when no general is
/// present. RTW engine formula (Feral docs): army-wide bonus =
/// 2 + command + influence/2 + TroopMorale, i.e. a bare captain still
/// floors at +2 for the whole army. The nearby aura radius at zero
/// command is ~6 m (6 + 7xcommand + 4xinfluence) — effectively nothing,
/// so the aura waits for real generals with stars.
const LEADER_ALIVE: f32 = 2.0;
/// The commander's OWN regiment carries far more: MEASURED +8 (devlog
/// 0057, effect id 3 — it appeared on the general's bodyguard and on no
/// other unit, matching Feral's RTW documentation exactly).
const LEADER_SELF: f32 = 8.0;
/// Leader falls: -8 for a few seconds, then -2 for the rest of the
/// battle and the alive bonus is gone (MTW official table; M2TW's own
/// shock size unextracted).
const LEADER_SHOCK: f32 = -8.0;
const LEADER_SHOCK_S: f32 = 10.0;
const LEADER_LOST_PERM: f32 = -2.0;

/// Display normalization for bars/cards: full at the unit's calm base,
/// empty exactly at the rout line.
pub fn morale01(gd: &crate::orders::GroupData) -> f32 {
    let base = TYPES[gd.kind as usize].base_morale;
    ((gd.morale - ROUT_AT) / (base - ROUT_AT)).clamp(0.0, 1.0)
}

/// MTW discipline weighting of a routing unit in view: elites and
/// drilled troops count men LESS disciplined than themselves as half a
/// unit; undrilled men count every rout in full ("disciplined units
/// care little for routing peasants").
fn rout_weight(observer_kind: u8, router_kind: u8) -> f32 {
    let od = TYPES[observer_kind as usize].discipline;
    let rd = TYPES[router_kind as usize].discipline;
    if od < 1.0 && rd > od { 0.5 } else { 1.0 }
}

/// Measured casualty ladder (fraction lost -> level penalty).
fn casualty_penalty(lost_frac: f32) -> f32 {
    let mut pen = 0.0;
    for (at, amount) in CASUALTY_STEPS {
        if lost_frac >= at {
            pen = amount;
        }
    }
    pen
}

/// Per-tick regiment morale update (serial; ~200 rows). Runs after the
/// damage apply pass so `recent_deaths`/`recent_kills` are this tick's
/// tally, and after fatigue so the fatigue penalty is current.
#[allow(clippy::too_many_arguments)] // bevy system params
pub fn update_morale(
    mut groups: ResMut<Groups>,
    field: Res<InfluenceField>,
    time: Res<Time>,
    mut readout: ResMut<MoraleReadout>,
    mut tick: Local<u32>,
    mut deaths_ema: Local<Vec<f32>>,
    mut kills_ema: Local<Vec<f32>>,
    mut leader_lost: Local<[u32; 2]>,
) {
    *tick += 1;
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }
    let n = groups.list.len();
    readout.0.resize(n, MoraleFactors::default());
    deaths_ema.resize(n, 0.0);
    kills_ema.resize(n, 0.0);

    // Leadership: every army is LED (a captain when no general). The
    // command regiment is assigned once — the first heavy regiment
    // (the captain rides with the armor), else the first alive.
    for t in 0..2u8 {
        if !groups.list.iter().any(|g| g.leader && g.team == t) {
            let pick = groups
                .list
                .iter()
                .position(|g| {
                    g.team == t && g.count > 0 && g.kind == crate::unit_types::KIND_HEAVY
                })
                .or_else(|| groups.list.iter().position(|g| g.team == t && g.count > 0));
            if let Some(i) = pick {
                groups.list[i].leader = true;
            }
        }
    }
    // Army-wide leadership term per team: +2 while the command regiment
    // stands; when it breaks, the death shock then the permanent loss.
    // A rallied command regiment takes back up its standard.
    let mut leader_term = [0.0f32; 2];
    for t in 0..2usize {
        let standing = groups
            .list
            .iter()
            .any(|g| g.leader && g.team == t as u8 && g.count > 0 && !g.state.is_broken());
        if standing {
            if leader_lost[t] != 0 {
                info!("team {t} commander returns to the field");
            }
            leader_lost[t] = 0;
            leader_term[t] = LEADER_ALIVE;
        } else if groups.list.iter().any(|g| g.leader && g.team == t as u8) {
            if leader_lost[t] == 0 {
                leader_lost[t] = *tick;
                info!("team {t} LOSES ITS COMMANDER");
            }
            let since = (*tick - leader_lost[t]) as f32 * dt;
            leader_term[t] = if since < LEADER_SHOCK_S {
                LEADER_SHOCK
            } else {
                LEADER_LOST_PERM
            };
        }
    }

    // Snapshot broken/steady centroids for the neighbor terms (state
    // from last tick is fine — morale contagion is not latency-sensitive).
    let broken_centroids: Vec<(u8, u8, Vec2)> = groups
        .list
        .iter()
        .filter(|g| g.state.is_broken() && g.count > 0)
        .map(|g| (g.team, g.kind, g.centroid))
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
            g.recent_kills = 0;
            readout.0[gi] = MoraleFactors::default();
            continue;
        }
        let params = &TYPES[g.kind as usize];

        // Smoothed melee exchange (deaths suffered vs kills made, per
        // second): raw per-tick tallies are unreadable spikes.
        let k = (dt / EXCHANGE_TAU).min(1.0);
        deaths_ema[gi] += (g.recent_deaths as f32 / dt - deaths_ema[gi]) * k;
        kills_ema[gi] += (g.recent_kills as f32 / dt - kills_ema[gi]) * k;
        g.recent_deaths = 0;
        g.recent_kills = 0;

        // --- The modifier sum: identical for steady and routing
        // regiments — the level IS the state of mind; the state machine
        // below just reads it.
        let f = &mut readout.0[gi];
        *f = MoraleFactors::default();
        f.base = params.base_morale;

        // Casualties: permanent for the battle, but a plateau, not a
        // pump — when the killing stops, the penalty stops growing.
        let lost = 1.0 - g.count as f32 / g.initial_count.max(1) as f32;
        f.casualties = casualty_penalty(lost);

        // Winning/losing the melee exchange.
        let dr = deaths_ema[gi] / g.count as f32;
        let kr = kills_ema[gi] / g.count as f32;
        let total = dr + kr;
        if total > 1e-6 {
            let net = (kr - dr) / total;
            let intensity = (total / EXCHANGE_FULL_RATE).min(1.0);
            f.exchange = net
                * intensity
                * if net < 0.0 { -EXCHANGE_LOSING } else { EXCHANGE_WINNING };
        }

        // Outflanked. 8 probes on a ring around the centroid; a sector
        // is hostile only where enemy mass DOMINATES own mass (see
        // FLANK_DOMINANCE — absolute density alone made every line
        // fight read as encircled). The largest hostile-free arc is the
        // safe side: a full frontal contact (3 adjacent sectors) scores
        // exactly 0, a pincer ~0.4, encirclement 1.
        let ring_r = (g.radius + FLANK_MARGIN).max(FLANK_RING_R);
        let mut hostile = [false; 8];
        let mut any_hostile = false;
        for (s, h) in hostile.iter_mut().enumerate() {
            let a = s as f32 / 8.0 * std::f32::consts::TAU;
            let p = g.centroid + Vec2::new(a.cos(), a.sin()) * ring_r;
            let e = field.density(1 - g.team, p);
            *h = e > FLANK_T && e > field.density(g.team, p) * FLANK_DOMINANCE;
            any_hostile |= *h;
        }
        let mut safe_run = 0usize;
        let mut run = 0usize;
        for s in 0..16 {
            if hostile[s % 8] {
                run = 0;
            } else {
                run += 1;
                safe_run = safe_run.max(run);
            }
        }
        let safe_arc = safe_run.min(8) as f32 * 45.0;
        let flanked01 = ((225.0 - safe_arc) / 225.0).clamp(0.0, 1.0);
        f.flanked01 = flanked01;

        // Locally outnumbered (blurred density ratio at the centroid):
        // minor and hard to trigger — a clogged line is NOT panic.
        let own = field.density(g.team, g.centroid);
        let enemy = field.density(1 - g.team, g.centroid);
        let outnumbered = enemy / (own + 0.1) > 3.0;

        // Discipline damps the SHOCK terms (flanked, contagion,
        // outnumbered) — the documented M2TW discipline semantic;
        // casualties and exhaustion spare no one.
        let disc = params.discipline;
        f.flanked = MORALE_FLANKED * flanked01 * disc;
        f.outnumbered = if outnumbered { MORALE_OUTNUMBERED * disc } else { 0.0 };

        // Routing friendlies nearby shake resolve; routing ENEMIES in
        // view lift it. Both use the documented class weighting and
        // saturate at two weighted units — discipline decides how much
        // each router COUNTS (rout_weight), so no extra multiplier here.
        let rout_friends: f32 = broken_centroids
            .iter()
            .filter(|(t, _, c)| *t == g.team && c.distance(g.centroid) < NEIGHBOR_R)
            .map(|(_, k, _)| rout_weight(g.kind, *k))
            .sum();
        f.contagion = rout_friends.min(CONTAGION_SAT) * CONTAGION_PER;
        let rout_enemies: f32 = broken_centroids
            .iter()
            .filter(|(t, _, c)| *t != g.team && c.distance(g.centroid) < NEIGHBOR_R)
            .map(|(_, k, _)| rout_weight(g.kind, *k))
            .sum();
        f.rout_enemies = rout_enemies.min(CONTAGION_SAT) * ROUT_ENEMY_PER;

        // Steady friends nearby secure the flanks — full +4 with three
        // of them and a quiet ring, eroded as the ring turns hostile.
        let friends = steady_centroids
            .iter()
            .filter(|(i, t, c)| *i != gi && *t == g.team && c.distance(g.centroid) < NEIGHBOR_R)
            .count()
            .min(3);
        f.friends = friends as u32;
        f.support = SUPPORT_MAX * friends as f32 / 3.0 * (1.0 - flanked01);

        // Calm: no enemy anywhere near. This is also the rally pull —
        // a regiment that outruns its pursuers finds its feet again.
        if !any_hostile && !g.engaged && !g.enemy_near {
            f.no_enemy = NO_ENEMY_BONUS;
        }

        // Fighting in disarray: the formation-shape term, engaged only —
        // a scattered regiment dressing its ranks in peace is fine, the
        // same scatter in melee is how formations die.
        if g.engaged {
            let disorder01 = ((g.disorder - DISORDER_FREE) / (DISORDER_FULL - DISORDER_FREE))
                .clamp(0.0, 1.0);
            f.disorder = MORALE_DISORDER * disorder01;
        }

        // Weary men waver (MTW1 fatigue table).
        f.fatigue = crate::fatigue::morale_penalty(g.fatigue);

        // Braced walls stand.
        if crate::formation::wall_kind(g) != 0 {
            f.wall = WALL_BONUS;
        }

        // The army's commander: present, freshly fallen, or lost. His
        // own regiment stands on the measured +8 as well.
        f.leader = leader_term[g.team as usize];
        if g.leader && !g.state.is_broken() {
            f.leader += LEADER_SELF;
        }

        // The rout lock: a broken regiment carries a heavy penalty that
        // decays, so it keeps running until it has genuinely collected
        // itself (measured -50 on break; the engine holds it flat).
        if let RegState::Routing { since } = g.state {
            let elapsed = (*tick - since) as f32 * dt;
            let k = 1.0 - (elapsed / ROUT_LOCK_DECAY_S).clamp(0.0, 1.0);
            f.rout_lock = ROUT_LOCK * k;
        }

        let eff = f.base
            + f.casualties
            + f.exchange
            + f.outnumbered
            + f.flanked
            + f.contagion
            + f.rout_enemies
            + f.support
            + f.no_enemy
            + f.disorder
            + f.fatigue
            + f.wall
            + f.leader
            + f.rout_lock;
        f.effective = eff;
        g.morale = eff;

        match g.state {
            RegState::Steady => {
                g.shaken = eff <= SHAKEN_AT;
                g.wavering = eff <= WAVER_AT;
                if eff <= ROUT_AT {
                    let held = g.break_ticks as f32 * dt;
                    if held >= BREAK_HOLD_S {
                        g.state = RegState::Routing { since: *tick };
                        g.order = None;
                        g.break_ticks = 0;
                        info!(
                            "regiment {gi} BREAKS at morale {eff:.1} ({} of {} left)",
                            g.count, g.initial_count
                        );
                    } else {
                        g.break_ticks = g.break_ticks.saturating_add(1);
                    }
                } else {
                    g.break_ticks = 0;
                }
            }
            RegState::Routing { since } => {
                g.wavering = false;
                g.shaken = false;
                if (g.count as f32) < g.initial_count as f32 * SHATTER_FRAC {
                    g.state = RegState::Shattered;
                    info!("regiment {gi} shatters");
                } else if (*tick - since) as f32 * dt > RALLY_DELAY && eff > RALLY_AT {
                    // The situation genuinely improved: clear of enemies,
                    // contagion faded, the level climbed back. Rally.
                    g.state = RegState::Steady;
                    g.anchor = g.centroid;
                    // Re-dress the ranks where the rout ended, facing
                    // back the way they fled from.
                    g.facing = if g.team == 0 { 0.0 } else { std::f32::consts::PI };
                    g.reform = true;
                    info!("regiment {gi} rallies at morale {eff:.1} ({} left)", g.count);
                }
            }
            RegState::Shattered => {
                g.wavering = false;
                g.shaken = false;
            }
        }
    }
}
