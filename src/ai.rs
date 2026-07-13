//! Minimal skirmish AI for the enemy side (team 1) + battle outcome.
//!
//! Every couple of seconds, each idle Steady enemy regiment attack-moves
//! onto a player regiment, choosing the closest one but penalizing targets
//! other regiments already committed to (greedy spread → the AI forms a
//! line instead of a pile). Engaged regiments keep pressing (the engaged
//! flag blocks order arrival); routing is morale's business.
//!
//! FL_AI=0 disables. The AI also stands down while any FL_TEST_* script
//! is driving orders, so tests stay deterministic.

use bevy::prelude::*;

use crate::orders::Groups;

/// AI decision cadence.
const THINK_PERIOD: f32 = 2.0;
/// Distance penalty per regiment already targeting the same player
/// regiment (meters-equivalent).
const SPREAD_PENALTY: f32 = 40.0;
const AI_TEAM: u8 = 1;

/// Set once when one side has no fighting regiments left.
#[derive(Resource, Default)]
pub struct BattleOutcome(pub Option<u8>);

pub struct AiPlugin;

impl Plugin for AiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BattleOutcome>()
            .add_systems(Update, (ai_think, check_victory));
    }
}

fn ai_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        if std::env::var("FL_AI").is_ok_and(|v| v == "0") {
            return false;
        }
        // Test scripts own the orders.
        !["FL_TEST_FRONT", "FL_TEST_ORDERS", "FL_TEST_SURROUND", "FL_TEST_ROUT"]
            .iter()
            .any(|k| std::env::var(k).is_ok())
    })
}

fn ai_think(
    time: Res<Time>,
    mut groups: ResMut<Groups>,
    outcome: Res<BattleOutcome>,
    mut next: Local<f32>,
    mut targets: Local<Vec<Option<usize>>>,
) {
    if !ai_enabled() || outcome.0.is_some() {
        return;
    }
    let t = time.elapsed_secs();
    // Stagger the first advance a little so the player gets oriented.
    if t < *next || t < 5.0 {
        return;
    }
    *next = t + THINK_PERIOD;

    targets.resize(groups.list.len(), None);

    // Drop assignments whose target regiment is gone or broken.
    let alive_player = |groups: &Groups, g: usize| {
        let gr = &groups.list[g];
        gr.team != AI_TEAM && gr.count > 0 && !gr.state.is_broken()
    };
    for g in 0..groups.list.len() {
        if let Some(tg) = targets[g]
            && !alive_player(&groups, tg)
        {
            targets[g] = None;
        }
    }
    // Commitment load per player regiment (for the spread penalty).
    let mut load = vec![0u32; groups.list.len()];
    for (g, tg) in targets.iter().enumerate() {
        if groups.list[g].team == AI_TEAM
            && let Some(tg) = tg
        {
            load[*tg] += 1;
        }
    }

    for g in 0..groups.list.len() {
        let gr = &groups.list[g];
        if gr.team != AI_TEAM
            || gr.count == 0
            || gr.state.is_broken()
            || gr.order.is_some()
            || gr.engaged
        {
            continue;
        }
        let from = gr.centroid;
        let Some((best, _)) = (0..groups.list.len())
            .filter(|&tg| alive_player(&groups, tg))
            .map(|tg| {
                let cost =
                    from.distance(groups.list[tg].centroid) + SPREAD_PENALTY * load[tg] as f32;
                (tg, cost)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
        else {
            continue;
        };
        targets[g] = Some(best);
        load[best] += 1;
        groups.list[g].order = Some(crate::orders::Order::Attack(best as u32));
    }
}

/// A side is defeated when it has no Steady regiment with units left.
fn check_victory(
    groups: Res<Groups>,
    mut outcome: ResMut<BattleOutcome>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    if outcome.0.is_some() || groups.list.is_empty() {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next || t < 10.0 {
        return;
    }
    *next = t + 1.0;
    let fighting = |team: u8| {
        groups
            .list
            .iter()
            .any(|g| g.team == team && g.count > 0 && !g.state.is_broken())
    };
    match (fighting(0), fighting(1)) {
        (true, false) => {
            outcome.0 = Some(0);
            info!("VICTORY — the enemy army is broken");
        }
        (false, true) => {
            outcome.0 = Some(1);
            info!("DEFEAT — your army is broken");
        }
        (false, false) => {
            outcome.0 = Some(2);
            info!("MUTUAL DESTRUCTION — both armies are broken");
        }
        _ => {}
    }
}
