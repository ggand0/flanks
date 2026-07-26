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
/// At-ease engagement range: an idle regiment (either team, not in HOLD)
/// attacks an unbroken enemy regiment that closes to this distance.
const AT_EASE_R: f32 = 40.0;

/// Set once when one side has no fighting regiments left.
#[derive(Resource, Default)]
pub struct BattleOutcome(pub Option<u8>);

pub struct AiPlugin;

impl Plugin for AiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BattleOutcome>()
            .add_systems(
                Update,
                (
                    // Stand down while the player deploys: no enemy
                    // pre-orders, no auto-engagement across the line.
                    (ai_think, auto_engage)
                        .in_set(crate::game_state::BattleInputSet)
                        .run_if(crate::game_state::deployment_done),
                    check_victory,
                ),
            );
    }
}

/// Test scripts own the orders — every automatic order source stands down.
fn scripts_active() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        [
            "FL_TEST_FRONT",
            "FL_TEST_ORDERS",
            "FL_TEST_SURROUND",
            "FL_TEST_ROUT",
            "FL_TEST_FORM",
            "FL_TEST_DIR",
            "FL_TEST_CHARGE",
        ]
            .iter()
            .any(|k| std::env::var(k).is_ok())
    })
}

fn ai_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // FL_ENEMY_STATIC=1 / FL_ARENA=1: practice-dummy modes — the enemy
    // spawns in hold-position (regiments.rs) and the strategy AI stands
    // down, so regiments defend where they stand but never advance.
    *ON.get_or_init(|| {
        !std::env::var("FL_AI").is_ok_and(|v| v == "0")
            && std::env::var("FL_ENEMY_STATIC").is_err()
            && std::env::var("FL_ARENA").is_err()
            && !scripts_active()
    })
}

fn ai_think(
    time: Res<Time>,
    mut groups: ResMut<Groups>,
    outcome: Res<BattleOutcome>,
    config: Res<crate::game_state::BattleConfig>,
    mut next: Local<f32>,
    mut targets: Local<Vec<Option<usize>>>,
) {
    if !config.ai_enabled || !ai_enabled() || outcome.0.is_some() {
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

/// At ease (M2TW): an idle Steady regiment of EITHER team — no order, not
/// engaged, not in hold-position mode — attacks an unbroken enemy regiment
/// that comes within AT_EASE_R on its own initiative, and stands down when
/// an auto-engaged target breaks or dies instead of pursuing it across the
/// map (pursuit is for deliberate player orders). Runs even with FL_AI=0:
/// this is regiment self-preservation, not army strategy — it is also what
/// makes rallied enemies in the AI-less sandbox fight back instead of
/// staring. Auto orders are flagged so the UI click/horn stays silent.
fn auto_engage(time: Res<Time>, mut groups: ResMut<Groups>, mut next: Local<f32>) {
    if scripts_active() {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next || t < 5.0 {
        return;
    }
    *next = t + 1.0;

    let snapshot: Vec<(u8, usize, Vec2, bool)> = groups
        .list
        .iter()
        .map(|g| (g.team, g.count, g.centroid, g.state.is_broken()))
        .collect();
    for g in 0..groups.list.len() {
        let gd = &groups.list[g];
        if gd.count == 0 || gd.state.is_broken() || gd.hold {
            continue;
        }
        // Stand down a finished AUTO attack (target broken or wiped).
        if gd.auto_order
            && let Some(crate::orders::Order::Attack(tg)) = gd.order
        {
            let (_, count, _, broken) = snapshot[tg as usize];
            if count == 0 || broken {
                let gd = &mut groups.list[g];
                gd.order = None;
                gd.auto_order = false;
                gd.anchor = gd.centroid;
                info!("regiment {g} stands down (at-ease target finished)");
            }
            continue;
        }
        if gd.order.is_some() || gd.engaged {
            continue;
        }
        let team = gd.team;
        let from = gd.centroid;
        let Some((tg, d2)) = snapshot
            .iter()
            .enumerate()
            .filter(|(_, (tt, count, _, broken))| *tt != team && *count > 0 && !broken)
            .map(|(tg, (_, _, c, _))| (tg, c.distance_squared(from)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
        else {
            continue;
        };
        if d2 < AT_EASE_R * AT_EASE_R {
            let gd = &mut groups.list[g];
            gd.order = Some(crate::orders::Order::Attack(tg as u32));
            gd.auto_order = true;
            let dir = snapshot[tg].2 - gd.centroid;
            if gd.shape == crate::formation::FormShape::Rect && dir.length() > 1.0 {
                gd.facing = crate::formation::facing_of(dir);
                gd.reform = true;
            }
            info!("regiment {g} engages regiment {tg} at ease");
        }
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
