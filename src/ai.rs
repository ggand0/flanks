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

use crate::game_state::{BattleClock, PerBattle};
use crate::orders::Groups;

/// AI decision cadence.
const THINK_PERIOD: f32 = 2.0;
/// Distance penalty per regiment already targeting the same player
/// regiment (meters-equivalent).
const SPREAD_PENALTY: f32 = 40.0;
const AI_TEAM: u8 = 1;
/// The automatic attack's reach (`auto_engage`): when the battle set-up
/// turns it on, an idle regiment of either team, not in hold mode,
/// attacks an unbroken enemy regiment whose centre comes this close and
/// with no friendly regiment between them.
const AT_EASE_R: f32 = 40.0;

/// Set once when one side has no fighting regiments left.
#[derive(Resource, Default)]
pub struct BattleOutcome(pub Option<u8>);

pub struct AiPlugin;

impl Plugin for AiPlugin {
    fn build(&self, app: &mut App) {
        // On the sim tick, which does not run while the player deploys
        // (no enemy pre-orders, no automatic attack across the line) or
        // while the game is paused. The victory check goes first: the AI
        // stands down on the tick the battle is decided.
        app.init_resource::<BattleOutcome>().add_systems(
            FixedUpdate,
            (check_victory, ai_think, auto_engage)
                .chain()
                .in_set(crate::game_state::ScriptSet::Ai),
        );
    }
}

/// Test scripts own the orders — every automatic order source stands
/// down. Shared with game_state (the old private env list here went
/// stale the moment new scenarios were added: the Archery range's
/// "static" holders charged because FL_TEST_ARCHERY wasn't on it).
fn scripts_active() -> bool {
    crate::game_state::scripts_active()
}

fn ai_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // FL_ENEMY_STATIC=1 / FL_ARENA=1: practice-dummy modes — the enemy
    // spawns in hold-position (regiments.rs, scenes/arena.rs) and the
    // strategy AI stands down, so regiments defend where they stand but
    // never advance.
    // Launch-time knobs stay cached; the scenario check must NOT be
    // (menu buttons swap scenarios between battles).
    let static_on = *ON.get_or_init(|| {
        !std::env::var("FL_AI").is_ok_and(|v| v == "0")
            && std::env::var("FL_ENEMY_STATIC").is_err()
            && std::env::var("FL_ARENA").is_err()
    });
    static_on && !scripts_active()
}

fn ai_think(
    clock: Res<BattleClock>,
    mut groups: ResMut<Groups>,
    outcome: Res<BattleOutcome>,
    config: Res<crate::game_state::BattleConfig>,
    mut state: Local<PerBattle<(f32, Vec<Option<usize>>)>>,
) {
    if !config.ai_enabled || !ai_enabled() || outcome.0.is_some() {
        return;
    }
    let (next, targets) = state.of(&clock);
    let t = clock.secs();
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
    for target in targets.iter_mut() {
        if let Some(tg) = *target
            && !alive_player(&groups, tg)
        {
            *target = None;
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

    for (g, target) in targets.iter_mut().enumerate() {
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
        // Archers weigh armour: an attack order for them is a fire
        // order (the stand-off), and arrows respect armour and shields,
        // so the AI's bows prefer soft targets over near ones. 12 m of
        // march per protection point trades cheaply against the 2-4x
        // damage swing.
        let archer = gr.kind == crate::unit_types::KIND_ARCHER;
        let Some((best, _)) = (0..groups.list.len())
            .filter(|&tg| alive_player(&groups, tg))
            .map(|tg| {
                let mut cost =
                    from.distance(groups.list[tg].centroid) + SPREAD_PENALTY * load[tg] as f32;
                if archer {
                    let pt = &crate::unit_types::TYPES[groups.list[tg].kind as usize];
                    cost += 12.0 * (pt.armour + pt.shield);
                }
                (tg, cost)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
        else {
            continue;
        };
        *target = Some(best);
        load[best] += 1;
        groups.list[g].order = Some(crate::orders::Order::Attack(best as u32));
    }
}

/// The automatic attack, when the battle set-up turns it on
/// (`BattleConfig::at_ease_attack`, the menu's "Units attack on their
/// own", `FL_AT_EASE=1`): once a second, an idle regiment of either team
/// (no order, not engaged, not in hold mode) attacks the nearest unbroken
/// enemy regiment it has a clear way to (no living friendly regiment's
/// centre between them, `Groups::front_on_outline`), if that one's centre
/// is within AT_EASE_R of its own, and stands down when a target it took
/// this way breaks or dies instead of pursuing it across the map. It runs
/// with the enemy AI on or off. Its orders are
/// flagged so the order click and horn stay silent. When the option is
/// off it gives no order and stands nobody down: an order it gave earlier
/// stays until the player changes it or its target dies.
fn auto_engage(
    clock: Res<BattleClock>,
    config: Res<crate::game_state::BattleConfig>,
    mut groups: ResMut<Groups>,
    mut next: Local<PerBattle<f32>>,
) {
    if scripts_active() || !config.at_ease_attack {
        return;
    }
    let next = next.of(&clock);
    let t = clock.secs();
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
        // Archers with their bows never CHARGE at ease: their
        // fire-at-will already engages at bow range, and skirmish mode
        // handles proximity. (Out of ammo they still defend themselves
        // when reached.) With swords out they engage as infantry does.
        if gd.kind == crate::unit_types::KIND_ARCHER && !gd.melee {
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
        // Only an enemy it has a clear way to: no living friendly
        // regiment's centre in the corridor between them.
        let Some((tg, d2)) = snapshot
            .iter()
            .enumerate()
            .filter(|(_, (tt, count, _, broken))| *tt != team && *count > 0 && !broken)
            .filter(|(tg, _)| groups.front_on_outline(g, *tg).is_none())
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
    clock: Res<BattleClock>,
    mut next: Local<PerBattle<f32>>,
) {
    if outcome.0.is_some() || groups.list.is_empty() {
        return;
    }
    let next = next.of(&clock);
    let t = clock.secs();
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
