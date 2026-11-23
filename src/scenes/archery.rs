//! FL_TEST_ARCHERY: the archer sandbox, its blocked and side marches,
//! and its log.

use bevy::prelude::*;

use crate::orders::{GroupData, Groups};
use crate::regiments::spawn_regiment;
use crate::terrain::Terrain;
use crate::unit_types::{KIND_ARCHER, KIND_HEAVY, KIND_LIGHT, KIND_SPEAR};
use crate::units::Units;

/// FL_TEST_ARCHERY: the archer sandbox (also on the menu's debug list).
/// Two blue archer regiments stand behind a friendly light screen. The
/// enemy is fully STATIC: the middle orange block holds INSIDE bow
/// range (fire-at-will opens up within seconds), the flank blocks hold
/// just outside it (attack-order the archers onto one to watch the
/// stand-off halt and the arcs over the screen's heads). Nothing
/// chases. FL_ARCHERY_ATTACK=1 adds the old deep attacker for the
/// fire-at-will / lead / skirmish demo.
pub(crate) fn spawn_archery_test(units: &mut Units, terrain: &Terrain, groups: &mut Groups) {
    let size = crate::util::env_or("FL_REG_SIZE", 500_usize).max(50);
    let mut list = Vec::new();
    if std::env::var("FL_ARCHERY_BLOCKED").is_ok() {
        spawn_archery_blocked(units, terrain, &mut list, size);
        groups.list = list;
        return;
    }
    if let Ok(v) = std::env::var("FL_ARCHERY_SIDE") {
        spawn_archery_side(units, terrain, &mut list, size, v == "2");
        groups.list = list;
        return;
    }
    // Blue: light screen up front, two archer blocks behind it.
    spawn_regiment(units, terrain, &mut list, 0, KIND_LIGHT, Vec2::new(0.0, -70.0), size, -1.0);
    list[0].hold = true;
    for x in [-30.0, 30.0] {
        spawn_regiment(units, terrain, &mut list, 0, KIND_ARCHER, Vec2::new(x, -90.0), size, -1.0);
    }
    // Orange: middle block in range (~110 m), flanks just outside.
    // FL_ARCHERY_TARGET=heavy|spear|archer swaps the middle block's kind
    // (the calibration target: kills per volley by armour class). Bowmen
    // there shoot back.
    let mid_kind = match std::env::var("FL_ARCHERY_TARGET").as_deref() {
        Ok("heavy") => KIND_HEAVY,
        Ok("spear") => KIND_SPEAR,
        Ok("archer") => KIND_ARCHER,
        _ => KIND_LIGHT,
    };
    // FL_ARCHERY_MELEE=1: archers behind a line in melee. The static blocks
    // stand beyond bow range and an orange spear block attacks the end
    // of the screen, so the one enemy in the archers' range is in melee
    // and its open flank and rear are what the men must find.
    let melee = std::env::var("FL_ARCHERY_MELEE").is_ok();
    let far = if melee { 70.0 } else { 0.0 };
    for (x, z, kind) in [
        (-60.0, 40.0 + far, KIND_LIGHT),
        (0.0, 20.0 + far, mid_kind),
        (60.0, 40.0 + far, KIND_LIGHT),
    ] {
        spawn_regiment(units, terrain, &mut list, 1, kind, Vec2::new(x, z), size, 1.0);
        let g = list.len() - 1;
        list[g].hold = true;
    }
    // Optional attacker from the west flank, ordered onto the west archer
    // regiment with no friendly block in its way: the skirmish run, or
    // with FL_ARCHERY_NOSKIRMISH=1 the men turning their bows on the
    // block coming for them.
    if std::env::var("FL_ARCHERY_ATTACK").is_ok() {
        spawn_regiment(units, terrain, &mut list, 1, KIND_LIGHT, Vec2::new(-130.0, -30.0), size, 1.0);
        let g = list.len() - 1;
        list[g].order = Some(crate::orders::Order::Attack(1));
        list[g].auto_order = true;
    }
    if std::env::var("FL_ARCHERY_NOSKIRMISH").is_ok() {
        for g in [1, 2] {
            list[g].skirmish = false;
        }
    }
    if melee {
        spawn_regiment(units, terrain, &mut list, 1, KIND_SPEAR, Vec2::new(75.0, -45.0), size, 1.0);
        let g = list.len() - 1;
        list[g].order = Some(crate::orders::Order::Attack(0));
        list[g].auto_order = true;
    }
    // FL_ARCHERY_ORDER=1: both archer regiments ordered at the east
    // flank block, in range by the edge gap: the stand-off must hold at
    // once, no march, no re-dressing.
    if std::env::var("FL_ARCHERY_ORDER").is_ok() {
        for g in [1, 2] {
            list[g].order = Some(crate::orders::Order::Attack(5));
        }
    }
    // FL_ARCHERY_SWORDS=1: both archer regiments with swords out,
    // ordered at the middle block: a charge like any infantry's, no
    // stand-off, no shooting.
    if std::env::var("FL_ARCHERY_SWORDS").is_ok() {
        for g in [1, 2] {
            list[g].melee = true;
            list[g].order = Some(crate::orders::Order::Attack(4));
        }
    }
    groups.list = list;
    info!(
        "[archery] 2 archer regiments behind a screen; orange holds STATIC — middle block in \
         bow range, flanks beyond it (attack-order for the stand-off); FL_ARCHERY_ATTACK=1 \
         adds a charging attacker"
    );
}

/// FL_ARCHERY_BLOCKED=1: the attack order's close-in march with things
/// in the way. The two archer regiments are ordered at an orange block
/// far beyond their range; a friendly block holds across their path, so
/// the men stream round it into a column; a small orange block stands
/// short of the firing place, in the way. The march must stop before
/// that block and shoot it down, then go on to the firing place and
/// shoot the ordered target from there, with no man walking into either
/// block and none walking back. Indices as the base scene's: 0 the
/// friendly block, 1 and 2 the archers, 3 to 5 orange, 5 the target, 4
/// the block in the way.
fn spawn_archery_blocked(
    units: &mut Units,
    terrain: &Terrain,
    list: &mut Vec<GroupData>,
    size: usize,
) {
    spawn_regiment(units, terrain, list, 0, KIND_LIGHT, Vec2::new(0.0, -110.0), size, -1.0);
    list[0].hold = true;
    for x in [-30.0, 30.0] {
        spawn_regiment(units, terrain, list, 0, KIND_ARCHER, Vec2::new(x, -150.0), size, -1.0);
    }
    for (x, z, n) in [(-120.0, 160.0, size), (0.0, -10.0, size / 5), (0.0, 160.0, size)] {
        spawn_regiment(units, terrain, list, 1, KIND_LIGHT, Vec2::new(x, z), n, 1.0);
        let g = list.len() - 1;
        list[g].hold = true;
    }
    for g in [1, 2] {
        list[g].order = Some(crate::orders::Order::Attack(5));
    }
    if std::env::var("FL_ARCHERY_NOSKIRMISH").is_ok() {
        for g in [1, 2] {
            list[g].skirmish = false;
        }
    }
    info!(
        "[archery] blocked march: both archer regiments ordered at the far block (5) past a \
         friendly block across their path and a small orange block (4) short of the firing place"
    );
}

/// FL_ARCHERY_SIDE=1: the partly blocked march at a target off to the
/// side: an archer regiment ordered at an enemy block far off to the
/// right and ahead, a friendly block across the
/// left part of its path only, so part of the regiment is caught in the
/// friend while the rest streams past its end, and a closer enemy block
/// off the path. FL_ARCHERY_SIDE=2: the target walks away slowly, the
/// case of a target drifting about the edge of range. Indices as the
/// base scene's: 0 the friendly block, 1 and 2 the archers, 3 to 5
/// orange, 5 the target, 4 the closer block.
fn spawn_archery_side(
    units: &mut Units,
    terrain: &Terrain,
    list: &mut Vec<GroupData>,
    size: usize,
    walks: bool,
) {
    spawn_regiment(units, terrain, list, 0, KIND_LIGHT, Vec2::new(-40.0, -100.0), size, -1.0);
    list[0].hold = true;
    for x in [-30.0, 30.0] {
        spawn_regiment(units, terrain, list, 0, KIND_ARCHER, Vec2::new(x, -150.0), size, -1.0);
    }
    for (x, z) in [(-120.0, 160.0), (0.0, 60.0), (150.0, 150.0)] {
        spawn_regiment(units, terrain, list, 1, KIND_LIGHT, Vec2::new(x, z), size, 1.0);
        let g = list.len() - 1;
        list[g].hold = true;
    }
    for g in [1, 2] {
        list[g].order = Some(crate::orders::Order::Attack(5));
    }
    if walks {
        list[5].hold = false;
        list[5].order = Some(crate::orders::Order::Move(Vec2::new(170.0, 230.0)));
    }
    if std::env::var("FL_ARCHERY_NOSKIRMISH").is_ok() {
        for g in [1, 2] {
            list[g].skirmish = false;
        }
    }
    info!(
        "[archery] side march: both archer regiments ordered at the far block (5) off to the \
         right, a friendly block across the left part of regiment 1's path, a closer orange \
         block (4) off the path{}",
        if walks { ", the target walking away" } else { "" }
    );
}

/// The blocked march's numbers for one archer regiment: the leading
/// man's distance to the target block's edge and to the block in the
/// way, the men swinging a weapon (in melee), the men inside the
/// friendly block (0), and where the regiment is against its firing
/// place.
fn blocked_march_report(groups: &Groups, units: &Units, g: u32) -> String {
    let gd = &groups.list[g as usize];
    let target = crate::formation::block_rect(&groups.list[5]);
    let way = crate::formation::block_rect(&groups.list[4]);
    let friend = crate::formation::block_rect(&groups.list[0]);
    let (mut near_t, mut near_w, mut melee, mut inside) = (f32::MAX, f32::MAX, 0usize, 0usize);
    for i in 0..units.len() {
        if units.group[i] != g || units.death_t[i] != 0 {
            continue;
        }
        let p = Vec2::new(units.pos[i].x, units.pos[i].z);
        near_t = near_t.min(target.distance(p));
        inside += usize::from(friend.distance(p) == 0.0);
        if groups.list[4].count > 0 {
            near_w = near_w.min(way.distance(p));
        }
        if units.swing[i] & crate::units::SWING_STATE_MASK == crate::units::SWING_WINDUP
            && units.swing[i] & crate::units::SWING_RANGED == 0
        {
            melee += 1;
        }
    }
    let place = gd.fire_place.map_or("none".to_string(), |p| {
        format!("{:.0} m off, anchor {:.0} m off", gd.centroid.distance(p), gd.anchor.distance(p))
    });
    format!(
        " | reg {g}: lead {:.0} m, nearest man {:.0} m from the target, {} from the block in \
         the way, {melee} in melee, {inside} in the friend, hold {:?}, place {place}",
        gd.lead,
        near_t,
        if near_w < f32::MAX { format!("{near_w:.0} m") } else { "-".to_string() },
        gd.skirmish_hold,
    )
}

/// FL_TEST_ARCHERY bookkeeping every 4 s: archer strength + ammo, the
/// live arrow pool, and orange's bleed.
pub(super) fn archery_log(
    groups: Res<Groups>,
    units: Res<Units>,
    arrows: Res<crate::arrows::Arrows>,
    astats: Res<crate::arrows::ArrowStats>,
    time: Res<Time>,
    mut next: Local<f32>,
) {
    if std::env::var("FL_TEST_ARCHERY").is_err() || groups.list.len() < 6 {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 4.0;
    // E engaged, B broken, S skirmishing, O under an order, F firing (the
    // card's mark), H standing off its attack target, h standing instead
    // of walking into an enemy block, M swords out, C charging.
    let flag = |gd: &crate::orders::GroupData| {
        format!(
            "{}{}{}{}{}{}{}{}{}",
            if gd.engaged { "E" } else { "-" },
            if gd.state.is_broken() { "B" } else { "-" },
            if gd.skirmishing { "S" } else { "-" },
            if gd.order.is_some() { "O" } else { "-" },
            if gd.firing { "F" } else { "-" },
            if gd.standoff { "H" } else { "-" },
            if gd.skirmish_hold.is_some() { "h" } else { "-" },
            if gd.melee { "M" } else { "-" },
            if gd.charging { "C" } else { "-" },
        )
    };
    // Men of an archer regiment drawing their bows this tick.
    let drawing = |g: u32| {
        units
            .group
            .iter()
            .zip(units.swing.iter())
            .filter(|&(&gg, &sw)| {
                gg == g
                    && sw & crate::units::SWING_STATE_MASK == crate::units::SWING_WINDUP
                    && sw & crate::units::SWING_RANGED != 0
            })
            .count()
    };
    let drawn = format!(
        " | drawing {}/{} | struck blue {} orange {}",
        drawing(1),
        drawing(2),
        astats.struck[0].iter().sum::<u32>(),
        astats.struck[1].iter().sum::<u32>(),
    );
    let attacker = groups
        .list
        .get(6)
        .map(|a| {
            format!(
                ", attacker {} ({}) at {:.0} m",
                a.count,
                flag(a),
                a.centroid.distance(groups.list[1].centroid)
            )
        })
        .unwrap_or_default();
    let blocked = if std::env::var("FL_ARCHERY_BLOCKED").is_ok()
        || std::env::var("FL_ARCHERY_SIDE").is_ok()
    {
        blocked_march_report(&groups, &units, 1) + &blocked_march_report(&groups, &units, 2)
    } else {
        String::new()
    };
    info!(
        "[archery] t={t:.0}s archers {} ({}, ammo {}) / {} ({}, ammo {}) | flight {} \
         (dropped {}) | screen {} | holders {}/{}/{}{attacker}{drawn}{blocked}",
        groups.list[1].count,
        flag(&groups.list[1]),
        groups.list[1].ammo_left,
        groups.list[2].count,
        flag(&groups.list[2]),
        groups.list[2].ammo_left,
        arrows.len(),
        astats.dropped,
        groups.list[0].count,
        groups.list[3].count,
        groups.list[4].count,
        groups.list[5].count,
    );
}
