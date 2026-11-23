//! FL_TEST_FRONT: both armies marched into contact, a held regiment,
//! then a salient.

use bevy::prelude::*;

use crate::orders::Groups;

/// FL_TEST_FRONT=1: march both armies of regiments into contact to form a
/// battle line, hold one regiment unordered near the front (drift watch),
/// then push it through the line as a salient.
pub(super) fn test_front_script(
    time: Res<Time>,
    mut groups: ResMut<Groups>,
    mut stage: Local<u32>,
    mut next_reorder: Local<f32>,
    mut watch: Local<Option<(u32, Vec2)>>,
) {
    if std::env::var("FL_TEST_FRONT").is_err() {
        return;
    }
    let t = time.elapsed_secs();

    // Stand-in for player/AI: idle unengaged regiments ATTACK their
    // nearest enemy regiment every 15 s so remnant pockets hunt each
    // other down (also exercises attack orders + charge phase).
    if *stage >= 1 && t > *next_reorder {
        *next_reorder = t + 15.0;
        let snapshot: Vec<(u8, usize, Vec2)> = groups
            .list
            .iter()
            .map(|g| (g.team, g.count, g.centroid))
            .collect();
        for (g, group) in groups.list.iter_mut().enumerate() {
            if watch.is_some_and(|(w, _)| w as usize == g) {
                continue; // drift-watch regiment must stay unordered
            }
            if group.count == 0 || group.engaged || group.order.is_some() {
                continue;
            }
            let nearest = snapshot
                .iter()
                .enumerate()
                .filter(|(_, (team, count, _))| *team != group.team && *count > 0)
                .min_by(|a, b| {
                    a.1 .2
                        .distance_squared(group.centroid)
                        .total_cmp(&b.1 .2.distance_squared(group.centroid))
                })
                .map(|(t, _)| t as u32);
            if let Some(t) = nearest {
                group.order = Some(crate::orders::Order::Attack(t));
            }
        }
    }

    match *stage {
        0 if t > 3.0 => {
            // Every regiment advances straight across the gap; blocks keep
            // their x, fronts collide near z = 0.
            for group in groups.list.iter_mut() {
                let dir: f32 = if group.team == 0 { 1.0 } else { -1.0 };
                group.order = Some(crate::orders::Order::Move(Vec2::new(group.anchor.x, dir * 10.0)));
            }
            info!("[front-test] all regiments ordered into contact");
            *stage = 1;
        }
        1 if t > 35.0 => {
            // Hold the blue regiment nearest a spot behind the front, NO
            // order: it must stand fast (units move only when commanded).
            let spot = Vec2::new(40.0, -60.0);
            if let Some((g, _)) = groups
                .list
                .iter()
                .enumerate()
                .filter(|(_, gr)| gr.team == 0 && gr.count > 0)
                .min_by(|a, b| {
                    a.1.centroid
                        .distance_squared(spot)
                        .total_cmp(&b.1.centroid.distance_squared(spot))
                })
            {
                let group = &mut groups.list[g];
                group.anchor = group.centroid;
                group.order = None;
                *watch = Some((g as u32, group.centroid));
                info!("[front-test] regiment {g} held near the front, NO order — watching drift");
            }
            *stage = 2;
        }
        2 if t > 55.0 => {
            if let Some((g, start)) = *watch {
                let drift = groups.list[g as usize].centroid.distance(start);
                info!("[front-test] held regiment {g} drift over 20s: {drift:.2} m");
                // Now push it through the line as a salient.
                groups.list[g as usize].order =
                    Some(crate::orders::Order::Move(Vec2::new(40.0, 120.0)));
                info!("[front-test] salient regiment {g} ordered through the line");
            }
            *stage = 3;
        }
        _ => {}
    }
}
