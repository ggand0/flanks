//! FL_TEST_SURROUND: an encircled pocket against a line fight, and its
//! log.

use bevy::prelude::*;

use crate::units::{Units, hash01, push_unit, spawn_seed};

/// FL_TEST_SURROUND: two equal blue detachments of light infantry, one
/// fully encircled, one fighting line-against-line, both against the same
/// odds. The surrounded pocket must die per-capita faster — that is the
/// acceptance test for swing combat (the old gather model traded 1:1).
/// Region split for bookkeeping: pocket at x = -200, line at x = +200.
/// Orange groups get press orders (units advance only under orders). A
/// block order translates a formation (it never converges one), so the
/// encirclement is four arc regiments each ordered inward past the pocket.
pub fn spawn_surround_test(
    units: &mut Units,
    terrain: &crate::terrain::Terrain,
    groups: &mut crate::orders::Groups,
) {
    use crate::orders::{GroupData, Order};
    use crate::unit_types::KIND_LIGHT;
    let pocket_c = Vec2::new(-200.0, 0.0);
    let mut seed = spawn_seed(1);
    let mut list: Vec<GroupData> = Vec::new();

    // Blue pocket (group 0): disc, holds.
    list.push(GroupData::new(0, KIND_LIGHT, pocket_c, 500));
    for _ in 0..500 {
        seed = seed.wrapping_add(1);
        let a = hash01(seed.wrapping_mul(3) + 1) * std::f32::consts::TAU;
        let r = 13.0 * hash01(seed.wrapping_mul(3) + 2).sqrt();
        let (x, z) = (pocket_c.x + r * a.cos(), pocket_c.y + r * a.sin());
        push_unit(units, terrain, seed, 0, KIND_LIGHT, 0, x, z, pocket_c);
    }
    // Orange annulus: four arc regiments (groups 1-4), each pressing its
    // anchor to the pocket center.
    for q in 0..4u32 {
        let a0 = q as f32 * std::f32::consts::FRAC_PI_2;
        let mid = a0 + std::f32::consts::FRAC_PI_4;
        let anchor = pocket_c + Vec2::new(mid.cos(), mid.sin()) * 20.5;
        let g = list.len() as u32;
        let mut gd = GroupData::new(1, KIND_LIGHT, anchor, 500);
        gd.order = Some(Order::Move(pocket_c));
        list.push(gd);
        for _ in 0..500 {
            seed = seed.wrapping_add(1);
            let a = a0 + hash01(seed.wrapping_mul(3) + 1) * std::f32::consts::FRAC_PI_2;
            let rr = 14.0f32 * 14.0
                + (27.0f32 * 27.0 - 14.0 * 14.0) * hash01(seed.wrapping_mul(3) + 2);
            let r = rr.sqrt();
            let (x, z) = (pocket_c.x + r * a.cos(), pocket_c.y + r * a.sin());
            push_unit(units, terrain, seed, 1, KIND_LIGHT, g, x, z, anchor);
        }
    }
    // Line fight: blue block (group 5) holds; orange block (group 6)
    // presses 27 m south through it.
    let blue_anchor = Vec2::new(200.0, -6.0);
    let orange_anchor = Vec2::new(200.0, 21.0);
    let block = |units: &mut Units,
                 seed: &mut u32,
                     team: u8,
                     group: u32,
                     anchor: Vec2,
                     depth: f32,
                     n: usize| {
        for _ in 0..n {
            *seed = seed.wrapping_add(1);
            let x = 200.0 + (hash01(seed.wrapping_mul(3) + 1) - 0.5) * 60.0;
            let z = anchor.y + (hash01(seed.wrapping_mul(3) + 2) - 0.5) * depth;
            push_unit(units, terrain, *seed, team, KIND_LIGHT, group, x, z, anchor);
        }
    };
    list.push(GroupData::new(0, KIND_LIGHT, blue_anchor, 500));
    block(units, &mut seed, 0, 5, blue_anchor, 10.0, 500);
    let mut press = GroupData::new(1, KIND_LIGHT, orange_anchor, 2000);
    // Press to CONTACT, not through: the fight must stay frontal (the
    // pocket is the surrounded case; this is the control).
    press.order = Some(Order::Move(Vec2::new(orange_anchor.x, 13.0)));
    list.push(press);
    block(units, &mut seed, 1, 6, orange_anchor, 40.0, 2000);

    groups.list = list;
}

/// FL_TEST_SURROUND bookkeeping: per-capita survival of the encircled
/// pocket (x < 0) vs the line fight (x > 0). Swing combat's acceptance
/// criterion is the pocket dying clearly faster.
pub(super) fn surround_test_log(units: Res<Units>, time: Res<Time>, mut next: Local<f32>) {
    if std::env::var("FL_TEST_SURROUND").is_err() {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 2.0;
    let mut pocket = 0usize;
    let mut line = 0usize;
    for i in 0..units.len() {
        if units.team[i] == 0 && units.death_t[i] == 0 {
            if units.pos[i].x < 0.0 {
                pocket += 1;
            } else {
                line += 1;
            }
        }
    }
    info!("[surround-test] t={t:.0}s blue alive: pocket {pocket}/500, line {line}/500");
}
