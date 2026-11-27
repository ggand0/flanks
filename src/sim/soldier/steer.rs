//! From his wish to his step: his legs and feet, whether he stands, the
//! facing, the slide along blocked ground and the step onto the terrain.

use bevy::math::Vec3Swizzles;
use bevy::prelude::*;

use super::*;

const SEP_STRENGTH: f32 = 60.0;
/// Whether a walker on the walk to his place has stopped walking: he is
/// there, within STEP_STOP; he is a holding soldier parked inside the hold
/// deadzone (`drive`); or his own rule told him to stand where he is this
/// tick (`told_to_stand`: `hold_the_frame` zeroed his wish, a holding
/// soldier of a fighting regiment with a comrade between him and his
/// mark). A wish that is merely small is not arrival.
#[inline]
fn has_arrived(place_dist: f32, parked: bool, told_to_stand: bool) -> bool {
    place_dist <= STEP_STOP || parked || told_to_stand
}
/// Whether a soldier on the walk to his place stands this tick, for his
/// bracing and his stance: a wish at or over the speed of a step makes him
/// a walker; under it a standing soldier still stands, and a walking one
/// stands again only when he has arrived at his place.
#[inline]
fn still_stands(was_stood: bool, wish_under_line: bool, arrived: bool) -> bool {
    wish_under_line && (was_stood || arrived)
}
/// Under the friction contact, a soldier surrounded by bodies does not
/// sprint, he walks: with other bodies overlapping his own on all four
/// sides his wish is this long at most (m/s, a human walk), and he goes at
/// his full speed once a side of him is free (`steer`). How far he is
/// surrounded is read from the side that is pressed least (`press_shares`,
/// CROWD_DEPTH), so the wish goes over from his full speed to the walk
/// without a jump.
const CROWD_WALK: f32 = 1.5;
/// His wish in a press: `surrounded` is how deep the least pressed of his
/// four sides is in a body (0 free, 1 in full). Free, his wish is
/// untouched; surrounded in full it is the walk at most; between the two
/// its length lies between them. The direction is kept, and a wish under
/// the walk is untouched.
#[inline]
fn press_walk(surrounded: f32, wish: Vec2, walk: f32) -> Vec2 {
    let len = wish.length();
    if surrounded <= 0.0 || len <= walk {
        wish
    } else {
        wish * ((len + (walk - len) * surrounded.min(1.0)) / len)
    }
}
/// The friction contact's feet: each tick they take back up to this much
/// (m/s^2) of the speed the contacts have put into him.
const FEET: f32 = 7.0;
/// The feet's take this tick: up to FEET x dt along what the contacts put
/// into his speed.
#[inline]
fn feet_take(imposed: Vec2, dt: f32) -> Vec2 {
    let il = imposed.length();
    if il > 1e-12 { imposed * (il.min(FEET * dt) / il) } else { Vec2::ZERO }
}
/// A soldier thrown clear of the bodies catches his footing: with no body
/// overlapping his, the speed he has no wish for (all of it when he wishes
/// to go nowhere, the part against his wish when he does) his feet keep
/// taking while it is over a step's pace (STEP_PACE), within what is left
/// of the feet's take of one tick (FEET x dt less `taken`). Among bodies
/// nothing changes: there the total his feet take back from the contacts
/// stays a step's pace (`held_total`), and only his flight over open
/// ground is cut short.
#[inline]
fn catch_footing(v: Vec2, wish: Vec2, taken: f32, dt: f32) -> Vec2 {
    let wl = wish.length();
    let unwished = if wl < 1e-3 { v } else { wish * (v.dot(wish).min(0.0) / (wl * wl)) };
    let ul = unwished.length();
    if ul > STEP_PACE {
        unwished * ((ul - STEP_PACE).min((FEET * dt - taken).max(0.0)) / ul)
    } else {
        Vec2::ZERO
    }
}
/// Whether a man of a regiment standing ready stands, below the speed of
/// his steps. Under the friction contact it is where he is trying to go,
/// so a push he gets does not take him out of the ready stance and only
/// his own walking does; else it is how fast he moved last tick. For his
/// bracing (`steer`) that wish leaves out the sidestep of a soldier's try
/// at a side. It is the line a wish crosses, not yet whether he stands:
/// on the walk to his place with nothing else to do a walker stays one
/// until he arrives (`still_stands`).
#[inline]
pub(super) fn stands(friction: bool, kept: Vec2, wish: Vec2) -> bool {
    let v = if friction { wish } else { kept };
    v.length_squared() < STEP_FACE_SPEED * STEP_FACE_SPEED
}
/// Whether his wish this tick is the drive to his place and nothing else
/// (`still_stands`): alive and not broken, his regiment not in its melee
/// (so no joining), himself in formation, no enemy in his reach or in his
/// mind, and no blow, parry or draw under way. His regiment may be
/// engaged: a soldier in the rear ranks of a fighting regiment with no
/// enemy in reach or in mind walks to his place like any other. Fight
/// footwork (`close_in`, `decide_to_join`, `swing`) steers only a soldier
/// this is false for; `hold_the_frame` may still stop him where he stands,
/// which ends his walk (`has_arrived`). The scan reads it before
/// `decide_to_join` and `swing` run, so it sees his formation and his
/// swing as they were kept from last tick.
#[inline]
pub(super) fn on_his_way(s: &Soldier, st: &Step) -> bool {
    !st.dying
        && !st.routed
        && !st.fight.in_melee
        && !st.fight.committed
        && st.fight.join_fp.is_none()
        && st.foe.best_idx == u32::MAX
        && !st.foe.memo_valid
        && *s.swing & crate::units::SWING_STATE_MASK == crate::units::SWING_READY
}
/// What the contacts have put into his speed and his feet have not taken
/// back, after this tick's step: the old total plus the kept speed less
/// the legs' speed, never more than a step's pace.
#[inline]
fn held_total(imposed: Vec2, kept: Vec2, legs_v: Vec2) -> Vec2 {
    (imposed + (kept - legs_v)).clamp_length_max(STEP_PACE)
}
const STEER_GAIN: f32 = 3.0;
const MAX_ACCEL: f32 = 50.0;
/// What a man's legs can do: the steering toward his wish is capped at
/// this acceleration (m/s^2), about what a man in armour with a shield
/// manages from a stand. The gain alone would give 28 m/s^2, three g, at
/// every start and stop, and a lane behind an opening gap would surge
/// and stop as one body. The spring's push and the contact are not legs
/// and keep MAX_ACCEL.
pub(crate) const LEG_ACCEL: f32 = 4.0;
/// How far a lone soldier who wants no speed (a staggered man, `steer`)
/// goes from a speed `u` before he is at rest, ticking his stopping rule
/// under the friction contact: each tick his feet take FEET x dt off what
/// a contact put into his speed, which starts at `u` held to STEP_PACE
/// (`held_total`), and then his legs brake toward zero, STEER_GAIN times
/// his speed but LEG_ACCEL at most. Once his feet are done and his legs
/// are under their cap his speed falls by the same share every tick, and
/// what is left of the way is that geometric sum.
fn rest_distance(u: f32, dt: f32) -> f32 {
    let mut v = u;
    let mut held = u.min(STEP_PACE);
    let mut way = 0.0;
    while held > 0.0 || STEER_GAIN * v > LEG_ACCEL {
        let cut = held.min(FEET * dt);
        v -= cut;
        held -= cut;
        v -= (STEER_GAIN * v).clamp(-LEG_ACCEL, LEG_ACCEL) * dt;
        way += v * dt;
    }
    way + v * (1.0 - STEER_GAIN * dt) / STEER_GAIN
}
/// The speed at which a lone soldier who wants no speed comes to rest `d`
/// metres on (`rest_distance`): under the friction contact the charge
/// blow's shove on his speed, which knocks him back as far as the one-tick
/// jump without it moves him (sim/damage.rs).
/// The way grows with the speed, so the speed is found by halving.
pub(crate) fn shove_speed(d: f32, dt: f32) -> f32 {
    if d <= 0.0 || dt <= 0.0 {
        return 0.0;
    }
    let mut hi = 1.0;
    while rest_distance(hi, dt) < d && hi < 1000.0 {
        hi *= 2.0;
    }
    let mut lo = 0.0;
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if rest_distance(mid, dt) < d { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}
/// Facing turn smoothing (fraction/s of the remaining angle) — the
/// settle behavior for SMALL corrections.
const YAW_RATE: f32 = 10.0;
/// A blooded soldier (one whose combat memo holds a living enemy) keeps
/// facing the fight while that enemy stands within this range, instead
/// of dressing back to the ordered facing between swing cycles. Fresh
/// men still hold the line — this is memory of a fight, not proximity
/// awareness, so rear ambushes keep their first-blood advantage.
const KEEP_FACING_R: f32 = 6.0;
/// Hard angular speed cap (rad/s). The smoothing above is exponential,
/// so before this cap a 180° about-face finished in ~0.2 s — rear-
/// attacked soldiers whipped around between two swings and rear
/// attacks played out as frontal fights. A burdened man in ranks
/// turns ~180° in about a second: pi rad/s. Small angles never hit
/// the cap, so marching wheels and duel tracking feel unchanged.
const TURN_SPEED_MAX: f32 = std::f32::consts::PI;
/// Below this ground speed a standing formation's soldier keeps his
/// facing while he moves: he steps sideways or back, he does not turn
/// to walk (M2TW's shuffle keeps facing). The renderer plays a skinned
/// kind's step clips below it.
pub(crate) const STEP_FACE_SPEED: f32 = 1.05;
/// A standing man plants his feet: a push must exceed this (m/s^2 of
/// separation push) before it moves him. About a 10 cm squeeze from one
/// neighbor. A man holding his place (`planted` in steer) keeps the grip
/// while he steps back to it, and his feet take the first STAND_GRIP x dt
/// of a shared shove too, each man 0.7 to 1.3 of it: a man brushed by a
/// passer stays in his file instead of being carried off. Running men
/// are unaffected, a charge's shove is far above it, and true overlap is
/// still corrected.
pub(crate) const STAND_GRIP: f32 = 6.0;
/// A standing man's stance (his lean, arms and shield) takes the first
/// STANCE metres of a net squeeze before he gives ground to it. Under the
/// friction contact it is how deep a pair's overlap his brace holds
/// (`braced_set`, contact.rs).
pub(super) const STANCE: f32 = 0.10;
/// His own 0.7 to 1.3 of a man-to-man quantity (his grip).
#[inline]
fn own_spread(i: usize) -> f32 {
    0.7 + 0.6 * crate::units::hash01((i as u32).wrapping_mul(0x51ED).wrapping_add(77))
}
/// A regiment in its charge phase runs the last stretch home.
const CHARGE_SPEED_BOOST: f32 = 1.15;
/// Walls advance at a deliberate pace — breaking into a run breaks the wall.
const WALL_SPEED_FRAC: f32 = 0.72;

/// From drive to velocity: the wall and charge paces, fatigue, the
/// stagger, the crowd's damping, a standing man's grip on the ground and
/// his stance against a squeeze, the acceleration and speed caps, and the
/// shared speed with a body he runs into. Under the friction contact:
/// the crowd walk and whether he stands, the feet's take and his legs
/// alone, the damping and the speed cap.
#[inline]
pub(super) fn steer(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { speed, orders, wall, charging, fat_speed, fat_nocharge, dt, .. } = *f;
    let i = s.i;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    // Holding his place: his regiment has no order and he is not broken,
    // dying, in its melee or out of formation for a fight.
    let planted = orders[gi].is_none() && !dying && !routed && !st.fight.in_melee && !st.fight.committed;
    let push = st.scan.push;
    let jam = st.walk.jam;
    let mut desired = st.desired;
    let friction = friction_contact();
    crowd_walk(f, st, friction, &mut desired);
    if friction {
        keep_stood(s, st, desired);
    }
    // Formation pace: walls advance deliberately (running breaks a wall), the
    // charge phase runs the last stretch home. Broken/dying already excluded
    // from both states by construction.
    if wall[gi] != 0 {
        desired *= WALL_SPEED_FRAC;
    } else if charging[gi] && !dying && !routed && !fat_nocharge[gi] {
        desired *= CHARGE_SPEED_BOOST;
    }
    desired *= fat_speed[gi];
    // A staggered man reels where the blow left him: no steering, no closing,
    // until the stun runs out. The shove that staggered him still resolves
    // through separation — he is a body, not an actor.
    if *s.swing & crate::units::SWING_STAGGERED != 0 {
        desired = Vec2::ZERO;
    }

    let mut v = s.vel.xz();
    if friction {
        // His feet take back part of what the contacts put into his speed,
        // off his speed and off the total alike, before his legs act.
        let cut = feet_take(*s.imposed, dt);
        v -= cut;
        *s.imposed -= cut;
        // Thrown clear of the bodies he catches his footing
        // (`catch_footing`). A staggered soldier reels and does not.
        if st.scan.corr.length_squared() < 1e-12 && *s.swing & crate::units::SWING_STAGGERED == 0 {
            v -= catch_footing(v, desired, cut.length(), dt);
        }
    }
    // Jammed units stop shoving entirely: at full jam the crowd is
    // quasi-static and overlap resolution is purely positional — force-based
    // separation in a wedged mass only produces bang-bang oscillation.
    let mut push_a = push * (SEP_STRENGTH * (1.0 - jam));
    // A standing man plants his feet: small pushes do not move him, real
    // shoves do (less the grip). Without it any squeeze turned straight into
    // sliding, and a fight's jostle rippled back through packed ranks. His
    // stance takes the first STANCE of a net squeeze the same way: he gives
    // ground only to what is beyond it. A man holding his place keeps the
    // grip while he steps back to it.
    // Under the friction contact there is no spring, grip or stance: his
    // legs alone move him, and the setting apart is applied in full.
    let standing = desired.length_squared() < 1e-6;
    if !friction && (standing || planted) {
        let pl = push_a.length();
        push_a *= if pl > STAND_GRIP { (pl - STAND_GRIP) / pl } else { 0.0 };
    }
    if !friction && standing && st.scan.overlap_depth > 0.0 && *s.swing & crate::units::SWING_STAGGERED == 0 {
        st.scan.corr *= ((st.scan.overlap_depth - STANCE) / st.scan.overlap_depth).max(0.0);
    }
    let mut legs = (desired - v) * STEER_GAIN;
    let l2 = legs.length_squared();
    if l2 > LEG_ACCEL * LEG_ACCEL {
        legs *= LEG_ACCEL / l2.sqrt();
    }
    let mut accel = if friction { legs } else { legs + push_a };
    let a2 = accel.length_squared();
    if a2 > MAX_ACCEL * MAX_ACCEL {
        accel *= MAX_ACCEL / a2.sqrt();
    }

    let mut new_v = v + accel * dt;
    // Viscous damping in the press: bleeds the spring energy that otherwise
    // ping-pongs between neighbors every tick.
    new_v *= 1.0 - 0.4 * st.walk.body_jam;
    // Slight overspeed under crowd pressure; a man squeezing through a
    // crack goes at a walk at most.
    let vmax = if squeezing(*s.stuck) { SQUEEZE_CAP } else { speed[i] * 1.15 };
    let v2 = new_v.length_squared();
    if v2 > vmax * vmax {
        new_v *= vmax / v2.sqrt();
    }
    // The contact. Shared speed: a man gives up his share by weight of the
    // closing speed on the bodies he overlaps, measured against their own
    // movement, and each of the two does the same, so a runner and the man
    // he hits move on together and a shove travels through a press. The
    // friction contact does not have it.
    let dir = if friction { Vec2::ZERO } else { st.scan.overlap_dir };
    if dir != Vec2::ZERO {
        let into = (new_v - st.scan.overlap_vel).dot(-dir);
        if into > 0.0 {
            // A man holding his place loses less of it: his feet take the
            // first part of the shove.
            let mut lost = into * st.scan.overlap_share;
            if planted {
                lost = (lost - STAND_GRIP * dt * own_spread(i)).max(0.0);
            }
            new_v += dir * lost;
        }
    }
    st.walk.new_v = new_v;
}

/// The crowd walk (`steer`, CROWD_WALK): his wish cut to a walk when
/// bodies press on all four sides of him.
#[inline]
fn crowd_walk(f: &Field, st: &mut Step, friction: bool, desired: &mut Vec2) {
    let Field { charging, fat_nocharge, .. } = *f;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    // The crowd walk under the friction contact: with bodies overlapping
    // his own on all four sides he does not sprint, he walks, by the side
    // pressed least, without a jump. This is the one point every stage's
    // wish has reached (the drive, the join, the closing in, the sidestep),
    // so his whole wish with what the crowd's brake took off it put back is
    // cut to the walk, and both parts by the same share, as if it had been
    // cut before the brake: the brake and the standing test (`keep_stood`)
    // see the walk. A routed soldier flees, a dying one is not steering, and a
    // regiment in its charge runs by order.
    let charge_run = charging[gi] && !fat_nocharge[gi];
    if friction && !dying && !routed && !charge_run {
        let whole = *desired + st.walk.fade;
        let p4 = st.scan.press4;
        let cut = press_walk(p4[0].min(p4[1]).min(p4[2]).min(p4[3]), whole, CROWD_WALK);
        if cut != whole {
            let k = cut.length() / whole.length();
            *desired *= k;
            st.walk.fade *= k;
        }
    }
}

/// Whether he stands this tick (`steer`), kept in his stood bit under the
/// friction contact.
#[inline]
fn keep_stood(s: &mut Soldier, st: &Step, desired: Vec2) {
    let dying = st.dying;
    // Whether he stands, for his bracing next tick (`brace_pairs`) and his
    // stance (`scan`). The line is his own wish as it comes here, after
    // every stage that sets it, with what the crowd's brake took off it put
    // back and the sidestep of his try at a side taken out, before the
    // paces and caps of `steer`, under the speed of a step (`stands`). A push he
    // gets does not count, and neither does that sidestep: a jammed soldier
    // who tries a side still stands, or his weight against his own side
    // would change each time a try begins or ends. A wish at or over the
    // line makes him a walker. Under it, a soldier under fight footwork
    // stands at once; a soldier on the walk to his place with nothing else
    // to do (`on_his_way`) who was walking keeps walking until he arrives
    // there, within STEP_STOP or parked in the hold deadzone, or until his
    // own rule tells him to stand where he is: a holding soldier of a
    // fighting regiment shoved off his mark with a comrade between him and
    // it, whose wish `hold_the_frame` zeroed (`has_arrived`,
    // `still_stands`). That holds in the rear ranks of a fighting regiment
    // too. His wish slides with his distance to his place and crosses the
    // line some metres out, and the few centimetres a shove moves a jammed
    // soldier there would else change his weight and stance again and
    // again.
    let side = st.walk.side;
    let under = stands(true, Vec2::ZERO, desired + st.walk.fade - side);
    let stood = if on_his_way(s, st) {
        let arrived = has_arrived(st.walk.place_dist, st.walk.parked, st.sight.slot_blocked);
        still_stands(*s.stood, under, arrived)
    } else if st.fight.joining {
        // A soldier walking to the fight is a walker until the still
        // men ahead have stopped him behind a comrade in his way: the
        // brake has taken his wish under the speed of a step. From
        // then he stands, and `drive` gives him no wish while that
        // comrade is in his way. Keeping his elbow room (`elbow_room`) he
        // stands only once `drive` has stopped him by sight.
        under || (!elbow_room() && st.sight.way_blocked && stands(true, Vec2::ZERO, desired - side))
    } else {
        under
    };
    // The tick he leaves his place to join the fight he sets off: a
    // walker from then, whatever stood him at his place. Else a soldier
    // who stood within a step of the man ahead when he joined would
    // count as arrived at once and never close up, and the regiment's
    // ranks would end at two different spacings.
    let stood = stood && !(st.fight.set_off && joins_as_walker());
    // A falling soldier keeps his stood bit as it was when he was
    // struck (`fall`).
    if !(dying && fall()) {
        *s.stood = stood;
    }
}

/// Which way he looks: the man he swings at, the enemy in reach when he
/// is fighting or just struck, where he is going when out of formation,
/// the foe he remembers close by, else the line's facing, at the human
/// turn rate. Under the facing rule (`face_keep`) whether he stands or
/// walks is read from his wish, not his speed, and walking he faces his
/// wish.
#[inline]
pub(super) fn face(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { pos_prev, team, threat, form_face, dt, .. } = *f;
    let i = s.i;
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    let committed = st.fight.committed;
    let memo_dir = st.sight.memo_dir;
    let best_idx = st.foe.best_idx;
    let face_target = st.fight.face_target;
    let new_v = st.walk.new_v;
    // Facing priority: locked wind-up target > nearest enemy in reach >
    // movement direction. Fighters keep eyes on the enemy even while the
    // crowd shoves them; only routing/unengaged units face their velocity.
    // yaw_prev snapshots the pre-update angle so the renderer can interpolate
    // (yaw stepped once per tick otherwise — visible facing snaps at high
    // fps).
    *s.yaw_prev = *s.yaw;
    let wish2 = st.desired.length_squared();
    let face_dir = match face_target {
        Some(t) => t - p,
        // A man IN his swing cycle faces the fight (a formed one too — he is
        // the fighting rim), and a man JUST STRUCK turns toward the blow
        // (flash). A man merely NEAR an enemy does not: turning on proximity
        // raced the attacker's wind-up and had every rear-approached victim
        // frontal by first blood — the whole point of facing, gone. So the
        // first hit lands in the back, spins its victim, and THEN he answers.
        // Unformed units (Blob, no facing claim) still turn on proximity.
        None if !routed
            && best_idx != u32::MAX
            && ((*s.swing & crate::units::SWING_STATE_MASK)
                != crate::units::SWING_READY
                || *s.flash > 0
                || form_face[gi] == Vec2::ZERO) =>
        {
            pos_prev[best_idx as usize].xz() - p
        }
        // Out of formation with no enemy in reach: he faces where he is
        // going, the enemy he remembers or the fight he heads for, standing
        // or walking. Without this a blocked joiner fell through to the
        // formed man's rule below and stood facing the line's front with the
        // fight beside him.
        None if !face_keep() && !routed && committed && memo_dir != Vec2::ZERO => memo_dir,
        // Under the facing rule (`face_keep`), the four arms below and the
        // walking arm at the end.
        // Standing or stepping, he turns to the enemy he remembers once
        // that enemy is within his threat distance, in his rank or out.
        None if face_keep()
            && !routed
            && wish2 < STEP_FACE_SPEED * STEP_FACE_SPEED
            && (*s.target as usize) < pos_prev.len()
            && team[*s.target as usize] != team[i]
            && f.death_snap[*s.target as usize] == 0
            && pos_prev[*s.target as usize].xz().distance_squared(p) < threat_radius() * threat_radius() =>
        {
            pos_prev[*s.target as usize].xz() - p
        }
        // In his rank, standing at his place, he dresses to the formation's
        // facing.
        None if face_keep()
            && !routed
            && !committed
            && wish2 < STEP_FACE_SPEED * STEP_FACE_SPEED
            && form_face[gi] != Vec2::ZERO
            && (st.walk.parked || st.walk.place_dist <= STEP_STOP) =>
        {
            form_face[gi]
        }
        // Standing still with his back to the way his regiment faces, in
        // his rank or out of it, he turns round to it.
        None if face_keep()
            && !routed
            && wish2 < 0.2 * 0.2
            && form_face[gi] != Vec2::ZERO
            && Vec2::new(s.yaw.sin(), s.yaw.cos()).dot(form_face[gi]) < 0.0 =>
        {
            form_face[gi]
        }
        // Else, standing or stepping, he keeps the facing he has: a formed
        // soldier short of his place, and a soldier out of his rank who has
        // stopped. (An unformed regiment's standing watch is further down.)
        None if face_keep()
            && !routed
            && wish2 < STEP_FACE_SPEED * STEP_FACE_SPEED
            && (form_face[gi] != Vec2::ZERO || committed) =>
        {
            Vec2::ZERO
        }
        // Blooded and the enemy still close: a man who has traded blows keeps
        // facing the fight while his last foe (combat memo, validated by team
        // and distance like the closing drive) stands within KEEP_FACING_R —
        // no parade dressing with a sword a few strides away. A player reform
        // takes hold once the ground near him clears. Fresh men fall through
        // and hold the ordered line.
        None if !routed
            && new_v.length_squared() < STEP_FACE_SPEED * STEP_FACE_SPEED
            && form_face[gi] != Vec2::ZERO
            && (*s.target as usize) < pos_prev.len()
            && team[*s.target as usize] != team[i]
            && pos_prev[*s.target as usize]
                .xz()
                .distance_squared(p)
                < KEEP_FACING_R * KEEP_FACING_R =>
        {
            pos_prev[*s.target as usize].xz() - p
        }
        // Standing in formation: HOLD the ordered facing. M2TW rule — a
        // formed unit never rotates itself toward a threat (it goes "ready"
        // in place; the render brace pose keys off enemy_near, not yaw);
        // facing is the player's job, and leaving a flank open is supposed to
        // cost.
        None if !routed
            && new_v.length_squared() < STEP_FACE_SPEED * STEP_FACE_SPEED
            && form_face[gi] != Vec2::ZERO =>
        {
            form_face[gi]
        }
        // Standing watch WITHOUT a formation claim (Blob mobs, rallied
        // remnants): face the enemy mass instead of keeping a stale yaw.
        None if !routed
            && new_v.length_squared() < STEP_FACE_SPEED * STEP_FACE_SPEED
            && threat[gi] != Vec2::ZERO =>
        {
            threat[gi]
        }
        // Walking, he faces the way he means to walk: a speed the bodies
        // round him put into him does not turn him.
        None if face_keep() && !routed => st.desired,
        None => new_v,
    };
    let min_len2 = if face_target.is_some() || best_idx != u32::MAX {
        1e-4 // enemies can be close; still face them
    } else {
        0.25 // velocity facing ignores micro-drift
    };
    // A staggered man cannot even turn — the stun freezes his facing, so a
    // charge's second blow finds the same back the first one hit.
    if !dying
        && *s.swing & crate::units::SWING_STAGGERED == 0
        && face_dir.length_squared() > min_len2
    {
        let target_yaw = face_dir.x.atan2(face_dir.y);
        let diff = (target_yaw - *s.yaw + std::f32::consts::PI)
            .rem_euclid(std::f32::consts::TAU)
            - std::f32::consts::PI;
        let step = diff * (YAW_RATE * dt).min(1.0);
        *s.yaw += step.clamp(-TURN_SPEED_MAX * dt, TURN_SPEED_MAX * dt);
    }
}

/// `FL_EDGE_STOP=0` leaves a man the velocity an edge of impassable
/// ground stopped (`slide`): he is then turned to it and keeps it.
fn edge_stop() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("FL_EDGE_STOP").is_ok_and(|v| v == "0"))
}

/// A man's velocity after a step that an edge stopped on the axes given:
/// on such an axis it loses the part that points the way the step went.
/// A velocity that points away from the edge stays.
#[inline]
fn stopped_velocity(v: Vec2, step: Vec2, stop_x: bool, stop_z: bool) -> Vec2 {
    Vec2::new(
        if stop_x && v.x * step.x > 0.0 { 0.0 } else { v.x },
        if stop_z && v.y * step.y > 0.0 { 0.0 } else { v.y },
    )
}

/// The step: velocity into a new place, clamped to the field, sliding
/// along impassable ground. Where an edge stops the step on an axis, his
/// velocity loses that axis too: he keeps no speed into a rail or a wall
/// he cannot pass. Before `face`, so he is turned to the way he moves,
/// not to the way he was pushed at the edge.
#[inline]
pub(super) fn slide(f: &Field, i: usize, st: &mut Step) {
    let Field { terrain, pos_prev, dt, bounds_min, bounds_max, .. } = *f;
    let new_v = st.walk.new_v;
    let corr = st.scan.corr;
    let mut nx = (pos_prev[i].x + new_v.x * dt + corr.x)
        .clamp(bounds_min.x, bounds_max.x);
    let mut nz = (pos_prev[i].z + new_v.y * dt + corr.y)
        .clamp(bounds_min.y, bounds_max.y);
    // Impassable ground (terrace risers, gorge walls, crater lips, a
    // wall's edges): wall-slide, keeping the axis that stays on walkable
    // ground and dropping the one that doesn't, so crowds flow along the
    // obstacle. His body stops at the edge, half the body distance
    // (`Terrain::body_radius`) out from his centre.
    let (px, pz) = (pos_prev[i].x, pos_prev[i].z);
    let r = terrain.body_radius();
    let step = Vec2::new(nx - px, nz - pz);
    if !terrain.body_free(nx, nz, r, step.x, step.y) {
        let (stop_x, stop_z) = if terrain.body_free(nx, pz, r, step.x, 0.0) {
            nz = pz;
            (false, true)
        } else if terrain.body_free(px, nz, r, 0.0, step.y) {
            nx = px;
            (true, false)
        } else {
            nx = px;
            nz = pz;
            (true, true)
        };
        if edge_stop() {
            st.walk.new_v = stopped_velocity(new_v, step, stop_x, stop_z);
        }
    }
    // An edge across his motion over blocked ground (a crowd pushed him
    // against a wall's face, then he walked along it): the step takes him
    // off it, so he climbs a stair up its middle instead of with his
    // shoulder on the wall.
    if r > 0.0 {
        let off = terrain.body_overlap(nx, nz, r);
        if off != Vec2::ZERO && !terrain.blocked_at(nx + off.x, nz + off.y) {
            nx = (nx + off.x).clamp(bounds_min.x, bounds_max.x);
            nz = (nz + off.y).clamp(bounds_min.y, bounds_max.y);
        }
    }
    st.walk.next = Vec2::new(nx, nz);
}

/// The step taken: his velocity kept for the next tick, and his body set
/// down on the terrain where the step ended.
#[inline]
pub(super) fn integrate(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { terrain, pos_prev, kind, dt, .. } = *f;
    let (new_v, next) = (st.walk.new_v, st.walk.next);
    *s.vel = Vec3::new(new_v.x, 0.0, new_v.y);
    *s.pos = Vec3::new(
        next.x,
        terrain.height_at(next.x, next.y) + crate::unit_types::half_height(kind[s.i] as usize),
        next.y,
    );
    // The friction contact: his kept speed is the move he made, and what
    // the setting apart, the field's edge and the step off blocked ground
    // did to it beyond his legs' speed is added to the total his feet take
    // back. His legs' speed is the one `slide` left, after an edge he could
    // not pass stopped it, so a rail's stop is not a push his feet take back.
    if friction_contact() && dt > 0.0 {
        let p = pos_prev[s.i];
        let kept = (next - Vec2::new(p.x, p.z)) / dt;
        *s.vel = Vec3::new(kept.x, 0.0, kept.y);
        *s.imposed = held_total(*s.imposed, kept, new_v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lone man shoved at `shove_speed(d)` who wants no speed, stepped
    /// tick by tick with his feet and his legs, comes to rest d metres on.
    #[test]
    fn a_shoved_man_comes_to_rest_at_the_shove_distance() {
        let dt = 1.0 / 30.0;
        for d in [0.1125_f32, 0.225, 0.45, 0.675] {
            let u = shove_speed(d, dt);
            let mut v = Vec2::new(u, 0.0);
            let mut imposed = v.clamp_length_max(STEP_PACE);
            let mut x = Vec2::ZERO;
            for _ in 0..600 {
                let cut = feet_take(imposed, dt);
                v -= cut;
                imposed -= cut;
                let mut legs = -v * STEER_GAIN;
                if legs.length() > LEG_ACCEL {
                    legs *= LEG_ACCEL / legs.length();
                }
                v += legs * dt;
                x += v * dt;
            }
            assert!(v.length() < 1e-3, "{d}: still moving at {}", v.length());
            assert!((x.length() - d).abs() < 0.03, "{d}: at rest {} on at {u} m/s", x.length());
        }
    }

    /// A soldier with a side free keeps his full wish; surrounded in full
    /// he walks, the same way; half surrounded his wish lies half way; a
    /// wish under the walk is untouched.
    #[test]
    fn press_walk_goes_from_the_full_wish_to_the_walk() {
        let wish = Vec2::new(6.0, 8.0);
        assert_eq!(press_walk(0.0, wish, 1.5), wish);
        let cut = press_walk(1.0, wish, 1.5);
        assert!((cut.length() - 1.5).abs() < 1e-5);
        assert!((cut.normalize() - wish.normalize()).length() < 1e-6);
        let half = press_walk(0.5, wish, 1.5);
        assert!((half.length() - 5.75).abs() < 1e-5);
        let slow = Vec2::new(0.6, -0.8);
        assert_eq!(press_walk(1.0, slow, 1.5), slow);
    }

    /// The feet take FEET x dt along the total, or the whole of a smaller
    /// total, and nothing of none.
    #[test]
    fn feet_take_is_capped_along_the_total() {
        let dt = 1.0 / 30.0;
        let big = Vec2::new(0.6, -0.8);
        let cut = feet_take(big, dt);
        assert!((cut.length() - FEET * dt).abs() < 1e-6);
        assert!((cut.normalize() - big.normalize()).length() < 1e-6);
        let small = Vec2::new(0.05, 0.0);
        assert_eq!(feet_take(small, dt), small);
        assert_eq!(feet_take(Vec2::ZERO, dt), Vec2::ZERO);
    }

    /// Without the friction contact a man stands by his kept speed; with
    /// it a man pushed at 2 m/s who wishes to go nowhere stands, and a still
    /// man who wishes to walk at 3 m/s does not.
    #[test]
    fn stands_reads_the_wish_under_friction() {
        let fast = Vec2::new(2.0, 0.0);
        let walk = Vec2::new(0.0, 3.0);
        assert!(!stands(false, fast, Vec2::ZERO));
        assert!(stands(false, Vec2::ZERO, walk));
        assert!(stands(false, Vec2::new(0.5, 0.0), walk));
        assert!(stands(true, fast, Vec2::ZERO));
        assert!(!stands(true, Vec2::ZERO, walk));
    }

    /// A jammed soldier who waits at under a step's pace still stands while
    /// he tries a side: the sidestep at STEP_PACE across his wish takes the
    /// wish over the step's speed, and `steer` leaves it out.
    #[test]
    fn a_sidestep_does_not_make_him_a_walker() {
        let wish = Vec2::new(0.0, 0.9);
        let side = Vec2::new(STEP_PACE, 0.0);
        assert!(stands(true, Vec2::ZERO, wish));
        assert!(!stands(true, Vec2::ZERO, wish + side));
        assert!(stands(true, Vec2::ZERO, wish + side - side));
    }

    /// On the walk to his place: a walker whose wish has dropped
    /// under the speed of a step far from his place still walks; arriving
    /// within STEP_STOP he stands; a standing soldier shoved 5 cm stays
    /// standing; a wish over the line makes him a walker. A jammed soldier
    /// shoved back and forth 5 cm across the distance where his wish
    /// crosses the line changes once, not every time. A walker told to
    /// stand where he is (his wish zeroed, a comrade between him and his
    /// mark) stands though he is far from his place, and walks again once
    /// his wish reaches the line; a walker whose wish is merely small has
    /// not arrived.
    #[test]
    fn a_walker_stands_again_only_at_his_place() {
        // The drive's wish of a soldier of 6 m/s at `d` metres from his place.
        let wish = |d: f32| Vec2::new(0.0, 6.0 * (d / ARRIVE_RADIUS).min(1.0));
        let under = |d: f32| stands(true, Vec2::ZERO, wish(d));
        let arrived = |d: f32| has_arrived(d, false, false);
        assert!(has_arrived(0.3, false, false));
        assert!(has_arrived(5.0, true, false));
        assert!(!has_arrived(0.4, false, false));
        // Told to stand 5 m from his place: his wish is zero, he stands.
        let told = has_arrived(5.0, false, true);
        assert!(told && stands(true, Vec2::ZERO, Vec2::ZERO));
        assert!(still_stands(false, stands(true, Vec2::ZERO, Vec2::ZERO), told));
        // The way clears: his wish back over the line, he walks again.
        assert!(!still_stands(true, under(10.0), arrived(10.0)));
        // Merely slowed to a crawl 5 m out, not told to stand: he walks on.
        assert!(!still_stands(false, stands(true, Vec2::ZERO, Vec2::new(0.0, 0.1)), arrived(5.0)));
        assert!(under(5.0) && !arrived(5.0));
        assert!(!still_stands(false, under(5.0), arrived(5.0)));
        assert!(still_stands(false, under(0.3), arrived(0.3)));
        assert!(still_stands(true, under(5.05), arrived(5.05)));
        assert!(still_stands(true, under(4.95), arrived(4.95)));
        assert!(!under(10.0));
        assert!(!still_stands(true, under(10.0), arrived(10.0)));
        assert!(!still_stands(false, under(10.0), arrived(10.0)));
        // The line is at 35 x 1.05 / 6 = 6.125 m.
        let mut stood = true;
        let mut changes = 0;
        for k in 0..60 {
            let d = if k % 2 == 0 { 6.10 } else { 6.15 };
            let next = still_stands(stood, under(d), arrived(d));
            changes += (next != stood) as u32;
            stood = next;
        }
        assert_eq!(changes, 1);
        assert!(!stood);
    }

    /// The total gains the kept speed less the legs' speed and stops at a
    /// step's pace.
    #[test]
    fn held_total_adds_the_contact_and_caps_it() {
        let t = held_total(Vec2::new(0.1, 0.0), Vec2::new(1.0, 0.2), Vec2::new(1.0, 0.0));
        assert!((t - Vec2::new(0.1, 0.2)).length() < 1e-6);
        let t = held_total(Vec2::new(0.7, 0.0), Vec2::new(0.0, 0.0), Vec2::new(-1.0, 0.0));
        assert!((t.length() - STEP_PACE).abs() < 1e-6);
        assert!(t.x > 0.0 && t.y == 0.0);
    }

    /// Thrown clear, his feet take the speed he has no wish for while it
    /// is over a step's pace: all of it with no wish, the part against his
    /// wish with one, and nothing of a speed along his wish or under a
    /// step's pace; never more than the feet's take of one tick.
    #[test]
    fn catch_footing_takes_the_unwished_speed_over_a_step() {
        let dt = 1.0 / 30.0;
        let near = |a: Vec2, b: Vec2| (a - b).length() < 1e-6;
        let t = catch_footing(Vec2::new(-5.0, 0.0), Vec2::ZERO, 0.0, dt);
        assert!(near(t, Vec2::new(-FEET * dt, 0.0)));
        let t = catch_footing(Vec2::new(-5.0, 0.0), Vec2::ZERO, FEET * dt, dt);
        assert!(near(t, Vec2::ZERO));
        let t = catch_footing(Vec2::new(-(STEP_PACE + 0.1), 0.0), Vec2::ZERO, 0.0, dt);
        assert!(near(t, Vec2::new(-0.1, 0.0)));
        let t = catch_footing(Vec2::new(-0.5, 0.0), Vec2::ZERO, 0.0, dt);
        assert!(near(t, Vec2::ZERO));
        let t = catch_footing(Vec2::new(-5.0, 2.0), Vec2::new(3.0, 0.0), 0.0, dt);
        assert!(near(t, Vec2::new(-FEET * dt, 0.0)));
        let t = catch_footing(Vec2::new(5.0, 2.0), Vec2::new(3.0, 0.0), 0.0, dt);
        assert!(near(t, Vec2::ZERO));
    }

    /// A man pushed at a rail on z while he climbs along x: the rail
    /// stops the z of his step, and his velocity keeps only the climb.
    #[test]
    fn a_stopped_axis_loses_the_speed_into_the_edge() {
        let v = stopped_velocity(Vec2::new(1.2, 10.0), Vec2::new(0.04, 0.33), false, true);
        assert_eq!(v, Vec2::new(1.2, 0.0));
        let v = stopped_velocity(Vec2::new(-3.0, -2.0), Vec2::new(-0.1, -0.07), true, true);
        assert_eq!(v, Vec2::ZERO);
    }

    /// The setting apart pushed his step into the edge while his own
    /// velocity points away from it: that velocity stays.
    #[test]
    fn a_velocity_away_from_the_edge_stays() {
        let v = stopped_velocity(Vec2::new(1.2, -0.5), Vec2::new(0.04, 0.02), false, true);
        assert_eq!(v, Vec2::new(1.2, -0.5));
    }

    /// An axis no edge stopped keeps its speed whatever the step.
    #[test]
    fn a_free_axis_keeps_its_speed() {
        let v = stopped_velocity(Vec2::new(1.2, 10.0), Vec2::new(0.04, 0.33), false, false);
        assert_eq!(v, Vec2::new(1.2, 10.0));
    }
}
