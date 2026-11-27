//! The size and weight of a soldier's body: his width by pose and turn,
//! a pair's distances, the masses, the contact switch, the bracing and
//! the fall.

use crate::unit_types::TYPES;

pub(crate) const SEP_RADIUS: f32 = 1.4;
/// Below this distance overlap is resolved positionally, for a man in
/// his ready stance (at ease `body_ease`). A soldier is 0.75 to 1.1 m
/// across with his arms, shield and weapon. Never more than a pair's own
/// rest distance, so the men of a wall keep their 1.05 m files.
/// The contact between two soldiers reads it through `ready_body`.
/// FL_BODY=<metres> sets it, up to the separation distance.
const BODY_DISTANCE: f32 = 1.2;
pub(crate) fn body_distance() -> f32 {
    static D: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *D.get_or_init(|| crate::util::env_or("FL_BODY", BODY_DISTANCE).clamp(0.5, SEP_RADIUS))
}
/// The body of a soldier in his ready stance in the contact between two
/// soldiers (`widths`): under the friction contact (`friction_contact`)
/// it is the body at ease (BODY_EASE, as FL_BODY_EASE sets it), and
/// `body_distance` without it. Under the friction contact a soldier stops
/// pushing into still soldiers when they are at his width with arms and
/// shield (BRAKE_NEAR, 0.9 m). A ready body wider than that would stand
/// inside every neighbour at the fighting line, and every turn or death
/// there would throw the soldiers around it. The wall clearance and the
/// places on a walk keep reading `body_distance`.
#[inline]
pub(crate) fn ready_body() -> f32 {
    static D: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *D.get_or_init(|| {
        ready_body_of(friction_contact(), crate::util::env_or("FL_BODY_EASE", BODY_EASE), body_distance())
    })
}
/// `ready_body` from its inputs: under the friction contact the body at
/// ease, else the body distance.
fn ready_body_of(friction: bool, ease: f32, body: f32) -> f32 {
    (if friction { ease } else { body }).clamp(0.5, SEP_RADIUS)
}
/// Crowd-density yield: a man's drive fades to nothing as the crowd around
/// him (his neighbours' closeness summed, one each at a touch) rises
/// between the two values of `crowd_brake`. Scalar density can't cancel
/// out the way opposing push vectors do, so this is what stops a
/// goal-seeking crowd from compressing itself into overlap: packed
/// interior units genuinely stop shoving. CROWD_SLOW, six neighbours at
/// 1.12 m, is the crowd above which a pressing man with nobody in reach
/// stops looking for a far enemy (`acquire`).
pub(super) const CROWD_SLOW: f32 = 1.2;
/// The fade is set by the men's real size, not by the body distance: in
/// his ready stance a man's drive eases from six neighbours at arm's length, BRAKE_FAR of
/// the 1.70 m man, to nothing when six are pressed against him at
/// BRAKE_NEAR, his width with arms and shield. Both scale with the unit
/// scale. A band set from the body distance would be 11 cm of squeeze at
/// a 1.2 m body, and the drive of a whole lane would come back at once
/// whenever a gap opened at its front.
const BRAKE_FAR: f32 = 1.3;
const BRAKE_NEAR: f32 = 0.9;
/// The crowd brake of a man in his ready stance.
pub(crate) fn crowd_brake() -> (f32, f32) {
    let w = &widths()[3];
    (w.slow, w.stop)
}
/// The body distance at ease: a man standing, walking or running with
/// his arms down is narrower than one in his ready stance with weapon
/// and shield up (`ready_body`); under the friction contact the two are
/// the same. The spring rest at ease stays the same 0.2 m over the body
/// (`Width`): it is measured from BODY_DISTANCE, not from the body in the
/// ready stance. FL_BODY_EASE=<metres> sets it, up to the body in the
/// ready stance.
const BODY_EASE: f32 = 0.9;
pub(crate) fn body_ease() -> f32 {
    static D: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *D.get_or_init(|| crate::util::env_or("FL_BODY_EASE", BODY_EASE).clamp(0.5, ready_body()))
}
/// The spring rest at ease (metres).
pub(crate) fn sep_ease() -> f32 {
    body_ease() + SEP_RADIUS - BODY_DISTANCE
}
/// Ticks a man takes from at ease into his ready stance and back, the
/// change of his width: half a second, the length of the stance clip,
/// so a block that turns to fight widens as a step and not a jolt.
pub(crate) const POSE_READY: u8 = 15;
/// A regiment's stance (sim/job.rs `stance_of`), as the pose pass draws
/// its men: at ease; in the ready stance while a man stands (an enemy
/// near, an attack, a charge); in it whatever he does (fighting, a
/// wall).
pub(crate) const STANCE_EASE: u8 = 0;
pub(crate) const STANCE_READY: u8 = 1;
pub(crate) const STANCE_FIGHT: u8 = 2;
/// His pose as the grid carries it (spatial.rs META_POSE): 0 at ease to
/// 3 ready, the ramp in thirds. Both men of a pair read each other's
/// width from it and their own from the same rounding, so the pair's
/// distances are the same from either side. The pose byte carries the
/// stance's ramp in its low four bits and the turn of his shoulders in its
/// high four (`turn_level`).
#[inline]
pub(crate) fn pose_level(pose: u8) -> usize {
    RAMP_LEVELS[(pose & POSE_STANCE) as usize] as usize
}
/// A ramp of POSE_READY ticks in thirds. A table, not a division: the grid
/// rebuild reads it for every man every tick.
const RAMP_LEVELS: [u8; 16] = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3];
pub(super) const POSE_STANCE: u8 = 0x0f;
pub(super) const POSE_TURN_SHIFT: u8 = 4;
/// How far he has turned his shoulders to squeeze through a crack
/// (`try_a_side`): 0 square to 3 turned. He turns over the stance's half
/// second and turns back square over the same, so a man narrows as he
/// reaches a crack and widens as he comes out, and a block whose squeezes
/// end together does not jump wider in one tick.
pub(crate) fn turn_level(pose: u8) -> usize {
    RAMP_LEVELS[(pose >> POSE_TURN_SHIFT) as usize] as usize
}
/// His width level (`widths`): his pose level and his turn level together.
#[inline]
pub(crate) fn width_level(pose: u8) -> usize {
    pose_level(pose) + 4 * turn_level(pose)
}
/// A man's size at one pose level: his body distance and spring rest
/// (a pair's are the mean of the two men's), and the crowd at which his
/// drive starts to fade and is gone, from six men at his body plus
/// 0.1 m to six at his body less 0.2 m at ease, and BRAKE_FAR to
/// BRAKE_NEAR in his ready stance, each against his own spring rest.
pub(crate) struct Width {
    pub body: f32,
    pub sep: f32,
    pub slow: f32,
    pub stop: f32,
}
/// The width levels: his pose level (0 at ease to 3 ready) plus four times
/// his turn level (0 square to 3 turned). Turned, he narrows across his
/// way toward his depth, SQUEEZE_DEPTH, body and spring alike, and his
/// brake band toward a man's at ease, a third at each level.
pub(crate) const WIDTH_LEVELS: usize = 16;
pub(crate) const SQUEEZE_DEPTH: f32 = 0.5;
#[inline]
pub(crate) fn widths() -> &'static [Width; WIDTH_LEVELS] {
    static W: std::sync::OnceLock<[Width; WIDTH_LEVELS]> = std::sync::OnceLock::new();
    W.get_or_init(|| widths_of(body_ease(), ready_body(), crate::unit_types::unit_scale()))
}
/// The width levels from the body at ease, the body in the ready stance
/// and the unit scale (`widths`).
fn widths_of(body_ease: f32, ready_body: f32, scale: f32) -> [Width; WIDTH_LEVELS] {
    let sep_ease = body_ease + SEP_RADIUS - BODY_DISTANCE;
    let pose = |level: usize| {
        let t = level as f32 / 3.0;
        let mix = |ease: f32, ready: f32| ease * (1.0 - t) + ready * t;
        let body = mix(body_ease, ready_body);
        let sep = mix(sep_ease, SEP_RADIUS);
        let at = |d: f32| 6.0 * (1.0 - (d * scale).min(sep) / sep);
        Width {
            body,
            sep,
            slow: at(mix(body_ease + 0.1, BRAKE_FAR)),
            stop: at(mix(body_ease - 0.2, BRAKE_NEAR)),
        }
    };
    std::array::from_fn(|level| {
        let (w, turn) = (pose(level % 4), level / 4);
        if turn == 0 {
            return w;
        }
        let (t, ease) = (turn as f32 / 3.0, pose(0));
        let mix = |square: f32, turned: f32| square + (turned - square) * t;
        Width {
            body: mix(w.body, SQUEEZE_DEPTH),
            sep: mix(w.sep, SQUEEZE_DEPTH),
            slow: mix(w.slow, ease.slow),
            stop: mix(w.stop, ease.stop),
        }
    })
}
/// A pair of men by their width levels: their spring rest, its inverse,
/// and the distance under which their bodies overlap (`scan`). Looked
/// up per neighbour instead of averaged and divided per neighbour: the
/// scan is the kernel's cost and dividing would take a third of it. A
/// pair with a turned man in it rests nearer the touch, at it when he is
/// fully turned (by the more turned of the two): turned sideways and
/// pressing into a crack he has no air between his shoulder and the men
/// beside him, and with the other man's spring reaching past the touch a
/// crack narrowed by earlier passers held every squeezer short of it for
/// good.
pub(crate) struct PairWidth {
    pub sep: f32,
    pub inv_sep: f32,
    pub hard: f32,
}
#[inline]
pub(crate) fn pair_widths() -> &'static [[PairWidth; WIDTH_LEVELS]; WIDTH_LEVELS] {
    static T: std::sync::OnceLock<[[PairWidth; WIDTH_LEVELS]; WIDTH_LEVELS]> = std::sync::OnceLock::new();
    T.get_or_init(|| std::array::from_fn(|a| std::array::from_fn(|b| pair_width_of(a, 1.0, b, 1.0))))
}
/// A pair of men by their width levels and the share of his body each
/// still has (`fall_share`, 1 for a living soldier): each man's body and
/// spring rest times his share, then the pair's as in the table. A share
/// of 1 is the table's entry to the bit. The same from either side.
#[inline]
pub(crate) fn pair_width_of(a: usize, share_a: f32, b: usize, share_b: f32) -> PairWidth {
    pair_of(widths(), a, share_a, b, share_b)
}
/// `pair_width_of` on a given set of width levels.
#[inline]
fn pair_of(w: &[Width; WIDTH_LEVELS], a: usize, share_a: f32, b: usize, share_b: f32) -> PairWidth {
    let mut sep = 0.5 * (w[a].sep * share_a + w[b].sep * share_b);
    let hard = (0.5 * (w[a].body * share_a + w[b].body * share_b)).min(sep);
    sep += (hard - sep) * ((a / 4).max(b / 4) as f32 / 3.0);
    PairWidth { sep, inv_sep: 1.0 / sep, hard }
}
/// Under the friction contact, a soldier who is struck down falls over his
/// death clock instead of in single ticks. While he dies he keeps the
/// weight he had when he was struck: his stance, the turn of his shoulders
/// and his stood bit stay as they were (`scan`, `try_a_side`, `steer`), and
/// he is braced in the setting apart as a living soldier in that state
/// (`brace_level`). And his body goes down: its width and spring rest
/// shrink with the share of his death clock that is left, whole on his
/// first tick down and nothing on his last (`fall_share`), so the men
/// around him close over the place he fell a little each tick, and nothing
/// overlaps him when he is swept. Without the friction contact a dying
/// soldier weighs nothing from the tick he is struck and his whole body
/// stays until it is swept.
#[inline]
pub(crate) fn fall() -> bool {
    friction_contact()
}
/// The share of his body a soldier still has, from his death clock as the
/// grid carries it (spatial.rs META_FALL): 1 alive and on his first tick
/// down (DEATH_TICKS), falling a step a tick to nothing on his last tick
/// in the grid (2: on the tick his clock reaches 1 he is swept). Both
/// soldiers of a pair read it from the same clock, so they read the same.
#[inline]
pub(crate) fn fall_share(death_t: u8) -> f32 {
    const LAST: u8 = 2;
    if death_t == 0 {
        return 1.0;
    }
    (death_t.saturating_sub(LAST) as f32 / (crate::sim::damage::DEATH_TICKS - LAST) as f32).min(1.0)
}
/// A pair of men by their kinds: the mass weight of the push on the
/// first (2 m_o / (m_i + m_o)) and the other's share of a closing speed
/// (m_o / (m_i + m_o)).
#[inline]
pub(crate) fn mass_pairs() -> &'static [[(f32, f32); crate::unit_types::NUM_KINDS]; crate::unit_types::NUM_KINDS] {
    static T: std::sync::OnceLock<[[(f32, f32); crate::unit_types::NUM_KINDS]; crate::unit_types::NUM_KINDS]> =
        std::sync::OnceLock::new();
    T.get_or_init(|| {
        std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                let (mi, mo) = (TYPES[a].mass, TYPES[b].mass);
                (2.0 * mo / (mi + mo), mo / (mi + mo))
            })
        })
    })
}
/// The friction contact, on by default; FL_CONTACT=share turns it off for
/// the shared contact, where a man and the body he runs into share the
/// closing speed by weight (half each between equal men), so a shove
/// travels through a press and a deep mass pushes a line. Under the
/// friction contact there is no spring between men, no grip, no
/// stance and no contact speed. The setting apart moves him, his kept
/// speed is where he ended up, and what the contacts put into that speed
/// his feet take back at FEET (`steer`, `integrate`).
/// The one thing that stands in for a stance is the bracing: in the
/// setting apart a man braced against a push (`brace_pairs`) holds his
/// ground, and the man who walks into him takes most of it.
#[inline]
pub(crate) fn friction_contact() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| !std::env::var("FL_CONTACT").is_ok_and(|v| v == "share"))
}
/// Under the friction contact, how many times his weight a man braced
/// against a push counts for when two bodies are set apart: his feet are
/// planted and his weight is behind his shield, so the man who walks into
/// him is the one who gives. He is braced as far as he is in his ready
/// stance (`pose_level`, a third of the way at each level) while he stands
/// (`steer`, the `stood` column), whatever his regiment's order. He
/// becomes a walker when his own wish to move reaches the speed of a step.
/// On the walk to his place with nothing else to do (`on_his_way`), in a
/// fighting regiment's rear ranks too, he stands again only once he has
/// arrived there, within STEP_STOP, or his own rule tells him to stand
/// where he is (`has_arrived`, `still_stands`); a soldier under fight
/// footwork stands again as soon as his wish drops under that speed. So a
/// jammed soldier on his way, whose wish slides with the few centimetres a
/// shove moves him, does not change his weight back and forth. The
/// sidestep of a jammed soldier's try at a side (`try_a_side`) is not part
/// of that wish: trying a side, a standing soldier still stands. This is
/// the standing bracing, the one that meets a charge, and it counts
/// against any body. A walking soldier does not have it (he may have the
/// walking bracing against the other side, `BRACE_BASIC`), and neither
/// does a dying man unless he falls over his death clock (`fall`),
/// keeping what he had. A push does not take him out of it, only his own
/// walking does. This is a man bracing his body, not a spearwall's braced
/// points. FL_BRACE=<number> sets it, from 1 (his own weight, as without
/// it) to 20.
const BRACE: f32 = 4.0;
fn brace() -> f32 {
    static B: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *B.get_or_init(|| crate::util::env_or("FL_BRACE", BRACE).clamp(1.0, 20.0))
}
/// Under the friction contact, the walking bracing: how many times his
/// weight a soldier in his ready stance counts for against a body of the
/// other side while he walks or runs. He keeps his weight behind his
/// shield as he goes, so an enemy who walks into him does not pass through
/// him as through a walker. Against his own side a walking soldier weighs
/// as a walker. It is by his pose level as `BRACE` is, and a soldier whose
/// regiment is under a Move order (sim/job.rs `moving`) or a dying soldier
/// does not have it (a falling one keeps it, `fall`). A soldier who stands
/// has the standing bracing (`BRACE`) against every body instead. Both
/// sides have it alike: it reads only whether the two bodies are of one
/// side.
const BRACE_BASIC: f32 = 4.0;
/// His brace level in a pair: 0 not braced (weight 1); 1 to 3 standing,
/// braced by his pose level (`BRACE`); 4 to 6 walking, braced against the
/// other side by his pose level, 4 + pose level - 1 (`BRACE_BASIC`).
const BRACE_LEVELS: usize = 7;
/// Where the walking bracing's levels start (`BRACE_LEVELS`).
const BRACE_BASIC_FROM: usize = 4;
/// How many times his mass a man at brace level `b` counts for: standing,
/// 1 + (brace - 1) x pose level / 3; walking, 1 + (basic - 1) x pose
/// level / 3.
fn brace_mult(b: usize, brace: f32, basic: f32) -> f32 {
    if b < BRACE_BASIC_FROM {
        1.0 + (brace - 1.0) * b as f32 / 3.0
    } else {
        1.0 + (basic - 1.0) * (b - BRACE_BASIC_FROM + 1) as f32 / 3.0
    }
}
/// The weight of the setting apart on a man of mass `mi` at brace level
/// `bi` from one of mass `mo` at level `bo`, with a fully braced man
/// counting `brace` (standing) or `basic` (walking) times his mass:
/// 2 m_o / (m_i + m_o) over the braced masses. Two men braced alike share
/// it by their own masses, exactly as `mass_pairs`, so the weights of a
/// pair sum to 2 whatever their levels.
fn brace_weight(mi: f32, mo: f32, bi: usize, bo: usize, brace: f32, basic: f32) -> f32 {
    if bi == bo {
        return 2.0 * mo / (mi + mo);
    }
    let (mi, mo) = (mi * brace_mult(bi, brace, basic), mo * brace_mult(bo, brace, basic));
    2.0 * mo / (mi + mo)
}
/// 4 kinds x 4 kinds x 7 levels x 7 levels of f32: 3,136 bytes, read by
/// one row (his kind, his level) per soldier.
pub(super) type BraceTable = [BraceRow; crate::unit_types::NUM_KINDS];
/// One kind's row of the table: by the other's kind, his level and the
/// other's.
pub(super) type BraceRow = [[[f32; BRACE_LEVELS]; BRACE_LEVELS]; crate::unit_types::NUM_KINDS];
fn brace_table(brace: f32, basic: f32) -> BraceTable {
    std::array::from_fn(|a| {
        std::array::from_fn(|o| {
            std::array::from_fn(|bi| {
                std::array::from_fn(|bo| brace_weight(TYPES[a].mass, TYPES[o].mass, bi, bo, brace, basic))
            })
        })
    })
}
/// The friction contact's weights of the setting apart, by his kind, the
/// other's kind, his brace level and the other's: a table read per
/// overlapping neighbour instead of a division (`pair_widths`).
#[inline]
pub(super) fn brace_pairs() -> &'static BraceTable {
    static T: std::sync::OnceLock<BraceTable> = std::sync::OnceLock::new();
    T.get_or_init(|| brace_table(brace(), BRACE_BASIC))
}
/// A soldier's brace level in a pair under the friction contact, from what
/// both soldiers of the pair read alike in the grid: standing (`stood`),
/// his pose level against anyone; walking, against the other side
/// (`cross`) and not under a Move order, the walking bracing's level for
/// his pose; else 0. A dying soldier is 0, unless he falls over his death
/// clock (`fall`): then he is braced as a living soldier in the same
/// state. `moving` is called only when the walking bracing could count,
/// so a neighbour's regiment is read only then.
#[inline]
pub(super) fn brace_level(
    pose_lv: usize,
    stood: bool,
    dying: bool,
    fall: bool,
    cross: bool,
    moving: impl FnOnce() -> bool,
) -> usize {
    if dying && !fall {
        0
    } else if stood {
        pose_lv
    } else if cross && pose_lv > 0 && !moving() {
        BRACE_BASIC_FROM + pose_lv - 1
    } else {
        0
    }
}
/// Under the friction contact the crowd's brake reads the still men ahead
/// of his wish (`crowd_still`), against a band half as wide. FL_BRAKE=density
/// keeps the brake on every neighbour's closeness, as without it.
#[inline]
pub(super) fn still_brake() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| friction_contact() && !std::env::var("FL_BRAKE").is_ok_and(|v| v == "density"))
}
/// A soldier walking to the fight stops by sight an arm's length
/// (SEP_RADIUS) behind a comrade in his way, not a body and a short step
/// behind him: he stands once his look sees that comrade, whether or not
/// the bodies have stopped him yet, so he never walks into the back ahead.
/// Standing there he takes his elbow room with his own legs (`take_room`).
/// FL_ROOM=0 turns it off; only with the still brake.
#[inline]
pub(super) fn elbow_room() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| still_brake() && crate::util::env_or("FL_ROOM", 1u32) != 0)
}
/// A soldier walking to the fight who stands behind comrades steps into
/// an opening among the bodies just ahead of him that fits him
/// (`look_around`, `opening_ahead`, `step_into_opening`). FL_JOIN_OPEN=0
/// turns it off; only with the still brake.
#[inline]
pub(super) fn join_open() -> bool {
    static J: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *J.get_or_init(|| still_brake() && crate::util::env_or("FL_JOIN_OPEN", 1u32) != 0)
}
/// A soldier in his rank walking to his place who is stopped behind a
/// comrade that has left his rank and stands there leaves his rank too and
/// walks on to the fight (`look_around`, `decide_to_join`, `join_way`).
/// FL_JOIN_TOUCH=0 turns it off; only with the still brake.
#[inline]
pub(super) fn join_touch() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| still_brake() && crate::util::env_or("FL_JOIN_TOUCH", 1u32) != 0)
}
/// The opening read counts open ground beside the nearest rank of bodies
/// ahead as room too, not only the air between two of them
/// (`opening_ahead`). FL_OPEN_SIDE=0 turns it off.
#[inline]
pub(super) fn open_side() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| crate::util::env_or("FL_OPEN_SIDE", 1u32) != 0)
}
/// An opening fits him only with his elbow room beside his body, and a
/// body nearer than that to his aim has taken it (`opening_ahead`,
/// `aim_taken`). FL_OPEN_ROOMY=0 turns it off.
#[inline]
pub(super) fn open_roomy() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| crate::util::env_or("FL_OPEN_ROOMY", 1u32) != 0)
}
/// A held regiment's soldiers leave their ranks for its fight too and walk
/// straight up to its fighting line (`begin`, `look_around`); they close
/// on an enemy only within 2.2 m, as before (`join_way`). FL_HOLD_JOIN=0
/// turns it off.
#[inline]
pub(super) fn hold_join() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| crate::util::env_or("FL_HOLD_JOIN", 1u32) != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The (standing, walking) strengths the table tests run over.
    const BRACE_CASES: [(f32, f32); 6] =
        [(1.0, 1.0), (BRACE, 1.0), (BRACE, BRACE_BASIC), (BRACE, 2.0), (20.0, 20.0), (1.0, 20.0)];

    /// Two men braced alike, or any two with both strengths 1, share the
    /// setting apart exactly as the plain mass weights do, to the bit.
    #[test]
    fn equal_bracing_is_the_plain_mass_weight() {
        let plain = mass_pairs();
        for (brace, basic) in BRACE_CASES {
            let t = brace_table(brace, basic);
            for (ta, pa) in t.iter().zip(plain) {
                for (tao, pao) in ta.iter().zip(pa) {
                    for (b, row) in tao.iter().enumerate() {
                        assert_eq!(row[b].to_bits(), pao.0.to_bits());
                    }
                }
            }
        }
        let t = brace_table(1.0, 1.0);
        for (ta, pa) in t.iter().zip(plain) {
            for (tao, pao) in ta.iter().zip(pa) {
                for w in tao.iter().flatten() {
                    assert_eq!(w.to_bits(), pao.0.to_bits());
                }
            }
        }
    }

    /// The grid's stood bit sits above the turn of his shoulders, the fall
    /// field between it and his regiment, and no field reads another; the
    /// regiment keeps 17 bits.
    #[test]
    fn the_stood_bit_and_the_fall_field_leave_the_width_and_the_regiment_alone() {
        use crate::spatial::{
            META_FALL_SHIFT, META_GROUP_SHIFT, META_POSE_SHIFT, META_STOOD, META_TURN_SHIFT, meta_fall, meta_group,
            meta_width,
        };
        assert_eq!(META_STOOD, 1 << 9);
        assert_eq!(META_FALL_SHIFT, 10);
        assert_eq!(META_GROUP_SHIFT, 15);
        let g = (1u32 << (32 - META_GROUP_SHIFT)) - 1;
        for stood in [false, true] {
            for d in [0, 2, 9, crate::sim::damage::DEATH_TICKS as u32, 31] {
                let meta = (3 << META_POSE_SHIFT)
                    | (2 << META_TURN_SHIFT)
                    | (stood as u32 * META_STOOD)
                    | (d << META_FALL_SHIFT)
                    | (g << META_GROUP_SHIFT);
                assert_eq!(meta_width(meta), 3 + 4 * 2);
                assert_eq!(meta_group(meta), g as usize);
                assert_eq!((meta & META_STOOD) != 0, stood);
                assert_eq!(meta_fall(meta), d as u8);
            }
        }
    }

    /// A living soldier has his whole body; a falling one has it whole on his
    /// first tick down, half of it half way through his clock, and nothing
    /// on his last tick in the grid, a sixteenth less each tick.
    #[test]
    fn the_fall_takes_his_body_down_over_his_death_clock() {
        let first = crate::sim::damage::DEATH_TICKS;
        assert_eq!(fall_share(0), 1.0);
        assert_eq!(fall_share(first), 1.0);
        assert_eq!(fall_share((first + 2) / 2), 0.5);
        assert_eq!(fall_share(2), 0.0);
        for d in 3..=first {
            let step = fall_share(d) - fall_share(d - 1);
            assert!((step - 1.0 / (first - 2) as f32).abs() < 1e-6, "{d}");
        }
    }

    /// Under the friction contact the body in the ready stance is the body
    /// at ease, at every pose level; without it, it is the body distance.
    #[test]
    fn the_ready_body_under_friction_is_the_body_at_ease() {
        let ready = ready_body_of(true, BODY_EASE, BODY_DISTANCE);
        assert_eq!(ready, BODY_EASE);
        let w = widths_of(BODY_EASE, ready, 1.0);
        for (level, w) in w[..4].iter().enumerate() {
            assert_eq!(w.body.to_bits(), BODY_EASE.to_bits(), "{level}");
        }
        assert_eq!(ready_body_of(false, BODY_EASE, BODY_DISTANCE), BODY_DISTANCE);
    }

    /// A pair with a falling soldier in it is as far apart on his first tick
    /// down as two living soldiers, to the bit, and closes to the living
    /// soldier's half alone on his last; the pair reads the same from either
    /// side, and at every share between it lies between the two.
    #[test]
    #[allow(clippy::needless_range_loop)] // the indices are kinds and levels of more than one table
    fn a_falling_body_closes_to_the_living_half() {
        let w = widths();
        let table = pair_widths();
        let first = crate::sim::damage::DEATH_TICKS;
        for a in 0..WIDTH_LEVELS {
            for b in 0..WIDTH_LEVELS {
                let whole = pair_width_of(a, 1.0, b, fall_share(first));
                assert_eq!(whole.sep.to_bits(), table[a][b].sep.to_bits());
                assert_eq!(whole.inv_sep.to_bits(), table[a][b].inv_sep.to_bits());
                assert_eq!(whole.hard.to_bits(), table[a][b].hard.to_bits());
                let gone = pair_width_of(a, 1.0, b, fall_share(2));
                let hard = (0.5 * w[a].body).min(0.5 * w[a].sep);
                let sep = 0.5 * w[a].sep + (hard - 0.5 * w[a].sep) * ((a / 4).max(b / 4) as f32 / 3.0);
                assert!((gone.hard - hard).abs() < 1e-6, "{a} {b}");
                assert!((gone.sep - sep).abs() < 1e-6, "{a} {b}");
                for d in 2..=first {
                    let one = pair_width_of(a, 1.0, b, fall_share(d));
                    let other = pair_width_of(b, fall_share(d), a, 1.0);
                    assert_eq!(one.sep.to_bits(), other.sep.to_bits());
                    assert_eq!(one.hard.to_bits(), other.hard.to_bits());
                    assert!(one.hard <= table[a][b].hard && one.hard >= gone.hard - 1e-6);
                }
            }
        }
    }

    /// Falling, a dying soldier is braced as a living one with the same
    /// pose, stood bit and side; without the fall he is not braced at all.
    #[test]
    fn a_falling_soldier_keeps_his_brace() {
        for p in 0..4 {
            for stood in [false, true] {
                for cross in [false, true] {
                    for moving in [false, true] {
                        let living = brace_level(p, stood, false, false, cross, || moving);
                        let falling = brace_level(p, stood, true, true, cross, || moving);
                        let fallen = brace_level(p, stood, true, false, cross, || moving);
                        assert_eq!(falling, living, "{p} {stood} {cross} {moving}");
                        assert_eq!(fallen, 0);
                        assert_eq!(brace_level(p, stood, false, true, cross, || moving), living);
                    }
                }
            }
        }
    }

    /// A fully braced man against an unbraced one of his own kind takes
    /// 2 / (1 + BRACE) of the setting apart and the other 2 BRACE / (1 + BRACE);
    /// the same with BRACE_BASIC for a fully braced walking man.
    #[test]
    fn a_braced_man_takes_little_of_the_setting_apart() {
        let t = brace_table(BRACE, BRACE_BASIC);
        let full_walk = BRACE_BASIC_FROM + 2;
        for (k, tk) in t.iter().enumerate() {
            let own = &tk[k];
            assert!((own[3][0] - 2.0 / (1.0 + BRACE)).abs() < 1e-6);
            assert!((own[0][3] - 2.0 * BRACE / (1.0 + BRACE)).abs() < 1e-6);
            assert!((own[full_walk][0] - 2.0 / (1.0 + BRACE_BASIC)).abs() < 1e-6);
            assert!((own[0][full_walk] - 2.0 * BRACE_BASIC / (1.0 + BRACE_BASIC)).abs() < 1e-6);
        }
    }

    /// For any pair, whatever their kinds and levels: the two weights sum
    /// to 2, as the plain weights do, so the pair is set apart by the same
    /// distance in all; and each man's braced mass times his weight is the
    /// other's, as m_i w_i = m_o w_o holds for the plain weights.
    #[test]
    fn a_pair_is_set_apart_by_the_same_distance_in_all() {
        for (brace, basic) in BRACE_CASES {
            let t = brace_table(brace, basic);
            let braced = |k: usize, b: usize| TYPES[k].mass * brace_mult(b, brace, basic);
            for (a, ta) in t.iter().enumerate() {
                for (o, tao) in ta.iter().enumerate() {
                    for (bi, row) in tao.iter().enumerate() {
                        for (bo, &wi) in row.iter().enumerate() {
                            let wo = t[o][a][bo][bi];
                            assert!((wi + wo - 2.0).abs() < 1e-5, "{a} {o} {bi} {bo}: {wi} + {wo}");
                            let (mi, mo) = (braced(a, bi), braced(o, bo));
                            assert!((mi * wi - mo * wo).abs() < 1e-4 * mi.max(mo), "{a} {o} {bi} {bo}");
                        }
                    }
                }
            }
        }
    }

    /// The standing levels are the one-strength table, to the bit; and with
    /// the walking bracing at 1 its levels weigh as level 0 against anyone,
    /// to the bit.
    #[test]
    #[allow(clippy::needless_range_loop)] // the indices are kinds and levels of more than one table
    fn the_standing_levels_are_one_strength_and_a_walking_bracing_of_1_is_none() {
        for brace in [1.0, BRACE, 20.0] {
            let one = |mi: f32, mo: f32, bi: usize, bo: usize| {
                if bi == bo {
                    return 2.0 * mo / (mi + mo);
                }
                let braced = |m: f32, b: usize| m * (1.0 + (brace - 1.0) * b as f32 / 3.0);
                let (mi, mo) = (braced(mi, bi), braced(mo, bo));
                2.0 * mo / (mi + mo)
            };
            for basic in [1.0, BRACE_BASIC, 20.0] {
                let t = brace_table(brace, basic);
                for a in 0..crate::unit_types::NUM_KINDS {
                    for o in 0..crate::unit_types::NUM_KINDS {
                        for bi in 0..BRACE_BASIC_FROM {
                            for bo in 0..BRACE_BASIC_FROM {
                                let w = one(TYPES[a].mass, TYPES[o].mass, bi, bo);
                                assert_eq!(t[a][o][bi][bo].to_bits(), w.to_bits(), "{a} {o} {bi} {bo}");
                            }
                        }
                    }
                }
            }
            let t = brace_table(brace, 1.0);
            for a in 0..crate::unit_types::NUM_KINDS {
                for o in 0..crate::unit_types::NUM_KINDS {
                    for p in 0..3 {
                        for bo in 0..BRACE_LEVELS {
                            let walk = t[a][o][BRACE_BASIC_FROM + p][bo];
                            let bo0 = if bo >= BRACE_BASIC_FROM { 0 } else { bo };
                            assert_eq!(walk.to_bits(), t[a][o][0][bo0].to_bits(), "{a} {o} {p} {bo}");
                        }
                    }
                }
            }
        }
    }

    /// Standing, a soldier has his pose level against anyone; walking, the
    /// walking bracing's level for his pose only against the other side,
    /// only in a ready pose, not under a Move order; dying, nothing. With
    /// both bracings at the same strength a fully braced walking soldier
    /// weighs as a standing one.
    #[test]
    #[allow(clippy::needless_range_loop)] // the indices are kinds and levels of more than one table
    fn the_two_bracings_by_side_and_standing() {
        for p in 0..4 {
            for moving in [false, true] {
                for cross in [false, true] {
                    assert_eq!(brace_level(p, true, false, false, cross, || moving), p);
                    let walk = if cross && !moving && p > 0 { BRACE_BASIC_FROM + p - 1 } else { 0 };
                    assert_eq!(brace_level(p, false, false, false, cross, || moving), walk, "{p} {moving} {cross}");
                    for stood in [false, true] {
                        assert_eq!(brace_level(p, stood, true, false, cross, || moving), 0);
                    }
                }
            }
        }
        // `moving` is not read for a standing soldier or a friend.
        assert_eq!(brace_level(2, true, false, false, true, || unreachable!()), 2);
        assert_eq!(brace_level(2, false, false, false, false, || unreachable!()), 0);
        let t = brace_table(BRACE, BRACE);
        for a in 0..crate::unit_types::NUM_KINDS {
            for o in 0..crate::unit_types::NUM_KINDS {
                assert!((t[a][o][BRACE_BASIC_FROM + 2][0] - t[a][o][3][0]).abs() < 1e-6);
                assert!((t[a][o][BRACE_BASIC_FROM + 2][3] - t[a][o][3][3]).abs() < 1e-6);
            }
        }
    }
}
