//! One soldier's tick, stage by stage. `tick_chunk` runs every soldier
//! of a chunk through the stages below in order; `Step` carries what one
//! stage hands the next. The rules live in the stage comments.

use bevy::math::Vec3Swizzles;
use bevy::prelude::*;

use crate::sim::damage::DamageEvent;
use crate::spatial::SpatialGrid;
use crate::terrain::Terrain;
use crate::unit_types::TYPES;

pub(crate) const SEP_RADIUS: f32 = 1.4;
const SEP_STRENGTH: f32 = 60.0;
/// Base neighbor query radius: covers separation and sword reach. Kinds
/// with longer reach (spears) widen their own scan per unit — the sword
/// majority must not pay for the spear's reach.
const QUERY_RADIUS: f32 = 2.0;
/// Cap on summed separation push to avoid explosive forces deep in a crowd.
const SEP_PUSH_MAX: f32 = 2.5;
/// Below this distance overlap is resolved positionally, for a man in
/// his ready stance (at ease `body_ease`). A soldier is 0.75 to 1.1 m
/// across with his arms, shield and weapon. Never more than a pair's own
/// rest distance, so the men of a wall keep their 1.05 m files.
/// FL_BODY=<metres> sets it, up to the separation distance.
const BODY_DISTANCE: f32 = 1.2;
pub(crate) fn body_distance() -> f32 {
    static D: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *D.get_or_init(|| crate::util::env_or("FL_BODY", BODY_DISTANCE).clamp(0.5, SEP_RADIUS))
}
/// Crowd-density yield: a man's drive fades to nothing as the crowd around
/// him (his neighbours' closeness summed, one each at a touch) rises
/// between the two values of `crowd_brake`. Scalar density can't cancel
/// out the way opposing push vectors do, so this is what stops a
/// goal-seeking crowd from compressing itself into overlap: packed
/// interior units genuinely stop shoving. CROWD_SLOW, six neighbours at
/// 1.12 m, is the crowd above which a pressing man with nobody in reach
/// stops looking for a far enemy (`acquire`).
const CROWD_SLOW: f32 = 1.2;
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
/// and shield up (BODY_DISTANCE). The spring rest at ease stays the same
/// 0.2 m over the body (`Width`). FL_BODY_EASE=<metres> sets it, up to
/// the body in the ready stance.
const BODY_EASE: f32 = 0.9;
pub(crate) fn body_ease() -> f32 {
    static D: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *D.get_or_init(|| crate::util::env_or("FL_BODY_EASE", BODY_EASE).clamp(0.5, body_distance()))
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
pub(crate) fn pose_level(pose: u8) -> usize {
    RAMP_LEVELS[(pose & POSE_STANCE) as usize] as usize
}
/// A ramp of POSE_READY ticks in thirds. A table, not a division: the grid
/// rebuild reads it for every man every tick.
const RAMP_LEVELS: [u8; 16] = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3];
const POSE_STANCE: u8 = 0x0f;
const POSE_TURN_SHIFT: u8 = 4;
/// How far he has turned his shoulders to squeeze through a crack
/// (`try_a_side`): 0 square to 3 turned. He turns over the stance's half
/// second and turns back square over the same, so a man narrows as he
/// reaches a crack and widens as he comes out, and a block whose squeezes
/// end together does not jump wider in one tick.
pub(crate) fn turn_level(pose: u8) -> usize {
    RAMP_LEVELS[(pose >> POSE_TURN_SHIFT) as usize] as usize
}
/// His width level (`widths`): his pose level and his turn level together.
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
pub(crate) fn widths() -> &'static [Width; WIDTH_LEVELS] {
    static W: std::sync::OnceLock<[Width; WIDTH_LEVELS]> = std::sync::OnceLock::new();
    W.get_or_init(|| {
        let scale = crate::unit_types::unit_scale();
        let pose = |level: usize| {
            let t = level as f32 / 3.0;
            let mix = |ease: f32, ready: f32| ease * (1.0 - t) + ready * t;
            let body = mix(body_ease(), body_distance());
            let sep = mix(sep_ease(), SEP_RADIUS);
            let at = |d: f32| 6.0 * (1.0 - (d * scale).min(sep) / sep);
            Width {
                body,
                sep,
                slow: at(mix(body_ease() + 0.1, BRAKE_FAR)),
                stop: at(mix(body_ease() - 0.2, BRAKE_NEAR)),
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
    })
}
/// A pair of men by their width levels: their spring rest, its inverse,
/// and the distance under which their bodies overlap (`scan`). Looked
/// up per neighbour instead of averaged and divided per neighbour: the
/// scan is the kernel's cost and the divisions were a third of it. A
/// pair with a turned man in it rests nearer the touch, at it when he is
/// fully turned (by the more turned of the two): turned sideways and
/// pressing into a crack he has no air between his shoulder and the men
/// beside him, and with the other man's spring reaching past the touch a
/// crack narrowed by earlier passers held every squeezer short of it for
/// good (toy devlog 011).
pub(crate) struct PairWidth {
    pub sep: f32,
    pub inv_sep: f32,
    pub hard: f32,
}
pub(crate) fn pair_widths() -> &'static [[PairWidth; WIDTH_LEVELS]; WIDTH_LEVELS] {
    static T: std::sync::OnceLock<[[PairWidth; WIDTH_LEVELS]; WIDTH_LEVELS]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let w = widths();
        std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                let mut sep = 0.5 * (w[a].sep + w[b].sep);
                let hard = (0.5 * (w[a].body + w[b].body)).min(sep);
                sep += (hard - sep) * ((a / 4).max(b / 4) as f32 / 3.0);
                PairWidth { sep, inv_sep: 1.0 / sep, hard }
            })
        })
    })
}
/// A pair of men by their kinds: the mass weight of the push on the
/// first (2 m_o / (m_i + m_o)) and the other's share of a closing speed
/// (m_o / (m_i + m_o)).
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
/// The contact between a man and the body he runs into. By default the
/// two share the closing speed by weight (half each between equal men),
/// so a shove travels through a press and a deep mass pushes a line.
/// FL_CONTACT=wall plays the soft wall instead: the runner's legs lose
/// the closing speed over his give and the man hit is handed none of it.
fn share_contact() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| !std::env::var("FL_CONTACT").is_ok_and(|v| v == "wall"))
}
/// The contact between men: yield and collide by default. A man's legs
/// alone set his velocity (no spring, no grip, no stance, no shared
/// speed, no density brake); his feet take back what his legs do not
/// want; the bodies are set apart by position after the step
/// (`set_apart`), and what the setting apart could not take away of a
/// closing is a collision (`collide`). FL_CONTACT_OLD=1 plays the old
/// contact: the spring, the grip, the stance, the shared speed or the
/// soft wall, the brake and the CORR_GAIN set-apart.
pub(crate) fn contact_old() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("FL_CONTACT_OLD").is_ok_and(|v| v == "1"))
}
/// His feet (m/s^2): each tick they take back up to FEET x dt of the
/// velocity his legs do not want, the part across his drive and the
/// excess along it, with no memory. A thrown man is stopped by his feet;
/// a man walking as he intends feels nothing.
const FEET: f32 = 7.0;
/// The setting apart of the yield-and-collide contact: SET_APART_PASSES
/// passes over the tick's touching pairs, each removing every overlap
/// whole, each man of a pair moving the other man's mass share of it,
/// his corrections averaged over the men he touches and multiplied by
/// SET_APART_WEIGHT. One summed pass overshoots in a chain; one averaged
/// pass crawls; eight at 1.5 carry a crush about eight ranks a tick.
/// FL_PASSES=<n> sets the passes, 1 to 16; two by default, the toy's
/// hard2 (round 50), with the still-man brake keeping the press from
/// crushing.
const SET_APART_PASSES: usize = 2;
pub(crate) fn set_apart_passes() -> usize {
    static P: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *P.get_or_init(|| crate::util::env_or("FL_PASSES", SET_APART_PASSES).clamp(1, 16))
}
/// The still-man brake of the yield-and-collide contact (`scan`,
/// `yield_to_crowd`): his drive fades for the crowd of standing men ahead
/// of his wish, by the old brake's band, and never under the step pace.
/// FL_BRAKE=0 turns it off (no brake, as before it).
pub(crate) fn still_brake() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| !std::env::var("FL_BRAKE").is_ok_and(|v| v == "0"))
}
/// The still-man crowd's crawl (m/s): a neighbour ahead who actually
/// moved along a man's way over the last tick at CRAWL or faster counts
/// nothing in it; one who stood, or came at him, counts in full; between,
/// by the share of the crawl he did not make.
const CRAWL: f32 = 0.3;
const SET_APART_WEIGHT: f32 = 1.5;
/// A neighbour is on his list of the tick's touching pairs when the two
/// would stand within the pair's body distance plus TOUCH_SLACK after
/// each steps a tick at his velocity of the tick's start: the slack is
/// for the legs' change of speed and the passes' moves (metres). Men
/// walking side by side farther apart than that list nothing.
const TOUCH_SLACK: f32 = 0.10;
/// The scan's cut of its candidates (`scan`): metres past the farthest a
/// neighbour can be and still count.
const CUT_MARGIN: f32 = 0.01;
/// The speed (m/s) the scan's cut is drawn for (`scan`): over a man's
/// run at 1.15 times the fastest kind's speed.
const CUT_FAST: f32 = 12.0;
/// The widest pair's body distance and spring rest over the pair table
/// (`pair_widths`), for the scan's cut.
fn pair_cuts() -> (f32, f32) {
    static C: std::sync::OnceLock<(f32, f32)> = std::sync::OnceLock::new();
    *C.get_or_init(|| {
        let (mut hard, mut sep) = (0.0f32, 0.0f32);
        for row in pair_widths() {
            for w in row {
                hard = hard.max(w.hard);
                sep = sep.max(w.sep);
            }
        }
        (hard, sep)
    })
}
/// One neighbour a man may touch this tick (`set_apart`, `collide`): his
/// rank in the grid's order (`SpatialGrid::sorted`), and the pair's body
/// distance and the other man's share of the pair's mass, m_o / (m_i +
/// m_o), as a row of `touch_table` (`code`): eight bytes a pair instead
/// of twelve, the passes' stream over the pairs being their cost in a
/// crush.
#[derive(Clone, Copy)]
pub(crate) struct Touch {
    pub b: u32,
    code: u16,
}
/// The touching pairs' body distances, by the man's width level, the
/// neighbour's and whether both stand in a wall, and their mass shares,
/// by the two men's kinds (`Touch::code`): the values `scan` computes for
/// the pair, looked up.
pub(crate) struct TouchTable {
    hard: [f32; 2 * WIDTH_LEVELS * WIDTH_LEVELS],
    share: [f32; crate::unit_types::NUM_KINDS * crate::unit_types::NUM_KINDS],
}
pub(crate) fn touch_table() -> &'static TouchTable {
    static T: std::sync::OnceLock<TouchTable> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let (pairs, masses) = (pair_widths(), mass_pairs());
        let k = crate::unit_types::NUM_KINDS;
        TouchTable {
            hard: std::array::from_fn(|c| {
                let hard = pairs[c / (2 * WIDTH_LEVELS)][(c / 2) % WIDTH_LEVELS].hard;
                if c % 2 == 1 { hard.min(WALL_SEP_RADIUS) } else { hard }
            }),
            share: std::array::from_fn(|c| masses[c / k][c % k].1),
        }
    })
}
impl Touch {
    #[inline]
    fn new(b: u32, my_level: usize, o_width: usize, both_wall: bool, my_kind: usize, o_kind: usize) -> Touch {
        let hard = (my_level * WIDTH_LEVELS + o_width) * 2 + both_wall as usize;
        let share = my_kind * crate::unit_types::NUM_KINDS + o_kind;
        Touch { b, code: ((hard << 4) | share) as u16 }
    }
}
impl TouchTable {
    #[inline]
    fn hard(&self, t: &Touch) -> f32 {
        self.hard[(t.code >> 4) as usize & (2 * WIDTH_LEVELS * WIDTH_LEVELS - 1)]
    }
    #[inline]
    fn share(&self, t: &Touch) -> f32 {
        self.share[(t.code & 15) as usize]
    }
}
const _: () = assert!(crate::unit_types::NUM_KINDS * crate::unit_types::NUM_KINDS == 16);
const _: () = assert!((2 * WIDTH_LEVELS * WIDTH_LEVELS).is_power_of_two() && 2 * WIDTH_LEVELS * WIDTH_LEVELS <= 1 << 12);
const STEER_GAIN: f32 = 3.0;
const MAX_ACCEL: f32 = 50.0;
/// What a man's legs can do: the steering toward his wish is capped at
/// this acceleration (m/s^2), about what a man in armour with a shield
/// manages from a stand. The gain alone would give 28 m/s^2, three g, at
/// every start and stop, and a lane behind an opening gap would surge
/// and stop as one body. The spring's push and the contact are not legs
/// and keep MAX_ACCEL.
pub(crate) const LEG_ACCEL: f32 = 4.0;
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
/// Positional overlap resolution: the share of a pair's overlap each man
/// is moved back per tick, and the per-tick cap on total correction
/// distance. A packed man overlaps up to six bodies and their corrections
/// add up, so the share stays under one sixth: over it, six bodies throw
/// him past the middle of them and he comes back on the next tick, every
/// tick. 90% of an overlap is gone in 7 ticks.
const CORR_GAIN: f32 = 0.15;
const CORR_MAX: f32 = 0.1;
/// Units slow down proportionally inside this distance to the target.
const ARRIVE_RADIUS: f32 = 35.0;
/// How close to his slot a parked man stands without stepping (metres):
/// rigid slots sit exactly at the separation rest distance, so a dressed
/// rank is force-free and can afford a tight tolerance; with 1.5 the
/// ranks never finished dressing. At 0.7 a man brushed by a passer
/// stayed wherever he landed and the next passer shoved him further.
pub(crate) const HOLD_DEADZONE: f32 = 0.3;
/// A holding soldier who leaves his mark by more than the hold deadzone
/// walks back at this pace at least, and keeps going until he is within
/// STEP_STOP of it: a real step, never a creep. M2TW's slowest
/// locomotion is its shuffle, 0.75 to 0.9 m/s.
pub(crate) const STEP_PACE: f32 = 0.8;
pub(crate) const STEP_STOP: f32 = 0.35;
/// Below this ground speed a standing formation's soldier keeps his
/// facing while he moves: he steps sideways or back, he does not turn
/// to walk (M2TW's shuffle keeps facing). The renderer plays a skinned
/// kind's step clips below it.
pub(crate) const STEP_FACE_SPEED: f32 = 1.05;
/// Going to an enemy he can see: M2TW's ready-stance `advance`
/// (1.06 m/s) for the last few meters or with a comrade close ahead,
/// its `combat_jog` (2.87 m/s) from further out over open ground.
pub(crate) const ADVANCE_PACE: f32 = 1.06;
const COMBAT_JOG_PACE: f32 = 2.87;
const JOG_BEYOND: f32 = 3.0;
/// A man blocked by a comrade on the way to his enemy sidesteps toward
/// an open lane in some one-second windows (M2TW's isSideStepping),
/// and waits in the others.
const SIDESTEP_WINDOW: u32 = 30;
/// A man fights at his weapon's length. A blade lands with its far end,
/// so he stands where that end meets his enemy: he closes to this share
/// of his reach, and steps back to it from an enemy who has come inside
/// it when the ground behind him is open.
const FIGHT_SHARE: f32 = 0.9;
/// He lets an enemy this far inside his fighting distance before he
/// steps back, and then steps all the way out to it: a real step, as on
/// the way back to his mark.
const BACK_SLACK: f32 = 0.3;
/// Man-ticks of that step back since the room log last read them
/// (FL_LOG_ROOM).
pub(crate) static STEP_BACK_TICKS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);
/// Share of those windows in which a blocked man sidesteps
/// (FL_SIDESTEP, 0 to 1). A play-testing knob.
fn sidestep_chance() -> f32 {
    static P: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *P.get_or_init(|| crate::util::env_or("FL_SIDESTEP", 0.2))
}
/// A standing man plants his feet: a push must exceed this (m/s^2 of
/// separation push) before it moves him. About a 10 cm squeeze from one
/// neighbor. A man holding his place (`planted` in steer) keeps the grip
/// while he steps back to it, and his feet take the first STAND_GRIP x dt
/// of a shared shove too, each man 0.7 to 1.3 of it: a man brushed by a
/// passer stays in his file instead of being carried off. Running men
/// are unaffected, a charge's shove is far above it, and true overlap is
/// still corrected.
pub(crate) const STAND_GRIP: f32 = 6.0;
/// The soft wall (FL_CONTACT=wall): a man's legs carry him this far into
/// the body he runs into (his own give, 0.7 to 1.3 of it), losing their
/// closing speed on the way, and nothing of his speed is handed to the
/// man he hits: the press and that man's feet take it. A stop on one tick
/// is the twitch (he is set back and walks in again).
const GIVE: f32 = 0.08;
/// A standing man's stance (his lean, arms and shield) takes the first
/// STANCE metres of a net squeeze before he gives ground to it.
const STANCE: f32 = 0.10;
#[inline]
fn give(i: usize) -> f32 {
    GIVE * own_spread(i)
}
/// His own 0.7 to 1.3 of a man-to-man quantity (his give, his grip).
#[inline]
fn own_spread(i: usize) -> f32 {
    0.7 + 0.6 * crate::units::hash01((i as u32).wrapping_mul(0x51ED).wrapping_add(77))
}

/// A man of a regiment in melee who sees no enemy joins the fight when
/// he sees the comrade beside him run toward it: in each half-second
/// window he reacts with this chance (FL_JOIN_REACT overrides), so the
/// line rolls up from the contact outward, a man at a time, as in
/// M2TW when a deep unit hits a wide line. Once going he keeps going.
const JOIN_WINDOW: u32 = 15;
fn join_react_chance() -> f32 {
    static P: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *P.get_or_init(|| crate::util::env_or("FL_JOIN_REACT", 0.2_f32).clamp(0.0, 1.0))
}
/// Ground speed toward the fight that reads as "going to it". A regiment
/// starts a charge only when its centre closes on its target this fast
/// (frontline.rs).
pub(crate) const GOING_SPEED: f32 = 1.0;
/// How far a man notices a comrade of his own regiment running to the
/// fight (on his far look, below).
const JOIN_SEE_R: f32 = 6.0;
/// A man does not re-read the comrades around him 30 times a second.
/// He looks at the comrades beside him and in his way every LOOK_TICKS
/// (about a quarter second, a human reaction time), in his own rhythm,
/// and acts on what he last saw (the `sight` column) in between. His
/// far look (enemies in sight, comrades running to the fight) keeps the
/// acquisition's own 1-in-8 rhythm.
const LOOK_TICKS: u32 = 8;
/// `sight` bits: a comrade blocks his way, his left, his right; a
/// comrade ahead; a comrade between him and his mark; a comrade close
/// by running to the fight; one further out (far look); a man close
/// behind him.
const SIGHT_WAY: u16 = 1;
pub(crate) const SIGHT_LEFT: u16 = 1 << 1;
pub(crate) const SIGHT_RIGHT: u16 = 1 << 2;
const SIGHT_AHEAD: u16 = 1 << 3;
const SIGHT_MARK: u16 = 1 << 4;
const SIGHT_GO: u16 = 1 << 5;
const SIGHT_GO_FAR: u16 = 1 << 6;
const SIGHT_BACK: u16 = 1 << 7;
/// An enemy stands in his weapon's reach (the scan's finding, every
/// tick): the archers' search never aims at him (sim/mod.rs
/// `clear_members`).
pub(crate) const SIGHT_FOE: u16 = 1 << 8;
/// Bit 9 is free.
/// An archer's knife: bits 10 to 14 count down (one every fourth tick)
/// from the last tick an enemy came close to him (`KNIFE_HOLD`), and
/// `SIGHT_KNIFE` is set while the knife is out, by that count or by his
/// regiment's swords-out order. With it out he draws no bow, and the
/// renderer shows the knife and hides the bow (render_units_gpu.rs).
const SIGHT_KNIFE_SHIFT: u16 = 10;
/// His aim is a ring aim (`choose_opening`): kept beside or behind him
/// until he is on it, unlike a wide-look aim, which is dropped once past.
const SIGHT_RING: u16 = 1 << 9;
/// Ring aims set this tick, for the straggler log (regiments.rs).
pub(crate) static RING_AIMS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
const SIGHT_KNIFE_COUNT: u16 = 0x1f << SIGHT_KNIFE_SHIFT;
pub(crate) const SIGHT_KNIFE: u16 = 1 << 15;
/// The count an enemy close to him sets: 31 steps of four ticks, about
/// 4 s at 30 ticks a second, before he puts the knife away.
const KNIFE_HOLD: u16 = 31;
/// How close an enemy block's edge comes before he takes his knife out:
/// about two seconds of a charge, one bow draw.
pub(crate) const KNIFE_EDGE: f32 = 10.0;
/// Even with nobody going in sight, a man joins once his regiment has
/// been in melee longer than his own patience, spread between these
/// (seconds; FL_JOIN_PATIENCE sets the upper end): the fight's noise and
/// the officers carry further than sight. Far men join late, never not.
const JOIN_PATIENCE_MIN: f32 = 25.0;
fn join_patience_max() -> f32 {
    static D: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *D.get_or_init(|| crate::util::env_or("FL_JOIN_PATIENCE", 50.0_f32).max(JOIN_PATIENCE_MIN))
}

/// Sight radius for a man of a fighting regiment looking for an enemy
/// to go to (FL_SEEK_R, meters). A play-testing knob.
fn seek_radius() -> f32 {
    static R: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *R.get_or_init(|| crate::util::env_or("FL_SEEK_R", 15.0))
}

/// How near an enemy of any regiment must be, coming at a man whose
/// regiment fights one ordered target (sim/job.rs `focus`), before he
/// turns to fight him (FL_THREAT_R, meters). Beyond it he goes after the
/// target's men only. A play-testing knob: M2TW's seek radius is not
/// known (devlog 0120). At the combat jog a man covers 4 m in 1.4 s, a
/// quarter second of which is his next look and a second a full turn.
fn threat_radius() -> f32 {
    static R: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *R.get_or_init(|| crate::util::env_or("FL_THREAT_R", 4.0))
}

/// Routing units flee at this fraction of their speed: fleeing at
/// exactly max speed made pursuit a zero-kill treadmill.
const ROUT_FLEE_FRAC: f32 = 0.9;
/// Sparse-fight acquisition radius. Two depleted formations interleave
/// with gaps wider than QUERY_RADIUS and the fight stalls (hits/tick 0
/// with both sides standing at "the front"). Pressing units whose normal
/// scan finds no enemy look this far on a staggered 1-in-8-tick cadence
/// and lock a closing target; swings still require reach.
const WIDE_ACQUIRE_R: f32 = 4.0;
/// Speed fraction above which a newly started swing counts as a charge.
/// The momentum bonus (attacker's charge_bonus attack points) resolves in
/// the damage pass — a braced SPEARWALL nullifies it (points beat momentum).
const CHARGE_SPEED_FRAC: f32 = 0.6;
/// A regiment in its charge phase runs the last stretch home.
const CHARGE_SPEED_BOOST: f32 = 1.15;
/// Walls advance at a deliberate pace — breaking into a run breaks the wall.
const WALL_SPEED_FRAC: f32 = 0.72;
/// Separation rest distance between two SAME-TEAM units who are both in a
/// wall stance: shoulder to shoulder. Without this the physics pushes a
/// wall back out to normal spacing and the tight slots never happen.
const WALL_SEP_RADIUS: f32 = 1.05;
/// The braced spear is a PHYSICAL hazard, not a stance rule: a
/// spearwall soldier's leveled point covers a line along his facing,
/// from POINT_MIN (get inside it and the spear is useless) out to his
/// reach, this wide. A body crossing that line at charge speed runs
/// onto the point — a damage event from the spearman powered by the
/// CHARGER's own momentum, and a stop (stagger). A charger who slips
/// between the lines (rank edge, dead wielder, hole in the wall)
/// reaches the spearman and swings with his full charge bonus: gaps
/// are real, coverage at 1.05 m wall pitch is ~2/3 of the frontage.
const SPEAR_POINT_MIN: f32 = 0.9;
pub(crate) const SPEAR_LINE_HALF_W: f32 = 0.35;

/// Global damage multiplier. FL_COMBAT_SCALE overrides (fast test battles).
/// What every soldier reads and none writes: the job's shared inputs.
#[derive(Clone, Copy)]
pub(crate) struct Field<'a> {
    pub grid: &'a SpatialGrid,
    pub terrain: &'a Terrain,
    /// Positions at the tick's start.
    pub pos_prev: &'a [Vec3],
    pub speed: &'a [f32],
    pub team: &'a [u8],
    pub kind: &'a [u8],
    pub group: &'a [u32],
    pub home: &'a [Vec2],
    /// Facings at the tick's start, only when a spearwall stands on the field.
    pub yaw_snap: &'a [f32],
    // Per regiment.
    pub orders: &'a [Option<Vec2>],
    pub anchors: &'a [Vec2],
    pub broken: &'a [bool],
    pub press: &'a [bool],
    /// Per regiment: walking through a friend that opened for it.
    pub passing: &'a [bool],
    /// Per regiment: under a Move order (sim/job.rs `moving`).
    pub moving: &'a [bool],
    /// Per regiment: the stance its men are drawn in (STANCE_*).
    pub stance: &'a [u8],
    /// Per regiment: the one enemy regiment its men fight (sim/job.rs
    /// `focus`), None for any.
    pub focus: &'a [Option<u32>],
    pub engaged: &'a [bool],
    pub contact: &'a [bool],
    pub fight_point: &'a [Option<Vec2>],
    pub melee_ticks: &'a [u32],
    pub hold: &'a [bool],
    pub threat: &'a [Vec2],
    pub form_face: &'a [Vec2],
    pub wall: &'a [u8],
    pub charging: &'a [bool],
    pub fat_speed: &'a [f32],
    pub fat_nocharge: &'a [bool],
    pub shoot_at: &'a [Option<crate::sim::job::ShootAt>],
    pub target_members: &'a [Vec<u32>],
    /// Per regiment under search: its clear men, ascending (sim/mod.rs
    /// `clear_members`).
    pub clear_members: &'a [Vec<u32>],
    /// Per archer regiment: the blocks its men search in, nearest first.
    pub search_blocks: &'a [Vec<u32>],
    /// Per regiment: why its men take their knives out (archers only).
    pub sidearm: &'a [crate::sim::job::Sidearm],
    /// The death column at the tick's start: another man's life, read
    /// by the archer who picks him.
    pub death_snap: &'a [u8],
    pub blocks: &'a [Vec<(Vec2, f32, f32)>; 2],
    pub faces_spearwall: [bool; 2],
    pub dt: f32,
    pub tick_seed: u32,
    pub tick: u32,
    pub bounds_min: Vec2,
    pub bounds_max: Vec2,
    /// Per regiment: its route, and the grid (sim/route.rs).
    pub paths: crate::sim::route::KernelRoutes<'a>,
}

/// One soldier's own row: the columns his tick writes.
pub(crate) struct Soldier<'a> {
    pub i: usize,
    /// His new position (the output column).
    pub pos: &'a mut Vec3,
    pub vel: &'a mut Vec3,
    pub yaw: &'a mut f32,
    pub yaw_prev: &'a mut f32,
    pub target: &'a mut u32,
    pub swing: &'a mut u8,
    pub swing_t: &'a mut u8,
    pub flash: &'a mut u8,
    pub death_t: &'a mut u8,
    pub ammo: &'a mut u8,
    pub out_form: &'a mut bool,
    pub sight: &'a mut u16,
    pub way: &'a mut Vec2,
    pub chain: &'a mut u8,
    pub pose: &'a mut u8,
    pub quarry: &'a mut u32,
    pub stuck: &'a mut u16,
    pub aim: &'a mut Vec2,
}

/// The rows of one chunk of soldiers, in index order.
pub(crate) struct Rows<'a> {
    pub start: usize,
    pub pos: &'a mut [Vec3],
    pub vel: &'a mut [Vec3],
    pub yaw: &'a mut [f32],
    pub yaw_prev: &'a mut [f32],
    pub target: &'a mut [u32],
    pub swing: &'a mut [u8],
    pub swing_t: &'a mut [u8],
    pub flash: &'a mut [u8],
    pub death_t: &'a mut [u8],
    pub ammo: &'a mut [u8],
    pub out_form: &'a mut [bool],
    pub sight: &'a mut [u16],
    pub way: &'a mut [Vec2],
    pub chain: &'a mut [u8],
    pub pose: &'a mut [u8],
    pub quarry: &'a mut [u32],
    pub stuck: &'a mut [u16],
    pub aim: &'a mut [Vec2],
}

/// What a chunk emits beside its rows: landed swings and loosed arrows.
pub(crate) struct ChunkOut<'a> {
    pub events: &'a mut Vec<DamageEvent>,
    pub arrows: &'a mut Vec<crate::arrows::ArrowSpawn>,
    /// The chunk's men's touching pairs for the yield-and-collide
    /// contact, man after man, and for each row the end of his run in it.
    pub touch: &'a mut Vec<Touch>,
    pub touch_end: &'a mut [u32],
    /// The chunk's men braked by the still-man crowd (jam over 0) and
    /// held up at the step pace by the brake's floor.
    pub braked: u32,
    pub held: u32,
    /// Scratch for the scan: the ranks of a man's candidates past its cut.
    pub cands: Vec<u32>,
}

/// What one stage of a soldier's tick hands the next.
struct Step {
    /// His position at the tick's start, on the ground plane.
    p: Vec2,
    /// His regiment.
    gi: usize,
    my_kind: usize,
    dying: bool,
    routed: bool,
    /// Where he wants to go, as a velocity; stages add to it and scale it.
    desired: Vec2,
    // The scan.
    push: Vec2,
    corr: Vec2,
    /// Which way the overlap correction points, before its deadband and
    /// cap, how deep the net overlap is along it in metres, and the ground
    /// velocity of the bodies he overlaps, weighted by how far each sets
    /// him back.
    overlap_dir: Vec2,
    overlap_depth: f32,
    overlap_vel: Vec2,
    /// His share by weight of a closing speed on them (the shared speed).
    overlap_share: f32,
    crowd: f32,
    /// The still-man crowd (the yield-and-collide contact's brake): the
    /// neighbours within the spring on the half-plane ahead of his wish,
    /// by their closeness, each times how little he actually moved along
    /// the wish last tick against CRAWL. 0 for a man with no wish.
    crowd_still: f32,
    /// The still-man brake's floor held his drive up at the step pace
    /// (`yield_to_crowd`).
    held: bool,
    /// The crowd ahead of him on his way: the neighbours within the
    /// spring whose direction is within 60 degrees of his wish, by their
    /// closeness (`try_a_side`).
    ahead: f32,
    /// The crowd at which his drive starts to fade and is gone, for his
    /// width (`Width`).
    brake: (f32, f32),
    best_idx: u32,
    sticky: bool,
    prev_target: u32,
    impale_idx: u32,
    at_charge_speed: bool,
    scan_r: f32,
    my_team_bit: u32,
    // His regiment's melee and his part in it.
    in_melee: bool,
    may_leave: bool,
    committed: bool,
    join_fp: Option<Vec2>,
    // What he remembers and sees.
    memo_valid: bool,
    memo_d2: f32,
    memo_dir: Vec2,
    side_dir: Vec2,
    watch_go: bool,
    way_blocked: bool,
    left_blocked: bool,
    right_blocked: bool,
    comrade_ahead: bool,
    slot_blocked: bool,
    back_blocked: bool,
    // The swing and the steering.
    face_target: Option<Vec2>,
    /// His drive before the crowd's brake faded it (`try_a_side`).
    wish: Vec2,
    jam: f32,
    new_v: Vec2,
    // The choice of opening and the squeeze (`choose_opening`,
    // `try_a_side`): whether he reads the comrades ahead this tick, his
    // normal body and spring (not squeezed), whether the scan saw an
    // obstacle ahead and the opening it found among them (across and
    // along his way, its fit), a body at each of his sides, and the
    // opening he aims at (its fit: 0 none, 1 his depth, 2 his width;
    // `gap`: it fits his depth only; `gap_near`: that crack within his
    // spring).
    open_on: bool,
    body0: f32,
    sep0: f32,
    saw: bool,
    near: (f32, f32, u8),
    /// The near opening's air across his way and the far side of its
    /// passage along it (`choose_opening`'s taken crack).
    near_span: (f32, f32, f32),
    between: bool,
    aim_fit: u8,
    gap: bool,
    gap_near: bool,
}

/// Run every soldier of a chunk through his tick.
pub(crate) fn tick_chunk(f: &Field, rows: Rows, out: &mut ChunkOut) {
    for j in 0..rows.pos.len() {
        let mut s = Soldier {
            i: rows.start + j,
            pos: &mut rows.pos[j],
            vel: &mut rows.vel[j],
            yaw: &mut rows.yaw[j],
            yaw_prev: &mut rows.yaw_prev[j],
            target: &mut rows.target[j],
            swing: &mut rows.swing[j],
            swing_t: &mut rows.swing_t[j],
            flash: &mut rows.flash[j],
            death_t: &mut rows.death_t[j],
            ammo: &mut rows.ammo[j],
            out_form: &mut rows.out_form[j],
            sight: &mut rows.sight[j],
            way: &mut rows.way[j],
            chain: &mut rows.chain[j],
            pose: &mut rows.pose[j],
            quarry: &mut rows.quarry[j],
            stuck: &mut rows.stuck[j],
            aim: &mut rows.aim[j],
        };
        let mut st = begin(f, &mut s);
        drive(f, &mut s, &mut st);
        scan(f, &mut s, &mut st, out);
        out.touch_end[j] = out.touch.len() as u32;
        look_around(f, &mut s, &mut st);
        choose_opening(f, &mut s, &mut st);
        hold_the_frame(f, &mut st);
        acquire(f, &mut s, &mut st);
        decide_to_join(f, &mut s, &mut st);
        swing(f, &mut s, &mut st, out);
        yield_to_crowd(&mut st);
        out.braked += (st.jam > 0.0) as u32;
        out.held += st.held as u32;
        try_a_side(f, &mut s, &mut st);
        close_in(f, &mut s, &mut st);
        steer(f, &mut s, &mut st);
        face(f, &mut s, &mut st);
        integrate(f, &mut s, &mut st);
    }
}

/// The tick's bookkeeping before any decision: who he is, whether he is
/// dying or broken, the hit flash and the death countdown, and his part
/// in his regiment's melee.
#[inline]
fn begin(f: &Field, s: &mut Soldier) -> Step {
    let Field { pos_prev, kind, group, broken, fight_point, hold, moving, .. } = *f;
    let i = s.i;
    let p = pos_prev[i].xz();
    let my_kind = kind[i] as usize;
    let dying = *s.death_t > 0;
    // Hit flash decays here (set by the serial apply pass). A twitch code
    // (sim/damage.rs TWITCH_TICKS) counts its low nibble down and goes
    // at once.
    *s.flash = if *s.flash >= 16 {
        if *s.flash & 15 > 1 { *s.flash - 1 } else { 0 }
    } else {
        s.flash.saturating_sub(1)
    };
    // Corpses play out their death anim: no orders, no combat; they stay as
    // an obstacle until swept.
    if dying && *s.death_t > 1 {
        *s.death_t -= 1;
    }
    let gi = group[i] as usize;
    let routed = broken[gi];
    // In melee: his regiment has a fight. Out of formation (out_form): he
    // left his slot to fight, his own state, not his regiment's (M2TW's
    // isInFormation). It sticks while he sees an enemy or his regiment has
    // a fight to head for (decide_to_join); a Move order, a hold order, a
    // rout and death bring him back. Fighting takes him out of formation.
    // Keyed on the regiment's melee instead, a man whose regiment had no
    // fight on its clock, with an enemy standing off his flank, went to
    // him and was pulled back to his slot, over and over (devlog 0212).
    let in_melee = fight_point[gi].is_some() && !routed && !dying;
    let may_leave = !routed && !dying && !hold[gi] && !moving[gi];
    if !may_leave {
        *s.out_form = false;
    } else if *s.swing & crate::units::SWING_STATE_MASK
        != crate::units::SWING_READY
        && *s.swing & crate::units::SWING_STATE_MASK
            != crate::units::SWING_DEFEND
        && *s.swing & crate::units::SWING_RANGED == 0
    {
        *s.out_form = true;
    }
    if !in_melee {
        *s.sight &= !(SIGHT_GO | SIGHT_GO_FAR);
    }
    let committed = *s.out_form;
    let join_fp = if in_melee { fight_point[gi] } else { None };
    Step {
        p,
        gi,
        my_kind,
        dying,
        routed,
        desired: Vec2::ZERO,
        push: Vec2::ZERO,
        corr: Vec2::ZERO,
        overlap_dir: Vec2::ZERO,
        overlap_depth: 0.0,
        overlap_vel: Vec2::ZERO,
        overlap_share: 0.0,
        crowd: 0.0,
        crowd_still: 0.0,
        held: false,
        ahead: 0.0,
        brake: (0.0, 0.0),
        best_idx: u32::MAX,
        sticky: false,
        prev_target: u32::MAX,
        impale_idx: u32::MAX,
        at_charge_speed: false,
        scan_r: 0.0,
        my_team_bit: 0,
        in_melee,
        may_leave,
        committed,
        join_fp,
        memo_valid: false,
        memo_d2: f32::MAX,
        memo_dir: Vec2::ZERO,
        side_dir: Vec2::ZERO,
        watch_go: false,
        way_blocked: false,
        left_blocked: false,
        right_blocked: false,
        comrade_ahead: false,
        slot_blocked: false,
        back_blocked: false,
        face_target: None,
        wish: Vec2::ZERO,
        jam: 0.0,
        new_v: Vec2::ZERO,
        open_on: false,
        body0: 0.0,
        sep0: 0.0,
        saw: false,
        near: (0.0, 0.0, 0),
        near_span: (0.0, 0.0, 0.0),
        between: false,
        aim_fit: 0,
        gap: false,
        gap_near: false,
    }
}

/// Where he wants to go on his own account: a broken man flees for his
/// map edge, a formed man walks to his slot under his regiment's order, at
/// a pace set by slope and water, with a deadzone so a parked man does not
/// jitter and a step that finishes once begun.
#[inline]
fn drive(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { terrain, speed, team, home, orders, anchors, paths, .. } = *f;
    let i = s.i;
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    // Units move ONLY under orders. A regiment order is one point rigidly
    // translated by each unit's `home` offset (the block moves; it never
    // converges). No order = hold at the anchor — a standing order: units
    // drift back to their slot at reduced gain inside the block. Enemy
    // contact is pure physics: cross-team separation blocks, crowd yield
    // stops the shove, combat thins the block. The "front line" is where that
    // collision is.
    let mut desired = Vec2::ZERO;
    if !dying && routed {
        // Broken: flee toward the own map edge with a per-unit lateral
        // scatter — slightly SLOWER than formed pursuers (0.9x): fleeing at
        // exactly max speed made pursuit a zero-kill treadmill (gap frozen
        // forever, counts flat for 30-45 s).
        let flee_z: f32 = if team[i] == 0 { -1.0 } else { 1.0 };
        let lat = (crate::units::hash01((i as u32).wrapping_mul(17) + 3) - 0.5)
            * 0.7;
        let dir = Vec2::new(lat, flee_z).normalize_or_zero();
        let slope = terrain.slope_at(p.x, p.y);
        desired = dir
            * speed[i]
            * ROUT_FLEE_FRAC
            * terrain.wade_mult(p.x, p.y)
            / (1.0 + 3.0 * slope * slope);
    } else if !dying {
        let holding = orders[gi].is_none();
        let mut goal = orders[gi].unwrap_or(anchors[gi]) + home[i];
        // On a route, a man whose straight way to his slot is blocked
        // follows the route instead (sim/route.rs); short of the last leg
        // he keeps his pace. He reads the route every LOOK_TICKS, when he
        // reaches the point he chose, or when he has none, and walks on
        // what he chose between reads.
        let mut through = false;
        match (paths.routes.get(gi).and_then(Option::as_ref), paths.nav) {
            (Some(route), Some(nav)) => {
                let frame = orders[gi].unwrap_or(anchors[gi]);
                let way = *s.way;
                let kept = !way.x.is_nan();
                let at_way = kept && way.x.is_finite() && p.distance_squared(way) < 4.0;
                if !kept || at_way || (i as u32).wrapping_add(f.tick).is_multiple_of(LOOK_TICKS) {
                    let (t, th) = crate::sim::route::man_target(nav, terrain, route, frame, p, goal, i);
                    *s.way = if t == goal && !th { Vec2::INFINITY } else if t == goal { Vec2::NEG_INFINITY } else { t };
                    (goal, through) = (t, th);
                } else if way.x.is_infinite() {
                    through = way.x < 0.0;
                } else {
                    (goal, through) = (way, true);
                }
            }
            _ => *s.way = Vec2::NAN,
        }
        let to_goal = goal - p;
        let dist = to_goal.length();
        // Hold deadzone: parked units don't jitter around their slot point
        // (HOLD_DEADZONE). A soldier already stepping finishes the step
        // (STEP_STOP) instead of stalling at the deadzone's edge.
        let stepping = s.vel.xz().length_squared()
            > (0.5 * STEP_PACE) * (0.5 * STEP_PACE);
        let deadzone = if stepping { STEP_STOP } else { HOLD_DEADZONE };
        if !(holding && dist < deadzone) {
            // Slope penalty: steep ground is slow ground. Wading the river is
            // slow too (the bridge deck is dry: full speed).
            let slope = terrain.slope_at(p.x, p.y);
            let slope_mult = 1.0 / (1.0 + 3.0 * slope * slope);
            let (gain, arrive) = if holding {
                (0.6, 10.0)
            } else {
                (1.0, ARRIVE_RADIUS)
            };
            let mut desired_speed = speed[i]
                * slope_mult
                * terrain.wade_mult(p.x, p.y)
                * gain
                * if through { 1.0 } else { (dist / arrive).min(1.0) };
            if holding {
                desired_speed = desired_speed
                    .max(STEP_PACE * slope_mult * terrain.wade_mult(p.x, p.y));
            }
            if dist > 1e-3 {
                desired = to_goal * (desired_speed / dist);
            }
        }
    }
    st.desired = desired;
}

/// The fused neighbour scan every man runs every tick over the 2 m box:
/// separation pushes and overlap corrections, the crowd around him, the
/// nearest living enemy in his weapon's reach (sticky to the one he already
/// fights), and, at charge speed against a spearwall, the braced point his
/// body runs onto, which is a damage event from the spearman.
#[inline]
fn scan(f: &Field, s: &mut Soldier, st: &mut Step, out: &mut ChunkOut) {
    let Field {
        grid,
        speed,
        team,
        yaw_snap,
        wall,
        stance,
        faces_spearwall,
        tick_seed,
        orders,
        engaged,
        anchors,
        home,
        passing,
        ..
    } = *f;
    let i = s.i;
    let params = &TYPES[st.my_kind];
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    // Fused neighbor scan: separation physics + nearest living enemy in reach
    // (swing targeting). Scalar over cache-ordered SortedUnits — an 8-wide
    // SIMD variant measured slower here: candidate runs are too short for
    // lane occupancy.
    let mut push = Vec2::ZERO;
    let mut corr = Vec2::ZERO;
    let mut overlap_vel = Vec2::ZERO;
    let mut overlap_share = 0.0f32;
    let mut overlap = 0.0f32;
    let mut crowd = 0.0f32;
    let mut crowd_still = 0.0f32;
    let mut ahead = 0.0f32;
    let wish_dir = st.desired.normalize_or_zero();
    let my_team_bit = (team[i] as u32) * crate::spatial::META_TEAM;
    let my_wall = wall[gi] != 0;
    // His width and each neighbour's by their poses and the turn of their
    // shoulders at the tick's start, the pair's rest, its inverse and its
    // overlap distance from a table, and the pair's mass weights by kind.
    let widths = widths();
    let squeeze_on = squeezing(*s.stuck);
    let pose_lv = pose_level(*s.pose);
    let my_level = width_level(*s.pose);
    let mine = &widths[my_level];
    // The choice of opening's read (`choose_opening`): only a man walking
    // somewhere, not fighting, reads the comrades ahead; a man squeezing
    // feels the bodies at his sides.
    let body0 = widths[pose_lv].body;
    let far = (anchors[gi] + home[i]).distance_squared(p) > TRY_FROM_SLOT * TRY_FROM_SLOT;
    let walking = orders[gi].is_some() || far;
    // Only a man with a reason to pick an opening: his regiment walking
    // through a friend that opened for it, or himself a straggler of a
    // holding regiment. Read for every walking man, it cost 5.6 ms of
    // the 200k step in a front battle where nobody passes anyone.
    let reason = passing[gi] || (orders[gi].is_none() && far);
    let open_on = opening_on()
        && walking
        && reason
        && !engaged[gi]
        && !dying
        && !st.routed
        && !st.in_melee
        && !st.committed
        && st.join_fp.is_none()
        && st.desired.length_squared() > 0.25;
    // Every obstacle within his reach is in the row nearest him (the band
    // reaches past OPEN_SEE), so the scan keeps the nearest two on each
    // side as it goes.
    let mut row = Row::default();
    let mut row_at = f32::INFINITY;
    let (mut side_left, mut side_right) = (false, false);
    let pairs = &pair_widths()[my_level];
    let masses = &mass_pairs()[st.my_kind];
    let gain = CORR_GAIN;
    let reach2 = params.reach * params.reach;
    let prev_target = *s.target;
    let mut best_d2 = f32::MAX;
    let mut best_idx = u32::MAX;
    let mut sticky = false;
    // Moving at charge speed with unspent momentum: braced enemy spears in
    // the path are a collision hazard, and the scan must see out to SPEAR
    // reach, not just mine. A man already run through (flash; a twitch
    // code of 16 up is a render cue, not a wound) or reeling is not
    // re-impaled every tick: a spear is a point, not an aura.
    let spear_reach = TYPES[crate::unit_types::KIND_SPEAR as usize].reach;
    let cs = speed[i] * CHARGE_SPEED_FRAC;
    let at_charge_speed = faces_spearwall[team[i] as usize]
        && !dying
        && (*s.flash == 0 || *s.flash >= 16)
        && *s.swing & crate::units::SWING_STAGGERED == 0
        && s.vel.xz().length_squared() > cs * cs;
    let mut impale_idx = u32::MAX;
    let mut impale_d = f32::MAX;
    // The yield-and-collide contact's touching pairs: every neighbour
    // whose body he would touch at their stepped places, both men a tick
    // on at their velocities of the tick's start.
    let touching = !contact_old();
    // His stepped place from the same numbers the grid hands his
    // neighbours for him (his place and velocity at the tick's start), so
    // both men of a pair compute the same distance and list it or not.
    let dt = f.dt;
    let my_step = p + s.vel.xz() * dt;
    let scan_r = if at_charge_speed {
        QUERY_RADIUS.max(params.reach).max(spear_reach)
    } else {
        QUERY_RADIUS.max(params.reach)
    };
    // The farthest a neighbour can be and still count for anything below,
    // a friend and a foe: the touching pair at the widest pair's body
    // distance with both men closing at full speed (no neighbour is
    // faster than the grid's fastest), a friend within SEP_RADIUS, a foe
    // within the widest pair's spring rest, his weapon's reach, or a
    // braced spear's point. Every test below needs the neighbour nearer
    // than that, so a man farther off is out before any of them, and the
    // box of cells shrinks to it, each row of the box to the cells the
    // disc of it reaches (a fast pair may touch from beyond the old
    // radius: the box never grows past it). CUT_MARGIN is far above the
    // rounding of a place on the map.
    //
    // A few men thrown out of a crush run far faster than anyone walks,
    // and the fastest sets the touching pair's reach for every man: the
    // cut is drawn for neighbours no faster than CUT_FAST, and a man
    // beyond it goes on to the tests only when he is faster.
    let (cut_hard, cut_sep) = pair_cuts();
    let my_speed = s.vel.xz().length();
    let touch_at = |v: f32| if touching { cut_hard + TOUCH_SLACK + (my_speed + v) * dt } else { 0.0 };
    let spear_cut = if at_charge_speed { (spear_reach * spear_reach + SPEAR_LINE_HALF_W * SPEAR_LINE_HALF_W).sqrt() } else { 0.0 };
    let foe_at = cut_sep.max(params.reach).max(spear_cut);
    let touch_cut = touch_at(grid.vel_max());
    let same_cut = SEP_RADIUS.max(touch_cut) + CUT_MARGIN;
    let foe_cut = foe_at.max(touch_cut) + CUT_MARGIN;
    // The box drawn for the fastest man on the field: the look round him
    // reads it (`look_around`).
    let scan_r = scan_r.min(same_cut.max(foe_cut));
    // His own neighbourhood, the cells of that box: whether any man of
    // the other side stands in them, and the fastest man in them. With no
    // enemy there every enemy test below fails, and no man there is
    // faster than its fastest, so the cut and the box are drawn for the
    // friends alone and for that speed: the walk lists fewer men and every
    // one who counts.
    let (foes, box_vel2, box_men) = grid.box_foes_vel2(p, scan_r, my_team_bit != 0);
    let box_vel = box_vel2.sqrt();
    let fast = box_vel.min(CUT_FAST);
    let (touch_box, touch_near) = (touch_at(box_vel), touch_at(fast));
    let same_near = SEP_RADIUS.max(touch_near) + CUT_MARGIN;
    let foe_near = foe_at.max(touch_near) + CUT_MARGIN;
    let (same_cut2, foe_cut2) = (same_cut * same_cut, foe_cut * foe_cut);
    let (same_near2, foe_near2, fast2) = (same_near * same_near, foe_near * foe_near, fast * fast);
    let same_box = SEP_RADIUS.max(touch_box) + CUT_MARGIN;
    let disc_r = if foes { same_box.max(foe_at.max(touch_box) + CUT_MARGIN) } else { same_box };
    let walk_r = scan_r.min(disc_r);
    // First the candidates past the cut, by rank, in the walk's order:
    // which of them pass is a coin toss in a press, so each is written
    // and kept by the test's outcome instead of by a branch on it. The
    // box holds every man the walk visits.
    out.cands.clear();
    out.cands.reserve(box_men);
    let kept = out.cands.as_mut_ptr();
    let cut = crate::spatial::KeepCut {
        team_bit: my_team_bit,
        me: i as u32,
        same_near2,
        same_cut2,
        foe_near2,
        foe_cut2,
        fast2,
    };
    // SAFETY: the box holds every man the walk visits, and the list has
    // room for all of them.
    let n_kept = unsafe { grid.kept_in_disc(p, walk_r, disc_r, &cut, kept) };
    let sorted = grid.sorted();
    // Then the touching pairs, written and kept by the test's outcome as
    // well, and the candidates kept again for the rest of the tests: a
    // friend beyond the widest spring rest counts for nothing in them.
    // Most of a scan's candidates in a press are such men (the cells span
    // more than the circle).
    if touching {
        out.touch.reserve(n_kept);
    }
    // The list's end held in locals over the loop, so it stays in
    // registers: the list's length is set once after it.
    let (touch_at_end, mut n_touch) = (out.touch.as_mut_ptr(), out.touch.len());
    let my_kind = st.my_kind;
    let mut n_on = 0usize;
    for j in 0..n_kept {
        // SAFETY: j < n_kept, the list's length.
        let o_rank = unsafe { *kept.add(j) };
        let o = &sorted[o_rank as usize];
        let (o_vel, _) = grid.vel_move(o_rank as usize);
        let same = (o.meta & crate::spatial::META_TEAM) == my_team_bit;
        if touching {
            let w = &pairs[crate::spatial::meta_width(o.meta)];
            let both_wall = my_wall && same && (o.meta & crate::spatial::META_WALL) != 0;
            let hard = if both_wall { w.hard.min(WALL_SEP_RADIUS) } else { w.hard };
            let reach = hard + TOUCH_SLACK;
            let touches = (my_step - (o.xz() + o_vel * dt)).length_squared() < reach * reach;
            let (o_width, o_kind) = (crate::spatial::meta_width(o.meta), crate::spatial::meta_kind(o.meta));
            // SAFETY: the slot at `n_touch` is within the capacity
            // reserved above (one for each kept candidate), and the length
            // takes it only once it is written.
            unsafe { touch_at_end.add(n_touch).write(Touch::new(o_rank, my_level, o_width, both_wall, my_kind, o_kind)) };
            n_touch += touches as usize;
        }
        let d2 = (p - o.xz()).length_squared();
        // SAFETY: n_on <= j, a slot already read.
        unsafe { *kept.add(n_on) = o_rank };
        n_on += !(same & (d2 >= SEP_RADIUS * SEP_RADIUS)) as usize;
    }
    // SAFETY: the slots up to `n_touch` and the first `n_on` are written.
    unsafe {
        out.touch.set_len(n_touch);
        out.cands.set_len(n_on);
    }
    for &o_rank in out.cands.iter() {
        let o = &sorted[o_rank as usize];
        let (o_vel, o_move) = grid.vel_move(o_rank as usize);
        let d = p - o.xz();
        let d2 = d.length_squared();
        let same = (o.meta & crate::spatial::META_TEAM) == my_team_bit;
        let enemy = !same && (o.meta & crate::spatial::META_DYING) == 0;
        if enemy && d2 < reach2 {
            sticky |= o.idx == prev_target;
            if d2 < best_d2 {
                best_d2 = d2;
                best_idx = o.idx;
            }
        }
        // Spear-line collision: he is a braced enemy spearman, and my body is
        // crossing his leveled point while I close at speed.
        if at_charge_speed
            && enemy
            && (o.meta & crate::spatial::META_WALL) != 0
            && crate::spatial::meta_kind(o.meta)
                == crate::unit_types::KIND_SPEAR as usize
        {
            let ys = yaw_snap[o.idx as usize];
            let f = Vec2::new(ys.sin(), ys.cos());
            let rel = d; // spearman -> me (d = p - o.xz())
            let fwd_d = rel.dot(f);
            let lat = (rel.x * f.y - rel.y * f.x).abs();
            let closing = s.vel.xz().dot(d) < 0.0;
            if fwd_d > SPEAR_POINT_MIN
                && fwd_d < spear_reach
                && lat < SPEAR_LINE_HALF_W
                && closing
                && fwd_d < impale_d
            {
                impale_d = fwd_d;
                impale_idx = o.idx;
            }
        }
        // Two same-team units both in a wall STANCE rest shoulder to shoulder
        // (symmetric predicate: both sides compute the same radius). Ordinary
        // ranks keep parade spacing even while fighting — the seams between
        // files are what enemy bodies flow into.
        let cross = (o.meta & crate::spatial::META_TEAM) != my_team_bit;
        // No packing rule for fighting regiments: M2TW keeps its formation
        // grid during melee, and observably loosens it. The fighting crowd's
        // spacing is slots plus body collision, nothing else. A
        // shoulder-to-shoulder press rest for fighting pairs sealed the very
        // seams the intermix needs: a pressed front's gaps shrank to about
        // 1.05 m against 0.95 m bodies and symmetric fights collapsed to a
        // two-rank duel line. It must not come back.
        // The comrades ahead for the choice of opening: a queue (a man of
        // a regiment under an order moving his way) is not an obstacle; a
        // holding man always is.
        if same && (o.meta & crate::spatial::META_DYING) == 0 && (open_on | squeeze_on) {
            let to = -d;
            let along = to.dot(wish_dir);
            let lat = wish_dir.x * to.y - wish_dir.y * to.x;
            let o_body = widths[crate::spatial::meta_pose(o.meta)].body;
            if open_on && along > OPEN_AHEAD {
                let holding = orders[crate::spatial::meta_group(o.meta)].is_none();
                let queue = !holding && o_vel.dot(wish_dir) > QUEUE_SPEED;
                if !queue {
                    row.add(lat, 0.5 * o_body, along);
                    row_at = row_at.min(along);
                }
            }
            if squeeze_on {
                let near = 0.5 * (body0 + o_body) - BETWEEN_SLACK;
                if d2 < near * near && along.abs() < 1.0 {
                    side_left |= lat > 0.0;
                    side_right |= lat < 0.0;
                }
            }
        }
        let pair = &pairs[crate::spatial::meta_width(o.meta)];
        let (sep_r, inv_sep, hard) = if !cross && my_wall && (o.meta & crate::spatial::META_WALL) != 0 {
            (WALL_SEP_RADIUS, 1.0 / WALL_SEP_RADIUS, pair.hard.min(WALL_SEP_RADIUS))
        } else {
            (pair.sep, pair.inv_sep, pair.hard)
        };
        if d2 < sep_r * sep_r && d2 > 1e-8 {
            // Mass-weighted: heavies shove lights aside.
            let (mw, share) = masses[crate::spatial::meta_kind(o.meta)];
            let len = d2.sqrt();
            let inv_len = 1.0 / len;
            let w = 1.0 - len * inv_sep;
            // The yield-and-collide contact reads none of the overlap's
            // correction (`steer`).
            if !touching && len < hard {
                // Overlap is resolved POSITIONALLY only. A hard force boost
                // here as well made packed crowds oscillate at the
                // acceleration cap, two solvers fighting over the same
                // overlap, every frame.
                let back = (hard - len) * gain * mw;
                corr += d * (back * inv_len);
                overlap_vel += o_vel * back;
                overlap_share += share * back;
                overlap += back;
            }
            push += d * (w * mw * inv_len);
            ahead += w * (d.dot(wish_dir) < -0.5 * len) as u32 as f32;
            // Jam density: COMPRESSED pairs only — a neighbor resting AT his
            // pair's rest distance contests nothing, so a formed man's
            // slot-keeping is never faded by the settled enemies one stride
            // away, which made a line yield and stay ragged.
            crowd += w;
            // The still-man crowd: a neighbour on the half-plane ahead of
            // his wish (`d` points from the neighbour to him), by how
            // little he moved along it last tick.
            // Weighed by 1 or 0 without a branch (ahead of him or not is a
            // coin toss): a 0 added changes nothing, since the sum is never
            // -0.
            if touching {
                let on = (d.dot(wish_dir) < 0.0) as u32 as f32;
                crowd_still += w * on * (1.0 - o_move.dot(wish_dir) / CRAWL).clamp(0.0, 1.0);
            }
        }
    }

    // Ran onto a braced spear: the collision is a damage event from the
    // SPEARMAN, resolved with everything else in the serial apply (which also
    // stops the runner). One point, one wound — the nearest line crossed this
    // tick.
    if impale_idx != u32::MAX {
        let jit = 0.85
            + 0.3
                * crate::units::hash01(
                    tick_seed ^ (i as u32).wrapping_mul(0x7A31),
                );
        out.events.push(DamageEvent {
            victim: i as u32,
            attacker: impale_idx,
            jit,
            charge: false,
            impale: true,
            warn: 0,
            chain: 0,
        });
    }
    // His mark for the archers (sim/mod.rs `clear_members`): every tick
    // whether an enemy stands in his reach.
    if !dying {
        if best_idx != u32::MAX {
            *s.sight |= SIGHT_FOE;
        } else {
            *s.sight &= !SIGHT_FOE;
        }
    }
    // His stance for the next tick, as he is drawn: into the ready stance
    // while his regiment fights, while it stands ready and he stands (the
    // pose pass's guard, below the speed of his steps), or with an enemy
    // in his reach; else back at ease, walking or running. A step of the
    // ramp a tick.
    let ready = match stance[gi] {
        STANCE_FIGHT => true,
        STANCE_READY => s.vel.xz().length_squared() < STEP_FACE_SPEED * STEP_FACE_SPEED,
        _ => false,
    };
    let stance_ramp = *s.pose & POSE_STANCE;
    let stance_ramp = if ready || best_idx != u32::MAX {
        (stance_ramp + 1).min(POSE_READY)
    } else {
        stance_ramp.saturating_sub(1)
    };
    *s.pose = (*s.pose & !POSE_STANCE) | stance_ramp;
    st.brake = (mine.slow, mine.stop);
    // An archer's knife: out while his regiment has its swords out, and
    // for a few seconds after an enemy came close to him, one in his
    // reach, an enemy block's edge within KNIFE_EDGE of him, or with his
    // quiver empty an enemy block in his regiment's bow range.
    if !dying && st.my_kind == crate::unit_types::KIND_ARCHER as usize {
        let arm = &f.sidearm[gi];
        let close = best_idx != u32::MAX
            || (*s.ammo == 0 && arm.foe_in_range)
            || arm.blocks.iter().any(|b| b.distance(p) <= KNIFE_EDGE);
        let mut count = (*s.sight & SIGHT_KNIFE_COUNT) >> SIGHT_KNIFE_SHIFT;
        if close {
            count = KNIFE_HOLD;
        } else if count > 0 && f.tick.wrapping_add(i as u32).is_multiple_of(4) {
            count -= 1;
        }
        let out = if arm.swords || count > 0 { SIGHT_KNIFE } else { 0 };
        *s.sight = (*s.sight & !(SIGHT_KNIFE_COUNT | SIGHT_KNIFE)) | count << SIGHT_KNIFE_SHIFT | out;
    }
    st.push = push;
    st.corr = corr;
    if overlap > 0.0 {
        st.overlap_vel = overlap_vel / overlap;
        st.overlap_share = overlap_share / overlap;
    }
    st.crowd = crowd;
    st.crowd_still = crowd_still;
    st.ahead = ahead;
    st.best_idx = best_idx;
    st.sticky = sticky;
    st.prev_target = prev_target;
    st.impale_idx = impale_idx;
    st.at_charge_speed = at_charge_speed;
    st.scan_r = scan_r;
    st.my_team_bit = my_team_bit;
    st.open_on = open_on;
    st.body0 = body0;
    st.sep0 = widths[pose_lv].sep;
    st.saw = row_at.is_finite();
    if open_on && st.saw {
        let (pref, coin) = side_pref(i, *s.stuck);
        let (aim_lat, aim_along, fit, span) = openings(&row, row_at, OPEN_SEE, body0, SQUEEZE_DEPTH, false, pref, coin);
        st.near = (aim_lat, aim_along, fit);
        st.near_span = span;
    }
    st.between = side_left && side_right;
}

/// The ring read (`choose_opening`): how far round him a hemmed man reads
/// the men for a crack, and how close to a ring aim counts as on it.
const RING_SEE: f32 = 2.0;
const RING_ON: f32 = 0.3;
/// The widest open side (radians between two neighbours round him) a man
/// may have and still count as hemmed in: a third of a turn.
const RING_OPEN_SPAN: f32 = 2.0 * std::f32::consts::FRAC_PI_3;
/// FL_OPEN_RING=0 turns the ring read off.
fn open_ring() -> bool {
    static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *K.get_or_init(|| crate::util::env_or("FL_OPEN_RING", 1u32) != 0)
}

/// The side a man leans to among equal openings: the side he chose last
/// while he remembers it, and his own coin, the same every tick.
#[inline]
fn side_pref(i: usize, stuck: u16) -> (f32, f32) {
    let pref = if (stuck & STUCK_MEM) != 0 { if stuck & STUCK_LEFT != 0 { 1.0 } else { -1.0 } } else { 0.0 };
    let coin = if crate::units::hash01((i as u32).wrapping_mul(0x7A31).wrapping_add(13)) > 0.5 { 1.0 } else { -1.0 };
    (pref, coin)
}

/// The choice of opening (OPEN_SEE above): from the obstacles ahead his
/// scan saw, and on his look tick from a wider look, the opening he aims
/// at; his drive turned toward it at its pace; the side he chose
/// remembered as the try remembers its side. Only a man walking somewhere
/// with no enemy in mind and his regiment not fighting (the try's test):
/// fighting men keep their own footwork.
#[inline]
fn choose_opening(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { grid, orders, tick, anchors, home, .. } = *f;
    let i = s.i;
    let going = st.open_on && st.best_idx == u32::MAX && !st.memo_valid;
    let ring_flag = *s.sight & SIGHT_RING != 0;
    *s.sight &= !SIGHT_RING;
    if !going {
        *s.aim = Vec2::NAN;
        return;
    }
    let p = st.p;
    let want = st.desired.length();
    let wd = st.desired / want;
    let perp = Vec2::new(-wd.y, wd.x);
    let mut v = *s.stuck;
    // The near read, from the scan.
    let (mut aim_lat, mut aim_along, mut fit) = st.near;
    if fit > 0 {
        *s.aim = Vec2::NAN;
    }
    // A man he follows still takes up his space: a near crack that fits
    // only his depth is taken by any comrade ahead of him, one he follows
    // or not, whose body overlaps its air short of its passage's far side.
    // He may still aim at a taken crack, which is following, but never
    // turns his shoulders for it. Only for a man with such a crack in his
    // aim, so a second look at the comrades within the scan's box.
    let mut taken = false;
    if fit == 1 {
        let (lo, hi, far) = st.near_span;
        let my_team_bit = st.my_team_bit;
        let widths = widths();
        grid.for_each_candidate(p, QUERY_RADIUS, |o| {
            if taken
                || o.idx as usize == i
                || (o.meta & crate::spatial::META_TEAM) != my_team_bit
                || (o.meta & crate::spatial::META_DYING) != 0
            {
                return;
            }
            let to = o.xz() - p;
            let along = to.dot(wd);
            if along <= OPEN_AHEAD || to.length_squared() >= QUERY_RADIUS * QUERY_RADIUS {
                return;
            }
            let lat = wd.x * to.y - wd.y * to.x;
            let r = 0.5 * widths[crate::spatial::meta_pose(o.meta)].body;
            taken = along - r < far && lat + r > lo && lat - r < hi;
        });
    }
    // The wide look: nothing near fits and he saw somebody in his way.
    // The ring (FL_OPEN_RING): a man of a holding regiment far from his
    // place with nobody ahead fitting him reads the men all round him within RING_SEE on his look
    // tick, the crack between each two neighbours round the ring, and aims
    // at the one that fits him nearest his way, whichever side it lies.
    // A man hemmed in a dressed block at an angle to his way sees no pair
    // ahead of his line, and the cracks round him run between a man ahead
    // and a man abeam or behind (devlog 0357).
    let look = tick.wrapping_add((i as u32).wrapping_mul(7)).is_multiple_of(OPEN_LOOK_TICKS);
    // Only a man whose regiment holds: a marching man far from his place
    // is the whole regiment on the march, and a lane walker who read the
    // ring aimed beside himself and jammed the walk pass.
    let hemmed = open_ring()
        && orders[st.gi].is_none()
        && (anchors[st.gi] + home[i]).distance_squared(p) > TRY_FROM_SLOT * TRY_FROM_SLOT;
    let mut ring_aim = false;
    if fit == 0 && st.saw && hemmed && look {
        let my_team_bit = st.my_team_bit;
        let widths = widths();
        let mut men: [(f32, Vec2, f32); 16] = [(0.0, Vec2::ZERO, 0.0); 16];
        let mut n = 0usize;
        let mut own = 0usize;
        grid.for_each_candidate_vel(p, RING_SEE, |o, o_vel| {
            if n == men.len()
                || o.idx as usize == i
                || (o.meta & crate::spatial::META_TEAM) != my_team_bit
                || (o.meta & crate::spatial::META_DYING) != 0
            {
                return;
            }
            let to = o.xz() - p;
            if to.length_squared() >= RING_SEE * RING_SEE {
                return;
            }
            let holding = orders[crate::spatial::meta_group(o.meta)].is_none();
            if !holding && o_vel.dot(wd) > QUEUE_SPEED {
                return;
            }
            let ang = (wd.x * to.y - wd.y * to.x).atan2(to.dot(wd));
            men[n] = (ang, to, 0.5 * widths[crate::spatial::meta_pose(o.meta)].body);
            n += 1;
            own += usize::from(crate::spatial::meta_group(o.meta) == st.gi);
        });
        // Hemmed in: men on every side of him, no open side wider than
        // RING_OPEN_SPAN, and most of them another regiment's. A man in a
        // lane or at a block's edge has an open side and walks; a man
        // inside his own block (an attacking block the moment it arrives)
        // is among his comrades; the ring is for the man left in a cell of
        // another regiment's planted men.
        let mut open_span = std::f32::consts::TAU;
        if n >= 3 {
            men[..n].sort_by(|a, b| a.0.total_cmp(&b.0));
            open_span = (0..n)
                .map(|k| if k + 1 == n { men[0].0 + std::f32::consts::TAU - men[k].0 } else { men[k + 1].0 - men[k].0 })
                .fold(0.0f32, f32::max);
        }
        if n >= 3 && open_span <= RING_OPEN_SPAN && 2 * own < n {
            let mut best = f32::INFINITY;
            let mut best_point = Vec2::ZERO;
            let mut best_fit = 0u8;
            for k in 0..n {
                let (a0, q0, h0) = men[k];
                let (a1, q1, h1) = men[(k + 1) % n];
                let span = if k + 1 == n { a1 + std::f32::consts::TAU - a0 } else { a1 - a0 };
                let (point, wf) = if span > std::f32::consts::FRAC_PI_2 {
                    // Open ground between them: the way between at the
                    // nearer man's distance.
                    let mid = a0 + 0.5 * span;
                    let dir = wd * mid.cos() + perp * mid.sin();
                    (dir * q0.length().min(q1.length()).max(st.sep0), 2u8)
                } else {
                    let chord = q1 - q0;
                    let len = chord.length();
                    let air = len - h0 - h1;
                    let wf = if air >= st.body0 - OPEN_SLACK {
                        2
                    } else if air >= SQUEEZE_DEPTH {
                        1
                    } else {
                        0
                    };
                    if wf == 0 || len < 1e-3 {
                        continue;
                    }
                    (q0 + chord * ((h0 + 0.5 * air) / len), wf)
                };
                let off = (wd.x * point.y - wd.y * point.x).atan2(point.dot(wd)).abs();
                let cost = off + if wf == 1 { SQUEEZE_DETOUR } else { 0.0 };
                if cost < best {
                    best = cost;
                    best_point = point;
                    best_fit = wf;
                }
            }
            if best_fit > 0 {
                *s.aim = p + best_point;
                v = (v & !STUCK_WIDE_DEPTH) | ((best_fit == 1) as u16 * STUCK_WIDE_DEPTH);
                ring_aim = true;
                RING_AIMS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    // The wide look: nothing near fits and he saw somebody in his way.
    if fit == 0 && !ring_aim && st.saw && look {
        let (pref, coin) = side_pref(i, v);
        let my_team_bit = st.my_team_bit;
        let widths = widths();
        let read = |o: &crate::spatial::SortedUnit, o_vel: Vec2| -> Option<(f32, f32, f32)> {
            if o.idx as usize == i
                || (o.meta & crate::spatial::META_TEAM) != my_team_bit
                || (o.meta & crate::spatial::META_DYING) != 0
            {
                return None;
            }
            let to = o.xz() - p;
            let along = to.dot(wd);
            if along <= OPEN_AHEAD || to.length_squared() >= OPEN_FAR * OPEN_FAR {
                return None;
            }
            let holding = orders[crate::spatial::meta_group(o.meta)].is_none();
            if !holding && o_vel.dot(wd) > QUEUE_SPEED {
                return None;
            }
            let lat = wd.x * to.y - wd.y * to.x;
            Some((along, lat, 0.5 * widths[crate::spatial::meta_pose(o.meta)].body))
        };
        // One pass over the grid: the comrades in his sight into a buffer,
        // then the nearest and the row from it. Two passes cost 3.0 ms of
        // the 200k step; the buffer overflowing (a press denser than
        // WIDE_SEEN men in 4 m) falls back to the second pass.
        let mut wide_at = f32::INFINITY;
        let mut seen: [(f32, f32, f32); WIDE_SEEN] = [(0.0, 0.0, 0.0); WIDE_SEEN];
        let mut n_seen = 0usize;
        let mut overflow = false;
        grid.for_each_candidate_vel(p, OPEN_FAR, |o, o_vel| {
            if let Some(m) = read(o, o_vel) {
                wide_at = wide_at.min(m.0);
                if n_seen < WIDE_SEEN {
                    seen[n_seen] = m;
                    n_seen += 1;
                } else {
                    overflow = true;
                }
            }
        });
        let mut wide = Row::default();
        if overflow {
            grid.for_each_candidate_vel(p, OPEN_FAR, |o, o_vel| {
                if let Some((along, lat, half)) = read(o, o_vel)
                    && along < wide_at + OPEN_BAND
                {
                    wide.add(lat, half, along);
                }
            });
        } else {
            for &(along, lat, half) in &seen[..n_seen] {
                if along < wide_at + OPEN_BAND {
                    wide.add(lat, half, along);
                }
            }
        }
        let (w_lat, w_along, w_fit, _) = openings(&wide, wide_at, OPEN_FAR, st.body0, SQUEEZE_DEPTH, true, pref, coin);
        *s.aim = if w_fit > 0 { p + wd * w_along + perp * w_lat } else { Vec2::NAN };
        v = (v & !STUCK_WIDE_DEPTH) | ((w_fit == 1) as u16 * STUCK_WIDE_DEPTH);
    }
    // A near fit wins; else the wide aim while it is still ahead of him,
    // or a ring aim within RING_SEE whichever way it lies, until he is on
    // it.
    let mut ring_kept = false;
    if fit == 0 && s.aim.x.is_finite() {
        let rel = *s.aim - p;
        let r_along = rel.dot(wd);
        let d2 = rel.length_squared();
        let near_ring = (ring_flag || ring_aim) && d2 < RING_SEE * RING_SEE && d2 > RING_ON * RING_ON;
        if r_along > OPEN_AHEAD || near_ring {
            aim_lat = rel.dot(perp);
            aim_along = r_along;
            fit = if v & STUCK_WIDE_DEPTH != 0 { 1 } else { 2 };
            ring_kept = r_along <= OPEN_AHEAD;
            if near_ring {
                *s.sight |= SIGHT_RING;
            }
        } else {
            *s.aim = Vec2::NAN;
        }
    }
    st.aim_fit = fit;
    st.gap = fit == 1 && !taken;
    st.gap_near = st.gap && aim_along < st.sep0;
    if fit > 0 {
        // The side he chose, remembered as the try remembers its side.
        if aim_lat.abs() > 0.1 {
            v = (v & !(STUCK_MEM | STUCK_LEFT)) | (SIDE_MEMORY << STUCK_MEM_SHIFT) | ((aim_lat > 0.0) as u16 * STUCK_LEFT);
        }
        // His drive turned toward the aim, at its pace; a ring aim beside
        // or behind him is driven at straight.
        let to = if ring_kept { wd * aim_along + perp * aim_lat } else { wd * aim_along.max(OPEN_MIN_AHEAD) + perp * aim_lat };
        let len = to.length();
        if len > 1e-6 {
            st.desired = to * (want / len);
        }
    }
    *s.stuck = v;
}

/// What he remembers and what he sees of his comrades: the enemy he
/// remembers if still in sight, the way he is going, and every eighth tick
/// a look at the men around him (a comrade in his way, to his left or
/// right, ahead, between him and his mark, or running to the fight; with
/// an enemy in reach, a comrade to either side of his way to him and a
/// man close behind him), remembered in the sight bits until the next
/// look.
#[inline]
fn look_around(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { grid, pos_prev, team, group, orders, engaged, focus, tick, .. } = *f;
    let i = s.i;
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    let in_melee = st.in_melee;
    let committed = st.committed;
    let join_fp = st.join_fp;
    let desired = st.desired;
    let prev_target = st.prev_target;
    let best_idx = st.best_idx;
    let scan_r = st.scan_r;
    let my_team_bit = st.my_team_bit;
    // The enemy he remembers, if still in sight: he tracks him while he has
    // no enemy in reach, and seeing him is what makes a man still in
    // formation react. A man fighting someone in reach out of formation needs
    // neither, and skips the lookup.
    // With his regiment focused on one target (sim/job.rs `focus`), a man
    // of another regiment is remembered only within his threat distance:
    // one who is not on him he lets go, and follows his orders.
    let memo = prev_target as usize;
    let mut memo_d2 = f32::MAX;
    let memo_valid = ((best_idx == u32::MAX && !dying && !routed)
        || (in_melee && !committed))
        && memo < pos_prev.len()
        && team[memo] != team[i]
        && {
            let d2 = pos_prev[memo].xz().distance_squared(p);
            memo_d2 = d2;
            d2 < (seek_radius() + 1.0) * (seek_radius() + 1.0)
                && focus[gi].is_none_or(|t| {
                    group[memo] == t || d2 < threat_radius() * threat_radius()
                })
        };
    // Which way he is going: to that enemy, else to his regiment's fight.
    let memo_dir = if memo_valid {
        (pos_prev[memo].xz() - p).normalize_or_zero()
    } else if let Some(fp) = join_fp {
        (fp - p).normalize_or_zero()
    } else {
        Vec2::ZERO
    };
    // The blocking test toward a holding man's own mark.
    let slot_dir = if orders[gi].is_none() && engaged[gi] && !routed {
        desired.normalize_or_zero()
    } else {
        Vec2::ZERO
    };
    // With an enemy in reach his way is to that enemy, and he looks at the
    // room he has to fight him.
    let fighter = best_idx != u32::MAX && !dying && !routed;
    let to_enemy = if fighter { pos_prev[best_idx as usize].xz() - p } else { Vec2::ZERO };
    let fight_dir = to_enemy.normalize_or_zero();
    // Open lanes to either side of his way.
    let side_dir = if fighter {
        Vec2::new(-fight_dir.y, fight_dir.x)
    } else {
        Vec2::new(-memo_dir.y, memo_dir.x)
    };
    // His look at the comrades around him on his way (to his enemy, to the
    // fight, or back to his mark), in his own rhythm: a separate pass over
    // the neighbors of the separation scan, run only by men who are going
    // somewhere. He looks for a lane only while he has no enemy in reach, and
    // watches his comrades go only while he sees no enemy of his own.
    let look_lanes = memo_dir != Vec2::ZERO && best_idx == u32::MAX && !dying && !routed;
    let watch_go = in_melee && !committed && !memo_valid;
    let slot_on = slot_dir != Vec2::ZERO;
    if (i as u32).wrapping_add(tick).is_multiple_of(LOOK_TICKS) {
        let mut seen = 0u16;
        if look_lanes || watch_go || slot_on {
            let (mut way, mut left, mut right, mut ahead_any, mut mark, mut go) =
                (false, false, false, false, false, false);
            // Branch-free: neighbors of both teams interleave in a melee, and
            // branching on each one mispredicts.
            grid.for_each_candidate_vel(p, scan_r, |o, ov| {
                let comrade = (o.idx as usize != i)
                    & ((o.meta & crate::spatial::META_TEAM) == my_team_bit);
                let d = p - o.xz();
                let d2 = d.length_squared();
                let reach = 0.707 * d2.sqrt();
                let near = d2 < SEP_RADIUS * SEP_RADIUS;
                let going = ov.dot(memo_dir);
                // A comrade ahead: he walks up to him instead of jogging, and
                // at arm's length the man blocks his way. One already walking
                // away the same way is neither: men heading for the same
                // fight move together instead of each waiting for the next.
                let away = going > 0.5;
                let ahead = comrade & look_lanes & ((-d).dot(memo_dir) > reach) & !away;
                ahead_any |= ahead;
                way |= ahead & near;
                // A comrade of his own regiment close by, running to the
                // fight: the sight that makes him follow.
                go |= comrade
                    & watch_go
                    & (crate::spatial::meta_group(o.meta) == gi)
                    & (going > GOING_SPEED);
                let lateral = (-d).dot(side_dir);
                let beside = comrade & look_lanes & near;
                left |= beside & (lateral > reach);
                right |= beside & (lateral < -reach);
                mark |= comrade & slot_on & near & ((-d).dot(slot_dir) > reach);
            });
            seen = (way as u16 * SIGHT_WAY)
                | (left as u16 * SIGHT_LEFT)
                | (right as u16 * SIGHT_RIGHT)
                | (ahead_any as u16 * SIGHT_AHEAD)
                | (mark as u16 * SIGHT_MARK)
                | (go as u16 * SIGHT_GO);
        }
        if fighter {
            // The room he has to fight the enemy in reach: a comrade close
            // to either side of his way to the enemy, and any man close
            // behind him.
            let (mut left, mut right, mut back) = (false, false, false);
            grid.for_each_candidate(p, scan_r, |o| {
                let other = o.idx as usize != i;
                let comrade = other & ((o.meta & crate::spatial::META_TEAM) == my_team_bit);
                let to = o.xz() - p;
                let d2 = to.length_squared();
                let len = d2.sqrt();
                let quarter = 0.707 * len;
                let near = d2 < SEP_RADIUS * SEP_RADIUS;
                let along = to.dot(fight_dir);
                let lateral = to.dot(side_dir);
                left |= comrade & near & (lateral > quarter);
                right |= comrade & near & (lateral < -quarter);
                back |= other & near & (along < -quarter);
            });
            seen |= (left as u16 * SIGHT_LEFT)
                | (right as u16 * SIGHT_RIGHT)
                | (back as u16 * SIGHT_BACK);
        }
        *s.sight = (*s.sight
            & (SIGHT_GO_FAR | SIGHT_FOE | SIGHT_KNIFE_COUNT | SIGHT_KNIFE))
            | seen;
    }
    let way_blocked = *s.sight & SIGHT_WAY != 0;
    let left_blocked = *s.sight & SIGHT_LEFT != 0;
    let right_blocked = *s.sight & SIGHT_RIGHT != 0;
    let comrade_ahead = *s.sight & SIGHT_AHEAD != 0;
    let slot_blocked = *s.sight & SIGHT_MARK != 0;
    st.back_blocked = *s.sight & SIGHT_BACK != 0;
    st.memo_valid = memo_valid;
    st.memo_d2 = memo_d2;
    st.memo_dir = memo_dir;
    st.side_dir = side_dir;
    st.watch_go = watch_go;
    st.way_blocked = way_blocked;
    st.left_blocked = left_blocked;
    st.right_blocked = right_blocked;
    st.comrade_ahead = comrade_ahead;
    st.slot_blocked = slot_blocked;
}

/// Stuck ticks before a man tries a side (half a second); how long a try
/// lasts at most (a second) and at least (a real step, not a twitch);
/// how long he remembers the side he tried, in eighths of a second (four
/// seconds).
const STUCK_TICKS: u16 = 15;
const TRY_TICKS: u16 = 30;
const TRY_MIN: u16 = 6;
const SIDE_MEMORY: u16 = 15;
/// The crowd ahead of him (`Step::ahead`) that counts as someone in his
/// way, and the sideways lean of his separation push that picks his
/// side over his memory and his coin.
const AHEAD_MIN: f32 = 0.1;
const LEAN_MIN: f32 = 0.05;
/// A holding man this far from his slot is walking somewhere too, and
/// tries a side when stuck: a man left inside a friend that closed, or
/// carried off by a friend walking through, works his way out instead of
/// standing against the gripping men for good.
const TRY_FROM_SLOT: f32 = 5.0;
/// The `stuck` column: the stuck ticks (bits 0 to 3), the try's ticks
/// left (4 to 8), the side memory in eighths of a second (9 to 12), the
/// remembered side (13, set for his left), squeezing (14; then bits 0 to
/// 8 count his ticks in the squeeze instead, since a squeezing man is
/// neither counting nor trying), and whether the opening of his last wide
/// look fits only his depth (15, `choose_opening`).
const STUCK_COUNT: u16 = 0x000f;
const STUCK_TRY_SHIFT: u16 = 4;
const STUCK_TRY: u16 = 0x1f << STUCK_TRY_SHIFT;
const STUCK_MEM_SHIFT: u16 = 9;
const STUCK_MEM: u16 = 0x0f << STUCK_MEM_SHIFT;
pub(crate) const STUCK_LEFT: u16 = 1 << 13;
const STUCK_SQUEEZE: u16 = 1 << 14;
const STUCK_SQUEEZE_TICKS: u16 = 0x01ff;
const STUCK_WIDE_DEPTH: u16 = 1 << 15;
/// The stuck ticks and the try's ticks left packed in `stuck`.
pub(crate) fn stuck_count(v: u16) -> u16 {
    if squeezing(v) { 0 } else { v & STUCK_COUNT }
}
pub(crate) fn stuck_trying(v: u16) -> u16 {
    if squeezing(v) { 0 } else { (v & STUCK_TRY) >> STUCK_TRY_SHIFT }
}
/// Whether he is squeezing through a crack, and for how many ticks.
pub(crate) fn squeezing(v: u16) -> bool {
    v & STUCK_SQUEEZE != 0
}
fn squeeze_ticks(v: u16) -> u16 {
    if squeezing(v) { v & STUCK_SQUEEZE_TICKS } else { 0 }
}

/// The squeeze (docs/plans/032): a man going through a crack narrower
/// than he is wide turns his shoulders and is SQUEEZE_DEPTH across
/// (`turn_level`). It lasts from the crack's mouth until he is through:
/// it ends once he has no body within his normal width at each side
/// (less BETWEEN_SLACK, within a metre along his way) and either he aims
/// at no crack or he moves freely at nine tenths of the advance pace with
/// no crack within his spring, after TRY_MIN ticks, or after
/// SQUEEZE_STALL ticks with no progress, whatever is beside him: a man
/// wedged in a press turns back square (toy devlog 015). While
/// it lasts his legs push along his wish with their full effort, not
/// faded by the crowd's brake, and his speed is capped at SQUEEZE_CAP,
/// a walk.
const BETWEEN_SLACK: f32 = 0.05;
const SQUEEZE_STALL: u16 = 60;
const SQUEEZE_CAP: f32 = 2.0;
/// The choice of opening (docs/plans/033): a man going somewhere, with no
/// enemy in his reach and his regiment not fighting (the try's own test),
/// reads the comrades ahead of him within OPEN_SEE (the scan's same-team
/// reach) and aims at an opening between them that fits him, before he
/// reaches them. A comrade of a regiment under an order who moves his way
/// faster than QUEUE_SPEED is a queue he follows, not an obstacle; a
/// holding man is an obstacle whatever a shove does to him, and a man
/// squeezing who has stopped is a body like any other. Of the obstacles he takes the row nearest him (the men within
/// OPEN_BAND along of the nearest one, a rank's pitch and a bit, so the
/// two files of a pair offset by a rank read as one row) and in it the
/// nearest two on each side of his way. The air between neighbouring men
/// of the row fits his width (his normal body, less OPEN_SLACK), else his
/// depth (the squeeze), else nothing. For each fitting opening his centre
/// goes to its point nearest his way; the nearest wins, a squeeze counting
/// as SQUEEZE_DETOUR metres of detour, the side he chose last worth
/// OPEN_SIDE_KEEP, his own coin breaking a tie. His drive keeps its pace
/// and turns toward that point, OPEN_MIN_AHEAD ahead at least. The near
/// read takes only openings between two men it has seen. When nothing
/// near fits and he saw an obstacle, he looks out to OPEN_FAR once in
/// OPEN_LOOK_TICKS, in his own rhythm, where the open ground beyond the
/// last man on each side counts too, and keeps what he found (`aim`)
/// until his next look, until it is behind him or until his near read
/// fits. A man he follows still takes up his space: a crack that fits
/// only his depth with any comrade inside its passage is taken, and he
/// does not turn his shoulders for it, though he may aim at it, which is
/// following (toy devlog 015). Measured in the toy first (toy devlogs 012,
/// 013 and 015).
const OPEN_SEE: f32 = SEP_RADIUS;
const OPEN_FAR: f32 = 4.0;
const OPEN_BAND: f32 = 1.5;
const OPEN_AHEAD: f32 = 0.05;
const QUEUE_SPEED: f32 = 0.3;
const SQUEEZE_DETOUR: f32 = 1.0;
const OPEN_LOOK_TICKS: u32 = 8;
/// The comrades the wide look keeps from its one pass over the grid.
const WIDE_SEEN: usize = 96;
const OPEN_SLACK: f32 = 0.02;
const OPEN_SIDE_KEEP: f32 = 0.2;
const OPEN_MIN_AHEAD: f32 = 0.3;

fn opening_on() -> bool {
    static O: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *O.get_or_init(|| crate::util::env_or("FL_OPENING", 1u32) != 0)
}

/// The row nearest a man: the nearest two obstacles on each side of his
/// way by their offset across it (left positive), each with his half
/// body and his far side along the way (his offset along it plus his half
/// body).
#[derive(Default)]
struct Row {
    left: [(f32, f32, f32); 2],
    right: [(f32, f32, f32); 2],
    n_left: u8,
    n_right: u8,
}
impl Row {
    #[inline]
    fn add(&mut self, lat: f32, half: f32, along: f32) {
        let (side, n, key) = if lat >= 0.0 {
            (&mut self.left, &mut self.n_left, lat)
        } else {
            (&mut self.right, &mut self.n_right, -lat)
        };
        let man = (key, half, along + half);
        if *n == 0 || key < side[0].0 {
            side[1] = side[0];
            side[0] = man;
            *n = (*n + 1).min(2);
        } else if *n == 1 || key < side[1].0 {
            side[1] = man;
            *n = 2;
        }
    }
}

/// The openings a man sees in a row at `row_at` along his way, out to
/// `see`: ahead between the nearest man on each side, left between the
/// two nearest on his left, right between the two on his right, and, when
/// `beyond` (the wide look), the open ground past the last man seen on a
/// side up to the edge of his sight at that distance. Returns the aim
/// across and along his way, its fit (0 none, 1 his depth, 2 his width),
/// and the picked opening's air across his way with the far side of its
/// passage along it (the farther of its two men's far sides).
#[allow(clippy::too_many_arguments)]
fn openings(
    row: &Row,
    row_at: f32,
    see: f32,
    body: f32,
    depth: f32,
    beyond: bool,
    pref: f32,
    coin: f32,
) -> (f32, f32, u8, (f32, f32, f32)) {
    let none = (f32::NAN, f32::NAN, f32::NAN);
    if row.n_left == 0 && row.n_right == 0 {
        return (0.0, 0.0, 0, none);
    }
    let edge = (see * see - row_at.min(see) * row_at.min(see)).max(0.0).sqrt();
    let far = if beyond { edge } else { f32::NAN };
    let (l1, l1r, l1f) = if row.n_left >= 1 { row.left[0] } else { none };
    let (r1, r1r, r1f) = if row.n_right >= 1 { row.right[0] } else { none };
    let (hi_l, l2f) = if row.n_left >= 2 { (row.left[1].0 - row.left[1].1, row.left[1].2) } else { (far, f32::NAN) };
    let (lo_r, r2f) = if row.n_right >= 2 { (-(row.right[1].0 - row.right[1].1), row.right[1].2) } else { (-far, f32::NAN) };
    // f32::max takes the number over a NaN: a side with one man seen has
    // his far side.
    let ahead = (
        if row.n_right >= 1 { -(r1 - r1r) } else { -far },
        if row.n_left >= 1 { l1 - l1r } else { far },
        true,
        l1f.max(r1f),
    );
    let left = (l1 + l1r, hi_l, row.n_left >= 1, l1f.max(l2f));
    let right = (lo_r, -(r1 + r1r), row.n_right >= 1, r1f.max(r2f));
    let sign = |x: f32| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 };
    let (mut best, mut aim_lat, mut fit, mut span) = (f32::INFINITY, 0.0, 0u8, none);
    for (lo, hi, ok, pass) in [ahead, left, right] {
        if !ok {
            continue;
        }
        // NaN air (an edge the near read does not see) fits nothing.
        let air = hi - lo;
        let fw = air >= body - OPEN_SLACK;
        let fd = !fw && air >= depth - OPEN_SLACK;
        if !(fw || fd) {
            continue;
        }
        let r = 0.5 * if fw { body } else { depth };
        let aim = (lo + r).max(0.0).min(hi - r);
        let aim = if aim.is_finite() { aim } else { 0.0 };
        let cost = aim.abs() + if fd { SQUEEZE_DETOUR } else { 0.0 }
            - if pref != 0.0 && sign(aim) == pref { OPEN_SIDE_KEEP } else { 0.0 }
            - if sign(aim) == coin { 0.01 } else { 0.0 };
        if cost < best {
            best = cost;
            aim_lat = aim;
            fit = if fw { 2 } else { 1 };
            span = (lo, hi, pass);
        }
    }
    (aim_lat, if fit > 0 { row_at.min(see) } else { 0.0 }, fit, span)
}

/// A man going somewhere who finds himself stuck tries a side. Stuck is
/// felt, not seen: he wants to walk and has moved at under half his wish,
/// and slower than a step, for STUCK_TICKS with bodies pressed against
/// him ahead (the net push would not do: in a press the men behind cancel
/// the men in front). He
/// tries the side his separation push leans to, the side with more room;
/// when it leans nowhere, the side he tried within the last SIDE_MEMORY,
/// else his own coin: a sidestep at STEP_PACE on top of whatever the
/// crowd leaves of his drive, not faded by the crowd's brake, so a man
/// packed in still shuffles sideways. The step ends as soon as the way
/// ahead of him is clear, after TRY_MIN ticks, and at TRY_TICKS at the
/// latest: he steps into the gap and walks, he does not step through it
/// into the next man (that was the pinball between the columns). Pressed
/// forward and free to slide sideways, he slides into the gap beside the
/// man in his way, and the man behind him does the same a moment later.
/// The blind sidestep is only for a man with no opening to aim at
/// (`choose_opening`): reaching a crack he aims at that fits only his
/// depth, he turns his shoulders at once and squeezes through
/// (`turn_level`). Only a man walking somewhere, under his regiment's
/// order or back to a slot more than TRY_FROM_SLOT off, with no enemy in
/// mind and his regiment not fighting: men walking to a fight and the
/// rear ranks of a fighting regiment keep their own rules (`close_in`),
/// so the melee is as it was, and a squeeze ends when the fight reaches
/// him. His state rides the `stuck` column (STUCK_*).
#[inline]
fn try_a_side(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { orders, engaged, anchors, home, tick, .. } = *f;
    let i = s.i;
    let wish = st.wish;
    let want = wish.length();
    let walking = orders[st.gi].is_some()
        || (anchors[st.gi] + home[i]).distance_squared(st.p) > TRY_FROM_SLOT * TRY_FROM_SLOT;
    let going = walking
        && !engaged[st.gi]
        && !st.dying
        && !st.routed
        && !st.in_melee
        && !st.committed
        && st.best_idx == u32::MAX
        && !st.memo_valid
        && st.join_fp.is_none()
        && want > 0.5;
    if !going {
        *s.stuck = 0;
        turn_shoulders(s);
        return;
    }
    let dir = wish / want;
    let left_dir = Vec2::new(-dir.y, dir.x);
    let v = *s.stuck;
    let mut squeeze = squeeze_ticks(v);
    let mut count = stuck_count(v);
    let mut trying = stuck_trying(v);
    let mut mem = (v & STUCK_MEM) >> STUCK_MEM_SHIFT;
    let mut left = v & STUCK_LEFT != 0;
    let v_along = s.vel.xz().dot(dir);
    // The squeeze (SQUEEZE_STALL above): it ends once no body is at each
    // of his sides and either no crack is ahead or he walks freely, or
    // once he has made no progress for SQUEEZE_STALL ticks, whatever is
    // beside him: a man wedged in a press turns back square.
    if squeeze > 0 {
        let free = v_along >= 0.9 * ADVANCE_PACE && !st.gap_near;
        let stalled = squeeze >= SQUEEZE_STALL && v_along < 0.1;
        let through = !st.between && squeeze >= TRY_MIN && (!st.gap || free);
        squeeze = if through || stalled { 0 } else { (squeeze + 1).min(STUCK_SQUEEZE_TICKS) };
    }
    if squeeze > 0 {
        // His legs at their full effort along his wish: he pushes with
        // all he has, at a walk at most (`steer`).
        st.desired = wish;
    } else if trying > 0 {
        trying -= 1;
        // The way ahead clear after the first TRY_MIN ticks: the step is
        // done, he walks.
        if trying <= TRY_TICKS - TRY_MIN && st.ahead < AHEAD_MIN {
            trying = 0;
        }
        let side = if left { left_dir } else { -left_dir };
        st.desired += side * STEP_PACE;
        mem = SIDE_MEMORY;
    } else {
        // Stuck: pressed, and not even stepping. Under half his wish
        // alone, a lane walker at the sprint behind a slower man counted
        // stuck at a jog and zigzagged down the lane. A crack he aims at
        // that fits only his depth presses him too.
        let pressed = st.ahead > AHEAD_MIN || st.gap;
        let stuck = pressed && v_along < (0.5 * want).min(STEP_PACE);
        count = if stuck { (count + 1).min(STUCK_TICKS) } else { count.saturating_sub(1) };
        // He turns his shoulders as he reaches the crack he aims at, no
        // stuck wait; the blind sidestep only with no opening to aim at.
        // A straggler of a holding regiment whose aim has not moved him
        // (stuck at its top, a ring or wide aim into a standing man's
        // spring) drops it and tries a side; the ring reads again after
        // the step. Passers keep their aims: a lane queue stands still too.
        let dead_aim = st.aim_fit != 0
            && orders[st.gi].is_none()
            && (anchors[st.gi] + home[i]).distance_squared(st.p) > TRY_FROM_SLOT * TRY_FROM_SLOT;
        let start = count >= STUCK_TICKS && !st.gap_near && (st.aim_fit == 0 || dead_aim);
        if start {
            if dead_aim {
                *s.aim = Vec2::NAN;
                *s.sight &= !SIGHT_RING;
            }
            let lean = st.push.dot(left_dir);
            left = if lean.abs() > LEAN_MIN {
                lean > 0.0
            } else if mem > 0 {
                left
            } else {
                crate::units::hash01((i as u32).wrapping_mul(0x2545).wrapping_add(tick)) > 0.5
            };
            trying = TRY_TICKS;
            count = 0;
            mem = SIDE_MEMORY;
        } else {
            if st.gap_near {
                squeeze = 1;
                count = 0;
            }
            if mem > 0 && tick.is_multiple_of(8) {
                mem -= 1;
            }
        }
    }
    let low = if squeeze > 0 { STUCK_SQUEEZE | squeeze } else { count | (trying << STUCK_TRY_SHIFT) };
    *s.stuck = low | (mem << STUCK_MEM_SHIFT) | (left as u16 * STUCK_LEFT) | (v & STUCK_WIDE_DEPTH);
    turn_shoulders(s);
}

/// The turn of his shoulders follows his squeeze a step a tick
/// (`turn_level`): toward turned while he squeezes, back toward square
/// once he does not.
#[inline]
fn turn_shoulders(s: &mut Soldier) {
    let turn = *s.pose >> POSE_TURN_SHIFT;
    let turn = if squeezing(*s.stuck) { (turn + 1).min(POSE_READY) } else { turn.saturating_sub(1) };
    *s.pose = (*s.pose & POSE_STANCE) | (turn << POSE_TURN_SHIFT);
}

/// Two things a man in melee does not do for his slot: step back away
/// from his regiment's fight to dress, or press into the comrade between
/// him and his mark.
#[inline]
fn hold_the_frame(f: &Field, st: &mut Step) {
    let Field { contact, form_face, .. } = *f;
    let gi = st.gi;
    let routed = st.routed;
    let slot_blocked = st.slot_blocked;
    let mut desired = st.desired;
    // A regiment fighting on its contact frame never steps backward, away
    // from its fight, to dress its ranks: a man ahead of his mark (bunched up
    // behind a stopped front, or shoved forward) stays; he only steps forward
    // or sideways to it. Without this a charging block walked backward to
    // reopen its ranks the moment it made contact. A regiment struck from
    // behind has its fight at its back, so it keeps stepping back to hold its
    // ground.
    if contact[gi] && !routed && form_face[gi] != Vec2::ZERO {
        let back = desired.dot(form_face[gi]);
        if back < 0.0 {
            desired -= form_face[gi] * back;
        }
    }
    // In melee, holding men step back to their marks only through open
    // ground: shoved off his mark with a comrade between him and it, a man
    // stands where he is until the way clears (the comrade dies, steps up or
    // moves back to his own mark) instead of pressing into the man's back.
    if slot_blocked {
        desired = Vec2::ZERO;
    }
    st.desired = desired;
}

/// The far look, every eighth tick: a pressing man with nobody in reach
/// and open ground around him remembers the nearest enemy within his sight
/// (15 m for a fighting regiment, 4 m on the approach; with his regiment
/// focused on one target, its men, and any man within his threat distance
/// coming at him), and a man still in formation notices a comrade of his
/// regiment running to the fight within 6 m.
#[inline]
fn acquire(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { grid, press, engaged, focus, tick_seed, .. } = *f;
    let i = s.i;
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    let best_idx = st.best_idx;
    let crowd = st.crowd;
    let watch_go = st.watch_go;
    let memo_dir = st.memo_dir;
    let my_team_bit = st.my_team_bit;
    // Sparse-fight acquisition (see WIDE_ACQUIRE_R): a pressing unit with an
    // empty scan and open space around it memoizes a farther enemy in
    // `target` so the closing drive below can restore contact. Gated hard
    // (press + no near enemy + low crowd + 1/8 cadence) to stay off the 200k
    // hot path.
    let acquire_crowd_lim = CROWD_SLOW;
    // A man of a fighting regiment looks as far as he can see (seek_radius)
    // on his far look; on the approach the old short scan stands.
    let far_look = (i as u32).wrapping_add(tick_seed).is_multiple_of(8);
    if far_look {
        *s.sight &= !SIGHT_GO_FAR;
    }
    if best_idx == u32::MAX
        && !dying
        && !routed
        && press[gi]
        && crowd < acquire_crowd_lim
        && far_look
    {
        let look = if engaged[gi] { seek_radius() } else { WIDE_ACQUIRE_R };
        let mut far_d2 = look * look;
        if let Some(t) = focus[gi] {
            // His regiment fights one ordered target (sim/job.rs `focus`):
            // he looks for its men anywhere in his sight, and for any
            // enemy within his threat distance who is coming at him,
            // closing faster than GOING_SPEED. A man of another regiment
            // standing off or going elsewhere is not his business. The
            // same cells as the look below, with each man's velocity.
            let threat2 = threat_radius() * threat_radius();
            grid.for_each_candidate_vel(p, look, |o, ov| {
                let enemy = (o.meta & crate::spatial::META_TEAM) != my_team_bit
                    && (o.meta & crate::spatial::META_DYING) == 0;
                if enemy {
                    let d = p - o.xz();
                    let d2 = d.length_squared();
                    let closing = ov.dot(d);
                    let counts = crate::spatial::meta_group(o.meta) as u32 == t
                        || (d2 < threat2
                            && closing > 0.0
                            && closing * closing > GOING_SPEED * GOING_SPEED * d2);
                    if counts && d2 < far_d2 {
                        far_d2 = d2;
                        *s.target = o.idx;
                    }
                }
            });
        } else {
            grid.for_each_candidate(p, look, |o| {
                let enemy = (o.meta & crate::spatial::META_TEAM) != my_team_bit
                    && (o.meta & crate::spatial::META_DYING) == 0;
                if enemy {
                    let d2 = (p - o.xz()).length_squared();
                    if d2 < far_d2 {
                        far_d2 = d2;
                        *s.target = o.idx;
                    }
                }
            });
        }
        // Further than the neighbor scan, within JOIN_SEE_R: a comrade of his
        // own regiment running to the fight.
        if watch_go
            && grid.any_candidate_vel(p, look.min(JOIN_SEE_R), |o, ov| {
                o.idx as usize != i
                    && crate::spatial::meta_group(o.meta) == gi
                    && (p - o.xz()).length_squared() < JOIN_SEE_R * JOIN_SEE_R
                    && ov.dot(memo_dir) > GOING_SPEED
            })
        {
            *s.sight |= SIGHT_GO_FAR;
        }
    }
}

/// Leaving formation to fight, in his own time: a chance per half-second
/// window while he sees an enemy or a comrade going, and patience as the
/// backstop. Once out he never dresses on his slot while he sees an enemy
/// or his regiment's melee runs.
#[inline]
fn decide_to_join(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { melee_ticks, engaged, tick, .. } = *f;
    let i = s.i;
    let in_melee = st.in_melee;
    let may_leave = st.may_leave;
    let gi = st.gi;
    let memo_valid = st.memo_valid;
    let memo_d2 = st.memo_d2;
    let best_idx = st.best_idx;
    let mut desired = st.desired;
    let committed = st.committed;
    let mut committed = committed;
    // Out of formation with his regiment's melee over, or never begun: he
    // comes back to his slot once he sees no enemy, nobody in reach and no
    // remembered enemy within his closing range (close_in). While his
    // regiment's melee runs he stays out until it ends, heading for the
    // fight when he sees nobody.
    if committed && !in_melee {
        let look = if engaged[gi] { seek_radius() } else { WIDE_ACQUIRE_R };
        let sees = best_idx != u32::MAX
            || (memo_valid && memo_d2 < (look + 1.0) * (look + 1.0));
        if !sees {
            committed = false;
            *s.out_form = false;
        }
    }
    // Leaving formation to fight, in his own time: he reacts to an enemy in
    // sight or a comrade seen running to the fight (a chance per half-second
    // window), or his patience with the fight's noise runs out. Near men go
    // first and the line rolls up outward. With no melee on his regiment's
    // clock (a lone enemy at the flank, a thin contact) he goes only for an
    // enemy within the short look (WIDE_ACQUIRE_R): the men nearest him
    // leave the ranks for him, the rest stay in formation.
    if may_leave && !committed {
        let patience = (JOIN_PATIENCE_MIN
            + (join_patience_max() - JOIN_PATIENCE_MIN)
                * crate::units::hash01((i as u32).wrapping_mul(0x3C6E) ^ 0xF372))
            * 30.0;
        let saw_comrade_go = *s.sight & (SIGHT_GO | SIGHT_GO_FAR) != 0;
        let sees = if in_melee {
            memo_valid || saw_comrade_go
        } else {
            memo_valid && memo_d2 < WIDE_ACQUIRE_R * WIDE_ACQUIRE_R
        };
        let react = sees && {
            let window = (tick.wrapping_add((i as u32).wrapping_mul(11)) / JOIN_WINDOW)
                .wrapping_mul(0x9E37_79B1);
            crate::units::hash01(window ^ (i as u32).wrapping_mul(0x6A09))
                < join_react_chance()
        };
        if react || (in_melee && melee_ticks[gi] as f32 > patience) {
            committed = true;
            *s.out_form = true;
        }
    }
    // Out of formation he never dresses on his slot: he fights, goes for an
    // enemy he sees, heads for the fight, or stands where he is when
    // blocked. Inferring this from his speed made a man who slowed down run
    // back to his slot and come out again, over and over.
    if committed {
        desired = Vec2::ZERO;
    }
    st.committed = committed;
    st.desired = desired;
}

/// The swing state machine: an archer's draw and loose, a wind-up and
/// its strike (a damage event, resolved by the serial apply), the recovery,
/// and a ready man's choice of target in his forward half-plane.
#[inline]
fn swing(f: &Field, s: &mut Soldier, st: &mut Step, out: &mut ChunkOut) {
    let Field { terrain, pos_prev, speed, team, group, broken, shoot_at, target_members, blocks, tick_seed, .. } = *f;
    let i = s.i;
    let params = &TYPES[st.my_kind];
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    let sticky = st.sticky;
    let prev_target = st.prev_target;
    let best_idx = st.best_idx;
    let my_kind = st.my_kind;
    let mut desired = st.desired;
    // Swing state machine. All writes are to this unit's own row; damage goes
    // through the chunk event buffer.
    let mut face_target = None;
    if !dying {
        match *s.swing & crate::units::SWING_STATE_MASK {
            crate::units::SWING_WINDUP
                if *s.swing & crate::units::SWING_RANGED != 0 =>
            {
                // Drawing the bow: feet planted, eyes on the target block.
                // A man whose knife came out lets the draw go.
                desired = Vec2::ZERO;
                let knife = *s.sight & SIGHT_KNIFE != 0;
                if let Some(shot) = shoot_at[gi].as_ref().filter(|_| !knife) {
                    let q = *s.quarry as usize;
                    face_target =
                        Some(if q < pos_prev.len() { pos_prev[q].xz() } else { shot.c });
                    if *s.swing_t == 0 {
                        let h = |k: u32| {
                            crate::units::hash01(
                                tick_seed
                                    ^ (i as u32).wrapping_mul(k).wrapping_add(k),
                            )
                        };
                        // Aim: the man he drew at (every man has his own
                        // aim target) while he lives, else another man of
                        // the block; the footprint-disc spot stands in only
                        // if every member died this tick. Then the range-
                        // INDEPENDENT landing scatter and a lead for the
                        // block's drift over the flight.
                        let members = &target_members[shot.t];
                        let q = *s.quarry as usize;
                        let base = if q < pos_prev.len()
                            && team[q] != team[i]
                            && f.death_snap[q] == 0
                        {
                            pos_prev[q].xz()
                        } else if members.is_empty() {
                            let ang =
                                h(0x1F3B) * std::f32::consts::TAU;
                            let rad = shot.r * 0.85 * h(0x2E5D).sqrt();
                            shot.c + Vec2::new(ang.cos(), ang.sin()) * rad
                        } else {
                            let m = members[(h(0x1F3B)
                                * members.len() as f32)
                                as usize
                                % members.len()]
                                as usize;
                            pos_prev[m].xz()
                        };
                        let sigma = crate::unit_types::missile::AIM_SPREAD;
                        let mut aim = base
                            + Vec2::new(
                                (h(0x3B7F) + h(0x45A3) - 1.0) * sigma,
                                (h(0x5DC1) + h(0x6B8D) - 1.0) * sigma,
                            );
                        aim += shot.vel * (aim.distance(p) / 30.0);
                        // Launch above the body-hit band (arrows.rs): a
                        // shaft leaving at head height is inside its own
                        // shooter's hit cylinder at flight-time zero and
                        // kills him.
                        let over = crate::unit_types::half_height(my_kind)
                            + crate::arrows::LAUNCH_OVER_HEAD;
                        let from = Vec3::new(p.x, pos_prev[i].y + over, p.y);
                        let to = Vec3::new(
                            aim.x,
                            terrain.height_at(aim.x, aim.y)
                                + crate::arrows::AIM_HEIGHT * crate::unit_types::tallest(),
                            aim.y,
                        );
                        out.arrows.push(crate::arrows::ArrowSpawn {
                            pos: from,
                            vel: crate::arrows::solve_launch(
                                from,
                                to,
                                &blocks[team[i] as usize],
                                terrain,
                            ),
                            team: team[i],
                            group: group[i],
                        });
                        *s.ammo = s.ammo.saturating_sub(1);
                        *s.swing = crate::units::SWING_RECOVER;
                        // Reload: the M2TW volley cycle is animation-bound at
                        // ~10 s; the jitter keeps later volleys ragged.
                        *s.swing_t =
                            (crate::unit_types::missile::RELOAD_TICKS as f32
                                * (0.8 + 0.25 * h(0x77F1)))
                                .min(255.0)
                                as u8;
                    } else {
                        *s.swing_t -= 1;
                    }
                } else {
                    // Target gone mid-draw, or the knife out: ease off and
                    // reassess shortly.
                    *s.swing = crate::units::SWING_RECOVER;
                    *s.swing_t = crate::unit_types::missile::CANCEL_TICKS;
                }
            }
            crate::units::SWING_WINDUP => {
                // Feet planted while winding up — EXCEPT against a routing
                // target: the cut-down happens at a run, or the runner is 3 m
                // gone by the strike tick and every blow whiffs (the pursuit
                // treadmill).
                let t = *s.target as usize;
                let target_routed =
                    t < pos_prev.len() && broken[group[t] as usize];
                if !target_routed {
                    desired *= 0.25;
                }
                if t < pos_prev.len() {
                    face_target = Some(pos_prev[t].xz());
                }
                if *s.swing_t == 0 {
                    // Strike lands; validity (still alive, still in reach,
                    // still an enemy) is checked in the apply pass — a dodged
                    // or dead target is a whiff. A cut lands harder.
                    let cut = params.is_cut(*s.swing);
                    let mut jit =
                        0.85 + 0.3 * crate::units::hash01(tick_seed ^ (i as u32));
                    if cut {
                        jit *= params.cut_weight();
                    }
                    out.events.push(DamageEvent {
                        victim: *s.target,
                        attacker: i as u32,
                        jit,
                        charge: *s.swing & crate::units::SWING_CHARGE != 0,
                        impale: false,
                        warn: 0,
                        chain: 0,
                    });
                    *s.swing = crate::units::SWING_RECOVER;
                    let cjit = (1.0 - crate::unit_types::COOLDOWN_JITTER)
                        + 2.0
                            * crate::unit_types::COOLDOWN_JITTER
                            * crate::units::hash01(
                                tick_seed ^ (i as u32).wrapping_mul(0x9E37),
                            );
                    *s.swing_t = (params.cooldown_ticks as f32 * cjit) as u8;
                    if cut {
                        *s.swing_t = (*s.swing_t).max(params.cut_return_ticks);
                    }
                } else {
                    *s.swing_t -= 1;
                }
            }
            crate::units::SWING_DEFEND => {
                // Answering a swing he saw coming (sim/damage.rs): feet
                // planted, eyes on the attacker, the shield or the blade
                // up to meet the blow, then back on guard. The damage
                // apply set this and may end it early: a parry that
                // stopped the blow turns into the counter's wind-up.
                desired *= 0.25;
                let t = *s.target as usize;
                if t < pos_prev.len() {
                    face_target = Some(pos_prev[t].xz());
                }
                // A parry that stopped the blow counters: a stab at
                // the man he parried. In his kind's window he launches
                // it with a chance each tick by his skill, at the
                // window's end for sure.
                let stopped = *s.swing & crate::sim::damage::GUARD_STOPPED != 0;
                let elapsed = crate::sim::damage::PARRY_LEN.saturating_sub(*s.swing_t);
                let (from, by) = crate::sim::damage::COUNTER_WINDOW[st.my_kind];
                let launch = stopped
                    && elapsed >= from
                    && (elapsed >= by
                        || crate::units::hash01(tick_seed ^ (i as u32).wrapping_mul(0x7F4A_7C15))
                            < params.defence_skill * crate::sim::damage::counter_per_tick());
                if launch {
                    *s.swing = crate::units::SWING_WINDUP
                        | (*s.swing & crate::units::SWING_STAGGER_IMMUNE);
                    *s.swing_t = ((params.windup_ticks as f32
                        * crate::sim::damage::counter_windup())
                        .ceil() as u8)
                        .clamp(1, params.windup_ticks);
                    // The man he parried is warned like any target, one
                    // deeper in the exchange: he may answer from the
                    // bind, with less chance each time (sim/damage.rs).
                    out.events.push(DamageEvent {
                        victim: *s.target,
                        attacker: i as u32,
                        jit: 0.0,
                        charge: false,
                        impale: false,
                        warn: *s.swing_t,
                        chain: s.chain.saturating_add(1),
                    });
                } else if *s.swing_t == 0 {
                    *s.swing = crate::units::SWING_READY
                        | (*s.swing & crate::units::SWING_STAGGER_IMMUNE);
                } else {
                    *s.swing_t -= 1;
                }
            }
            crate::units::SWING_RECOVER => {
                if *s.swing_t == 0 {
                    // A stagger that just wore off leaves one free pass
                    // against the next one (anti-stunlock); a plain recovery
                    // carries an unspent pass forward.
                    let immune = if *s.swing
                        & crate::units::SWING_STAGGERED
                        != 0
                    {
                        crate::units::SWING_STAGGER_IMMUNE
                    } else {
                        *s.swing & crate::units::SWING_STAGGER_IMMUNE
                    };
                    *s.swing = crate::units::SWING_READY | immune;
                } else {
                    *s.swing_t -= 1;
                }
            }
            _ => {
                // Ready: pick a target from the scan. Stick with the previous
                // one when still in reach (duels), else nearest. Routing
                // units never start attacks (they still defend nothing —
                // pursuit is free hits).
                let chosen = if sticky { prev_target } else { best_idx };
                // No eyes in the back of his head: a man only opens on a
                // target in his forward half-plane. Being struck tells him
                // where to turn (the flash facing below), but it does NOT let
                // him swing backward over his shoulder — he attacks once he
                // has turned far enough, at the human turn-speed cap. Without
                // this gate every rear-approached victim counter-wound-up on
                // proximity and was face-on before the first blow landed; the
                // rear sector never fired in practice.
                let aware = chosen != u32::MAX && {
                    let t = chosen as usize;
                    t < pos_prev.len() && {
                        let to_t = pos_prev[t].xz() - p;
                        let fwd = Vec2::new(
                            s.yaw.sin(),
                            s.yaw.cos(),
                        );
                        fwd.dot(to_t) >= 0.0
                    }
                };
                if chosen != u32::MAX && !routed && aware {
                    *s.target = chosen;
                    // Arriving at speed = a charging blow: momentum converts
                    // to damage + a bigger lunge (render reads the flag).
                    let v2 = s.vel.xz().length_squared();
                    let cs = speed[i] * CHARGE_SPEED_FRAC;
                    // Attack style for this swing: 0 = stab, 1 = the classic
                    // overhead swing (2 = slash exists in the shader but
                    // benched). A kind with a cut draws 1 as its slow cut;
                    // its 2 is the overhead swing at the stab's pace, drawn
                    // once the overhead is a clip of its own (the code's
                    // swing over the guard turns the blade sideways). Spears
                    // only ever thrust, and a charging man with a cut stabs:
                    // a slow cut at a run carries him into the man before it
                    // lands.
                    let charge = v2 > cs * cs;
                    let style = if my_kind
                        == crate::unit_types::KIND_SPEAR as usize
                        || (charge && params.cut_windup_ticks > 0)
                    {
                        0
                    } else {
                        (crate::units::hash01(
                            tick_seed ^ (i as u32).wrapping_mul(0x51ED),
                        ) * 2.0) as u8
                    };
                    let style = style << crate::units::SWING_STYLE_SHIFT;
                    *s.swing = if charge {
                        crate::units::SWING_WINDUP
                            | crate::units::SWING_CHARGE
                            | style
                    } else {
                        crate::units::SWING_WINDUP | style
                    };
                    *s.swing_t = params.windup_for(*s.swing);
                    // The target is told of the swing now, so that he
                    // can answer it with a block or a parry timed to the
                    // strike (sim/damage.rs).
                    out.events.push(DamageEvent {
                        victim: chosen,
                        attacker: i as u32,
                        jit: 0.0,
                        charge: false,
                        impale: false,
                        warn: *s.swing_t,
                        chain: 0,
                    });
                } else if my_kind == crate::unit_types::KIND_ARCHER as usize
                    && !routed
                    && *s.ammo > 0
                    && *s.sight & SIGHT_KNIFE == 0
                    && s.vel.xz().length_squared() < 4.0
                    && let Some(shot) = &shoot_at[gi]
                    && let Some(q) = pick_quarry(f, s, p, gi, shot)
                {
                    // Nock and draw at his man (foot archers shoot standing
                    // only; the speed test keeps a marching or skirmishing
                    // man's bow on his back). Style bits stay 0: the stab
                    // pull-back IS the string draw.
                    *s.quarry = q;
                    *s.swing = crate::units::SWING_WINDUP
                        | crate::units::SWING_RANGED;
                    *s.swing_t = crate::unit_types::missile::DRAW_TICKS
                        + (crate::units::hash01(
                            tick_seed ^ (i as u32).wrapping_mul(0x2C9F),
                        ) * 20.0) as u8;
                }
            }
        }
        // A reloading archer contacted in melee drops the reload: he defends
        // at knife tempo instead of standing through the 8 s bow cycle.
        if my_kind == crate::unit_types::KIND_ARCHER as usize
            && best_idx != u32::MAX
            && *s.swing & crate::units::SWING_STATE_MASK
                == crate::units::SWING_RECOVER
            && *s.swing_t > params.cooldown_ticks
        {
            *s.swing_t = params.cooldown_ticks;
        }
    }
    st.desired = desired;
    st.face_target = face_target;
}

/// The share of the archers who keep shooting at the same man while he
/// lives and stays in reach; the rest pick afresh for every shot, as men
/// in the chaos of a battle do.
const KEEP_QUARRY_SHARE: f32 = 1.0 / 3.0;
/// An archer without a man to shoot looks for one every this many ticks.
const QUARRY_LOOK_TICKS: u32 = 8;
/// Random members he tries in one look.
const QUARRY_TRIES: u32 = 3;

/// Whether `m` is a man this archer may shoot: an enemy of a block his
/// regiment shoots at, alive, within his reach and, in a search, among
/// his block's clear men (sim/mod.rs `clear_members`).
#[inline]
fn quarry_ok(
    f: &Field,
    p: Vec2,
    my_team: u8,
    reach2: f32,
    shot: &crate::sim::job::ShootAt,
    blocks: &[u32],
    m: u32,
) -> bool {
    let Field { pos_prev, team, group, death_snap, clear_members, .. } = *f;
    let mi = m as usize;
    mi < pos_prev.len()
        && team[mi] != my_team
        && death_snap[mi] == 0
        && blocks.contains(&group[mi])
        && pos_prev[mi].xz().distance_squared(p) <= reach2
        && (!shot.search || clear_members[group[mi] as usize].binary_search(&m).is_ok())
}

/// The man an archer draws at: the one he keeps, when he is one of the
/// men who keep theirs and the man still serves; else a new pick from
/// his regiment's blocks (sim/job.rs `shoot_at`, `search_blocks`): a
/// random living member, or in a search a random clear member, the
/// nearest block first, a few tries a look and a look every few ticks.
/// None: he waits with the bow down.
#[inline]
fn pick_quarry(
    f: &Field,
    s: &Soldier,
    p: Vec2,
    gi: usize,
    shot: &crate::sim::job::ShootAt,
) -> Option<u32> {
    let Field { team, kind, target_members, clear_members, search_blocks, tick, tick_seed, .. } =
        *f;
    let i = s.i;
    let my_team = team[i];
    let reach = crate::unit_types::bow_reach(kind[i]);
    let reach2 = reach * reach;
    let single = [shot.t as u32];
    let blocks: &[u32] = if shot.search && !search_blocks[gi].is_empty() {
        &search_blocks[gi]
    } else {
        &single
    };
    let keeps =
        crate::units::hash01((i as u32).wrapping_mul(0x9E37) ^ 0x1B3D) < KEEP_QUARRY_SHARE;
    if keeps && quarry_ok(f, p, my_team, reach2, shot, blocks, *s.quarry) {
        return Some(*s.quarry);
    }
    if !tick.wrapping_add((i as u32).wrapping_mul(7)).is_multiple_of(QUARRY_LOOK_TICKS) {
        return None;
    }
    for k in 0..QUARRY_TRIES {
        let b = blocks[k as usize % blocks.len()] as usize;
        let list: &[u32] = if shot.search { &clear_members[b] } else { &target_members[b] };
        if list.is_empty() {
            continue;
        }
        let h = crate::units::hash01(
            tick_seed
                ^ (i as u32).wrapping_mul(0x7F4A_7C15).wrapping_add(k.wrapping_mul(0x2545_F491)),
        );
        let m = list[((h * list.len() as f32) as usize).min(list.len() - 1)];
        if quarry_ok(f, p, my_team, reach2, shot, blocks, m) {
            return Some(m);
        }
    }
    None
}

/// What the crowd does to his drive: overlap corrections and pushes are
/// clamped, and in a packed crowd the drive fades out so the mass cannot
/// keep compressing itself.
#[inline]
fn yield_to_crowd(st: &mut Step) {
    let crowd = st.crowd;
    let mut corr = st.corr;
    let mut push = st.push;
    let mut desired = st.desired;
    let corr_len2 = corr.length_squared();
    if corr_len2 > 1e-12 {
        st.overlap_dir = corr / corr_len2.sqrt();
        st.overlap_depth = corr_len2.sqrt() / CORR_GAIN;
    }
    // Every correction is applied, the sub-centimetre ones too: a threshold
    // turns a steady lean into jumps of a centimetre.
    if corr_len2 > CORR_MAX * CORR_MAX {
        corr *= CORR_MAX / corr_len2.sqrt();
    }
    let push_len = push.length();
    if push_len > SEP_PUSH_MAX {
        push *= SEP_PUSH_MAX / push_len;
    }
    // Yield in dense crowds: goal drive fades out entirely so the mass can't
    // keep compressing itself; `jam` (0 = free, 1 = packed) also damps the
    // response below.
    // The yield-and-collide contact brakes for the still-man crowd
    // (`crowd_still`) by the same band, and its fade never takes his drive
    // under the step pace: a wish already slower is left as it is. `jam`
    // still fades close_in's step toward a remembered enemy and its jog to
    // the fight point. FL_BRAKE=0: no brake, nothing slows a man for the
    // crowd ahead of him but the bodies themselves.
    let (slow, stop) = st.brake;
    let old = contact_old();
    let jam = if old {
        ((crowd - slow) / (stop - slow)).clamp(0.0, 1.0)
    } else if still_brake() {
        ((st.crowd_still - slow) / (stop - slow)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    st.wish = desired;
    if old {
        desired *= 1.0 - jam;
    } else if jam > 0.0 {
        let wl = desired.length();
        if wl > 1e-6 {
            let faded = wl * (1.0 - jam);
            let held = wl.min(STEP_PACE);
            st.held = held > faded;
            desired = desired / wl * faded.max(held);
        }
    }
    st.corr = corr;
    st.push = push;
    st.jam = jam;
    st.desired = desired;
}

/// A waiting man's sidestep: in some one-second windows he steps toward
/// whichever side is open, at the shuffle's pace, and in the others he
/// stands.
#[inline]
fn sidestep(i: usize, tick: u32, left_blocked: bool, right_blocked: bool, side_dir: Vec2) -> Vec2 {
    let window = (tick.wrapping_add((i as u32).wrapping_mul(7)) / SIDESTEP_WINDOW)
        .wrapping_mul(0x2545_F491);
    let roll = crate::units::hash01(window ^ (i as u32).wrapping_mul(0x9E37));
    let chance = sidestep_chance();
    let side = match (left_blocked, right_blocked) {
        _ if roll >= chance => Vec2::ZERO,
        (false, true) => side_dir,
        (true, false) => -side_dir,
        (false, false) if roll < 0.5 * chance => side_dir,
        (false, false) => -side_dir,
        (true, true) => Vec2::ZERO,
    };
    side * STEP_PACE
}

/// His footwork around the enemy in reach: closing to his weapon's
/// length and stepping back to it from an enemy who has come inside it.
/// With nobody in reach, going to the enemy he
/// remembers through open ground (waiting behind a comrade, sidestepping
/// now and then), or, with nobody to close on, joining his regiment's
/// fight at the jog.
#[inline]
fn close_in(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { pos_prev, speed, team, group, broken, wall, engaged, hold, moving, focus, tick, .. } = *f;
    let i = s.i;
    let params = &TYPES[st.my_kind];
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    let committed = st.committed;
    let memo_valid = st.memo_valid;
    let join_fp = st.join_fp;
    let best_idx = st.best_idx;
    let crowd = st.crowd;
    let jam = st.jam;
    let way_blocked = st.way_blocked;
    let left_blocked = st.left_blocked;
    let right_blocked = st.right_blocked;
    let comrade_ahead = st.comrade_ahead;
    let side_dir = st.side_dir;
    let mut desired = st.desired;
    let mut d_surge = Vec2::ZERO;
    // Fighters close the last meter to swing range. Only active when an enemy
    // is ALREADY in reach — this is combat execution (like the wind-up foot
    // plant), not steering; it bypasses the jam yield on purpose so front
    // lines stay joined instead of settling at the separation standoff just
    // outside sword range. Close toward the LOCKED swing target when there is
    // one — the per-tick nearest enemy flips in a clog and flip-flopping the
    // close direction reads as twitch. Falls back to the far-acquisition memo
    // in `target` when the near scan is empty AND the unit is in open space —
    // in a dense press the memo would let second ranks drive through the jam
    // and compress the crowd into overlap. The memo may be stale after
    // death sweeps reindex, so it is validated as "some enemy within closing
    // range" — a legitimate closing target regardless of identity. Only a
    // man out of formation goes to a remembered enemy: a formed man seeing
    // one leaves formation first (decide_to_join), so the slot never pulls
    // against the chase.
    let (close_to, memo_close) = if *s.swing & crate::units::SWING_STATE_MASK
        != crate::units::SWING_READY
        && (*s.target as usize) < pos_prev.len()
    {
        (*s.target, false)
    } else if best_idx != u32::MAX {
        (best_idx, false)
    } else if crowd < st.brake.0 && committed {
        (*s.target, true)
    } else {
        (u32::MAX, false)
    };
    // A man sent somewhere by a Move order closes on nobody: he walks on,
    // striking only a man in front of him (the swing above). With his
    // regiment focused on one target he closes on a man of another
    // regiment only within his threat distance (look_around).
    if !dying
        && !routed
        && !moving[gi]
        && (close_to as usize) < pos_prev.len()
        && team[close_to as usize] != team[i]
        && focus[gi].is_none_or(|t| {
            group[close_to as usize] == t
                || pos_prev[close_to as usize].xz().distance_squared(p)
                    < threat_radius() * threat_radius()
        })
    {
        let to_enemy = pos_prev[close_to as usize].xz() - p;
        let dist = to_enemy.length();
        // Held regiments fight at arm's length only.
        let look = if engaged[gi] { seek_radius() } else { WIDE_ACQUIRE_R };
        let max_close = if hold[gi] { 2.2 } else { look + 1.0 };
        // He closes on an enemy in reach to his weapon's length. Going to
        // a seen enemy, each man stops at his own distance, between that
        // and just inside his reach.
        let own = crate::units::hash01((i as u32).wrapping_mul(0x3C1B) ^ 0x51F7);
        let fight = FIGHT_SHARE * params.reach;
        let stop = if memo_close { fight + 0.8 * (params.reach - fight) * own } else { fight };
        if dist > stop && dist < max_close {
            let mut urge = ((dist - stop) / 0.8).clamp(0.0, 1.0);
            // The surge toward a REMEMBERED enemy (no one in reach yet) is
            // steering, not combat execution: it yields to the jam like all
            // steering, so the press brakes on genuine body-pack. Ungated
            // this factor is always 1 (the memo gate above already required
            // a crowd under the brake's start, i.e. jam == 0).
            if memo_close {
                urge *= 1.0 - jam;
                // He steps toward a remembered enemy only through open
                // ground. With a comrade in the way he stands and waits for
                // room instead of leaning on the man's back (M2TW's crowded
                // soldier).
                if way_blocked {
                    urge = 0.0;
                    // Waiting; in some windows he sidesteps toward whichever
                    // side is open.
                    desired += sidestep(i, tick, left_blocked, right_blocked, side_dir);
                }
            }
            let pace = if !memo_close {
                speed[i] * 0.35
            } else if dist > JOG_BEYOND && !comrade_ahead {
                COMBAT_JOG_PACE
            } else {
                ADVANCE_PACE
            };
            desired += to_enemy * (pace * urge / dist);
            d_surge = to_enemy * (pace * urge / dist);
        } else if !memo_close
            && committed
            && wall[gi] == 0
            && !st.back_blocked
            && *s.swing & crate::units::SWING_STATE_MASK != crate::units::SWING_WINDUP
            && !broken[group[close_to as usize] as usize]
            && dist > 1e-3
        {
            // The enemy has come inside his weapon's length: out of
            // formation, with open ground behind him and no blow under way,
            // he steps back to it. He lets the man well inside first, and a
            // step once begun goes all the way out. A man in formation holds
            // his place, and nobody backs off a broken enemy.
            let stepping = s.vel.xz().dot(to_enemy) < -0.5 * STEP_PACE * dist;
            let inside = if stepping { fight } else { fight - BACK_SLACK };
            if dist < inside {
                desired -= to_enemy * (STEP_PACE / dist);
                STEP_BACK_TICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    // Joining: no enemy to close on, so he heads for the enemy unit his
    // regiment fights, jogging over open ground, walking with a comrade close
    // ahead, waiting or sidestepping when blocked. He picks a soldier to
    // fight once one is in sight (the acquisition above).
    if committed
        && !memo_valid
        && let Some(fp) = join_fp
        && best_idx == u32::MAX
        && d_surge == Vec2::ZERO
        && *s.swing & crate::units::SWING_STATE_MASK == crate::units::SWING_READY
    {
        let to_fp = fp - p;
        let dist = to_fp.length();
        if dist > 1.0 {
            let dir = to_fp / dist;
            desired = if !way_blocked {
                let pace = if comrade_ahead { ADVANCE_PACE } else { COMBAT_JOG_PACE };
                dir * pace * (1.0 - jam)
            } else {
                sidestep(i, tick, left_blocked, right_blocked, side_dir)
            };
        }
    }
    st.desired = desired;
}

/// From drive to velocity: the wall and charge paces, fatigue, the
/// stagger, the crowd's damping, a standing man's grip on the ground and
/// his stance against a squeeze, the acceleration and speed caps, and the
/// give of his legs at a body he runs into.
#[inline]
fn steer(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { speed, orders, wall, charging, fat_speed, fat_nocharge, dt, .. } = *f;
    let i = s.i;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    // Holding his place: his regiment has no order and he is not broken,
    // dying, in its melee or out of formation for a fight.
    let planted = orders[gi].is_none() && !dying && !routed && !st.in_melee && !st.committed;
    let push = st.push;
    let jam = st.jam;
    let mut desired = st.desired;
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

    let v = s.vel.xz();
    if !contact_old() {
        st.new_v = legs_and_feet(desired, v, if squeezing(*s.stuck) { SQUEEZE_CAP } else { speed[i] * 1.15 }, dt);
        st.corr = Vec2::ZERO;
        return;
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
    let standing = desired.length_squared() < 1e-6;
    if standing || planted {
        let pl = push_a.length();
        push_a *= if pl > STAND_GRIP { (pl - STAND_GRIP) / pl } else { 0.0 };
    }
    if standing && st.overlap_depth > 0.0 && *s.swing & crate::units::SWING_STAGGERED == 0 {
        st.corr *= ((st.overlap_depth - STANCE) / st.overlap_depth).max(0.0);
    }
    let mut legs = (desired - v) * STEER_GAIN;
    let l2 = legs.length_squared();
    if l2 > LEG_ACCEL * LEG_ACCEL {
        legs *= LEG_ACCEL / l2.sqrt();
    }
    let mut accel = legs + push_a;
    let a2 = accel.length_squared();
    if a2 > MAX_ACCEL * MAX_ACCEL {
        accel *= MAX_ACCEL / a2.sqrt();
    }

    let mut new_v = v + accel * dt;
    // Viscous damping in the press: bleeds the spring energy that otherwise
    // ping-pongs between neighbors every tick.
    new_v *= 1.0 - 0.4 * jam;
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
    // he hits move on together and a shove travels through a press. Soft
    // wall (FL_CONTACT=wall): toward the bodies he overlaps a man's legs
    // lose their closing speed over his give, all of it left at the touch,
    // none at the give, never carrying him past it in one tick, and never
    // more than his own speed into them, so a column on the march is not
    // held up; the man he hits is handed nothing and is moved by the
    // setting apart alone, as pressure.
    let dir = st.overlap_dir;
    if dir != Vec2::ZERO && share_contact() {
        let into = (new_v - st.overlap_vel).dot(-dir);
        if into > 0.0 {
            // A man holding his place loses less of it: his feet take the
            // first part of the shove.
            let mut lost = into * st.overlap_share;
            if planted {
                lost = (lost - STAND_GRIP * dt * own_spread(i)).max(0.0);
            }
            new_v += dir * lost;
        }
    } else if dir != Vec2::ZERO {
        let own = new_v.dot(-dir);
        let into = (new_v - st.overlap_vel).dot(-dir);
        let closing = into.min(own);
        if closing > 0.0 {
            let give = give(i);
            let left = (give - st.overlap_depth).max(0.0);
            let allowed = (closing * left / give).min(left / dt);
            new_v += dir * (closing - allowed);
        }
    }
    st.new_v = new_v;
}

/// The yield-and-collide contact's velocity: his legs toward his drive at
/// their acceleration, his feet taking back up to FEET x dt of what his
/// legs do not want (his velocity across his drive and the excess along
/// it), under the acceleration cap and his speed cap. No spring, no grip,
/// no stance, no shared speed: the bodies are settled by position after
/// the step (`set_apart`, `collide`).
#[inline]
fn legs_and_feet(desired: Vec2, v: Vec2, vmax: f32, dt: f32) -> Vec2 {
    let mut legs = (desired - v) * STEER_GAIN;
    let l2 = legs.length_squared();
    if l2 > LEG_ACCEL * LEG_ACCEL {
        legs *= LEG_ACCEL / l2.sqrt();
    }
    let want = desired.length();
    let unwanted = if want < 1e-6 {
        v
    } else {
        let d_hat = desired / want;
        v - d_hat * v.dot(d_hat).clamp(0.0, want)
    };
    let ul = unwanted.length();
    if ul > 1e-12 {
        let bleed = ul.min(FEET * dt);
        legs -= unwanted * (bleed / ul / dt);
    }
    let a2 = legs.length_squared();
    if a2 > MAX_ACCEL * MAX_ACCEL {
        legs *= MAX_ACCEL / a2.sqrt();
    }
    let mut new_v = v + legs * dt;
    let v2 = new_v.length_squared();
    if v2 > vmax * vmax {
        new_v *= vmax / v2.sqrt();
    }
    new_v
}

/// Which way he looks: the man he swings at, the enemy in reach when he
/// is fighting or just struck, where he is going when out of formation,
/// the foe he remembers close by, else the line's facing, at the human
/// turn rate.
#[inline]
fn face(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { pos_prev, team, threat, form_face, dt, .. } = *f;
    let i = s.i;
    let p = st.p;
    let gi = st.gi;
    let dying = st.dying;
    let routed = st.routed;
    let committed = st.committed;
    let memo_dir = st.memo_dir;
    let best_idx = st.best_idx;
    let face_target = st.face_target;
    let new_v = st.new_v;
    // Facing priority: locked wind-up target > nearest enemy in reach >
    // movement direction. Fighters keep eyes on the enemy even while the
    // crowd shoves them; only routing/unengaged units face their velocity.
    // yaw_prev snapshots the pre-update angle so the renderer can interpolate
    // (yaw stepped once per tick otherwise — visible facing snaps at high
    // fps).
    *s.yaw_prev = *s.yaw;
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
        None if !routed && committed && memo_dir != Vec2::ZERO => memo_dir,
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

/// The step: velocity into position, clamped to the field, sliding along
/// impassable ground, set down on the terrain.
#[inline]
fn integrate(f: &Field, s: &mut Soldier, st: &mut Step) {
    let Field { terrain, pos_prev, kind, dt, bounds_min, bounds_max, .. } = *f;
    let i = s.i;
    let new_v = st.new_v;
    let corr = st.corr;
    *s.vel = Vec3::new(new_v.x, 0.0, new_v.y);
    let to = Vec2::new(pos_prev[i].x + new_v.x * dt + corr.x, pos_prev[i].z + new_v.y * dt + corr.y);
    let n = ground_step(terrain, bounds_min, bounds_max, pos_prev[i].xz(), to);
    *s.pos = Vec3::new(
        n.x,
        terrain.height_at(n.x, n.y) + crate::unit_types::half_height(kind[i] as usize),
        n.y,
    );
}

/// A move from `from` to `to` on the ground plane, clamped to the field,
/// sliding along impassable ground.
#[inline]
fn ground_step(terrain: &Terrain, bounds_min: Vec2, bounds_max: Vec2, from: Vec2, to: Vec2) -> Vec2 {
    let mut nx = to.x.clamp(bounds_min.x, bounds_max.x);
    let mut nz = to.y.clamp(bounds_min.y, bounds_max.y);
    // Impassable ground (terrace risers, gorge walls, crater lips):
    // wall-slide — keep the axis that stays on walkable ground, drop the one
    // that doesn't, so crowds flow along the obstacle.
    if terrain.blocked_at(nx, nz) {
        let (px, pz) = (from.x, from.y);
        if !terrain.blocked_at(nx, pz) {
            nz = pz;
        } else if !terrain.blocked_at(px, nz) {
            nx = px;
        } else {
            nx = px;
            nz = pz;
        }
    }
    Vec2::new(nx, nz)
}

/// A run of rows of one kernel chunk for the yield-and-collide
/// contact's passes: the chunk's men's touching pairs
/// (`ChunkOut::touch`), the end of each row's run in them, and where the
/// first row's run begins.
pub(crate) struct TouchRows<'a> {
    pub start: usize,
    pub touch: &'a [Touch],
    pub touch_end: &'a [u32],
    pub from0: u32,
}

impl TouchRows<'_> {
    #[inline]
    fn run(&self, j: usize) -> &[Touch] {
        let from = if j == 0 { self.from0 } else { self.touch_end[j - 1] } as usize;
        &self.touch[from..self.touch_end[j] as usize]
    }
}

/// A place buffer of the contact phase, by rank in the grid's order
/// (`SpatialGrid::sorted`), which its tasks write at scattered slots: a
/// man's slot is his rank, and only the task that holds the man writes it.
#[derive(Clone, Copy)]
pub(crate) struct Places(pub *mut Vec2);
unsafe impl Send for Places {}
unsafe impl Sync for Places {}

/// Where each man of a chunk stands before the first pass of the setting
/// apart, in both place buffers, at his rank: a man with a touching pair
/// at his legs' step from the tick's start, unclamped, as the step asks
/// (`q` only; the first pass writes his `next`); any other man where the
/// kernel set him down, in both, since no pass moves him or writes his
/// place again. Returns how many men of the chunk have a touching pair.
///
/// # Safety
/// `q` and `next` hold a slot for every rank, and no one else reads or
/// writes the chunk's men's slots meanwhile.
pub(crate) unsafe fn set_apart_start(f: &Field, rows: &TouchRows, vel: &[Vec3], pos: &[Vec3], q: Places, next: Places) -> u32 {
    let rank_of = f.grid.rank_of();
    let mut listed = 0;
    for j in 0..pos.len() {
        let i = rows.start + j;
        let r = rank_of[i] as usize;
        // SAFETY: the caller's.
        unsafe {
            if rows.run(j).is_empty() {
                *q.0.add(r) = pos[j].xz();
                *next.0.add(r) = pos[j].xz();
            } else {
                listed += 1;
                *q.0.add(r) = f.pos_prev[i].xz() + vel[j].xz() * f.dt;
            }
        }
    }
    listed
}

/// One pass of the setting apart, for one chunk (`set_apart_passes`):
/// every man reads the overlaps with the men of his touching pairs where
/// the last pass left everyone (`cur`, by rank), removes each whole, moving the
/// other man's mass share of it away from him, averages his corrections
/// over the men he touches, multiplies them by SET_APART_WEIGHT and
/// steps by them on the ground (`next`). All men move together after the
/// pass. The first pass steps from the tick's start, the rest from where
/// the last pass left him. A man touching nobody is skipped: both
/// buffers already hold his place (`set_apart_start`).
///
/// # Safety
/// `next` holds a slot for every rank, and no one else reads or writes
/// the chunk's men's slots meanwhile.
pub(crate) unsafe fn set_apart(f: &Field, rows: &TouchRows, first: bool, cur: &[Vec2], next: Places) {
    let tab = touch_table();
    let rank_of = f.grid.rank_of();
    for j in 0..rows.touch_end.len() {
        let run = rows.run(j);
        if run.is_empty() {
            continue;
        }
        let i = rows.start + j;
        let r = rank_of[i] as usize;
        let me = cur[r];
        let mut corr = Vec2::ZERO;
        let mut count = 0u32;
        // Every pair worked out and weighed by `on`, 1 for a pair that
        // overlaps and 0 for any other, without a branch: a 0 added to
        // the sums changes nothing, since neither is ever -0. The distance
        // is held off 0 so the numbers stay finite; an overlapping pair is
        // farther than that already. Four pairs at a time, with the same
        // arithmetic, summed one after the other in their order.
        #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
        let mut k = 0;
        #[cfg(target_arch = "x86_64")]
        while k + 4 <= run.len() {
            let t = &run[k..k + 4];
            let q = [cur[t[0].b as usize], cur[t[1].b as usize], cur[t[2].b as usize], cur[t[3].b as usize]];
            let (mut px, mut pz) = ([0.0f32; 4], [0.0f32; 4]);
            // SAFETY: SSE2 is part of every x86_64; the stores write the
            // two local arrays.
            let on = unsafe {
                use std::arch::x86_64::*;
                let qx = _mm_set_ps(q[3].x, q[2].x, q[1].x, q[0].x);
                let qz = _mm_set_ps(q[3].y, q[2].y, q[1].y, q[0].y);
                let hard = _mm_set_ps(tab.hard(&t[3]), tab.hard(&t[2]), tab.hard(&t[1]), tab.hard(&t[0]));
                let share = _mm_set_ps(tab.share(&t[3]), tab.share(&t[2]), tab.share(&t[1]), tab.share(&t[0]));
                let (dx, dz) = (_mm_sub_ps(_mm_set1_ps(me.x), qx), _mm_sub_ps(_mm_set1_ps(me.y), qz));
                let l2 = _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dz, dz));
                let tiny = _mm_set1_ps(1e-12);
                let m = _mm_and_ps(_mm_cmplt_ps(l2, _mm_mul_ps(hard, hard)), _mm_cmpgt_ps(l2, tiny));
                let l = _mm_sqrt_ps(_mm_max_ps(l2, tiny));
                let w = _mm_mul_ps(_mm_div_ps(_mm_mul_ps(_mm_sub_ps(hard, l), share), l), _mm_and_ps(m, _mm_set1_ps(1.0)));
                _mm_storeu_ps(px.as_mut_ptr(), _mm_mul_ps(dx, w));
                _mm_storeu_ps(pz.as_mut_ptr(), _mm_mul_ps(dz, w));
                _mm_movemask_ps(m)
            };
            for lane in 0..4 {
                corr.x += px[lane];
                corr.y += pz[lane];
            }
            count += on.count_ones();
            k += 4;
        }
        for t in &run[k..] {
            let dq = me - cur[t.b as usize];
            let l2 = dq.length_squared();
            let hard = tab.hard(t);
            let on = (l2 < hard * hard) & (l2 > 1e-12);
            let l = l2.max(1e-12).sqrt();
            corr += dq * ((hard - l) * tab.share(t) / l * on as u32 as f32);
            count += on as u32;
        }
        let from = if first { f.pos_prev[i].xz() } else { me };
        let n = if count == 0 && !first {
            me
        } else {
            let corr = corr * (SET_APART_WEIGHT / count.max(1) as f32);
            ground_step(f.terrain, f.bounds_min, f.bounds_max, from, me + corr)
        };
        // SAFETY: the caller's.
        unsafe { *next.0.add(r) = n };
    }
}

/// The collision after the setting apart, for one chunk (places by rank,
/// the columns by index): each man's move
/// over the tick, from the tick's start to where the last pass left him;
/// for every pair still touching there and closing on those moves, he
/// takes the other man's mass share of the closing speed along the
/// line between them, summed over his pairs and added to his legs'
/// velocity. Parting or sliding pairs give nothing. His position is set
/// down on the ground. Returns the chunk's men's pairs still overlapping
/// there, each counted from the man's side.
pub(crate) fn collide(f: &Field, rows: &TouchRows, cur: &[Vec2], vel: &mut [Vec3], pos: &mut [Vec3]) -> u32 {
    let inv_dt = 1.0 / f.dt;
    let tab = touch_table();
    let (sorted, rank_of) = (f.grid.sorted(), f.grid.rank_of());
    let mut overlapping = 0;
    for j in 0..vel.len() {
        let i = rows.start + j;
        let run = rows.run(j);
        if run.is_empty() {
            continue;
        }
        let me = cur[rank_of[i] as usize];
        let u = (me - f.pos_prev[i].xz()) * inv_dt;
        let mut dv = Vec2::ZERO;
        // Weighed by `on` without a branch, as in `set_apart`: 1 for a
        // pair that overlaps and closes, 0 for any other. Four pairs at a
        // time, as there.
        #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
        let mut k = 0;
        #[cfg(target_arch = "x86_64")]
        while k + 4 <= run.len() {
            let t = &run[k..k + 4];
            let bs = [t[0].b as usize, t[1].b as usize, t[2].b as usize, t[3].b as usize];
            let q = [cur[bs[0]], cur[bs[1]], cur[bs[2]], cur[bs[3]]];
            let o = [&sorted[bs[0]], &sorted[bs[1]], &sorted[bs[2]], &sorted[bs[3]]];
            let (mut nvx, mut nvz) = ([0.0f32; 4], [0.0f32; 4]);
            // SAFETY: SSE2 is part of every x86_64; the stores write the
            // two local arrays.
            let hit = unsafe {
                use std::arch::x86_64::*;
                let qx = _mm_set_ps(q[3].x, q[2].x, q[1].x, q[0].x);
                let qz = _mm_set_ps(q[3].y, q[2].y, q[1].y, q[0].y);
                let sx = _mm_set_ps(o[3].x, o[2].x, o[1].x, o[0].x);
                let sz = _mm_set_ps(o[3].z, o[2].z, o[1].z, o[0].z);
                let hard = _mm_set_ps(tab.hard(&t[3]), tab.hard(&t[2]), tab.hard(&t[1]), tab.hard(&t[0]));
                let share = _mm_set_ps(tab.share(&t[3]), tab.share(&t[2]), tab.share(&t[1]), tab.share(&t[0]));
                let (dx, dz) = (_mm_sub_ps(_mm_set1_ps(me.x), qx), _mm_sub_ps(_mm_set1_ps(me.y), qz));
                let l2 = _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dz, dz));
                let tiny = _mm_set1_ps(1e-12);
                let overlaps = _mm_and_ps(_mm_cmplt_ps(l2, _mm_mul_ps(hard, hard)), _mm_cmpgt_ps(l2, tiny));
                let l = _mm_sqrt_ps(_mm_max_ps(l2, tiny));
                let (nx, nz) = (_mm_div_ps(dx, l), _mm_div_ps(dz, l));
                let idt = _mm_set1_ps(inv_dt);
                let (ubx, ubz) = (_mm_mul_ps(_mm_sub_ps(qx, sx), idt), _mm_mul_ps(_mm_sub_ps(qz, sz), idt));
                let c = _mm_add_ps(
                    _mm_mul_ps(_mm_sub_ps(ubx, _mm_set1_ps(u.x)), nx),
                    _mm_mul_ps(_mm_sub_ps(ubz, _mm_set1_ps(u.y)), nz),
                );
                let on = _mm_and_ps(overlaps, _mm_cmpgt_ps(c, _mm_setzero_ps()));
                let w = _mm_mul_ps(_mm_mul_ps(c, share), _mm_and_ps(on, _mm_set1_ps(1.0)));
                _mm_storeu_ps(nvx.as_mut_ptr(), _mm_mul_ps(nx, w));
                _mm_storeu_ps(nvz.as_mut_ptr(), _mm_mul_ps(nz, w));
                _mm_movemask_ps(overlaps)
            };
            for lane in 0..4 {
                dv.x += nvx[lane];
                dv.y += nvz[lane];
            }
            overlapping += hit.count_ones();
            k += 4;
        }
        for t in &run[k..] {
            let b = t.b as usize;
            let dq = me - cur[b];
            let l2 = dq.length_squared();
            let hard = tab.hard(t);
            let overlaps = (l2 < hard * hard) & (l2 > 1e-12);
            overlapping += overlaps as u32;
            let n = dq / l2.max(1e-12).sqrt();
            let u_b = (cur[b] - sorted[b].xz()) * inv_dt;
            let c = (u_b - u).dot(n);
            let on = overlaps & (c > 0.0);
            dv += n * (c * tab.share(t) * on as u32 as f32);
        }
        vel[j].x += dv.x;
        vel[j].z += dv.y;
        pos[j] = Vec3::new(
            me.x,
            f.terrain.height_at(me.x, me.y) + crate::unit_types::half_height(f.kind[i] as usize),
            me.y,
        );
    }
    overlapping
}

