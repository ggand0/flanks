//! FL_PLAY_LOG=1: a record of a played battle for finding a defect seen in
//! play, so one play is enough. Files go to `tmp/runs/play-logs/<start
//! time>/`:
//!
//! - `regs.log`: every five seconds of battle, one line per regiment: its
//!   team, kind, men, state, where its men are and where it is sent (its
//!   centroid, anchor, order and the goal its men's places are laid
//!   about), its fight (engaged, contact frame, melee clock, charge),
//!   archers' stand-off and skirmish, its spacing and files, how far its
//!   men spread, and how many stand off their places and how far.
//! - `snap-<n>.txt`, written when the player presses F9: every living
//!   soldier at that moment (regiment, position and height, his place,
//!   his velocity, his target, out of formation or not), every regiment's
//!   line as in `regs.log`, and where the camera looks, so the regiment
//!   on screen can be found in it.
//!
//! The order log (`[order]` lines, orders.rs `log_order`) goes to the
//! game's own output; with the output saved to a file the play's orders
//! are there too.

use bevy::prelude::*;
use std::io::Write;

use crate::orders::{Groups, Order};
use crate::units::Units;

pub struct PlayLogPlugin;

impl Plugin for PlayLogPlugin {
    fn build(&self, app: &mut App) {
        if !on() {
            return;
        }
        app.init_resource::<PlayLog>().add_systems(Update, (log_regiments, snapshot));
    }
}

fn on() -> bool {
    std::env::var("FL_PLAY_LOG").is_ok_and(|v| v != "0")
}

/// How often `regs.log` gets its lines (seconds of battle).
const EVERY_S: f32 = 5.0;

#[derive(Resource, Default)]
struct PlayLog {
    dir: Option<std::path::PathBuf>,
    regs: Option<std::fs::File>,
    /// Battle time of the last `regs.log` lines (s).
    last: f32,
    snaps: u32,
}

impl PlayLog {
    /// The play's folder, made on first use.
    fn dir(&mut self) -> std::path::PathBuf {
        self.dir
            .get_or_insert_with(|| {
                let stamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let dir = std::path::PathBuf::from(format!("tmp/runs/play-logs/{stamp}"));
                if let Err(e) = std::fs::create_dir_all(&dir) {
                    warn!("[play-log] cannot make {}: {e}", dir.display());
                }
                info!("[play-log] writing to {}", dir.display());
                dir
            })
            .clone()
    }
}

/// One regiment's line (the module docs).
fn regiment_line(groups: &Groups, units: &Units, g: usize, t: f32) -> String {
    let gd = &groups.list[g];
    let goal = groups.goal(g);
    let at = goal.unwrap_or(gd.anchor);
    let (mut out_form, mut off_1, mut off_5, mut sum, mut max, mut n) = (0, 0, 0, 0.0f32, 0.0f32, 0usize);
    for i in 0..units.len() {
        if units.group[i] as usize != g || units.death_t[i] != 0 {
            continue;
        }
        let p = Vec2::new(units.pos[i].x, units.pos[i].z);
        let d = p.distance(at + units.home[i]);
        out_form += usize::from(units.out_form[i]);
        off_1 += usize::from(d > 1.0);
        off_5 += usize::from(d > 5.0);
        sum += d;
        max = max.max(d);
        n += 1;
    }
    let order = match gd.order {
        Some(Order::Move(p)) => format!("move ({:.0}, {:.0})", p.x, p.y),
        Some(Order::Attack(e)) => {
            let ed = &groups.list[e as usize];
            format!(
                "attack g{e} ({} men{}, at ({:.0}, {:.0}))",
                ed.count,
                if ed.state.is_broken() { ", broken" } else { "" },
                ed.centroid.x,
                ed.centroid.y
            )
        }
        None => "none".into(),
    };
    let goal = goal.map_or("none".to_string(), |p| format!("({:.0}, {:.0})", p.x, p.y));
    format!(
        "t {t:.0} g{g} team {} {} men {} broken {} | centroid ({:.0}, {:.0}) anchor ({:.0}, {:.0}) facing {:.2} | order {order} goal {goal} | \
         engaged {} contact {} melee {} charging {} crashing {} retarget {} | standoff {} skirmish {} skirmish_hold {} hold {} | \
         spacing {:?} shape {:?} files {} lane {} | reach {:.1} lead {:.1} | out of formation {out_form}, off place >1 m {off_1}, >5 m {off_5}, mean {:.1} m, most {max:.1} m",
        gd.team,
        crate::unit_types::kind_name(gd.kind),
        gd.count,
        gd.state.is_broken(),
        gd.centroid.x,
        gd.centroid.y,
        gd.anchor.x,
        gd.anchor.y,
        gd.facing,
        gd.engaged,
        gd.contact,
        gd.melee_ticks,
        gd.charging,
        gd.crashing,
        gd.retarget,
        gd.standoff,
        gd.skirmish,
        gd.skirmish_hold.is_some(),
        gd.hold,
        gd.spacing,
        gd.shape,
        gd.files,
        gd.lane,
        gd.reach,
        gd.lead,
        if n > 0 { sum / n as f32 } else { 0.0 },
    )
}

/// `regs.log`, every EVERY_S seconds of battle.
fn log_regiments(mut log: ResMut<PlayLog>, groups: Res<Groups>, units: Res<Units>, time: Res<Time<Virtual>>) {
    if groups.list.is_empty() || units.len() == 0 {
        return;
    }
    let t = time.elapsed_secs();
    if t < log.last + EVERY_S && log.regs.is_some() {
        return;
    }
    log.last = t;
    if log.regs.is_none() {
        let path = log.dir().join("regs.log");
        log.regs = std::fs::File::create(&path).map_err(|e| warn!("[play-log] {}: {e}", path.display())).ok();
    }
    let Some(file) = log.regs.as_mut() else { return };
    for g in 0..groups.list.len() {
        let _ = writeln!(file, "{}", regiment_line(&groups, &units, g, t));
    }
}

/// F9: `snap-<n>.txt` (the module docs).
fn snapshot(
    keys: Res<ButtonInput<KeyCode>>,
    mut log: ResMut<PlayLog>,
    groups: Res<Groups>,
    units: Res<Units>,
    time: Res<Time<Virtual>>,
    camera: Query<&GlobalTransform, With<Camera3d>>,
) {
    if !keys.just_pressed(KeyCode::F9) || groups.list.is_empty() {
        return;
    }
    log.snaps += 1;
    let path = log.dir().join(format!("snap-{}.txt", log.snaps));
    let Ok(mut file) = std::fs::File::create(&path) else {
        warn!("[play-log] cannot write {}", path.display());
        return;
    };
    let t = time.elapsed_secs();
    if let Some(cam) = camera.iter().next() {
        let (p, f) = (cam.translation(), cam.forward());
        // Where the camera's middle line meets the plane y = 0: the spot
        // it looks at, near enough.
        let ahead = if f.y < -1e-3 { p + *f * (p.y / -f.y) } else { p };
        let _ = writeln!(
            file,
            "camera at ({:.0}, {:.0}, {:.0}) looking at about ({:.0}, {:.0})",
            p.x, p.y, p.z, ahead.x, ahead.z
        );
    }
    let _ = writeln!(file, "battle time {t:.1} s\n\n# regiments");
    for g in 0..groups.list.len() {
        let _ = writeln!(file, "{}", regiment_line(&groups, &units, g, t));
    }
    let _ = writeln!(file, "\n# soldiers: g i x height z place_x place_z vx vz target out_of_formation hp");
    for i in 0..units.len() {
        if units.death_t[i] != 0 {
            continue;
        }
        let g = units.group[i] as usize;
        let at = groups.goal(g).unwrap_or(groups.list[g].anchor) + units.home[i];
        let (p, v) = (units.pos[i], units.vel[i]);
        let target = if (units.target[i] as usize) < units.len() { units.target[i] as i64 } else { -1 };
        let _ = writeln!(
            file,
            "{g} {i} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {target} {} {:.0}",
            p.x, p.y, p.z, at.x, at.y, v.x, v.z, units.out_form[i], units.hp[i]
        );
    }
    info!("[play-log] F9: snapshot {} at {t:.1} s in {}", log.snaps, path.display());
}
