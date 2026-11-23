//! The scripted test scenes and the diagnostic logs. A scene file holds
//! what one FL_ switch turns on: its spawn (called from the battle
//! spawn in regiments.rs), the script that gives its orders and the log
//! that prints its numbers. logs.rs and shake.rs hold the logs any
//! battle can print.

use bevy::prelude::*;

pub(crate) mod archery;
pub(crate) mod arena;
pub(crate) mod charge;
pub(crate) mod direction;
mod form;
mod front;
pub(crate) mod join;
mod logs;
mod orders;
pub(crate) mod pass;
pub(crate) mod path;
pub(crate) mod pile;
pub(crate) mod push;
pub(crate) mod retarget;
pub(crate) mod rout;
mod shake;
pub(crate) mod surround;

use archery::archery_log;
use arena::arena_log;
use charge::charge_test_log;
use direction::dir_test_log;
use form::test_form_script;
use front::test_front_script;
use join::join_test_log;
use logs::{disorder_log, melee_diag_log, room_log, straggler_log};
use orders::test_orders_script;
use pass::{pass_line_script, routpass_test_log};
use pile::pile_test_log;
use push::push_test_log;
use retarget::test_retarget_script;
use rout::rout_test_log;
use shake::shake_log;
use surround::surround_test_log;

/// Unit spacing inside a regiment block in the test scenarios' layouts.
const SPACING: f32 = crate::formation::BASE_SPACING;

/// The scenes' systems. Each returns at once unless its FL_ switch is
/// set; the path scenes register theirs in `path::PathTestPlugin`.
pub struct ScenesPlugin;

impl Plugin for ScenesPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(path::PathTestPlugin)
            .add_systems(Update, surround_test_log)
            .add_systems(
                Update,
                (rout_test_log, dir_test_log, arena_log, charge_test_log, pile_test_log, melee_diag_log, join_test_log, routpass_test_log, pass_line_script, push_test_log, archery_log, room_log, shake_log, straggler_log),
            )
            .add_systems(Update, (test_orders_script, test_retarget_script))
            .add_systems(Update, (test_form_script, disorder_log))
            .add_systems(Update, test_front_script);
    }
}
