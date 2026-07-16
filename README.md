# cascade ("frontline")

A Medieval 2: Total War style mass battle prototype in Rust and Bevy 0.19.
Two armies of 100 regiments (1000 soldiers each) fight at a steady 30 Hz
sim tick and interactive framerates. 200k soldiers is the standard test
load.

## What is implemented (as of the battle-feel merge)

Engine:

- Soldiers live in structure-of-arrays buffers, never as per-entity ECS
  objects. One instanced draw call per unit kind renders the whole army.
- Unit meshes are code-built low-poly cuboid figures (knights and
  men-at-arms) with per-part team coloring. All animation runs in the
  vertex shader: walk cycles, attack swings, braced approach, victory
  cheers, death falls, corpse poses.
- A counting-sort spatial hash grid rebuilds every tick and feeds a
  parallel movement and combat kernel: goal steering, mass-weighted
  separation, crowd jam yield, positional overlap resolution.
- Deformable heightmap terrain with chunked remeshing, plus a density
  field that draws the live front line as a contour.

Game:

- Regiments are permanent groups. An order is a single point; each
  soldier offsets it by his home slot, so blocks move without dissolving.
- Controls: lasso or loop selection, right click to move or attack,
  Backspace to halt, ctrl+1..9 control groups, WASD and edge pan camera,
  scroll zoom, middle drag rotate, G debug overlays, R restart.
- Swing-timer melee with wind-up, reach checks, cooldown jitter, and a
  momentum bonus for hits landed at charging speed. Attack orders chase
  a target regiment, trigger a charge phase inside 60 m, and fire war
  cries on the way in.
- Per-regiment morale drained by casualties, being outflanked (the
  dominant factor, measured on a density ring), routing neighbors, and
  local odds, damped by nearby steady allies. Broken regiments rout,
  can rally, or shatter and flee the field. Pursuers cut runners down.
- A simple skirmish AI attacks with spread targeting, and the battle
  ends in victory, defeat, or mutual destruction.
- Regiment banners show kind, strength, morale, and selection. A hover
  inspect panel breaks down live morale factors. Corpses stay where men
  fell.
- Battle audio mixes crossfaded intensity beds with budgeted one-shots:
  steel, screams, horns, drums, war cries, rout wails, and stings.

Verification is log-based: FL_TEST_* environment knobs run scripted
acceptance scenarios (front line formation, order cohesion, encirclement,
rout behavior) whose outcomes are checked from the log rather than by
eye. FL_UNITS, FL_REG_SIZE, FL_COMBAT_SCALE, FL_AI and similar knobs
configure sandbox battles.

## Build and run

```
cargo run --profile opt-dev
```

Rust 1.95 or newer. See CLAUDE.md and devlogs/ (untracked) for
development notes.
