# FLANKS

Real-time massed medieval battles in Rust and Bevy 0.19. Two armies
of regiments fight at a steady 30 Hz sim tick and interactive
framerates, every soldier individually simulated. Army size is
selectable in the menu and scales with your hardware, from tens of
thousands of soldiers up to hundreds of thousands.

The battle model starts from measured Medieval 2: Total War behavior
(swing-timer melee, morale, fatigue, routs), but the point is the
scale: battles an order of magnitude past what the classics ran, and a
foundation for going beyond them rather than recreating one.

## Features

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
- Map art: a meandering river carved into the heightfield (animated
  faceted water shader, foam banks, a gorge where it exits the
  mountains). The river is wadeable everywhere at reduced speed; a
  stone bridge gives a dry full-speed crossing. Highlands above ~15 m
  are terraced into angled plateau steps whose risers are impassable
  (M2TW-style mountain framing). Wheat-field patches and chunk-merged
  low-poly forests fill the outskirts.

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
configure sandbox battles. FL_MAP=river enables the map-art terrain
(river/terraces/vegetation; classic map is the default); FL_VOLUME=0
mutes.

## Build and run

```
cargo run --profile opt-dev
```

Rust 1.95 or newer.

## Assets

There are no external art assets: unit meshes, terrain, water,
vegetation, and all animation are generated in code. Sound effects and
audio beds are AI-generated (ElevenLabs), plus one marching-boots loop
from freesound.org.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
