# FLANKS

FLANKS is a real-time medieval battle game written in Rust and Bevy.
Every soldier on the field is individually simulated, and battles can scale up to a few hundred thousand soldiers (200k tested on an RTX 3090). The combat model builds on mechanics measured from Medieval 2: Total War, with the goal of going beyond the classics rather than recreating them. It is an early prototype under active development.

## Features

- Mass battles: two armies of up to 100 regiments, 1,000 soldiers each
- Regiment-based orders that keep formations intact: lasso selection, move and attack orders, battle lines drawn by dragging
- Formations: shield wall, spear wall, loose order, hold position
- Melee combat with swing timers, directional defense, charge impact, and physical spear walls
- Morale and fatigue based on values measured from the M2TW engine: regiments waver, rout, rally, or shatter
- Skirmish AI opponent and battle outcomes
- Unit cards, regiment banners, and a live morale inspect panel
- Battle audio: layered battle din, steel, screams, horns, and war cries
- Low-poly soldiers animated on the GPU, optimized to render the whole field at interactive framerates
- Army size selectable in the menu, from 20k to 200k total soldiers depending on your hardware

## Build and run

```sh
cargo run --profile opt-dev
```

Requires Rust 1.95 or newer.

## Controls

| Action                          | Input                     |
|---------------------------------|---------------------------|
| Select regiments                | LMB drag (lasso or loop)  |
| Move / attack                   | Right click               |
| Draw a battle line              | RMB drag                  |
| Halt selection                  | Backspace                 |
| Shield / spear wall             | F                         |
| Loose order                     | L                         |
| Blob (mob) formation            | B                         |
| Hold position                   | H                         |
| Control groups                  | Ctrl + 1..9 store, 1..9 recall |
| Pan camera                      | WASD or screen edges      |
| Zoom / rotate camera            | Scroll / middle drag      |
| Pause                           | Esc                       |
| Debug overlays                  | G                         |

A set of `FL_*` environment variables configure sandbox battles and scripted test scenarios (army size, AI on/off, random seed, and so on). `FL_MAP=river` enables an experimental map with a river and vegetation; `FL_VOLUME=0` mutes the game.

## Assets

There are no external art assets: unit meshes, terrain, and all animation are generated in code. Sound effects are AI-generated (ElevenLabs), plus one [marching loop from Pixabay](https://pixabay.com/sound-effects/people-marching-loop-32908/). Audio files are covered by their respective licenses, not the source license below.

## License

FLANKS is licensed under either of

- Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license
  ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
