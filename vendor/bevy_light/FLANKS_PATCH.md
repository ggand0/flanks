# bevy_light 0.19.0, patched for flanks

Written by Claude Opus 5.5.

- Changed: in `check_dir_light_mesh_visibility` (src/lib.rs), the loop that gathers each thread's per-cascade lists skips a list shorter than the cascade count instead of indexing past it.
- Why: only the threads that work in a frame resize their lists to the light's cascades; a thread idle that frame keeps an older, shorter list, so adding a cascade while the game runs (flanks gives the sun a fourth on maps with a wall) panicked with an index out of bounds.
- A shorter list means that thread saw no entity for the missing cascades this frame, so skipping it drops nothing.
- Everything else is the crates.io 0.19.0 source as published; the root Cargo.toml points `[patch.crates-io] bevy_light` here.
- This copy can go once upstream sizes the lists on every thread (or bounds-checks the gather) in the Bevy release flanks moves to.
