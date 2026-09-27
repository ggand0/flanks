# Grassland trees

Four original project models and textures, stored as GLB files through Git LFS. The grassland map displays them at their authored scale along its western margin.

| File | Height | Near triangles | Middle triangles | Far triangles |
|---|---:|---:|---:|---:|
| `oak.glb` | 13.164 m | 2,672 | 636 | 4 |
| `mature_oak.glb` | 13.719 m | 2,912 | 636 | 4 |
| `leaning_oak.glb` | 9.001 m | 2,340 | 560 | 4 |
| `silver_birch.glb` | 15.482 m | 1,936 | 450 | 4 |

Each file contains `L0`, `L1` and `card` nodes under `variant_0`, with identity transforms, metres as units, +Y up and the origin at the trunk base. `TEXCOORD_0` samples the embedded atlas. `TEXCOORD_1` stores bend strength and a foliage marker. `COLOR_0` carries linear material colour with opaque vertex alpha; texture alpha supplies leaf cutouts.

The mature oak uses subdued green foliage with a separate opaque bark texture and continuous mirrored bark coordinates. Its near and middle levels each use two material primitives. The other trees use combined bark and foliage atlases. All far levels use crossed planes with baked colour and volume-normal atlases; the planes include transparent padding beyond the tree bounds.

The renderer selects detail by projected height, with hysteresis and a mesh fallback for steep overhead views. Vegetation is visual only and does not alter terrain or pathfinding. Fetch Git LFS objects when cloning to obtain the models and embedded textures.
