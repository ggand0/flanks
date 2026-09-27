# Grassland vegetation

Four original project tree shapes, four low shrub versions and their textures, stored as GLB files through Git LFS. The grassland map displays them at their authored scale along its western margin.

| File | Height | Near triangles | Middle triangles | Far triangles |
|---|---:|---:|---:|---:|
| `oak.glb` | 13.164 m | 2,672 | 636 | 4 |
| `mature_oak.glb` | 13.719 m | 2,912 | 636 | 4 |
| `leaning_oak.glb` | 9.001 m | 2,340 | 560 | 4 |
| `silver_birch.glb` | 15.482 m | 1,936 | 450 | 4 |
| `shrub_b.glb` | 0.800 m | 416 | 120 | 4 |
| `shrub_b_v2.glb` | 0.800 m | 894 | 120 | 4 |
| `shrub_b_v3.glb` | 0.800 m | 1,086 | 120 | 4 |
| `shrub_b_v4.glb` | 0.800 m | 1,854 | 120 | 4 |

Each file contains `L0`, `L1` and `card` nodes under `variant_0`, with identity transforms, metres as units, +Y up and the origin at the trunk base. `TEXCOORD_0` samples the embedded atlas. `TEXCOORD_1` stores bend strength and a foliage marker. `COLOR_0` carries linear material colour with opaque vertex alpha; texture alpha supplies leaf cutouts.

All shapes use separate opaque bark textures and continuous mirrored bark coordinates. Near and middle levels each use two material primitives, one for cutout foliage and one for bark. All far levels use crossed planes with baked colour and volume-normal atlases; the planes include transparent padding beyond the tree bounds.

Each oak shape comes in natural and lighter green: `oak.glb` / `oak_lighter.glb`, `mature_oak.glb` / `mature_oak_lighter.glb`, and `leaning_oak.glb` / `leaning_oak_lighter.glb`. Palette variants share geometry, dimensions and triangle counts. Each has its own matching far-colour bake. The map displays the six oaks in those pairs, followed by `silver_birch_warm.glb`, a silver birch with warmer green foliage. `silver_birch.glb` retains the cooler grey-green palette for other maps. Both birch palettes share geometry and bark, with their own matching far-colour bakes.

Files ending in `_original.glb` preserve the earlier pale colours. The mature oak's original-colour variant uses corrected opaque bark; the upright oak, leaning oak and birch originals retain their earlier combined material. These files are not placed on the map.

All shrubs are 2.5 m wide and 1.85 m deep. `shrub_b.glb` has rounded foliage lobes. `shrub_b_v2.glb` has 224 branch-attached leaf sprays and a more open crown. `shrub_b_v3.glb` adds 96 small inner shoots. `shrub_b_v4.glb` fills more of the canopy with another 384 fine shoots, preserving the earlier foliage, bark and outer bounds; it appears between the smaller oak and warm birch in the specimen row. They share the same middle mesh and far cards. Each uses 1024-square foliage colour, 256-square opaque bark, and 512 by 256 far-colour and normal atlases.

The renderer selects detail by projected maximum mesh dimension, with hysteresis and a mesh fallback for steep overhead views. Vegetation is visual only and does not alter terrain or pathfinding. Fetch Git LFS objects when cloning to obtain the models and embedded textures.
