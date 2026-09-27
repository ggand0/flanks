# Vegetation

Four original project tree shapes, low shrub variants and their textures, stored as GLB files through Git LFS. Grassland displays a specimen row at authored scale along its western margin. Sandbox places six oaks, two warm birches and twenty low shrubs in irregular groups around the centre, with varied rotation and scale.

| File | Height | Near triangles | Middle triangles | Far triangles |
|---|---:|---:|---:|---:|
| `oak.glb` | 13.164 m | 2,670 | 690 | 4 |
| `oak_trunk.glb` | 13.164 m | 2,670 | 690 | 4 |
| `oak_trunk_light.glb` | 13.164 m | 2,670 | 690 | 4 |
| `mature_oak.glb` | 13.719 m | 2,960 | 690 | 4 |
| `mature_oak_trunk.glb` | 13.719 m | 2,960 | 690 | 4 |
| `leaning_oak.glb` | 9.001 m | 2,344 | 664 | 4 |
| `silver_birch.glb` | 15.482 m | 1,936 | 450 | 4 |
| `shrub_b.glb` | 0.800 m | 416 | 120 | 4 |
| `shrub_b_v2.glb` | 0.800 m | 894 | 120 | 4 |
| `shrub_b_v3.glb` | 0.800 m | 1,086 | 120 | 4 |
| `shrub_b_v4.glb` | 0.800 m | 1,854 | 120 | 4 |
| `shrub_b_sandbox.glb` | 0.800 m | 1,854 | 574 | 4 |

Each file contains `L0`, `L1` and `card` nodes under `variant_0`, with identity transforms, metres as units, +Y up and the origin at the trunk base. `TEXCOORD_0` samples the embedded atlas. `TEXCOORD_1` stores bend strength and a foliage marker. `COLOR_0` carries linear material colour with opaque vertex alpha; texture alpha supplies leaf cutouts.

All shapes use separate opaque bark textures and continuous mirrored bark coordinates. Near and middle levels each use two material primitives, one for cutout foliage and one for bark. All far levels use crossed planes with baked colour and volume-normal atlases; the planes include transparent padding beyond the tree bounds.

Each oak shape comes in natural and lighter green: `oak.glb` / `oak_lighter.glb`, `mature_oak.glb` / `mature_oak_lighter.glb`, and `leaning_oak.glb` / `leaning_oak_lighter.glb`. Palette variants share geometry, dimensions and triangle counts. Each has its own matching far-colour bake. Grassland displays the six oaks in those pairs, followed by `silver_birch_warm.glb`, a silver birch with warmer green foliage. `silver_birch.glb` retains the cooler grey-green palette for other maps. Both birch palettes share geometry and bark, with their own matching far-colour bakes.

Files ending in `_original.glb` preserve the earlier pale colours. The mature oak's original-colour variant uses corrected opaque bark; the upright oak, leaning oak and birch originals retain their earlier combined material. These files are not placed on the map.

All shrubs are 2.5 m wide and 1.85 m deep. `shrub_b.glb` has rounded foliage lobes. `shrub_b_v2.glb` has 224 branch-attached leaf sprays and a more open crown. `shrub_b_v3.glb` adds 96 small inner shoots. `shrub_b_v4.glb` fills more of the canopy with another 384 fine shoots, preserving the earlier foliage, bark and outer bounds; it appears between the smaller oak and warm birch in the specimen row. They share the same middle mesh and far cards. Each uses 1024-square foliage colour, 256-square opaque bark, and 512 by 256 far-colour and normal atlases.

`shrub_b_sandbox.glb` retains the v4 near mesh and textures. Its middle mesh uses 176 enlarged leaf sprays and the same woody branches; its far colour and normals are baked from v4. Grassland keeps `shrub_b_v4.glb`.

All six oak shape/foliage combinations share fine warm-grey bark, joined woody branches, uneven root flares and darker base weathering. Root dimensions scale with trunk radius. Bark coordinates follow branch length at 1.8 m per tile with mirrored wrapping. Near and middle foliage retains its authored geometry and colours; each far level includes the matching wood and foliage. Both Grassland and Sandbox use these assets.

`mature_oak_trunk.glb` and `oak_trunk.glb` retain the darker grey-brown bark on the revised mature and upright trunks for other maps. `oak_trunk_light.glb` preserves the standalone upright light-bark specimen. These three files are not loaded by the default scene.

The renderer selects detail by projected maximum mesh dimension including instance scale, with hysteresis and a mesh fallback for steep overhead views. Vegetation is visual only and does not alter terrain or pathfinding. Fetch Git LFS objects when cloning to obtain the models and embedded textures.
