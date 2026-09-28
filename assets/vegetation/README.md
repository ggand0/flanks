# Vegetation

Four original project tree shapes, low shrub variants and their textures, stored as GLB files through Git LFS. Grassland plants about 130 trees and 165 shrubs at the default army gap: a broken oak and birch edge along the western margin, shorter groups along the drier eastern margin, a short hedge with an opening where the dry ground crosses the army gap, and a few low shrub groups in the rear margins, all with their whole crowns clear of the deployment zones and the central 640 m of the gap at the largest scale their kind is drawn at. Two copses belong at the outer ends of the army gap; until trees collide with soldiers they stand in the rear corners of the deployment zones, beyond the ranks. River plants about 350 trees and 410 shrubs where its forest noise runs high, away from the middle of the field and the river: woods with a core of broad oaks, an edge of leaning oaks and birches over shrubs and a fringe of scrub, birches on the higher ground, and small shrub groups along both banks. Sandbox places six oaks, two warm birches, twenty low shrubs and one upright shrub in irregular groups around the centre, with varied rotation and scale.

| File | Height | Near triangles | Middle triangles | Far triangles |
|---|---:|---:|---:|---:|
| `oak.glb` | 13.164 m | 2,670 | 690 | 4 |
| `oak_pale.glb` | 13.164 m | 2,670 | 690 | 4 |
| `oak_trunk.glb` | 13.164 m | 2,670 | 690 | 4 |
| `oak_trunk_light.glb` | 13.164 m | 2,670 | 690 | 4 |
| `mature_oak.glb` | 13.719 m | 2,960 | 690 | 4 |
| `mature_oak_pale.glb` | 13.719 m | 2,960 | 690 | 4 |
| `mature_oak_trunk.glb` | 13.719 m | 2,960 | 690 | 4 |
| `leaning_oak.glb` | 9.001 m | 2,344 | 664 | 4 |
| `leaning_oak_pale.glb` | 9.001 m | 2,344 | 664 | 4 |
| `silver_birch.glb` | 15.482 m | 1,936 | 450 | 4 |
| `shrub_a.glb` | 1.700 m | 2,548 | 868 | 4 |
| `shrub_b.glb` | 0.800 m | 416 | 120 | 4 |
| `shrub_b_v2.glb` | 0.800 m | 894 | 120 | 4 |
| `shrub_b_v3.glb` | 0.800 m | 1,086 | 120 | 4 |
| `shrub_b_v4.glb` | 0.800 m | 1,854 | 120 | 4 |
| `shrub_b_sandbox.glb` | 0.800 m | 1,854 | 574 | 4 |

Each file contains `L0`, `L1` and `card` nodes under `variant_0`, with identity transforms, metres as units, +Y up and the origin at the trunk base. `TEXCOORD_0` samples the embedded atlas. `TEXCOORD_1` stores bend strength and a foliage marker. `COLOR_0` carries linear material colour with opaque vertex alpha; texture alpha supplies leaf cutouts.

All shapes use separate opaque bark textures and continuous mirrored bark coordinates. Near and middle levels each use two material primitives, one for cutout foliage and one for bark. All far levels use crossed planes with baked colour and volume-normal atlases; the planes include transparent padding beyond the tree bounds.

Each oak shape comes in natural and lighter green: `oak.glb` / `oak_lighter.glb`, `mature_oak.glb` / `mature_oak_lighter.glb`, and `leaning_oak.glb` / `leaning_oak_lighter.glb`. Palette variants share geometry, dimensions and triangle counts. Each has its own matching far-colour bake. `silver_birch_warm.glb` is a silver birch with warmer green foliage. `silver_birch.glb` retains the cooler grey-green palette for other maps. Both birch palettes share geometry and bark, with their own matching far-colour bakes.

Three additional files, `oak_pale.glb`, `mature_oak_pale.glb` and `leaning_oak_pale.glb`, restore the archived pale foliage colours on the current warm-grey trunks. Their geometry and triangle counts match the corresponding natural/lighter pair; distant colour is baked for each pale variant. Together these provide nine current-trunk oak variants. Grassland and Sandbox use the pale and lighter oaks, mixed at random.

Files ending in `_original.glb` preserve the earlier pale colours. The mature oak's original-colour variant uses corrected opaque bark; the upright oak, leaning oak and birch originals retain their earlier combined material. These files are not placed on the map.

The low B shrubs are 2.5 m wide and 1.85 m deep. `shrub_b.glb` has rounded foliage lobes. `shrub_b_v2.glb` has 224 branch-attached leaf sprays and a more open crown. `shrub_b_v3.glb` adds 96 small inner shoots. `shrub_b_v4.glb` fills more of the canopy with another 384 fine shoots, preserving the earlier foliage, bark and outer bounds. They share the same middle mesh and far cards. Each uses 1024-square foliage colour, 256-square opaque bark, and 512 by 256 far-colour and normal atlases.

`shrub_b_sandbox.glb` retains the v4 near mesh and textures. Its middle mesh uses 176 enlarged leaf sprays and the same woody branches; its far colour and normals are baked from v4. Grassland and Sandbox use it for the low shrubs.

All nine current oak shape/foliage combinations share fine warm-grey bark, joined woody branches, uneven root flares and darker base weathering. Root dimensions scale with trunk radius. Bark coordinates follow branch length at 1.8 m per tile with mirrored wrapping. Near and middle foliage retains its authored geometry and colours; each far level includes the matching wood and foliage. Both Grassland and Sandbox use these assets.

`mature_oak_trunk.glb` and `oak_trunk.glb` retain the darker grey-brown bark on the revised mature and upright trunks for other maps. `oak_trunk_light.glb` preserves the standalone upright light-bark specimen. These three files are not loaded by the default scene.

`shrub_a.glb` is an upright multi-stem shrub, 2.2 m wide, 1.85 m deep and 1.7 m tall. Seven stems and fourteen forks support 840 fine leaf sprays, using B's foliage and bark textures. Its middle mesh has 210 enlarged sprays and matching wood; far colour and normals are baked from its own near canopy. Sandbox places one specimen at (14,14); on Grassland it is about one shrub in six, as upright accents and in the hedge. [Build instructions](../../tools/blender/shrub_a/README.md).

The renderer selects detail by projected maximum mesh dimension including instance scale, with hysteresis and a mesh fallback for steep overhead views. Vegetation is visual only and does not alter terrain or pathfinding. Fetch Git LFS objects when cloning to obtain the models and embedded textures.
