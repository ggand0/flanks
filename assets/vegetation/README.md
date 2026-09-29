# Vegetation

Original project trees and shrubs, stored as GLB files through Git LFS: three oak shapes in two foliage colours each, a silver birch and two shrubs.

Grassland plants about 130 trees and 165 shrubs at the default army gap: a broken oak and birch edge along the western margin, shorter groups along the drier eastern margin, a short hedge with an opening where the dry ground crosses the army gap, and a few low shrub groups in the rear margins, all with their whole crowns clear of the deployment zones and the central 640 m of the gap at the largest scale their kind is drawn at. Two copses stand in the rear corners of the deployment zones, beyond the ranks. River plants about 350 trees and 410 shrubs where its forest noise runs high, away from the middle of the field and the river: woods with a core of broad oaks, an edge of leaning oaks and birches over shrubs and a fringe of scrub, birches on the higher ground, and small shrub groups along both banks. Sandbox places six oaks, two warm birches, twenty low shrubs and one upright shrub in irregular groups around the centre, with varied rotation and scale.

| File | Height | Near triangles | Middle triangles | Far triangles |
|---|---:|---:|---:|---:|
| `oak_pale.glb`, `oak_lighter.glb` | 13.164 m | 2,670 | 690 | 4 |
| `mature_oak_pale.glb`, `mature_oak_lighter.glb` | 13.719 m | 2,960 | 690 | 4 |
| `leaning_oak_pale.glb`, `leaning_oak_lighter.glb` | 9.001 m | 2,344 | 664 | 4 |
| `silver_birch_warm.glb` | 15.482 m | 1,936 | 450 | 4 |
| `shrub_a.glb` | 1.700 m | 2,548 | 868 | 4 |
| `shrub_b_sandbox.glb` | 0.800 m | 1,854 | 574 | 4 |

Each file contains `L0`, `L1` and `card` nodes under `variant_0`, with identity transforms, metres as units, +Y up and the origin at the trunk base. `TEXCOORD_0` samples the embedded atlas. `TEXCOORD_1` stores bend strength and a foliage marker. `COLOR_0` carries linear material colour with opaque vertex alpha; texture alpha supplies leaf cutouts.

All shapes use separate opaque bark textures and continuous mirrored bark coordinates. Near and middle levels each use two material primitives, one for cutout foliage and one for bark. All far levels use crossed planes with baked colour and volume-normal atlases; the planes include transparent padding beyond the tree bounds.

Each oak shape comes in pale and lighter green foliage, `_pale` and `_lighter`, sharing geometry, dimensions and triangle counts, each with its own matching far-colour bake. The maps mix the two at random. All six share fine warm-grey bark, joined woody branches, uneven root flares and darker base weathering. Root dimensions scale with trunk radius. Bark coordinates follow branch length at 1.8 m per tile with mirrored wrapping. `silver_birch_warm.glb` is a silver birch with warmer green foliage.

`shrub_b_sandbox.glb` is the low spreading shrub, 2.5 m wide, 1.85 m deep and 0.8 m tall: branch-attached leaf sprays and fine inner shoots over the woody branches. Its middle mesh uses 176 enlarged leaf sprays and the same branches, and its far colour and normals are baked from the near canopy. It uses 1024-square foliage colour, 256-square opaque bark, and 512 by 256 far-colour and normal atlases.

`shrub_a.glb` is an upright multi-stem shrub, 2.2 m wide, 1.85 m deep and 1.7 m tall. Seven stems and fourteen forks support 840 fine leaf sprays, using the low shrub's foliage and bark textures. Its middle mesh has 210 enlarged sprays and matching wood; far colour and normals are baked from its own near canopy. Sandbox places one specimen at (14,14); on Grassland it is about one shrub in six, as upright accents and in the hedge. [Build instructions](../../tools/blender/shrub_a/README.md).

The renderer selects detail by projected maximum mesh dimension including instance scale, with hysteresis and a mesh fallback for steep overhead views. Vegetation is visual only and does not alter terrain or pathfinding. Fetch Git LFS objects when cloning to obtain the models and embedded textures.
