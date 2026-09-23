# Distant unit meshes

These Blender scripts append a closed L2 mesh to the installed man-at-arms, spearman or archer GLB. They preserve the original L0 binary data, atlas, material and pivot nodes. Each level uses the same texture atlas and part IDs. The spearman and archer builds also copy the existing animation sidecar.

| Kind | L2 triangles | Parts retained |
|---|---:|---|
| Man-at-arms | 248 | Body, legs, arms, shield and sword |
| Spearman | 250 | Body, legs, shield arm, jointed spear arm and spear |
| Archer | 250 | Body, torso, head, legs, both arm chains, bow grip and bow limbs |

The archer omits the held arrow and both string halves. The hip quiver and its arrow fan remain. Human head height is 1.80 m; the spear reaches 2.52 m.

Run each build in a separate background Blender process:

```sh
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/build_man_at_arms_l2.py
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/build_spearman_l2.py
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/build_archer_l2.py
```

Outputs go to `assets_dev/<kind>/lod_l2_v1/`. Append `-- --out path/to/kind.glb` to choose another destination. Inputs are the GLBs and animation sidecars in `assets/units/`; the build does not require an authoring scene or a baked PNG outside those GLBs. The component labels under `source_labels/` are bound to each input GLB by SHA-256, so changing the input asset requires updating its labels.

Local atlas samples establish each triangle's colour. A visibility pass across eight facings at 50 degrees then fits the mean colour and team share of each part to L0, in linear light. L2 stores those samples as constant per-triangle atlas UVs. The embedded atlas stays unchanged.

Inspect an exported candidate:

```sh
python3 tools/blender/lods/verify.py \
  assets_dev/spearman/lod_l2_v1/spearman.glb \
  --source assets/units/spearman.glb
python3 tools/blender/glb_inspect.py assets_dev/spearman/lod_l2_v1/spearman.glb
python3 tools/blender/inspect_surfaces.py assets_dev/spearman/lod_l2_v1/spearman.glb
```

The verifier checks binary preservation, channels, scale, part membership, pivots, closed geometry and shield boundaries. Existing L0 defects remain separate from the L2 report.

Render and compose the size comparisons:

```sh
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/render_review.py -- \
  assets_dev/spearman/lod_l2_v1/spearman.blend
python3 tools/blender/lods/compose_review.py assets_dev/spearman/lod_l2_v1
```

`render_motion.py` compares the existing attacks for L0 and L2; `check_joints.py` checks a 0 to 60 degree pivot sweep. Both accept the candidate blend after `--`. `compose_motion.py` creates paired frames for a video. The rendered reviews enable backface culling, including when an inherited material is double-sided. Python inspection and composition scripts require NumPy and Pillow.

## Knight L1 and L3

`build_knight_l1_l3.py` reads the installed knight GLB containing L0 and L2, then appends L1 and L3 while preserving all original buffer bytes, nodes, materials and atlas data. L1 has 678 triangles and keeps body, both arms, both legs and weapon, with a visor, closed shield rim, wood back and straps. L3 has 52 triangles, all part 0: a tapered torso, flat-crowned five-sided helm, two leg masses and a closed triangular shield. Its atlas colours are fitted to the complete L0's visible colour and team share across eight facings.

```sh
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/build_knight_l1_l3.py
python3 tools/blender/lods/verify_levels.py \
  assets_dev/knight/lod_l1_l3_v1/knight.glb --source assets/units/knight.glb
python3 tools/blender/glb_inspect.py assets_dev/knight/lod_l1_l3_v1/knight.glb
python3 tools/blender/inspect_surfaces.py assets_dev/knight/lod_l1_l3_v1/knight.glb
python3 tools/blender/lods/measure_visibility.py assets_dev/knight/lod_l1_l3_v1/knight.glb
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/render_levels.py -- assets_dev/knight/lod_l1_l3_v1/knight.glb
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/check_joints.py -- assets_dev/knight/lod_l1_l3_v1/knight.blend --level L1
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/render_motion.py -- assets_dev/knight/lod_l1_l3_v1/knight.blend --level L1
python3 tools/blender/lods/compose_motion.py assets_dev/knight/lod_l1_l3_v1
ffmpeg -framerate 15 -i assets_dev/knight/lod_l1_l3_v1/motion_paired/%03d.png \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \
  assets_dev/knight/lod_l1_l3_v1/motion_comparison.mp4
python3 tools/blender/lods/compose_levels.py assets_dev/knight/lod_l1_l3_v1
```

Open `assets_dev/knight/lod_l1_l3_v1/review.html` for all four levels, pixel-size comparisons, silhouette overlays and sword motion. The output directory can be changed with the builder's `--out` argument.

## Infantry and archer L1 and L3

`build_unit_l1_l3.py` appends authored L1 and L3 to the installed man-at-arms, spearman or archer. Original L0/L2 buffer data, atlas, materials and pivot nodes remain unchanged. Spearman and archer sidecars are copied without modification.

| Kind | L1 triangles | L3 triangles |
|---|---:|---:|
| Man-at-arms | 674 | 48 |
| Spearman | 664 | 56 |
| Archer | 680 | 56 |

L1 keeps the animated parts and equipment outlines. The archer retains both arm chains, torso, head, grip, bow limbs and quiver arrow fan; it omits the held arrow and two string halves. L3 uses only part 0, with simplified headgear, torso and legs plus a shield, spear shaft or bow. All new surfaces are closed.

```sh
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/build_unit_l1_l3.py -- --kind man_at_arms
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/build_unit_l1_l3.py -- --kind spearman
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/build_unit_l1_l3.py -- --kind archer
```

Outputs go to `assets_dev/<kind>/lod_l1_l3_v1/`. Use `--out` for another destination or `--source` for another input GLB. The `*_l0_l2.json` manifests retain the original JSON chunk and binary length, allowing `source_base.py` to recover the same source from a subsequently installed four-level asset. Recovery checks the original metadata and complete SHA-256 before building. Changes to L0, L2 or the atlas require refreshed labels.

Use the knight inspection and review commands above with the corresponding kind and output paths. Motion videos use 90 frames at 15 fps for man-at-arms, 64 at 20 fps for spearman, and 150 at 15 fps for archer. `verify_levels.py` checks byte preservation and sidecar identity as well as the new levels. `render_levels.py` compares L0, L1, L2 and L3 at 20, 8 and 3 pixels, from front, rear and side, with both team colours.

Atlas fitting first matches each part's visible mean. The archer L1 also fits the complete figure to account for changes in the projected share of each part. The correction stays within each component's atlas samples; it does not change the atlas or mask format.
