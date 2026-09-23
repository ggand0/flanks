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

Outputs go to `assets_dev/<kind>/lod_l2_v1/`. Append `-- --out path/to/kind.glb` to choose another destination. Inputs are the GLBs and animation sidecars in `assets/units/`; the build does not require an authoring scene or a baked PNG outside those GLBs. The component labels under `source_labels/` are bound to each L0 by SHA-256, so changing the input asset requires updating its labels.

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
