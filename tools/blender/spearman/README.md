# Spearman textured v3: jointed spear arm and stab

This directory contains reproducible sources; generated files go to
`assets_dev/spearman/rebuild_v3/` by default.

`review_motion.py` renders side and oblique frame sequences and saves a playable
96-frame `spear_stab_review.blend` at 30 fps. Encode the video with the command
below, then open the generated `review.html` for speed controls and scrubbing.

The source files and motion tables are tracked; generated scenes and GLBs
remain in ignored working directories.

## Changes and evidence

- Replace the weapon arm's continuous bendable sleeve with explicitly assigned
  upper arm, forearm and hand. The shoulder socket belongs to the body, the
  elbow cover to the upper arm, and the wrist cover to the hand. All are closed
  volumes with their adjacent attachment rings buried inside them.
- Author a bent carry/guard with travel available for a stab. Keep the actual
  anatomical wrist distinct from the point where the fingers grip the shaft.
- Use chained rotations. All arm triangles remain rigid; the wrist and spear
  move with the elbow and shoulder. The spear rotates about its contact inside
  the fist. During carry-to-guard it slides through that contact; throughout
  the attack it stays held 0.45 m from the butt.
- Author grip targets and bake them into two small 33-pose angle tables. Runtime
  evaluation needs two table samples, interpolation and joint transforms, not
  an IK solver. Endpoint-only interpolation produced a measured 12 cm dip,
  which is why the intermediate samples are necessary.
- Keep the face, body, shield, kit and legs from v2. Rebuild the atlas from the
  same project-generated source textures because the arm geometry/UVs changed.

Actual GLB measurements: L0 **2,954 triangles**, character **1.800 m**, one opaque
material, embedded 2048² RGBA atlas, `COLOR_0` VEC4 and both UV channels. Every
vertex has a part; every triangle belongs entirely to one rigid part. There
are no authored L1–L3 meshes at this stage.

Across **2,504 poses** covering carry-to-guard, wind-up, strike and recovery:

| Measurement | Result |
|---|---:|
| Forward spear-tip stroke | 0.49487 m |
| Tip vertical deviation during the stroke | 0.002031 m |
| Minimum shoulder attachment inset | 0.012071 m |
| Minimum elbow attachment inset | 0.010607 m |
| Minimum wrist attachment inset | 0.002250 m |
| Maximum adjacent-joint disagreement | < 1e-12 m |
| Maximum grip/shaft-axis disagreement | < 1e-12 m |
| Maximum triangle-edge length change | < 1e-12 m |
| Pose mismatch at all three phase boundaries | 0 |

The tiny numerical errors above use float64 authoring math. They are **not a
claim of GPU validation**. `check_motion.py` reads the GLB bytes, verifies that
every attachment probe exists in the exported mesh, tests containment in the
actual joint-cover convex polyhedra, and tests every triangle edge over the
full sampled cycle. See `motion_validation.json` for exported vertex coverage
and all pivots. The static render alone is not the evidence.

## Contract

Existing IDs retain their meanings. `arm_spear` (4) contains the upper weapon
arm only; `weapon` (8) contains only the spear. The forearm and hand are their
own parts:

| ID | Part | Pivot, glTF XYZ in metres |
|---|---|---|
| 4 | arm_spear | (-0.224, 1.435, 0.000) |
| 9 | forearm_spear | (-0.310, 1.200, -0.070) |
| 10 | hand_spear | (-0.360, 1.205, 0.160) |
| 8 | weapon | (-0.373, 1.185, 0.219) |

`TEXCOORD_1` stores the integer part ID and that part's pivot height, with the
Blender V flip handled before export. `joint_elbow` and `joint_wrist` are also
exported for inspection. The engine reads the stab from `spear_stab.json`,
installed next to the model as `spearman.stab.json`.

The motion covers the stationary weapon-arm stab. Shieldwall shoulder
motion, sword attacks, locomotion and full-body stepping are separate tasks.

## Rebuild

Run in a separate headless Blender process; these scripts never touch the
user's open Blender scene:

```sh
blender --background --factory-startup --threads 4 --python-exit-code 1 --python tools/blender/spearman/build_spearman.py
python3 tools/blender/glb_inspect.py assets_dev/spearman/rebuild_v3/spearman.glb
blender --background --factory-startup --threads 4 --python-exit-code 1 --python tools/blender/spearman/review_motion.py
python3 tools/blender/spearman/check_motion.py
blender --background --factory-startup --threads 4 --python-exit-code 1 --python tools/blender/spearman/review_motion.py -- --verify-only
ffmpeg -v error -framerate 30 -i assets_dev/spearman/rebuild_v3/frames_side/%03d.png -framerate 30 -i assets_dev/spearman/rebuild_v3/frames_oblique/%03d.png -filter_complex '[0:v][1:v]hstack=inputs=2' -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart assets_dev/spearman/rebuild_v3/spear_stab.mp4
```

`build_spearman.py` uses the spec's exact export call. `motion.py` is the source
of the pose tables; `review_motion.py` applies it to the reimported GLB for
viewing. All geometry and texture inputs were generated for this project.

The Blender scripts accept `-- --out-dir <directory>`; the system-Python
validator accepts `--out-dir <directory>`. They read source PNGs from this
tracked directory and use `tools/blender/glb_inspect.py`, so they do not need
an earlier ignored model directory. Blender supplies its NumPy dependency;
`check_motion.py` needs NumPy and SciPy in system Python.

The JSON motion table is checked against `motion.export_clip()` by the validator.
Regenerate `spear_stab.json` whenever `motion.py` changes. The `.blend` is a
review scene; the runtime deliverable is the static GLB plus these pose tables.
