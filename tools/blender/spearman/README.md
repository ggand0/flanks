# Spearman textured v3: L0 arm articulation prototype

L0 motion reviewed positively by the owner on 2026-09-23. Engine integration
and final LOD acceptance remain pending. This directory contains reproducible
sources; generated files go to `assets_dev/spearman/rebuild_v3/` by default.
The original reviewed outputs remain in `assets_dev/spearman/textured_v3/`.

`review_motion.py` renders side and oblique frame sequences and saves a playable
96-frame `spear_stab_review.blend` at 30 fps. Encode the video with the command
below, then open the generated `review.html` for speed controls and scrubbing.

The asset is **not compatible with the current engine without the integration
described below**. No `src/`, renderer, accepted GLB or earlier generator was
edited for this prototype. The source files and motion tables are tracked; generated scenes and GLBs remain in ignored working directories.
L1–L3 and full acceptance remain pending.

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

## Proposed contract extension

Existing IDs retain their meanings. `arm_spear` (4) now contains the upper
weapon arm only; `weapon` (8) remains only the spear. Add:

| ID | Part | Pivot, glTF XYZ in metres |
|---|---|---|
| 4 | arm_spear | (-0.224, 1.435, 0.000) |
| 9 | forearm_spear | (-0.310, 1.200, -0.070) |
| 10 | hand_spear | (-0.360, 1.205, 0.160) |
| 8 | weapon | (-0.373, 1.185, 0.219) |

IDs 9 and 10 are proposed additions for this isolated prototype, not changes
to the approved spec. The byte format is unchanged: `TEXCOORD_1` stores the
integer part ID and that part's pivot height, with the Blender V flip handled
before export. `joint_elbow` and `joint_wrist` are also exported for inspection.

## Engine integration, pending permission

1. `src/unit_glb.rs`: recognize the two additional part names, load the wrist
   and new joint positions, validate the complete rig, and preserve membership
   in derived LODs. Select the new animation only for assets carrying that rig.
2. `src/unit_meshes.rs` and both renderer bucket layouts: extend the rig uniform
   with the wrist information and an explicit articulation flag. Preserve the
   current fallback path for legacy assets/code-built meshes.
3. `src/shaders/unit_instancing.wgsl`: use exact part membership and the shared
   shoulder → elbow → wrist chain. Replace the broad `part > 7.5` weapon test
   for this path, since it would incorrectly treat IDs 9/10 as weapons. Apply
   upper-body transforms consistently to all these parts. Sample the authored
   wind-up/recovery tables from `spear_stab.json`.
4. Attack signals in `src/render_units.rs` / `src/shaders/unit_build.wgsl`:
   preserve combat readiness at wind-up start. Currently negative standby
   values carry readiness, but positive wind-up does not: at zero progress the
   spear's `level` becomes zero. Do not hide that discontinuity with mesh
   overlap. Preserve the existing uncommitted follow-through work; provide
   normalized phase consistently for CPU and GPU paths, including charges.
5. Review in game using a development path override, including low frame rates,
   interrupted attacks, readiness transitions, moving attacks and derived LODs.
   Accept the L0 motion before authoring final LODs or promoting the GLB.

This prototype covers the stationary weapon-arm stab. Shieldwall shoulder
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
