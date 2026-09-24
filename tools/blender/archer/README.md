# Archer held pose and raise

Build a textured 1.80 m archer with a side-on shooting stance. The 1.5-second clip raises a held, nocked bow from horizontal to 35 degrees: 25 degrees of torso lean and 10 degrees of bow-arm lift. The drawing arm holds its pose relative to the torso. There is no string-pulling or release event.

A separate 0.6-second release clip starts at the raised endpoint. The held arrow disappears at time zero, the string returns to its braced position over 75 ms, and the drawing hand follows back and outward beside the jaw. The bow arm holds aim for 180 ms before the torso and bow arm each settle by 2 degrees.

The 5.8-second reload lowers the bow, reaches to the right-hip quiver, extracts an arrow and brings it around the outside of the shoulder to the string. The elbow stays low during pickup and early extraction. After extraction, the hand follows one inward curve in front of the chest toward the string. The arrow turns forward with the hand. Nocking completes at 3.8 seconds. The final two seconds return to the horizontal drawn-bow pose, ready for the raise.

From the repository root:

```sh
blender --background --factory-startup --python-exit-code 1 --python tools/blender/archer/build_archer.py
python3 tools/blender/archer/check_motion.py
blender --background --factory-startup --python-exit-code 1 --python tools/blender/archer/review_motion.py
blender --background --factory-startup --python-exit-code 1 --python tools/blender/archer/check_scene.py
ffmpeg -y -framerate 30 -i assets_dev/archer/rebuild_raise_v4/frames_oblique/%03d.png -framerate 30 -i assets_dev/archer/rebuild_raise_v4/frames_side/%03d.png -filter_complex hstack=inputs=2 -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart assets_dev/archer/rebuild_raise_v4/archer_raise.mp4
```

Open `assets_dev/archer/rebuild_raise_v4/review.html` for playback and frame stepping, or `archer_raise_review.blend` for the 85-frame sequence. The mesh has 2,987 L0 triangles and one embedded 2048² atlas. The Python validation script requires NumPy and SciPy.

## Parts and transforms

| IDs | Parts |
|---|---|
| 0, 2, 3 | Lower body and legs |
| 1, 9, 10 | Draw upper arm, forearm and hand |
| 6, 11, 12 | Bow upper arm, forearm and hand |
| 8, 14, 15 | Bow grip and limbs |
| 13, 17 | Upper and lower string segments |
| 16 | Held arrow |
| 18, 19 | Head and torso |

Every non-body part has a `pivot_<part>` node. `TEXCOORD_1` holds its integer part ID and pivot height; the atlas alpha is team tint.

`archer.raise.json` uses version 3 with 65-sample `raise` and `release` tables and a 257-sample `reload` table. The engine reads it installed next to the model as `archer.shoot.json`. Each row contains six local XYZW quaternions (draw shoulder, elbow, wrist, then bow shoulder, elbow, wrist), followed by bow yaw, limb bend, arrow visibility, body yaw, torso pitch and bow pitch. Interpolate quaternions along the shortest path and normalize them before composing parent and child rotations. Sample each clip using its own row count. `motion.py` defines the complete rigid transforms in glTF metres, +Y up and +Z forward. The release event hides the held arrow immediately; a projectile is rendered separately by the game.

`arrow_samples.reload` contains 257 rows with a nock XYZ position, XYZW orientation and attachment weight. These transforms are local to the torso, before body yaw and torso pitch. Blend the free arrow transform toward the bowstring-derived arrow transform using that weight, with shortest-path normalized quaternion interpolation. This lets the arrow follow the drawing hand during extraction and transfer, then attach to the string without a jump. Visibility is a discrete event: the held arrow becomes visible 1.65 seconds into reload. The bundled arrows inside the quiver remain decorative geometry.

The authoring solver uses a shared anatomical elbow axis and aligns the wrist with the forearm. The drawing-arm rotations remain constant during the lift. Body yaw establishes the stance; torso pitch raises the upper body around the waist, while the head faces the shot. Shoulder covers belong to the torso, elbow covers to the upper arms and wrist covers to the hands. The held arrow and two fixed-length string halves follow their nock and bow contacts.

The motion check samples 1,001 poses per clip and measures joint coverage, hinge axes, wrist alignment, torso clearance, arrow-shaft clearance, grip contact, string length, rigid edges and clip boundaries.

## Release preview

Pass `--release` to the preview and saved-scene checker to play both clips. The sequence holds horizontally for 0.4 seconds, raises for 1.5 seconds, holds for 0.3 seconds, releases for 0.6 seconds and holds the final pose. The shot occurs at 2.20 seconds. Both views include the full relaxed bow.

```sh
blender --background --factory-startup --python-exit-code 1 --python tools/blender/archer/review_motion.py -- --out-dir assets_dev/archer/release_v5 --release
blender --background --factory-startup --python-exit-code 1 --python tools/blender/archer/check_scene.py -- --out-dir assets_dev/archer/release_v5 --release
ffmpeg -y -framerate 30 -i assets_dev/archer/release_v5/frames_oblique/%03d.png -framerate 30 -i assets_dev/archer/release_v5/frames_side/%03d.png -filter_complex hstack=inputs=2 -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart assets_dev/archer/release_v5/archer_release.mp4
```

The preview expects `archer.glb` in the specified directory. Build it with `build_archer.py -- --out <directory>/archer.glb`, then run `check_motion.py --out-dir <directory>` to write the tables and measurements. The release preview contains 109 frames at 30 fps. Its player provides a replay-release button, slow playback and frame stepping.

Pass `--baseline <previous-table.json>` to `check_motion.py` to verify exact equality of every clip present in an earlier table.

## Reload preview

Pass `--reload` to preview the release endpoint, 5.8-second reload and following raise. Reload begins at 0.40 seconds, horizontal ready begins at 6.20 seconds and the raise starts at 6.60 seconds. The sequence has 265 frames at 30 fps.

```sh
python3 tools/blender/archer/check_motion.py --out-dir assets_dev/archer/reload_v11 --baseline assets_dev/archer/reload_v11/accepted_raise_release.json --reload-baseline assets_dev/archer/reload_v11/previous_reload.json --reload-unchanged-from 3.83
blender --background --factory-startup --python-exit-code 1 --python tools/blender/archer/review_motion.py -- --out-dir assets_dev/archer/reload_v11 --reload
blender --background --factory-startup --python-exit-code 1 --python tools/blender/archer/check_scene.py -- --out-dir assets_dev/archer/reload_v11 --reload
ffmpeg -y -framerate 30 -i assets_dev/archer/reload_v11/frames_oblique/%03d.png -framerate 30 -i assets_dev/archer/reload_v11/frames_side/%03d.png -filter_complex hstack=inputs=2 -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart assets_dev/archer/reload_v11/archer_reload.mp4
```

Pass `--reload-baseline <previous-table.json>` to check that the arrow path and reload poses from 2.65 seconds onward match a previous version. Use `--reload-unchanged-from <seconds>` to select another boundary. The motion check also measures draw-forearm clearance and verifies a lowered elbow during early extraction.
