# Diagonal sword slash

`sword_slash.py` authors a right-handed cut from a high right guard toward the low left side. The 0.65-second cycle holds the high guard from 0.19 to 0.24 seconds, reaches contact at 0.36 seconds and returns to a low sword-ready pose. The hold adds 0.05 seconds without shortening the active cut or recovery. The active cut rotates the shoulder while continuously extending the elbow. The blade initially aligns its cutting edge with the travel of a point along the blade. From 0.33 to 0.37 seconds the hand and sword roll 52 degrees around the blade axis, presenting the edge through the finish. This authored follow-through departs from exact edge-to-velocity alignment. The roll releases from 0.41 to 0.51 seconds during recovery. The hand continues below chest height toward the opposite hip. It samples 131 poses and exports local XYZW shoulder, elbow and grip rotations, torso yaw and shield yaw.

`SwordSlash(nodes)` reads `pivot_arm_weapon`, `joint_elbow`, `pivot_weapon` and `pivot_arm_shield`. Hand targets are relative to the shoulder and measured upper-arm-plus-forearm length, so rigs with different arm proportions can use the same motion path. Rest swords point along glTF +Z. Part IDs for arm, sword and shield can be supplied to the constructor; the defaults are 1, 8 and 5.

The deformation adapter supports the knight and man-at-arms convention of a connected sleeve plus a separate weapon. It blends the sleeve around the elbow and grip, applies the grip rotation to the hand and sword together, and distributes a small torso twist above the waist while keeping the legs fixed. This differs from a rigid part per bone. A unit with separately tagged forearm and hand parts needs its own deformation adapter; a spear's geometry and contact path also require a separate weapon profile. Do not apply sword-tip targets to a long spear unchanged.

Interpolation uses shortest-path normalized quaternions. `sample()` and `deform()` define the playback contract. This table is independent of the existing pitch-only stab and overhead attack format.

## Knight preview

From the repository root:

```sh
blender --background --factory-startup --python-exit-code 1 --python tools/blender/knight/prepare_slash.py
python3 assets_dev/_setup_smoke/glb_inspect.py assets_dev/knight/diagonal_slash_v4/knight.glb
blender --background --factory-startup --python-exit-code 1 --python tools/blender/knight/check_slash.py
blender --background --factory-startup --python-exit-code 1 --python tools/blender/knight/review_slash.py -- --view oblique
blender --background --factory-startup --python-exit-code 1 --python tools/blender/knight/review_slash.py -- --view gameplay
ffmpeg -y -framerate 30 -i assets_dev/knight/diagonal_slash_v4/frames_oblique/%03d.png -framerate 30 -i assets_dev/knight/diagonal_slash_v4/frames_gameplay/%03d.png -filter_complex hstack=inputs=2 -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart assets_dev/knight/diagonal_slash_v4/knight_diagonal_slash.mp4
```

The three Blender scripts accept `--out-dir`. Preparation reads the installed knight and adds a closed leather cuff overlap using its existing atlas. The preview contains 108 frames at 30 fps: one cut at normal speed followed by one at half speed. The right view uses a camera 50 degrees above the horizon. The saved `.blend` contains the same motion. `knight.slash.json` stores the sampled rig and clip; `motion_validation.json` records grip and hinge errors, rigid sword lengths, fixed feet, loop closure and sword intersections with the body, shield and legs.
