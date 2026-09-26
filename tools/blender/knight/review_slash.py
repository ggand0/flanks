"""Render the measured knight rig with a reusable diagonal sword clip.

Run in background Blender. Use --poses for key frames or --view to render one view.
"""
import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[2]
ROOT = REPO / "assets_dev/knight/diagonal_slash_v4"
sys.path.insert(0, str(SOURCE.parent / "melee"))
sys.path.insert(0, str(REPO / "tools/blender/knight"))
import build_knight_textured as support
from sword_slash import SwordSlash

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=Path, default=ROOT)
parser.add_argument("--poses", action="store_true")
parser.add_argument("--view", choices=["oblique", "gameplay"])
args = parser.parse_args(
    sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
)

ROOT = args.out_dir.resolve()


def seconds_at(frame):
    time = frame / 30
    slow_start = 0.35 + SwordSlash.duration + 0.70
    return (
        np.clip(time - 0.35, 0, SwordSlash.duration)
        if time < slow_start
        else np.clip((time - slow_start) * 0.5, 0, SwordSlash.duration)
    )


def main():
    assert bpy.app.background
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(ROOT / "knight.glb"))
    obj = bpy.data.objects["L0"]
    for other in bpy.data.objects:
        if other.type == "MESH" and other != obj:
            other.hide_render = True
    mesh = obj.data
    positions = np.array([(v.co.x, v.co.z, -v.co.y) for v in mesh.vertices])
    parts = np.zeros(len(positions), int)
    for loop in mesh.loops:
        parts[loop.vertex_index] = round(mesh.uv_layers[1].data[loop.index].uv.x)
    nodes = {
        o.name: [o.location.x, o.location.z, -o.location.y]
        for o in bpy.data.objects
        if o.name.startswith(("pivot_", "joint_"))
    }
    rig = SwordSlash(nodes)
    (ROOT / "knight.slash.json").write_text(json.dumps(rig.export(), indent=2) + "\n")
    scene = bpy.context.scene
    support.render_setup()
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.film_transparent = False
    scene.render.resolution_x, scene.render.resolution_y = 640, 720
    scene.render.image_settings.color_mode = "RGB"
    scene.render.fps = 30
    obj.data.materials[0] = support.tint_material(mesh.materials[0], support.TEAM_RED)
    obj.data.materials[0].use_backface_culling = True
    obj.shape_key_add(name="Basis")
    mesh.shape_keys.use_relative = False
    expected = []
    frames = int(np.ceil((0.35 + 3 * rig.duration + 0.70 + 0.60) * 30))
    for frame in range(frames):
        p = rig.deform(positions, parts, rig.sample(seconds_at(frame)))
        coords = p[:, [0, 2, 1]] * [1, -1, 1]
        expected.append(coords.astype(np.float32))
        key = obj.shape_key_add(name=f"Frame_{frame:03d}")
        key.interpolation = "KEY_LINEAR"
        key.data.foreach_set("co", coords.astype(np.float32).reshape(-1))
        mesh.shape_keys.eval_time = key.frame
        mesh.shape_keys.keyframe_insert("eval_time", frame=frame + 1)
    for fc in (
        mesh.shape_keys.animation_data.action.layers[0].strips[0].channelbags[0].fcurves
    ):
        for point in fc.keyframe_points:
            point.interpolation = "LINEAR"
    scene.frame_start, scene.frame_end = 1, frames
    max_error = 0.0
    for frame, target in enumerate(expected):
        scene.frame_set(frame + 1)
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        actual = np.array([v.co[:] for v in evaluated.data.vertices])
        max_error = max(max_error, float(np.max(abs(actual - target))))
    assert max_error < 1e-5, max_error
    support.camera(8, -25, (0, -0.12, 1.18), 3.25)
    scene.frame_set(17)
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / "knight_slash_review.blend"))
    (ROOT / "saved_scene_validation.json").write_text(
        json.dumps(
            {
                "frames": frames,
                "max_position_error_m": max_error,
                "playback": [1, 0.5],
            },
            indent=2,
        )
        + "\n"
    )
    for view, elevation, azimuth in [("oblique", 8, -25), ("gameplay", 50, -35)]:
        if args.view and args.view != view:
            continue
        folder = ROOT / f"frames_{view}"
        folder.mkdir(exist_ok=True)
        support.camera(elevation, azimuth, (0, -0.12, 1.18), 3.25)
        diagnostic = [0] + [
            round((0.35 + t) * 30)
            for t in [
                0.12,
                rig.hold_start,
                rig.hold_end,
                rig.hold_end + 0.04,
                rig.hold_end + 0.08,
                rig.hold_end + 0.12,
                rig.hold_end + 0.16,
                0.53,
                rig.duration,
            ]
        ]
        for frame in diagnostic if args.poses else range(frames):
            scene.frame_set(frame + 1)
            scene.render.filepath = str(folder / f"{frame:03d}.png")
            bpy.ops.render.render(write_still=True)
    print("SLASH_REVIEW_DONE", flush=True)


if __name__ == "__main__":
    main()
