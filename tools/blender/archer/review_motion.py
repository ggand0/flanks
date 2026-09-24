"""Render the exported archer throughout draw and release, and save playback.

Run in a fresh background Blender. Pass --poses for diagnostic frames.
"""
import argparse
import json
from pathlib import Path
import sys
import bpy
import numpy as np

SOURCE = Path(__file__).resolve().parent
DEFAULT_OUT = SOURCE.parents[2] / "assets_dev/archer/rebuild_raise_v4"
parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
parser.add_argument("--poses", action="store_true")
parser.add_argument("--release", action="store_true")
parser.add_argument("--reload", action="store_true")
parser.add_argument("--view", choices=["oblique", "side"])
args = parser.parse_args(
    sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
)
ROOT = args.out_dir.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SOURCE))
import motion
import build_support as support

support.ROOT = ROOT


def pose_at(seconds):
    if args.reload:
        if seconds < 0.4:
            return motion.sample("release", 1)
        if seconds <= 6.2:
            return motion.sample("reload", (seconds - 0.4) / motion.RELOAD_DURATION)
        return motion.sample("raise", (seconds - 6.6) / 1.5)
    if args.release and seconds >= 2.2:
        return motion.sample("release", (seconds - 2.2) / 0.6)
    return motion.sample("raise", (seconds - 0.4) / 1.5)


def main():
    assert bpy.app.background
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(ROOT / "archer.glb"))
    scene = bpy.context.scene
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    obj = next(o for o in scene.objects if o.type == "MESH")
    mesh = obj.data
    parts = np.zeros(len(mesh.vertices), dtype=int)
    for loop in mesh.loops:
        parts[loop.vertex_index] = round(mesh.uv_layers[1].data[loop.index].uv.x)
    coords = np.array([v.co[:] for v in mesh.vertices])
    positions = np.stack([coords[:, 0], coords[:, 2], -coords[:, 1]], axis=1)
    mesh.materials[0] = support.tint_material(mesh.materials[0], support.TEAM_RED)
    support.render_setup()
    scene.render.film_transparent = False
    scene.world.color = (0.12, 0.12, 0.12)
    scene.render.resolution_x = 640
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = 30
    # Absolute shape keys reproduce the sampled rigid transforms in the saved
    # scene, including the two string segments and arrow visibility.
    obj.shape_key_add(name="Basis")
    obj.data.shape_keys.use_relative = False
    expected = []
    frame_count = 265 if args.reload else 109 if args.release else 85
    clip_name = "reload" if args.reload else "release" if args.release else "raise"
    camera_target = (0, 0, 1.25) if args.release or args.reload else (0, 0, 1.13)
    camera_scale = 3.05 if args.release or args.reload else 2.75
    for frame in range(frame_count):
        posed = motion.deform(positions, parts, pose_at(frame / 30))
        posed_blender = np.stack([posed[:, 0], -posed[:, 2], posed[:, 1]], axis=1)
        key = obj.shape_key_add(name=f"Frame_{frame:03d}")
        key.interpolation = "KEY_LINEAR"
        key.data.foreach_set("co", posed_blender.astype(np.float32).reshape(-1))
        obj.data.shape_keys.eval_time = key.frame
        obj.data.shape_keys.keyframe_insert("eval_time", frame=frame + 1)
        expected.append(posed_blender.astype(np.float32))
    for fc in (
        obj.data.shape_keys.animation_data.action.layers[0]
        .strips[0]
        .channelbags[0]
        .fcurves
    ):
        for point in fc.keyframe_points:
            point.interpolation = "LINEAR"
    scene.frame_start, scene.frame_end = 1, frame_count
    support.camera(8, -38, camera_target, camera_scale)
    scene.frame_set(58)
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / f"archer_{clip_name}_review.blend"))
    np.savez_compressed(ROOT / "expected_scene.npz", positions=np.array(expected))
    for view, az in [("oblique", -38), ("side", -88)]:
        if args.view and args.view != view:
            continue
        folder = ROOT / ("frames_" + view)
        folder.mkdir(exist_ok=True)
        support.camera(8, az, camera_target, camera_scale)
        diagnostic = (
            [0, 57, 65, 66, 67, 68, 70, 73, 79, 84, 108]
            if args.release
            else [0, 12, 24, 36, 48, 57, 70, 84]
        )
        if args.reload:
            diagnostic = [
                0,
                28,
                43,
                53,
                62,
                66,
                70,
                81,
                94,
                105,
                117,
                126,
                144,
                162,
                186,
                243,
                264,
            ]
        for frame in diagnostic if args.poses else range(frame_count):
            scene.frame_set(frame + 1)
            scene.render.filepath = str(folder / f"{frame:03d}.png")
            bpy.ops.render.render(write_still=True)
    html_name = (
        "review_reload.html"
        if args.reload
        else "review_release.html"
        if args.release
        else "review.html"
    )
    (ROOT / "review.html").write_text((SOURCE / html_name).read_text())
    print("ARCHER_REVIEW_DONE", flush=True)


if __name__ == "__main__":
    main()
