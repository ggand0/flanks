"""Render two translated shuffle cycles from the exported pose table.

Run: blender --background --factory-startup --python-exit-code 1 --python
tools/blender/knight/review_shuffle.py -- --out-dir assets_dev/knight/shuffle_v1
"""

import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
sys.path.insert(0, str(REPO / "tools/blender/knight"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_knight_textured as support
from shuffle import Shuffle, sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO / "assets_dev/knight/shuffle_v1")
    parser.add_argument("--source", type=Path, default=REPO / "assets/units/knight.glb")
    parser.add_argument("--poses", action="store_true")
    parser.add_argument("--direction", choices=["left", "right"])
    parser.add_argument("--view", choices=["back", "oblique"])
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    output = args.out_dir.resolve()
    assert bpy.app.background
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.source.resolve()))
    obj = bpy.data.objects["L0"]
    for other in bpy.data.objects:
        if other.type == "MESH" and other != obj:
            other.hide_render = True
    mesh = obj.data
    positions = np.array([(v.co.x, v.co.z, -v.co.y) for v in mesh.vertices])
    parts = np.zeros(len(positions), int)
    for loop in mesh.loops:
        parts[loop.vertex_index] = round(mesh.uv_layers[1].data[loop.index].uv.x)
    nodes = {o.name: [o.location.x, o.location.z, -o.location.y] for o in bpy.data.objects if o.name.startswith("pivot_")}
    rig = Shuffle(nodes)
    table = json.loads((output / "knight.shuffle.json").read_text())
    support.render_setup()
    scene = bpy.context.scene
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.eevee.taa_render_samples = 24
    scene.render.film_transparent = False
    scene.render.resolution_x, scene.render.resolution_y = 720, 540
    scene.render.image_settings.color_mode = "RGB"
    scene.render.fps = 24
    mesh.materials[0] = support.tint_material(mesh.materials[0], support.TEAM_RED)
    mesh.materials[0].use_backface_culling = True
    floor_mat = bpy.data.materials.new("Ground")
    floor_mat.diffuse_color = (0.12, 0.145, 0.17, 1)
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.002))
    bpy.context.object.data.materials.append(floor_mat)
    line_mat = bpy.data.materials.new("Ground_grid")
    line_mat.diffuse_color = (0.22, 0.25, 0.28, 1)
    for x in np.arange(-2.4, 2.41, 0.3):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(float(x), 0, -0.001))
        line = bpy.context.object
        line.scale = (0.007, 5.0, 0.001)
        line.data.materials.append(line_mat)
    obj.shape_key_add(name="Basis")
    mesh.shape_keys.use_relative = False
    frames = 48
    keys = []
    for frame in range(frames):
        key = obj.shape_key_add(name=f"Frame_{frame:03d}")
        key.interpolation = "KEY_LINEAR"
        keys.append(key)
        mesh.shape_keys.eval_time = key.frame
        mesh.shape_keys.keyframe_insert("eval_time", frame=frame + 1)
    for fc in mesh.shape_keys.animation_data.action.layers[0].strips[0].channelbags[0].fcurves:
        for point in fc.keyframe_points:
            point.interpolation = "LINEAR"
    scene.frame_start, scene.frame_end = 1, frames
    validation = {}
    for name, sign in [("left", 1), ("right", -1)]:
        if args.direction and name != args.direction:
            continue
        expected = []
        for frame, key in enumerate(keys):
            cycles = float(np.clip((frame / 24.0 - 0.25) / rig.duration, 0.0, 2.0))
            phase = cycles % 1.0
            posed = rig.deform(positions, parts, sample(table, name, phase))
            posed[:, 0] += sign * (cycles * rig.distance - rig.distance)
            coords = (posed[:, [0, 2, 1]] * [1, -1, 1]).astype(np.float32)
            key.data.foreach_set("co", coords.reshape(-1))
            expected.append(coords)
        max_error = 0.0
        for frame, target in enumerate(expected):
            scene.frame_set(frame + 1)
            evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            actual = np.array([v.co[:] for v in evaluated.data.vertices])
            max_error = max(max_error, float(np.max(abs(actual - target))))
        assert max_error < 1e-5, max_error
        validation[name] = {"frames": frames, "saved_playback_max_error_m": max_error}
        for view, elevation, azimuth in [("back", 8, 180), ("oblique", 28, 145)]:
            if args.view and view != args.view:
                continue
            support.camera(elevation, azimuth, (0, -0.04, 0.85), 3.35)
            scene.frame_set(1)
            if view == "back":
                bpy.ops.wm.save_as_mainfile(filepath=str(output / f"knight_shuffle_{name}.blend"))
            folder = output / f"frames_{name}_{view}"
            folder.mkdir(exist_ok=True)
            for frame in [6, 9, 12, 15, 18, 21, 24] if args.poses else range(frames):
                scene.frame_set(frame + 1)
                scene.render.filepath = str(folder / f"{frame:03d}.png")
                bpy.ops.render.render(write_still=True)
        print("SHUFFLE_REVIEW", name, validation[name], flush=True)
    (output / f"saved_scene_validation_{args.direction or 'both'}.json").write_text(json.dumps(validation, indent=2) + "\n")


if __name__ == "__main__":
    main()
