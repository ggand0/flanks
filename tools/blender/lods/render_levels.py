"""Render all authored levels at identical framing and pixel sizes.

blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/render_levels.py -- candidate.glb
"""

import argparse
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "knight"))
import build_knight_textured as review

parser = argparse.ArgumentParser()
parser.add_argument("glb", type=Path)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
assert bpy.app.background
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(args.glb.resolve()))
folder = args.glb.resolve().parent
objects = [bpy.data.objects[name] for name in ["L0", "L1", "L2", "L3"]]
material = objects[0].data.materials[0]
for node in material.node_tree.nodes:
    if node.type == "TEX_IMAGE":
        node.image.alpha_mode = "CHANNEL_PACKED"
red = review.tint_material(material, review.TEAM_RED)
blue = review.tint_material(material, review.TEAM_BLUE)
for mat in [red, blue]:
    mat.use_backface_culling = True
review.render_setup()
scene = bpy.context.scene
scene.render.threads_mode = "FIXED"
scene.render.threads = 4
side_azimuth = 65 if args.glb.stem == "archer" else -65
for azimuth, label in [(25, "front"), (180, "back"), (side_azimuth, "side")]:
    cam = review.camera(50, azimuth)
    inverse = cam.matrix_world.inverted()
    points = np.array([tuple(inverse @ v.co) for v in objects[0].data.vertices])
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    cam.location += cam.rotation_euler.to_matrix() @ Vector((center[0], center[1], 0))
    span = high[1] - low[1]
    for team_name, mat in [("red", red), ("blue", blue)]:
        if team_name == "blue" and label != "front":
            continue
        for obj in objects:
            obj.data.materials[0] = mat
        for size in [256, 60, 20, 8, 3]:
            resolution = 320 if size == 256 else 80 if size == 60 else 64
            scene.render.resolution_x = scene.render.resolution_y = resolution
            cam.data.ortho_scale = span * resolution / size
            for obj in objects:
                for other in objects:
                    other.hide_render = other != obj
                scene.render.filepath = str(
                    folder / f"{obj.name}_{team_name}_{label}_{size}px.png"
                )
                bpy.ops.render.render(write_still=True)
print("LEVEL_REVIEW_DONE", flush=True)
