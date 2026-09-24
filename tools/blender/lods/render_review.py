"""Render L0 and L2 at identical framing from a 50 degree camera."""
import math
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils import Vector

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("blend", type=Path)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
HERE = args.blend.resolve().parent
SOURCE = Path(__file__).resolve().parents[1] / "knight"
sys.path.insert(0, str(SOURCE))
import build_knight_textured as build

bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
build.render_setup()
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.threads_mode = "FIXED"
scene.render.threads = 6
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = True
scene.render.resolution_percentage = 100
l0, l2 = [bpy.data.objects[n] for n in ["L0", "L2"]]
mat = l0.data.materials[0]
red = build.tint_material(mat, build.TEAM_RED)
blue = build.tint_material(mat, build.TEAM_BLUE)
for obj in [l0, l2]:
    obj.data.materials[0] = red

for azimuth, label in [
    (25, "front"),
    (180, "back"),
    (65 if "archer" in args.blend.name else -65, "side"),
]:
    cam = build.camera(50, azimuth)
    inverse = cam.matrix_world.inverted()
    points = [inverse @ v.co for v in l0.data.vertices]
    low = np.min([tuple(p) for p in points], axis=0)
    high = np.max([tuple(p) for p in points], axis=0)
    center = (low + high) / 2
    cam.location += cam.rotation_euler.to_matrix() @ Vector((center[0], center[1], 0))
    span = high[1] - low[1]
    for size in [256, 20, 8, 3]:
        resolution = 320 if size == 256 else 64
        cam.data.ortho_scale = span * resolution / size
        scene.render.resolution_x = scene.render.resolution_y = resolution
        for obj, other in [(l0, l2), (l2, l0)]:
            obj.hide_render, other.hide_render = False, True
            scene.render.filepath = str(HERE / f"{obj.name}_{label}_{size}px.png")
            bpy.ops.render.render(write_still=True)
    if label == "front":
        for obj in [l0, l2]:
            obj.data.materials[0] = blue
        for size in [20, 8, 3]:
            cam.data.ortho_scale = span * 64 / size
            scene.render.resolution_x = scene.render.resolution_y = 64
            for obj, other in [(l0, l2), (l2, l0)]:
                obj.hide_render, other.hide_render = False, True
                scene.render.filepath = str(HERE / f"{obj.name}_blue_{size}px.png")
                bpy.ops.render.render(write_still=True)
        for obj in [l0, l2]:
            obj.data.materials[0] = red
print("REVIEW_RENDERS_DONE", flush=True)
