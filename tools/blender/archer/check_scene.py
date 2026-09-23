"""Reload the saved Blender scene and check every playback frame."""
from pathlib import Path
import argparse
import json
import sys
import bpy
import numpy as np

SOURCE = Path(__file__).resolve().parent
DEFAULT_OUT = SOURCE.parents[2] / "assets_dev/archer/rebuild_raise_v4"
parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
parser.add_argument("--release", action="store_true")
parser.add_argument("--reload", action="store_true")
args = parser.parse_args(
    sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
)
ROOT = args.out_dir.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
clip_name = "reload" if args.reload else "release" if args.release else "raise"
bpy.ops.wm.open_mainfile(filepath=str(ROOT / f"archer_{clip_name}_review.blend"))
expected = np.load(ROOT / "expected_scene.npz")["positions"]
obj = bpy.data.objects["L0"]
max_error = 0.0
for frame, points in enumerate(expected, 1):
    bpy.context.scene.frame_set(frame)
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    positions = np.array([v.co[:] for v in mesh.vertices])
    max_error = max(max_error, float(np.max(abs(positions - points))))
    evaluated.to_mesh_clear()
# Blender evaluates shape-key time in float32. Allow ten micrometres.
assert max_error < 1e-5, max_error
(ROOT / "saved_scene_validation.json").write_text(
    json.dumps({"frames": len(expected), "max_position_error_m": max_error}, indent=2)
    + "\n"
)
print("SAVED_SCENE_ERROR", max_error, flush=True)
# Inspect the rear with back-face culling enabled.
sys.path.insert(0, str(SOURCE))
import build_support as support

support.ROOT = ROOT
for mat in obj.data.materials:
    mat.use_backface_culling = True
bpy.context.scene.frame_set(58)
support.camera(12, 150, (0, -0.23, 1.18), 2.75)
bpy.context.scene.render.filepath = str(ROOT / "back_culling.png")
bpy.ops.render.render(write_still=True)
