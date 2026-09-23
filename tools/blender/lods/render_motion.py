"""Compare L0 and L2 through existing attack motions in headless Blender.

blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/render_motion.py -- candidate.blend
"""
import argparse
import json
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/blender/knight"))
import build_knight_textured as review

parser = argparse.ArgumentParser()
parser.add_argument("blend", type=Path)
parser.add_argument("--check-only", action="store_true")
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
folder = args.blend.resolve().parent
kind = args.blend.stem
bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
objects = [bpy.data.objects[name] for name in ["L0", "L2"]]
positions, parts = {}, {}
for obj in objects:
    positions[obj.name] = np.array(
        [(v.co.x, v.co.z, -v.co.y) for v in obj.data.vertices]
    )
    ids = np.zeros(len(obj.data.vertices), int)
    for loop in obj.data.loops:
        ids[loop.vertex_index] = round(obj.data.uv_layers["part"].data[loop.index].uv.x)
    parts[obj.name] = ids
nodes = {
    o.name: [o.location.x, o.location.z, -o.location.y]
    for o in bpy.data.objects
    if o.name.startswith(("pivot_", "joint_"))
}
if kind == "man_at_arms":
    source = Path(__file__).resolve().parent
    sys.path.insert(0, str(source))
    import sword_motion as motion_preview

    rigs = {
        name: motion_preview.SwordRig(nodes, positions[name], parts[name])
        for name in positions
    }
    frames = 90
    frame_times = np.arange(frames) / 15

    def deform(name, frame):
        rig = rigs[name]
        return rig.deform(
            positions[name], parts[name], rig.pose(*motion_preview.timeline(frame * 2))
        )

    links = [("sword shoulder", 0, 1), ("shield shoulder", 0, 5), ("sword grip", 1, 8)]
    scale, target = 3.2, (0, -0.12, 1.25)
elif kind == "spearman":
    source = ROOT / "tools/blender/spearman"
    sys.path.insert(0, str(source))
    import motion

    table = json.loads((folder / "spearman.stab.json").read_text())
    assert table == json.loads(json.dumps(motion.export_clip()))
    frames = 64
    frame_times = np.arange(frames) / 20

    def deform(name, frame):
        return motion.deform(
            positions[name], parts[name], motion.sample(frame / 20, intro=True)
        )

    links = [
        ("spear shoulder", 0, 4),
        ("spear elbow", 4, 9),
        ("spear wrist", 9, 10),
        ("spear grip", 10, 8),
        ("shield shoulder", 0, 5),
    ]
    scale, target = 3.6, (0, -0.70, 1.22)
else:
    assert kind == "archer"
    source = ROOT / "tools/blender/archer"
    sys.path.insert(0, str(source))
    import motion

    table = json.loads((folder / "archer.shoot.json").read_text())
    rebuilt = motion.export()
    table_difference = max(
        float(
            np.max(
                np.abs(
                    np.asarray(table["samples"][phase])
                    - np.asarray(rebuilt["samples"][phase])
                )
            )
        )
        for phase in table["samples"]
    )
    assert table_difference < 1e-6, table_difference
    motion.TABLES = {
        phase: np.asarray(samples) for phase, samples in table["samples"].items()
    }
    motion.ARROW_TABLE = np.asarray(table["arrow_samples"]["reload"])
    # Carry, raise, release, the full reload, and the following raise.
    frames = 150
    frame_times = np.arange(frames) / 15

    def pose(seconds):
        if seconds < 0.4:
            return motion.sample("raise", 0)
        if seconds < 1.9:
            return motion.sample("raise", (seconds - 0.4) / 1.5)
        if seconds < 2.5:
            return motion.sample("release", (seconds - 1.9) / 0.6)
        if seconds < 8.3:
            return motion.sample("reload", (seconds - 2.5) / 5.8)
        return motion.sample("raise", (seconds - 8.3) / 1.5)

    def deform(name, frame):
        return motion.deform(positions[name], parts[name], pose(frame / 15))

    links = [
        ("waist", 0, 19),
        ("neck", 19, 18),
        ("draw shoulder", 19, 1),
        ("draw elbow", 1, 9),
        ("draw wrist", 9, 10),
        ("bow shoulder", 19, 6),
        ("bow elbow", 6, 11),
        ("bow wrist", 11, 12),
        ("bow grip", 12, 8),
        ("bow upper", 8, 14),
        ("bow lower", 8, 15),
    ]
    scale, target = 3.05, (0, -0.05, 1.25)

review.render_setup()
scene = bpy.context.scene
scene.render.threads_mode = "FIXED"
scene.render.threads = 6
scene.render.resolution_x = scene.render.resolution_y = 320
scene.render.resolution_percentage = 100
scene.render.image_settings.color_mode = "RGBA"
review.camera(15, -45, target, scale)
material = review.tint_material(objects[0].data.materials[0], review.TEAM_RED)
for obj in objects:
    obj.data.materials[0] = material
frames_dir = folder / "motion_frames"
frames_dir.mkdir(exist_ok=True)
report = {
    "frames": frames,
    "duration_seconds": float(frame_times[-1]),
    "motion_source": str(source),
    "sidecar_matches_source": kind != "man_at_arms",
    "render_backface_culling": True,
    "joint_overlap": {label: [] for label, _, _ in links},
}
low = objects[1]
triangles = {
    pid: [
        list(p.vertices) for p in low.data.polygons if parts["L2"][p.vertices[0]] == pid
    ]
    for pid in set(parts["L2"])
}
for frame in range(frames):
    for obj in objects:
        posed = deform(obj.name, frame)
        blender = posed[:, [0, 2, 1]] * [1, -1, 1]
        obj.data.vertices.foreach_set("co", blender.astype(np.float32).reshape(-1))
        obj.data.update()
        if obj.name == "L2":
            trees = {
                pid: BVHTree.FromPolygons(blender.tolist(), faces, all_triangles=True)
                for pid, faces in triangles.items()
            }
            for label, a, b in links:
                report["joint_overlap"][label].append(len(trees[a].overlap(trees[b])))
        if not args.check_only:
            obj.hide_render = False
            next(other for other in objects if other != obj).hide_render = True
            scene.render.filepath = str(frames_dir / f"{obj.name}_{frame:03d}.png")
            bpy.ops.render.render(write_still=True)
report["frames_without_surface_intersection"] = {
    label: [i for i, value in enumerate(values) if not value]
    for label, values in report["joint_overlap"].items()
}
(folder / "motion_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print("MOTION_DONE", report["frames_without_surface_intersection"], flush=True)
