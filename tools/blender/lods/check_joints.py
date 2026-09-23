"""Check L1 or L2 part overlap during 0 to 60 degree rigid rotations.

blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/check_joints.py -- candidate.blend
"""
import argparse
import json
from pathlib import Path
import sys
import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
import math

parser = argparse.ArgumentParser()
parser.add_argument("blend", type=Path)
parser.add_argument("--level", default="L2", choices=["L1", "L2"])
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
obj = bpy.data.objects[args.level]
positions = [v.co.copy() for v in obj.data.vertices]
parts = [obj.vertex_groups[v.groups[0].group].name for v in obj.data.vertices]
faces = {
    name: [list(p.vertices) for p in obj.data.polygons if parts[p.vertices[0]] == name]
    for name in set(parts)
}
kind = args.blend.stem
parents = {"leg_l": "body", "leg_r": "body", "arm_shield": "body"}
if kind in ("knight", "man_at_arms"):
    parents.update({"arm_weapon": "body", "weapon": "arm_weapon"})
elif kind == "spearman":
    parents.update(
        {
            "arm_spear": "body",
            "forearm_spear": "arm_spear",
            "hand_spear": "forearm_spear",
            "weapon": "hand_spear",
        }
    )
else:
    parents = {
        "leg_l": "body",
        "leg_r": "body",
        "torso": "body",
        "head": "torso",
        "arm_weapon": "torso",
        "forearm_draw": "arm_weapon",
        "hand_draw": "forearm_draw",
        "arm_bow": "torso",
        "forearm_bow": "arm_bow",
        "hand_bow": "forearm_bow",
        "weapon": "hand_bow",
        "bow_upper": "weapon",
        "bow_lower": "weapon",
    }
report = {"axis": "Blender +X", "degrees": [0, 10, 20, 30, 40, 50, 60], "parts": {}}
for part, parent in parents.items():
    pivot = bpy.data.objects["pivot_" + part].location
    affected = {part}
    while True:
        expanded = affected | {child for child, p in parents.items() if p in affected}
        if expanded == affected:
            break
        affected = expanded
    result = []
    for angle in report["degrees"]:
        rotation = Matrix.Rotation(math.radians(angle), 3, "X")
        moved = [
            pivot + rotation @ (p - pivot) if name in affected else p
            for p, name in zip(positions, parts)
        ]
        tree_a = BVHTree.FromPolygons(moved, faces[parent], all_triangles=True)
        tree_b = BVHTree.FromPolygons(moved, faces[part], all_triangles=True)
        result.append(len(tree_a.overlap(tree_b)))
    report["parts"][part] = {"parent": parent, "intersecting_triangle_pairs": result}
(folder := args.blend.resolve().parent).joinpath(
    "joint_sweep_validation.json"
).write_text(json.dumps(report, indent=2) + "\n")
failures = {
    part: data
    for part, data in report["parts"].items()
    if not all(data["intersecting_triangle_pairs"])
}
print("JOINT_SWEEP", kind, "failures", failures, flush=True)
assert not failures, failures
