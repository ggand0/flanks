"""Check the exported slash rig, mesh intersections and sampled motion.

Run in background Blender after prepare_knight.py.
"""
import argparse
import json
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[2]
ROOT = REPO / "assets_dev/knight/diagonal_slash_v4"
sys.path.insert(0, str(SOURCE.parent / "melee"))
from sword_slash import SwordSlash, matrix

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=Path, default=ROOT)
args = parser.parse_args(
    sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
)
ROOT = args.out_dir.resolve()
ROOT.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(ROOT / "knight.glb"))
obj = bpy.data.objects["L0"]
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
faces = {
    pid: [list(p.vertices) for p in mesh.polygons if parts[p.vertices[0]] == pid]
    for pid in set(parts)
}
errors = []
collisions = {"body": [], "shield": [], "legs": []}
max_grip_error = max_hinge_error = max_sword_edge_error = 0.0
max_wrist = 0.0
min_flex, max_flex = 180.0, 0.0
edges = np.array([(f[i], f[(i + 1) % len(f)]) for f in faces[8] for i in range(len(f))])
rest_lengths = np.linalg.norm(positions[edges[:, 1]] - positions[edges[:, 0]], axis=1)
for seconds in np.linspace(0, rig.duration, 241):
    pose = rig.sample(seconds)
    r1, r2, r3, elbow, grip = rig.transforms(pose)
    flex = np.degrees(
        np.arccos(
            np.clip(
                np.dot((elbow - rig.shoulder) / rig.l1, (grip - elbow) / rig.l2), -1, 1
            )
        )
    )
    min_flex, max_flex = min(min_flex, flex), max(max_flex, flex)
    max_wrist = max(
        max_wrist, float(2 * np.degrees(np.arccos(np.clip(abs(pose[11]), 0, 1))))
    )
    max_hinge_error = max(
        max_hinge_error,
        float(np.linalg.norm(matrix(pose[4:8]) @ rig.hinge - rig.hinge)),
    )
    contact = rig.deform(np.array([rig.grip, rig.grip]), np.array([1, 8]), pose)
    max_grip_error = max(max_grip_error, float(np.linalg.norm(contact[1] - contact[0])))
    p = rig.deform(positions, parts, pose)
    max_sword_edge_error = max(
        max_sword_edge_error,
        float(
            np.max(
                abs(
                    np.linalg.norm(p[edges[:, 1]] - p[edges[:, 0]], axis=1)
                    - rest_lengths
                )
            )
        ),
    )
    trees = {
        pid: BVHTree.FromPolygons(p.tolist(), faces[pid], all_triangles=False)
        for pid in [0, 2, 3, 5, 8]
    }
    for label, pids in [("body", [0]), ("shield", [5]), ("legs", [2, 3])]:
        if any(trees[8].overlap(trees[pid]) for pid in pids):
            collisions[label].append(round(float(seconds), 5))
    assert (
        np.max(abs(p[np.isin(parts, [2, 3])] - positions[np.isin(parts, [2, 3])]))
        < 1e-12
    )
boundary = float(
    np.max(
        abs(
            rig.deform(positions, parts, rig.sample(0))
            - rig.deform(positions, parts, rig.sample(rig.duration))
        )
    )
)
hold_reference = rig.deform(positions, parts, rig.sample(rig.hold_start))
hold_error = max(
    float(np.max(abs(rig.deform(positions, parts, rig.sample(t)) - hold_reference)))
    for t in np.linspace(rig.hold_start, rig.hold_end, 29)
)
contact_pose = rig.sample(rig.contact_time)
contact_transform = rig.transforms(contact_pose)
contact_grip = contact_transform[4]
contact_blade = contact_transform[2] @ np.array([0, 0, 1])
cut_flexion, cut_tip_heights, edge_errors = [], [], []
early_edge_errors = []
for seconds in np.linspace(rig.hold_end + 0.01, rig.hold_end + 0.15, 57):
    pose = rig.sample(seconds)
    _, _, _, elbow, grip = rig.transforms(pose)
    cut_flexion.append(
        float(
            np.arccos(
                np.clip(
                    np.dot((elbow - rig.shoulder) / rig.l1, (grip - elbow) / rig.l2),
                    -1,
                    1,
                )
            )
        )
    )
    basis = rig.deform(
        np.array([rig.grip, rig.grip + [1, 0, 0], rig.grip + [0, 0, 1]]),
        np.array([8, 8, 8]),
        pose,
    )
    edge, axis = basis[1] - basis[0], basis[2] - basis[0]
    cut_tip_heights.append(float((basis[0] + rig.reach * axis)[1]))
    point = np.array([rig.grip + [0, 0, rig.reach]])
    before = rig.deform(point, np.array([8]), rig.sample(seconds - 0.0005))[0]
    after = rig.deform(point, np.array([8]), rig.sample(seconds + 0.0005))[0]
    velocity = after - before
    velocity -= axis * np.dot(velocity, axis)
    velocity /= np.linalg.norm(velocity)
    edge_errors.append(
        float(np.degrees(np.arccos(np.clip(abs(np.dot(edge, velocity)), 0, 1))))
    )
    if seconds <= rig.finish_roll_start:
        early_edge_errors.append(edge_errors[-1])
report = {
    "samples": 241,
    "duration_seconds": rig.duration,
    "contact_seconds": rig.contact_time,
    "raised_hold_seconds": [rig.hold_start, rig.hold_end],
    "max_raised_hold_position_error_m": hold_error,
    "contact_grip_height_m": float(contact_grip[1]),
    "contact_blade_downward_angle_degrees": float(
        np.degrees(np.arcsin(-contact_blade[1]))
    ),
    "maximum_blade_edge_to_sweep_angle_degrees": max(edge_errors),
    "maximum_edge_to_sweep_angle_before_finish_roll_degrees": max(early_edge_errors),
    "cut_elbow_straightens_continuously": bool(np.all(np.diff(cut_flexion) < 0)),
    "cut_blade_point_descends_continuously": bool(np.all(np.diff(cut_tip_heights) < 0)),
    "max_grip_error_m": max_grip_error,
    "max_elbow_hinge_axis_error": max_hinge_error,
    "elbow_flexion_degrees": [min_flex, max_flex],
    "max_wrist_rotation_degrees": max_wrist,
    "max_weapon_edge_error_m": max_sword_edge_error,
    "loop_boundary_error_m": boundary,
    "weapon_surface_intersection_seconds": collisions,
    "feet_unchanged": True,
    "joints": rig.export()["joints"],
    "height_m": float(positions[:, 1].max()),
    "triangles": {
        o.name: sum(len(p.vertices) - 2 for p in o.data.polygons)
        for o in bpy.data.objects
        if o.type == "MESH"
    },
}
(ROOT / "motion_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
assert max_grip_error < 1e-12 and max_hinge_error < 1e-12
assert max_sword_edge_error < 1e-12 and boundary < 1e-12
assert max_flex < 150
assert hold_error < 1e-12
assert np.all(np.diff(cut_flexion) < 0)
assert np.all(np.diff(cut_tip_heights) < 0)
assert max(early_edge_errors) < 8
assert not any(collisions.values()), collisions
