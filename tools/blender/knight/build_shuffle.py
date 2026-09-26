"""Write and validate the knight's left/right shuffle pose tables.

Run: python3 tools/blender/knight/build_shuffle.py --out-dir assets_dev/knight/shuffle_v1
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
sys.path.insert(0, str(REPO / "tools/blender"))
from inspect_surfaces import load
from shuffle import Shuffle, sample


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    doc, positions, uv, triangles = load(source)
    nodes = {n["name"]: n.get("translation", [0, 0, 0]) for n in doc["nodes"]}
    parts = np.rint(uv[:, 0]).astype(int)
    rig = Shuffle(nodes)
    table = rig.export(hashlib.sha256(source.read_bytes()).hexdigest())
    path = output / "knight.shuffle.json"
    path.write_text(json.dumps(table, indent=2) + "\n")
    table = json.loads(path.read_text())
    metrics = {"poses_per_direction": 1001, "directions": {}}
    for name, direction in [("left", 1), ("right", -1)]:
        contact_error = 0.0
        target_error = 0.0
        ground_min = 0.0
        angle_max = np.zeros(14)
        mirror_error = 0.0
        for phase in np.linspace(0, 1, 1001):
            row = sample(table, name, phase)
            angle_max = np.maximum(angle_max, np.abs(row))
            posed = rig.deform(positions, parts, row)
            ground_min = min(ground_min, float(posed[:, 1].min()))
            ankles = rig.joint_positions(row)[:, 2]
            target_error = max(target_error, float(np.max(np.linalg.norm(ankles - rig.targets(phase, direction), axis=1))))
            for leg, part in enumerate([2, 3]):
                intervals = table["clips"][name]["contact_phase_intervals"]["left" if leg == 0 else "right"]
                for start, end in intervals:
                    if start <= phase <= end:
                        mask = (parts == part) & (positions[:, 1] < 0.001)
                        expected = positions[mask].copy()
                        expected[:, 0] += direction * rig.distance * (1.0 if start > 0 else 0.0)
                        world = posed[mask] + [direction * phase * rig.distance, 0, 0]
                        contact_error = max(contact_error, float(np.max(np.linalg.norm(world - expected, axis=1))))
            other = sample(table, "right" if name == "left" else "left", phase)
            expected = np.r_[row[:4] * [-1, 1, 1, -1], row[9:14] * [-1, 1, 1, 1, -1], row[4:9] * [-1, 1, 1, 1, -1]]
            mirror_error = max(mirror_error, float(np.max(np.abs(other - expected))))
        endpoint_error = max(float(np.max(np.abs(rig.deform(positions, parts, sample(table, name, p)) - positions))) for p in [0, 1])
        assert endpoint_error < 1e-9
        assert contact_error < 0.0005, contact_error
        assert target_error < 0.0005, target_error
        assert ground_min > -0.0005, ground_min
        assert mirror_error < 1e-9
        metrics["directions"][name] = {
            "planted_sole_max_error_m": contact_error,
            "ankle_target_max_error_m": target_error,
            "minimum_vertex_height_m": ground_min,
            "standing_endpoint_max_error_m": endpoint_error,
            "mirror_max_error": mirror_error,
            "max_pelvis_dip_m": float(angle_max[1]),
            "max_hip_roll_deg": float(np.degrees(max(angle_max[4], angle_max[9]))),
            "max_knee_flex_deg": float(np.degrees(max(angle_max[6], angle_max[11]))),
        }
    levels = {}
    for node in doc["nodes"]:
        if "mesh" not in node:
            continue
        _, pos, part_uv, tris = load(source, node["mesh"])
        level_parts = np.rint(part_uv[:, 0]).astype(int)
        min_height = 0.0
        if node["name"] != "L3":
            for name in ["left", "right"]:
                for phase in np.linspace(0, 1, 257):
                    posed = rig.deform(pos, level_parts, sample(table, name, phase))
                    min_height = min(min_height, float(posed[:, 1].min()))
            assert min_height > -0.0005, (node["name"], min_height)
        levels[node["name"]] = {
            "triangles": len(tris), "height_m": float(np.ptp(pos[:, 1])),
            "part_vertex_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(np.rint(part_uv[:, 0]), return_counts=True))},
            "minimum_posed_vertex_height_m": min_height,
            "shuffle_deformation": node["name"] != "L3",
        }
    metrics["source_levels"] = levels
    metrics["source_glb_sha256"] = table["source_glb_sha256"]
    metrics["table_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=REPO / "assets/units/knight.glb")
    parser.add_argument("--out-dir", type=Path, default=REPO / "assets_dev/knight/shuffle_v1")
    args = parser.parse_args()
    build(args.source.resolve(), args.out_dir.resolve())
