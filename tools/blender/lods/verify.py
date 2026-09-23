"""Verify L2 channels, closed geometry, and preservation of a source GLB.

python3 tools/blender/lods/verify.py candidate.glb --source original.glb
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glb_inspect
import inspect_surfaces


def verify(path, source):
    old, old_blob = glb_inspect.load(str(source))
    doc, blob = glb_inspect.load(str(path))
    assert blob[: len(old_blob)] == old_blob
    for key in ["materials", "images", "textures", "samplers", "asset"]:
        assert doc.get(key) == old.get(key), key
    for key in ["meshes", "nodes", "accessors", "bufferViews"]:
        assert doc[key][: len(old[key])] == old[key], key
    assert len(doc["meshes"]) == 2
    assert len(doc["materials"]) == len(doc["images"]) == len(doc["textures"]) == 1
    assert doc["materials"][0].get("alphaMode", "OPAQUE") == "OPAQUE"
    names = {
        0: "body",
        1: "arm_weapon",
        2: "leg_l",
        3: "leg_r",
        4: "arm_spear",
        5: "arm_shield",
        6: "arm_bow",
        8: "weapon",
        9: "forearm_spear",
        10: "hand_spear",
        11: "forearm_bow",
        12: "hand_bow",
        13: "bow_string",
        14: "bow_upper",
        15: "bow_lower",
        16: "arrow",
        17: "bow_string_lower",
        18: "head",
        19: "torso",
    }
    if "archer" in path.name:
        names.update({9: "forearm_draw", 10: "hand_draw"})
    pivots = {
        n["name"]: n.get("translation", [0, 0, 0])
        for n in doc["nodes"]
        if n.get("name", "").startswith(("pivot_", "joint_"))
    }
    report = {
        "file": str(path),
        "source": str(source),
        "L0_data_unchanged": True,
        "atlas_unchanged": True,
        "nodes_unchanged": True,
        "inherited_double_sided_material": doc["materials"][0].get(
            "doubleSided", False
        ),
        "pivots_gltf_xyz": pivots,
        "levels": {},
    }
    source_parts = set()
    for index, mesh in enumerate(doc["meshes"]):
        node = next(n for n in doc["nodes"] if n.get("mesh") == index)
        assert all(
            k not in node for k in ["translation", "rotation", "scale", "matrix"]
        )
        primitive = mesh["primitives"][0]
        attrs = primitive["attributes"]
        assert primitive["material"] == 0
        assert set(attrs) == {
            "POSITION",
            "NORMAL",
            "COLOR_0",
            "TEXCOORD_0",
            "TEXCOORD_1",
        }
        assert doc["accessors"][attrs["COLOR_0"]]["type"] == "VEC4"
        values = lambda key: np.array(
            glb_inspect.accessor_values(doc, blob, attrs[key])[1]
        )
        pos, parts, colors, uv = [
            values(k) for k in ["POSITION", "TEXCOORD_1", "COLOR_0", "TEXCOORD_0"]
        ]
        tri = np.array(
            glb_inspect.accessor_values(doc, blob, primitive["indices"])[1], int
        ).reshape(-1, 3)
        assert np.all(parts[:, 0] == np.rint(parts[:, 0]))
        assert np.all(parts[tri] == parts[tri[:, 0, None]])
        assert np.allclose(colors[:, :3], 1)
        assert np.all((uv >= 0) & (uv <= 1))
        head_height = pos[np.isin(parts[:, 0], [0, 18]), 1].max()
        assert abs(pos[:, 1].min()) < 1e-6 and abs(head_height - 1.8) < 1e-6
        for part, height in np.unique(parts, axis=0):
            expected = pivots["pivot_" + names[int(part)]][1] if part else 0
            assert abs(height - expected) < 1e-6, (part, height, expected)
        counts = Counter(int(x) for x in parts[tri[:, 0], 0])
        surface = inspect_surfaces.inspect(path, mesh_index=index)
        if node["name"] == "L0":
            source_parts = set(counts)
        else:
            assert 150 <= len(tri) <= 250, len(tri)
            assert set(counts) == source_parts - (
                {13, 16, 17} if "archer" in path.name else set()
            )
            for part, data in surface["parts"].items():
                assert all(
                    data[k] == 0
                    for k in [
                        "open_edges",
                        "nonmanifold_edges",
                        "same_winding_edges",
                        "degenerate_triangles",
                    ]
                ), (part, data)
                assert data["signed_volume_m3"] > 0
        report["levels"][node["name"]] = {
            "triangles": len(tri),
            "exported_vertices": len(pos),
            "head_height_m": float(head_height),
            "total_height_m": float(pos[:, 1].max()),
            "ground_m": float(pos[:, 1].min()),
            "triangles_by_part": {names[p]: c for p, c in sorted(counts.items())},
            "part_id_and_pivot_height": np.unique(parts, axis=0).tolist(),
            "topology_totals": {
                k: sum(p[k] for p in surface["parts"].values())
                for k in [
                    "open_edges",
                    "nonmanifold_edges",
                    "same_winding_edges",
                    "degenerate_triangles",
                ]
            },
        }
    manifest = json.loads((path.parent / "L2_surfaces.json").read_text())
    lookup = defaultdict(set)
    for i, face in enumerate(manifest["faces"]):
        for vertex in face:
            lookup[
                inspect_surfaces.key(
                    np.asarray(manifest["vertices"][vertex], dtype=np.float32)
                )
            ].add(i)
    labels = []
    for face in tri:
        matches = set.intersection(
            *(lookup[inspect_surfaces.key(pos[v])] for v in face)
        )
        selected = {manifest["components"][i] for i in matches}
        assert len(selected) == 1, selected
        labels.append(selected.pop())
    labels = np.array(labels)
    if "heater_face" in labels:
        for key, selected in [
            ("shield_board", ["heater_face", "heater_wood", "heater_rim"]),
            ("shield_straps", ["heater_straps"]),
        ]:
            data, _ = inspect_surfaces.topology(pos, tri[np.isin(labels, selected)])
            assert all(
                data[k] == 0
                for k in [
                    "open_edges",
                    "nonmanifold_edges",
                    "same_winding_edges",
                    "degenerate_triangles",
                ]
            )
            assert data["signed_volume_m3"] > 0
            report[key] = data
    report["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (path.parent / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("glb", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.glb, args.source), indent=2))
