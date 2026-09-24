"""Check four-level GLBs and exact preservation of their installed source.

python3 tools/blender/lods/verify_levels.py candidate.glb --source installed.glb
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

parser = argparse.ArgumentParser()
parser.add_argument("glb", type=Path)
parser.add_argument("--source", type=Path, required=True)
args = parser.parse_args()
old, old_blob = glb_inspect.load(args.source)
doc, blob = glb_inspect.load(args.glb)
assert blob[: len(old_blob)] == old_blob
for key in ["materials", "images", "textures", "samplers", "asset"]:
    assert doc.get(key) == old.get(key), key
for key in ["meshes", "nodes", "accessors", "bufferViews"]:
    assert doc[key][: len(old[key])] == old[key], key
nodes = {n["name"]: n for n in doc["nodes"] if "mesh" in n}
assert set(nodes) == {"L0", "L1", "L2", "L3"}
assert len(doc["materials"]) == len(doc["images"]) == len(doc["textures"]) == 1
assert doc["materials"][0].get("alphaMode", "OPAQUE") == "OPAQUE"
source_node = next(n for n in old["nodes"] if n.get("name") == "L0")
_, _, original_parts, _ = inspect_surfaces.load(args.source, source_node["mesh"])
heights = {int(p): h for p, h in original_parts}
report = {
    "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
    "source_data_unchanged": True,
    "atlas_unchanged": True,
    "inherited_double_sided_material": doc["materials"][0].get("doubleSided", False),
    "sha256": hashlib.sha256(args.glb.read_bytes()).hexdigest(),
    "pivots_gltf_xyz": {
        n["name"]: n.get("translation", [0, 0, 0])
        for n in doc["nodes"]
        if n.get("name", "").startswith(("pivot_", "joint_"))
    },
    "levels": {},
}
for name in ["L0", "L1", "L2", "L3"]:
    node = nodes[name]
    assert not any(
        key in node for key in ["translation", "rotation", "scale", "matrix"]
    )
    _, pos, uv1, tri = inspect_surfaces.load(args.glb, node["mesh"])
    mesh = doc["meshes"][node["mesh"]]
    assert len(mesh["primitives"]) == 1
    prim = mesh["primitives"][0]
    assert prim["material"] == 0
    colors_acc, colors = glb_inspect.accessor_values(
        doc, blob, prim["attributes"]["COLOR_0"]
    )
    assert colors_acc["type"] == "VEC4" and np.allclose(np.array(colors)[:, :3], 1)
    assert np.all(uv1[tri] == uv1[tri[:, 0, None]])
    assert abs(pos[:, 1].min()) < 1e-6
    head_height = pos[np.isin(uv1[:, 0], [0, 18]), 1].max()
    if name != "L3":
        assert abs(head_height - 1.8) < 1e-6
    for pid, height in np.unique(uv1, axis=0):
        assert pid == int(pid) and height == heights[int(pid)]
    if name == "L1":
        optional = {13, 16, 17} if args.glb.stem == "archer" else set()
        assert 600 <= len(tri) <= 800
        assert set(uv1[:, 0]) <= set(heights)
        assert set(heights) - set(uv1[:, 0]) <= optional
    if name == "L3":
        assert 24 <= len(tri) <= 60 and np.all(uv1 == [0, 0])
    topology, _ = inspect_surfaces.topology(pos, tri)
    entry = {
        "triangles": len(tri),
        "vertices": len(pos),
        "height_m": float(pos[:, 1].max()),
        "parts": {
            str(k): int(v) for k, v in Counter(uv1[tri[:, 0], 0].astype(int)).items()
        },
        "part_id_and_pivot_height": np.unique(uv1, axis=0).tolist(),
        "topology": topology,
    }
    if name in ["L1", "L3"]:
        manifest = json.loads((args.glb.parent / f"{name}_surfaces.json").read_text())
        lookup = defaultdict(set)
        for i, face in enumerate(manifest["faces"]):
            for v in face:
                lookup[
                    inspect_surfaces.key(
                        np.asarray(manifest["vertices"][v], np.float32)
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
        if name == "L3":
            head_vertices = tri[np.isin(labels, ["head", "hat", "helm_shell"])].reshape(
                -1
            )
            assert len(head_vertices)
            head_height = pos[head_vertices, 1].max()
            assert abs(head_height - 1.8) < 1e-6
        for metric in [
            "open_edges",
            "nonmanifold_edges",
            "same_winding_edges",
            "degenerate_triangles",
        ]:
            assert topology[metric] == 0, (name, metric, topology)
        entry["components"] = {
            label: int(np.count_nonzero(labels == label))
            for label in sorted(set(labels))
        }
        entry["shield"], _ = inspect_surfaces.topology(
            pos, tri[np.isin(labels, ["heater_face", "heater_wood", "heater_rim"])]
        )
        assert (
            entry["shield"]["open_edges"]
            == entry["shield"]["nonmanifold_edges"]
            == entry["shield"]["same_winding_edges"]
            == 0
        )
        if entry["shield"]["triangles"]:
            assert entry["shield"]["signed_volume_m3"] > 0
    entry["head_height_m"] = float(head_height)
    report["levels"][name] = entry
suffix = {"spearman": ".stab.json", "archer": ".shoot.json"}.get(args.glb.stem)
if suffix:
    assert (
        args.glb.with_suffix(suffix).read_bytes()
        == args.source.with_suffix(suffix).read_bytes()
    )
    report["sidecar_unchanged"] = True
(args.glb.parent / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
