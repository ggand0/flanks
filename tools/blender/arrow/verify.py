"""Check the exported arrow's dimensions, channels and closed components.

python3 tools/blender/arrow/verify.py assets_dev/arrow/arrow.glb
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glb_inspect
import inspect_surfaces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("glb", type=Path)
    args = parser.parse_args()
    doc, blob = glb_inspect.load(args.glb)
    assert len(doc["meshes"]) == len(doc["nodes"]) == 1
    node = doc["nodes"][0]
    assert node["name"] == "L0" and node["mesh"] == 0
    assert node.get("translation", [0, 0, 0]) == [0, 0, 0]
    assert node.get("scale", [1, 1, 1]) == [1, 1, 1]
    assert node.get("rotation", [0, 0, 0, 1]) == [0, 0, 0, 1]
    assert "matrix" not in node
    assert not doc.get("textures") and not doc.get("images")
    assert len(doc["materials"]) == 1
    material = doc["materials"][0]
    assert material.get("alphaMode", "OPAQUE") == "OPAQUE"
    assert not material.get("doubleSided", False)
    assert len(doc["meshes"][0]["primitives"]) == 1
    prim = doc["meshes"][0]["primitives"][0]
    color_accessor, colors = glb_inspect.accessor_values(
        doc, blob, prim["attributes"]["COLOR_0"]
    )
    assert color_accessor["type"] == "VEC4"
    assert all(color[3] == 0 for color in colors)
    _, pos, uv, triangles = inspect_surfaces.load(args.glb)
    assert np.all(uv == [7.0, 0.0])
    assert 0 < len(triangles) <= 40
    lower, upper = pos.min(axis=0), pos.max(axis=0)
    assert np.isclose(lower[2], -0.350, atol=1e-6)
    assert np.isclose(upper[2], 0.410, atol=1e-6)
    assert np.allclose(pos[pos[:, 2] == upper[2]][:, :2], 0, atol=1e-6)
    manifest = json.loads(args.glb.with_suffix(".manifest.json").read_text())
    source_pos = np.asarray(manifest["vertices"])
    labels = {}
    for face, name in zip(manifest["faces"], manifest["components"]):
        key = tuple(sorted(inspect_surfaces.key(source_pos[i]) for i in face))
        labels[key] = name
    components = {}
    for triangle in triangles:
        name = labels[tuple(sorted(inspect_surfaces.key(pos[i]) for i in triangle))]
        components.setdefault(name, []).append(triangle)
    assert set(components) == {
        "shaft_and_nock",
        "bodkin",
        "fletching_1",
        "fletching_2",
        "fletching_3",
    }
    report = {}
    for name, faces in components.items():
        metrics, _ = inspect_surfaces.topology(pos, np.asarray(faces))
        for key in (
            "open_edges",
            "nonmanifold_edges",
            "same_winding_edges",
            "degenerate_triangles",
        ):
            assert metrics[key] == 0, (name, key, metrics[key])
        assert metrics["signed_volume_m3"] > 0, name
        report[name] = metrics
    result = {
        "file": str(args.glb),
        "triangles": len(triangles),
        "exported_vertices": len(pos),
        "bounds_gltf_xyz_m": [lower.tolist(), upper.tolist()],
        "length_m": float(upper[2] - lower[2]),
        "shaft_endpoints_z_m": manifest["shaft_endpoints_z_m"],
        "origin_at_shaft_midpoint": True,
        "nock_depth_m": manifest["nock_depth_m"],
        "part_id_pivot_height": [7, 0],
        "color_type": color_accessor["type"],
        "alpha_min_max": [0, 0],
        "backface_culling": True,
        "components": report,
    }
    args.glb.with_suffix(".validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
