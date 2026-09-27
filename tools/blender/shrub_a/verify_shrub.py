"""Inspect exported tree geometry and opaque bark/cutout foliage contracts.

Run: python3 verify_shrub.py shrub_a/tree.glb
"""
from collections import Counter
import io
import json
import math
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import glb_inspect
from glb_inspect import accessor_values


def levels(document):
    return {
        n["name"]: document["meshes"][n["mesh"]]["primitives"]
        for n in document["nodes"]
        if n.get("name") in ("L0", "L1", "card")
    }


def positions(document, blob, primitives):
    vertices = set()
    for primitive in primitives:
        _, points = accessor_values(document, blob, primitive["attributes"]["POSITION"])
        vertices.update(tuple(p) for p in points)
    return vertices


def main():
    path = Path(sys.argv[1])
    species = path.parent.name
    document, blob = glb_inspect.load(str(path))
    results = {}
    for node in document["nodes"]:
        assert node.get("translation", [0, 0, 0]) == [0, 0, 0]
        assert node.get("rotation", [0, 0, 0, 1]) == [0, 0, 0, 1]
        assert node.get("scale", [1, 1, 1]) == [1, 1, 1]
        assert "skin" not in node and "matrix" not in node
    for name, primitives in levels(document).items():
        expected = 1 if name == "card" else 2
        assert len(primitives) == expected
        points, triangle_count, modes = [], 0, []
        for primitive in primitives:
            values = {}
            for attr, kind in [
                ("POSITION", "VEC3"),
                ("NORMAL", "VEC3"),
                ("TEXCOORD_0", "VEC2"),
                ("TEXCOORD_1", "VEC2"),
                ("COLOR_0", "VEC4"),
            ]:
                accessor = primitive["attributes"][attr]
                assert document["accessors"][accessor]["type"] == kind
                _, values[attr] = accessor_values(document, blob, accessor)
                assert all(math.isfinite(float(v)) for p in values[attr] for v in p)
            assert all(abs(c[3] - 1) < 1e-5 for c in values["COLOR_0"])
            assert all(0 <= v <= 1 for p in values["TEXCOORD_1"] for v in p)
            _, indices = accessor_values(document, blob, primitive["indices"])
            assert len(indices) % 3 == 0
            assert all(0 <= p[0] < len(values["POSITION"]) for p in indices)
            triangle_count += len(indices) // 3
            points.extend(values["POSITION"])
            mat = document["materials"][primitive["material"]]
            mode = mat.get("alphaMode", "OPAQUE")
            modes.append(mode)
            if mode == "OPAQUE":
                assert not mat.get("doubleSided", False)
                assert all(p[1] == 0 for p in values["TEXCOORD_1"])
                vertices = [tuple(round(v, 6) for v in p) for p in values["POSITION"]]
                edges = Counter()
                for start in range(0, len(indices), 3):
                    triangle = [vertices[int(indices[start + i][0])] for i in range(3)]
                    assert len(set(triangle)) == 3, "degenerate bark triangle"
                    for a, b in zip(triangle, triangle[1:] + triangle[:1]):
                        edges[(a, b)] += 1
                assert all(
                    count == 1 and edges[(b, a)] == 1 for (a, b), count in edges.items()
                ), "open or inconsistently wound bark"

            else:
                assert mode == "MASK" and mat.get("alphaCutoff", 0.5) == 0.5
                assert mat["doubleSided"]
                assert all(p[1] == 1 for p in values["TEXCOORD_1"])
            tex = mat["pbrMetallicRoughness"]["baseColorTexture"]
            assert tex.get("texCoord", 0) == 0
            image = document["images"][document["textures"][tex["index"]]["source"]]
            view = document["bufferViews"][image["bufferView"]]
            start = view.get("byteOffset", 0)
            im = Image.open(io.BytesIO(blob[start : start + view["byteLength"]]))
            assert im.mode == "RGBA"
            if mode == "OPAQUE":
                assert im.getchannel("A").getextrema() == (255, 255)
            if name == "card":
                assert "normalTexture" in mat
        assert sorted(modes) == (["MASK"] if name == "card" else ["MASK", "OPAQUE"])
        budget = {"L0": 2800, "L1": 900, "card": 4}
        assert triangle_count <= budget[name]
        minimum = [min(p[a] for p in points) for a in range(3)]
        maximum = [max(p[a] for p in points) for a in range(3)]
        assert abs(minimum[1]) < 1e-5
        results[name] = dict(
            triangles=triangle_count,
            height=maximum[1] - minimum[1],
            min=minimum,
            max=maximum,
            materials=modes,
        )
    near, middle = results["L0"], results["L1"]
    assert abs(near["height"] - 1.7) < 1e-5
    assert abs(near["max"][0] - near["min"][0] - 2.2) < 1e-5
    for axis in range(3):
        a = near["max"][axis] - near["min"][axis]
        b = middle["max"][axis] - middle["min"][axis]
        assert abs(a - b) / a < 0.11, ("LOD envelope", axis, a, b)
    (path.parent / "verified_glb.json").write_text(json.dumps(results, indent=2) + "\n")
    print(species, json.dumps(results))


if __name__ == "__main__":
    main()
