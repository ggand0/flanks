"""Inspect each GLB mesh's topology and facing area across attribute seams.

Run: python3 tools/blender/inspect_surfaces.py unit.glb --out report.json
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import struct
import numpy as np


def load(path, mesh_index=0):
    raw = path.read_bytes()
    assert raw[:4] == b"glTF"
    size = struct.unpack_from("<I", raw, 12)[0]
    doc = json.loads(raw[20 : 20 + size])
    offset = 20 + size
    length = struct.unpack_from("<I", raw, offset)[0]
    blob = raw[offset + 8 : offset + 8 + length]

    def accessor(index):
        a = doc["accessors"][index]
        view = doc["bufferViews"][a["bufferView"]]
        dtype = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[a["componentType"]]
        count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        width = np.dtype(dtype).itemsize
        data = np.ndarray(
            (a["count"], count),
            dtype=dtype,
            buffer=blob,
            offset=view.get("byteOffset", 0) + a.get("byteOffset", 0),
            strides=(view.get("byteStride", width * count), width),
        ).copy()
        if a.get("normalized"):
            data = data.astype(float) / np.iinfo(dtype).max
        return data

    positions, parts, triangles = [], [], []
    offset = 0
    for prim in doc["meshes"][mesh_index]["primitives"]:
        assert prim.get("mode", 4) == 4, "Expected triangles"
        attributes = prim["attributes"]
        position = accessor(attributes["POSITION"]).astype(float)
        positions.append(position)
        parts.append(accessor(attributes["TEXCOORD_1"]))
        indices = (
            accessor(prim["indices"]).astype(int).reshape(-1, 3)
            if "indices" in prim else np.arange(len(position)).reshape(-1, 3)
        )
        triangles.append(indices + offset)
        offset += len(position)
    return doc, np.concatenate(positions), np.concatenate(parts), np.concatenate(triangles)


def key(p):
    return tuple(int(round(float(x) * 1e6)) for x in p)


def topology(position, triangles):
    edges = defaultdict(list)
    for triangle in triangles:
        points = [key(position[i]) for i in triangle]
        for a, b in zip(points, points[1:] + points[:1]):
            edges[tuple(sorted((a, b)))].append(1 if a < b else -1)
    xyz = position[triangles]
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    area = np.linalg.norm(cross, axis=1) * 0.5
    open_edges = [e for e, v in edges.items() if len(v) == 1]
    return {
        "triangles": len(triangles),
        "open_edges": len(open_edges),
        "nonmanifold_edges": sum(len(v) > 2 for v in edges.values()),
        "same_winding_edges": sum(len(v) == 2 and sum(v) != 0 for v in edges.values()),
        "forward_facing_area_m2": float(area[cross[:, 2] > 1e-10].sum()),
        "backward_facing_area_m2": float(area[cross[:, 2] < -1e-10].sum()),
        "forward_projected_area_m2": float(np.maximum(cross[:, 2], 0).sum() * 0.5),
        "backward_projected_area_m2": float(np.maximum(-cross[:, 2], 0).sum() * 0.5),
        "signed_volume_m3": float(
            np.einsum("ij,ij->i", xyz[:, 0], np.cross(xyz[:, 1], xyz[:, 2])).sum() / 6
        ),
        "degenerate_triangles": int(np.count_nonzero(area < 1e-12)),
    }, open_edges


def inspect(path, manifest_path=None, mesh_index=0):
    doc, pos, uv, tri = load(path, mesh_index)
    parts = np.rint(uv[:, 0]).astype(int)
    assert np.all(parts[tri] == parts[tri[:, 0, None]])
    culling = all(
        not material.get("doubleSided", False) for material in doc["materials"]
    )
    report = {
        "file": str(path),
        "weld_tolerance_m": 1e-6,
        "backface_culling": culling,
        "parts": {},
    }
    for part in sorted(set(parts)):
        selected = tri[parts[tri[:, 0]] == part]
        report["parts"][str(part)], _ = topology(pos, selected)
    below = tri[(parts[tri[:, 0]] == 5) & np.all(pos[tri, 1] < 1.0, axis=1)]
    report["shield_below_arm"], _ = topology(pos, below)
    report["nodes"] = {
        n["name"]: n.get("translation", [0, 0, 0])
        for n in doc["nodes"]
        if n.get("name", "").startswith(("pivot_", "joint_"))
    }
    if manifest_path:
        assert culling, "The shield must work with back-face culling"
        source = json.loads(manifest_path.read_text())
        vertex_faces = defaultdict(set)
        for i, face in enumerate(source["faces"]):
            for v in face:
                # Blender mesh positions use float32 before glTF export.
                vertex_faces[
                    key(np.asarray(source["vertices"][v], dtype=np.float32))
                ].add(i)
        labels = []
        for triangle in tri:
            possible = set.intersection(*(vertex_faces[key(pos[v])] for v in triangle))
            assert possible, ("Triangle not matched to source", triangle)
            names = {source["components"][i] for i in possible}
            assert len(names) == 1, names
            labels.append(names.pop())
        labels = np.array(labels)
        shell = tri[np.isin(labels, ["heater_face", "heater_wood", "heater_rim"])]
        report["shield_board"], _ = topology(pos, shell)
        board = report["shield_board"]
        assert (
            board["open_edges"]
            == board["nonmanifold_edges"]
            == board["same_winding_edges"]
            == 0
        ), board
        assert board["signed_volume_m3"] > 0
        assert (
            abs(
                board["forward_projected_area_m2"] - board["backward_projected_area_m2"]
            )
            < 1e-6
        )
        report["shield_components"] = {}
        report["open_edge_components"] = {}
        for label in sorted(set(labels)):
            data, edges = topology(pos, tri[labels == label])
            if label.startswith("heater_"):
                report["shield_components"][label] = data
            if edges:
                report["open_edge_components"][label] = {
                    "open_edges": len(edges),
                    "bounds_m": [
                        list(np.min(np.array(edges) / 1e6, axis=(0, 1))),
                        list(np.max(np.array(edges) / 1e6, axis=(0, 1))),
                    ],
                }
        # Fittings and straps are separate closed solids; the board includes
        # its material boundaries when testing shared edge winding.
        for label in ["heater_fittings", "heater_straps"]:
            data = report["shield_components"][label]
            assert (
                data["open_edges"]
                == data["nonmanifold_edges"]
                == data["same_winding_edges"]
                == 0
            ), data
        assert not any(x["degenerate_triangles"] for x in report["parts"].values())
    return report


def inspect_levels(path, manifest_path=None):
    doc, _, _, _ = load(path)
    if len(doc["meshes"]) == 1:
        return inspect(path, manifest_path)
    levels = {}
    for index, mesh in enumerate(doc["meshes"]):
        names = [n.get("name") for n in doc["nodes"] if n.get("mesh") == index]
        name = next((n for n in names if n in {"L0", "L1", "L2", "L3"}),
                    mesh.get("name", str(index)))
        assert name not in levels, f"Duplicate mesh name: {name}"
        levels[name] = inspect(path, manifest_path if name == "L0" else None, index)
    return {"file": str(path), "levels": levels}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("glb", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = inspect_levels(args.glb, args.manifest)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ["nodes", "open_edge_components", "shield_components"]
            },
            indent=2,
        )
    )
