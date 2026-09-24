"""Build closed LOD shapes and sample their colours from a source GLB atlas.

Imported by the unit LOD build scripts. Coordinates in Blender use Z up.
"""
import json
import hashlib
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import struct
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/blender"))
import glb_inspect

WHITE = (1, 1, 1, 0)
NAMES = {
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


class Geometry:
    def __init__(self, source):
        self.vertices, self.faces, self.colors = [], [], []
        self.parts, self.smooth, self.components = [], [], []
        self.part, self.label = "body", "body"
        self.source = source

    def add(self, points, faces, color, smooth=False, mail=False):
        start = len(self.vertices)
        self.vertices.extend(tuple(p) for p in points)
        self.parts.extend([self.part] * len(points))
        for k, face in enumerate(faces):
            self.faces.append(tuple(start + i for i in face))
            if mail:
                # Small tonal facets suggest a woven surface, not huge rings.
                v = (0.95, 1.03, 0.99, 1.02, 0.97, 1.04)[k % 6]
                c = tuple(x * v for x in color[:3]) + (color[3],)
            else:
                c = color
            self.colors.append(c)
            self.smooth.append(smooth)
            self.components.append(self.label)

    def loft(self, rings, n, color, smooth=False, mail=False, cap=True):
        # Each ring is (z, center_x, center_y, radius_x, radius_y).
        points = []
        for z, x, y, rx, ry in rings:
            for i in range(n):
                angle = 2 * math.pi * i / n
                points.append((x + rx * math.sin(angle), y - ry * math.cos(angle), z))
        faces = []
        for r in range(len(rings) - 1):
            for i in range(n):
                a, b = r * n + i, r * n + (i + 1) % n
                faces.append((a, b, b + n, a + n))
        if cap:
            faces.extend(
                [
                    tuple(reversed(range(n))),
                    tuple((len(rings) - 1) * n + i for i in range(n)),
                ]
            )
        self.add(points, faces, color, smooth, mail)

    def tube(self, centers, radii, n, color, smooth=True, mail=False):
        points = []
        for k, center in enumerate(centers):
            tangent = Vector(centers[min(k + 1, len(centers) - 1)]) - Vector(
                centers[max(0, k - 1)]
            )
            tangent.normalize()
            reference = Vector((0, 1, 0)) if abs(tangent.y) < 0.9 else Vector((1, 0, 0))
            u = tangent.cross(reference).normalized()
            v = tangent.cross(u).normalized()
            for i in range(n):
                a = i * 2 * math.pi / n
                points.append(
                    Vector(center) + radii[k] * (u * math.cos(a) + v * math.sin(a))
                )
        faces = [tuple(reversed(range(n)))]
        for k in range(len(centers) - 1):
            for i in range(n):
                a, b = k * n + i, k * n + (i + 1) % n
                faces.append((a, b, b + n, a + n))
        faces.append(tuple((len(centers) - 1) * n + i for i in range(n)))
        self.add(points, faces, color, smooth, mail)

    def box(self, center, size, color):
        x, y, z = center
        a, b, c = (d / 2 for d in size)
        self.add(
            [
                (x + i * a, y + j * b, z + k * c)
                for i, j, k in [
                    (-1, -1, -1),
                    (1, -1, -1),
                    (1, 1, -1),
                    (-1, 1, -1),
                    (-1, -1, 1),
                    (1, -1, 1),
                    (1, 1, 1),
                    (-1, 1, 1),
                ]
            ],
            [
                (0, 3, 2, 1),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (1, 2, 6, 5),
                (2, 3, 7, 6),
                (3, 0, 4, 7),
            ],
            color,
        )

    def mesh(self, material):
        mesh = bpy.data.meshes.new("L2_geometry")
        mesh.from_pydata(self.vertices, [], self.faces)
        mesh.update()
        obj = bpy.data.objects.new("L2", mesh)
        bpy.context.scene.collection.objects.link(obj)
        mesh.materials.append(material)
        for name in sorted(set(self.parts)):
            group = obj.vertex_groups.new(name=name)
            group.add(
                [i for i, part in enumerate(self.parts) if part == name], 1, "REPLACE"
            )
        colors = mesh.color_attributes.new(
            name="Col", type="FLOAT_COLOR", domain="CORNER"
        )
        mesh.color_attributes.active_color = colors
        mesh.color_attributes.render_color_index = 0
        mesh.uv_layers.new(name="UVMap")
        uv1 = mesh.uv_layers.new(name="part")
        face_ids = mesh.attributes.new(name="source_face", type="INT", domain="FACE")
        for face, smooth in zip(mesh.polygons, self.smooth):
            face.use_smooth = smooth
            face_ids.data[face.index].value = face.index
            for loop in face.loop_indices:
                part = self.parts[mesh.loops[loop].vertex_index]
                pid = self.source.ids[part]
                uv1.data[loop].uv = (pid, 1 - self.source.heights[pid])
                colors.data[loop].color = WHITE
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        mesh.uv_layers.active_index = 0
        return obj


def position_key(position):
    return tuple(int(round(float(v) * 1e6)) for v in position)


class Source:
    def __init__(self, path, out_dir, manifest=None):
        self.path = path
        self.doc, self.blob = glb_inspect.load(str(path))
        doc, blob = self.doc, self.blob
        index = next(n["mesh"] for n in doc["nodes"] if n.get("name") == "L0")
        self.primitive = doc["meshes"][index]["primitives"][0]
        attrs = self.primitive["attributes"]
        read = lambda index: np.array(glb_inspect.accessor_values(doc, blob, index)[1])
        self.positions_gltf = read(attrs["POSITION"])
        self.positions = self.positions_gltf[:, [0, 2, 1]] * [1, -1, 1]
        self.triangles = read(self.primitive["indices"]).astype(int).reshape(-1, 3)
        self.uv = read(attrs["TEXCOORD_0"]) * [1, -1] + [0, 1]
        self.parts = read(attrs["TEXCOORD_1"])
        self.heights = {int(p): float(h) for p, h in self.parts}
        self.names = NAMES.copy()
        if "archer" in path.name:
            self.names.update({9: "forearm_draw", 10: "hand_draw"})
        self.ids = {name: pid for pid, name in self.names.items()}
        self.pivots = {
            n["name"][6:]: Vector(
                (n["translation"][0], -n["translation"][2], n["translation"][1])
            )
            for n in doc["nodes"]
            if n.get("name", "").startswith("pivot_")
        }
        self.labels = None
        if manifest is not None and "triangle_labels" in manifest:
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest()
                == manifest["source_sha256"]
            ), "Component labels must match the source GLB"
            self.labels = [
                manifest["component_names"][i] for i in manifest["triangle_labels"]
            ]
            assert len(self.labels) == len(self.triangles)
        elif manifest is not None:
            lookup = defaultdict(set)
            for i, face in enumerate(manifest["faces"]):
                for vertex in face:
                    lookup[
                        position_key(
                            np.asarray(manifest["vertices"][vertex], dtype=np.float32)
                        )
                    ].add(i)
            self.labels = []
            for face in self.triangles:
                matches = set.intersection(
                    *(lookup[position_key(self.positions_gltf[v])] for v in face)
                )
                labels = {manifest["components"][i] for i in matches}
                assert len(labels) == 1, ("Unmatched source triangle", face, labels)
                self.labels.append(labels.pop())
        image = doc["images"][0]
        view = doc["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        atlas_path = out_dir / "source_atlas.png"
        atlas_path.write_bytes(blob[start : start + view["byteLength"]])
        self.image = bpy.data.images.load(str(atlas_path), check_existing=False)
        self.image.alpha_mode = "CHANNEL_PACKED"
        width, height = self.image.size
        pixels = np.array(self.image.pixels[:], np.float32).reshape(height, width, 4)
        rgb = pixels[:, :, :3]
        pixels[:, :, :3] = np.where(
            rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4
        )
        self.pixels = pixels
        self.material = atlas_material(self.image)
        self.reference = self.make_reference(read)

    def make_reference(self, read):
        mesh = bpy.data.meshes.new("L0_reference")
        mesh.from_pydata(self.positions.tolist(), [], self.triangles.tolist())
        mesh.update()
        obj = bpy.data.objects.new("L0", mesh)
        bpy.context.scene.collection.objects.link(obj)
        mesh.materials.append(self.material)
        uv = mesh.uv_layers.new(name="UVMap")
        part = mesh.uv_layers.new(name="part")
        colors = mesh.color_attributes.new(
            name="Col", type="FLOAT_COLOR", domain="CORNER"
        )
        mesh.color_attributes.active_color = colors
        for pid in sorted(set(self.parts[:, 0].astype(int))):
            obj.vertex_groups.new(name=self.names[pid]).add(
                np.flatnonzero(self.parts[:, 0] == pid).tolist(), 1, "REPLACE"
            )
        for face in mesh.polygons:
            face.use_smooth = True
            for loop in face.loop_indices:
                vertex = mesh.loops[loop].vertex_index
                uv.data[loop].uv = self.uv[vertex]
                pid, pivot = self.parts[vertex]
                part.data[loop].uv = (pid, 1 - pivot)
                colors.data[loop].color = (1, 1, 1, 0)
        normals = read(self.primitive["attributes"]["NORMAL"])[:, [0, 2, 1]] * [
            1,
            -1,
            1,
        ]
        mesh.normals_split_custom_set_from_vertices(normals.tolist())
        for node in self.doc["nodes"]:
            if not node.get("name", "").startswith(("pivot_", "joint_")):
                continue
            empty = bpy.data.objects.new(node["name"], None)
            x, y, z = node.get("translation", [0, 0, 0])
            empty.location = (x, -z, y)
            bpy.context.scene.collection.objects.link(empty)
        return obj


def atlas_material(image):
    material = bpy.data.materials.new("atlas_opaque")
    material.use_nodes = True
    material.use_backface_culling = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.74
    bsdf.inputs["Metallic"].default_value = 0.08
    texture = material.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.extension = "EXTEND"
    uv = material.node_tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    material.node_tree.links.new(uv.outputs[0], texture.inputs["Vector"])
    material.node_tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    return material


def atlas_projection(obj, geometry, source, categories):
    points = [Vector(p) for p in source.positions]
    pixels = source.pixels
    height, width = pixels.shape[:2]
    weights = []
    for a in range(10):
        for b in range(10 - a):
            for offset in [1 / 3, 2 / 3] if a + b < 9 else [1 / 3]:
                u, v = (a + offset) / 10, (b + offset) / 10
                weights.append((u, v, 1 - u - v))
    weights = np.asarray(weights)
    trees, faces, candidates = {}, {}, {}
    # Existing mask transitions allow a far face to retain a fractional team share.
    y, x = np.nonzero((pixels[:, :, 3] > 0.2) & (pixels[:, :, 3] < 0.98))
    transition_uv = np.column_stack(((x + 0.5) / width, (y + 0.5) / height))
    transition_values = pixels[y, x]
    for label, source_labels in categories.items():
        if isinstance(source_labels, int):
            selected = source.parts[source.triangles[:, 0], 0] == source_labels
        else:
            selected = np.isin(source.labels, source_labels)
        faces[label] = source.triangles[selected]
        assert len(faces[label]), label
        trees[label] = BVHTree.FromPolygons(
            points, faces[label].tolist(), all_triangles=True
        )
        coords = np.einsum("wk,fkc->fwc", weights, source.uv[faces[label]]).reshape(
            -1, 2
        )
        x = np.clip((coords[:, 0] * width).astype(int), 0, width - 1)
        y = np.clip((coords[:, 1] * height).astype(int), 0, height - 1)
        uv = np.column_stack(((x + 0.5) / width, (y + 0.5) / height))
        values = pixels[y, x]
        if np.max(values[:, 3]) > 0.5:
            uv = np.concatenate([uv, transition_uv])
            values = np.concatenate([values, transition_values])
        candidates[label] = uv, values
    face_ids = obj.data.attributes["source_face"]
    uv0 = obj.data.uv_layers["UVMap"]
    colors = obj.data.color_attributes.active_color
    for face in obj.data.polygons:
        label = geometry.components[face_ids.data[face.index].value]
        vertices = [obj.data.vertices[v].co for v in face.vertices]
        samples = []
        for weight in weights:
            p = sum((v * w for v, w in zip(vertices, weight)), Vector())
            hit, _, index, _ = trees[label].ray_cast(
                p + face.normal * 0.5, -face.normal, 1.0
            )
            if hit is None:
                hit, _, index, _ = trees[label].find_nearest(p)
            triangle = faces[label][index]
            coords = source.uv[triangle]
            q = barycentric_transform(
                hit,
                *[points[v] for v in triangle],
                *[Vector((u, v, 0)) for u, v in coords],
            )
            x = min(width - 1, max(0, int(q.x * width)))
            y = min(height - 1, max(0, int(q.y * height)))
            samples.append(pixels[y, x])
        target = np.mean(samples, axis=0)
        uv, values = candidates[label]
        best = int(np.argmin(np.sum((values - target) ** 2 * [1, 1, 1, 0.4], axis=1)))
        for loop in face.loop_indices:
            uv0.data[loop].uv = uv[best]
            colors.data[loop].color = (1, 1, 1, float(values[best, 3]))
    return candidates


def fit_visibility(obj, geometry, source, candidates, out, match_global=False):
    """Match per-part visible means across eight facings without changing the atlas."""
    from measure_visibility import raster

    doc, blob = glb_inspect.load(str(out))
    levels = {n["name"]: doc["meshes"][n["mesh"]] for n in doc["nodes"] if "mesh" in n}
    primitive = levels[obj.name]["primitives"][0]
    positions = np.array(
        glb_inspect.accessor_values(doc, blob, primitive["attributes"]["POSITION"])[1]
    )
    triangles = np.array(
        glb_inspect.accessor_values(doc, blob, primitive["indices"])[1], int
    ).reshape(-1, 3)
    source_sums, source_counts = {}, {}
    counts = np.zeros(len(triangles))
    # Source pixels are stored bottom-up by Blender, glTF UVs are top-down.
    pixels = source.pixels[::-1]
    for azimuth in range(0, 360, 45):
        rgba, parts = raster(doc, blob, levels["L0"], azimuth, pixels)
        if obj.name == "L3":
            parts = np.where(parts >= 0, 0, -1)
        for pid in np.unique(parts[parts >= 0]):
            values = rgba[parts == pid]
            source_sums[pid] = source_sums.get(pid, np.zeros(4)) + values.sum(axis=0)
            source_counts[pid] = source_counts.get(pid, 0) + len(values)
        _, _, faces = raster(
            doc, blob, levels[obj.name], azimuth, pixels, face_indices=True
        )
        counts += np.bincount(faces[faces >= 0], minlength=len(triangles))
    lookup = {}
    for face in obj.data.polygons:
        points = [obj.data.vertices[v].co for v in face.vertices]
        key = tuple(sorted(position_key((v.x, v.z, -v.y)) for v in points))
        lookup[key] = face.index
    visibility = np.zeros(len(obj.data.polygons))
    for triangle, weight in zip(triangles, counts):
        key = tuple(sorted(position_key(positions[v]) for v in triangle))
        visibility[lookup[key]] += weight
    face_ids = obj.data.attributes["source_face"]
    labels = [
        geometry.components[face_ids.data[p.index].value] for p in obj.data.polygons
    ]
    part_ids = np.array(
        [source.ids[geometry.parts[p.vertices[0]]] for p in obj.data.polygons]
    )
    uv0 = obj.data.uv_layers["UVMap"]
    colors = obj.data.color_attributes.active_color
    height, width = source.pixels.shape[:2]

    def current_values():
        values = []
        for face in obj.data.polygons:
            u, v = uv0.data[face.loop_indices[0]].uv
            values.append(
                source.pixels[
                    min(height - 1, int(v * height)), min(width - 1, int(u * width))
                ]
            )
        return np.array(values)

    report = {}
    for pid in sorted(set(part_ids)):
        selected = np.flatnonzero(part_ids == pid)
        weights = visibility[selected]
        if weights.sum() == 0:
            continue
        weights = weights / weights.sum()
        desired = source_sums[pid] / source_counts[pid]
        values = current_values()[selected]
        initial = weights @ values
        for iteration in range(4):
            current = weights @ values
            ratio = np.divide(desired, current, out=np.ones(4), where=current > 1e-8)
            targets = np.clip(values * ratio, 0, 1)
            # Full-mask faces stay full; mixed boundary faces absorb the residual.
            residual = desired[3] - weights @ targets[:, 3]
            flexible = (values[:, 3] > 0.01) & (values[:, 3] < 0.99)
            available = weights[flexible].sum()
            if available > 0:
                targets[flexible, 3] = np.clip(
                    targets[flexible, 3] + residual / available, 0, 1
                )
            for row, face_index in enumerate(selected):
                uv, pool = candidates[labels[face_index]]
                best = int(
                    np.argmin(
                        np.sum((pool - targets[row]) ** 2 * [1, 1, 1, 0.7], axis=1)
                    )
                )
                values[row] = pool[best]
                face = obj.data.polygons[face_index]
                for loop in face.loop_indices:
                    uv0.data[loop].uv = uv[best]
                    colors.data[loop].color = (1, 1, 1, float(values[row, 3]))
        report[source.names[pid]] = {
            "source_mean": desired.tolist(),
            "initial_mean": initial.tolist(),
            "fitted_mean": (weights @ values).tolist(),
        }
    if match_global:
        # Simplified parts cover different pixel shares even after matching each part.
        # Fit the complete figure with the same component-restricted atlas samples.
        weights = visibility / visibility.sum()
        desired = sum(source_sums.values()) / sum(source_counts.values())
        values = current_values()
        initial = weights @ values
        for iteration in range(4):
            current = weights @ values
            ratio = np.divide(desired, current, out=np.ones(4), where=current > 1e-8)
            targets = np.clip(values * ratio, 0, 1)
            for face_index, face in enumerate(obj.data.polygons):
                if visibility[face_index] == 0:
                    continue
                uv, pool = candidates[labels[face_index]]
                best = int(
                    np.argmin(
                        np.sum(
                            (pool - targets[face_index]) ** 2 * [1, 1, 1, 0.7], axis=1
                        )
                    )
                )
                values[face_index] = pool[best]
                for loop in face.loop_indices:
                    uv0.data[loop].uv = uv[best]
                    colors.data[loop].color = (1, 1, 1, float(pool[best, 3]))
        report["whole_figure"] = {
            "source_mean": desired.tolist(),
            "initial_mean": initial.tolist(),
            "fitted_mean": (weights @ values).tolist(),
        }
    report_name = (
        "colour_fit.json" if obj.name == "L2" else f"{obj.name}_colour_fit.json"
    )
    (out.parent / report_name).write_text(json.dumps(report, indent=2) + "\n")


def merge_level(source, level, out, level_name="L2"):
    doc, blob = glb_inspect.load(str(source))
    extra, data = glb_inspect.load(str(level))
    doc = deepcopy(doc)
    assert not any(n.get("name") == level_name for n in doc["nodes"]), level_name
    accessor_offset = len(doc["accessors"])
    views = {}
    for index in sorted({a["bufferView"] for a in extra["accessors"]}):
        view = deepcopy(extra["bufferViews"][index])
        source_start = view.get("byteOffset", 0)
        blob += b"\0" * (-len(blob) % 4)
        view["byteOffset"] = len(blob)
        views[index] = len(doc["bufferViews"])
        doc["bufferViews"].append(view)
        blob += data[source_start : source_start + view["byteLength"]]
    for accessor in extra["accessors"]:
        accessor = deepcopy(accessor)
        accessor["bufferView"] = views[accessor["bufferView"]]
        doc["accessors"].append(accessor)
    mesh = deepcopy(extra["meshes"][0])
    mesh["name"] = level_name
    for primitive in mesh["primitives"]:
        primitive["material"] = 0
        primitive["indices"] += accessor_offset
        primitive["attributes"] = {
            key: value + accessor_offset
            for key, value in primitive["attributes"].items()
        }
    doc["nodes"].append({"name": level_name, "mesh": len(doc["meshes"])})
    doc["meshes"].append(mesh)
    doc["scenes"][doc.get("scene", 0)]["nodes"].append(len(doc["nodes"]) - 1)
    doc["buffers"][0]["byteLength"] = len(blob)
    blob += b"\0" * (-len(blob) % 4)
    encoded = json.dumps(doc, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    out.write_bytes(
        struct.pack("<III", 0x46546C67, 2, 28 + len(encoded) + len(blob))
        + struct.pack("<I4s", len(encoded), b"JSON")
        + encoded
        + struct.pack("<I4s", len(blob), b"BIN\0")
        + blob
    )


def finish(source, geometry, categories, out):
    obj = geometry.mesh(source.material)
    assert 150 <= len(obj.data.polygons) <= 250, len(obj.data.polygons)
    candidates = atlas_projection(obj, geometry, source, categories)
    assert all(
        len(v.groups) == 1 and v.groups[0].weight == 1 for v in obj.data.vertices
    )
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    path = str(out.with_name("L2_export.glb"))
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_normals=True,
        export_texcoords=True,
        export_vertex_color="ACTIVE",
    )
    merge_level(source.path, Path(path), out)
    fit_visibility(obj, geometry, source, candidates, out)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_normals=True,
        export_texcoords=True,
        export_vertex_color="ACTIVE",
    )
    merge_level(source.path, Path(path), out)
    manifest = {
        "vertices": [(x, z, -y) for x, y, z in geometry.vertices],
        "faces": geometry.faces,
        "components": geometry.components,
        "parts": geometry.parts,
    }
    (out.parent / "L2_surfaces.json").write_text(json.dumps(manifest) + "\n")
    bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix(".blend")))
    print("L2_BUILD_DONE", str(out), len(obj.data.polygons), flush=True)
