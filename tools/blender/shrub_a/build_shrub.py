"""Build an upright multi-stem shrub with near, middle and crossed far detail.

Run in headless Blender with --python build_shrub.py.
Read shared textures from assets/vegetation/shrub_b_sandbox.glb.
Run from the repository root; --out sets the generated output directory.
"""
import argparse
import json
import math
import random
import sys
import struct
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shrub_mesh as mesh

PROFILE = dict(
    height=1.7,
    width=2.2,
    frame=2.7,
    budgets=[2800, 900, 4],
    leaf_tint=(0.66, 0.73, 0.72),
)


def structure():
    paths = [
        [
            (-0.09, 0.02, 0),
            (-0.16, 0.00, 0.32),
            (-0.33, 0.02, 0.88),
            (-0.43, 0.07, 1.39),
        ],
        [
            (0.04, -0.04, 0),
            (0.08, -0.12, 0.31),
            (0.32, -0.22, 0.81),
            (0.42, -0.25, 1.13),
        ],
        [
            (-0.02, -0.07, 0),
            (-0.19, -0.22, 0.27),
            (-0.47, -0.39, 0.67),
            (-0.63, -0.43, 0.94),
        ],
        [(0.10, 0.00, 0), (0.29, 0.02, 0.27), (0.61, 0.13, 0.59), (0.78, 0.16, 0.77)],
        [(0.04, 0.08, 0), (0.13, 0.29, 0.33), (0.26, 0.52, 0.90), (0.22, 0.60, 1.24)],
        [
            (-0.07, 0.07, 0),
            (-0.26, 0.19, 0.23),
            (-0.51, 0.39, 0.57),
            (-0.69, 0.48, 0.75),
        ],
        [(0.01, 0.01, 0), (-0.01, -0.03, 0.44), (0.04, 0.03, 1.05), (0.08, 0.14, 1.34)],
    ]
    stems = [
        ([Vector(p) for p in path], [0.040, 0.031, 0.014, 0.0025]) for path in paths
    ]
    forks = []
    for index, (points, _) in enumerate(stems):
        for side in [-1, 1]:
            joint = points[1].lerp(points[2], 0.14 if side == -1 else 0.52)
            angle = index * 2.39996 + side * 0.95
            reach = 0.48 if side == -1 else 0.34
            tip = joint + Vector(
                (
                    math.cos(angle) * reach,
                    math.sin(angle) * reach,
                    0.17 if side == -1 else 0.34,
                )
            )
            mid = joint.lerp(tip, 0.52) + Vector((0, 0, -0.025))
            forks.append(([joint, mid, tip], [0.018, 0.010, 0.002]))
    return stems, forks


class Shoots(mesh.Builder):
    def __init__(self):
        super().__init__()
        self.shoots = []

    def shoot(self, *args, **kwargs):
        first = len(self.vertices)
        super().shoot(*args, **kwargs)
        self.shoots.append((first, len(self.vertices)))


def near_geometry():
    stems, forks = structure()
    buf = Shoots()
    for points, radii in stems:
        buf.tube(points, radii, 5)
    for points, radii in forks:
        buf.tube(points, radii, 3)
    rng = random.Random(3107)
    for index, (points, _) in enumerate(stems + forks):
        count = 40
        for j in range(count):
            # Alternate sprays around each branch while leaving the short basal wood visible.
            t = (0.15 if index < len(stems) else 0.08) + (
                0.83 if index < len(stems) else 0.89
            ) * (j + rng.uniform(0.2, 0.8)) / count
            segment = min(int(t * (len(points) - 1)), len(points) - 2)
            fraction = t * (len(points) - 1) - segment
            base = points[segment].lerp(points[segment + 1], fraction)
            tangent = (points[segment + 1] - points[segment]).normalized()
            side = tangent.cross(Vector((0, 0, 1))).normalized()
            around = tangent.cross(side).normalized()
            angle = j * 2.39996 + index * 0.77
            outward = side * math.cos(angle) + around * math.sin(angle)
            direction = (
                tangent * rng.uniform(0.3, 0.65) + outward * rng.uniform(0.65, 1.0)
            ).normalized()
            direction.z = max(direction.z, -0.12 if base.z > 0.4 else 0.22)
            length = rng.uniform(0.28, 0.43) * (0.84 if t > 0.83 else 1.0)
            width = length * rng.uniform(0.82, 1.02)
            roll = rng.uniform(-1.3, 1.3)
            first = len(buf.vertices)
            buf.shoot(
                base,
                direction,
                length,
                width,
                roll,
                mesh.LEAF_RECTS[(index + j) % 4],
                rng.uniform(0.88, 1.05),
                folded=(index * count + j) % 4 == 0,
            )
            # Shorten low sprays as a whole to keep leaf shapes clear of the soil.
            fit = 1.0
            for p in buf.vertices[first:]:
                if p[2] < 0.035:
                    fit = min(fit, (base.z - 0.035) / (base.z - p[2]))
            assert fit > 0
            buf.vertices[first:] = [
                tuple(base[k] + fit * (p[k] - base[k]) for k in range(3))
                for p in buf.vertices[first:]
            ]
    coords = np.array(buf.vertices)
    scale = [
        2.2 / np.ptp(coords[:, 0]),
        1.85 / np.ptp(coords[:, 1]),
        1.7 / coords[:, 2].max(),
    ]
    buf.vertices = [tuple(p[k] * scale[k] for k in range(3)) for p in buf.vertices]
    buf.normals = [
        tuple(Vector(tuple(n[k] / scale[k] for k in range(3))).normalized())
        for n in buf.normals
    ]
    buf.weights = [
        (max(0, min(1, p[2] / 1.7)) ** 2, marker[1])
        for p, marker in zip(buf.vertices, buf.weights)
    ]
    return buf


def middle_geometry(near):
    buf = mesh.Builder()
    for face, material in zip(near.faces, near.material_ids):
        if material == 1:
            buf.face(
                [Vector(near.vertices[i]) for i in face],
                [near.uvs[i] for i in face],
                [near.normals[i] for i in face],
                near.colors[face[0]],
            )
    lower = np.min(near.vertices, axis=0)
    upper = np.max(near.vertices, axis=0)
    for sample in range(210):
        start, end = near.shoots[round(sample * (len(near.shoots) - 1) / 209)]
        ids = [start + i for i in ([0, 5, 6, 3] if end - start == 8 else [0, 1, 2, 3])]
        points = [Vector(near.vertices[i]) for i in ids]
        base = (points[0] + points[1]) * 0.5
        expanded = [base + (p - base) * 1.80 for p in points]
        fit = 1.0
        for p in expanded:
            for axis in range(3):
                delta = p[axis] - base[axis]
                if delta > 1e-8:
                    fit = min(fit, (upper[axis] - base[axis]) / delta)
                elif delta < -1e-8:
                    fit = min(fit, (lower[axis] - base[axis]) / delta)
        buf.face(
            [base + (p - base) * fit for p in expanded],
            [near.uvs[i] for i in ids],
            [near.normals[i] for i in ids],
            near.colors[ids[0]],
            leaf=True,
        )
    return buf


def load_materials(path, out):
    """Extract the shared generated foliage and bark from the shipped B shrub."""
    data = path.read_bytes()
    assert data[:4] == b"glTF"
    offset, chunks = 12, {}
    while offset < len(data):
        length, tag = struct.unpack_from("<I4s", data, offset)
        chunks[tag] = data[offset + 8 : offset + 8 + length]
        offset += 8 + length
    document = json.loads(chunks[b"JSON"])
    blob = chunks[b"BIN\0"]
    node = next(n for n in document["nodes"] if n.get("name") == "L0")
    materials = {}
    for primitive in document["meshes"][node["mesh"]]["primitives"]:
        material = document["materials"][primitive["material"]]
        mode = material.get("alphaMode", "OPAQUE")
        index = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        image = document["images"][document["textures"][index]["source"]]
        view = document["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        filename = out / (
            "source_bark.png" if mode == "OPAQUE" else "source_foliage.png"
        )
        filename.write_bytes(blob[start : start + view["byteLength"]])
        texture = bpy.data.images.load(str(filename))
        texture.pack()
        materials[mode] = mesh.material(mode.lower(), texture, opaque=mode == "OPAQUE")
    return [materials["MASK"], materials["OPAQUE"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("assets_dev/vegetation/shrub_a_rebuild/shrub_a"),
    )
    parser.add_argument(
        "--texture-source",
        type=Path,
        default=Path("assets/vegetation/shrub_b_sandbox.glb"),
    )
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    mesh.CFG.update(PROFILE)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    materials = load_materials(args.texture_source, out)
    geometry = near_geometry()
    near = geometry.mesh("L0", materials)
    middle = middle_geometry(geometry).mesh("L1", materials)
    middle.hide_render = True
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_percentage = 100
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs[0].default_value = (0.3, 0.35, 0.4, 1)
    background.inputs[1].default_value = 0.35
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    far = mesh.bake_card(scene, near, out)
    far.hide_render = True
    root = bpy.data.objects.new("variant_0", None)
    scene.collection.objects.link(root)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in [root, near, middle, far]:
        if obj != root:
            obj.parent = root
        obj.select_set(True)
    bpy.context.view_layer.objects.active = near
    bpy.ops.export_scene.gltf(
        filepath=str(out / "tree.glb"),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_normals=True,
        export_texcoords=True,
        export_vertex_color="ACTIVE",
    )
    report = dict(
        shoots=len(geometry.shoots),
        stems=7,
        forks=14,
        near_wood_triangles=geometry.wood_triangles,
        leaf_tint=PROFILE["leaf_tint"],
    )
    for obj, budget in zip([near, middle, far], PROFILE["budgets"]):
        obj.data.calc_loop_triangles()
        coords = np.array([v.co[:] for v in obj.data.vertices])
        report[obj.name] = dict(
            triangles=len(obj.data.loop_triangles),
            dimensions=np.ptp(coords, axis=0).tolist(),
        )
        assert len(obj.data.loop_triangles) <= budget
    (out / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    scene.collection.objects.link(sun)
    sun.data.energy = 2.2
    sun.rotation_euler = (0.6, -0.35, -0.6)
    scene.render.resolution_x = 900
    scene.render.resolution_y = 900
    for obj in [near, middle, far]:
        for other in [near, middle, far]:
            other.hide_render = obj != other
        mesh.camera_at(scene, (0, 0, 0.85), 0.6, math.radians(20))
        scene.render.filepath = str(out / f"preview_{obj.name}.png")
        bpy.ops.render.render(write_still=True)
    near.hide_render = False
    middle.hide_render = True
    far.hide_render = True
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "tree.blend"))


if __name__ == "__main__":
    main()
