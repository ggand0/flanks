"""Build a 0.76 m arrow with closed geometry and vertex colours.

blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/arrow/build_arrow.py -- --out assets_dev/arrow/arrow.glb
"""

import argparse
import json
import math
from pathlib import Path
import sys

import bmesh
import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[3]
WOOD = (0.32, 0.17, 0.065, 0.0)
NOCK = (0.14, 0.075, 0.028, 0.0)
STEEL = (0.24, 0.28, 0.31, 0.0)
FEATHER = (0.82, 0.77, 0.63, 0.0)


def make_material():
    material = bpy.data.materials.new("Arrow_vertex_colour")
    material.use_nodes = True
    material.use_backface_culling = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Roughness"].default_value = 0.72
    colour = material.node_tree.nodes.new("ShaderNodeVertexColor")
    colour.layer_name = "Color"
    material.node_tree.links.new(colour.outputs["Color"], shader.inputs["Base Color"])
    return material


def build():
    vertices, faces, colours, labels = [], [], [], []

    def solid(points, polygons, label, colour):
        offset = len(vertices)
        # Author in exported XYZ: Blender maps (x, -z, y) back to Y-up.
        vertices.extend((x, -z, y) for x, y, z in points)
        for polygon in polygons:
            for i in range(1, len(polygon) - 1):
                faces.append(
                    tuple(offset + p for p in (polygon[0], polygon[i], polygon[i + 1]))
                )
                colours.append(colour)
                labels.append(label)

    ring = [
        (
            0.005 * math.cos(math.radians(18 + i * 72)),
            0.005 * math.sin(math.radians(18 + i * 72)),
        )
        for i in range(5)
    ]
    radius_x = max(x for x, _ in ring)
    # The rear cap is two sloping planes meeting in an 8 mm string notch.
    rear = [(x, y, -0.350 + 0.008 * (1 - abs(x) / radius_x)) for x, y in ring]
    rear.append((0.0, ring[3][1], -0.342))
    points = rear + [(x, y, 0.350) for x, y in ring]
    polygons = [
        (0, 6, 7, 1),
        (1, 7, 8, 2),
        (2, 8, 9, 3),
        (3, 9, 10, 4, 5),
        (4, 10, 6, 0),
        (6, 10, 9, 8, 7),
        (0, 1, 5, 4),
        (1, 2, 3, 5),
    ]
    solid(points, polygons, "shaft_and_nock", WOOD)
    for i in range(len(colours) - 4, len(colours)):
        colours[i] = NOCK

    solid(
        [
            (-0.0075, 0, 0.346),
            (0, 0.0075, 0.346),
            (0.0075, 0, 0.346),
            (0, -0.0075, 0.346),
            (0, 0, 0.410),
        ],
        [(0, 1, 2, 3), (0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0)],
        "bodkin",
        STEEL,
    )

    for vane in range(3):
        angle = math.radians(90 + vane * 120)
        radial = (math.cos(angle), math.sin(angle))
        tangent = (-radial[1], radial[0])

        def feather_point(radius, offset, z):
            return (
                radius * radial[0] + offset * tangent[0],
                radius * radial[1] + offset * tangent[1],
                z,
            )

        solid(
            [
                feather_point(0.003, 0, -0.316),
                feather_point(0.003, 0, -0.171),
                feather_point(0.020, -0.00065, -0.291),
                feather_point(0.020, 0.00065, -0.291),
            ],
            [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)],
            f"fletching_{vane + 1}",
            FEATHER,
        )

    mesh = bpy.data.meshes.new("Arrow_geometry")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("L0", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(make_material())
    obj.vertex_groups.new(name="projectile").add(
        list(range(len(vertices))), 1.0, "REPLACE"
    )
    mesh.uv_layers.new(name="UVMap")
    part = mesh.uv_layers.new(name="part")
    color = mesh.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="CORNER")
    mesh.color_attributes.active_color = color
    for polygon, rgba in zip(mesh.polygons, colours):
        for loop in polygon.loop_indices:
            part.data[loop].uv = (7.0, 1.0)
            color.data[loop].color = rgba
    normals = []
    for polygon in mesh.polygons:
        for loop in polygon.loop_indices:
            point = mesh.vertices[mesh.loops[loop].vertex_index].co
            # Radial shaft normals keep its five-sided section round in light.
            normal = (
                Vector((point.x, 0, point.z)).normalized()
                if polygon.index < 11
                else polygon.normal
            )
            normals.append(normal)
    mesh.normals_split_custom_set(normals)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    assert len(mesh.polygons) == 36
    manifest = {
        "vertices": [(v.co.x, v.co.z, -v.co.y) for v in mesh.vertices],
        "faces": [list(p.vertices) for p in mesh.polygons],
        "components": labels,
        "shaft_endpoints_z_m": [-0.350, 0.350],
        "nock_depth_m": 0.008,
        "fletching_radial_tip_m": 0.020,
    }
    return obj, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "assets_dev/arrow/arrow.glb")
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    assert bpy.app.background, "Use a separate background Blender process"
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = "METRIC"
    obj, manifest = build()
    path = str(args.out.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_normals=True,
        export_texcoords=True,
        export_vertex_color="ACTIVE",
    )
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out.with_suffix(".blend").resolve()))
    print("ARROW", path, "triangles", len(obj.data.polygons), flush=True)


if __name__ == "__main__":
    main()
