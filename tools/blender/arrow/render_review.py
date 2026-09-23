"""Render the exported arrow, box comparison and soldier scale reference.

blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/arrow/render_review.py -- assets_dev/arrow/arrow.glb
"""

import argparse
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_arrow import make_material

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/blender"))
sys.path.insert(0, str(ROOT / "tools/blender/knight"))
import glb_inspect
import build_knight_textured as review


def read_arrow(path):
    doc, blob = glb_inspect.load(path)
    prim = doc["meshes"][0]["primitives"][0]

    def values(name):
        return glb_inspect.accessor_values(doc, blob, prim["attributes"][name])[1]

    positions = [(x, -z, y) for x, y, z in values("POSITION")]
    indices = [
        int(v[0]) for v in glb_inspect.accessor_values(doc, blob, prim["indices"])[1]
    ]
    mesh = bpy.data.meshes.new("Exported_arrow")
    mesh.from_pydata(
        positions, [], [indices[i : i + 3] for i in range(0, len(indices), 3)]
    )
    mesh.update()
    colors = mesh.color_attributes.new(
        name="Color", type="FLOAT_COLOR", domain="CORNER"
    )
    rgba = values("COLOR_0")
    for loop in mesh.loops:
        colors.data[loop.index].color = rgba[loop.vertex_index]
    mesh.normals_split_custom_set_from_vertices(
        [(x, -z, y) for x, y, z in values("NORMAL")]
    )
    obj = bpy.data.objects.new("Exported_L0", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(make_material())
    return obj


def camera(target, direction, width, up=(0, 0, 1)):
    scene = bpy.context.scene
    if scene.camera is None:
        scene.camera = bpy.data.objects.new(
            "ReviewCamera", bpy.data.cameras.new("ReviewCamera")
        )
        scene.collection.objects.link(scene.camera)
    cam = scene.camera
    back = Vector(direction).normalized()
    right = Vector(up).cross(back).normalized()
    vertical = back.cross(right)
    cam.rotation_euler = Matrix((right, vertical, back)).transposed().to_euler()
    cam.location = Vector(target) + 4 * back
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = width
    bpy.context.view_layer.update()


def render(folder, name, width=960, height=300):
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.filepath = str(folder / f"{name}.png")
    bpy.ops.render.render(write_still=True)


def box_reference():
    objects = []
    for label, center, half, color in [
        ("shaft", (0, 0, 0), (0.014, 0.014, 0.36), (0.44, 0.30, 0.18, 1)),
        ("head", (0, 0, 0.385), (0.022, 0.022, 0.035), (0.88, 0.91, 0.97, 1)),
        ("fletching", (0, 0, -0.31), (0.03, 0.03, 0.06), (0.88, 0.86, 0.78, 1)),
    ]:
        x, y, z = center
        bpy.ops.mesh.primitive_cube_add(size=2, location=(x, -z, y))
        obj = bpy.context.object
        obj.name = "Box_reference_" + label
        obj.scale = (half[0], half[2], half[1])
        material = bpy.data.materials.new(obj.name)
        material.use_nodes = True
        material.use_backface_culling = True
        material.node_tree.nodes.get("Principled BSDF").inputs[
            "Base Color"
        ].default_value = color
        obj.data.materials.append(material)
        objects.append(obj)
    return objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("glb", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    assert bpy.app.background
    bpy.ops.wm.read_factory_settings(use_empty=True)
    folder = args.glb.resolve().parent
    arrow = read_arrow(args.glb)
    review.render_setup()
    scene = bpy.context.scene
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    camera((0, -0.03, 0), (-1, 0, 0), 0.88)
    render(folder, "side")
    camera((0, -0.03, 0), (0, 0, 1), 0.88, up=(1, 0, 0))
    render(folder, "top")
    camera((0, 0.28, 0), (0, 1, 0), 0.065)
    render(folder, "rear", 600, 600)
    camera((0, 0.255, 0), (-1, 0.9, 0.45), 0.205)
    render(folder, "fletching_closeup", 800, 600)
    camera((0, -0.365, 0), (-1, -0.3, 0.55), 0.105)
    render(folder, "head_closeup", 800, 600)
    camera((0, 0.344, 0), (0, 1, 1), 0.029)
    render(folder, "nock_closeup", 600, 600)
    camera((0, -0.03, 0), (-1, 0, 0), 0.76 * 64 / 10)
    render(folder, "ten_px", 64, 64)
    arrow.hide_render = True
    boxes = box_reference()
    camera((0, -0.03, 0), (-1, 0, 0), 0.88)
    render(folder, "boxes_side")
    camera((0, -0.025, 0), (-1, 0, 0), 0.79 * 64 / 10)
    render(folder, "boxes_ten_px", 64, 64)
    for obj in boxes:
        obj.hide_render = True

    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(ROOT / "assets/units/archer.glb"))
    imported = set(bpy.data.objects) - before
    for obj in imported:
        obj.hide_render = obj.type == "MESH" and obj.name != "L0"
        if obj.type == "MESH" and obj.name == "L0":
            obj.data.materials[0] = review.tint_material(
                obj.data.materials[0], review.TEAM_RED
            )
            obj.data.materials[0].use_backface_culling = True
    arrow.hide_render = False
    # Rotate only the review instance upright beside the 1.80 m soldier.
    from math import pi

    arrow.rotation_euler.x = -pi / 2
    arrow.location = (0.62, -0.05, 0.350)
    camera((0.20, 0, 0.90), (0.25, -1, 0.16), 1.85)
    render(folder, "soldier_scale", 850, 1050)
    print("ARROW_REVIEW_DONE", flush=True)


if __name__ == "__main__":
    main()
