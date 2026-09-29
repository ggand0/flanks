"""Build shrub mesh parts and far textures for build_shrub.py.

Run through Blender with --python tools/blender/shrub_a/build_shrub.py.
"""
import math

import bpy
import numpy as np
from mathutils import Vector

CFG = {}
TAU = 2.0 * math.pi
LEAF_RECTS = [
    (0.015, 0.505, 0.485, 0.995),
    (0.515, 0.505, 0.985, 0.995),
    (0.015, 0.005, 0.485, 0.495),
    (0.515, 0.005, 0.985, 0.495),
]


class Builder:
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.uvs = []
        self.colors = []
        self.normals = []
        self.weights = []
        self.wood_triangles = 0
        self.material_ids = []

    def face(self, points, uvs, normals=None, color=(1, 1, 1, 1), leaf=False):
        start = len(self.vertices)
        self.vertices.extend(tuple(p) for p in points)
        self.faces.append(tuple(range(start, start + len(points))))
        self.material_ids.append(0 if leaf else 1)
        self.uvs.extend(uvs)
        self.colors.extend([color] * len(points))
        face_normal = (points[1] - points[0]).cross(points[2] - points[0]).normalized()
        self.normals.extend(normals or [tuple(face_normal)] * len(points))
        self.weights.extend(
            [
                (max(0, min(1, p.z / CFG["height"])) ** 2, 1.0 if leaf else 0.0)
                for p in points
            ]
        )
        if not leaf:
            self.wood_triangles += len(points) - 2

    def tube(self, points, radii, sides):
        rings = []
        normals = []
        for i, (p, radius) in enumerate(zip(points, radii)):
            tangent = points[min(i + 1, len(points) - 1)] - points[max(0, i - 1)]
            tangent.normalize()
            ref = Vector((0, 1, 0)) if abs(tangent.y) < 0.9 else Vector((1, 0, 0))
            a = tangent.cross(ref).normalized()
            b = tangent.cross(a).normalized()
            ns = [
                (a * math.cos(TAU * j / sides) + b * math.sin(TAU * j / sides))
                for j in range(sides)
            ]
            ring = [
                p + n * radius * (1 + 0.07 * math.sin(j * 2.3 + i))
                for j, n in enumerate(ns)
            ]
            if i == 0 and abs(p.z) < 0.0001:
                for vertex in ring:
                    vertex.z = 0.0
            rings.append(ring)
            normals.append(ns)
        for i in range(len(points) - 1):
            # Mirrored coordinates meet continuously around the tube and at each ring.
            for j in range(sides):
                k = (j + 1) % sides
                u0 = 1 - abs(2 * j / sides - 1)
                u1 = 1 - abs(2 * (j + 1) / sides - 1)
                v0, v1 = i % 2, 1 - i % 2
                self.face(
                    [rings[i][j], rings[i][k], rings[i + 1][k], rings[i + 1][j]],
                    [(u0, v0), (u1, v0), (u1, v1), (u0, v1)],
                    [
                        normals[i][j],
                        normals[i][k],
                        normals[i + 1][k],
                        normals[i + 1][j],
                    ],
                )
        self.face(list(reversed(rings[0])), [(0.5, 0.5)] * sides)
        self.face(rings[-1], [(0.5, 0.5)] * sides)

    def shoot(self, base, direction, length, width, roll, rect, tint, folded):
        forward = direction.normalized()
        side = forward.cross(Vector((0, 0, 1))).normalized()
        normal = side.cross(forward).normalized()
        side, normal = (
            side * math.cos(roll) + normal * math.sin(roll),
            normal * math.cos(roll) - side * math.sin(roll),
        )
        tip = base + forward * length
        # A shallow fold gives each attached shoot depth from oblique viewpoints.
        left0 = base - side * width * 0.20 - normal * width * 0.06
        right0 = base + side * width * 0.20 - normal * width * 0.06
        left1 = tip - side * width / 2 - normal * width * 0.23
        right1 = tip + side * width / 2 - normal * width * 0.08
        u0, v0, u1, v1 = rect
        um = (u0 + u1) / 2
        color = tuple(tint * v for v in CFG["leaf_tint"]) + (1,)
        panels = [
            ([left0, base, tip, left1], [(u0, v0), (um, v0), (um, v1), (u0, v1)], -1),
            ([base, right0, right1, tip], [(um, v0), (u1, v0), (u1, v1), (um, v1)], 1),
        ]
        if not folded:
            panels = [
                (
                    [left0, right0, right1, left1],
                    [(u0, v0), (u1, v0), (u1, v1), (u0, v1)],
                    0,
                )
            ]
        for points, uvs, sign in panels:
            normals = []
            for _ in points:
                n = (
                    normal * 0.60 + Vector((0, 0, 0.75)) + side * sign * 0.16
                ).normalized()
                normals.append(tuple(n))
            self.face(points, uvs, normals, color, leaf=True)

    def mesh(self, name, materials):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(self.vertices, [], self.faces)
        mesh.update()
        uv0 = mesh.uv_layers.new(name="UVMap")
        uv1 = mesh.uv_layers.new(name="wind")
        col = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="CORNER")
        mesh.color_attributes.active_color = col
        for loop in mesh.loops:
            index = loop.vertex_index
            uv0.data[loop.index].uv = self.uvs[index]
            # Blender's exporter flips V; store its complement so GLB has the weight.
            bend, flutter = self.weights[index]
            uv1.data[loop.index].uv = (bend, 1 - flutter)
            col.data[loop.index].color = self.colors[index]
        for poly, material_id in zip(mesh.polygons, self.material_ids):
            poly.use_smooth = True
            poly.material_index = material_id
        mesh.normals_split_custom_set_from_vertices(self.normals)
        bake_normals = mesh.attributes.new(
            name="bake_normal", type="FLOAT_VECTOR", domain="POINT"
        )
        for value, normal in zip(bake_normals.data, self.normals):
            value.vector = normal
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        for mat in materials:
            obj.data.materials.append(mat)
        return obj


def material(name, image, opaque=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = opaque
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = 0.92
    bsdf.inputs["Specular IOR Level"].default_value = 0.15
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    color = nodes.new("ShaderNodeVertexColor")
    color.layer_name = "Col"
    multiply = nodes.new("ShaderNodeMixRGB")
    multiply.name = "LeafAlbedo"
    multiply.blend_type = "MULTIPLY"
    multiply.inputs[0].default_value = 1
    links.new(tex.outputs["Color"], multiply.inputs[1])
    links.new(color.outputs["Color"], multiply.inputs[2])
    links.new(multiply.outputs[0], bsdf.inputs["Base Color"])
    if opaque:
        return mat
    cutoff = nodes.new("ShaderNodeMath")
    cutoff.operation = "GREATER_THAN"
    cutoff.inputs[1].default_value = 0.5
    links.new(tex.outputs["Alpha"], cutoff.inputs[0])
    links.new(cutoff.outputs[0], bsdf.inputs["Alpha"])
    return mat


def camera_at(scene, target, yaw, elevation, distance=40):
    cam = scene.camera
    offset = (
        Vector(
            (
                math.sin(yaw) * math.cos(elevation),
                -math.cos(yaw) * math.cos(elevation),
                math.sin(elevation),
            )
        )
        * distance
    )
    cam.location = Vector(target) + offset
    cam.rotation_euler = (
        (Vector(target) - cam.location).to_track_quat("-Z", "Y").to_euler()
    )


def bake_card(scene, oak, out_dir):
    # Bake colour and the authored volume normals separately, for live far lighting.
    bark_mat = oak.data.materials[1]
    bark_nodes, bark_links = bark_mat.node_tree.nodes, bark_mat.node_tree.links
    bark_output = next(n for n in bark_nodes if n.type == "OUTPUT_MATERIAL")
    bark_bsdf = next(n for n in bark_nodes if n.type == "BSDF_PRINCIPLED")
    bark_tex = next(n for n in bark_nodes if n.type == "TEX_IMAGE")
    bark_emit = bark_nodes.new("ShaderNodeEmission")
    bark_links.new(bark_emit.outputs[0], bark_output.inputs["Surface"])
    bark_attr = bark_nodes.new("ShaderNodeAttribute")
    bark_attr.attribute_name = "bake_normal"
    bark_remap = bark_nodes.new("ShaderNodeVectorMath")
    bark_remap.operation = "MULTIPLY_ADD"
    bark_remap.inputs[1].default_value = (0.5, 0.5, 0.5)
    bark_remap.inputs[2].default_value = (0.5, 0.5, 0.5)
    bark_combine = bark_nodes.new("ShaderNodeCombineXYZ")
    bark_links.new(bark_combine.outputs[0], bark_remap.inputs[0])
    bark_dots = []
    for channel in ["X", "Y", "Z"]:
        dot = bark_nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        bark_links.new(bark_attr.outputs["Vector"], dot.inputs[0])
        bark_links.new(dot.outputs["Value"], bark_combine.inputs[channel])
        bark_dots.append(dot)
    mat = oak.data.materials[0]
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    tex = next(n for n in nodes if n.type == "TEX_IMAGE")
    cutoff = next(n for n in nodes if n.type == "MATH")
    emit = nodes.new("ShaderNodeEmission")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(cutoff.outputs[0], mix.inputs[0])
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(emit.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "bake_normal"
    combine = nodes.new("ShaderNodeCombineXYZ")
    remap = nodes.new("ShaderNodeVectorMath")
    remap.operation = "MULTIPLY_ADD"
    remap.inputs[1].default_value = (0.5, 0.5, 0.5)
    remap.inputs[2].default_value = (0.5, 0.5, 0.5)
    links.new(combine.outputs[0], remap.inputs[0])
    dots = []
    for channel in ["X", "Y", "Z"]:
        dot = nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        links.new(attr.outputs["Vector"], dot.inputs[0])
        links.new(dot.outputs["Value"], combine.inputs[channel])
        dots.append(dot)
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = CFG["frame"]
    scene.render.resolution_x = 256
    scene.render.resolution_y = 256
    scene.render.film_transparent = True
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    baked = {}
    for mode in ["color", "normal"]:
        scene.view_settings.view_transform = "Standard" if mode == "color" else "Raw"
        links.new(
            nodes["LeafAlbedo"].outputs[0] if mode == "color" else remap.outputs[0],
            emit.inputs["Color"],
        )
        bark_links.new(
            bark_nodes["LeafAlbedo"].outputs[0]
            if mode == "color"
            else bark_remap.outputs[0],
            bark_emit.inputs["Color"],
        )
        pixels = []
        for i in range(2):
            basis = (
                [(1, 0, 0), (0, 0, 1), (0, -1, 0)]
                if i == 0
                else [(0, 1, 0), (0, 0, 1), (1, 0, 0)]
            )
            for dot, axis in zip(bark_dots, basis):
                dot.inputs[1].default_value = axis
            for dot, axis in zip(dots, basis):
                dot.inputs[1].default_value = axis
            camera_at(scene, (0, 0, CFG["frame"] / 2), i * math.pi / 2, 0.0)
            path = out_dir / f"card_{mode}_view_{i}.png"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            rendered = bpy.data.images.load(str(path), check_existing=False)
            if mode == "normal":
                rendered.colorspace_settings.name = "Non-Color"
            data = np.empty(256 * 256 * 4, dtype=np.float32)
            rendered.pixels.foreach_get(data)
            pixels.append(data.reshape(256, 256, 4))
        atlas = np.concatenate(pixels, axis=1)
        image = bpy.data.images.new(
            f"tree_far_{mode}_atlas", width=512, height=256, alpha=True
        )
        if mode == "normal":
            image.colorspace_settings.name = "Non-Color"
        image.pixels.foreach_set(atlas.ravel())
        image.filepath_raw = str(out_dir / f"tree_far_{mode}_atlas.png")
        image.file_format = "PNG"
        image.save()
        image.pack()
        baked[mode] = image
    bark_links.new(bark_bsdf.outputs[0], bark_output.inputs["Surface"])
    for node in [bark_emit, bark_attr, bark_remap, bark_combine] + bark_dots:
        bark_nodes.remove(node)
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    for node in [emit, transparent, mix, attr, combine, remap] + dots:
        nodes.remove(node)
    scene.view_settings.view_transform = "Standard"
    far_mat = material("oak_far", baked["color"])
    far_nodes = far_mat.node_tree.nodes
    far_bsdf = next(n for n in far_nodes if n.type == "BSDF_PRINCIPLED")
    normal_tex = far_nodes.new("ShaderNodeTexImage")
    normal_tex.image = baked["normal"]
    normal_map = far_nodes.new("ShaderNodeNormalMap")
    far_mat.node_tree.links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
    far_mat.node_tree.links.new(normal_map.outputs["Normal"], far_bsdf.inputs["Normal"])
    buf = Builder()
    for i in range(2):
        half = CFG["frame"] / 2
        a = Vector((half, 0, 0)) if i == 0 else Vector((0, half, 0))
        low = Vector((0, 0, 0))
        high = Vector((0, 0, CFG["frame"]))
        u0, u1 = i * 0.5, (i + 1) * 0.5
        buf.face(
            [low - a, low + a, high + a, high - a],
            [(u0, 0), (u1, 0), (u1, 1), (u0, 1)],
            leaf=True,
        )
    return buf.mesh("card", [far_mat])
