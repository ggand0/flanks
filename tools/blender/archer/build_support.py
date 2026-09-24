"""Shared baking and review lighting for the textured archer."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import types

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import geometry_archer as geo

TEAM_RED = (0.58, 0.060, 0.032)
TEAM_BLUE = (0.055, 0.21, 0.46)


def node(mat, kind):
    return mat.node_tree.nodes.new(kind)


from materials import category, source_material


def unwrap(obj, g, size):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.data.uv_layers.active_index = 0
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(70), island_margin=0.006)
    bpy.ops.object.mode_set(mode="OBJECT")
    # Allocate the exposed face one continuous, larger island. Automatic
    # area-based packing undersamples the eyelids/lips even in a 2K atlas.
    uv = obj.data.uv_layers[0]
    source = obj.data.attributes["source_face"]
    front = []
    for polygon in obj.data.polygons:
        if g.components[source.data[polygon.index].value] != "face":
            continue
        if any(obj.data.vertices[v].co.y > 0.03 for v in polygon.vertices):
            continue  # The closure lies inside the coif and retains its UVs.
        for loop in polygon.loop_indices:
            co = obj.data.vertices[obj.data.loops[loop].vertex_index].co
            uv.data[loop].uv = (2 + co.x, co.z)
            front.append(loop)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(rotate=True, margin=0.006)
    bpy.ops.object.mode_set(mode="OBJECT")
    uv = obj.data.uv_layers[0]
    coords = [uv.data[i].uv for i in front]
    bounds = [
        (max(p[a] for p in coords) - min(p[a] for p in coords)) * size for a in [0, 1]
    ]
    assert min(bounds) > 200, "Insufficient face texel allocation"
    return {
        "front_island_bounds_px": bounds,
        "atlas_size_px": size,
        "continuous_front_projection": True,
    }


def bake_image(obj, materials, size, mode):
    image = bpy.data.images.new(
        f"Bake_{mode}", width=size, height=size, alpha=True, float_buffer=True
    )
    image.colorspace_settings.name = "Non-Color"
    for mat, emit, result, target, mask in materials:
        tree = mat.node_tree
        for link in list(emit.inputs["Color"].links):
            tree.links.remove(link)
        if mode == "color":
            tree.links.new(result, emit.inputs["Color"])
        elif mode == "mask":
            tree.links.new(mask, emit.inputs["Color"])
        else:
            ao = node(mat, "ShaderNodeAmbientOcclusion")
            ao.inputs["Distance"].default_value = 0.13
            ao.only_local = True
            ao.samples = 16
            tree.links.new(ao.outputs["Color"], emit.inputs["Color"])
        target.image = image
        tree.nodes.active = target
    print("BAKE_START", mode, flush=True)
    bpy.ops.object.bake(type="EMIT", margin=12, use_clear=True)
    pixels = np.empty(size * size * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    print("BAKE_DONE", mode, flush=True)
    return pixels.reshape((size, size, 4))


def atlas_material(image):
    mat = bpy.data.materials.new("archer_atlas_opaque")
    mat.use_nodes = True
    principled = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    principled.inputs["Roughness"].default_value = 0.74
    principled.inputs["Metallic"].default_value = 0.08
    uv = node(mat, "ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    tex = node(mat, "ShaderNodeTexImage")
    tex.image = image
    tex.extension = "EXTEND"
    mat.node_tree.links.new(uv.outputs[0], tex.inputs["Vector"])
    vertex = node(mat, "ShaderNodeVertexColor")
    vertex.layer_name = "Col"
    multiply = node(mat, "ShaderNodeMixRGB")
    multiply.blend_type = "MULTIPLY"
    multiply.inputs[0].default_value = 1
    mat.node_tree.links.new(tex.outputs["Color"], multiply.inputs[1])
    mat.node_tree.links.new(vertex.outputs["Color"], multiply.inputs[2])
    mat.node_tree.links.new(multiply.outputs[0], principled.inputs["Base Color"])
    # Alpha is a team mask. It must never connect to Principled Alpha.
    return mat


def tint_material(material, team):
    mat = material.copy()
    mat.name = "Review_team_tint"
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    tex = next(n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE")
    blend = node(mat, "ShaderNodeMixRGB")
    blend.blend_type = "MIX"
    blend.inputs[1].default_value = (1, 1, 1, 1)
    blend.inputs[2].default_value = tuple(team) + (1,)
    mat.node_tree.links.new(tex.outputs["Alpha"], blend.inputs[0])
    multiply = node(mat, "ShaderNodeMixRGB")
    multiply.blend_type = "MULTIPLY"
    multiply.inputs[0].default_value = 1
    mat.node_tree.links.new(tex.outputs["Color"], multiply.inputs[1])
    mat.node_tree.links.new(blend.outputs[0], multiply.inputs[2])
    mat.node_tree.links.new(multiply.outputs[0], bsdf.inputs["Base Color"])
    return mat


def camera(elevation, azimuth, target=(0, -0.23, 0.94), scale=2.23):
    scene = bpy.context.scene
    if scene.camera is None:
        data = bpy.data.cameras.new("ReviewCamera")
        scene.camera = bpy.data.objects.new("ReviewCamera", data)
        scene.collection.objects.link(scene.camera)
    cam = scene.camera
    el, az = map(math.radians, (elevation, azimuth))
    target = Vector(target)
    cam.location = target + 6 * Vector(
        (math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))
    )
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = scale
    bpy.context.view_layer.update()
    return cam


def render_setup():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.resolution_percentage = 100
    scene.world = bpy.data.worlds.new("ReviewWorld")
    scene.world.use_nodes = True
    background = next(n for n in scene.world.node_tree.nodes if n.type == "BACKGROUND")
    background.inputs["Color"].default_value = (0.22, 0.25, 0.30, 1)
    background.inputs["Strength"].default_value = 0.35
    scene.view_settings.view_transform = "AgX"
    for name, loc, power, size in [
        ("Key", (-3, -4, 6), 520, 4),
        ("Fill", (3, -1, 3), 180, 4),
        ("Rim", (1, 3, 4), 400, 3),
    ]:
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.shape, data.size = power, "DISK", size
        light = bpy.data.objects.new(name, data)
        light.location = loc
        light.rotation_euler = (
            (Vector((0, 0, 1)) - light.location).to_track_quat("-Z", "Y").to_euler()
        )
        scene.collection.objects.link(light)


def render(name, width=840, height=960):
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.filepath = str(ROOT / name)
    bpy.ops.render.render(write_still=True)
