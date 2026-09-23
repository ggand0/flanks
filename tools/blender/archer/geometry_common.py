"""FLANKS rigid geometry helpers. Original project geometry.

Run only in a fresh background Blender process, never the user's live scene.
Writes exclusively below --out's directory. No external assets or downloads.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import bpy
import bmesh
from mathutils import Vector

PARTS = {
    "body": 0,
    "arm_weapon": 1,
    "leg_l": 2,
    "leg_r": 3,
    "arm_spear": 4,
    "arm_shield": 5,
    "arm_bow": 6,
}
PIVOTS = {
    "leg_l": (0.115, 0, 0.91),
    "leg_r": (-0.115, 0, 0.91),
    "arm_weapon": (-0.205, 0, 1.435),
    "arm_bow": (0.205, 0, 1.435),
}

# Linear RGB plus team blend amount. Opaque even when team amount is zero.
MAIL = (0.075, 0.088, 0.099, 0.0)
STEEL = (0.23, 0.26, 0.285, 0.16)
EDGE = (0.42, 0.445, 0.45, 0.12)
DARK = (0.012, 0.017, 0.019, 0.0)
CLOTH = (0.34, 0.052, 0.032, 1.0)
HEM = (0.27, 0.035, 0.022, 1.0)
IVORY = (0.64, 0.56, 0.38, 0.0)
LEATHER = (0.072, 0.036, 0.019, 0.0)
LEATHER_EDGE = (0.115, 0.067, 0.032, 0.0)
BRASS = (0.37, 0.25, 0.10, 0.0)


class Geometry:
    def __init__(self):
        self.vertices, self.faces, self.colors = [], [], []
        self.parts, self.smooth, self.components = [], [], []
        self.part, self.label = "body", "body"

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

    def tube(self, centers, radii, n, color, smooth=True, mail=False, cap=True):
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
        faces = [tuple(reversed(range(n)))] if cap else []
        for k in range(len(centers) - 1):
            for i in range(n):
                a, b = k * n + i, k * n + (i + 1) % n
                faces.append((a, b, b + n, a + n))
        if cap:
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

    def stud(self, center, r, color, normal=(0, -1, 0)):
        normal = Vector(normal).normalized()
        u = normal.cross(Vector((0, 0, 1))).normalized()
        if u.length < 0.1:
            u = Vector((1, 0, 0))
        v = normal.cross(u)
        center = Vector(center)
        points = [
            center + r * (u * math.cos(i * math.pi / 2) + v * math.sin(i * math.pi / 2))
            for i in range(4)
        ]
        points.append(center + normal * r * 0.45)
        self.add(points, [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)], color)

    def mail_link(self, center, normal, radius=0.009):
        # Fine mail now lives in the texture atlas.
        pass

    def mesh(self, material):
        mesh = bpy.data.meshes.new("archer_L0_geometry")
        mesh.from_pydata(self.vertices, [], self.faces)
        mesh.update()
        obj = bpy.data.objects.new("L0", mesh)
        bpy.context.scene.collection.objects.link(obj)
        mesh.materials.append(material)
        for name in PARTS:
            if name in self.parts:
                group = obj.vertex_groups.new(name=name)
                group.add(
                    [i for i, p in enumerate(self.parts) if p == name], 1, "REPLACE"
                )
        col = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="CORNER")
        mesh.color_attributes.active_color = col
        mesh.attributes.active_color = col
        uv0 = mesh.uv_layers.new(name="UVMap")
        uv1 = mesh.uv_layers.new(name="part")
        for polygon, color, smooth in zip(mesh.polygons, self.colors, self.smooth):
            polygon.use_smooth = smooth
            for index in polygon.loop_indices:
                vertex = mesh.loops[index].vertex_index
                part = self.parts[vertex]
                col.data[index].color = color
                uv0.data[index].uv = (0, 0)
                # glTF exporter transforms v to 1-v. Pre-compensate so the
                # ACTUAL GLB stores the pivot height, not 1 minus its height.
                height = PIVOTS[part][2] if part != "body" else 0
                uv1.data[index].uv = (PARTS[part], 1 - height)
        source_face = mesh.attributes.new(name="source_face", type="INT", domain="FACE")
        for polygon in mesh.polygons:
            source_face.data[polygon.index].value = polygon.index
        mesh.uv_layers.active_index = 0
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        return obj
