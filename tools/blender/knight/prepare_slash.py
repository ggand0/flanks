"""Add a closed wrist overlap to the knight for three-dimensional sword motion.

Run in background Blender. Keeps the existing atlas and both LOD meshes.
"""
import argparse
import sys
from pathlib import Path
import math
import bpy
import numpy as np

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[2]
ROOT = REPO / "assets_dev/knight/diagonal_slash_v4"
parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=Path, default=ROOT)
args = parser.parse_args(
    sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
)
ROOT = args.out_dir.resolve()
ROOT.mkdir(parents=True, exist_ok=True)

assert bpy.app.background
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(REPO / "assets/units/knight.glb"))
obj = bpy.data.objects["L0"]
mesh = obj.data
probe = np.array([-0.427, -0.110, 1.140])
faces = [
    p for p in mesh.polygons if round(mesh.uv_layers[1].data[p.loop_start].uv.x) == 1
]
face = min(faces, key=lambda p: np.linalg.norm(np.array(p.center) - probe))
uv = np.mean([mesh.uv_layers[0].data[i].uv[:] for i in face.loop_indices], axis=0)
color = np.mean(
    [mesh.color_attributes.active_color.data[i].color[:] for i in face.loop_indices],
    axis=0,
)
start = np.array([-0.349, -0.045, 1.190])
end = np.array([-0.396, -0.180, 1.100])
axis = (end - start) / np.linalg.norm(end - start)
side = np.cross(axis, [0, 0, 1])
side /= np.linalg.norm(side)
other = np.cross(axis, side)
vertices = []
for t, radius in [(0, 0.081), (0.5, 0.075), (1, 0.063)]:
    center = start * (1 - t) + end * t
    for i in range(12):
        angle = i * 2 * math.pi / 12
        vertices.append(
            center + radius * (math.cos(angle) * side + math.sin(angle) * other)
        )
polygons = []
for ring in range(2):
    for i in range(12):
        a, b = ring * 12 + i, ring * 12 + (i + 1) % 12
        polygons.append((a, b, b + 12, a + 12))
polygons.extend([tuple(reversed(range(12))), tuple(range(24, 36))])
cap_mesh = bpy.data.meshes.new("WristOverlap")
cap_mesh.from_pydata(vertices, [], polygons)
cap_mesh.update()
cap = bpy.data.objects.new("WristOverlap", cap_mesh)
bpy.context.scene.collection.objects.link(cap)
atlas = cap.data.uv_layers.new(name=mesh.uv_layers[0].name)
part = cap.data.uv_layers.new(name=mesh.uv_layers[1].name)
col = cap.data.color_attributes.new(
    name=mesh.color_attributes.active_color.name, type="FLOAT_COLOR", domain="CORNER"
)
cap.data.color_attributes.active_color = col
for loop in cap.data.loops:
    atlas.data[loop.index].uv = uv
    part.data[loop.index].uv = (1, 1 - 1.435)
    col.data[loop.index].color = color
for p in cap.data.polygons:
    p.use_smooth = True
cap.data.materials.append(mesh.materials[0])
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
cap.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.join()
for o in bpy.context.scene.objects:
    o.select_set(True)
bpy.ops.export_scene.gltf(
    filepath=str(ROOT / "knight.glb"),
    export_format="GLB",
    use_selection=True,
    export_apply=True,
    export_normals=True,
    export_texcoords=True,
    export_vertex_color="ACTIVE",
)
print("KNIGHT_WRIST_OVERLAP_DONE", flush=True)
