"""Build and bake the articulated archer with a fresh background Blender.

blender -b --factory-startup --python-exit-code 1 --python build_archer.py
"""
import argparse
import json
from pathlib import Path
import sys
import bpy
import numpy as np

SOURCE = Path(__file__).resolve().parent
DEFAULT_OUT = SOURCE.parents[2] / "assets_dev/archer/rebuild_raise_v4"
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "archer.glb")
parser.add_argument("--reuse-bake", action="store_true")
args = parser.parse_args(
    sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
)
ROOT = args.out.resolve().parent
ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SOURCE))
import geometry_archer as geo
import build_support as support

support.ROOT = ROOT
import motion


def main():
    assert bpy.app.background
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 8
    g = geo.build()
    obj = g.mesh(bpy.data.materials.new("temporary"))
    print("TRIANGLES", len(obj.data.polygons), flush=True)
    assert len(obj.data.polygons) <= 3000
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    keys = sorted(
        {support.category(c, rgba) for c, rgba in zip(g.components, g.colors)}
    )
    materials = [support.source_material(*key) for key in keys]
    obj.data.materials.clear()
    for mat, *rest in materials:
        obj.data.materials.append(mat)
    source = obj.data.attributes["source_face"]
    for p in obj.data.polygons:
        i = source.data[p.index].value
        p.material_index = keys.index(support.category(g.components[i], g.colors[i]))
    support.unwrap(obj, g, 2048)
    if not args.reuse_bake:
        base = support.bake_image(obj, materials, 2048, "color")
        mask = support.bake_image(obj, materials, 2048, "mask")
        ao = support.bake_image(obj, materials, 2048, "ao")
        rgba = np.ones_like(base)
        linear = np.clip(base[:, :, :3] * (0.48 + 0.52 * ao[:, :, :1]), 0, 1)
        rgba[:, :, :3] = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * np.power(linear, 1 / 2.4) - 0.055,
        )
        rgba[:, :, 3] = np.clip(mask[:, :, 0], 0, 1)
        atlas = bpy.data.images.new("ArcherAtlas", width=2048, height=2048, alpha=True)
        atlas.colorspace_settings.name = "sRGB"
        atlas.alpha_mode = "CHANNEL_PACKED"
        atlas.pixels.foreach_set(rgba.reshape(-1))
        atlas.filepath_raw = str(ROOT / "archer_atlas.png")
        atlas.file_format = "PNG"
        atlas.save()
    atlas = bpy.data.images.load(str(ROOT / "archer_atlas.png"), check_existing=False)
    atlas.alpha_mode = "CHANNEL_PACKED"
    mat = support.atlas_material(atlas)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for p in obj.data.polygons:
        p.material_index = 0
    color = obj.data.color_attributes.active_color
    for c in color.data:
        c.color = (1, 1, 1, c.color[3])
    obj.data.color_attributes.render_color_index = obj.data.color_attributes.find(
        color.name
    )
    for name, loc in geo.PIVOTS.items():
        empty = bpy.data.objects.new("pivot_" + name, None)
        empty.location = loc
        scene.collection.objects.link(empty)
        empty.select_set(True)
    path = str(args.out.resolve())
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_normals=True,
        export_texcoords=True,
        export_vertex_color="ACTIVE",
    )
    assert all(
        len(v.groups) == 1 and v.groups[0].weight == 1 for v in obj.data.vertices
    )
    assert all(p.area > 1e-12 for p in obj.data.polygons)
    (ROOT / "joint_attachment_probes.json").write_text(
        json.dumps(geo.PROBES, indent=2) + "\n"
    )
    (ROOT / "archer.raise.json").write_text(
        json.dumps(motion.export(), indent=2) + "\n"
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / "archer.blend"))
    print("ARCHER_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
