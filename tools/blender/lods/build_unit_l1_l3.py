"""Append authored infantry or archer L1/L3 to an installed L0/L2 GLB.

blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/build_unit_l1_l3.py -- --kind man_at_arms
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, Source, atlas_projection, fit_visibility, merge_level
from build_knight_l1_l3 import export_level
import infantry_levels
from source_base import resolve_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind", choices=["man_at_arms", "spearman", "archer"], required=True
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    out = (
        args.out or ROOT / f"assets_dev/{args.kind}/lod_l1_l3_v1/{args.kind}.glb"
    ).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    assert bpy.app.background
    bpy.ops.wm.read_factory_settings(use_empty=True)
    manifest = json.loads(
        (
            Path(__file__).resolve().parent / f"source_labels/{args.kind}_l0_l2.json"
        ).read_text()
    )
    input_path = (args.source or ROOT / f"assets/units/{args.kind}.glb").resolve()
    source = Source(
        resolve_source(input_path, out.parent, manifest), out.parent, manifest
    )
    geometry_module = infantry_levels
    if args.kind == "archer":
        import archer_levels

        geometry_module = archer_levels
    categories = geometry_module.categories()
    combined = source.path
    for name, budget in [("L1", (600, 800)), ("L3", (24, 60))]:
        g = geometry_module.geometry(source, args.kind, name)
        obj = g.mesh(source.material)
        obj.name, obj.data.name = name, args.kind + "_" + name
        assert budget[0] <= len(obj.data.polygons) <= budget[1], (
            name,
            len(obj.data.polygons),
        )
        used = set(g.components)
        candidates = atlas_projection(
            obj, g, source, {k: v for k, v in categories.items() if k in used}
        )
        level_path = out.parent / f"{name}_export.glb"
        merged_path = out.parent / f"{name}_combined.glb"
        export_level(obj, level_path)
        merge_level(combined, level_path, merged_path, name)
        fit_visibility(
            obj,
            g,
            source,
            candidates,
            merged_path,
            match_global=args.kind == "archer" and name == "L1",
        )
        export_level(obj, level_path)
        merge_level(combined, level_path, merged_path, name)
        combined = merged_path
        (out.parent / f"{name}_surfaces.json").write_text(
            json.dumps(
                {
                    "vertices": [(x, z, -y) for x, y, z in g.vertices],
                    "faces": g.faces,
                    "components": g.components,
                    "parts": g.parts,
                }
            )
            + "\n"
        )
        print(name, "TRIANGLES", len(obj.data.polygons), flush=True)
    out.write_bytes(combined.read_bytes())
    suffix = {"spearman": ".stab.json", "archer": ".shoot.json"}.get(args.kind)
    if suffix:
        shutil.copyfile(input_path.with_suffix(suffix), out.with_suffix(suffix))
    bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix(".blend")))


if __name__ == "__main__":
    main()
