"""Build man-at-arms L2 alongside the original L0 and atlas.

blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/build_man_at_arms_l2.py -- \
  --out assets_dev/man_at_arms/lod_l2_v1/man_at_arms.glb
"""
import argparse
import json
from pathlib import Path
import sys
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, Geometry, Source, WHITE, finish
import soldier


def geometry(source):
    g = Geometry(source)
    soldier.torso(g)
    soldier.head(g)
    soldier.legs(g)
    soldier.arm(g, -1, "arm_weapon")
    soldier.arm(g, 1, "arm_shield")
    soldier.shield(g)
    g.part, g.label = "body", "scabbard"
    g.tube(
        [(0.189, 0.022, 1.008), (0.245, 0.152, 0.209)], [0.027, 0.012], 3, WHITE, False
    )
    g.part, g.label = "weapon", "blade"
    x, z = -0.390, 1.084
    points = [
        (x - 0.024, -0.308, z),
        (x, -0.308, z + 0.0045),
        (x + 0.024, -0.308, z),
        (x, -0.308, z - 0.0045),
        (x - 0.0205, -0.825, z),
        (x, -0.825, z + 0.0035),
        (x + 0.0205, -0.825, z),
        (x, -0.825, z - 0.0035),
        (x, -1.1, z),
    ]
    g.add(
        points,
        [(3, 2, 1, 0)]
        + [(i, (i + 1) % 4, (i + 1) % 4 + 4, i + 4) for i in range(4)]
        + [(i + 4, (i + 1) % 4 + 4, 8) for i in range(4)],
        WHITE,
    )
    g.label = "grip"
    g.tube([(x, -0.145, z), (x, -0.308, z)], [0.024, 0.016], 3, WHITE)
    g.label = "guard"
    g.tube(
        [(x - 0.102, -0.300, z), (x + 0.102, -0.300, z)],
        [0.012, 0.012],
        3,
        WHITE,
        False,
    )
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "assets_dev/man_at_arms/lod_l2_v1/man_at_arms.glb",
    )
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    assert bpy.app.background
    bpy.ops.wm.read_factory_settings(use_empty=True)
    source_path = ROOT / "assets/units/man_at_arms.glb"
    manifest = json.loads(
        (Path(__file__).resolve().parent / "source_labels/man_at_arms.json").read_text()
    )
    source = Source(source_path, out.parent, manifest)
    categories = soldier.categories()
    categories.update(
        {
            "arm_weapon": categories["arm_shield"],
            "scabbard": ["scabbard"],
            "blade": ["sword_blade"],
            "grip": ["sword_grip", "sword_hilt"],
            "guard": ["sword_hilt"],
        }
    )
    finish(source, geometry(source), categories, out)


if __name__ == "__main__":
    main()
