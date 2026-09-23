"""Build spearman L2 with the existing articulated spear arm and atlas.

blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/build_spearman_l2.py -- \
  --out assets_dev/spearman/lod_l2_v1/spearman.glb
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, Geometry, Source, WHITE, finish
import soldier


def octahedron(g, center, radii):
    center = Vector(center)
    points = [center + Vector((sign * radii[0], 0, 0)) for sign in [-1, 1]]
    points += [center + Vector((0, sign * radii[1], 0)) for sign in [-1, 1]]
    points += [center + Vector((0, 0, sign * radii[2])) for sign in [-1, 1]]
    faces = [(x, y, z) for x in [0, 1] for y in [2, 3] for z in [4, 5]]
    g.add(points, faces, WHITE, True)


def geometry(source):
    g = Geometry(source)
    soldier.torso(g)
    soldier.head(g)
    soldier.legs(g)
    soldier.arm(g, 1, "arm_shield")
    soldier.shield(g)
    shoulder, elbow, wrist, grip = [
        source.pivots[n] for n in ["arm_spear", "forearm_spear", "hand_spear", "weapon"]
    ]
    g.part, g.label = "body", "socket"
    octahedron(g, shoulder, (0.099, 0.099, 0.099))
    g.part, g.label = "arm_spear", "upper_arm"
    g.tube([shoulder, elbow], [0.072, 0.060], 3, WHITE, True)
    octahedron(g, elbow, (0.078, 0.078, 0.078))
    g.part, g.label = "forearm_spear", "forearm"
    g.tube([elbow, wrist], [0.061, 0.038], 3, WHITE, True)
    g.part, g.label = "hand_spear", "hand"
    octahedron(g, wrist.lerp(grip, 0.33), (0.053, 0.065, 0.052))
    g.part, g.label = "weapon", "shaft"
    x, y, _ = grip
    g.tube([(x, y, 0.02), (x, y, 2.30)], [0.018, 0.014], 4, WHITE, True)
    g.label = "spearhead"
    points = [
        (x - 0.035, y, 2.35),
        (x, y - 0.006, 2.35),
        (x + 0.035, y, 2.35),
        (x, y + 0.006, 2.35),
        (x, y, 2.282),
        (x, y, 2.52),
    ]
    faces = [(i, (i + 1) % 4, tip) for i in range(4) for tip in [4, 5]]
    g.add(points, faces, WHITE)
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=ROOT / "assets_dev/spearman/lod_l2_v1/spearman.glb"
    )
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    assert bpy.app.background
    bpy.ops.wm.read_factory_settings(use_empty=True)
    source_path = ROOT / "assets/units/spearman.glb"
    manifest = json.loads(
        (Path(__file__).resolve().parent / "source_labels/spearman.json").read_text()
    )
    source = Source(source_path, out.parent, manifest)
    categories = soldier.categories()
    categories.update(
        {
            "socket": ["mail_shoulder_socket"],
            "upper_arm": 4,
            "forearm": 9,
            "hand": 10,
            "shaft": ["spear_shaft"],
            "spearhead": ["spear_steel"],
        }
    )
    finish(source, geometry(source), categories, out)
    shutil.copyfile(
        source_path.with_suffix(".stab.json"), out.with_suffix(".stab.json")
    )


if __name__ == "__main__":
    main()
