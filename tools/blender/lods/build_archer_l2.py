"""Build archer L2 with articulated arms, bow limbs, torso and head.

blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/build_archer_l2.py -- \
  --out assets_dev/archer/lod_l2_v1/archer.glb
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import bpy
from mathutils import Vector

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, Geometry, Source, WHITE, finish
from build_spearman_l2 import octahedron
import soldier


def capped_cone(g, rings, n, tip):
    points = [
        (
            cx + rx * math.sin(2 * math.pi * i / n),
            cy - ry * math.cos(2 * math.pi * i / n),
            z,
        )
        for z, cx, cy, rx, ry in rings
        for i in range(n)
    ]
    points.append(tip)
    faces = [tuple(reversed(range(n)))]
    faces += [
        (r * n + i, r * n + (i + 1) % n, (r + 1) * n + (i + 1) % n, (r + 1) * n + i)
        for r in range(len(rings) - 1)
        for i in range(n)
    ]
    start = (len(rings) - 1) * n
    faces += [(start + i, start + (i + 1) % n, len(points) - 1) for i in range(n)]
    g.add(points, faces, WHITE, True)


def geometry(source):
    g = Geometry(source)
    g.part, g.label = "body", "lower_tunic"
    g.loft([(0.505, 0, 0, 0.224, 0.132), (1.13, 0, 0, 0.170, 0.127)], 6, WHITE, True)
    g.part, g.label = "torso", "upper_tunic"
    capped_cone(
        g,
        [(1.00, 0, 0, 0.171, 0.115), (1.33, 0, 0.005, 0.258, 0.163)],
        6,
        (0, 0.018, 1.553),
    )
    g.part, g.label = "head", "head"
    capped_cone(
        g,
        [(1.48, 0, 0.006, 0.074, 0.083), (1.69, 0, 0.005, 0.108, 0.114)],
        6,
        (0, 0.008, 1.8),
    )
    soldier.legs(g)
    for upper, forearm, hand, sign in [
        ("arm_weapon", "forearm_draw", "hand_draw", -1),
        ("arm_bow", "forearm_bow", "hand_bow", 1),
    ]:
        s, e, w = [source.pivots[name] for name in [upper, forearm, hand]]
        grip = Vector((sign * 0.365, -0.190, 0.900))
        g.part, g.label = "torso", "upper_tunic"
        octahedron(g, s, (0.097, 0.097, 0.097))
        g.part, g.label = upper, upper
        g.tube([s, e], [0.083, 0.063], 3, WHITE, True)
        octahedron(g, e, (0.077, 0.077, 0.077))
        g.part, g.label = forearm, forearm
        g.tube([e, w], [0.068, 0.040], 3, WHITE, True)
        g.part, g.label = hand, hand
        octahedron(g, w.lerp(grip, 0.35), (0.047, 0.061, 0.048))
    grip = source.pivots["weapon"]
    g.part, g.label = "weapon", "grip"
    g.tube(
        [grip + Vector((0, 0, -0.046)), grip + Vector((0, 0, 0.046))],
        [0.018, 0.018],
        3,
        WHITE,
        True,
    )
    for sign, part in [(1, "bow_upper"), (-1, "bow_lower")]:
        g.part, g.label = part, part
        centers = [
            grip + Vector((0, front, sign * height))
            for height, front in [(0.04, 0), (0.52, 0.085), (0.87, 0.190)]
        ]
        g.tube(centers, [0.018, 0.0105, 0.0036], 3, WHITE, True)
    g.part, g.label = "body", "quiver"
    g.tube(
        [(-0.265, 0.134, 0.49), (-0.245, 0.155, 1.035)], [0.040, 0.058], 3, WHITE, True
    )
    g.label = "quiver_arrows"
    points = [
        (-0.279, 0.13, 1.205),
        (-0.211, 0.13, 1.218),
        (-0.211, 0.18, 1.231),
        (-0.279, 0.18, 1.218),
        (-0.245, 0.155, 1.035),
    ]
    g.add(points, [(0, 1, 2, 3)] + [(i, (i + 1) % 4, 4) for i in range(4)], WHITE)
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=ROOT / "assets_dev/archer/lod_l2_v1/archer.glb"
    )
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    assert bpy.app.background
    bpy.ops.wm.read_factory_settings(use_empty=True)
    source_path = ROOT / "assets/units/archer.glb"
    manifest = json.loads(
        (Path(__file__).resolve().parent / "source_labels/archer.json").read_text()
    )
    source = Source(source_path, out.parent, manifest)
    categories = {
        "lower_tunic": ["tunic", "tunic_seam", "belt", "patch"],
        "upper_tunic": ["tunic", "tunic_sleeve", "hood_cape", "hood_fold", "neck"],
        "head": 18,
        "leg_l": 2,
        "leg_r": 3,
        "arm_weapon": 1,
        "forearm_draw": 9,
        "hand_draw": 10,
        "arm_bow": 6,
        "forearm_bow": 11,
        "hand_bow": 12,
        "grip": 8,
        "bow_upper": 14,
        "bow_lower": 15,
        "quiver": ["quiver", "quiver_rim", "quiver_hanger"],
        "quiver_arrows": ["arrow_shaft", "fletching"],
    }
    finish(source, geometry(source), categories, out)
    shutil.copyfile(
        source_path.with_suffix(".shoot.json"), out.with_suffix(".shoot.json")
    )


if __name__ == "__main__":
    main()
