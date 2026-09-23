"""Append closed knight L1 and L3 meshes to the installed L0/L2 GLB.

blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/lods/build_knight_l1_l3.py
"""

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    ROOT,
    Geometry,
    Source,
    WHITE,
    atlas_projection,
    fit_visibility,
    merge_level,
)


def outlined(g, rings, outline, smooth=False):
    n = len(outline)
    points = [
        (x * rx + cx, y * ry + cy, z) for z, cx, cy, rx, ry in rings for x, y in outline
    ]
    faces = [
        tuple(reversed(range(n))),
        tuple((len(rings) - 1) * n + i for i in range(n)),
    ]
    faces += [
        (r * n + i, r * n + (i + 1) % n, (r + 1) * n + (i + 1) % n, (r + 1) * n + i)
        for r in range(len(rings) - 1)
        for i in range(n)
    ]
    g.add(points, faces, WHITE, smooth)


OCTAGON = [
    (-1, -0.5),
    (-0.65, -1),
    (0.65, -1),
    (1, -0.5),
    (1, 0.5),
    (0.65, 1),
    (-0.65, 1),
    (-1, 0.5),
]


def shield(g, detailed):
    g.part = "arm_shield" if detailed else "body"
    tangent, normal = Vector((0.9063, 0.4226, 0)), Vector((0.4226, -0.9063, 0))

    def at(u, z, depth=0):
        return (
            Vector((0.412, -0.207, z))
            + tangent * u
            + normal * (0.057 * (1 - (u / 0.236) ** 2) + depth)
        )

    border = [
        (-0.219, 1.415),
        (0.219, 1.415),
        (0.236, 1.353),
        (0.213, 1.16),
        (0.150, 0.971),
        (0, 0.749),
        (-0.150, 0.971),
        (-0.213, 1.16),
        (-0.236, 1.353),
    ]
    if detailed:
        n = len(border)
        inner = [(u * 0.975, 1.17 + (z - 1.17) * 0.985) for u, z in border]
        front = [at(u, z) for u, z in border + inner] + [at(0, 1.17)]
        face = [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)]
        face += [(i + n, (i + 1) % n + n, 2 * n) for i in range(n)]
    else:
        border = [(-0.236, 1.415), (0.236, 1.415), (0, 0.749)]
        front = [at(u, z) for u, z in border]
        face = [(0, 1, 2)]
        n = len(border)
    m = len(front)
    points = front + [p - normal * 0.023 for p in front]
    faces = face + [tuple(v + m for v in reversed(f)) for f in face]
    faces += [(i, i + m, (i + 1) % n + m, (i + 1) % n) for i in range(n)]
    start = len(g.faces)
    g.label = "heater_face"
    g.add(points, faces, WHITE, detailed)
    for i in range(len(face), len(faces)):
        g.components[start + i] = "heater_wood" if i < 2 * len(face) else "heater_rim"
    if detailed:
        # The outer band and the rim read as leather, including from behind.
        for i in range(n):
            g.components[start + i] = "heater_rim"
        g.label = "heater_straps"
        for z in [1.22, 1.06]:
            g.tube(
                [at(-0.075, z, -0.03), at(0.075, z, -0.065)],
                [0.015, 0.015],
                3,
                WHITE,
                False,
            )


def sword(g, detailed):
    g.part = "weapon" if detailed else "body"
    g.label = "arming_sword"
    x, z = -0.4, 1.084
    if not detailed:
        g.add(
            [
                (x - 0.027, -0.24, z),
                (x + 0.027, -0.24, z),
                (x, -0.24, z + 0.016),
                (x, -1.075, z),
            ],
            [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
            WHITE,
        )
        return
    points = [
        (x - 0.026, -0.32, z),
        (x, -0.32, z + 0.006),
        (x + 0.026, -0.32, z),
        (x, -0.32, z - 0.006),
        (x - 0.019, -0.85, z),
        (x, -0.85, z + 0.004),
        (x + 0.019, -0.85, z),
        (x, -0.85, z - 0.004),
        (x, -1.075, z),
    ]
    g.add(
        points,
        [(3, 2, 1, 0)]
        + [(i, (i + 1) % 4, (i + 1) % 4 + 4, i + 4) for i in range(4)]
        + [(i + 4, (i + 1) % 4 + 4, 8) for i in range(4)],
        WHITE,
    )
    g.tube([(x, -0.15, z), (x, -0.32, z)], [0.022, 0.018], 4, WHITE)
    g.tube(
        [(x - 0.119, -0.295, z), (x + 0.119, -0.295, z)],
        [0.015, 0.015],
        3,
        WHITE,
        False,
    )
    g.tube([(x, -0.148, z), (x, -0.168, z)], [0.030, 0.026], 3, WHITE, False)


def l1_geometry(source):
    g = Geometry(source)
    g.label = "mail_hauberk"
    g.loft(
        [
            (0.425, 0, 0, 0.224, 0.150),
            (0.66, 0, 0, 0.217, 0.145),
            (1.08, 0, 0, 0.176, 0.13),
            (1.465, 0, 0, 0.182, 0.108),
        ],
        8,
        WHITE,
        True,
    )
    g.label = "surcoat"
    start = len(g.faces)
    outlined(
        g,
        [
            (0.535, 0, 0, 0.214, 0.155),
            (0.66, 0, 0, 0.21, 0.151),
            (1.055, 0, 0, 0.175, 0.145),
            (1.478, 0, 0, 0.178, 0.111),
        ],
        OCTAGON,
        True,
    )
    g.components[start + 1] = "mail_coif"
    for ring in range(3):
        for side in [3, 7]:
            g.components[start + 2 + ring * 8 + side] = "mail_hauberk"
    g.label = "belt"
    outlined(g, [(0.984, 0, 0, 0.179, 0.15), (1.036, 0, 0, 0.178, 0.149)], OCTAGON)
    g.label = "mail_coif"
    g.loft(
        [
            (1.405, 0, 0, 0.157, 0.114),
            (1.49, 0, 0, 0.135, 0.112),
            (1.59, 0, 0, 0.095, 0.095),
        ],
        6,
        WHITE,
        True,
    )
    g.label = "helm_shell"
    outlined(
        g,
        [
            (1.51, 0, 0, 0.108, 0.117),
            (1.68, 0, 0, 0.111, 0.120),
            (1.8, 0, 0, 0.103, 0.108),
        ],
        OCTAGON,
    )
    g.label = "visor"
    g.box((0, -0.120, 1.686), (0.15, 0.008, 0.014), WHITE)
    for sign, part in [(1, "leg_l"), (-1, "leg_r")]:
        g.part, g.label = part, "boot"
        start = len(g.faces)
        outlined(
            g,
            [
                (0, sign * 0.112, -0.071, 0.063, 0.186),
                (0.12, sign * 0.112, -0.023, 0.067, 0.112),
                (0.40, sign * 0.112, 0, 0.074, 0.080),
                (0.55, sign * 0.112, 0, 0.077, 0.085),
                (0.95, sign * 0.112, 0, 0.108, 0.112),
            ],
            OCTAGON,
            True,
        )
        # Rings above the boot sample mail, including the hip overlap cap.
        g.components[start + 1] = "mail_chausses"
        for i in range(start + 2 + 2 * 8, len(g.faces)):
            g.components[i] = "mail_chausses"
    for sign, part in [(-1, "arm_weapon"), (1, "arm_shield")]:
        g.part, g.label = part, "mail_sleeve"
        start = len(g.faces)
        g.tube(
            [
                (sign * 0.208, 0, 1.423),
                (sign * 0.292, -0.004, 1.285),
                (sign * 0.337, -0.012, 1.215),
                (sign * 0.4, -0.217, 1.084),
            ],
            [0.092, 0.083, 0.082, 0.06],
            8,
            WHITE,
            True,
        )
        for i in range(start + 1 + 2 * 8, len(g.faces)):
            g.components[i] = "leather_cuff"
        g.label = "glove"
        g.tube(
            [(sign * 0.392, -0.165, 1.11), (sign * 0.4, -0.237, 1.084)],
            [0.059, 0.054],
            6,
            WHITE,
            True,
        )
    shield(g, True)
    sword(g, True)
    return g


def l3_geometry(source):
    g = Geometry(source)
    g.label = "mass"
    outline = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    outlined(
        g,
        [
            (0.425, 0, 0, 0.213, 0.132),
            (1.06, 0, 0, 0.182, 0.125),
            (1.493, 0, 0, 0.285, 0.104),
        ],
        outline,
        True,
    )
    g.label = "helm_shell"
    outlined(
        g,
        [(1.48, 0, 0, 0.111, 0.116), (1.8, 0, 0, 0.104, 0.108)],
        [(-0.7, -1), (0.7, -1), (1, 0.4), (0, 1), (-1, 0.4)],
        False,
    )
    g.label = "legs"
    for sign in [-1, 1]:
        x = sign * 0.112
        g.add(
            [
                (x - 0.065, -0.230, 0),
                (x + 0.065, -0.230, 0),
                (x, 0.095, 0),
                (x, 0, 0.71),
            ],
            [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
            WHITE,
            True,
        )
    shield(g, False)
    return g


def export_level(obj, path):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_normals=True,
        export_texcoords=True,
        export_vertex_color="ACTIVE",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=ROOT / "assets_dev/knight/lod_l1_l3_v1/knight.glb"
    )
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    assert bpy.app.background
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    manifest = json.loads(
        (Path(__file__).resolve().parent / "source_labels/knight.json").read_text()
    )
    source = Source(ROOT / "assets/units/knight.glb", out.parent, manifest)
    categories = {label: [label] for label in set(source.labels)}
    categories.update(
        {
            "helm_shell": ["helm_shell", "great_helm"],
            "visor": ["helm_interior"],
            "boot": ["boot", "boot_trim"],
            "mass": ["surcoat", "belt", "mail_hauberk", "mail_sleeve"],
            "legs": ["boot", "boot_trim", "mail_chausses"],
        }
    )
    combined = source.path
    for name, builder, budget in [
        ("L1", l1_geometry, (600, 800)),
        ("L3", l3_geometry, (24, 60)),
    ]:
        g = builder(source)
        obj = g.mesh(source.material)
        obj.name, obj.data.name = name, "knight_" + name
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
        fit_visibility(obj, g, source, candidates, merged_path)
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
    bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix(".blend")))


if __name__ == "__main__":
    main()
