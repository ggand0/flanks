"""Construct closed infantry L1 and L3 geometry for the unit LOD builder.

Imported by build_unit_l1_l3.py; dimensions are Blender metres, Z up.
"""

import math
from mathutils import Vector

from common import Geometry, WHITE
from build_knight_l1_l3 import OCTAGON, outlined, shield as knight_shield
from build_archer_l2 import capped_cone
from build_spearman_l2 import octahedron
import build_man_at_arms_l2
import soldier


def copy_parts(g, other, names):
    selected = [
        i for i, face in enumerate(other.faces) if other.parts[face[0]] in names
    ]
    used = sorted({v for i in selected for v in other.faces[i]})
    offset = len(g.vertices)
    mapping = {old: offset + i for i, old in enumerate(used)}
    g.vertices.extend(other.vertices[i] for i in used)
    g.parts.extend(other.parts[i] for i in used)
    for i in selected:
        g.faces.append(tuple(mapping[v] for v in other.faces[i]))
        g.colors.append(other.colors[i])
        g.smooth.append(other.smooth[i])
        g.components.append(other.components[i])


def shield(g, detailed):
    start = len(g.vertices)
    knight_shield(g, detailed)
    tangent, normal = Vector((0.9063, 0.4226, 0)), Vector((0.4226, -0.9063, 0))
    for i in range(start, len(g.vertices)):
        point = Vector(g.vertices[i])
        local = point - Vector((0.412, -0.207, 0))
        u = local.dot(tangent)
        point += Vector((-0.018, 0.001, 0)) + tangent * (u * (0.205 / 0.236 - 1))
        point -= normal * (0.009 * (1 - (u / 0.236) ** 2))
        point.z = 0.744 + (point.z - 0.749) * (0.660 / 0.666)
        g.vertices[i] = tuple(point)


def legs(g):
    for sign, part in [(1, "leg_l"), (-1, "leg_r")]:
        g.part, g.label = part, part
        outlined(
            g,
            [
                (0, sign * 0.112, -0.0675, 0.055, 0.1475),
                (0.12, sign * 0.112, -0.017, 0.061, 0.090),
                (0.25, sign * 0.112, 0, 0.067, 0.074),
                (0.50, sign * 0.108, 0, 0.070, 0.081),
                (0.95, sign * 0.075, 0, 0.083, 0.088),
            ],
            OCTAGON,
            True,
        )


def head(g, detailed):
    g.part, g.label = "body", "head"
    if detailed:
        g.loft(
            [
                (1.405, 0, 0, 0.142, 0.110),
                (1.49, 0, 0.012, 0.117, 0.105),
                (1.68, 0, 0.005, 0.102, 0.110),
            ],
            6,
            WHITE,
            True,
        )
        g.label = "face"
        outlined(
            g,
            [(1.49, 0, -0.087, 0.056, 0.040), (1.66, 0, -0.088, 0.068, 0.042)],
            [(-1, -0.55), (0, -1), (1, -0.55), (1, 0.5), (-1, 0.5)],
            True,
        )
        g.label = "hat"
        g.loft(
            [
                (1.653, 0, 0, 0.192, 0.188),
                (1.663, 0, 0, 0.192, 0.188),
                (1.672, 0, 0.003, 0.115, 0.111),
                (1.754, 0, 0.004, 0.092, 0.088),
                (1.8, 0, 0.004, 0.035, 0.033),
            ],
            10,
            WHITE,
            True,
        )
    else:
        capped_cone(g, [(1.435, 0, 0, 0.113, 0.108)], 4, (0, 0.005, 1.715))
        g.label = "hat"
        capped_cone(g, [(1.655, 0, 0, 0.192, 0.188)], 4, (0, 0.004, 1.8))


def torso(g, detailed):
    g.part, g.label = "body", "torso"
    outline = OCTAGON if detailed else [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    rings = [
        (0.535, 0, 0, 0.209, 0.143),
        (0.635, 0, 0, 0.208, 0.146),
        (1.045, 0, 0, 0.168, 0.136),
        (1.47, 0, 0, 0.184, 0.112),
    ]
    if not detailed:
        rings = [
            (0.505, 0, 0, 0.200, 0.122),
            (1.045, 0, 0, 0.165, 0.117),
            (1.49, 0, 0, 0.280, 0.102),
        ]
    start = len(g.faces)
    outlined(g, rings, outline, True)
    if detailed:
        g.components[start + 1] = "head"
        g.label = "belt"
        outlined(g, [(0.988, 0, 0, 0.174, 0.143), (1.044, 0, 0, 0.173, 0.142)], OCTAGON)


def arm(g, sign, part):
    g.part, g.label = part, part
    g.tube(
        [
            (sign * 0.208, 0, 1.423),
            (sign * 0.288, -0.005, 1.30),
            (sign * 0.333, -0.016, 1.228),
            (sign * 0.39, -0.205, 1.085),
        ],
        [0.091, 0.083, 0.075, 0.053],
        8,
        WHITE,
        True,
    )
    g.label = "hand"
    g.tube(
        [(sign * 0.383, -0.166, 1.107), (sign * 0.39, -0.237, 1.084)],
        [0.052, 0.047],
        6,
        WHITE,
        True,
    )


def far_legs(g):
    g.part, g.label = "body", "legs"
    for sign in [-1, 1]:
        x = sign * 0.112
        g.add(
            [(x - 0.06, -0.215, 0), (x + 0.06, -0.215, 0), (x, 0.080, 0), (x, 0, 0.75)],
            [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
            WHITE,
            True,
        )


def spear(g, source, detailed):
    grip = source.pivots["weapon"]
    x, y, _ = grip
    g.part, g.label = ("weapon" if detailed else "body"), "shaft"
    g.tube(
        [(x, y, 0.02), (x, y, 2.30 if detailed else 2.52)],
        [0.018, 0.014 if detailed else 0.005],
        8 if detailed else 3,
        WHITE,
        True,
    )
    if detailed:
        g.label = "spearhead"
        points = [
            (x - 0.035, y, 2.35),
            (x, y - 0.006, 2.35),
            (x + 0.035, y, 2.35),
            (x, y + 0.006, 2.35),
            (x, y, 2.282),
            (x, y, 2.52),
        ]
        g.add(
            points, [(i, (i + 1) % 4, tip) for i in range(4) for tip in [4, 5]], WHITE
        )


def geometry(source, kind, level):
    detailed = level == "L1"
    g = Geometry(source)
    torso(g, detailed)
    head(g, detailed)
    if detailed:
        legs(g)
        arm(g, 1, "arm_shield")
        if kind == "man_at_arms":
            arm(g, -1, "arm_weapon")
            copy_parts(g, build_man_at_arms_l2.geometry(source), {"weapon"})
        else:
            s, e, w, grip = [
                source.pivots[n]
                for n in ["arm_spear", "forearm_spear", "hand_spear", "weapon"]
            ]
            g.part, g.label = "body", "socket"
            octahedron(g, s, (0.099, 0.099, 0.099))
            g.part, g.label = "arm_spear", "upper_arm"
            g.tube([s, e], [0.072, 0.060], 6, WHITE, True)
            octahedron(g, e, (0.078, 0.078, 0.078))
            g.part, g.label = "forearm_spear", "forearm"
            g.tube([e, w], [0.061, 0.038], 6, WHITE, True)
            g.part, g.label = "hand_spear", "hand"
            octahedron(g, w.lerp(grip, 0.33), (0.053, 0.065, 0.052))
            spear(g, source, True)
        g.part, g.label = "body", "scabbard"
        g.tube(
            [(0.189, 0.022, 1.008), (0.245, 0.152, 0.209)],
            [0.027, 0.012],
            4,
            WHITE,
            False,
        )
    else:
        far_legs(g)
        if kind == "spearman":
            spear(g, source, False)
    shield(g, detailed)
    return g


def categories():
    result = soldier.categories()
    result.update(
        {
            "arm_weapon": result["arm_shield"],
            "face": ["face"],
            "hand": ["hand"],
            "belt": ["belt"],
            "scabbard": ["scabbard"],
            "blade": ["sword_blade"],
            "grip": ["sword_grip", "sword_hilt"],
            "guard": ["sword_hilt"],
            "socket": ["mail_shoulder_socket"],
            "upper_arm": 4,
            "forearm": 9,
            "shaft": ["spear_shaft"],
            "spearhead": ["spear_steel"],
            "legs": ["hose", "shoe", "shoe_edge"],
        }
    )
    return result
