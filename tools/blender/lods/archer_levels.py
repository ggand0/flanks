"""Construct closed archer L1/L3 with articulated arms and bow limbs.

Imported by build_unit_l1_l3.py; dimensions are Blender metres, Z up.
"""

from mathutils import Vector

from common import Geometry, WHITE
from build_knight_l1_l3 import OCTAGON, outlined
from build_archer_l2 import capped_cone
from build_spearman_l2 import octahedron
from infantry_levels import legs, far_legs


def geometry(source, kind, level):
    g = Geometry(source)
    detailed = level == "L1"
    g.part, g.label = "body", "lower_tunic"
    if detailed:
        outlined(
            g,
            [
                (0.505, 0, 0, 0.224, 0.132),
                (0.66, 0, 0, 0.211, 0.137),
                (0.98, 0, 0, 0.171, 0.128),
                (1.13, 0, 0, 0.170, 0.127),
            ],
            OCTAGON,
            True,
        )
        g.label = "belt"
        outlined(g, [(1.016, 0, 0, 0.174, 0.131), (1.069, 0, 0, 0.173, 0.131)], OCTAGON)
        g.part, g.label = "torso", "upper_tunic"
        start = len(g.faces)
        capped_cone(
            g,
            [
                (1.00, 0, 0, 0.171, 0.115),
                (1.29, 0, 0.003, 0.213, 0.143),
                (1.40, 0, 0.01, 0.240, 0.148),
            ],
            8,
            (0, 0.018, 1.553),
        )
        for index in range(start, len(g.faces)):
            if min(g.vertices[v][2] for v in g.faces[index]) >= 1.29:
                g.components[index] = "cape"
        g.part, g.label = "head", "head"
        capped_cone(
            g,
            [
                (1.48, 0, 0.006, 0.074, 0.083),
                (1.60, 0, 0.005, 0.105, 0.108),
                (1.715, 0, 0.005, 0.108, 0.111),
            ],
            8,
            (0, 0.008, 1.8),
        )
        g.label = "face"
        outlined(
            g,
            [(1.515, 0, -0.107, 0.055, 0.034), (1.688, 0, -0.110, 0.070, 0.034)],
            [(-1, -0.5), (0, -1), (1, -0.5), (1, 0.5), (0, 1), (-1, 0.5)],
            True,
        )
        legs(g)
        for upper, forearm, hand, sign in [
            ("arm_weapon", "forearm_draw", "hand_draw", -1),
            ("arm_bow", "forearm_bow", "hand_bow", 1),
        ]:
            s, e, w = [source.pivots[name] for name in [upper, forearm, hand]]
            grip = Vector((sign * 0.365, -0.190, 0.900))
            g.part, g.label = "torso", "shoulder"
            octahedron(g, s, (0.097, 0.097, 0.097))
            g.part, g.label = upper, upper
            g.tube([s, s.lerp(e, 0.50), e], [0.083, 0.077, 0.063], 6, WHITE, True)
            octahedron(g, e, (0.077, 0.077, 0.077))
            g.part, g.label = forearm, forearm
            g.tube([e, e.lerp(w, 0.55), w], [0.068, 0.055, 0.040], 6, WHITE, True)
            g.part, g.label = hand, hand
            octahedron(g, w.lerp(grip, 0.35), (0.047, 0.061, 0.048))
    else:
        outlined(
            g,
            [
                (0.505, 0, 0, 0.213, 0.124),
                (1.045, 0, 0, 0.170, 0.12),
                (1.515, 0, 0.007, 0.278, 0.114),
            ],
            [(-1, -1), (1, -1), (1, 1), (-1, 1)],
            True,
        )
        g.label = "head"
        capped_cone(g, [(1.48, 0, 0.006, 0.108, 0.113)], 4, (0, 0.008, 1.8))
        far_legs(g)

    grip = source.pivots["weapon"]
    if detailed:
        g.part, g.label = "weapon", "grip"
        g.tube(
            [grip + Vector((0, 0, -0.046)), grip + Vector((0, 0, 0.046))],
            [0.018, 0.018],
            6,
            WHITE,
            True,
        )
        for sign, part in [(1, "bow_upper"), (-1, "bow_lower")]:
            g.part, g.label = part, part
            centers = [
                grip + Vector((0, front, sign * height))
                for height, front in [
                    (0.04, 0),
                    (0.33, 0.040),
                    (0.62, 0.124),
                    (0.87, 0.190),
                ]
            ]
            g.tube(centers, [0.018, 0.014, 0.009, 0.0036], 6, WHITE, True)
    else:
        g.part, g.label = "body", "bow"
        g.tube(
            [grip + Vector((0, 0.19, -0.87)), grip, grip + Vector((0, 0.19, 0.87))],
            [0.004, 0.019, 0.004],
            3,
            WHITE,
            True,
        )

    g.part, g.label = "body", "quiver"
    if detailed:
        g.tube(
            [(-0.265, 0.134, 0.49), (-0.254, 0.145, 0.78), (-0.245, 0.155, 1.035)],
            [0.040, 0.050, 0.058],
            6,
            WHITE,
            True,
        )
        g.label = "quiver_arrows"
        for x, top in [(-0.274, 1.205), (-0.245, 1.231), (-0.215, 1.218)]:
            g.add(
                [
                    (x - 0.014, 0.13, top),
                    (x + 0.014, 0.13, top),
                    (x, 0.185, top + 0.007),
                    (x, 0.155, 1.035),
                ],
                [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)],
                WHITE,
            )
    else:
        g.tube(
            [(-0.265, 0.134, 0.49), (-0.245, 0.155, 1.231)],
            [0.040, 0.047],
            3,
            WHITE,
            True,
        )
        g.components[-1] = "quiver_arrows"
    return g


def categories():
    return {
        "lower_tunic": ["tunic", "tunic_seam", "belt", "patch"],
        "upper_tunic": ["tunic"],
        "cape": ["hood_cape", "hood_fold", "neck"],
        "shoulder": ["tunic_sleeve"],
        "head": ["linen_coif", "coif_tie"],
        "face": ["face"],
        "belt": ["belt"],
        "leg_l": 2,
        "leg_r": 3,
        "legs": ["hose", "shoe", "shoe_edge"],
        "arm_weapon": 1,
        "forearm_draw": 9,
        "hand_draw": 10,
        "arm_bow": 6,
        "forearm_bow": 11,
        "hand_bow": 12,
        "grip": 8,
        "bow_upper": 14,
        "bow_lower": 15,
        "bow": ["bow_wood", "bow_grip", "bow_nock"],
        "quiver": ["quiver", "quiver_rim", "quiver_hanger"],
        "quiver_arrows": ["arrow_shaft", "fletching"],
    }
