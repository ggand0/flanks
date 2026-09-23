"""Construct the shared kettle hat, clothing and shield for distant infantry."""
import math
from mathutils import Vector
from common import WHITE


def torso(g):
    g.part, g.label = "body", "torso"
    rings = [
        (0.535, 0.209, 0.128),
        (0.635, 0.208, 0.146),
        (1.045, 0.168, 0.136),
        (1.47, 0.184, 0.112),
    ]
    points = [
        (x * width * 0.94, y * depth * 0.91, z)
        for z, width, depth in rings
        for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    ]
    faces = [(3, 2, 1, 0), (12, 13, 14, 15)]
    faces += [
        (r * 4 + i, r * 4 + (i + 1) % 4, (r + 1) * 4 + (i + 1) % 4, (r + 1) * 4 + i)
        for r in range(3)
        for i in range(4)
    ]
    g.add(points, faces, WHITE, True)


def head(g):
    g.part, g.label = "body", "head"
    g.loft([(1.405, 0, 0, 0.142, 0.110), (1.695, 0, 0, 0.105, 0.118)], 4, WHITE, True)
    g.label = "hat"
    n = 6
    points = [
        (rx * math.sin(2 * math.pi * i / n), cy - ry * math.cos(2 * math.pi * i / n), z)
        for z, rx, ry, cy in [(1.655, 0.192, 0.188, 0), (1.735, 0.115, 0.111, 0.003)]
        for i in range(n)
    ]
    points.append((0, 0.004, 1.8))
    faces = [tuple(reversed(range(n)))]
    faces += [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)]
    faces += [(i + n, (i + 1) % n + n, 2 * n) for i in range(n)]
    g.add(points, faces, WHITE, True)


def legs(g):
    for sign, part in [(1, "leg_l"), (-1, "leg_r")]:
        g.part, g.label = part, part
        rings = [
            (0, 0.112, 0.043, -0.215, 0.080),
            (0.15, 0.112, 0.042, -0.060, 0.068),
            (0.43, 0.112, 0.059, -0.064, 0.065),
            (0.95, 0.075, 0.068, -0.078, 0.078),
        ]
        points = []
        for z, center, width, front, back in rings:
            points.extend(
                [
                    (sign * center - width, front, z),
                    (sign * center + width, front, z),
                    (sign * center + width, back, z),
                    (sign * center - width, back, z),
                ]
            )
        faces = [(3, 2, 1, 0), (12, 13, 14, 15)]
        faces += [
            (r * 4 + i, r * 4 + (i + 1) % 4, (r + 1) * 4 + (i + 1) % 4, (r + 1) * 4 + i)
            for r in range(3)
            for i in range(4)
        ]
        g.add(points, faces, WHITE, True)


def arm(g, sign, part):
    g.part, g.label = part, part
    g.tube(
        [
            (sign * 0.208, 0, 1.423),
            (sign * 0.333, -0.016, 1.228),
            (sign * 0.39, -0.205, 1.085),
        ],
        [0.091, 0.073, 0.050],
        4,
        WHITE,
        True,
    )


def shield(g):
    g.part = "arm_shield"
    tangent = Vector((0.9063, 0.4226, 0))
    normal = Vector((0.4226, -0.9063, 0))

    def at(u, z, depth=0):
        return (
            Vector((0.394, -0.206, z))
            + tangent * u
            + normal * (0.048 * (1 - (u / 0.205) ** 2) + depth)
        )

    border = [
        (-0.190, 1.404),
        (0.190, 1.404),
        (0.204, 1.342),
        (0.181, 1.152),
        (0.130, 0.956),
        (0, 0.744),
        (-0.130, 0.956),
        (-0.181, 1.152),
        (-0.204, 1.342),
    ]
    n, m = len(border), len(border) + 1
    front = [at(u, z) for u, z in border] + [at(0, 1.17)]
    points = front + [p - normal * 0.023 for p in front]
    faces = [(n, i, (i + 1) % n) for i in range(n)]
    faces += [(n + m, (i + 1) % n + m, i + m) for i in range(n)]
    faces += [(i, i + m, (i + 1) % n + m, (i + 1) % n) for i in range(n)]
    g.label = "heater_face"
    start = len(g.faces)
    g.add(points, faces, WHITE, True)
    for i in range(n, 3 * n):
        g.components[start + i] = "heater_wood" if i < 2 * n else "heater_rim"
    g.label = "heater_straps"
    for z in [1.22, 1.06]:
        g.tube(
            [at(-0.075, z, -0.03), at(0.075, z, -0.065)],
            [0.015, 0.015],
            3,
            WHITE,
            False,
        )


def categories():
    return {
        "torso": ["surcoat", "mail_body", "quilt_hem", "belt"],
        "head": ["mail_coif", "face", "neck", "arming_cap"],
        "hat": ["kettle_hat", "helmet_band"],
        "leg_l": 2,
        "leg_r": 3,
        "arm_shield": ["mail_sleeve", "quilt_sleeve", "quilt_cuff", "hand"],
        "heater_face": ["heater_face"],
        "heater_wood": ["heater_wood"],
        "heater_rim": ["heater_rim"],
        "heater_straps": ["heater_straps"],
    }
