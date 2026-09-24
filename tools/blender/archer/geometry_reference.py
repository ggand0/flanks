"""Original kettle-hatted sergeant. Five rigid parts; no animation work."""
import math
from collections import Counter
import numpy as np
from mathutils import Vector
from geometry_common import (
    Geometry,
    PARTS,
    PIVOTS,
    MAIL,
    STEEL,
    EDGE,
    DARK,
    CLOTH,
    LEATHER,
    LEATHER_EDGE,
    IVORY,
    BRASS,
)

SKIN = (0.36, 0.235, 0.16, 0.0)
QUILT = (0.46, 0.415, 0.32, 0.0)
HOSE = (0.12, 0.102, 0.076, 0.0)
STITCH = (0.23, 0.187, 0.13, 0.0)


def split_skirt(g, rings, color, n=16, split_rows=2, smooth=True, cap=True):
    points = []
    for z, rx, ry in rings:
        for i in range(n):
            a = 2 * math.pi * (i + 0.5) / n
            points.append((rx * math.sin(a), -ry * math.cos(a), z))
    faces = []
    for r in range(len(rings) - 1):
        for i in range(n):
            if r < split_rows and i in [n // 2 - 1, n - 1]:
                continue
            a, b = r * n + i, r * n + (i + 1) % n
            faces.append((a, b, b + n, a + n))
    if cap:
        faces.append(tuple((len(rings) - 1) * n + i for i in range(n)))
    g.add(points, faces, color, smooth)


def clothing(g):
    g.part, g.label = "body", "quilt_hem"
    split_skirt(
        g,
        [(0.535, 0.213, 0.127), (0.58, 0.200, 0.120), (0.66, 0.189, 0.112)],
        QUILT,
        16,
        2,
        cap=False,
    )
    g.label = "mail_body"
    split_skirt(
        g,
        [
            (0.585, 0.213, 0.137),
            (0.69, 0.205, 0.138),
            (0.83, 0.184, 0.12),
            (1.04, 0.160, 0.116),
            (1.20, 0.177, 0.133),
            (1.38, 0.197, 0.13),
            (1.46, 0.148, 0.095),
        ],
        MAIL,
        12,
        2,
    )
    g.label = "surcoat"
    # Tailored front and back panels; a shallow opening is covered by the coif.
    ts = [-1, -0.72, -0.40, -0.12, 0.12, 0.40, 0.72, 1]
    rows = [
        (0.622, 0.208, 0.149),
        (0.78, 0.199, 0.146),
        (0.93, 0.177, 0.132),
        (1.045, 0.165, 0.127),
        (1.20, 0.180, 0.150),
        (1.36, 0.193, 0.151),
        (1.445, 0.187, 0.121),
    ]
    points = []
    for sign in [-1, 1]:
        for r, (z, w, d) in enumerate(rows):
            for i, t in enumerate(ts):
                fold = [0, 0.010, -0.002, 0.009, 0.009, -0.002, 0.010, 0][i] * (
                    1 if r < 3 else 0.30
                )
                h = z + (
                    [0, 0.005, 0.003, 0.018, 0.018, 0.003, 0.005, 0][i] if r == 0 else 0
                )
                if r == 6 and abs(t) < 0.5:
                    h -= 0.063 if abs(t) < 0.2 else 0.031
                points.append(
                    (w * t, sign * (d * math.sqrt(1 - 0.34 * t * t) + fold), h)
                )
    faces = []
    for offset in [0, 56]:
        for r in range(6):
            for i in range(7):
                if r < 2 and i == 3:
                    continue
                a = offset + r * 8 + i
                faces.append((a, a + 1, a + 9, a + 8))
    for columns in [(0, 1, 2), (5, 6, 7)]:
        previous = [48 + i for i in columns]
        for y in [-0.05, 0, 0.05]:
            current = []
            for i in columns:
                current.append(len(points))
                points.append(
                    (
                        ts[i] * 0.187,
                        y,
                        1.467 + 0.003 * abs(ts[i]) - 0.004 * abs(y) / 0.05,
                    )
                )
            for k in range(2):
                faces.append((previous[k], previous[k + 1], current[k + 1], current[k]))
            previous = current
        back = [104 + i for i in columns]
        for k in range(2):
            faces.append((previous[k], previous[k + 1], back[k + 1], back[k]))
    g.add(points, faces, CLOTH, True)
    # Narrow self-colored binding along the lower edges, without decoration.
    for side, offset in [(-1, 0), (1, 56)]:
        for a, b in [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]:
            pa, pb = Vector(points[offset + a]), Vector(points[offset + b])
            shift = Vector((0, side * 0.002, 0.008))
            g.add([pa, pb, pb + shift, pa + shift], [(0, 1, 2, 3)], CLOTH)


def belts(g):
    g.part, g.label = "body", "belt"
    outline = []
    for sign, ts in [
        (-1, [-1, -0.82, -0.5, -0.17, 0.17, 0.5, 0.82, 1]),
        (1, [1, 0.82, 0.5, 0.17, -0.17, -0.5, -0.82, -1]),
    ]:
        outline.extend(
            (t * 0.177, sign * (0.133 * math.sqrt(1 - 0.34 * t * t) + 0.008))
            for t in ts
        )
    points = [(x, y, z) for z in [1.026, 1.060] for x, y in outline]
    g.add(
        points,
        [(i, (i + 1) % 16, (i + 1) % 16 + 16, i + 16) for i in range(16)],
        LEATHER,
    )
    g.box((-0.025, -0.147, 1.044), (0.049, 0.009, 0.035), BRASS)
    g.box((-0.025, -0.153, 1.044), (0.033, 0.004, 0.019), LEATHER)
    g.box((-0.025, -0.157, 1.044), (0.003, 0.003, 0.022), BRASS)
    g.box((0.019, -0.146, 0.984), (0.019, 0.007, 0.108), LEATHER)
    for z in [0.957, 0.981, 1.005]:
        g.stud((0.019, -0.151, z), 0.0023, EDGE)
    # Empty leather scabbard at the left hip; the short sword is held forward.
    g.label = "scabbard"
    g.tube(
        [(0.189, 0.022, 1.008), (0.206, 0.086, 0.78), (0.224, 0.14, 0.50)],
        [0.025, 0.024, 0.019],
        6,
        LEATHER,
        False,
    )
    g.tube([(0.224, 0.14, 0.50), (0.225, 0.144, 0.477)], [0.021, 0.008], 6, EDGE, False)
    g.tube(
        [(0.189, 0.022, 1.008), (0.191, 0.028, 0.980)],
        [0.027, 0.027],
        6,
        LEATHER_EDGE,
        False,
    )


def kettle_hat(g):
    g.part, g.label = "body", "kettle_hat"
    # Rolled oval brim, with a modest downward slope. Rounded crown above it.
    n = 20
    g.loft(
        [
            (1.684, 0, 0, 0.113, 0.120),
            (1.729, 0, 0.002, 0.106, 0.116),
            (1.770, 0, 0.004, 0.078, 0.091),
            (1.794, 0, 0.004, 0.036, 0.044),
            (1.800, 0, 0.004, 0.010, 0.013),
        ],
        n,
        STEEL,
        True,
        cap=False,
    )
    g.add(
        [
            (
                0.010 * math.sin(2 * math.pi * i / n),
                0.004 - 0.013 * math.cos(2 * math.pi * i / n),
                1.800,
            )
            for i in range(n)
        ],
        [tuple(range(n))],
        STEEL,
        True,
    )
    rings = [
        (1.693, 0.114, 0.123),
        (1.658, 0.175, 0.187),
        (1.653, 0.176, 0.188),
        (1.684, 0.111, 0.120),
    ]
    points = []
    for z, rx, ry in rings:
        for i in range(n):
            a = 2 * math.pi * i / n
            points.append(
                (rx * math.sin(a), -ry * math.cos(a), z + 0.004 * math.cos(a * 2))
            )
    faces = []
    for r in range(len(rings)):
        nxt = (r + 1) % len(rings)
        for i in range(n):
            faces.append(
                (r * n + i, r * n + (i + 1) % n, nxt * n + (i + 1) % n, nxt * n + i)
            )
    g.add(points, faces, STEEL, True)
    g.label = "helmet_band"
    g.loft(
        [(1.695, 0, 0.001, 0.115, 0.123), (1.705, 0, 0.001, 0.115, 0.123)],
        n,
        EDGE,
        False,
        cap=False,
    )
    for i in range(10):
        a = 2 * math.pi * (i + 0.5) / 10
        g.stud(
            (0.115 * math.sin(a), -0.123 * math.cos(a), 1.700),
            0.0027,
            EDGE,
            (math.sin(a), -math.cos(a), 0),
        )


def head_and_coif(g):
    g.part, g.label = "body", "face"
    # A continuous facial surface: chin, labiomental fold, lower lip, mouth,
    # upper lip, nasal wings/tip, recessed sockets and a projecting brow.
    # Each half-row runs from the centre to the side of the head. The rear
    # closure is entirely inside the coif: spend the triangles on the face.
    # Columns carry (x, y, z offset) relative to their row height.
    rows = [
        (
            1.505,
            [
                (0, -0.055, 0),
                (0.014, -0.054, 0),
                (0.029, -0.046, 0.001),
                (0.040, -0.029, 0.003),
                (0.045, 0.006, 0.005),
            ],
        ),
        (
            1.533,
            [
                (0, -0.088, 0),
                (0.016, -0.089, 0),
                (0.031, -0.080, 0.001),
                (0.045, -0.057, 0.003),
                (0.055, -0.004, 0.009),
            ],
        ),
        (
            1.550,
            [
                (0, -0.094, 0),
                (0.016, -0.093, 0),
                (0.032, -0.087, -0.001),
                (0.052, -0.065, -0.002),
                (0.065, -0.005, 0),
            ],
        ),
        (
            1.566,
            [
                (0, -0.103, 0),
                (0.012, -0.102, 0),
                (0.028, -0.094, 0.001),
                (0.053, -0.069, 0),
                (0.070, -0.006, -0.001),
            ],
        ),
        (
            1.571,
            [
                (0, -0.101, 0),
                (0.011, -0.101, 0.0006),
                (0.028, -0.094, -0.001),
                (0.054, -0.071, 0),
                (0.072, -0.006, -0.001),
            ],
        ),
        (
            1.578,
            [
                (0, -0.105, -0.001),
                (0.010, -0.106, 0),
                (0.027, -0.097, -0.002),
                (0.053, -0.074, 0),
                (0.074, -0.005, 0),
            ],
        ),
        (
            1.596,
            [
                (0, -0.112, -0.003),
                (0.012, -0.114, 0),
                (0.029, -0.095, 0),
                (0.053, -0.080, 0.004),
                (0.076, -0.004, 0.004),
            ],
        ),
        (
            1.606,
            [
                (0, -0.129, 0),
                (0.008, -0.125, -0.001),
                (0.029, -0.098, 0.002),
                (0.054, -0.085, 0.009),
                (0.077, -0.003, 0.010),
            ],
        ),
        (
            1.635,
            [
                (0, -0.114, 0),
                (0.010, -0.105, -0.001),
                (0.032, -0.094, 0),
                (0.052, -0.082, 0.001),
                (0.077, -0.001, 0.003),
            ],
        ),
        (
            1.652,
            [
                (0, -0.110, -0.002),
                (0.014, -0.111, -0.003),
                (0.033, -0.104, 0.001),
                (0.054, -0.080, 0),
                (0.077, 0, 0.002),
            ],
        ),
        (
            1.710,
            [
                (0, -0.095, 0),
                (0.014, -0.094, 0),
                (0.031, -0.081, 0),
                (0.051, -0.054, 0),
                (0.070, 0.007, 0),
            ],
        ),
    ]
    points = []
    for r, (z, half) in enumerate(rows):
        profile = [(-x, y, dz) for x, y, dz in reversed(half[1:])] + half
        for x, y, dz in profile:
            # Sub-millimetre asymmetry in the mouth and cheeks, without a
            # permanent snarl or a separate floating nose/lip shell.
            asym = 0.0007 * (x / 0.077) if 2 <= r <= 7 else 0
            points.append((x, y, z + dz + asym))
    # Five extra vertices resolve the bridge/sidewalls instead of stretching
    # a single wedge from the eyes to the tip. Neighbouring pentagons include
    # the edge vertices, keeping the surface connected without T junctions.
    bridge = {c: len(points) + i for i, c in enumerate(range(2, 7))}
    points.extend(
        [
            (-0.023, -0.101, 1.620),
            (-0.006, -0.119, 1.621),
            (0, -0.123, 1.621),
            (0.006, -0.119, 1.621),
            (0.023, -0.101, 1.620),
        ]
    )
    faces = []
    for r in range(10):
        for c in range(8):
            a = r * 9 + c
            if r == 7 and 2 <= c <= 5:
                faces.extend(
                    [
                        (a, a + 1, bridge[c + 1], bridge[c]),
                        (bridge[c], bridge[c + 1], a + 10, a + 9),
                    ]
                )
            elif r == 7 and c == 1:
                faces.append((a, a + 1, bridge[2], a + 10, a + 9))
            elif r == 7 and c == 6:
                faces.append((a, a + 1, a + 10, a + 9, bridge[6]))
            else:
                faces.append((a, a + 1, a + 10, a + 9))
    boundary = list(range(9)) + [r * 9 + 8 for r in range(1, 11)]
    boundary += list(range(97, 89, -1)) + [r * 9 for r in range(9, 0, -1)]
    rear = len(points)
    points.append((0, 0.070, 1.614))
    faces += [
        (a, rear, boundary[(i + 1) % len(boundary)]) for i, a in enumerate(boundary)
    ]
    g.add(points, faces, SKIN, True)
    g.part, g.label = "body", "mail_coif"
    rings = [
        (1.402, 0.140, 0.109),
        (1.456, 0.121, 0.102),
        (1.514, 0.093, 0.093),
        (1.542, 0.093, 0.101),
        (1.602, 0.100, 0.110),
        (1.670, 0.105, 0.114),
        (1.704, 0.102, 0.111),
    ]
    n = 20
    points = []
    for z, rx, ry in rings:
        for i in range(n):
            a = 2 * math.pi * (i - n // 2) / n
            points.append((rx * math.sin(a), -ry * math.cos(a), z))
    faces = []
    for r in range(len(rings) - 1):
        for i in range(n):
            a = 2 * math.pi * (i + 0.5 - n // 2) / n
            if r >= 2 and abs(a) < 0.96:
                continue
            j, k = r * n + i, r * n + (i + 1) % n
            faces.append((j, k, k + n, j + n))
    g.add(points, faces, MAIL, True)
    g.label = "arming_cap"
    g.loft(
        [(1.68, 0, 0.005, 0.098, 0.104), (1.724, 0, 0.005, 0.095, 0.102)],
        12,
        QUILT,
        True,
        cap=False,
    )


def arms(g):
    for sign, part in [(-1, "arm_spear"), (1, "arm_shield")]:
        g.part, g.label = part, "quilt_sleeve"
        centers = [
            (sign * 0.185, 0, 1.414),
            (sign * 0.230, 0, 1.417),
            (sign * 0.275, 0, 1.38),
            (sign * 0.306, 0.006, 1.315),
            (sign * 0.333, -0.016, 1.228),
            (sign * 0.352, -0.069, 1.16),
            (sign * 0.378, -0.148, 1.108),
        ]
        g.tube(centers[2:], [0.073, 0.066, 0.061, 0.057, 0.048], 10, QUILT, True)
        g.label = "mail_sleeve"
        g.tube(
            centers[:4] + [(sign * 0.315, 0.001, 1.29)],
            [0.070, 0.080, 0.083, 0.076, 0.073],
            10,
            MAIL,
            True,
        )
        g.label = "quilt_cuff"
        g.tube(
            [(sign * 0.369, -0.128, 1.121), (sign * 0.378, -0.149, 1.107)],
            [0.052, 0.050],
            10,
            QUILT,
            True,
            cap=False,
        )
        g.label = "hand"
        g.tube(
            [
                (sign * 0.377, -0.146, 1.110),
                (sign * 0.390, -0.185, 1.092),
                (sign * 0.39, -0.217, 1.085),
            ],
            [0.038, 0.047, 0.037],
            8,
            SKIN,
            True,
        )
        g.tube(
            [(sign * 0.362, -0.174, 1.085), (sign * 0.363, -0.219, 1.065)],
            [0.018, 0.017],
            6,
            SKIN,
            True,
        )


def legs_and_shoes(g):
    for sign, part in [(1, "leg_l"), (-1, "leg_r")]:
        x = sign * 0.112
        g.part, g.label = part, "hose"
        g.loft(
            [
                (0.115, x, 0.003, 0.043, 0.054),
                (0.30, x, 0.004, 0.060, 0.067),
                (0.41, x, 0.004, 0.069, 0.074),
                (0.49, x, -0.006, 0.064, 0.069),
                (0.56, x, -0.012, 0.069, 0.073),
                (0.69, sign * 0.100, 0, 0.076, 0.080),
                (0.95, sign * 0.075, 0, 0.080, 0.090),
            ],
            10,
            HOSE,
            True,
        )
        g.label = "shoe"
        stations = [
            (0.080, 0.043, 0.085),
            (0.028, 0.055, 0.130),
            (-0.055, 0.057, 0.110),
            (-0.139, 0.044, 0.071),
            (-0.205, 0.021, 0.041),
            (-0.229, 0.008, 0.028),
        ]
        points = []
        for y, w, h in stations:
            points.extend(
                (x + xx, y, z)
                for xx, z in [
                    (-w, 0.011),
                    (-w, h * 0.48),
                    (-w * 0.65, h * 0.90),
                    (0, h),
                    (w * 0.65, h * 0.90),
                    (w, h * 0.48),
                    (w, 0.011),
                ]
            )
        faces = []
        for r in range(len(stations) - 1):
            for i in range(7):
                a, b = r * 7 + i, r * 7 + (i + 1) % 7
                faces.append((a, b, b + 7, a + 7))
        faces.extend([tuple(reversed(range(7))), tuple(35 + i for i in range(7))])
        g.add(points, faces, LEATHER, True)
        # Low quarters and a turnshoe seam, never a tall boot.
        g.loft(
            [
                (0.073, x, 0.010, 0.052, 0.064),
                (0.135, x, 0.008, 0.050, 0.062),
                (0.157, x, 0.007, 0.049, 0.061),
            ],
            10,
            LEATHER,
            True,
            cap=False,
        )
        g.label = "shoe_edge"
        outline = [(x - w, y) for y, w, h in stations] + [
            (x + w, y) for y, w, h in reversed(stations)
        ]
        n = len(outline)
        points = [(xx, y, z) for z in [0, 0.010] for xx, y in outline]
        faces = [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)]
        faces.extend([tuple(reversed(range(n))), tuple(n + i for i in range(n))])
        g.add(points, faces, LEATHER_EDGE)
        g.loft(
            [(0.150, x, 0.007, 0.051, 0.063), (0.157, x, 0.007, 0.051, 0.063)],
            10,
            LEATHER_EDGE,
            True,
            cap=False,
        )


def heater(g):
    g.part, g.label = "arm_shield", "heater_face"
    tangent = Vector((0.9063, 0.4226, 0))
    normal = Vector((0.4226, -0.9063, 0))
    center = Vector((0.394, -0.206, 1.092))

    def at(u, z, offset=0):
        return (
            Vector((0.394, -0.206, z))
            + tangent * u
            + normal * (0.048 * (1 - (u / 0.205) ** 2) + offset)
        )

    rows = [
        (1.404, 0.190),
        (1.342, 0.204),
        (1.152, 0.181),
        (0.956, 0.130),
        (0.783, 0.051),
        (0.744, 0.004),
    ]
    points = [at(u, z) for z, w in rows for u in [-w, -w * 0.5, 0, w * 0.5, w]]
    faces = [
        (r * 5 + i, r * 5 + i + 1, r * 5 + i + 6, r * 5 + i + 5)
        for r in range(5)
        for i in range(4)
    ]
    g.add(points, faces, CLOTH, True)
    g.label = "heater_wood"
    g.add([p - normal * 0.023 for p in points], faces, LEATHER_EDGE, True)
    g.label = "heater_rim"
    boundary = [0, 1, 2, 3, 4, 9, 14, 19, 24, 29, 28, 27, 26, 25, 20, 15, 10, 5]
    for k, a in enumerate(boundary):
        b = boundary[(k + 1) % len(boundary)]
        pa, pb = Vector(points[a]), Vector(points[b])
        qa, qb = (
            pa.lerp(center, 0.038) + normal * 0.003,
            pb.lerp(center, 0.038) + normal * 0.003,
        )
        g.add([pa + normal * 0.003, pb + normal * 0.003, qb, qa], [(0, 1, 2, 3)], IVORY)
        g.add(
            [pa, pb, pb - normal * 0.023, pa - normal * 0.023], [(0, 1, 2, 3)], LEATHER
        )
    g.label = "heater_fittings"
    for u, z in [
        (-0.167, 1.372),
        (0.167, 1.372),
        (-0.151, 1.145),
        (0.151, 1.145),
        (0, 0.788),
    ]:
        g.stud(at(u, z, 0.009), 0.003, EDGE, normal)
    # Rear grip and arm strap, visible in the back review.
    for z, offset in [(1.22, 0.0), (1.06, 0.04)]:
        g.tube(
            [at(-0.075, z, -0.044), at(0.075, z - offset, -0.044)],
            [0.013, 0.013],
            6,
            LEATHER,
            True,
        )


def spear(g):
    g.part, g.label = "arm_spear", "spear_shaft"
    x, y = -0.390, -0.205
    # 2.50 m complete weapon, butt 2 cm clear of the ground in this rest pose.
    g.tube(
        [(x, y, 0.02), (x, y, 1.10), (x, y, 2.26)],
        [0.016, 0.015, 0.012],
        8,
        LEATHER_EDGE,
        True,
    )
    g.label = "spear_steel"
    g.tube([(x, y, 2.215), (x, y, 2.308)], [0.017, 0.013], 8, STEEL, True)
    g.tube([(x, y, 0.02), (x, y, 0.10)], [0.017, 0.017], 6, STEEL, False)
    points = []
    for z, w, t in [(2.282, 0.009, 0.005), (2.35, 0.035, 0.006), (2.425, 0.027, 0.004)]:
        points.extend([(x - w, y, z), (x, y - t, z), (x + w, y, z), (x, y + t, z)])
    faces = []
    for r in range(2):
        for i in range(4):
            a, b = r * 4 + i, r * 4 + (i + 1) % 4
            faces.append((a, b, b + 4, a + 4))
    points.append((x, y, 2.52))
    faces.extend((8 + i, 8 + (i + 1) % 4, 12) for i in range(4))
    g.add(points, faces, STEEL)


def sidearm_hilt(g):
    g.part, g.label = "body", "sidearm_hilt"
    g.tube(
        [(0.188, 0.020, 1.00), (0.177, -0.006, 1.114)], [0.014, 0.014], 6, LEATHER, True
    )
    g.tube(
        [(0.176, -0.008, 1.111), (0.174, -0.013, 1.131)], [0.026, 0.026], 8, EDGE, False
    )
    g.box((0.188, 0.020, 1.008), (0.136, 0.016, 0.016), EDGE)


def build():
    g = Geometry()
    clothing(g)
    belts(g)
    head_and_coif(g)
    kettle_hat(g)
    arms(g)
    legs_and_shoes(g)
    heater(g)
    spear(g)
    sidearm_hilt(g)
    return g


def measurements(g):
    result = {}
    for label in sorted(set(g.components)):
        ids = {i for face, c in zip(g.faces, g.components) if c == label for i in face}
        points = np.array([g.vertices[i] for i in ids])
        result[label] = {
            "min_xyz": points.min(axis=0).tolist(),
            "max_xyz": points.max(axis=0).tolist(),
            "triangles": sum(
                len(f) - 2 for f, c in zip(g.faces, g.components) if c == label
            ),
        }
    return result
