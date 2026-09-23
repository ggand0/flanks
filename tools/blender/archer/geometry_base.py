"""Original levy archer: linen coif, lowered wool hood, tunic and hip quiver."""
import math
import numpy as np
from mathutils import Vector
import geometry_reference as ref
from face_geometry import build_face
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

SKIN = ref.SKIN
LINEN = (0.56, 0.50, 0.39, 0)
WOOL = (0.16, 0.092, 0.049, 0)
HORN = (0.052, 0.040, 0.026, 0)


def retain(g, source, labels, part_map=None):
    selected = [i for i, c in enumerate(source.components) if c in labels]
    ids = sorted({v for i in selected for v in source.faces[i]})
    offset = len(g.vertices)
    mapping = {v: offset + i for i, v in enumerate(ids)}
    g.vertices.extend(source.vertices[v] for v in ids)
    g.parts.extend((part_map or {}).get(source.parts[v], source.parts[v]) for v in ids)
    g.faces.extend(tuple(mapping[v] for v in source.faces[i]) for i in selected)
    for name in ["colors", "smooth", "components"]:
        getattr(g, name).extend(getattr(source, name)[i] for i in selected)


def tunic(g):
    g.part, g.label = "body", "tunic"
    n = 20
    rows = [
        (0.51, 0.203, 0.129),
        (0.58, 0.198, 0.125),
        (0.73, 0.182, 0.122),
        (0.94, 0.153, 0.110),
        (1.035, 0.146, 0.106),
        (1.12, 0.163, 0.124),
        (1.30, 0.180, 0.128),
        (1.425, 0.178, 0.105),
        (1.47, 0.107, 0.084),
    ]
    pts = []
    for r, (z, rx, ry) in enumerate(rows):
        for i in range(n):
            a = 2 * math.pi * (i + 0.5) / n
            fold = (0.009 * math.cos(a * 7) + 0.004 * math.sin(a * 3)) * (
                1 if r < 4 else 0.45
            )
            pts.append(
                (
                    (rx + fold) * math.sin(a),
                    -(ry + fold) * math.cos(a),
                    z + (0.005 * math.cos(a * 5) if r == 0 else 0),
                )
            )
    faces = []
    for r in range(len(rows) - 1):
        for i in range(n):
            if r < 2 and i in [n - 1, n // 2 - 1]:
                continue
            a, b = r * n + i, r * n + (i + 1) % n
            faces.append((a, b, b + n, a + n))
    faces.append(tuple((len(rows) - 1) * n + i for i in range(n)))
    g.add(pts, faces, CLOTH, True)
    # Raised seam strips use the same worn wool, not decorative trim.
    g.label = "tunic_seam"
    for side in [-1, 1]:
        p = [Vector(pts[r * n + (0 if side < 0 else n - 1)]) for r in range(5)]
        for a, b in zip(p, p[1:]):
            d = Vector((side * 0.005, -0.001, 0))
            g.add([a, b, b + d, a + d], [(0, 1, 2, 3)], CLOTH, True)
    g.label = "patch"
    patch = []
    for z in [0.610, 0.679]:
        for x in [-0.159, -0.130, -0.101]:
            rx = float(np.interp(z, [r[0] for r in rows], [r[1] for r in rows]))
            ry = float(np.interp(z, [r[0] for r in rows], [r[2] for r in rows]))
            a = math.asin(x / rx)
            fold = 0.009 * math.cos(a * 7) + 0.004 * math.sin(a * 3)
            patch.append((x, -(ry + fold) * math.cos(a) - 0.006, z))
    g.add(patch, [(0, 1, 4, 3), (1, 2, 5, 4)], WOOL[:3] + (0.35,), True)


def head(g):
    build_face(g, SKIN)
    g.part, g.label = "body", "neck"
    g.loft(
        [(1.45, 0, 0.012, 0.048, 0.052), (1.553, 0, 0.012, 0.048, 0.052)], 8, SKIN, True
    )
    g.part, g.label = "body", "linen_coif"
    # Open-front linen shell, with a rounded crown and narrow cheek bindings.
    rows = [
        (1.51, 0.072, 0.082),
        (1.55, 0.088, 0.102),
        (1.62, 0.096, 0.108),
        (1.69, 0.097, 0.110),
        (1.728, 0.088, 0.103),
        (1.765, 0.065, 0.080),
        (1.793, 0.030, 0.042),
        (1.80, 0.006, 0.008),
    ]
    n = 18
    pts = []
    for r, (z, rx, ry) in enumerate(rows):
        for i in range(n):
            a = 2 * math.pi * (i - n // 2) / n
            forward = 0.006 if r in [3, 4] and abs(a) < 1.3 else 0
            pts.append((rx * math.sin(a), 0.008 - ry * math.cos(a) - forward, z))
    faces = []
    for r in range(len(rows) - 1):
        for i in range(n):
            a = 2 * math.pi * (i + 0.5 - n // 2) / n
            if r < 3 and abs(a) < 0.91:
                continue
            j, k = r * n + i, r * n + (i + 1) % n
            faces.append((j, k, k + n, j + n))
    faces.append(tuple((len(rows) - 1) * n + i for i in range(n)))
    g.add(pts, faces, LINEN, True)
    g.label = "linen_binding"
    left = [
        (
            -rx * math.sin(0.99),
            0.008 - ry * math.cos(0.99) - (0.009 if z > 1.68 else 0.003),
            z,
        )
        for z, rx, ry in rows[:4]
    ]
    arch = [
        (0.097 * math.sin(a), 0.002 - 0.114 * math.cos(a), 1.695 + 0.003 * math.cos(a))
        for a in [-0.66, -0.33, 0, 0.33, 0.66]
    ]
    right = [(-x, y, z) for x, y, z in reversed(left)]
    binding = left + arch + right
    g.tube(binding, [0.003] * len(binding), 3, LINEN, True)
    # Cloth ties at the chin, with short hanging ends.
    g.label = "coif_tie"
    for sign in [-1, 1]:
        g.tube(
            [
                (sign * 0.058, -0.053, 1.517),
                (sign * 0.012, -0.102, 1.494),
                (sign * 0.008, -0.119, 1.455),
                (sign * 0.026, -0.129, 1.432),
            ],
            [0.0025] * 4,
            4,
            LINEN,
            True,
        )


def hood(g):
    g.part, g.label = "body", "hood_cape"
    n = 20
    pts = []
    for r, (z, rx, ry) in enumerate(
        [
            (1.305, 0.243, 0.161),
            (1.39, 0.223, 0.145),
            (1.47, 0.138, 0.106),
            (1.502, 0.091, 0.089),
        ]
    ):
        for i in range(n):
            a = 2 * math.pi * (i + 0.5) / n
            dz = (0.027 * abs(math.cos(a)) + 0.006 * math.cos(a * 5)) if r == 0 else 0
            # Short cape settles over the shoulder rather than becoming a cloak.
            pts.append((rx * math.sin(a), -ry * math.cos(a), z + dz))
    faces = [
        (r * n + i, r * n + (i + 1) % n, (r + 1) * n + (i + 1) % n, (r + 1) * n + i)
        for r in range(3)
        for i in range(n)
    ]
    g.add(pts, faces, WOOL, True)
    g.label = "hood_fold"
    # Folded hood on the upper back; no long tail.
    g.add(
        [
            (-0.090, 0.080, 1.515),
            (0.090, 0.080, 1.515),
            (-0.112, 0.147, 1.455),
            (0.112, 0.147, 1.455),
            (-0.083, 0.179, 1.392),
            (0.083, 0.179, 1.392),
            (0, 0.176, 1.343),
            (0, 0.139, 1.472),
        ],
        [(0, 1, 7), (0, 7, 2), (1, 3, 7), (2, 7, 4), (7, 3, 5, 4), (4, 5, 6)],
        WOOL,
        True,
    )
    g.tube(
        [
            (-0.109, 0.069, 1.489),
            (-0.075, 0.127, 1.467),
            (0, 0.143, 1.449),
            (0.075, 0.127, 1.467),
            (0.109, 0.069, 1.489),
        ],
        [0.015, 0.019, 0.016, 0.019, 0.015],
        6,
        WOOL,
        True,
    )


def arms(g):
    for sign, part in [(-1, "arm_weapon"), (1, "arm_bow")]:
        g.part, g.label = part, "tunic_sleeve"
        centres = [
            (sign * 0.173, 0, 1.42),
            (sign * 0.218, 0.006, 1.395),
            (sign * 0.264, 0.018, 1.29),
            (sign * 0.303, 0.016, 1.195),
            (sign * 0.324, -0.036, 1.10),
            (sign * 0.352, -0.121, 1.015),
        ]
        g.tube(centres, [0.065, 0.074, 0.067, 0.057, 0.046, 0.037], 10, CLOTH, True)
        g.label = "linen_cuff"
        g.tube(
            [(sign * 0.350, -0.111, 1.028), (sign * 0.369, -0.163, 0.986)],
            [0.049, 0.042],
            8,
            LINEN,
            True,
        )
        g.label = "hand"
        g.tube(
            [
                (sign * 0.369, -0.16, 0.990),
                (sign * 0.390, -0.199, 0.962),
                (sign * 0.395, -0.238, 0.951),
            ],
            [0.033, 0.041, 0.031],
            8,
            SKIN,
            True,
        )
        g.tube(
            [(sign * 0.362, -0.186, 0.954), (sign * 0.368, -0.23, 0.932)],
            [0.015, 0.015],
            6,
            SKIN,
            True,
        )
        if sign == 1:
            g.label = "bracer"
            g.tube(
                [
                    (0.319, -0.025, 1.123),
                    (0.339, -0.075, 1.065),
                    (0.356, -0.128, 1.012),
                ],
                [0.059, 0.056, 0.051],
                8,
                LEATHER,
                False,
                cap=False,
            )
            g.label = "bracer_strap"
            for c in [
                [(0.323, -0.036, 1.112), (0.328, -0.048, 1.099)],
                [(0.350, -0.109, 1.034), (0.354, -0.122, 1.021)],
            ]:
                g.tube(
                    c,
                    [0.060 if c[0][2] > 1.1 else 0.054] * 2,
                    6,
                    LEATHER_EDGE,
                    False,
                    cap=False,
                )


def belt_and_knife(g):
    g.part, g.label = "body", "belt"
    g.loft(
        [(1.014, 0, 0, 0.158, 0.122), (1.041, 0, 0, 0.158, 0.122)],
        20,
        LEATHER,
        False,
        cap=False,
    )
    g.box((-0.014, -0.124, 1.027), (0.041, 0.009, 0.033), BRASS)
    g.box((-0.014, -0.13, 1.027), (0.028, 0.005, 0.020), LEATHER)
    g.box((0.031, -0.118, 0.928), (0.018, 0.008, 0.167), LEATHER)
    g.label = "knife_sheath"
    g.tube(
        [(0.168, 0.025, 1.014), (0.205, 0.017, 0.80), (0.211, 0.015, 0.761)],
        [0.022, 0.018, 0.007],
        6,
        LEATHER,
        False,
    )
    g.label = "knife_grip"
    g.tube(
        [(0.168, 0.025, 1.014), (0.153, 0.03, 1.102)],
        [0.014, 0.013],
        6,
        LEATHER_EDGE,
        False,
    )
    g.label = "knife_metal"
    g.box((0.165, 0.025, 1.016), (0.061, 0.016, 0.011), STEEL)
    g.tube([(0.153, 0.03, 1.10), (0.152, 0.03, 1.113)], [0.016, 0.016], 6, STEEL, False)


def bow(g):
    g.part, g.label = "arm_bow", "bow_wood"
    x = 0.395
    stations = [
        (0.035, -0.055, 0.003, 0.004),
        (0.18, -0.097, 0.005, 0.009),
        (0.38, -0.16, 0.009, 0.012),
        (0.64, -0.225, 0.013, 0.017),
        (0.89, -0.245, 0.015, 0.020),
        (0.995, -0.245, 0.015, 0.020),
        (1.23, -0.211, 0.012, 0.016),
        (1.49, -0.137, 0.008, 0.011),
        (1.69, -0.073, 0.0045, 0.007),
        (1.79, -0.055, 0.003, 0.004),
    ]
    profile = [
        (-1, -0.45),
        (1, -0.45),
        (1, 0.1),
        (0.65, 0.65),
        (0, 0.85),
        (-0.65, 0.65),
        (-1, 0.1),
    ]
    pts = [(x + u * w, y + v * t, z) for z, y, w, t in stations for u, v in profile]
    n = len(profile)
    faces = [
        (r * n + i, r * n + (i + 1) % n, (r + 1) * n + (i + 1) % n, (r + 1) * n + i)
        for r in range(len(stations) - 1)
        for i in range(n)
    ]
    faces.extend(
        [
            tuple(reversed(range(n))),
            tuple((len(stations) - 1) * n + i for i in range(n)),
        ]
    )
    g.add(pts, faces, LEATHER_EDGE, True)
    g.label = "bow_grip"
    g.loft(
        [(0.904, x, -0.245, 0.016, 0.017), (0.980, x, -0.245, 0.016, 0.017)],
        8,
        LEATHER,
        True,
    )
    g.label = "bow_string"
    g.tube([(x, -0.055, 0.041), (x, -0.055, 1.784)], [0.0012, 0.0012], 4, LINEN, True)
    g.label = "bow_nock"
    for z in [0.04, 1.785]:
        g.tube(
            [(x, -0.055, z - 0.006), (x, -0.055, z + 0.006)],
            [0.004, 0.003],
            5,
            HORN,
            False,
        )


def quiver(g):
    g.part, g.label = "body", "quiver"
    bottom = Vector((-0.265, 0.134, 0.49))
    top = Vector((-0.245, 0.155, 1.035))
    n = 12
    pts = []
    for c, r in [
        (bottom, 0.035),
        (top, 0.050),
        (top + Vector((0, 0, -0.015)), 0.043),
        (bottom + Vector((0, 0, 0.02)), 0.030),
    ]:
        pts.extend(
            tuple(
                c
                + Vector(
                    (
                        r * math.sin(i * 2 * math.pi / n),
                        r * math.cos(i * 2 * math.pi / n),
                        0,
                    )
                )
            )
            for i in range(n)
        )
    faces = []
    for r in range(3):
        for i in range(n):
            a, b = r * n + i, r * n + (i + 1) % n
            faces.append((a, b, b + n, a + n))
    faces.append(tuple(reversed(range(n))))
    g.add(pts, faces, LEATHER, True)
    g.label = "quiver_rim"
    g.loft(
        [(1.014, top.x, top.y, 0.052, 0.052), (1.037, top.x, top.y, 0.052, 0.052)],
        12,
        LEATHER_EDGE,
        True,
        cap=False,
    )
    g.label = "quiver_hanger"
    for x, y in [(-0.20, 0.13), (-0.185, 0.18)]:
        g.tube(
            [(x, y, 1.03), (-0.225, y, 1.064), (-0.246, y, 1.025)],
            [0.009] * 3,
            4,
            LEATHER,
            False,
        )
    # Six complete shafts inside the quiver; their fletched tails form a fan.
    for i in range(6):
        a = 2 * math.pi * i / 6
        end = Vector(
            (
                top.x + 0.030 * math.sin(a),
                top.y + 0.029 * math.cos(a),
                1.205 + 0.013 * (i % 3),
            )
        )
        start = Vector(
            (bottom.x + 0.014 * math.sin(a), bottom.y + 0.014 * math.cos(a), 0.531)
        )
        g.label = "arrow_shaft"
        g.tube([start, end], [0.0028, 0.0028], 4, LEATHER_EDGE, True)
        g.label = "fletching"
        axis = (end - start).normalized()
        u = Vector((1, 0, 0))
        v = axis.cross(u).normalized()
        u = v.cross(axis).normalized()
        for k in range(3):
            radial = u * math.cos(k * 2 * math.pi / 3) + v * math.sin(
                k * 2 * math.pi / 3
            )
            c = end - axis * 0.009
            p = [
                c,
                c - axis * 0.067,
                c - axis * 0.060 + radial * 0.010,
                c - axis * 0.014 + radial * 0.012,
            ]
            # Paired feather faces remain visible from either side with an opaque material.
            offset = axis.cross(radial) * 0.0006
            g.add(
                [q + offset for q in p] + [q - offset for q in p],
                [(0, 1, 2, 3), (7, 6, 5, 4)],
                LINEN,
                False,
            )
        g.label = "arrowhead"
        # Alternating broadheads and narrow bodkins, pointing down inside
        # the quiver. No loose arrow is welded to the drawing hand.
        if i % 2 == 0:
            p = [
                start - axis * 0.027,
                start + u * 0.008,
                start - u * 0.008,
                start + axis * 0.004,
                start - axis * 0.005 + v * 0.0015,
                start - axis * 0.005 - v * 0.0015,
            ]
            g.add(
                p,
                [
                    (0, 1, 4),
                    (1, 3, 4),
                    (3, 2, 4),
                    (2, 0, 4),
                    (1, 0, 5),
                    (3, 1, 5),
                    (2, 3, 5),
                    (0, 2, 5),
                ],
                STEEL,
            )
        else:
            p = [
                start + u * 0.003 + v * 0.003,
                start - u * 0.003 + v * 0.003,
                start - u * 0.003 - v * 0.003,
                start + u * 0.003 - v * 0.003,
                start - axis * 0.030,
            ]
            g.add(p, [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4), (3, 2, 1, 0)], STEEL)


def build():
    g = Geometry()
    tunic(g)
    head(g)
    hood(g)
    arms(g)
    belt_and_knife(g)
    temp = Geometry()
    ref.legs_and_shoes(temp)
    # Hidden upper hose must fit inside the tunic's narrower waist and folds.
    ids = {
        v
        for face, label in zip(temp.faces, temp.components)
        if label == "hose"
        for v in face
    }
    for i in ids:
        x, y, z = temp.vertices[i]
        t = max(0, min(1, (z - 0.60) / 0.35))
        temp.vertices[i] = (x * (1 - 0.15 * t), y * (1 - 0.10 * t), z)
    retain(g, temp, {"hose", "shoe", "shoe_edge"})
    bow(g)
    quiver(g)
    return g


measurements = ref.measurements
