"""Build a levy archer with two covered rigid arm chains and hinged bow limbs."""
import math
import numpy as np
from mathutils import Vector
import geometry_common as common
import geometry_base as base
import motion
from geometry_common import Geometry, CLOTH, LEATHER, LEATHER_EDGE, BRASS, STEEL

PARTS = common.PARTS
PARTS.clear()
PARTS.update({name: pid for pid, name in motion.PARTS.items()})
PIVOTS = common.PIVOTS
PIVOTS.clear()
PIVOTS.update(
    {motion.PARTS[pid]: (p[0], -p[2], p[1]) for pid, p in motion.JOINTS.items()}
)
PROBES = []


def xyz(p):
    return Vector((p[0], -p[2], p[1]))


def ball(g, center, radius, color):
    c = xyz(center)
    # Icosahedron covers have a known insphere and no open pole seams.
    golden = (1 + math.sqrt(5)) / 2
    points = []
    for a in [-1, 1]:
        for b in [-golden, golden]:
            points.extend([(0, a, b), (a, b, 0), (b, 0, a)])
    points = np.array(points, dtype=float)
    points *= radius / np.linalg.norm(points[0])
    faces = []
    edge = np.linalg.norm(points[0] - points[3])
    distances = [
        np.linalg.norm(a - b) for i, a in enumerate(points) for b in points[i + 1 :]
    ]
    edge = min(d for d in distances if d > 1e-6)
    for a in range(12):
        for b in range(a + 1, 12):
            for d in range(b + 1, 12):
                if all(
                    abs(np.linalg.norm(points[i] - points[j]) - edge) < 1e-6
                    for i, j in [(a, b), (b, d), (d, a)]
                ):
                    faces.append((a, b, d))
    vertices = [c + Vector(p) for p in points]
    g.add(vertices, faces, color, True)
    return [[p.x, p.z, -p.y] for p in vertices]


def sleeve(g, points, radii, n, color, joint_probes):
    start = len(g.vertices)
    g.tube([xyz(p) for p in points], radii, n, color, True)
    for offset, probe in joint_probes:
        coords = g.vertices[start + offset * n : start + (offset + 1) * n]
        probe["seams"].append(
            {"part": PARTS[g.part], "points": [[x, z, -y] for x, y, z in coords]}
        )


def arms(g):
    for upper, fore, hand, grip in motion.CHAIN:
        s, e, w = [motion.JOINTS[k] for k in [upper, fore, hand]]
        probes = []
        for joint, p, radius, cover, color in [
            ("shoulder", s, 0.091, 19, CLOTH),
            ("elbow", e, 0.069, upper, CLOTH),
            ("wrist", w, 0.042, hand, base.SKIN),
        ]:
            g.part = motion.PARTS[cover]
            g.label = "hand" if joint == "wrist" else "tunic_sleeve"
            caps = ball(g, p, radius, color)
            probe = {
                "joint": f"{upper}_{joint}",
                "cap_part": cover,
                "cap_vertices": caps,
                "seams": [],
            }
            probes.append(probe)
            PROBES.append(probe)
        g.part, g.label = motion.PARTS[upper], "tunic_sleeve"
        sleeve(
            g,
            [s, s * 0.58 + e * 0.42, e],
            [0.057, 0.073, 0.042],
            8,
            CLOTH,
            [(0, probes[0]), (2, probes[1])],
        )
        g.part = motion.PARTS[fore]
        sleeve(
            g,
            [e, e * 0.45 + w * 0.55, w],
            [0.042, 0.052, 0.024],
            8,
            CLOTH,
            [(0, probes[1]), (2, probes[2])],
        )
        g.label = "linen_cuff"
        # The cuff ends at the same buried wrist ring as the sleeve.
        sleeve(
            g, [e * 0.13 + w * 0.87, w], [0.038, 0.025], 6, base.LINEN, [(1, probes[2])]
        )
        if upper == 6:
            g.label = "bracer"
            sleeve(
                g,
                [e * 0.67 + w * 0.33, e * 0.24 + w * 0.76],
                [0.054, 0.046],
                6,
                LEATHER,
                [],
            )
        g.part, g.label = motion.PARTS[hand], "hand"
        direction = (grip - w) / np.linalg.norm(grip - w)
        sleeve(
            g,
            [w, grip - direction * 0.016, grip + direction * 0.018],
            [0.021, 0.033, 0.023],
            6,
            base.SKIN,
            [(0, probes[2])],
        )
        g.tube(
            [xyz(grip + [-0.021, 0.014, -0.018]), xyz(grip + [-0.024, -0.014, 0.012])],
            [0.011, 0.010],
            5,
            base.SKIN,
            True,
        )


def bow(g):
    grip = motion.GB
    g.part, g.label = "weapon", "bow_grip"
    g.tube(
        [xyz(grip + [0, -0.046, 0]), xyz(grip + [0, 0.046, 0])],
        [0.017, 0.017],
        6,
        LEATHER,
        True,
    )
    for pid, sign in [(14, 1), (15, -1)]:
        g.part, g.label = motion.PARTS[pid], "bow_wood"
        stations = [
            (0.040, 0.0, 0.015),
            (0.26, -0.023, 0.013),
            (0.52, -0.085, 0.009),
            (0.72, -0.150, 0.006),
            (0.87, -0.19, 0.003),
        ]
        points = [xyz(grip + [0, sign * y, z]) for y, z, r in stations]
        g.tube(points, [r for y, z, r in stations], 4, LEATHER_EDGE, True)
        g.label = "bow_nock"
        tip = grip + np.array([0, sign * 0.87, -0.19])
        g.tube(
            [xyz(tip + [0, -0.006, 0]), xyz(tip + [0, 0.006, 0])],
            [0.004, 0.004],
            5,
            base.HORN,
            True,
        )
    for pid, tip in [(13, motion.TOP), (17, motion.BOTTOM)]:
        g.part, g.label = motion.PARTS[pid], "bow_string"
        g.tube([xyz(motion.NOCK), xyz(tip)], [0.0012, 0.0012], 3, base.LINEN, True)
    g.part, g.label = "arrow", "arrow_shaft"
    nock = motion.JOINTS[16]
    g.tube(
        [xyz(nock), xyz(nock + [0, 0, 0.82])], [0.0028, 0.0028], 4, LEATHER_EDGE, True
    )
    g.label = "arrowhead"
    g.tube(
        [xyz(nock + [0, 0, 0.81]), xyz(nock + [0, 0, 0.85])],
        [0.006, 0.0003],
        4,
        STEEL,
        False,
    )
    g.label = "fletching"
    for k in range(3):
        radial = (
            np.array([math.cos(k * 2 * math.pi / 3), math.sin(k * 2 * math.pi / 3), 0])
            * 0.011
        )
        vertices = [
            nock + [0, 0, 0.014],
            nock + [0, 0, 0.075],
            nock + [0, 0, 0.065] + radial,
            nock + [0, 0, 0.025] + radial,
        ]
        # Closed thin wedges preserve back-face culling.
        across = np.cross(radial, [0, 0, 1]) * 0.045
        g.add(
            [xyz(v + across) for v in vertices] + [xyz(v - across) for v in vertices],
            [
                (0, 1, 2, 3),
                (7, 6, 5, 4),
                (0, 4, 5, 1),
                (1, 5, 6, 2),
                (2, 6, 7, 3),
                (3, 7, 4, 0),
            ],
            base.LINEN,
        )


def separate_upper_body(g):
    head = {"face", "linen_coif", "linen_binding", "coif_tie"}
    upper = {"tunic", "tunic_seam", "neck", "hood_cape", "hood_fold"}
    vertices, parts, faces = [], [], []
    remap = {}
    for face, label in zip(g.faces, g.components):
        part = g.parts[face[0]]
        if part == "body" and label in head:
            part = "head"
        elif (
            part == "body"
            and label in upper
            and min(g.vertices[i][2] for i in face) >= 1.035 - 1e-7
        ):
            part = "torso"
        new = []
        for i in face:
            key = (part, i)
            if key not in remap:
                remap[key] = len(vertices)
                vertices.append(g.vertices[i])
                parts.append(part)
            new.append(remap[key])
        faces.append(tuple(new))
    g.vertices, g.parts, g.faces = vertices, parts, faces


def waist_cover(g):
    g.part, g.label = "body", "tunic"
    center = xyz(motion.JOINTS[19])
    radius, n = 0.175, 12
    points = [center + Vector((0, 0, -radius))]
    for latitude in [-math.pi / 4, 0, math.pi / 4]:
        points.extend(
            center
            + radius
            * Vector(
                (
                    math.cos(latitude) * math.cos(i * 2 * math.pi / n),
                    math.cos(latitude) * math.sin(i * 2 * math.pi / n),
                    math.sin(latitude),
                )
            )
            for i in range(n)
        )
    points.append(center + Vector((0, 0, radius)))
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend(
            [(0, 1 + i, 1 + j), (1 + 2 * n + j, 1 + 2 * n + i, len(points) - 1)]
        )
        for row in range(2):
            a, b = 1 + row * n + i, 1 + row * n + j
            faces.append((a, b, b + n, a + n))
    # A cloth overlap follows the waist proportions, rather than making a
    # spherical belly. The upper and lower tunic cover its ends.
    points = [
        center
        + Vector(((p - center).x * 1.02, (p - center).y * 0.70, (p - center).z * 0.68))
        for p in points
    ]
    g.add(points, faces, CLOTH, True)


def build():
    PROBES.clear()
    original = base.build()
    g = Geometry()
    labels = {
        label
        for face, label in zip(original.faces, original.components)
        if all(original.parts[v] in ["body", "leg_l", "leg_r"] for v in face)
    }
    # Arrowheads at the bottom of the closed hip quiver are fully concealed.
    labels.discard("arrowhead")
    base.retain(g, original, labels)
    separate_upper_body(g)
    waist_cover(g)
    arms(g)
    bow(g)
    return g


measurements = base.measurements
