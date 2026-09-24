"""Reusable adult face scaffold. Metres, Blender -Y forward, body part.

Landmarks follow the reference's lean cheeks, straight nasal bridge and
closed mouth. A continuous front surface provides a single texture island.
Rows bend around the eyelids, alae and lips instead of forming a round muzzle.
The rear closure is hidden inside the coif; no hat geometry is included here.
"""


def build_face(g, skin):
    # Each half row is (x, forward y, height z), from centre to right temple.
    rows = [
        [
            (0, -0.059, 1.512),
            (0.016, -0.058, 1.513),
            (0.032, -0.047, 1.522),
            (0.048, -0.026, 1.538),
            (0.060, 0.011, 1.555),
        ],
        [
            (0, -0.101, 1.533),
            (0.018, -0.099, 1.534),
            (0.036, -0.084, 1.541),
            (0.052, -0.058, 1.548),
            (0.067, 0.005, 1.564),
        ],
        [
            (0, -0.098, 1.551),
            (0.015, -0.098, 1.552),
            (0.033, -0.081, 1.556),
            (0.052, -0.058, 1.570),
            (0.071, 0.004, 1.580),
        ],
        [
            (0, -0.108, 1.565),
            (0.013, -0.108, 1.566),
            (0.027, -0.096, 1.570),
            (0.050, -0.065, 1.581),
            (0.074, 0.002, 1.593),
        ],
        [
            (0, -0.106, 1.572),
            (0.012, -0.107, 1.573),
            (0.027, -0.096, 1.574),
            (0.052, -0.067, 1.589),
            (0.076, 0.001, 1.605),
        ],
        [
            (0, -0.109, 1.578),
            (0.010, -0.110, 1.580),
            (0.025, -0.098, 1.580),
            (0.052, -0.075, 1.600),
            (0.078, 0.001, 1.617),
        ],
        [
            (0, -0.116, 1.593),
            (0.012, -0.119, 1.596),
            (0.020, -0.102, 1.597),
            (0.052, -0.086, 1.613),
            (0.077, 0.002, 1.628),
        ],
        [
            (0, -0.138, 1.606),
            (0.008, -0.131, 1.605),
            (0.019, -0.103, 1.605),
            (0.052, -0.091, 1.622),
            (0.076, 0.003, 1.637),
        ],
        [
            (0, -0.129, 1.623),
            (0.010, -0.118, 1.622),
            (0.024, -0.093, 1.623),
            (0.046, -0.092, 1.627),
            (0.075, 0.004, 1.646),
        ],
        [
            (0, -0.119, 1.636),
            (0.014, -0.101, 1.635),
            (0.033, -0.095, 1.632),
            (0.050, -0.083, 1.635),
            (0.075, 0.005, 1.655),
        ],
        [
            (0, -0.113, 1.646),
            (0.014, -0.105, 1.640),
            (0.033, -0.096, 1.639),
            (0.051, -0.085, 1.638),
            (0.075, 0.006, 1.666),
        ],
        [
            (0, -0.111, 1.656),
            (0.015, -0.113, 1.650),
            (0.034, -0.110, 1.649),
            (0.053, -0.089, 1.646),
            (0.076, 0.007, 1.678),
        ],
        [
            (0, -0.108, 1.681),
            (0.018, -0.108, 1.677),
            (0.036, -0.102, 1.674),
            (0.055, -0.078, 1.674),
            (0.075, 0.008, 1.695),
        ],
        [
            (0, -0.095, 1.717),
            (0.016, -0.092, 1.715),
            (0.034, -0.083, 1.713),
            (0.053, -0.060, 1.712),
            (0.070, 0.010, 1.715),
        ],
    ]
    points = []
    for row in rows:
        points.extend([(-x, y, z) for x, y, z in reversed(row[1:])] + row)
    n = 9
    faces = [
        (r * n + i, r * n + i + 1, (r + 1) * n + i + 1, (r + 1) * n + i)
        for r in range(len(rows) - 1)
        for i in range(n - 1)
    ]
    # Closed perimeter: avoids open surfaces while spending only 42 triangles
    # on the part concealed by the unchanged coif.
    edge = list(range(n)) + [r * n + n - 1 for r in range(1, len(rows))]
    edge += list(range((len(rows) - 1) * n + n - 2, (len(rows) - 1) * n - 1, -1))
    edge += [r * n for r in reversed(range(1, len(rows) - 1))]
    back = len(points)
    points.append((0, 0.065, 1.615))
    faces.extend((edge[i], back, edge[(i + 1) % len(edge)]) for i in range(len(edge)))
    g.part, g.label = "body", "face"
    g.add(points, faces, skin, True)
