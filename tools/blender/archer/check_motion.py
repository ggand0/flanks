"""Validate exported attributes and articulated motion against GLB geometry."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

SOURCE = Path(__file__).resolve().parent
DEFAULT_OUT = SOURCE.parents[2] / "assets_dev/archer/rebuild_raise_v4"
parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
parser.add_argument("--baseline", type=Path)
parser.add_argument("--reload-baseline", type=Path)
parser.add_argument("--reload-unchanged-from", type=float, default=2.65)
args = parser.parse_args(
    sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
)
ROOT = args.out_dir.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SOURCE))
import motion as m

sys.path.insert(0, str(SOURCE.parent))
import glb_inspect


def load(path):
    doc, blob = glb_inspect.load(str(path))

    def acc(i):
        a = doc["accessors"][i]
        b = doc["bufferViews"][a["bufferView"]]
        dtype = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[a["componentType"]]
        count = glb_inspect.NCOMP[a["type"]]
        size = np.dtype(dtype).itemsize
        array = np.ndarray(
            (a["count"], count),
            dtype=dtype,
            buffer=blob,
            offset=a.get("byteOffset", 0) + b.get("byteOffset", 0),
            strides=(b.get("byteStride", count * size), size),
        ).copy()
        if a.get("normalized"):
            array = array.astype(float) / np.iinfo(dtype).max
        return array

    prim = doc["meshes"][0]["primitives"][0]
    at = prim["attributes"]
    return (
        doc,
        acc(at["POSITION"]).astype(float),
        acc(at["TEXCOORD_1"]),
        acc(prim["indices"]).astype(int).reshape(-1, 3),
    )


def main():
    doc, pos, uv, tri = load(ROOT / "archer.glb")
    part = np.rint(uv[:, 0]).astype(int)
    assert set(part) == set(m.PARTS)
    assert np.max(abs(uv[:, 0] - part)) < 1e-6
    assert np.all(part[tri] == part[tri[:, :1]])
    assert len(tri) <= 3000
    assert abs(pos[part == 18, 1].max() - 1.8) < 1e-6
    assert abs(pos[:, 1].min()) < 1e-6
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    assert doc["accessors"][attrs["COLOR_0"]]["type"] == "VEC4"
    assert "TEXCOORD_0" in attrs
    assert len(doc["materials"]) == 1
    assert doc["materials"][0].get("alphaMode", "OPAQUE") == "OPAQUE"
    nodes = {n["name"]: np.array(n.get("translation", [0, 0, 0])) for n in doc["nodes"]}
    for pid, point in m.JOINTS.items():
        np.testing.assert_allclose(
            nodes["pivot_" + m.PARTS[pid]], point, atol=1e-6, rtol=0
        )
        assert np.max(abs(uv[part == pid, 1] - point[1])) < 1e-6
    probes = json.loads((ROOT / "joint_attachment_probes.json").read_text())
    trees = {pid: cKDTree(pos[part == pid]) for pid in set(part)}
    for probe in probes:
        assert trees[probe["cap_part"]].query(probe["cap_vertices"])[0].max() < 1e-6
        probe["planes"] = ConvexHull(probe["cap_vertices"]).equations
        for seam in probe["seams"]:
            assert trees[seam["part"]].query(seam["points"])[0].max() < 1e-6
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edge0 = np.linalg.norm(pos[edges[:, 1]] - pos[edges[:, 0]], axis=1)
    max_edge = max_joint = max_string = max_contact = max_grip = 0.0
    worst = {p["joint"]: -np.inf for p in probes}
    trajectory = []
    elevations = []
    max_hinge_error = max_wrist_angle = 0.0
    elbow_angles = {1: [], 6: []}
    hand_clearance = float("inf")
    forearm_clearance = float("inf")
    extraction_elbow_height = -float("inf")
    # Outside the convex envelope also means outside its enclosed torso surfaces.
    torso_hull = ConvexHull(pos[part == 19]).equations
    arrow_hulls = {
        pid: ConvexHull(pos[part == pid]).equations for pid in [1, 9, 11, 18, 19]
    }
    arrow_clearance = {pid: float("inf") for pid in arrow_hulls}
    free_arrow_grip_error = 0.0
    samples = []
    samples += [
        (phase, float(t), m.sample(phase, t))
        for phase in m.TABLES
        for t in np.linspace(0, 1, 1001)
    ]
    for phase, t, pose in samples:
        xf, (grips, top, bottom, nock) = m.transforms(pose)
        if phase == "reload" and pose[26] >= 0.5:
            if t * m.RELOAD_DURATION < 3.50:
                held_point = m.transform(m.JOINTS[16] + [0, 0, 0.04], xf[16])
                free_arrow_grip_error = max(
                    free_arrow_grip_error, float(np.linalg.norm(held_point - grips[0]))
                )
            shaft = m.JOINTS[16] + np.outer(np.linspace(0.08, 0.85, 129), [0, 0, 1])
            shaft = m.transform(shaft, xf[16])
            for pid, hull in arrow_hulls.items():
                r, tr = xf[pid]
                local = (shaft - tr) @ r
                distance = np.max(local @ hull[:, :3].T + hull[:, 3], axis=1).min()
                arrow_clearance[pid] = min(arrow_clearance[pid], float(distance))
        arrow_axis = xf[16][0] @ np.array([0, 0, 1])
        if phase == "raise":
            elevations.append(
                float(
                    np.degrees(
                        np.arctan2(arrow_axis[1], np.linalg.norm(arrow_axis[[0, 2]]))
                    )
                )
            )
        apply = lambda p, pid: m.transform(p, xf[pid])
        for a, b, c, g in m.CHAIN:
            max_joint = max(
                max_joint,
                np.linalg.norm(apply(m.JOINTS[a], a) - apply(m.JOINTS[a], 19)),
                np.linalg.norm(apply(m.JOINTS[b], a) - apply(m.JOINTS[b], b)),
                np.linalg.norm(apply(m.JOINTS[c], b) - apply(m.JOINTS[c], c)),
            )
        for a, b, c, g in m.CHAIN:
            s0, e0, w0 = [m.JOINTS[k] for k in [a, b, c]]
            s1, e1, w1, g1 = [
                apply(point, pid) for point, pid in [(s0, a), (e0, b), (w0, c), (g, c)]
            ]
            elbow_angles[a].append(
                float(
                    np.degrees(
                        np.arccos(
                            np.clip(np.dot(m.unit(e1 - s1), m.unit(w1 - e1)), -1, 1)
                        )
                    )
                )
            )
            wrist_angle = float(
                np.degrees(
                    np.arccos(np.clip(np.dot(m.unit(w1 - e1), m.unit(g1 - w1)), -1, 1))
                )
            )
            max_wrist_angle = max(max_wrist_angle, wrist_angle)
            axis = m.unit(np.cross(e0 - s0, w0 - e0))
            relative = xf[a][0].T @ xf[b][0]
            max_hinge_error = max(
                max_hinge_error, float(np.linalg.norm(relative @ axis - axis))
            )
        rt, tt = xf[19]
        hand_local = (apply(pos[part == 10], 10) - tt) @ rt
        outside = np.max(hand_local @ torso_hull[:, :3].T + torso_hull[:, 3], axis=1)
        hand_clearance = min(hand_clearance, float(outside.min()))
        forearm_local = (apply(pos[part == 9], 9) - tt) @ rt
        outside = np.max(forearm_local @ torso_hull[:, :3].T + torso_hull[:, 3], axis=1)
        forearm_clearance = min(forearm_clearance, float(outside.min()))
        if phase == "reload" and 1.60 <= t * m.RELOAD_DURATION <= 1.80:
            elbow_local = (apply(m.JOINTS[9], 9) - tt) @ rt
            extraction_elbow_height = max(
                extraction_elbow_height, float(elbow_local[1] - m.JOINTS[1][1])
            )
        max_grip = max(max_grip, np.linalg.norm(apply(m.GB, 12) - apply(m.GB, 8)))
        for end, pid, original in [(top, 13, m.TOP), (bottom, 17, m.BOTTOM)]:
            max_string = max(
                max_string,
                abs(np.linalg.norm(end - nock) - 0.87),
                np.linalg.norm(apply(original, pid) - end),
                np.linalg.norm(apply(m.NOCK, pid) - nock),
            )
        if phase == "raise" or (phase == "reload" and t * m.RELOAD_DURATION >= 3.80):
            contact = grips[0] - m.rx(-pose[28]) @ np.array([0, 0.015, 0])
            max_contact = max(max_contact, np.linalg.norm(contact - nock))
            trajectory.append(grips[0].tolist())
        for probe in probes:
            r, trans = xf[probe["cap_part"]]
            planes = probe["planes"]
            for seam in probe["seams"]:
                local = (apply(seam["points"], seam["part"]) - trans) @ r
                distance = np.max(local @ planes[:, :3].T + planes[:, 3])
                worst[probe["joint"]] = max(worst[probe["joint"]], float(distance))
        posed = m.deform(pos, part, pose)
        active = (
            np.ones(len(edges), dtype=bool)
            if pose[26] >= 0.5
            else part[edges[:, 0]] != 16
        )
        lengths = np.linalg.norm(posed[edges[:, 1]] - posed[edges[:, 0]], axis=1)
        max_edge = max(max_edge, float(np.max(abs(lengths[active] - edge0[active]))))
    boundary = np.max(np.ptp(m.TABLES["raise"][:, :12], axis=0))
    assert max_edge < 1e-12 and max_joint < 1e-12 and max_grip < 1e-12
    assert max_string < 1e-7
    assert max_contact < 0.001, max_contact
    assert boundary < 1e-12
    assert all(v < 0 for v in worst.values()), worst
    assert max_hinge_error < 1e-12
    assert max_wrist_angle < 0.1
    assert max(elbow_angles[1]) < 150
    assert hand_clearance > 0, hand_clearance
    assert forearm_clearance > 0, forearm_clearance
    assert extraction_elbow_height < -0.10, extraction_elbow_height
    assert abs(elevations[0]) < 0.001
    assert abs(elevations[-1] - 35) < 0.001
    assert min(np.diff(elevations)) >= -1e-9
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())["samples"]
        for phase, rows in baseline.items():
            np.testing.assert_array_equal(m.TABLES[phase], np.array(rows))
    later_reload_error = None
    if args.reload_baseline:
        previous = json.loads(args.reload_baseline.read_text())
        old_rows = np.array(previous["samples"]["reload"])
        old_arrows = np.array(previous["arrow_samples"]["reload"])
        first = int(
            np.ceil(
                args.reload_unchanged_from / m.RELOAD_DURATION * (len(old_rows) - 1)
            )
        )
        np.testing.assert_array_equal(m.ARROW_TABLE[first:], old_arrows[first:])
        later_reload_error = 0.0
        for i in range(first, len(old_rows)):
            old_pose = np.concatenate([old_rows[i], old_arrows[i]])
            new_pose = m.sample("reload", i / (len(old_rows) - 1))
            error = float(
                np.max(
                    abs(m.deform(pos, part, old_pose) - m.deform(pos, part, new_pose))
                )
            )
            later_reload_error = max(later_reload_error, error)
        assert later_reload_error < 1e-12, later_reload_error
    clip_boundaries = {}
    for before_phase, after_phase in [("release", "reload"), ("reload", "raise")]:
        before_mesh = m.deform(pos, part, m.sample(before_phase, 1))
        after_mesh = m.deform(pos, part, m.sample(after_phase, 0))
        error = float(np.max(abs(before_mesh - after_mesh)))
        clip_boundaries[f"{before_phase}_to_{after_phase}"] = error
        assert error < 1e-12, clip_boundaries
    assert min(arrow_clearance.values()) > 0.003, arrow_clearance
    assert free_arrow_grip_error < 0.001, free_arrow_grip_error
    before, after = m.sample("raise", 1), m.sample("release", 0)
    keep = np.arange(30) != 26
    boundary_error = float(np.max(abs(before[keep] - after[keep])))
    assert boundary_error < 1e-12
    assert before[26] == 1 and np.all(m.TABLES["release"][:, 26] == 0)
    assert abs(m.sample("release", 1)[25]) < 1e-12
    release_hand = [m.transforms(m.sample("release", t))[1][0][0] for t in [0, 1]]
    report = {
        "max_elbow_hinge_axis_error": max_hinge_error,
        "max_physical_wrist_bend_degrees": max_wrist_angle,
        "elbow_flexion_degrees": {k: [min(v), max(v)] for k, v in elbow_angles.items()},
        "minimum_draw_hand_clearance_from_convex_torso_m": hand_clearance,
        "minimum_draw_forearm_clearance_from_convex_torso_m": forearm_clearance,
        "max_extraction_elbow_height_relative_to_shoulder_m": extraction_elbow_height,
        "later_reload_position_error_m": later_reload_error,
        "aim_elevation_degrees": [elevations[0], elevations[-1]],
        "samples": len(samples),
        "triangles": {"L0": len(tri), "L1": None, "L2": None, "L3": None},
        "height_m": float(pos[part == 18, 1].max()),
        "part_vertices": {m.PARTS[k]: int(sum(part == k)) for k in sorted(m.PARTS)},
        "pivots_gltf_xyz": {
            name: p.tolist() for name, p in nodes.items() if name.startswith("pivot_")
        },
        "max_edge_length_change_m": max_edge,
        "max_joint_separation_m": float(max_joint),
        "max_bow_grip_error_m": float(max_grip),
        "max_string_attachment_or_length_error_m": float(max_string),
        "max_draw_finger_string_error_m": float(max_contact),
        "min_joint_attachment_inset_m": {k: -v for k, v in worst.items()},
        "drawing_arm_local_rotation_change": float(boundary),
        "raise_table_matches_baseline": True if args.baseline else None,
        "clips_matching_baseline": list(baseline) if args.baseline else [],
        "reload_boundary_position_errors_m": clip_boundaries,
        "reload_arrow_shaft_clearance_m": {
            m.PARTS[k]: v for k, v in arrow_clearance.items()
        },
        "max_free_arrow_grip_error_m": free_arrow_grip_error,
        "raise_release_boundary_error_excluding_visibility": boundary_error,
        "release_hand_displacement_m": float(
            np.linalg.norm(release_hand[1] - release_hand[0])
        ),
        "arrow_visibility": "Hidden at release; appears 1.65 seconds into reload during extraction; nocked at 3.8 seconds.",
        "scope": "L0 raise, 0.6-second release and 5.8-second hip-quiver reload.",
    }
    (ROOT / "motion_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "archer.raise.json").write_text(json.dumps(m.export(), indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
