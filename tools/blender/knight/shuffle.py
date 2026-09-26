"""Author square-stance side steps in glTF metres, +Y up and +Z forward.

Used by build_shuffle.py and review_shuffle.py with the shipped knight mesh.
"""

import math

import numpy as np


def smooth(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def ease(value):
    value = np.clip(value, 0.0, 1.0)
    return value ** 3 * (10.0 - 15.0 * value + 6.0 * value ** 2)


def rotate(points, angle, axis):
    points = np.asarray(points)
    result = points.copy()
    c, s = np.cos(angle), np.sin(angle)
    a, b = (1, 2) if axis == "x" else (0, 1)
    result[..., a] = c * points[..., a] - s * points[..., b]
    result[..., b] = s * points[..., a] + c * points[..., b]
    return result


class Shuffle:
    distance = 0.60
    duration = 0.75
    samples = 129
    columns = [
        "pelvis_x_m", "pelvis_y_m", "pelvis_z_m", "torso_roll_z_rad",
        "left_hip_roll_z_rad", "left_hip_pitch_x_rad", "left_knee_pitch_x_rad",
        "left_ankle_pitch_x_rad", "left_ankle_roll_z_rad",
        "right_hip_roll_z_rad", "right_hip_pitch_x_rad", "right_knee_pitch_x_rad",
        "right_ankle_pitch_x_rad", "right_ankle_roll_z_rad",
    ]

    def __init__(self, nodes):
        self.hips = np.array([nodes["pivot_leg_l"], nodes["pivot_leg_r"]])
        self.knees = self.hips.copy()
        self.ankles = self.hips.copy()
        self.knees[:, 1] *= 0.50
        self.ankles[:, 1] *= 0.10
        self.waist = np.array([0.0, float(np.mean(self.hips[:, 1])), 0.0])

    def targets(self, phase, direction):
        targets = self.ankles.copy()
        for leg in range(2):
            leading = leg == (0 if direction == 1 else 1)
            start, end = (0.04, 0.48) if leading else (0.52, 0.96)
            t = np.clip((phase - start) / (end - start), 0.0, 1.0)
            targets[leg, 0] += direction * self.distance * (ease(t) - phase)
            targets[leg, 1] += (0.050 if leading else 0.075) * math.sin(math.pi * t) ** 2
        return targets

    def author(self, phase, direction):
        phase = float(np.clip(phase, 0.0, 1.0))
        envelope = math.sin(math.pi * phase) ** 2
        pelvis = np.array([
            -direction * 0.018 * math.sin(2.0 * math.pi * phase) * envelope,
            -0.073 * envelope,
            0.0,
        ])
        torso_roll = direction * 0.022 * math.sin(2.0 * math.pi * phase) * envelope
        values = [*pelvis, torso_roll]
        for leg, ankle in enumerate(self.targets(phase, direction)):
            delta = ankle - (self.hips[leg] + pelvis)
            roll = math.atan2(delta[0], -delta[1])
            sagittal_height = math.hypot(delta[0], delta[1])
            reach = float(np.linalg.norm(delta))
            thigh = self.hips[leg, 1] - self.knees[leg, 1]
            shank = self.knees[leg, 1] - self.ankles[leg, 1]
            assert reach <= thigh + shank + 1e-9, (phase, leg, reach)
            lead = math.acos(np.clip((thigh ** 2 + reach ** 2 - shank ** 2) / (2 * thigh * reach), -1, 1))
            flex = math.pi - math.acos(np.clip((thigh ** 2 + shank ** 2 - reach ** 2) / (2 * thigh * shank), -1, 1))
            pitch = math.atan2(-delta[2], sagittal_height) - lead
            values.extend([roll, pitch, flex, -pitch - flex, -roll])
        if phase in (0.0, 1.0):
            values = [0.0] * len(values)
        return np.array(values)

    def deform(self, positions, parts, row):
        posed = positions.copy()
        for leg, part in enumerate([2, 3]):
            mask = parts == part
            points = positions[mask].copy()
            hip, knee, ankle = self.hips[leg], self.knees[leg], self.ankles[leg]
            roll, pitch, flex, ankle_pitch, ankle_roll = row[4 + 5 * leg:9 + 5 * leg]
            down = (hip[1] - points[:, 1]) / hip[1]
            knee_weight = smooth((down - 0.40) / 0.20)
            ankle_weight = smooth((down - 0.84) / 0.08)
            points = ankle + rotate(rotate(points - ankle, ankle_roll * ankle_weight, "z"), ankle_pitch * ankle_weight, "x")
            points = knee + rotate(points - knee, flex * knee_weight, "x")
            posed[mask] = hip + rotate(rotate(points - hip, pitch, "x"), roll, "z")
        body = (parts != 2) & (parts != 3)
        weight = smooth((positions[body, 1] - self.waist[1]) / 0.35)
        posed[body] = self.waist + rotate(posed[body] - self.waist, row[3] * weight, "z")
        return posed + row[:3]

    def joint_positions(self, row):
        result = []
        for leg in range(2):
            roll, pitch, flex = row[4 + 5 * leg:7 + 5 * leg]
            hip, knee, ankle = self.hips[leg], self.knees[leg], self.ankles[leg]
            knee_pose = hip + rotate(rotate(knee - hip, pitch, "x"), roll, "z")
            ankle_pose = knee_pose + rotate(rotate(ankle - knee, pitch + flex, "x"), roll, "z")
            result.append(np.array([hip, knee_pose, ankle_pose]) + row[:3])
        return np.array(result)

    def export(self, source_hash):
        return {
            "version": 1,
            "space": "glTF metres; +Y up, +Z forward, +X soldier left",
            "rotation": "right-handed radians; hip Rz(roll)*Rx(pitch), then knee Rx, then ankle Rx(pitch)*Rz(roll); column vectors",
            "source_glb_sha256": source_hash,
            "cycle_distance_m": self.distance,
            "cycle_duration_s": self.duration,
            "reference_speed_m_s": self.distance / self.duration,
            "interpolation": "linear between 129 uniform samples inclusive of duplicate endpoints; index = phase * 128",
            "phase": "accumulated absolute sideways ground distance in asset metres / cycle_distance_m; root translation is external",
            "parts": {"0": "body", "1": "arm_weapon", "2": "leg_l", "3": "leg_r", "5": "arm_shield", "8": "weapon"},
            "joints_gltf_xyz": {
                "left_hip": self.hips[0].tolist(), "right_hip": self.hips[1].tolist(),
                "left_knee": self.knees[0].tolist(), "right_knee": self.knees[1].tolist(),
                "left_ankle": self.ankles[0].tolist(), "right_ankle": self.ankles[1].tolist(),
                "waist": self.waist.tolist(),
            },
            "deformation": {
                "down": "(rest_hip_y - rest_vertex_y) / rest_hip_y",
                "knee_weight": "smoothstep(0.40, 0.60, down)",
                "ankle_weight": "smoothstep(0.84, 0.92, down)",
                "leg_order": "about rest ankle: Rz(ankle_roll*w), then Rx(ankle_pitch*w); about rest knee: Rx(knee_pitch*w); about rest hip: Rx(hip_pitch), then Rz(hip_roll)",
                "torso": "non-leg parts: rotate about waist by torso_roll * smoothstep(waist_y, waist_y+0.35, rest_vertex_y)",
                "pelvis": "add pelvis XYZ to every posed vertex after joint rotations; do not add walk bob or walk leg pose",
                "normals": "apply the same weighted rotations in the same order, without pivot translations, then normalize",
                "L3": "keep the all-body L3 static relative to its externally moving root; no shuffle dip",
                "scale": "multiply all metre values, pivots and cycle distance by the model-to-world scale; angles do not scale",
            },
            "columns": self.columns,
            "clips": {
                name: {
                    "direction_gltf_xyz": [direction, 0, 0],
                    "contact_phase_intervals": {
                        "left": [[0.0, 0.04], [0.48, 1.0]] if direction == 1 else [[0.0, 0.52], [0.96, 1.0]],
                        "right": [[0.0, 0.52], [0.96, 1.0]] if direction == 1 else [[0.0, 0.04], [0.48, 1.0]],
                    },
                    "samples": [self.author(p, direction).tolist() for p in np.linspace(0, 1, self.samples)],
                }
                for name, direction in [("left", 1), ("right", -1)]
            },
        }


def sample(table, direction, phase):
    rows = np.asarray(table["clips"][direction]["samples"])
    cursor = np.clip(phase, 0.0, 1.0) * (len(rows) - 1)
    index = min(int(cursor), len(rows) - 2)
    return rows[index] * (1 - (cursor - index)) + rows[index + 1] * (cursor - index)
