"""Author a right-handed diagonal sword cut from a rig's measured arm markers.

Uses glTF metres, +Y up and +Z forward. Only NumPy is required.
"""
import math
import numpy as np


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def rotation(axis, angle):
    x, y, z = unit(axis)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def align(a, b):
    a, b = unit(a), unit(b)
    cross = np.cross(a, b)
    return (
        rotation(cross, math.acos(np.clip(np.dot(a, b), -1, 1)))
        if np.linalg.norm(cross) > 1e-9
        else np.eye(3)
    )


def frame(direction, hinge):
    u, h = unit(direction), unit(hinge)
    return np.column_stack([u, h, np.cross(u, h)])


def quaternion(r):
    # The largest eigenvector gives XYZW without a trace singularity at pi.
    k = np.array(
        [
            [
                r[0, 0] - r[1, 1] - r[2, 2],
                r[1, 0] + r[0, 1],
                r[2, 0] + r[0, 2],
                r[2, 1] - r[1, 2],
            ],
            [
                r[1, 0] + r[0, 1],
                r[1, 1] - r[0, 0] - r[2, 2],
                r[2, 1] + r[1, 2],
                r[0, 2] - r[2, 0],
            ],
            [
                r[2, 0] + r[0, 2],
                r[2, 1] + r[1, 2],
                r[2, 2] - r[0, 0] - r[1, 1],
                r[1, 0] - r[0, 1],
            ],
            [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], np.trace(r)],
        ]
    )
    q = np.linalg.eigh(k)[1][:, -1]
    return q if q[3] >= 0 else -q


def matrix(q):
    x, y, z, w = unit(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def curve(keys, seconds, stops=(), reach_limit=None):
    times = np.array([k[0] for k in keys])
    values = np.array([k[1:] for k in keys])
    i = min(max(np.searchsorted(times, seconds, side="right") - 1, 0), len(keys) - 2)
    dt = times[i + 1] - times[i]
    t = np.clip((seconds - times[i]) / dt, 0, 1)
    slopes = np.zeros_like(values)
    slopes[1:-1] = (values[2:] - values[:-2]) / (times[2:] - times[:-2])[:, None]
    for stop in stops:
        slopes[np.isclose(times, stop)] = 0
    if reach_limit is not None:
        # Bound both Bezier handles at each key so the entire hand curve stays
        # reachable, with a shared tangent on either side of the key.
        for j in range(1, len(keys) - 1):
            scale = 1.0
            for dt_control in [times[j - 1] - times[j], times[j + 1] - times[j]]:
                delta = slopes[j] * dt_control / 3
                a = np.dot(delta, delta)
                b = 2 * np.dot(values[j], delta)
                c = np.dot(values[j], values[j]) - reach_limit ** 2
                if a > 1e-12:
                    scale = min(scale, (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a))
            slopes[j] *= scale
    return (
        (2 * t ** 3 - 3 * t * t + 1) * values[i]
        + (t ** 3 - 2 * t * t + t) * dt * slopes[i]
        + (-2 * t ** 3 + 3 * t * t) * values[i + 1]
        + (t ** 3 - t * t) * dt * slopes[i + 1]
    )


class SwordSlash:
    duration = 0.65
    contact_time = 0.36
    hold_start = 0.19
    hold_end = 0.24
    finish_roll_start = 0.33

    def __init__(self, nodes, arm_part=1, weapon_part=8, shield_part=5):
        self.arm_part, self.weapon_part, self.shield_part = (
            arm_part,
            weapon_part,
            shield_part,
        )
        self.shoulder = np.array(nodes["pivot_arm_weapon"], dtype=float)
        self.elbow = np.array(nodes["joint_elbow"], dtype=float)
        self.grip = np.array(nodes["pivot_weapon"], dtype=float)
        self.shield = np.array(nodes["pivot_arm_shield"], dtype=float)
        self.upper, self.fore = self.elbow - self.shoulder, self.grip - self.elbow
        self.l1, self.l2 = np.linalg.norm(self.upper), np.linalg.norm(self.fore)
        self.reach = self.l1 + self.l2
        self.hinge = unit(np.cross(self.upper, self.fore))
        self.waist_y = 0.72 * self.shoulder[1]
        self.table = np.array(
            [self.author(t) for t in np.linspace(0, self.duration, 131)]
        )
        for i in range(1, len(self.table)):
            for j in range(0, 12, 4):
                if np.dot(self.table[i - 1, j : j + 4], self.table[i, j : j + 4]) < 0:
                    self.table[i, j : j + 4] *= -1

    def solve_arm(self, hand):
        target = self.shoulder + np.array(hand) * self.reach
        delta = target - self.shoulder
        distance = np.linalg.norm(delta)
        assert abs(self.l1 - self.l2) < distance < self.reach, distance
        direction = delta / distance
        pole = unit([-1, -0.65, -0.10])
        side = unit(pole - direction * np.dot(pole, direction))
        along = (self.l1 ** 2 - self.l2 ** 2 + distance ** 2) / (2 * distance)
        elbow = (
            self.shoulder
            + along * direction
            + math.sqrt(self.l1 ** 2 - along ** 2) * side
        )
        u, f = unit(elbow - self.shoulder), unit(target - elbow)
        hinge = unit(np.cross(u, f))
        return (
            frame(u, hinge) @ frame(self.upper, self.hinge).T,
            frame(f, hinge) @ frame(self.fore, self.hinge).T,
        )

    @staticmethod
    def mix_rotation(a, b, amount):
        qa, qb = quaternion(a), quaternion(b)
        if np.dot(qa, qb) < 0:
            qb = -qb
        return matrix(qa * (1 - amount) + qb * amount)

    def cut_kinematics(self, phase):
        upper_a, fore_a = self.solve_arm([-0.42, 0.60, 0.18])
        upper_b, fore_b = self.solve_arm([0.45, -0.70, 0.48])
        upper = self.mix_rotation(upper_a, upper_b, phase)
        elbow = self.mix_rotation(upper_a.T @ fore_a, upper_b.T @ fore_b, phase)
        fore = upper @ elbow
        grip = self.shoulder + upper @ self.upper + fore @ self.fore
        start, end = unit([0.42, 0.84, 0.34]), unit([0.42, -0.79, 0.44])
        axis = unit(np.cross(start, end))
        angle = math.acos(np.clip(np.dot(start, end), -1, 1))
        blade = rotation(axis, phase * angle) @ start
        return upper, fore, grip, blade

    def cut_pose(self, phase):
        upper, fore, grip, blade = self.cut_kinematics(phase)
        before, after = [
            self.cut_kinematics(p)
            for p in [max(phase - 0.001, 0), min(phase + 0.001, 1)]
        ]
        velocity = (after[2] + self.reach * after[3]) - (
            before[2] + self.reach * before[3]
        )
        edge = unit(velocity - blade * np.dot(velocity, blade))
        # Blade width is local X and its face normal is local Y. Both cutting
        # edges lie in the swept plane; the broad face does not slap the path.
        weapon = np.column_stack([edge, np.cross(blade, edge), blade])
        return upper, fore, weapon

    def author(self, seconds):
        clip_seconds = seconds
        # Add anticipation without shortening the active cut or its recovery.
        if self.hold_start < seconds <= self.hold_end:
            seconds = self.hold_start
        elif seconds > self.hold_end:
            seconds -= self.hold_end - self.hold_start
        # Hand targets are offsets from the shoulder in total arm lengths.
        hand = curve(
            [
                (0.00, -0.34, -0.70, 0.32),
                (0.12, -0.44, 0.30, 0.34),
                (0.19, -0.42, 0.60, 0.18),
                (0.27, 0.24, -0.54, 0.68),
                (0.35, 0.45, -0.70, 0.48),
                (0.47, -0.08, -0.75, 0.46),
                (0.60, -0.34, -0.70, 0.32),
            ],
            seconds,
            stops=(self.hold_start, 0.35),
            reach_limit=0.985,
        )
        blade = unit(
            curve(
                [
                    (0.00, -0.10, -0.88, 0.46),
                    (0.12, 0.30, 0.84, 0.45),
                    (0.19, 0.42, 0.84, 0.34),
                    (0.27, 0.46, -0.67, 0.58),
                    (0.35, 0.42, -0.79, 0.44),
                    (0.47, 0.08, -0.91, 0.40),
                    (0.60, -0.10, -0.88, 0.46),
                ],
                seconds,
                stops=(self.hold_start, 0.35),
            )
        )
        r1, r2 = self.solve_arm(hand)
        weapon_r = align(r2 @ [0, 0, 1], blade) @ r2
        if 0.19 <= seconds <= 0.35:
            r1, r2, weapon_r = self.cut_pose(smooth((seconds - 0.19) / 0.16))
        elif seconds < 0.19:
            weapon_r = self.mix_rotation(
                weapon_r, self.cut_pose(0)[2], smooth((seconds - 0.12) / 0.07)
            )
        else:
            weapon_r = self.mix_rotation(
                self.cut_pose(1)[2], weapon_r, smooth((seconds - 0.35) / 0.12)
            )
        # Turn the hand and blade about the grip's longitudinal axis during
        # follow-through, then release the roll as the arm recovers.
        finish_roll = (
            np.radians(52)
            * smooth((clip_seconds - self.finish_roll_start) / 0.04)
            * (1 - smooth((clip_seconds - 0.41) / 0.10))
        )
        weapon_r = weapon_r @ rotation([0, 0, 1], finish_roll)
        torso, shield = np.radians(
            curve(
                [
                    (0.0, 0, 0),
                    (0.19, -9, 8),
                    (0.32, 9, 22),
                    (0.43, 5, 14),
                    (0.60, 0, 0),
                ],
                seconds,
                stops=(self.hold_start, 0.35),
            )
        )
        return np.concatenate(
            [
                quaternion(r1),
                quaternion(r1.T @ r2),
                quaternion(r2.T @ weapon_r),
                [torso, shield],
            ]
        )

    def sample(self, seconds):
        at = np.clip(seconds / self.duration, 0, 1) * (len(self.table) - 1)
        i = min(int(at), len(self.table) - 2)
        row = self.table[i] * (1 - (at - i)) + self.table[i + 1] * (at - i)
        for j in range(0, 12, 4):
            row[j : j + 4] = unit(row[j : j + 4])
        return row

    def transforms(self, pose):
        r1 = matrix(pose[:4])
        r2 = r1 @ matrix(pose[4:8])
        r3 = r2 @ matrix(pose[8:12])
        elbow = self.shoulder + r1 @ self.upper
        grip = elbow + r2 @ self.fore
        return r1, r2, r3, elbow, grip

    def deform(self, positions, parts, pose):
        points = np.asarray(positions)
        result = points.copy()
        r1, r2, r3, elbow, grip = self.transforms(pose)
        arm = parts == self.arm_part
        p = points[arm]
        past = ((p - self.elbow) @ self.upper) / (self.upper @ self.upper)
        bend = smooth((past + 0.15) / 0.30)
        upper = self.shoulder + (p - self.shoulder) @ r1.T
        fore = elbow + (p - self.elbow) @ r2.T
        moved = upper * (1 - bend[:, None]) + fore * bend[:, None]
        distal = ((p - self.elbow) @ self.fore) / (self.fore @ self.fore)
        wrist = smooth((distal - 0.58) / 0.30)
        hand = grip + (p - self.grip) @ r3.T
        result[arm] = moved * (1 - wrist[:, None]) + hand * wrist[:, None]
        weapon = parts == self.weapon_part
        result[weapon] = grip + (points[weapon] - self.grip) @ r3.T
        shield = parts == self.shield_part
        rs = rotation([0, 1, 0], pose[13])
        result[shield] = self.shield + (points[shield] - self.shield) @ rs.T
        # Feet and hips stay planted; mail and cloth distribute a small twist
        # between belt and shoulders instead of rotating the waist alone.
        weight = smooth(
            (points[:, 1] - self.waist_y) / (self.shoulder[1] - self.waist_y)
        )
        weight[np.isin(parts, [self.arm_part, self.weapon_part, self.shield_part])] = 1
        weight[np.isin(parts, [2, 3])] = 0
        angle = pose[12] * weight
        x, z = result[:, 0].copy(), result[:, 2].copy()
        result[:, 0] = x * np.cos(angle) + z * np.sin(angle)
        result[:, 2] = -x * np.sin(angle) + z * np.cos(angle)
        return result

    def export(self):
        return {
            "version": 1,
            "clip": "diagonal_slash",
            "duration": self.duration,
            "contact_time": self.contact_time,
            "raised_hold_seconds": [self.hold_start, self.hold_end],
            "space": "glTF metres, +Y up, +Z forward",
            "channels": [
                "shoulder_xyzw",
                "elbow_xyzw",
                "grip_xyzw",
                "torso_yaw",
                "shield_yaw",
            ],
            "joints": {
                "shoulder": self.shoulder.tolist(),
                "elbow": self.elbow.tolist(),
                "grip": self.grip.tolist(),
                "shield": self.shield.tolist(),
            },
            "parts": {
                "arm": self.arm_part,
                "weapon": self.weapon_part,
                "shield": self.shield_part,
            },
            "waist_y": self.waist_y,
            "samples": self.table.tolist(),
            "deformation": "Connected sleeve elbow and wrist blends; distributed torso yaw. See sword_slash.py.",
        }
