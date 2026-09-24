"""Mirror the shader's standing sword attacks for asset-side motion review.

Coordinates are exported GLB XYZ in metres. Runtime uses the existing shader,
not this module or a new clip. Locomotion, charge and shieldwall are excluded.
"""
import math
import numpy as np

SCALE = 1 / 1.8
HALF_HEIGHT = 0.5


def smooth(low, high, value):
    t = np.clip((value - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def turn(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack(
        [v[..., 0] * c + v[..., 1] * s, -v[..., 0] * s + v[..., 1] * c], axis=-1
    )


def angle(v):
    return math.atan2(v[1], -v[0])


def direction(a):
    return np.array([-math.cos(a), math.sin(a)])


def wrap(a):
    return a - 2 * math.pi * math.floor((a + math.pi) / (2 * math.pi))


def local(points):
    out = np.asarray(points, dtype=float) * SCALE
    out[..., 1] -= HALF_HEIGHT
    return out


class SwordRig:
    def __init__(self, nodes, positions, parts):
        self.shoulder = local(nodes["pivot_arm_weapon"])[1:]
        self.elbow = local(nodes["joint_elbow"])[1:]
        self.grip = local(nodes["pivot_weapon"])[1:]
        weapon = local(positions[parts == 8])[:, 1:] - self.grip
        tip = weapon[np.argmax(np.sum(weapon * weapon, axis=1))]
        self.tip = tip / np.linalg.norm(tip)
        self.upper = self.elbow - self.shoulder
        self.fore = self.grip - self.elbow
        self.l1, self.l2 = np.linalg.norm(self.upper), np.linalg.norm(self.fore)
        self.reach = self.l1 + self.l2

    def pose(self, style, phase, following=False):
        wound = phase * phase
        settle = 1 - smooth(0.25, 1.0, phase)
        raised = settle if following else smooth(0, 0.55, wound)
        chop = settle if following else smooth(0.55, 1, wound)
        lunge = settle if following else wound
        rest = angle(self.tip)
        goal, aim, whole_turn = self.grip.copy(), rest, 0.0
        if style == "stab":
            goal += (
                direction(rest) * self.reach * (-0.4 * raised * (1 - chop) + 0.7 * chop)
            )
            aim += 0.15 * raised - 0.1 * chop
        else:
            assert style == "overhead"
            whole_turn = 1.9 * raised - 2.5 * chop
        to = goal - self.shoulder
        distance = np.clip(
            np.linalg.norm(to), abs(self.l1 - self.l2) + 1e-4, self.reach - 1e-4
        )
        lead = math.acos(
            np.clip(
                (self.l1 * self.l1 + distance * distance - self.l2 * self.l2)
                / (2 * self.l1 * distance),
                -1,
                1,
            )
        )
        toward = angle(to)
        upper = toward - lead
        fore = angle(distance * direction(toward) - self.l1 * direction(upper))
        shoulder = wrap(upper - angle(self.upper))
        fore_turn = wrap(fore - angle(self.fore))
        return (
            shoulder + whole_turn,
            wrap(fore_turn - shoulder),
            wrap(aim - rest - fore_turn),
            float(raised),
            float(chop),
            float(lunge),
        )

    def deform(self, positions, parts, pose):
        points = local(positions)
        result = points.copy()
        shoulder, elbow_turn, wrist, raised, chop, lunge = pose
        selected = np.isin(parts, [1, 8])
        weapon = parts[selected] == 8
        yz = points[selected, 1:]
        past = ((yz - self.elbow) @ self.upper) / (self.upper @ self.upper)
        weight = np.where(weapon, 1.0, smooth(-0.15, 0.15, past))
        yz = self.shoulder + turn(yz - self.shoulder, shoulder)
        moved_elbow = self.shoulder + turn(self.upper, shoulder)
        yz = moved_elbow + turn(yz - moved_elbow, elbow_turn * weight)
        moved_grip = moved_elbow + turn(self.fore, shoulder + elbow_turn)
        yz[weapon] = moved_grip + turn(yz[weapon] - moved_grip, wrist)
        result[selected, 1:] = yz
        shield = parts == 5
        shield_joint = np.array([1.435 * SCALE - HALF_HEIGHT, 0])
        result[shield, 1:] = shield_joint + turn(
            points[shield, 1:] - shield_joint, 0.12 * raised
        )
        result[:, 2] += (
            (0.30 * lunge + 0.25 * chop) * np.clip(result[:, 1] + 0.5, 0, 1.5) * 0.3
        )
        result[:, 1] += HALF_HEIGHT
        return result / SCALE


def timeline(frame):
    """Two attacks at half speed: 0.3 s windup, 0.6 s follow-through."""
    style = "stab" if frame < 90 else "overhead"
    seconds = (frame % 90) / 60
    if seconds < 0.3:
        return style, 0.0, False
    if seconds < 0.6:
        return style, (seconds - 0.3) / 0.3, False
    return style, min((seconds - 0.6) / 0.6, 1.0), True
