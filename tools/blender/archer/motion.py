"""Author two rigid arm chains and a strung bow in glTF metres.

Run with Python to export uniformly sampled local joint rotations. Runtime
uses forward kinematics only. Bow limbs rotate about the grip shoulders;
the two string halves retain their lengths as the limbs bend.
"""
import json
import math
from pathlib import Path
import numpy as np

PARTS = {
    0: "body",
    1: "arm_weapon",
    2: "leg_l",
    3: "leg_r",
    6: "arm_bow",
    8: "weapon",
    9: "forearm_draw",
    10: "hand_draw",
    11: "forearm_bow",
    12: "hand_bow",
    13: "bow_string",
    14: "bow_upper",
    15: "bow_lower",
    16: "arrow",
    17: "bow_string_lower",
    18: "head",
    19: "torso",
}
JOINTS = {
    1: [-0.205, 1.435, 0],
    9: [-0.29, 1.155, -0.01],
    10: [-0.35, 0.925, 0.13],
    6: [0.205, 1.435, 0],
    11: [0.29, 1.155, -0.01],
    12: [0.35, 0.925, 0.13],
    8: [0.365, 0.900, 0.19],
    14: [0.365, 0.94, 0.19],
    15: [0.365, 0.86, 0.19],
    13: [0.365, 0.900, 0],
    16: [0.365, 0.900, 0],
    2: [0.115, 0.91, 0],
    3: [-0.115, 0.91, 0],
}
JOINTS[17] = JOINTS[13][:]
JOINTS[18] = [0, 1.485, 0]
JOINTS[19] = [0, 1.035, 0]
JOINTS = {k: np.array(v, dtype=float) for k, v in JOINTS.items()}
GD = np.array([-0.365, 0.900, 0.19])
GB = JOINTS[8]
TOP = np.array([GB[0], GB[1] + 0.87, 0.0])
BOTTOM = np.array([GB[0], GB[1] - 0.87, 0.0])
NOCK = JOINTS[13]
CHAIN = [(1, 9, 10, GD), (6, 11, 12, GB)]


def unit(v):
    return np.asarray(v) / np.linalg.norm(v)


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def euler(v):
    x, y, z = v
    cx, sx, cy, sy, cz, sz = (
        math.cos(x),
        math.sin(x),
        math.cos(y),
        math.sin(y),
        math.cos(z),
        math.sin(z),
    )
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ]
    )


def angles(r):
    return np.array(
        [
            math.atan2(r[2, 1], r[2, 2]),
            math.asin(np.clip(-r[2, 0], -1, 1)),
            math.atan2(r[1, 0], r[0, 0]),
        ]
    )


def align(a, b):
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    assert c > -0.99999
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


BODY_YAW = math.radians(-65)
BODY_ROTATION = euler([0, BODY_YAW, 0])
ANCHOR = np.array([-0.095, 1.550, -0.015])
NOCK_WORLD = ANCHOR - np.array([0, 0.015, 0])
DRAW_LENGTH = 0.750
RELOAD_DURATION = 5.8
PICKUP_VISIBLE_SECONDS = 1.65


def reload_state(seconds):
    released_bow = NOCK_WORLD + rx(-math.radians(8)) @ np.array(
        [-0.020, 0, DRAW_LENGTH]
    )
    low_bow = np.array([-0.180, 1.300, 0.450])
    yaw = math.atan2(-0.020, DRAW_LENGTH)
    shot_axis = euler([0, yaw, 0]) @ np.array([0, 0, 1])
    low_nock = low_bow - shot_axis * 0.19
    low_hand = low_nock + [0, 0.015, 0]
    lower = smooth(seconds / 0.95)
    torso_pitch = math.radians(23) * (1 - lower)
    bow_pitch = math.radians(8) * (1 - lower)
    bow = released_bow * (1 - lower) + low_bow * lower
    hand = (
        BODY_ROTATION @ draw_grip(early_reload_angles(seconds))
        if seconds <= 3.80
        else low_hand
    )
    rotate = smooth((seconds - 1.80) / 1.20)
    tilt = math.radians(12) * smooth((seconds - 2.22) / 0.15)
    tilt *= 1 - smooth((seconds - 2.70) / 0.25)
    tilt += (
        math.radians(24)
        * smooth((seconds - 1.65) / 0.15)
        * (1 - smooth((seconds - 2.30) / 0.30))
    )
    outward_tilt = math.radians(25) * smooth((seconds - 1.80) / 0.25)
    outward_tilt *= 1 - smooth((seconds - 2.70) / 0.25)
    down = BODY_ROTATION @ np.array(
        [
            -math.sin(outward_tilt),
            -math.cos(outward_tilt) * math.cos(tilt),
            -math.cos(outward_tilt) * math.sin(tilt),
        ]
    )
    toward_bow = smooth((seconds - 2.70) / 1.0)
    forward_axis = unit(
        (1 - toward_bow) * (BODY_ROTATION @ np.array([0, 0, 1]))
        + toward_bow * shot_axis
    )
    direction = unit(
        down * math.cos(rotate * math.pi / 2)
        + forward_axis * math.sin(rotate * math.pi / 2)
    )
    arrow_nock = hand - direction * 0.04
    attachment = smooth((seconds - 3.50) / 0.30)
    depth = 0.19
    if seconds >= 3.80:
        draw = smooth((seconds - 3.80) / 2.0)
        bow = low_bow * (1 - draw) + (NOCK_WORLD + [-0.020, 0, DRAW_LENGTH]) * draw
        hand = low_hand * (1 - draw) + ANCHOR * draw
        clearance = BODY_ROTATION @ np.array([0, 0, 0.055 * 4 * draw * (1 - draw)])
        bow += clearance
        hand += clearance
        depth = np.linalg.norm(bow - (hand - [0, 0.015, 0]))
    bend = 0.0 if seconds <= 3.80 else limb_angle(depth)
    return hand, bow, torso_pitch, bow_pitch, bend, arrow_nock, direction, attachment


def bone_frame(direction, hinge):
    direction, hinge = unit(direction), unit(hinge)
    return np.column_stack([direction, hinge, np.cross(direction, hinge)])


def reload_draw_pole(seconds):
    # Keep the elbow below the quiver grip while extracting the arrow. A fixed
    # outward pole rolls the humerus as the hand crosses shoulder height.
    amount = smooth((seconds - 1.00) / 0.60)
    amount *= 1 - smooth((seconds - 1.80) / 0.45)
    if amount == 0:
        return np.array([-0.3, -0.1, -1])
    down = BODY_ROTATION @ unit([-0.6, -1.0, 0.1])
    outward = unit([-0.3, -0.1, -1])
    return down * amount + outward * (1 - amount)


def solve(chain, contact_world, pole_world):
    # The elbow is one hinge. Orient the shoulder's whole bend plane first,
    # then flex the forearm about the same anatomical axis. Keep the wrist
    # straight instead of independently pointing the hand at an IK target.
    a, b, c, grip = chain
    s, e, w = [JOINTS[k] for k in chain[:3]]
    contact = BODY_ROTATION.T @ contact_world
    pole = BODY_ROTATION.T @ np.array(pole_world)
    upper, fore, palm = e - s, w - e, grip - w
    l1, lf, lh = [np.linalg.norm(v) for v in [upper, fore, palm]]
    l2 = lf + lh
    delta = contact - s
    distance = np.linalg.norm(delta)
    assert abs(l1 - l2) < distance < l1 + l2
    direction = delta / distance
    side = unit(pole - direction * np.dot(pole, direction))
    along = (l1 * l1 - l2 * l2 + distance * distance) / (2 * distance)
    ep = s + along * direction + math.sqrt(l1 * l1 - along * along) * side
    fore_direction = unit(contact - ep)
    wp = ep + lf * fore_direction
    h0 = unit(np.cross(upper, fore))
    h = unit(np.cross(ep - s, wp - ep))
    r1 = bone_frame(ep - s, h) @ bone_frame(upper, h0).T
    r2 = bone_frame(wp - ep, h) @ bone_frame(fore, h0).T
    r3 = align(r2 @ palm, fore_direction) @ r2
    return np.concatenate([angles(r1), angles(r1.T @ r2), angles(r2.T @ r3)])


def draw_grip(joints):
    s, e, w = [JOINTS[k] for k in [1, 9, 10]]
    r1 = euler(joints[:3])
    r2 = r1 @ euler(joints[3:6])
    r3 = r2 @ euler(joints[6:9])
    return s + r1 @ (e - s) + r2 @ (w - e) + r3 @ (GD - w)


def early_reload_angles(seconds):
    # Swing moves the upper arm; the signed roll carries the forearm forward.
    # Interpolate them separately so a combined quaternion rotation cannot
    # raise the elbow while rolling the hand behind the shoulder.
    if seconds > 1.80:
        return forward_transfer_angles(seconds)
    keys = [
        (0.0, ANCHOR + [-0.040, 0.020, -0.025], [-0.3, -0.1, -1], 0),
        (1.35, BODY_ROTATION @ [-0.245, 1.225, -0.184], reload_draw_pole(1.35), 1),
        (
            1.80,
            BODY_ROTATION @ [-0.500, 1.570, 0.020],
            BODY_ROTATION @ [-1, -0.5, 0.2],
            0,
        ),
    ]
    upper = JOINTS[9] - JOINTS[1]
    fore = JOINTS[10] - JOINTS[9]
    palm = GD - JOINTS[10]
    h0 = unit(np.cross(upper, fore))

    def components(key):
        _, contact, pole, turns = key
        a = solve(CHAIN[0], contact, pole)
        r1 = euler(a[:3])
        r2 = r1 @ euler(a[3:6])
        u, f = r1 @ unit(upper), r2 @ unit(fore)
        flex = math.acos(np.clip(np.dot(u, f), -1, 1))
        bend = unit(f - u * np.dot(u, f))
        reference = unit(np.array([0, 0, 1]) - u * u[2])
        roll = math.atan2(np.dot(u, np.cross(reference, bend)), np.dot(reference, bend))
        return u, flex, roll + turns * 2 * math.pi

    for a, b in zip(keys, keys[1:]):
        if seconds <= b[0]:
            ua, fa, ra = components(a)
            ub, fb, rb = components(b)
            t = smooth((seconds - a[0]) / (b[0] - a[0]))
            u = unit(ua * (1 - t) + ub * t)
            flex, roll = fa * (1 - t) + fb * t, ra * (1 - t) + rb * t
            reference = unit(np.array([0, 0, 1]) - u * u[2])
            bend = reference * math.cos(roll) + np.cross(u, reference) * math.sin(roll)
            f = u * math.cos(flex) + bend * math.sin(flex)
            h = unit(np.cross(u, f))
            r1 = bone_frame(u, h) @ bone_frame(upper, h0).T
            r2 = bone_frame(f, h) @ bone_frame(fore, h0).T
            r3 = align(r2 @ palm, f) @ r2
            return np.concatenate([angles(r1), angles(r1.T @ r2), angles(r2.T @ r3)])
    return solve(CHAIN[0], keys[-1][1], keys[-1][2])


def forward_transfer_angles(seconds):
    # Turn toward the string as soon as extraction ends. The curve passes in
    # front of the chest without extending the arm farther out to the side.
    t = smooth((seconds - 1.80) / 2.0)
    start = np.array([-0.500, 1.570, 0.020])
    shot_axis = euler([0, math.atan2(-0.020, DRAW_LENGTH), 0]) @ np.array([0, 0, 1])
    end = BODY_ROTATION.T @ (np.array([-0.180, 1.315, 0.450]) - shot_axis * 0.19)
    control_a = np.array([-0.440, 1.550, 0.380])
    control_b = np.array([-0.100, 1.360, 0.380])
    hand = (
        (1 - t) ** 3 * start
        + 3 * (1 - t) ** 2 * t * control_a
        + 3 * (1 - t) * t ** 2 * control_b
        + t ** 3 * end
    )
    initial = solve(CHAIN[0], BODY_ROTATION @ start, BODY_ROTATION @ [-1, -0.5, 0.2])
    upper = euler(initial[:3]) @ (JOINTS[9] - JOINTS[1])
    pole = (1 - t) * (BODY_ROTATION @ unit(upper)) + t * unit([-0.3, -0.1, -1])
    return solve(CHAIN[0], BODY_ROTATION @ hand, pole)


def limb_angle(depth):
    target = GB + np.array([0, 0, -depth])
    pivot = JOINTS[14]
    lo, hi = 0.0, 1.0
    for _ in range(48):
        a = (lo + hi) / 2
        tip = pivot + rx(-a) @ (TOP - pivot)
        if np.linalg.norm(tip - target) > 0.87:
            lo = a
        else:
            hi = a
    return (lo + hi) / 2


def authored(phase, t):
    if phase == "reload":
        if t <= 0:
            return authored("release", 1)
        if t >= 1:
            return authored("raise", 0)
        seconds = t * RELOAD_DURATION
        hand, bow, torso, pitch, bend, _, _, _ = reload_state(seconds)
        joints = np.concatenate(
            [
                early_reload_angles(seconds)
                if seconds <= 3.80
                else solve(CHAIN[0], hand, [-0.3, -0.1, -1]),
                solve(CHAIN[1], bow, [1, -0.4, 0]),
            ]
        )
        yaw = math.atan2(-0.020, DRAW_LENGTH)
        visible = float(seconds >= PICKUP_VISIBLE_SECONDS)
        return np.concatenate([joints, [yaw, bend, visible, BODY_YAW, torso, pitch]])
    if phase == "release":
        seconds = np.clip(t, 0, 1) * 0.6
        pose = authored("raise", 1).copy()
        # Keep the hand beside the jaw as the elbow follows through backward.
        hand = ANCHOR + smooth(seconds / 0.22) * np.array([-0.040, 0.020, -0.025])
        settle = smooth((seconds - 0.18) / 0.42)
        pose[22] -= math.radians(2) * settle
        pose[23] -= math.radians(2) * settle
        bow = NOCK_WORLD + rx(-pose[23]) @ np.array([-0.020, 0, DRAW_LENGTH])
        pose[:9] = solve(CHAIN[0], hand, [-0.3, -0.1, -1])
        pose[9:18] = solve(CHAIN[1], bow, [1, -0.4, 0])
        # The string returns to its braced position while the bow arm holds aim.
        pose[19] *= 1 - smooth(seconds / 0.075)
        pose[20] = 0.0
        return pose
    assert phase == "raise"
    amount = smooth(t)
    torso_pitch = math.radians(25) * amount
    bow_pitch = math.radians(10) * amount
    # The hand anchor and drawing-arm joints remain fixed in the torso frame.
    local_aim = rx(-bow_pitch)
    bow = NOCK_WORLD + local_aim @ np.array([-0.020, 0, DRAW_LENGTH])
    joints = np.concatenate(
        [
            solve(CHAIN[0], ANCHOR, [-0.3, -0.1, -1]),
            solve(CHAIN[1], bow, [1, -0.4, 0]),
        ]
    )
    yaw = math.atan2(-0.020, DRAW_LENGTH)
    bend = limb_angle(math.hypot(DRAW_LENGTH, 0.020))
    return np.concatenate([joints, [yaw, bend, 1.0, BODY_YAW, torso_pitch, bow_pitch]])


def quaternion(matrix):
    # Eigenvector form avoids a special case at 180 degrees.
    r = matrix
    k = (
        np.array(
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
                [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], r.trace()],
            ]
        )
        / 3
    )
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    return q if q[3] >= 0 else -q


def qmatrix(q):
    x, y, z, w = unit(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def make_table(phase):
    rows = []
    intervals = 256 if phase == "reload" else 64
    for i in range(intervals + 1):
        pose = authored(phase, i / intervals)
        qs = [quaternion(euler(pose[j : j + 3])) for j in range(0, 18, 3)]
        if rows:
            qs = [
                q if np.dot(q, rows[-1][j * 4 : j * 4 + 4]) >= 0 else -q
                for j, q in enumerate(qs)
            ]
        rows.append(np.concatenate(qs + [pose[18:]]))
    return np.array(rows)


TABLES = {name: make_table(name) for name in ["raise", "release", "reload"]}


def make_arrow_table():
    rows = []
    for seconds in np.linspace(0, RELOAD_DURATION, len(TABLES["reload"])):
        *_, nock, direction, attachment = reload_state(seconds)
        q = quaternion(BODY_ROTATION.T @ align([0, 0, 1], direction))
        if rows and np.dot(q, rows[-1][3:7]) < 0:
            q = -q
        rows.append(np.concatenate([BODY_ROTATION.T @ nock, q, [attachment]]))
    return np.array(rows)


ARROW_TABLE = make_arrow_table()
CARRY = np.array([0.0, 0.0, 0.0, 1.0] * 6 + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def sample(phase, p):
    table = TABLES[phase]
    intervals = len(table) - 1
    t = np.clip(p, 0, 1) * intervals
    i = min(int(t), intervals - 1)
    pose = table[i] * (1 - (t - i)) + table[i + 1] * (t - i)
    for j in range(0, 24, 4):
        pose[j : j + 4] = unit(pose[j : j + 4])
    if phase == "reload":
        arrow = ARROW_TABLE[i] * (1 - (t - i)) + ARROW_TABLE[i + 1] * (t - i)
        arrow[3:7] = unit(arrow[3:7])
        pose[26] = float(np.clip(p, 0, 1) * RELOAD_DURATION >= PICKUP_VISIBLE_SECONDS)
        pose = np.concatenate([pose, arrow])
    return pose


def transforms(pose):
    xf = {k: (np.eye(3), np.zeros(3)) for k in PARTS}
    grips = []
    for offset, chain in [(0, CHAIN[0]), (12, CHAIN[1])]:
        a, b, c, grip = chain
        s, e, w = [JOINTS[k] for k in chain[:3]]
        r1 = qmatrix(pose[offset : offset + 4])
        r2 = r1 @ qmatrix(pose[offset + 4 : offset + 8])
        r3 = r2 @ qmatrix(pose[offset + 8 : offset + 12])
        ep = s + r1 @ (e - s)
        wp = ep + r2 @ (w - e)
        gp = wp + r3 @ (grip - w)
        xf.update({a: (r1, s - r1 @ s), b: (r2, ep - r2 @ e), c: (r3, wp - r3 @ w)})
        grips.append(gp)
    yaw, bend, visible = pose[24:27]
    yaw_body, torso_pitch, bow_pitch = pose[27:30]
    body = euler([0, yaw_body, 0])
    rb = body.T @ rx(-bow_pitch) @ euler([0, yaw, 0])
    xf[8] = rb, grips[1] - rb @ GB
    for pid, sign in [(14, -1), (15, 1)]:
        r = rb @ rx(sign * bend)
        p = JOINTS[pid]
        pp = grips[1] + rb @ (p - GB)
        xf[pid] = r, pp - r @ p
    top = transform(TOP, xf[14])
    bottom = transform(BOTTOM, xf[15])
    mid = (top + bottom) / 2
    halfspan = np.linalg.norm(top - bottom) / 2
    sag = math.sqrt(max(0, 0.87 ** 2 - halfspan ** 2))
    nock = mid - rb @ np.array([0, 0, sag])
    for pid, end, original in [(13, top, TOP), (17, bottom, BOTTOM)]:
        r = align(original - NOCK, end - nock)
        xf[pid] = r, nock - r @ NOCK
    # The string intersection follows its two fixed segment lengths.
    arrow_r = align([0, 0, 1], grips[1] + rb @ np.array([0.020, 0, 0]) - nock)
    if len(pose) > 30:
        amount = pose[37]
        q = quaternion(arrow_r)
        free_q = pose[33:37]
        if np.dot(q, free_q) < 0:
            free_q = -free_q
        arrow_r = qmatrix(free_q * (1 - amount) + q * amount)
        arrow_origin = pose[30:33] * (1 - amount) + nock * amount
    else:
        arrow_origin = nock
    xf[16] = arrow_r, arrow_origin - arrow_r @ JOINTS[16]
    waist = JOINTS[19]
    pitch = rx(-torso_pitch)
    upper_parent = pitch @ body
    upper_translation = waist - pitch @ waist
    for pid, (r, translation) in list(xf.items()):
        if pid in [0, 2, 3]:
            xf[pid] = body @ r, body @ translation
        elif pid == 18:
            neck = JOINTS[18]
            xf[pid] = pitch, upper_translation + upper_parent @ neck - pitch @ neck
        else:
            xf[pid] = upper_parent @ r, upper_parent @ translation + upper_translation
    world = lambda p: upper_parent @ p + upper_translation
    return xf, ([world(g) for g in grips], world(top), world(bottom), world(nock))


def transform(p, xf):
    r, t = xf
    return np.asarray(p) @ r.T + t


def deform(pos, parts, pose):
    xf, (grips, top, bottom, nock) = transforms(pose)
    out = np.array(pos, copy=True)
    for pid in PARTS:
        mask = parts == pid
        out[mask] = transform(pos[mask], xf[pid])
    if pose[26] < 0.5:
        out[parts == 16] = nock
    return out


def export():
    return {
        "version": 3,
        "space": "glTF metres, +Y up, +Z forward",
        "parts": PARTS,
        "joints": {k: v.tolist() for k, v in JOINTS.items()},
        "channels": [
            "draw_shoulder_xyzw",
            "draw_elbow_xyzw",
            "draw_wrist_xyzw",
            "bow_shoulder_xyzw",
            "bow_elbow_xyzw",
            "bow_wrist_xyzw",
            "bow_yaw",
            "limb_bend",
            "arrow_visible",
            "body_yaw",
            "torso_pitch",
            "bow_pitch",
        ],
        "rotation": "local unit quaternions xyzw, parent times child, shortest-path normalized linear interpolation",
        "durations": {"raise": 1.5, "release": 0.6, "reload": RELOAD_DURATION},
        "events": {
            "release": [{"time": 0.0, "event": "loose", "arrow_visible": False}],
            "reload": [
                {
                    "time": PICKUP_VISIBLE_SECONDS,
                    "event": "arrow_emerges",
                    "arrow_visible": True,
                },
                {"time": 3.8, "event": "nocked"},
            ],
        },
        "arrow_space": "torso-local glTF metres, before body yaw and torso pitch",
        "arrow_channels": ["nock_xyz", "orientation_xyzw", "string_attachment"],
        "arrow_samples": {"reload": ARROW_TABLE.tolist()},
        "discrete_channels": ["arrow_visible"],
        "samples": {k: v.tolist() for k, v in TABLES.items()},
    }


if __name__ == "__main__":
    Path(__file__).with_name("archer.raise.json").write_text(
        json.dumps(export(), indent=2) + "\n"
    )
    for phase in TABLES:
        xf, data = transforms(sample(phase, 1))
        print(phase, data)
