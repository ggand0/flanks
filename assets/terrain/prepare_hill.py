"""Author the hill heightmap at 2 m spacing.

Run: python3 assets/terrain/prepare_hill.py
Requires NumPy and Pillow. Writes a linear 16-bit greyscale PNG; the game
reads its vertical scale from hill.json. The first row is the northern edge.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


CROWN_M = 45.0
FACE_M = 48.0
JOIN_M = 8.0
FACE_DEGREES = 37.5


def surface(x, z):
    """A gentle lower approach followed by a short, steep upper face."""
    crest_z = -160 + 12 * np.sin(x / 180) + 6 * np.sin(x / 75)
    crest_height = CROWN_M - 2.2 * (1 - np.cos(x / 115)) - 1.5 * (np.abs(x) / 512)**4
    distance = z - crest_z
    lower_length = 280 - (crest_z + FACE_M)
    foot_m = 80.0
    face_grade = np.tan(np.radians(FACE_DEGREES))
    # Solve the lower grade so the joined profile reaches the authored crown.
    lower_grade = (crest_height - face_grade * (FACE_M - JOIN_M)) / (
        lower_length - foot_m / 2 + JOIN_M / 2
    )
    travel = np.clip(280 - z, 0, lower_length)
    lower_rise = np.where(travel < foot_m, travel**2 / (2 * foot_m), travel - foot_m / 2)
    lower_height = lower_grade * lower_rise
    face_base = lower_grade * (lower_length - foot_m / 2)

    # Integrate smoothstep grades at the toe and crown; the middle stays 37.5 degrees.
    q = np.clip(FACE_M - distance, 0, FACE_M)
    toe_u = np.clip(q / JOIN_M, 0, 1)
    toe = lower_grade * q + (face_grade - lower_grade) * JOIN_M * (toe_u**3 - 0.5 * toe_u**4)
    toe_height = JOIN_M * (lower_grade + face_grade) / 2
    middle = toe_height + face_grade * (q - JOIN_M)
    cap_u = np.clip((q - (FACE_M - JOIN_M)) / JOIN_M, 0, 1)
    cap = toe_height + face_grade * (FACE_M - 2 * JOIN_M) + face_grade * JOIN_M * (
        cap_u - cap_u**3 + 0.5 * cap_u**4
    )
    face_rise = np.where(q < JOIN_M, toe, np.where(q > FACE_M - JOIN_M, cap, middle))
    height = np.where(distance <= FACE_M, face_base + face_rise, lower_height)
    rear = np.clip(-distance / 224, 0, 1)
    height = np.where(distance < 0, crest_height * (1 - 0.11 * rear**2), height)

    # Retain the lower rolls, fading them out before the steep face.
    envelope = np.minimum(np.clip(travel / 60, 0, 1), np.clip((distance - FACE_M) / 40, 0, 1))
    envelope = envelope**2 * (3 - 2 * envelope)
    rolls = (3.4 * np.exp(-((x + 230) / 40)**2 - ((z - 110) / 24)**2)
             + 2.6 * np.exp(-((x - 190) / 50)**2 - ((z - 45) / 25)**2)
             + 3.0 * np.exp(-((x + 5) / 42)**2 - ((z + 55) / 28)**2)
             - 1.5 * np.exp(-((x + 40) / 100)**2 - ((z - 125) / 80)**2))
    return np.clip(height + rolls * envelope, 0, CROWN_M)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "hill_height.png")
    args = parser.parse_args()
    x, z = np.meshgrid(np.arange(-512, 513, 2), np.arange(-384, 385, 2))
    heights = surface(x, z)
    pixels = np.rint(heights / CROWN_M * 65535).astype(np.uint16)
    Image.fromarray(pixels).save(args.out)
    decoded = np.asarray(Image.open(args.out)).astype(np.float64) * (CROWN_M / 65535)
    error = np.max(np.abs(decoded - heights))
    assert error <= CROWN_M / 65535 / 2 + 1e-10
    dz, dx = np.gradient(decoded, 2)
    angles = np.degrees(np.arctan(np.hypot(dx, dz)))
    crest_z = -160 + 12 * np.sin(x / 180) + 6 * np.sin(x / 75)
    distance = z - crest_z
    face = (np.abs(x) <= 440) & (distance >= JOIN_M + 2) & (distance <= FACE_M - JOIN_M - 2)
    lower = (np.abs(x) <= 440) & (distance >= FACE_M + 4) & (z <= 280)
    assert angles[face].min() >= 35 and angles.max() <= 40
    print(json.dumps({
        "file": str(args.out),
        "dimensions": [513, 385],
        "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "height_min_max_m": [float(decoded.min()), float(decoded.max())],
        "face_horizontal_run_m": FACE_M,
        "face_join_m": JOIN_M,
        "face_angle_min_max_degrees": [float(angles[face].min()), float(angles[face].max())],
        "lower_angle_median_p95_max_degrees": np.percentile(angles[lower], [50, 95, 100]).tolist(),
        "maximum_angle_degrees": float(angles.max()),
        "height_at_centre_crest_m": float(decoded[112, 256]),
        "height_at_face_foot_m": float(decoded[136, 256]),
        "height_at_valley_m": float(decoded[332, 256]),
        "quantization_error_max_m": float(error),
    }, indent=2))


if __name__ == "__main__":
    main()
