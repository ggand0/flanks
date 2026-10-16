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
CLIMB_M = 180.0
LOWER_GRADE = 0.025


def surface(x, z):
    """A broad climb with a rounded shoulder and an uneven ridge."""
    crest_z = -160 + 12 * np.sin(x / 180) + 6 * np.sin(x / 75)
    crest_height = CROWN_M - 2.2 * (1 - np.cos(x / 115)) - 1.5 * (np.abs(x) / 512)**4
    climb = CLIMB_M + 18 * np.sin(x / 145)
    distance = z - crest_z
    lower_length = 280 - (crest_z + climb)
    foot_m = 80.0
    travel = np.clip(280 - z, 0, lower_length)
    lower_rise = np.where(travel < foot_m, travel**2 / (2 * foot_m), travel - foot_m / 2)
    base = LOWER_GRADE * (lower_length - foot_m / 2)

    # Integrate a changing grade over the whole climb, with no straight face.
    toe_m = climb * 2 / 3
    crown_m = climb - toe_m
    peak_grade = 2 * (crest_height - base - LOWER_GRADE * toe_m / 2) / climb
    q = np.clip(climb - distance, 0, climb)
    u = np.clip(q / toe_m, 0, 1)
    toe = LOWER_GRADE * q + (peak_grade - LOWER_GRADE) * toe_m * (u**3 - 0.5 * u**4)
    toe_height = toe_m * (LOWER_GRADE + peak_grade) / 2
    v = np.clip((q - toe_m) / crown_m, 0, 1)
    crown = toe_height + peak_grade * crown_m * (v - v**3 + 0.5 * v**4)
    rise = np.where(q < toe_m, toe, crown)
    height = np.where(distance <= climb, base + rise, LOWER_GRADE * lower_rise)
    rear = np.clip(-distance / 224, 0, 1)
    height = np.where(distance < 0, crest_height * (1 - 0.11 * rear**2), height)

    # Carry the low rolls onto the climb, easing them away from the crown.
    envelope = np.minimum(np.clip((280 - z) / 60, 0, 1), np.clip(distance / 50, 0, 1))
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
    assert 18 < angles.max() < 28
    assert decoded[192, 256] > 6
    assert decoded[152, 256] > 20
    print(json.dumps({
        "file": str(args.out),
        "dimensions": [513, 385],
        "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "height_min_max_m": [float(decoded.min()), float(decoded.max())],
        "centre_climb_horizontal_run_m": CLIMB_M,
        "climb_horizontal_run_min_max_m": [CLIMB_M - 18, CLIMB_M + 18],
        "maximum_angle_degrees": float(angles.max()),
        "centre_maximum_angle_degrees": float(angles[:, 256].max()),
        "centre_height_at_z_m": {str(zz): float(decoded[(zz + 384)//2, 256]) for zz in [-160, -120, -80, -40, 0, 20, 280]},
        "quantization_error_max_m": float(error),
    }, indent=2))


if __name__ == "__main__":
    main()
