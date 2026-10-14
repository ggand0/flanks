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


def surface(x, z):
    """Heights in metres, with a 36 m crown and a southern valley at zero."""
    crest_z = -160 + 12 * np.sin(x / 180) + 6 * np.sin(x / 75)
    crest_height = 36 - 2.2 * (1 - np.cos(x / 115)) - 1.5 * (np.abs(x) / 512)**4
    length = 280 - crest_z
    travel = np.clip(280 - z, 0, length)
    ramp = 80.0
    rise = np.where(travel < ramp, travel**2 / (2 * ramp), travel - ramp / 2)
    rise = np.where(travel > length - ramp,
                    length - ramp - (length - travel)**2 / (2 * ramp), rise)
    profile = rise / (length - ramp)
    rear = np.clip((crest_z - z) / 224, 0, 1)
    profile = np.where(z < crest_z, 1 - 0.11 * rear**2, profile)

    # These unequal rolls fade before the foot and the deployment crown.
    envelope = np.minimum(np.clip(travel / 60, 0, 1), np.clip((length - travel) / 50, 0, 1))
    envelope = envelope**2 * (3 - 2 * envelope)
    rolls = (3.4 * np.exp(-((x + 230) / 40)**2 - ((z - 110) / 24)**2)
             + 2.6 * np.exp(-((x - 190) / 50)**2 - ((z - 45) / 25)**2)
             + 3.0 * np.exp(-((x + 5) / 42)**2 - ((z + 55) / 28)**2)
             - 1.5 * np.exp(-((x + 40) / 100)**2 - ((z - 125) / 80)**2))
    return np.clip(profile * crest_height + rolls * envelope, 0, 36)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "hill_height.png")
    args = parser.parse_args()
    x, z = np.meshgrid(np.arange(-512, 513, 2), np.arange(-384, 385, 2))
    heights = surface(x, z)
    pixels = np.rint(heights / 36 * 65535).astype(np.uint16)
    Image.fromarray(pixels).save(args.out)
    decoded = np.asarray(Image.open(args.out)).astype(np.float64) * (36 / 65535)
    assert np.max(np.abs(decoded - heights)) <= 36 / 65535 / 2 + 1e-10
    dz, dx = np.gradient(decoded, 2)
    grades = np.hypot(dx, dz)
    assert grades.max() < 0.25
    print(json.dumps({
        "file": str(args.out),
        "dimensions": [513, 385],
        "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "height_min_max_m": [float(decoded.min()), float(decoded.max())],
        "maximum_grade_percent": float(grades.max() * 100),
        "height_at_centre_crest_m": float(decoded[112, 256]),
        "height_at_valley_m": float(decoded[332, 256]),
        "quantization_error_max_m": float(np.max(np.abs(decoded - heights))),
    }, indent=2))


if __name__ == "__main__":
    main()
