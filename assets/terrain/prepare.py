"""Pack the Poly Haven ground maps into mipmapped KTX2 textures.

Requires NumPy, Pillow and Khronos toktx 4.4.2. Run from any directory:
python3 assets/terrain/prepare.py --downloads /tmp/terrain-downloads --toktx toktx
Source JPEGs must match sources.json. Download URLs and checksums are in that file.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from PIL import Image
import numpy as np


def pasture_detail(source):
    """Keep neutral grass detail; landscape color comes from the coverage map."""
    srgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
    log_light = np.log(np.maximum(luminance, 0.001))
    height, width = luminance.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    # Periodic filtering preserves the tile seam and removes clumps larger
    # than about 0.4 m when the texture covers 6 m of ground.
    gaussian = np.exp(-2.0 * np.pi ** 2 * 32.0 ** 2 * (fx * fx + fy * fy))
    low = np.fft.ifft2(np.fft.fft2(log_light) * gaussian).real
    neutral = np.clip(0.5 + (log_light - low) * 0.3, 0.02, 0.98)
    encoded = np.where(neutral <= 0.0031308, neutral * 12.92,
                       1.055 * neutral ** (1.0 / 2.4) - 0.055)
    pixels = np.round(encoded * 255.0).astype(np.uint8)
    return Image.fromarray(pixels).convert("RGBA")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--toktx", default="toktx")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sources = json.loads((root / "sources.json").read_text())
    with tempfile.TemporaryDirectory(prefix="flanks-ground-") as temporary:
        for layer, source in sources.items():
            maps = {}
            for name, spec in source["maps"].items():
                path = args.downloads / spec["file"]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != spec["sha256"]:
                    raise ValueError(f"Checksum mismatch: {path}")
                maps[name] = Image.open(path)

            albedo = maps["Diffuse"].convert("RGBA")
            if layer == "pasture":
                albedo = pasture_detail(maps["Diffuse"])
            normal = maps["nor_gl"].convert("RGBA")
            normal.putalpha(maps["Rough"].convert("L"))
            for suffix, pixels, transfer in [
                ("color", albedo, "srgb"),
                ("normal_roughness", normal, "linear"),
            ]:
                png = Path(temporary) / f"{layer}_{suffix}.png"
                pixels.save(png)
                destination = root / f"{layer}_{suffix}.ktx2"
                subprocess.run(
                    [args.toktx, "--t2", "--genmipmap", "--filter", "box",
                     "--assign_oetf", transfer,
                     "--target_type", "RGBA", "--zcmp", "9",
                     str(destination), str(png)],
                    check=True,
                )
                print(f"{destination.name}: {destination.stat().st_size} bytes")


if __name__ == "__main__":
    main()
