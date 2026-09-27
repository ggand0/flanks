"""Pack the unique grassland albedo with an exposed-earth detail mask in alpha.

Run python3 assets/terrain/prepare_layout.py --toktx /path/to/toktx.
The source artwork covers the full 1024 by 768 metre battlefield once.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image


def transition(low, high, value):
    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toktx", default="toktx")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    spec = json.loads((root / "grassland_layout.source.json").read_text())
    source_path = root / spec["file"]
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"Checksum mismatch: {source_path}")
    source = Image.open(source_path).convert("RGB")
    if list(source.size) != spec["pixels"]:
        raise ValueError(f"Image size mismatch: {source_path}")
    rgb = np.asarray(source, dtype=np.float32) / 255.0
    saturation = (rgb.max(axis=2) - rgb.min(axis=2)) / np.maximum(rgb.max(axis=2), 0.001)
    red_excess = (rgb[:, :, 0] - rgb[:, :, 1]) / np.maximum(rgb[:, :, 0] + rgb[:, :, 1], 0.001)
    # Pale warm openings receive soil detail. Their source color stays intact;
    # this mask only chooses the close scan and normal, never distant color.
    soil = (1.0 - transition(0.20, 0.48, saturation)) * transition(0.02, 0.09, red_excess)
    packed = source.convert("RGBA")
    packed.putalpha(Image.fromarray(np.round(soil * 255.0).astype(np.uint8)))
    destination = root / "grassland_layout_color.ktx2"
    with tempfile.TemporaryDirectory(prefix="flanks-layout-") as temporary:
        png = Path(temporary) / "layout.png"
        packed.save(png)
        subprocess.run(
            [args.toktx, "--t2", "--genmipmap", "--filter", "box",
             "--assign_oetf", "srgb", "--target_type", "RGBA", "--zcmp", "9",
             str(destination), str(png)],
            check=True,
        )
    print(f"{destination.name}: {source.width} x {source.height}, {destination.stat().st_size} bytes")


if __name__ == "__main__":
    main()
