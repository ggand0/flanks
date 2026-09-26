"""Pack the Poly Haven ground maps into mipmapped KTX2 textures.

Requires Pillow and Khronos toktx 4.4.2. Run from any directory:
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
