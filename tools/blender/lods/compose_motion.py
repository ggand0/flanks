"""Compose L0/L2 attack frames into paired frames and a diagnostic sequence.

python3 tools/blender/lods/compose_motion.py assets_dev/kind/lod_l2_v1
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
args = parser.parse_args()
folder = args.folder.resolve()
report = json.loads((folder / "motion_validation.json").read_text())
levels = report.get("levels", ["L0", "L2"])
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
paired = folder / "motion_paired"
paired.mkdir(exist_ok=True)
for frame in range(report["frames"]):
    canvas = Image.new("RGB", (640, 360), (32, 35, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (15, 12),
        f'{folder.parent.name.replace("_", " ").title()}    {" / ".join(levels)}    frame {frame}',
        font=font,
        fill="white",
    )
    for column, level in enumerate(levels):
        image = Image.open(folder / "motion_frames" / f"{level}_{frame:03d}.png")
        canvas.paste(image, (column * 320, 40), image)
    canvas.save(paired / f"{frame:03d}.png")
indices = np.linspace(0, report["frames"] - 1, 16).astype(int)
sheet = Image.new("RGB", (1280, 720), (32, 35, 40))
for tile, frame in enumerate(indices):
    image = Image.open(paired / f"{frame:03d}.png").resize(
        (320, 180), Image.Resampling.LANCZOS
    )
    sheet.paste(image, (tile % 4 * 320, tile // 4 * 180))
sheet.save(folder / "motion_sequence.png")
print("COMPOSED", folder, report["frames"], "frames")
