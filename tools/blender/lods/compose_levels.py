"""Compose four-level size, silhouette and colour comparisons.

python3 tools/blender/lods/compose_levels.py assets_dev/knight/lod_l1_l3_v1
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
folder = parser.parse_args().folder
kind = folder.parent.name
kind_name = {"man_at_arms": "Man-at-arms"}.get(kind, kind.title())
motion_frames = {"spearman": 64, "archer": 150}.get(kind, 90)
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = ImageFont.truetype(font_path, 19)
small = ImageFont.truetype(font_path, 15)
title = ImageFont.truetype(font_path, 28)
background = (28, 33, 40)
levels = ["L0", "L1", "L2", "L3"]
validation = json.loads((folder / "validation.json").read_text())
visibility = json.loads((folder / "visibility.json").read_text())
reference = visibility[next(k for k in visibility if k not in levels[1:])]
metrics = {}
for level in levels:
    entry = reference if level == "L0" else visibility[level]
    metrics[level] = {
        "triangles": validation["levels"][level]["triangles"],
        "mean_rgb_relative_percent": (
            (np.array(entry["linear_rgb"]) / reference["linear_rgb"] - 1) * 100
        ).tolist(),
        "team_share_percent": 100 * entry["team_share"],
        "silhouettes": {},
    }


def stamp(canvas, image, xy):
    canvas.paste(image, xy, image)


sheet = Image.new("RGB", (1120, 1000), background)
draw = ImageDraw.Draw(sheet)
draw.text((24, 15), f"{kind_name}: authored levels", font=title, fill="white")
for col, level in enumerate(levels):
    label = f'{level}: {metrics[level]["triangles"]} triangles'
    draw.text((col * 280 + 20, 62), label, font=font, fill="white")
for row, view in enumerate(["front", "back", "side"]):
    for col, level in enumerate(levels):
        image = Image.open(folder / f"{level}_red_{view}_256px.png").convert("RGBA")
        image = image.resize((280, 280), Image.Resampling.LANCZOS)
        stamp(sheet, image, (col * 280, 95 + row * 295))
    draw.text((22, 95 + row * 295), view.title(), font=small, fill="#b4c0d0")
sheet.save(folder / "review.png")

sizes = Image.new("RGB", (1120, 1120), background)
draw = ImageDraw.Draw(sizes)
draw.text((24, 14), f"{kind_name} at 20, 8 and 3 pixels", font=title, fill="white")
draw.text(
    (24, 55),
    "Each row uses identical framing. Enlargements preserve native pixels.",
    font=small,
    fill="#b4c0d0",
)
for col, level in enumerate(levels):
    draw.text((col * 280 + 118, 85), level, font=font, fill="white")
for row, (team, view) in enumerate(
    [("red", "front"), ("blue", "front"), ("red", "back"), ("red", "side")]
):
    y = 120 + row * 245
    draw.text((22, y), f"{team.title()} / {view}", font=small, fill="#b4c0d0")
    for col, level in enumerate(levels):
        for i, size in enumerate([20, 8, 3]):
            image = Image.open(folder / f"{level}_{team}_{view}_{size}px.png").convert(
                "RGBA"
            )
            stamp(sizes, image, (col * 280 + 14 + i * 85, y + 26))
            draw.text(
                (col * 280 + 37 + i * 85, y + 96), str(size), font=small, fill="white"
            )
            side = 28 if size == 20 else 14 if size == 8 else 8
            crop = image.crop(
                (32 - side // 2, 32 - side // 2, 32 + side // 2, 32 + side // 2)
            )
            scale = 3 if size == 20 else 6 if size == 8 else 10
            enlarged = crop.resize(
                (side * scale, side * scale), Image.Resampling.NEAREST
            )
            stamp(sizes, enlarged, (col * 280 + 5 + i * 90, y + 124))
sizes.save(folder / "size_review.png")

overlays = Image.new("RGB", (960, 710), background)
draw = ImageDraw.Draw(overlays)
draw.text(
    (20, 10),
    "Silhouettes: L0 blue, candidate orange, overlap white",
    font=font,
    fill="white",
)
for row, level in enumerate(["L1", "L3"]):
    for col, view in enumerate(["front", "back", "side"]):
        a = np.array(Image.open(folder / f"L0_red_{view}_256px.png"))[:, :, 3] > 127
        b = (
            np.array(Image.open(folder / f"{level}_red_{view}_256px.png"))[:, :, 3]
            > 127
        )
        rgb = np.zeros((*a.shape, 4), np.uint8)
        rgb[a] = [80, 150, 255, 255]
        rgb[b] = [255, 150, 70, 255]
        rgb[a & b] = [235, 235, 225, 255]
        stamp(overlays, Image.fromarray(rgb), (col * 320, 50 + row * 330))
        iou = float(np.count_nonzero(a & b) / np.count_nonzero(a | b))
        metrics[level]["silhouettes"][view] = {
            "iou": iou,
            "coverage_ratio_to_L0": float(b.sum() / a.sum()),
        }
        draw.text(
            (col * 320 + 16, 50 + row * 330),
            f"{level}, {view}: {iou:.1%}",
            font=small,
            fill="white",
        )
overlays.save(folder / "silhouettes.png")
(folder / "comparison.json").write_text(json.dumps(metrics, indent=2) + "\n")
rows = "".join(
    f'<tr><td>{level}</td><td>{v["triangles"]}</td><td>{v["team_share_percent"]:.2f}%</td><td>{max(abs(x) for x in v["mean_rgb_relative_percent"]):.2f}%</td></tr>'
    for level, v in metrics.items()
)
(folder / "review.html").write_text(
    f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{kind_name} L1 and L3</title><style>body{{max-width:1120px;margin:32px auto;padding:0 20px;background:#1c2128;color:#e5eaf1;font:17px/1.6 system-ui}}img{{width:100%;height:auto}}a{{color:#9dccff}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #48515e}}video{{width:100%;max-width:800px}}summary{{cursor:pointer;padding:16px 0}}</style>
<h1>{kind_name} L1 and L3</h1><p>L1 keeps the animated parts and equipment silhouette. L3 is a static {metrics['L3']['triangles']}-triangle silhouette, all part 0. Both use the original atlas. Existing L0 and L2 data are preserved byte for byte.</p>
<p><a href="{kind}.glb">Four-level GLB</a> · <a href="validation.json">Geometry and channels</a> · <a href="comparison.json">Colour and silhouettes</a></p>
<table><tr><th>Level</th><th>Triangles</th><th>Visible team share</th><th>Largest mean RGB change from L0</th></tr>{rows}</table>
<p>Colour measurements average eight facings at 50° in linear light. Pixel sizes use identical L0 framing and include held equipment.</p>
<h2>At playing sizes</h2><img src="size_review.png" alt="Four levels in red and blue at 20, 8 and 3 pixels">
<details open><summary>Front, rear and side</summary><img src="review.png" alt="All four levels from three directions"></details>
<details><summary>Silhouette overlays</summary><img src="silhouettes.png" alt="L1 and L3 silhouettes against L0"></details>
<h2>L1 motion</h2><video controls loop muted playsinline preload="metadata" src="motion_comparison.mp4" poster="motion_paired/000.png"></video>
<p>L0 left, L1 right. Existing attacks, {motion_frames} sampled frames. <a href="motion_sequence.png">Frame sequence</a> · <a href="motion_validation.json">Joint checks</a></p>
</html>"""
)
print(folder / "review.html")
