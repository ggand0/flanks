"""Compose aligned renders and silhouette overlays for the knight L2."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
args = parser.parse_args()
HERE = args.folder.resolve()
validation = json.loads((HERE / "validation.json").read_text())
counts = [validation["levels"][name]["triangles"] for name in ["L0", "L2"]]
kind = HERE.parent.name.replace("_", " ").title()
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
sheet = Image.new("RGB", (1200, 1000), (32, 35, 40))
draw = ImageDraw.Draw(sheet)
draw.text(
    (24, 14),
    f"{kind}: L0 / L2     |     50 degree elevation     |     {counts[0]} / {counts[1]} triangles",
    font=font,
    fill="white",
)
metrics = {}
for col, facing in enumerate(["front", "back", "side"]):
    x = col * 400
    draw.text((x + 15, 50), facing.replace("_", " "), font=font, fill="white")
    for j, level in enumerate(["L0", "L2"]):
        im = Image.open(HERE / f"{level}_{facing}_256px.png")
        im = im.resize((200, 200), Image.Resampling.LANCZOS)
        sheet.paste(im, (x + j * 200, 78), im)
        draw.text((x + j * 200 + 80, 282), level, font=font, fill="white")
    a = np.array(Image.open(HERE / f"L0_{facing}_256px.png"))[:, :, 3] / 255
    b = np.array(Image.open(HERE / f"L2_{facing}_256px.png"))[:, :, 3] / 255
    common = np.minimum(a, b)
    overlay = np.zeros((*a.shape, 3), np.float32) + [32, 35, 40]
    for weight, color in [
        (common, [200, 206, 210]),
        (np.maximum(a - b, 0), [0, 200, 255]),
        (np.maximum(b - a, 0), [255, 96, 140]),
    ]:
        overlay = (
            overlay * (1 - weight[:, :, None]) + np.array(color) * weight[:, :, None]
        )
    image = Image.fromarray(np.uint8(overlay))
    image.save(HERE / f"overlay_{facing}.png")
    sheet.paste(image, (x + 40, 320))
    iou = common.sum() / np.maximum(a, b).sum()
    metrics[facing] = {
        "silhouette_iou": float(iou),
        "coverage_ratio_L2_to_L0": float(b.sum() / a.sum()),
    }
    draw.text((x + 16, 650), f"Silhouette overlap {iou:.1%}", fill="white", font=small)
    for row, size in enumerate([20, 8, 3]):
        y = 695 + row * 88
        draw.text((x + 10, y + 20), f"{size}px", fill="white", font=small)
        for j, level in enumerate(["L0", "L2"]):
            im = Image.open(HERE / f"{level}_{facing}_{size}px.png")
            crop = im.crop((16, 16, 48, 48)).resize((96, 96), Image.Resampling.NEAREST)
            sheet.paste(crop, (x + 58 + j * 150, y - 12), crop)
            sheet.paste(im, (x + 140 + j * 150, y + 4), im)
draw.text(
    (24, 978),
    "Overlay: grey shared, cyan L0 only, pink L2 only. Small renders: 3x nearest and native pixels.",
    font=small,
    fill="white",
)
sheet.save(HERE / "review.png")
(HERE / "silhouette_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(metrics)

size_sheet = Image.new("RGB", (1000, 480), (32, 35, 40))
draw = ImageDraw.Draw(size_sheet)
draw.text(
    (20, 14),
    f"{kind} at 50 degrees | Native pixels and nearest-neighbour enlargement",
    font=font,
    fill="white",
)
for row, color in enumerate(["front", "blue"]):
    for col, (size, zoom) in enumerate([(20, 6), (8, 12), (3, 24)]):
        x, y = 20 + col * 330, 70 + row * 205
        draw.text(
            (x, y),
            f'{"Red" if row == 0 else "Blue"}, {size}px | L0 / L2',
            font=font,
            fill="white",
        )
        for j, level in enumerate(["L0", "L2"]):
            image = Image.open(HERE / f"{level}_{color}_{size}px.png")
            edge = max(size + 6, 8)
            low = 32 - edge // 2
            crop = image.crop((low, low, low + edge, low + edge))
            big = crop.resize((edge * zoom, edge * zoom), Image.Resampling.NEAREST)
            size_sheet.paste(big, (x + j * 155, y + 30), big)
            size_sheet.paste(image, (x + j * 155 + 70, y + 125), image)
size_sheet.save(HERE / "size_review.png")
