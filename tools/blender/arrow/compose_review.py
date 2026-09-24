"""Compose arrow renders into a labelled review sheet and HTML page.

python3 tools/blender/arrow/compose_review.py assets_dev/arrow
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
folder = parser.parse_args().folder
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = ImageFont.truetype(font_path, 19)
small = ImageFont.truetype(font_path, 15)
title = ImageFont.truetype(font_path, 30)
sheet = Image.new("RGB", (1280, 1380), (28, 33, 40))
draw = ImageDraw.Draw(sheet)
draw.text((28, 20), "Arrow projectile", font=title, fill="white")
draw.text(
    (28, 65),
    "36 triangles  |  0.76 m  |  closed geometry  |  vertex colours  |  backface culling",
    font=font,
    fill="#bdc7d5",
)


def panel(name, label, rect):
    x, y, width, height = rect
    draw.rounded_rectangle((x, y, x + width, y + height), radius=8, fill=(37, 43, 52))
    draw.text((x + 12, y + 8), label, font=font, fill="white")
    image = Image.open(folder / f"{name}.png").convert("RGBA")
    if name in ("side", "top", "boxes_side"):
        image = image.crop((0, 100, image.width, 200))
    elif name == "soldier_scale":
        bounds = image.getbbox()
        image = image.crop(
            (max(0, bounds[0] - 25), 0, min(image.width, bounds[2] + 25), image.height)
        )
    image.thumbnail((width - 16, height - 38), Image.Resampling.LANCZOS)
    sheet.paste(
        image,
        (x + (width - image.width) // 2, y + 34 + (height - 38 - image.height) // 2),
        image,
    )


panel("side", "Side: new arrow", (24, 110, 900, 165))
panel(
    "boxes_side", "Side: existing three-box mesh, same metre scale", (24, 285, 900, 165)
)
panel("top", "Top", (24, 460, 900, 165))
panel("soldier_scale", "Beside a 1.80 m soldier", (940, 110, 316, 700))
panel("fletching_closeup", "Three solid fletching vanes", (24, 640, 438, 315))
panel("head_closeup", "Steel bodkin head", (478, 640, 446, 315))
panel("rear", "Rear: looking along +Z", (24, 970, 300, 320))
panel("nock_closeup", "Carved V nock", (340, 970, 300, 320))
draw.text((674, 990), "Approximately 10 px long", font=font, fill="white")
draw.text(
    (674, 1023),
    "Native pixels, then 12x nearest-neighbour enlargement",
    font=small,
    fill="#bdc7d5",
)
for row, (name, label) in enumerate((("ten_px", "New"), ("boxes_ten_px", "Boxes"))):
    y = 1080 + row * 100
    draw.text((680, y + 20), label, font=font, fill="white")
    image = Image.open(folder / f"{name}.png").convert("RGBA")
    sheet.paste(image, (765, y), image)
    crop = image.crop((24, 29, 40, 35)).resize((192, 72), Image.Resampling.NEAREST)
    sheet.paste(crop, (855, y - 4), crop)
draw.text(
    (28, 1330),
    "True-scale asset. Runtime readability scaling is separate. The soldier and its quiver are unchanged.",
    font=font,
    fill="#bdc7d5",
)
sheet.save(folder / "review.png")
links = "".join(
    f'<a href="{name}.png">{label}</a> '
    for name, label in [
        ("side", "Side"),
        ("top", "Top"),
        ("rear", "Rear"),
        ("head_closeup", "Head"),
        ("fletching_closeup", "Fletching"),
        ("nock_closeup", "Nock"),
        ("ten_px", "10 pixels"),
        ("soldier_scale", "Scale"),
    ]
)
(folder / "review.html").write_text(
    f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arrow projectile</title><style>body{{background:#1c2128;color:#e9edf4;font:17px/1.6 system-ui;margin:28px auto;max-width:1280px;padding:0 18px}}a{{color:#9ed3ff;margin-right:15px}}img{{width:100%;height:auto}}p{{max-width:1000px}}</style>
<h1>Arrow projectile</h1><p>36 triangles, 0.76 m long. Wooden shaft, carved nock, steel bodkin and three closed fletching vanes. One L0 mesh; part ID 7, pivot height 0, vertex alpha 0. No textures.</p>
<p><a href="arrow.glb">GLB</a><a href="arrow.blend">Blender scene</a><a href="arrow.validation.json">Validation</a></p>
<img src="review.png" alt="Arrow comparison views and scale review"><p>{links}</p>
<p>The box reference reproduces the existing procedural mesh dimensions and colours. Side views use the same metre scale; the small comparisons each frame the complete arrow at 10 pixels. All views use backface culling. The soldier is shown at 1.80 m without changing its asset.</p></html>
"""
)
print(folder / "review.png")
