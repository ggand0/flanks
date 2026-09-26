"""Assemble shuffle videos, GIFs and foot-contact review sheets.

Run: python3 tools/blender/knight/compose_shuffle.py --out-dir assets_dev/knight/shuffle_v1
"""

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flanks-shuffle-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from shuffle import Shuffle, sample

REPO = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
sys.path.insert(0, str(REPO / "tools/blender"))
from inspect_surfaces import load


def diagram(rig, table, direction):
    sign = 1 if direction == "left" else -1
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 3.4), dpi=100)
    colors = ["#2878cc", "#d34e45"]
    phases = np.linspace(0, 1, 301)
    joints = np.array([rig.joint_positions(sample(table, direction, p)) for p in phases])
    joints[:, :, :, 0] += sign * phases[:, None, None] * rig.distance
    for index in np.linspace(0, 300, 11).astype(int):
        for leg, color in enumerate(colors):
            path = joints[index, leg]
            axes[0].plot(path[:, 0], path[:, 2], color=color, alpha=0.15 + 0.75 * phases[index], lw=1)
    for leg, color in enumerate(colors):
        foot = joints[:, leg, 2]
        axes[0].plot(foot[:, 0], foot[:, 2], ":", color=color, lw=1)
        for start, end in table["clips"][direction]["contact_phase_intervals"]["left" if leg == 0 else "right"]:
            center = rig.ankles[leg].copy()
            if start > 0:
                center[0] += sign * rig.distance
            outline = np.array([[-0.050, -0.090], [0.050, -0.090], [0.060, 0.05], [0.025, 0.21], [0, 0.258], [-0.025, 0.21], [-0.060, 0.05]])
            outline += center[[0, 2]]
            axes[0].add_patch(Polygon(outline, facecolor=color, edgecolor=color, alpha=0.25))
            label = f'{"L" if leg == 0 else "R"} {start * rig.duration:.2f}-{end * rig.duration:.2f}s'
            axes[0].text(center[0], -0.16 - 0.09 * leg, label, color=color, ha="center", fontsize=8)
        axes[1].plot(phases * rig.duration, foot[:, 1] - rig.ankles[leg, 1], color=color, label=f'{"Left" if leg == 0 else "Right"} foot')
        axes[2].plot(phases * rig.duration, foot[:, 0], color=color)
    dip = [-sample(table, direction, p)[1] for p in phases]
    axes[1].plot(phases * rig.duration, dip, "--", color="#303030", label="Pelvis dip")
    axes[2].plot(phases * rig.duration, sign * phases * rig.distance, "--", color="#303030", label="Root")
    axes[0].set(title="Top view: planted footprints and leg strobe", xlabel="Asset X (m), positive = soldier left", ylabel="Forward Z (m)")
    axes[0].set_aspect("equal")
    axes[0].set_ylim(-0.34, 0.33)
    axes[0].autoscale_view()
    axes[1].set(title="Foot lift and pelvis dip", xlabel="Seconds", ylabel="Metres")
    axes[1].legend(fontsize=8)
    axes[2].set(title="World sideways travel: plateaus = planted", xlabel="Seconds", ylabel="Asset X (m)")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO / "assets_dev/knight/shuffle_v1")
    args = parser.parse_args()
    output = args.out_dir.resolve()
    doc, _, _, _ = load(REPO / "assets/units/knight.glb")
    nodes = {n["name"]: n.get("translation", [0, 0, 0]) for n in doc["nodes"]}
    rig = Shuffle(nodes)
    table = json.loads((output / "knight.shuffle.json").read_text())
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25)
    for direction in ["left", "right"]:
        sheet = Image.new("RGB", (1680, 930), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((24, 15), f"Knight shuffle {direction}: 0.60 m / 0.75 s (0.80 m/s)", font=title_font, fill="black")
        draw.text((24, 52), "Square stance, lead foot out then trailing foot closes. Fixed cameras and translated root. Blue = left foot, red = right.", font=font, fill="black")
        for row, view in enumerate(["back", "oblique"]):
            for index, frame in enumerate([6, 9, 12, 15, 18, 21, 24]):
                tile = Image.open(output / f"frames_{direction}_{view}/{frame:03d}.png").resize((240, 180), Image.Resampling.LANCZOS)
                x, y = index * 240, 110 + row * 230
                sheet.paste(tile, (x, y))
                draw.text((x + 8, y - 25), f"{view} {(frame - 6) / 24:.3f} s", font=font, fill="black")
        plots = diagram(rig, table, direction)
        plots.save(output / f"knight_shuffle_{direction}_footprints.png")
        sheet.paste(plots, (0, 565))
        sheet.save(output / f"knight_shuffle_{direction}_sheet.png")
        for view in ["back", "oblique"]:
            sequence = output / f"frames_{direction}_{view}"
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(sequence / "%03d.png"),
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(output / f"knight_shuffle_{direction}_{view}.mp4"),
            ], check=True)
            frames = [Image.open(sequence / f"{i:03d}.png").resize((600, 450), Image.Resampling.LANCZOS) for i in range(48)]
            frames[0].save(output / f"knight_shuffle_{direction}_{view}.gif", save_all=True, append_images=frames[1:], duration=[40, 40, 40, 40, 40, 50] * 8, loop=0)
    inputs = [output / f"knight_shuffle_{direction}_{view}.mp4" for view in ["back", "oblique"] for direction in ["left", "right"]]
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for path in inputs:
        command.extend(["-i", str(path)])
    command.extend([
        "-filter_complex", "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack,drawtext=text='SHUFFLE LEFT':x=24:y=20:fontsize=26:fontcolor=white,drawtext=text='SHUFFLE RIGHT':x=744:y=20:fontsize=26:fontcolor=white[v]",
        "-map", "[v]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / "knight_shuffle_review.mp4"),
    ])
    subprocess.run(command, check=True)
    print("Wrote two review sheets, four GIFs and five MP4s")


if __name__ == "__main__":
    main()
