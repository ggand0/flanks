"""Draw the sprite sheet of the far soldiers' cards (FL_SPRITE).

Each kind is drawn from the level the far soldiers stand next to, posed by
the game itself: the tool writes the instance records of every frame it
wants, runs the game with FL_SPRITE_BAKE (src/sprite_bake.rs), which poses
every corner of the mesh with the draw's own pose code and writes the
placed corners back, and draws each frame from every view with the
model's atlas.

The card is lit in the game's shader by the same sky and sun as the mesh
(unit_instancing.wgsl `sky_and_sun`), so the sheet holds no light: per
texel the colour outside the team cloth, the cloth's grey, the normal in
the soldier's own frame and the cut-out.

Sheet (assets/units/sprites.png, sRGB RGBA8): two halves side by side.
Left half: rgb the colour outside the cloth (atlas colour times one less
the team mask), alpha the cut-out (the share of the texel the figure
covers). Right half, the same texel half a sheet to the right: rgb the
normal, n * 0.5 + 0.5 written so that the sRGB decode gives it back, alpha
the cloth's grey (atlas grey times the team mask times the largest channel
of the kind's cloth hue), which the shader tints by the team colour times
that hue. Every texel outside the figure carries the nearest figure
texel's values, so the mips have no fringe.

Each kind has one cell per (camera height, direction around him, frame), in
that order. A cell shows him orthographic from that view, x along the
camera's right and y along its up, and is just large enough for him: its
box, in metres from his centre (half his height above his feet), goes to
the layout with its place in the sheet. The cells are packed in shelves.
The layout goes to assets/units/sprites.json, which the game reads
(src/unit_glb.rs `SpriteLayout`).

Frames: a family per thing a man can be doing (FAMILIES, the order of the
ids the shader picks by, unit_instancing.wgsl `sprite_frame`), each a few
frames over its phase. A family a kind does not have takes the frames of
the one it falls back to.

Run from the repo root:

    python3 tools/sprite_sheet.py --game <game binary or launcher> [--work <dir>]

The game is started with FL_SPRITE_BAKE=<work dir> and the variables of
--env, and ends by itself once it has written the placed corners. With
--bake-done the tool reads an existing <work dir>/bake_out.bin instead.
Needs numpy, numba and Pillow.
"""
import argparse
import io
import json
import math
import os
import struct
import subprocess
import sys

import numba
import numpy as np
from PIL import Image

KINDS = ["knight", "man_at_arms", "spearman", "archer"]

# Pixels per metre of the sheet's base level, and the supersampling of
# the drawing.
PX_PER_M = 12.0
SS = 4
# Cells are a multiple of this many pixels wide and high and start on
# such a multiple, so the first mips of a cell take nothing from its
# neighbours.
CELL_ALIGN = 8
# Empty pixels kept around the figure in a cell at least.
MARGIN = 2

# Camera heights above the horizon, degrees, and directions around the man.
ELEVATIONS = [3.0, 14.0, 30.0, 55.0]
DIRECTIONS = 8

# Ground speeds, m/s, the gaits are drawn at, and the speeds the shader
# switches at. The pose sizes a man's gait to his speed (unit_pose.wgsl
# `gait_at`): up to about 1.5 m/s his stride grows, and from there to about
# 4 m/s his feet leave the ground for longer, from a walk into a run; above
# that it stays the same. So four gaits: a slow walk, the walk, a jog and
# the run, each switched to halfway between its neighbours in that change
# (the stride at a 0.86 m leg for the slow walk and the walk, the share of
# the run for the jog and the run). Below STAND_BELOW a man shows his
# standing frames.
GAITS = ["slow", "walk", "jog", "run"]
GAIT_SPEEDS = [0.8, 1.5, 2.6, 6.0]
GAIT_SWITCH = [1.1, 2.2, 3.3]
STAND_BELOW = 0.35

GAIT_FRAMES = 8
ATTACK_WINDUP = 4
ATTACK_FOLLOW = [0.25, 0.65]
SHOT_DRAW = [1 / 6, 1 / 2, 5 / 6]
SHOT_RELEASE = [0.25, 0.75]
SHOT_RELOAD = [1 / 6, 1 / 2, 5 / 6]
STAGGER = [0.45, 0.85]
CHEER_FRAMES = 4
# A charging blow lands at a run: its wind-up and follow-through frames,
# each with the legs at these places in the gait's cycle.
CHARGE_GAIT = [0.0, 0.5]

# The families, in the order of the shader's ids (unit_instancing.wgsl
# `sprite_frame`). Each gait has three: plain, with the spear or blade
# levelled, and an archer's with his knife out and his bow put away.
FAMILIES = ["stand", "ready", "wall", "bowup", "stand_knife"]
for _g in GAITS:
    FAMILIES += [_g, _g + "_level", _g + "_knife"]
FAMILIES += ["stab", "swing", "shot", "stagger", "cheer", "charge"]
# Per kind, the families it has, and what the others fall back to.
_LEVELLED = [g for gait in GAITS for g in (gait, gait + "_level")]
HAS = {
    "knight": ["stand", "wall"] + _LEVELLED + ["stab", "swing", "stagger", "cheer", "charge"],
    "man_at_arms": ["stand", "wall"] + _LEVELLED + ["stab", "swing", "stagger", "cheer", "charge"],
    "spearman": ["stand", "ready", "wall"] + _LEVELLED + ["stab", "stagger", "cheer", "charge"],
    "archer": ["stand", "bowup", "stand_knife"] + [g for gait in GAITS for g in (gait, gait + "_knife")]
    + ["shot", "stab", "swing", "stagger", "cheer", "charge"],
}
FALLBACK = {
    "ready": "stand", "wall": "stand", "bowup": "stand", "stand_knife": "stand",
    "swing": "stab", "shot": "stand", "stab": "stand",
}
for _g in GAITS:
    FALLBACK[_g + "_level"] = _g
    FALLBACK[_g + "_knife"] = _g

TAU = 2.0 * math.pi
# render_units.rs CELEBRATE_BASE and RANGED_BASE, FOLLOW_BASE and FOLLOW_SPAN.
CELEBRATE_BASE = 12.0
RANGED_BASE = 16.0
FOLLOW_BASE = 1.4
FOLLOW_SPAN = 0.59


def record(speed=0.0, z=0.0, band=0.0, wall=0.0, gait=0.0, w4=0.0, psw=0.0, time=0.0):
    """One instance record (render_units.rs InstanceData) at yaw 0 and the
    origin, seed 0, white: pos_scale, color, anim, anim2, and its time."""
    return [0.0, 0.0, 0.0, psw, 1.0, 1.0, 1.0, 0.0, 0.0, speed, z, 0.0, band, wall, gait, w4], time


def family_records(kind, fam):
    """The records of one family's frames, in frame order."""
    archer = kind == "archer"
    gait = [k / GAIT_FRAMES for k in range(GAIT_FRAMES)]
    if fam == "stand":
        return [record()]
    if fam == "ready":
        return [record(band=0.5)]
    if fam == "wall":
        return [record(band=0.5, wall=1.0)]
    if fam == "bowup":
        return [record(band=0.5, psw=1.0)]
    for g, speed in zip(GAITS, GAIT_SPEEDS):
        if fam == g:
            return [record(speed=speed, gait=x) for x in gait]
        if fam == g + "_level":
            return [record(speed=speed, band=0.5 if g != "run" else 1.0, gait=x) for x in gait]
        if fam == g + "_knife":
            return [record(speed=speed, psw=-1.0, gait=x) for x in gait]
    if fam == "stand_knife":
        return [record(psw=-1.0)]
    if fam == "charge":
        # The charging blow: the swing digit plus 3 (render_units.rs), a
        # stab at his parts' levels, at a run with the spear levelled.
        psw = -1.0 if archer else 0.0
        out = []
        for g in CHARGE_GAIT:
            out += [record(speed=GAIT_SPEEDS[-1], band=1.0, gait=g, z=6.0 + (k + 0.5) / ATTACK_WINDUP, psw=psw)
                    for k in range(ATTACK_WINDUP)]
            out += [record(speed=GAIT_SPEEDS[-1], band=1.0, gait=g, z=6.0 + FOLLOW_BASE + FOLLOW_SPAN * f, psw=psw)
                    for f in ATTACK_FOLLOW]
        return out
    if fam in ("stab", "swing"):
        digit = 0.0 if fam == "stab" else 1.0
        psw = -1.0 if archer else 0.0
        out = [record(band=0.5, z=digit * 2 + (k + 0.5) / ATTACK_WINDUP, psw=psw) for k in range(ATTACK_WINDUP)]
        out += [record(band=0.5, z=digit * 2 + FOLLOW_BASE + FOLLOW_SPAN * f, psw=psw) for f in ATTACK_FOLLOW]
        return out
    if fam == "shot":
        out = []
        for clip, xs in ((0, SHOT_DRAW), (1, SHOT_RELEASE), (2, SHOT_RELOAD)):
            out += [record(band=0.5, z=RANGED_BASE + clip * 2 + x, psw=1.0) for x in xs]
        return out
    if fam == "stagger":
        return [record(w4=s) for s in STAGGER]
    if fam == "cheer":
        return [record(z=CELEBRATE_BASE + 0.5, time=(k / CHEER_FRAMES) * TAU / 9.0) for k in range(CHEER_FRAMES)]
    raise ValueError(fam)


def kind_frames(kind):
    """The kind's frames in sheet order and per family its (first frame,
    frame count), with the fallbacks resolved."""
    frames = []
    span = {}
    for fam in FAMILIES:
        if fam in HAS[kind]:
            recs = family_records(kind, fam)
            span[fam] = (len(frames), len(recs))
            frames += recs
    for fam in FAMILIES:
        f = fam
        while f not in span:
            f = FALLBACK[f]
        span[fam] = span[f]
    return frames, span


def read_glb(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + n])
    off = 20 + n
    bn = struct.unpack("<I", b[off:off + 4])[0]
    blob = b[off + 8:off + 8 + bn]
    return j, blob


def read_atlas(path):
    """The model's atlas in linear light, rgb, its team mask, and the
    cloth's hue: the cloth's mean colour over its mean grey, scaled so its
    largest channel is 1, and that scale."""
    j, blob = read_glb(path)
    img = j["images"][0]
    bv = j["bufferViews"][img["bufferView"]]
    start = bv.get("byteOffset", 0)
    a = np.asarray(Image.open(io.BytesIO(blob[start:start + bv["byteLength"]])).convert("RGBA")).astype(np.float32) / 255.0
    rgb = srgb_to_lin(a[..., :3]).astype(np.float32)
    m = a[..., 3]
    cloth = (rgb * m[..., None]).reshape(-1, 3).sum(axis=0)
    hue = cloth / cloth.mean()
    scale = float(hue.max())
    return rgb, m.copy(), scale, hue / scale


def srgb_to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def write_bake_input(path):
    rows = []
    for k, kind in enumerate(KINDS):
        frames, _ = kind_frames(kind)
        for rec, time in frames:
            rows.append([float(k), time, 0.0, 0.0] + rec)
    np.asarray(rows, dtype=np.float32).tofile(path)


def read_bake_output(path):
    """Per kind: placed positions (frames, corners, 3), normals, the atlas
    coordinates (corners, 2) and the part ids (corners)."""
    b = open(path, "rb").read()
    off = 0
    out = []
    for kind in KINDS:
        corners, records = struct.unpack("<II", b[off:off + 8])
        off += 8
        a = np.frombuffer(b, dtype=np.float32, count=records * corners * 8, offset=off).reshape(records, corners, 2, 4)
        off += records * corners * 32
        frames, _ = kind_frames(kind)
        assert records == len(frames), f"{kind}: {records} records posed, {len(frames)} asked"
        uvbits = a[0, :, 0, 3].copy().view(np.uint32)
        uv = np.stack([(uvbits & 0xFFFF) / 65535.0, (uvbits >> 16) / 65535.0], axis=1)
        out.append((a[:, :, 0, :3].astype(np.float64), a[:, :, 1, :3].astype(np.float64), uv, a[0, :, 1, 3].copy()))
    return out


def view_basis(az, el):
    """Toward the camera, its right and its up, in the soldier's frame
    (his facing +Z, unit_pose.wgsl `place`)."""
    d = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
    right = np.array([d[2], 0.0, -d[0]])
    right /= np.linalg.norm(right)
    up = np.cross(right, -d)
    return d, right, up


@numba.njit(cache=True)
def raster(sx, sy, sz, nrm, uv, w, h, atlas_rgb, atlas_mask, out_rgb, out_mask, out_nrm, out_cov, zbuf):
    """Triangles (corner triples) in pixel space onto the buffers: the
    nearest (largest sz) wins. Colour and mask from the atlas, bilinear;
    the normal blended from the corners as the vertex light is."""
    ah, aw = atlas_mask.shape
    for t in range(0, sx.shape[0], 3):
        x0, x1, x2 = sx[t], sx[t + 1], sx[t + 2]
        y0, y1, y2 = sy[t], sy[t + 1], sy[t + 2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        lo_x = max(int(math.floor(min(x0, x1, x2))), 0)
        hi_x = min(int(math.ceil(max(x0, x1, x2))), w - 1)
        lo_y = max(int(math.floor(min(y0, y1, y2))), 0)
        hi_y = min(int(math.ceil(max(y0, y1, y2))), h - 1)
        for py in range(lo_y, hi_y + 1):
            cy = py + 0.5
            for px in range(lo_x, hi_x + 1):
                cx = px + 0.5
                a = ((y1 - y2) * (cx - x2) + (x2 - x1) * (cy - y2)) / den
                b = ((y2 - y0) * (cx - x2) + (x0 - x2) * (cy - y2)) / den
                c = 1.0 - a - b
                if a < 0.0 or b < 0.0 or c < 0.0:
                    continue
                z = a * sz[t] + b * sz[t + 1] + c * sz[t + 2]
                if z <= zbuf[py, px]:
                    continue
                zbuf[py, px] = z
                u = a * uv[t, 0] + b * uv[t + 1, 0] + c * uv[t + 2, 0]
                v = a * uv[t, 1] + b * uv[t + 1, 1] + c * uv[t + 2, 1]
                fx = (u % 1.0) * aw - 0.5
                fy = (v % 1.0) * ah - 0.5
                ix = int(math.floor(fx))
                iy = int(math.floor(fy))
                tx = fx - ix
                ty = fy - iy
                for ch in range(4):
                    acc = 0.0
                    for dy in range(2):
                        for dx in range(2):
                            qx = min(max(ix + dx, 0), aw - 1)
                            qy = min(max(iy + dy, 0), ah - 1)
                            wgt = (tx if dx == 1 else 1.0 - tx) * (ty if dy == 1 else 1.0 - ty)
                            if ch < 3:
                                acc += wgt * atlas_rgb[qy, qx, ch]
                            else:
                                acc += wgt * atlas_mask[qy, qx]
                    if ch < 3:
                        out_rgb[py, px, ch] = acc
                    else:
                        out_mask[py, px] = acc
                for ch in range(3):
                    out_nrm[py, px, ch] = a * nrm[t, ch] + b * nrm[t + 1, ch] + c * nrm[t + 2, ch]
                out_cov[py, px] = 1.0


def draw_cell(pos, nrm, uv, view, atlas):
    """One frame from one view, in a cell just large enough for it: the
    cell's box in metres from his centre (x0, x1 along the camera's right,
    y0, y1 along its up), and (rgb outside the cloth, cloth grey, normal,
    cut-out) at the cell's size, linear values."""
    az, el = view
    d, right, up = view_basis(az, el)
    x = pos @ right
    y = pos @ up
    # The box on the pixel grid, the margin around it, grown to whole
    # multiples of CELL_ALIGN about its middle.
    px0 = math.floor(x.min() * PX_PER_M) - MARGIN
    px1 = math.ceil(x.max() * PX_PER_M) + MARGIN
    py0 = math.floor(y.min() * PX_PER_M) - MARGIN
    py1 = math.ceil(y.max() * PX_PER_M) + MARGIN
    cw, ch = align(px1 - px0), align(py1 - py0)
    px0 -= (cw - (px1 - px0)) // 2
    py0 -= (ch - (py1 - py0)) // 2
    box = (px0 / PX_PER_M, (px0 + cw) / PX_PER_M, py0 / PX_PER_M, (py0 + ch) / PX_PER_M)
    w, h = cw * SS, ch * SS
    k = PX_PER_M * SS
    sx = (x - box[0]) * k
    sy = (box[3] - y) * k
    sz = pos @ d
    rgb = np.zeros((h, w, 3))
    mask = np.zeros((h, w))
    n = np.zeros((h, w, 3))
    cov = np.zeros((h, w))
    zbuf = np.full((h, w), -1e9)
    raster(sx, sy, sz, nrm, uv, w, h, atlas[0], atlas[1], rgb, mask, n, cov, zbuf)
    grey = rgb.mean(axis=2)
    plain = rgb * (1.0 - mask[..., None])
    cloth = grey * mask * atlas[2]

    def down(a):
        return a.reshape(ch, SS, cw, SS, *a.shape[2:]).sum(axis=(1, 3))
    c = down(cov)
    inv = 1.0 / np.maximum(c, 1e-9)
    return box, (
        down(plain * cov[..., None]) * inv[..., None],
        down(cloth * cov) * inv,
        down(n * cov[..., None]) * inv[..., None],
        c / (SS * SS),
    )


def dilate(values, filled):
    """Fill every empty pixel with the mean of its filled neighbours, ring
    by ring outward, so the mips have no fringe."""
    values = values.copy()
    filled = filled.copy()
    while not filled.all():
        acc = np.zeros_like(values)
        cnt = np.zeros(filled.shape)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            f = np.zeros_like(filled)
            v = np.zeros_like(values)
            sy0, sy1 = max(dy, 0), filled.shape[0] + min(dy, 0)
            sx0, sx1 = max(dx, 0), filled.shape[1] + min(dx, 0)
            f[sy0:sy1, sx0:sx1] = filled[sy0 - dy:sy1 - dy, sx0 - dx:sx1 - dx]
            v[sy0:sy1, sx0:sx1] = values[sy0 - dy:sy1 - dy, sx0 - dx:sx1 - dx]
            acc += v * f.reshape(f.shape + (1,) * (values.ndim - 2))
            cnt += f
        grow = ~filled & (cnt > 0)
        if not grow.any():
            break
        values[grow] = acc[grow] / cnt[grow].reshape(-1, *([1] * (values.ndim - 2)))
        filled |= grow
    return values


def align(v):
    return int(math.ceil(v / CELL_ALIGN) * CELL_ALIGN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", help="the game binary, or a launcher that runs it with the environment it is given")
    ap.add_argument("--work", default="tmp/sprite-bake")
    ap.add_argument("--units", default="assets/units")
    ap.add_argument("--out", default="assets/units/sprites.png")
    ap.add_argument("--bake-done", action="store_true", help="read the placed corners already in --work")
    ap.add_argument("--env", nargs="*", default=[], help="KEY=VALUE pairs for the game")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    work = os.path.abspath(args.work)
    if not args.bake_done:
        if not args.game:
            sys.exit("--game is needed to pose the frames (or --bake-done)")
        write_bake_input(os.path.join(work, "bake_in.bin"))
        out = os.path.join(work, "bake_out.bin")
        if os.path.exists(out):
            os.replace(out, out + ".old")
        env = dict(os.environ, FL_SPRITE_BAKE=work)
        env["FL_SPRITE"] = "0"
        for pair in args.env:
            key, value = pair.split("=", 1)
            env[key] = value
        subprocess.run([args.game], env=env, check=False)
        if not os.path.exists(out):
            sys.exit(f"the game wrote no {out}")
    posed = read_bake_output(os.path.join(work, "bake_out.bin"))

    views = [(TAU * d / DIRECTIONS, math.radians(e)) for e in ELEVATIONS for d in range(DIRECTIONS)]
    kinds = []
    cells = []
    for k, kind in enumerate(KINDS):
        pos, nrm, uv, part = posed[k]
        frames, span = kind_frames(kind)
        atlas = read_atlas(os.path.join(args.units, f"{kind}.glb"))
        first = len(cells)
        for view in views:
            for f in range(len(frames)):
                box, planes = draw_cell(pos[f], nrm[f], uv, view, atlas)
                cells.append(dict(kind=k, box=box, planes=planes))
        kinds.append(dict(name=kind, frames=len(frames), span=span, first=first, cloth=atlas[3]))
        sizes = [c["planes"][3].shape for c in cells[first:]]
        print(f"{kind}: {len(frames)} frames x {len(views)} views, cells {min(s[1] for s in sizes)}..{max(s[1] for s in sizes)} x {min(s[0] for s in sizes)}..{max(s[0] for s in sizes)} px")

    # Shelves: the cells by height, tallest first, left to right in rows
    # across a half as wide as the power of two that keeps the sheet near
    # square.
    total = sum(c["planes"][3].size for c in cells)
    half_w = 1 << int(math.ceil(math.log2(math.sqrt(total / 2.0) * 1.05)))
    order = sorted(range(len(cells)), key=lambda i: -cells[i]["planes"][3].shape[0])
    x = y = shelf = 0
    for i in order:
        ch, cw = cells[i]["planes"][3].shape
        if x + cw > half_w:
            x, y, shelf = 0, y + shelf, 0
        cells[i]["at"] = (x, y)
        x += cw
        shelf = max(shelf, ch)
    height = align(y + shelf)
    width = 2 * half_w
    sheet = np.zeros((height, width, 4), dtype=np.uint8)
    for c in cells:
        plain, cloth, n, cov = c["planes"]
        ch, cw = cov.shape
        cx, cy = c["at"]
        filled = cov > 0
        if not filled.any():
            filled[0, 0] = True
        plain = dilate(plain, filled)
        cloth = dilate(cloth, filled)
        n = dilate(n, filled)
        left = np.concatenate([lin_to_srgb(plain), cov[..., None]], axis=2)
        right = np.concatenate([lin_to_srgb(np.clip(n * 0.5 + 0.5, 0, 1)), np.clip(cloth, 0, 1)[..., None]], axis=2)
        sheet[cy:cy + ch, cx:cx + cw] = (np.clip(left, 0, 1) * 255 + 0.5).astype(np.uint8)
        sheet[cy:cy + ch, half_w + cx:half_w + cx + cw] = (np.clip(right, 0, 1) * 255 + 0.5).astype(np.uint8)
    layout = dict(
        width=width, height=height, px_per_m=PX_PER_M, elevations=ELEVATIONS, directions=DIRECTIONS,
        families=FAMILIES, speeds=dict(stand_below=STAND_BELOW, gaits=GAIT_SPEEDS, switch=GAIT_SWITCH),
        kinds=[],
    )
    for kd in kinds:
        mine = cells[kd["first"]:kd["first"] + kd["frames"] * len(views)]
        layout["kinds"].append(dict(
            name=kd["name"], frames=kd["frames"],
            families=[list(kd["span"][f]) for f in FAMILIES],
            cloth=[round(float(v), 5) for v in kd["cloth"]],
            # Per cell: its rect in the left half, pixels (x, y, width,
            # height), and its box in metres (x0, x1, y0, y1).
            cells=[[c["at"][0], c["at"][1], c["planes"][3].shape[1], c["planes"][3].shape[0]]
                   + [round(v, 5) for v in c["box"]] for c in mine],
        ))
    Image.fromarray(sheet, "RGBA").save(args.out, optimize=True)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(layout, f, separators=(",", ":"))
        f.write("\n")
    raw = width * height * 4
    print(f"sheet {width}x{height}: {raw} bytes a level, {raw * 4 // 3} with its mips; file {os.path.getsize(args.out)} bytes")


if __name__ == "__main__":
    main()
