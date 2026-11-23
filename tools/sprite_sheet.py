"""Draw the placeholder sprite sheet of the far soldiers' cards (FL_SPRITE).

For each soldier kind it rasterizes one level of the unit model
(assets/units/<kind>.glb) from the front, orthographic, with the model's
own atlas: rgb the atlas colour lit by the game's sky and sun terms
(unit_instancing.wgsl `sky_and_sun`, the sun as main.rs `sun_rotation`),
the team mask from the atlas alpha. Four times supersampled, then
averaged down.

The sheet is 8 cells of CW x CH pixels in one row. Cells 0..3 are the
kinds (knight, man-at-arms, spearman, archer): rgb colour with the team
cloth grey, alpha the cut-out. Cells 4..7 are the same kinds' team masks
in alpha (rgb white). The feet stand on the cell's bottom row, the figure
centred, PX_PER_M pixels a metre.

It prints each kind's quad in model metres and its uv rect, the table
`SPRITE_CARDS` in src/unit_glb.rs. Run from the repo root:

    python3 tools/sprite_sheet.py                  # writes assets/units/sprites.png from L0
    python3 tools/sprite_sheet.py <units dir> <out png> [level name]

Needs numpy and Pillow.
"""
import json, struct, sys
import numpy as np
from PIL import Image
import io

KINDS = ["knight", "man_at_arms", "spearman", "archer"]
CW, CH = 128, 256
PX_PER_M = 88.0
SS = 4
SUN = np.array([0.472, 0.682, 0.560])
SUN /= np.linalg.norm(SUN)


def read_glb(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + n])
    off = 20 + n
    bn = struct.unpack("<I", b[off:off + 4])[0]
    blob = b[off + 8:off + 8 + bn]
    return j, blob


def accessor(j, blob, i):
    a = j["accessors"][i]
    bv = j["bufferViews"][a["bufferView"]]
    comps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
    dt = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}[a["componentType"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    size = np.dtype(dt).itemsize
    stride = bv.get("byteStride", comps * size)
    raw = np.frombuffer(blob, dtype=np.uint8, count=stride * (a["count"] - 1) + comps * size, offset=start)
    out = np.empty((a["count"], comps), dtype=dt)
    for k in range(a["count"]):
        out[k] = np.frombuffer(raw[k * stride:k * stride + comps * size].tobytes(), dtype=dt)
    if a.get("normalized") and dt != np.float32:
        out = out.astype(np.float32) / np.iinfo(dt).max
    return out


def srgb_to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def render(path, level):
    j, blob = read_glb(path)
    node = [nd for nd in j["nodes"] if nd.get("name") == level][0]
    img = j["images"][0]
    bv = j["bufferViews"][img["bufferView"]]
    atlas = np.asarray(Image.open(io.BytesIO(blob[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]])).convert("RGBA")).astype(np.float32) / 255.0
    ah, aw = atlas.shape[:2]
    tris = []
    for prim in j["meshes"][node["mesh"]]["primitives"]:
        at = prim["attributes"]
        pos = accessor(j, blob, at["POSITION"]).astype(np.float64)
        nrm = accessor(j, blob, at["NORMAL"]).astype(np.float64)
        uv = accessor(j, blob, at["TEXCOORD_0"]).astype(np.float64)
        idx = accessor(j, blob, prim["indices"]).reshape(-1).astype(np.int64)
        tris.append((pos, nrm, uv, idx))
    allpos = np.concatenate([t[0] for t in tris])
    lo, hi = allpos.min(0), allpos.max(0)
    d = PX_PER_M * SS
    W, H = CW * SS, CH * SS
    # Screen x = model +X (the camera stands at +Z looking at his front), y down from the cell top; the feet (model
    # y = 0) on the bottom row, x = 0 at the middle column.
    col = np.zeros((H, W, 3))
    mask = np.zeros((H, W))
    cov = np.zeros((H, W))
    zbuf = np.full((H, W), -1e9)
    for pos, nrm, uv, idx in tris:
        for t in range(0, len(idx), 3):
            ia, ib, ic = idx[t], idx[t + 1], idx[t + 2]
            P = pos[[ia, ib, ic]]
            sx = W / 2 + P[:, 0] * d
            sy = H - P[:, 1] * d
            x0, x1 = int(max(np.floor(sx.min()), 0)), int(min(np.ceil(sx.max()), W - 1))
            y0, y1 = int(max(np.floor(sy.min()), 0)), int(min(np.ceil(sy.max()), H - 1))
            if x1 < x0 or y1 < y0:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            den = (sy[1] - sy[2]) * (sx[0] - sx[2]) + (sx[2] - sx[1]) * (sy[0] - sy[2])
            if abs(den) < 1e-12:
                continue
            w0 = ((sy[1] - sy[2]) * (xs - sx[2]) + (sx[2] - sx[1]) * (ys - sy[2])) / den
            w1 = ((sy[2] - sy[0]) * (xs - sx[2]) + (sx[0] - sx[2]) * (ys - sy[2])) / den
            w2 = 1 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            z = w0 * P[0, 2] + w1 * P[1, 2] + w2 * P[2, 2]
            yy, xx = ys[inside].astype(int), xs[inside].astype(int)
            zz = z[inside]
            closer = zz > zbuf[yy, xx]
            if not closer.any():
                continue
            yy, xx, zz = yy[closer], xx[closer], zz[closer]
            a, b, c = w0[inside][closer], w1[inside][closer], w2[inside][closer]
            U = a * uv[ia, 0] + b * uv[ib, 0] + c * uv[ic, 0]
            V = a * uv[ia, 1] + b * uv[ib, 1] + c * uv[ic, 1]
            N = a[:, None] * nrm[ia] + b[:, None] * nrm[ib] + c[:, None] * nrm[ic]
            N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
            tx = np.clip((U % 1.0) * aw, 0, aw - 1).astype(int)
            ty = np.clip((V % 1.0) * ah, 0, ah - 1).astype(int)
            tex = atlas[ty, tx]
            sky = 0.30 + 0.20 * (0.5 + 0.5 * N[:, 1])
            sun = 0.65 * np.maximum(N @ SUN, 0.0)
            zbuf[yy, xx] = zz
            # Colour in linear light; the cloth (team mask) grey at its own brightness, so the shader's tint by the
            # team colour gives the cloth its colour as the atlas tint does on a mesh.
            lin = srgb_to_lin(tex[:, :3])
            col[yy, xx] = lin * (sky + sun)[:, None]
            mask[yy, xx] = tex[:, 3]
            cov[yy, xx] = 1.0
    # Average down, colour weighted by coverage.
    def down(a):
        return a.reshape(CH, SS, CW, SS, *a.shape[2:]).mean(axis=(1, 3))
    c = down(cov)
    rgb = down(col * cov[..., None]) / np.maximum(c[..., None], 1e-9)
    m = down(mask * cov) / np.maximum(c, 1e-9)
    return rgb, m, c, lo, hi


def main():
    units = sys.argv[1] if len(sys.argv) > 1 else "assets/units"
    out = sys.argv[2] if len(sys.argv) > 2 else "assets/units/sprites.png"
    level = sys.argv[3] if len(sys.argv) > 3 else "L0"
    sheet = np.zeros((CH, CW * 8, 4))
    for k, name in enumerate(KINDS):
        rgb, m, c, lo, hi = render(f"{units}/{name}.glb", level)
        # The mips average rgb without weighting by alpha, so the colour runs out into the empty pixels around the
        # figure: no dark fringe on a far card.
        filled = c > 0.0
        for _ in range(CW):
            if filled.all():
                break
            acc = np.zeros_like(rgb)
            n = np.zeros(c.shape)
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                f = np.roll(filled, (dy, dx), axis=(0, 1))
                acc += np.roll(rgb, (dy, dx), axis=(0, 1)) * f[..., None]
                n += f
            grow = ~filled & (n > 0)
            rgb[grow] = acc[grow] / n[grow][:, None]
            filled = filled | grow
        sheet[:, k * CW:(k + 1) * CW, :3] = lin_to_srgb(rgb)
        sheet[:, k * CW:(k + 1) * CW, 3] = c
        sheet[:, (4 + k) * CW:(5 + k) * CW, :3] = 1.0
        sheet[:, (4 + k) * CW:(5 + k) * CW, 3] = m * c
        ys, xs = np.nonzero(c > 0.02)
        # Quad in model metres (x left to right, y feet up), and the uv rect over the whole sheet, a pixel of margin.
        px0, px1 = max(xs.min() - 1, 0), min(xs.max() + 2, CW)
        py0, py1 = max(ys.min() - 1, 0), CH
        qx0, qx1 = (px0 - CW / 2) / PX_PER_M, (px1 - CW / 2) / PX_PER_M
        qy0, qy1 = (CH - py1) / PX_PER_M, (CH - py0) / PX_PER_M
        u0, u1 = (k * CW + px0) / (8 * CW), (k * CW + px1) / (8 * CW)
        v0, v1 = py0 / CH, py1 / CH
        print(f"    // {name}: model {lo.round(3)} to {hi.round(3)}")
        print(f"    [{qx0:.4f}, {qx1:.4f}, {qy0:.4f}, {qy1:.4f}, {u0:.5f}, {u1:.5f}, {v0:.5f}, {v1:.5f}],")
    Image.fromarray((np.clip(sheet, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA").save(out)


if __name__ == "__main__":
    main()
