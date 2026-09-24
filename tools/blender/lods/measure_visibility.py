"""Measure visible atlas colour and team coverage from eight 50 degree views."""
import io
import json
import math
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import glb_inspect


def raster(doc, blob, mesh, azimuth, pixels, size=192, face_indices=False):
    primitive = mesh["primitives"][0]
    get = lambda index: np.array(glb_inspect.accessor_values(doc, blob, index)[1])
    pos = get(primitive["attributes"]["POSITION"])
    uv = get(primitive["attributes"]["TEXCOORD_0"])
    ids = get(primitive["attributes"]["TEXCOORD_1"])[:, 0].astype(int)
    tri = get(primitive["indices"]).astype(int).reshape(-1, 3)
    az, el = np.radians([azimuth, 50])
    toward = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    right = np.array([np.cos(az), 0, -np.sin(az)])
    up = np.cross(toward, right)
    p = pos @ np.array([right, up, toward]).T
    p[:, :2] = (p[:, :2] - [0, 0.56]) * (size / 2.25) + size / 2
    zbuffer = np.full((size, size), -np.inf)
    rgba = np.zeros((size, size, 4))
    part = np.full((size, size), -1)
    height, width = pixels.shape[:2]
    visible_faces = np.full((size, size), -1)
    for face_index, face in enumerate(tri):
        a, b, c = p[face]
        cross = np.cross(b - a, c - a)
        if cross[2] <= 1e-10:
            continue
        low = np.maximum(0, np.floor(p[face, :2].min(axis=0)).astype(int))
        high = np.minimum(size - 1, np.ceil(p[face, :2].max(axis=0)).astype(int))
        if np.any(high < low):
            continue
        x, y = np.meshgrid(
            np.arange(low[0], high[0] + 1) + 0.5, np.arange(low[1], high[1] + 1) + 0.5
        )
        ab, ac = b - a, c - a
        u = ((x - a[0]) * ac[1] - (y - a[1]) * ac[0]) / cross[2]
        v = (ab[0] * (y - a[1]) - ab[1] * (x - a[0])) / cross[2]
        z = a[2] + ab[2] * u + ac[2] * v
        window = np.s_[low[1] : high[1] + 1, low[0] : high[0] + 1]
        keep = (u >= 0) & (v >= 0) & (u + v <= 1) & (z > zbuffer[window])
        if not keep.any():
            continue
        tex = (
            uv[face[0]]
            + u[:, :, None] * (uv[face[1]] - uv[face[0]])
            + v[:, :, None] * (uv[face[2]] - uv[face[0]])
        )
        tx, ty = tex[:, :, 0] * width - 0.5, tex[:, :, 1] * height - 0.5
        ix, iy = np.floor(tx).astype(int), np.floor(ty).astype(int)
        fx, fy = (tx - ix)[:, :, None], (ty - iy)[:, :, None]
        sample = np.zeros((*u.shape, 4))
        for dx, dy, weight in [
            (0, 0, (1 - fx) * (1 - fy)),
            (1, 0, fx * (1 - fy)),
            (0, 1, (1 - fx) * fy),
            (1, 1, fx * fy),
        ]:
            sample += (
                pixels[np.clip(iy + dy, 0, height - 1), np.clip(ix + dx, 0, width - 1)]
                * weight
            )
        zbuffer[window][keep] = z[keep]
        rgba[window][keep] = sample[keep]
        part[window][keep] = ids[face[0]]
        visible_faces[window][keep] = face_index
    return (rgba, part, visible_faces) if face_indices else (rgba, part)


def main():
    from PIL import Image
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("glb", type=Path)
    path = parser.parse_args().glb
    doc, blob = glb_inspect.load(str(path))
    view = doc["bufferViews"][doc["images"][0]["bufferView"]]
    start = view.get("byteOffset", 0)
    pixels = (
        np.asarray(
            Image.open(io.BytesIO(blob[start : start + view["byteLength"]]))
        ).astype(float)
        / 255
    )
    pixels[:, :, :3] = np.where(
        pixels[:, :, :3] <= 0.04045,
        pixels[:, :, :3] / 12.92,
        ((pixels[:, :, :3] + 0.055) / 1.055) ** 2.4,
    )
    report = {}
    for mesh in doc["meshes"]:
        collected = []
        per_part = {}
        views = {}
        for azimuth in range(0, 360, 45):
            rgba, parts = raster(doc, blob, mesh, azimuth, pixels)
            visible = rgba[parts >= 0]
            collected.append(visible)
            views[azimuth] = {
                "pixels": len(visible),
                "team_share": float(visible[:, 3].mean()),
                "linear_rgb": visible[:, :3].mean(axis=0).tolist(),
            }
            for part in np.unique(parts[parts >= 0]):
                per_part.setdefault(str(part), []).append(rgba[parts == part])
        mean = np.concatenate(collected).mean(axis=0)
        report[mesh["name"]] = {
            "linear_rgb": mean[:3].tolist(),
            "team_share": float(mean[3]),
            "views": views,
            "parts": {
                key: np.concatenate(value).mean(axis=0).tolist()
                for key, value in per_part.items()
            },
        }
    (path.parent / "visibility.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: {k: v for k, v in value.items() if k != "views"}
                for key, value in report.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
