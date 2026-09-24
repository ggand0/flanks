"""Read a GLB's attribute table without Blender: what actually shipped.

    python3 glb_inspect.py out/smoke.glb
"""

import json
import os
import struct
import sys

COMPONENT = {5126: ("f", 4), 5123: ("H", 2), 5121: ("B", 1), 5125: ("I", 4)}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def load(path):
    data = open(path, "rb").read()
    _, _, total = struct.unpack_from("<III", data, 0)
    off, chunks = 12, {}
    while off < total:
        clen, ctype = struct.unpack_from("<II", data, off)
        tag = struct.pack("<I", ctype).decode().strip("\x00")
        chunks[tag] = data[off + 8: off + 8 + clen]
        off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)
    return json.loads(chunks["JSON"].decode()), chunks.get("BIN")


def accessor_values(j, bin_, idx):
    acc = j["accessors"][idx]
    bv = j["bufferViews"][acc["bufferView"]]
    ncomp = NCOMP[acc["type"]]
    code, size = COMPONENT[acc["componentType"]]
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or ncomp * size
    norm = 65535.0 if acc["componentType"] == 5123 else (255.0 if acc["componentType"] == 5121 else 1.0)
    if not acc.get("normalized", False):
        norm = 1.0
    vals = [struct.unpack_from("<" + code * ncomp, bin_, base + i * stride) for i in range(acc["count"])]
    return acc, [tuple(c / norm for c in v) for v in vals]


def dump(paths):
    for path in paths:
        j, bin_ = load(path)
        print(f"== {os.path.basename(path)}")
        for mesh in j["meshes"]:
            for prim in mesh["primitives"]:
                for name, idx in sorted(prim["attributes"].items()):
                    acc, vals = accessor_values(j, bin_, idx)
                    uniq = sorted({tuple(round(c, 3) for c in v) for v in vals})
                    print(f"   {name:<12} {acc['type']:<7} n={acc['count']:<4} "
                          f"distinct={len(uniq):<3} {uniq[:4]}")
        print(f"   nodes={[n.get('name') for n in j['nodes']]}")
        print(f"   materials={[(m.get('name'), m.get('alphaMode')) for m in j.get('materials', [])]}")


if __name__ == "__main__":
    dump(sys.argv[1:])
