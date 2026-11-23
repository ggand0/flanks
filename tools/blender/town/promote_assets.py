"""Install the wall, tower, stair and gatehouse models and the town houses
into assets/town/.

Every source GLB embeds its whole atlas (4096 by 4096 PNG, about 30 MB),
so the nine files would carry two atlases nine times. This copies each
atlas once and rewrites each GLB to reference it by file name instead of
embedding it: the image's buffer view is dropped and the binary buffer
compacted. Geometry, UVs, normals, materials and node transforms are
copied byte for byte. Each embedded image must match its atlas file
exactly (SHA-256), so a GLB and an atlas from different iterations cannot
be paired.

Run from the game root:

    python3 tools/blender/town/promote_assets.py
"""

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WALL = Path('assets_dev/wall/gatehouse_textured_v1')
TOWN = Path('assets_dev/town/merchant_textured_v1')

# Source file, installed name.
MODELS = [
    (WALL / 'wall_unit.glb', 'wall_unit.glb'),
    (WALL / 'wall_unit_landing.glb', 'wall_unit_landing.glb'),
    (WALL / 'stair_ramp.glb', 'stair_ramp.glb'),
    (WALL / 'drum_tower.glb', 'drum_tower.glb'),
    (WALL / 'gatehouse.glb', 'gatehouse.glb'),
    (TOWN / 'humble_house_shared.glb', 'humble_house.glb'),
    (TOWN / 'market_house_shared.glb', 'market_house.glb'),
    (TOWN / 'workshop_house_shared.glb', 'workshop_house.glb'),
    (TOWN / 'merchant_house.glb', 'merchant_house.glb'),
]
ATLASES = [
    WALL / 'wall_stone_atlas.png',
    WALL / 'gate_wood_atlas.png',
    TOWN / 'town_colour_atlas.png',
]


def read_glb(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from('<III', data, 0)
    assert magic == 0x46546C67 and version == 2 and length == len(data), path
    json_len, json_type = struct.unpack_from('<II', data, 12)
    assert json_type == 0x4E4F534A, path
    doc = json.loads(data[20:20 + json_len])
    bin_len, bin_type = struct.unpack_from('<II', data, 20 + json_len)
    assert bin_type == 0x004E4942, path
    start = 28 + json_len
    return doc, data[start:start + bin_len]


def write_glb(path, doc, blob):
    text = json.dumps(doc, separators=(',', ':')).encode()
    text += b' ' * (-len(text) % 4)
    blob += b'\0' * (-len(blob) % 4)
    total = 12 + 8 + len(text) + 8 + len(blob)
    out = struct.pack('<III', 0x46546C67, 2, total)
    out += struct.pack('<II', len(text), 0x4E4F534A) + text
    out += struct.pack('<II', len(blob), 0x004E4942) + blob
    path.write_bytes(out)


def strip_images(doc, blob, atlas_hashes):
    """Point every image at its atlas file and drop its bytes."""
    views = doc['bufferViews']
    dropped = set()
    for image in doc['images']:
        view = views[image['bufferView']]
        start = view.get('byteOffset', 0)
        digest = hashlib.sha256(blob[start:start + view['byteLength']]).hexdigest()
        name = image['name'] + '.png'
        if atlas_hashes.get(name) != digest:
            raise SystemExit(f"embedded {name} does not match the atlas file")
        dropped.add(image.pop('bufferView'))
        image['uri'] = name
        image.pop('mimeType', None)
    remap, kept, out = {}, [], bytearray()
    for i, view in enumerate(views):
        if i in dropped:
            continue
        start = view.get('byteOffset', 0)
        out += b'\0' * (-len(out) % 4)
        view = dict(view, byteOffset=len(out))
        out += blob[start:start + view['byteLength']]
        remap[i] = len(kept)
        kept.append(view)
    doc['bufferViews'] = kept
    for accessor in doc['accessors']:
        accessor['bufferView'] = remap[accessor['bufferView']]
    doc['buffers'] = [{'byteLength': len(out)}]
    return bytes(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='assets/town')
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for atlas in ATLASES:
        source = ROOT / atlas
        hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source, out / source.name)
        print(f"{source.name}: {source.stat().st_size} bytes")
    for source, name in MODELS:
        doc, blob = read_glb(ROOT / source)
        blob = strip_images(doc, blob, hashes)
        write_glb(out / name, doc, blob)
        print(f"{name}: {(out / name).stat().st_size} bytes, from {source}")


if __name__ == '__main__':
    main()
