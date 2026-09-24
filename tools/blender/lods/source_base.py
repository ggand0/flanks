"""Recover an unchanged L0/L2 source after authored levels have been installed."""

import hashlib
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glb_inspect


def resolve_source(path, out_dir, manifest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() == manifest["source_sha256"]:
        return path
    original_json = manifest["source_json_chunk"].encode("utf-8")
    original = json.loads(original_json)
    current, blob = glb_inspect.load(path)
    for key in ["materials", "images", "textures", "samplers", "asset"]:
        assert current.get(key) == original.get(key), f"Source changed: {key}"
    for key in ["meshes", "nodes", "accessors", "bufferViews"]:
        assert (
            current[key][: len(original[key])] == original[key]
        ), f"Source changed: {key}"
    blob = blob[: manifest["source_bin_length"]]
    restored = (
        struct.pack("<III", 0x46546C67, 2, 28 + len(original_json) + len(blob))
        + struct.pack("<II", len(original_json), 0x4E4F534A)
        + original_json
        + struct.pack("<II", len(blob), 0x004E4942)
        + blob
    )
    assert (
        hashlib.sha256(restored).hexdigest() == manifest["source_sha256"]
    ), "Original source geometry or atlas changed; update its component labels"
    destination = out_dir / "source" / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(restored)
    return destination
