# Unit asset builds

These scripts reproduce original unit geometry and textures for FLANKS. They run in a fresh **background** Blender process and write generated files to ignored `assets_dev/` directories. They never overwrite `assets/units/` automatically or connect to a live Blender session.

## Approved knight L0

`assets/units/knight.glb` is the textured knight these scripts build. It contains one L0 mesh: 2,716 triangles, 1.80 m, five rigid parts, one opaque material and one embedded 2048² RGBA atlas. Authored L1 to L3 are still to come.

Tested with Blender 5.2.2 LTS, Python 3.10, NumPy and Pillow. Generate source tiles with system Python, then bake/export/render in Blender:

```bash
python3 tools/blender/knight/make_materials.py
blender --background --factory-startup --threads 4 --python-exit-code 1 \
  --python tools/blender/knight/build_knight_textured.py
python3 tools/blender/glb_inspect.py assets_dev/knight/rebuild/knight_textured.glb
python3 tools/blender/knight/check_texture.py
python3 tools/blender/knight/compose_review.py
```

Default output is `assets_dev/knight/rebuild/`, including the GLB, external atlas, authoring Blend, measurements and review renders. For another destination, pass `--out <directory>/texture_sources` to the tile generator and `-- --out-dir <directory>` after the Blender script. The two postprocessors take `--out-dir <directory>` directly. Use `-- --reuse-bake` only when geometry, UVs and texture sources have not changed.

`geometry_knight.py` holds the base geometry; `revision_geometry.py` replaces its helm, mail skirt and boots with the current ones. `make_materials.py` creates deterministic original material tiles. `build_knight_textured.py` bakes color/local AO/team mask, uses the required export call, checks the GLB bytes, then re-imports that file for all renders. `check_texture.py` checks the actual embedded image and exported part coverage.

A rebuild matches the committed GLB in every exported mesh attribute and node.

## Renderer contract

- +Y up, +Z forward, origin on the ground. The engine handles display scale and any anatomical mirroring.
- `TEXCOORD_0`: packed atlas UVs. `TEXCOORD_1`: `(part ID, pivot height in metres)`; do not invert the height again.
- IDs: body 0, weapon arm 1, left leg 2, right leg 3, shield arm 5, the sword 8.
- Pivot nodes: hips `(±0.115, 0.910, 0)`; weapon shoulder `(-0.224, 1.435, 0)`; shield shoulder `(0.224, 1.435, 0)`; the sword's grip `pivot_weapon` `(-0.40, 1.084, 0.21)`. The `joint_elbow` empty at `(-0.337, 1.215, 0.012)` is where the engine bends the weapon arm.
- `COLOR_0`: VEC4, white RGB and preserved vertex team alpha. Use atlas alpha for the textured path, including untinted woven trim within the cloth.
- Atlas RGB is sRGB color with local AO; alpha is a linear team mask, **not transparency**. Keep the material opaque and retain RGB where the mask is zero.

In linear color space, before lighting:

```text
base = atlas.rgb * mix(vec3(1), team_tint, atlas.a)
```

The shared atlas uses 16 MiB as RGBA8, about 21.3 MiB with full mips. There are no normal maps or animation clips. Geometry, patterns and material tiles were generated for this project; no external model or reference-image pixels are included.
