# Ground textures

The source textures are released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) by [Poly Haven](https://polyhaven.com/license).

| Layer | Source | Maps |
|---|---|---|
| Pasture | [Grass Ground](https://polyhaven.com/a/grass_ground) | 1K diffuse, OpenGL normal, roughness |
| Stony soil | [Grass Path 2](https://polyhaven.com/a/grass_path_2) | 1K diffuse, OpenGL normal, roughness |
| Exposed earth | [Brown Mud Dry](https://polyhaven.com/a/brown_mud_dry) | 1K diffuse, OpenGL normal, roughness |

`sources.json` records exact download URLs, sizes and checksums. `prepare.py` converts the maps to KTX2 with a full mip chain and Zstandard compression. Color uses sRGB; normal XYZ and roughness A use linear channels. Source JPEGs are not required at runtime.

The pasture color map contains neutral detail centered on 0.5 in linear space. The preparation script removes broad color and lighting patches with a periodic filter; the terrain coverage field supplies those scales. The shader preserves the detail's variance when blending rotated samples and uses separate turf and close grass scales.
