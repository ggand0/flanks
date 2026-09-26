# Ground textures

The three Poly Haven source textures are released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) by [Poly Haven](https://polyhaven.com/license).

| Layer | Source | Maps |
|---|---|---|
| Pasture | [Grass Ground](https://polyhaven.com/a/grass_ground) | 1K diffuse, OpenGL normal, roughness; 2.51 m square |
| Stony soil | [Grass Path 2](https://polyhaven.com/a/grass_path_2) | 1K diffuse, OpenGL normal, roughness |
| Exposed earth | [Brown Mud Dry](https://polyhaven.com/a/brown_mud_dry) | 1K diffuse, OpenGL normal, roughness |

`sources.json` records exact download URLs, sizes and checksums. `prepare.py` converts the maps to KTX2 with a full mip chain and Zstandard compression; `--layers pasture` repacks just the grass. Color uses sRGB; normal XYZ and roughness A use linear channels. Source JPEGs are not required at runtime.

The classic grassland uses `pasture_natural_color.ktx2`, which preserves the original Grass Ground colors at 2.51 m per tile. `grassland_layout.png` is original artwork created for this project. It covers the battlefield once, supplying the larger grass and exposed-earth arrangement. `grassland_layout.prompt.txt` describes the source artwork, and `grassland_layout.source.json` records its size and checksum. Run `python3 assets/terrain/prepare_layout.py --toktx /path/to/toktx` to pack it. RGB stores the artwork in sRGB, and alpha selects grass or exposed-earth detail. Alpha does not control distant color. Native-scale scan detail is normalized by its linear mean so mip filtering preserves the layout's color.

The river material uses `pasture_color.ktx2`, containing neutral detail centered on 0.5 in linear space. Its preparation removes broad color and lighting patches with a periodic filter, and a procedural coverage field supplies those scales. Grass Path 2 is used only by the river material.
