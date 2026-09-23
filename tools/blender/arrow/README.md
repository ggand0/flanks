# Arrow projectile

A 0.76 m medieval arrow in one mesh named `L0`: 18 triangles for the wooden shaft and carved V nock, 6 for the steel bodkin and 4 for each of three solid fletching vanes. Total: 36 triangles. The five-sided shaft has radial shading normals. The origin is the shaft midpoint; +Y is up and the point faces +Z. Shaft ends are at Z = -0.350 and +0.350 m; the point reaches +0.410 m.

The GLB uses one opaque, culled material and no textures. Vertex colour is `COLOR_0` VEC4 with alpha 0 throughout. Every `TEXCOORD_1` value is `(7, 0)`. The ivory fletching supplies the light rear silhouette. Every component has a closed boundary, including the vanes and nock.

```sh
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/arrow/build_arrow.py -- --out assets_dev/arrow/arrow.glb
python3 tools/blender/glb_inspect.py assets_dev/arrow/arrow.glb
python3 tools/blender/inspect_surfaces.py assets_dev/arrow/arrow.glb
python3 tools/blender/arrow/verify.py assets_dev/arrow/arrow.glb
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/arrow/render_review.py -- assets_dev/arrow/arrow.glb
python3 tools/blender/arrow/compose_review.py assets_dev/arrow
```

The build also writes a Blender scene and component manifest. The verifier checks the exported bytes, channels, dimensions and each closed component. Review renders include side, top, rear, head, fletching, nock, 10-pixel and soldier-scale views, plus the existing three-box mesh at the same scale. Open `assets_dev/arrow/review.html` for the combined sheet. Rendering the scale reference reads `assets/units/archer.glb`; the arrow build is independent of soldier assets. Python verification and composition require NumPy and Pillow.
