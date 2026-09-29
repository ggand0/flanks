# Upright shrub A

Build a 1.7 m tall, 2.2 m wide, 1.85 m deep multi-stem shrub. Seven stems and fourteen forks carry 840 leaf sprays. Near, middle and far levels contain 2,548, 868 and 4 triangles. Foliage stays close to the ground; the outer tips form an uneven crown.

The build extracts the project's original foliage and bark textures from `assets/vegetation/shrub_b_sandbox.glb`. Fetch that file through Git LFS first. Blender 5.2 and NumPy are required; raw verification also uses Pillow. No private authoring scene or external texture is needed.

From the repository root:

```sh
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/shrub_a/build_shrub.py
python3 tools/blender/glb_inspect.py assets_dev/vegetation/shrub_a_rebuild/shrub_a/tree.glb
python3 tools/blender/shrub_a/verify_shrub.py assets_dev/vegetation/shrub_a_rebuild/shrub_a/tree.glb
```

Use `-- --out <directory>` after the Blender script to change the output location. It writes a GLB, Blender scene, reports, extracted source textures, far atlases and previews; it does not replace the shipped asset. `--texture-source <GLB>` selects another B shrub file with the same foliage/bark material contract.

`build_shrub.py` defines branch paths and deterministic leaf placement. `shrub_mesh.py` provides tubes, folded sprays, materials and colour/volume-normal baking. The middle mesh keeps the wood and 210 enlarged sprays; far detail uses two crossed planes. The raw verifier checks bounds, budgets, finite attributes, material modes and closed bark winding.
