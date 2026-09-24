# Spearman texture inputs

These PNG tiles were generated for this project. They contain no downloaded
model textures or reference-image pixels. They are the source tiles used by
the textured v2 spearman and retained for the v3 arm revision.

`materials.py` uses these files to bake the single embedded RGBA atlas. Its
alpha channel stores team tint, not opacity. Committing the original tiles
keeps the build independent of ignored working directories.
