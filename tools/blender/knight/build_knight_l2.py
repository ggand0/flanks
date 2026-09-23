"""Build the knight L2 beside the preserved textured L0.

blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender/knight/build_knight_l2.py -- \
  --out assets_dev/knight/lod_l2_v1/knight.glb

Reads the authoring scene, component manifest and atlas from
assets_dev/knight/textured_v3_visor/. The source GLB's existing data is preserved.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import struct
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'assets_dev/knight/textured_v3_visor'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT / 'tools/blender'))
import geometry_knight as geo
import build_knight_textured as source_build
import glb_inspect


def geometry():
    g = geo.Geometry()
    g.label = 'surcoat'
    outline = [(-1, -.7), (0, -1), (1, -.7), (1, .7), (0, 1), (-1, .7)]
    rings = [(.535, .215, .155), (1.055, .175, .140), (1.478, .177, .110)]
    points = [(x * rx, y * ry, z) for z, rx, ry in rings for x, y in outline]
    faces = [tuple(reversed(range(6))), tuple(range(12, 18))]
    faces += [(r * 6 + i, r * 6 + (i + 1) % 6,
               (r + 1) * 6 + (i + 1) % 6, (r + 1) * 6 + i)
              for r in range(2) for i in range(6)]
    g.add(points, faces, geo.CLOTH, True)
    for i in [2 + r * 6 + side for r in range(2) for side in [2, 5]]:
        g.components[i] = 'mail_hauberk'
        g.colors[i] = geo.MAIL
    g.label = 'mail_hauberk'
    g.loft([(.425, 0, 0, .224, .150), (.66, 0, 0, .217, .145)], 4, geo.MAIL, True)
    g.label = 'mail_coif'
    g.loft([(1.40, 0, 0, .157, .114), (1.59, 0, 0, .09, .09)], 3, geo.MAIL, True)
    g.label = 'helm_shell'
    # The crown stays flat and keeps the source's frontal width.
    outline = [(-1, 0), (-.65, -1), (.65, -1), (1, 0), (.65, 1), (-.65, 1)]
    points = [(x * rx, y * ry, z) for z, rx, ry in
              [(1.513, .108, .117), (1.80, .103, .108)] for x, y in outline]
    faces = [tuple(reversed(range(6))), tuple(range(6, 12))]
    faces += [(i, (i + 1) % 6, (i + 1) % 6 + 6, i + 6) for i in range(6)]
    g.add(points, faces, geo.STEEL)
    for sign, part in [(1, 'leg_l'), (-1, 'leg_r')]:
        g.part, g.label = part, 'boot'
        start = len(g.faces)
        vertex_start = len(g.vertices)
        g.loft([(0, sign * .112, -.074, .065, .184),
                (.12, sign * .112, -.025, .066, .105),
                (.445, sign * .112, .002, .078, .083),
                (.95, sign * .112, 0, .108, .112)], 4, geo.LEATHER, True)
        for ring, coords in enumerate([[(-.044, -.238), (.044, -.238), (.055, .095), (-.055, .095)],
                                      [(-.064, -.083), (.064, -.083), (.057, .075), (-.057, .075)]]):
            for j, (xx, yy) in enumerate(coords):
                g.vertices[vertex_start + ring * 4 + j] = (sign * .112 + xx, yy, ring * .12)
        for ring, (height, width, depth) in enumerate([(.445, .078, .083), (.95, .108, .112)], 2):
            for j, (xx, yy) in enumerate([(-1,-1), (1,-1), (1,1), (-1,1)]):
                g.vertices[vertex_start + ring * 4 + j] = (sign * .112 + xx * width * .80, yy * depth * .80, height)
        for i in range(start + 8, start + 12):
            g.components[i] = 'mail_chausses'
            g.colors[i] = geo.MAIL
        g.components[len(g.faces) - 1] = 'mail_chausses'
    for sign, part in [(-1, 'arm_weapon'), (1, 'arm_shield')]:
        g.part, g.label = part, 'mail_sleeve'
        g.tube([(sign * .208, 0, 1.423), (sign * .337, -.012, 1.215),
                (sign * .4, -.23, 1.084)], [.092, .080, .058], 4, geo.MAIL, True)
        for i in range(len(g.faces) - 5, len(g.faces)):
            g.components[i] = 'leather_cuff'
            g.colors[i] = geo.LEATHER
    g.part, g.label = 'arm_shield', 'heater_face'
    tangent = Vector((.9063, .4226, 0))
    normal = Vector((.4226, -.9063, 0))
    def shield(u, z, depth=0):
        return Vector((.412, -.207, z)) + tangent * u + normal * (
            .057 * (1 - (u / .236) ** 2) + depth)
    border = [(-.219, 1.415), (.219, 1.415), (.236, 1.353),
              (.213, 1.16), (.150, .971), (0, .749), (-.150, .971), (-.213, 1.16), (-.236, 1.353)]
    n = len(border)
    front = [shield(u, z) for u, z in border] + [shield(0, 1.19)]
    m = n + 1
    points = front + [p - normal * .023 for p in front]
    start = len(g.faces)
    faces = [(n, i, (i + 1) % n) for i in range(n)]
    faces += [(n + m, (i + 1) % n + m, i + m) for i in range(n)]
    faces += [(i, i + m, (i + 1) % n + m, (i + 1) % n) for i in range(n)]
    g.add(points, faces, geo.CLOTH, True)
    for i in range(n, 3 * n):
        g.components[start + i] = 'heater_wood' if i < 2 * n else 'heater_rim'
        g.colors[start + i] = geo.LEATHER
    g.label = 'heater_straps'
    for z in [1.22, 1.06]:
        g.tube([shield(-.075, z, -.03), shield(.075, z, -.065)],
               [.015, .015], 3, geo.LEATHER, False)
    g.part, g.label = 'weapon', 'arming_sword'
    x, z = -.4, 1.084
    points = [(x - .026, -.32, z), (x, -.32, z + .006),
              (x + .026, -.32, z), (x, -.32, z - .006),
              (x - .019, -.85, z), (x, -.85, z + .004),
              (x + .019, -.85, z), (x, -.85, z - .004), (x, -1.075, z)]
    g.add(points, [(3, 2, 1, 0)] + [(i, (i + 1) % 4, (i + 1) % 4 + 4, i + 4) for i in range(4)]
          + [(i + 4, (i + 1) % 4 + 4, 8) for i in range(4)], geo.EDGE)
    g.tube([(x, -.15, z), (x, -.32, z)], [.022, .018], 3, geo.LEATHER)
    g.tube([(x - .119, -.295, z), (x + .119, -.295, z)], [.015, .015], 3, geo.EDGE, False)
    return g


def atlas_projection(obj, g, l0):
    """Sample the existing atlas at the mean local colour of each far face."""
    manifest = json.loads((SOURCE / 'source_surfaces.json').read_text())
    face_ids = l0.data.attributes['source_face']
    uv = l0.data.uv_layers['UVMap']
    categories = {}
    for face in l0.data.polygons:
        label = manifest['components'][face_ids.data[face.index].value]
        entries = categories.setdefault(label, [])
        entries.append((list(face.vertices), [uv.data[i].uv.copy() for i in face.loop_indices],
                        face.normal.copy()))
    categories['surcoat'] += categories['belt'] + categories['mail_hauberk']
    categories['helm_shell'] += categories['great_helm']
    categories['boot'] += categories['boot_trim']
    verts = [v.co.copy() for v in l0.data.vertices]
    trees = {label: BVHTree.FromPolygons(verts, [e[0] for e in entries], all_triangles=True)
             for label, entries in categories.items()}
    atlas = bpy.data.images.load(str(SOURCE / 'knight_atlas.png'), check_existing=True)
    atlas.alpha_mode = 'CHANNEL_PACKED'
    width, height = atlas.size
    pixels = np.array(atlas.pixels[:], dtype=np.float32).reshape(height, width, 4)
    # Byte image pixels are encoded RGB; compare and average in linear light.
    rgb = pixels[:, :, :3]
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    pixels = np.concatenate([linear, pixels[:, :, 3:]], axis=2)
    candidates = {}
    mask = pixels[:, :, 3]
    transition_y, transition_x = np.nonzero((mask > .2) & (mask < .98))
    transitions = [((x + .5) / width, (y + .5) / height, pixels[y, x])
                   for x, y in zip(transition_x, transition_y)]
    weights = []
    for a in range(10):
        for b in range(10 - a):
            u, v = (a + 1 / 3) / 10, (b + 1 / 3) / 10
            weights.append((u, v, 1 - u - v))
            if a + b < 9:
                u, v = (a + 2 / 3) / 10, (b + 2 / 3) / 10
                weights.append((u, v, 1 - u - v))
    for label, entries in categories.items():
        samples = []
        for ids, coords, normal in entries:
            for weights_ in weights:
                sample = sum((p * w for p, w in zip(coords, weights_)), Vector((0, 0)))
                px, py = int(sample.x * width), int(sample.y * height)
                px, py = min(width - 1, max(0, px)), min(height - 1, max(0, py))
                samples.append(((px + .5) / width, (py + .5) / height, pixels[py, px]))
        candidates[label] = samples + transitions if label in ['surcoat', 'heater_face'] else samples
    uv0 = obj.data.uv_layers['UVMap']
    color = obj.data.color_attributes.active_color
    for face in obj.data.polygons:
        source_id = obj.data.attributes['source_face'].data[face.index].value
        label = g.components[source_id]
        samples = []
        points = [obj.data.vertices[v].co for v in face.vertices]
        for weights_ in weights:
            p = sum((v * w for v, w in zip(points, weights_)), Vector())
            hit, _, index, _ = trees[label].ray_cast(p + face.normal * .5, -face.normal, 1.0)
            if hit is None:
                hit, _, index, _ = trees[label].find_nearest(p)
            ids, coords, _ = categories[label][index]
            q = barycentric_transform(hit, *[verts[v] for v in ids],
                                      *[Vector((u.x, u.y, 0)) for u in coords])
            px = min(width - 1, max(0, int(q.x * width)))
            py = min(height - 1, max(0, int(q.y * height)))
            samples.append(pixels[py, px])
        target = np.mean(samples, axis=0)
        available = candidates[label]
        values = np.array([s[2] for s in available])
        best = int(np.argmin(np.sum((values - target) ** 2 * [1, 1, 1, .4], axis=1)))
        u, v, rgba = available[best]
        for loop in face.loop_indices:
            uv0.data[loop].uv = (u, v)
            color.data[loop].color = (1, 1, 1, float(rgba[3]))
    return source_build.atlas_material(atlas)


def merge_level(source, level, out):
    """Append exported level accessors while retaining the original GLB data."""
    doc, blob = glb_inspect.load(str(source))
    extra, data = glb_inspect.load(str(level))
    doc = deepcopy(doc)
    accessor_offset = len(doc['accessors'])
    views = {}
    for index in sorted({a['bufferView'] for a in extra['accessors']}):
        view = deepcopy(extra['bufferViews'][index])
        source_start = view.get('byteOffset', 0)
        blob += b'\0' * (-len(blob) % 4)
        view['byteOffset'] = len(blob)
        views[index] = len(doc['bufferViews'])
        doc['bufferViews'].append(view)
        blob += data[source_start:source_start + view['byteLength']]
    for accessor in extra['accessors']:
        accessor = deepcopy(accessor)
        accessor['bufferView'] = views[accessor['bufferView']]
        doc['accessors'].append(accessor)
    mesh = deepcopy(extra['meshes'][0])
    mesh['name'] = 'L2'
    for primitive in mesh['primitives']:
        primitive['material'] = 0
        primitive['indices'] += accessor_offset
        primitive['attributes'] = {key: value + accessor_offset
                                   for key, value in primitive['attributes'].items()}
    mesh_index = len(doc['meshes'])
    doc['meshes'].append(mesh)
    node_index = len(doc['nodes'])
    doc['nodes'].append({'name': 'L2', 'mesh': mesh_index})
    doc['scenes'][doc.get('scene', 0)]['nodes'].append(node_index)
    doc['buffers'][0]['byteLength'] = len(blob)
    blob += b'\0' * (-len(blob) % 4)
    encoded = json.dumps(doc, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 4)
    out.write_bytes(struct.pack('<III', 0x46546C67, 2, 28 + len(encoded) + len(blob))
                    + struct.pack('<I4s', len(encoded), b'JSON') + encoded
                    + struct.pack('<I4s', len(blob), b'BIN\0') + blob)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'assets_dev/knight/lod_l2_v1/knight.glb')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    assert bpy.app.background
    bpy.ops.wm.open_mainfile(filepath=str(SOURCE / 'knight_textured.blend'))
    l0 = bpy.data.objects['L0']
    g = geometry()
    obj = g.mesh(l0.data.materials[0])
    obj.name = 'L2'
    obj.data.name = 'knight_L2'
    mat = atlas_projection(obj, g, l0)
    obj.data.materials[0] = mat
    l0.data.materials[0] = mat
    assert 150 <= len(obj.data.polygons) <= 250, len(obj.data.polygons)
    assert all(len(v.groups) == 1 and v.groups[0].weight == 1 for v in obj.data.vertices)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    level_path = str(out.with_name('L2_export.glb'))
    bpy.ops.export_scene.gltf(
        filepath=level_path, export_format="GLB", use_selection=True,
        export_apply=True, export_normals=True, export_texcoords=True,
        export_vertex_color="ACTIVE",
    )
    merge_level(SOURCE / 'knight_textured.glb', Path(level_path), out)
    counts = Counter(g.components[obj.data.attributes['source_face'].data[p.index].value]
                     for p in obj.data.polygons)
    (out.parent / 'components.json').write_text(json.dumps(dict(counts), indent=2) + '\n')
    manifest = {'vertices': [(x, z, -y) for x, y, z in g.vertices],
                'faces': g.faces, 'components': g.components, 'parts': g.parts}
    (out.parent / 'L2_surfaces.json').write_text(json.dumps(manifest) + '\n')
    bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix('.blend')))
    print('L2_TRIANGLES', len(obj.data.polygons), dict(counts), flush=True)


if __name__ == '__main__':
    main()
