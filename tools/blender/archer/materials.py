"""Original sergeant textures, with neutral team areas and plain livery."""
from pathlib import Path
import bpy
import geometry_archer as geo

ROOT = Path(__file__).resolve().parent


def category(component, rgba):
    if component in ["tunic", "tunic_seam", "tunic_sleeve"]:
        return "cloth", 1.0
    if component in ["hood_cape", "hood_fold"]:
        return "wool", 0.0
    if component == "patch":
        return "patch", 0.35
    if component in ["linen_coif", "linen_binding", "coif_tie", "linen_cuff"]:
        return "linen", 0.0
    if component == "fletching":
        return "feather", 0.0
    if component == "bow_string":
        return "linen", 0.0
    if component == "bow_nock":
        return "dark", 0.0
    if component in ["bow_wood", "arrow_shaft"]:
        return "wood", 0.0
    if component == "face":
        return "face", 0.0
    if component in ["hand", "neck"]:
        return "skin", 0.0
    if component == "hose":
        return "hose", 0.0
    if component in ["knife_metal", "arrowhead"]:
        return "steel", round(rgba[3], 3)
    if rgba == geo.BRASS:
        return "brass", 0.0
    if component in ["shoe_edge", "bracer_strap", "quiver_rim"]:
        return "leather_trim", 0.0
    return "leather", 0.0


def source_material(kind, alpha):
    mat = bpy.data.materials.new("bake_" + kind)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    links.new(emit.outputs[0], out.inputs["Surface"])
    coord = nodes.new("ShaderNodeTexCoord")
    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs[3].default_value = {
        "mail": 7.5,
        "cloth": 8,
        "paint": 3,
        "quilt": 3.2,
        "hose": 6,
        "skin": 3,
        "leather": 4,
        "steel": 3,
        "wool": 4,
        "linen": 4,
        "feather": 2,
        "patch": 6,
    }.get(kind, 3)
    links.new(coord.outputs["Object"], scale.inputs[0])
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(
        str(ROOT / "texture_sources" / f"{kind}.png"), check_existing=True
    )
    texture.projection = "BOX"
    texture.projection_blend = 0.12
    texture.extension = "REPEAT"
    links.new(scale.outputs["Vector"], texture.inputs["Vector"])
    if kind == "face":
        texture.projection = "FLAT"
        texture.extension = "EXTEND"
        separate = nodes.new("ShaderNodeSeparateXYZ")
        links.new(coord.outputs["Object"], separate.inputs[0])
        combine = nodes.new("ShaderNodeCombineXYZ")
        for src, dst, low, high in [("X", "X", -0.085, 0.085), ("Z", "Y", 1.50, 1.74)]:
            map_range = nodes.new("ShaderNodeMapRange")
            map_range.inputs["From Min"].default_value = low
            map_range.inputs["From Max"].default_value = high
            links.new(separate.outputs[src], map_range.inputs["Value"])
            if src == "Z":
                # Fit the generated albedo landmarks to the anatomical mesh.
                # Linear curve preserves the raster unchanged; the coordinates
                # map chin, lips, nostrils, tip, eyes and brow to their mesh rows.
                curve = nodes.new("ShaderNodeFloatCurve")
                curve.mapping.initialize()
                points = curve.mapping.curves[0].points
                points[0].location = (0, 0)
                points[-1].location = (1, 1)
                for z, v in [
                    (1.530, 0.054),
                    (1.572, 0.299),
                    (1.595, 0.433),
                    (1.606, 0.475),
                    (1.635, 0.686),
                    (1.650, 0.733),
                ]:
                    points.new((z - 1.50) / 0.24, v)
                for point in points:
                    point.handle_type = "VECTOR"
                curve.mapping.update()
                links.new(map_range.outputs[0], curve.inputs["Value"])
                links.new(curve.outputs[0], combine.inputs[dst])
            else:
                links.new(map_range.outputs[0], combine.inputs[dst])
        links.new(combine.outputs[0], texture.inputs["Vector"])
    result = texture.outputs["Color"]
    if kind in ["cloth", "wool", "linen", "hose"]:
        xyz = nodes.new("ShaderNodeSeparateXYZ")
        links.new(coord.outputs["Object"], xyz.inputs[0])
        hem = nodes.new("ShaderNodeMapRange")
        hem.inputs["From Min"].default_value = 0.12 if kind == "hose" else 0.50
        hem.inputs["From Max"].default_value = 0.36 if kind == "hose" else 0.85
        hem.inputs["To Min"].default_value = 0.57
        hem.inputs["To Max"].default_value = 1.0
        links.new(xyz.outputs["Z"], hem.inputs["Value"])
        multiply = nodes.new("ShaderNodeMixRGB")
        multiply.blend_type = "MULTIPLY"
        multiply.inputs[0].default_value = 1.0
        links.new(result, multiply.inputs[1])
        links.new(hem.outputs[0], multiply.inputs[2])
        result = multiply.outputs[0]
    links.new(result, emit.inputs["Color"])
    mask = nodes.new("ShaderNodeValue").outputs[0]
    mask.default_value = alpha
    target = nodes.new("ShaderNodeTexImage")
    target.name = "BakeTarget"
    nodes.active = target
    return mat, emit, result, target, mask
