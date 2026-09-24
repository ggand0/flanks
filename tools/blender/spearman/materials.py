"""Original sergeant textures, with neutral team areas and plain livery."""
from pathlib import Path
import bpy
import geometry_spearman as geo

ROOT=Path(__file__).resolve().parent


def category(component,rgba):
    if component.startswith('mail'):
        return 'mail',0.
    if component.startswith('quilt') or component=='arming_cap':
        return 'quilt',0.
    if component=='surcoat':
        return 'cloth',1.
    if component=='face':
        return 'face',0.
    if component.startswith('hand'):
        return 'skin',0.
    if component=='face_detail':
        return ('dark' if rgba==geo.DARK else 'stubble'),0.
    if component=='hose':
        return 'hose',0.
    if component=='heater_face':
        return 'paint',1.
    if component=='heater_wood' or component=='spear_shaft':
        return 'wood',0.
    if component=='heater_rim':
        return ('rawhide' if rgba==geo.IVORY else 'leather'),0.
    if rgba==geo.BRASS:
        return 'brass',0.
    if component=='shoe_edge':
        return 'leather_trim',0.
    if rgba in [geo.LEATHER,geo.LEATHER_EDGE]:
        return 'leather',0.
    return 'steel',round(rgba[3],3)


def source_material(kind,alpha):
    mat=bpy.data.materials.new('bake_'+kind)
    mat.use_nodes=True
    mat.node_tree.nodes.clear()
    nodes,links=mat.node_tree.nodes,mat.node_tree.links
    out=nodes.new('ShaderNodeOutputMaterial')
    emit=nodes.new('ShaderNodeEmission')
    links.new(emit.outputs[0],out.inputs['Surface'])
    coord=nodes.new('ShaderNodeTexCoord')
    scale=nodes.new('ShaderNodeVectorMath')
    scale.operation='SCALE'
    scale.inputs[3].default_value={'mail':7.5,'cloth':8,'paint':3,'quilt':3.2,'hose':6,'skin':3,'leather':4,'steel':3}.get(kind,3)
    links.new(coord.outputs['Object'],scale.inputs[0])
    texture=nodes.new('ShaderNodeTexImage')
    texture.image=bpy.data.images.load(str(ROOT/'texture_sources'/f'{kind}.png'),check_existing=True)
    texture.projection='BOX'
    texture.projection_blend=.12
    texture.extension='REPEAT'
    links.new(scale.outputs['Vector'],texture.inputs['Vector'])
    if kind=='face':
        texture.projection='FLAT'
        texture.extension='EXTEND'
        separate=nodes.new('ShaderNodeSeparateXYZ')
        links.new(coord.outputs['Object'],separate.inputs[0])
        combine=nodes.new('ShaderNodeCombineXYZ')
        for src,dst,low,high in [('X','X',-.085,.085),('Z','Y',1.50,1.74)]:
            map_range=nodes.new('ShaderNodeMapRange')
            map_range.inputs['From Min'].default_value=low
            map_range.inputs['From Max'].default_value=high
            links.new(separate.outputs[src],map_range.inputs['Value'])
            links.new(map_range.outputs[0],combine.inputs[dst])
        links.new(combine.outputs[0],texture.inputs['Vector'])
    result=texture.outputs['Color']
    if kind in ['cloth','quilt','hose']:
        xyz=nodes.new('ShaderNodeSeparateXYZ')
        links.new(coord.outputs['Object'],xyz.inputs[0])
        hem=nodes.new('ShaderNodeMapRange')
        hem.inputs['From Min'].default_value=.12 if kind=='hose' else .54
        hem.inputs['From Max'].default_value=.36 if kind=='hose' else .85
        hem.inputs['To Min'].default_value=.66
        hem.inputs['To Max'].default_value=1.
        links.new(xyz.outputs['Z'],hem.inputs['Value'])
        multiply=nodes.new('ShaderNodeMixRGB')
        multiply.blend_type='MULTIPLY'
        multiply.inputs[0].default_value=1.
        links.new(result,multiply.inputs[1])
        links.new(hem.outputs[0],multiply.inputs[2])
        result=multiply.outputs[0]
    links.new(result,emit.inputs['Color'])
    mask=nodes.new('ShaderNodeValue').outputs[0]
    mask.default_value=alpha
    target=nodes.new('ShaderNodeTexImage')
    target.name='BakeTarget'
    nodes.active=target
    return mat,emit,result,target,mask
