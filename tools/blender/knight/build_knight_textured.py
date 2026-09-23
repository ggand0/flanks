"""Build the textured knight L0. Run only in a fresh background Blender.

Writes to an output directory, never to the committed asset. The bake
materials collapse to one opaque material and one RGBA atlas in the GLB.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

SOURCE=Path(__file__).resolve().parent
REPO=SOURCE.parents[2]
ROOT=REPO/"assets_dev/knight/rebuild"
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,str(SOURCE.parent))
import geometry_knight as geo

TEAM_RED=(.58,.060,.032)
TEAM_BLUE=(.055,.21,.46)


def node(mat,kind):
    return mat.node_tree.nodes.new(kind)


def tweak_geometry(g):
    import revision_geometry
    return revision_geometry.build(g)


def category(component,rgba):
    alpha=round(rgba[3],3)
    if component.startswith('mail'):
        return 'mail',0.
    if component=='surcoat':
        return 'cloth',1.
    if component=='boot_trim':
        return 'leather_trim',0.
    if rgba[:3]==geo.DARK[:3]:
        return 'dark',0.
    if rgba[:3]==geo.BRASS[:3]:
        return 'brass',0.
    if component=='heater_shield':
        if alpha==1:
            return 'paint',1.
        if rgba[:3]==geo.IVORY[:3]:
            return 'steel',0.
        return 'wood',0.
    if component in ['glove','belt','boot','boot_high','leather_cuff']:
        return 'leather',0.
    if rgba[:3]==geo.LEATHER[:3]:
        return 'leather',0.
    return 'steel',alpha


def source_material(kind,alpha):
    mat=bpy.data.materials.new(f'bake_{kind}_{alpha}')
    mat.use_nodes=True
    mat.node_tree.nodes.clear()
    link=mat.node_tree.links.new
    out=node(mat,'ShaderNodeOutputMaterial')
    emit=node(mat,'ShaderNodeEmission')
    link(emit.outputs[0],out.inputs['Surface'])
    coords=node(mat,'ShaderNodeTexCoord')
    scale=node(mat,'ShaderNodeVectorMath')
    scale.operation='SCALE'
    scale.inputs[3].default_value={'mail':7.8,'cloth':10.,'paint':3.,'leather':4.,'steel':2.8,'wood':3.5}.get(kind,3.)
    link(coords.outputs['Object'],scale.inputs[0])
    texture=node(mat,'ShaderNodeTexImage')
    texture.image=bpy.data.images.load(str(ROOT/'texture_sources'/f'{kind}.png'),check_existing=True)
    texture.projection='BOX'
    texture.projection_blend=.12
    texture.extension='REPEAT'
    link(scale.outputs['Vector'],texture.inputs['Vector'])
    result=texture.outputs['Color']
    team=node(mat,'ShaderNodeValue')
    team.outputs[0].default_value=alpha
    mask=team.outputs[0]
    if kind=='cloth':
        separate=node(mat,'ShaderNodeSeparateXYZ')
        link(coords.outputs['Object'],separate.inputs[0])
        hem=node(mat,'ShaderNodeMapRange')
        hem.inputs['From Min'].default_value=.535
        hem.inputs['From Max'].default_value=.88
        hem.inputs['To Min'].default_value=.68
        hem.inputs['To Max'].default_value=1.
        link(separate.outputs['Z'],hem.inputs['Value'])
        multiply=node(mat,'ShaderNodeMixRGB')
        multiply.blend_type='MULTIPLY'
        multiply.inputs[0].default_value=1
        link(result,multiply.inputs[1])
        link(hem.outputs[0],multiply.inputs[2])
        result=multiply.outputs[0]
        # Original woven lozenge repeat, positioned in object space before
        # baking. The finished GLB needs no procedural shader or extra mesh.
        def math_node(operation,a,b=None):
            n=node(mat,'ShaderNodeMath')
            n.operation=operation
            for i,value in enumerate([a,b]):
                if value is None:
                    continue
                if isinstance(value,(float,int)):
                    n.inputs[i].default_value=value
                else:
                    link(value,n.inputs[i])
            return n.outputs[0]
        x=math_node('MULTIPLY',separate.outputs['X'],1/.050)
        x=math_node('FRACT',x)
        x=math_node('MULTIPLY',math_node('ABSOLUTE',math_node('SUBTRACT',x,.5)),2.)
        z=math_node('ABSOLUTE',math_node('SUBTRACT',separate.outputs['Z'],.613))
        zunit=math_node('DIVIDE',z,.023)
        diamond=math_node('LESS_THAN',math_node('ABSOLUTE',math_node('SUBTRACT',math_node('ADD',x,zunit),.85)),.115)
        diamond=math_node('MULTIPLY',diamond,math_node('LESS_THAN',z,.023))
        edges=math_node('LESS_THAN',math_node('ABSOLUTE',math_node('SUBTRACT',z,.030)),.0025)
        pattern=math_node('MAXIMUM',diamond,edges)
        linen=node(mat,'ShaderNodeMixRGB')
        linen.blend_type='MULTIPLY'
        linen.inputs[0].default_value=1.
        link(texture.outputs['Color'],linen.inputs[1])
        linen.inputs[2].default_value=(.74,.63,.39,1.)
        mix=node(mat,'ShaderNodeMixRGB')
        link(pattern,mix.inputs[0])
        link(result,mix.inputs[1])
        link(linen.outputs[0],mix.inputs[2])
        result=mix.outputs[0]
        mask=math_node('SUBTRACT',1.,pattern)
    link(result,emit.inputs['Color'])
    target=node(mat,'ShaderNodeTexImage')
    target.name='BakeTarget'
    mat.node_tree.nodes.active=target
    mat['team_alpha']=alpha
    return mat,emit,result,target,mask


def unwrap(obj):
    bpy.context.view_layer.objects.active=obj
    obj.select_set(True)
    obj.data.uv_layers.active_index=0
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(70),island_margin=.006)
    bpy.ops.object.mode_set(mode='OBJECT')


def bake_image(obj,materials,size,mode):
    image=bpy.data.images.new(f'Bake_{mode}',width=size,height=size,alpha=True,float_buffer=True)
    image.colorspace_settings.name='Non-Color'
    for mat,emit,result,target,mask in materials:
        tree=mat.node_tree
        for link in list(emit.inputs['Color'].links):
            tree.links.remove(link)
        if mode=='color':
            tree.links.new(result,emit.inputs['Color'])
        elif mode=='mask':
            tree.links.new(mask,emit.inputs['Color'])
        else:
            ao=node(mat,'ShaderNodeAmbientOcclusion')
            ao.inputs['Distance'].default_value=.13
            ao.only_local=True
            ao.samples=16
            tree.links.new(ao.outputs['Color'],emit.inputs['Color'])
        target.image=image
        tree.nodes.active=target
    print('BAKE_START',mode,flush=True)
    bpy.ops.object.bake(type='EMIT',margin=12,use_clear=True)
    pixels=np.empty(size*size*4,dtype=np.float32)
    image.pixels.foreach_get(pixels)
    print('BAKE_DONE',mode,flush=True)
    return pixels.reshape((size,size,4))


def atlas_material(image):
    mat=bpy.data.materials.new('knight_atlas_opaque')
    mat.use_nodes=True
    principled=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    principled.inputs['Roughness'].default_value=.74
    principled.inputs['Metallic'].default_value=.08
    uv=node(mat,'ShaderNodeUVMap')
    uv.uv_map='UVMap'
    tex=node(mat,'ShaderNodeTexImage')
    tex.image=image
    tex.extension='EXTEND'
    mat.node_tree.links.new(uv.outputs[0],tex.inputs['Vector'])
    vertex=node(mat,'ShaderNodeVertexColor')
    vertex.layer_name='Col'
    multiply=node(mat,'ShaderNodeMixRGB')
    multiply.blend_type='MULTIPLY'
    multiply.inputs[0].default_value=1
    mat.node_tree.links.new(tex.outputs['Color'],multiply.inputs[1])
    mat.node_tree.links.new(vertex.outputs['Color'],multiply.inputs[2])
    mat.node_tree.links.new(multiply.outputs[0],principled.inputs['Base Color'])
    # Alpha is a team mask. It must never connect to Principled Alpha.
    return mat


def tint_material(material,team):
    mat=material.copy()
    mat.name='Review_team_tint'
    bsdf=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    tex=next(n for n in mat.node_tree.nodes if n.type=='TEX_IMAGE')
    blend=node(mat,'ShaderNodeMixRGB')
    blend.blend_type='MIX'
    blend.inputs[1].default_value=(1,1,1,1)
    blend.inputs[2].default_value=tuple(team)+(1,)
    mat.node_tree.links.new(tex.outputs['Alpha'],blend.inputs[0])
    multiply=node(mat,'ShaderNodeMixRGB')
    multiply.blend_type='MULTIPLY'
    multiply.inputs[0].default_value=1
    mat.node_tree.links.new(tex.outputs['Color'],multiply.inputs[1])
    mat.node_tree.links.new(blend.outputs[0],multiply.inputs[2])
    mat.node_tree.links.new(multiply.outputs[0],bsdf.inputs['Base Color'])
    return mat


def verify(obj,path,size):
    import glb_inspect
    doc,blob=glb_inspect.load(str(path))
    assert len(doc['meshes'])==1 and len(doc['materials'])==1
    assert len(doc['images'])==1 and len(doc['textures'])==1
    material=doc['materials'][0]
    assert material.get('alphaMode','OPAQUE')=='OPAQUE'
    assert material['pbrMetallicRoughness']['baseColorTexture']['index']==0
    primitive=doc['meshes'][0]['primitives'][0]
    attrs=primitive['attributes']
    assert doc['accessors'][attrs['COLOR_0']]['type']=='VEC4'
    _,colors=glb_inspect.accessor_values(doc,blob,attrs['COLOR_0'])
    assert all(all(abs(v-1)<1e-5 for v in c[:3]) for c in colors)
    assert min(c[3] for c in colors)==0 and max(c[3] for c in colors)==1
    _,uv=glb_inspect.accessor_values(doc,blob,attrs['TEXCOORD_1'])
    pairs=sorted({tuple(round(c,6) for c in v) for v in uv})
    expected=sorted([(0.,0.),(1.,1.435),(2.,.91),(3.,.91),(5.,1.435)])
    assert pairs==expected,(pairs,expected)
    _,uv0=glb_inspect.accessor_values(doc,blob,attrs['TEXCOORD_0'])
    assert len(set(uv0))>100
    assert all(-1e-5<=v<=1.00001 for p in uv0 for v in p)
    _,position=glb_inspect.accessor_values(doc,blob,attrs['POSITION'])
    assert abs(max(p[1] for p in position)-1.8)<1e-5
    assert abs(min(p[1] for p in position))<1e-5
    triangles=doc['accessors'][primitive['indices']]['count']//3
    assert 2000<=triangles<=3000
    pivots={n['name']:n['translation'] for n in doc['nodes'] if n.get('name','').startswith('pivot_')}
    assert len(pivots)==4
    for part,(x,y,z) in geo.PIVOTS.items():
        assert all(abs(a-b)<1e-5 for a,b in zip(pivots['pivot_'+part],(x,z,-y)))
    image=doc['images'][0]
    assert image['mimeType']=='image/png'
    view=doc['bufferViews'][image['bufferView']]
    start=view.get('byteOffset',0)
    png=blob[start:start+view['byteLength']]
    assert png[:8]==b'\x89PNG\r\n\x1a\n'
    assert int.from_bytes(png[16:20],'big')==size and int.from_bytes(png[20:24],'big')==size
    assert png[25]==6, 'Embedded PNG must have RGBA channels'
    assert all(len(v.groups)==1 and v.groups[0].weight==1 for v in obj.data.vertices)
    assert all(p.area>1e-12 for p in obj.data.polygons)
    return {'stage':'Textured L0 review only','triangles':triangles,
            'height_m':max(p[1] for p in position),'ground_m':min(p[1] for p in position),
            'source_vertices':len(obj.data.vertices),'exported_vertices':len(position),
            'part_vertices':dict(Counter(obj.vertex_groups[v.groups[0].group].name for v in obj.data.vertices)),
            'pivots_gltf_xyz':pivots,'part_uv_pairs':pairs,'material_count':1,'alpha_mode':'OPAQUE',
            'atlas_size':[size,size],'embedded_images':1,'atlas_format':'RGBA PNG',
            'atlas_rgb':'sRGB base color with baked local AO; neutral team regions',
            'atlas_alpha':'Linear team tint mask, not transparency',
            'COLOR_0':'VEC4, RGB white so glTF does not multiply the atlas by old colors',
            'TEXCOORD_0':'Packed atlas UVs','TEXCOORD_1':'Unchanged (part ID, pivot height)',
            'team_formula':'linear_atlas_rgb * mix(vec3(1), linear_team_tint, atlas_alpha)',
            'normal_maps':False,'insignia':False,'engine_texture_support_required':True}


def camera(elevation,azimuth,target=(0,-.07,.91),scale=2.32):
    scene=bpy.context.scene
    if scene.camera is None:
        data=bpy.data.cameras.new('ReviewCamera')
        scene.camera=bpy.data.objects.new('ReviewCamera',data)
        scene.collection.objects.link(scene.camera)
    cam=scene.camera
    el,az=map(math.radians,(elevation,azimuth))
    target=Vector(target)
    cam.location=target+6*Vector((math.sin(az)*math.cos(el),-math.cos(az)*math.cos(el),math.sin(el)))
    cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.data.type='ORTHO'
    cam.data.ortho_scale=scale
    bpy.context.view_layer.update()
    return cam


def render_setup():
    scene=bpy.context.scene
    scene.render.engine='BLENDER_EEVEE'
    scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGBA'
    scene.render.film_transparent=True
    scene.render.resolution_percentage=100
    scene.world=bpy.data.worlds.new('ReviewWorld')
    scene.world.use_nodes=True
    background=next(n for n in scene.world.node_tree.nodes if n.type=='BACKGROUND')
    background.inputs['Color'].default_value=(.22,.25,.30,1)
    background.inputs['Strength'].default_value=.35
    scene.view_settings.view_transform='AgX'
    for name,loc,power,size in [('Key',(-3,-4,6),520,4),('Fill',(3,-1,3),180,4),('Rim',(1,3,4),400,3)]:
        data=bpy.data.lights.new(name,'AREA')
        data.energy,data.shape,data.size=power,'DISK',size
        light=bpy.data.objects.new(name,data)
        light.location=loc
        light.rotation_euler=(Vector((0,0,1))-light.location).to_track_quat('-Z','Y').to_euler()
        scene.collection.objects.link(light)


def render(name,width=840,height=960):
    scene=bpy.context.scene
    scene.render.resolution_x,scene.render.resolution_y=width,height
    scene.render.filepath=str(ROOT/name)
    bpy.ops.render.render(write_still=True)


def main():
    global ROOT
    ap=argparse.ArgumentParser()
    ap.add_argument('--size',type=int,default=2048,choices=[1024,2048])
    ap.add_argument('--reuse-bake',action='store_true')
    ap.add_argument('--out-dir',type=Path,default=ROOT)
    args=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    ROOT=args.out_dir.resolve()
    ROOT.mkdir(parents=True,exist_ok=True)
    if not bpy.app.background:
        raise RuntimeError('Run in a separate background Blender process')
    started=time.monotonic()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    scene.render.threads_mode='FIXED'
    scene.render.threads=4
    scene.render.engine='CYCLES'
    scene.cycles.device='CPU'
    scene.cycles.samples=16
    g=tweak_geometry(geo.build())
    dummy=bpy.data.materials.new('temporary')
    obj=g.mesh(dummy)
    print('GEOMETRY',len(obj.data.polygons),'triangles',flush=True)
    assert 2000<=len(obj.data.polygons)<=3000, 'L0 must stay inside the agreed triangle budget'
    assert all(p.area>1e-12 for p in obj.data.polygons), 'Degenerate source triangle'
    obj.name='L0'
    obj.data.name='knight_textured_L0'
    bpy.context.view_layer.objects.active=obj
    obj.select_set(True)
    obj.data.materials.clear()
    keys=sorted({category(c,rgba) for c,rgba in zip(g.components,g.colors)})
    materials=[source_material(*key) for key in keys]
    for mat,*_ in materials:
        obj.data.materials.append(mat)
    face_source=obj.data.attributes['source_face']
    for p in obj.data.polygons:
        i=face_source.data[p.index].value
        p.material_index=keys.index(category(g.components[i],g.colors[i]))
    unwrap(obj)
    atlas_path=ROOT/'knight_atlas.png'
    if args.reuse_bake:
        atlas=bpy.data.images.load(str(atlas_path),check_existing=True)
        assert tuple(atlas.size)==(args.size,args.size)
    else:
        base=bake_image(obj,materials,args.size,'color')
        mask=bake_image(obj,materials,args.size,'mask')
        ao=bake_image(obj,materials,args.size,'ao')
        rgba=np.ones_like(base)
        linear=np.clip(base[:,:,:3]*(.48+.52*ao[:,:,:1]),0,1)
        # Bake targets are linear float buffers. Byte-image pixels expect
        # encoded samples, so encode RGB explicitly before writing the PNG.
        rgba[:,:,:3]=np.where(linear<=.0031308,linear*12.92,1.055*np.power(linear,1/2.4)-.055)
        rgba[:,:,3]=np.clip(mask[:,:,0],0,1)
        atlas=bpy.data.images.new('KnightAtlas',width=args.size,height=args.size,alpha=True)
        atlas.colorspace_settings.name='sRGB'
        atlas.alpha_mode='CHANNEL_PACKED'
        atlas.pixels.foreach_set(rgba.reshape(-1))
        atlas.filepath_raw=str(atlas_path)
        atlas.file_format='PNG'
        atlas.save()
        # Reload the saved PNG so the export and renders use identical bytes.
        atlas=bpy.data.images.load(str(atlas_path),check_existing=False)
    atlas.alpha_mode='CHANNEL_PACKED'
    mat=atlas_material(atlas)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for p in obj.data.polygons:
        p.material_index=0
    color=obj.data.color_attributes.active_color
    # Keep exactly one color layer. With the measured Blender 5.2 exporter,
    # exporting a second layer overwrites its material-to-color association
    # and replaces COLOR_0 with white including alpha. Geometry source retains
    # the original fixed colors, so no auxiliary layer is necessary here.
    for src in color.data:
        rgba=tuple(src.color)
        src.color=(1,1,1,rgba[3])
    obj.data.color_attributes.active_color=color
    obj.data.attributes.active_color=color
    obj.data.color_attributes.render_color_index=obj.data.color_attributes.find(color.name)
    for attribute in obj.data.color_attributes:
        print('SOURCE_COLOR',attribute.name,'alpha',sorted({round(c.color[3],3) for c in attribute.data}),flush=True)
    print('ACTIVE_COLOR',obj.data.color_attributes.active_color.name,flush=True)
    for part,loc in geo.PIVOTS.items():
        empty=bpy.data.objects.new('pivot_'+part,None)
        empty.location=loc
        empty.empty_display_size=.045
        scene.collection.objects.link(empty)
        empty.select_set(True)
    path=str(ROOT/'knight_textured.glb')
    bpy.ops.export_scene.gltf(
        filepath=path, export_format="GLB", use_selection=True,
        export_apply=True, export_normals=True, export_texcoords=True,
        export_vertex_color="ACTIVE",
    )
    report=verify(obj,Path(path),args.size)
    import revision_geometry
    report['revision']='v2: oval great helm, extended mail skirt, tapered riding boots, woven geometric hem'
    report['shape_measurements_blender_xyz']=revision_geometry.measurements(g)
    report['pattern']='Original woven diamond border, with an untinted flax-colored thread mask'
    report['build_seconds_before_renders']=time.monotonic()-started
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('EXPORTED',json.dumps(report),flush=True)
    render_setup()
    camera(12,25)
    obj.data.materials[0]=tint_material(mat,TEAM_RED)
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'knight_textured.blend'))
    # Visual checks use the GLB itself, after its independent byte assertions.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.threads_mode='FIXED'
    bpy.context.scene.render.threads=4
    bpy.ops.import_scene.gltf(filepath=path)
    obj=next(o for o in bpy.context.scene.objects if o.type=='MESH')
    imported=obj.data.materials[0]
    render_setup()
    camera(12,25)
    obj.data.materials[0]=tint_material(imported,TEAM_RED)
    render('knight_red.png')
    camera(15,155)
    render('knight_back.png')
    camera(12,25)
    obj.data.materials[0]=tint_material(imported,TEAM_BLUE)
    render('knight_blue.png')
    camera(8,85)
    render('knight_side.png')
    camera(8,25,(0,-.025,1.642),.42)
    render('detail_helm.png',800,800)
    camera(10,45,(0,-.05,.315),.78)
    render('detail_boots.png',800,800)
    camera(3,0,(0,-.05,.65),.71)
    render('detail_hem.png',800,800)
    obj.data.materials[0]=tint_material(imported,TEAM_RED)
    cam=camera(50,0)
    rotation=cam.matrix_world.to_quaternion().inverted()
    points=[rotation@(obj.matrix_world@v.co) for v in obj.data.vertices]
    extent=max(p.y for p in points)-min(p.y for p in points)
    mid=(max(p.y for p in points)+min(p.y for p in points))/2
    current=(rotation@cam.location).y
    cam.location+=cam.matrix_world.to_quaternion()@Vector((0,mid-current,0))
    for px in [60,20,8,3]:
        cam.data.ortho_scale=extent*256/px
        render(f'L0_{px}px.png',256,256)
    report['render_source']='Re-imported GLB, with explicit team tint preview shader'
    report['screen_checks']={'elevation_degrees':50,'targets_px':[60,20,8,3],'all_geometry':'L0'}
    report['total_build_seconds']=time.monotonic()-started
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('TEXTURED_KNIGHT_DONE',flush=True)


if __name__=='__main__':
    main()
