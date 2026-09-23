"""Textured L0 v3 spearman review. Run only in a fresh background Blender.

All outputs go to --out-dir, defaulting to assets_dev/spearman/rebuild_v3/. Temporary bake materials collapse to one
opaque material and one RGBA atlas. The new arm parts require engine support.
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
ROOT=SOURCE.parents[2]/"assets_dev/spearman/rebuild_v3"
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,str(SOURCE.parent))
import geometry_spearman as geo

TEAM_RED=(.58,.060,.032)
TEAM_BLUE=(.055,.21,.46)


def node(mat,kind):
    return mat.node_tree.nodes.new(kind)


from materials import category, source_material


def unwrap(obj,g,size):
    bpy.context.view_layer.objects.active=obj
    obj.select_set(True)
    obj.data.uv_layers.active_index=0
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(70),island_margin=.006)
    bpy.ops.object.mode_set(mode='OBJECT')
    # Allocate the exposed face one continuous, larger island. Automatic
    # area-based packing undersamples the eyelids/lips even in a 2K atlas.
    uv=obj.data.uv_layers[0]
    source=obj.data.attributes['source_face']
    front=[]
    for polygon in obj.data.polygons:
        if g.components[source.data[polygon.index].value]!='face':
            continue
        if any(obj.data.vertices[v].co.y>.03 for v in polygon.vertices):
            continue  # The closure lies inside the coif and retains its UVs.
        for loop in polygon.loop_indices:
            co=obj.data.vertices[obj.data.loops[loop].vertex_index].co
            uv.data[loop].uv=(2+co.x,co.z)
            front.append(loop)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.pack_islands(rotate=True,margin=.006)
    bpy.ops.object.mode_set(mode='OBJECT')
    uv=obj.data.uv_layers[0]
    coords=[uv.data[i].uv for i in front]
    bounds=[(max(p[a] for p in coords)-min(p[a] for p in coords))*size for a in [0,1]]
    assert min(bounds)>200, 'Insufficient face texel allocation'
    return {'front_island_bounds_px':bounds,'atlas_size_px':size,'continuous_front_projection':True}


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
    mat=bpy.data.materials.new('spearman_atlas_opaque')
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
    expected=sorted([(0.,0.)]+[(float(geo.PARTS[k]),round(v[2],6)) for k,v in geo.PIVOTS.items()])
    assert pairs==expected,(pairs,expected)
    _,uv0=glb_inspect.accessor_values(doc,blob,attrs['TEXCOORD_0'])
    assert len(set(uv0))>100
    assert all(-1e-5<=v<=1.00001 for p in uv0 for v in p)
    _,position=glb_inspect.accessor_values(doc,blob,attrs['POSITION'])
    character_top=max(p[1] for p,part in zip(position,uv) if part[0]==0)
    assert abs(character_top-1.8)<1e-5
    assert abs(max(p[1] for p in position)-2.52)<1e-5
    assert abs(min(p[1] for p in position))<1e-5
    triangles=doc['accessors'][primitive['indices']]['count']//3
    assert 2000<=triangles<=3000
    pivots={n['name']:n['translation'] for n in doc['nodes'] if n.get('name','').startswith(('pivot_','joint_'))}
    assert len(pivots)==len(geo.PIVOTS)+len(geo.JOINTS)
    for name,(x,y,z) in [('pivot_'+k,v) for k,v in geo.PIVOTS.items()]+[('joint_'+k,v) for k,v in geo.JOINTS.items()]:
        assert all(abs(a-b)<1e-5 for a,b in zip(pivots[name],(x,z,-y)))
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
            'height_m':character_top,'overall_height_m':max(p[1] for p in position),'ground_m':min(p[1] for p in position),
            'source_vertices':len(obj.data.vertices),'exported_vertices':len(position),
            'part_vertices':dict(Counter(obj.vertex_groups[v.groups[0].group].name for v in obj.data.vertices)),
            'pivots_gltf_xyz':pivots,'part_uv_pairs':pairs,'material_count':1,'alpha_mode':'OPAQUE',
            'atlas_size':[size,size],'embedded_images':1,'atlas_format':'RGBA PNG',
            'atlas_rgb':'sRGB base color with baked local AO; neutral team regions',
            'atlas_alpha':'Linear team tint mask, not transparency',
            'COLOR_0':'VEC4, RGB white so glTF does not multiply the atlas by old colors',
            'TEXCOORD_0':'Packed atlas UVs','TEXCOORD_1':'Part ID and pivot height, including proposed IDs 9 and 10',
            'team_formula':'linear_atlas_rgb * mix(vec3(1), linear_team_tint, atlas_alpha)',
            'normal_maps':False,'insignia':False,'engine_texture_support_required':True}


def camera(elevation,azimuth,target=(0,-.07,1.21),scale=3.03):
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
    g=geo.build()
    # Preserve earlier build directories; this is the isolated v3 prototype.
    preserved={'scope':'Weapon arm articulation, shoulder socket, spear position only; original texture sources retained'}
    dummy=bpy.data.materials.new('temporary')
    obj=g.mesh(dummy)
    print('GEOMETRY',len(obj.data.polygons),'triangles',flush=True)
    print('COMPONENT_TRIANGLES',{k:v['triangles'] for k,v in geo.measurements(g).items()},flush=True)
    assert 2000<=len(obj.data.polygons)<=3000, 'L0 must stay inside the agreed triangle budget'
    assert all(p.area>1e-12 for p in obj.data.polygons), 'Degenerate source triangle'
    obj.name='L0'
    obj.data.name='spearman_L0'
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
    face_uv=unwrap(obj,g,args.size)
    atlas_path=ROOT/'spearman_atlas.png'
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
        atlas=bpy.data.images.new('SpearmanAtlas',width=args.size,height=args.size,alpha=True)
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
    for name,loc in [('pivot_'+k,v) for k,v in geo.PIVOTS.items()]+[('joint_'+k,v) for k,v in geo.JOINTS.items()]:
        empty=bpy.data.objects.new(name,None)
        empty.location=loc
        empty.empty_display_size=.045
        scene.collection.objects.link(empty)
        empty.select_set(True)
    path=str(ROOT/'spearman.glb')
    bpy.ops.export_scene.gltf(
        filepath=path, export_format="GLB", use_selection=True,
        export_apply=True, export_normals=True, export_texcoords=True,
        export_vertex_color="ACTIVE",
    )
    report=verify(obj,Path(path),args.size)
    report['revision']='Spearman textured L0 v3: explicit upper arm, forearm, wrist and grip chain (requires engine integration)'
    report['contract_extension']={'9':'forearm_spear','10':'hand_spear'}
    report['face_revision_scope']=preserved
    report['face_uv']=face_uv
    report['shape_measurements_blender_xyz']=geo.measurements(g)
    report['pattern']='Plain team cloth and shield, no insignia'
    report['build_seconds_before_renders']=time.monotonic()-started
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('EXPORTED',report['triangles'],'triangles;',report['height_m'],'m character;',report['exported_vertices'],'exported vertices',flush=True)
    render_setup()
    camera(12,25)
    obj.data.materials[0]=tint_material(mat,TEAM_RED)
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'spearman.blend'))
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
    render('spearman_red.png')
    camera(15,155)
    render('spearman_back.png')
    camera(12,25)
    obj.data.materials[0]=tint_material(imported,TEAM_BLUE)
    render('spearman_blue.png')
    camera(8,85)
    render('spearman_side.png')
    camera(6,15,(0,-.015,1.618),.48)
    render('detail_helm.png',800,800)
    camera(12,40,(0,-.05,.18),.43)
    render('detail_shoes.png',800,800)
    camera(3,0,(0,-.05,.65),.71)
    render('detail_hem.png',800,800)
    obj.data.materials[0]=tint_material(imported,TEAM_RED)
    cam=camera(50,0)
    rotation=cam.matrix_world.to_quaternion().inverted()
    # Fit the soldier silhouette to the requested pixel height. The spear
    # is taller than the head and is deliberately excluded from this measure.
    points=[rotation@Vector(p) for p,part in zip(g.vertices,g.parts) if part in ["body","leg_l","leg_r"]]
    extent=max(p.y for p in points)-min(p.y for p in points)
    mid=(max(p.y for p in points)+min(p.y for p in points))/2
    current=(rotation@cam.location).y
    cam.location+=cam.matrix_world.to_quaternion()@Vector((0,mid-current,0))
    for px in [60,20,8,3]:
        cam.data.ortho_scale=extent*256/px
        render(f'L0_{px}px.png',256,256)
    report['render_source']='Re-imported GLB, with explicit team tint preview shader'
    report['screen_checks']={'elevation_degrees':50,'targets_px':[60,20,8,3],'all_geometry':'L0','height_basis':'Character body and legs; spear excluded from target height'}
    report['total_build_seconds']=time.monotonic()-started
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('TEXTURED_SPEARMAN_DONE',flush=True)


if __name__=='__main__':
    main()
