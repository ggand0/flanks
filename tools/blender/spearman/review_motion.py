"""Render the actual exported GLB under motion.py's rigid transforms.

Separate background Blender only. Save a playable hierarchy with sampled
object animation, then render the complete cycle from two fixed cameras.
The runtime GLB remains the unposed single-mesh asset.
"""
from pathlib import Path
import argparse, sys, json, math
import bpy
import numpy as np
from mathutils import Matrix, Vector

SOURCE=Path(__file__).resolve().parent
ROOT=SOURCE.parents[2]/"assets_dev/spearman/rebuild_v3"
sys.path.insert(0,str(SOURCE))
import motion
import build_spearman as build
import geometry_spearman as geo


def gltf(p):
    p=np.asarray(p)
    return np.stack([p[...,0],p[...,2],-p[...,1]],axis=-1)


def probes():
    g=geo.build()
    def points(label, part):
        ids={i for face,name in zip(g.faces,g.components) if name==label for i in face if g.parts[i]==part}
        return gltf([g.vertices[i] for i in sorted(ids)])
    result=[]
    for joint,center,label,cap_part,attachments in [
        ('shoulder',motion.S,'mail_shoulder_socket','body',[('mail_sleeve','arm_spear',motion.E-motion.S)]),
        ('elbow',motion.E,'quilt_elbow_joint','arm_spear',[('mail_sleeve','arm_spear',motion.E-motion.S),('quilt_sleeve','forearm_spear',motion.W-motion.E)]),
        ('wrist',motion.W,'hand_wrist_joint','hand_spear',[('quilt_sleeve','forearm_spear',motion.W-motion.E),('quilt_cuff','forearm_spear',motion.W-motion.E),('hand','hand_spear',np.array([-.013,-.018,.039]))])]:
        caps=points(label,cap_part)
        seams=[]
        for lab,part,axis in attachments:
            pts=points(lab,part)
            distance=np.abs((pts-center)@(axis/np.linalg.norm(axis)))
            pts=pts[distance<1e-6]
            assert len(pts)>=6,(joint,lab,len(pts))
            seams.append({'part':geo.PARTS[part],'points':pts.tolist()})
        result.append({'joint':joint,'center':center.tolist(),'cap_part':geo.PARTS[cap_part],
                       'cap_vertices':caps.tolist(),'seams':seams})
    (ROOT/'joint_attachment_probes.json').write_text(json.dumps(result,indent=2)+'\n')


def bake_preview_animation(objects):
    scene=bpy.context.scene
    c=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    for part in objects.values():
        part.animation_data_clear()
        # Imported glTF objects use quaternion mode. Keying Euler channels
        # without changing mode saves translations but no playable rotations.
        part.rotation_mode='XYZ'
    scene.render.fps=30;scene.frame_start=1;scene.frame_end=96
    for frame in range(1,97):
        for pid,(r,t) in motion.transforms(motion.sample((frame-1)/30))[0].items():
            part=objects[pid]
            mat=np.eye(4);mat[:3,:3]=c@r@c.T;mat[:3,3]=c@t
            part.matrix_world=Matrix(mat.tolist())
            part.keyframe_insert('location',frame=frame)
            part.keyframe_insert('rotation_euler',frame=frame)
    for part in objects.values():
        action=part.animation_data.action
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for curve in bag.fcurves:
                        for key in curve.keyframe_points:
                            key.interpolation='LINEAR'
    scene.frame_set(1)


def main():
    global ROOT
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir',type=Path,default=ROOT)
    ap.add_argument('--verify-only',action='store_true')
    ap.add_argument('--rebake-scene-only',action='store_true')
    args=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    ROOT=args.out_dir.resolve()
    build.ROOT=ROOT
    assert bpy.app.background
    if args.rebake_scene_only:
        bpy.ops.wm.open_mainfile(filepath=str(ROOT/'spear_stab_review.blend'))
        objects={pid:bpy.data.objects[motion.PARTS[pid]] for pid in [4,9,10,8]}
        bake_preview_animation(objects)
        bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'spear_stab_review.blend'))
        return
    if args.verify_only:
        bpy.ops.wm.open_mainfile(filepath=str(ROOT/'spear_stab_review.blend'))
        scene=bpy.context.scene
        c=np.array([[1,0,0],[0,0,-1],[0,1,0]])
        maximum=0.
        for frame in range(1,97):
            scene.frame_set(frame)
            for pid,(r,t) in motion.transforms(motion.sample((frame-1)/30))[0].items():
                actual=np.array(bpy.data.objects[motion.PARTS[pid]].matrix_world)
                expected=np.eye(4);expected[:3,:3]=c@r@c.T;expected[:3,3]=c@t
                maximum=max(maximum,float(np.max(abs(actual-expected))))
        assert maximum<1e-6,maximum
        report={'frames_checked':96,'max_saved_scene_matrix_error':maximum,'tolerance':1e-6}
        (ROOT/'saved_scene_validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print('SAVED_SCENE_VALID',report,flush=True)
        return
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    scene.render.threads_mode='FIXED';scene.render.threads=4
    bpy.ops.import_scene.gltf(filepath=str(ROOT/'spearman.glb'))
    obj=next(o for o in scene.objects if o.type=='MESH')
    mesh=obj.data
    uv=mesh.uv_layers[1]  # glTF carries channel order, not authoring UV names.
    parts=np.zeros(len(mesh.vertices),dtype=int)
    for loop in mesh.loops:
        parts[loop.vertex_index]=round(uv.data[loop.index].uv.x)
    coords=np.array([v.co[:] for v in mesh.vertices])
    source=gltf(coords)
    mat=build.tint_material(mesh.materials[0],build.TEAM_RED)
    mesh.materials[0]=mat
    build.render_setup()
    scene.render.film_transparent=False
    bg=next(n for n in scene.world.node_tree.nodes if n.type=='BACKGROUND')
    bg.inputs['Color'].default_value=(.10,.12,.15,1)
    bg.inputs['Strength'].default_value=.55
    # Export independent measurements and portable motion tables.
    probes()
    (ROOT/'spear_stab.json').write_text(json.dumps(motion.export_clip(),indent=2)+'\n')
    np.savez(ROOT/'review_mesh.npz',positions=source,parts=parts)

    def apply(pose):
        posed=motion.deform(source,parts,pose)
        blender=np.stack([posed[:,0],-posed[:,2],posed[:,1]],axis=1)
        mesh.vertices.foreach_set('co',blender.astype(np.float32).reshape(-1))
        mesh.update()

    # Label-free actual frames; compose captions outside the rendered image.
    for label,pose in [('carry',motion.CARRY),('guard',motion.GUARD),('draw',motion.DRAW),
                       ('midstrike',motion.attack(windup=.80)),('strike',motion.STRIKE),
                       ('recover',motion.attack(recovery=.5))]:
        apply(pose)
        build.camera(8,-78,(0,-.80,1.08),3.45)
        build.render('pose_'+label+'.png',1000,660)
    # Both side and front-oblique views throughout windup, strike and recovery.
    for view,az in [('side',-88),('oblique',-38)]:
        folder=ROOT/('frames_'+view)
        folder.mkdir(exist_ok=True)
        build.camera(10,az,(0,-.80,1.02),3.50)
        for frame in range(48):
            apply(motion.sample(frame/30))
            build.render(str(folder/f'{frame:03d}.png'),840,600)
    # Restore the base geometry, split for preview object transform animation.
    mesh.vertices.foreach_set('co',coords.astype(np.float32).reshape(-1));mesh.update()
    for name in motion.PARTS.values():
        vg=obj.vertex_groups.new(name=name)
        pid=next(k for k,v in motion.PARTS.items() if v==name)
        ids=np.where(parts==pid)[0].tolist()
        if ids: vg.add(ids,1.,'REPLACE')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True);bpy.context.view_layer.objects.active=obj
    # Each part is one rigid object, animated by the same matrices as the proof.
    objects={}
    for pid in [4,9,10,8]:
        bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='DESELECT');bpy.ops.object.mode_set(mode='OBJECT')
        for v in obj.data.vertices:
            v.select=any(obj.vertex_groups[g.group].name==motion.PARTS[pid] for g in v.groups)
        before=set(bpy.data.objects)
        bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.separate(type='SELECTED');bpy.ops.object.mode_set(mode='OBJECT')
        part=(set(bpy.data.objects)-before).pop()
        part.name=motion.PARTS[pid];objects[pid]=part
        bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);bpy.context.view_layer.objects.active=obj
    bake_preview_animation(objects)
    build.camera(10,-38,(0,-.80,1.02),3.50)
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'spear_stab_review.blend'))
    (ROOT/'review.html').write_text((SOURCE/'review.html').read_text())
    print('MOTION_REVIEW_DONE',flush=True)


if __name__=='__main__':
    main()
