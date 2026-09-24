"""Byte-level asset checks plus full-cycle geometry/attachment measurements.

No Blender import is used here. Includes the actual GLB mesh and verifies that
the authoring probes exist in its exported positions before testing coverage.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
from scipy.spatial import ConvexHull,cKDTree

SOURCE=Path(__file__).resolve().parent
ROOT=SOURCE.parents[2]/"assets_dev/spearman/rebuild_v3"
sys.path.insert(0,str(SOURCE))
import motion as m
sys.path.insert(0,str(SOURCE.parent))
import glb_inspect


def load(path):
    doc,blob=glb_inspect.load(str(path))
    def acc(i):
        a=doc['accessors'][i];b=doc['bufferViews'][a['bufferView']]
        dtype={5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
        count=glb_inspect.NCOMP[a['type']];size=np.dtype(dtype).itemsize
        array=np.ndarray((a['count'],count),dtype=dtype,buffer=blob,
                         offset=a.get('byteOffset',0)+b.get('byteOffset',0),
                         strides=(b.get('byteStride',count*size),size)).copy()
        if a.get('normalized'): array=array.astype(float)/np.iinfo(dtype).max
        return array
    prim=doc['meshes'][0]['primitives'][0];at=prim['attributes']
    return doc,acc(at['POSITION']).astype(float),acc(at['TEXCOORD_1']),acc(prim['indices']).astype(int).reshape(-1,3)


def main():
    global ROOT
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir',type=Path,default=ROOT)
    ROOT=ap.parse_args().out_dir.resolve()
    saved=json.loads((SOURCE/'spear_stab.json').read_text())
    current=json.loads(json.dumps(m.export_clip()))
    # Blender's and system Python's NumPy builds can differ by one float64
    # rounding bit. Check the motion to sub-GPU precision, not byte equality.
    for key in ['windup_samples','recovery_samples']:
        np.testing.assert_allclose(saved.pop(key),current.pop(key),rtol=0,atol=1e-12)
    saved_keys,current_keys=saved.pop('keys_radians'),current.pop('keys_radians')
    assert saved_keys.keys()==current_keys.keys()
    for key in saved_keys:
        np.testing.assert_allclose(saved_keys[key],current_keys[key],rtol=0,atol=1e-12)
    assert saved==current, 'Regenerate spear_stab.json after changing motion.py'
    doc,pos,uv,tri=load(ROOT/'spearman.glb')
    part=np.rint(uv[:,0]).astype(int)
    assert np.max(abs(uv[:,0]-part))<1e-6
    assert set(part)==set(m.PARTS)
    assert np.all(part[tri[:,0]]==part[tri[:,1]]) and np.all(part[tri[:,0]]==part[tri[:,2]])
    assert 2000<=len(tri)<=3000
    assert abs(pos[part==0,1].max()-1.8)<1e-6 and abs(pos[:,1].min())<1e-6
    at=doc['meshes'][0]['primitives'][0]['attributes']
    assert doc['accessors'][at['COLOR_0']]['type']=='VEC4'
    assert len(doc['materials'])==1 and doc['materials'][0].get('alphaMode','OPAQUE')=='OPAQUE'
    nodes={n['name']:np.array(n.get('translation',[0,0,0])) for n in doc['nodes']}
    for pid,joint in m.JOINTS.items():
        assert np.max(abs(nodes['pivot_'+m.PARTS[pid]]-joint))<1e-6
        assert np.max(abs(uv[part==pid,1]-joint[1]))<1e-6
    probes=json.loads((ROOT/'joint_attachment_probes.json').read_text())
    trees={pid:cKDTree(pos[part==pid]) for pid in set(part)}
    for probe in probes:
        assert trees[probe['cap_part']].query(probe['cap_vertices'])[0].max()<1e-6
        hull=ConvexHull(probe['cap_vertices'])
        probe['planes']=hull.equations
        for seam in probe['seams']:
            assert trees[seam['part']].query(seam['points'])[0].max()<1e-6
    # 1001 points in each phase, plus carry transitions, at identical shared
    # boundaries. Include intermediate angle extrema and a repeated cycle.
    samples=[m.CARRY]
    samples += [m.CARRY+(m.GUARD-m.CARRY)*m.smooth(t) for t in np.linspace(0,1,501)]
    samples += [m.attack(windup=t) for t in np.linspace(0,1,1001)]
    samples += [m.attack(recovery=t) for t in np.linspace(0,1,1001)]
    max_edge=max_joint=max_contact=max_ortho=0.
    worst_coverage={p['joint']:-np.inf for p in probes}
    edges=np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]])
    edge0=np.linalg.norm(pos[edges[:,1]]-pos[edges[:,0]],axis=1)
    tip=pos[part==8][np.argmax(pos[part==8,1])]
    trajectories=[]
    for k,pose in enumerate(samples):
        xf,(s,e,w,g)=m.transforms(pose)
        xf[0]=(np.eye(3),np.zeros(3))
        transform=lambda points,pid:np.asarray(points)@xf[pid][0].T+xf[pid][1]
        # Shared joint positions under each adjacent rigid transform.
        max_joint=max(max_joint,np.linalg.norm(transform(m.S,4)-m.S),
                      np.linalg.norm(transform(m.E,4)-transform(m.E,9)),
                      np.linalg.norm(transform(m.W,9)-transform(m.W,10)))
        slide=(m.G[1]-.020-.450)*pose[3]
        held=m.G-np.array([0,slide,0])
        max_contact=max(max_contact,np.linalg.norm(transform(held,8)-transform(m.G,10)))
        for r,t in xf.values(): max_ortho=max(max_ortho,np.max(abs(r.T@r-np.eye(3))))
        for probe in probes:
            rc,tc=xf[probe['cap_part']]
            planes=probe['planes']
            for seam in probe['seams']:
                local=(transform(seam['points'],seam['part'])-tc)@rc
                distance=np.max(local@planes[:,:3].T+planes[:,3])
                worst_coverage[probe['joint']]=max(worst_coverage[probe['joint']],float(distance))
        # Checking every triangle edge across every sample also catches an
        # accidentally shared vertex or a mixed-part triangle.
        posed=m.deform(pos,part,pose)
        max_edge=max(max_edge,float(np.max(abs(np.linalg.norm(posed[edges[:,1]]-posed[edges[:,0]],axis=1)-edge0))))
        if 502<=k<1503:
            trajectories.append(transform(tip,8))
    assert max_joint<1e-12 and max_contact<1e-12 and max_edge<1e-12
    assert all(v<0 for v in worst_coverage.values()),worst_coverage
    trajectories=np.array(trajectories)
    assert np.ptp(trajectories[:,1])<.0021
    continuity={
        'guard_to_windup':np.linalg.norm(m.attack(windup=0)-m.GUARD),
        'windup_to_recovery':np.linalg.norm(m.attack(windup=1)-m.attack(recovery=0)),
        'recovery_to_guard':np.linalg.norm(m.attack(recovery=1)-m.GUARD)}
    assert max(continuity.values())<1e-12
    preserved=ROOT/'preserved_inputs.json'
    if preserved.exists():
        for file,expected in json.loads(preserved.read_text()).items():
            assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==expected,file
    report={'samples':len(samples),'triangles':{'L0':len(tri),'L1':None,'L2':None,'L3':None},
            'stage':'L0 articulation prototype; no runtime integration or LOD acceptance claimed',
            'character_height_m':float(pos[part==0,1].max()),
            'part_coverage':'100%; every triangle has exactly one rigid part',
            'exported_vertices_per_part':{m.PARTS[pid]:int(sum(part==pid)) for pid in sorted(set(part))},
            'pivot_positions_gltf_xyz':{k:v.tolist() for k,v in nodes.items() if k.startswith(('pivot_','joint_'))},
            'max_joint_separation_m':float(max_joint),'max_grip_axis_error_m':float(max_contact),
            'max_triangle_edge_length_change_m':max_edge,'max_rotation_orthogonality_error':float(max_ortho),
            'minimum_attachment_inset_m':{k:-v for k,v in worst_coverage.items()},
            'thrust_travel_m':float(np.ptp(trajectories[:,2])),
            'thrust_vertical_deviation_m':float(np.ptp(trajectories[:,1])),
            'phase_boundary_pose_errors':{k:float(v) for k,v in continuity.items()},
            'grip_distance_from_butt_m':.450,'original_inputs_unchanged':True if preserved.exists() else None,
            'scope_limits':['Stationary spear stab and carry-to-guard only','Shieldwall motion unchanged',
                            'No game-side shader execution yet','L1-L3 not authored in this stage']}
    (ROOT/'motion_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
