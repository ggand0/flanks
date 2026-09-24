"""Check the actual GLB image payload and create RGB/mask inspection copies."""
import io
import argparse
import json
from collections import Counter
import struct
from pathlib import Path
import sys
import numpy as np
from PIL import Image

SOURCE=Path(__file__).resolve().parent
ROOT=SOURCE.parents[2]/"assets_dev/knight/rebuild"
sys.path.insert(0,str(SOURCE.parent))
import glb_inspect


def main():
    global ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",type=Path,default=ROOT)
    ROOT=parser.parse_args().out_dir
    doc,blob=glb_inspect.load(str(ROOT/'knight_textured.glb'))
    image=doc['images'][0]
    view=doc['bufferViews'][image['bufferView']]
    start=view.get('byteOffset',0)
    encoded=blob[start:start+view['byteLength']]
    embedded=Image.open(io.BytesIO(encoded))
    external=Image.open(ROOT/'knight_atlas.png')
    pixels=np.array(embedded)
    assert embedded.mode=='RGBA' and embedded.size==(2048,2048)
    assert np.array_equal(pixels,np.array(external))
    alpha=pixels[:,:,3]
    assert alpha.min()==0 and alpha.max()==255
    preserved=int(np.count_nonzero((alpha==0)&(pixels[:,:,:3].max(axis=2)>16)))
    assert preserved>10000, 'Non-team material RGB must survive zero team alpha'
    attrs=doc['meshes'][0]['primitives'][0]['attributes']
    assert 'COLOR_1' not in attrs, 'Use only one color layer with this exporter'
    _,colors=glb_inspect.accessor_values(doc,blob,attrs['COLOR_0'])
    assert min(c[3] for c in colors)==0 and max(c[3] for c in colors)==1
    assert all(all(abs(c-1)<1e-5 for c in rgba[:3]) for rgba in colors)
    embedded.convert('RGB').resize((1024,1024)).save(ROOT/'atlas_rgb_preview.png')
    embedded.getchannel('A').resize((1024,1024)).save(ROOT/'atlas_team_mask.png')
    report=json.loads((ROOT/'validation.json').read_text())
    primitive=doc['meshes'][0]['primitives'][0]
    _,parts=glb_inspect.accessor_values(doc,blob,attrs['TEXCOORD_1'])
    # The shared inspection helper normalizes all uint16 values for color
    # display. Indices are integers, never normalized; decode their bytes.
    accessor=doc['accessors'][primitive['indices']]
    view=doc['bufferViews'][accessor['bufferView']]
    code,width={5121:('B',1),5123:('H',2),5125:('I',4)}[accessor['componentType']]
    start=view.get('byteOffset',0)+accessor.get('byteOffset',0)
    stride=view.get('byteStride',width)
    assert not accessor.get('normalized',False)
    indices=[struct.unpack_from('<'+code,blob,start+i*stride)[0] for i in range(accessor['count'])]
    assert min(indices)>=0 and max(indices)<len(parts)
    assert all(parts[indices[i]]==parts[indices[i+1]]==parts[indices[i+2]] for i in range(0,len(indices),3))
    report['exported_part_vertices_by_id']=dict(sorted(Counter(int(p[0]) for p in parts).items()))
    report['part_triangles_by_id']=dict(sorted(Counter(int(parts[indices[i]][0]) for i in range(0,len(indices),3)).items()))
    assert set(report['part_triangles_by_id'])=={0,1,2,3,5}
    assert sum(report['part_triangles_by_id'].values())==report['triangles']
    report['all_triangles_have_one_part']=True
    report['image_byte_checks']={'dimensions':list(embedded.size),'mode':embedded.mode,
        'embedded_matches_external':True,'alpha_min':int(alpha.min()),'alpha_max':int(alpha.max()),
        'distinct_alpha_values':len(np.unique(alpha)),'non_team_pixels_with_rgb_preserved':preserved,
        'png_bytes':len(encoded),'rgba8_memory_mib':pixels.nbytes/1024**2,
        'rgba8_with_full_mips_mib':pixels.nbytes*4/3/1024**2}
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['image_byte_checks'],indent=2))


if __name__=='__main__':
    main()
