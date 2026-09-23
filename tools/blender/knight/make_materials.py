"""Original, deterministic material tiles used to bake the knight atlas.

No external images, models, downloads or image-generation services.
"""
from pathlib import Path
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[3] / "assets_dev/knight/rebuild/texture_sources"
SIZE = 512
RNG = np.random.default_rng(230923)


def noise(scale):
    size=max(2,SIZE//scale)
    image=Image.fromarray(RNG.integers(0,256,(size,size),dtype=np.uint8))
    return np.asarray(image.resize((SIZE,SIZE),Image.Resampling.BICUBIC),dtype=float)/255-.5


def tile(color, variation=1):
    signal=15*noise(48)+9*noise(14)+6*noise(3)+3*noise(1)
    return np.clip(np.array(color)[None,None,:]+signal[:,:,None]*variation,0,255).astype('uint8')


def save(name,pixels):
    image=pixels if isinstance(pixels,Image.Image) else Image.fromarray(pixels)
    image.save(ROOT/f'{name}.png')


def main():
    global ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=ROOT)
    ROOT=parser.parse_args().out
    ROOT.mkdir(parents=True,exist_ok=True)
    # Neutral cloth stores illumination and weave. Team tint is applied later.
    pixels=tile((208,205,197),.9).astype(float)
    grid_y,grid_x=np.indices((SIZE,SIZE))
    weave=2.5*np.cos(grid_x*math.pi)+2*np.cos(grid_y*math.pi/2)+1.3*np.sin((grid_x+grid_y)*math.pi/2)
    pixels+=weave[:,:,None]
    save('cloth',np.clip(pixels,0,255).astype('uint8'))

    image=Image.fromarray(tile((53,43,34),1.1))
    draw=ImageDraw.Draw(image)
    for i in range(950):
        x,y=RNG.integers(0,SIZE,2)
        length=int(RNG.integers(2,15))
        c=(79,66,50) if i%3 else (33,28,22)
        draw.line((int(x),int(y),int(x+length),int(y-2)),fill=c,width=1)
    save('leather',image.filter(ImageFilter.GaussianBlur(.22)))
    # Raised leather binding stays subtly lighter than the boot shaft.
    save('leather_trim',tile((79,57,36),1.1))

    image=Image.fromarray(tile((135,143,150),1.55))
    draw=ImageDraw.Draw(image)
    for i in range(240):
        x,y=map(int,RNG.integers(0,SIZE,2))
        length=int(RNG.integers(3,45))
        c=(176,180,181) if i%2 else (83,89,95)
        draw.line((x,y,x+length,y+int(RNG.integers(-7,8))),fill=c,width=1)
    save('steel',image)
    save('brass',tile((139,109,57),1.3))
    save('dark',tile((11,15,18),.22))

    pixels=tile((76,52,32),1.1).astype(float)
    grain=8*np.sin(grid_x*.13+2*np.sin(grid_y*.006))+4*np.cos(grid_x*.38+np.sin(grid_y*.016))
    pixels+=grain[:,:,None]
    save('wood',np.clip(pixels,0,255).astype('uint8'))

    image=Image.fromarray(tile((190,188,181),.9))
    draw=ImageDraw.Draw(image)
    for i in range(100):
        x,y=map(int,RNG.integers(0,SIZE,2))
        length=int(RNG.integers(2,15))
        draw.line((x,y,x+1,y+length),fill=(143,141,135),width=1)
    save('paint',image)

    # Overlapping, alternating rows of oval links. Ring shadows and restrained
    # highlights are baked detail, with no geometry or normal map per link.
    image=Image.fromarray(tile((27,32,36),.4))
    draw=ImageDraw.Draw(image)
    spacing_x,spacing_y=32,16
    for row,y in enumerate(range(-48,SIZE+48,spacing_y)):
        shift=16 if row%2 else 0
        for x in range(-64,SIZE+64,spacing_x):
            x+=shift
            tilt=3 if row%2 else -3
            points=[]
            for k in range(40):
                a=k*2*math.pi/40
                points.append((x+13*math.cos(a)+tilt*math.sin(a),y+10*math.sin(a)))
            dark=[(a+1,b+2) for a,b in points]
            draw.line(dark+[dark[0]],fill=(8,12,15),width=5)
            draw.line(points+[points[0]],fill=(79,90,98),width=3)
            draw.line(points[19:36],fill=(126,136,141),width=1)
            draw.line(points[1:17],fill=(44,53,62),width=1)
    save('mail',image)
    print('Wrote nine original material tiles:',ROOT)


if __name__=='__main__':
    main()
