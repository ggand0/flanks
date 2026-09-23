"""Compose the Blender renders into review sheets, without retouching."""
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]/"assets_dev/knight/rebuild"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def text(draw, pos, value, size=20, fill="#c1cbcf", bold=False):
    draw.text(pos,value,font=ImageFont.truetype(BOLD if bold else FONT,size),fill=fill)


def paste_render(canvas, name, box):
    image = Image.open(ROOT/name).convert("RGBA")
    image.thumbnail(box[2:],Image.Resampling.LANCZOS)
    canvas.alpha_composite(image,(box[0]+(box[2]-image.width)//2,box[1]))


def main():
    global ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",type=Path,default=ROOT)
    ROOT=parser.parse_args().out_dir
    report=json.loads((ROOT/"validation.json").read_text())
    canvas=Image.new("RGBA",(1680,900),"#202b30")
    draw=ImageDraw.Draw(canvas)
    text(draw,(48,28),"FLANKS  /  KNIGHT V2",32,"#f0e8d6",True)
    text(draw,(48,76),"L0 shape review  ·  oval helm  ·  longer mail skirt  ·  tapered boots  ·  woven diamond trim",20)
    for x,name,title in [(24,"knight_red.png","FRONT / RED TEAM"),
                          (576,"knight_back.png","BACK / RED TEAM"),
                          (1128,"knight_blue.png","FRONT / BLUE TEAM")]:
        paste_render(canvas,name,(x,118,528,640))
        text(draw,(x+26,778),title,19,"#e2d7bc",True)
    draw.line((48,824,1632,824),fill="#546267",width=1)
    text(draw,(48,842),f"{report['triangles']:,} triangles  |  1.80 m  |  5 rigid parts  |  1 opaque material  |  baked color + AO + team mask",21)
    canvas.convert("RGB").save(ROOT/"knight_review.png")

    canvas=Image.new("RGBA",(1800,750),"#202b30")
    draw=ImageDraw.Draw(canvas)
    text(draw,(35,25),"V2  /  SHAPE AND TEXTURE DETAILS",30,"#f0e8d6",True)
    for x,name,title in [(0,"detail_helm.png","OVAL GREAT HELM"),
                          (600,"detail_hem.png","LOWER MAIL HEM / WOVEN BAND"),
                          (1200,"detail_boots.png","TAPERED TOE / TALL LEATHER SHAFT")]:
        paste_render(canvas,name,(x+20,90,560,560))
        text(draw,(x+28,680),title,20,"#e2d7bc",True)
    canvas.convert("RGB").save(ROOT/"knight_details.png")

    canvas=Image.new("RGBA",(1200,590),"#202b30")
    draw=ImageDraw.Draw(canvas)
    text(draw,(30,20),"L0 SCREEN-SIZE CHECKS",28,"#f0e8d6",True)
    text(draw,(30,61),"50° above the horizon. All four use L0 geometry. Lower-detail meshes are pending review.",18)
    coverage={}
    for index,(px,factor) in enumerate([(60,3),(20,8),(8,20),(3,45)]):
        x=index*300
        source=Image.open(ROOT/f"L0_{px}px.png").convert("RGBA")
        alpha=source.getchannel("A")
        bbox=alpha.point(lambda value:255 if value>=128 else 0).getbbox()
        height=bbox[3]-bbox[1]
        coverage[str(px)]={"target_height_px":px,"raster_height_at_half_alpha":height}
        draw.rectangle((x+20,107,x+279,277),fill="#414b3d")
        native=source.crop((0,43,256,213))
        canvas.alpha_composite(native,(x+22,107))
        text(draw,(x+28,286),f"{px} px target / native pixels",17,"#e2d7bc",True)
        crop=source.crop((bbox[0]-1,bbox[1]-1,bbox[2]+1,bbox[3]+1))
        crop=crop.resize((crop.width*factor,crop.height*factor),Image.Resampling.NEAREST)
        canvas.alpha_composite(crop,(x+(300-crop.width)//2,326))
        text(draw,(x+28,532),f"{factor}× nearest-neighbor enlargement",14)
    measured=", ".join(str(coverage[str(px)]['raster_height_at_half_alpha']) for px in [60,20,8,3])
    text(draw,(30,565),f"Measured heights at 50% alpha: {measured} px. Edge antialiasing can reduce coverage.",16)
    canvas.convert("RGB").save(ROOT/"knight_size_checks.png")
    report["raster_coverage"]=coverage
    (ROOT/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print("Review sheets saved. Measured raster coverage:",coverage)


if __name__=="__main__":
    main()
