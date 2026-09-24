"""Original kettle-hatted spearman with articulated weapon arm for v3 review."""
import math
from collections import Counter
import numpy as np
from mathutils import Vector
from geometry_common import Geometry, PARTS, PIVOTS, JOINTS, MAIL, STEEL, EDGE, DARK, CLOTH, LEATHER, LEATHER_EDGE, IVORY, BRASS

SKIN=(.36,.235,.16,0.)
QUILT=(.46,.415,.32,0.)
HOSE=(.12,.102,.076,0.)
STITCH=(.23,.187,.13,0.)


def split_skirt(g,rings,color,n=16,split_rows=2,smooth=True,cap=True):
    points=[]
    for z,rx,ry in rings:
        for i in range(n):
            a=2*math.pi*(i+.5)/n
            points.append((rx*math.sin(a),-ry*math.cos(a),z))
    faces=[]
    for r in range(len(rings)-1):
        for i in range(n):
            if r<split_rows and i in [n//2-1,n-1]:
                continue
            a,b=r*n+i,r*n+(i+1)%n
            faces.append((a,b,b+n,a+n))
    if cap:
        faces.append(tuple((len(rings)-1)*n+i for i in range(n)))
    g.add(points,faces,color,smooth)


def clothing(g):
    g.part,g.label='body','quilt_hem'
    split_skirt(g,[(.535,.213,.127),(.58,.200,.120),(.66,.189,.112)],QUILT,16,2,cap=False)
    g.label='mail_body'
    split_skirt(g,[(.585,.213,.137),(.69,.205,.138),(.83,.184,.12),
                   (1.04,.160,.116),(1.20,.177,.133),(1.38,.197,.13),
                   (1.46,.148,.095)],MAIL,12,2)
    g.label='surcoat'
    # Tailored front and back panels; a shallow opening is covered by the coif.
    ts=[-1,-.72,-.40,-.12,.12,.40,.72,1]
    rows=[(.622,.208,.149),(.78,.199,.146),(.93,.177,.132),
          (1.045,.165,.127),(1.20,.180,.150),(1.36,.193,.151),(1.445,.187,.121)]
    points=[]
    for sign in [-1,1]:
        for r,(z,w,d) in enumerate(rows):
            for i,t in enumerate(ts):
                fold=[0,.010,-.002,.009,.009,-.002,.010,0][i]*(1 if r<3 else .30)
                h=z+([0,.005,.003,.018,.018,.003,.005,0][i] if r==0 else 0)
                if r==6 and abs(t)<.5:
                    h-=.063 if abs(t)<.2 else .031
                points.append((w*t,sign*(d*math.sqrt(1-.34*t*t)+fold),h))
    faces=[]
    for offset in [0,56]:
        for r in range(6):
            for i in range(7):
                if r<2 and i==3:
                    continue
                a=offset+r*8+i
                faces.append((a,a+1,a+9,a+8))
    for columns in [(0,1,2),(5,6,7)]:
        previous=[48+i for i in columns]
        for y in [-.05,0,.05]:
            current=[]
            for i in columns:
                current.append(len(points))
                points.append((ts[i]*.187,y,1.467+.003*abs(ts[i])-.004*abs(y)/.05))
            for k in range(2):
                faces.append((previous[k],previous[k+1],current[k+1],current[k]))
            previous=current
        back=[104+i for i in columns]
        for k in range(2):
            faces.append((previous[k],previous[k+1],back[k+1],back[k]))
    g.add(points,faces,CLOTH,True)
    # Narrow self-colored binding along the lower edges, without decoration.
    for side,offset in [(-1,0),(1,56)]:
        for a,b in [(0,1),(1,2),(2,3),(4,5),(5,6),(6,7)]:
            pa,pb=Vector(points[offset+a]),Vector(points[offset+b])
            shift=Vector((0,side*.002,.008))
            g.add([pa,pb,pb+shift,pa+shift],[(0,1,2,3)],CLOTH)


def belts(g):
    g.part,g.label='body','belt'
    outline=[]
    for sign,ts in [(-1,[-1,-.82,-.5,-.17,.17,.5,.82,1]),(1,[1,.82,.5,.17,-.17,-.5,-.82,-1])]:
        outline.extend((t*.177,sign*(.133*math.sqrt(1-.34*t*t)+.008)) for t in ts)
    points=[(x,y,z) for z in [1.026,1.060] for x,y in outline]
    g.add(points,[(i,(i+1)%16,(i+1)%16+16,i+16) for i in range(16)],LEATHER)
    g.box((-.025,-.147,1.044),(.049,.009,.035),BRASS)
    g.box((-.025,-.153,1.044),(.033,.004,.019),LEATHER)
    g.box((-.025,-.157,1.044),(.003,.003,.022),BRASS)
    g.box((.019,-.146,.984),(.019,.007,.108),LEATHER)
    for z in [.957,.981,1.005]:
        g.stud((.019,-.151,z),.0023,EDGE)
    # Empty leather scabbard at the left hip; the short sword is held forward.
    g.label='scabbard'
    g.tube([(.189,.022,1.008),(.206,.086,.78),(.224,.14,.50)], [.025,.024,.019],6,LEATHER,False)
    g.tube([(.224,.14,.50),(.225,.144,.477)],[.021,.008],6,EDGE,False)
    g.tube([(.189,.022,1.008),(.191,.028,.980)],[.027,.027],6,LEATHER_EDGE,False)


def kettle_hat(g):
    g.part,g.label='body','kettle_hat'
    # Rolled oval brim, with a modest downward slope. Rounded crown above it.
    n=20
    g.loft([(1.684,0,0,.113,.120),(1.729,0,.002,.106,.116),
            (1.770,0,.004,.078,.091),(1.794,0,.004,.036,.044),
            (1.800,0,.004,.010,.013)],n,STEEL,True,cap=False)
    g.add([(.010*math.sin(2*math.pi*i/n),.004-.013*math.cos(2*math.pi*i/n),1.800) for i in range(n)],
          [tuple(range(n))],STEEL,True)
    rings=[(1.693,.114,.123),(1.658,.175,.187),
           (1.653,.176,.188),(1.684,.111,.120)]
    points=[]
    for z,rx,ry in rings:
        for i in range(n):
            a=2*math.pi*i/n
            points.append((rx*math.sin(a),-ry*math.cos(a),z+.004*math.cos(a*2)))
    faces=[]
    for r in range(len(rings)):
        nxt=(r+1)%len(rings)
        for i in range(n):
            faces.append((r*n+i,r*n+(i+1)%n,nxt*n+(i+1)%n,nxt*n+i))
    g.add(points,faces,STEEL,True)
    g.label='helmet_band'
    g.loft([(1.695,0,.001,.115,.123),(1.705,0,.001,.115,.123)],n,EDGE,False,cap=False)
    for i in range(10):
        a=2*math.pi*(i+.5)/10
        g.stud((.115*math.sin(a),-.123*math.cos(a),1.700),.0027,EDGE,(math.sin(a),-math.cos(a),0))


def head_and_coif(g):
    g.part,g.label='body','face'
    # A continuous facial surface: chin, labiomental fold, lower lip, mouth,
    # upper lip, nasal wings/tip, recessed sockets and a projecting brow.
    # Each half-row runs from the centre to the side of the head. The rear
    # closure is entirely inside the coif: spend the triangles on the face.
    # Columns carry (x, y, z offset) relative to their row height.
    rows=[
        (1.505,[(0,-.055,0),(.014,-.054,0),(.029,-.046,.001),(.040,-.029,.003),(.045,.006,.005)]),
        (1.533,[(0,-.088,0),(.016,-.089,0),(.031,-.080,.001),(.045,-.057,.003),(.055,-.004,.009)]),
        (1.550,[(0,-.094,0),(.016,-.093,0),(.032,-.087,-.001),(.052,-.065,-.002),(.065,-.005,0)]),
        (1.566,[(0,-.103,0),(.012,-.102,0),(.028,-.094,.001),(.053,-.069,0),(.070,-.006,-.001)]),
        (1.571,[(0,-.101,0),(.011,-.101,.0006),(.028,-.094,-.001),(.054,-.071,0),(.072,-.006,-.001)]),
        (1.578,[(0,-.105,-.001),(.010,-.106,0),(.027,-.097,-.002),(.053,-.074,0),(.074,-.005,0)]),
        (1.596,[(0,-.112,-.003),(.012,-.114,0),(.029,-.095,0),(.053,-.080,.004),(.076,-.004,.004)]),
        (1.606,[(0,-.129,0),(.008,-.125,-.001),(.029,-.098,.002),(.054,-.085,.009),(.077,-.003,.010)]),
        (1.635,[(0,-.114,0),(.010,-.105,-.001),(.032,-.094,0),(.052,-.082,.001),(.077,-.001,.003)]),
        (1.652,[(0,-.110,-.002),(.014,-.111,-.003),(.033,-.104,.001),(.054,-.080,0),(.077,0,.002)]),
        (1.710,[(0,-.095,0),(.014,-.094,0),(.031,-.081,0),(.051,-.054,0),(.070,.007,0)]),
    ]
    points=[]
    for r,(z,half) in enumerate(rows):
        profile=[(-x,y,dz) for x,y,dz in reversed(half[1:])]+half
        for x,y,dz in profile:
            # Sub-millimetre asymmetry in the mouth and cheeks, without a
            # permanent snarl or a separate floating nose/lip shell.
            asym=.0007*(x/.077) if 2<=r<=7 else 0
            points.append((x,y,z+dz+asym))
    # Five extra vertices resolve the bridge/sidewalls instead of stretching
    # a single wedge from the eyes to the tip. Neighbouring pentagons include
    # the edge vertices, keeping the surface connected without T junctions.
    bridge={c:len(points)+i for i,c in enumerate(range(2,7))}
    points.extend([(-.023,-.101,1.620),(-.006,-.119,1.621),
                   (0,-.123,1.621),(.006,-.119,1.621),(.023,-.101,1.620)])
    faces=[]
    for r in range(10):
        for c in range(8):
            a=r*9+c
            if r==7 and 2<=c<=5:
                faces.extend([(a,a+1,bridge[c+1],bridge[c]),
                              (bridge[c],bridge[c+1],a+10,a+9)])
            elif r==7 and c==1:
                faces.append((a,a+1,bridge[2],a+10,a+9))
            elif r==7 and c==6:
                faces.append((a,a+1,a+10,a+9,bridge[6]))
            else:
                faces.append((a,a+1,a+10,a+9))
    boundary=list(range(9))+[r*9+8 for r in range(1,11)]
    boundary+=list(range(97,89,-1))+[r*9 for r in range(9,0,-1)]
    rear=len(points)
    points.append((0,.070,1.614))
    faces += [(a,rear,boundary[(i+1)%len(boundary)]) for i,a in enumerate(boundary)]
    g.add(points,faces,SKIN,True)
    g.part,g.label='body','mail_coif'
    rings=[(1.402,.140,.109),(1.456,.121,.102),(1.514,.093,.093),
           (1.542,.093,.101),(1.602,.100,.110),(1.670,.105,.114),(1.704,.102,.111)]
    n=20
    points=[]
    for z,rx,ry in rings:
        for i in range(n):
            a=2*math.pi*(i-n//2)/n
            points.append((rx*math.sin(a),-ry*math.cos(a),z))
    faces=[]
    for r in range(len(rings)-1):
        for i in range(n):
            a=2*math.pi*(i+.5-n//2)/n
            if r>=2 and abs(a)<.96:
                continue
            j,k=r*n+i,r*n+(i+1)%n
            faces.append((j,k,k+n,j+n))
    g.add(points,faces,MAIL,True)
    g.label='arming_cap'
    g.loft([(1.68,0,.005,.098,.104),(1.724,0,.005,.095,.102)],12,QUILT,True,cap=False)


def arms(g):
    for sign,part in [(1,'arm_shield')]:
        g.part,g.label=part,'quilt_sleeve'
        centers=[(sign*.185,0,1.414),(sign*.230,0,1.417),(sign*.275,0,1.38),
                 (sign*.306,.006,1.315),(sign*.333,-.016,1.228),
                 (sign*.352,-.069,1.16),(sign*.378,-.148,1.108)]
        g.tube(centers[2:],[.073,.066,.061,.057,.048],10,QUILT,True)
        g.label='mail_sleeve'
        g.tube(centers[:4]+[(sign*.315,.001,1.29)],[.070,.080,.083,.076,.073],10,MAIL,True)
        g.label='quilt_cuff'
        g.tube([(sign*.369,-.128,1.121),(sign*.378,-.149,1.107)],[.052,.050],10,QUILT,True,cap=False)
        g.label='hand'
        g.tube([(sign*.377,-.146,1.110),(sign*.390,-.185,1.092),
                (sign*.39,-.217,1.085)],[.038,.047,.037],8,SKIN,True)
        g.tube([(sign*.362,-.174,1.085),(sign*.363,-.219,1.065)],[.018,.017],6,SKIN,True)
    articulated_spear_arm(g)


def joint_ball(g, center, radius, color, n=8):
    # Poles along X: our pitch hinge rotates each ring in its own plane.
    # 32 triangles for n=8. Closed volume, shared centre across the joint.
    c=Vector(center)
    points=[c+Vector((-radius,0,0))]
    for x in [-.5,.5]:
        rr=radius*math.sqrt(1-x*x)
        for i in range(n):
            a=2*math.pi*i/n
            points.append(c+Vector((radius*x,rr*math.cos(a),rr*math.sin(a))))
    points.append(c+Vector((radius,0,0)))
    faces=[]
    for i in range(n):
        j=(i+1)%n
        faces.extend([(0,1+j,1+i),(1+i,1+j,1+n+j,1+n+i),
                      (1+n+i,1+n+j,len(points)-1)])
    g.add(points,faces,color,True)


def articulated_spear_arm(g):
    s=Vector(PIVOTS['arm_spear'])
    e=Vector(PIVOTS['forearm_spear'])
    w=Vector(PIVOTS['hand_spear'])
    # A stationary shoulder socket is part of the torso. The upper sleeve
    # starts within it, so rotating the arm never carries the socket away.
    g.part,g.label='body','mail_shoulder_socket'
    joint_ball(g,s,.091,MAIL)
    g.part,g.label='arm_spear','mail_sleeve'
    g.tube([s,s.lerp(e,.38),e],[.061,.078,.042],8,MAIL,True,cap=False)
    g.label='quilt_elbow_joint'
    joint_ball(g,e,.067,QUILT)
    g.part,g.label='forearm_spear','quilt_sleeve'
    g.tube([e,e.lerp(w,.48),w],[.043,.055,.027],8,QUILT,True,cap=False)
    g.label='quilt_cuff'
    g.tube([e.lerp(w,.86),w],[.045,.030],8,QUILT,True,cap=False)
    g.part,g.label='hand_spear','hand_wrist_joint'
    joint_ball(g,w,.043,SKIN,n=6)
    g.label='hand'
    delta=w-Vector((-.377,-.146,1.110))
    def move(coords):
        return [Vector(v)+delta for v in coords]
    g.tube(move([(-.377,-.146,1.110),(-.390,-.185,1.092),
                 (-.39,-.217,1.085)]),[.028,.047,.037],8,SKIN,True)
    g.tube(move([(-.362,-.174,1.085),(-.363,-.219,1.065)]),[.018,.017],6,SKIN,True)


def legs_and_shoes(g):
    for sign,part in [(1,'leg_l'),(-1,'leg_r')]:
        x=sign*.112
        g.part,g.label=part,'hose'
        g.loft([(.115,x,.003,.043,.054),(.30,x,.004,.060,.067),(.41,x,.004,.069,.074),
                (.49,x,-.006,.064,.069),(.56,x,-.012,.069,.073),
                (.69,sign*.100,0,.076,.080),(.95,sign*.075,0,.080,.090)],10,HOSE,True)
        g.label='shoe'
        stations=[(.080,.043,.085),(.028,.055,.130),(-.055,.057,.110),
                  (-.139,.044,.071),(-.205,.021,.041),(-.229,.008,.028)]
        points=[]
        for y,w,h in stations:
            points.extend((x+xx,y,z) for xx,z in [(-w,.011),(-w,h*.48),(-w*.65,h*.90),
                          (0,h),(w*.65,h*.90),(w,h*.48),(w,.011)])
        faces=[]
        for r in range(len(stations)-1):
            for i in range(7):
                a,b=r*7+i,r*7+(i+1)%7
                faces.append((a,b,b+7,a+7))
        faces.extend([tuple(reversed(range(7))),tuple(35+i for i in range(7))])
        g.add(points,faces,LEATHER,True)
        # Low quarters and a turnshoe seam, never a tall boot.
        g.loft([(.073,x,.010,.052,.064),(.135,x,.008,.050,.062),
                (.157,x,.007,.049,.061)],10,LEATHER,True,cap=False)
        g.label='shoe_edge'
        outline=[(x-w,y) for y,w,h in stations]+[(x+w,y) for y,w,h in reversed(stations)]
        n=len(outline)
        points=[(xx,y,z) for z in [0,.010] for xx,y in outline]
        faces=[(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)]
        faces.extend([tuple(reversed(range(n))),tuple(n+i for i in range(n))])
        g.add(points,faces,LEATHER_EDGE)
        g.loft([(.150,x,.007,.051,.063),(.157,x,.007,.051,.063)],10,LEATHER_EDGE,True,cap=False)


def heater(g):
    g.part,g.label='arm_shield','heater_face'
    tangent=Vector((.9063,.4226,0))
    normal=Vector((.4226,-.9063,0))
    center=Vector((.394,-.206,1.092))
    def at(u,z,offset=0):
        return Vector((.394,-.206,z))+tangent*u+normal*(.048*(1-(u/.205)**2)+offset)
    rows=[(1.404,.190),(1.342,.204),(1.152,.181),(.956,.130),(.783,.051),(.744,.004)]
    points=[at(u,z) for z,w in rows for u in [-w,-w*.5,0,w*.5,w]]
    faces=[(r*5+i,r*5+i+1,r*5+i+6,r*5+i+5) for r in range(5) for i in range(4)]
    g.add(points,faces,CLOTH,True)
    g.label='heater_wood'
    g.add([p-normal*.023 for p in points],faces,LEATHER_EDGE,True)
    g.label='heater_rim'
    boundary=[0,1,2,3,4,9,14,19,24,29,28,27,26,25,20,15,10,5]
    for k,a in enumerate(boundary):
        b=boundary[(k+1)%len(boundary)]
        pa,pb=Vector(points[a]),Vector(points[b])
        qa,qb=pa.lerp(center,.038)+normal*.003,pb.lerp(center,.038)+normal*.003
        g.add([pa+normal*.003,pb+normal*.003,qb,qa],[(0,1,2,3)],IVORY)
        g.add([pa,pb,pb-normal*.023,pa-normal*.023],[(0,1,2,3)],LEATHER)
    g.label='heater_fittings'
    for u,z in [(-.167,1.372),(.167,1.372),(-.151,1.145),(.151,1.145),(0,.788)]:
        g.stud(at(u,z,.009),.003,EDGE,normal)
    # Rear grip and arm strap, visible in the back review.
    for z,offset in [(1.22,.0),(1.06,.04)]:
        g.tube([at(-.075,z,-.044),at(.075,z-offset,-.044)],[.013,.013],6,LEATHER,True)


def spear(g):
    g.part,g.label='weapon','spear_shaft'
    x,y,_=PIVOTS['weapon']
    # 2.50 m complete weapon, butt 2 cm clear of the ground in this rest pose.
    g.tube([(x,y,.02),(x,y,1.10),(x,y,2.26)],[.016,.015,.012],8,LEATHER_EDGE,True)
    g.label='spear_steel'
    g.tube([(x,y,2.215),(x,y,2.308)],[.017,.013],8,STEEL,True)
    g.tube([(x,y,.02),(x,y,.10)],[.017,.017],6,STEEL,False)
    points=[]
    for z,w,t in [(2.282,.009,.005),(2.35,.035,.006),(2.425,.027,.004)]:
        points.extend([(x-w,y,z),(x,y-t,z),(x+w,y,z),(x,y+t,z)])
    faces=[]
    for r in range(2):
        for i in range(4):
            a,b=r*4+i,r*4+(i+1)%4
            faces.append((a,b,b+4,a+4))
    points.append((x,y,2.52))
    faces.extend((8+i,8+(i+1)%4,12) for i in range(4))
    g.add(points,faces,STEEL)


def sidearm_hilt(g):
    g.part,g.label='body','sidearm_hilt'
    g.tube([(.188,.020,1.00),(.177,-.006,1.114)],[.014,.014],6,LEATHER,True)
    g.tube([(.176,-.008,1.111),(.174,-.013,1.131)],[.026,.026],8,EDGE,False)
    g.box((.188,.020,1.008),(.136,.016,.016),EDGE)


def build():
    g=Geometry()
    clothing(g)
    belts(g)
    head_and_coif(g)
    kettle_hat(g)
    arms(g)
    legs_and_shoes(g)
    heater(g)
    spear(g)
    sidearm_hilt(g)
    return g


def measurements(g):
    result={}
    for label in sorted(set(g.components)):
        ids={i for face,c in zip(g.faces,g.components) if c==label for i in face}
        points=np.array([g.vertices[i] for i in ids])
        result[label]={'min_xyz':points.min(axis=0).tolist(),'max_xyz':points.max(axis=0).tolist(),
                       'triangles':sum(len(f)-2 for f,c in zip(g.faces,g.components) if c==label)}
    return result
