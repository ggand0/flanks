"""Revised helm, mail skirt and boots for the knight L0."""
import math
import numpy as np
from mathutils import Vector
import geometry_knight as geo


def retain(g,excluded):
    result=geo.Geometry()
    old_to_new={}
    for face,color,smooth,label in zip(g.faces,g.colors,g.smooth,g.components):
        if label in excluded:
            continue
        mapped=[]
        for old in face:
            if old not in old_to_new:
                old_to_new[old]=len(result.vertices)
                result.vertices.append(g.vertices[old])
                result.parts.append(g.parts[old])
            mapped.append(old_to_new[old])
        result.faces.append(tuple(mapped))
        result.colors.append(color)
        result.smooth.append(smooth)
        result.components.append(label)
    return result


def hauberk(g):
    g.part,g.label='body','mail_hauberk'
    n=16
    rings=[(.425,.225,.149),(.50,.227,.151),(.63,.215,.145),
           (.73,.207,.135),(.84,.205,.13),(1.,.17,.12),
           (1.14,.175,.127),(1.29,.205,.135),(1.41,.224,.126),(1.465,.155,.102)]
    points=[]
    for z,rx,ry in rings:
        for i in range(n):
            angle=2*math.pi*(i+.5)/n
            points.append((rx*math.sin(angle),-ry*math.cos(angle),z))
    faces=[]
    for r in range(len(rings)-1):
        for i in range(n):
            # Narrow riding splits at front and back, ending below the hips.
            if r<3 and i in [7,15]:
                continue
            a,b=r*n+i,r*n+(i+1)%n
            faces.append((a,b,b+n,a+n))
    faces.append(tuple((len(rings)-1)*n+i for i in range(n)))
    g.add(points,faces,geo.MAIL,True)


def helmet(g):
    g.part,g.label='body','great_helm'
    # Oval crown, separate front plates and a shaped chin edge. The lower
    # section narrows slightly instead of flaring.
    outline=[(0,-1.08),(.40,-1.025),(.75,-.88),(.96,-.57),
             (1.,0),(.90,.60),(.65,.91),(.32,1.04),
             (0,1.075),(-.32,1.04),(-.65,.91),(-.90,.60),
             (-1.,0),(-.96,-.57),(-.75,-.88),(-.40,-1.025)]
    levels=[(1.510,.103,.107),(1.58,.108,.110),(1.675,.111,.111),
            (1.691,.111,.110),(1.783,.108,.106),(1.800,.103,.101)]
    def profile(z):
        return (float(np.interp(z,[v[0] for v in levels],[v[1] for v in levels])),
                float(np.interp(z,[v[0] for v in levels],[v[2] for v in levels])))
    def front(x,z,offset=.0015):
        rx,ry=profile(z)
        return float(np.interp(abs(x)/rx,[0,.40,.75,.96,1],[-1.08,-1.025,-.88,-.57,0]))*ry-offset
    points=[]
    for r,(z,rx,ry) in enumerate(levels):
        for x,y in outline:
            chin=(.013*abs(x)+.014*max(y,0)) if r==0 else 0
            points.append((x*rx,y*ry,z+chin))
    faces=[]
    for r in range(len(levels)-1):
        for i in range(16):
            # Open ocular strip across the front. Nasal reinforcement divides
            # it into two slits; the dark recess below sits behind it.
            if r==2 and i in [0,1,14,15]:
                continue
            a,b=r*16+i,r*16+(i+1)%16
            faces.append((a,b,b+16,a+16))
    g.add(points,faces,geo.STEEL)
    g.add(points[-16:],[tuple(range(16))],geo.STEEL)
    for side in [-1,1]:
        g.add([(side*x,front(side*x,z,-.010),z) for x,z in
               [(0,1.673),(.089,1.673),(.089,1.693),(0,1.693)]],[(0,1,2,3)],geo.DARK)
        # Riveted brow reinforcement sits on the real front surface.
        # Follow each change of face plane. A single chord cuts inside the
        # convex shell and hides parts of the brow.
        brow=[]
        for z in [1.700,1.713]:
            rx,_=profile(z)
            brow.extend((side*x,front(side*x,z,.0025),z) for x in [.010,.40*rx,.75*rx,.088])
        g.add(brow,[(i,i+1,i+5,i+4) for i in range(3)],geo.BRASS)
        for z in [1.592,1.618,1.644]:
            for x in [.031,.055,.078]:
                x*=side
                r=.004
                points=[(x,front(x,z-r),z-r),(x+r,front(x+r,z),z),
                        (x,front(x,z+r),z+r),(x-r,front(x-r,z),z)]
                g.add(points,[(0,1,2,3)],geo.DARK)
        for z in [1.56,1.62,1.709,1.765]:
            x=side*.094
            g.stud((x,front(x,z,.003),z),.003,geo.EDGE,(side*.5,-.866,0))
    # Narrow nasal plate follows the central ridge down to the chin.
    zrows=[1.534,1.58,1.675,1.714,1.781]
    points=[(x,front(x,z,.003),z) for z in zrows for x in [-.008,.008]]
    g.add(points,[(i,i+1,i+3,i+2) for i in range(0,8,2)],geo.BRASS)
    for z in [1.545,1.631,1.710,1.765]:
        g.stud((0,front(0,z,.005),z),.0033,geo.BRASS)
    # A thin structural crown band.
    points=[]
    for z in [1.775,1.783]:
        rx,ry=profile(z)
        points.extend((x*(rx+.001),y*(ry+.001),z) for x,y in outline)
    g.add(points,[(i,(i+1)%16,(i+1)%16+16,i+16) for i in range(16)],geo.EDGE)


def boots(g):
    for sign,part in [(1,'leg_l'),(-1,'leg_r')]:
        g.part,g.label=part,'boot'
        x=sign*.112
        # Cross sections run from heel to a restrained point. This gives the
        # vamp and toe their own outline instead of an elliptical blob.
        stations=[(.095,.050,.115),(.042,.061,.165),(-.050,.064,.125),
                  (-.137,.051,.094),(-.209,.027,.056),(-.258,.008,.036)]
        points=[]
        for y,w,h in stations:
            ring=[(-w,.018),(-w,.018+(h-.018)*.44),(-w*.62,h*.94),
                  (0,h),(w*.62,h*.94),(w,.018+(h-.018)*.44),(w,.018),(0,.018)]
            points.extend((x+xx,y,z) for xx,z in ring)
        faces=[]
        for r in range(len(stations)-1):
            for i in range(8):
                a,b=r*8+i,r*8+(i+1)%8
                faces.append((a,b,b+8,a+8))
        faces.extend([tuple(reversed(range(8))),tuple((len(stations)-1)*8+i for i in range(8))])
        g.add(points,faces,geo.LEATHER,True)
        # Thin sole with a heel, not a thick modern rounded outsole.
        outline=[(x-w,y) for y,w,h in stations]+[(x+w,y) for y,w,h in reversed(stations)]
        n=len(outline)
        points=[(xx,y,z) for z in [0,.017] for xx,y in outline]
        faces=[(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)]
        faces.extend([tuple(reversed(range(n))),tuple(n+i for i in range(n))])
        g.add(points,faces,geo.LEATHER_EDGE)
        shaft=[(.10,x,.008,.060,.075),(.18,x,.010,.060,.073),
               (.28,x,.010,.072,.080),(.36,x,.006,.082,.087),
               (.445,x,.002,.078,.083)]
        g.loft(shaft,10,geo.LEATHER,True,cap=False)
        g.label='boot_trim'
        def shaft_section(z):
            values=[float(np.interp(z,[r[0] for r in shaft],[r[c] for r in shaft])) for c in [2,3,4]]
            cy,rx,ry=values
            return (z,x,cy,rx+.0035,ry+.0035)
        for low,high in [(.428,.447),(.183,.196),(.342,.355)]:
            # Matching section count and angle makes the binding follow the
            # shaft instead of passing through it at the polygon corners.
            g.loft([shaft_section(low),shaft_section(high)],10,geo.LEATHER_EDGE,True,cap=False)


def chausses(g):
    for sign,part in [(1,'leg_l'),(-1,'leg_r')]:
        g.part,g.label=part,'mail_chausses'
        x=sign*.112
        # The boot encloses the lower leg. Terminate the hidden mail inside
        # its cuff; keeping an entire second shin caused visible intersections.
        g.loft([(.405,x,-.003,.069,.072),(.47,x,-.006,.070,.074),
                (.54,x,-.020,.077,.079),(.66,x,0,.089,.096),
                (.8,x,0,.103,.111),(.95,x,0,.108,.112)],12,geo.MAIL,True,True)


def build(g):
    g=retain(g,{'great_helm','mail_hauberk','mail_chausses','boot'})
    cloth={i for f,c in zip(g.faces,g.components) if c=='surcoat' for i in f}
    for i in cloth:
        x,y,z=g.vertices[i]
        if z<.94:
            z-=.100*max(0,min(1,(.94-z)/.305))
        g.vertices[i]=(x,y,z)
    hauberk(g)
    helmet(g)
    boots(g)
    chausses(g)
    for sign,part in [(-1,'arm_weapon'),(1,'arm_shield')]:
        g.part,g.label=part,'leather_cuff'
        g.tube([(sign*.351,-.052,1.18),(sign*.37,-.107,1.139),
                (sign*.390,-.164,1.105)],[.071,.066,.058],8,geo.LEATHER,True)
    return g


def measurements(g):
    def bounds(label):
        ids={i for face,c in zip(g.faces,g.components) if c==label for i in face}
        points=np.array([g.vertices[i] for i in ids])
        return {'min_xyz':points.min(axis=0).tolist(),'max_xyz':points.max(axis=0).tolist()}
    return {'great_helm':bounds('great_helm'),'mail_hauberk':bounds('mail_hauberk'),
            'surcoat':bounds('surcoat'),'boots':bounds('boot'),'boot_trim':bounds('boot_trim'),
            'helmet_shell_width_m':.222,'helmet_crown_width_m':.206,
            'helmet_lower_shell_width_m':.206,'mail_hem_m':.425,'surcoat_hem_m':.535,
            'boot_top_m':.447,'toe_tip_half_width_m':.008,'boot_length_m':.353,
            'animation_work':'None; existing rigid assignments and pivots retained'}
