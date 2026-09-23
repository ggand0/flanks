"""FLANKS knight L0 geometry, original to this project.

Run only in a fresh background Blender process, never in a live session.
Writes exclusively below --out's directory. No external assets or downloads.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import bpy
import bmesh
from mathutils import Vector

PARTS = {"body": 0, "arm_weapon": 1, "leg_l": 2, "leg_r": 3,
         "arm_spear": 4, "arm_shield": 5, "arm_bow": 6}
PIVOTS = {"leg_l": (0.115, 0, 0.91), "leg_r": (-0.115, 0, 0.91),
          "arm_weapon": (-0.224, 0, 1.435), "arm_shield": (0.224, 0, 1.435)}

# Linear RGB plus team blend amount. Opaque even when team amount is zero.
MAIL = (0.075, 0.088, 0.099, 0.0)
STEEL = (0.23, 0.26, 0.285, 0.16)
EDGE = (0.42, 0.445, 0.45, 0.12)
DARK = (0.012, 0.017, 0.019, 0.0)
CLOTH = (0.34, 0.052, 0.032, 1.0)
HEM = (0.27, 0.035, 0.022, 1.0)
IVORY = (0.64, 0.56, 0.38, 0.0)
LEATHER = (0.072, 0.036, 0.019, 0.0)
LEATHER_EDGE = (0.115, 0.067, 0.032, 0.0)
BRASS = (0.37, 0.25, 0.10, 0.0)


class Geometry:
    def __init__(self):
        self.vertices, self.faces, self.colors = [], [], []
        self.parts, self.smooth, self.components = [], [], []
        self.part, self.label = "body", "body"

    def add(self, points, faces, color, smooth=False, mail=False):
        start = len(self.vertices)
        self.vertices.extend(tuple(p) for p in points)
        self.parts.extend([self.part] * len(points))
        for k, face in enumerate(faces):
            self.faces.append(tuple(start + i for i in face))
            if mail:
                # Small tonal facets suggest a woven surface, not huge rings.
                v = (0.95, 1.03, 0.99, 1.02, 0.97, 1.04)[k % 6]
                c = tuple(x * v for x in color[:3]) + (color[3],)
            else:
                c = color
            self.colors.append(c)
            self.smooth.append(smooth)
            self.components.append(self.label)

    def loft(self, rings, n, color, smooth=False, mail=False, cap=True):
        # Each ring is (z, center_x, center_y, radius_x, radius_y).
        points = []
        for z, x, y, rx, ry in rings:
            for i in range(n):
                angle = 2 * math.pi * i / n
                points.append((x + rx * math.sin(angle), y - ry * math.cos(angle), z))
        faces = []
        for r in range(len(rings) - 1):
            for i in range(n):
                a, b = r * n + i, r * n + (i + 1) % n
                faces.append((a, b, b + n, a + n))
        if cap:
            faces.extend([tuple(reversed(range(n))), tuple((len(rings) - 1) * n + i for i in range(n))])
        self.add(points, faces, color, smooth, mail)

    def tube(self, centers, radii, n, color, smooth=True, mail=False):
        points = []
        for k, center in enumerate(centers):
            tangent = Vector(centers[min(k + 1, len(centers) - 1)]) - Vector(centers[max(0, k - 1)])
            tangent.normalize()
            reference = Vector((0, 1, 0)) if abs(tangent.y) < .9 else Vector((1, 0, 0))
            u = tangent.cross(reference).normalized()
            v = tangent.cross(u).normalized()
            for i in range(n):
                a = i * 2 * math.pi / n
                points.append(Vector(center) + radii[k] * (u * math.cos(a) + v * math.sin(a)))
        faces = [tuple(reversed(range(n)))]
        for k in range(len(centers) - 1):
            for i in range(n):
                a, b = k * n + i, k * n + (i + 1) % n
                faces.append((a, b, b + n, a + n))
        faces.append(tuple((len(centers) - 1) * n + i for i in range(n)))
        self.add(points, faces, color, smooth, mail)

    def box(self, center, size, color):
        x, y, z = center
        a, b, c = (d / 2 for d in size)
        self.add([(x + i*a, y + j*b, z + k*c) for i, j, k in
                  [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                   (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]],
                 [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)], color)

    def stud(self, center, r, color, normal=(0, -1, 0)):
        normal = Vector(normal).normalized()
        u = normal.cross(Vector((0, 0, 1))).normalized()
        if u.length < 0.1:
            u = Vector((1, 0, 0))
        v = normal.cross(u)
        center = Vector(center)
        points = [center + r*(u*math.cos(i*math.pi/2)+v*math.sin(i*math.pi/2)) for i in range(4)]
        points.append(center + normal*r*0.45)
        self.add(points, [(0,1,4),(1,2,4),(2,3,4),(3,0,4)], color)

    def mail_link(self, center, normal, radius=.009):
        # Fine mail now lives in the texture atlas.
        pass

    def mesh(self, material):
        mesh = bpy.data.meshes.new("knight_L0_geometry")
        mesh.from_pydata(self.vertices, [], self.faces)
        mesh.update()
        obj = bpy.data.objects.new("L0", mesh)
        bpy.context.scene.collection.objects.link(obj)
        mesh.materials.append(material)
        for name in PARTS:
            if name in self.parts:
                group = obj.vertex_groups.new(name=name)
                group.add([i for i, p in enumerate(self.parts) if p == name], 1, "REPLACE")
        col = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="CORNER")
        mesh.color_attributes.active_color = col
        mesh.attributes.active_color = col
        uv0 = mesh.uv_layers.new(name="UVMap")
        uv1 = mesh.uv_layers.new(name="part")
        for polygon, color, smooth in zip(mesh.polygons, self.colors, self.smooth):
            polygon.use_smooth = smooth
            for index in polygon.loop_indices:
                vertex = mesh.loops[index].vertex_index
                part = self.parts[vertex]
                col.data[index].color = color
                uv0.data[index].uv = (0, 0)
                # glTF exporter transforms v to 1-v. Pre-compensate so the
                # GLB stores the pivot height, not 1 minus its height.
                height = PIVOTS[part][2] if part != "body" else 0
                uv1.data[index].uv = (PARTS[part], 1 - height)
        source_face = mesh.attributes.new(name="source_face", type="INT", domain="FACE")
        for polygon in mesh.polygons:
            source_face.data[polygon.index].value = polygon.index
        mesh.uv_layers.active_index = 0
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        return obj


def build():
    g = Geometry()
    g.label = "mail_hauberk"
    g.loft([(0.70,0,0,.20,.125),(.84,0,0,.205,.13),(1.00,0,0,.17,.12),
            (1.14,0,0,.175,.127),(1.29,0,0,.205,.135),(1.41,0,0,.224,.126),
            (1.465,0,0,.155,.102)], 16, MAIL, True, True)
    g.label = "mail_coif"
    g.loft([(1.405,0,0,.157,.114),(1.455,0,0,.124,.102),
            (1.53,0,0,.092,.092),(1.59,0,0,.09,.084)], 16, MAIL, True, True)

    # Tailored sleeveless surcoat: sculpted chest, belt pinch and heavy skirt.
    # Front and back share shoulder seams, with open armholes.
    g.label = "surcoat"
    xs = [-1, -.72, -.40, -.12, .12, .40, .72, 1]
    rows = [(.635,.213,.143),(.78,.207,.145),(.93,.187,.134),
            (1.045,.169,.125),(1.18,.18,.145),(1.34,.202,.153),(1.44,.194,.114)]
    cloth_points, cloth_faces = [], []
    for front in [True, False]:
        points = []
        sign = -1 if front else 1
        for r, (z,w,d) in enumerate(rows):
            for i,t in enumerate(xs):
                # Pleats are restrained above the belt and deepen at the hem.
                fold = [0,.012,-.004,.011,.011,-.004,.012,0][i] * (1 if r < 3 else .35)
                zz = z
                if r == 0:
                    zz += [0,.012,.005,.032,.032,.005,.012,0][i]
                if r == len(rows)-1 and abs(t) < .5:
                    zz -= .075 if abs(t) < .2 else .037
                points.append((w*t, sign*(d*math.sqrt(max(.2,1-.34*t*t))+fold), zz))
        faces = []
        for r in range(len(rows)-1):
            for i in range(len(xs)-1):
                if i == 3 and r < 2: # riding split, with mail visible beneath
                    continue
                a = r*len(xs)+i
                faces.append((a,a+1,a+1+len(xs),a+len(xs)))
        offset=len(cloth_points)
        cloth_points.extend(points)
        cloth_faces.extend(tuple(i+offset for i in f) for f in faces)
        # Narrow woven hem, following the skirt edge and split.
        for a,b in [(0,1),(1,2),(2,3),(4,5),(5,6),(6,7)]:
            pa,pb = Vector(points[a]),Vector(points[b])
            shift = Vector((0,sign*.002,.014))
            g.add([pa,pb,pb+shift,pa+shift],[(0,1,2,3)],HEM)
    # Connect to the exact chest/back vertices rather than overlaying straps.
    # This avoids z-fighting and gives the cloth one continuous shoulder seam.
    for columns in [(0,1,2),(5,6,7)]:
        previous=[48+i for i in columns]
        for y in [-.055,0,.055]:
            current=[]
            for i in columns:
                current.append(len(cloth_points))
                cloth_points.append((xs[i]*.194,y,1.478 + .004*abs(xs[i]) - .008*(abs(y)/.055)))
            for k in range(2):
                cloth_faces.append((previous[k],previous[k+1],current[k+1],current[k]))
            previous=current
        back=[104+i for i in columns]
        for k in range(2):
            cloth_faces.append((previous[k],previous[k+1],back[k+1],back[k]))
    g.add(cloth_points,cloth_faces,CLOTH,True)

    g.label = "belt"
    belt_outline=[]
    for sign,xx in [(-1,xs),(1,list(reversed(xs)))]:
        belt_outline.extend((t*.18,sign*(.137*math.sqrt(1-.34*t*t)+.008)) for t in xx)
    g.add([(x,y,z) for z in [1.029,1.06] for x,y in belt_outline],
          [(i,(i+1)%16,(i+1)%16+16,i+16) for i in range(16)],LEATHER)
    g.box((-.025,-.145,1.046),(.063,.011,.037),BRASS)
    g.box((-.025,-.152,1.046),(.043,.006,.021),LEATHER)
    g.box((-.025,-.157,1.046),(.004,.005,.023),BRASS)
    g.box((.04,-.144,.985),(.026,.009,.114),LEATHER)
    for z in [.95,.975,1.0]:
        g.stud((.04,-.150,z),.0027,BRASS)

    # Mail chausses follow thigh, knee, calf and ankle instead of cylinders.
    for sign,part in [(1,"leg_l"),(-1,"leg_r")]:
        g.part, g.label = part, "mail_chausses"
        x = sign*.112
        g.loft([(.14,x,0,.055,.065),(.23,x,.005,.058,.067),
                (.35,x,.006,.076,.079),(.47,x,-.006,.07,.074),
                (.54,x,-.02,.077,.079),(.66,x,0,.089,.096),
                (.8,x,0,.103,.111),(.95,x,0,.108,.112)], 12, MAIL, True, True)
        g.label = "boot"
        g.loft([(.015,x,-.057,.067,.133),(.043,x,-.064,.075,.143),
                (.092,x,-.056,.073,.141),(.14,x,-.005,.062,.078),
                (.21,x,.006,.059,.068)],10,LEATHER,True)
        g.loft([(0,x,-.06,.069,.136),(.018,x,-.06,.071,.139)],10,LEATHER_EDGE)
        g.loft([(.166,x,.004,.062,.071),(.183,x,.004,.062,.071)],10,LEATHER_EDGE)
        g.label="mail_links"
        for row in range(8):
            z=.245+row*.044
            # Front and outer quarter of each shin. Below the surcoat hem.
            radius=.060+(.016*math.sin((z-.23)/.34*math.pi))
            for angle in [-.60,0,.60]:
                a=angle+(.13 if row%2 else 0)
                g.mail_link((x+radius*math.sin(a),.001-(radius+.006)*math.cos(a),z),
                            (math.sin(a),-math.cos(a),0),.008)

    # Whole arms are rigid parts, with buried shoulder caps for rotation.
    for sign,part in [(-1,"arm_weapon"),(1,"arm_shield")]:
        g.part, g.label = part,"mail_sleeve"
        centers = [(sign*.183,0,1.420),(sign*.230,0,1.423),(sign*.279,0,1.387),
                   (sign*.31,.008,1.32),(sign*.337,-.012,1.215),
                   (sign*.359,-.075,1.155),(sign*.39,-.163,1.105)]
        radii=[.066,.081,.081,.076,.069,.064,.052]
        g.tube(centers,radii,12,MAIL,True,True)
        g.label="mail_links"
        for k in [2,3,4]:
            for along in [.2,.55,.86]:
                c=Vector(centers[k]).lerp(Vector(centers[k+1]),along)
                radius=radii[k]*(1-along)+radii[k+1]*along
                for angle in [-.4,.4]:
                    n=Vector((math.sin(angle),-math.cos(angle),0))
                    g.mail_link(c+n*(radius+.001),n,.008)
        g.label = "glove"
        g.tube([(sign*.384,-.144,1.112),(sign*.4,-.205,1.092),
                (sign*.4,-.24,1.083)], [.057,.057,.042],8,LEATHER,True)
        g.tube([(sign*.365,-.20,1.075),(sign*.365,-.24,1.064)], [.024,.018],6,LEATHER,True)

    g.part, g.label = "body", "great_helm"
    # Chamfered barrel cross-section, center ridge on the face.
    outline = [(0,-1.04),(.69,-.91),(1,-.48),(1,.46),(.69,.90),
               (0,1),(-.69,.90),(-1,.46),(-1,-.48),(-.69,-.91)]
    levels = [(1.513,.114,.106),(1.571,.128,.119),(1.675,.13,.125),
              (1.771,.12,.118),(1.8,.105,.104)]
    points = [(x*rx,y*ry,z) for z,rx,ry in levels for x,y in outline]
    faces=[]
    for r in range(len(levels)-1):
        for i in range(10):
            a,b=r*10+i,r*10+(i+1)%10
            faces.append((a,b,b+10,a+10))
    faces.extend([tuple(reversed(range(10))),tuple(40+i for i in range(10))])
    g.add(points,faces,STEEL)
    # Horizontal oculars, pitched onto the front's two planes.
    for side in [-1,1]:
        p=[(side*.014,-.1321,1.67),(side*.087,-.121,1.67),
           (side*.087,-.121,1.686),(side*.014,-.1321,1.686)]
        g.add(p,[(0,1,2,3)],DARK)
        g.add([(side*.012,-.134,1.689),(side*.091,-.122,1.689),
               (side*.091,-.122,1.697),(side*.012,-.134,1.697)],[(0,1,2,3)],EDGE)
    # Raised brass nasal and brow reinforcement.
    g.add([(-.01,-.131,1.543),(.01,-.131,1.543),(.009,-.137,1.71),
           (-.009,-.137,1.71)],[(0,1,2,3)],BRASS)
    for side in [-1,1]:
        g.add([(side*.009,-.136,1.707),(side*.093,-.119,1.707),
               (side*.093,-.119,1.722),(side*.009,-.136,1.722)],[(0,1,2,3)],BRASS)
        for x,z in [(.036,1.601),(.062,1.601),(.036,1.623),(.062,1.623),(.084,1.623)]:
            xx=side*x
            yy=-.134+abs(xx)*.17
            g.add([(xx-.0035,yy,z-.0035),(xx+.0035,yy,z-.0035),
                   (xx+.0035,yy,z+.0035),(xx-.0035,yy,z+.0035)],[(0,1,2,3)],DARK)
        for z in [1.548,1.581,1.644,1.711,1.755]:
            g.stud((side*.095,-.116,z),.0035,EDGE)
    for z in [1.552,1.651,1.715]:
        g.stud((0,-.139,z),.0035,BRASS)
    # Small rim follows all ten faces, with highlights at its upper lip.
    for z,rx,ry in [(1.522,.117,.11),(1.773,.121,.119)]:
        pp=[(x*rx,y*ry,zz) for zz in [z,z+.008] for x,y in outline]
        ff=[(i,(i+1)%10,(i+1)%10+10,i+10) for i in range(10)]
        g.add(pp,ff,EDGE)

    g.part,g.label="arm_shield","heater_shield"
    # Heater shield as a curved shallow shell, including a wooden back.
    tangent=Vector((.9063,.4226,0))
    normal=Vector((.4226,-.9063,0))
    def shield(u,z,offset=0):
        return Vector((.412,-.207,0))+tangent*u+normal*(.057*(1-(u/.236)**2)+offset)+Vector((0,0,z))
    shield_rows=[(1.415,.219),(1.353,.236),(1.16,.213),(.971,.15),(.793,.056),(.749,.004)]
    pp=[]
    for z,w in shield_rows:
        for u in [-w,-w*.5,0,w*.5,w]:
            pp.append(shield(u,z))
    ff=[]
    for r in range(len(shield_rows)-1):
        for i in range(4):
            a=r*5+i
            ff.append((a,a+1,a+6,a+5))
    g.add(pp,ff,CLOTH,True)
    g.add([p-normal*.023 for p in pp],list(reversed(ff)),LEATHER_EDGE,True)
    boundary=[0,1,2,3,4,9,14,19,24,29,28,27,26,25,20,15,10,5]
    for k,a in enumerate(boundary):
        b=boundary[(k+1)%len(boundary)]
        pa,pb=Vector(pp[a]),Vector(pp[b])
        # Inset rim, ivory rawhide on the front, dark thickness around side.
        ca=Vector((.412,-.207,1.135))
        qa,qb=pa.lerp(ca,.043)+normal*.004,pb.lerp(ca,.043)+normal*.004
        g.add([pa+normal*.004,pb+normal*.004,qb,qa],[(0,1,2,3)],IVORY)
        g.add([pa,pb,pb-normal*.023,pa-normal*.023],[(0,1,2,3)],LEATHER)
    for u,z in [(-.192,1.385),(.192,1.385),(-.179,1.164),(.179,1.164),(0,.79)]:
        g.stud(shield(u,z,.01),.004,BRASS,normal)

    g.part,g.label="arm_weapon","arming_sword"
    x,z=-.40,1.084
    # Grip behind the guard. A flattened diamond blade has a real ridge.
    g.tube([(x,-.175,z),(x,-.298,z)],[.018,.015],8,LEATHER)
    g.tube([(x,-.15,z),(x,-.173,z)],[.033,.029],8,BRASS,False)

    g.tube([(x-.119,-.278,z-.014),(x-.078,-.295,z),
            (x,-.31,z+.005),(x+.078,-.295,z),(x+.119,-.278,z-.014)],
           [.012,.012,.014,.012,.012],6,EDGE,False)
    points=[]
    for y,w,t in [(-.32,.026,.006),(-.39,.026,.0055),(-.85,.019,.004),(-1.015,.010,.003)]:
        points.extend([(x-w,y,z),(x,y,z+t),(x+w,y,z),(x,y,z-t)])
    faces=[]
    for r in range(3):
        for i in range(4):
            a,b=r*4+i,r*4+(i+1)%4
            faces.append((a,b,b+4,a+4))
    points.append((x,-1.075,z))
    faces.extend((12+i,12+(i+1)%4,16) for i in range(4))
    g.add(points,faces,EDGE)
    return g

