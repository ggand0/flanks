"""Spearman arm prototype, metres and glTF (+Y up, +Z forward).

Explicit rigid membership, no per-vertex spatial classification. Solve IK at
authoring time; export joint angle keys for inexpensive runtime interpolation.
The shaft follows the hand's contact point and turns inside the fist. Sliding
only changes the point along the shaft held by the closed hand while lowering
from vertical carry.
"""
import math
import numpy as np

S = np.array([-.224, 1.435, 0.])
E = np.array([-.310, 1.200, -.070])
W = np.array([-.360, 1.205, .160])
G = np.array([-.373, 1.185, .219])
JOINTS = {4:S, 9:E, 10:W, 8:G}
PARTS = {0:'body', 2:'leg_l', 3:'leg_r', 4:'arm_spear',
         5:'arm_shield', 8:'weapon', 9:'forearm_spear', 10:'hand_spear'}


def smooth(t):
    t=np.clip(t,0.,1.)
    return t*t*(3-2*t)


def rotation(a):
    c,s=math.cos(a),math.sin(a)
    return np.array([[1.,0,0],[0,c,s],[0,-s,c]])


def angle(v):
    return math.atan2(v[2],-v[1])


def wrap(a):
    return (a+math.pi)%(2*math.pi)-math.pi


def solve(grip, level=1.):
    # Hand rotation relative to vertical carry. Grip is the hand contact,
    # whereas W is the actual anatomical wrist: they must not be conflated.
    hand_turn=-math.radians(20)*level
    wrist=np.asarray(grip)-rotation(hand_turn)@(G-W)
    upper,fore=E-S,W-E
    l1,l2=np.linalg.norm(upper[1:]),np.linalg.norm(fore[1:])
    to=wrist-S
    d=np.linalg.norm(to[1:])
    assert abs(l1-l2)+1e-5 < d < l1+l2-1e-5, (grip,d,l1+l2)
    lead=math.acos((l1*l1+d*d-l2*l2)/(2*l1*d))
    sh=angle(to)-lead-angle(upper)
    elbow=S+rotation(sh)@upper
    ft=angle(wrist-elbow)-angle(fore)
    return np.array([wrap(sh),wrap(ft-sh),wrap(hand_turn-ft),level])


GUARD=solve([G[0],1.190,.170])
DRAW=solve([G[0],1.190,-.100])
STRIKE=solve([G[0],1.190,.395])
CARRY=np.zeros(4)
WINDUP_SECONDS=10/30
RECOVERY_SECONDS=.6


def authored_pose(windup=None, recovery=None):
    guard=np.array([G[0],1.190,.170])
    draw=np.array([G[0],1.190,-.100])
    strike=np.array([G[0],1.190,.395])
    if windup is not None:
        t=float(np.clip(windup,0,1))
        a,b,u=(guard,draw,t/.6) if t<=.6 else (draw,strike,(t-.6)/.4)
        return solve(a+(b-a)*smooth(u))
    if recovery is not None:
        return solve(strike+(guard-strike)*smooth((recovery-.08)/.84))
    return GUARD.copy()


# Uniform samples let the shader fetch two adjacent poses, with no IK and no
# key search per vertex. More than just endpoint rotations: those make the
# spear tip dip 12 cm halfway through this stroke.
WINDUP_TABLE=np.array([authored_pose(windup=i/32) for i in range(33)])
RECOVERY_TABLE=np.array([authored_pose(recovery=i/32) for i in range(33)])


def attack(windup=None, recovery=None):
    if windup is None and recovery is None:
        return GUARD.copy()
    table=WINDUP_TABLE if windup is not None else RECOVERY_TABLE
    t=float(np.clip(windup if windup is not None else recovery,0,1))*32
    i=min(int(t),31)
    return table[i]+(table[i+1]-table[i])*(t-i)


def sample(seconds, intro=False):
    if intro:
        if seconds<.65:
            return CARRY+(GUARD-CARRY)*smooth(seconds/.65)
        seconds-=.65
    t=seconds%1.6
    if t<.3:
        return GUARD.copy()
    if t<.3+WINDUP_SECONDS:
        return attack(windup=(t-.3)/WINDUP_SECONDS)
    if t<.3+WINDUP_SECONDS+RECOVERY_SECONDS:
        return attack(recovery=(t-.3-WINDUP_SECONDS)/RECOVERY_SECONDS)
    return GUARD.copy()


def transforms(pose):
    sh,el,wr,level=pose
    r1,r2,r3=rotation(sh),rotation(sh+el),rotation(sh+el+wr)
    ep=S+r1@(E-S)
    wp=ep+r2@(W-E)
    gp=wp+r3@(G-W)
    # Butt distance: 1.165 m upright, 0.450 m levelled. Same source mesh.
    slide=(G[1]-.020-.450)*level
    rw=rotation(-math.pi/2*level)
    return {4:(r1,S-r1@S),9:(r2,ep-r2@E),
            10:(r3,wp-r3@W),8:(rw,gp-rw@G+rw@np.array([0,slide,0]))}, (S,ep,wp,gp)


def deform(positions,parts,pose):
    out=np.array(positions,copy=True)
    for part,(r,t) in transforms(pose)[0].items():
        mask=parts==part
        out[mask]=out[mask]@r.T+t
    return out


def export_clip():
    return {'version':1,'space':'glTF metres, +Y up, +Z forward',
            'rotation':'pitch about +X, positive sends +Z toward +Y',
            'interpolation':'linear between adjacent uniform samples in each 33-entry table; carry uses smoothstep',
            'parts':PARTS,'joints_gltf_xyz':{str(k):v.tolist() for k,v in JOINTS.items()},
            'keys_radians':{k:v.tolist() for k,v in [('carry',CARRY),('guard',GUARD),('draw',DRAW),('strike',STRIKE)]},
            'windup_samples':WINDUP_TABLE.tolist(),'recovery_samples':RECOVERY_TABLE.tolist(),
            'windup_keys':[[0,'guard'],[.6,'draw'],[1,'strike']],
            'recovery_keys':[[0,'strike'],[.08,'strike'],[.92,'guard'],[1,'guard']],
            'windup_seconds':WINDUP_SECONDS,'recovery_seconds':RECOVERY_SECONDS,
            'butt_to_hand_levelled_m':.450,'weapon_slide_levelled_m':G[1]-.020-.450,
            'integration_note':'Use combat readiness throughout wind-up and recovery; do not reset to vertical carry at positive attack=0.'}
