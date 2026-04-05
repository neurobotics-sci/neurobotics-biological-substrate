"""bubo/shared/kinematics/biomimetic_ik.py — v10.17"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class BiomechanicalJoint:
    name: str; dof_label: str; muscle_group: str
    axis: np.ndarray; offset_m: np.ndarray
    q: float = 0.0; q_min: float = -1.57; q_max: float = 1.57
    stiffness: float = 10.0; damping: float = 1.0


def _rot(axis, q):
    c,s=np.cos(q),np.sin(q); t=1-c; x,y,z=axis
    return np.array([[t*x*x+c,t*x*y-s*z,t*x*z+s*y],
                     [t*x*y+s*z,t*y*y+c,t*y*z-s*x],
                     [t*x*z-s*y,t*y*z+s*x,t*z*z+c]])


class DampedLeastSquaresIK:
    def __init__(self, joints: List[BiomechanicalJoint], damping=0.03, max_iter=60, tol=5e-4):
        self.joints=joints; self.lam=damping; self.max_iter=max_iter; self.tol=tol; self.n=len(joints)
    def forward_kinematics(self) -> Tuple[np.ndarray,List]:
        R=np.eye(3); p=np.zeros(3); jpos=[p.copy()]
        for j in self.joints:
            p=p+R@j.offset_m; R=R@_rot(j.axis,j.q); jpos.append(p.copy())
        return p,jpos
    def _jacobian(self):
        ee,jpos=self.forward_kinematics(); J=np.zeros((3,self.n)); R=np.eye(3)
        for i,j in enumerate(self.joints):
            z=R@j.axis; J[:,i]=np.cross(z,ee-jpos[i]); R=R@_rot(j.axis,j.q)
        return J
    def _limit_grad(self):
        g=np.zeros(self.n)
        for i,j in enumerate(self.joints):
            mid=(j.q_min+j.q_max)/2; r=j.q_max-j.q_min
            if r>0: g[i]=-((j.q-mid)/(r/2))**3
        return g*0.1
    def solve(self, target):
        for _ in range(self.max_iter):
            ee,_=self.forward_kinematics(); err=target-ee
            if np.linalg.norm(err)<self.tol: break
            J=self._jacobian(); JJT=J@J.T+self.lam**2*np.eye(3)
            dq=J.T@np.linalg.solve(JJT,err)
            null=np.eye(self.n)-J.T@np.linalg.pinv(J.T); dq+=null@self._limit_grad()
            for i,j in enumerate(self.joints): j.q=float(np.clip(j.q+dq[i],j.q_min,j.q_max))
        return np.array([j.q for j in self.joints])


def make_human_arm(side, shoulder_origin):
    sg=1.0 if side=="R" else -1.0
    return DampedLeastSquaresIK([
        BiomechanicalJoint("shoulder_flex","shoulder_flexion","deltoid_ant",np.array([1.,0.,0.]),shoulder_origin,q_min=np.radians(-60),q_max=np.radians(180),stiffness=15.0),
        BiomechanicalJoint("shoulder_abd","shoulder_abduction","deltoid_mid",np.array([0.,0.,1.]),np.zeros(3),q_min=np.radians(-30*sg),q_max=np.radians(180),stiffness=12.0),
        BiomechanicalJoint("shoulder_rot","shoulder_rotation","subscapularis",np.array([0.,1.,0.]),np.zeros(3),q_min=np.radians(-90),q_max=np.radians(90),stiffness=8.0),
        BiomechanicalJoint("elbow_flex","elbow_flexion","biceps",np.array([1.,0.,0.]),np.array([0.,-0.30,0.]),q_min=0.0,q_max=np.radians(150),stiffness=20.0),
        BiomechanicalJoint("wrist_flex","wrist_flexion","FCR",np.array([1.,0.,0.]),np.array([0.,-0.28,0.]),q_min=np.radians(-80),q_max=np.radians(70),stiffness=12.0),
        BiomechanicalJoint("wrist_rad","wrist_radial","FCR_rad",np.array([0.,0.,1.]),np.zeros(3),q_min=np.radians(-30),q_max=np.radians(20),stiffness=10.0),
        BiomechanicalJoint("forearm_ps","forearm_pron","pronator",np.array([0.,1.,0.]),np.zeros(3),q_min=np.radians(-85),q_max=np.radians(90),stiffness=8.0),
    ])


def make_human_leg(side, hip_origin):
    sg=1.0 if side=="R" else -1.0
    return DampedLeastSquaresIK([
        BiomechanicalJoint("hip_flex","hip_flexion","iliopsoas",np.array([1.,0.,0.]),hip_origin,q_min=np.radians(-20),q_max=np.radians(120),stiffness=25.0),
        BiomechanicalJoint("hip_abd","hip_abduction","gluteus_med",np.array([0.,0.,1.]),np.zeros(3),q_min=np.radians(-45*sg),q_max=np.radians(45),stiffness=20.0),
        BiomechanicalJoint("hip_rot","hip_rotation","TFL",np.array([0.,1.,0.]),np.zeros(3),q_min=np.radians(-45),q_max=np.radians(45),stiffness=15.0),
        BiomechanicalJoint("knee_flex","knee_flexion","quadriceps",np.array([1.,0.,0.]),np.array([0.,-0.43,0.]),q_min=np.radians(-135),q_max=0.0,stiffness=30.0),
        BiomechanicalJoint("ankle_df","ankle_dorsiflexion","tibialis",np.array([1.,0.,0.]),np.array([0.,-0.41,0.]),q_min=np.radians(-50),q_max=np.radians(20),stiffness=18.0),
        BiomechanicalJoint("subtalar","subtalar_inv","tibialis_post",np.array([0.,0.,1.]),np.zeros(3),q_min=np.radians(-35),q_max=np.radians(15),stiffness=10.0),
    ])


def make_cervical_spine():
    seg=np.array([0.,0.034,0.])
    joints=[]
    for name,ax,qm in [("C12_flex",[1,0,0],15),("C12_rot",[0,1,0],45),("C34_flex",[1,0,0],12),
                        ("C34_lat",[0,0,1],12),("C56_flex",[1,0,0],10),("C56_rot",[0,1,0],10),("C7_flex",[1,0,0],8)]:
        joints.append(BiomechanicalJoint(name,name,"SCM",np.array(ax,dtype=float),seg,
                                          q_min=np.radians(-qm),q_max=np.radians(qm),stiffness=5.0,damping=0.5))
    return DampedLeastSquaresIK(joints, damping=0.02)


def make_eye(side):
    return DampedLeastSquaresIK([
        BiomechanicalJoint("horizontal","LR_MR","lat_med_rectus",np.array([0.,1.,0.]),np.zeros(3),q_min=np.radians(-45),q_max=np.radians(45),stiffness=2.0,damping=0.2),
        BiomechanicalJoint("vertical","SR_IR","sup_inf_rectus",np.array([1.,0.,0.]),np.zeros(3),q_min=np.radians(-35),q_max=np.radians(35),stiffness=2.0,damping=0.2),
    ], damping=0.01, max_iter=20)
