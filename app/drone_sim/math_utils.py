import numpy as np
from numpy import sin, cos, tan, sqrt, pi

def quat2Dcm(q):
    """
    Quaternion to Direction Cosine Matrix
    """
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*q1**2 - 2*q3**2, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*q1**2 - 2*q2**2]
    ])

def quatToYPR_ZYX(q):
    """
    Quaternion to Yaw, Pitch, Roll (ZYX convention)
    """
    q0, q1, q2, q3 = q
    yaw = np.arctan2(2*(q1*q2 + q0*q3), q0**2 + q1**2 - q2**2 - q3**2)
    pitch = np.arcsin(-2*(q1*q3 - q0*q2))
    roll = np.arctan2(2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2)
    return np.array([yaw, pitch, roll])

def RotToQuat(R):
    """
    Rotation Matrix to Quaternion
    """
    tr = np.trace(R)
    if tr > 0:
        S = sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S = sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

def quatMultiply(q, r):
    """
    Quaternion Multiplication
    """
    q0, q1, q2, q3 = q
    r0, r1, r2, r3 = r
    return np.array([
        r0*q0 - r1*q1 - r2*q2 - r3*q3,
        r0*q1 + r1*q0 - r2*q3 + r3*q2,
        r0*q2 + r1*q3 + r2*q0 - r3*q1,
        r0*q3 - r1*q2 + r2*q1 + r3*q0
    ])

def inverse(q):
    """
    Quaternion Inverse
    """
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)

def vectNormalize(v):
    """
    Vector Normalization
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def mixerFM(quad, thrust, rateCtrl):
    """
    Mixer: Force/Moment to Motor Speeds squared
    """
    fm = np.array([thrust, rateCtrl[0], rateCtrl[1], rateCtrl[2]])
    w2 = np.dot(quad.params["mixerFMinv"], fm)
    w2 = np.clip(w2, quad.params["minWmotor"]**2, quad.params["maxWmotor"]**2)
    return np.sqrt(w2)
    
def makeMixerFM(params, orient="NED"):
    dxm = params["dxm"]
    dym = params["dym"]
    kTh = params["kTh"]
    kTo = params["kTo"] 

    if orient == "NED":
        mixerFM = np.array([[    kTh,      kTh,      kTh,      kTh],
                            [dym*kTh, -dym*kTh,  -dym*kTh, dym*kTh],
                            [dxm*kTh,  dxm*kTh, -dxm*kTh, -dxm*kTh],
                            [   -kTo,      kTo,     -kTo,      kTo]])
    else: # ENU
        mixerFM = np.array([[     kTh,      kTh,      kTh,     kTh],
                            [ dym*kTh, -dym*kTh, -dym*kTh, dym*kTh],
                            [-dxm*kTh, -dxm*kTh,  dxm*kTh, dxm*kTh],
                            [     kTo,     -kTo,      kTo,    -kTo]])
    return mixerFM
