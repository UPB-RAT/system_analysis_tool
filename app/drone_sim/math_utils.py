import numpy as np

def get_rotation_matrix(phi, theta, psi):
    """
    Calculates the ZYX rotation matrix (Body to World) as requested.
    phi: roll, theta: pitch, psi: yaw (in radians)
    """
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    R = np.array([
        [c_psi * c_theta, c_psi * s_theta * s_phi - s_psi * c_phi, c_psi * s_theta * c_phi + s_psi * s_phi],
        [s_psi * c_theta, s_psi * s_theta * s_phi + c_psi * c_phi, s_psi * s_theta * c_phi - c_psi * s_phi],
        [-s_theta,       c_theta * s_phi,                c_theta * c_phi]
    ])
    return R

def get_angular_jacobian(phi, theta):
    """
    Transforms body angular rates (p, q, r) to Euler angle rates (phi_dot, theta_dot, psi_dot).
    As specified: dot(Euler) = J_inv * [p, q, r]
    """
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    t_theta = np.tan(theta)

    # Note: Singularity at theta = +/- 90 degrees handled by small epsilon if needed in dynamics
    J_inv = np.array([
        [1, s_phi * t_theta, c_phi * t_theta],
        [0, c_phi,          -s_phi],
        [0, s_phi / c_theta, c_phi / c_theta]
    ])
    return J_inv

def vect_normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v

def mixerFM(quad, thrust, rateCtrl):
    """
    Mixed Force/Moment to Motor Speeds squared
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

    # Building mapping matrix based on:
    # T = k * sum(w_i^2)
    # tx = l * k * (w4^2 - w2^2)
    # ty = l * k * (w3^2 - w1^2)
    # tz = b * (w1^2 - w2^2 + w3^2 - w4^2)
    
    mixerFM = np.array([
        [kTh,  kTh,  kTh,  kTh],      # Total Thrust
        [0.0, -dym*kTh, 0.0, dym*kTh], # Roll (matches tx = k*dym*(w4^2 - w2^2))
        [-dxm*kTh, 0.0, dxm*kTh, 0.0], # Pitch (matches ty = k*dxm*(w3^2 - w1^2))
        [kTo, -kTo, kTo, -kTo]         # Yaw (matches tz = b*(w1^2 - w2^2 + w3^2 - w4^2))
    ])
    return mixerFM
