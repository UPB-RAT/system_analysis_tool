import numpy as np
from numpy import sin, cos, sqrt, pi
from numpy.linalg import norm
from .math_utils import vect_normalize, get_rotation_matrix, mixerFM

class Control:
    def __init__(self, quad, params, yawType, orient="NED"):
        self.params = params
        self.orient = orient
        self.sDesCalc = np.zeros(12) # Reduced from 16, no quaternions
        self.w_cmd = np.ones(4) * quad.params["w_hover"]
        self.thr_int = np.zeros(3)
        
        # Gains (Reduced to prevent 'rippling' oscillations)
        self.pos_P_gain = np.array([2.0, 2.0, 2.0])
        self.vel_P_gain = np.array([10.0, 10.0, 8.0])
        self.vel_D_gain = np.array([1.0, 1.0, 1.0])
        self.vel_I_gain = np.array([2.0, 2.0, 2.0])
        self.att_P_gain = np.array([10.0, 10.0, 2.0])
        self.rate_P_gain = np.array([0.1, 0.1, 0.1])
        self.rate_D_gain = np.array([0.02, 0.02, 0.05])
        
        if yawType == 0:
            self.att_P_gain[2] = 0
            
        self.setYawWeight()
        
        self.pos_sp = np.zeros(3)
        self.vel_sp = np.zeros(3)
        self.acc_sp = np.zeros(3)
        self.thrust_sp = np.zeros(3)
        self.eul_sp = np.zeros(3)
        self.pqr_sp = np.zeros(3)
        self.yawFF = 0.0
        
        self.velMaxAll = 5.0
        self.tiltMax = 50.0 * pi/180.0
        self.rateMax = np.array([200.0, 200.0, 150.0]) * pi/180.0

    def setYawWeight(self):
        roll_pitch_gain = 0.5 * (self.att_P_gain[0] + self.att_P_gain[1])
        self.yaw_w = np.clip(self.att_P_gain[2] / roll_pitch_gain, 0.0, 1.0)
        self.att_P_gain[2] = roll_pitch_gain

    def controller(self, traj, quad, sDes, Ts):
        self.pos_sp[:] = sDes[0:3]
        self.vel_sp[:] = sDes[3:6]
        self.acc_sp[:] = sDes[6:9]
        self.thrust_sp[:] = sDes[9:12]
        self.eul_sp[:] = sDes[12:15]
        self.pqr_sp[:] = sDes[15:18]
        self.yawFF = sDes[18]

        if traj.ctrlType == "xyz_pos":
            self.z_pos_control(quad, Ts)
            self.xy_pos_control(quad, Ts)
        
        self.saturateVel()
        self.z_vel_control(quad, Ts)
        self.xy_vel_control(quad, Ts)
        self.thrustToAttitude(quad, Ts)
        self.attitude_control(quad, Ts)
        self.rate_control(quad, Ts)

        # Mixer
        self.w_cmd = mixerFM(quad, norm(self.thrust_sp), self.rateCtrl)

        self.sDesCalc[0:3] = self.pos_sp
        self.sDesCalc[3:6] = self.vel_sp
        self.sDesCalc[6:9] = self.thrust_sp
        self.sDesCalc[9:12] = self.rate_sp

    def z_pos_control(self, quad, Ts):
        pos_z_error = self.pos_sp[2] - quad.pos[2]
        self.vel_sp[2] += self.pos_P_gain[2] * pos_z_error

    def xy_pos_control(self, quad, Ts):
        pos_xy_error = self.pos_sp[0:2] - quad.pos[0:2]
        self.vel_sp[0:2] += self.pos_P_gain[0:2] * pos_xy_error

    def saturateVel(self):
        totalVel_sp = norm(self.vel_sp)
        if totalVel_sp > self.velMaxAll:
            self.vel_sp = self.vel_sp / totalVel_sp * self.velMaxAll

    def z_vel_control(self, quad, Ts):
        vel_z_error = self.vel_sp[2] - quad.vel[2]
        if self.orient == "NED":
            thrust_z_sp = self.vel_P_gain[2]*vel_z_error - self.vel_D_gain[2]*quad.vel_dot[2] + quad.params["mB"]*(self.acc_sp[2] - quad.params["g"]) + self.thr_int[2]
            uMax, uMin = -quad.params["minThr"], -quad.params["maxThr"]
        else: # ENU
            thrust_z_sp = self.vel_P_gain[2]*vel_z_error - self.vel_D_gain[2]*quad.vel_dot[2] + quad.params["mB"]*(self.acc_sp[2] + quad.params["g"]) + self.thr_int[2]
            uMax, uMin = quad.params["maxThr"], quad.params["minThr"]

        stop_int_D = (thrust_z_sp >= uMax and vel_z_error >= 0.0) or (thrust_z_sp <= uMin and vel_z_error <= 0.0)
        if not stop_int_D:
            self.thr_int[2] += self.vel_I_gain[2] * vel_z_error * Ts * self.params.get("useIntegral", False)
            self.thr_int[2] = min(abs(self.thr_int[2]), quad.params["maxThr"]) * np.sign(self.thr_int[2])
        self.thrust_sp[2] = np.clip(thrust_z_sp, uMin, uMax)

    def xy_vel_control(self, quad, Ts):
        vel_xy_error = self.vel_sp[0:2] - quad.vel[0:2]
        thrust_xy_sp = self.vel_P_gain[0:2]*vel_xy_error - self.vel_D_gain[0:2]*quad.vel_dot[0:2] + quad.params["mB"]*(self.acc_sp[0:2]) + self.thr_int[0:2]

        thrust_max_xy_tilt = abs(self.thrust_sp[2]) * np.tan(self.tiltMax)
        thrust_max_xy = sqrt(max(0, quad.params["maxThr"]**2 - self.thrust_sp[2]**2))
        thrust_max_xy = min(thrust_max_xy, thrust_max_xy_tilt)

        self.thrust_sp[0:2] = thrust_xy_sp
        if norm(self.thrust_sp[0:2]) > thrust_max_xy:
            self.thrust_sp[0:2] = thrust_xy_sp / norm(self.thrust_sp[0:2]) * thrust_max_xy
        
        arw_gain = 2.0 / self.vel_P_gain[0:2]
        vel_err_lim = vel_xy_error - (thrust_xy_sp - self.thrust_sp[0:2]) * arw_gain
        self.thr_int[0:2] += self.vel_I_gain[0:2] * vel_err_lim * Ts * self.params.get("useIntegral", False)

    def thrustToAttitude(self, quad, Ts):
        yaw_sp = self.eul_sp[2]
        # In our Euler-first model, we find the body-z vector needed to achieve thrust_sp
        body_z = vect_normalize(self.thrust_sp)
        if self.orient == "NED":
            body_z = vect_normalize(-self.thrust_sp)
        
        # Calculate desired orientation (R_sp)
        y_C = np.array([-np.sin(yaw_sp), np.cos(yaw_sp), 0.0])
        body_x = vect_normalize(np.cross(y_C, body_z))
        body_y = np.cross(body_z, body_x)
        
        # Desired Euler angles (extracted from Rotation matrix)
        # ZYX convention: R = Rz(psi)Ry(theta)Rx(phi)
        # sin(theta) = -R[2,0]
        self.theta_sp = np.arcsin(np.clip(-body_x[2], -1, 1))
        self.phi_sp = np.arctan2(body_y[2], body_z[2])
        self.psi_sp = yaw_sp

    def attitude_control(self, quad, Ts):
        # Desired Rates p, q, r
        e_phi = self.phi_sp - quad.state[3]
        e_theta = self.theta_sp - quad.state[4]
        e_psi = self.psi_sp - quad.state[5]
        
        # KEY FIX: Angle wrapping for Yaw
        # Prevents the drone from rotating the long way around (flickering/nutting out)
        e_psi = (e_psi + np.pi) % (2 * np.pi) - np.pi
        
        # Simple P-control for orientation
        self.rate_sp = np.array([
            self.att_P_gain[0] * e_phi,
            self.att_P_gain[1] * e_theta,
            self.att_P_gain[2] * e_psi
        ])
        self.rate_sp = np.clip(self.rate_sp, -self.rateMax, self.rateMax)

    def rate_control(self, quad, Ts):
        # Rate control remains simple P; D term removed to reduce high-frequency ripple
        rate_error = self.rate_sp - quad.omega
        self.rateCtrl = self.rate_P_gain * rate_error
