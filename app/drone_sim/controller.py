import numpy as np
from numpy import sin, cos, sqrt, pi
from numpy.linalg import norm
from .math_utils import vectNormalize, RotToQuat, quatMultiply, inverse, quat2Dcm, mixerFM

class Control:
    def __init__(self, quad, params, yawType, orient="NED"):
        self.params = params
        self.orient = orient
        self.sDesCalc = np.zeros(16)
        self.w_cmd = np.ones(4) * quad.params["w_hover"]
        self.thr_int = np.zeros(3)
        
        # Gains (can be made configurable)
        self.pos_P_gain = np.array([1.0, 1.0, 1.0])
        self.vel_P_gain = np.array([5.0, 5.0, 4.0])
        self.vel_D_gain = np.array([0.5, 0.5, 0.5])
        self.vel_I_gain = np.array([5.0, 5.0, 5.0])
        self.att_P_gain = np.array([8.0, 8.0, 1.5])
        self.rate_P_gain = np.array([1.5, 1.5, 1.0])
        self.rate_D_gain = np.array([0.04, 0.04, 0.1])
        
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
        self.sDesCalc[9:13] = self.qd
        self.sDesCalc[13:16] = self.rate_sp

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
        body_z = -vectNormalize(self.thrust_sp)
        if self.orient == "ENU":
            body_z = -body_z
        y_C = np.array([-sin(yaw_sp), cos(yaw_sp), 0.0])
        body_x = vectNormalize(np.cross(y_C, body_z))
        body_y = np.cross(body_z, body_x)
        R_sp = np.array([body_x, body_y, body_z]).T
        self.qd_full = RotToQuat(R_sp)

    def attitude_control(self, quad, Ts):
        e_z = quad.dcm[:,2]
        e_z_d = -vectNormalize(self.thrust_sp)
        if self.orient == "ENU":
            e_z_d = -e_z_d

        qe_red = np.zeros(4)
        qe_red[0] = np.dot(e_z, e_z_d) + sqrt(norm(e_z)**2 * norm(e_z_d)**2)
        qe_red[1:4] = np.cross(e_z, e_z_d)
        qe_red = vectNormalize(qe_red)
        
        self.qd_red = quatMultiply(qe_red, quad.quat)
        q_mix = quatMultiply(inverse(self.qd_red), self.qd_full)
        q_mix = q_mix * np.sign(q_mix[0])
        q_mix[0] = np.clip(q_mix[0], -1.0, 1.0)
        q_mix[3] = np.clip(q_mix[3], -1.0, 1.0)
        self.qd = quatMultiply(self.qd_red, np.array([cos(self.yaw_w*np.arccos(q_mix[0])), 0, 0, sin(self.yaw_w*np.arcsin(q_mix[3]))]))
        self.qe = quatMultiply(inverse(quad.quat), self.qd)
        self.rate_sp = (2.0*np.sign(self.qe[0])*self.qe[1:4]) * self.att_P_gain
        self.rate_sp += quat2Dcm(inverse(quad.quat))[:,2] * self.yawFF
        self.rate_sp = np.clip(self.rate_sp, -self.rateMax, self.rateMax)

    def rate_control(self, quad, Ts):
        rate_error = self.rate_sp - quad.omega
        self.rateCtrl = self.rate_P_gain * rate_error - self.rate_D_gain * quad.omega_dot
