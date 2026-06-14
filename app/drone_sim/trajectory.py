import numpy as np
from numpy import pi
from numpy.linalg import norm

class Trajectory:
    def __init__(self, quad, ctrlType, trajSelect, waypoints_data=None):
        self.ctrlType = ctrlType
        self.xyzType = trajSelect[0]
        self.yawType = trajSelect[1]
        self.averVel = trajSelect[2]

        if waypoints_data is None:
            self.t_wps, self.wps, self.y_wps, self.v_wp = self.default_waypoints()
        else:
            self.t_wps, self.wps, self.y_wps, self.v_wp = waypoints_data

        self.end_reached = 0
        self.T_segment = np.diff(self.t_wps)
        
        if self.ctrlType == "xyz_pos":
            if self.averVel == 1:
                distance_segment = norm(self.wps[1:] - self.wps[:-1], axis=1)
                self.T_segment = distance_segment / self.v_wp
                self.t_wps = np.zeros(len(self.T_segment) + 1)
                self.t_wps[1:] = np.cumsum(self.T_segment)
            
            if 3 <= self.xyzType <= 6:
                self.deriv_order = int(self.xyzType - 2)
                self.coeff_x = self.minSomethingTraj(self.wps[:,0], self.T_segment, self.deriv_order)
                self.coeff_y = self.minSomethingTraj(self.wps[:,1], self.T_segment, self.deriv_order)
                self.coeff_z = self.minSomethingTraj(self.wps[:,2], self.T_segment, self.deriv_order)
        
        if self.yawType == 4:
            self.y_wps = np.zeros(len(self.t_wps))
        
        self.current_heading = quad.psi
        self.sDes = np.zeros(19)

    def default_waypoints(self):
        v_average = 1.6
        t_wps = np.array([3, 2, 2, 2, 2, 2])
        t_wps = np.cumsum(np.insert(t_wps, 0, 0)) # [0, 3, 5, 7, 9, 11, 13]
        wp = np.array([
            [0, 0, 0],
            [2, 2, 1],
            [-2, 3, -3],
            [-2, -1, -3],
            [3, -2, 1],
            [0, 0, 0]
        ])
        yaw = np.array([0, 20, -90, 120, 45, 0]) * pi/180.0
        return t_wps, wp, yaw, v_average

    def desiredState(self, t, Ts, quad):
        desPos = np.zeros(3)
        desVel = np.zeros(3)
        desAcc = np.zeros(3)
        desThr = np.zeros(3)
        desEul = np.zeros(3)
        desPQR = np.zeros(3)
        desYawRate = 0.

        if self.ctrlType == "xyz_pos":
            if self.xyzType == 0:
                desPos = self.wps[0]
            elif self.xyzType == 2: # Interp
                if t >= self.t_wps[-1]:
                    self.t_idx = -1
                    desPos = self.wps[-1]
                else:
                    self.t_idx = np.where(t <= self.t_wps)[0][0] - 1
                    scale = (t - self.t_wps[self.t_idx]) / (self.t_wps[self.t_idx+1] - self.t_wps[self.t_idx])
                    desPos = (1 - scale) * self.wps[self.t_idx] + scale * self.wps[self.t_idx+1]
            elif 3 <= self.xyzType <= 6: # Min something
                nb_coeff = self.deriv_order * 2
                if t == 0:
                    self.t_idx = 0
                    desPos = self.wps[0]
                elif t >= self.t_wps[-1]:
                    self.t_idx = -1
                    desPos = self.wps[-1]
                else:
                    self.t_idx = np.where(t <= self.t_wps)[0][0] - 1
                    scale = (t - self.t_wps[self.t_idx])
                    start, end = nb_coeff * self.t_idx, nb_coeff * (self.t_idx + 1)
                    t0 = self.get_poly_cc(nb_coeff, 0, scale)
                    desPos = np.array([self.coeff_x[start:end].dot(t0), self.coeff_y[start:end].dot(t0), self.coeff_z[start:end].dot(t0)])
                    t1 = self.get_poly_cc(nb_coeff, 1, scale)
                    desVel = np.array([self.coeff_x[start:end].dot(t1), self.coeff_y[start:end].dot(t1), self.coeff_z[start:end].dot(t1)])
                    t2 = self.get_poly_cc(nb_coeff, 2, scale)
                    desAcc = np.array([self.coeff_x[start:end].dot(t2), self.coeff_y[start:end].dot(t2), self.coeff_z[start:end].dot(t2)])

            # Yaw
            if self.yawType == 2: # Interp
                if t >= self.t_wps[-1]:
                    desEul[2] = self.y_wps[-1]
                else:
                    scale = (t - self.t_wps[self.t_idx]) / (self.t_wps[self.t_idx+1] - self.t_wps[self.t_idx])
                    desEul[2] = (1 - scale) * self.y_wps[self.t_idx] + scale * self.y_wps[self.t_idx+1]
                    desYawRate = (desEul[2] - self.current_heading) / Ts
                    self.current_heading = desEul[2]
            elif self.yawType == 3: # Follow
                if t > 0:
                    desEul[2] = np.arctan2(desVel[1], desVel[0])
                    if (np.sign(desEul[2]) != np.sign(self.current_heading) and abs(desEul[2]-self.current_heading) >= 2*pi-0.1):
                        self.current_heading += np.sign(desEul[2]) * 2*pi
                    desYawRate = (desEul[2] - self.current_heading) / Ts
                    self.current_heading = desEul[2]

        self.sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate))
        return self.sDes

    def get_poly_cc(self, n, k, t):
        cc = np.ones(n)
        D = np.linspace(n-1, 0, n)
        for i in range(n):
            for j in range(k):
                cc[i] = cc[i] * D[i]
                D[i] = max(0, D[i] - 1)
        for i, c in enumerate(cc):
            cc[i] = c * np.power(t, D[i])
        return cc

    def minSomethingTraj(self, waypoints, times, order):
        n = len(waypoints) - 1
        nb_coeff = order * 2
        A = np.zeros([nb_coeff*n, nb_coeff*n])
        B = np.zeros(nb_coeff*n)
        for i in range(n):
            B[i] = waypoints[i]
            B[i + n] = waypoints[i+1]
            A[i, nb_coeff*i:nb_coeff*(i+1)] = self.get_poly_cc(nb_coeff, 0, 0)
            A[i+n, nb_coeff*i:nb_coeff*(i+1)] = self.get_poly_cc(nb_coeff, 0, times[i])
        for k in range(1, order):
            A[2*n+k-1, :nb_coeff] = self.get_poly_cc(nb_coeff, k, 0)
            A[2*n+(order-1)+k-1, -nb_coeff:] = self.get_poly_cc(nb_coeff, k, times[-1])
        for i in range(n-1):
            for k in range(1, nb_coeff-1):
                row = 2*n + 2*(order-1) + i*(nb_coeff-2) + k - 1
                A[row, i*nb_coeff:(i+2)*nb_coeff] = np.concatenate((self.get_poly_cc(nb_coeff, k, times[i]), -self.get_poly_cc(nb_coeff, k, 0)))
        return np.linalg.solve(A, B)
