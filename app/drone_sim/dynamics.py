import numpy as np
from numpy import sin, cos, tan, pi, sign
from scipy.integrate import ode
from .math_utils import quat2Dcm, quatToYPR_ZYX

class Quadcopter:
    def __init__(self, params, Ti=0, orient="NED"):
        self.params = params
        self.orient = orient
        
        # Initial State
        self.state = self.init_state()
        self.pos   = self.state[0:3]
        self.quat  = self.state[3:7]
        self.vel   = self.state[7:10]
        self.omega = self.state[10:13]
        self.wMotor = np.array([self.state[13], self.state[15], self.state[17], self.state[19]])
        self.vel_dot = np.zeros(3)
        self.omega_dot = np.zeros(3)
        self.acc = np.zeros(3)

        self.extended_state()
        self.forces()

        # Set Integrator
        self.integrator = ode(self.state_dot).set_integrator('dopri5', first_step='0.00005', atol='10e-6', rtol='10e-6')
        self.integrator.set_initial_value(self.state, Ti)

    def init_state(self):
        s = np.zeros(21)
        # Default hovering state
        s[3] = 1.0 # q0
        w_hover = self.params["w_hover"]
        s[13] = s[15] = s[17] = s[19] = w_hover
        return s

    def extended_state(self):
        self.dcm = quat2Dcm(self.quat)
        YPR = quatToYPR_ZYX(self.quat)
        self.euler = YPR[::-1] 
        self.psi, self.theta, self.phi = YPR

    def forces(self):
        self.thr = self.params["kTh"] * self.wMotor**2
        self.tor = self.params["kTo"] * self.wMotor**2

    def state_dot(self, t, state):
        cmd = self.current_cmd if hasattr(self, 'current_cmd') and self.current_cmd is not None else np.ones(4) * self.params["w_hover"]
        wind = self.current_wind if hasattr(self, 'current_wind') and self.current_wind is not None else Wind('None')
        mB = self.params["mB"]
        g = self.params["g"]
        dxm = self.params["dxm"]
        dym = self.params["dym"]
        IB = self.params["IB"]
        IBxx, IByy, IBzz = IB[0,0], IB[1,1], IB[2,2]
        Cd = self.params["Cd"]
        kTh, kTo = self.params["kTh"], self.params["kTo"]
        tau, damp, kp = self.params["tau"], self.params["damp"], self.params["kp"]
        minWmotor, maxWmotor = self.params["minWmotor"], self.params["maxWmotor"]
        IRzz = self.params.get("IRzz", 2.7e-5)
        uP = 1 if self.params.get("usePrecession", False) else 0

        # State Vector
        x, y, z = state[0:3]
        q0, q1, q2, q3 = state[3:7]
        xdot, ydot, zdot = state[7:10]
        p, q, r = state[10:13]
        wM1, wdotM1 = state[13:15]
        wM2, wdotM2 = state[15:17]
        wM3, wdotM3 = state[17:19]
        wM4, wdotM4 = state[19:21]

        # Motor Dynamics
        uMotor = cmd
        wddotM1 = (-2.0*damp*tau*wdotM1 - wM1 + kp*uMotor[0])/(tau**2)
        wddotM2 = (-2.0*damp*tau*wdotM2 - wM2 + kp*uMotor[1])/(tau**2)
        wddotM3 = (-2.0*damp*tau*wdotM3 - wM3 + kp*uMotor[2])/(tau**2)
        wddotM4 = (-2.0*damp*tau*wdotM4 - wM4 + kp*uMotor[3])/(tau**2)

        wMotor = np.clip(np.array([wM1, wM2, wM3, wM4]), minWmotor, maxWmotor)
        thrust = kTh * wMotor**2
        torque = kTo * wMotor**2
        ThrM1, ThrM2, ThrM3, ThrM4 = thrust
        TorM1, TorM2, TorM3, TorM4 = torque

        # Wind (simplified or passed)
        velW, qW1, qW2 = wind.randomWind(t)

        if self.orient == "NED":
            DynamicsDot = np.array([
                [xdot],
                [ydot],
                [zdot],
                [-0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r],
                [0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r],
                [0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r],
                [-0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r],
                [(Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)*cos(qW2) - xdot)**2 - 2*(q0*q2 + q1*q3)*(sum(thrust)))/mB],
                [(Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)*cos(qW2) - ydot)**2 + 2*(q0*q1 - q2*q3)*(sum(thrust)))/mB],
                [(-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot)**2 - (sum(thrust))*(q0**2 - q1**2 - q2**2 + q3**2) + g*mB)/mB],
                [((IByy - IBzz)*q*r - uP*IRzz*(wM1 - wM2 + wM3 - wM4)*q + (ThrM1 - ThrM2 - ThrM3 + ThrM4)*dym)/IBxx],
                [((IBzz - IBxx)*p*r + uP*IRzz*(wM1 - wM2 + wM3 - wM4)*p + (ThrM1 + ThrM2 - ThrM3 - ThrM4)*dxm)/IByy],
                [((IBxx - IByy)*p*q - TorM1 + TorM2 - TorM3 + TorM4)/IBzz]
            ])
        else: # ENU
            # Simplified ENU from quad.py
            DynamicsDot = np.array([
                [xdot],
                [ydot],
                [zdot],
                [-0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r],
                [0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r],
                [0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r],
                [-0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r],
                [(Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)*cos(qW2) - xdot)**2 + 2*(q0*q2 + q1*q3)*(sum(thrust)))/mB],
                [(Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)*cos(qW2) - ydot)**2 - 2*(q0*q1 - q2*q3)*(sum(thrust)))/mB],
                [(-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot)**2 + (sum(thrust))*(q0**2 - q1**2 - q2**2 + q3**2) - g*mB)/mB],
                [((IByy - IBzz)*q*r + uP*IRzz*(wM1 - wM2 + wM3 - wM4)*q + (ThrM1 - ThrM2 - ThrM3 + ThrM4)*dym)/IBxx],
                [((IBzz - IBxx)*p*r - uP*IRzz*(wM1 - wM2 + wM3 - wM4)*p + (-ThrM1 - ThrM2 + ThrM3 + ThrM4)*dxm)/IByy],
                [((IBxx - IBzz)*p*q + TorM1 - TorM2 + TorM3 - TorM4)/IBzz]
            ])

        sdot = np.zeros(21)
        sdot[0:13] = DynamicsDot.flatten()
        sdot[13] = wdotM1
        sdot[14] = wddotM1
        sdot[15] = wdotM2
        sdot[16] = wddotM2
        sdot[17] = wdotM3
        sdot[18] = wddotM3
        sdot[19] = wdotM4
        sdot[20] = wddotM4
        return sdot

    def update(self, t, Ts, cmd, wind):
        prev_vel = self.vel
        prev_omega = self.omega
        self.current_cmd = cmd
        self.current_wind = wind
        self.state = self.integrator.integrate(t, t+Ts)
        self.pos = self.state[0:3]
        self.quat = self.state[3:7]
        self.vel = self.state[7:10]
        self.omega = self.state[10:13]
        self.wMotor = np.array([self.state[13], self.state[15], self.state[17], self.state[19]])
        self.vel_dot = (self.vel - prev_vel)/Ts
        self.omega_dot = (self.omega - prev_omega)/Ts
        self.extended_state()
        self.forces()
