import numpy as np
from .math_utils import get_rotation_matrix, get_angular_jacobian

class Quadcopter:
    """
    Quadcopter dynamics using Newton-Euler equations and explicit integration.
    State vector (12): [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
    """
    def __init__(self, params, orient="NED"):
        self.params = params
        self.orient = orient
        self.m = params["mB"]
        self.g = params["g"]
        self.I = params["IB"]
        self.I_inv = np.linalg.inv(self.I)
        self.k = params["kTh"]
        self.b = params["kTo"]
        self.L_x = params["dxm"]
        self.L_y = params["dym"]
        
        # Initial State: [pos(3), euler(3), vel(3), pqr(3)]
        self.state = np.zeros(12)
        # Standard hover setup
        self.psi, self.theta, self.phi = 0.0, 0.0, 0.0
        self.pos = self.state[0:3]
        self.euler = self.state[3:6]
        self.vel = self.state[6:9]
        self.omega = self.state[9:12]
        self.vel_dot = np.zeros(3)

    def get_forces_and_moments(self, rotor_speeds):
        """
        Calculates Total Thrust (T) and Control Moments (tau) from rotor speeds.
        """
        w_sq = np.square(rotor_speeds)
        
        # Total Thrust (directed along body-z)
        T = self.k * np.sum(w_sq)
        
        # Control Moments
        # Assumed layout: 1: Front-Right (+x, +y), 2: Front-Left (+x, -y), 3: Back-Left (-x, -y), 4: Back-Right (-x, +y)
        # Note: Adjusting based on common quad-x layout or user's provided logic
        # tau_x (Roll): Difference between left and right rotors
        # tau_y (Pitch): Difference between front and back rotors
        # Following provided math: tx = lk(w4^2 - w2^2), ty = lk(w3^2 - w1^2)
        tau_x = self.L_y * self.k * (w_sq[3] - w_sq[1]) 
        tau_y = self.L_x * self.k * (w_sq[2] - w_sq[0])
        # Yaw: Reactive torques
        tau_z = self.b * (w_sq[0] - w_sq[1] + w_sq[2] - w_sq[3])
        
        return T, np.array([tau_x, tau_y, tau_z])

    def calculate_accelerations(self, state, rotor_speeds):
        phi, theta, psi = state[3:6]
        v_world = state[6:9]
        p, q, r = state[9:12]
        
        T, tau = self.get_forces_and_moments(rotor_speeds)
        R = get_rotation_matrix(phi, theta, psi)
        
        # 1. Linear Acceleration
        # Thrust is in body frame [0, 0, T]
        thrust_body = np.array([0, 0, T])
        if self.orient == "NED":
            # In NED, force is Up (negative z), Gravity is [0, 0, g]
            thrust_body = np.array([0, 0, -T])
            gravity = np.array([0, 0, self.g])
        else: # ENU
            # In ENU, force is Up (positive z), Gravity is [0, 0, -g]
            thrust_body = np.array([0, 0, T])
            gravity = np.array([0, 0, -self.g])
            
        accel_linear = (R @ thrust_body) / self.m + gravity
        
        # Add drag if Cd exists
        if self.params.get("Cd", 0) > 0:
            accel_linear -= (self.params["Cd"] / self.m) * v_world
            
        # 2. Angular Acceleration
        # dot(p,q,r) = I_inv * (tau - omega x (I * omega))
        pqr = state[9:12]
        accel_angular = self.I_inv @ (tau - np.cross(pqr, self.I @ pqr))
        
        return accel_linear, accel_angular

    def update(self, t, dt, rotor_speeds, wind=None):
        """
        Explicit integration step (0.01s typically)
        """
        accel_linear, accel_angular = self.calculate_accelerations(self.state, rotor_speeds)
        
        # 1. Update Velocities
        self.state[6:9] += accel_linear * dt
        self.state[9:12] += accel_angular * dt
        self.vel_dot = accel_linear # Store for controller access
        
        # 2. Update Position
        self.state[0:3] += self.state[6:9] * dt
        
        # 3. Update Orientation (using Jacobian)
        phi, theta = self.state[3], self.state[4]
        if abs(np.cos(theta)) < 1e-3: theta += 1e-3 # Avoid singularity
        J_inv = get_angular_jacobian(phi, theta)
        euler_rates = J_inv @ self.state[9:12]
        self.state[3:6] += euler_rates * dt
        
        # Synch local properties
        self.pos = self.state[0:3]
        self.euler = self.state[3:6]
        self.vel = self.state[6:9]
        self.omega = self.state[9:12]
        self.psi = self.euler[2] # For controller/trajectory compatibility
