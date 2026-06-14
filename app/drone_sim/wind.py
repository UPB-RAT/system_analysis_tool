import numpy as np
from numpy import pi, sin, cos

class Wind:
    def __init__(self, mode='None', avg_speed=2.0, heading=90, elevation=-15):
        self.mode = mode
        self.avg_speed = avg_speed
        self.heading = heading * pi/180.0
        self.elevation = elevation * pi/180.0
        
    def randomWind(self, t):
        if self.mode == 'None':
            return 0, 0, 0
        
        # Simplified wind model
        v = self.avg_speed + 0.5 * sin(t)
        return v, self.heading, self.elevation
