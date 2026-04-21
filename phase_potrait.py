import numpy as np
import matplotlib.pyplot as plt

# 1. System Parameters
R, L, C = 2, 1, 1

# 2. Define the coordinate grid
x1_range = np.linspace(-2, 2, 20)
x2_range = np.linspace(-2, 2, 20)
X1, X2 = np.meshgrid(x1_range, x2_range)

# 3. Define the system of ODEs
# dx1/dt = (1/C) * x2
# dx2/dt = -(1/L) * x1 - (R/L) * x2
DX1 = (1/C) * X2
DX2 = -(1/L) * X1 - (R/L) * np.sin(X2)

# 4. Create the plot
plt.figure(figsize=(8, 8))

# Streamplot draws the continuous trajectories
plt.streamplot(X1, X2, DX1, DX2, color='royalblue', density=1.2)

# Quiver draws the specific direction vectors
plt.quiver(X1, X2, DX1, DX2, color='red', alpha=0.3)

# Formatting the chart
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('$x_1$ (Capacitor Voltage/Charge)')
plt.ylabel('$x_2$ (Inductor Current)')
plt.title(f'Phase Portrait: LC Circuit (R={R}, L={L}, C={C})')
plt.grid(alpha=0.3)
plt.gca().set_aspect('equal')

plt.show()