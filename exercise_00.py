"""
## Exercise 00:
Find a numerical solution to the following differential equations with the associated initial conditions.
Expand the requested time horizon until the solution reaches a steady state. Show a plot of the states (x(t) and/or y(t)).
Report the final value of each state as t---> inf.

Problem 1:
dy/dt = -y(t) + 1
y(0) = 0


Problem 2:
5 * dy/dt = -y(t) + u(t)
y(0) = 1
u(t) steps from 0 to 2 at t=10


Problem 3:
Solve for x(t) and y(t) and show that solutions are equivalent and show that the solutions are equivalent.
dx/dt = 3 exp(-t)
dy/dt = 3 - y(t)
x(0) = 0
y(0) = 0

Problem 4:
2*dx/dt = -x(t) + u(t)
5*dy/dt = -y(t) + x(t)
u = 2*s(t-5)
x(0) = 0
y(0) = 0
where s(t-5) is a step function that changes from zero to one at t=5.
When it is multiplied by two, it changes from zero to two at that same time, t=5.
.
"""

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

"""

Since we're looking for a unified framework to solve all 4 ODEs
We define a general representation for the ODE
This will be our system dynamics

Let the representation be:

    dz/dt = A*z + B*u

    A, B are coefficient matrices
    u is common for input function
    (This representation is chosen based on manual inspection of the 4 ODEs)

    
Now we find values of A, B, z, u for each ODE 

Problem 1:
A = [-1]
z = [x]
B = [1]
u = [1]

----------------------------
Problem 2:
A = [-0.2]
z = [x]
B = [1]
u = u (u(t) steps from 0 to 2 at t=10)

----------------------------
Problem 3:
A = [   0   0
        0   -1       ]

z = [   x
        y   ]

B = [   3   0
        0   3        ]

u = [   exp(-t)
        1            ]

----------------------------
Problem 4:
A = [   -0.5    0
        0.2   -0.2   ]

z = [   x
        y   ]

B = [   0.5
        0   ]

u = 2*s(t-5)
x(0) = 0
y(0) = 0
where s(t-5) is a step function that changes from zero to one at t=5.
When it is multiplied by two, it changes from zero to two at that same time, t=5.

        
"""


# Define parameters for each problem

# Problem 1

A1 = np.array([[-1]])
B1 = np.array([[1]])
z10 = np.array([0])


def u1(t):
    return np.array([1])


# Problem 2

A2 = np.array([[-0.2]])
B2 = np.array([[0.2]])
z20 = np.array([1])


def u2(t):
    u = 0 if t < 10 else 2
    return np.array([u])


# Problem 3

A3 = np.array([[0, 0], [0, -1]])
B3 = np.array([[3, 0], [0, 3]])
z30 = np.array([0, 0])


def u3(t):
    return np.array([np.exp(-t), 1])


# Problem 4

A4 = np.array([[-0.5, 0], [0.2, -0.2]])
B4 = np.array([[0.5], [0]])
z40 = np.array([0, 0])


def u4(t):
    s = 0 if t < 5 else 1
    u = 2 * s
    return np.array([u])


###########################################################

# System Dynamics => General ODE representation framework


def system_dynamics(z, t, A: np.ndarray, B: np.ndarray, u_func):
    z = np.array(z)
    u = np.array(u_func(t))
    dzdt = A @ z + B @ u
    return dzdt


def run(A, B, u_func, z0, T=40):
    t = np.linspace(0, T, 1000)

    sol = odeint(system_dynamics, z0, t, args=(A, B, u_func))

    return t, sol


###########################################################

# Problem 1

t, sol = run(A1, B1, u1, z10, 10)
plt.plot(t, sol, label="x(t)")
plt.xlabel("time")
plt.ylabel("z(t)")
plt.legend()
plt.show()

# # Problem 2

t, sol = run(A2, B2, u2, z20, 40)
u = [u2(t_) for t_ in t]
plt.plot(t, u, label="u")
plt.plot(t, sol, label="x(t)")
plt.xlabel("time")
plt.ylabel("z(t)")
plt.legend()
plt.show()

# # Problem 3

t, sol = run(A3, B3, u3, z30, 20)
u = [u3(t_) for t_ in t]
# plt.plot(t, u, label="u")
plt.plot(t, sol[:, 0], label="x(t)")
plt.plot(t, sol[:, 1], label="y(t)")
plt.xlabel("time")
plt.ylabel("z(t)")
plt.legend()
plt.show()

# Problem 4

t, sol = run(A4, B4, u4, z40, 50)
u = [u4(t_) for t_ in t]
plt.plot(t, u, "--", label="u")
plt.plot(t, sol[:, 0], label="x(t)")
plt.plot(t, sol[:, 1], label="y(t)")
plt.xlabel("time")
plt.ylabel("z(t)")
plt.legend()
plt.show()


## Hints:
"""
Problem 1:
# function that returns dy/dt
def model(y,t):
    dydt = -y + 1.0
    return dydt

# initial condition
y0 = 0

# time points
t = np.linspace(0,5)

# solve ODE
y = odeint(model,y0,t)

# plot results
plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()


Problem 2: 
# function that returns dy/dt
def model(y,t):
    # u steps from 0 to 2 at t=10
    if t<10.0:
        u = 0
    else:
        u = 2
    dydt = (-y + u)/5.0
    return dydt

# initial condition
y0 = 1

# time points
t = np.linspace(0,40,1000)

# solve ODE
y = odeint(model,y0,t)

# plot results
plt.plot(t,y,'r-',label='Output (y(t))')
plt.plot([0,10,10,40],[0,0,2,2],'b-',label='Input (u(t))')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()

Problem 3:

# function that returns dz/dt
def model(z,t):
    dxdt = 3.0 * np.exp(-t)
    dydt = -z[1] + 3
    dzdt = [dxdt,dydt]
    return dzdt

# initial condition
z0 = [0,0]

# time points
t = np.linspace(0,5)

# solve ODE
z = odeint(model,z0,t)

# plot results
plt.plot(t,z[:,0],'b-',label=r'$\frac{dx}{dt}=3 \; \exp(-t)$')
plt.plot(t,z[:,1],'r--',label=r'$\frac{dy}{dt}=-y+3$')
plt.ylabel('response')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()

Problem 4:
# function that returns dz/dt
def model(z,t,u):
    x = z[0]
    y = z[1]
    dxdt = (-x + u)/2.0
    dydt = (-y + x)/5.0
    dzdt = [dxdt,dydt]
    return dzdt

# initial condition
z0 = [0,0]

# number of time points
n = 401

# time points
t = np.linspace(0,40,n)

# step input
u = np.zeros(n)
# change to 2.0 at time = 5.0
u[51:] = 2.0

# store solution
x = np.empty_like(t)
y = np.empty_like(t)
# record initial conditions
x[0] = z0[0]
y[0] = z0[1]

# solve ODE
for i in range(1,n):
    # span for next time step
    tspan = [t[i-1],t[i]]
    # solve for next step
    z = odeint(model,z0,tspan,args=(u[i],))
    # store solution for plotting
    x[i] = z[1][0]
    y[i] = z[1][1]
    # next initial condition
    z0 = z[1] 

# plot results
plt.plot(t,u,'g:',label='u(t)')
plt.plot(t,x,'b-',label='x(t)')
plt.plot(t,y,'r--',label='y(t)')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()
"""
