from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt




## Example 1:
'''
    An example of using odeint is with the following differential equation with parameter k=0.3, the initial condition y(0)= 5 and the following differential equation.
    dy/dt = -k y(t)
    
    TODO: Solve it using odeint lib
'''
# function that returns dy/dt
def model(y,t):
    k = 0.3
    dydt = -k * y
    return dydt

# initial condition
y0 = 5

# time points
t = np.linspace(0,20)

# solve ODE
y = odeint(model,y0,t)

# plot results
plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()

## Example 2:
'''
    Additional Input Arguments:
    An optional fourth input is args that allows additional information to be passed into the model function. The args input is a tuple sequence of values. The argument is now an input to the model function by including an addition argument.

    y = odeint(model, y0, t, args)
    args: Additional inputs to the model.

'''

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dy/dt
def model(y,t,k):
    dydt = -k * y
    return dydt

# initial condition
y0 = 5

# time points
t = np.linspace(0,20)

# solve ODEs
k = 0.1
y1 = odeint(model,y0,t,args=(k,))
k = 0.2
y2 = odeint(model,y0,t,args=(k,))
k = 0.5
y3 = odeint(model,y0,t,args=(k,))

# plot results
plt.plot(t,y1,'r-',linewidth=2,label='k=0.1')
plt.plot(t,y2,'b--',linewidth=2,label='k=0.2')
plt.plot(t,y3,'g:',linewidth=2,label='k=0.5')
plt.xlabel('time')
plt.ylabel('y(t)')
plt.legend()
plt.show()