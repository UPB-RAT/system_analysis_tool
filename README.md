## Solve Differential Equations in Python

Differential equations are solved in Python with the Scipy.integrate package using function odeint or solve_ivp. Another Python package that solves differential equations is GEKKO. See this link for the same tutorial in GEKKO versus ODEINT. ODEINT requires three inputs:
```bash
    y = odeint(model, y0, t)
```
1. model: Function name that returns derivative values at requested y and t values as dydt = model(y,t)

2. y0: Initial conditions of the differential states

3. t: Time points at which the solution should be reported. Additional internal points are often calculated to maintain accuracy of the solution but are not reported.

## Using the GUI

* The ui is built using [Streamlit](https://streamlit.io/). 
* Install it along with other packages.

```bash 
pip install -r requirements.txt
```

* Run it. 

```bash 
streamlit run app/main.py
```

---

### ODE Solver Examples

| Example | ODE System                                        | Initial Conditions | Input Function / Notes                    |
| ------- | ------------------------------------------------- | ------------------ | ----------------------------------------- |
| **1**   | `dy/dt = -y + u1`                                 | `y=0`              | `u1 = 1`                                  |
| **2**   | `dy/dt = 0.2*(-y + u1`)                           | `y=1`              | `u1 = 0 if t < 10 else 2`                 |
| **3**   | `dx/dt = 3*u1`<br>`dy/dt = 3*u2 - y`              | `x=0,y=0`          | `u1 = np.exp(-t)` <br> `u2 = 1`           |
| **4**   | `dx/dt = 0.5*(-x + u1)`<br>`dy/dt = 0.2*(-y + x)` | `x=0,y=0`          | `s = 0 if t < 5 else 1` <br> `u1 = 2 * s` |


### Phase Analysis Examples

| Example | OD Equations                                                                            |
| ------- | --------------------------------------------------------------------------------------- |
| **1**   | `dx/dt = (x - y)*(x**2 + y**2 - 1)` <br> `dy/dt = (x + y)*(x**2 + y**2 - 1)`            |
| **2**   | `dx/dt = -x + y*(1 + x)` <br> `dy/dt = -x*(1+x)`                                        |
| **3**   | `dx/dt = -x + y` <br> `dy/dt = 0.1*x - 2*y - x**2 - 0.1*x**3`                           |
| **4**   | `dx/dt = y` <br> `dy/dt = -x + y * (1 - 3*x**2 - 2*y**2)`                               |
| **5**   | `dx/dt = y` <br> `dy/dt = -x + (x**3)/6 - y `                                           |
| **6**   | `dx/dt = (1-x)*x - (2*x*y)/(1+x)` <br> `dy/dt = (1 - y/(1 + x)) * y`                    |
| **7**   | `dx1/dt = 2*x1 - x1*x2` <br> `dx2/dt = 2*x1**2 - x2`                                    |
| **8**   | `dx1/dt = x2 + x1 * (1 - x1**2 - x2**2)` <br> `dx2/dt = -x1 + x2 * (1 - x1**2 - x2**2)` |


