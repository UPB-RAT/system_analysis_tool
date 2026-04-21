import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import plotly.graph_objects as go
import os, sys

from ode import extract_expressions, run_linear_system_dynamics


######################################

# config

ode_types = ["Linear ODE", "Nonlinear ODE"]
min_time = 1
max_time = 100

min_points = 100
max_points = 5000


######################################


st.set_page_config(page_title="ODE Solver", layout="wide")
st.title("ODE Solver")
T = st.sidebar.slider("Time Horizon", min_time, max_time, int(max_time * 0.4))

######################################


st.header("Linear System: dz/dt = A z + B u")

st.markdown(
    """### Important Instructions

* Enter the ODE expressions in Python syntax.
* All input functions and constants must be defined as variables starting with "u" as follows u1, u2 ... un, where n in the state dimension.
* u1, u2... must be explicitly defines in the input function section.
* Example:
```python
dx/dt = 0.5*(-x + u1)
dy/dt = 0.2*(-y + x)
```
"""
)


st.divider()


col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area(
        "Enter ODEs (one per line)", "dx/dt = 0.5*(-x + u1)\ndy/dt = 0.2*(-y + x)"
    )

    initial_conditions_str = st.text_input("Initial Condition", "x=0,y=0")


with col2:

    st.write(
        """Define the Input functions `u(t)` using Python syntax.

- Input parameter: `t` (time)  
- Make sure to store the final values in u1, u2... etc

**Example:**
```python
u1 = 0 if t < 10 else 2"""
    )

    u_func_str = st.text_area(
        "Enter input function", "s = 0 if t < 5 else 1\nu1 = 2 * s"
    )


######################################


if st.button("Solve ODE System"):

    try:

        st.divider()

        equations = user_input.strip().split("\n")
        if not equations or any("=" not in eq for eq in equations):
            st.error("Invalid ODE format. Use: dx/dt = ...")
            st.stop()

        expressions, variables = extract_expressions(equations)
        variables_list = sorted(
            [v for v in variables if v.name != "u"], key=lambda x: x.name
        )

        state_dimension = len(equations)

        initial_conditions_map = {}
        try:
            for item in initial_conditions_str.split(","):
                key, value = item.split("=")
                initial_conditions_map[key.strip()] = float(value.strip())
        except:
            st.error("Invalid initial conditions format. Example: x=0, y=1")
            st.stop()

        initial_conditions = [initial_conditions_map[v.name] for v in variables_list]

        z0 = initial_conditions

        # Find A and B

        u_symbols = [sp.Symbol(f"u{i+1}") for i in range(state_dimension)]

        def u_func(t):
            localsParameter = {"t": t}
            exec(u_func_str, globals(), localsParameter)

            u = []
            for i in range(state_dimension):
                var = f"u{i+1}"
                if var in localsParameter.keys():
                    u.append(localsParameter[var])
                else:
                    u.append(1)
            return np.array(u)

        A = pd.DataFrame(np.zeros((state_dimension, state_dimension)))
        test_u = u_func(0)
        input_dim = test_u.shape[0]
        B = pd.DataFrame(np.zeros((state_dimension, input_dim)))

        u_symbol = sp.Symbol("u")

        for k, expression in enumerate(expressions):
            for j, v in enumerate(variables_list):

                A.loc[k, j] = expression.coeff(v)

            # B.loc[k, 0] = expression.coeff(u_symbol)
            for i, u_sym in enumerate(u_symbols):
                B.loc[k, i] = float(expression.coeff(u_sym))

        # solve linear sys dyn

        t, sol = run_linear_system_dynamics(A, B, u_func, z0, T)

        # plot

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("System Identification")

            st.metric("State Dimension", state_dimension)

            st.markdown("**State Expressions**")
            st.latex(sp.latex(expressions))

            st.markdown("**State Variables**")
            st.latex(sp.latex(variables_list))

            st.divider()

            st.subheader("State-Space Model")

            st.markdown("#### $\\dot{z} = Az + Bu$")

            colA, colB = st.columns(2)

            with colA:
                st.markdown("**Matrix A**")
                st.latex(sp.latex(sp.Matrix(A)))

                st.markdown("**State Vector $z$**")
                st.latex(sp.latex(variables_list))

            with colB:
                st.markdown("**Matrix B**")
                st.latex(sp.latex(sp.Matrix(B)))

                st.markdown("**Control Input $u$**")
                lines = u_func_str.strip().split("\n")
                vars = []
                for line in lines:
                    lhs = line.split("=")[0].strip()
                    if "u" in lhs:
                        vars.append(lhs)

                vars.extend(["1"] * (state_dimension - len(vars)))
                u_latex = sp.Matrix([sp.Symbol(v) for v in vars])
                st.latex(r"u = " + sp.latex(u_latex))

            # st.divider()

        with col2:
            st.subheader("Visualize")

            num_states = sol.shape[1]

            fig = go.Figure()

            for i in range(num_states):
                fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name=f"z{i}(t)"))

            frames = []
            for k in range(len(t)):
                frame_data = []
                for i in range(num_states):
                    frame_data.append(
                        go.Scatter(x=t[: k + 1], y=sol[: k + 1, i], mode="lines")
                    )
                frames.append(go.Frame(data=frame_data, name=str(k)))

            fig.frames = frames

            fig.update_layout(
                title="ODE Simulation",
                xaxis_title="Time",
                yaxis_title="States",
                updatemenus=[
                    {
                        "type": "buttons",
                        "buttons": [
                            {
                                "label": "▶️ Simulate",
                                "method": "animate",
                                "args": [
                                    None,
                                    {
                                        "frame": {"duration": 50, "redraw": True},
                                        "fromcurrent": True,
                                    },
                                ],
                            },
                            {
                                "label": "⏸ Pause",
                                "method": "animate",
                                "args": [
                                    [None],
                                    {"frame": {"duration": 0}, "mode": "immediate"},
                                ],
                            },
                        ],
                    }
                ],
            )

            # Show in Streamlit
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)
        st.error(f"❌ Error: {str(e)} ({exc_tb.tb_lineno})")
        st.stop()

st.divider()


######################################
