from scipy.integrate import odeint
import numpy as np
import sympy as sp
from pprint import pprint
from typing import Optional, List
import matplotlib.pyplot as plt

# import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go


###################################


# linear ode


def linear_system_dynamics(z, t, A: np.ndarray, B: np.ndarray, u_func):
    z = np.array(z)
    u = np.array(u_func(t))
    dzdt = A @ z + B @ u
    return dzdt


def run_linear_system_dynamics(A, B, u_func, z0, T=40):
    t = np.linspace(0, T, 1000)

    sol = odeint(linear_system_dynamics, z0, t, args=(A, B, u_func))

    return t, sol


def plot_ode():
    pass


###################################


# linearize ODEs & phase analysis


def extract_expressions(equations: List[str]):

    variables = set()
    expressions = []

    for eq in equations:
        eq = eq.replace("^", "**")
        lhs, r = eq.split("=")
        r = r.strip()

        # add logic for things like 5*dx/dt = wtvr

        var = lhs.replace("d", "").replace("t", "").replace("/", "").strip()
        var_symbol = sp.Symbol(var)

        expr = sp.sympify(r)
        expressions.append(expr)
        variables.update({var_symbol})

    return expressions, variables


def linearize_odes(equations: str | List[str]):

    # Based on  Week 2_Phase_plane_Analysis.pdf

    if isinstance(equations, str):
        equations = [equations]

    expressions, variables = extract_expressions(equations)

    # print("Expressions: ", expressions)
    # print(type(expressions[0]))

    # find equilibrium points

    equilibrium_points = sp.solve(expressions, variables)
    # print("Equilibrium points: ", equilibrium_points)
    # print(type(list(equilibrium[0].keys())[0]))

    t = sp.Symbol("t")

    # find jacobian

    variables_list = sorted([v for v in variables if v != t], key=lambda x: x.name)
    jacobian = sp.Matrix(expressions).jacobian(variables_list)

    # print("Variables Set: ", variables)
    # print("Variables List: ", variables_list)

    return (
        expressions,
        variables,
        equilibrium_points,
        jacobian,
    )


def check_stability_at_point(jacobian, point, variables_list):

    jacobian_at_point = jacobian.subs(point)

    eigenvals = jacobian_at_point.eigenvals()
    stability = "Unknown"
    real_parts = [sp.re(ev) for ev in eigenvals.keys()]

    if all(r < 0 for r in real_parts):
        stability = "Asymptotically stable"
    elif all(r > 0 for r in real_parts):
        stability = "Unstable (repelling)"
    elif any(r > 0 for r in real_parts) and any(r < 0 for r in real_parts):
        stability = "UNSTABLE (Saddle point)"
    elif all(r <= 0 for r in real_parts):
        stability = "STABLE"
    else:
        stability = "Inconclusive"

    x0 = sp.Matrix([point[v.name] for v in variables_list])
    x = sp.Matrix(variables_list)
    linear_system = jacobian_at_point * (x - x0)

    return stability, jacobian_at_point, linear_system


def phase_portrait(expressions, variables, scale=3, add_quiver=True):

    x1_range = np.linspace(-1 * scale, scale, 20)
    x2_range = np.linspace(-1 * scale, scale, 20)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    f = sp.lambdify(variables, expressions[0], "numpy")
    g = sp.lambdify(variables, expressions[1], "numpy")

    DX1 = f(X1, X2)
    DX2 = g(X1, X2)

    fig = ff.create_streamline(
        x1_range,
        x2_range,
        DX1,
        DX2,
        line_color="RoyalBlue",
        density=1.2,
        arrow_scale=0.1,
        name="streamline",
    )

    if add_quiver:
        fig.add_traces(
            data=ff.create_quiver(
                X1,
                X2,
                DX1,
                DX2,
                line_color="yellow",
                opacity=0.3,
                arrow_scale=0.1,
                scale=0.05,
                name="quiver",
            ).data,
        )

    fig.update_layout(
        height=600,
        xaxis_title=variables[0].name,
        yaxis_title=variables[1].name,
    )

    return (fig, f, g)


def point_animation(expressions, variables, scale=2, initial_point=[1.5, 1.5], T=10):

    z0 = initial_point

    fig, f, g = phase_portrait(expressions, variables, scale, add_quiver=False)

    dot_trace_index = len(fig.data)

    def system(z, t):
        x1, x2 = z
        return [f(x1, x2), g(x1, x2)]

    t = np.linspace(0, T, 200)
    sol = odeint(system, z0, t)

    x = sol[:, 0]
    y = sol[:, 1]

    fig.add_trace(
        go.Scatter(
            x=[x[0]],
            y=[y[0]],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="state",
        )
    )

    frames = []
    for i in range(len(t)):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=[x[i]],
                        y=[y[i]],
                        mode="markers",
                        marker=dict(size=10, color="red"),
                    )
                ],
                traces=[dot_trace_index],
            )
        )

    fig.frames = frames

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 40, "redraw": False},
                                "fromcurrent": True,
                            },
                        ],
                    )
                ],
            )
        ],
    )

    return fig


if __name__ == "__main__":

    # equations = ["dx/dt = x*(1-x)"]
    equations = ["dx/dt = x - x*y", "dy/dt = y*(x - 1)"]
    # equations = ["dx/dt = x", "dy/dt = -y + x*z", "dz/dt = -z"]

    # result = linearize_odes(equations)

    expressions, variables = extract_expressions(equations)

    variables_tuple = tuple(sorted([v for v in variables], key=lambda x: x.name))
    fig, ax = plot_phase_portrait(expressions, variables_tuple)
    plt.show()
    # Formatting
