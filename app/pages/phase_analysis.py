import streamlit as st
import sympy as sp

from ode import (
    linearize_odes, 
    phase_portrait, 
    check_stability_at_point, 
    point_animation)


#######################################################


st.title("Phase Portait Analyzer")


def initialize_st_states(
    keys=[
        "variables",
        "variables_list",
        "jacobian",
        "expressions",
        "equilibrium_points",
        "jacobian_at_point",
        "linear_system",
        "phase_potrait_fig",
        "point_animation_fig",
        "stability",
    ]
):
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = None


initialize_st_states()
#######################################################


# num_states = st.number_input("Enter number of states", 1, 3, 2)
num_states = 2

user_input = st.text_area(
    "Enter ODEs (one per line)", "dx/dt = x - x*y\ndy/dt = y*(x - 1)"
)


if st.button("Linearize"):

    equations = user_input.strip().split("\n")

    if num_states != len(equations):
        st.error("Number of states must match the number of ODEs !")
    else:

        (
            expressions,
            variables,
            equilibrium_points,
            jacobian,
        ) = linearize_odes(equations)

        if len(variables) != num_states:
            st.error("Number of unique variables must match the state !")
        else:
            st.session_state.variables = variables
            st.session_state.variables_list = sorted(
                [v for v in st.session_state["variables"]], key=lambda x: x.name
            )
            st.session_state.jacobian = jacobian
            st.session_state.expressions = expressions
            st.session_state.equilibrium_points = equilibrium_points



variables = st.session_state.get("variables")
jacobian = st.session_state.get("jacobian")
eps = st.session_state.get("equilibrium_points")


col1, col2 = st.columns([1.3, 1])

with col1:
    st.markdown("### System Definition")

    if variables is not None:
        st.latex(sp.latex(variables))

    st.markdown("### Linearization")

    if jacobian is not None:
        st.latex(sp.latex(jacobian))

with col2:
    st.markdown("### Equilibrium Points")

    if eps:
        for i, point in enumerate(eps):
            with st.container(border=True):
                st.markdown(f"**Point {i + 1}**")
                st.latex(sp.latex(point))

st.divider()


j = st.session_state["jacobian"]
vl = st.session_state["variables_list"]

if j:
    st.subheader("System Behaviour Analysis")
    user_input = st.text_area(
        "Enter point (comma separated. Eg: x=1,y=1)", "x=0.5,y=0.5"
    )

    if st.button("Analyze"):
        point = {
            x.split("=")[0]: x.split("=")[1] for x in user_input.strip().split(",")
        }

        stability, jacobian_at_point, linear_system = check_stability_at_point(
            j, point, vl
        )

        initial_point = [ point[x.name] for x in st.session_state['variables_list'] ]

        variables_tuple = tuple(st.session_state["variables_list"])
        pa_fig = point_animation(st.session_state["expressions"], variables_tuple, scale=4, initial_point=initial_point)

        st.session_state["point_animation_fig"] = pa_fig
        st.session_state["jacobian_at_point"] = jacobian_at_point
        st.session_state["linear_system"] = linear_system
        st.session_state["stability"] = stability


fig = st.session_state.get("point_animation_fig")
stability = st.session_state.get("stability")
jacobian = st.session_state.get("jacobian_at_point")
linear = st.session_state.get("linear_system")

colA, colB = st.columns([3, 1])

with colA:
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No phase portrait available yet.")

with colB:
    st.subheader("System Info")

    if stability:
        st.markdown(
            f"### :orange-badge[:material/star: {stability}]"
        )

    if jacobian is not None:
        with st.expander("Jacobian at point", expanded=True):
            st.latex(sp.latex(jacobian))

    if linear is not None:
        with st.expander("Linearized system", expanded=True):
            st.latex(sp.latex(linear))


st.divider()


if st.session_state["variables_list"]:
    st.subheader("Phase Potrait Visualizer")
    scale = st.slider("Select scale", 0.5, 5.0, 2.0, 0.5)

    if st.button("Plot"):
        variables_tuple = tuple(st.session_state["variables_list"])

        fig = phase_portrait(
            st.session_state["expressions"], variables_tuple, scale
        )[0]
        st.session_state["phase_potrait_fig"] = fig

if st.session_state["phase_potrait_fig"]:
    # st.pyplot(st.session_state["fig"], width="content")
        st.plotly_chart(st.session_state["phase_potrait_fig"], use_container_width=True)
