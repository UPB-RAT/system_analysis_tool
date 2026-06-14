import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from drone_sim import Quadcopter, Control, Trajectory, Wind, makeMixerFM

st.set_page_config(page_title="3D Drone Simulator", layout="wide")

st.title("🚁 3D Drone Simulator")
st.markdown("""
This tool simulates a 3D quadcopter drone following a trajectory. 
Adjust parameters in the sidebar to see how they affect the drone's performance and stability.
""")

# --- Sidebar Configuration ---
st.sidebar.header("Drone Parameters")
mB = st.sidebar.slider("Mass (kg)", 0.5, 3.0, 1.2, 0.1)
g = st.sidebar.slider("Gravity (m/s²)", 8.0, 12.0, 9.81, 0.01)
Cd = st.sidebar.slider("Drag Coefficient (Cd)", 0.0, 1.0, 0.1, 0.05)

st.sidebar.header("Simulation Settings")
Tf = st.sidebar.slider("Simulation Duration (s)", 5, 30, 15, 1)
Ts = 0.01 # 10ms timestep for the main loop

st.sidebar.header("Trajectory")
traj_type_options = {
    "Static Hover": [0, 0, 1],
    "Simple Interpolation": [2, 2, 1],
    "Minimum Velocity": [3, 3, 1],
    "Minimum Acceleration": [4, 3, 1],
    "Minimum Jerk": [5, 3, 1],
    "Minimum Snap": [6, 3, 1]
}
traj_name = st.sidebar.selectbox("Trajectory Type", list(traj_type_options.keys()), index=4)
traj_select = traj_type_options[traj_name]

# --- Initialize Parameters ---
params = {
    "mB": mB,
    "g": g,
    "dxm": 0.16,
    "dym": 0.16,
    "dzm": 0.05,
    "IB": np.array([[0.0123, 0, 0], [0, 0.0123, 0], [0, 0, 0.0224]]),
    "IRzz": 2.7e-5,
    "Cd": Cd,
    "kTh": 1.076e-5,
    "kTo": 1.632e-7,
    "minWmotor": 75,
    "maxWmotor": 925,
    "tau": 0.015,
    "kp": 1.0,
    "damp": 1.0,
    "useIntegral": False,
    "minThr": 0.1 * 4,
    "maxThr": 9.18 * 4,
}

# Calculate Hover Speed
params["w_hover"] = np.sqrt((mB * g / 4.0) / params["kTh"])
params["mixerFM"] = makeMixerFM(params, orient="NED")
params["mixerFMinv"] = np.linalg.inv(params["mixerFM"])

# --- Run Simulation ---
if st.button("🚀 Run Simulation"):
    with st.spinner("Simulating..."):
        # Initialize
        quad = Quadcopter(params, Ti=0, orient="NED")
        traj = Trajectory(quad, "xyz_pos", traj_select)
        ctrl = Control(quad, params, traj.yawType, orient="NED")
        wind = Wind('None')

        numSteps = int(Tf / Ts)
        t_all = np.zeros(numSteps)
        pos_all = np.zeros((numSteps, 3))
        vel_all = np.zeros((numSteps, 3))
        euler_all = np.zeros((numSteps, 3))
        w_cmd_all = np.zeros((numSteps, 4))
        target_pos_all = np.zeros((numSteps, 3))

        t = 0.0
        for i in range(numSteps):
            sDes = traj.desiredState(t, Ts, quad)
            ctrl.controller(traj, quad, sDes, Ts)
            quad.update(t, Ts, ctrl.w_cmd, wind)
            
            t_all[i] = t
            pos_all[i] = quad.pos.copy()
            vel_all[i] = quad.vel.copy()
            euler_all[i] = quad.euler.copy() * (180.0 / np.pi)
            w_cmd_all[i] = ctrl.w_cmd.copy()
            target_pos_all[i] = sDes[0:3].copy()
            t += Ts

        st.success(f"Simulation completed! Flight reviewed in animation.")

        st.divider()
        st.subheader("📊 Simulation Results")

        # Layout: 3D Plot on the left, telemetry on the right
        col_main, col_data = st.columns([2, 1])

        with col_main:
            st.markdown("### 3D Trajectory Fly-through")
            # Create high-performance Plotly animation
            step_anim = max(2, int(0.05 / Ts))
            indices = np.arange(0, numSteps, step_anim)
            
            fig3d = go.Figure(
                data=[
                    go.Scatter3d(x=target_pos_all[:,0], y=target_pos_all[:,1], z=-target_pos_all[:,2],
                               mode='lines', line=dict(color='red', width=2, dash='dot'), name='Planned Trajectory'),
                    go.Scatter3d(x=traj.wps[:,0], y=traj.wps[:,1], z=-traj.wps[:,2],
                               mode='markers', marker=dict(size=4, color='darkred'), name='Waypoints'),
                    go.Scatter3d(x=[pos_all[0,0]], y=[pos_all[0,1]], z=[-pos_all[0,2]],
                               mode='lines', line=dict(color='blue', width=4), name='Actual Path'),
                    go.Scatter3d(x=[pos_all[0,0]], y=[pos_all[0,1]], z=[-pos_all[0,2]],
                               mode='markers', marker=dict(size=10, color='darkblue', symbol='diamond'), name='Drone')
                ],
                layout=go.Layout(
                    scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Alt (m)', aspectmode='cube'),
                    height=700, margin=dict(l=0, r=0, b=0, t=30),
                    updatemenus=[dict(
                        type="buttons",
                        font=dict(color="#2c3e50"), # Dark color for visibility
                        bgcolor="rgba(255, 255, 255, 0.8)", # Light background for contrast
                        buttons=[
                            dict(label="▶️ Play Flight", method="animate", args=[None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}]),
                            dict(label="⏸️ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
                        ]
                    )]
                ),
                frames=[go.Frame(data=[
                    go.Scatter3d(x=pos_all[:k+1,0], y=pos_all[:k+1,1], z=-pos_all[:k+1,2]),
                    go.Scatter3d(x=[pos_all[k,0]], y=[pos_all[k,1]], z=[-pos_all[k,2]])
                ], traces=[2, 3], name=f'f{k}') for k in indices]
            )
            st.plotly_chart(fig3d, use_container_width=True)

        with col_data:
            st.markdown("### Telemetry")
            # Euler Angles
            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(x=t_all, y=euler_all[:, 0], name='Roll'))
            fig_e.add_trace(go.Scatter(x=t_all, y=euler_all[:, 1], name='Pitch'))
            fig_e.add_trace(go.Scatter(x=t_all, y=euler_all[:, 2], name='Yaw'))
            fig_e.update_layout(title="Orientation (deg)", height=320, margin=dict(l=10, r=10, b=0, t=40))
            st.plotly_chart(fig_e, use_container_width=True)

            # Motor Commands
            fig_m = go.Figure()
            for i in range(4):
                fig_m.add_trace(go.Scatter(x=t_all, y=w_cmd_all[:, i], name=f'M{i+1}'))
            fig_m.update_layout(title="Motor Radians/s", height=320, margin=dict(l=10, r=10, b=0, t=40))
            st.plotly_chart(fig_m, use_container_width=True)

        # Bottom section: Position Tracking
        st.markdown("### Position Tracking")
        fig_pos = make_subplots(rows=1, cols=3, subplot_titles=("X (m)", "Y (m)", "Altitude (m)"))
        fig_pos.add_trace(go.Scatter(x=t_all, y=pos_all[:, 0], name='Actual'), row=1, col=1)
        fig_pos.add_trace(go.Scatter(x=t_all, y=target_pos_all[:, 0], name='Ref', line=dict(dash='dash')), row=1, col=1)
        fig_pos.add_trace(go.Scatter(x=t_all, y=pos_all[:, 1], name='Actual'), row=1, col=2)
        fig_pos.add_trace(go.Scatter(x=t_all, y=target_pos_all[:, 1], name='Ref', line=dict(dash='dash')), row=1, col=2)
        fig_pos.add_trace(go.Scatter(x=t_all, y=-pos_all[:, 2], name='Actual'), row=1, col=3)
        fig_pos.add_trace(go.Scatter(x=t_all, y=-target_pos_all[:, 2], name='Ref', line=dict(dash='dash')), row=1, col=3)
        fig_pos.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig_pos, use_container_width=True)

else:
    st.info("Adjust parameters in the sidebar and click **Run Simulation** to begin.")
