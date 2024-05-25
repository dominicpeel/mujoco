import numpy as np
import mujoco
import mujoco.viewer
from robot_descriptions import go2_mj_description

# Load the model from the XML path
model = mujoco.MjModel.from_xml_path(go2_mj_description.PACKAGE_PATH + "/scene.xml")
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

motor_names = [
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
]


def control_leg_motors(model, data, t):
    # Frequency and amplitude for alternating leg movements
    freq = 1
    amp = 3

    for i, motor_name in enumerate(motor_names):
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_name)
        phase_offset = i * np.pi / 6.0
        data.ctrl[actuator_id] = amp * np.sin(2 * np.pi * freq * t + phase_offset)


with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = data.time
    while viewer.is_running():
        # Compute elapsed time
        t = data.time - start_time

        if t > 0.5:
            control_leg_motors(model, data, t)

        # Advance the simulation
        mujoco.mj_step(model, data)

        # Sync the viewer
        viewer.sync()
