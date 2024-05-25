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


def stand(model, data, t):
    # Define the target positions for the standing pose over time
    target_positions = {
        "FR_hip": -np.pi / 4,
        "FR_thigh": -np.pi / 4,
        "FR_calf": -np.pi / 4,
        "FL_hip": -np.pi / 4,
        "FL_thigh": -np.pi / 4,
        "FL_calf": -np.pi / 4,
        "RR_hip": -np.pi / 4,
        "RR_thigh": -np.pi / 4,
        "RR_calf": -np.pi / 4,
        "RL_hip": -np.pi / 4,
        "RL_thigh": -np.pi / 4,
        "RL_calf": -np.pi / 4,
    }

    # Define the duration over which the transition should occur
    transition_duration = 5.0  # 5 seconds

    for motor_name, target_position in target_positions.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, motor_name)
        current_position = data.qpos[joint_id]

        # Calculate the new position based on the elapsed time
        new_position = current_position + (target_position - current_position) * min(
            t / transition_duration, 1.0
        )
        data.qpos[joint_id] = new_position


def score(data):
    # Calculate the distance in the +x direction
    distance_x = data.qpos[0]  # Assuming the first element in qpos is the x position
    return distance_x


wait_time = 1
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = data.time
    while viewer.is_running():
        # Compute elapsed time
        t = data.time - start_time - wait_time

        if t > 0:
            stand(model, data, t)

        # Advance the simulation
        mujoco.mj_step(model, data)

        # Print the score
        print("Score (distance in +x):", score(data))

        # Sync the viewer
        viewer.sync()
