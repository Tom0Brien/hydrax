import jax

import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.tasks.g1.g1_locomotion import G1Locomotion
from hydrax.algs.predictive_sampling import PredictiveSampling
from hydrax.risk import AverageCost
from hydrax.simulation.deterministic import run_interactive

def main():
    # Initialize task (now includes goal marker and soccer ball in XML)
    print("Initializing G1Locomotion task...")
    task = G1Locomotion()
    
    # Initialize controller
    print("Initializing PredictiveSampling controller...")
    ctrl = PredictiveSampling(
        task=task,
        num_samples=16,  # Reduced for faster testing
        noise_level=0.5,
        seed=0,
        plan_horizon=0.25,
        num_knots=5,
        iterations=1,
    )
    
    # Define the model used for simulation
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Use home keyframe
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
        # Initialize ctrl to default pose so last_act at t=0 is zeros
        mj_data.ctrl[:] = task._default_pose
    else:
        print("Warning: 'knees_bent' keyframe not found.")
    
    # Set soccer ball initial position (away from robot at origin)
    # Soccer ball has a freejoint: 7 DOFs (xyz position + quaternion)
    # It comes after robot root (7) + robot joints (29) in qpos
    soccer_ball_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "soccer_ball")
    if soccer_ball_id != -1:
        # Set ball at x=2.5, y=0.5, z=0.117 (radius above ground)
        mj_data.qpos[36:36+3] = [2.5, 0.5, 0.117]
        # Quaternion identity (no rotation)
        mj_data.qpos[36+3:36+7] = [1.0, 0.0, 0.0, 0.0]
        print(f"Soccer ball positioned at: x=2.5, y=0.5, z=0.117")
    
    # Set goal by moving mocap body (like pusht)
    # Goal: x=3.0, y=1.0, theta=0.0
    if len(mj_data.mocap_pos) > 0:
        mj_data.mocap_pos[0] = [3.0, 1.0, 0.05]  # x, y, z (on ground)
        # Quaternion for theta=0.0 (facing forward)
        mj_data.mocap_quat[0] = [1.0, 0.0, 0.0, 0.0]
        print(f"Goal set to: x=3.0, y=1.0, theta=0.0")

    # Run the interactive simulation
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=50,
        show_traces=False,
        record_video=True,
    )


if __name__ == "__main__":
    main()
