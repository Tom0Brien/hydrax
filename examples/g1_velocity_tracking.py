import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.tasks.g1.g1_velocity_tracking import G1VelocityTracking
from hydrax.algs.cem import CEM

from hydrax.simulation.deterministic import run_interactive

def main():
    # Initialize task with a target velocity (e.g. 0.6 m/s forward, 0.2 rad/s turn)
    target_vel = jnp.array([0.6, 0.0, 0.2])
    print(f"Initializing G1VelocityTracking task with target velocity: {target_vel}")
    task = G1VelocityTracking(target_velocity=target_vel)
    
    # Initialize controller
    print("Initializing PredictiveSampling controller...")
    ctrl = CEM(
        task=task,
        num_samples=32,
        num_elites=8,
        sigma_start=0.5,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=4,
    )
    
    
    # Define the model used for simulation
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Use home keyframe if available
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
        # Initialize ctrl to default pose
        mj_data.ctrl[:] = task._default_pose
    else:
        print("Warning: 'knees_bent' keyframe not found.")
    
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
