import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.tasks.g1.g1_velocity_tracking import G1VelocityTracking
from hydrax.tasks.g1.g1_velocity_tracking_augmented import G1VelocityTrackingAugmented
from hydrax.algs.icem import iCEM

from hydrax.simulation.deterministic import run_interactive

def main():
    # Initialize task with a target velocity
    target_vel = jnp.array([0.5, 0.0, 0.0])
    print(f"Initializing G1VelocityTracking task with target velocity: {target_vel}")
    task = G1VelocityTracking(target_velocity=target_vel)
    task_aug = G1VelocityTrackingAugmented(target_velocity=target_vel)
    
    # Initialize iCEM controller
    print("Initializing iCEM controller...")
    ctrl = iCEM(
        task=task,
        num_samples=32,
        num_elites=8,
        sigma_start=0.5,
        sigma_min=0.05,
        alpha=0.1,              # Momentum smoothing for stable updates
        noise_beta=2.0,         # Colored noise for smooth locomotion trajectories
        fraction_elites_reused=0.3,  # Reuse 30% of elites
        shift_elites=True,      # Warm-start with shifted trajectories
        use_best_action=True,   # Execute best action instead of mean
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=4,
        iterations=1,
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
