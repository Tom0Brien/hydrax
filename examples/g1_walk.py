import jax

import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.tasks.g1.g1_locomotion import G1Locomotion
from hydrax.algs.predictive_sampling import PredictiveSampling
from hydrax.risk import AverageCost
from hydrax.simulation.deterministic import run_interactive

def main():
    # Initialize task
    print("Initializing G1Locomotion task...")
    task = G1Locomotion()
    
    # Initialize controller
    print("Initializing PredictiveSampling controller...")
    ctrl = PredictiveSampling(
        task=task,
        num_samples=1, # Reduced for CPU verification
        noise_level=0.2,
        num_knots=1,
        plan_horizon=0.5,
        iterations=1,
    )
    
    # Define the model used for simulation
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Use home keyframe
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    else:
        print("Warning: 'knees_bent' keyframe not found, using default pose.")
        # Fallback: set joints if possible, but careful with shape
        # mj_data.qpos[7:] = np.array(task._default_pose) 
        pass

    # Run the interactive simulation
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=50,
        show_traces=False,
    )


if __name__ == "__main__":
    main()
