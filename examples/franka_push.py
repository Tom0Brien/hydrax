"""Interactive Franka push example with SPC residuals.

This example demonstrates the Franka push task where a trained RL policy
handles the base pushing behavior and SPC optimizes residuals for adaptation.

Supports different object geometries: cube (default), square, tblock.
"""

import argparse
import jax
import mujoco
import numpy as np
from mujoco import mjx

from hydrax.algs.cem import CEM
from hydrax.algs.ccem import CCEM
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.franka import FrankaPushGeometry


def main() -> None:
    """Run Franka push task: robot pushes object to target."""
    parser = argparse.ArgumentParser(description="Interactive Franka push with SPC")
    parser.add_argument(
        "--geometry", 
        type=str, 
        default="cube",
        choices=["cube", "square", "tblock"],
        help="Object geometry to use (default: cube)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization"
    )
    args = parser.parse_args()
    
    # Initialize task with specified geometry
    print(f"Initializing FrankaPush task with geometry: {args.geometry}")
    task = FrankaPushGeometry(geometry=args.geometry, use_rl_policy=True)
    
    # Initialize controller
    ctrl = CCEM(
        task=task,
        num_samples=128,
        num_elites=8,
        sigma_start=0.3,  # Small - residuals should be small
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=25,
    )
    
    mj_model = task.mj_model
    rng = jax.random.PRNGKey(args.seed)
    
    # Use unified reset interface on task
    # mj_data, mjx_data = task.reset(rng)
    
    # Set target outside workspace
    mj_data = mujoco.MjData(mj_model)
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    mj_data.qpos[13] = 0.65
    mj_data.qpos[14] = 0.0
    mj_data.qpos[16:20] = np.array([1.0, 0.0, 0.0, 0.0])
    rand_y = np.random.uniform(-0.1, 0.1)
    mj_data.mocap_pos[0] = np.array([0.775, rand_y, 0.0306525])
    mj_data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0])

    
    print(f"Reset with seed: {args.seed}")
    print(f"Object position: {mj_data.qpos[13:16]}")
    print(f"Target position: {mj_data.mocap_pos[0]}")

    # Print initial state
    print(f"\nTask info:")
    print(f"  - Geometry: {args.geometry}")
    print(f"  - Control dim (residuals): {task.nu}")
    print(f"  - Control frequency: {1/task.ctrl_dt:.0f} Hz")
    print(f"  - Residual bounds: [{task.u_min[0]:.2f}, {task.u_max[0]:.2f}]")
    print(f"\nStarting interactive simulation...")
    print("  - Drag the target mocap to move the goal")
    print("  - RL policy + SPC residuals will push the object")

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

