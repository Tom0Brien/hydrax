"""Interactive Cartpole swingup example with SPC residuals.

This example demonstrates the CartpoleSwingup task where a trained RL policy
handles the base control and SPC optimizes residuals for improved swingup.
"""

import argparse
import jax
import mujoco
import numpy as np
from mujoco import mjx

import sys
import os
sys.path.append(os.getcwd())

from hydrax.algs.cem import CEM
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.cartpole import CartpoleSwingup


def main() -> None:
    """Run CartpoleSwingup task: swing up and balance the pole."""
    parser = argparse.ArgumentParser(description="Interactive Cartpole swingup with SPC")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization"
    )
    parser.add_argument(
        "--no-policy",
        action="store_true",
        help="Run without RL policy (pure SPC control)"
    )
    args = parser.parse_args()
    
    # Initialize task
    print(f"Initializing CartpoleSwingup task...")
    task = CartpoleSwingup(use_rl_policy=not args.no_policy)
    
    # Initialize controller
    ctrl = CEM(
        task=task,
        num_samples=128,
        num_elites=8,
        sigma_start=0.4,
        sigma_min=0.01,
        explore_fraction=0.5,
        plan_horizon=1.0,  # 1s horizon
        spline_type="cubic",
        num_knots=6,
        iterations=3,
    )
    
    mj_model = task.mj_model
    rng = jax.random.PRNGKey(args.seed)
    
    # Reset environment
    mj_data, _ = task.reset(rng)
    
    print(f"Reset with seed: {args.seed}")
    
    # Print task info
    print(f"\nTask info:")
    print(f"  - Control dim (residuals): {task.nu}")
    print(f"  - Control frequency: {1/task.ctrl_dt:.0f} Hz")
    print(f"  - Residual bounds: [{task.u_min[0]:.2f}, {task.u_max[0]:.2f}]")
    print(f"  - Goal: Swing up and balance the pole")
    print(f"\nStarting interactive simulation...")
    print("  - Watch the cartpole swing up!")
    print("  - RL policy + SPC residuals optimize swing trajectory")

    # Run the interactive simulation
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=100,  # 100Hz control (matches training)
        show_traces=False,
        record_video=True,
    )


if __name__ == "__main__":
    main()
