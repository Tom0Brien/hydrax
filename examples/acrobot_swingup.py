"""Interactive Acrobot swingup example with SPC residuals.

This example demonstrates the AcrobotSwingup task where a trained RL policy
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
from hydrax.tasks.acrobot import AcrobotSwingup


def main() -> None:
    """Run AcrobotSwingup task: swing the tip to reach the target."""
    parser = argparse.ArgumentParser(description="Interactive Acrobot swingup with SPC")
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
    print(f"Initializing AcrobotSwingup task...")
    task = AcrobotSwingup(use_rl_policy=not args.no_policy)
    
    # Initialize controller
    ctrl = CEM(
        task=task,
        num_samples=128,
        num_elites=8,
        sigma_start=0.4,
        sigma_min=0.01,
        explore_fraction=0.5,
        plan_horizon=1,  # 500ms horizon
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
    print(f"  - Target: swing tip to reach the ball above")
    print(f"\nStarting interactive simulation...")
    print("  - Watch the acrobot swing up!")
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
