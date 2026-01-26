"""Interactive Cheetah run example with SPC residuals.

This example demonstrates the CheetahRun task where a trained RL policy
handles the base locomotion and SPC optimizes residuals for improved running.
"""

import argparse
import jax
import mujoco
import numpy as np
from mujoco import mjx

from hydrax.algs.cem import CEM
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.cheetah import CheetahRun


def main() -> None:
    """Run CheetahRun task: cheetah runs as fast as possible."""
    parser = argparse.ArgumentParser(description="Interactive Cheetah run with SPC")
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
    print(f"Initializing CheetahRun task...")
    task = CheetahRun(use_rl_policy=not args.no_policy)
    
    # Initialize controller
    ctrl = CEM(
        task=task,
        num_samples=64,
        num_elites=8,
        sigma_start=0.2,  # Slightly larger for locomotion
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=0.3,  # 300ms horizon
        spline_type="zero",
        num_knots=10,
    )
    
    mj_model = task.mj_model
    rng = jax.random.PRNGKey(args.seed)
    
    # Reset environment
    mj_data, _ = task.reset(rng)
    
    print(f"Reset with seed: {args.seed}")
    print(f"Initial position: x={mj_data.qpos[0]:.3f}")
    
    # Print task info
    print(f"\nTask info:")
    print(f"  - Control dim (residuals): {task.nu}")
    print(f"  - Control frequency: {1/task.ctrl_dt:.0f} Hz")
    print(f"  - Residual bounds: [{task.u_min[0]:.2f}, {task.u_max[0]:.2f}]")
    print(f"  - Target speed: {10.0} m/s")
    print(f"\nStarting interactive simulation...")
    print("  - Watch the cheetah run!")
    print("  - RL policy + SPC residuals optimize running speed")

    # Run the interactive simulation
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=100,  # 100Hz control
        show_traces=False,
        record_video=True,
    )


if __name__ == "__main__":
    main()
