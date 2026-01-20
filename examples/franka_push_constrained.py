"""Interactive demo of the constrained Franka push task.

This example demonstrates the benefit of combining:
1. Pre-trained RL policy (learned pushing behavior)
2. CCEM (Constrained CEM) for online constraint satisfaction

The task is to push a cube to a target pose while keeping it within
a safe zone (green boundary). The RL policy was trained without this
constraint, so it may push the box outside. CCEM corrects for this.

Controls:
    - Double click on the target, then drag with [ctrl + right-click]
    - Observe: RL alone may push box outside green zone
    - CCEM keeps the box inside while still tracking the target

Usage:
    python examples/franka_push_constrained.py [--rl-only] [--ccem-only]
"""

import argparse
import mujoco

from hydrax.algs import CCEM, CEM
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.franka.franka_push_constrained import FrankaPushConstrained


def main():
    parser = argparse.ArgumentParser(
        description="Constrained Franka Push: RL + CCEM demo"
    )
    parser.add_argument(
        "--rl-only", action="store_true",
        help="Use RL policy alone (no CCEM planning)"
    )
    parser.add_argument(
        "--ccem-only", action="store_true",
        help="Use CCEM alone (no RL policy)"
    )
    parser.add_argument(
        "--safe-zone-size", type=float, default=0.2,
        help="Half-size of safe zone in meters (default: 0.15)"
    )
    args = parser.parse_args()

    # Determine mode
    use_rl = not args.ccem_only
    use_ccem = not args.rl_only
    
    mode_str = "RL + CCEM" if (use_rl and use_ccem) else ("RL only" if use_rl else "CCEM only")
    print(f"\n{'='*60}")
    print(f"Constrained Franka Push Demo: {mode_str}")
    print(f"{'='*60}")
    print(f"Safe zone: ±{args.safe_zone_size}m from center (0.525, 0.0)")
    print(f"RL target region: [0.4, 0.65] x [-0.2, 0.2] (larger than safe zone)")
    print(f"\nThe RL policy may try to push outside the green zone.")
    print(f"CCEM will correct this to keep the box inside.")
    print(f"{'='*60}\n")

    # Create the constrained task
    task = FrankaPushConstrained(
        geometry="square",  # Square cube (0.05073 x 0.05073 x 0.05073)
        use_rl_policy=use_rl,
        safe_zone_center=(0.525, 0.0),
        safe_zone_half_size=args.safe_zone_size,
    )

    # Create the CCEM controller if using online planning
    if use_ccem:
        ctrl = CCEM(
            task,
            num_samples=128,
            num_elites=16,
            sigma_start=0.1,  # Small - residuals should be small
            sigma_min=0.05,
            explore_fraction=0.5,
            plan_horizon=0.5,
            spline_type="zero",
            num_knots=6,
        )
    else:
        # RL only - no CCEM controller
        ctrl = None

    # Get the MuJoCo model and data
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    
    # Reset to home keyframe (proper initial joint configuration)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, mj_model.keyframe("home").id)
    mujoco.mj_forward(mj_model, mj_data)  # Compute xquat for box
    
    # Set target position just outside the safe zone (to the left)
    # Safe zone: y in [-0.2, 0.2], so y=-0.25 is outside
    target_pos = [0.525, -0.25, 0.03]  # x=center, y=outside left, z=table height
    mj_data.mocap_pos[0] = target_pos
    
    # Set target orientation to match box's initial orientation
    box_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_quat = mj_data.xquat[box_body_id].copy()
    mj_data.mocap_quat[0] = box_quat

    # Run interactive simulation
    print("Starting interactive simulation...")
    print(f"Target set at y={target_pos[1]}m (outside safe zone boundary at y=-0.15)")
    print("Watch: CCEM should stop the box at the boundary!\\n")
    
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
