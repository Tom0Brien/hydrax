"""Interactive demo of the constrained Franka push task with obstacle avoidance.

This example demonstrates the benefit of combining:
1. Pre-trained RL policy (learned pushing behavior)
2. CCEM (Constrained CEM) for online obstacle avoidance

The task is to push a cube to a target pose while avoiding a cylinder
obstacle in the workspace. The RL policy was trained without this
obstacle, so CCEM is needed to avoid collisions.

Controls:
    - Double click on the target, then drag with [ctrl + right-click]
    - Observe: RL alone may push box into the obstacle
    - CCEM plans around the obstacle while still tracking the target

Usage:
    python examples/franka_push_constrained.py [--rl-only] [--ccem-only]
"""

import argparse
import mujoco

from hydrax.algs import CCEM, CEM, ALCEM
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.franka.franka_push_constrained import FrankaPushConstrained
from hydrax.tasks.franka import apply_perturbation, PerturbationConfig


def main():
    parser = argparse.ArgumentParser(
        description="Constrained Franka Push: RL + CCEM obstacle avoidance demo"
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
        "--safety-margin", type=float, default=0.02,
        help="Safety margin around obstacle in meters (default: 0.02)"
    )
    args = parser.parse_args()

    # Determine mode
    use_rl = not args.ccem_only
    use_ccem = not args.rl_only
    
    mode_str = "RL + CCEM" if (use_rl and use_ccem) else ("RL only" if use_rl else "CCEM only")
    print(f"\n{'='*60}")
    print(f"Constrained Franka Push Demo: {mode_str}")
    print(f"{'='*60}")
    print(f"Obstacle: Red cylinder at (0.55, 0.0) with radius 0.04m")
    print(f"Safety margin: {args.safety_margin}m")
    print(f"\nThe RL policy was trained without this obstacle.")
    print(f"CCEM will plan around it to avoid collisions.")
    print(f"{'='*60}\n")

    # Create the constrained task with obstacle
    task = FrankaPushConstrained(
        geometry="square",  # Rectangular cube
        use_rl_policy=use_rl,
        safety_margin=args.safety_margin,
    )
    
    # Apply slippery perturbation (friction 0.3x)
    perturbation = PerturbationConfig(
        name="slippery",
        mass_scale=1.0,
        friction_scale=1.0,
        geometry="square",
    )
    # apply_perturbation(task.mj_model, perturbation, task=task)
    print(f"Applied perturbation: {perturbation.name} (friction={perturbation.friction_scale}x)")

    # Create the CCEM controller if using online planning
    if use_ccem:
        ctrl = CCEM(
            task,
            num_samples=256,
            num_elites=16,
            sigma_start=0.3,
            sigma_min=0.05,
            explore_fraction=0.5,
            plan_horizon=1.0,
            spline_type="zero",
            num_knots=6,
            num_randomizations=1,
        )
        # ctrl = ALCEM(
        #     task=task,
        #     num_samples=128,
        #     num_elites=16,
        #     sigma_start=0.1,
        #     sigma_min=0.01,
        #     explore_fraction=0.5,
        #     num_knots=6,
        #     spline_type="zero",
        #     plan_horizon=0.5,
        #     # Augmented Lagrangian parameters
        #     lambda_init=0.0,      # Initial Lagrange multiplier
        #     rho_init=1.0,         # Initial penalty coefficient
        #     rho_max=1000.0,       # Max penalty coefficient
        #     adapt_rho=True,       # Automatically increase rho if violations persist
        # )
    else:
        # RL only - no CCEM controller
        ctrl = None

    # Get the MuJoCo model and data
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    
    # Reset to home keyframe (proper initial joint configuration)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, mj_model.keyframe("home").id)

    # Move box to the right of obstacle (set position via qpos, not xpos)
    # Box freejoint qpos: indices 13-15 = position (x,y,z), 16-19 = quaternion (w,x,y,z)
    box_pos = [0.40, -0.3, 0.03]  # Left of obstacle, will push right
    mj_data.qpos[13:16] = box_pos
    
    # Set obstacle position (static body, modify model.body_pos)
    # Obstacle is at y=0.0 in XML, between box (y=-0.15) and target (y=0.15)
    obstacle_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
    obstacle_pos = [0.55, 0.0, 0.08]  # Same as XML: x=0.55, y=0.0, z=0.08 (half-height)
    mj_model.body_pos[obstacle_body_id] = obstacle_pos
    
    # Run forward to compute derived quantities (xpos, xquat, etc.)
    mujoco.mj_forward(mj_model, mj_data)

    # Set target position on the other side of the obstacle
    # Box starts at y=-0.15, obstacle at y=0.0, target at y=0.15
    # This forces the robot to push around the cylinder
    target_pos = [0.60, 0.3, 0.03]  # Past the obstacle in y direction
    mj_data.mocap_pos[0] = target_pos
    
    # Set target orientation to match box's initial orientation
    box_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_quat = mj_data.xquat[box_body_id].copy()
    mj_data.mocap_quat[0] = box_quat

    # Run interactive simulation
    print("Starting interactive simulation...")
    print(f"Target set at {target_pos} (requires avoiding the obstacle)")
    print("Watch: CCEM should plan a path around the red cylinder!\n")
    
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
