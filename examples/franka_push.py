"""Interactive Franka push cube example with SPC residuals.

This example demonstrates the Franka push cube task where a trained RL policy
handles the base pushing behavior and SPC optimizes residuals for adaptation.
"""

import mujoco

from hydrax.algs.cem import CEM
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.franka import FrankaPushCube


def main() -> None:
    """Run Franka push cube task: robot pushes cube to target."""
    # Initialize task
    print("Initializing FrankaPushCube task...")
    task = FrankaPushCube()
    
    # Initialize controller
    ctrl = CEM(
        task=task,
        num_samples=1,
        num_elites=1,
        sigma_start=0.05,  # Small - residuals should be small
        sigma_min=0.01,
        explore_fraction=0.5,
        plan_horizon=1,
        spline_type="zero",
        num_knots=4,
    )
    
    # Define the model used for simulation
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    
    # Use home keyframe
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
        print(f"Reset to 'home' keyframe")
    else:
        print("Warning: 'home' keyframe not found.")
    
    # Set initial target position for the cube (via mocap body)
    if len(mj_data.mocap_pos) > 0:
        # Target: push cube to position in front of robot
        mj_data.mocap_pos[0] = [0.5, 0.1, 0.02]  # x, y, z (slightly above ground)
        mj_data.mocap_quat[0] = [1.0, 0.0, 0.0, 0.0]  # No rotation
        print(f"Target position set to: {mj_data.mocap_pos[0]}")
    
    # Run forward to compute derived quantities
    mujoco.mj_forward(mj_model, mj_data)
    
    # Print initial state
    print(f"\nTask info:")
    print(f"  - Control dim (residuals): {task.nu}")
    print(f"  - Control frequency: {1/task.ctrl_dt:.0f} Hz")
    print(f"  - Residual bounds: [{task.u_min[0]:.2f}, {task.u_max[0]:.2f}]")
    print(f"\nStarting interactive simulation...")
    print("  - Drag the green target mocap to move the goal")
    print("  - RL policy + SPC residuals will push the cube")

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
