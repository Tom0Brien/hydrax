import mujoco

from evosax.algorithms.distribution_based import CMA_ES

from hydrax.algs import CEM, Evosax
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.g1.g1_soccer import G1Soccer


def main() -> None:
    """Run G1 soccer task: robot pushes ball to goal."""
    # Initialize task
    print("Initializing G1Soccer task...")
    task = G1Soccer()
    
    # Initialize controller
    # print("Initializing controller...")
    ctrl = CEM(
        task=task,
        num_samples=32,
        num_elites=8,
        sigma_start=0.5,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=1,
        spline_type="zero",
        num_knots=4,
    )
    # print("Initializing Evosax controller...")
    # ctrl = Evosax(
    #     task,
    #     CMA_ES,
    #     num_samples=32,
    #     plan_horizon=0.75,
    #     spline_type="zero",
    #     num_knots=6,
    # )
    
    # Define the model used for simulation
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Use knees_bent keyframe
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
        # Initialize ctrl to default pose
        mj_data.ctrl[:] = task._default_pose
    else:
        print("Warning: 'knees_bent' keyframe not found.")
    
    # Set soccer ball initial position (between robot and goal)
    # Robot at origin, goal at (3, 1), ball at (1.5, 0.5)
    ball_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, "soccer_ball"
    )
    if ball_id != -1:
        # Ball at x=1.5, y=0.5, z=0.117 (radius above ground)
        mj_data.qpos[36:36+3] = [2, 0.5, 0.117]
        # Quaternion identity (no rotation)
        mj_data.qpos[36+3:36+7] = [1.0, 0.0, 0.0, 0.0]
        print(f"Soccer ball positioned at: {mj_data.qpos[36:36+3]}")
    
    # Set goal position via mocap body
    # Goal: push ball to (3, 1)
    if len(mj_data.mocap_pos) > 0:
        mj_data.mocap_pos[0] = [4.75, 0.0, 0.05]  # x, y, z (on ground)
        # Quaternion for theta=0.0 (facing forward)
        mj_data.mocap_quat[0] = [1.0, 0.0, 0.0, 0.0]
        print(f"Goal set to: {mj_data.mocap_pos[0]}")
        print(
            "Task: Robot should push ball from "
            f"({mj_data.qpos[36:36+3]} to goal at {mj_data.mocap_pos[0]}"
        )

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

