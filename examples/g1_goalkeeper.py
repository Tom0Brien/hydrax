import mujoco
import numpy as np

from hydrax.algs import CEM, MPPI
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.g1.g1_goalkeeper import G1Goalkeeper


def main() -> None:
    """Run G1 goalkeeper task: robot intercepts ball moving toward goal."""
    # Initialize task
    print("Initializing G1Goalkeeper task...")
    task = G1Goalkeeper()
    
    # Initialize controller
    print("Initializing controller...")
    ctrl = MPPI(
        task=task,
        num_samples=32,
        noise_level=1,
        temperature=0.01,
        plan_horizon=1,
        num_knots=5,
        iterations=2,
    )
    
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
    
    # Position robot in front of goal (goalkeeper position)
    # Red goal is at x=-4.5, position robot at x=-4.5
    mj_data.qpos[0] = -4.5  # x position
    mj_data.qpos[1] = 0.0  # y position (center)
    print(f"Robot positioned at goalkeeper position: {mj_data.qpos[:3]}")
    
    # Randomize soccer ball starting position and velocity
    # Ball starts at random position, moving toward goal
    rng = np.random.default_rng()
    
    # Ball position: somewhere in positive x region, with some randomness
    ball_x = rng.uniform(-1.5, -0.5)  # Random x (closer to robot)
    ball_y = rng.uniform(-2.0, 2.0)  # Random y across width
    ball_z = 0.117  # Ball radius (on ground)
    
    # Ball velocity: directed toward red goal with randomness
    # Red goal is at x=-4.5, y=0
    goal_pos = np.array([-4.5, 0.0])
    ball_pos_xy = np.array([ball_x, ball_y])
    
    # Direction toward goal
    to_goal = goal_pos - ball_pos_xy
    to_goal_norm = to_goal / (np.linalg.norm(to_goal) + 1e-6)
    
    # Random speed (2-6 m/s) and add some lateral variation
    speed = rng.uniform(8.5, 8.5)
    lateral_angle = rng.uniform(-0.3, 0.3)  # ±17 degrees
    
    # Rotate direction by lateral angle
    cos_a = np.cos(lateral_angle)
    sin_a = np.sin(lateral_angle)
    direction = np.array([
        cos_a * to_goal_norm[0] - sin_a * to_goal_norm[1],
        sin_a * to_goal_norm[0] + cos_a * to_goal_norm[1]
    ])
    
    ball_vx = direction[0] * speed
    ball_vy = direction[1] * speed
    ball_vz = rng.uniform(3.5, 5.5)  # Small upward velocity for realism
    
    # Set ball state
    ball_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, "soccer_ball"
    )
    if ball_id != -1:
        # Position (qpos indices 36-42: 3 pos + 4 quat)
        mj_data.qpos[36:36+3] = [ball_x, ball_y, ball_z]
        mj_data.qpos[36+3:36+7] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
        
        # Velocity (qvel indices 35-40: 3 linear + 3 angular)
        mj_data.qvel[35:35+3] = [ball_vx, ball_vy, ball_vz]
        mj_data.qvel[35+3:35+6] = [0.0, 0.0, 0.0]  # No angular velocity
        
        print(
            f"\nBall positioned at: "
            f"x={ball_x:.2f}, y={ball_y:.2f}, z={ball_z:.3f}"
        )
        print(
            f"Ball velocity: vx={ball_vx:.2f} m/s, "
            f"vy={ball_vy:.2f} m/s, vz={ball_vz:.2f} m/s"
        )
        print(f"Ball speed: {speed:.2f} m/s toward goal")
        
        # Estimate time to goal
        dist_to_goal = np.linalg.norm(goal_pos - ball_pos_xy)
        time_to_goal = dist_to_goal / speed
        print(f"Estimated time to goal: {time_to_goal:.2f} seconds")
        print(
            "\nTask: Robot must intercept and stop the ball "
            "before it enters the goal!"
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

