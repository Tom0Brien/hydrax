"""Simple script to run a single constrained push rollout and plot the trajectory.

Generates a 2D top-down plot showing:
- Box start position (pink)
- Target/goal position (light green)
- Obstacle (light blue cylinder)
- Box trajectory (dashed line)

Usage:
    python examples/franka_push_constrained_plot.py
"""

import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mujoco
from mujoco import mjx
import numpy as np

import sys
import os
sys.path.append(os.getcwd())

from hydrax.tasks.franka import FrankaPushConstrained
from hydrax.algs import CCEM


# Colors from the reference image
COLOR_OBJECT = "#E8B4B8"      # Pink/rose for object
COLOR_OBSTACLE = "#A8D8EA"    # Light blue for obstacle
COLOR_GOAL = "#C5E1A5"        # Light green for goal
COLOR_TRAJECTORY = "#666666"  # Gray for trajectory


def run_single_rollout(
    task: FrankaPushConstrained,
    controller: CCEM,
    duration: float = 6.0,
    frequency: float = 50.0,
    seed: int = 42,
):
    """Run a single rollout and collect box trajectory."""
    model = task.model
    n_substeps = task.n_substeps
    
    # Initialize from home keyframe
    home_qpos = jnp.array([
        -0.182772, 0.146282, 0.172246, -2.24238, -0.0788546, 2.45127, 0.0160022,  # arm (7)
        0.8, 0.8, 0.8, 0.8, 0.8, 0.8,  # gripper (6)
        0.60, -0.3, 0.03,  # box position (3)
        0.0, 0.0, 0.0, 1.0,  # box quaternion wxyz (4)
    ])
    home_ctrl = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.82])
    
    # Target position
    target_pos = jnp.array([0.60, 0.3, 0.03])
    
    # Create mjx data
    mjx_data = mjx.make_data(model, nconmax=256, njmax=256)
    mjx_data = mjx_data.replace(
        qpos=home_qpos,
        qvel=jnp.zeros_like(mjx_data.qvel),
        ctrl=home_ctrl,
        mocap_pos=jnp.array([[target_pos[0], target_pos[1], target_pos[2]]]),
        mocap_quat=jnp.array([[1.0, 0.0, 0.0, 0.0]]),
        time=jnp.array(0.0),
    )
    mjx_data = mjx.forward(model, mjx_data)
    
    # Initialize controller
    policy_params = controller.init_params(initial_knots=None, seed=seed)
    
    # JIT compile
    jit_optimize = jax.jit(controller.optimize)
    jit_interp_func = jax.jit(controller.interp_func)
    
    @jax.jit
    def step_env(data, ctrl):
        data = task.apply_control(data, ctrl)
        def single_step(d, _):
            return mjx.step(model, d), None
        data = jax.lax.scan(single_step, data, None, n_substeps)[0]
        return data
    
    # Warmup
    _, _ = jit_optimize(mjx_data, policy_params)
    
    # Timing
    replan_period = 1.0 / frequency
    sim_dt = task.dt
    sim_steps_per_replan = max(1, int(replan_period / sim_dt))
    num_replans = int(duration * frequency)
    
    # Collect trajectory
    obj_body_id = task._obj_body
    box_trajectory = []
    
    # Get start position
    start_pos = np.array(mjx_data.xpos[obj_body_id, :2])
    
    print("Running rollout...")
    start_time = time.time()
    
    for step in range(num_replans):
        # Optimize
        policy_params, _ = jit_optimize(mjx_data, policy_params)
        
        # Get controls
        t_curr = mjx_data.time
        tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
        knots = policy_params.mean[None, ...]
        us = jit_interp_func(tq, policy_params.tk, knots)[0]
        
        # Step simulation
        for i in range(sim_steps_per_replan):
            mjx_data = step_env(mjx_data, us[i])
            box_pos = np.array(mjx_data.xpos[obj_body_id, :2])
            box_trajectory.append(box_pos.copy())
    
    elapsed = time.time() - start_time
    print(f"Rollout completed in {elapsed:.1f}s")
    
    # Get final position
    final_pos = np.array(mjx_data.xpos[obj_body_id, :2])
    
    return {
        "start_pos": start_pos,
        "target_pos": np.array(target_pos[:2]),
        "trajectory": np.array(box_trajectory),
        "final_pos": final_pos,
        "obstacle_pos": np.array([0.60, 0.0]),
        "obstacle_radius": 0.04,
        "obj_size": 0.03,  # box half-size
    }


def plot_trajectory(data: dict, output_path: str = "franka_push_trajectory.png"):
    """Generate a clean 2D top-down plot of the trajectory (rotated 90 deg).
    
    The plot is rotated so that Y becomes the horizontal axis and X becomes vertical.
    This better matches the visual perspective of the task.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set up the plot (swapped axes for 90 deg rotation)
    ax.set_aspect('equal')
    ax.set_xlabel('Y (m)', fontsize=12)
    ax.set_ylabel('X (m)', fontsize=12)
    ax.set_title('Constrained Push Task: Box Trajectory', fontsize=14)
    
    # Helper to swap coordinates (rotate 90 deg: x,y -> y,x)
    def rot(pos):
        return (pos[1], pos[0])
    
    # Draw obstacle (cylinder from top = circle)
    obstacle = plt.Circle(
        rot(data["obstacle_pos"]), 
        data["obstacle_radius"],
        facecolor=COLOR_OBSTACLE,
        edgecolor='#5DADE2',
        linewidth=2,
        label='Obstacle',
        zorder=2,
    )
    ax.add_patch(obstacle)
    
    # Draw start position (box = square)
    obj_size = data["obj_size"]
    start_rot = rot(data["start_pos"])
    start_rect = patches.Rectangle(
        (start_rot[0] - obj_size, start_rot[1] - obj_size),
        2 * obj_size, 2 * obj_size,
        facecolor=COLOR_OBJECT,
        edgecolor='#C0392B',
        linewidth=2,
        label='Start',
        zorder=3,
    )
    ax.add_patch(start_rect)
    
    # Draw goal position (box = square)
    goal_rot = rot(data["target_pos"])
    goal_rect = patches.Rectangle(
        (goal_rot[0] - obj_size, goal_rot[1] - obj_size),
        2 * obj_size, 2 * obj_size,
        facecolor=COLOR_GOAL,
        edgecolor='#27AE60',
        linewidth=2,
        label='Goal',
        zorder=3,
    )
    ax.add_patch(goal_rect)
    
    # Draw trajectory (swap x,y -> y,x)
    traj = data["trajectory"]
    ax.plot(
        traj[:, 1], traj[:, 0],  # Swapped: y, x
        color=COLOR_TRAJECTORY,
        linestyle='--',
        linewidth=2,
        label='Trajectory',
        zorder=1,
    )
    
    # Draw final position marker
    final_rot = rot(data["final_pos"])
    ax.scatter(
        final_rot[0], final_rot[1],
        color='#2C3E50',
        s=50,
        marker='x',
        linewidths=2,
        label='Final',
        zorder=4,
    )
    
    # Add labels (centered directly above each object)
    ax.annotate('Object', start_rot, 
                textcoords="offset points", xytext=(0, 35),
                fontsize=13, fontweight='bold', color='#C0392B', ha='center')
    ax.annotate('Goal', goal_rot, 
                textcoords="offset points", xytext=(0, 35),
                fontsize=13, fontweight='bold', color='#27AE60', ha='center')
    ax.annotate('Obstacle', rot(data["obstacle_pos"]), 
                textcoords="offset points", xytext=(0, 35),
                fontsize=13, fontweight='bold', color='#5DADE2', ha='center')
    
    # Set axis limits with some padding (using swapped coordinates)
    all_y = np.concatenate([traj[:, 0], [data["start_pos"][0], data["target_pos"][0], data["obstacle_pos"][0]]])
    all_x = np.concatenate([traj[:, 1], [data["start_pos"][1], data["target_pos"][1], data["obstacle_pos"][1]]])
    padding = 0.1
    ax.set_xlim(all_x.min() - padding, all_x.max() + padding)
    ax.set_ylim(all_y.min() - padding, all_y.max() + padding)
    
    # Legend
    ax.legend(loc='lower left', fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    plt.show()



def main():
    print("\n" + "="*60)
    print("Constrained Push: Single Rollout Trajectory Plot")
    print("="*60)
    
    # Create task
    task = FrankaPushConstrained(
        geometry="square",
        use_rl_policy=True,
        safety_margin=0.02,
    )
    
    # Update obstacle position in model
    obstacle_body_id = task._obstacle_body
    obstacle_pos = jnp.array([0.60, 0.0, 0.08])
    new_body_pos = task.model.body_pos.at[obstacle_body_id].set(obstacle_pos)
    task.model = task.model.tree_replace({"body_pos": new_body_pos})
    
    # Create controller
    controller = CCEM(
        task=task,
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
    
    print(f"Box start: (0.60, -0.30)")
    print(f"Target: (0.60, 0.30)")
    print(f"Obstacle: (0.60, 0.00) with radius 0.04m")
    print("="*60 + "\n")
    
    # Run rollout
    data = run_single_rollout(task, controller, duration=6.0)
    
    # Compute final distance
    final_dist = np.linalg.norm(data["final_pos"] - data["target_pos"])
    print(f"\nFinal distance to target: {final_dist:.3f}m")
    
    # Plot
    plot_trajectory(data)


if __name__ == "__main__":
    main()
