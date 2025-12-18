import time
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Optional, Dict

# Add project root to path if needed
import sys
import os
sys.path.append(os.getcwd())

from hydrax.tasks.g1.g1_navigation import G1Navigation
from hydrax.algs.cem import CEM

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)


def run_navigation_rollout(
    task: Any,
    controller: Any,
    goal_pos: np.ndarray,
    duration: float = 10.0,
    frequency: float = 50.0,
) -> Dict[str, np.ndarray]:
    """Run a navigation rollout and return trajectory and cost data.
    
    Args:
        task: Task instance
        controller: Controller (MPC or None for RL only)
        goal_pos: Goal position [x, y, theta]
        duration: Duration of rollout in seconds
        frequency: Control frequency in Hz
        
    Returns:
        Dictionary with trajectory and cost data
    """
    print(f"Running navigation rollout for {duration}s to goal {goal_pos}...")
    
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    
    # Initialize state at knees_bent keyframe
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
        mj_data.ctrl[:] = task._default_pose
    
    # Set goal using mocap body (similar to pusht)
    # Convert theta to quaternion (rotation around z-axis)
    theta = goal_pos[2]
    quat = np.array([np.cos(theta/2), 0, 0, np.sin(theta/2)])  # [qw, qx, qy, qz]
    mj_data.mocap_pos[0, :] = np.array([goal_pos[0], goal_pos[1], 0.0])
    mj_data.mocap_quat[0, :] = quat
    
    # Setup timing
    replan_period = 1.0 / frequency
    sim_dt = mj_model.opt.timestep
    sim_steps_per_replan = int(replan_period / sim_dt)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    
    # Initialize controller
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos,
        mocap_quat=mj_data.mocap_quat
    )
    
    if hasattr(controller, "init_params"):
        # MPC Controller
        print("Initializing MPC controller...")
        policy_params = controller.init_params(initial_knots=None, seed=SEED)
        jit_optimize = jax.jit(controller.optimize)
        jit_interp_func = jax.jit(controller.interp_func)
        
        # Warmup
        print("Warming up controller...")
        policy_params, _ = jit_optimize(mjx_data, policy_params)
        policy_params, _ = jit_optimize(mjx_data, policy_params)
    else:
        # RL Only (not applicable for navigation, but included for completeness)
        policy_params = None
        jit_optimize = None
        jit_interp_func = None
        
    # Data logging
    times = []
    positions = []  # (x, y)
    orientations = []  # theta
    costs = []
    
    # Main loop
    num_replans = int(duration * frequency)
    
    start_time = time.time()
    
    for step in range(num_replans):
        # 1. Replan (if MPC)
        if policy_params is not None:
            # Update mjx_data from current mj_data
            mjx_data = mjx.put_data(mj_model, mj_data)
            mjx_data = mjx_data.replace(
                mocap_pos=mj_data.mocap_pos, 
                mocap_quat=mj_data.mocap_quat
            )
            
            policy_params, _ = jit_optimize(mjx_data, policy_params)
            
            # Interpolate controls for this period
            t_curr = mj_data.time
            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            tk = policy_params.tk
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]
        else:
            # RL Only: no planning, just use zero velocity commands
            us = np.zeros((sim_steps_per_replan, task.nu))
            
        # 2. Simulate substeps
        n_substeps = getattr(task, "n_substeps", 1)
        
        for i in range(sim_steps_per_replan):
            # Update control
            should_update_ctrl = (n_substeps == 1) or (i % n_substeps == 0)
            
            if should_update_ctrl:
                # Sync mjx_data for apply_control
                mjx_data = mjx.put_data(mj_model, mj_data)
                mjx_data = mjx_data.replace(
                    mocap_pos=mj_data.mocap_pos, 
                    mocap_quat=mj_data.mocap_quat
                )
                
                # Apply control
                ctrl_input = jnp.array(us[i])
                mjx_data = task.apply_control(mjx_data, ctrl_input)
                
                # Sync back to mj_data
                mj_data.ctrl[:] = np.array(mjx_data.ctrl)
            
            # Step physics
            mujoco.mj_step(mj_model, mj_data)
            
            # Log data
            times.append(mj_data.time)
            
            # Position (x, y)
            positions.append(mj_data.qpos[:2].copy())
            
            # Orientation (theta from quaternion)
            quat = mj_data.qpos[3:7]  # [qw, qx, qy, qz]
            # Extract yaw angle from quaternion
            # theta = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
            qw, qx, qy, qz = quat
            theta = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
            orientations.append(theta)
            
            # Compute cost
            mjx_data = mjx.put_data(mj_model, mj_data)
            mjx_data = mjx_data.replace(
                mocap_pos=mj_data.mocap_pos, 
                mocap_quat=mj_data.mocap_quat
            )
            cost = float(task.running_cost(mjx_data, jnp.array(us[i])))
            costs.append(cost)
            
        # Progress
        if step % 10 == 0:
            current_pos = mj_data.qpos[:2]
            dist_to_goal = np.linalg.norm(current_pos - goal_pos[:2])
            print(f"Step {step}/{num_replans}, Distance to goal: {dist_to_goal:.3f}m", end="\r")
            
    print(f"\nRollout complete. Time: {time.time() - start_time:.2f}s")
    
    return {
        "time": np.array(times),
        "positions": np.array(positions),
        "orientations": np.array(orientations),
        "costs": np.array(costs),
        "goal_pos": goal_pos,
    }


def main():
    # Define goal position: [x, y, theta]
    goal_pos = np.array([3.0, 1.0, 1.571/2])  # 3m forward, 2m left, facing forward
    duration = 5.0
    
    results = {}
    
    # Scenario: SPC (CEM-based MPC)
    print("\n--- SPC Navigation Experiment ---")
    task = G1Navigation()
    
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
    
    results["SPC"] = run_navigation_rollout(task, ctrl, goal_pos, duration)
    
    # Print final metrics
    print("\n" + "="*50)
    print("Navigation Metrics")
    print("-" * 50)
    
    res = results["SPC"]
    final_pos = res["positions"][-1]
    final_theta = res["orientations"][-1]
    
    pos_error = np.linalg.norm(final_pos - goal_pos[:2])
    theta_error = np.abs(final_theta - goal_pos[2])
    total_cost = np.sum(res["costs"])
    avg_cost = np.mean(res["costs"])
    
    print(f"Final position error: {pos_error:.4f} m")
    print(f"Final orientation error: {theta_error:.4f} rad")
    print(f"Total cumulative cost: {total_cost:.2f}")
    print(f"Average cost: {avg_cost:.4f}")
    print("="*50 + "\n")
    
    # Plotting
    print("Plotting results...")
    
    # Set publication-quality parameters (matching g1_velocity_tracking_experiment.py)
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'lines.linewidth': 2.5,
        'text.usetex': False,
    })
    
    # Color scheme (matching g1_velocity_tracking_experiment.py)
    colors = {
        "RL": "#F48B96",    # Pink/salmon
        "CEM": "#90CCEB",   # Light blue
        "iCEM": "#D1E7BE"   # Light green
    }
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: 2D Top-down trajectory
    ax1 = plt.subplot(1, 2, 1)
    
    positions = res["positions"]
    goal = res["goal_pos"]
    
    # Plot trajectory with CEM color
    ax1.plot(positions[:, 0], positions[:, 1], 
             color=colors["CEM"], linewidth=3, label='SPC trajectory')
    
    # Plot start position (using iCEM color)
    ax1.plot(positions[0, 0], positions[0, 1], 
             'o', color=colors["iCEM"], markersize=12, label='Start', 
             markeredgecolor='black', markeredgewidth=1.5)
    
    # Plot goal position (using RL color)
    ax1.plot(goal[0], goal[1], 
             '*', color=colors["RL"], markersize=20, label='Goal', 
             markeredgecolor='black', markeredgewidth=1.5)
    
    # Plot final position
    ax1.plot(positions[-1, 0], positions[-1, 1], 
             's', color=colors["CEM"], markersize=10, label='Final', 
             markeredgecolor='black', markeredgewidth=1.5)
    
    ax1.set_xlabel('X Position (m)', fontsize=14)
    ax1.set_ylabel('Y Position (m)', fontsize=14)
    ax1.set_title('2D Top-down Trajectory', fontsize=14)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Cost over time
    ax2 = plt.subplot(1, 2, 2)
    
    # Use CEM color for instantaneous cost
    ax2.plot(res["time"], res["costs"], 
             color=colors["CEM"], linewidth=3, label='Instantaneous cost')
    
    # Also plot cumulative cost
    cumulative_costs = np.cumsum(res["costs"])
    ax2_twin = ax2.twinx()
    ax2_twin.plot(res["time"], cumulative_costs, 
                  '--', color=colors["RL"], linewidth=2.5, label='Cumulative cost')
    
    ax2.set_xlabel('Time (s)', fontsize=14)
    ax2.set_ylabel('Instantaneous Cost', fontsize=14, color=colors["CEM"])
    ax2_twin.set_ylabel('Cumulative Cost', fontsize=14, color=colors["RL"])
    ax2.set_title('Cost over Time', fontsize=14)
    ax2.tick_params(axis='y', labelcolor=colors["CEM"])
    ax2_twin.tick_params(axis='y', labelcolor=colors["RL"])
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    output_path = "g1_navigation_experiment.eps"
    plt.savefig(output_path, format='eps', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Also save as PNG for easier viewing
    output_path_png = "g1_navigation_experiment.png"
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight')
    print(f"Plot also saved to {output_path_png}")


if __name__ == "__main__":
    main()
