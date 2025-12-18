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

from hydrax.tasks.g1.g1_soccer import G1Soccer
from hydrax.tasks.g1.g1_soccer_augmented import G1SoccerAugmented
from hydrax.algs.cem import CEM

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)


def run_soccer_rollout(
    task: Any,
    controller: Any,
    ball_init_pos: np.ndarray,
    goal_pos: np.ndarray,
    duration: float = 15.0,
    frequency: float = 50.0,
) -> Dict[str, np.ndarray]:
    """Run a soccer rollout and return trajectory and cost data.
    
    Args:
        task: Task instance (G1Soccer or G1SoccerAugmented)
        controller: CEM controller
        ball_init_pos: Initial ball position [x, y]
        goal_pos: Goal position [x, y]
        duration: Duration of rollout in seconds
        frequency: Control frequency in Hz
        
    Returns:
        Dictionary with trajectory and cost data
    """
    print(f"Running soccer rollout for {duration}s...")
    
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    
    # Initialize state at knees_bent keyframe
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
        mj_data.ctrl[:] = task._default_pose
    
    # Set soccer ball initial position
    mj_data.qpos[36:36+3] = [ball_init_pos[0], ball_init_pos[1], 0.117]  # x, y, z (radius)
    mj_data.qpos[36+3:36+7] = [1.0, 0.0, 0.0, 0.0]  # Quaternion identity
    
    # Set goal position via mocap body
    mj_data.mocap_pos[0, :] = np.array([goal_pos[0], goal_pos[1], 0.05])
    mj_data.mocap_quat[0, :] = np.array([1.0, 0.0, 0.0, 0.0])
    
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
    
    print("Initializing MPC controller...")
    policy_params = controller.init_params(initial_knots=None, seed=SEED)
    jit_optimize = jax.jit(controller.optimize)
    jit_interp_func = jax.jit(controller.interp_func)
    
    # Warmup
    print("Warming up controller...")
    policy_params, _ = jit_optimize(mjx_data, policy_params)
    policy_params, _ = jit_optimize(mjx_data, policy_params)
    
    # Data logging
    times = []
    robot_positions = []  # Robot (x, y)
    ball_positions = []   # Ball (x, y)
    costs = []
    
    # Main loop
    num_replans = int(duration * frequency)
    
    start_time = time.time()
    
    for step in range(num_replans):
        # 1. Replan
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
            
            # Robot position (x, y)
            robot_positions.append(mj_data.qpos[:2].copy())
            
            # Ball position (x, y)
            ball_positions.append(mj_data.qpos[36:38].copy())
            
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
            ball_pos = mj_data.qpos[36:38]
            dist_to_goal = np.linalg.norm(ball_pos - goal_pos)
            print(f"Step {step}/{num_replans}, Ball dist to goal: {dist_to_goal:.3f}m", end="\r")
            
    print(f"\nRollout complete. Time: {time.time() - start_time:.2f}s")
    
    return {
        "time": np.array(times),
        "robot_positions": np.array(robot_positions),
        "ball_positions": np.array(ball_positions),
        "costs": np.array(costs),
        "ball_init_pos": ball_init_pos,
        "goal_pos": goal_pos,
    }


def main():
    # Define initial ball and goal positions
    ball_init_pos = np.array([2.0, 0.5])  # Start with ball slightly offset
    goal_pos = np.array([4.75, 0.0])     # Goal position
    duration = 5.0
    
    results = {}
    
    # Scenario 1: Baseline (3D velocity commands only)
    print("\n" + "="*60)
    print("Scenario 1: Baseline Soccer (3D velocity commands)")
    print("="*60)
    
    task_baseline = G1Soccer()
    
    ctrl_baseline = CEM(
        task=task_baseline,
        num_samples=32,
        num_elites=8,
        sigma_start=0.5,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=1,
        spline_type="zero",
        num_knots=4,
    )
    
    results["Baseline"] = run_soccer_rollout(
        task_baseline, ctrl_baseline, ball_init_pos, goal_pos, duration
    )
    
    # Scenario 2: Augmented (15D with leg residuals)
    print("\n" + "="*60)
    print("Scenario 2: Augmented Soccer (15D with leg residuals)")
    print("="*60)
    
    task_augmented = G1SoccerAugmented()
    
    ctrl_augmented = CEM(
        task=task_augmented,
        num_samples=32,
        num_elites=8,
        sigma_start=0.5,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=1,
        spline_type="zero",
        num_knots=4,
    )
    
    results["Augmented"] = run_soccer_rollout(
        task_augmented, ctrl_augmented, ball_init_pos, goal_pos, duration
    )
    
    # Print final metrics
    print("\n" + "="*60)
    print("Soccer Metrics Comparison")
    print("-" * 60)
    
    for name, res in results.items():
        final_ball_pos = res["ball_positions"][-1]
        ball_to_goal_dist = np.linalg.norm(final_ball_pos - goal_pos)
        total_cost = np.sum(res["costs"])
        avg_cost = np.mean(res["costs"])
        
        print(f"\n{name}:")
        print(f"  Final ball-to-goal distance: {ball_to_goal_dist:.4f} m")
        print(f"  Total cumulative cost: {total_cost:.2f}")
        print(f"  Average cost: {avg_cost:.4f}")
    
    print("="*60 + "\n")
    
    # Plotting
    print("Plotting results...")
    
    # Set publication-quality parameters
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
    
    # Color scheme
    colors = {
        "Baseline": "#90CCEB",    # Light blue
        "Augmented": "#F48B96",   # Pink/salmon
        "Start": "#D1E7BE",       # Light green
        "Goal": "#FFD700",        # Gold
    }
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: 2D Top-down trajectory (ball positions)
    ax1 = plt.subplot(1, 2, 1)
    
    for name, res in results.items():
        ball_positions = res["ball_positions"]
        color = colors[name]
        
        # Plot ball trajectory
        ax1.plot(ball_positions[:, 0], ball_positions[:, 1], 
                 color=color, linewidth=3, label=f'{name} ball trajectory',
                 alpha=0.8)
        
        # Plot final ball position
        ax1.plot(ball_positions[-1, 0], ball_positions[-1, 1], 
                 'o', color=color, markersize=10, 
                 markeredgecolor='black', markeredgewidth=1.5)
    
    # Plot initial ball position
    ax1.plot(ball_init_pos[0], ball_init_pos[1], 
             'o', color=colors["Start"], markersize=12, label='Ball start', 
             markeredgecolor='black', markeredgewidth=1.5)
    
    # Plot goal position
    ax1.plot(goal_pos[0], goal_pos[1], 
             '*', color=colors["Goal"], markersize=20, label='Goal', 
             markeredgecolor='black', markeredgewidth=1.5)
    
    ax1.set_xlabel('X Position (m)', fontsize=14)
    ax1.set_ylabel('Y Position (m)', fontsize=14)
    ax1.set_title('Ball Trajectory (Top-down)', fontsize=14)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Cost over time
    ax2 = plt.subplot(1, 2, 2)
    
    for name, res in results.items():
        color = colors[name]
        ax2.plot(res["time"], res["costs"], 
                 color=color, linewidth=3, label=f'{name}',
                 alpha=0.8)
    
    ax2.set_xlabel('Time (s)', fontsize=14)
    ax2.set_ylabel('Instantaneous Cost', fontsize=14)
    ax2.set_title('Cost over Time', fontsize=14)
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = "g1_soccer_experiment.eps"
    plt.savefig(output_path, format='eps', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Also save as PNG for easier viewing
    output_path_png = "g1_soccer_experiment.png"
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight')
    print(f"Plot also saved to {output_path_png}")


if __name__ == "__main__":
    main()

