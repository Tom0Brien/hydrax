"""Experiment comparing RL policy vs RL + SPC for Franka push cube task.

This experiment compares:
1. Baseline RL policy (zero residuals)
2. RL + SPC (CEM optimizing residuals)

Metrics: Box-to-target distance over time, success rate, task completion time.
"""

import time
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict

import sys
import os
sys.path.append(os.getcwd())

from hydrax.tasks.franka import FrankaPushGeometry
from hydrax.algs.cem import CEM

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)


def run_rollout(
    task: Any,
    controller: Any,
    reset_seed: int = 42,
    duration: float = 10.0,
    frequency: float = 50.0,
) -> Dict[str, np.ndarray]:
    """Run a deterministic rollout and return data.
    
    Args:
        task: FrankaPushCube task
        controller: CEM controller or None for RL-only
        reset_seed: Random seed for playground env reset (ensures reproducibility)
        duration: Rollout duration in seconds
        frequency: Control frequency in Hz
        
    Returns:
        Dictionary with time series data
    """
    print(f"Running rollout for {duration}s...")
    
    # Use unified reset interface
    rng = jax.random.PRNGKey(reset_seed)
    mj_data, mjx_data = task.reset(rng)
    
    mj_model = task.mj_model
    
    # Get the randomized target position from the reset state
    target_pos = np.array(mj_data.mocap_pos[0])
    
    print(f"Reset with seed: {reset_seed}")
    print(f"Box position: {mj_data.qpos[13:16]}")
    print(f"Target position: {target_pos}")
    
    # Setup timing
    replan_period = 1.0 / frequency
    sim_dt = mj_model.opt.timestep
    sim_steps_per_replan = int(replan_period / sim_dt)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    
    # mjx_data already created by reset
    
    if controller is not None:
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
        # RL Only - zero residuals
        policy_params = None
        jit_optimize = None
        jit_interp_func = None
        
    # Data logging
    times = []
    box_positions = []
    box_target_distances = []
    gripper_positions = []
    
    # Get object body id
    obj_body_id = task._obj_body
    gripper_site_id = task._gripper_site
    
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
            # RL Only: zero residuals
            us = np.zeros((sim_steps_per_replan, task.nu))
            
        # 2. Simulate substeps
        n_substeps = getattr(task, "n_substeps", 1)
        
        for i in range(sim_steps_per_replan):
            # Update control
            should_update_ctrl = (n_substeps == 1) or (i % n_substeps == 0)
            
            if should_update_ctrl:
                # Sync mjx_data for apply_control
                mjx_data = mjx.put_data(mj_model, mj_data)
                
                # Apply control (RL policy + residuals)
                ctrl_input = jnp.array(us[i])
                mjx_data = task.apply_control(mjx_data, ctrl_input)
                
                # Sync back to mj_data
                mj_data.ctrl[:] = np.array(mjx_data.ctrl)
            
            # Step physics
            mujoco.mj_step(mj_model, mj_data)
            
            # Log data
            times.append(mj_data.time)
            
            # Box position
            box_pos = mj_data.xpos[obj_body_id].copy()
            box_positions.append(box_pos)
            
            # Box-to-target distance (XY only)
            dist = np.linalg.norm(box_pos[:2] - target_pos[:2])
            box_target_distances.append(dist)
            
            # Gripper position
            gripper_pos = mj_data.site_xpos[gripper_site_id].copy()
            gripper_positions.append(gripper_pos)
            
        # Progress
        if step % 10 == 0:
            print(f"Step {step}/{num_replans}", end="\r")
            
    print(f"\nRollout complete. Time: {time.time() - start_time:.2f}s")
    
    return {
        "time": np.array(times),
        "box_pos": np.array(box_positions),
        "box_target_dist": np.array(box_target_distances),
        "gripper_pos": np.array(gripper_positions),
        "target_pos": target_pos,  # Now comes from playground reset
    }


def main():
    """Run comparison experiment."""
    # Experiment parameters
    reset_seed = 42  # Same seed for both conditions (fair comparison)
    duration = 8.0  # seconds
    
    results = {}
    
    # Scenario 1: RL Policy Alone (zero residuals)
    print("\n" + "="*50)
    print("Scenario 1: RL Policy Alone")
    print("="*50)
    task1 = FrankaPushGeometry(geometry="cube", use_rl_policy=True)
    results["RL"] = run_rollout(task1, None, reset_seed=reset_seed, duration=duration)
    
    # Scenario 2: RL + SPC (CEM)
    print("\n" + "="*50)
    print("Scenario 2: RL + SPC (CEM)")
    print("="*50)
    task2 = FrankaPushGeometry(geometry="cube", use_rl_policy=True)
    ctrl2 = CEM(
        task=task2,
        num_samples=128,
        num_elites=16,
        sigma_start=0.1,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=6,
    )
    results["RL+SPC"] = run_rollout(task2, ctrl2, reset_seed=reset_seed, duration=duration)
    
    # Calculate metrics
    print("\n" + "="*60)
    print(f"{'Metric':<30} | {'RL':<12} | {'RL+SPC':<12}")
    print("-" * 60)
    
    for name, res in results.items():
        # Skip first 1s for transient
        mask = res["time"] > 1.0
        if not np.any(mask):
            mask = np.ones_like(res["time"], dtype=bool)
        
        # Mean distance after transient
        mean_dist = np.mean(res["box_target_dist"][mask])
        
        # Final distance
        final_dist = res["box_target_dist"][-1]
        
        # Minimum distance achieved
        min_dist = np.min(res["box_target_dist"][mask])
        
        # Time to reach within 5cm
        within_5cm = res["box_target_dist"] < 0.05
        if np.any(within_5cm):
            time_to_5cm = res["time"][np.argmax(within_5cm)]
        else:
            time_to_5cm = float('inf')
        
        results[name]["metrics"] = {
            "mean_dist": mean_dist,
            "final_dist": final_dist,
            "min_dist": min_dist,
            "time_to_5cm": time_to_5cm,
        }
    
    # Print metrics
    print(f"{'Mean Distance (m)':<30} | {results['RL']['metrics']['mean_dist']:<12.4f} | {results['RL+SPC']['metrics']['mean_dist']:<12.4f}")
    print(f"{'Final Distance (m)':<30} | {results['RL']['metrics']['final_dist']:<12.4f} | {results['RL+SPC']['metrics']['final_dist']:<12.4f}")
    print(f"{'Min Distance (m)':<30} | {results['RL']['metrics']['min_dist']:<12.4f} | {results['RL+SPC']['metrics']['min_dist']:<12.4f}")
    
    time_rl = results['RL']['metrics']['time_to_5cm']
    time_spc = results['RL+SPC']['metrics']['time_to_5cm']
    print(f"{'Time to 5cm (s)':<30} | {time_rl if time_rl != float('inf') else 'N/A':<12} | {time_spc if time_spc != float('inf') else 'N/A':<12}")
    print("="*60 + "\n")
    
    # Plotting
    print("Plotting results...")
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'lines.linewidth': 2.5,
    })
    
    # Color scheme
    colors = {
        "RL": "#F48B96",      # Pink/salmon
        "RL+SPC": "#90CCEB",  # Light blue
    }
    
    # Figure 1: Distance over time
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    
    # Plot 1: Box-to-target distance
    ax = axes[0]
    for name, res in results.items():
        ax.plot(res["time"], res["box_target_dist"], 
                color=colors[name], lw=2.5, label=name)
    ax.axhline(y=0.05, color='g', linestyle='--', label='Success threshold (5cm)')
    ax.set_ylabel("Box-Target Distance (m)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Push Task Performance: Box Distance to Target")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    
    # Plot 2: XY Trajectory of box
    ax = axes[1]
    for name, res in results.items():
        ax.plot(res["box_pos"][:, 0], res["box_pos"][:, 1], 
                color=colors[name], lw=2.5, label=f"{name} Box")
    # Plot target (get from results)
    target_pos = results["RL"]["target_pos"]
    ax.scatter([target_pos[0]], [target_pos[1]], s=200, c='green', marker='*', 
               label='Target', zorder=10)
    # Plot initial box position
    ax.scatter([results["RL"]["box_pos"][0, 0]], [results["RL"]["box_pos"][0, 1]], 
               s=100, c='black', marker='s', label='Start', zorder=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Box Trajectory (Top View)")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = "franka_push_comparison.eps"
    plt.savefig(output_path, format='eps', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Also save as PNG for quick viewing
    plt.savefig("franka_push_comparison.png", format='png', dpi=150, bbox_inches='tight')
    print("Plot also saved as franka_push_comparison.png")
    
    plt.show()


if __name__ == "__main__":
    main()
