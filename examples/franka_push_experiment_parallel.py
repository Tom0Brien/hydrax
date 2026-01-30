"""Parallel experiment for Franka push cube task using JAX vectorization.

This script runs multiple environments in parallel using JAX vmap/scan,
leveraging MJX for GPU-accelerated simulation.

Compares:
1. Baseline policy (zero residuals)
2. Policy-guided CEM (CEM optimizing residuals around policy)
3. CEM Only (no policy)

Usage:
    python franka_push_experiment_parallel.py --num_evals 16 --num_envs 4 --duration 5.0
"""

import argparse
import math
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math

import sys
import os
sys.path.append(os.getcwd())

from hydrax.tasks.franka import FrankaPushGeometry
from hydrax.algs.cem import CEM, CEMParams


class ParallelRolloutData(NamedTuple):
    """Data collected from parallel rollouts."""
    time: jax.Array  # (num_steps,)
    box_pos: jax.Array  # (num_envs, num_steps, 3)
    box_quat: jax.Array  # (num_envs, num_steps, 4)
    box_target_dist: jax.Array  # (num_envs, num_steps)
    box_ori_error: jax.Array  # (num_envs, num_steps) - orientation error in radians
    gripper_pos: jax.Array  # (num_envs, num_steps, 3)
    target_pos: jax.Array  # (num_envs, 3)
    target_quat: jax.Array  # (num_envs, 4)
    running_cost: jax.Array  # (num_envs, num_steps) - per-step running cost


def quat_angle_error(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """Compute orientation error between two quaternions in radians.
    
    Args:
        q1: Quaternion (w, x, y, z) shape (..., 4)
        q2: Quaternion (w, x, y, z) shape (..., 4)
        
    Returns:
        Angular error in radians, shape (...)
    """
    # q_diff = q1 * q2^-1
    q2_inv = q2.at[..., 1:].multiply(-1)  # Conjugate for unit quaternion
    
    # Quaternion multiplication
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2_inv[..., 0], q2_inv[..., 1], q2_inv[..., 2], q2_inv[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    # Angle from quaternion: 2 * arccos(|w|)
    # Use arcsin for better numerical stability near identity
    sin_half_angle = jnp.sqrt(x**2 + y**2 + z**2)
    angle = 2.0 * jnp.arcsin(jnp.clip(sin_half_angle, 0.0, 1.0))
    
    return angle


def create_batched_reset(task: FrankaPushGeometry, num_envs: int):
    """Create a function to reset multiple environments in parallel."""
    @jax.jit
    def batch_reset(rng: jax.Array) -> mjx.Data:
        """Reset num_envs environments in parallel."""
        rngs = jax.random.split(rng, num_envs)
        return jax.vmap(task.mjx_reset)(rngs)
    
    return batch_reset


def create_parallel_step_fn(task: FrankaPushGeometry):
    """Create a function to step multiple environments in parallel."""
    @jax.jit
    def parallel_step(mjx_data: mjx.Data, ctrl_batch: jax.Array) -> mjx.Data:
        """Step all environments with given controls."""
        def step_single(data, ctrl):
            data = task.apply_control(data, ctrl)
            return task.step(task.model, data)
        
        return jax.vmap(step_single)(mjx_data, ctrl_batch)
    
    return parallel_step


def run_parallel_rollout(
    task: FrankaPushGeometry,
    controller: CEM | None,
    num_envs: int,
    base_seed: int = 42,
    duration: float = 10.0,
    frequency: float = 50.0,
) -> ParallelRolloutData:
    """Run parallel rollouts across multiple environments.
    
    Args:
        task: FrankaPushGeometry task
        controller: CEM controller or None for RL-only
        num_envs: Number of parallel environments
        base_seed: Base random seed
        duration: Rollout duration in seconds
        frequency: Control frequency in Hz
        
    Returns:
        ParallelRolloutData with collected data
    """
    print(f"Running parallel rollout: {num_envs} envs for {duration}s...")
    
    # Create batched functions
    batch_reset = create_batched_reset(task, num_envs)
    parallel_step = create_parallel_step_fn(task)
    
    # Reset all environments
    rng = jax.random.PRNGKey(base_seed)
    mjx_data = batch_reset(rng)
    
    # Get target positions and orientations from reset state
    target_pos = mjx_data.mocap_pos[:, 0, :]  # (num_envs, 3)
    target_quat = mjx_data.mocap_quat[:, 0, :]  # (num_envs, 4)
    
    print(f"Reset complete. Target positions shape: {target_pos.shape}")
    
    # Timing setup
    replan_period = 1.0 / frequency
    sim_dt = task.dt
    n_substeps = task.n_substeps
    sim_steps_per_replan = int(replan_period / sim_dt)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    num_replans = int(duration * frequency)
    
    # Initialize controller if provided
    if controller is not None:
        # Initialize policy params for each environment
        policy_params_list = [
            controller.init_params(initial_knots=None, seed=base_seed + i)
            for i in range(num_envs)
        ]
        policy_params_batch = jax.tree.map(
            lambda *xs: jnp.stack(xs, axis=0), 
            *policy_params_list
        )
        
        jit_optimize = jax.jit(controller.optimize)
        jit_interp_func = jax.jit(controller.interp_func)
        
        # Warmup with first env
        print("Warming up controller...")
        single_data = jax.tree.map(lambda x: x[0], mjx_data)
        single_params = jax.tree.map(lambda x: x[0], policy_params_batch)
        _, _ = jit_optimize(single_data, single_params)
        _, _ = jit_optimize(single_data, single_params)
        
        # Batched optimize
        @jax.jit
        def batched_optimize(mjx_data_batch, params_batch):
            """Optimize all environments in parallel."""
            return jax.vmap(jit_optimize)(mjx_data_batch, params_batch)
    else:
        policy_params_batch = None
    
    # Data collection arrays
    all_times = []
    all_box_pos = []
    all_box_quat = []
    all_box_target_dist = []
    all_box_ori_error = []
    all_gripper_pos = []
    all_running_cost = []
    
    # JIT compile running cost function for batch evaluation
    @jax.jit
    def compute_running_cost_batch(mjx_data_batch, ctrl_batch):
        """Compute running cost for all environments."""
        return jax.vmap(task.running_cost)(mjx_data_batch, ctrl_batch)
    
    # Get body/site IDs
    obj_body_id = task._obj_body
    gripper_site_id = task._gripper_site
    
    start_time = time.time()
    
    for step in range(num_replans):
        # Replan controls
        if controller is not None:
            policy_params_batch, _ = batched_optimize(mjx_data, policy_params_batch)
            
            t_curr = mjx_data.time[0]
            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            
            def get_controls_for_env(tk, mean):
                knots = mean[None, ...]
                us = jit_interp_func(tq, tk, knots)[0]
                return us
            
            us_batch = jax.vmap(get_controls_for_env)(
                policy_params_batch.tk, 
                policy_params_batch.mean
            )
        else:
            us_batch = jnp.zeros((num_envs, sim_steps_per_replan, task.nu))
        
        # Simulate substeps
        for i in range(sim_steps_per_replan):
            ctrl_batch = us_batch[:, i, :]
            mjx_data = parallel_step(mjx_data, ctrl_batch)
            
            # Log data
            all_times.append(float(mjx_data.time[0]))
            
            # Box positions and quaternions
            box_pos = mjx_data.xpos[:, obj_body_id, :]  # (num_envs, 3)
            box_quat = mjx_data.xquat[:, obj_body_id, :]  # (num_envs, 4)
            all_box_pos.append(box_pos)
            all_box_quat.append(box_quat)
            
            # Position error (XY only)
            dist = jnp.linalg.norm(box_pos[:, :2] - target_pos[:, :2], axis=1)
            all_box_target_dist.append(dist)
            
            # Orientation error
            ori_error = quat_angle_error(box_quat, target_quat)
            all_box_ori_error.append(ori_error)
            
            # Gripper positions
            gripper_pos = mjx_data.site_xpos[:, gripper_site_id, :]
            all_gripper_pos.append(gripper_pos)
            
            # Running cost
            step_cost = compute_running_cost_batch(mjx_data, ctrl_batch)
            all_running_cost.append(step_cost)
        
        if step % 10 == 0:
            print(f"Step {step}/{num_replans}", end="\r")
    
    elapsed = time.time() - start_time
    print(f"\nRollout complete. Time: {elapsed:.2f}s ({num_envs * duration / elapsed:.1f}x realtime)")
    
    return ParallelRolloutData(
        time=jnp.array(all_times),
        box_pos=jnp.stack(all_box_pos, axis=1),
        box_quat=jnp.stack(all_box_quat, axis=1),
        box_target_dist=jnp.stack(all_box_target_dist, axis=1),
        box_ori_error=jnp.stack(all_box_ori_error, axis=1),
        gripper_pos=jnp.stack(all_gripper_pos, axis=1),
        target_pos=target_pos,
        target_quat=target_quat,
        running_cost=jnp.stack(all_running_cost, axis=1),
    )


def run_batched_experiment(
    task: FrankaPushGeometry,
    controller: CEM | None,
    num_evals: int,
    num_envs: int,
    base_seed: int,
    duration: float,
) -> ParallelRolloutData:
    """Run experiment in batches until num_evals is reached.
    
    Args:
        task: Task instance
        controller: Controller or None
        num_evals: Total number of evaluations to run
        num_envs: Number of parallel environments per batch
        base_seed: Base seed
        duration: Duration per rollout
        
    Returns:
        Combined ParallelRolloutData from all batches
    """
    num_batches = math.ceil(num_evals / num_envs)
    actual_evals = num_batches * num_envs
    
    print(f"Running {num_batches} batches of {num_envs} envs = {actual_evals} total evals")
    
    all_data = []
    for batch_idx in range(num_batches):
        batch_seed = base_seed + batch_idx * num_envs * 1000
        data = run_parallel_rollout(
            task, controller,
            num_envs=num_envs,
            base_seed=batch_seed,
            duration=duration,
        )
        all_data.append(data)
    
    # Concatenate all batches
    combined = ParallelRolloutData(
        time=all_data[0].time,  # Same for all
        box_pos=jnp.concatenate([d.box_pos for d in all_data], axis=0),
        box_quat=jnp.concatenate([d.box_quat for d in all_data], axis=0),
        box_target_dist=jnp.concatenate([d.box_target_dist for d in all_data], axis=0),
        box_ori_error=jnp.concatenate([d.box_ori_error for d in all_data], axis=0),
        gripper_pos=jnp.concatenate([d.gripper_pos for d in all_data], axis=0),
        target_pos=jnp.concatenate([d.target_pos for d in all_data], axis=0),
        target_quat=jnp.concatenate([d.target_quat for d in all_data], axis=0),
        running_cost=jnp.concatenate([d.running_cost for d in all_data], axis=0),
    )
    
    return combined


def compute_metrics(data: ParallelRolloutData) -> dict:
    """Compute metrics from parallel rollout data.
    
    Metrics:
    - Final position error (median + IQR)
    - Final orientation error (median + IQR)
    - Total running cost (accumulated over episode)
    - Time to success (dist < 3cm AND ori < 10°)
    - Success rate (based on final state)
    """
    times = np.array(data.time)
    dists = np.array(data.box_target_dist)  # (num_envs, num_steps)
    ori_errors = np.array(data.box_ori_error)  # (num_envs, num_steps)
    running_costs = np.array(data.running_cost)  # (num_envs, num_steps)
    
    num_envs = dists.shape[0]
    
    # Final state metrics
    final_dist_per_env = dists[:, -1]
    final_ori_per_env = ori_errors[:, -1]  # in radians
    
    # Total running cost (accumulated over episode)
    total_running_cost_per_env = np.sum(running_costs, axis=1)
    
    # Helper for median + IQR stats
    def median_iqr(arr):
        median = float(np.median(arr))
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        return median, q1, q3
    
    # Final position with median/IQR
    final_dist_median, final_dist_q1, final_dist_q3 = median_iqr(final_dist_per_env)
    
    # Final orientation with median/IQR (in degrees)
    final_ori_deg_per_env = final_ori_per_env * 180 / np.pi
    final_ori_median, final_ori_q1, final_ori_q3 = median_iqr(final_ori_deg_per_env)
    
    # Total running cost with median/IQR
    total_cost_median, total_cost_q1, total_cost_q3 = median_iqr(total_running_cost_per_env)
    
    # Success thresholds: dist < 3cm AND ori < 10 degrees
    pos_threshold = 0.05  # 3cm position error
    ori_threshold = 15.0 * np.pi / 180  # 10 degrees orientation error
    
    # Time to success: first time BOTH position and orientation are within threshold
    def time_to_success(dist_seq, ori_seq):
        within_pos = dist_seq < pos_threshold
        within_ori = ori_seq < ori_threshold
        success_mask = within_pos & within_ori
        if np.any(success_mask):
            return times[np.argmax(success_mask)]
        return float('inf')
    
    time_to_success_per_env = np.array([
        time_to_success(dists[i], ori_errors[i]) for i in range(num_envs)
    ])
    
    # Time to success with median/IQR (only for successful runs)
    successful_times = time_to_success_per_env[time_to_success_per_env < float('inf')]
    if len(successful_times) > 0:
        time_success_median, time_success_q1, time_success_q3 = median_iqr(successful_times)
    else:
        time_success_median = float('inf')
        time_success_q1 = float('inf')
        time_success_q3 = float('inf')
    
    # Success rate (based on final state)
    success_per_env = (final_dist_per_env < pos_threshold) & (final_ori_per_env < ori_threshold)
    print("Success rate:", np.mean(success_per_env))
    return {
        # Final position (median + IQR)
        "final_dist": final_dist_median,
        "final_dist_q1": final_dist_q1,
        "final_dist_q3": final_dist_q3,
        
        # Final orientation in degrees (median + IQR)
        "final_ori_error": final_ori_median,
        "final_ori_error_q1": final_ori_q1,
        "final_ori_error_q3": final_ori_q3,
        
        # Total running cost (median + IQR)
        "total_running_cost": total_cost_median,
        "total_running_cost_q1": total_cost_q1,
        "total_running_cost_q3": total_cost_q3,
        
        # Time to success (median + IQR)
        "time_to_success": time_success_median,
        "time_to_success_q1": time_success_q1,
        "time_to_success_q3": time_success_q3,
        
        # Success rate
        "success_rate": float(np.mean(success_per_env)),
        
        # Count
        "num_evals": num_envs,
    }


def plot_bar_comparison(results: dict, output_path: str):
    """Create bar plot comparing metrics across approaches.
    
    Uses median with IQR (25th-75th percentile) for error bars.
    """
    approaches = ["Policy", "CEM", "Policy-guided CEM"]
    colors = {"Policy": "#F48B96", "CEM": "#9ACD32", "Policy-guided CEM": "#90CCEB"}
    
    # Metrics to plot: (key, label, is_percentage, format_str)
    metrics = [
        ("final_dist", "Final Position Error (m)", False, ".3f"),
        ("final_ori_error", "Final Orientation Error (°)", False, ".1f"),
        ("total_running_cost", "Total Running Cost", False, ".1f"),
        ("time_to_success", "Time to Success (s)", False, ".2f"),
        ("success_rate", "Success Rate (%)", True, ".1f"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for ax, (metric, label, is_percentage, fmt) in zip(axes, metrics):
        values = [results[a]["metrics"][metric] for a in approaches]
        
        # Handle inf values for time_to_success
        display_values = [v if v != float('inf') else 0 for v in values]
        
        if is_percentage:
            # Success rate doesn't have IQR, just show the value
            display_values = [v * 100 for v in display_values]
            yerr = None
        else:
            # Get Q1 and Q3 for asymmetric error bars
            q1s = [results[a]["metrics"].get(f"{metric}_q1", display_values[i]) for i, a in enumerate(approaches)]
            q3s = [results[a]["metrics"].get(f"{metric}_q3", display_values[i]) for i, a in enumerate(approaches)]
            # Handle inf values
            q1s = [v if v != float('inf') else 0 for v in q1s]
            q3s = [v if v != float('inf') else 0 for v in q3s]
            # Error bars: lower = value - Q1, upper = Q3 - value
            yerr = [[max(0, display_values[i] - q1s[i]) for i in range(len(approaches))],
                    [max(0, q3s[i] - display_values[i]) for i in range(len(approaches))]]
        
        x = np.arange(len(approaches))
        bars = ax.bar(x, display_values, yerr=yerr, capsize=5,
                      color=[colors[a] for a in approaches],
                      edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(approaches, rotation=15, ha='right')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, None)  # Ensure y-axis starts at 0
        
        # Add value labels on bars
        for i, (bar, val, orig_val) in enumerate(zip(bars, display_values, values)):
            height = bar.get_height()
            # Position label above the upper error bar
            if yerr is not None and max(display_values) > 0:
                label_y = height + yerr[1][i] + 0.01 * max(display_values)
            else:
                label_y = height + 2
            
            # Show "N/A" for inf values
            if orig_val == float('inf'):
                ax.text(bar.get_x() + bar.get_width()/2., label_y,
                       'N/A', ha='center', va='bottom', fontsize=10)
            elif is_percentage:
                ax.text(bar.get_x() + bar.get_width()/2., label_y,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., label_y,
                       f'{val:{fmt}}', ha='center', va='bottom', fontsize=10)
    
    # Hide the last (empty) subplot
    axes[-1].set_visible(False)
    
    plt.suptitle(f"Franka Push Comparison - Median + IQR (n={results['Policy']['metrics']['num_evals']} evals)", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(output_path, format='png', dpi=150, bbox_inches='tight')
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Bar plot saved to {output_path} and {pdf_path}")
    
    return fig


def generate_latex_table(results: dict, output_path: str):
    """Generate a LaTeX formatted table of all metrics.
    
    Uses median with IQR (Q1-Q3) for error bars.
    Metrics: Final position/orientation, total running cost, time to success, success rate.
    
    Args:
        results: Dictionary with results for each approach
        output_path: Path to save the .tex file
    """
    # Order: RL, CEM, Residual CEM
    approaches = ["Policy", "CEM", "Policy-guided CEM"]
    
    def fmt(val, q1=None, q3=None, precision=4):
        if val == float('inf'):
            return "N/A"
        if q1 is not None and q3 is not None and q1 != float('inf') and q3 != float('inf'):
            return f"${val:.{precision}f}$ ({q1:.{precision}f}-{q3:.{precision}f})"
        return f"${val:.{precision}f}$"
    
    def fmt_pct(val):
        return f"${val*100:.1f}\\%$"
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Franka Push Task Performance Comparison (Median with IQR)}",
        r"\label{tab:franka_push_results}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Metric & Policy & CEM & Policy-guided CEM \\",
        r"\midrule",
    ]
    
    # Final position error
    vals = [fmt(results[a]["metrics"]["final_dist"], 
               results[a]["metrics"].get("final_dist_q1"),
               results[a]["metrics"].get("final_dist_q3"), precision=3) 
            for a in approaches]
    lines.append(f"Final Position Error (m) & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    
    # Final orientation error
    vals = [fmt(results[a]["metrics"]["final_ori_error"], 
               results[a]["metrics"].get("final_ori_error_q1"),
               results[a]["metrics"].get("final_ori_error_q3"), precision=1) 
            for a in approaches]
    lines.append(f"Final Orientation Error ($^\\circ$) & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    
    lines.append(r"\midrule")
    
    # Total running cost
    vals = [fmt(results[a]["metrics"]["total_running_cost"], 
               results[a]["metrics"].get("total_running_cost_q1"),
               results[a]["metrics"].get("total_running_cost_q3"), precision=1) 
            for a in approaches]
    lines.append(f"Total Running Cost & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    
    # Time to success
    vals = [fmt(results[a]["metrics"]["time_to_success"], 
               results[a]["metrics"].get("time_to_success_q1"),
               results[a]["metrics"].get("time_to_success_q3"), precision=2) 
            for a in approaches]
    lines.append(f"Time to Success (s) & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    
    # Success rate
    vals = [fmt_pct(results[a]["metrics"]["success_rate"]) for a in approaches]
    lines.append(f"Success Rate & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1mm}",
        r"\footnotesize{Success = position error $<$ 3cm AND orientation error $<$ 10$^\circ$}",
        r"\end{table}",
    ])
    
    latex_content = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX table saved to {output_path}")
    print("\n--- LaTeX Table ---")
    print(latex_content)
    print("-------------------\n")
    
    return latex_content


def main():
    parser = argparse.ArgumentParser(description="Parallel Franka Push Experiment")
    parser.add_argument("--num_evals", type=int, default=16, 
                        help="Total number of evaluations (will be rounded up to multiple of num_envs)")
    parser.add_argument("--num_envs", type=int, default=4, 
                        help="Number of parallel environments per batch")
    parser.add_argument("--duration", type=float, default=8.0, 
                        help="Rollout duration in seconds")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Base random seed")
    args = parser.parse_args()
    
    # Round up num_evals to multiple of num_envs
    num_evals = math.ceil(args.num_evals / args.num_envs) * args.num_envs
    
    print(f"\n{'='*60}")
    print(f"Parallel Franka Push Experiment")
    print(f"  num_evals: {num_evals} (requested {args.num_evals})")
    print(f"  num_envs: {args.num_envs}")
    print(f"  duration: {args.duration}s")
    print(f"  seed: {args.seed}")
    print(f"{'='*60}\n")
    
    results = {}
    
    # Scenario 1: RL Policy Alone
    print("\n" + "="*50)
    print("Scenario 1: Policy Alone")
    print("="*50)
    task1 = FrankaPushGeometry(geometry="tblock", use_rl_policy=True)
    data1 = run_batched_experiment(
        task1, None,
        num_evals=num_evals,
        num_envs=args.num_envs, 
        base_seed=args.seed, 
        duration=args.duration
    )
    results["Policy"] = {"data": data1, "metrics": compute_metrics(data1)}
    
    # Scenario 2: CEM Only (run second for ordering)
    print("\n" + "="*50)
    print("Scenario 2: CEM Only (no policy)")
    print("="*50)
    task2 = FrankaPushGeometry(geometry="cube", use_rl_policy=False)
    ctrl2 = CEM(
        task=task2,
        num_samples=96,
        num_elites=8,
        sigma_start=0.1,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=6,
        seed=args.seed,
    )
    data2 = run_batched_experiment(
        task2, ctrl2,
        num_evals=num_evals,
        num_envs=args.num_envs, 
        base_seed=args.seed, 
        duration=args.duration
    )
    results["CEM"] = {"data": data2, "metrics": compute_metrics(data2)}
    
    # Scenario 3: Residual CEM
    print("\n" + "="*50)
    print("Scenario 3: Policy-guided CEM")
    print("="*50)
    task3 = FrankaPushGeometry(geometry="cube", use_rl_policy=True)
    ctrl3 = CEM(
        task=task3,
        num_samples=96,
        num_elites=8,
        sigma_start=0.1,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=6,
        seed=args.seed,
    )
    data3 = run_batched_experiment(
        task3, ctrl3,
        num_evals=num_evals,
        num_envs=args.num_envs, 
        base_seed=args.seed, 
        duration=args.duration
    )
    results["Policy-guided CEM"] = {"data": data3, "metrics": compute_metrics(data3)}
    
    # Order for display: Policy, CEM, Policy-guided CEM
    approaches = ["Policy", "CEM", "Policy-guided CEM"]
    
    # Print metrics table
    print("\n" + "="*110)
    print(f"{'Metric':<30} | {'Policy':<24} | {'CEM':<24} | {'Policy-guided CEM':<24}")
    print("-" * 110)
    
    def fmt_metric_iqr(val, q1=None, q3=None):
        if val == float('inf'):
            return "N/A"
        if q1 is not None and q3 is not None and q1 != float('inf') and q3 != float('inf'):
            return f"{val:.4f} ({q1:.4f}-{q3:.4f})"
        return f"{val:.4f}"
    
    # Final position error
    print(f"{'Final Position Error (m)':<30} | " + " | ".join(
        f"{fmt_metric_iqr(results[a]['metrics']['final_dist'], results[a]['metrics'].get('final_dist_q1'), results[a]['metrics'].get('final_dist_q3')):<24}"
        for a in approaches))
    
    # Final orientation error
    print(f"{'Final Ori Error (°)':<30} | " + " | ".join(
        f"{fmt_metric_iqr(results[a]['metrics']['final_ori_error'], results[a]['metrics'].get('final_ori_error_q1'), results[a]['metrics'].get('final_ori_error_q3')):<24}"
        for a in approaches))
    
    # Total running cost
    print(f"{'Total Running Cost':<30} | " + " | ".join(
        f"{fmt_metric_iqr(results[a]['metrics']['total_running_cost'], results[a]['metrics'].get('total_running_cost_q1'), results[a]['metrics'].get('total_running_cost_q3')):<24}"
        for a in approaches))
    
    # Time to success
    print(f"{'Time to Success (s)':<30} | " + " | ".join(
        f"{fmt_metric_iqr(results[a]['metrics']['time_to_success'], results[a]['metrics'].get('time_to_success_q1'), results[a]['metrics'].get('time_to_success_q3')):<24}"
        for a in approaches))
    
    # Success rate
    print(f"{'Success Rate':<30} | " + " | ".join(
        f"{results[a]['metrics']['success_rate']:<24.2%}"
        for a in approaches))
    
    print("="*110)
    print("Success criteria: position error < 3cm AND orientation error < 10°\n")
    
    # Generate LaTeX table
    generate_latex_table(results, "franka_push_results.tex")
    
    # Plotting
    print("Generating plots...")
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'lines.linewidth': 2.0,
    })
    
    colors = {"Policy": "#F48B96", "CEM": "#9ACD32", "Policy-guided CEM": "#90CCEB"}
    
    # Plot 1: Position error over time (individual figure)
    fig1, ax = plt.subplots(figsize=(10, 6))
    for name in approaches:
        data = results[name]["data"]
        times = np.array(data.time)
        dists = np.array(data.box_target_dist)
        
        mean = np.mean(dists, axis=0)
        std = np.std(dists, axis=0)
        
        ax.plot(times, mean, color=colors[name], lw=2.5, label=name)
        ax.fill_between(times, mean - std, mean + std, color=colors[name], alpha=0.2)
    
    ax.axhline(y=0.05, color='g', linestyle='--', label='Success (5cm)', alpha=0.7)
    ax.set_ylabel("Position Error (m)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Box-Target Distance (n={num_evals})")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig("franka_push_position_error.png", format='png', dpi=150, bbox_inches='tight')
    plt.savefig("franka_push_position_error.pdf", format='pdf', bbox_inches='tight')
    print("Position error plot saved")
    
    # Plot 2: Orientation error over time (individual figure)
    fig2, ax = plt.subplots(figsize=(10, 6))
    for name in approaches:
        data = results[name]["data"]
        times = np.array(data.time)
        ori_errors = np.array(data.box_ori_error) * 180 / np.pi
        
        mean = np.mean(ori_errors, axis=0)
        std = np.std(ori_errors, axis=0)
        
        ax.plot(times, mean, color=colors[name], lw=2.5, label=name)
        ax.fill_between(times, mean - std, mean + std, color=colors[name], alpha=0.2)
    
    ax.set_ylabel("Orientation Error (°)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Box Orientation Error (n={num_evals})")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig("franka_push_orientation_error.png", format='png', dpi=150, bbox_inches='tight')
    plt.savefig("franka_push_orientation_error.pdf", format='pdf', bbox_inches='tight')
    print("Orientation error plot saved")
    
    # Plot 3: Final error bar chart (individual figure)
    fig3, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(approaches))
    
    final_dists = [results[a]["metrics"]["final_dist"] for a in approaches]
    final_dist_q1 = [results[a]["metrics"]["final_dist_q1"] for a in approaches]
    final_dist_q3 = [results[a]["metrics"]["final_dist_q3"] for a in approaches]
    
    # Asymmetric error bars: [lower, upper] where lower = value - Q1, upper = Q3 - value
    yerr = [[final_dists[i] - final_dist_q1[i] for i in range(len(approaches))],
            [final_dist_q3[i] - final_dists[i] for i in range(len(approaches))]]
    
    bars = ax.bar(x, final_dists, yerr=yerr, capsize=5,
                  color=[colors[a] for a in approaches], edgecolor='black', linewidth=1.5)
    
    ax.axhline(y=0.05, color='g', linestyle='--', label='Success (5cm)', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.set_ylabel("Final Position Error (m)")
    ax.set_title(f"Final Error Comparison - Median + IQR (n={num_evals})")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, None)
    
    # Add value labels above upper error bar
    for i, (bar, val) in enumerate(zip(bars, final_dists)):
        label_y = val + yerr[1][i] + 0.005
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig("franka_push_final_error_bar.png", format='png', dpi=150, bbox_inches='tight')
    plt.savefig("franka_push_final_error_bar.pdf", format='pdf', bbox_inches='tight')
    print("Final error bar plot saved")
    
    # Plot 4: Full bar comparison (multiple metrics)
    plot_bar_comparison(results, "franka_push_metrics_comparison.png")
    
    plt.show()


if __name__ == "__main__":
    main()

