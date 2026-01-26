"""Parallel experiment for Cheetah run task using JAX vectorization.

This script runs multiple environments in parallel using JAX vmap/scan,
leveraging MJX for GPU-accelerated simulation.

Compares:
1. Baseline policy (zero residuals)
2. Policy-guided CEM (CEM optimizing residuals around policy)

Usage:
    python cheetah_experiment.py --num_evals 16 --num_envs 4 --duration 5.0
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

import sys
import os
sys.path.append(os.getcwd())

from hydrax.tasks.cheetah import CheetahRun
from hydrax.algs.cem import CEM, CEMParams


class ParallelRolloutData(NamedTuple):
    """Data collected from parallel rollouts."""
    time: jax.Array  # (num_steps,)
    speed: jax.Array  # (num_envs, num_steps)
    position: jax.Array  # (num_envs, num_steps)
    qpos: jax.Array  # (num_envs, num_steps, nq)


def create_batched_reset(task: CheetahRun, num_envs: int):
    """Create a function to reset multiple environments in parallel."""
    @jax.jit
    def batch_reset(rng: jax.Array) -> mjx.Data:
        """Reset num_envs environments in parallel."""
        rngs = jax.random.split(rng, num_envs)
        return jax.vmap(task.mjx_reset)(rngs)
    
    return batch_reset


def create_parallel_step_fn(task: CheetahRun):
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
    task: CheetahRun,
    controller: CEM | None,
    num_envs: int,
    base_seed: int = 42,
    duration: float = 10.0,
    frequency: float = 100.0,
) -> ParallelRolloutData:
    """Run parallel rollouts across multiple environments.
    
    Args:
        task: CheetahRun task
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
    
    print(f"Reset complete. Starting rollout...")
    
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
    all_speeds = []
    all_positions = []
    all_qpos = []
    
    # Get sensor address for speed
    sensor_adr = task.mj_model.sensor_adr[task._torso_subtreelinvel_sensor]
    
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
            
            # Speed (x-velocity from sensor)
            speed = mjx_data.sensordata[:, sensor_adr]  # (num_envs,)
            all_speeds.append(speed)
            
            # Position (x-coordinate)
            position = mjx_data.qpos[:, 0]  # (num_envs,)
            all_positions.append(position)
            
            # Full qpos for reference
            all_qpos.append(mjx_data.qpos)
        
        if step % 20 == 0:
            print(f"Step {step}/{num_replans}", end="\r")
    
    elapsed = time.time() - start_time
    print(f"\nRollout complete. Time: {elapsed:.2f}s ({num_envs * duration / elapsed:.1f}x realtime)")
    
    return ParallelRolloutData(
        time=jnp.array(all_times),
        speed=jnp.stack(all_speeds, axis=1),
        position=jnp.stack(all_positions, axis=1),
        qpos=jnp.stack(all_qpos, axis=1),
    )


def run_batched_experiment(
    task: CheetahRun,
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
        speed=jnp.concatenate([d.speed for d in all_data], axis=0),
        position=jnp.concatenate([d.position for d in all_data], axis=0),
        qpos=jnp.concatenate([d.qpos for d in all_data], axis=0),
    )
    
    return combined


def compute_metrics(data: ParallelRolloutData, target_speed: float = 10.0) -> dict:
    """Compute metrics from parallel rollout data.
    
    Uses median and IQR (interquartile range) for robust statistics.
    """
    times = np.array(data.time)
    speeds = np.array(data.speed)  # (num_envs, num_steps)
    positions = np.array(data.position)  # (num_envs, num_steps)
    
    # Skip first 1s for transient/stabilization
    mask = times > 1.0
    if not np.any(mask):
        mask = np.ones_like(times, dtype=bool)
    
    num_envs = speeds.shape[0]
    
    # Speed metrics (per environment, then aggregate)
    mean_speed_per_env = np.mean(speeds[:, mask], axis=1)
    max_speed_per_env = np.max(speeds[:, mask], axis=1)
    final_speed_per_env = speeds[:, -1]
    
    # Position metrics (distance traveled)
    initial_pos = positions[:, 0]
    final_pos = positions[:, -1]
    distance_traveled = final_pos - initial_pos
    
    # Reward-like metric: how close to target speed (10 m/s)
    # Tolerance function: reward = 1 when speed >= target, linearly decreasing below
    def speed_reward(speed):
        return np.clip(speed / target_speed, 0.0, 1.0)
    
    mean_reward_per_env = np.mean(speed_reward(speeds[:, mask]), axis=1)
    
    # Helper for median + IQR stats
    def median_iqr(arr):
        median = float(np.median(arr))
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        return median, q1, q3
    
    # Speed metrics with median/IQR
    mean_speed_median, mean_speed_q1, mean_speed_q3 = median_iqr(mean_speed_per_env)
    max_speed_median, max_speed_q1, max_speed_q3 = median_iqr(max_speed_per_env)
    final_speed_median, final_speed_q1, final_speed_q3 = median_iqr(final_speed_per_env)
    
    # Distance metrics
    distance_median, distance_q1, distance_q3 = median_iqr(distance_traveled)
    
    # Reward metrics
    reward_median, reward_q1, reward_q3 = median_iqr(mean_reward_per_env)
    
    # Success: sustained speed > 8 m/s for the duration
    success_threshold = 0.8 * target_speed  # 80% of target
    success_per_env = mean_speed_per_env > success_threshold
    
    return {
        # Speed metrics (median + IQR)
        "mean_speed": mean_speed_median,
        "mean_speed_q1": mean_speed_q1,
        "mean_speed_q3": mean_speed_q3,
        "max_speed": max_speed_median,
        "max_speed_q1": max_speed_q1,
        "max_speed_q3": max_speed_q3,
        "final_speed": final_speed_median,
        "final_speed_q1": final_speed_q1,
        "final_speed_q3": final_speed_q3,
        
        # Distance metrics
        "distance": distance_median,
        "distance_q1": distance_q1,
        "distance_q3": distance_q3,
        
        # Reward metrics
        "mean_reward": reward_median,
        "mean_reward_q1": reward_q1,
        "mean_reward_q3": reward_q3,
        
        # Success metrics
        "success_rate": float(np.mean(success_per_env)),
        
        # Count
        "num_evals": num_envs,
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel Cheetah Run Experiment")
    parser.add_argument("--num_evals", type=int, default=16, 
                        help="Total number of evaluations (will be rounded up to multiple of num_envs)")
    parser.add_argument("--num_envs", type=int, default=4, 
                        help="Number of parallel environments per batch")
    parser.add_argument("--duration", type=float, default=5.0, 
                        help="Rollout duration in seconds")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Base random seed")
    args = parser.parse_args()
    
    # Round up num_evals to multiple of num_envs
    num_evals = math.ceil(args.num_evals / args.num_envs) * args.num_envs
    
    print(f"\n{'='*60}")
    print(f"Parallel Cheetah Run Experiment")
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
    task1 = CheetahRun(use_rl_policy=True)
    data1 = run_batched_experiment(
        task1, None,
        num_evals=num_evals,
        num_envs=args.num_envs, 
        base_seed=args.seed, 
        duration=args.duration
    )
    results["Policy"] = {"data": data1, "metrics": compute_metrics(data1)}
    
    # Scenario 2: Policy-guided CEM
    print("\n" + "="*50)
    print("Scenario 2: Policy-guided CEM")
    print("="*50)
    task2 = CheetahRun(use_rl_policy=True)
    ctrl2 = CEM(
        task=task2,
        num_samples=64,
        num_elites=8,
        sigma_start=0.2,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=0.3,
        spline_type="zero",
        num_knots=10,
    )
    data2 = run_batched_experiment(
        task2, ctrl2,
        num_evals=num_evals,
        num_envs=args.num_envs, 
        base_seed=args.seed, 
        duration=args.duration
    )
    results["Policy-guided CEM"] = {"data": data2, "metrics": compute_metrics(data2)}
    
    # Order for display
    approaches = ["Policy", "Policy-guided CEM"]
    
    # Print metrics table
    print("\n" + "="*80)
    print(f"{'Metric':<25} | {'Policy':<25} | {'Policy-guided CEM':<25}")
    print("-" * 80)
    
    def fmt_metric(val, q1=None, q3=None):
        if q1 is not None and q3 is not None:
            return f"{val:.2f} ({q1:.2f}-{q3:.2f})"
        return f"{val:.2f}"
    
    for metric in ["mean_speed", "max_speed", "final_speed"]:
        print(f"{metric + ' (m/s)':<25} | " + " | ".join(
            f"{fmt_metric(results[a]['metrics'][metric], results[a]['metrics'].get(metric+'_q1'), results[a]['metrics'].get(metric+'_q3')):<25}"
            for a in approaches))
    
    print(f"{'distance (m)':<25} | " + " | ".join(
        f"{fmt_metric(results[a]['metrics']['distance'], results[a]['metrics'].get('distance_q1'), results[a]['metrics'].get('distance_q3')):<25}"
        for a in approaches))
    
    print(f"{'mean_reward':<25} | " + " | ".join(
        f"{fmt_metric(results[a]['metrics']['mean_reward'], results[a]['metrics'].get('mean_reward_q1'), results[a]['metrics'].get('mean_reward_q3')):<25}"
        for a in approaches))
    
    print(f"{'success_rate':<25} | " + " | ".join(
        f"{results[a]['metrics']['success_rate']:<25.2%}"
        for a in approaches))
    
    print("="*80 + "\n")
    
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
    
    colors = {"Policy": "#F48B96", "Policy-guided CEM": "#90CCEB"}
    
    # Plot 1: Speed over time
    fig1, ax = plt.subplots(figsize=(10, 6))
    for name in approaches:
        data = results[name]["data"]
        times = np.array(data.time)
        speeds = np.array(data.speed)
        
        mean = np.mean(speeds, axis=0)
        std = np.std(speeds, axis=0)
        
        ax.plot(times, mean, color=colors[name], lw=2.5, label=name)
        ax.fill_between(times, mean - std, mean + std, color=colors[name], alpha=0.2)
    
    ax.axhline(y=10.0, color='g', linestyle='--', label='Target (10 m/s)', alpha=0.7)
    ax.set_ylabel("Speed (m/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Cheetah Running Speed (n={num_evals})")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig("cheetah_speed.png", format='png', dpi=150, bbox_inches='tight')
    plt.savefig("cheetah_speed.pdf", format='pdf', bbox_inches='tight')
    print("Speed plot saved to cheetah_speed.png/pdf")
    
    # Plot 2: Position over time
    fig2, ax = plt.subplots(figsize=(10, 6))
    for name in approaches:
        data = results[name]["data"]
        times = np.array(data.time)
        positions = np.array(data.position)
        
        mean = np.mean(positions, axis=0)
        std = np.std(positions, axis=0)
        
        ax.plot(times, mean, color=colors[name], lw=2.5, label=name)
        ax.fill_between(times, mean - std, mean + std, color=colors[name], alpha=0.2)
    
    ax.set_ylabel("Position (m)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Cheetah Distance Traveled (n={num_evals})")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cheetah_position.png", format='png', dpi=150, bbox_inches='tight')
    plt.savefig("cheetah_position.pdf", format='pdf', bbox_inches='tight')
    print("Position plot saved to cheetah_position.png/pdf")
    
    # Plot 3: Bar chart comparison
    fig3, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    metrics_to_plot = [
        ("mean_speed", "Mean Speed (m/s)"),
        ("distance", "Distance Traveled (m)"),
        ("mean_reward", "Mean Reward"),
    ]
    
    for ax, (metric, label) in zip(axes, metrics_to_plot):
        values = [results[a]["metrics"][metric] for a in approaches]
        q1s = [results[a]["metrics"].get(f"{metric}_q1", values[i]) for i, a in enumerate(approaches)]
        q3s = [results[a]["metrics"].get(f"{metric}_q3", values[i]) for i, a in enumerate(approaches)]
        
        yerr = [[values[i] - q1s[i] for i in range(len(approaches))],
                [q3s[i] - values[i] for i in range(len(approaches))]]
        
        x = np.arange(len(approaches))
        bars = ax.bar(x, values, yerr=yerr, capsize=5,
                      color=[colors[a] for a in approaches],
                      edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(approaches, rotation=15, ha='right')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, None)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            label_y = height + yerr[1][i] + 0.02 * max(values)
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(f"Cheetah Run Comparison (n={num_evals})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("cheetah_comparison.png", format='png', dpi=150, bbox_inches='tight')
    plt.savefig("cheetah_comparison.pdf", format='pdf', bbox_inches='tight')
    print("Comparison plot saved to cheetah_comparison.png/pdf")
    
    print("\nExperiment complete!")
    plt.show()


if __name__ == "__main__":
    main()
