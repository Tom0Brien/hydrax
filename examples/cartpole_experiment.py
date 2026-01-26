"""Parallel experiment for Cartpole swingup task using JAX vectorization.

This script runs multiple environments in parallel using JAX vmap/scan,
leveraging MJX for GPU-accelerated simulation.

Compares:
1. Baseline policy (zero residuals)
2. Policy-guided CEM (CEM optimizing residuals around policy)

Usage:
    python cartpole_experiment.py --num_evals 16 --num_envs 4 --duration 10.0
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

from hydrax.tasks.cartpole import CartpoleSwingup
from hydrax.algs.cem import CEM, CEMParams


class ParallelRolloutData(NamedTuple):
    """Data collected from parallel rollouts."""
    time: jax.Array  # (num_steps,)
    pole_angle_cos: jax.Array  # (num_envs, num_steps)
    cart_position: jax.Array  # (num_envs, num_steps)
    qpos: jax.Array  # (num_envs, num_steps, nq)


def create_batched_reset(task: CartpoleSwingup, num_envs: int):
    """Create a function to reset multiple environments in parallel."""
    @jax.jit
    def batch_reset(rng: jax.Array) -> mjx.Data:
        """Reset num_envs environments in parallel."""
        rngs = jax.random.split(rng, num_envs)
        return jax.vmap(task.mjx_reset)(rngs)
    
    return batch_reset


def create_parallel_step_fn(task: CartpoleSwingup):
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
    task: CartpoleSwingup,
    controller: CEM | None,
    num_envs: int,
    base_seed: int = 42,
    duration: float = 10.0,
    frequency: float = 100.0,
) -> ParallelRolloutData:
    """Run parallel rollouts across multiple environments."""
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
        
        # Warmup
        print("Warming up controller...")
        single_data = jax.tree.map(lambda x: x[0], mjx_data)
        single_params = jax.tree.map(lambda x: x[0], policy_params_batch)
        _, _ = jit_optimize(single_data, single_params)
        _, _ = jit_optimize(single_data, single_params)
        
        @jax.jit
        def batched_optimize(mjx_data_batch, params_batch):
            return jax.vmap(jit_optimize)(mjx_data_batch, params_batch)
    else:
        policy_params_batch = None
    
    # Data collection arrays
    all_times = []
    all_pole_angle_cos = []
    all_cart_positions = []
    all_qpos = []
    
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
            
            # Pole angle cos (zz component of pole body, index 2)
            pole_angle_cos = mjx_data.xmat[:, 2, 2, 2]
            all_pole_angle_cos.append(pole_angle_cos)
            
            # Cart position
            cart_position = mjx_data.qpos[:, 0]
            all_cart_positions.append(cart_position)
            
            # Qpos
            all_qpos.append(mjx_data.qpos)
        
        if step % 20 == 0:
            print(f"Step {step}/{num_replans}", end="\r")
    
    elapsed = time.time() - start_time
    print(f"\nRollout complete. Time: {elapsed:.2f}s ({num_envs * duration / elapsed:.1f}x realtime)")
    
    return ParallelRolloutData(
        time=jnp.array(all_times),
        pole_angle_cos=jnp.stack(all_pole_angle_cos, axis=1),
        cart_position=jnp.stack(all_cart_positions, axis=1),
        qpos=jnp.stack(all_qpos, axis=1),
    )


def run_batched_experiment(
    task: CartpoleSwingup,
    controller: CEM | None,
    num_evals: int,
    num_envs: int,
    base_seed: int,
    duration: float,
) -> ParallelRolloutData:
    """Run experiment in batches until num_evals is reached."""
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
    
    combined = ParallelRolloutData(
        time=all_data[0].time,
        pole_angle_cos=jnp.concatenate([d.pole_angle_cos for d in all_data], axis=0),
        cart_position=jnp.concatenate([d.cart_position for d in all_data], axis=0),
        qpos=jnp.concatenate([d.qpos for d in all_data], axis=0),
    )
    
    return combined


def compute_metrics(data: ParallelRolloutData) -> dict:
    """Compute metrics from parallel rollout data."""
    times = np.array(data.time)
    pole_angle_cos = np.array(data.pole_angle_cos)
    cart_positions = np.array(data.cart_position)
    
    # Skip first 1s for transient
    mask = times > 1.0
    if not np.any(mask):
        mask = np.ones_like(times, dtype=bool)
    
    num_envs = pole_angle_cos.shape[0]
    
    # Upright metrics: pole_angle_cos = 1 when upright
    upright_per_env = (pole_angle_cos[:, mask] + 1) / 2
    mean_upright_per_env = np.mean(upright_per_env, axis=1)
    max_upright_per_env = np.max(upright_per_env, axis=1)
    final_upright_per_env = (pole_angle_cos[:, -1] + 1) / 2
    
    # Cart centering metrics
    mean_cart_dist_per_env = np.mean(np.abs(cart_positions[:, mask]), axis=1)
    
    # Time to reach upright (pole_angle_cos > 0.9)
    def time_to_upright(cos_seq, threshold=0.9):
        within = cos_seq > threshold
        if np.any(within):
            return times[np.argmax(within)]
        return float('inf')
    
    time_to_reach = np.array([time_to_upright(pole_angle_cos[i]) for i in range(num_envs)])
    
    # Combined reward (simplified)
    centered = np.exp(-0.5 * (cart_positions[:, mask] / 2.0)**2)
    combined_reward = upright_per_env * (1 + centered) / 2
    mean_reward_per_env = np.mean(combined_reward, axis=1)
    
    # Helper for median + IQR stats
    def median_iqr(arr):
        median = float(np.median(arr))
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        return median, q1, q3
    
    # Upright metrics with median/IQR
    mean_upright_median, mean_upright_q1, mean_upright_q3 = median_iqr(mean_upright_per_env)
    max_upright_median, max_upright_q1, max_upright_q3 = median_iqr(max_upright_per_env)
    final_upright_median, final_upright_q1, final_upright_q3 = median_iqr(final_upright_per_env)
    
    # Reward metrics
    reward_median, reward_q1, reward_q3 = median_iqr(mean_reward_per_env)
    
    # Success: sustained upright (mean > 0.9) AND cart near center
    success_per_env = (mean_upright_per_env > 0.9) & (mean_cart_dist_per_env < 0.5)
    
    return {
        "mean_upright": mean_upright_median,
        "mean_upright_q1": mean_upright_q1,
        "mean_upright_q3": mean_upright_q3,
        "max_upright": max_upright_median,
        "max_upright_q1": max_upright_q1,
        "max_upright_q3": max_upright_q3,
        "final_upright": final_upright_median,
        "final_upright_q1": final_upright_q1,
        "final_upright_q3": final_upright_q3,
        "time_to_upright": float(np.median(time_to_reach[time_to_reach < float('inf')])) 
            if np.any(time_to_reach < float('inf')) else float('inf'),
        "mean_reward": reward_median,
        "mean_reward_q1": reward_q1,
        "mean_reward_q3": reward_q3,
        "success_rate": float(np.mean(success_per_env)),
        "num_evals": num_envs,
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel Cartpole Swingup Experiment")
    parser.add_argument("--num_evals", type=int, default=16)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    num_evals = math.ceil(args.num_evals / args.num_envs) * args.num_envs
    
    print(f"\n{'='*60}")
    print(f"Parallel Cartpole Swingup Experiment")
    print(f"  num_evals: {num_evals}")
    print(f"  num_envs: {args.num_envs}")
    print(f"  duration: {args.duration}s")
    print(f"  seed: {args.seed}")
    print(f"{'='*60}\n")
    
    results = {}
    
    # Scenario 1: RL Policy Alone
    print("\n" + "="*50)
    print("Scenario 1: Policy Alone")
    print("="*50)
    task1 = CartpoleSwingup(use_rl_policy=True)
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
    task2 = CartpoleSwingup(use_rl_policy=True)
    ctrl2 = CEM(
        task=task2,
        num_samples=128,
        num_elites=8,
        sigma_start=0.4,
        sigma_min=0.01,
        explore_fraction=0.5,
        plan_horizon=1.0,
        spline_type="cubic",
        num_knots=6,
        iterations=3,
    )
    data2 = run_batched_experiment(
        task2, ctrl2,
        num_evals=num_evals,
        num_envs=args.num_envs, 
        base_seed=args.seed, 
        duration=args.duration
    )
    results["Policy-guided CEM"] = {"data": data2, "metrics": compute_metrics(data2)}
    
    approaches = ["Policy", "Policy-guided CEM"]
    
    # Print metrics table
    print("\n" + "="*80)
    print(f"{'Metric':<25} | {'Policy':<25} | {'Policy-guided CEM':<25}")
    print("-" * 80)
    
    def fmt_metric(val, q1=None, q3=None):
        if val == float('inf'):
            return "N/A"
        if q1 is not None and q3 is not None:
            return f"{val:.3f} ({q1:.3f}-{q3:.3f})"
        return f"{val:.3f}"
    
    for metric in ["mean_upright", "max_upright", "final_upright"]:
        print(f"{metric:<25} | " + " | ".join(
            f"{fmt_metric(results[a]['metrics'][metric], results[a]['metrics'].get(metric+'_q1'), results[a]['metrics'].get(metric+'_q3')):<25}"
            for a in approaches))
    
    print(f"{'time_to_upright (s)':<25} | " + " | ".join(
        f"{fmt_metric(results[a]['metrics']['time_to_upright']):<25}"
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
        'lines.linewidth': 2.0,
    })
    
    colors = {"Policy": "#F48B96", "Policy-guided CEM": "#90CCEB"}
    
    # Plot 1: Pole angle cos over time
    fig1, ax = plt.subplots(figsize=(10, 6))
    for name in approaches:
        data = results[name]["data"]
        times = np.array(data.time)
        pole_cos = np.array(data.pole_angle_cos)
        
        mean = np.mean(pole_cos, axis=0)
        std = np.std(pole_cos, axis=0)
        
        ax.plot(times, mean, color=colors[name], lw=2.5, label=name)
        ax.fill_between(times, mean - std, mean + std, color=colors[name], alpha=0.2)
    
    ax.axhline(y=1.0, color='g', linestyle='--', label='Upright', alpha=0.7)
    ax.axhline(y=-1.0, color='r', linestyle='--', label='Hanging', alpha=0.7)
    ax.set_ylabel("Pole Angle Cosine")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Cartpole Pole Angle (n={num_evals})")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.2, 1.2)
    plt.tight_layout()
    plt.savefig("cartpole_angle.png", format='png', dpi=150, bbox_inches='tight')
    plt.savefig("cartpole_angle.pdf", format='pdf', bbox_inches='tight')
    print("Angle plot saved to cartpole_angle.png/pdf")
    
    # Plot 2: Cart position over time
    fig2, ax = plt.subplots(figsize=(10, 6))
    for name in approaches:
        data = results[name]["data"]
        times = np.array(data.time)
        cart_pos = np.array(data.cart_position)
        
        mean = np.mean(cart_pos, axis=0)
        std = np.std(cart_pos, axis=0)
        
        ax.plot(times, mean, color=colors[name], lw=2.5, label=name)
        ax.fill_between(times, mean - std, mean + std, color=colors[name], alpha=0.2)
    
    ax.axhline(y=0.0, color='g', linestyle='--', label='Center', alpha=0.7)
    ax.set_ylabel("Cart Position (m)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Cartpole Cart Position (n={num_evals})")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cartpole_position.png", format='png', dpi=150, bbox_inches='tight')
    plt.savefig("cartpole_position.pdf", format='pdf', bbox_inches='tight')
    print("Position plot saved to cartpole_position.png/pdf")
    
    # Plot 3: Bar chart comparison
    fig3, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    metrics_to_plot = [
        ("mean_upright", "Mean Upright"),
        ("final_upright", "Final Upright"),
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
        ax.set_ylim(0, 1.1)
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            label_y = height + yerr[1][i] + 0.02
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(f"Cartpole Swingup Comparison (n={num_evals})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("cartpole_comparison.png", format='png', dpi=150, bbox_inches='tight')
    plt.savefig("cartpole_comparison.pdf", format='pdf', bbox_inches='tight')
    print("Comparison plot saved to cartpole_comparison.png/pdf")
    
    print("\nExperiment complete!")
    plt.show()


if __name__ == "__main__":
    main()
