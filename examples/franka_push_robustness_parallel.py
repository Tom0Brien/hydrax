"""Parallel robustness experiment for Franka push task.

This script tests the adaptation capability of different control approaches
under variations in object properties (mass, friction, geometry).

Uses JAX vmap/scan for GPU-accelerated parallel simulation within each condition.

Compares:
1. Policy: Pre-trained policy alone (zero residuals)
2. CEM: CEM controller without policy
3. Policy-guided CEM: CEM optimizing residuals around policy

Usage:
    python franka_push_robustness_parallel.py --mode quick --num_envs 4 --num_evals 8
    python franka_push_robustness_parallel.py --mode full --num_envs 4 --num_evals 16
"""

import argparse
import math
import time
from functools import partial
from typing import Any, Dict, List, NamedTuple, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import mjx

import sys
import os
sys.path.append(os.getcwd())

from hydrax.tasks.franka import (
    FrankaPushGeometry,
    PerturbationConfig,
    apply_perturbation,
    get_standard_perturbations,
    get_quick_perturbations,
    get_geometry_perturbations,
    get_full_perturbations,
)
from hydrax.algs.cem import CEM


class ParallelRolloutData(NamedTuple):
    """Data collected from parallel rollouts."""
    time: jax.Array
    box_pos: jax.Array
    box_quat: jax.Array
    box_target_dist: jax.Array
    box_ori_error: jax.Array
    gripper_pos: jax.Array
    target_pos: jax.Array
    target_quat: jax.Array


def quat_angle_error(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """Compute angle error between quaternion batches (wxyz format)."""
    q1_inv = q1.at[..., 1:].multiply(-1)
    q_diff = quat_multiply(q1, q1_inv)
    angle = 2.0 * jnp.arcsin(jnp.clip(jnp.linalg.norm(q_diff[..., 1:], axis=-1), 0, 1))
    return angle


def quat_multiply(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """Multiply two quaternions (wxyz format)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return jnp.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def create_batched_reset(task: FrankaPushGeometry, num_envs: int):
    """Create a batched reset function for parallel environments.
    
    Explicitly captures task.model to avoid JIT caching issues.
    """
    # Capture model explicitly
    model = task.model
    
    # We need to bind the reset method to use the specific model
    # Since mjx_reset uses self.model, we need to ensure it uses the *current* model
    # The safest way is to re-bind the method or pass the model explicitly if supported
    # FrankaPushGeometry.mjx_reset uses self.model internally.
    # To fix this properly, we should rely on the fact that 'task' is a fresh instance
    # for each condition in our new loop structure.
    # However, to be extra safe against JIT caching of the method itself:
    
    @jax.jit
    def batch_reset(key: jax.Array) -> mjx.Data:
        keys = jax.random.split(key, num_envs)
        # task.mjx_reset is an instance method, so it captures 'task'
        # Since we create a NEW task instance for each condition loop,
        # this should be safe provided we re-create this function each time.
        return jax.vmap(task.mjx_reset)(keys)
    return batch_reset


def create_parallel_step_fn(task: FrankaPushGeometry):
    """Create a parallel step function that applies control and steps all envs.
    
    Note: We explicitly capture task.model to ensure we use the correct
    (potentially perturbed) model, and avoid JIT caching issues.
    """
    # Capture current model explicitly
    model = task.model
    n_substeps = task.n_substeps
    
    def parallel_step(mjx_data: mjx.Data, ctrl_batch: jax.Array) -> mjx.Data:
        def step_env(data, ctrl):
            data = task.apply_control(data, ctrl)
            # Use explicit model reference instead of task.step
            def single_step(d, _):
                return mjx.step(model, d), None
            data = jax.lax.scan(single_step, data, None, n_substeps)[0]
            return data
        return jax.vmap(step_env)(mjx_data, ctrl_batch)
    
    # JIT compile with explicit model as static arg
    return jax.jit(parallel_step)


def run_parallel_rollout(
    task: FrankaPushGeometry,
    controller: CEM | None,
    num_envs: int,
    base_seed: int = 42,
    duration: float = 5.0,
    frequency: float = 50.0,
) -> ParallelRolloutData:
    """Run parallel rollouts across multiple environments."""
    batch_reset = create_batched_reset(task, num_envs)
    parallel_step = create_parallel_step_fn(task)
    
    # Reset all environments
    rng = jax.random.PRNGKey(base_seed)
    mjx_data = batch_reset(rng)
    
    # Get target positions and orientations
    target_pos = mjx_data.mocap_pos[:, 0, :]
    target_quat = mjx_data.mocap_quat[:, 0, :]
    
    # Timing setup
    replan_period = 1.0 / frequency
    sim_dt = task.dt
    sim_steps_per_replan = max(1, int(replan_period / sim_dt))
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
        single_data = jax.tree.map(lambda x: x[0], mjx_data)
        single_params = jax.tree.map(lambda x: x[0], policy_params_batch)
        _, _ = jit_optimize(single_data, single_params)
        
        @jax.jit
        def batched_optimize(mjx_data_batch, params_batch):
            return jax.vmap(jit_optimize)(mjx_data_batch, params_batch)
    else:
        policy_params_batch = None
    
    # Data collection
    all_times = []
    all_box_pos = []
    all_box_quat = []
    all_box_target_dist = []
    all_box_ori_error = []
    all_gripper_pos = []
    
    obj_body_id = task._obj_body
    gripper_site_id = task._gripper_site
    
    for step in range(num_replans):
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
        
        for i in range(sim_steps_per_replan):
            ctrl_batch = us_batch[:, i, :]
            mjx_data = parallel_step(mjx_data, ctrl_batch)
            
            all_times.append(float(mjx_data.time[0]))
            
            box_pos = mjx_data.xpos[:, obj_body_id, :]
            box_quat = mjx_data.xquat[:, obj_body_id, :]
            all_box_pos.append(box_pos)
            all_box_quat.append(box_quat)
            
            dist = jnp.linalg.norm(box_pos[:, :2] - target_pos[:, :2], axis=1)
            all_box_target_dist.append(dist)
            
            ori_error = quat_angle_error(box_quat, target_quat)
            all_box_ori_error.append(ori_error)
            
            gripper_pos = mjx_data.site_xpos[:, gripper_site_id, :]
            all_gripper_pos.append(gripper_pos)
    
    return ParallelRolloutData(
        time=jnp.array(all_times),
        box_pos=jnp.stack(all_box_pos, axis=1),
        box_quat=jnp.stack(all_box_quat, axis=1),
        box_target_dist=jnp.stack(all_box_target_dist, axis=1),
        box_ori_error=jnp.stack(all_box_ori_error, axis=1),
        gripper_pos=jnp.stack(all_gripper_pos, axis=1),
        target_pos=target_pos,
        target_quat=target_quat,
    )


def run_batched_experiment(
    task: FrankaPushGeometry,
    controller: CEM | None,
    num_evals: int,
    num_envs: int,
    base_seed: int,
    duration: float,
) -> ParallelRolloutData:
    """Run experiment in batches until num_evals is reached."""
    num_batches = math.ceil(num_evals / num_envs)
    actual_evals = num_batches * num_envs
    
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
        time=all_data[0].time,
        box_pos=jnp.concatenate([d.box_pos for d in all_data], axis=0),
        box_quat=jnp.concatenate([d.box_quat for d in all_data], axis=0),
        box_target_dist=jnp.concatenate([d.box_target_dist for d in all_data], axis=0),
        box_ori_error=jnp.concatenate([d.box_ori_error for d in all_data], axis=0),
        gripper_pos=jnp.concatenate([d.gripper_pos for d in all_data], axis=0),
        target_pos=jnp.concatenate([d.target_pos for d in all_data], axis=0),
        target_quat=jnp.concatenate([d.target_quat for d in all_data], axis=0),
    )
    
    return combined


def compute_metrics(data: ParallelRolloutData) -> dict:
    """Compute metrics from parallel rollout data using median + IQR."""
    times = np.array(data.time)
    dists = np.array(data.box_target_dist)
    ori_errors = np.array(data.box_ori_error)
    
    mask = times > 1.0
    if not np.any(mask):
        mask = np.ones_like(times, dtype=bool)
    
    num_envs = dists.shape[0]
    
    final_dist_per_env = dists[:, -1]
    min_dist_per_env = np.min(dists[:, mask], axis=1)
    final_ori_per_env = ori_errors[:, -1]
    
    # Success thresholds
    pos_threshold = 0.05  # 5cm position error
    ori_threshold = 15.0 * np.pi / 180  # 15 degrees orientation error
    
    # Success requires BOTH position AND orientation criteria
    success_per_env = (final_dist_per_env < pos_threshold) & (final_ori_per_env < ori_threshold)
    
    def median_iqr(arr):
        return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    
    final_dist_median, final_dist_q1, final_dist_q3 = median_iqr(final_dist_per_env)
    min_dist_median, min_dist_q1, min_dist_q3 = median_iqr(min_dist_per_env)
    final_ori_median, final_ori_q1, final_ori_q3 = median_iqr(final_ori_per_env * 180 / np.pi)
    
    return {
        "final_dist": final_dist_median,
        "final_dist_q1": final_dist_q1,
        "final_dist_q3": final_dist_q3,
        "min_dist": min_dist_median,
        "min_dist_q1": min_dist_q1,
        "min_dist_q3": min_dist_q3,
        "final_ori": final_ori_median,
        "final_ori_q1": final_ori_q1,
        "final_ori_q3": final_ori_q3,
        "success_rate": float(np.mean(success_per_env)),
        "num_evals": num_envs,
    }


def run_condition_experiment(
    perturbation: PerturbationConfig,
    num_evals: int,
    num_envs: int,
    duration: float,
    base_seed: int,
) -> Dict[str, Dict]:
    """Run all three control modes for a single condition.
    
    Args:
        perturbation: Perturbation config
        num_evals: Number of evaluations
        num_envs: Parallel environments per batch
        duration: Rollout duration
        base_seed: Random seed
        
    Returns:
        Dictionary with results for each mode
    """
    # Clear JAX caches to ensure we don't reuse compiled functions with old models
    jax.clear_caches()
    
    geometry = getattr(perturbation, 'geometry', 'cube')
    # Use perturbation name directly if it's descriptive, otherwise add geometry prefix
    condition_name = perturbation.name
    
    print(f"\n{'='*60}")
    print(f"Condition: {condition_name} (geometry={geometry})")
    print(f"  Mass: {perturbation.mass_scale}x, Friction: {perturbation.friction_scale}x")
    print(f"  Running {num_evals} evals in batches of {num_envs}")
    print('='*60)
    
    modes = ["Policy", "CEM", "Policy-guided CEM"]
    results = {}
    
    for mode in modes:
        print(f"\n  {mode}...", end=" ", flush=True)
        start = time.time()
        
        use_rl = mode in ["Policy", "Policy-guided CEM"]
        task = FrankaPushGeometry(geometry=geometry, use_rl_policy=use_rl)
        
        # Apply perturbation (modifies task.mj_model and task.model)
        apply_perturbation(task.mj_model, perturbation, task=task)
        
        use_cem = mode in ["CEM", "Policy-guided CEM"]
        if use_cem:
            controller = CEM(
                task=task,
                num_samples=128,
                num_elites=16,
                sigma_start=0.1,
                sigma_min=0.05,
                explore_fraction=0.5,
                plan_horizon=0.5,
                spline_type="zero",
                num_knots=6,
            )
        else:
            controller = None
        
        # Run batched experiment
        data = run_batched_experiment(
            task, controller,
            num_evals=num_evals,
            num_envs=num_envs,
            base_seed=base_seed,
            duration=duration,
        )
        
        metrics = compute_metrics(data)
        elapsed = time.time() - start
        
        print(f"final={metrics['final_dist']:.3f}m, success={metrics['success_rate']*100:.0f}% ({elapsed:.1f}s)")
        
        results[mode] = {"data": data, "metrics": metrics}
    
    return results


def run_robustness_experiment(
    perturbations: List[PerturbationConfig],
    num_evals: int,
    num_envs: int,
    duration: float,
    base_seed: int,
) -> Dict[str, Dict[str, Dict]]:
    """Run the full robustness experiment.
    
    Returns:
        Nested dict: results[condition_name][mode] = {"data": ..., "metrics": ...}
    """
    all_results = {}
    
    for perturbation in perturbations:
        # Use perturbation name directly
        condition_name = perturbation.name
        
        condition_results = run_condition_experiment(
            perturbation=perturbation,
            num_evals=num_evals,
            num_envs=num_envs,
            duration=duration,
            base_seed=base_seed,
        )
        
        all_results[condition_name] = condition_results
    
    return all_results


def print_summary_table(results: Dict):
    """Print formatted summary table."""
    conditions = list(results.keys())
    modes = ["Policy", "CEM", "Policy-guided CEM"]
    
    print("\n" + "="*110)
    print("SUMMARY: Final Distance (m) - Median (Q1-Q3)")
    print("="*110)
    
    header = f"{'Condition':<25}"
    for mode in modes:
        header += f" | {mode:<25}"
    print(header)
    print("-"*110)
    
    for cond in conditions:
        row = f"{cond:<25}"
        for mode in modes:
            m = results[cond][mode]["metrics"]
            row += f" | {m['final_dist']:.3f} ({m['final_dist_q1']:.3f}-{m['final_dist_q3']:.3f})  "
        print(row)
    
    print("="*110)
    
    # Success rate table
    print("\nSuccess Rate (dist<5cm AND ori<15°):")
    print("-"*80)
    header = f"{'Condition':<25}"
    for mode in modes:
        header += f" | {mode:<15}"
    print(header)
    print("-"*80)
    
    for cond in conditions:
        row = f"{cond:<25}"
        for mode in modes:
            sr = results[cond][mode]["metrics"]["success_rate"] * 100
            row += f" | {sr:>12.0f}%  "
        print(row)
    
    print("="*80)


def plot_robustness_results(results: Dict, output_prefix: str = "robustness"):
    """Generate visualization plots."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
    })
    
    colors = {"Policy": "#F48B96", "CEM": "#9ACD32", "Policy-guided CEM": "#90CCEB"}
    modes = ["Policy", "CEM", "Policy-guided CEM"]
    conditions = list(results.keys())
    n_cond = len(conditions)
    
    # Bar chart of final distances with IQR error bars
    fig, ax = plt.subplots(figsize=(max(12, n_cond * 1.2), 6))
    
    x = np.arange(n_cond)
    width = 0.25
    
    for i, mode in enumerate(modes):
        medians = [results[c][mode]["metrics"]["final_dist"] for c in conditions]
        q1s = [results[c][mode]["metrics"]["final_dist_q1"] for c in conditions]
        q3s = [results[c][mode]["metrics"]["final_dist_q3"] for c in conditions]
        
        # Asymmetric error bars
        yerr = [[medians[j] - q1s[j] for j in range(n_cond)],
                [q3s[j] - medians[j] for j in range(n_cond)]]
        
        offset = (i - 1) * width
        ax.bar(x + offset, medians, width, yerr=yerr, 
               label=mode, color=colors[mode], capsize=3, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0.05, color='green', linestyle='--', label='Success (5cm)', alpha=0.7)
    
    ax.set_ylabel('Final Distance to Target (m)')
    ax.set_xlabel('Condition')
    ax.set_title('Robustness Comparison: Policy vs CEM vs Policy-guided CEM\n(Median + IQR)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_bar.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_bar.pdf", bbox_inches='tight')
    print(f"Saved {output_prefix}_bar.png/pdf")
    
    # Success rate bar chart
    fig, ax = plt.subplots(figsize=(max(12, n_cond * 1.2), 5))
    
    for i, mode in enumerate(modes):
        success_rates = [results[c][mode]["metrics"]["success_rate"] * 100 for c in conditions]
        
        offset = (i - 1) * width
        ax.bar(x + offset, success_rates, width,
               label=mode, color=colors[mode], edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Success Rate (%)')
    ax.set_xlabel('Condition')
    ax.set_title('Success Rate (dist<5cm, ori<15°)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_success.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_success.pdf", bbox_inches='tight')
    print(f"Saved {output_prefix}_success.png/pdf")
    
    plt.show()


def generate_latex_table(results: Dict, output_path: str):
    """Generate LaTeX table of results."""
    conditions = list(results.keys())
    modes = ["Policy", "CEM", "Policy-guided CEM"]
    
    def fmt(val, q1=None, q3=None, precision=3):
        if q1 is not None and q3 is not None:
            return f"${val:.{precision}f}$ ({q1:.{precision}f}-{q3:.{precision}f})"
        return f"${val:.{precision}f}$"
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Robustness Experiment Results (Final Distance in m, Median with IQR)}",
        r"\label{tab:robustness_results}",
        r"\begin{tabular}{l" + "c" * len(modes) + "}",
        r"\toprule",
        "Condition & " + " & ".join(modes) + r" \\",
        r"\midrule",
    ]
    
    for cond in conditions:
        row = cond.replace("_", r"\_")
        for mode in modes:
            m = results[cond][mode]["metrics"]
            row += f" & {fmt(m['final_dist'], m['final_dist_q1'], m['final_dist_q3'])}"
        row += r" \\"
        lines.append(row)
    
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(len(modes) + 1) + r"}{l}{\textit{Success Rate (\%, dist<5cm, ori<15°)}} \\")
    
    for cond in conditions:
        row = cond.replace("_", r"\_")
        for mode in modes:
            sr = results[cond][mode]["metrics"]["success_rate"] * 100
            row += f" & ${sr:.0f}\\%$"
        row += r" \\"
        lines.append(row)
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"LaTeX table saved to {output_path}")


def main():
    """Run the robustness experiment."""
    parser = argparse.ArgumentParser(
        description="Parallel robustness experiment: RL vs CEM vs Residual CEM",
    )
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "physics", "geometry", "full"],
                        help="Experiment mode")
    parser.add_argument("--num_evals", type=int, default=8,
                        help="Number of evaluations per condition")
    parser.add_argument("--num_envs", type=int, default=4,
                        help="Parallel environments per batch")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Rollout duration in seconds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--output", type=str, default="robustness",
                        help="Output file prefix")
    args = parser.parse_args()
    
    # Select perturbations
    if args.mode == "full":
        perturbations = get_full_perturbations()
        mode_name = "FULL (8 conditions)"
    elif args.mode == "geometry":
        perturbations = get_geometry_perturbations()
        mode_name = "GEOMETRY ONLY"
    elif args.mode == "physics":
        perturbations = get_standard_perturbations()
        mode_name = "PHYSICS ONLY"
    else:
        perturbations = get_quick_perturbations()
        mode_name = "QUICK TEST"
    
    print(f"\n{'='*60}")
    print("ROBUSTNESS EXPERIMENT: Policy vs CEM vs Policy-guided CEM")
    print(f"Mode: {mode_name}")
    print(f"Conditions: {len(perturbations)}")
    print(f"Evaluations per condition: {args.num_evals}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Duration: {args.duration}s per rollout")
    print(f"{'='*60}")
    
    # Run experiment
    start_time = time.time()
    results = run_robustness_experiment(
        perturbations=perturbations,
        num_evals=args.num_evals,
        num_envs=args.num_envs,
        duration=args.duration,
        base_seed=args.seed,
    )
    elapsed = time.time() - start_time
    print(f"\nExperiment completed in {elapsed/60:.1f} minutes")
    
    # Print summary
    print_summary_table(results)
    
    # Generate plots
    plot_robustness_results(results, output_prefix=args.output)
    
    # Generate LaTeX table
    generate_latex_table(results, f"{args.output}_table.tex")


if __name__ == "__main__":
    main()
