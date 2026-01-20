"""Domain Randomization experiment for Franka push task.

This script tests whether domain randomization (DR) in the SPC planner
(without privileged knowledge of the true dynamics) improves robustness
for the residual CEM + policy architecture.

Compares:
1. Policy: Pre-trained policy alone (zero residuals)
2. Policy-guided CEM: CEM with policy but NO domain randomization
3. Policy-guided CEM (DR): CEM with policy AND domain randomization

The CEM controllers use domain randomization over mass and friction,
without knowing the true perturbation applied to the real environment.

Usage:
    python franka_push_dr_experiment.py --mode quick --num_envs 4 --num_evals 8
    python franka_push_dr_experiment.py --mode full --num_envs 4 --num_evals 16
"""

import argparse
import math
import time
from dataclasses import dataclass
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

from hydrax.tasks.franka import FrankaPushGeometry
from hydrax.algs.cem import CEM
from hydrax.risk import ConditionalValueAtRisk


@dataclass
class PhysicsPerturbation:
    """Physics perturbation configuration."""
    name: str
    mass_scale: float = 1.0
    friction_scale: float = 1.0


def get_perturbations(mode: str) -> List[PhysicsPerturbation]:
    """Get perturbation conditions based on experiment mode."""
    if mode == "quick":
        return [
            PhysicsPerturbation("nominal", 1.0, 1.0),
            PhysicsPerturbation("heavy", 2.0, 1.0),
            PhysicsPerturbation("slippery", 1.0, 0.3),
        ]
    elif mode == "full":
        return [
            PhysicsPerturbation("nominal", 1.0, 1.0),
            PhysicsPerturbation("heavy", 2.0, 1.0),
            PhysicsPerturbation("light", 0.5, 1.0),
            PhysicsPerturbation("slippery", 1.0, 0.3),
            PhysicsPerturbation("sticky", 1.0, 2.0),
            PhysicsPerturbation("heavy_slippery", 2.0, 0.3),
            PhysicsPerturbation("light_sticky", 0.5, 2.0),
        ]
    else:
        return [PhysicsPerturbation("nominal", 1.0, 1.0)]


def apply_physics_perturbation(
    mj_model: mujoco.MjModel,
    perturbation: PhysicsPerturbation,
    task: FrankaPushGeometry,
) -> None:
    """Apply physics perturbation to the MuJoCo model.
    
    Modifies the model in-place and updates task.model (mjx.Model).
    """
    # Get object body and geom IDs
    obj_body_id = task._obj_body
    obj_geom_id = task._obj_geom
    
    # Apply mass scaling
    if perturbation.mass_scale != 1.0:
        mj_model.body_mass[obj_body_id] *= perturbation.mass_scale
        # Update inertia proportionally (assuming uniform density)
        mj_model.body_inertia[obj_body_id] *= perturbation.mass_scale
    
    # Apply friction scaling
    if perturbation.friction_scale != 1.0:
        mj_model.geom_friction[obj_geom_id, 0] *= perturbation.friction_scale
    
    # Recreate mjx.Model with updated parameters
    task.model = mjx.put_model(mj_model)


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
    q2_inv = q2.at[..., 1:].multiply(-1)
    
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2_inv[..., 0], q2_inv[..., 1], q2_inv[..., 2], q2_inv[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    sin_half_angle = jnp.sqrt(x**2 + y**2 + z**2)
    angle = 2.0 * jnp.arcsin(jnp.clip(sin_half_angle, 0.0, 1.0))
    return angle


def create_batched_reset(task: FrankaPushGeometry, num_envs: int):
    """Create a batched reset function for parallel environments."""
    @jax.jit
    def batch_reset(key: jax.Array) -> mjx.Data:
        keys = jax.random.split(key, num_envs)
        return jax.vmap(task.mjx_reset)(keys)
    return batch_reset


def create_parallel_step_fn(task: FrankaPushGeometry):
    """Create a parallel step function."""
    model = task.model
    n_substeps = task.n_substeps
    
    def parallel_step(mjx_data: mjx.Data, ctrl_batch: jax.Array) -> mjx.Data:
        def step_env(data, ctrl):
            data = task.apply_control(data, ctrl)
            def single_step(d, _):
                return mjx.step(model, d), None
            data = jax.lax.scan(single_step, data, None, n_substeps)[0]
            return data
        return jax.vmap(step_env)(mjx_data, ctrl_batch)
    
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
    
    rng = jax.random.PRNGKey(base_seed)
    mjx_data = batch_reset(rng)
    
    target_pos = mjx_data.mocap_pos[:, 0, :]
    target_quat = mjx_data.mocap_quat[:, 0, :]
    
    replan_period = 1.0 / frequency
    sim_dt = task.dt
    sim_steps_per_replan = max(1, int(replan_period / sim_dt))
    num_replans = int(duration * frequency)
    
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
    perturbation: PhysicsPerturbation,
    num_evals: int,
    num_envs: int,
    duration: float,
    base_seed: int,
    num_randomizations: int = 4,
) -> Dict[str, Dict]:
    """Run all three control modes for a single perturbation condition.
    
    Key difference from robustness experiment:
    - CEM uses domain randomization (num_randomizations > 1) 
    - CEM's internal model is NOT updated with the perturbation
    - Only the real simulation environment has the perturbation
    """
    jax.clear_caches()
    
    print(f"\n{'='*60}")
    print(f"Condition: {perturbation.name}")
    print(f"  Real env: mass={perturbation.mass_scale}x, friction={perturbation.friction_scale}x")
    print(f"  DR uses {num_randomizations} domain randomizations")
    print(f"  Running {num_evals} evals in batches of {num_envs}")
    print('='*60)
    
    modes = ["Policy", "Policy-guided CEM", "Policy-guided CEM (DR)"]
    results = {}
    
    for mode in modes:
        print(f"\n  {mode}...", end=" ", flush=True)
        start = time.time()
        
        # Create fresh task - all modes use the RL policy
        task = FrankaPushGeometry(geometry="cube", use_rl_policy=True)
        
        # Apply perturbation to the SIMULATION model (real environment)
        apply_physics_perturbation(task.mj_model, perturbation, task)
        
        # Create controller if needed
        use_cem = mode in ["Policy-guided CEM", "Policy-guided CEM (DR)"]
        if use_cem:
            # Determine number of randomizations based on mode
            n_rand = num_randomizations if mode == "Policy-guided CEM (DR)" else 1
            controller = CEM(
                task=task,
                num_samples=64,
                num_elites=16,
                sigma_start=0.1,
                sigma_min=0.05,
                explore_fraction=0.5,
                plan_horizon=0.5,
                spline_type="zero",
                num_knots=6,
                num_randomizations=n_rand,
                risk_strategy=ConditionalValueAtRisk(alpha=0.25) if n_rand > 1 else None,
            )
        else:
            controller = None
        
        # Run experiment
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


def run_dr_experiment(
    perturbations: List[PhysicsPerturbation],
    num_evals: int,
    num_envs: int,
    duration: float,
    base_seed: int,
    num_randomizations: int = 4,
) -> Dict[str, Dict[str, Dict]]:
    """Run the full domain randomization experiment."""
    all_results = {}
    
    for perturbation in perturbations:
        condition_results = run_condition_experiment(
            perturbation=perturbation,
            num_evals=num_evals,
            num_envs=num_envs,
            duration=duration,
            base_seed=base_seed,
            num_randomizations=num_randomizations,
        )
        all_results[perturbation.name] = condition_results
    
    return all_results


def print_summary_table(results: Dict):
    """Print formatted summary table."""
    conditions = list(results.keys())
    modes = ["Policy", "Policy-guided CEM", "Policy-guided CEM (DR)"]
    
    print("\n" + "="*110)
    print("SUMMARY: Final Distance (m) - Median (Q1-Q3)")
    print("="*110)
    
    header = f"{'Condition':<20}"
    for mode in modes:
        header += f" | {mode:<25}"
    print(header)
    print("-"*110)
    
    for cond in conditions:
        row = f"{cond:<20}"
        for mode in modes:
            m = results[cond][mode]["metrics"]
            row += f" | {m['final_dist']:.3f} ({m['final_dist_q1']:.3f}-{m['final_dist_q3']:.3f})  "
        print(row)
    
    print("="*110)
    
    # Success rate table
    print("\nSuccess Rate (final distance <5cm AND orientation <15°):")
    print("-"*80)
    header = f"{'Condition':<20}"
    for mode in modes:
        header += f" | {mode:<18}"
    print(header)
    print("-"*80)
    
    for cond in conditions:
        row = f"{cond:<20}"
        for mode in modes:
            sr = results[cond][mode]["metrics"]["success_rate"] * 100
            row += f" | {sr:>15.0f}%  "
        print(row)
    
    print("="*80)


def plot_dr_results(results: Dict, output_prefix: str = "dr_experiment"):
    """Generate visualization plots."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
    })
    
    colors = {"Policy": "#F48B96", "Policy-guided CEM": "#9ACD32", "Policy-guided CEM (DR)": "#90CCEB"}
    modes = ["Policy", "Policy-guided CEM", "Policy-guided CEM (DR)"]
    conditions = list(results.keys())
    n_cond = len(conditions)
    
    # Bar chart of final distances with IQR error bars
    fig, ax = plt.subplots(figsize=(max(12, n_cond * 1.5), 6))
    
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
    ax.set_xlabel('Perturbation Condition')
    ax.set_title('Domain Randomization Experiment: Effect of DR on Policy-guided CEM\n(Median + IQR)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_distance.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_distance.pdf", bbox_inches='tight')
    print(f"Saved {output_prefix}_distance.png/pdf")
    
    # Success rate bar chart
    fig, ax = plt.subplots(figsize=(max(12, n_cond * 1.5), 5))
    
    for i, mode in enumerate(modes):
        success_rates = [results[c][mode]["metrics"]["success_rate"] * 100 for c in conditions]
        
        offset = (i - 1) * width
        ax.bar(x + offset, success_rates, width,
               label=mode, color=colors[mode], edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Success Rate (%)')
    ax.set_xlabel('Perturbation Condition')
    ax.set_title('Success Rate (dist<5cm, ori<15°): Domain Randomization Experiment')
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
    modes = ["Policy", "Policy-guided CEM", "Policy-guided CEM (DR)"]
    
    def fmt(val, q1=None, q3=None, precision=3):
        if q1 is not None and q3 is not None:
            return f"${val:.{precision}f}$ ({q1:.{precision}f}-{q3:.{precision}f})"
        return f"${val:.{precision}f}$"
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Domain Randomization Experiment Results (Final Distance in m, Median with IQR)}",
        r"\label{tab:dr_results}",
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
    lines.append(r"\multicolumn{" + str(len(modes) + 1) + r"}{l}{\textit{Success Rate (\%)}} \\")
    
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
    """Run the domain randomization experiment."""
    parser = argparse.ArgumentParser(
        description="Domain Randomization Experiment: RL vs CEM-DR vs Residual CEM-DR",
    )
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "full"],
                        help="Experiment mode")
    parser.add_argument("--num_evals", type=int, default=8,
                        help="Number of evaluations per condition")
    parser.add_argument("--num_envs", type=int, default=4,
                        help="Parallel environments per batch")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Rollout duration in seconds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--num_randomizations", type=int, default=4,
                        help="Number of domain randomizations for CEM")
    parser.add_argument("--output", type=str, default="dr_experiment",
                        help="Output file prefix")
    args = parser.parse_args()
    
    perturbations = get_perturbations(args.mode)
    
    print(f"\n{'='*60}")
    print("DOMAIN RANDOMIZATION EXPERIMENT")
    print("Comparing: Policy vs Policy-guided CEM vs Policy-guided CEM (DR)")
    print(f"{'='*60}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Perturbation conditions: {len(perturbations)}")
    print(f"Evaluations per condition: {args.num_evals}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Domain randomizations: {args.num_randomizations}")
    print(f"Duration: {args.duration}s per rollout")
    print(f"{'='*60}")
    print("\nNOTE: CEM uses domain randomization WITHOUT privileged knowledge")
    print("      of the true perturbation applied to the real environment.")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    results = run_dr_experiment(
        perturbations=perturbations,
        num_evals=args.num_evals,
        num_envs=args.num_envs,
        duration=args.duration,
        base_seed=args.seed,
        num_randomizations=args.num_randomizations,
    )
    elapsed = time.time() - start_time
    print(f"\nExperiment completed in {elapsed/60:.1f} minutes")
    
    # Print summary
    print_summary_table(results)
    
    # Generate plots
    plot_dr_results(results, output_prefix=args.output)
    
    # Generate LaTeX table
    generate_latex_table(results, f"{args.output}_table.tex")


if __name__ == "__main__":
    main()
