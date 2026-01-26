"""Domain Randomization Ablation Experiment for Franka push task.

This script investigates whether domain randomization (DR) provides tangible
benefits for the policy-guided CEM controller by comparing multiple DR/risk
strategy configurations under identical initial conditions.

Compares:
1. No DR (Baseline): Policy-guided CEM without domain randomization
2. DR + Average: DR with average cost aggregation
3. DR + CVaR: DR with Conditional Value-at-Risk (α=0.25)
4. DR + WorstCase: DR with worst-case (maximum) cost
5. DR + ExpWeight: DR with exponential weighted average (γ=1.0)

All variants use the same policy-guided CEM architecture with identical seeds
to ensure fair comparison.

Usage:
    python franka_push_dr_ablation.py --mode quick --num_envs 4 --num_evals 16
    python franka_push_dr_ablation.py --mode full --num_envs 4 --num_evals 32
"""

import argparse
import math
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

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
from hydrax.risk import (
    AverageCost,
    ConditionalValueAtRisk,
    WorstCase,
    ExponentialWeightedAverage,
    RiskStrategy,
)


@dataclass
class PhysicsPerturbation:
    """Physics perturbation configuration."""
    name: str
    mass_scale: float = 1.0
    friction_scale: float = 1.0


@dataclass
class ControllerVariant:
    """Configuration for a controller variant."""
    name: str
    num_randomizations: int
    risk_strategy: Optional[RiskStrategy]
    color: str
    use_cem: bool = True  # If False, just use RL policy (no CEM)
    
    def __repr__(self) -> str:
        if not self.use_cem:
            return f"{self.name} (RL only, no CEM)"
        dr_str = f"DR={self.num_randomizations}"
        risk_str = self.risk_strategy.__class__.__name__ if self.risk_strategy else "None"
        return f"{self.name} ({dr_str}, risk={risk_str})"


def get_controller_variants() -> List[ControllerVariant]:
    """Get all controller variants to compare."""
    return [
        # Baseline: RL policy only (no CEM)
        ControllerVariant(
            name="RL Only",
            num_randomizations=1,
            risk_strategy=None,
            color="#808080",  # Gray
            use_cem=False,
        ),
        # CEM without domain randomization
        ControllerVariant(
            name="CEM (No DR)",
            num_randomizations=1,
            risk_strategy=None,
            color="#1f77b4",  # Blue
            use_cem=True,
        ),
        ControllerVariant(
            name="DR + Average",
            num_randomizations=4,
            risk_strategy=AverageCost(),
            color="#ff7f0e",  # Orange
        ),
        ControllerVariant(
            name="DR + CVaR",
            num_randomizations=4,
            risk_strategy=ConditionalValueAtRisk(alpha=0.25),
            color="#2ca02c",  # Green
        ),
        ControllerVariant(
            name="DR + WorstCase",
            num_randomizations=4,
            risk_strategy=WorstCase(),
            color="#d62728",  # Red
        ),
        ControllerVariant(
            name="DR + ExpWeight",
            num_randomizations=4,
            risk_strategy=ExponentialWeightedAverage(gamma=1.0),
            color="#9467bd",  # Purple
        ),
    ]


def get_perturbations(mode: str) -> List[PhysicsPerturbation]:
    """Get perturbation conditions based on experiment mode."""
    if mode == "quick":
        return [
PhysicsPerturbation("slippery", 1.0, 0.3),
            PhysicsPerturbation("heavy_slippery", 2.0, 0.3),
        ]
    elif mode == "full":
        return [
            PhysicsPerturbation("nominal", 1.0, 1.0),
            PhysicsPerturbation("light", 0.5, 1.0),
            PhysicsPerturbation("heavy", 2.0, 1.0),
            PhysicsPerturbation("slippery", 1.0, 0.3),
            PhysicsPerturbation("heavy_slippery", 2.0, 0.3),
        ]
    else:
        return [PhysicsPerturbation("nominal", 1.0, 1.0)]


def apply_physics_perturbation(
    mj_model: mujoco.MjModel,
    perturbation: PhysicsPerturbation,
    task: FrankaPushGeometry,
) -> mjx.Model:
    """Apply physics perturbation to the MuJoCo model for SIMULATION ONLY.
    
    IMPORTANT: This returns a SEPARATE perturbed model for simulation.
    The task.model (used by CEM controller) remains UNCHANGED (nominal).
    This ensures the controller does NOT have privileged knowledge of the perturbation.
    
    Args:
        mj_model: MuJoCo model to copy and perturb
        perturbation: Perturbation configuration
        task: Task (used only to get body/geom IDs, NOT modified)
        
    Returns:
        mjx.Model: Perturbed model for simulation (NOT the controller's model)
    """
    import copy
    
    # Create a COPY of the model for simulation - don't modify the original!
    sim_mj_model = copy.deepcopy(mj_model)
    
    obj_body_id = task._obj_body
    obj_geom_id = task._obj_geom
    
    if perturbation.mass_scale != 1.0:
        sim_mj_model.body_mass[obj_body_id] *= perturbation.mass_scale
        sim_mj_model.body_inertia[obj_body_id] *= perturbation.mass_scale
    
    if perturbation.friction_scale != 1.0:
        sim_mj_model.geom_friction[obj_geom_id, 0] *= perturbation.friction_scale
    
    # Return perturbed mjx model for simulation
    # NOTE: task.model is NOT modified - controller keeps nominal model
    return mjx.put_model(sim_mj_model)


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


def create_parallel_step_fn(task: FrankaPushGeometry, sim_model: mjx.Model):
    """Create a parallel step function using the specified simulation model.
    
    Args:
        task: Task for apply_control and n_substeps
        sim_model: The mjx.Model to use for simulation (may be perturbed)
                   This is SEPARATE from task.model used by the CEM controller.
    """
    n_substeps = task.n_substeps
    
    def parallel_step(mjx_data: mjx.Data, ctrl_batch: jax.Array) -> mjx.Data:
        def step_env(data, ctrl):
            data = task.apply_control(data, ctrl)
            def single_step(d, _):
                return mjx.step(sim_model, d), None  # Use sim_model, not task.model!
            data = jax.lax.scan(single_step, data, None, n_substeps)[0]
            return data
        return jax.vmap(step_env)(mjx_data, ctrl_batch)
    
    return jax.jit(parallel_step)


def run_parallel_rollout(
    task: FrankaPushGeometry,
    controller: Optional[CEM],
    sim_model: mjx.Model,
    num_envs: int,
    base_seed: int = 42,
    duration: float = 5.0,
    frequency: float = 50.0,
) -> ParallelRolloutData:
    """Run parallel rollouts across multiple environments.
    
    Args:
        task: The task (controller uses task.model for planning - NOMINAL)
        controller: CEM controller, or None for RL-only baseline
        sim_model: Model for simulation (may be PERTURBED)
        num_envs: Number of parallel environments
        base_seed: Random seed
        duration: Rollout duration
        frequency: Control frequency
    """
    batch_reset = create_batched_reset(task, num_envs)
    parallel_step = create_parallel_step_fn(task, sim_model)  # Use sim_model for stepping!
    
    rng = jax.random.PRNGKey(base_seed)
    mjx_data = batch_reset(rng)
    
    target_pos = mjx_data.mocap_pos[:, 0, :]
    target_quat = mjx_data.mocap_quat[:, 0, :]
    
    replan_period = 1.0 / frequency
    sim_dt = task.dt
    sim_steps_per_replan = max(1, int(replan_period / sim_dt))
    num_replans = int(duration * frequency)
    
    # Setup for CEM controller (if provided)
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
            # CEM: optimize and get residual controls
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
            # RL-only: zero residuals (RL policy in apply_control handles everything)
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
    controller: Optional[CEM],
    sim_model: mjx.Model,
    num_evals: int,
    num_envs: int,
    base_seed: int,
    duration: float,
) -> ParallelRolloutData:
    """Run experiment in batches until num_evals is reached.
    
    Args:
        task: Task (controller uses task.model - NOMINAL)
        controller: CEM controller, or None for RL-only baseline
        sim_model: Model for simulation (may be PERTURBED)
        num_evals: Total evaluations
        num_envs: Parallel environments per batch
        base_seed: Random seed
        duration: Rollout duration
    """
    num_batches = math.ceil(num_evals / num_envs)
    
    all_data = []
    for batch_idx in range(num_batches):
        batch_seed = base_seed + batch_idx * num_envs * 1000
        data = run_parallel_rollout(
            task, controller, sim_model,
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


def create_controller(
    task: FrankaPushGeometry,
    variant: ControllerVariant,
) -> CEM:
    """Create a CEM controller for the given variant."""
    return CEM(
        task=task,
        num_samples=64,
        num_elites=16,
        sigma_start=0.1,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=6,
        num_randomizations=variant.num_randomizations,
        risk_strategy=variant.risk_strategy,
    )


def run_ablation_experiment(
    perturbations: List[PhysicsPerturbation],
    variants: List[ControllerVariant],
    num_evals: int,
    num_envs: int,
    duration: float,
    base_seed: int,
) -> Dict[str, Dict[str, Dict]]:
    """Run the full DR ablation experiment.
    
    IMPORTANT: The controller's internal model (task.model) remains at NOMINAL values.
    Only the simulation uses the perturbed model. This ensures NO privileged knowledge.
    
    Returns:
        Nested dict: results[perturbation_name][variant_name] = {"data": ..., "metrics": ...}
    """
    all_results = {}
    
    for perturbation in perturbations:
        jax.clear_caches()
        
        print(f"\n{'='*70}")
        print(f"Perturbation: {perturbation.name}")
        print(f"  Real env: mass={perturbation.mass_scale}x, friction={perturbation.friction_scale}x")
        print(f"  Controller model: NOMINAL (no privileged knowledge)")
        print(f"  Running {num_evals} evals in batches of {num_envs}")
        print('='*70)
        
        condition_results = {}
        
        for variant in variants:
            if variant.use_cem:
                print(f"\n  {variant.name} (CEM, DR={variant.num_randomizations})...", end=" ", flush=True)
            else:
                print(f"\n  {variant.name} (RL only, no CEM)...", end=" ", flush=True)
            start = time.time()
            
            # Create fresh task - task.model stays NOMINAL
            task = FrankaPushGeometry(geometry="cube", use_rl_policy=True)
            
            # Create SEPARATE perturbed model for SIMULATION ONLY
            # task.model is NOT modified - controller keeps nominal model
            sim_model = apply_physics_perturbation(task.mj_model, perturbation, task)
            
            # Create controller (or None for RL-only)
            if variant.use_cem:
                controller = create_controller(task, variant)
            else:
                controller = None  # RL-only baseline
            
            # Run experiment with PERTURBED sim_model for simulation
            # Controller plans with NOMINAL task.model (no privileged knowledge!)
            data = run_batched_experiment(
                task, controller, sim_model,
                num_evals=num_evals,
                num_envs=num_envs,
                base_seed=base_seed,  # Same seed for all variants!
                duration=duration,
            )
            
            metrics = compute_metrics(data)
            elapsed = time.time() - start
            
            print(f"final={metrics['final_dist']:.3f}m, success={metrics['success_rate']*100:.0f}% ({elapsed:.1f}s)")
            
            condition_results[variant.name] = {"data": data, "metrics": metrics}
        
        all_results[perturbation.name] = condition_results
    
    return all_results


def print_summary_table(results: Dict, variants: List[ControllerVariant]):
    """Print formatted summary table."""
    conditions = list(results.keys())
    variant_names = [v.name for v in variants]
    
    print("\n" + "="*130)
    print("SUMMARY: Final Distance (m) - Median (Q1-Q3)")
    print("="*130)
    
    # Header
    header = f"{'Condition':<20}"
    for name in variant_names:
        header += f" | {name:<20}"
    print(header)
    print("-"*130)
    
    for cond in conditions:
        row = f"{cond:<20}"
        for name in variant_names:
            m = results[cond][name]["metrics"]
            row += f" | {m['final_dist']:.3f} ({m['final_dist_q1']:.2f}-{m['final_dist_q3']:.2f})"
        print(row)
    
    print("="*130)
    
    # Success rate table
    print("\nSuccess Rate (final distance <5cm AND orientation <15°):")
    print("-"*100)
    header = f"{'Condition':<20}"
    for name in variant_names:
        header += f" | {name:<15}"
    print(header)
    print("-"*100)
    
    for cond in conditions:
        row = f"{cond:<20}"
        for name in variant_names:
            sr = results[cond][name]["metrics"]["success_rate"] * 100
            row += f" | {sr:>12.0f}%  "
        print(row)
    
    print("="*100)


def plot_ablation_results(
    results: Dict,
    variants: List[ControllerVariant],
    output_prefix: str = "dr_ablation",
):
    """Generate visualization plots."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 9,
    })
    
    conditions = list(results.keys())
    variant_names = [v.name for v in variants]
    colors = {v.name: v.color for v in variants}
    n_cond = len(conditions)
    n_variants = len(variants)
    
    # Bar chart of final distances with IQR error bars
    fig, ax = plt.subplots(figsize=(max(14, n_cond * 3), 7))
    
    x = np.arange(n_cond)
    width = 0.15
    
    for i, name in enumerate(variant_names):
        medians = [results[c][name]["metrics"]["final_dist"] for c in conditions]
        q1s = [results[c][name]["metrics"]["final_dist_q1"] for c in conditions]
        q3s = [results[c][name]["metrics"]["final_dist_q3"] for c in conditions]
        
        yerr = [[medians[j] - q1s[j] for j in range(n_cond)],
                [q3s[j] - medians[j] for j in range(n_cond)]]
        
        offset = (i - (n_variants - 1) / 2) * width
        ax.bar(x + offset, medians, width, yerr=yerr, 
               label=name, color=colors[name], capsize=2, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0.05, color='green', linestyle='--', label='Success (5cm)', alpha=0.7)
    
    ax.set_ylabel('Final Distance to Target (m)')
    ax.set_xlabel('Perturbation Condition')
    ax.set_title('DR Ablation: Comparing Risk Strategies\n(Median + IQR, same initial conditions)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_distance.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_distance.pdf", bbox_inches='tight')
    print(f"Saved {output_prefix}_distance.png/pdf")
    
    # Success rate bar chart
    fig, ax = plt.subplots(figsize=(max(14, n_cond * 3), 6))
    
    for i, name in enumerate(variant_names):
        success_rates = [results[c][name]["metrics"]["success_rate"] * 100 for c in conditions]
        
        offset = (i - (n_variants - 1) / 2) * width
        ax.bar(x + offset, success_rates, width,
               label=name, color=colors[name], edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Success Rate (%)')
    ax.set_xlabel('Perturbation Condition')
    ax.set_title('DR Ablation: Success Rate Comparison (dist<5cm, ori<15°)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_success.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_success.pdf", bbox_inches='tight')
    print(f"Saved {output_prefix}_success.png/pdf")
    
    # Heatmap showing relative performance
    fig, ax = plt.subplots(figsize=(12, 6))
    
    success_matrix = np.array([
        [results[c][v.name]["metrics"]["success_rate"] * 100 for c in conditions]
        for v in variants
    ])
    
    im = ax.imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(n_cond))
    ax.set_yticks(np.arange(n_variants))
    ax.set_xticklabels(conditions)
    ax.set_yticklabels(variant_names)
    
    # Add text annotations
    for i in range(n_variants):
        for j in range(n_cond):
            text = ax.text(j, i, f"{success_matrix[i, j]:.0f}%",
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('Success Rate Heatmap: DR Strategy vs Perturbation')
    ax.set_xlabel('Perturbation Condition')
    ax.set_ylabel('Controller Variant')
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Success Rate (%)', rotation=-90, va="bottom")
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_heatmap.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_heatmap.pdf", bbox_inches='tight')
    print(f"Saved {output_prefix}_heatmap.png/pdf")
    
    plt.show()


def generate_latex_table(
    results: Dict,
    variants: List[ControllerVariant],
    output_path: str,
):
    """Generate LaTeX table of results."""
    conditions = list(results.keys())
    variant_names = [v.name for v in variants]
    
    def fmt_dist(m):
        return f"${m['final_dist']:.3f}$"
    
    def fmt_sr(m):
        sr = m['success_rate'] * 100
        return f"${sr:.0f}\\%$"
    
    # Find best performer for each condition (highest success rate)
    def get_best_variant(cond):
        best_sr = -1
        best_name = None
        for name in variant_names:
            sr = results[cond][name]["metrics"]["success_rate"]
            if sr > best_sr:
                best_sr = sr
                best_name = name
        return best_name
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{DR Ablation: Comparing Risk Strategies (Same Initial Conditions)}",
        r"\label{tab:dr_ablation}",
        r"\begin{tabular}{l" + "c" * len(variant_names) + "}",
        r"\toprule",
        "Condition & " + " & ".join([n.replace("_", r"\_").replace("+", r"\texttt{+}") for n in variant_names]) + r" \\",
        r"\midrule",
    ]
    
    # Final distance section
    lines.append(r"\multicolumn{" + str(len(variant_names) + 1) + r"}{l}{\textit{Final Distance (m)}} \\")
    for cond in conditions:
        row = cond.replace("_", r"\_")
        for name in variant_names:
            m = results[cond][name]["metrics"]
            row += f" & {fmt_dist(m)}"
        row += r" \\"
        lines.append(row)
    
    lines.append(r"\midrule")
    
    # Success rate section
    lines.append(r"\multicolumn{" + str(len(variant_names) + 1) + r"}{l}{\textit{Success Rate (\%)}} \\")
    for cond in conditions:
        best = get_best_variant(cond)
        row = cond.replace("_", r"\_")
        for name in variant_names:
            m = results[cond][name]["metrics"]
            sr_str = fmt_sr(m)
            if name == best:
                sr_str = r"\textbf{" + sr_str + "}"
            row += f" & {sr_str}"
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
    """Run the DR ablation experiment."""
    parser = argparse.ArgumentParser(
        description="DR Ablation Experiment: Comparing Risk Strategies",
    )
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "full"],
                        help="Experiment mode")
    parser.add_argument("--num_evals", type=int, default=16,
                        help="Number of evaluations per condition")
    parser.add_argument("--num_envs", type=int, default=4,
                        help="Parallel environments per batch")
    parser.add_argument("--duration", type=float, default=8.0,
                        help="Rollout duration in seconds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (same for all variants)")
    parser.add_argument("--output", type=str, default="dr_ablation",
                        help="Output file prefix")
    args = parser.parse_args()
    
    perturbations = get_perturbations(args.mode)
    variants = get_controller_variants()
    
    print(f"\n{'='*70}")
    print("DOMAIN RANDOMIZATION ABLATION EXPERIMENT")
    print("Investigating whether DR provides tangible benefits")
    print(f"{'='*70}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Perturbation conditions: {len(perturbations)}")
    print(f"Controller variants: {len(variants)}")
    print(f"Evaluations per condition: {args.num_evals}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Duration: {args.duration}s per rollout")
    print(f"Base seed: {args.seed} (SAME for all variants)")
    print(f"{'='*70}")
    print("\nController variants:")
    for v in variants:
        print(f"  - {v}")
    print(f"{'='*70}")
    print("\nNOTE: All variants use IDENTICAL initial conditions (same seed)")
    print("      to ensure fair comparison of DR/risk strategy effects.")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    results = run_ablation_experiment(
        perturbations=perturbations,
        variants=variants,
        num_evals=args.num_evals,
        num_envs=args.num_envs,
        duration=args.duration,
        base_seed=args.seed,
    )
    elapsed = time.time() - start_time
    print(f"\nExperiment completed in {elapsed/60:.1f} minutes")
    
    # Print summary
    print_summary_table(results, variants)
    
    # Generate plots
    plot_ablation_results(results, variants, output_prefix=args.output)
    
    # Generate LaTeX table
    generate_latex_table(results, variants, f"{args.output}_table.tex")


if __name__ == "__main__":
    main()
