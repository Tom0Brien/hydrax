"""Parameter sweep experiment for DR risk strategy and distribution tuning.

This script performs a grid search over:
1. ExpWeight gamma values (risk aversion parameter)
2. DR distribution ranges for mass and friction

The goal is to find optimal DR parameters for the Franka push task,
particularly for the challenging heavy and slippery perturbations.

Usage:
    python franka_push_dr_sweep.py --mode quick --num_evals 16
    python franka_push_dr_sweep.py --mode full --num_evals 32
"""

import argparse
import copy
import math
import time
from dataclasses import dataclass, field
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
from hydrax.risk import ExponentialWeightedAverage


@dataclass
class PhysicsPerturbation:
    """Physics perturbation for the 'real' environment."""
    name: str
    mass_scale: float = 1.0
    friction_scale: float = 1.0


@dataclass
class DRConfig:
    """Configuration for domain randomization distribution."""
    name: str
    mass_range: Tuple[float, float] = (0.5, 2.0)
    friction_range: Tuple[float, float] = (0.3, 2.0)
    
    def __repr__(self) -> str:
        return f"{self.name} (mass={self.mass_range}, fric={self.friction_range})"


@dataclass
class SweepConfig:
    """Configuration for a single sweep point."""
    gamma: float
    dr_config: DRConfig
    num_randomizations: int = 4
    
    @property
    def name(self) -> str:
        return f"γ={self.gamma}, {self.dr_config.name}"


def get_gamma_values() -> List[float]:
    """Get gamma values to sweep over."""
    return [0.0, 0.5, 1.0, 2.0, 5.0]


def get_dr_configs() -> List[DRConfig]:
    """Get DR distribution configurations to sweep over."""
    return [
        DRConfig(
            name="narrow",
            mass_range=(0.75, 1.5),
            friction_range=(0.5, 1.5),
        ),
        DRConfig(
            name="default",
            mass_range=(0.5, 2.0),
            friction_range=(0.3, 2.0),
        ),
        DRConfig(
            name="wide",
            mass_range=(0.25, 4.0),
            friction_range=(0.1, 3.0),
        ),
        DRConfig(
            name="asymmetric_heavy",
            mass_range=(0.5, 4.0),  # Bias toward heavy
            friction_range=(0.3, 2.0),
        ),
        DRConfig(
            name="asymmetric_slippery",
            mass_range=(0.5, 2.0),
            friction_range=(0.1, 2.0),  # Bias toward slippery
        ),
    ]


def get_perturbations(mode: str) -> List[PhysicsPerturbation]:
    """Get perturbation conditions to test on."""
    if mode == "quick":
        return [
            PhysicsPerturbation("heavy", 2.0, 1.0),
            PhysicsPerturbation("slippery", 1.0, 0.3),
        ]
    elif mode == "full":
        return [
            PhysicsPerturbation("nominal", 1.0, 1.0),
            PhysicsPerturbation("heavy", 2.0, 1.0),
            PhysicsPerturbation("slippery", 1.0, 0.3),
            PhysicsPerturbation("heavy_slippery", 2.0, 0.3),
        ]
    else:
        return [PhysicsPerturbation("nominal", 1.0, 1.0)]


def get_sweep_configs(mode: str) -> List[SweepConfig]:
    """Generate sweep configurations."""
    
    if mode == "quick":
        # Quick mode: Fix gamma=1.0, sweep over all DR configs
        dr_configs = get_dr_configs()
        configs = [SweepConfig(gamma=1.0, dr_config=dr_config) for dr_config in dr_configs]
    else:
        # Full mode: Sweep both gamma and DR configs
        gammas = get_gamma_values()
        dr_configs = get_dr_configs()
        configs = []
        for gamma in gammas:
            for dr_config in dr_configs:
                configs.append(SweepConfig(gamma=gamma, dr_config=dr_config))
    
    return configs


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


def apply_physics_perturbation(
    mj_model: mujoco.MjModel,
    perturbation: PhysicsPerturbation,
    task: FrankaPushGeometry,
) -> mjx.Model:
    """Apply physics perturbation to create a separate simulation model."""
    sim_mj_model = copy.deepcopy(mj_model)
    
    obj_body_id = task._obj_body
    obj_geom_id = task._obj_geom
    
    if perturbation.mass_scale != 1.0:
        sim_mj_model.body_mass[obj_body_id] *= perturbation.mass_scale
        sim_mj_model.body_inertia[obj_body_id] *= perturbation.mass_scale
    
    if perturbation.friction_scale != 1.0:
        sim_mj_model.geom_friction[obj_geom_id, 0] *= perturbation.friction_scale
    
    return mjx.put_model(sim_mj_model)


def create_batched_reset(task: FrankaPushGeometry, num_envs: int):
    """Create a batched reset function for parallel environments."""
    @jax.jit
    def batch_reset(key: jax.Array) -> mjx.Data:
        keys = jax.random.split(key, num_envs)
        return jax.vmap(task.mjx_reset)(keys)
    return batch_reset


def create_parallel_step_fn(task: FrankaPushGeometry, sim_model: mjx.Model):
    """Create a parallel step function using the specified simulation model."""
    n_substeps = task.n_substeps
    
    def parallel_step(mjx_data: mjx.Data, ctrl_batch: jax.Array) -> mjx.Data:
        def step_env(data, ctrl):
            data = task.apply_control(data, ctrl)
            def single_step(d, _):
                return mjx.step(sim_model, d), None
            data = jax.lax.scan(single_step, data, None, n_substeps)[0]
            return data
        return jax.vmap(step_env)(mjx_data, ctrl_batch)
    
    return jax.jit(parallel_step)


def run_parallel_rollout(
    task: FrankaPushGeometry,
    controller: CEM,
    sim_model: mjx.Model,
    num_envs: int,
    base_seed: int = 42,
    duration: float = 5.0,
    frequency: float = 50.0,
) -> ParallelRolloutData:
    """Run parallel rollouts across multiple environments."""
    batch_reset = create_batched_reset(task, num_envs)
    parallel_step = create_parallel_step_fn(task, sim_model)
    
    rng = jax.random.PRNGKey(base_seed)
    mjx_data = batch_reset(rng)
    
    target_pos = mjx_data.mocap_pos[:, 0, :]
    target_quat = mjx_data.mocap_quat[:, 0, :]
    
    replan_period = 1.0 / frequency
    sim_dt = task.dt
    sim_steps_per_replan = max(1, int(replan_period / sim_dt))
    num_replans = int(duration * frequency)
    
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
    controller: CEM,
    sim_model: mjx.Model,
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
    """Compute metrics from parallel rollout data."""
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
    
    pos_threshold = 0.05
    ori_threshold = 15.0 * np.pi / 180
    
    success_per_env = (final_dist_per_env < pos_threshold) & (final_ori_per_env < ori_threshold)
    
    def median_iqr(arr):
        return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    
    final_dist_median, final_dist_q1, final_dist_q3 = median_iqr(final_dist_per_env)
    
    return {
        "final_dist": final_dist_median,
        "final_dist_q1": final_dist_q1,
        "final_dist_q3": final_dist_q3,
        "success_rate": float(np.mean(success_per_env)),
        "num_evals": num_envs,
    }


def create_controller(
    task: FrankaPushGeometry,
    sweep_config: SweepConfig,
) -> CEM:
    """Create a CEM controller for the given sweep configuration."""
    # Set DR ranges on task
    task.dr_mass_range = sweep_config.dr_config.mass_range
    task.dr_friction_range = sweep_config.dr_config.friction_range
    
    # Create risk strategy (gamma=0 means average cost)
    if sweep_config.gamma == 0.0:
        from hydrax.risk import AverageCost
        risk_strategy = AverageCost()
    else:
        risk_strategy = ExponentialWeightedAverage(gamma=sweep_config.gamma)
    
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
        num_randomizations=sweep_config.num_randomizations,
        risk_strategy=risk_strategy,
    )


def run_sweep_experiment(
    perturbations: List[PhysicsPerturbation],
    sweep_configs: List[SweepConfig],
    num_evals: int,
    num_envs: int,
    duration: float,
    base_seed: int,
) -> Dict[str, Dict[str, Dict]]:
    """Run the parameter sweep experiment."""
    all_results = {}
    
    for perturbation in perturbations:
        print(f"\n{'='*70}")
        print(f"Perturbation: {perturbation.name}")
        print(f"  Real env: mass={perturbation.mass_scale}x, friction={perturbation.friction_scale}x")
        print('='*70)
        
        condition_results = {}
        
        for config in sweep_configs:
            jax.clear_caches()
            
            print(f"\n  {config.name}...", end=" ", flush=True)
            start = time.time()
            
            # Create fresh task
            task = FrankaPushGeometry(geometry="cube", use_rl_policy=True)
            
            # Create perturbed simulation model
            sim_model = apply_physics_perturbation(task.mj_model, perturbation, task)
            
            # Create controller with sweep config
            controller = create_controller(task, config)
            
            # Run experiment
            data = run_batched_experiment(
                task, controller, sim_model,
                num_evals=num_evals,
                num_envs=num_envs,
                base_seed=base_seed,
                duration=duration,
            )
            
            metrics = compute_metrics(data)
            elapsed = time.time() - start
            
            print(f"success={metrics['success_rate']*100:.0f}% ({elapsed:.1f}s)")
            
            condition_results[config.name] = {"config": config, "metrics": metrics}
        
        all_results[perturbation.name] = condition_results
    
    return all_results


def print_sweep_results(results: Dict, sweep_configs: List[SweepConfig]):
    """Print formatted sweep results."""
    print("\n" + "="*100)
    print("PARAMETER SWEEP RESULTS: Success Rate (%)")
    print("="*100)
    
    conditions = list(results.keys())
    config_names = [c.name for c in sweep_configs]
    
    # Find best config for each perturbation
    print("\nBest configurations per perturbation:")
    print("-"*60)
    for cond in conditions:
        best_name = None
        best_sr = -1
        for name in config_names:
            if name in results[cond]:
                sr = results[cond][name]["metrics"]["success_rate"]
                if sr > best_sr:
                    best_sr = sr
                    best_name = name
        print(f"  {cond}: {best_name} ({best_sr*100:.0f}%)")
    
    # Detailed table
    print("\n" + "-"*100)
    for cond in conditions:
        print(f"\n{cond}:")
        for name in config_names:
            if name in results[cond]:
                sr = results[cond][name]["metrics"]["success_rate"] * 100
                dist = results[cond][name]["metrics"]["final_dist"]
                print(f"  {name:<40}: {sr:>5.0f}% (dist={dist:.3f}m)")


def plot_gamma_sweep(
    results: Dict,
    sweep_configs: List[SweepConfig],
    output_prefix: str = "dr_sweep",
):
    """Plot gamma sweep results."""
    conditions = list(results.keys())
    
    # Extract unique gamma values
    gammas = sorted(set(c.gamma for c in sweep_configs))
    
    # Group by DR config
    dr_configs = {}
    for config in sweep_configs:
        if config.dr_config.name not in dr_configs:
            dr_configs[config.dr_config.name] = []
        dr_configs[config.dr_config.name].append(config)
    
    # Plot gamma vs success rate for each perturbation
    fig, axes = plt.subplots(1, len(conditions), figsize=(5*len(conditions), 5))
    if len(conditions) == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(dr_configs)))
    
    for ax, cond in zip(axes, conditions):
        for (dr_name, dr_color) in zip(dr_configs.keys(), colors):
            gamma_vals = []
            success_rates = []
            
            for config in dr_configs[dr_name]:
                config_name = config.name
                if config_name in results[cond]:
                    gamma_vals.append(config.gamma)
                    success_rates.append(results[cond][config_name]["metrics"]["success_rate"] * 100)
            
            if gamma_vals:
                ax.plot(gamma_vals, success_rates, 'o-', color=dr_color, label=dr_name, markersize=8)
        
        ax.set_xlabel("Gamma (γ)")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title(f"Perturbation: {cond}")
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_gamma.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_gamma.pdf", bbox_inches='tight')
    print(f"Saved {output_prefix}_gamma.png/pdf")
    
    plt.show()


def plot_heatmap(
    results: Dict,
    sweep_configs: List[SweepConfig],
    output_prefix: str = "dr_sweep",
):
    """Plot heatmap of gamma vs DR config for each perturbation."""
    conditions = list(results.keys())
    
    gammas = sorted(set(c.gamma for c in sweep_configs))
    dr_names = sorted(set(c.dr_config.name for c in sweep_configs))
    
    for cond in conditions:
        # Build success rate matrix
        matrix = np.zeros((len(gammas), len(dr_names)))
        
        for i, gamma in enumerate(gammas):
            for j, dr_name in enumerate(dr_names):
                config_name = f"γ={gamma}, {dr_name}"
                if config_name in results[cond]:
                    matrix[i, j] = results[cond][config_name]["metrics"]["success_rate"] * 100
                else:
                    matrix[i, j] = np.nan
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        ax.set_xticks(np.arange(len(dr_names)))
        ax.set_yticks(np.arange(len(gammas)))
        ax.set_xticklabels(dr_names, rotation=45, ha='right')
        ax.set_yticklabels([f"γ={g}" for g in gammas])
        
        for i in range(len(gammas)):
            for j in range(len(dr_names)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.0f}%",
                           ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title(f'Success Rate: {cond} perturbation')
        ax.set_xlabel('DR Distribution')
        ax.set_ylabel('Risk Aversion (γ)')
        
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Success Rate (%)', rotation=-90, va="bottom")
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_heatmap_{cond}.png", dpi=150, bbox_inches='tight')
        print(f"Saved {output_prefix}_heatmap_{cond}.png")
    
    plt.show()


def main():
    """Run the parameter sweep experiment."""
    parser = argparse.ArgumentParser(
        description="DR Parameter Sweep: Tuning gamma and distribution ranges",
    )
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "full"],
                        help="Experiment mode")
    parser.add_argument("--num_evals", type=int, default=16,
                        help="Number of evaluations per config")
    parser.add_argument("--num_envs", type=int, default=4,
                        help="Parallel environments per batch")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Rollout duration in seconds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--output", type=str, default="dr_sweep",
                        help="Output file prefix")
    args = parser.parse_args()
    
    perturbations = get_perturbations(args.mode)
    sweep_configs = get_sweep_configs(args.mode)
    
    print(f"\n{'='*70}")
    print("DR PARAMETER SWEEP EXPERIMENT")
    print(f"{'='*70}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Perturbations: {[p.name for p in perturbations]}")
    print(f"Sweep configs: {len(sweep_configs)}")
    print(f"Evaluations per config: {args.num_evals}")
    print(f"Duration: {args.duration}s")
    print(f"{'='*70}")
    print("\nSweep configurations:")
    for cfg in sweep_configs:
        print(f"  - {cfg}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    results = run_sweep_experiment(
        perturbations=perturbations,
        sweep_configs=sweep_configs,
        num_evals=args.num_evals,
        num_envs=args.num_envs,
        duration=args.duration,
        base_seed=args.seed,
    )
    elapsed = time.time() - start_time
    print(f"\nExperiment completed in {elapsed/60:.1f} minutes")
    
    # Print results
    print_sweep_results(results, sweep_configs)
    
    # Generate plots
    plot_gamma_sweep(results, sweep_configs, output_prefix=args.output)
    
    if args.mode == "full":
        plot_heatmap(results, sweep_configs, output_prefix=args.output)


if __name__ == "__main__":
    main()
