"""Unified robustness experiment: RL vs RL+SPC under physics and geometry variations.

This experiment compares the robustness of:
1. RL policy alone (zero residuals)
2. RL + SPC (CEM optimizing residuals)

Under various perturbations:
- Physics: Mass variations, friction variations
- Geometry: Cube (training), Square, T-block

The hypothesis is that SPC can adapt online to compensate for
model mismatch, improving robustness compared to RL alone.
"""

import time
import argparse
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

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


@dataclass
class ExperimentConfig:
    """Configuration for the robustness experiment."""
    duration: float = 8.0       # seconds per rollout
    frequency: float = 50.0     # control frequency Hz
    num_seeds: int = 3          # number of random seeds per condition
    success_threshold: float = 0.05  # meters (5cm)


def sync_mj_to_mjx(mjx_data: mjx.Data, mj_data: mujoco.MjData) -> mjx.Data:
    """Sync mujoco.MjData state to mjx.Data without contact count issues."""
    return mjx_data.replace(
        qpos=jnp.array(mj_data.qpos),
        qvel=jnp.array(mj_data.qvel),
        ctrl=jnp.array(mj_data.ctrl),
        mocap_pos=jnp.array(mj_data.mocap_pos),
        mocap_quat=jnp.array(mj_data.mocap_quat),
        time=jnp.array(mj_data.time),
    )


def run_single_rollout(
    geometry: str,
    perturbation: PerturbationConfig,
    reset_seed: int,
    config: ExperimentConfig,
    use_spc: bool = False,
) -> Dict[str, Any]:
    """Run a single rollout with specified geometry and perturbation.
    
    Args:
        geometry: Object geometry ("cube", "square", "tblock")
        perturbation: Physics perturbation config
        reset_seed: Random seed for environment reset
        config: Experiment configuration
        use_spc: If True, use RL+SPC; if False, RL only
        
    Returns:
        Dictionary with rollout data and metrics
    """
    mode = "RL+SPC" if use_spc else "RL only"
    print(f"    {geometry}/{perturbation.name} ({mode})...", end=" ", flush=True)
    
    # Create task with specified geometry
    task = FrankaPushGeometry(geometry=geometry, use_rl_policy=True)
    mj_model = task.mj_model
    
    # Use unified reset
    rng = jax.random.PRNGKey(reset_seed)
    mj_data, mjx_data = task.reset(rng)
    
    # Apply physics perturbation (must happen before creating controller)
    apply_perturbation(mj_model, perturbation, task=task)
    
    # Re-run forward after perturbation
    mujoco.mj_forward(mj_model, mj_data)
    
    target_pos = np.array(mj_data.mocap_pos[0])
    
    # Create controller if using SPC
    if use_spc:
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
    
    # Setup timing
    replan_period = 1.0 / config.frequency
    sim_dt = mj_model.opt.timestep
    sim_steps_per_replan = max(1, int(replan_period / sim_dt))
    
    # Update mjx_data from current mj_data state
    mjx_data = sync_mj_to_mjx(mjx_data, mj_data)
    
    if controller is not None:
        policy_params = controller.init_params(initial_knots=None, seed=reset_seed)
        jit_optimize = jax.jit(controller.optimize)
        jit_interp_func = jax.jit(controller.interp_func)
        
        # Warmup
        policy_params, _ = jit_optimize(mjx_data, policy_params)
        policy_params, _ = jit_optimize(mjx_data, policy_params)
    else:
        policy_params = None
        jit_optimize = None
        jit_interp_func = None
    
    # Data logging
    times = []
    box_target_distances = []
    
    obj_body_id = task._obj_body
    n_substeps = getattr(task, "n_substeps", 1)
    num_replans = int(config.duration * config.frequency)
    
    # Main loop
    for step in range(num_replans):
        if policy_params is not None:
            mjx_data = sync_mj_to_mjx(mjx_data, mj_data)
            policy_params, _ = jit_optimize(mjx_data, policy_params)
            
            t_curr = mj_data.time
            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            tk = policy_params.tk
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]
        else:
            us = np.zeros((sim_steps_per_replan, task.nu))
        
        for i in range(sim_steps_per_replan):
            should_update_ctrl = (n_substeps == 1) or (i % n_substeps == 0)
            
            if should_update_ctrl:
                mjx_data = sync_mj_to_mjx(mjx_data, mj_data)
                ctrl_input = jnp.array(us[i])
                mjx_data = task.apply_control(mjx_data, ctrl_input)
                mj_data.ctrl[:] = np.array(mjx_data.ctrl)
            
            mujoco.mj_step(mj_model, mj_data)
            
            times.append(mj_data.time)
            box_pos = mj_data.xpos[obj_body_id]
            dist = np.linalg.norm(box_pos[:2] - target_pos[:2])
            box_target_distances.append(dist)
    
    # Compute metrics
    times = np.array(times)
    distances = np.array(box_target_distances)
    
    mask = times > 1.0
    if not np.any(mask):
        mask = np.ones_like(times, dtype=bool)
    
    metrics = {
        "mean_dist": np.mean(distances[mask]),
        "final_dist": distances[-1],
        "min_dist": np.min(distances[mask]),
    }
    
    within_threshold = distances < config.success_threshold
    if np.any(within_threshold):
        metrics["time_to_success"] = times[np.argmax(within_threshold)]
        metrics["success"] = True
    else:
        metrics["time_to_success"] = float('inf')
        metrics["success"] = False
    
    print(f"final={metrics['final_dist']:.3f}m")
    
    return {
        "time": times,
        "distances": distances,
        "target_pos": target_pos,
        "metrics": metrics,
        "geometry": geometry,
        "perturbation": perturbation.name,
        "seed": reset_seed,
    }


def run_experiment(
    perturbations: List[PerturbationConfig],
    config: ExperimentConfig,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Dict[str, List[Dict]]]:
    """Run the full robustness experiment.
    
    Args:
        perturbations: List of perturbation configs (may include geometry variants)
        config: Experiment configuration
        seeds: List of random seeds
        
    Returns:
        Nested dict: results[method][condition_name] = list of rollout results
    """
    if seeds is None:
        seeds = list(range(42, 42 + config.num_seeds))
    
    results = {"RL": {}, "RL+SPC": {}}
    
    for perturbation in perturbations:
        # Use geometry from perturbation config, default to "cube"
        geometry = getattr(perturbation, 'geometry', 'cube')
        condition_name = f"{geometry}_{perturbation.name}"
        
        print(f"\n{'='*60}")
        print(f"Condition: {condition_name}")
        print(f"  Geometry: {geometry}")
        print(f"  Mass scale: {perturbation.mass_scale}x")
        print(f"  Friction scale: {perturbation.friction_scale}x")
        print('='*60)
        
        results["RL"][condition_name] = []
        results["RL+SPC"][condition_name] = []
        
        for seed in seeds:
            print(f"\n  Seed {seed}:")
            
            # RL Only
            rollout_rl = run_single_rollout(
                geometry=geometry,
                perturbation=perturbation,
                reset_seed=seed,
                config=config,
                use_spc=False,
            )
            results["RL"][condition_name].append(rollout_rl)
            
            # RL + SPC
            rollout_spc = run_single_rollout(
                geometry=geometry,
                perturbation=perturbation,
                reset_seed=seed,
                config=config,
                use_spc=True,
            )
            results["RL+SPC"][condition_name].append(rollout_spc)
    
    return results


def compute_summary_stats(results: Dict) -> Dict:
    """Compute summary statistics across seeds for each condition."""
    summary = {}
    
    for method in ["RL", "RL+SPC"]:
        summary[method] = {}
        for condition, rollouts in results[method].items():
            metrics_list = [r["metrics"] for r in rollouts]
            
            summary[method][condition] = {
                "mean_final_dist": np.mean([m["final_dist"] for m in metrics_list]),
                "std_final_dist": np.std([m["final_dist"] for m in metrics_list]),
                "mean_min_dist": np.mean([m["min_dist"] for m in metrics_list]),
                "success_rate": np.mean([m["success"] for m in metrics_list]),
            }
    
    return summary


def print_summary_table(summary: Dict):
    """Print a formatted summary table."""
    print("\n" + "="*85)
    print("SUMMARY: Mean Final Distance (m) ± Std")
    print("="*85)
    
    conditions = list(summary["RL"].keys())
    
    print(f"{'Condition':<25} | {'RL':<20} | {'RL+SPC':<20} | {'Δ':<10}")
    print("-"*85)
    
    for cond in conditions:
        rl_mean = summary["RL"][cond]["mean_final_dist"]
        rl_std = summary["RL"][cond]["std_final_dist"]
        spc_mean = summary["RL+SPC"][cond]["mean_final_dist"]
        spc_std = summary["RL+SPC"][cond]["std_final_dist"]
        
        improvement = (rl_mean - spc_mean) / rl_mean * 100 if rl_mean > 0 else 0
        
        print(f"{cond:<25} | {rl_mean:.4f} ± {rl_std:.4f}   | "
              f"{spc_mean:.4f} ± {spc_std:.4f}   | {improvement:+.1f}%")
    
    print("="*85)
    
    # Success rate table
    print("\nSuccess Rate (reaching <5cm):")
    print("-"*60)
    print(f"{'Condition':<25} | {'RL':<12} | {'RL+SPC':<12}")
    print("-"*60)
    
    for cond in conditions:
        rl_sr = summary["RL"][cond]["success_rate"] * 100
        spc_sr = summary["RL+SPC"][cond]["success_rate"] * 100
        print(f"{cond:<25} | {rl_sr:>10.0f}% | {spc_sr:>10.0f}%")
    
    print("="*60)


def plot_results(results: Dict, output_prefix: str = "robustness"):
    """Generate visualization plots."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
    })
    
    colors = {"RL": "#F48B96", "RL+SPC": "#90CCEB"}
    
    conditions = list(results["RL"].keys())
    n_cond = len(conditions)
    
    # Bar chart of final distances
    fig, ax = plt.subplots(figsize=(max(10, n_cond * 0.8), 5))
    
    x = np.arange(n_cond)
    width = 0.35
    
    rl_means = [np.mean([r["metrics"]["final_dist"] for r in results["RL"][c]]) 
                for c in conditions]
    rl_stds = [np.std([r["metrics"]["final_dist"] for r in results["RL"][c]]) 
               for c in conditions]
    
    spc_means = [np.mean([r["metrics"]["final_dist"] for r in results["RL+SPC"][c]]) 
                 for c in conditions]
    spc_stds = [np.std([r["metrics"]["final_dist"] for r in results["RL+SPC"][c]]) 
                for c in conditions]
    
    ax.bar(x - width/2, rl_means, width, yerr=rl_stds, 
           label='RL', color=colors["RL"], capsize=3)
    ax.bar(x + width/2, spc_means, width, yerr=spc_stds,
           label='RL+SPC', color=colors["RL+SPC"], capsize=3)
    
    ax.axhline(y=0.05, color='green', linestyle='--', label='Success threshold', alpha=0.7)
    
    ax.set_ylabel('Final Distance to Target (m)')
    ax.set_xlabel('Condition (Geometry_Perturbation)')
    ax.set_title('Robustness Comparison: RL vs RL+SPC')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_bar.png", dpi=150)
    print(f"Saved {output_prefix}_bar.png")
    
    # Time series plots
    n_cols = min(4, n_cond)
    n_rows = (n_cond + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), squeeze=False)
    
    for idx, cond in enumerate(conditions):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        # Plot first seed
        rl_data = results["RL"][cond][0]
        spc_data = results["RL+SPC"][cond][0]
        
        ax.plot(rl_data["time"], rl_data["distances"], 
                color=colors["RL"], label='RL', alpha=0.8)
        ax.plot(spc_data["time"], spc_data["distances"],
                color=colors["RL+SPC"], label='RL+SPC', alpha=0.8)
        ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.5)
        
        ax.set_title(cond, fontsize=9)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (m)')
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # Hide empty subplots
    for idx in range(n_cond, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_timeseries.png", dpi=150)
    print(f"Saved {output_prefix}_timeseries.png")
    
    plt.show()


def main():
    """Run the unified robustness experiment."""
    parser = argparse.ArgumentParser(
        description="Unified robustness experiment: RL vs RL+SPC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiment modes:
  --quick       : 3 physics perturbations on cube only
  --physics     : Full physics perturbations on cube only  
  --geometry    : Geometry variants with nominal physics
  --full        : All combinations (physics × geometry)
        """
    )
    parser.add_argument("--quick", action="store_true", 
                        help="Quick test with 3 perturbations")
    parser.add_argument("--physics", action="store_true",
                        help="Full physics perturbations (cube only)")
    parser.add_argument("--geometry", action="store_true",
                        help="Geometry variants (nominal physics)")
    parser.add_argument("--full", action="store_true",
                        help="All combinations")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds per condition")
    parser.add_argument("--duration", type=float, default=8.0,
                        help="Rollout duration in seconds")
    parser.add_argument("--output", type=str, default="robustness",
                        help="Output file prefix")
    args = parser.parse_args()
    
    # Experiment config
    config = ExperimentConfig(
        duration=args.duration,
        num_seeds=args.seeds,
    )
    
    # Select perturbations based on mode
    if args.full:
        perturbations = get_full_perturbations()
        mode_name = "FULL (physics × geometry)"
    elif args.geometry:
        perturbations = get_geometry_perturbations()
        mode_name = "GEOMETRY ONLY"
    elif args.physics:
        perturbations = get_standard_perturbations()
        mode_name = "PHYSICS ONLY (cube)"
    elif args.quick:
        perturbations = get_quick_perturbations()
        mode_name = "QUICK TEST"
    else:
        # Default: quick test
        perturbations = get_quick_perturbations()
        mode_name = "QUICK TEST (default)"
    
    print(f"\n{'='*60}")
    print(f"UNIFIED ROBUSTNESS EXPERIMENT")
    print(f"Mode: {mode_name}")
    print(f"Config: {config}")
    print(f"Conditions: {len(perturbations)}")
    print(f"{'='*60}")
    
    # Run experiment
    start_time = time.time()
    results = run_experiment(perturbations, config)
    elapsed = time.time() - start_time
    print(f"\nExperiment completed in {elapsed/60:.1f} minutes")
    
    # Compute and print summary
    summary = compute_summary_stats(results)
    print_summary_table(summary)
    
    # Generate plots
    plot_results(results, output_prefix=args.output)


if __name__ == "__main__":
    main()
