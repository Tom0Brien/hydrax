"""Parallel experiment for Franka push task with obstacle avoidance.

This script runs the constrained push task with standard CCEM (no domain randomization).

Randomizes:
- Box start position (along x-axis, ±0.10m from nominal)
- Target position (along x-axis, ±0.10m from nominal)

Metrics:
- Success rate (box reaches target within threshold)
- Constraint violations (box enters obstacle zone)
- Final distance to target

Usage:
    python examples/franka_push_constrained_experiment.py --num_evals 16 --num_envs 4
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

from hydrax.tasks.franka import FrankaPushConstrained
from hydrax.algs import CCEM


# Experiment configuration
# Experiment configuration
OBSTACLE_RADIUS = 0.04
# Adjusted positions to be within RL policy training distribution (y in [-0.2, 0.2])
# while still requiring obstacle avoidance
OBSTACLE_POS_NOMINAL = jnp.array([0.60, 0.0])  # x, y (centered)
BOX_POS_NOMINAL = jnp.array([0.60, -0.3, 0.03])  # y=-0.15 (inside training bounds)
TARGET_POS_NOMINAL = jnp.array([0.60, 0.3, 0.03])  # y=0.15 (inside training bounds)

# Randomization range along x-axis (±0.10m)
X_RANDOM_RANGE = 0.10


class ConstrainedRolloutData(NamedTuple):
    """Data collected from parallel constrained rollouts."""
    time: jax.Array
    box_pos: jax.Array
    box_quat: jax.Array
    box_target_dist: jax.Array
    box_ori_error: jax.Array
    gripper_pos: jax.Array
    target_pos: jax.Array
    target_quat: jax.Array
    obstacle_pos: jax.Array  # For each env
    constraint_violation: jax.Array  # Max violation per timestep per env


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


def create_randomized_reset(task: FrankaPushConstrained, num_envs: int, obstacle_x_offset: float = 0.0):
    """Create a reset function that randomizes box and target positions.
    
    Args:
        task: The constrained task
        num_envs: Number of parallel environments
        obstacle_x_offset: X-axis offset for obstacle position (applied to model, same for all envs)
    """
    model = task.model
    home_qpos = jnp.array([
        -0.182772, 0.146282, 0.172246, -2.24238, -0.0788546, 2.45127, 0.0160022,  # arm (7)
        0.8, 0.8, 0.8, 0.8, 0.8, 0.8,  # gripper (6)
        0.56784, -0.0253974, 0.0306525,  # box position (3)
        0.0, 0.0, 0.0, 1.0,  # box quaternion wxyz (4)
    ])
    home_ctrl = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.82])
    
    # Compute obstacle position for this batch
    obstacle_xy = OBSTACLE_POS_NOMINAL.at[0].add(obstacle_x_offset)
    
    # Update model with new obstacle position
    obstacle_body_id = task._obstacle_body
    obstacle_z = model.body_pos[obstacle_body_id, 2]
    new_body_pos = model.body_pos.at[obstacle_body_id].set(
        jnp.array([obstacle_xy[0], obstacle_xy[1], obstacle_z])
    )
    model = model.tree_replace({"body_pos": new_body_pos})
    
    @jax.jit
    def single_reset(rng: jax.Array):
        rng, rng_box_x, rng_target_x = jax.random.split(rng, 3)
        
        # Randomize x positions (±0.15m from nominal)
        box_x_offset = jax.random.uniform(rng_box_x, minval=-X_RANDOM_RANGE, maxval=X_RANDOM_RANGE)
        target_x_offset = jax.random.uniform(rng_target_x, minval=-X_RANDOM_RANGE, maxval=X_RANDOM_RANGE)
        
        # Compute actual positions
        box_pos = BOX_POS_NOMINAL.at[0].add(box_x_offset)
        target_pos = TARGET_POS_NOMINAL.at[0].add(target_x_offset)
        
        # Update qpos for box
        qpos = home_qpos.at[13:16].set(box_pos)
        
        # Create mjx.Data
        mjx_data = mjx.make_data(model, nconmax=256, njmax=256)
        mjx_data = mjx_data.replace(
            qpos=qpos,
            qvel=jnp.zeros_like(mjx_data.qvel),
            ctrl=home_ctrl,
            mocap_pos=jnp.array([[target_pos[0], target_pos[1], target_pos[2]]]),
            mocap_quat=jnp.array([[1.0, 0.0, 0.0, 0.0]]),  # Identity quaternion
            time=jnp.array(0.0),
        )
        
        # Run forward kinematics
        mjx_data = mjx.forward(model, mjx_data)
        
        return mjx_data
    
    @jax.jit
    def batch_reset(key: jax.Array):
        keys = jax.random.split(key, num_envs)
        mjx_data_batch = jax.vmap(single_reset)(keys)
        # Return obstacle_xy repeated for all envs (for metric computation)
        obstacle_xy_batch = jnp.tile(obstacle_xy[None, :], (num_envs, 1))
        return mjx_data_batch, obstacle_xy_batch
    
    return batch_reset


def create_parallel_step_fn(task: FrankaPushConstrained):
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


def compute_constraint_violation(
    box_pos_xy: jax.Array,
    obstacle_pos_xy: jax.Array,
    obstacle_radius: float,
    obj_radius: float,
    safety_margin: float,
) -> jax.Array:
    """Compute constraint violation (positive = violated)."""
    min_safe_dist = obstacle_radius + obj_radius + safety_margin
    dist_to_obstacle = jnp.linalg.norm(box_pos_xy - obstacle_pos_xy, axis=-1)
    violation = min_safe_dist - dist_to_obstacle
    return violation


def run_parallel_rollout(
    task: FrankaPushConstrained,
    controller: CCEM | None,
    num_envs: int,
    base_seed: int = 42,
    duration: float = 5.0,
    frequency: float = 50.0,
) -> ConstrainedRolloutData:
    """Run parallel rollouts with randomized positions."""
    batch_reset = create_randomized_reset(task, num_envs)
    parallel_step = create_parallel_step_fn(task)
    
    # Reset all environments
    rng = jax.random.PRNGKey(base_seed)
    mjx_data, obstacle_xy_batch = batch_reset(rng)
    
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
    all_constraint_violations = []
    
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
            
            # Compute constraint violation
            violation = compute_constraint_violation(
                box_pos[:, :2],
                obstacle_xy_batch,
                OBSTACLE_RADIUS,
                task.obj_radius,
                task.safety_margin,
            )
            all_constraint_violations.append(violation)
    
    return ConstrainedRolloutData(
        time=jnp.array(all_times),
        box_pos=jnp.stack(all_box_pos, axis=1),
        box_quat=jnp.stack(all_box_quat, axis=1),
        box_target_dist=jnp.stack(all_box_target_dist, axis=1),
        box_ori_error=jnp.stack(all_box_ori_error, axis=1),
        gripper_pos=jnp.stack(all_gripper_pos, axis=1),
        target_pos=target_pos,
        target_quat=target_quat,
        obstacle_pos=obstacle_xy_batch,
        constraint_violation=jnp.stack(all_constraint_violations, axis=1),
    )


def run_batched_experiment(
    task: FrankaPushConstrained,
    controller: CCEM | None,
    num_evals: int,
    num_envs: int,
    base_seed: int,
    duration: float,
) -> ConstrainedRolloutData:
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
    
    # Concatenate all batches
    combined = ConstrainedRolloutData(
        time=all_data[0].time,
        box_pos=jnp.concatenate([d.box_pos for d in all_data], axis=0),
        box_quat=jnp.concatenate([d.box_quat for d in all_data], axis=0),
        box_target_dist=jnp.concatenate([d.box_target_dist for d in all_data], axis=0),
        box_ori_error=jnp.concatenate([d.box_ori_error for d in all_data], axis=0),
        gripper_pos=jnp.concatenate([d.gripper_pos for d in all_data], axis=0),
        target_pos=jnp.concatenate([d.target_pos for d in all_data], axis=0),
        target_quat=jnp.concatenate([d.target_quat for d in all_data], axis=0),
        obstacle_pos=jnp.concatenate([d.obstacle_pos for d in all_data], axis=0),
        constraint_violation=jnp.concatenate([d.constraint_violation for d in all_data], axis=0),
    )
    
    return combined


def compute_metrics(data: ConstrainedRolloutData) -> dict:
    """Compute metrics including constraint violations."""
    times = np.array(data.time)
    dists = np.array(data.box_target_dist)
    ori_errors = np.array(data.box_ori_error)
    violations = np.array(data.constraint_violation)
    
    mask = times > 1.0
    if not np.any(mask):
        mask = np.ones_like(times, dtype=bool)
    
    num_envs = dists.shape[0]
    
    final_dist_per_env = dists[:, -1]
    min_dist_per_env = np.min(dists[:, mask], axis=1)
    final_ori_per_env = ori_errors[:, -1]
    
    # Constraint violation metrics
    max_violation_per_env = np.max(violations, axis=1)  # Max violation over time
    any_violation_per_env = max_violation_per_env > 0  # Did constraint get violated?
    violation_rate = np.mean(any_violation_per_env)
    
    # Success thresholds
    pos_threshold = 0.05  # 5cm position error
    ori_threshold = 15.0 * np.pi / 180  # 15 degrees orientation error
    
    # Success requires position AND orientation AND no constraint violation
    success_per_env = (
        (final_dist_per_env < pos_threshold) & 
        (final_ori_per_env < ori_threshold) &
        (~any_violation_per_env)
    )
    
    # Relaxed success (ignoring constraint violations)
    success_relaxed_per_env = (final_dist_per_env < pos_threshold) & (final_ori_per_env < ori_threshold)
    
    def median_iqr(arr):
        return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    
    final_dist_median, final_dist_q1, final_dist_q3 = median_iqr(final_dist_per_env)
    min_dist_median, min_dist_q1, min_dist_q3 = median_iqr(min_dist_per_env)
    final_ori_median, final_ori_q1, final_ori_q3 = median_iqr(final_ori_per_env * 180 / np.pi)
    max_viol_median, max_viol_q1, max_viol_q3 = median_iqr(max_violation_per_env)
    
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
        "success_rate_relaxed": float(np.mean(success_relaxed_per_env)),
        "violation_rate": float(violation_rate),
        "max_violation": float(np.mean(max_violation_per_env)),
        "max_violation_median": max_viol_median,
        "num_evals": num_envs,
    }


def run_constrained_experiment(
    num_evals: int,
    num_envs: int,
    duration: float,
    base_seed: int,
) -> Dict[str, Dict]:
    """Run the constrained push task with standard CCEM (no domain randomization)."""
    jax.clear_caches()
    
    print(f"\n{'='*70}")
    print("CONSTRAINED PUSH EXPERIMENT: Obstacle Avoidance (Standard CCEM)")
    print(f"{'='*70}")
    print(f"Randomizing x-position: ±{X_RANDOM_RANGE}m")
    print(f"Running {num_evals} evals in batches of {num_envs}")
    print('='*70)
    
    modes = [
        ("Policy-guided CCEM", True),
    ]
    results = {}
    
    for mode_name, use_ccem in modes:
        print(f"\n  {mode_name}...", end=" ", flush=True)
        start = time.time()
        
        # Create task (always with RL policy)
        task = FrankaPushConstrained(
            geometry="square",
            use_rl_policy=True,
            safety_margin=0.02,
        )
        
        if use_ccem:
            controller = CCEM(
                task=task,
                num_samples=256,
                num_elites=16,
                sigma_start=0.3,
                sigma_min=0.05,
                explore_fraction=0.5,
                plan_horizon=1.0,
                spline_type="zero",
                num_knots=6,
                num_randomizations=1,
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
        
        print(f"success={metrics['success_rate']*100:.0f}%, "
              f"violations={metrics['violation_rate']*100:.0f}%, "
              f"final_dist={metrics['final_dist']:.3f}m ({elapsed:.1f}s)")
        
        results[mode_name] = {"data": data, "metrics": metrics}
    
    return results


def print_summary_table(results: Dict):
    """Print formatted summary table."""
    modes = list(results.keys())
    
    print("\n" + "="*90)
    print("SUMMARY: Constrained Push Experiment")
    print("="*90)
    
    print(f"\n{'Metric':<30}", end="")
    for mode in modes:
        print(f" | {mode:<18}", end="")
    print()
    print("-"*90)
    
    # Final distance
    print(f"{'Final Distance (m)':<30}", end="")
    for mode in modes:
        m = results[mode]["metrics"]
        print(f" | {m['final_dist']:.3f} ({m['final_dist_q1']:.3f}-{m['final_dist_q3']:.3f})", end="")
    print()
    
    # Success rate (strict)
    print(f"{'Success Rate (strict) %':<30}", end="")
    for mode in modes:
        sr = results[mode]["metrics"]["success_rate"] * 100
        print(f" | {sr:>16.0f}%", end="")
    print()
    
    # Success rate (relaxed)
    print(f"{'Success Rate (relaxed) %':<30}", end="")
    for mode in modes:
        sr = results[mode]["metrics"]["success_rate_relaxed"] * 100
        print(f" | {sr:>16.0f}%", end="")
    print()
    
    # Violation rate
    print(f"{'Constraint Violation Rate %':<30}", end="")
    for mode in modes:
        vr = results[mode]["metrics"]["violation_rate"] * 100
        print(f" | {vr:>16.0f}%", end="")
    print()
    
    # Max violation
    print(f"{'Max Violation (m, median)':<30}", end="")
    for mode in modes:
        mv = results[mode]["metrics"]["max_violation_median"]
        print(f" | {mv:>17.3f}", end="")
    print()
    
    print("="*90)
    print("\nNote: Strict success = dist<5cm AND ori<15° AND no constraint violations")
    print("      Relaxed success = dist<5cm AND ori<15° (ignoring violations)")


def plot_results(results: Dict, output_prefix: str = "constrained_experiment"):
    """Generate visualization plots."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
    })
    
    modes = list(results.keys())
    # Colors for metrics
    c_strict = "#90CCEB"  # Blue
    c_relaxed = "#B0E0E6" # Light Blue
    c_violation = "#F48B96" # Red
    c_dist = "#9ACD32"    # Green
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Bar chart 1: Success rates
    ax = axes[0]
    x = np.arange(len(modes))
    width = 0.35
    
    strict = [results[m]["metrics"]["success_rate"] * 100 for m in modes]
    relaxed = [results[m]["metrics"]["success_rate_relaxed"] * 100 for m in modes]
    
    ax.bar(x - width/2, strict, width, label='Strict', color=c_strict)
    ax.bar(x + width/2, relaxed, width, label='Relaxed', color=c_relaxed)
    
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Bar chart 2: Constraint violations
    ax = axes[1]
    violations = [results[m]["metrics"]["violation_rate"] * 100 for m in modes]
    bars = ax.bar(x, violations, color=c_violation)
    
    ax.set_ylabel('Violation Rate (%)')
    ax.set_title('Constraint Violation Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15, ha='right')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Bar chart 3: Final distance
    ax = axes[2]
    medians = [results[m]["metrics"]["final_dist"] for m in modes]
    q1s = [results[m]["metrics"]["final_dist_q1"] for m in modes]
    q3s = [results[m]["metrics"]["final_dist_q3"] for m in modes]
    
    yerr = [[medians[i] - q1s[i] for i in range(len(modes))],
            [q3s[i] - medians[i] for i in range(len(modes))]]
    
    ax.bar(x, medians, yerr=yerr, color=c_dist, capsize=5)
    ax.axhline(y=0.05, color='green', linestyle='--', label='Success threshold', alpha=0.7)
    
    ax.set_ylabel('Final Distance (m)')
    ax.set_title('Final Distance to Target')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_prefix}.pdf", bbox_inches='tight')
    print(f"\nSaved {output_prefix}.png/pdf")
    
    plt.show()


def main():
    """Run the constrained push experiment."""
    parser = argparse.ArgumentParser(
        description="Constrained Franka Push Experiment: RL vs CCEM vs CCEM+DR",
    )
    parser.add_argument("--num_evals", type=int, default=16,
                        help="Number of evaluations per condition")
    parser.add_argument("--num_envs", type=int, default=4,
                        help="Parallel environments per batch")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Rollout duration in seconds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--output", type=str, default="constrained_experiment",
                        help="Output file prefix")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("CONSTRAINED PUSH EXPERIMENT (Standard CCEM)")
    print(f"{'='*70}")
    print(f"Running: Policy-guided CCEM (no domain randomization)")
    print(f"Evaluations: {args.num_evals}, Parallel envs: {args.num_envs}")
    print(f"Duration: {args.duration}s per rollout")
    print(f"{'='*70}")
    
    # Run experiment
    start_time = time.time()
    results = run_constrained_experiment(
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
    plot_results(results, output_prefix=args.output)


if __name__ == "__main__":
    main()
