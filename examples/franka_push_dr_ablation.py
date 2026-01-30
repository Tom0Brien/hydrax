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
from hydrax.utils.video import VideoRecorder
from hydrax import ROOT


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
    use_cem: bool = True
    use_rl_policy: bool = True
    
    def __repr__(self) -> str:
        if not self.use_cem:
            return f"{self.name} (RL only)"
        if not self.use_rl_policy:
            return f"{self.name} (CEM only)"
        dr_str = f"DR={self.num_randomizations}"
        risk_str = self.risk_strategy.__class__.__name__ if self.risk_strategy else "None"
        return f"{self.name} ({dr_str}, risk={risk_str})"


def get_controller_variants() -> List[ControllerVariant]:
    """Get all controller variants to compare."""
    return [
        # --- Parametric Robustness Baselines ---
        ControllerVariant(
            name="RL Only",
            num_randomizations=1,
            risk_strategy=None,
            color="#808080",  # Gray
            use_cem=False,
            use_rl_policy=True,
        ),
        ControllerVariant(
            name="CEM Only",
            num_randomizations=1,
            risk_strategy=None,
            color="#daa520",  # Goldenrod
            use_cem=True,
            use_rl_policy=False,
        ),
        # --- Policy-Guided CEM (Nominal) ---
        ControllerVariant(
            name="Policy-guided CEM",
            num_randomizations=1,
            risk_strategy=None,
            color="#1f77b4",  # Blue
            use_cem=True,
            use_rl_policy=True,
        ),
        # --- DR Variants (Policy-Guided) ---
        ControllerVariant(
            name="DR + Average",
            num_randomizations=5,
            risk_strategy=AverageCost(),
            color="#ff7f0e",  # Orange
            use_cem=True,
            use_rl_policy=True,
        ),
        ControllerVariant(
            name="DR + CVaR",
            num_randomizations=5,
            risk_strategy=ConditionalValueAtRisk(alpha=0.25),
            color="#2ca02c",  # Green
            use_cem=True,
            use_rl_policy=True,
        ),
        ControllerVariant(
            name="DR + WorstCase",
            num_randomizations=5,
            risk_strategy=WorstCase(),
            color="#d62728",  # Red
            use_cem=True,
            use_rl_policy=True,
        ),
        ControllerVariant(
            name="DR + ExpWeight",
            num_randomizations=5,
            risk_strategy=ExponentialWeightedAverage(gamma=1.0),
            color="#9467bd",  # Purple
            use_cem=True,
            use_rl_policy=True,
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
            PhysicsPerturbation("light", 0.5, 1.0),
            PhysicsPerturbation("heavy", 2.0, 1.0),
            PhysicsPerturbation("slippery", 1.0, 0.5),
            PhysicsPerturbation("sticky", 1.0, 1.5),
            PhysicsPerturbation("light_slippery", 0.5, 0.5),
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
    running_cost: jax.Array
    qpos: jax.Array
    qvel: jax.Array
    mocap_pos: jax.Array
    mocap_quat: jax.Array


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
    def parallel_step(mjx_data: mjx.Data, ctrl_batch: jax.Array) -> mjx.Data:
        def step_env(data, ctrl):
            data = task.apply_control(data, ctrl)
            return task.step(sim_model, data)  # Use task.step which handles n_substeps
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
    all_running_cost = []
    all_qpos = []
    all_qvel = []
    all_mocap_pos = []
    all_mocap_quat = []
    
    # JIT compile running cost function for batch evaluation
    @jax.jit
    def compute_running_cost_batch(mjx_data_batch, ctrl_batch):
        """Compute running cost for all environments."""
        return jax.vmap(task.running_cost)(mjx_data_batch, ctrl_batch)
    
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
        
        # Take the first control for the duration of the control step
        ctrl_batch = us_batch[:, 0, :]
        
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
        
        # Running cost
        step_cost = compute_running_cost_batch(mjx_data, ctrl_batch)
        all_running_cost.append(step_cost)

        all_qpos.append(mjx_data.qpos)
        all_qvel.append(mjx_data.qvel)
        all_mocap_pos.append(mjx_data.mocap_pos)
        all_mocap_quat.append(mjx_data.mocap_quat)
    
    # Pad to ensure correct length if needed (though shouldn't be with fixed steps)
    
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
        qpos=jnp.stack(all_qpos, axis=1),
        qvel=jnp.stack(all_qvel, axis=1),
        mocap_pos=jnp.stack(all_mocap_pos, axis=1),
        mocap_quat=jnp.stack(all_mocap_quat, axis=1),
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
        running_cost=jnp.concatenate([d.running_cost for d in all_data], axis=0),
        qpos=jnp.concatenate([d.qpos for d in all_data], axis=0),
        qvel=jnp.concatenate([d.qvel for d in all_data], axis=0),
        mocap_pos=jnp.concatenate([d.mocap_pos for d in all_data], axis=0),
        mocap_quat=jnp.concatenate([d.mocap_quat for d in all_data], axis=0),
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
    
    # Final state metrics
    final_dist_per_env = dists[:, -1]
    final_ori_per_env = ori_errors[:, -1]
    
    # Total running cost
    total_running_cost_per_env = np.sum(data.running_cost, axis=1)
    
    # Success thresholds
    pos_threshold = 0.05  # 5cm position error
    ori_threshold = 15.0 * np.pi / 180  # 15 degrees orientation error
    
    # Success requires BOTH position AND orientation criteria
    success_per_env = (final_dist_per_env < pos_threshold) & (final_ori_per_env < ori_threshold)
    
    # Time to success
    def time_to_success_fn(dist_seq, ori_seq):
        within_pos = dist_seq < pos_threshold
        within_ori = ori_seq < ori_threshold
        success_mask = within_pos & within_ori
        if np.any(success_mask):
            return times[np.argmax(success_mask)]
        return float('inf')
    
    time_to_success_per_env = np.array([
        time_to_success_fn(dists[i], ori_errors[i]) for i in range(num_envs)
    ])
    
    def median_iqr(arr):
        return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    
    final_dist_median, final_dist_q1, final_dist_q3 = median_iqr(final_dist_per_env)
    final_ori_median, final_ori_q1, final_ori_q3 = median_iqr(final_ori_per_env * 180 / np.pi)
    total_cost_median, total_cost_q1, total_cost_q3 = median_iqr(total_running_cost_per_env)
    
    # Time to success stats (only for successful runs)
    successful_times = time_to_success_per_env[time_to_success_per_env < float('inf')]
    if len(successful_times) > 0:
        time_success_median, time_success_q1, time_success_q3 = median_iqr(successful_times)
    else:
        time_success_median, time_success_q1, time_success_q3 = float('inf'), float('inf'), float('inf')
    
    return {
        "final_dist": final_dist_median,
        "final_dist_q1": final_dist_q1,
        "final_dist_q3": final_dist_q3,
        
        "final_ori_error": final_ori_median,
        "final_ori_error_q1": final_ori_q1,
        "final_ori_error_q3": final_ori_q3,
        
        "total_running_cost": total_cost_median,
        "total_running_cost_q1": total_cost_q1,
        "total_running_cost_q3": total_cost_q3,
        
        "time_to_success": time_success_median,
        "time_to_success_q1": time_success_q1,
        "time_to_success_q3": time_success_q3,
        
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
        num_elites=8,
        sigma_start=0.1,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=6,
        num_randomizations=variant.num_randomizations,
        risk_strategy=variant.risk_strategy,
    )


def render_trajectory(
    task: FrankaPushGeometry,
    qpos_traj: jax.Array,
    qvel_traj: jax.Array,
    mocap_pos_traj: jax.Array,
    mocap_quat_traj: jax.Array,
    output_path: str,
    width: int = 720,
    height: int = 480,
):
    """Render a trajectory to a video file."""
    # Create video recorder
    fps = 1.0 / task.ctrl_dt  # Use control frequency (50Hz)
    recorder = VideoRecorder(
        output_dir=os.path.dirname(output_path),
        width=width,
        height=height,
        fps=fps,
    )
    
    # Ensure model visual offscreen buffer is compatible
    task.mj_model.vis.global_.offwidth = width
    task.mj_model.vis.global_.offheight = height
    
    if not recorder.start():
        print("Failed to start video recorder")
        return

    renderer = mujoco.Renderer(task.mj_model, height=height, width=width)
    mj_data = mujoco.MjData(task.mj_model)
    
    # Convert JAX arrays to numpy for MuJoCo
    qpos_np = np.array(qpos_traj)
    qvel_np = np.array(qvel_traj)
    mocap_pos_np = np.array(mocap_pos_traj)
    mocap_quat_np = np.array(mocap_quat_traj)
    
    for i in range(len(qpos_np)):
        mj_data.qpos[:] = qpos_np[i]
        mj_data.qvel[:] = qvel_np[i]
        mj_data.mocap_pos[:] = mocap_pos_np[i]
        mj_data.mocap_quat[:] = mocap_quat_np[i]
        mujoco.mj_forward(task.mj_model, mj_data)
        
        renderer.update_scene(mj_data)
        frame = renderer.render()
        recorder.add_frame(frame.tobytes())
        
    recorder.stop()
    
    # Rename the file to the desired output path if needed
    if recorder.video_path and os.path.exists(recorder.video_path) and recorder.video_path != output_path:
        os.rename(recorder.video_path, output_path)


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
            task = FrankaPushGeometry(geometry="cube", use_rl_policy=variant.use_rl_policy)

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

            # Render trajectories
            video_dir = os.path.join(ROOT, "recordings", "dr_ablation", perturbation.name, variant.name)
            os.makedirs(video_dir, exist_ok=True)
            
            print(f"  Rendering videos to {video_dir}...")
            for i in range(data.qpos.shape[0]):
                video_path = os.path.join(video_dir, f"env_{i}.mp4")
                render_trajectory(
                    task,
                    data.qpos[i],
                    data.qvel[i],
                    data.mocap_pos[i],
                    data.mocap_quat[i],
                    video_path
                )
            
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
    
    print("\nTime to Success (s):")
    print("-"*100)
    print(header)
    print("-"*100)
    
    for cond in conditions:
        row = f"{cond:<20}"
        for name in variant_names:
            m = results[cond][name]["metrics"]
            ts = m["time_to_success"]
            if ts == float('inf'):
                row += f" | {'N/A':>13}"
            else:
                row += f" | {ts:>13.2f}"
        print(row)
    print("="*100)



def plot_ablation_results(
    results: Dict,
    variants: List[ControllerVariant],
    output_prefix: str = "dr_ablation",
    title: str = "Comparison Results",
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
    width = 0.8 / n_variants
    
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
    ax.set_title(f'{title}: Final Distance\n(Median + IQR, same initial conditions)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_distance.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_distance.pdf", bbox_inches='tight')
    print(f"Saved {output_prefix}_distance.png/pdf")
    
    # Bar chart of final orientation error
    fig, ax = plt.subplots(figsize=(max(14, n_cond * 3), 7))
    
    for i, name in enumerate(variant_names):
        medians = [results[c][name]["metrics"]["final_ori_error"] for c in conditions]
        q1s = [results[c][name]["metrics"]["final_ori_error_q1"] for c in conditions]
        q3s = [results[c][name]["metrics"]["final_ori_error_q3"] for c in conditions]
        
        yerr = [[medians[j] - q1s[j] for j in range(n_cond)],
                [q3s[j] - medians[j] for j in range(n_cond)]]
        
        offset = (i - (n_variants - 1) / 2) * width
        ax.bar(x + offset, medians, width, yerr=yerr, 
               label=name, color=colors[name], capsize=2, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=15.0, color='green', linestyle='--', label='Success (15°)', alpha=0.7)
    
    ax.set_ylabel('Final Orientation Error (°)')
    ax.set_xlabel('Perturbation Condition')
    ax.set_title(f'{title}: Final Orientation Error\n(Median + IQR)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_orientation.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_orientation.pdf", bbox_inches='tight')
    print(f"Saved {output_prefix}_orientation.png/pdf")
    
    # Bar chart of total running cost
    fig, ax = plt.subplots(figsize=(max(14, n_cond * 3), 7))
    
    for i, name in enumerate(variant_names):
        medians = [results[c][name]["metrics"]["total_running_cost"] for c in conditions]
        q1s = [results[c][name]["metrics"]["total_running_cost_q1"] for c in conditions]
        q3s = [results[c][name]["metrics"]["total_running_cost_q3"] for c in conditions]
        
        yerr = [[medians[j] - q1s[j] for j in range(n_cond)],
                [q3s[j] - medians[j] for j in range(n_cond)]]
        
        offset = (i - (n_variants - 1) / 2) * width
        ax.bar(x + offset, medians, width, yerr=yerr, 
               label=name, color=colors[name], capsize=2, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Total Running Cost')
    ax.set_xlabel('Perturbation Condition')
    ax.set_title(f'{title}: Total Running Cost\n(Median + IQR)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_cost.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_cost.pdf", bbox_inches='tight')
    print(f"Saved {output_prefix}_cost.png/pdf")
    
    # Bar chart of time to success
    fig, ax = plt.subplots(figsize=(max(14, n_cond * 3), 7))
    
    for i, name in enumerate(variant_names):
        medians = [results[c][name]["metrics"]["time_to_success"] for c in conditions]
        # Handle Inf for plotting (set to 0 or max + constant)
        # For this plot we'll just plot what we have, ignoring Inf
        plot_medians = [m if m != float('inf') else 0 for m in medians]
        
        q1s = [results[c][name]["metrics"]["time_to_success_q1"] for c in conditions]
        q3s = [results[c][name]["metrics"]["time_to_success_q3"] for c in conditions]
        
        # Adjust IQR for Inf
        q1s = [v if v != float('inf') else 0 for v in q1s]
        q3s = [v if v != float('inf') else 0 for v in q3s]
        
        yerr = [[max(0, plot_medians[j] - q1s[j]) for j in range(n_cond)],
                [max(0, q3s[j] - plot_medians[j]) for j in range(n_cond)]]
        
        offset = (i - (n_variants - 1) / 2) * width
        bars = ax.bar(x + offset, plot_medians, width, yerr=yerr, 
               label=name, color=colors[name], capsize=2, edgecolor='black', linewidth=0.5)
        
        # Label Inf bars
        for j, m in enumerate(medians):
            if m == float('inf'):
                ax.text(x[j] + offset, 0.1, "N/A", ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_ylabel('Time to Success (s)')
    ax.set_xlabel('Perturbation Condition')
    ax.set_title(f'{title}: Time to Success\n(Median + IQR, successful runs only)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_time.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_time.pdf", bbox_inches='tight')
    print(f"Saved {output_prefix}_time.png/pdf")
    
    # Success rate bar chart
    fig, ax = plt.subplots(figsize=(max(14, n_cond * 3), 6))
    
    for i, name in enumerate(variant_names):
        success_rates = [results[c][name]["metrics"]["success_rate"] * 100 for c in conditions]
        
        offset = (i - (n_variants - 1) / 2) * width
        ax.bar(x + offset, success_rates, width,
               label=name, color=colors[name], edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Success Rate (%)')
    ax.set_xlabel('Perturbation Condition')
    ax.set_title(f'{title}: Success Rate (dist<5cm, ori<15°)')
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
    
    ax.set_title(f'{title}: Success Rate Heatmap')
    ax.set_xlabel('Perturbation Condition')
    ax.set_ylabel('Controller Variant')
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Success Rate (%)', rotation=-90, va="bottom")
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_heatmap.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_heatmap.pdf", bbox_inches='tight')
    print(f"Saved {output_prefix}_heatmap.png/pdf")
    
    # plt.show()


def generate_latex_table(
    results: Dict,
    variants: List[ControllerVariant],
    output_path: str,
    caption: str = "Comparison Results",
):
    """Generate LaTeX table of results."""
    conditions = list(results.keys())
    variant_names = [v.name for v in variants]
    
    def fmt_dist(m):
        return f"${m['final_dist']:.3f}$ ({m['final_dist_q1']:.2f}-{m['final_dist_q3']:.2f})"
        
    def fmt_ori(m):
        return f"${m['final_ori_error']:.1f}$ ({m['final_ori_error_q1']:.1f}-{m['final_ori_error_q3']:.1f})"
        
    def fmt_cost(m):
        return f"${m['total_running_cost']:.1f}$ ({m['total_running_cost_q1']:.1f}-{m['total_running_cost_q3']:.1f})"
        
    def fmt_time(m):
        if m['time_to_success'] == float('inf'):
            return "N/A"
        return f"${m['time_to_success']:.2f}$ ({m['time_to_success_q1']:.2f}-{m['time_to_success_q3']:.2f})"
    
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
        r"\caption{" + caption + r"}",
        r"\label{tab:results}",
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
    
    # Final orientation section
    lines.append(r"\multicolumn{" + str(len(variant_names) + 1) + r"}{l}{\textit{Final Orientation Error ($^\circ$)}} \\")
    for cond in conditions:
        row = cond.replace("_", r"\_")
        for name in variant_names:
            m = results[cond][name]["metrics"]
            row += f" & {fmt_ori(m)}"
        row += r" \\"
        lines.append(row)
        
    lines.append(r"\midrule")
    
    # Running cost section
    lines.append(r"\multicolumn{" + str(len(variant_names) + 1) + r"}{l}{\textit{Total Running Cost}} \\")
    for cond in conditions:
        row = cond.replace("_", r"\_")
        for name in variant_names:
            m = results[cond][name]["metrics"]
            row += f" & {fmt_cost(m)}"
        row += r" \\"
        lines.append(row)
        
    lines.append(r"\midrule")
    
    # Time to success section
    lines.append(r"\multicolumn{" + str(len(variant_names) + 1) + r"}{l}{\textit{Time to Success (s)}} \\")
    for cond in conditions:
        row = cond.replace("_", r"\_")
        for name in variant_names:
            m = results[cond][name]["metrics"]
            row += f" & {fmt_time(m)}"
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
    parser.add_argument("--num_evals", type=int, default=24,
                        help="Number of evaluations per condition")
    parser.add_argument("--num_envs", type=int, default=24,
                        help="Parallel environments per batch")
    parser.add_argument("--duration", type=float, default=8.0,
                        help="Rollout duration in seconds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (same for all variants)")
    parser.add_argument("--output", type=str, default="dr_ablation",
                        help="Output file prefix")
    args = parser.parse_args()
    
    perturbations = get_perturbations(args.mode)
    all_variants = get_controller_variants()
    
    print(f"\n{'='*70}")
    print(f"Comparison Experiment")
    print(f"  Mode: {args.mode}")
    print(f"  Conditions: {[p.name for p in perturbations]}")
    print(f"  Variants: {[v.name for v in all_variants]}")
    print(f"{'='*70}\n")
    
    # Run full experiment for all variants
    all_results = run_ablation_experiment(
        perturbations,
        all_variants,
        args.num_evals,
        args.num_envs,
        args.duration,
        args.seed,
    )
    
    # --- Analysis 1: Parametric Robustness ---
    # Compare RL Only vs CEM Only vs Policy-guided CEM
    print("\n" + "="*80)
    print("ANALYSIS 1: Parametric Robustness (RL vs CEM vs Policy-guided CEM)")
    print("="*80)
    
    robustness_names = ["RL Only", "CEM Only", "Policy-guided CEM"]
    robustness_variants = [v for v in all_variants if v.name in robustness_names]
    
    if robustness_variants:
        print_summary_table(all_results, robustness_variants)
        plot_ablation_results(
            all_results, 
            robustness_variants, 
            output_prefix=f"{args.output}_robustness", 
            title="Parametric Robustness"
        )
        generate_latex_table(
            all_results, 
            robustness_variants, 
            f"{args.output}_robustness.tex",
            caption="Parametric Robustness Comparison"
        )
    
    # --- Analysis 2: DR Benefits ---
    # Compare Policy-guided CEM against DR variants
    # Exclude RL Only and CEM Only
    print("\n" + "="*80)
    print("ANALYSIS 2: DR Benefits (Policy-guided CEM +/- DR)")
    print("="*80)
    
    dr_variants = [v for v in all_variants if v.name not in ["RL Only", "CEM Only"]]
    
    if dr_variants:
        print_summary_table(all_results, dr_variants)
        plot_ablation_results(
            all_results, 
            dr_variants, 
            output_prefix=f"{args.output}_dr_benefit", 
            title="DR Benefit Analysis"
        )
        generate_latex_table(
            all_results, 
            dr_variants, 
            f"{args.output}_dr_benefit.tex",
            caption="Domain Randomization Benefit Comparison"
        )
    
    print("\nExperiment Complete!")


if __name__ == "__main__":
    main()
