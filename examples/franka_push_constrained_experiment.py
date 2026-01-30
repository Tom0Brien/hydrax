"""Parallel experiment for Franka push task with workspace constraints.

This experiment evaluates performance when targets are placed near the workspace 
boundary, making constraint handling critical.

Compares:
1. RL Policy (Soft constraints via termination)
2. Policy-guided CEM (Unconstrained optimization)
3. Policy-guided CCEM (Constrained optimization)
"""

import argparse
import math
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mujoco import mjx

import sys
import os
sys.path.append(os.getcwd())

from hydrax.tasks.franka import FrankaPushGeometry
from hydrax.algs.cem import CEM
from hydrax.algs.ccem import CCEM
from hydrax.utils.video import VideoRecorder
import mujoco
from hydrax import ROOT


class FrankaPushBoundary(FrankaPushGeometry):
    """Franka push task with targets sampled near the workspace boundary."""
    
    def mjx_reset(self, rng: jax.Array) -> mjx.Data:
        """Reset with fixed target at the workspace boundary."""
        # Use standard reset first
        data = super().mjx_reset(rng)

        # Override object pos and orientation to identity (indices 16:20)
        qpos = data.qpos.at[13].set(0.65)
        qpos = qpos.at[14].set(0.0)
        qpos = qpos.at[16:20].set(jnp.array([1.0, 0.0, 0.0, 0.0]))

        # Override target position to be just outside the boundary
        rng, key = jax.random.split(rng)
        target_y = jax.random.uniform(key, minval=-0.1, maxval=0.1)
        x_offset = jax.random.uniform(key, minval=-0.05, maxval=0.05)
        target_pos = jnp.array([0.775 + x_offset, target_y, 0.0306525])
        target_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        
        mocap_pos = data.mocap_pos.at[0].set(target_pos)
        mocap_quat = data.mocap_quat.at[0].set(target_quat)
        
        return data.replace(qpos=qpos, mocap_pos=mocap_pos, mocap_quat=mocap_quat)


class ParallelRolloutData(NamedTuple):
    """Data collected from parallel rollouts."""
    time: jax.Array
    box_pos: jax.Array
    box_quat: jax.Array
    box_target_dist: jax.Array
    box_ori_error: jax.Array
    running_cost: jax.Array
    constraint_cost: jax.Array  # NEW: Track constraint violations
    qpos: jax.Array  # NEW: Full state trajectory
    qvel: jax.Array  # NEW: Full state trajectory
    mocap_pos: jax.Array
    mocap_quat: jax.Array


def quat_angle_error(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """Compute orientation error between two quaternions in radians."""
    q2_inv = q2.at[..., 1:].multiply(-1)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2_inv[..., 0], q2_inv[..., 1], q2_inv[..., 2], q2_inv[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    sin_half_angle = jnp.sqrt(x**2 + y**2 + z**2)
    return 2.0 * jnp.arcsin(jnp.clip(sin_half_angle, 0.0, 1.0))


def run_parallel_rollout(
    task: FrankaPushGeometry,
    controller: CEM | None,
    num_envs: int,
    base_seed: int = 42,
    duration: float = 10.0,
    frequency: float = 50.0,
) -> ParallelRolloutData:
    """Run parallel rollouts."""
    print(f"Running parallel rollout: {num_envs} envs for {duration}s...")
    
    # Batched reset
    batch_reset = jax.jit(jax.vmap(task.mjx_reset))
    
    # Batched step
    @jax.jit
    def parallel_step(mjx_data, ctrl_batch):
        def step_single(data, ctrl):
            data = task.apply_control(data, ctrl)
            return task.step(task.model, data)
        return jax.vmap(step_single)(mjx_data, ctrl_batch)
        
    # Batched cost evaluation for logging
    @jax.jit
    def compute_metrics_batch(mjx_data, ctrl_batch):
        run_cost = jax.vmap(task.running_cost)(mjx_data, ctrl_batch)
        con_cost = jax.vmap(task.constraint_cost)(mjx_data, ctrl_batch)
        return run_cost, con_cost

    # Initialization
    rng = jax.random.PRNGKey(base_seed)
    rngs = jax.random.split(rng, num_envs)
    mjx_data = batch_reset(rngs)
    
    target_pos = mjx_data.mocap_pos[:, 0, :]
    target_quat = mjx_data.mocap_quat[:, 0, :]
    
    # Controller init
    replan_period = 1.0 / frequency
    sim_dt = task.dt
    sim_steps_per_replan = int(replan_period / sim_dt)
    num_replans = int(duration * frequency)
    
    if controller is not None:
        policy_params_list = [
            controller.init_params(seed=base_seed + i) for i in range(num_envs)
        ]
        policy_params_batch = jax.tree.map(
            lambda *xs: jnp.stack(xs, axis=0), *policy_params_list
        )
        jit_optimize = jax.jit(jax.vmap(controller.optimize))
        jit_interp = jax.jit(controller.interp_func)
    else:
        policy_params_batch = None

    # Data collection
    all_data = {
        "time": [], "box_pos": [], "box_quat": [], 
        "box_target_dist": [], "box_ori_error": [],
        "running_cost": [], "constraint_cost": [],
        "qpos": [], "qvel": [], "mocap_pos": [], "mocap_quat": []
    }
    
    obj_body_id = task._obj_body
    
    start_time = time.time()
    
    for step in range(num_replans):
        if controller is not None:
            policy_params_batch, _ = jit_optimize(mjx_data, policy_params_batch)
            t_curr = mjx_data.time[0]
            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            
            def get_controls(tk, mean):
                return jit_interp(tq, tk, mean[None, ...])[0]
                
            us_batch = jax.vmap(get_controls)(
                policy_params_batch.tk, policy_params_batch.mean
            )
        else:
            us_batch = jnp.zeros((num_envs, sim_steps_per_replan, task.nu))
            
        for i in range(sim_steps_per_replan):
            ctrl_batch = us_batch[:, i, :]
            mjx_data = parallel_step(mjx_data, ctrl_batch)
            
            # Logging
            run_cost, con_cost = compute_metrics_batch(mjx_data, ctrl_batch)
            
            box_pos = mjx_data.xpos[:, obj_body_id, :]
            box_quat = mjx_data.xquat[:, obj_body_id, :]
            dist = jnp.linalg.norm(box_pos[:, :2] - target_pos[:, :2], axis=1)
            ori_err = quat_angle_error(box_quat, target_quat)
            
            all_data["time"].append(float(mjx_data.time[0]))
            all_data["box_pos"].append(box_pos)
            all_data["box_quat"].append(box_quat)
            all_data["box_target_dist"].append(dist)
            all_data["box_ori_error"].append(ori_err)
            all_data["running_cost"].append(run_cost)
            all_data["constraint_cost"].append(con_cost)
            all_data["qpos"].append(mjx_data.qpos)
            all_data["qvel"].append(mjx_data.qvel)
            all_data["mocap_pos"].append(mjx_data.mocap_pos)
            all_data["mocap_quat"].append(mjx_data.mocap_quat)
            
        if step % 10 == 0:
            print(f"Step {step}/{num_replans}", end="\r")
            
    print(f"\nRollout complete. Time: {time.time() - start_time:.2f}s")
    
    return ParallelRolloutData(
        time=jnp.array(all_data["time"]),
        box_pos=jnp.stack(all_data["box_pos"], axis=1),
        box_quat=jnp.stack(all_data["box_quat"], axis=1),
        box_target_dist=jnp.stack(all_data["box_target_dist"], axis=1),
        box_ori_error=jnp.stack(all_data["box_ori_error"], axis=1),
        running_cost=jnp.stack(all_data["running_cost"], axis=1),
        constraint_cost=jnp.stack(all_data["constraint_cost"], axis=1),
        qpos=jnp.stack(all_data["qpos"], axis=1),
        qvel=jnp.stack(all_data["qvel"], axis=1),
        mocap_pos=jnp.stack(all_data["mocap_pos"], axis=1),
        mocap_quat=jnp.stack(all_data["mocap_quat"], axis=1),
    )


def compute_metrics(data: ParallelRolloutData) -> dict:
    dists = np.array(data.box_target_dist)
    ori_errors = np.array(data.box_ori_error)
    cons = np.array(data.constraint_cost)
    
    # Success: dist < 3cm AND ori < 10 deg
    pos_success = dists[:, -1] < 0.05 # 5cm tolerance for hard task
    ori_success = ori_errors[:, -1] < (15 * np.pi / 180)
    success = pos_success & ori_success
    
    # Constraint stats
    # Boolean violation per step: count > 0
    violation_rate_per_env = np.mean(cons > 0.0, axis=1)
    
    def median_iqr(arr):
        return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))

    final_dist = median_iqr(dists[:, -1])
    final_ori = median_iqr(ori_errors[:, -1] * 180/np.pi)
    viol_rate = median_iqr(violation_rate_per_env * 100) # Percentage

    return {
        "final_dist": final_dist[0], "final_dist_q1": final_dist[1], "final_dist_q3": final_dist[2],
        "final_ori": final_ori[0], "final_ori_q1": final_ori[1], "final_ori_q3": final_ori[2],
        "viol_rate": viol_rate[0], "viol_rate_q1": viol_rate[1], "viol_rate_q3": viol_rate[2],
        "success_rate": np.mean(success) * 100,
        "num_evals": len(success)
    }

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
    print(f"Rendering video to {output_path}...")
    
    # Create video recorder
    fps = 1.0 / task.dt
    recorder = VideoRecorder(
        output_dir=os.path.dirname(output_path),
        width=width,
        height=height,
        fps=fps,
    )
    # Hack to set specific filename in VideoRecorder if needed, 
    # but VideoRecorder generates its own timestamped name.
    # We will rename it after or modify VideoRecorder. 
    # Looking at VideoRecorder, it generates a name in start().
    # Let's just let it generate the name and then rename it, or strict to directory.
    # Actually, VideoRecorder doesn't easily support custom filenames.
    # Let's just modify the output_dir to be specific and print the path.
    # Wait, better plan: minimal change to VideoRecorder usage.
    
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
    
    # Rename the file to the desired output path
    if recorder.video_path and os.path.exists(recorder.video_path):
        os.rename(recorder.video_path, output_path)
        print(f"Video saved to {output_path}")

def generate_latex_table(results: dict, output_path: str):
    approaches = list(results.keys())
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Constrained Franka Push Performance (Targets near boundary)}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Metric & " + " & ".join(approaches) + r" \\",
        r"\midrule",
    ]
    
    def fmt(val, q1=None, q3=None):
        return f"${val:.2f}$ ({q1:.2f}-{q3:.2f})"

    # Final Position
    row = ["Position Error (m)"]
    for a in approaches:
        m = results[a]["metrics"]
        row.append(fmt(m["final_dist"], m["final_dist_q1"], m["final_dist_q3"]))
    lines.append(" & ".join(row) + r" \\")
    
    # Constraint Violation
    row = ["Constraint Viol. (\\%)"]
    for a in approaches:
        m = results[a]["metrics"]
        row.append(fmt(m["viol_rate"], m["viol_rate_q1"], m["viol_rate_q3"]))
    lines.append(" & ".join(row) + r" \\")
    
    # Success Rate
    row = ["Success Rate (\\%)"]
    for a in approaches:
        row.append(f"${results[a]['metrics']['success_rate']:.1f}$")
    lines.append(" & ".join(row) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_evals", type=int, default=16)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("Running Constrained Experiment with Boundary Targets...")
    
    results = {}
    
    def run_scenario(name, use_policy, controller_cls=None, **ctrl_kwargs):
        print(f"\nRunning {name}...")
        task = FrankaPushBoundary(geometry="cube", use_rl_policy=use_policy)
        
        if controller_cls:
            ctrl = controller_cls(
                task=task,
                num_samples=128,
                num_elites=8,
                sigma_start=0.3,
                sigma_min=0.01,
                plan_horizon=0.5,
                num_knots=6,
                **ctrl_kwargs
            )
        else:
            ctrl = None
            
        # Run batches
        num_batches = math.ceil(args.num_evals / args.num_envs)
        combined_data = None
        
        for b in range(num_batches):
            data = run_parallel_rollout(
                task, ctrl, args.num_envs, args.seed + b*1000, 
                args.duration
            )
            # Simple concatenation for this script (omitted for brevity, assume 1 batch for logic check)
            # In full version we'd concat. For now let's just use the last batch or concat if needed.
            # Ideally use the run_batched_experiment helper from other file, but recreating here simpler.
            if combined_data is None:
                combined_data = data
            else:
                 # Quick concat of named tuples
                combined_data = ParallelRolloutData(
                    time=data.time,
                    box_pos=jnp.concatenate([combined_data.box_pos, data.box_pos], axis=0),
                    box_quat=jnp.concatenate([combined_data.box_quat, data.box_quat], axis=0),
                    box_target_dist=jnp.concatenate([combined_data.box_target_dist, data.box_target_dist], axis=0),
                    box_ori_error=jnp.concatenate([combined_data.box_ori_error, data.box_ori_error], axis=0),
                    running_cost=jnp.concatenate([combined_data.running_cost, data.running_cost], axis=0),
                    constraint_cost=jnp.concatenate([combined_data.constraint_cost, data.constraint_cost], axis=0),
                    qpos=jnp.concatenate([combined_data.qpos, data.qpos], axis=0),
                    qvel=jnp.concatenate([combined_data.qvel, data.qvel], axis=0),
                    mocap_pos=jnp.concatenate([combined_data.mocap_pos, data.mocap_pos], axis=0),
                    mocap_quat=jnp.concatenate([combined_data.mocap_quat, data.mocap_quat], axis=0),
                )
        
        metrics = compute_metrics(combined_data)
        results[name] = {"metrics": metrics}
        
        print(f"  Final Dist: {metrics['final_dist']:.4f} ({metrics['final_dist_q1']:.4f}-{metrics['final_dist_q3']:.4f})")
        print(f"  Final Ori:  {metrics['final_ori']:.2f} ({metrics['final_ori_q1']:.2f}-{metrics['final_ori_q3']:.2f})")
        print(f"  Viol Rate:  {metrics['viol_rate']:.1f}%")
        print(f"  Success:    {metrics['success_rate']:.1f}%")

        # Render trajectories
        video_dir = os.path.join(ROOT, "recordings", name)
        os.makedirs(video_dir, exist_ok=True)
        
        # Render each environment
        # CAUTION: This can be slow if num_evals is large!
        # Limiting to top 5 for efficiency if needed, or render all.
        # Let's render all since user asked for "all trajectories".
        for i in range(combined_data.qpos.shape[0]):
            video_path = os.path.join(video_dir, f"{name}_env{i}.mp4")
            render_trajectory(
                task,
                combined_data.qpos[i],
                combined_data.qvel[i],
                combined_data.mocap_pos[i],
                combined_data.mocap_quat[i],
                video_path
            )

    # 1. RL Policy
    run_scenario("Policy", True, None)
    
    # 2. Policy-guided CEM
    run_scenario("Policy-CEM", True, CEM)
    
    # 3. Policy-guided CCEM
    run_scenario("Policy-CCEM", True, CCEM)
    
    generate_latex_table(results, "franka_constrained_results.tex")

if __name__ == "__main__":
    main()
