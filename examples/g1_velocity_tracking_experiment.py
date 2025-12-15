import time
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Optional, Dict

# Add project root to path if needed (though running as module is better)
import sys
import os
sys.path.append(os.getcwd())

from hydrax.tasks.g1.g1_velocity_tracking import G1VelocityTracking
from hydrax.tasks.g1.g1_velocity_tracking_augmented import G1VelocityTrackingAugmented
from hydrax.algs.predictive_sampling import PredictiveSampling
from hydrax.algs.cem import CEM

def compute_actual_velocity(qvel: np.ndarray, qpos: np.ndarray) -> np.ndarray:
    """Compute base velocity in base frame from world qvel and qpos."""
    # qpos[3:7] is [qw, qx, qy, qz]
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    
    # Inverse rotation (conjugate for unit quaternion)
    # We want to rotate world velocity TO base frame
    # R_base_to_world = q
    # v_base = R_world_to_base * v_world = q_inv * v_world
    
    # Quaternion multiplication logic or rotation matrix
    # R = ...
    # Let's use a simple rotation formula
    
    # v_world
    vx, vy, vz = qvel[0], qvel[1], qvel[2]
    
    # Rotate vector v by quaternion inverse q_inv = [w, -x, -y, -z]
    # v' = q_inv * v * q
    # Standard formula for rotating vector v by quaternion q is v' = v + 2*cross(q.xyz, cross(q.xyz, v) + q.w*v)
    # Here we rotate by q_inv.
    
    # q_inv components
    iw, ix, iy, iz = qw, -qx, -qy, -qz
    
    # Cross product 1: q_xyz x v
    c1x = iy * vz - iz * vy
    c1y = iz * vx - ix * vz
    c1z = ix * vy - iy * vx
    
    # Cross product 2: q_xyz x (c1 + q_w * v)
    # term = c1 + iw * v
    tx = c1x + iw * vx
    ty = c1y + iw * vy
    tz = c1z + iw * vz
    
    c2x = iy * tz - iz * ty
    c2y = iz * tx - ix * tz
    c2z = ix * ty - iy * tx
    
    # Result = v + 2 * c2
    v_base_x = vx + 2 * c2x
    v_base_y = vy + 2 * c2y
    v_base_z = vz + 2 * c2z
    
    # Angular velocity
    # qvel[3:6] is usually in local body frame for free joints in MuJoCo?
    # Wait, for freejoint, qvel is: 0-2 linear (world), 3-5 angular (local/body frame) IF conaffinity is set?
    # No, MuJoCo qvel for free joint:
    # 0-2: linear velocity in WORLD frame
    # 3-5: angular velocity in LOCAL/BODY frame (usually)
    # Let's verify.
    # "The velocity of a free joint is represented by 6 numbers... linear velocity in global frame... angular velocity in local frame."
    # So qvel[3:6] is ALREADY w_base!
    
    w_base_z = qvel[5]
    
    return np.array([v_base_x, v_base_y, w_base_z])

def run_rollout(
    task: Any,
    controller: Any,
    duration: float = 4.0,
    frequency: float = 50.0,
) -> Dict[str, np.ndarray]:
    """Run a deterministic rollout and return data."""
    print(f"Running rollout for {duration}s...")
    
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    
    # Initialize state
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
        mj_data.ctrl[:] = task._default_pose
    
    # Setup timing
    replan_period = 1.0 / frequency
    sim_dt = mj_model.opt.timestep
    sim_steps_per_replan = int(replan_period / sim_dt)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    
    # Initialize controller
    mjx_data = mjx.put_data(mj_model, mj_data)
    
    if hasattr(controller, "init_params"):
        # MPC Controller
        print("Initializing MPC controller...")
        # init_params takes (initial_knots, seed)
        policy_params = controller.init_params(initial_knots=None, seed=0)
        jit_optimize = jax.jit(controller.optimize)
        jit_interp_func = jax.jit(controller.interp_func)
        
        # Warmup
        print("Warming up controller...")
        policy_params, _ = jit_optimize(mjx_data, policy_params)
        policy_params, _ = jit_optimize(mjx_data, policy_params)
    else:
        # RL Only
        policy_params = None
        jit_optimize = None
        jit_interp_func = None
        
    # Data logging
    times = []
    target_vels = []
    actual_vels = []
    
    # Main loop
    num_replans = int(duration * frequency)
    
    start_time = time.time()
    
    for step in range(num_replans):
        # 1. Replan (if MPC)
        if policy_params is not None:
            # Update mjx_data from current mj_data
            mjx_data = mjx.put_data(mj_model, mj_data)
            mjx_data = mjx_data.replace(
                mocap_pos=mj_data.mocap_pos, 
                mocap_quat=mj_data.mocap_quat
            )
            
            policy_params, _ = jit_optimize(mjx_data, policy_params)
            
            # Interpolate controls for this period
            t_curr = mj_data.time
            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            tk = policy_params.tk
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]
        else:
            # RL Only: constant target velocity command
            us = np.tile(task.target_velocity, (sim_steps_per_replan, 1))
            
        # 2. Simulate substeps
        n_substeps = getattr(task, "n_substeps", 1)
        
        for i in range(sim_steps_per_replan):
            # Update control
            should_update_ctrl = (n_substeps == 1) or (i % n_substeps == 0)
            
            if should_update_ctrl:
                # Sync mjx_data for apply_control
                mjx_data = mjx.put_data(mj_model, mj_data)
                
                # Apply control
                ctrl_input = jnp.array(us[i])
                # Note: apply_control is not JITted here, running in eager mode
                mjx_data = task.apply_control(mjx_data, ctrl_input)
                
                # Sync back to mj_data
                mj_data.ctrl[:] = np.array(mjx_data.ctrl)
            
            # Step physics
            mujoco.mj_step(mj_model, mj_data)
            
            # Log data
            times.append(mj_data.time)
            target_vels.append(task.target_velocity)
            
            # Compute actual velocity
            # qvel: 0-2 (linear world), 3-5 (angular local)
            qvel = mj_data.qvel[:6]
            qpos = mj_data.qpos[:7]
            v_base = compute_actual_velocity(qvel, qpos)
            actual_vels.append(v_base)
            
        # Progress
        if step % 10 == 0:
            print(f"Step {step}/{num_replans}", end="\r")
            
    print(f"\nRollout complete. Time: {time.time() - start_time:.2f}s")
    
    return {
        "time": np.array(times),
        "target": np.array(target_vels),
        "actual": np.array(actual_vels)
    }

def main():
    target_vel = jnp.array([0.5, 0.0, 0.0])  # 0.5 m/s forward
    duration = 4.0
    
    results = {}
    
    # Scenario 1: RL Policy Alone
    print("\n--- Scenario 1: RL Policy Alone ---")
    task1 = G1VelocityTracking(target_velocity=target_vel)
    class RLController:
        pass
    ctrl1 = RLController()
    results["RL Only"] = run_rollout(task1, ctrl1, duration)
    
    # Scenario 2: RL Policy + MPC
    print("\n--- Scenario 2: RL Policy + MPC ---")
    task2 = G1VelocityTracking(target_velocity=target_vel)
    ctrl2 = CEM(
        task=task2,
        num_samples=64,
        num_elites=8,
        sigma_start=0.4,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=1,
        spline_type="zero",
        num_knots=4,
    )
    results["RL + MPC"] = run_rollout(task2, ctrl2, duration)
    
    # Scenario 3: RL Policy + MPC + Augmented
    print("\n--- Scenario 3: RL Policy + MPC + Augmented ---")
    task3 = G1VelocityTrackingAugmented(target_velocity=target_vel)
    ctrl3 = CEM(
        task=task3,
        num_samples=64,
        num_elites=8,
        sigma_start=0.4,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=1,
        spline_type="zero",
        num_knots=4,
    )
    results["RL + MPC + Aug"] = run_rollout(task3, ctrl3, duration)
    
    # Calculate and print metrics
    print("\n" + "="*60)
    print(f"{'Metric':<20} | {'RL Only':<12} | {'RL + MPC':<12} | {'RL + MPC + Aug':<12}")
    print("-" * 60)
    
    metrics = ["RMSE Vx", "RMSE Vy", "RMSE Vtheta", "Total RMSE"]
    
    for i, label in enumerate(["Vx", "Vy", "Vtheta"]):
        row = [f"RMSE {label}"]
        for name in ["RL Only", "RL + MPC", "RL + MPC + Aug"]:
            res = results[name]
            # Calculate RMSE for this component
            # Skip first 0.5s to allow for initial transient
            mask = res["time"] > 0.5
            if not np.any(mask):
                mask = np.ones_like(res["time"], dtype=bool)
                
            error = res["actual"][mask, i] - res["target"][mask, i]
            rmse = np.sqrt(np.mean(error**2))
            row.append(f"{rmse:.4f}")
        print(f"{row[0]:<20} | {row[1]:<12} | {row[2]:<12} | {row[3]:<12}")
        
    # Total RMSE
    row = ["Total RMSE"]
    for name in ["RL Only", "RL + MPC", "RL + MPC + Aug"]:
        res = results[name]
        mask = res["time"] > 0.5
        if not np.any(mask):
            mask = np.ones_like(res["time"], dtype=bool)
            
        error = res["actual"][mask] - res["target"][mask]
        rmse = np.sqrt(np.mean(np.sum(error**2, axis=1)))
        row.append(f"{rmse:.4f}")
    print(f"{row[0]:<20} | {row[1]:<12} | {row[2]:<12} | {row[3]:<12}")
    print("="*60 + "\n")

    # Collect metrics for plotting
    metrics_data = {
        "Vx": [],
        "Vy": [],
        "Vtheta": [],
        "Total": []
    }
    scenarios = ["RL Only", "RL + MPC", "RL + MPC + Aug"]
    
    for name in scenarios:
        res = results[name]
        mask = res["time"] > 0.5
        if not np.any(mask):
            mask = np.ones_like(res["time"], dtype=bool)
            
        # Component RMSEs
        for i, key in enumerate(["Vx", "Vy", "Vtheta"]):
            error = res["actual"][mask, i] - res["target"][mask, i]
            rmse = np.sqrt(np.mean(error**2))
            metrics_data[key].append(rmse)
            
        # Total RMSE
        error = res["actual"][mask] - res["target"][mask]
        rmse = np.sqrt(np.mean(np.sum(error**2, axis=1)))
        metrics_data["Total"].append(rmse)

    # Plotting Trajectories
    print("\nPlotting results...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    labels = ["Vx (m/s)", "Vy (m/s)", "Vtheta (rad/s)"]
    
    for i in range(3):
        ax = axes[i]
        # Plot target
        ax.plot(results["RL Only"]["time"], results["RL Only"]["target"][:, i], 
                'k--', label="Target", linewidth=2)
        
        # Plot actuals
        for name, res in results.items():
            ax.plot(res["time"], res["actual"][:, i], label=name)
            
        ax.set_ylabel(labels[i])
        ax.grid(True)
        if i == 0:
            ax.legend()
            
    axes[2].set_xlabel("Time (s)")
    plt.suptitle(f"G1 Velocity Tracking Performance\nTarget: {target_vel}")
    
    output_path = "g1_tracking_comparison.png"
    plt.savefig(output_path)
    print(f"Trajectory plot saved to {output_path}")
    
    # Plotting Bar Graph
    print("Plotting metrics bar graph...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(scenarios))
    width = 0.2
    multiplier = 0
    
    for attribute, measurement in metrics_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fmt='%.3f')
        multiplier += 1
        
    ax.set_ylabel('RMSE')
    ax.set_title('Velocity Tracking RMSE by Scenario')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, max([max(m) for m in metrics_data.values()]) * 1.2)
    
    metrics_path = "g1_tracking_metrics.png"
    plt.savefig(metrics_path)
    print(f"Metrics plot saved to {metrics_path}")

if __name__ == "__main__":
    main()
