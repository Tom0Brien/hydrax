"""Example demonstrating iCEM on pendulum swingup.

This script shows how to use the iCEM (improved Cross-Entropy Method) controller
and compares it with standard CEM to demonstrate the sample efficiency improvements.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mujoco import mjx

from hydrax.algs.cem import CEM
from hydrax.algs.icem import iCEM
from hydrax.tasks.pendulum import Pendulum


def run_closed_loop_mpc(controller, initial_state, sim_duration=5.0, name="Controller"):
    """Run closed-loop MPC simulation with receding horizon (JAX-accelerated).
    
    Uses jax.lax.scan for efficient JIT compilation of the entire simulation loop.
    
    Args:
        controller: The MPC controller (CEM or iCEM)
        initial_state: Initial state of the system
        sim_duration: Total simulation time in seconds
        name: Name of the controller for logging
        
    Returns:
        qpos: Array of positions over time, shape (num_steps+1, state_dim)
        qvel: Array of velocities over time, shape (num_steps+1, state_dim)
        actions: Array of actions executed, shape (num_steps, action_dim)
        costs: Array of instantaneous costs, shape (num_steps,)
        planning_costs: Array of predicted costs from MPC, shape (num_steps,)
        total_cost: Total accumulated cost
    """
    # Simulation setup
    ctrl_dt = controller.task.ctrl_dt
    num_steps = int(sim_duration / ctrl_dt)
    
    print(f"Compiling and running {name} in closed loop for {sim_duration}s ({num_steps} steps)...")
    
    # Initialize parameters
    params = controller.init_params(seed=42)
    
    # Get action bounds
    act_min = controller.task.mj_model.actuator_ctrlrange[:, 0]
    act_max = controller.task.mj_model.actuator_ctrlrange[:, 1]
    
    def scan_body(carry, _unused):
        """Single MPC step: optimize, execute, step physics."""
        state, params_carry = carry
        
        # MPC optimization
        params_new, rollouts = controller.optimize(state, params_carry)
        
        # Get action (first from plan)
        action = controller.get_action(params_new, t=0.0)
        
        # Clip to valid range
        action = jnp.clip(action, act_min, act_max)
        
        # Predicted cost from MPC
        planning_cost = jnp.min(jnp.sum(rollouts.costs, axis=1))
        
        # Apply control and step physics
        state_with_ctrl = controller.task.apply_control(state, action)
        next_state = controller.task.step(controller.task.model, state_with_ctrl)
        
        # Actual cost
        actual_cost = controller.task.running_cost(next_state, action)
        
        # Output for this timestep
        outputs = {
            'qpos': next_state.qpos,
            'qvel': next_state.qvel,
            'action': action,
            'cost': actual_cost,
            'planning_cost': planning_cost,
        }
        
        return (next_state, params_new), outputs
    
    # Run entire simulation with scan (JIT-compiled!)
    scan_fn = jax.jit(lambda s, p: jax.lax.scan(
        scan_body, (s, p), None, length=num_steps
    ))
    
    (final_state, final_params), outputs = scan_fn(initial_state, params)
    
    # Extract arrays
    qpos_traj = jnp.concatenate([initial_state.qpos[None], outputs['qpos']], axis=0)
    qvel_traj = jnp.concatenate([initial_state.qvel[None], outputs['qvel']], axis=0)
    actions = outputs['action']
    costs = outputs['cost']
    planning_costs = outputs['planning_cost']
    
    total_cost = float(jnp.sum(costs))
    total_planning_cost = float(jnp.sum(planning_costs))
    
    print(f"  {name}: Actual cost = {total_cost:.2f}, Predicted cost = {total_planning_cost:.2f}")
    
    return qpos_traj, qvel_traj, actions, costs, planning_costs, total_cost


def main():
    """Main function to run the comparison."""
    # Setup task
    task = Pendulum()
    state = mjx.make_data(task.model)

    # Common parameters
    common_params = {
        "task": task,
        "num_samples": 32,
        "num_elites": 8,
        "sigma_start": 1.0,
        "sigma_min": 0.1,
        "plan_horizon": 1.0,
        "spline_type": "zero",
        "num_knots": 11,
        "iterations": 3,
    }

    sim_duration = 5.0  # 5 seconds of simulation

    print("=" * 60)
    print("iCEM vs CEM Closed-Loop MPC Comparison")
    print("=" * 60)

    # Standard CEM
    print("\n1. Running standard CEM...")
    cem = CEM(**common_params)
    cem_qpos, cem_qvel, cem_actions, cem_costs, cem_planning_costs, cem_total_cost = run_closed_loop_mpc(
        cem, state, sim_duration=sim_duration, name="CEM"
    )

    # iCEM with colored noise
    print("\n2. Running iCEM (with all improvements)...")
    icem = iCEM(
        **common_params,
        alpha=0.1,
        noise_beta=2.0,
        fraction_elites_reused=0.3,
        shift_elites=True,
        keep_elites=False,
        use_best_action=True,
    )
    icem_qpos, icem_qvel, icem_actions, icem_costs, icem_planning_costs, icem_total_cost = run_closed_loop_mpc(
        icem, state, sim_duration=sim_duration, name="iCEM"
    )

    # iCEM with white noise (for comparison)
    print("\n3. Running iCEM with white noise (β=0)...")
    icem_white = iCEM(
        **common_params,
        alpha=0.1,
        noise_beta=0.0,  # White noise
        fraction_elites_reused=0.3,
        shift_elites=True,
        keep_elites=False,
        use_best_action=True,
    )
    icem_white_qpos, icem_white_qvel, icem_white_actions, icem_white_costs, icem_white_planning_costs, icem_white_total_cost = (
        run_closed_loop_mpc(icem_white, state, sim_duration=sim_duration, name="iCEM (white)")
    )

    # Create visualization
    print("\n4. Creating visualization...")
    fig = plt.figure(figsize=(14, 10))

    # Create time array
    times = jnp.arange(len(cem_costs)) * task.ctrl_dt

    # Cost comparison over time
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(times, cem_costs, label="CEM (actual)", alpha=0.7, linewidth=2)
    ax1.plot(times, icem_costs, label="iCEM (β=2.0)", alpha=0.7, linewidth=2)
    ax1.plot(times, icem_white_costs, label="iCEM (β=0.0)", alpha=0.7, linewidth=2, linestyle="--")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Instantaneous Cost")
    ax1.set_title("Tracking Performance Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # State trajectories - Angle
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(times, cem_qpos[:-1, 0], label="CEM", alpha=0.7, linewidth=2)
    ax2.plot(times, icem_qpos[:-1, 0], label="iCEM (β=2.0)", alpha=0.7, linewidth=2)
    ax2.plot(times, icem_white_qpos[:-1, 0], label="iCEM (β=0.0)", alpha=0.7, linewidth=2, linestyle="--")
    ax2.axhline(jnp.pi, color="black", linestyle=":", alpha=0.5, label="Upright")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(r"$\theta$ (rad)")
    ax2.set_title("Pendulum Angle")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Velocity trajectories
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(times, cem_qvel[:-1, 0], label="CEM", alpha=0.7, linewidth=2)
    ax3.plot(times, icem_qvel[:-1, 0], label="iCEM (β=2.0)", alpha=0.7, linewidth=2)
    ax3.plot(times, icem_white_qvel[:-1, 0], label="iCEM (β=0.0)", alpha=0.7, linewidth=2, linestyle="--")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel(r"$\dot{\theta}$ (rad/s)")
    ax3.set_title("Angular Velocity")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Control trajectories
    ax4 = plt.subplot(2, 2, 4)
    ax4.step(times, cem_actions[:, 0], where="post", label="CEM", alpha=0.7, linewidth=2)
    ax4.step(times, icem_actions[:, 0], where="post", label="iCEM (β=2.0)", alpha=0.7, linewidth=2)
    ax4.step(times, icem_white_actions[:, 0], where="post", label="iCEM (β=0.0)", alpha=0.7, linewidth=2, linestyle="--")
    ax4.axhline(-1.0, color="black", linestyle="--", alpha=0.3)
    ax4.axhline(1.0, color="black", linestyle="--", alpha=0.3)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Torque (N⋅m)")
    ax4.set_title("Control Input")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("icem_vs_cem_closed_loop.png", dpi=150)
    print("   Saved plot to: icem_vs_cem_closed_loop.png")
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Simulation Duration: {sim_duration}s")
    print(f"\nTotal Cost (lower is better):")
    print(f"  CEM:              {cem_total_cost:.2f}")
    print(f"  iCEM (β=2.0):     {icem_total_cost:.2f}")
    print(f"  iCEM (β=0.0):     {icem_white_total_cost:.2f}")
    print(f"\nCost Improvement vs CEM:")
    print(f"  iCEM (β=2.0):     {(cem_total_cost - icem_total_cost) / cem_total_cost * 100:.1f}%")
    print(f"  iCEM (β=0.0):     {(cem_total_cost - icem_white_total_cost) / cem_total_cost * 100:.1f}%")
    
    # Final state analysis
    cem_final_angle = float(cem_qpos[-1, 0])
    icem_final_angle = float(icem_qpos[-1, 0])
    target_angle = jnp.pi
    
    print(f"\nFinal Angle (target = {target_angle:.2f} rad):")
    print(f"  CEM:              {cem_final_angle:.2f} rad (error: {abs(cem_final_angle - target_angle):.3f})")
    print(f"  iCEM (β=2.0):     {icem_final_angle:.2f} rad (error: {abs(icem_final_angle - target_angle):.3f})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
