"""Verification script for passivity-regulated energy tank.

This script demonstrates that:
1. Residuals are scaled down when tank energy is low
2. Tank energy never goes negative (passivity guarantee)
3. When tank is full, residuals are applied at full strength
4. When tank depletes, residuals are attenuated

Passivity mechanism:
- In apply_control: scale = min(1.0, energy_available / energy_needed)
- scaled_residuals = residuals * scale
- This ensures: tank >= 0 always (passivity)
"""
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.getcwd())

from hydrax.tasks.g1.g1_soccer_augmented import G1SoccerAugmented


def main():
    print("=" * 60)
    print("Passivity-Regulated Energy Tank Verification")
    print("=" * 60)
    
    # Create task
    print("\n1. Creating G1SoccerAugmented task...")
    task = G1SoccerAugmented()
    
    print(f"   Tank capacity: {task.tank_capacity} J")
    print(f"   Tank initial: {task.tank_initial} J")
    print(f"   Recharge rate: {task.tank_recharge_rate} J/s")
    print(f"   ctrl_dt: {task.ctrl_dt} s")
    print(f"   Leg actuator kp (first 3): {np.array(task._leg_actuator_kp[:3])}")
    
    # Initialize state
    print("\n2. Initializing state...")
    mj_data = mujoco.MjData(task.mj_model)
    
    # Reset to keyframe
    key_id = mujoco.mj_name2id(task.mj_model, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(task.mj_model, mj_data, key_id)
    
    # JIT compile
    print("\n3. JIT compiling...")
    jit_step = jax.jit(task.step)
    jit_apply_control = jax.jit(task.apply_control)
    
    # Warmup
    mj_data.userdata[0] = task.tank_initial
    state = mjx.put_data(task.mj_model, mj_data)
    control = jnp.zeros(task.nu)
    state = jit_apply_control(state, control)
    state = jit_step(task.model, state)
    print("   Warmup complete")
    
    # Test: Apply large residuals and observe passivity regulation
    print("\n" + "=" * 60)
    print("Test: Passivity Regulation with Large Residuals")
    print("-" * 60)
    
    # Start with low tank to see regulation kick in
    mj_data.userdata[0] = 2.0  # Start with only 2J (20% of capacity)
    state = mjx.put_data(task.mj_model, mj_data)
    
    times = [float(state.time)]
    tank_levels = [float(state.userdata[0])]
    requested_residuals = []
    applied_residuals = []
    scale_factors = []
    
    num_steps = 200
    residual_val = 0.2  # Large residual to drain tank quickly
    
    kp = np.array(task._leg_actuator_kp)
    
    for i in range(num_steps):
        # Request large residuals
        residuals_request = jnp.full(12, residual_val)
        control = jnp.concatenate([jnp.zeros(3), residuals_request])
        
        # Apply control (this includes passivity scaling)
        state = jit_apply_control(state, control)
        
        # Check what residuals were actually applied
        actual_residuals = np.array(state.userdata[1:13])
        
        # Compute scale factor (actual / requested)
        scale = np.mean(np.abs(actual_residuals)) / residual_val if residual_val > 0 else 1.0
        
        requested_residuals.append(residual_val)
        applied_residuals.append(np.mean(np.abs(actual_residuals)))
        scale_factors.append(scale)
        
        # Step physics and update tank
        state = jit_step(task.model, state)
        
        times.append(float(state.time))
        tank_levels.append(float(state.userdata[0]))
    
    print(f"   Initial tank: 2.0 J (low to trigger regulation)")
    print(f"   Requested residual: {residual_val} rad")
    print(f"   Final tank: {tank_levels[-1]:.4f} J")
    print(f"   Min tank: {min(tank_levels):.4f} J (should be >= 0)")
    print(f"   Avg scale factor: {np.mean(scale_factors):.4f}")
    print(f"   Min scale factor: {np.min(scale_factors):.4f}")
    
    # Verify passivity
    if min(tank_levels) >= -1e-6:
        print("   ✓ PASSIVITY VERIFIED: Tank never went negative!")
    else:
        print(f"   ✗ PASSIVITY VIOLATED: Tank went to {min(tank_levels):.6f}")
    
    # Plot results
    print("\n4. Plotting results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Tank level over time
    ax1 = axes[0, 0]
    ax1.plot(times, tank_levels, 'b-', linewidth=2)
    ax1.axhline(y=task.tank_capacity, color='g', linestyle='--', alpha=0.7, label='Capacity')
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Empty (passivity limit)')
    ax1.fill_between(times, 0, tank_levels, alpha=0.3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Tank Energy (J)')
    ax1.set_title('Energy Tank Level (Passivity: tank ≥ 0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.5, task.tank_capacity + 1])
    
    # Plot 2: Requested vs Applied Residuals
    ax2 = axes[0, 1]
    ax2.plot(times[:-1], requested_residuals, 'b--', linewidth=2, label='Requested')
    ax2.plot(times[:-1], applied_residuals, 'r-', linewidth=2, label='Applied (scaled)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Residual Magnitude (rad)')
    ax2.set_title('Residual Attenuation for Passivity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Scale factor over time
    ax3 = axes[1, 0]
    ax3.plot(times[:-1], scale_factors, 'g-', linewidth=2)
    ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Full scale')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Scale Factor')
    ax3.set_title('Passivity Scale Factor (1.0 = full, <1 = attenuated)')
    ax3.set_ylim([0, 1.1])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Tank level vs Scale factor correlation
    ax4 = axes[1, 1]
    ax4.scatter(tank_levels[:-1], scale_factors, c=times[:-1], cmap='viridis', alpha=0.7)
    ax4.set_xlabel('Tank Energy (J)')
    ax4.set_ylabel('Scale Factor')
    ax4.set_title('Tank Level vs Residual Scaling')
    ax4.axvline(x=0, color='r', linestyle='--', label='Passivity limit')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Time (s)')
    
    plt.tight_layout()
    plt.savefig('passivity_verification.png', dpi=150)
    print("   Saved: passivity_verification.png")
    
    print("\n" + "=" * 60)
    print("Passivity Verification Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
