import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.tasks.g1.g1_velocity_tracking import G1VelocityTracking

class G1VelocityTrackingAugmented(G1VelocityTracking):
    """G1 velocity tracking with augmented action space (residuals).
    
    Extends G1VelocityTracking to include residual adjustments for leg joints.
    Action space (nu=15):
    - 0-2: Velocity commands (vx, vy, vtheta) for RL policy
    - 3-14: Residual adjustments for first 12 joints (legs)
    """
    
    def __init__(self, target_velocity: jax.Array = jnp.array([0.5, 0.0, 0.0])):
        """Initialize task with augmented action space."""
        super().__init__(target_velocity=target_velocity)
        
        # Augmented action space: 3 velocity + 12 leg residuals
        self.nu = 15
        self.u_min = jnp.concatenate([
            jnp.array([-1.0, -1.0, -1.0]),  # Velocity bounds
            jnp.full(12, -0.3)  # Residual bounds (±0.3 rad)
        ])
        self.u_max = jnp.concatenate([
            jnp.array([1.0, 1.0, 1.0]),  # Velocity bounds
            jnp.full(12, 0.3)  # Residual bounds (±0.3 rad)
        ])
        
        # Leg joint indices (first 12 of 29 robot joints)
        self._leg_joint_count = 12

    def apply_control(
        self,
        state: mjx.Data,
        control: jax.Array
    ) -> mjx.Data:
        """Apply hierarchical control with RL policy + leg residuals.
        
        Args:
            state: Current mjx.Data state
            control: 15D control [vx, vy, vtheta, leg_residuals_0-11]
            
        Returns:
            State with updated ctrl (motor targets + residuals)
        """
        # Extract velocity commands for RL policy (first 3)
        velocity_cmd = control[:3]
        
        # Extract leg joint residuals (next 12)
        leg_residuals = control[3:15]
        
        # Get base motor targets from RL policy using parent's method
        # Note: G1VelocityTracking inherits apply_control from G1Navigation
        state = super().apply_control(state, velocity_cmd)
        
        # Add residuals to first 12 motor targets (leg joints)
        motor_targets = state.ctrl
        leg_indices = slice(0, self._leg_joint_count)
        motor_targets_with_residuals = motor_targets.at[leg_indices].add(
            leg_residuals
        )
        
        # Clip to joint limits
        motor_targets_with_residuals = jnp.clip(
            motor_targets_with_residuals,
            self.mj_model.jnt_range[1:30, 0],
            self.mj_model.jnt_range[1:30, 1],
        )
        
        return state.replace(ctrl=motor_targets_with_residuals)

    def running_cost(
        self,
        state: mjx.Data,
        control: jax.Array
    ) -> jax.Array:
        """Cost to track desired velocity + residual regularization."""
        # Split control
        velocity_cmd = control[:3]
        leg_residuals = control[3:15]
        
        # 1. Velocity tracking cost (from parent logic)
        # We can't call super().running_cost directly because it expects 3D control
        # and does regularization on it. So we reimplement the tracking part.
        
        # Get robot quaternion (base to world)
        quat = state.qpos[3:7]
        quat_inv = jnp.array([quat[0], -quat[1], -quat[2], -quat[3]])
        
        # Get linear/angular velocity in base frame
        v_world = state.qvel[:3]
        v_base = mjx._src.math.rotate(v_world, quat_inv)
        w_world = state.qvel[3:6]
        w_base = mjx._src.math.rotate(w_world, quat_inv)
        
        actual_vel = jnp.array([v_base[0], v_base[1], w_base[2]])
        
        # Velocity tracking cost
        vel_error = actual_vel - self.target_velocity
        vel_cost = jnp.sum(vel_error**2)
        
        # 2. Control regularization
        # Regularize velocity command towards target (as in parent)
        cmd_cost = jnp.sum((velocity_cmd - self.target_velocity)**2) * 0.01
        
        # Regularize residuals towards zero
        residual_cost = jnp.sum(leg_residuals**2) * 0.0
        
        return vel_cost + cmd_cost + residual_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost."""
        # Pass zero residuals for terminal cost calculation
        dummy_control = jnp.concatenate([self.target_velocity, jnp.zeros(12)])
        return self.running_cost(state, dummy_control)
