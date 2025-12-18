import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.tasks.g1.g1_navigation import G1Navigation


class G1NavigationAugmented(G1Navigation):
    """G1 humanoid navigation task with augmented action space.
    
    Extends G1Navigation with augmented action space:
    - Actions 0-2: velocity commands (vx, vy, vtheta) for RL policy
    - Actions 3-14: residual adjustments for first 12 joints (leg joints)
    
    This hierarchical control enables fine-tuning leg movements for precise
    navigation while maintaining stable locomotion from the trained RL policy.
    """
    
    def __init__(self):
        """Initialize G1 navigation task with augmented action space."""
        super().__init__()
        
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
        # Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        # Right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        self._leg_joint_count = 12
        
        # Get torso site for height tracking
        self._torso_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso"
        )
        if self._torso_site_id == -1:
            raise ValueError("Torso site 'imu_in_torso' not found")
        
        # Target torso height (standing height)
        self.target_height = 0.75
    
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
    
    def _get_robot_position(self, state: mjx.Data) -> jax.Array:
        """Get robot XY position."""
        return state.qpos[:2]
    
    def _get_robot_quaternion(self, state: mjx.Data) -> jax.Array:
        """Get robot orientation quaternion."""
        return state.qpos[3:7]  # [qw, qx, qy, qz]
    
    def _get_goal_position(self, state: mjx.Data) -> jax.Array:
        """Get goal XY position from mocap body."""
        return state.mocap_pos[0, :2]
    
    def _get_goal_quaternion(self, state: mjx.Data) -> jax.Array:
        """Get goal orientation quaternion from mocap body."""
        return state.mocap_quat[0]
    
    def _get_torso_height(self, state: mjx.Data) -> jax.Array:
        """Get torso height above ground."""
        return state.site_xpos[self._torso_site_id, 2]
    
    def running_cost(
        self,
        state: mjx.Data,
        control: jax.Array
    ) -> jax.Array:
        """Cost to reach goal pose with augmented control.
        
        Cost drives six behaviors:
        1. Minimize position error to goal
        2. Minimize yaw orientation error to goal
        3. Keep robot upright (minimize roll/pitch tilt)
        4. Maintain torso height (prevent falling)
        5. Regularize velocity commands
        6. Regularize leg residuals (prevent destabilization)
        """
        # Get goal from mocap body
        goal_pos = self._get_goal_position(state)
        goal_quat = self._get_goal_quaternion(state)
        
        # Current robot pose
        robot_pos = self._get_robot_position(state)
        robot_quat = self._get_robot_quaternion(state)
        
        # Position error
        pos_error = robot_pos - goal_pos
        position_cost = jnp.sum(jnp.square(pos_error))
        
        # Orientation error (quaternion difference)
        quat_error = mjx._src.math.quat_sub(robot_quat, goal_quat)
        # Only care about yaw (z-axis rotation)
        orientation_cost = jnp.square(quat_error[3])  # qz component
        
        # Upright orientation: penalize tilt from vertical
        # Rotate the upright vector [0, 0, 1] by robot quaternion
        # If robot is upright, result should be [0, 0, 1]
        # If tilted, x and y components will be non-zero
        upright_vec = jnp.array([0.0, 0.0, 1.0])
        rotated_upright = mjx._src.math.rotate(upright_vec, robot_quat)
        upright_cost = jnp.sum(jnp.square(rotated_upright[:2]))  # x^2 + y^2
        
        # Upright posture: keep torso at target height
        height_err = self._get_torso_height(state) - self.target_height
        height_cost = jnp.square(height_err)
        
        # Control regularization
        velocity_cmd = control[:3]
        leg_residuals = control[3:15]
        
        velocity_cost = jnp.sum(jnp.square(velocity_cmd))
        residual_cost = jnp.sum(jnp.square(leg_residuals))
        
        # Weighted combination
        # Position (1.0), yaw orientation (1.0), upright (2.0 - critical for stability),
        # height (0.5 - prevent falling), velocity (0.01), residuals (0.05 - prevent instability)
        return (
            position_cost
            + orientation_cost
            + 2.0 * upright_cost
            + 0.5 * height_cost
            + 0.01 * velocity_cost
            + 0.05 * residual_cost
        )
    
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost (no control regularization)."""
        return self.running_cost(state, jnp.zeros(self.nu))

