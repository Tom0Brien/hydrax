import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.tasks.g1.g1_navigation import G1Navigation


class G1SoccerAugmented(G1Navigation):
    """G1 humanoid soccer task: push the ball to the goal position.
    
    Extends G1Navigation with augmented action space:
    - Actions 0-2: velocity commands (vx, vy, vtheta) for RL policy
    - Actions 3-14: residual adjustments for first 12 joints (leg joints)
    
    This hierarchical control enables fine-tuning leg movements for kicking
    while maintaining stable locomotion from the trained RL policy.
    """
    
    def __init__(self):
        """Initialize G1 soccer task with augmented action space."""
        super().__init__()
        
        # Get soccer ball body ID
        self._soccer_ball_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "soccer_ball"
        )
        if self._soccer_ball_id == -1:
            raise ValueError("Soccer ball body not found in model")
        
        # Soccer ball qpos indices (after robot: 7 root + 29 joints = 36)
        # Ball has freejoint: 7 DOFs (3 pos + 4 quat)
        self._ball_qpos_start = 36
        self._ball_qpos_end = 43
        
        # Get goal marker body ID
        self._goal_marker_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker"
        )
        if self._goal_marker_id == -1:
            raise ValueError("Goal marker body not found in model")
        
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
    
    def _get_ball_position(self, state: mjx.Data) -> jax.Array:
        """Get soccer ball XY position."""
        return state.qpos[self._ball_qpos_start:self._ball_qpos_start + 2]
    
    def _get_goal_position(self, state: mjx.Data) -> jax.Array:
        """Get goal XY position from mocap body."""
        return state.mocap_pos[0, :2]
    
    def _get_robot_position(self, state: mjx.Data) -> jax.Array:
        """Get robot XY position."""
        return state.qpos[:2]
    
    def _get_robot_yaw(self, state: mjx.Data) -> jax.Array:
        """Get robot yaw angle from quaternion."""
        # Robot quaternion is at qpos[3:7] as [qw, qx, qy, qz]
        quat = state.qpos[3:7]
        # Yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
        return jnp.arctan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2]**2 + quat[3]**2)
        )
    
    def _get_torso_height(self, state: mjx.Data) -> jax.Array:
        """Get torso height above ground."""
        return state.site_xpos[self._torso_site_id, 2]
    
    def _get_ball_to_goal_error(self, state: mjx.Data) -> jax.Array:
        """Position error from ball to goal."""
        ball_pos = self._get_ball_position(state)
        goal_pos = self._get_goal_position(state)
        return ball_pos - goal_pos
    
    def _get_desired_robot_position(self, state: mjx.Data) -> jax.Array:
        """Get desired robot position.
        
        Positions robot 0.5m behind ball along ball-to-goal vector.
        This enables approaching from behind to kick toward goal.
        """
        ball_pos = self._get_ball_position(state)
        goal_pos = self._get_goal_position(state)
        
        # Vector from ball to goal
        ball_to_goal = goal_pos - ball_pos
        ball_to_goal_dist = jnp.linalg.norm(ball_to_goal) + 1e-6
        ball_to_goal_dir = ball_to_goal / ball_to_goal_dist
        
        # Desired: 0.5m behind ball (opposite direction from goal)
        standoff_distance = 0.5
        desired_pos = ball_pos - standoff_distance * ball_to_goal_dir
        
        return desired_pos
    
    def _get_robot_positioning_error(
        self, state: mjx.Data
    ) -> tuple[jax.Array, jax.Array]:
        """Get robot positioning and orientation errors.
        
        Returns:
            position_error: Distance from desired position
            orientation_error: Yaw angle error (approach -> kick)
        """
        robot_pos = self._get_robot_position(state)
        robot_yaw = self._get_robot_yaw(state)
        
        goal_pos = self._get_goal_position(state)
        desired_pos = self._get_desired_robot_position(state)
        
        # Position error: distance from desired position
        position_error = robot_pos - desired_pos
        
        # Orientation: interpolate based on distance
        dist_to_desired = jnp.linalg.norm(position_error)
        
        # Far (>1m): orient to approach point
        # Close (<0.3m): orient to goal (for kicking)
        approach_angle = jnp.arctan2(
            desired_pos[1] - robot_pos[1],
            desired_pos[0] - robot_pos[0]
        )
        kick_angle = jnp.arctan2(
            goal_pos[1] - robot_pos[1],
            goal_pos[0] - robot_pos[0]
        )
        
        # Sigmoid: far -> 1 (approach), close -> 0 (kick)
        # Transition centered at 0.6m
        transition_weight = jax.nn.sigmoid(
            (dist_to_desired - 0.6) / 0.2
        )
        
        desired_yaw = (
            transition_weight * approach_angle
            + (1 - transition_weight) * kick_angle
        )
        
        # Yaw error wrapped to [-pi, pi]
        orientation_error = robot_yaw - desired_yaw
        orientation_error = jnp.arctan2(
            jnp.sin(orientation_error), jnp.cos(orientation_error)
        )
        
        return position_error, orientation_error
    
    def running_cost(
        self,
        state: mjx.Data,
        control: jax.Array
    ) -> jax.Array:
        """Cost to push ball to goal position.
        
        Cost drives six behaviors:
        1. Move ball to goal
        2. Position robot 0.5m behind ball
        3. Orient robot (approach -> kick transition)
        4. Maintain upright posture (prevent falling)
        5. Regularize velocity commands
        6. Regularize leg residuals (prevent destabilization)
        """
        # Main cost: ball to goal distance
        ball_to_goal_err = self._get_ball_to_goal_error(state)
        ball_to_goal_cost = jnp.sum(jnp.square(ball_to_goal_err))
        
        # Robot positioning: position and orientation
        pos_err, orient_err = self._get_robot_positioning_error(state)
        robot_position_cost = jnp.sum(jnp.square(pos_err))
        robot_orientation_cost = jnp.square(orient_err)
        
        # Upright posture: keep torso at target height
        height_err = self._get_torso_height(state) - self.target_height
        height_cost = jnp.square(height_err)
        
        # Control regularization
        velocity_cmd = control[:3]
        leg_residuals = control[3:15]
        
        velocity_cost = jnp.sum(jnp.square(velocity_cmd))
        residual_cost = jnp.sum(jnp.square(leg_residuals))
        
        # Weighted combination
        # Ball to goal (1.0), position (0.3), orient (0.2),
        # height (0.5 - prevent falling), velocity (0.01),
        # residuals (0.05 - prevent instability)
        return (
            ball_to_goal_cost
            + 0.3 * robot_position_cost
            + 0.2 * robot_orientation_cost
            + 0.5 * height_cost
            + 0.01 * velocity_cost
            + 0.05 * residual_cost
        )
    
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost (no control regularization)."""
        return self.running_cost(state, jnp.zeros(self.nu))

