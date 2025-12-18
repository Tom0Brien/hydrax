import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.tasks.g1.g1_navigation import G1Navigation


class G1Goalkeeper(G1Navigation):
    """G1 humanoid goalkeeper task: intercept and stop ball from goal.
    
    Extends G1Navigation with augmented action space:
    - Actions 0-2: velocity commands (vx, vy, vtheta) for RL policy
    - Actions 3-31: residual adjustments for all 29 robot joints
    
    This hierarchical control enables diving, reaching, and blocking with
    full body control while maintaining stable locomotion from trained policy.
    """
    
    def __init__(self):
        """Initialize G1 goalkeeper task with augmented action space."""
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
        
        # Soccer ball qvel indices (after robot: 6 root + 29 joints = 35)
        # Ball has 6 DOFs velocity (3 linear + 3 angular)
        self._ball_qvel_start = 35
        self._ball_qvel_end = 41
        
        # Get goal box geom ID for checking if ball is in goal
        self._goal_box_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "goal_box_red"
        )
        if self._goal_box_id == -1:
            raise ValueError("Goal box geom not found in model")
        
        # Augmented action space: 3 velocity + 29 joint residuals
        self.nu = 32
        self.u_min = jnp.concatenate([
            jnp.array([-1.0, -1.0, -1.0]),  # Velocity bounds
            jnp.full(29, -0.3)  # Residual bounds (±0.3 rad)
        ])
        self.u_max = jnp.concatenate([
            jnp.array([1.0, 1.0, 1.0]),  # Velocity bounds
            jnp.full(29, 0.3)  # Residual bounds (±0.3 rad)
        ])
        
        # Total robot joints (legs + torso + arms)
        self._num_joints = 29
        
        # Joint grouping: legs (0-11), upper body (12-28)
        self._leg_joint_count = 12
        self._upper_body_joint_count = 17
        
        # Get torso site for height tracking
        self._torso_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso"
        )
        if self._torso_site_id == -1:
            raise ValueError("Torso site 'imu_in_torso' not found")
        
        # Target torso height (standing height)
        self.target_height = 0.75
        
        # Goal line position (x-coordinate of red goal)
        self.goal_line_x = -4.5
    
    def apply_control(
        self,
        state: mjx.Data,
        control: jax.Array
    ) -> mjx.Data:
        """Apply hierarchical control with RL policy + joint residuals.
        
        Strategy:
        - Legs (0-11): RL policy + residuals (stable locomotion)
        - Upper body (12-28): default pose + residuals (free for blocking)
        
        Args:
            state: Current mjx.Data state
            control: 32D control [vx, vy, vtheta, joint_residuals_0-28]
            
        Returns:
            State with updated ctrl (motor targets + residuals)
        """
        # Extract velocity commands for RL policy (first 3)
        velocity_cmd = control[:3]
        
        # Extract joint residuals for all 29 joints (next 29)
        joint_residuals = control[3:32]
        
        # Get base motor targets from RL policy using parent's method
        # This sets targets for all 29 joints based on the policy
        state = super().apply_control(state, velocity_cmd)
        
        # Apply residuals differently for legs vs upper body
        motor_targets = state.ctrl
        
        # Legs (0-11): RL policy + residuals
        leg_slice = slice(0, self._leg_joint_count)
        motor_targets = motor_targets.at[leg_slice].add(
            joint_residuals[:self._leg_joint_count]
        )
        
        # Upper body (12-28): default pose + residuals (ignore RL policy)
        upper_slice = slice(self._leg_joint_count, self._num_joints)
        motor_targets = motor_targets.at[upper_slice].set(
            self._default_pose[upper_slice] +
            joint_residuals[self._leg_joint_count:]
        )
        
        # Clip to joint limits
        motor_targets = jnp.clip(
            motor_targets,
            self.mj_model.jnt_range[1:30, 0],
            self.mj_model.jnt_range[1:30, 1],
        )
        
        return state.replace(ctrl=motor_targets)
    
    def _get_ball_position(self, state: mjx.Data) -> jax.Array:
        """Get soccer ball XYZ position."""
        return state.qpos[self._ball_qpos_start:self._ball_qpos_start + 3]
    
    def _get_ball_velocity(self, state: mjx.Data) -> jax.Array:
        """Get soccer ball linear velocity."""
        return state.qvel[self._ball_qvel_start:self._ball_qvel_start + 3]
    
    def _get_robot_position(self, state: mjx.Data) -> jax.Array:
        """Get robot XYZ position."""
        return state.qpos[:3]
    
    def _get_torso_height(self, state: mjx.Data) -> jax.Array:
        """Get torso height above ground."""
        return state.site_xpos[self._torso_site_id, 2]
    
    def _is_ball_in_goal(self, state: mjx.Data) -> jax.Array:
        """Check if ball has crossed goal line into red goal box."""
        ball_pos = self._get_ball_position(state)
        # Red goal is at x=-4.5, box extends from -4.5 to -4.8
        in_goal = (ball_pos[0] < self.goal_line_x) & (
            ball_pos[0] > self.goal_line_x - 0.6
        )
        in_goal = in_goal & (jnp.abs(ball_pos[1]) < 1.2)  # Within goal width
        in_goal = in_goal & (ball_pos[2] < 1.8)  # Below crossbar
        return in_goal.astype(jnp.float32)
    
    def running_cost(
        self,
        state: mjx.Data,
        control: jax.Array
    ) -> jax.Array:
        """Cost to prevent ball from entering goal.
        
        Simplified cost that lets MPC discover optimal interception:
        1. Prevent ball from crossing goal line (PRIMARY)
        2. Keep ball away from goal (distance-based)
        3. Reduce ball velocity (stop the ball)
        4. Position robot between ball and goal
        5. Maintain upright posture
        6. Regularize controls
        """
        # PRIMARY: Huge penalty if ball is in goal
        ball_in_goal = self._is_ball_in_goal(state)
        goal_penalty = 1000.0 * ball_in_goal
        
        # Ball distance from goal (exponential penalty as ball approaches)
        ball_pos = self._get_ball_position(state)
        distance_to_goal = ball_pos[0] - self.goal_line_x
        # Exponential cost: very high near goal, lower far away
        ball_distance_cost = jnp.exp(-distance_to_goal / 2.0)
        
        # Ball velocity (want to stop it, especially when close to goal)
        ball_vel = self._get_ball_velocity(state)
        ball_velocity_cost = jnp.sum(jnp.square(ball_vel))
        # Weight velocity more when ball is close to goal
        proximity_weight = jnp.exp(-distance_to_goal / 1.0)
        weighted_velocity_cost = ball_velocity_cost * (
            1.0 + 2.0 * proximity_weight
        )
        
        # Robot-ball distance (want to be close, especially when ball near goal)
        robot_pos = self._get_robot_position(state)[:2]
        ball_pos_xy = ball_pos[:2]
        robot_to_ball_dist = jnp.linalg.norm(robot_pos - ball_pos_xy)
        # Higher weight when ball is close to goal
        distance_to_ball_cost = jnp.square(robot_to_ball_dist) * (
            1.0 + proximity_weight
        )
        
        # Robot should be between ball and goal (alignment cost)
        goal_pos = jnp.array([self.goal_line_x, 0.0])
        ball_to_goal = goal_pos - ball_pos_xy
        ball_to_robot = robot_pos - ball_pos_xy
        # Negative dot product means robot is between ball and goal
        alignment = jnp.dot(ball_to_goal, ball_to_robot)
        alignment_cost = jnp.maximum(0.0, -alignment) * 0.1
        
        # Upright posture: keep torso at target height
        height_err = self._get_torso_height(state) - self.target_height
        height_cost = jnp.square(height_err)
        
        # Control regularization
        velocity_cmd = control[:3]
        joint_residuals = control[3:32]
        
        # Split residuals: penalize leg residuals, allow upper body freedom
        leg_residuals = joint_residuals[:self._leg_joint_count]
        # upper_body_residuals = joint_residuals[self._leg_joint_count:]
        
        velocity_cost = jnp.sum(jnp.square(velocity_cmd))
        leg_residual_cost = jnp.sum(jnp.square(leg_residuals))
        # Don't penalize upper body residuals - free to use for blocking
        
        # Weighted combination - let MPC discover the strategy
        return (
            goal_penalty  # 1000.0 if ball in goal
            + 5.0 * ball_distance_cost  # Keep ball away (exponential)
            + 0.3 * weighted_velocity_cost  # Stop ball (weight near goal)
            + 0.3 * distance_to_ball_cost  # Get close (weight near goal)
            + alignment_cost  # Be between ball and goal
            + 0.0 * height_cost  # Stay upright
            + 0.00 * velocity_cost  # Regularize velocity
            + 0.05 * leg_residual_cost  # Regularize leg residuals only
        )
    
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost (no control regularization)."""
        return self.running_cost(state, jnp.zeros(self.nu))

