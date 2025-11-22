import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.tasks.g1.g1_locomotion import G1Locomotion


class G1Soccer(G1Locomotion):
    """G1 humanoid soccer task: push the ball to the goal position.
    
    Extends G1Locomotion to add a soccer ball pushing objective.
    The goal is defined by the mocap body position (like PushT).
    """
    
    def __init__(self):
        """Initialize G1 soccer task."""
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
        
        Cost drives four behaviors:
        1. Move ball to goal
        2. Position robot 0.5m behind ball
        3. Orient robot (approach -> kick transition)
        4. Regularize controls
        """
        # Main cost: ball to goal distance
        ball_to_goal_err = self._get_ball_to_goal_error(state)
        ball_to_goal_cost = jnp.sum(jnp.square(ball_to_goal_err))
        
        # Robot positioning: position and orientation
        pos_err, orient_err = self._get_robot_positioning_error(state)
        robot_position_cost = jnp.sum(jnp.square(pos_err))
        robot_orientation_cost = jnp.square(orient_err)
        
        # Control regularization
        ctrl_cost = jnp.sum(jnp.square(control))
        
        # Weighted combination
        # Ball to goal (1.0), position (0.3), orient (0.2), ctrl (0.01)
        return (
            ball_to_goal_cost
            + 0.3 * robot_position_cost
            + 0.2 * robot_orientation_cost
            + 0.01 * ctrl_cost
        )
    
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost (no control regularization)."""
        return self.running_cost(state, jnp.zeros(self.nu))

