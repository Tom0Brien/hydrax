import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.tasks.g1.g1_locomotion import G1Locomotion

class G1VelocityTracking(G1Locomotion):
    """G1 humanoid velocity tracking task.
    
    The goal is to track a desired velocity vector (vx, vy, vtheta).
    The control inputs are the velocity commands passed to the RL policy.
    """
    
    def __init__(self, target_velocity: jax.Array = jnp.array([0.5, 0.0, 0.0])):
        """Initialize task with target velocity.
        
        Args:
            target_velocity: Desired velocity (vx, vy, vtheta) in base frame.
        """
        super().__init__()
        self.target_velocity = jnp.asarray(target_velocity)

    def running_cost(
        self,
        state: mjx.Data,
        control: jax.Array
    ) -> jax.Array:
        """Cost to track desired velocity.
        
        Args:
            state: Current mjx.Data state
            control: Velocity command (vx, vy, vtheta) sent to policy
            
        Returns:
            Scalar cost
        """
        # Get robot quaternion (base to world)
        # qpos[3:7] is [qw, qx, qy, qz]
        quat = state.qpos[3:7]
        
        # Inverse quaternion (world to base)
        # For unit quaternion, inverse is conjugate: [w, -x, -y, -z]
        quat_inv = jnp.array([quat[0], -quat[1], -quat[2], -quat[3]])
        
        # Get linear velocity in world frame
        v_world = state.qvel[:3]
        
        # Rotate to base frame
        v_base = mjx._src.math.rotate(v_world, quat_inv)
        
        # Get angular velocity in world frame
        w_world = state.qvel[3:6]
        
        # Rotate to base frame
        w_base = mjx._src.math.rotate(w_world, quat_inv)
        
        # Construct actual velocity vector (vx, vy, vtheta)
        actual_vel = jnp.array([v_base[0], v_base[1], w_base[2]])
        
        # Velocity tracking cost
        vel_error = actual_vel - self.target_velocity
        vel_cost = jnp.sum(vel_error**2)
    
        ctrl_cost = jnp.sum((control - self.target_velocity)**2) * 0.1
        
        return vel_cost + ctrl_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost."""
        return self.running_cost(state, self.target_velocity)
