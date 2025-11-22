import functools
from typing import Any

import jax
import jax.numpy as jnp
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from mujoco import mjx

from hydrax.task_base import Task

# Checkpoint path for trained G1 joystick policy
CHECKPOINT_PATH = (
    "/home/tom/OneDrive/Phd/Papers/GPC/mujoco_playground/logs/"
    "G1JoystickFlatTerrain-20251120-162949/checkpoints/000202342400"
)
ENV_NAME = "G1JoystickFlatTerrain"

class G1Locomotion(Task):
    """G1 humanoid locomotion task using trained joystick policy.
    
    This task wraps the mujoco_playground G1 environment and its trained
    policy to enable MPC planning with velocity commands.
    """
    
    def __init__(self):
        """Initialize G1 locomotion task with environment and policy."""
        # Import here to avoid circular dependencies  # noqa: E402
        from mujoco_playground import registry, wrapper  # noqa: E402
        from mujoco_playground.config import locomotion_params  # noqa: E402
        
        # Create environment using registry (matches run_g1_interactive.py)
        env_cfg = registry.get_default_config(ENV_NAME)
        env_cfg["impl"] = "jax"
        self.env = registry.load(ENV_NAME, config=env_cfg)
        
        # Initialize Task base class with the environment's model
        super().__init__(self.env.mj_model)

        # Override nu to 3 (vx, vy, vtheta)
        self.nu = 3
        self.u_min = jnp.array([-1.0, -1.0, -1.0])
        self.u_max = jnp.array([1.0, 1.0, 1.0])

        # Load RL policy (matches run_g1_interactive.py pattern)
        self.inference_fn = self._load_policy(
            locomotion_params, wrapper
        )
        
        # Store environment attributes needed for control
        self._feet_floor_found_sensor = (
            self.env._feet_floor_found_sensor
        )
        self._default_pose = self.env._default_pose
        self._action_scale = self.env._config.action_scale
        
        # Set control frequency to 50Hz (matching RL policy training)
        self.ctrl_dt = 0.02
        # Recompute n_substeps after changing ctrl_dt
        self.n_substeps = max(1, round(self.ctrl_dt / self.dt))

    def _load_policy(
        self,
        locomotion_params: Any,
        wrapper: Any
    ) -> Any:
        """Load the RL policy from checkpoint.
        
        Uses the exact pattern from run_g1_interactive.py to properly
        load normalizer parameters and create the inference function.
        
        Returns:
            Jitted inference function
        """
        # Get PPO config
        ppo_params = locomotion_params.brax_ppo_config(ENV_NAME)
        ppo_params.num_timesteps = 0  # Just load, don't train
        
        # Set up network factory (matches run_g1_interactive.py)
        network_fn = ppo_networks.make_ppo_networks
        if hasattr(ppo_params, "network_factory"):
            network_factory = functools.partial(
                network_fn, **ppo_params.network_factory
            )
        else:
            network_factory = network_fn
        
        # Set up training function to load checkpoint
        training_params = dict(ppo_params)
        if "network_factory" in training_params:
            del training_params["network_factory"]
        
        checkpoint_path = epath.Path(CHECKPOINT_PATH).resolve()
        
        train_fn = functools.partial(
            ppo.train,
            **training_params,
            network_factory=network_factory,
            seed=42,
            restore_checkpoint_path=checkpoint_path,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
        
        # Load checkpoint
        # Returns (make_inference_fn, params, metrics)
        make_inference_fn, params, _ = train_fn(
            environment=self.env
        )
        inference_fn = make_inference_fn(params, deterministic=True)
        
        # JIT the inference function for efficiency
        return jax.jit(inference_fn)

    def apply_control(
        self,
        state: mjx.Data,
        control: jax.Array
    ) -> mjx.Data:
        """Apply control by querying policy and setting motor targets.
        
        This method takes a velocity command, computes the observation,
        queries the policy to get motor actions, and updates state.ctrl
        with the motor targets. Does NOT step the physics.
        
        Args:
            state: Current mjx.Data state
            control: Velocity command (vx, vy, vtheta)
            
        Returns:
            State with updated ctrl field (motor targets)
        """
        # Compute last action from current motor targets
        # motor_targets = default_pose + action * scale
        # => action = (motor_targets - default_pose) / scale
        last_act = (state.ctrl - self._default_pose) / self._action_scale
        
        # Get contact info for observation computation
        contact = jnp.array([
            state.sensordata[
                self.env.mj_model.sensor_adr[sensorid]
            ] > 0
            for sensorid in self._feet_floor_found_sensor
        ])
        
        # Compute gait phase based on current time
        # Use nominal gait frequency of 1.5 Hz (trained range [1.25, 1.5])
        gait_freq = 1.5
        phase_base = 2 * jnp.pi * state.time * gait_freq
        # Left and right legs offset by pi
        phase = jnp.array([phase_base, phase_base + jnp.pi])
        # Wrap phase to [-pi, pi]
        phase = jnp.fmod(phase + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        # Create info dict with all required keys for _get_obs
        info = {
            "command": control,
            "step": 0,
            "last_act": last_act,
            "feet_air_time": jnp.zeros(2),
            "rng": jax.random.PRNGKey(0),  # For noise (we use 0 noise)
            "phase": phase,
        }
        
        # Compute observation using environment's method
        obs = self.env._get_obs(state, info, contact)
        
        # Get action from policy
        # (matches run_g1_interactive.py pattern)
        rng = jax.random.PRNGKey(0)  # Deterministic
        act, _ = self.inference_fn(obs, rng)
        
        # Convert policy action to motor targets
        # Policy outputs actions in [-1, 1] range
        # Environment applies: motor_targets = default_pose + action * scale
        motor_targets = self._default_pose + act * self._action_scale
        
        # Clip motor targets to joint limits
        motor_targets = jnp.clip(
            motor_targets,
            self.mj_model.jnt_range[1:, 0],  # Skip freejoint
            self.mj_model.jnt_range[1:, 1],
        )
        
        # Return state with updated control
        return state.replace(ctrl=motor_targets)

    def running_cost(
        self,
        state: mjx.Data,
        control: jax.Array
    ) -> jax.Array:
        """Cost to reach a desired (x, y, theta) pose.
        
        The MPC will optimize velocity commands to minimize
        distance to target.
        """
        # Target pose (can be updated externally)
        target_pos = getattr(
            self, 'target_pos', jnp.array([2.0, 0.0])
        )
        target_theta = getattr(self, 'target_theta', 0.0)
        
        # Current pose
        pos = state.qpos[:2]  # x, y position
        
        # Extract yaw from quaternion
        # [qw, qx, qy, qz] = state.qpos[3:7]
        qw, qx, qy, qz = (
            state.qpos[3], state.qpos[4],
            state.qpos[5], state.qpos[6]
        )
        theta = jnp.arctan2(
            2 * (qw * qz + qx * qy),
            1 - 2 * (qy**2 + qz**2)
        )
        
        # Cost components
        dist_cost = jnp.sum((pos - target_pos)**2)
        theta_cost = (theta - target_theta)**2
        ctrl_cost = jnp.sum(control**2) * 0.01
        
        return dist_cost + theta_cost + ctrl_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost (no control regularization)."""
        return self.running_cost(state, jnp.zeros(self.nu))

