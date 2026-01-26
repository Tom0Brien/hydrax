"""HumanoidStand task for hydrax.

This module provides a hydrax task for the Humanoid Stand environment
trained in mujoco_playground, following the same API as other tasks.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.task_base import Task


# Path to mujoco_playground XML file
_XML_PATH = (
    Path(__file__).parent.parent.parent.parent.parent
    / "mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/humanoid.xml"
)

# Checkpoint path for trained RL policy
CHECKPOINT_PATH = (
    Path(__file__).parent.parent.parent.parent.parent
    / "mujoco_playground/logs/HumanoidStand-20260125-144955/checkpoints"
)

# Height of head above which stand reward is 1
_STAND_HEIGHT = 1.4


def _load_checkpoint_compat(path):
    """Load checkpoint with compatibility for different brax versions."""
    import numpy as np
    from etils import epath
    from orbax import checkpoint as ocp
    import jax
    import logging
    
    path = epath.Path(path)
    if not path.exists():
        raise ValueError(f'checkpoint path does not exist: {path.as_posix()}')
    
    logging.info('restoring from checkpoint %s', path.as_posix())
    
    metadata = ocp.PyTreeCheckpointer().metadata(path).item_metadata
    restore_args = jax.tree.map(
        lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata
    )
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    target = orbax_checkpointer.restore(
        path, ocp.args.PyTreeRestore(restore_args=restore_args), item=None
    )
    
    stats_dict = target[0]
    policy_params = target[1]
    
    logging.info(f'Checkpoint normalizer fields: {stats_dict.keys()}')
    
    if 'mean' in stats_dict:
        mean = np.array(stats_dict['mean'])
        std = np.array(stats_dict['std'])
    else:
        logging.warning("Old brax checkpoint format - normalization may not work correctly")
        mean = None
        std = None
    
    return {
        'mean': mean,
        'std': std,
        'policy_params': policy_params,
    }


class HumanoidStand(Task):
    """Humanoid stand task with RL policy + SPC residuals.
    
    The goal is to stand upright without falling over, while minimizing
    unnecessary movement.
    """
    
    def __init__(
        self,
        use_rl_policy: bool = True,
    ):
        """Initialize the HumanoidStand task.
        
        Args:
            use_rl_policy: If True, load RL policy for residual control.
                          If False, use direct control (no policy).
        """
        self._use_rl_policy = use_rl_policy
        
        # Load the XML model using mujoco_playground's asset loading
        if not _XML_PATH.exists():
            raise FileNotFoundError(f"XML file not found: {_XML_PATH}")
        
        # Load dm_control common assets
        from mujoco_playground._src.dm_control_suite import common
        assets = common.get_assets()
        
        # Load model
        xml_string = _XML_PATH.read_text()
        mj_model = mujoco.MjModel.from_xml_string(xml_string, assets=assets)
        
        # Initialize Task base class
        super().__init__(mj_model)
        
        # Control is 21 actuators
        self.nu = 21
        self.u_min = jnp.full(21, -1.0)
        self.u_max = jnp.full(21, 1.0)
        
        # Get body IDs
        self._head_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self._torso_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        
        # Get extremity body IDs
        extremities_ids = []
        for side in ("left_", "right_"):
            for limb in ("hand", "foot"):
                extremities_ids.append(
                    mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, side + limb)
                )
        self._extremities_ids = jnp.array(extremities_ids)
        
        # Get sensor ID for COM velocity
        self._torso_subtreelinvel_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_subtreelinvel"
        )
        
        # Load RL policy if requested
        if use_rl_policy:
            self.inference_fn = self._load_policy()
        else:
            self.inference_fn = None
        
        # Control frequency matches training (40Hz)
        self.ctrl_dt = 0.025
        self.n_substeps = max(1, round(self.ctrl_dt / self.dt))
    
    def _load_policy(self):
        """Load the trained RL policy for residual control."""
        from brax.training.agents.ppo import networks as ppo_networks
        from etils import epath
        import functools
        import json
        import jax.nn
        
        # Find latest checkpoint
        checkpoint_path = epath.Path(CHECKPOINT_PATH).resolve()
        if checkpoint_path.is_dir():
            latest_ckpts = list(checkpoint_path.glob("*"))
            numeric_ckpts = []
            for ckpt in latest_ckpts:
                try:
                    int(ckpt.name)
                    numeric_ckpts.append(ckpt)
                except ValueError:
                    continue
            if not numeric_ckpts:
                raise ValueError(f"No checkpoints found in {checkpoint_path}")
            numeric_ckpts.sort(key=lambda x: int(x.name))
            restore_checkpoint_path = numeric_ckpts[-1]
        else:
            restore_checkpoint_path = checkpoint_path
        
        print(f"Loading RL policy from: {restore_checkpoint_path}")
        
        config_path = restore_checkpoint_path / "ppo_network_config.json"
        with open(config_path) as f:
            net_config = json.load(f)
        
        observation_size = net_config["observation_size"]["shape"]
        action_size = net_config["action_size"]
        normalize_observations = net_config["normalize_observations"]
        
        # Build network factory with config from checkpoint
        network_factory_kwargs = net_config.get("network_factory_kwargs", {})
        
        # Convert activation string to actual function
        activation_map = {
            "silu": jax.nn.silu,
            "relu": jax.nn.relu, 
            "tanh": jnp.tanh,
            "swish": jax.nn.swish,
        }
        
        # Build kwargs for make_ppo_networks - only include supported params
        valid_kwargs = {}
        if 'policy_hidden_layer_sizes' in network_factory_kwargs:
            valid_kwargs['policy_hidden_layer_sizes'] = tuple(network_factory_kwargs['policy_hidden_layer_sizes'])
        if 'value_hidden_layer_sizes' in network_factory_kwargs:
            valid_kwargs['value_hidden_layer_sizes'] = tuple(network_factory_kwargs['value_hidden_layer_sizes'])
        if 'activation' in network_factory_kwargs:
            activation_str = network_factory_kwargs['activation']
            if activation_str in activation_map:
                valid_kwargs['activation'] = activation_map[activation_str]
        
        ppo_network = ppo_networks.make_ppo_networks(
            observation_size,
            action_size,
            preprocess_observations_fn=lambda x, y: x,
            **valid_kwargs
        )

        ckpt = _load_checkpoint_compat(restore_checkpoint_path)
        mean = ckpt['mean']
        std = ckpt['std']
        policy_params = ckpt['policy_params']
        
        def make_inference_fn(mean, std, policy_params, normalize_observations, ppo_network, action_size):
            if mean is not None and std is not None:
                def inference_fn(obs, rng):
                    if normalize_observations:
                        obs = (obs - mean) / (std + 1e-8)
                    output = ppo_network.policy_network.apply({}, policy_params, obs)
                    dist = ppo_network.parametric_action_distribution.create_dist(output)
                    action = dist.loc
                    action_std = dist.scale
                    return action, action_std
            else:
                def inference_fn(obs, rng):
                    output = ppo_network.policy_network.apply({}, policy_params, obs)
                    dist = ppo_network.parametric_action_distribution.create_dist(output)
                    action = dist.loc
                    action_std = dist.scale
                    return action, action_std
            return inference_fn
        
        inference_fn = make_inference_fn(mean, std, policy_params, normalize_observations, ppo_network, action_size)
        return jax.jit(inference_fn)
    
    def reset(self, rng: jax.Array) -> "Tuple[mujoco.MjData, mjx.Data]":
        """Reset the environment to default standing position.
        
        Args:
            rng: JAX random key
            
        Returns:
            Tuple of (mj_data, mjx_data) for simulation and controller
        """
        from typing import Tuple
        
        # Create fresh data with default pose (standing)
        mj_data = mujoco.MjData(self.mj_model)
        
        # Run forward to compute derived quantities
        mujoco.mj_forward(self.mj_model, mj_data)

        # Perturb joints 
        mj_data.qpos += jax.random.normal(rng, mj_data.qpos.shape) * 0.1
        
        # Create mjx.Data
        mjx_data = mjx.make_data(self.mj_model)
        mjx_data = mjx_data.replace(
            qpos=jnp.array(mj_data.qpos),
            qvel=jnp.array(mj_data.qvel),
            ctrl=jnp.array(mj_data.ctrl),
            time=jnp.array(0.0),
        )
        
        return mj_data, mjx_data
    
    def mjx_reset(self, rng: jax.Array) -> mjx.Data:
        """Pure MJX reset (fully jittable, can be vmapped for parallel envs).
        
        Args:
            rng: JAX random key
            
        Returns:
            mjx.Data with default initial state
        """
        # Create mjx.Data with default pose
        mjx_data = mjx.make_data(self.mj_model)
        mjx_data = mjx_data.replace(
            qpos=jnp.array(self.mj_model.qpos0),
            qvel=jnp.zeros(self.mj_model.nv),
            ctrl=jnp.zeros(self.mj_model.nu),
            time=jnp.array(0.0),
        )
        
        # Run forward kinematics
        mjx_data = mjx.forward(self.model, mjx_data)
        
        return mjx_data
    
    def _joint_angles(self, state: mjx.Data) -> jax.Array:
        """Returns the state without global orientation or position."""
        return state.qpos[7:]
    
    def _head_height(self, state: mjx.Data) -> jax.Array:
        """Returns the height of the head."""
        return state.xpos[self._head_body, 2]
    
    def _torso_upright(self, state: mjx.Data) -> jax.Array:
        """Returns projection from z-axes of torso to the z-axes of world."""
        return state.xmat[self._torso_body, 2, 2]
    
    def _torso_vertical_orientation(self, state: mjx.Data) -> jax.Array:
        """Returns the z-projection of the torso orientation matrix."""
        return state.xmat[self._torso_body, 2]
    
    def _center_of_mass_velocity(self, state: mjx.Data) -> jax.Array:
        """Returns the velocity of the center of mass in global coordinates."""
        sensor_adr = self.mj_model.sensor_adr[self._torso_subtreelinvel_sensor]
        return state.sensordata[sensor_adr:sensor_adr+3]
    
    def _extremities(self, state: mjx.Data) -> jax.Array:
        """Returns end effector positions in the egocentric frame."""
        torso_frame = state.xmat[self._torso_body]
        torso_pos = state.xpos[self._torso_body]
        torso_to_limb = state.xpos[self._extremities_ids] - torso_pos
        return torso_to_limb @ torso_frame
    
    def _get_obs(self, state: mjx.Data) -> jax.Array:
        """Compute observation from state (matching training format).
        
        Observation: joint_angles (14) + head_height (1) + extremities (12) + 
                    torso_orientation (3) + com_velocity (3) + qvel (27) = 67 dims
        """
        return jnp.concatenate([
            self._joint_angles(state),
            self._head_height(state).reshape(1),
            self._extremities(state).ravel(),
            self._torso_vertical_orientation(state),
            self._center_of_mass_velocity(state),
            state.qvel,
        ])
    
    def apply_control(self, state: mjx.Data, control: jax.Array) -> mjx.Data:
        """Apply control (RL policy + residuals or direct)."""
        if self.inference_fn is not None:
            obs = self._get_obs(state)
            rng = jax.random.PRNGKey(0)
            policy_action, _ = self.inference_fn(obs, rng)
            action_with_residuals = policy_action + control
        else:
            action_with_residuals = control
        
        # Clip to control limits
        ctrl = jnp.clip(action_with_residuals, self.u_min, self.u_max)
        
        return state.replace(ctrl=ctrl)
    
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Running cost for the humanoid stand task.
        
        Cost based on standing (head height + upright), not moving, and small control.
        """
        # Standing reward: head height above threshold
        head_height = self._head_height(state)
        standing = jnp.where(
            head_height >= _STAND_HEIGHT,
            1.0,
            jnp.maximum(0.0, head_height / _STAND_HEIGHT)
        )
        
        # Upright reward: torso z-axis aligned with world z-axis
        upright = self._torso_upright(state)
        upright_reward = jnp.where(
            upright >= 0.9,
            1.0,
            jnp.maximum(0.0, (upright + 1.0) / 2.9)  # Linear from -1 to 0.9
        )
        
        # Stand reward
        stand_reward = standing * upright_reward
        
        # Don't move reward: penalize horizontal velocity
        horizontal_velocity = self._center_of_mass_velocity(state)[:2]
        velocity_magnitude = jnp.linalg.norm(horizontal_velocity)
        dont_move = jnp.exp(-0.5 * (velocity_magnitude / 2.0)**2)
        
        # Small control reward
        if self.inference_fn is not None:
            obs = self._get_obs(state)
            rng = jax.random.PRNGKey(0)
            policy_action, _ = self.inference_fn(obs, rng)
            total_action = policy_action + control
        else:
            total_action = control
        action_magnitude = jnp.mean(jnp.abs(total_action))
        small_control = 1.0 - jnp.clip(action_magnitude, 0.0, 1.0)
        small_control = (4 + small_control) / 5
        
        # Combined reward
        total_reward = stand_reward * dont_move * small_control
        
        # Cost is 1 - reward
        task_cost = 1.0 - total_reward
        
        # Residual penalty (Information Theoretic regularization)
        if self.inference_fn is not None:
            obs = self._get_obs(state)
            rng = jax.random.PRNGKey(0)
            _, policy_std = self.inference_fn(obs, rng)
            # Mahalanobis distance
            inv_var = 1.0 / (jnp.square(policy_std) + 1e-6)
            residual_cost = jnp.sum(inv_var * jnp.square(control))
            lambda_reg = 0.001  # Lower regularization for high-dim action space
            residual_term = lambda_reg * residual_cost
        else:
            residual_term = 0.001 * jnp.sum(jnp.square(control))
        
        return task_cost + residual_term
    
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost."""
        return self.running_cost(state, jnp.zeros(self.nu))
