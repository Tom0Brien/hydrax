"""CheetahRun task for hydrax.

This module provides a hydrax task for the CheetahRun locomotion environment
trained in mujoco_playground, following the same API as FrankaPushGeometry.
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
    / "mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/cheetah.xml"
)

# Checkpoint path for trained RL policy
CHECKPOINT_PATH = (
    Path(__file__).parent.parent.parent.parent.parent
    / "mujoco_playground/logs/CheetahRun-20260124-212041/checkpoints"
)

# Target running speed (from dm_control_suite cheetah.py)
_RUN_SPEED = 10.0


def _load_checkpoint_compat(path):
    """Load checkpoint with compatibility for different brax versions.
    
    Extracts the essential data needed for inference:
    - mean/std for observation normalization  
    - policy network parameters
    
    Handles both newer brax (stores mean/std directly) and older versions.
    """
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
    
    # Extract normalizer stats and policy params
    # Target is a tuple/list: (normalizer_params, policy_params)
    stats_dict = target[0]
    policy_params = target[1]
    
    logging.info(f'Checkpoint normalizer fields: {stats_dict.keys()}')
    
    # Get mean/std for normalization
    if 'mean' in stats_dict:
        # Newer brax format - use mean/std directly
        mean = np.array(stats_dict['mean'])
        std = np.array(stats_dict['std'])
    else:
        # Older brax format - would need to compute from count/summed_variance
        # For now, assume no normalization if these aren't available
        logging.warning("Old brax checkpoint format - normalization may not work correctly")
        mean = None
        std = None
    
    return {
        'mean': mean,
        'std': std,
        'policy_params': policy_params,
    }


class CheetahRun(Task):
    """CheetahRun locomotion task with RL policy + SPC residuals.
    
    The cheetah's goal is to run as fast as possible. The RL policy provides
    base locomotion commands, and SPC optimizes residuals to improve performance.
    """
    
    def __init__(
        self,
        use_rl_policy: bool = True,
    ):
        """Initialize the CheetahRun task.
        
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
        
        # Control is the 6 leg actuators
        self.nu = 6
        self.u_min = jnp.full(6, -1.0)  # Control range from XML
        self.u_max = jnp.full(6, 1.0)
        
        # Store joint limits for reset
        self._lowers = jnp.array(mj_model.jnt_range[3:, 0])
        self._uppers = jnp.array(mj_model.jnt_range[3:, 1])
        
        # Get torso body ID for speed computation
        self._torso_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        
        # Get sensor ID for subtree linear velocity
        self._torso_subtreelinvel_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_subtreelinvel"
        )
        
        # Load RL policy if requested
        if use_rl_policy:
            self.inference_fn = self._load_policy()
        else:
            self.inference_fn = None
        
        # Control frequency matches training (100Hz)
        self.ctrl_dt = 0.01
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
        """Reset the environment with randomized joint positions.
        
        Matches the playground CheetahRun reset logic.
        
        Args:
            rng: JAX random key
            
        Returns:
            Tuple of (mj_data, mjx_data) for simulation and controller
        """
        from typing import Tuple
        
        rng, rng1 = jax.random.split(rng)
        
        # Create fresh data
        mj_data = mujoco.MjData(self.mj_model)
        
        # Randomize joint positions (joints 3+ are the hinge joints)
        # qpos: [rootx, rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot]
        random_joints = jax.random.uniform(
            rng1,
            (self.mj_model.nq - 3,),
            minval=self._lowers,
            maxval=self._uppers,
        )
        mj_data.qpos[:3] = 0.0  # Root position/rotation
        mj_data.qpos[3:] = random_joints
        
        # Run forward to compute derived quantities
        mujoco.mj_forward(self.mj_model, mj_data)
        
        # Stabilize for a bit (like playground does)
        for _ in range(200):
            mujoco.mj_step(self.mj_model, mj_data)
        mj_data.time = 0.0
        
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
            mjx.Data with randomized initial state
        """
        rng, rng1 = jax.random.split(rng)
        
        # Randomize joint positions
        random_joints = jax.random.uniform(
            rng1,
            (self.mj_model.nq - 3,),
            minval=self._lowers,
            maxval=self._uppers,
        )
        
        qpos = jnp.zeros(self.mj_model.nq)
        qpos = qpos.at[3:].set(random_joints)
        
        # Create mjx.Data
        mjx_data = mjx.make_data(self.mj_model)
        mjx_data = mjx_data.replace(
            qpos=qpos,
            qvel=jnp.zeros_like(mjx_data.qvel),
            ctrl=jnp.zeros(self.mj_model.nu),
            time=jnp.array(0.0),
        )
        
        # Run forward kinematics
        mjx_data = mjx.forward(self.model, mjx_data)
        
        # Stabilize
        def stabilize_step(data, _):
            data = data.replace(ctrl=jnp.zeros(self.mj_model.nu))
            return mjx.step(self.model, data), None
        
        mjx_data = jax.lax.scan(stabilize_step, mjx_data, None, 200)[0]
        mjx_data = mjx_data.replace(time=jnp.array(0.0))
        
        return mjx_data
    
    def _get_obs(self, state: mjx.Data) -> jax.Array:
        """Compute observation from state (matching training format).
        
        Observation: qpos[1:] (8 dims) + qvel (9 dims) = 17 dims
        Note: qpos[0] (rootx) is excluded as it's the global x position.
        """
        return jnp.concatenate([
            state.qpos[1:],  # Skip rootx (8 dims)
            state.qvel,      # All velocities (9 dims)
        ])
    
    def _get_speed(self, state: mjx.Data) -> jax.Array:
        """Get the forward running speed from sensor data."""
        # Get sensor data for subtree linear velocity
        # The sensor gives [vx, vy, vz], we want vx (forward speed)
        sensor_adr = self.mj_model.sensor_adr[self._torso_subtreelinvel_sensor]
        return state.sensordata[sensor_adr]  # x-axis velocity
    
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
        """Running cost for the cheetah run task.
        
        Reward for running fast, with residual penalty.
        """
        # Speed reward: want to maximize forward velocity toward target speed
        speed = self._get_speed(state)
        
        # Cost is negative of reward (we minimize cost)
        # Use tolerance function similar to playground
        speed_error = jnp.maximum(0.0, _RUN_SPEED - speed)
        speed_cost = speed_error / _RUN_SPEED  # Normalized [0, 1]
        
        # Residual penalty (Information Theoretic regularization)
        if self.inference_fn is not None:
            obs = self._get_obs(state)
            rng = jax.random.PRNGKey(0)
            _, policy_std = self.inference_fn(obs, rng)
            # Mahalanobis distance
            inv_var = 1.0 / (jnp.square(policy_std) + 1e-6)
            residual_cost = jnp.sum(inv_var * jnp.square(control))
            lambda_reg = 0.01  # Lower regularization for locomotion
            residual_term = lambda_reg * residual_cost
        else:
            residual_term = 0.01 * jnp.sum(jnp.square(control))
        
        return speed_cost + residual_term
    
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost."""
        return self.running_cost(state, jnp.zeros(self.nu))
