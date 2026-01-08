"""Franka Panda push cube task with RL policy + SPC residuals."""

import functools
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from etils import epath
from mujoco import mjx
from mujoco.mjx._src import math

from hydrax.task_base import Task

# Handle checkpoint loading with version compatibility
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




# Checkpoint path for trained Panda push cube policy
CHECKPOINT_PATH = str(
    Path(__file__).parent.parent.parent.parent.parent
    / "mujoco_playground/logs/PandaRobotiqPushCube-20260108-150347/checkpoints"
)
ENV_NAME = "PandaRobotiqPushCube"


class FrankaPushCube(Task):
    """Franka push cube task using trained RL policy with SPC residuals.
    
    This task wraps the mujoco_playground PandaRobotiqPushCube environment
    and its trained policy to enable MPC planning with joint residuals.
    
    The SPC controller outputs residuals that are added to the RL policy's
    actions, allowing online adaptation while leveraging the learned behavior.
    """
    
    def __init__(self):
        """Initialize Franka push cube task with environment and policy."""
        # Import here to avoid circular dependencies
        from mujoco_playground import registry, wrapper
        from mujoco_playground.config import manipulation_params
        
        # Create environment using registry
        env_cfg = registry.get_default_config(ENV_NAME)
        env_cfg["impl"] = "jax"
        self.env = registry.load(ENV_NAME, config=env_cfg)
        
        # Initialize Task base class with the environment's model
        super().__init__(self.env.mj_model)
        
        # Override nu to 7 (joint residuals for 7-DOF arm)
        self.nu = 7
        # Small residual bounds initially - policy should work out of the box
        self.u_min = jnp.full(7, -0.1)
        self.u_max = jnp.full(7, 0.1)
        
        # Load RL policy
        self.inference_fn = self._load_policy(manipulation_params, wrapper)
        
        # Store environment attributes needed for control
        self._action_scale = self.env._config.action_scale
        self._max_torque = self.env._max_torque
        self._gear = self.env._gear
        self._lowers = self.env._lowers
        self._uppers = self.env._uppers
        self._init_q = self.env._init_q
        self._robot_arm_qposadr = self.env._robot_arm_qposadr
        self._obj_body = self.env._obj_body
        self._gripper_site = self.env._gripper_site
        self._mocap_target = self.env._mocap_target
        
        # Set control frequency to 50Hz (matching RL policy)
        self.ctrl_dt = 0.02
        # Recompute n_substeps after changing ctrl_dt
        self.n_substeps = max(1, round(self.ctrl_dt / self.dt))
    
    def _load_policy(
        self,
        manipulation_params: Any,
        wrapper: Any
    ) -> Any:
        """Load the RL policy from checkpoint.
        
        Uses custom checkpoint loading to handle version compatibility
        between different brax versions (e.g., std_eps field).
        
        Returns:
            Jitted inference function
        """
        from brax.training.agents.ppo import networks as ppo_networks
        from brax.training import checkpoint as brax_checkpoint
        from brax.training.acme import running_statistics
        import json
        
        # Get PPO config for network factory setup
        ppo_params = manipulation_params.brax_ppo_config(ENV_NAME)
        
        # Set up network factory
        network_fn = ppo_networks.make_ppo_networks
        if hasattr(ppo_params, "network_factory"):
            network_factory = functools.partial(
                network_fn, **ppo_params.network_factory
            )
        else:
            network_factory = network_fn
        
        # Find latest checkpoint
        checkpoint_path = epath.Path(CHECKPOINT_PATH).resolve()
        if checkpoint_path.is_dir():
            latest_ckpts = list(checkpoint_path.glob("*"))
            latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
            # Only keep directories with numeric names
            numeric_ckpts = []
            for ckpt in latest_ckpts:
                try:
                    int(ckpt.name)
                    numeric_ckpts.append(ckpt)
                except ValueError:
                    continue
            if not numeric_ckpts:
                raise ValueError(
                    f"No numeric checkpoint directories found in {checkpoint_path}"
                )
            numeric_ckpts.sort(key=lambda x: int(x.name))
            latest_ckpt = numeric_ckpts[-1]
            restore_checkpoint_path = latest_ckpt
        else:
            restore_checkpoint_path = checkpoint_path
        
        print(f"Loading Franka push policy from: {restore_checkpoint_path}")
        
        # Load network config from checkpoint
        config_path = restore_checkpoint_path / "ppo_network_config.json"
        with open(config_path) as f:
            net_config = json.load(f)
        
        observation_size = net_config["observation_size"]["shape"]
        action_size = net_config["action_size"]
        normalize_observations = net_config["normalize_observations"]
        
        # Create the network - we'll handle normalization manually for compatibility
        ppo_network = network_factory(
            observation_size,
            action_size,
            preprocess_observations_fn=lambda x, y: x,  # Identity, we normalize manually
        )
        
        # Load checkpoint with version compatibility
        ckpt = _load_checkpoint_compat(restore_checkpoint_path)
        mean = ckpt['mean']
        std = ckpt['std']
        policy_params = ckpt['policy_params']
        
        # Create inference function
        # Note: brax FeedForwardNetwork.apply expects (processor_params, policy_params, obs)
        # The network outputs [mean, log_std] concatenated, we take just the mean for deterministic action
        def make_inference_fn(mean, std, policy_params, normalize_observations, ppo_network, action_size):
            if mean is not None and std is not None:
                # Normalize using stored mean/std
                def inference_fn(obs, rng):
                    if normalize_observations:
                        obs = (obs - mean) / (std + 1e-8)
                    # apply expects (processor_params, policy_params, obs)
                    output = ppo_network.policy_network.apply({}, policy_params, obs)
                    # Output is [action_mean, log_std], take just the mean for deterministic action
                    action = output[:action_size]
                    return action, {}  # Return (action, extras) like brax
            else:
                # No normalization available
                def inference_fn(obs, rng):
                    output = ppo_network.policy_network.apply({}, policy_params, obs)
                    action = output[:action_size]
                    return action, {}
            
            return inference_fn
        
        inference_fn = make_inference_fn(mean, std, policy_params, normalize_observations, ppo_network, action_size)
        
        # JIT the inference function
        return jax.jit(inference_fn)
    
    def _get_obs(self, state: mjx.Data) -> jax.Array:
        """Compute observation from state.
        
        Simplified observation matching key components from push_cube.py.
        Uses noisy=False version for MPC.
        """
        # Target position and orientation
        target_pos = state.mocap_pos[self._mocap_target, :].ravel()
        target_quat = state.mocap_quat[self._mocap_target, :].ravel()
        target_mat = math.quat_to_mat(target_quat)
        
        # Object position and orientation
        obj_pos = state.xpos[self._obj_body]
        obj_quat = state.xquat[self._obj_body]
        obj_mat = math.quat_to_mat(obj_quat)
        
        # Robot joint state (7 DOF arm)
        robot_qpos = state.qpos[:7]
        robot_qvel = state.qvel[:7]
        
        # Gripper position and orientation
        gripper_pos = state.site_xpos[self._gripper_site]
        gripper_mat = state.site_xmat[self._gripper_site]
        
        # Compute last action from current ctrl
        # ctrl = action * action_scale, so action = ctrl / action_scale
        # Note: ctrl has 8 elements (7 arm + 1 gripper), we only need first 7
        last_action = state.ctrl[:7] / self._action_scale
        
        # Assemble observation (matching push_cube.py structure)
        target_orientation = target_mat.ravel()[3:]
        obj_orientation = obj_mat.ravel()[3:]
        
        obs = jnp.concatenate([
            target_pos,
            target_orientation,
            last_action,  # Last action computed from state.ctrl
            robot_qpos,
            robot_qvel,
            gripper_pos,
            gripper_mat.ravel()[3:],
            obj_orientation,
            obj_pos,
        ])
        return obs
    
    def apply_control(
        self,
        state: mjx.Data,
        control: jax.Array
    ) -> mjx.Data:
        """Apply control by querying policy and adding residuals.
        
        Args:
            state: Current mjx.Data state
            control: Residuals to add to policy action (7D)
            
        Returns:
            State with updated ctrl field
        """
        # Get observation
        obs = self._get_obs(state)
        
        # Get action from policy
        rng = jax.random.PRNGKey(0)  # Deterministic
        policy_action, _ = self.inference_fn(obs, rng)
        
        # Add residuals to policy action
        action_with_residuals = policy_action + control
        
        # Convert to ctrl (same as push_cube.py step())
        ctrl = action_with_residuals * self._action_scale
        ctrl = jnp.clip(
            ctrl, -self._max_torque / self._gear, self._max_torque / self._gear
        )
        # Close the gripper (fixed at 0.82)
        ctrl = jnp.concatenate([ctrl, jnp.array([0.82])])
        ctrl = jnp.clip(ctrl, self._lowers, self._uppers)
        
        # Note: We don't update self._last_action here as it would cause JAX tracer leak.
        # For MPC planning, we use zeros for last_action in observation.
        
        return state.replace(ctrl=ctrl)
    
    def _get_box_target_cost(self, state: mjx.Data) -> jax.Array:
        """Cost for box distance to target position."""
        target_pos = state.mocap_pos[self._mocap_target, :].ravel()
        box_pos = state.xpos[self._obj_body]
        # Position error (XY only, matching push_cube.py)
        return jnp.sum(jnp.square(box_pos[:2] - target_pos[:2]))
    
    def _get_gripper_box_cost(self, state: mjx.Data) -> jax.Array:
        """Cost for gripper distance to box."""
        target_pos = state.mocap_pos[self._mocap_target, :].ravel()
        box_pos = state.xpos[self._obj_body]
        gripper_pos = state.site_xpos[self._gripper_site]
        
        # Offset gripper target to push from behind
        side_dir = box_pos - target_pos
        side_dir_norm = jnp.linalg.norm(side_dir) + 1e-6
        side_dir = side_dir / side_dir_norm * 0.1 * (side_dir_norm > 1e-3)
        box_side_pos = side_dir + box_pos
        
        return jnp.sum(jnp.square(box_side_pos - gripper_pos))
    
    def _get_orientation_cost(self, state: mjx.Data) -> jax.Array:
        """Cost for box orientation error."""
        target_quat = state.mocap_quat[self._mocap_target, :].squeeze()
        box_quat = state.xquat[self._obj_body]
        
        # Quaternion difference
        quat_diff = math.quat_mul(box_quat, math.quat_inv(target_quat))
        quat_diff = math.normalize(quat_diff)
        ori_error = 2.0 * jnp.arcsin(jnp.clip(math.norm(quat_diff[1:]), a_max=1.0))
        
        return jnp.square(ori_error)
    
    def running_cost(
        self,
        state: mjx.Data,
        control: jax.Array
    ) -> jax.Array:
        """Running cost based on push_cube.py reward.
        
        Cost terms (adapted from reward scales):
        - box_target: 8.0 weight
        - gripper_box: 2.0 weight
        - box_orientation: 6.0 weight
        - residual_regularization: 0.1 weight
        """
        # Box to target (main objective)
        box_target_cost = self._get_box_target_cost(state)
        
        # Gripper to box (approach cost)
        gripper_box_cost = self._get_gripper_box_cost(state)
        
        # Orientation cost
        orientation_cost = self._get_orientation_cost(state)
        
        # Residual regularization (keep residuals small)
        residual_cost = jnp.sum(jnp.square(control))
        
        # Weighted combination (matching reward scale ratios)
        return (
            8.0 * box_target_cost
            + 2.0 * gripper_box_cost
            + 6.0 * orientation_cost
            + 0.1 * residual_cost
        )
    
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost (no residual regularization)."""
        return self.running_cost(state, jnp.zeros(self.nu))
