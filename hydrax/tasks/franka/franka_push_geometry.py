"""Franka push task with configurable object geometry.

This module provides task variants for testing robustness with different
object geometries: cube (original), cylinder, and T-block.
"""

from pathlib import Path
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math

from hydrax.task_base import Task


# Path to mujoco_playground XML files
_XMLS_PATH = (
    Path(__file__).parent.parent.parent.parent.parent
    / "mujoco_playground/mujoco_playground/_src/manipulation/franka_emika_panda_robotiq/xmls"
)

# Available geometries and their XML files
# Note: MJX only supports box-box, box-plane collisions well
# "square" is a cube with equal sides (different from original rectangular cube)
GEOMETRY_XMLS = {
    "cube": "scene_panda_robotiq_cube.xml",
    "square": "scene_panda_robotiq_square.xml",  # Square cube (0.06 x 0.06 x 0.06)
    "tblock": "scene_panda_robotiq_tblock.xml",   # T-shaped (two boxes)
}

GeometryType = Literal["cube", "square", "tblock"]


class FrankaPushGeometry(Task):
    """Franka push task with configurable object geometry.
    
    Similar to FrankaPushCube but allows selecting different object geometries
    to test robustness of the RL+SPC approach under geometry mismatch.
    
    Note: The RL policy was trained on the cube. Using cylinder or tblock
    tests generalization capability.
    """
    
    def __init__(
        self,
        geometry: GeometryType = "cube",
        use_rl_policy: bool = True,
    ):
        """Initialize task with specified geometry.
        
        Args:
            geometry: Object geometry type ("cube", "cylinder", "tblock")
            use_rl_policy: If True, load RL policy for residual control.
                          If False, use direct control (no policy).
        """
        if geometry not in GEOMETRY_XMLS:
            raise ValueError(f"Unknown geometry '{geometry}'. "
                           f"Available: {list(GEOMETRY_XMLS.keys())}")
        
        self.geometry = geometry
        self._use_rl_policy = use_rl_policy
        
        # Load the XML model using mujoco_playground's asset loading
        xml_path = _XMLS_PATH / GEOMETRY_XMLS[geometry]
        if not xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {xml_path}")
        
        # Use mujoco_playground's asset collection to get all mesh files
        from mujoco_playground._src.manipulation.franka_emika_panda_robotiq.panda_robotiq import get_assets
        assets = get_assets()
        
        # Load model using from_xml_string with collected assets
        xml_string = xml_path.read_text()
        mj_model = mujoco.MjModel.from_xml_string(xml_string, assets=assets)
        
        # Initialize Task base class
        super().__init__(mj_model)
        
        # Override nu to 7 (joint residuals for 7-DOF arm)
        self.nu = 7
        self.u_min = jnp.full(7, -10.0)
        self.u_max = jnp.full(7, 10.0)
        
        # Store key body/geom/site IDs
        self._obj_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "box")
        self._obj_geom = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "box")
        self._gripper_site = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        self._mocap_target = 0  # First mocap body
        
        # Robot parameters (copied from PandaRobotiqPushCube defaults)
        self._action_scale = 0.1
        self._max_torque = jnp.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
        # Default gear ratios
        self._gear = jnp.ones(7)
        # Control limits
        self._lowers = jnp.array([-1.0] * 7 + [0.0])  # 7 arm + gripper
        self._uppers = jnp.array([1.0] * 7 + [1.0])
        
        # Load RL policy if requested
        if use_rl_policy:
            self.inference_fn = self._load_policy()
        else:
            self.inference_fn = None
        
        # Set control frequency to 50Hz
        self.ctrl_dt = 0.02
        self.n_substeps = max(1, round(self.ctrl_dt / self.dt))
    
    def _load_policy(self):
        """Load the cube-trained RL policy for residual control."""
        # Import the policy loading from FrankaPushCube
        from hydrax.tasks.franka.franka_push import _load_checkpoint_compat, CHECKPOINT_PATH
        from mujoco_playground.config import manipulation_params
        from brax.training.agents.ppo import networks as ppo_networks
        from etils import epath
        import functools
        import json
        
        ENV_NAME = "PandaRobotiqPushCube"
        ppo_params = manipulation_params.brax_ppo_config(ENV_NAME)
        
        network_fn = ppo_networks.make_ppo_networks
        if hasattr(ppo_params, "network_factory"):
            network_factory = functools.partial(network_fn, **ppo_params.network_factory)
        else:
            network_factory = network_fn
        
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
        
        ppo_network = network_factory(
            observation_size,
            action_size,
            preprocess_observations_fn=lambda x, y: x,
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
                    action = output[:action_size]
                    return action, {}
            else:
                def inference_fn(obs, rng):
                    output = ppo_network.policy_network.apply({}, policy_params, obs)
                    action = output[:action_size]
                    return action, {}
            return inference_fn
        
        inference_fn = make_inference_fn(mean, std, policy_params, normalize_observations, ppo_network, action_size)
        return jax.jit(inference_fn)
    
    def reset(self, rng: jax.Array) -> "Tuple[mujoco.MjData, mjx.Data]":
        """Reset the environment with randomized object and target positions.
        
        Matches the playground PandaRobotiqPushCube reset logic exactly.
        
        Args:
            rng: JAX random key
            
        Returns:
            Tuple of (mj_data, mjx_data) for simulation and controller
        """
        from typing import Tuple
        
        # Split RNG like playground does
        rng, rng_box1, rng_box2, rng_target, rng_robot_arm, rng_theta = jax.random.split(rng, 6)
        
        # Create fresh data and reset to home keyframe
        mj_data = mujoco.MjData(self.mj_model)
        key_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.mj_model, mj_data, key_id)
        
        # Randomize robot arm joint positions (matching playground)
        # Joint limits and percent limits from playground
        jnt_range = jnp.array([
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973],
        ])
        joint_range_init_percent_limit = jnp.array([0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3])
        
        # Add small random perturbation to arm joints (first 7 qpos)
        arm_noise = 0.3 * jax.random.uniform(
            rng_robot_arm,
            (7,),
            minval=jnt_range[:, 0] * joint_range_init_percent_limit,
            maxval=jnt_range[:, 1] * joint_range_init_percent_limit,
        )
        mj_data.qpos[:7] = mj_data.qpos[:7] + arm_noise
        
        # Get initial object position from keyframe (obj qpos at index 13-19)
        init_obj_pos = mj_data.qpos[13:16].copy()
        
        # Playground sampling bounds
        OBJ_SAMPLE_MIN = jnp.array([0.4, -0.2, -0.005])
        OBJ_SAMPLE_MAX = jnp.array([0.65, 0.2, 0.04])
        
        # Box position: offset=0.15 around init position, clipped to bounds
        box_offset = 0.15
        rng_box1_x, rng_box1_y = jax.random.split(rng_box1)
        box_x = float(jax.random.uniform(rng_box1_x, 
            minval=init_obj_pos[0] - box_offset * 0.4,
            maxval=init_obj_pos[0] + box_offset * 0.4))
        box_y = float(jax.random.uniform(rng_box1_y,
            minval=init_obj_pos[1] - box_offset,
            maxval=init_obj_pos[1] + box_offset))
        box_x = float(jnp.clip(box_x, OBJ_SAMPLE_MIN[0], OBJ_SAMPLE_MAX[0]))
        box_y = float(jnp.clip(box_y, OBJ_SAMPLE_MIN[1], OBJ_SAMPLE_MAX[1]))
        
        # Box quaternion: random rotation around Z axis
        box_theta = float(jax.random.uniform(rng_box2, minval=0, maxval=2*jnp.pi))
        box_quat = [float(jnp.cos(box_theta/2)), 0.0, 0.0, float(jnp.sin(box_theta/2))]
        
        # Set box position and quaternion
        mj_data.qpos[13] = box_x
        mj_data.qpos[14] = box_y
        mj_data.qpos[15] = init_obj_pos[2]  # Keep original z height
        mj_data.qpos[16:20] = box_quat
        
        # Target position: offset=0.05 around init position, clipped to bounds
        target_offset = 0.05
        rng_target_x, rng_target_y = jax.random.split(rng_target)
        target_x = float(jax.random.uniform(rng_target_x,
            minval=init_obj_pos[0] - target_offset * 0.4,
            maxval=init_obj_pos[0] + target_offset * 0.4))
        target_y = float(jax.random.uniform(rng_target_y,
            minval=init_obj_pos[1] - target_offset,
            maxval=init_obj_pos[1] + target_offset))
        target_x = float(jnp.clip(target_x, OBJ_SAMPLE_MIN[0], OBJ_SAMPLE_MAX[0]))
        target_y = float(jnp.clip(target_y, OBJ_SAMPLE_MIN[1], OBJ_SAMPLE_MAX[1]))
        target_z = init_obj_pos[2]  # Same height as object
        
        # Target quaternion: up to 45 degrees rotation combined with box rotation
        target_theta = float(jax.random.uniform(rng_theta, minval=0, maxval=45*jnp.pi/180))
        target_quat = [float(jnp.cos(target_theta/2)), 0.0, 0.0, float(jnp.sin(target_theta/2))]
        
        mj_data.mocap_pos[0] = [target_x, target_y, target_z]
        mj_data.mocap_quat[0] = target_quat
        
        # Run forward to compute derived quantities
        mujoco.mj_forward(self.mj_model, mj_data)
        
        # Create mjx.Data with larger contact buffer to avoid ncon errors
        mjx_data = mjx.make_data(self.mj_model, nconmax=256, njmax=256)
        mjx_data = mjx_data.replace(
            qpos=jnp.array(mj_data.qpos),
            qvel=jnp.array(mj_data.qvel),
            ctrl=jnp.array(mj_data.ctrl),
            mocap_pos=jnp.array(mj_data.mocap_pos),
            mocap_quat=jnp.array(mj_data.mocap_quat),
            time=jnp.array(mj_data.time),
        )
        
        return mj_data, mjx_data
    
    def _get_obs(self, state: mjx.Data) -> jax.Array:
        """Compute observation from state (matching cube policy expectations)."""
        target_pos = state.mocap_pos[self._mocap_target, :].ravel()
        target_quat = state.mocap_quat[self._mocap_target, :].ravel()
        target_mat = math.quat_to_mat(target_quat)
        
        obj_pos = state.xpos[self._obj_body]
        obj_quat = state.xquat[self._obj_body]
        obj_mat = math.quat_to_mat(obj_quat)
        
        robot_qpos = state.qpos[:7]
        robot_qvel = state.qvel[:7]
        
        gripper_pos = state.site_xpos[self._gripper_site]
        gripper_mat = state.site_xmat[self._gripper_site]
        
        last_action = state.ctrl[:7] / self._action_scale
        
        target_orientation = target_mat.ravel()[3:]
        obj_orientation = obj_mat.ravel()[3:]
        
        obs = jnp.concatenate([
            target_pos,
            target_orientation,
            last_action,
            robot_qpos,
            robot_qvel,
            gripper_pos,
            gripper_mat.ravel()[3:],
            obj_orientation,
            obj_pos,
        ])
        return obs
    
    def apply_control(self, state: mjx.Data, control: jax.Array) -> mjx.Data:
        """Apply control (RL policy + residuals or direct)."""
        if self.inference_fn is not None:
            obs = self._get_obs(state)
            rng = jax.random.PRNGKey(0)
            policy_action, _ = self.inference_fn(obs, rng)
            action_with_residuals = policy_action + control
        else:
            action_with_residuals = control
        
        ctrl = action_with_residuals * self._action_scale
        ctrl = jnp.clip(ctrl, -self._max_torque / self._gear, self._max_torque / self._gear)
        ctrl = jnp.concatenate([ctrl, jnp.array([0.82])])  # Close gripper
        ctrl = jnp.clip(ctrl, self._lowers, self._uppers)
        
        return state.replace(ctrl=ctrl)
    
    def _get_box_target_cost(self, state: mjx.Data) -> jax.Array:
        """Cost for object distance to target position."""
        target_pos = state.mocap_pos[self._mocap_target, :].ravel()
        obj_pos = state.xpos[self._obj_body]
        return jnp.sum(jnp.square(obj_pos[:2] - target_pos[:2]))
    
    def _get_gripper_obj_cost(self, state: mjx.Data) -> jax.Array:
        """Cost for gripper distance to object."""
        target_pos = state.mocap_pos[self._mocap_target, :].ravel()
        obj_pos = state.xpos[self._obj_body]
        gripper_pos = state.site_xpos[self._gripper_site]
        
        side_dir = obj_pos - target_pos
        side_dir_norm = jnp.linalg.norm(side_dir) + 1e-6
        side_dir = side_dir / side_dir_norm * 0.1 * (side_dir_norm > 1e-3)
        obj_side_pos = side_dir + obj_pos
        
        return jnp.sum(jnp.square(obj_side_pos - gripper_pos))
    
    def _get_orientation_cost(self, state: mjx.Data) -> jax.Array:
        """Cost for object orientation error."""
        target_quat = state.mocap_quat[self._mocap_target, :].squeeze()
        obj_quat = state.xquat[self._obj_body]
        
        quat_diff = math.quat_mul(obj_quat, math.quat_inv(target_quat))
        quat_diff = math.normalize(quat_diff)
        ori_error = 2.0 * jnp.arcsin(jnp.clip(math.norm(quat_diff[1:]), a_max=1.0))
        
        return jnp.square(ori_error)
    
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Running cost for the push task."""
        obj_target_cost = self._get_box_target_cost(state)
        gripper_obj_cost = self._get_gripper_obj_cost(state)
        orientation_cost = self._get_orientation_cost(state)
        residual_cost = jnp.sum(jnp.square(control))
        
        return (
            8.0 * obj_target_cost
            + 2.0 * gripper_obj_cost
            + 6.0 * orientation_cost
            + 0.1 * residual_cost
        )
    
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost."""
        return self.running_cost(state, jnp.zeros(self.nu))
