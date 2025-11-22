import json
import os
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
from brax.training.agents.ppo import networks as ppo_networks
from etils import epath
import orbax.checkpoint

from hydrax.task_base import Task

# Hardcoded path to the checkpoint
CHECKPOINT_PATH = "/home/tom/OneDrive/Phd/Papers/GPC/mujoco_playground/logs/G1JoystickFlatTerrain-20251120-162949/checkpoints/000202342400"
XML_PATH = "/home/tom/OneDrive/Phd/Papers/GPC/mujoco_playground/mujoco_playground/_src/locomotion/g1/xmls/scene_mjx_feetonly_flat_terrain.xml"

class G1Locomotion(Task):
    def __init__(self):
        # Load model
        mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
        super().__init__(mj_model)

        # Override nu to 3 (vx, vy, vtheta)
        self.nu = 3
        self.u_min = jnp.array([-1.0, -1.0, -1.0])
        self.u_max = jnp.array([1.0, 1.0, 1.0])

        # Load RL policy
        self.policy_params, self.inference_fn = self._load_policy()

        # Initialize constants for observation
        self._init_constants(mj_model)

    def _init_constants(self, mj_model):
        # Keyframe "knees_bent" for default pose
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
        if key_id == -1:
             # Fallback if keyframe not found, though it should be there
             self._default_pose = jnp.zeros(mj_model.nq - 7)
        else:
            self._default_pose = jnp.array(mj_model.key_qpos[key_id, 7:])

        # Indices
        self._torso_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self._pelvis_imu_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_pelvis")
        
        # Feet sites for phase reward (not strictly needed for obs, but good to have)
        self._feet_site_ids = [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "left_foot"),
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_foot"),
        ]

        # Sensor addresses for feet velocity
        # In joystick.py:
        # foot_linvel_sensor_adr = []
        # for site in consts.FEET_SITES:
        #   sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
        #   sensor_adr = self._mj_model.sensor_adr[sensor_id]
        #   sensor_dim = self._mj_model.sensor_dim[sensor_id]
        #   foot_linvel_sensor_adr.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
        # We can precompute these indices.
        # Assuming sensor names are "left_foot_global_linvel" and "right_foot_global_linvel"
        self._foot_linvel_indices = []
        for name in ["left_foot_global_linvel", "right_foot_global_linvel"]:
            id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            adr = mj_model.sensor_adr[id]
            dim = mj_model.sensor_dim[id]
            self._foot_linvel_indices.extend(range(adr, adr + dim))
        self._foot_linvel_indices = jnp.array(self._foot_linvel_indices)

        # Noise scales (from default_config in joystick.py)
        # We use 0.0 noise for planning
        self.noise_level = 0.0

    def _load_policy(self):
        # Load config
        config_path = os.path.join(CHECKPOINT_PATH, "ppo_network_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Reconstruct network
        # config["observation_size"] is a dict, we need the shape
        obs_size = config["observation_size"]["state"]["shape"][0] # 103
        action_size = config["action_size"] # 29
        
        # PPO networks factory
        # We need to convert list to tuple for hashability if needed, but make_ppo_networks takes kwargs
        network_factory_kwargs = config["network_factory_kwargs"]
        
        ppo_network = ppo_networks.make_ppo_networks(
            observation_size=obs_size,
            action_size=action_size,
            **network_factory_kwargs
        )
        
        # Initialize params to get structure
        rng = jax.random.PRNGKey(0)
        params = ppo_network.policy_network.init(rng)
        
        # Load params from checkpoint
        # The checkpoint contains 'params' which includes policy and value params.
        # We need to check the structure.
        # Usually brax saves the whole training state.
        # Let's try to load with orbax.
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        # We need a target structure. The training state typically has:
        # optimizer_state, params, normalizer_params, env_steps
        # We only care about params (policy) and normalizer_params.
        
        # To get the full structure, we might need to init the full PPO training state, 
        # but that requires creating the PPO agent.
        # Alternatively, we can try to restore with item=None and see what we get, 
        # but orbax might complain if we don't provide a schema.
        # However,        # Checkpoint structure: [normalizer_params, policy_params, value_params]
        restored = checkpointer.restore(CHECKPOINT_PATH)
        normalizer_params = restored[0]
        policy_params = restored[1]['params']
        
        # Flax apply expects {'params': ...}
        policy_params = {'params': policy_params}
        
        # Create inference function
        make_policy = ppo_networks.make_inference_fn(ppo_network)
        inference_fn = make_policy((normalizer_params, policy_params), deterministic=True)
        
        return (normalizer_params, policy_params), inference_fn

    def apply_control(self, state: mjx.Data, control: jax.Array) -> mjx.Data:
        # control is (3,) -> vx, vy, vtheta
        
        # Construct observation
        obs = self._get_obs(state, control)
        
        # Query policy
        rng = jax.random.PRNGKey(0) # Deterministic for planning
        # inference_fn takes (obs, key)
        action, _ = self.inference_fn(obs, rng)
        
        # Apply action (delta from default pose)
        # Action scale is 0.5 in default_config
        action_scale = 0.5
        motor_targets = self._default_pose + action * action_scale
        
        return state.replace(ctrl=motor_targets)

    def _get_obs(self, data: mjx.Data, command: jax.Array) -> jax.Array:
        # Match _get_obs in joystick.py
        
        # 1. linvel (local)
        # We need to compute local linvel.
        # global_linvel = data.cvel[self._torso_body_id][:3] # This is com vel? No, cvel is 6D.
        # joystick.py uses get_local_linvel(data, "pelvis")
        # "pelvis" body? In xml it's "torso_link" (ROOT_BODY).
        # Wait, joystick.py says:
        # self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        # But get_local_linvel uses "pelvis"?
        # In g1_constants.py, ROOT_BODY = "torso_link".
        # Let's assume "pelvis" is the name of the body or site?
        # In joystick.py: self._pelvis_imu_site_id = self._mj_model.site("imu_in_pelvis").id
        # get_local_linvel likely transforms global vel to local frame of sensor/body.
        
        # Let's implement equivalent logic.
        # Global linvel of torso
        subtree_com = data.subtree_com[self._torso_body_id]
        # This is position.
        # mjx doesn't have easy "get_local_linvel".
        # We can use sensor data if available.
        # "local_linvel" sensor?
        # g1_constants.py: LOCAL_LINVEL_SENSOR = "local_linvel"
        # Let's check if sensor exists.
        # If so, use data.sensordata.
        
        # 2. gyro
        # "gyro" sensor.
        
        # 3. gravity
        # "upvector" sensor (projected gravity).
        
        # 4. command
        # Passed in.
        
        # 5. joint angles (qpos[7:] - default_pose)
        
        # 6. joint vel (qvel[6:])
        
        # 7. last_act
        # data.ctrl
        
        # 8. phase
        # sin(phase), cos(phase)
        # phase = 2 * pi * time * freq
        # freq is randomized in training U(1.25, 1.5).
        # For planning, we should pick a nominal freq, say 1.375 or 1.5?
        # Or should we include freq in the state?
        # The policy expects phase as input.
        # In joystick.py: phase_dt = 2 * pi * dt * gait_freq
        # phase += phase_dt
        # We can compute phase from time: phase = 2 * pi * time * freq
        # But we need to handle the offset (phase is vector of 2 for left/right legs?).
        # joystick.py: phase = jp.array([0, jp.pi]) initially.
        # So left leg 0, right leg pi.
        # phase(t) = [2*pi*freq*t, 2*pi*freq*t + pi]
        # obs uses [cos(phase), sin(phase)] -> 4 dims.
        
        # Let's check sensors first.
        
        # Helper to get sensor data
        def get_sensor(name):
            id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if id == -1:
                raise ValueError(f"Sensor {name} not found")
            adr = self.mj_model.sensor_adr[id]
            dim = self.mj_model.sensor_dim[id]
            return data.sensordata[adr:adr+dim]

        linvel = get_sensor("local_linvel_pelvis")
        gyro = get_sensor("gyro_pelvis")
        gravity = get_sensor("upvector_pelvis")
        
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]
        last_act = data.ctrl
        
        # Phase
        gait_freq = 1.5 # Nominal
        phase_base = 2 * jnp.pi * data.time * gait_freq
        phase = jnp.array([phase_base, phase_base + jnp.pi])
        phase = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)])
        
        # Noise (0 for planning)
        
        obs = jnp.hstack([
            linvel,      # 3
            gyro,        # 3
            gravity,     # 3
            command,     # 3
            joint_angles - self._default_pose, # 29
            joint_vel,   # 29
            last_act,    # 29
            phase,       # 4
        ])
        
        return obs

    def apply_control_numpy(self, data: mujoco.MjData, control: np.ndarray) -> None:
        """Apply control for numpy-based simulation (e.g. run_interactive)."""
        # Construct observation from MjData
        # We need to replicate _get_obs but for MjData (numpy)
        
        def get_sensor(name):
            id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if id == -1:
                raise ValueError(f"Sensor {name} not found")
            adr = self.mj_model.sensor_adr[id]
            dim = self.mj_model.sensor_dim[id]
            return data.sensordata[adr:adr+dim]

        linvel = get_sensor("local_linvel_pelvis")
        gyro = get_sensor("gyro_pelvis")
        gravity = get_sensor("upvector_pelvis")
        
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]
        last_act = data.ctrl
        
        # Phase
        gait_freq = 1.5
        phase_base = 2 * np.pi * data.time * gait_freq
        phase = np.array([phase_base, phase_base + np.pi])
        phase = np.concatenate([np.cos(phase), np.sin(phase)])
        
        obs = np.hstack([
            linvel,
            gyro,
            gravity,
            control,
            joint_angles - np.array(self._default_pose),
            joint_vel,
            last_act,
            phase,
        ])
        
        # Run inference
        # inference_fn expects JAX arrays, but handles numpy.
        # We need a PRNG key.
        rng = jax.random.PRNGKey(0)
        action, _ = self.inference_fn(obs, rng)
        
        # Convert action to numpy
        action = np.array(action)
        
        # Apply action
        action_scale = 0.5
        motor_targets = np.array(self._default_pose) + action * action_scale
        
        data.ctrl[:] = motor_targets

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        # Goal: reach target (x, y, theta)
        # We can store target in self, or pass it?
        # Usually Task has fixed goal or we update it.
        # For now, let's assume target is (2, 0, 0) fixed, or we can make it a property.
        # Better: use control to drive towards target?
        # The user said: "goal be to reach a desired x,y,theta pose in the world."
        # The MPC produces velocity commands (control).
        # So the cost should penalize deviation from the path to the target.
        # Or simpler: Cost = distance to target + orientation error.
        # And the MPC will find velocity commands that minimize this cost.
        # Yes.
        
        target_pos = jnp.array([2.0, 0.0])
        target_theta = 0.0
        
        pos = state.qpos[:2]
        theta = jnp.arctan2(2 * (state.qpos[3] * state.qpos[6] + state.qpos[4] * state.qpos[5]), 
                            1 - 2 * (state.qpos[5]**2 + state.qpos[6]**2)) # Yaw from quat
        
        dist_cost = jnp.sum((pos - target_pos)**2)
        theta_cost = (theta - target_theta)**2
        
        # Regularize control
        ctrl_cost = jnp.sum(control**2) * 0.01
        
        return dist_cost + theta_cost + ctrl_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        return self.running_cost(state, jnp.zeros(3))

