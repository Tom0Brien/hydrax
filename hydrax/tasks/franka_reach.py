from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task
from mujoco.mjx._src.math import quat_sub


class FrankaReach(Task):
    """Franka to reach a target position."""

    def __init__(
        self,
    ):
        """Load the MuJoCo model and set task parameters.

        Args:
            planning_horizon: The number of control steps (T) to plan over.
            sim_steps_per_control_step: The number of simulation steps per control step.
            control_mode: The control mode to use.
                          CARTESIAN_SIMPLE_VI is recommended for Franka as it optimizes
                          only translational and rotational p-gains with d-gains automatically set.
            config: Optional dictionary with gain and control limit configurations. May include:
                         For GENERAL_VI mode:
                           'p_min', 'p_max', 'd_min', 'd_max'
                         For CARTESIAN_SIMPLE_VI mode:
                           'trans_p_min', 'trans_p_max', 'rot_p_min', 'rot_p_max'
                         For CARTESIAN mode (fixed gains and limits):
                           'trans_p', 'rot_p', 'pos_min', 'pos_max', 'rot_min', 'rot_max'
        """
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/franka_emika_panda/mjx_scene_reach.xml"
        )

        super().__init__(
            mj_model,
            trace_sites=["gripper"],
        )

        self.Kp = jnp.diag(jnp.array([100.0, 100.0, 100.0, 30.0, 30.0, 30.0]))
        self.Kd = 2.0 * jnp.sqrt(self.Kp)
        self.nullspace_stiffness = 0.0
        self.q_d_nullspace = jnp.array(
            [-0.196, -0.189, 0.182, -2.1, 0.0378, 1.91, 0.756, 0, 0]
        )
        self.u_min = jnp.array([-0, -1, 0.3, -3.14, -3.14, -3.14])
        self.u_max = jnp.array([1, 1, 1, 3.14, 3.14, 3.14])

        self.ee_site_id = mj_model.site("gripper").id
        self.reference_id = mj_model.site("reference").id
        # Get sensor ids
        self.gripper_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "gripper_position"
        )
        self.gripper_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "gripper_orientation"
        )

    def _get_gripper_position_err(self, state: mjx.Data) -> jax.Array:
        """Position of the gripper relative to the target grasp position."""
        gripper_position = self.model.sensor_adr[self.gripper_position_sensor]
        desired_position = state.mocap_pos[0]
        return (
            state.sensordata[gripper_position : gripper_position + 3]
            - desired_position
        )

    def _get_gripper_orientation_err(self, state: mjx.Data) -> jax.Array:
        """Orientation of the gripper relative to the target grasp orientation."""
        sensor_adr = self.model.sensor_adr[self.gripper_orientation_sensor]
        gripper_quat = state.sensordata[sensor_adr : sensor_adr + 4]

        # Quaternion subtraction gives us rotation relative to goal
        goal_quat = state.mocap_quat[0]
        return mjx._src.math.quat_sub(gripper_quat, goal_quat)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages target tracking."""
        position_cost = jnp.sum(
            jnp.square(self._get_gripper_position_err(state))
        )
        orientation_cost = jnp.sum(
            jnp.square(self._get_gripper_orientation_err(state))
        )

        # Penalize control effort (distance between reference and ee)
        control_cost = jnp.sum(
            jnp.square(state.ctrl[:3] - state.site_xpos[self.ee_site_id])
        )
        return (
            1e1 * position_cost + 1e0 * orientation_cost + 1e-2 * control_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""

        return self.running_cost(state, state.ctrl)
