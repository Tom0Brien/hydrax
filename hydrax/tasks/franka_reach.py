from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task
from hydrax.utils.math import mat_to_quat, quat_to_vel
from mujoco.mjx._src.math import quat_sub, quat_mul, quat_inv


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

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages target tracking."""
        # Use mocap position as the desired pose
        desired_position = state.mocap_pos[0]
        desired_orientation = state.mocap_quat[0]

        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.ee_site_id] - desired_position)
        )

        # Compute orientation error using the approach from impedance_controllers.py
        current_rot = state.site_xmat[self.ee_site_id].reshape((3, 3))
        current_quat = mat_to_quat(current_rot)
        current_quat_conj = quat_inv(current_quat)
        error_quat = quat_mul(desired_orientation, current_quat_conj)
        ori_error = quat_to_vel(error_quat, 1.0)

        orientation_cost = jnp.sum(jnp.square(ori_error))

        # Penalize control effort (distance between reference and ee)
        control_cost = jnp.sum(
            jnp.square(state.ctrl[:3] - state.site_xpos[self.ee_site_id])
        )
        return 1e1 * position_cost + 1e0 * orientation_cost + 1e-2 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""

        return self.running_cost(state, state.ctrl)
