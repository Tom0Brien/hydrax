from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src.math import quat_sub

from hydrax import ROOT
from hydrax.task_base import Task


class PushBox(Task):
    """Push a box to a desired pose."""

    def __init__(
        self,
    ):
        """Load the MuJoCo model and set task parameters.

        Args:
            planning_horizon: The number of control steps (T) to plan over.
            sim_steps_per_control_step: The number of simulation steps per control step.
            gain_mode: The gain optimization mode to use (NONE, INDIVIDUAL, or SIMPLE).
        """
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/pushbox/scene.xml"
        )

        super().__init__(
            mj_model,
            trace_sites=["pusher"],
        )

        # Get ids
        self.box_body_id = mj_model.body("box").id
        self.box_geom_id = mj_model.geom("box_geom").id
        self.pusher_id = mj_model.body("pusher").id
        self.box_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "box_position"
        )
        self.box_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "box_orientation"
        )

    def _get_box_position(self, state: mjx.Data) -> jax.Array:
        sensor_adr = self.model.sensor_adr[self.box_position_sensor]
        return state.sensordata[sensor_adr : sensor_adr + 3]

    def _get_box_orientation(self, state: mjx.Data) -> jax.Array:
        sensor_adr = self.model.sensor_adr[self.box_orientation_sensor]
        return state.sensordata[sensor_adr : sensor_adr + 4]

    def _get_box_orientation_err(self, state: mjx.Data) -> jax.Array:
        """Orientation error between box and goal as quaternion difference."""
        box_quat = self._get_box_orientation(state)
        goal_quat = state.mocap_quat[0]
        return quat_sub(box_quat, goal_quat)

    def _close_to_block_err(self, state: mjx.Data) -> jax.Array:
        """Position of the pusher relative to the desired pushing position."""
        # Use sensor-based box position
        current_box_pos = self._get_box_position(state)
        desired_box_pos = state.mocap_pos[0]
        box_to_goal = desired_box_pos - current_box_pos
        distance = jnp.linalg.norm(box_to_goal)
        direction = box_to_goal / jnp.maximum(distance, 1e-6)
        desired_pusher_pos = current_box_pos - 0.05 * direction
        pusher_pos = state.xpos[self.pusher_id]
        return pusher_pos - desired_pusher_pos

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        current_box_pos = self._get_box_position(state)
        box_orientation_err = self._get_box_orientation_err(state)
        desired_box_pos = state.mocap_pos[0]
        box_pos_cost = jnp.sum(jnp.square(current_box_pos - desired_box_pos))
        box_orientation_cost = jnp.sum(jnp.square(box_orientation_err))
        pusher_err = self._close_to_block_err(state)
        box_to_pusher_cost = jnp.sum(jnp.square(pusher_err))
        control_cost = jnp.sum(jnp.square(state.actuator_force))
        return (
            100.0 * box_pos_cost
            + 10.0 * box_orientation_cost
            + 40.0 * box_to_pusher_cost
            + 0.001 * control_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ℓ_T(x_T)."""

        return self.running_cost(
            state, jnp.zeros_like(self.model.actuator_ctrlrange)
        )

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the level of friction."""
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.1, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}
