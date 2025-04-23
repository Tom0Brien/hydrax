from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task
from hydrax.utils.math import mat_to_quat, orientation_error


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
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/pushbox/scene.xml")

        super().__init__(
            mj_model,
            trace_sites=["pusher"],
        )

        # Get sensor ids
        self.box_id = mj_model.body("box").id
        self.pusher_id = mj_model.body("pusher").id
        self.reference_id = mj_model.site("reference").id

    def _close_to_block_err(self, state: mjx.Data) -> jax.Array:
        """Position of the pusher relative to the desired pushing position."""
        # Get current box position
        current_box_pos = state.xpos[self.box_id]
        # Get desired box position from mocap
        desired_box_pos = state.mocap_pos[0]

        # Calculate direction vector from box to goal
        box_to_goal = desired_box_pos - current_box_pos
        # Normalize the direction vector
        distance = jnp.linalg.norm(box_to_goal)
        direction = box_to_goal / jnp.maximum(distance, 1e-6)  # Avoid division by zero
        # Calculate desired pusher position: 5cm back from box along this direction
        desired_pusher_pos = current_box_pos - 0.05 * direction  # 5cm offset

        # Get current pusher position
        pusher_pos = state.xpos[self.pusher_id]
        return pusher_pos - desired_pusher_pos

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        # Get current box position and orientation
        current_box_pos = state.xpos[self.box_id]
        box_rot = state.xmat[self.box_id].reshape((3, 3))
        box_quat = mat_to_quat(box_rot)

        # Get desired box position and orientation from mocap
        desired_box_pos = state.mocap_pos[0]
        desired_box_orientation = state.mocap_quat[0]

        # Calculate costs
        box_pos_cost = jnp.sum(jnp.square(current_box_pos - desired_box_pos))
        box_orientation_cost = jnp.sum(
            jnp.square(orientation_error(box_quat, desired_box_orientation, box_rot))
        )

        # Calculate pusher position cost
        pusher_err = self._close_to_block_err(state)
        box_to_pusher_cost = jnp.sum(jnp.square(pusher_err))

        control_cost = jnp.sum(jnp.square(state.actuator_force))
        return (
            100.0 * box_pos_cost  # Box position
            + 10.0 * box_orientation_cost  # Box orientation
            + 40.0 * box_to_pusher_cost  # Close to box cost
            + 0.001 * control_cost  # Control cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ℓ_T(x_T)."""

        return self.running_cost(state, jnp.zeros_like(self.model.actuator_ctrlrange))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the level of friction."""
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.1, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}
