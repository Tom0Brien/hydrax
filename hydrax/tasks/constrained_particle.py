from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class ConstrainedParticle(Task):
    """A particle task with a constraint to stay within a 0.1 x 0.1 box."""

    def __init__(self, box_size: float = 0.1) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/particle_constrained/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["pointmass"])
        self.pointmass_id = mj_model.site("pointmass").id
        self.box_size = box_size
        # Center of the box constraint - could be made adjustable
        self.box_center = jnp.zeros(3)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages target tracking."""
        state_cost = self.terminal_cost(state)
        control_cost = jnp.sum(jnp.square(control))
        return state_cost + 0.1 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.pointmass_id] - state.mocap_pos[0])
        )
        velocity_cost = jnp.sum(jnp.square(state.qvel))
        return 5.0 * position_cost + 0.1 * velocity_cost

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomly perturb the actuator gains."""
        multiplier = jax.random.uniform(
            rng, self.model.actuator_gainprm[:, 0].shape, minval=0.9, maxval=1.1
        )
        new_gains = self.model.actuator_gainprm[:, 0] * multiplier
        new_gains = self.model.actuator_gainprm.at[:, 0].set(new_gains)
        return {"actuator_gainprm": new_gains}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly shift the measured particle position."""
        shift = jax.random.uniform(rng, (2,), minval=-0.01, maxval=0.01)
        return {"qpos": data.qpos + shift}

    def constraint_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Constraint cost that's positive when outside the box, negative inside.

        For the constraint c(x,u) ≤ 0, we define the cost to be:
        - Negative (satisfying constraint) when particle is inside the box
        - Positive (violating constraint) when particle is outside the box
        """
        # Get current position
        position = state.site_xpos[self.pointmass_id]

        # Calculate distance from box center (only in x-y plane)
        distance = jnp.abs(position[:2] - self.box_center[:2])

        # Maximum distance from center in any dimension
        max_distance = jnp.max(distance)

        # Return cost: positive when outside, negative when inside
        # Magnitude of cost increases with distance from boundary
        return max_distance - self.box_size
