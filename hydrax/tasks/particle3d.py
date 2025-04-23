from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class Particle3D(Task):
    """A velocity-controlled 3D point mass that chases a target position."""

    def __init__(self):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/particle3d/scene.xml")

        super().__init__(
            mj_model,
            trace_sites=["particle"],
        )

        self.particle_id = mj_model.site("particle").id
        self.reference_id = mj_model.site("reference").id

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages target tracking."""
        terminal_cost = self.terminal_cost(state)
        return terminal_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.particle_id] - state.mocap_pos[0])
        )
        # control_cost = jnp.sum(jnp.square(state.actuator_force))

        # Penalize control effort
        control_cost = jnp.sum(
            jnp.square(state.ctrl - state.site_xpos[self.particle_id])
        )

        return 1e2 * position_cost + 1e1 * control_cost

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
        shift = jax.random.uniform(rng, (3,), minval=-0.01, maxval=0.01)
        shift = jnp.concatenate([shift, jnp.zeros(4)])
        return {"qpos": data.qpos + shift}
