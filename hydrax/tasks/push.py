from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class Push(Task):
    """Push an object to a goal position using an actuated robot."""

    def __init__(
        self, planning_horizon: int = 5, sim_steps_per_control_step: int = 10
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/push/scene.xml"
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["robot_site"],
        )

        # Get sensor id for the block's position (could be absolute or relative to goal,
        # depending on the sensor definition in the XML)
        self.object_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position"
        )

        # Retrieve the robot site's id to later access its position.
        self.robot_site_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SITE, "robot_site"
        )

    def _get_block_position(self, state: mjx.Data) -> jax.Array:
        """Return the block's position as measured by the sensor.

        Depending on the sensor configuration in the XML,
        this may be the block's absolute position or its error relative to the goal.
        """
        sensor_adr = self.model.sensor_adr[self.object_position_sensor]
        # Assuming the sensor returns a 3D position.
        return state.sensordata[sensor_adr : sensor_adr + 3]

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        This cost includes:
          - A cost on the object's (block's) position error (e.g. distance from goal),
          - A control regularization term, and
          - A cost on the distance between the robot and the block.
        """
        # Object (block) position cost (e.g., error relative to the goal)
        block_pos = self._get_block_position(state)
        position_cost = jnp.sum(jnp.square(block_pos))

        # Control regularization cost
        control_cost = jnp.sum(jnp.square(control))

        # Robot-block proximity cost:
        # Get the robot's current position from the site's positions.
        robot_pos = state.site_xpos[self.robot_site_id]
        # Compute squared distance between the robot and the block.
        proximity_cost = jnp.sum(jnp.square(robot_pos - block_pos))

        # Total running cost is the sum of the three cost terms.
        return 100*position_cost + 0.01 * control_cost + proximity_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ℓ_T(x_T)."""
        # Use zero control for terminal cost.
        return self.running_cost(state, jnp.zeros(self.model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the level of friction."""
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.1, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}  # Ensure this returns a dict
