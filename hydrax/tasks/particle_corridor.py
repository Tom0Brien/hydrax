from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class ParticleCorridor(Task):
    """A particle pushes a box out of an enclosed room through a door opening."""

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
            ROOT + "/models/particle_corridor/scene.xml"
        )

        super().__init__(
            mj_model,
            trace_sites=["particle", "box_site"],
        )

        self.particle_id = mj_model.site("particle").id
        self.reference_id = mj_model.site("reference").id
        self.door_center_id = mj_model.site("door_center").id

        # Sensor IDs for touch sensors
        self.box_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "box_position"
        )

        # Wall touch sensors
        self.left_wall_touch_top_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_wall_touch_top"
        )
        self.left_wall_touch_bottom_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_wall_touch_bottom"
        )
        self.right_wall_touch_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_wall_touch"
        )
        self.top_wall_touch_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "top_wall_touch"
        )
        self.bottom_wall_touch_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "bottom_wall_touch"
        )

        # Define parameters for wall collision detection
        self.wall_collision_penalty = 1e3  # Penalty for wall collisions

    def _get_box_position(self, state: mjx.Data) -> jax.Array:
        """Get the position of the box."""
        sensor_adr = self.model.sensor_adr[self.box_position_sensor]
        return state.sensordata[sensor_adr : sensor_adr + 2]  # Only x, y

    def _get_sensor_data(self, state: mjx.Data, sensor_id: int) -> jax.Array:
        """Gets sensor data given sensor ID."""
        sensor_adr = self.model.sensor_adr[sensor_id]
        sensor_dim = self.model.sensor_dim[sensor_id]
        return state.sensordata[sensor_adr : sensor_adr + sensor_dim]

    def _wall_collision(self, state: mjx.Data) -> jax.Array:
        """Calculate cost for box collision with walls using touch sensors.

        The touch sensor returns the normal force on the sensor site.
        A non-zero value indicates contact.
        """
        # Get touch data from all wall sensors
        left_wall_touch_top = self._get_sensor_data(
            state, self.left_wall_touch_top_sensor
        )
        left_wall_touch_bottom = self._get_sensor_data(
            state, self.left_wall_touch_bottom_sensor
        )
        right_wall_touch = self._get_sensor_data(state, self.right_wall_touch_sensor)
        top_wall_touch = self._get_sensor_data(state, self.top_wall_touch_sensor)
        bottom_wall_touch = self._get_sensor_data(state, self.bottom_wall_touch_sensor)

        # Calculate magnitude of touch forces
        left_top_touch_magnitude = jnp.sum(jnp.square(left_wall_touch_top))
        left_bottom_touch_magnitude = jnp.sum(jnp.square(left_wall_touch_bottom))
        right_touch_magnitude = jnp.sum(jnp.square(right_wall_touch))
        top_touch_magnitude = jnp.sum(jnp.square(top_wall_touch))
        bottom_touch_magnitude = jnp.sum(jnp.square(bottom_wall_touch))

        # Combine them in an array and check if any are greater than zero
        magnitudes = jnp.array(
            [
                left_top_touch_magnitude,
                left_bottom_touch_magnitude,
                right_touch_magnitude,
                top_touch_magnitude,
                bottom_touch_magnitude,
            ]
        )

        return jnp.where(jnp.any(magnitudes > 0.0), 1.0, 0.0)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost encourages target tracking and avoiding wall collisions."""
        # Particle tracking cost
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.particle_id] - state.mocap_pos[0])
        )

        # Control cost
        control_cost = jnp.sum(
            jnp.square(jnp.array([control[0], control[1], 0]) - state.mocap_pos[0])
        )

        # Wall collision cost based on touch sensors
        wall_cost = self._wall_collision(state)

        return (
            1e1 * position_cost
            + self.wall_collision_penalty * wall_cost
            + 1e0 * control_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost encourages pushing the box out through the door and tracking the target."""
        # Get the box position
        box_pos = self._get_box_position(state)

        # Door is at x=0.1, so we want the box to be beyond that (x > 0.1)
        box_escaped = jnp.where(box_pos[0] > 0.1, 0.0, 1.0)

        # Wall collision penalty
        wall_cost = self._wall_collision(state)

        # Particle position cost (should track the mocap target)
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.particle_id] - state.mocap_pos[0])
        )

        return 1e6 * wall_cost + 1e1 * position_cost  # 1e3 * box_escaped +

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
        return {"qpos": data.qpos.at[:2].add(shift)}
