from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task
from mujoco.mjx._src.math import quat_sub


class KinovaPush(Task):
    """Kinova Gen3 pushing a box to a target pose."""

    def __init__(
        self,
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/kinova_gen3/scene_box_push.xml"
        )

        super().__init__(
            mj_model,
            trace_sites=["gripper"],
        )

        self.ee_site_id = mj_model.site("gripper").id
        self.reference_id = mj_model.site("reference").id

        # Get sensor ids
        self.box_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "box_position"
        )
        self.box_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "box_orientation"
        )
        self.gripper_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "gripper_position"
        )
        self.gripper_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "gripper_orientation"
        )

        # Table constraint parameters
        self.table_center = jnp.array(
            [0.5, 0.0, 0.03]
        )  # Center of the table surface
        self.table_size = jnp.array(
            [0.35, 0.35]
        )  # Safe area size (slightly smaller than table)

    def _get_box_position_err(self, state: mjx.Data) -> jax.Array:
        """Position of the box relative to the target position."""
        sensor_adr = self.model.sensor_adr[self.box_position_sensor]
        box_position = state.sensordata[sensor_adr : sensor_adr + 3]
        desired_position = state.mocap_pos[0]
        return box_position - desired_position

    def _get_box_orientation_err(self, state: mjx.Data) -> jax.Array:
        """Orientation of the box relative to the target orientation."""
        sensor_adr = self.model.sensor_adr[self.box_orientation_sensor]
        box_quat = state.sensordata[sensor_adr : sensor_adr + 4]
        goal_quat = state.mocap_quat[0]
        return mjx._src.math.quat_sub(box_quat, goal_quat)

    def _get_gripper_position_err(self, state: mjx.Data) -> jax.Array:
        """Position of the gripper relative to the desired pushing position."""
        sensor_adr = self.model.sensor_adr[self.box_position_sensor]
        box_position = state.sensordata[sensor_adr : sensor_adr + 3]
        desired_position = state.mocap_pos[0]

        # Calculate direction vector from box to goal
        box_to_goal = desired_position - box_position
        # Normalize the direction vector
        distance = jnp.linalg.norm(box_to_goal)
        direction = box_to_goal / jnp.maximum(
            distance, 1e-6
        )  # Avoid division by zero
        # Calculate desired gripper position: 5cm back from box along this direction
        desired_gripper_pos = box_position - 0.05 * direction  # 5cm offset

        sensor_adr = self.model.sensor_adr[self.gripper_position_sensor]
        gripper_position = state.sensordata[sensor_adr : sensor_adr + 3]
        return gripper_position - desired_gripper_pos

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages pushing the box to the goal."""
        box_pos_cost = jnp.sum(jnp.square(self._get_box_position_err(state)))
        box_orientation_cost = jnp.sum(
            jnp.square(self._get_box_orientation_err(state))
        )
        gripper_pos_cost = jnp.sum(
            jnp.square(self._get_gripper_position_err(state))
        )
        control_cost = jnp.sum(jnp.square(state.actuator_force))

        return (
            1e3 * box_pos_cost  # Box position
            + 10.0 * box_orientation_cost  # Box orientation
            + 40.0 * gripper_pos_cost  # Close to box cost
            + 0.001 * control_cost  # Control effort
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return self.running_cost(state, state.ctrl)

    def constraint_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Constraint cost that's positive when the box is near/off the table edge."""
        sensor_adr = self.model.sensor_adr[self.box_position_sensor]
        box_position = state.sensordata[sensor_adr : sensor_adr + 3]

        # Calculate distance from table center (only in x-y plane)
        distance = jnp.abs(box_position[:2] - self.table_center[:2])

        # Maximum distance from center in any dimension
        max_distance = jnp.max(distance)

        # Return cost: positive when outside safe area, negative when inside
        # Magnitude increases with distance from boundary
        return max_distance - self.table_size[0]
