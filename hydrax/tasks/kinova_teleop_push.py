from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task
from mujoco.mjx._src.math import quat_sub


class KinovaTeleopPush(Task):
    """Kinova teleoperation with a pushable box on a table."""

    def __init__(
        self,
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/kinova_gen3/scene_teleop_push.xml"
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

        # Minimum height above table for gripper (5 cm)
        self.min_gripper_height = 0.0

    def _get_gripper_position_err(self, state: mjx.Data) -> jax.Array:
        """Position of the gripper relative to the target position."""
        sensor_adr = self.model.sensor_adr[self.gripper_position_sensor]
        gripper_position = state.sensordata[sensor_adr : sensor_adr + 3]
        desired_position = state.mocap_pos[0]
        return gripper_position - desired_position

    def _get_gripper_orientation_err(self, state: mjx.Data) -> jax.Array:
        """Orientation of the gripper relative to the target orientation."""
        sensor_adr = self.model.sensor_adr[self.gripper_orientation_sensor]
        gripper_quat = state.sensordata[sensor_adr : sensor_adr + 4]
        goal_quat = state.mocap_quat[0]
        return mjx._src.math.quat_sub(gripper_quat, goal_quat)

    def _get_box_position(self, state: mjx.Data) -> jax.Array:
        """Position of the box."""
        sensor_adr = self.model.sensor_adr[self.box_position_sensor]
        return state.sensordata[sensor_adr : sensor_adr + 3]

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages teleoperating the gripper to the goal."""
        # Gripper position and orientation tracking costs
        position_cost = jnp.sum(
            jnp.square(self._get_gripper_position_err(state))
        )
        orientation_cost = jnp.sum(
            jnp.square(self._get_gripper_orientation_err(state))
        )

        # Control effort cost
        control_cost = jnp.sum(jnp.square(state.actuator_force))

        return (
            100.0 * position_cost  # Gripper position
            + 10.0 * orientation_cost  # Gripper orientation
            + 0.001 * control_cost  # Control effort
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return self.running_cost(state, state.ctrl)

    def constraint_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Combined constraint cost for box on table and gripper height."""
        # Box constraint (positive when box is near/off the table edge)
        box_position = self._get_box_position(state)
        distance_from_center = jnp.abs(box_position[:2] - self.table_center[:2])
        max_distance = jnp.max(distance_from_center)
        box_constraint = max_distance - self.table_size[0]

        # Gripper height constraint (positive when gripper is too close to table)
        sensor_adr = self.model.sensor_adr[self.gripper_position_sensor]
        gripper_position = state.sensordata[sensor_adr : sensor_adr + 3]
        gripper_height = gripper_position[2]  # z-coordinate
        table_height = self.table_center[2]
        height_margin = self.min_gripper_height
        gripper_constraint = height_margin - (gripper_height - table_height)

        # Return the constraint violation
        return jnp.maximum(box_constraint, 0.0) + jnp.maximum(
            gripper_constraint, 0.0
        )
