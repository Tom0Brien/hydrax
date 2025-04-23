from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task
from hydrax.utils.math import mat_to_quat, eul_to_quat, orientation_error, mat_to_rpy


class FrankaPush(Task):
    """Franka pushing a box to a target pose."""

    def __init__(
        self,
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/franka_emika_panda/mjx_scene_box_push.xml"
        )

        super().__init__(
            mj_model,
            trace_sites=["gripper"],
        )

        # Cartesian control gains defaults
        self.Kp = jnp.diag(jnp.array([300.0, 300.0, 300.0, 50.0, 50.0, 50.0]))
        self.Kd = 2.0 * jnp.sqrt(self.Kp)
        self.nullspace_stiffness = 10.0
        self.q_d_nullspace = jnp.array(
            [-0.196, -0.189, 0.182, -2.1, 0.0378, 1.91, 0.756, 0, 0]
        )
        self.cartesian_u_min = jnp.array([-0, -1, 0.3, -3.14, -3.14, -3.14])
        self.cartesian_u_max = jnp.array([1, 1, 1, 3.14, 3.14, 3.14])

        self.ee_site_id = mj_model.site("gripper").id
        self.reference_id = mj_model.site("reference").id
        self.box_id = mj_model.body("box").id

        # Table constraint parameters
        self.table_center = jnp.array([0.5, 0.0, 0.39])  # Center of the table surface
        self.table_size = jnp.array(
            [0.35, 0.35]
        )  # Safe area size (slightly smaller than table)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages pushing the box to the goal."""
        state_cost = self.terminal_cost(state)
        control_cost = jnp.sum(jnp.square(state.actuator_force))
        return state_cost + 0.001 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        # Use mocap position as the desired box position
        desired_box_pos = state.mocap_pos[0]
        # Extract the mocap orientation as desired box orientation
        desired_box_orientation = state.mocap_quat[0]

        current_box_pos = state.xpos[self.box_id]
        box_rot = state.xmat[self.box_id].reshape((3, 3))
        box_quat = mat_to_quat(box_rot)
        box_pos_cost = jnp.sum(jnp.square(current_box_pos - desired_box_pos))
        box_orientation_cost = jnp.sum(
            jnp.square(orientation_error(box_quat, desired_box_orientation, box_rot))
        )

        # Desired gripper position: 5cm back from box along box-to-goal line
        # Calculate direction vector from box to goal
        box_to_goal = desired_box_pos - current_box_pos
        # Normalize the direction vector
        distance = jnp.linalg.norm(box_to_goal)
        direction = box_to_goal / jnp.maximum(distance, 1e-6)  # Avoid division by zero
        # Calculate desired gripper position: 5cm back from box along this direction
        desired_gripper_pos = current_box_pos - 0.05 * direction  # 5cm offset

        gripper_pos = state.site_xpos[self.ee_site_id]
        box_to_gripper_cost = jnp.sum(jnp.square(gripper_pos - desired_gripper_pos))

        # Desired gripper orientation (roll and pitch, yaw should be able to vary)
        gripper_rot = state.site_xmat[self.ee_site_id].reshape((3, 3))
        gripper_rpy = mat_to_rpy(gripper_rot)
        gripper_orientation_cost = jnp.sum(
            jnp.square(gripper_rpy[:2] - jnp.array([-3.14, 0.0]))
        )

        return (
            100.0 * box_pos_cost  # Box position
            + 10.0 * box_orientation_cost  # Box orientation
            + 40.0 * box_to_gripper_cost  # Close to box cost
            + 0 * gripper_orientation_cost  # Gripper orientation
        )

    def constraint_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Constraint cost that's positive when the box is near/off the table edge.

        For the constraint c(x,u) ≤ 0, we define the cost to be:
        - Negative (satisfying constraint) when box is safely on the table
        - Positive (violating constraint) when box is near/off the edge
        """
        # Get current box position
        box_position = state.xpos[self.box_id]

        # Calculate distance from table center (only in x-y plane)
        distance = jnp.abs(box_position[:2] - self.table_center[:2])

        # Maximum distance from center in any dimension
        max_distance = jnp.max(distance)

        # Return cost: positive when outside safe area, negative when inside
        # Magnitude increases with distance from boundary
        return max_distance - self.table_size[0]
