"""Franka push task with cylinder obstacle avoidance constraint.

This module provides a constrained version of FrankaPushGeometry that includes
a cylinder obstacle in the workspace that the pushed object must avoid.
This demonstrates the benefit of combining pre-trained RL policies with 
constrained online planning (CCEM) for obstacle avoidance.
"""

from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.tasks.franka.franka_push_geometry import FrankaPushGeometry, _XMLS_PATH

# Add obstacle scene to available geometries
OBSTACLE_GEOMETRY_XMLS = {
    "square": "scene_panda_robotiq_cube_obstacle.xml",
}

# Object half-sizes for each geometry (used for collision margin)
GEOMETRY_HALF_SIZES = {
    "cube": (0.07553, 0.11482),   # x, y half-sizes for rectangular cube
    "square": (0.05073, 0.05073), # x, y half-sizes for square cube
}

# Cylinder obstacle parameters
OBSTACLE_POS = jnp.array([0.55, 0.05])  # x, y position of cylinder center
OBSTACLE_RADIUS = 0.04  # radius of the cylinder


class FrankaPushConstrained(FrankaPushGeometry):
    """Franka push task with cylinder obstacle avoidance.
    
    The pushed object must avoid colliding with a fixed cylinder obstacle
    placed in the workspace. The constraint is based on the distance between
    the box center and the cylinder, accounting for both the cylinder radius
    and an approximation of the box extent.
    
    Obstacle parameters:
    - Position: (0.55, 0.05) - between typical box start and target
    - Radius: 0.04m - creates a meaningful obstacle in the workspace
    
    The constraint uses a safety margin to account for the box dimensions,
    approximating the box as a circle with radius equal to the max half-size.
    """
    
    def __init__(
        self,
        geometry: Literal["square"] = "square",
        use_rl_policy: bool = True,
        safety_margin: float = 0.02,
    ):
        """Initialize the obstacle avoidance push task.
        
        Args:
            geometry: Object geometry type (currently only "cube" supported)
            use_rl_policy: If True, load RL policy for residual control
            safety_margin: Additional margin beyond box+cylinder radii (meters)
        """
        # Store constraint parameters before calling super().__init__
        self.safety_margin = safety_margin
        self.obstacle_pos = OBSTACLE_POS
        self.obstacle_radius = OBSTACLE_RADIUS
        
        # Store object half-size for collision check
        if geometry not in GEOMETRY_HALF_SIZES:
            raise ValueError(f"Unknown geometry '{geometry}'")
        obj_half_size = GEOMETRY_HALF_SIZES[geometry]
        # Approximate box as circle with radius = max dimension
        self.obj_radius = jnp.sqrt(obj_half_size[0]**2 + obj_half_size[1]**2)
        
        # Minimum safe distance (cylinder radius + box radius + margin)
        self.min_safe_dist = self.obstacle_radius + self.obj_radius + self.safety_margin
        
        # Call parent __init__
        super().__init__(geometry=geometry, use_rl_policy=use_rl_policy)
        
        # Store obstacle body ID for potential future use
        self._obstacle_body = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "obstacle"
        )
    
    def _get_xml_path(self, geometry: str) -> Path:
        """Get the XML path for the obstacle scene."""
        if geometry not in OBSTACLE_GEOMETRY_XMLS:
            raise ValueError(f"Obstacle scene not available for geometry '{geometry}'. "
                           f"Available: {list(OBSTACLE_GEOMETRY_XMLS.keys())}")
        return _XMLS_PATH / OBSTACLE_GEOMETRY_XMLS[geometry]
    
    def constraint_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Constraint cost for obstacle avoidance.
        
        Computes the signed distance from the box to the obstacle cylinder.
        Uses the box center position and approximates the box as a circle.
        
        The obstacle position is read from state.xpos to support per-environment
        obstacle positions when using vmap.
        
        Returns:
            Positive value when box is too close to obstacle (constraint violated)
            Negative value when box is safely away from obstacle (constraint satisfied)
            Zero when box edge is exactly at the safety boundary
        """
        # Get box center position (x, y only)
        box_center = state.xpos[self._obj_body, :2]
        
        # Get obstacle position from state (supports per-env obstacle positions)
        obstacle_center = state.xpos[self._obstacle_body, :2]
        
        # Distance from box center to obstacle center
        dist_to_obstacle = jnp.linalg.norm(box_center - obstacle_center)
        
        # Constraint violation: positive when distance < min_safe_dist
        # Negative when safe, zero at boundary
        violation = self.min_safe_dist - dist_to_obstacle
        
        return violation

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Running cost with repulsive potential for obstacle avoidance.
        
        Extends the parent running cost with a smooth repulsive potential
        around the obstacle. This helps guide the optimizer around the obstacle
        rather than getting stuck in local minima where all forward paths
        are blocked.
        
        The repulsion uses an inverse-distance potential that:
        - Is zero outside the influence radius (2x min_safe_dist)
        - Increases smoothly as the box approaches the obstacle
        - Creates a gradient that pushes samples away from the obstacle
        """
        # Get base running cost from parent
        base_cost = super().running_cost(state, control)
        
        # Compute repulsive potential around obstacle
        box_xy = state.xpos[self._obj_body, :2]
        obstacle_xy = state.xpos[self._obstacle_body, :2]
        dist = jnp.linalg.norm(box_xy - obstacle_xy)
        
        # Influence radius: repulsion active within this distance
        influence_radius = self.min_safe_dist * 2.5
        
        # Smooth inverse-distance repulsion (Khatib-style potential field)
        # Repulsion = 0.5 * eta * (1/dist - 1/influence_radius)^2 when dist < influence_radius
        # This creates a smooth gradient pushing away from obstacle
        eta = 0.5  # Repulsion strength (tunable)
        repulsion = jnp.where(
            dist < influence_radius,
            0.5 * eta * jnp.square(1.0 / (dist + 1e-3) - 1.0 / influence_radius),
            0.0
        )
        
        return base_cost + repulsion
