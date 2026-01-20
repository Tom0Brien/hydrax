"""Franka push task with table boundary constraints.

This module provides a constrained version of FrankaPushGeometry that ensures
the pushed object stays within a safe zone on the table. This demonstrates the
benefit of combining pre-trained RL policies with constrained online planning (CCEM).
"""

from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.tasks.franka.franka_push_geometry import FrankaPushGeometry, _XMLS_PATH

# Add constrained scene to available geometries
CONSTRAINED_GEOMETRY_XMLS = {
    "cube": "scene_panda_robotiq_cube_constrained.xml",
    "square": "scene_panda_robotiq_square_constrained.xml",
}

# Object half-sizes for each geometry (used for face-based constraints)
GEOMETRY_HALF_SIZES = {
    "cube": (0.07553, 0.11482),   # x, y half-sizes for rectangular cube
    "square": (0.05073, 0.05073), # x, y half-sizes for square cube
}


class FrankaPushConstrained(FrankaPushGeometry):
    """Franka push task with safe zone constraints.
    
    The pushed object must stay within a defined safe zone on the table.
    The constraint is based on the FACE of the cube - i.e., the entire
    object must be inside the safe zone, not just its center.
    
    Safe zone parameters (default):
    - Center: (0.525, 0.0) - center of the table workspace
    - Half-size: 0.2m - object faces must stay within ±0.2m of center
    
    The constraint accounts for the object's geometry, so larger objects
    have less room to move before hitting the boundary.
    """
    
    def __init__(
        self,
        geometry: Literal["cube", "square"] = "square",
        use_rl_policy: bool = True,
        safe_zone_center: tuple = (0.525, 0.0),
        safe_zone_half_size: float = 0.2,
    ):
        """Initialize the constrained push task.
        
        Args:
            geometry: Object geometry type ("cube" or "square")
            use_rl_policy: If True, load RL policy for residual control
            safe_zone_center: (x, y) center of the safe zone in meters
            safe_zone_half_size: Half-size of the square safe zone in meters
                                (measured to where object FACE would hit boundary)
        """
        # Store constraint parameters before calling super().__init__
        self.safe_zone_center = jnp.array(safe_zone_center)
        self.safe_zone_half_size = safe_zone_half_size
        
        # Store object half-size for face-based constraint
        if geometry not in GEOMETRY_HALF_SIZES:
            raise ValueError(f"Unknown geometry '{geometry}'")
        self.obj_half_size = jnp.array(GEOMETRY_HALF_SIZES[geometry])
        
        # Override XML path to use constrained scene
        self._use_constrained_scene = True
        
        # Call parent __init__
        super().__init__(geometry=geometry, use_rl_policy=use_rl_policy)
    
    def _get_xml_path(self, geometry: str) -> Path:
        """Get the XML path for the constrained scene."""
        if geometry not in CONSTRAINED_GEOMETRY_XMLS:
            raise ValueError(f"Constrained scene not available for geometry '{geometry}'. "
                           f"Available: {list(CONSTRAINED_GEOMETRY_XMLS.keys())}")
        return _XMLS_PATH / CONSTRAINED_GEOMETRY_XMLS[geometry]
    
    def constraint_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Constraint cost for keeping the box FACE within the safe zone.
        
        The constraint checks if any face of the cube would exit the safe zone.
        This is more restrictive than just checking the center.
        
        Returns:
            Positive value when any face is outside safe zone (constraint violated)
            Negative value when all faces are inside safe zone (constraint satisfied)
            Zero when a face is exactly on the boundary
        """
        # Get box center position (x, y only)
        box_center = state.xpos[self._obj_body, :2]
        
        # Calculate position of box faces (center ± half_size)
        # For a square constraint, we only care about max extent in each direction
        # Note: This assumes axis-aligned box (no rotation consideration for simplicity)
        box_max = box_center + self.obj_half_size
        box_min = box_center - self.obj_half_size
        
        # Safe zone boundaries
        safe_max = self.safe_zone_center + self.safe_zone_half_size
        safe_min = self.safe_zone_center - self.safe_zone_half_size
        
        # Check violation: how far any face extends beyond safe zone
        # Positive = outside, negative = inside
        violation_max = jnp.max(box_max - safe_max)  # Right/top face exits
        violation_min = jnp.max(safe_min - box_min)  # Left/bottom face exits
        
        # Overall constraint: max violation from any direction
        return jnp.maximum(violation_max, violation_min)

