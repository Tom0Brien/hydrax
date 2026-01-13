"""Domain perturbation utilities for Franka push experiments.

This module provides tools for testing RL + SPC robustness under
varied physical parameters (mass, friction) and object geometries.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import mujoco
import numpy as np


@dataclass
class PerturbationConfig:
    """Configuration for a domain perturbation experiment.
    
    Physical parameters are specified as multiplicative factors relative
    to the nominal values used during RL policy training.
    
    Attributes:
        name: Human-readable name for this perturbation scenario
        mass_scale: Multiplicative factor for object mass (1.0 = nominal)
        friction_scale: Multiplicative factor for sliding friction (1.0 = nominal)
        geometry: Object geometry variant ("cube", "square", "tblock")
    """
    name: str
    mass_scale: float = 1.0
    friction_scale: float = 1.0
    geometry: str = "cube"  # Options: "square", "tblock"
    
    def __repr__(self) -> str:
        return (f"PerturbationConfig(name='{self.name}', "
                f"mass={self.mass_scale}x, friction={self.friction_scale}x, "
                f"geometry='{self.geometry}')")


def apply_perturbation(
    mj_model: mujoco.MjModel,
    config: PerturbationConfig,
    object_body_name: str = "box",
    object_geom_name: str = "box",
    task: "Task" = None,
) -> None:
    """Apply domain perturbations to a MuJoCo model in-place.
    
    Modifies the model's physical parameters according to the config.
    This should be called AFTER loading the model but BEFORE simulation.
    
    Args:
        mj_model: MuJoCo model to modify (modified in-place)
        config: Perturbation configuration
        object_body_name: Name of the object body in the model
        object_geom_name: Name of the object geometry in the model
        task: Optional Task instance - if provided, also updates task.model
              (the mjx.Model used for controller planning)
        
    Note:
        - Mass is stored in mj_model.body_mass
        - Friction is stored in mj_model.geom_friction (3 values: sliding, torsional, rolling)
        - After modifying mass, we call mj_setConst to update derived quantities
        - If task is provided, we regenerate task.model from the modified mj_model
    """
    from mujoco import mjx
    
    # Get body and geom IDs
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
    geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, object_geom_name)
    
    if body_id == -1:
        raise ValueError(f"Body '{object_body_name}' not found in model")
    if geom_id == -1:
        raise ValueError(f"Geom '{object_geom_name}' not found in model")
    
    # Store original values for logging
    original_mass = mj_model.body_mass[body_id]
    original_friction = mj_model.geom_friction[geom_id, 0]  # Sliding friction
    
    # Apply mass scaling
    if config.mass_scale != 1.0:
        mj_model.body_mass[body_id] *= config.mass_scale
        
    # Apply friction scaling (element 0 is sliding friction)
    if config.friction_scale != 1.0:
        mj_model.geom_friction[geom_id, 0] *= config.friction_scale
        # Optionally scale torsional and rolling friction too
        mj_model.geom_friction[geom_id, 1] *= config.friction_scale
        mj_model.geom_friction[geom_id, 2] *= config.friction_scale
    
    # If task is provided, update its mjx.Model for controller planning
    if task is not None:
        task.model = mjx.put_model(mj_model)
        print(f"  Updated task.model (mjx.Model) for controller planning")
    
    # Log the changes
    new_mass = mj_model.body_mass[body_id]
    new_friction = mj_model.geom_friction[geom_id, 0]
    print(f"Applied perturbation '{config.name}':")
    print(f"  Mass: {original_mass:.4f} -> {new_mass:.4f} ({config.mass_scale}x)")
    print(f"  Friction: {original_friction:.4f} -> {new_friction:.4f} ({config.friction_scale}x)")


def get_standard_perturbations() -> List[PerturbationConfig]:
    """Get a standard set of perturbation scenarios for benchmarking.
    
    Returns a list of configs testing:
    - Nominal (no change)
    - Mass variations (light, heavy)
    - Friction variations (slippery, sticky)
    - Combined perturbations
    """
    return [
        # Nominal
        PerturbationConfig(name="nominal", mass_scale=1.0, friction_scale=1.0),
        
        # Mass variations
        PerturbationConfig(name="light_object", mass_scale=0.5, friction_scale=1.0),
        PerturbationConfig(name="heavy_object", mass_scale=2.0, friction_scale=1.0),
        PerturbationConfig(name="very_heavy", mass_scale=4.0, friction_scale=1.0),
        
        # Friction variations
        PerturbationConfig(name="slippery", mass_scale=1.0, friction_scale=0.3),
        PerturbationConfig(name="sticky", mass_scale=1.0, friction_scale=2.0),
        
        # Combined (challenging)
        PerturbationConfig(name="heavy_slippery", mass_scale=2.0, friction_scale=0.3),
        PerturbationConfig(name="light_sticky", mass_scale=0.5, friction_scale=2.0),
    ]


def get_quick_perturbations() -> List[PerturbationConfig]:
    """Get a quick subset of perturbations for fast iteration."""
    return [
        PerturbationConfig(name="nominal", mass_scale=1.0, friction_scale=1.0),
        PerturbationConfig(name="heavy_object", mass_scale=2.0, friction_scale=1.0),
        PerturbationConfig(name="slippery", mass_scale=1.0, friction_scale=0.3),
    ]


def get_geometry_perturbations() -> List[PerturbationConfig]:
    """Get perturbations testing different object geometries.
    
    These test generalization to different object shapes while keeping
    physical parameters (mass, friction) at nominal values.
    """
    return [
        # Original trained geometry
        PerturbationConfig(name="cube", mass_scale=1.0, friction_scale=1.0, geometry="cube"),
        
        # Alternative geometries (tests shape generalization)
        PerturbationConfig(name="square", mass_scale=1.0, friction_scale=1.0, geometry="square"),
        PerturbationConfig(name="tblock", mass_scale=1.0, friction_scale=1.0, geometry="tblock"),
    ]


def get_full_perturbations() -> List[PerturbationConfig]:
    """Get comprehensive set including both physics and geometry perturbations."""
    perturbations = []
    
    # Physics perturbations on cube (original geometry)
    perturbations.extend(get_standard_perturbations())
    
    # Geometry perturbations with nominal physics
    for geom in ["square", "tblock"]:
        perturbations.append(
            PerturbationConfig(name=f"{geom}_nominal", 
                             mass_scale=1.0, friction_scale=1.0, geometry=geom)
        )
        # Also test geometry + physics combinations
        perturbations.append(
            PerturbationConfig(name=f"{geom}_heavy",
                             mass_scale=2.0, friction_scale=1.0, geometry=geom)
        )
        perturbations.append(
            PerturbationConfig(name=f"{geom}_slippery",
                             mass_scale=1.0, friction_scale=0.3, geometry=geom)
        )
    
    return perturbations

