"""Interactive simulation of the constrained particle tracking task.

This example demonstrates CCEM (Constrained CEM) on a particle that must
track a moving target while staying within a box constraint.

Usage:
    python examples/constrained_particle.py

Controls:
    - Double click on the green target, then drag with [ctrl + right-click]
    - The particle must stay within a 0.1 x 0.1 box centered at the origin
"""

import mujoco

from hydrax.algs import CCEM
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.constrained_particle import ConstrainedParticle

# Define the constrained task
task = ConstrainedParticle(box_size=0.1)

# Create the Constrained CEM controller
ctrl = CCEM(
    task,
    num_samples=512,  # Sample more trajectories to find feasible ones
    num_elites=20,  # Number of elite samples to keep
    sigma_start=0.3,  # Initial standard deviation
    sigma_min=0.05,  # Minimum standard deviation
    explore_fraction=0.5,  # Fraction of samples to keep at sigma_start
    num_randomizations=1,  # No domain randomization for simplicity
    plan_horizon=0.25,  # Planning horizon in seconds
    spline_type="zero",  # Zero-order hold for control interpolation
    num_knots=11,  # Number of control knots
)

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=False,
    max_traces=50,
)
