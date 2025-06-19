import argparse
import mujoco
from mujoco import mjx
import jax.numpy as jnp
from hydrax.algs import MPPI, CEM, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.pushbox import PushBox
from hydrax.identifiers import CEMIdentifier, CEMIdentifierParams
from jax import random

"""
Run an interactive simulation of the pushbox task with system identification.

This example demonstrates using system identification to learn the box's mass and friction
during control. Double click on the green target, then drag it around with 
[ctrl + right-click] to see the system adapt.
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run pushbox with system identification."
)
parser.add_argument(
    "--algorithm",
    default="cem",
    choices=["ps", "mppi", "cem"],
    help="Control algorithm",
)
parser.add_argument(
    "--frequency", type=float, default=50.0, help="Control frequency in Hz"
)
parser.add_argument(
    "--buffer_size",
    type=int,
    default=50,
    help="Number of transitions for system ID",
)
args = parser.parse_args()

# Define the task (cost and dynamics)
task = PushBox()

# Set up the controller based on algorithm
if args.algorithm == "ps":
    print("Using Predictive Sampling controller")
    controller = PredictiveSampling(
        task,
        num_samples=16,
        noise_level=0.1,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )
elif args.algorithm == "mppi":
    print("Using MPPI controller")
    controller = MPPI(
        task,
        num_samples=2000,
        noise_level=0.1,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )
else:  # Default to CEM
    print("Using CEM controller")
    controller = CEM(
        task,
        num_samples=64,
        num_elites=8,
        sigma_start=0.2,
        sigma_min=0.01,
        explore_fraction=0.5,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

# Create the system identifier using the new generalized structure
sigma_start = 0.2
identifier = CEMIdentifier(
    model_template=mjx.put_model(task.mj_model),
    apply_params_fn=task.apply_params_fn,
    param_dim=task.get_param_dim(),
    buffer_size=128,
    num_samples=128,
    num_elites=16,
    sigma_start=sigma_start,
    sigma_min=0.01,
    explore_fraction=0.5,
    seed=0,
)

# Initialize the identifier with a known initial parameter vector
# For PushBox: [box_mass, box_sliding_friction]
initial_params = jnp.array([0.15, 0.6])
initial_id_params = CEMIdentifierParams(
    mean=initial_params,
    cov=sigma_start * jnp.ones((2,)),
    rng=random.PRNGKey(0),
    iteration=0,
)
identifier.set_params(initial_id_params)

# Define the model used for simulation and perturb it
mj_model = task.mj_model
# Intentionally perturb the box mass and friction to test identification
mj_model.body_mass[task.box_body_id] = 0.3  # Increase mass
mj_model.geom_friction[task.box_geom_id, 0] *= 2  # Increase sliding friction

# Initialize data with default state
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    controller,
    mj_model,
    mj_data,
    frequency=args.frequency,
    show_traces=False,
    max_traces=5,
    identifier=identifier,
)
