import argparse
import mujoco
from mujoco import mjx
import jax.numpy as jnp
from hydrax.algs import MPPI, CEM, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.particle import Particle
from hydrax.cem_identifier import CEMIdentifier, IDParams
from jax import random

"""
Run an interactive simulation of the particle tracking task with system identification.

This example demonstrates using system identification to learn actuator gains
during control. Double click on the green target, then drag it around with 
[ctrl + right-click] to see the system adapt.
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run particle tracking with system identification."
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
task = Particle()

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
        num_samples=512,
        num_elites=20,
        sigma_start=0.2,
        sigma_min=0.1,
        explore_fraction=0.5,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

# Create the system identifier using the base class method
sigma_start = 0.1
identifier = CEMIdentifier(
    model_template=mjx.put_model(task.mj_model),
    apply_params_fn=task.apply_params_fn,
    param_dim=task.get_param_dim(),
    buffer_size=32,
    num_samples=128,
    num_elites=10,
    sigma_start=sigma_start,
    sigma_min=0.05,
    seed=0,
)

# Initialize the identifier with a known initial parameter vector
initial_gain = jnp.array([1.0, 1.0])
initial_id_params = IDParams(
    mean=initial_gain,
    cov=sigma_start * jnp.ones((2,)),
    rng=random.PRNGKey(0),
)
identifier.init_params(initial_id_params)

# Define the model used for simulation and perturb it
mj_model = task.mj_model
# Intentionally perturb the actuator gains to test identification
for i in range(mj_model.nu):
    mj_model.actuator_gainprm[i, 0] *= 1.5  # Increase gains

print("identifier.params():", identifier.params())

# Initialize data with default state
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    controller,
    mj_model,
    mj_data,
    frequency=args.frequency,
    show_traces=True,
    max_traces=5,
    identifier=identifier,
)
