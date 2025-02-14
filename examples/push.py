import mujoco
import argparse

from hydrax.algs import PredictiveSampling, MPPI, DIALMPC  # Import DIAL-MPC along with the other algorithms
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.push import Push

"""
Run an interactive simulation of the push task with predictive sampling, MPPI, or DIAL-MPC.
"""

# Define the task (which specifies the cost and dynamics)
task = Push()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the push task with different MPC algorithms."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("dialmpc", help="Diffusion-Inspired Annealing MPC")  # New option for DIAL-MPC
args = parser.parse_args()

# Set the controller based on command-line arguments
if args.algorithm == "ps" or args.algorithm is None:
    print("Running Predictive Sampling")
    ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.5)
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(task, num_samples=256, noise_level=0.5, temperature=1)
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model
mj_model.opt.timestep = 0.001
mj_model.opt.iterations = 100
mj_model.opt.ls_iterations = 50
mj_data = mujoco.MjData(mj_model)
mj_data.qpos = [0.1, 0.1, 0.0, 0.0]  # Initial state

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=True,
)
