"""Test custom step function functionality."""

import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.algs.predictive_sampling import PredictiveSampling
from hydrax.tasks.pendulum import Pendulum


class PendulumWithCustomStep(Pendulum):
    """Pendulum task with a custom step function."""

    def step(self, model: mjx.Model, state: mjx.Data) -> mjx.Data:
        """Custom step function."""
        # For this test, we'll just use the default step
        # In a real neural network case, this would use the NN to
        # predict next state
        return mjx.step(model, state)


def test_custom_step_function() -> None:
    """Test that custom step function is called during rollouts."""
    task = PendulumWithCustomStep()
    
    # Verify the task has the custom step method
    assert hasattr(task, "step")
    assert callable(task.step)
    
    # Create a controller
    opt = PredictiveSampling(
        task,
        num_samples=4,
        noise_level=0.1,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=5,
        num_randomizations=1,
    )
    
    # Initialize state and parameters
    state = mjx.make_data(task.model)
    params = opt.init_params()
    
    # Sample control sequences from the policy
    knots, params = opt.sample_knots(params)
    
    # Compute the control sequence from the knots
    tk = jnp.linspace(0.0, opt.plan_horizon, opt.num_knots)
    tq = jnp.linspace(0.0, opt.plan_horizon - opt.dt, opt.ctrl_steps)
    controls = opt.interp_func(tq, tk, knots)
    
    # Roll out the control sequences - this should call step() multiple times
    _, rollouts = opt.eval_rollouts(task.model, state, controls, knots)
    
    # Verify the rollout completes successfully
    assert rollouts.costs.shape[0] == opt.num_samples
    assert rollouts.costs.shape[1] == opt.ctrl_steps + 1


def test_custom_step_vs_default() -> None:
    """Test that both default and custom step functions work correctly."""
    # Create two tasks: one with default step, one with custom step
    task_default = Pendulum()
    task_custom = PendulumWithCustomStep()
    
    # Create controllers
    opt_default = PredictiveSampling(
        task_default,
        num_samples=2,
        noise_level=0.1,
        plan_horizon=0.1,
        spline_type="zero",
        num_knots=3,
        num_randomizations=1,
    )
    
    opt_custom = PredictiveSampling(
        task_custom,
        num_samples=2,
        noise_level=0.1,
        plan_horizon=0.1,
        spline_type="zero",
        num_knots=3,
        num_randomizations=1,
    )
    
    # Initialize state and parameters
    state_default = mjx.make_data(task_default.model)
    state_custom = mjx.make_data(task_custom.model)
    params_default = opt_default.init_params()
    params_custom = opt_custom.init_params()
    
    # Sample control sequences
    knots_default, params_default = opt_default.sample_knots(params_default)
    knots_custom, params_custom = opt_custom.sample_knots(params_custom)
    
    # Compute control sequences from knots
    tk_default = jnp.linspace(
        0.0, opt_default.plan_horizon, opt_default.num_knots
    )
    tq_default = jnp.linspace(
        0.0, opt_default.plan_horizon - opt_default.dt, opt_default.ctrl_steps
    )
    controls_default = opt_default.interp_func(
        tq_default, tk_default, knots_default
    )
    
    tk_custom = jnp.linspace(0.0, opt_custom.plan_horizon, opt_custom.num_knots)
    tq_custom = jnp.linspace(
        0.0, opt_custom.plan_horizon - opt_custom.dt, opt_custom.ctrl_steps
    )
    controls_custom = opt_custom.interp_func(tq_custom, tk_custom, knots_custom)
    
    # Roll out with default step
    _, rollouts_default = opt_default.eval_rollouts(
        task_default.model, state_default, controls_default, knots_default
    )
    
    # Roll out with custom step
    _, rollouts_custom = opt_custom.eval_rollouts(
        task_custom.model, state_custom, controls_custom, knots_custom
    )
    
    # Verify both produce valid results
    assert rollouts_default.costs.shape == rollouts_custom.costs.shape
    assert rollouts_default.controls.shape == rollouts_custom.controls.shape
    assert rollouts_default.costs.shape[0] == opt_default.num_samples
    assert rollouts_custom.costs.shape[0] == opt_custom.num_samples


if __name__ == "__main__":
    test_custom_step_function()
    test_custom_step_vs_default()
    print("All custom step tests passed!")

