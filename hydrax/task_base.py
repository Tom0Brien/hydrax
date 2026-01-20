from abc import ABC, abstractmethod
from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


class Task(ABC):
    """An abstract task interface, defining the dynamics and cost functions.

    The task is a discrete-time optimal control problem of the form

        minᵤ ϕ(x_{T+1}) + ∑ₜ ℓ(xₜ, uₜ)
        s.t. xₜ₊₁ = f(xₜ, uₜ)

    where the dynamics f(xₜ, uₜ) are defined by a MuJoCo model, and the costs
    ℓ(xₜ, uₜ) and ϕ(x_{T+1}) are defined by the task instance itself.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        trace_sites: Sequence[str] | None = None,
    ) -> None:
        """Set the model and simulation parameters.

        Args:
            mj_model: The MuJoCo model to use for simulation.
            trace_sites: A list of site names to visualize with traces.

        Note: many other simulator parameters, e.g., simulator time step,
              Newton iterations, etc., are set in the model itself.
        """
        assert isinstance(mj_model, mujoco.MjModel)
        self.mj_model = mj_model
        self.model = mjx.put_model(mj_model)

        # Set the control dimension (default to model's nu)
        self.nu = mj_model.nu

        # Set actuator limits
        self.u_min = jnp.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 0],
            -jnp.inf,
        )
        self.u_max = jnp.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 1],
            jnp.inf,
        )

        # Simulation timestep
        self.dt = mj_model.opt.timestep

        # Control timestep (how often to update control)
        # Defaults to simulation timestep (control updated every step)
        # Override to slower frequency (e.g., 0.02 for 50Hz) to hold control constant
        self.ctrl_dt = mj_model.opt.timestep
        
        # Number of simulation steps per control update
        self.n_substeps = max(1, round(self.ctrl_dt / self.dt))

        # Get site IDs for points we want to trace
        trace_sites = trace_sites or []
        self.trace_site_ids = jnp.array(
            [mj_model.site(name).id for name in trace_sites]
        )

    def apply_control(self, state: mjx.Data, control: jax.Array) -> mjx.Data:
        """Apply the control action to the state.

        Args:
            state: The current state xₜ.
            control: The control action uₜ.

        Returns:
            The state with the control action applied.
        """
        return state.replace(ctrl=control)

    @abstractmethod
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        Args:
            state: The current state xₜ.
            control: The control action uₜ.

        Returns:
            The scalar running cost ℓ(xₜ, uₜ)
        """
        pass

    @abstractmethod
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T).

        Args:
            state: The final state x_T.

        Returns:
            The scalar terminal cost ϕ(x_T).
        """
        pass

    def constraint_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The constraint cost c(xₜ, uₜ) for constrained optimization.

        Override in subclasses to implement constraints. The constraint is
        satisfied when cost <= 0, and violated when cost > 0.

        Used by constrained algorithms like CCEM to prioritize feasible solutions.

        Args:
            state: The current state xₜ.
            control: The control action uₜ.

        Returns:
            The constraint cost (scalar). Positive values indicate violation.
        """
        # By default, no constraints (always feasible)
        return jnp.zeros(())

    def get_trace_sites(self, state: mjx.Data) -> jax.Array:
        """Get the positions of the trace sites at the current time step.

        Args:
            state: The current state xₜ.

        Returns:
            The positions of the trace sites at the current time step.
        """
        if len(self.trace_site_ids) == 0:
            return jnp.zeros((0, 3))

        return state.site_xpos[self.trace_site_ids]

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Generate randomized model parameters for domain randomization.

        Returns a dictionary of randomized model parameters, that can be used
        with `mjx.Model.tree_replace` to create a new randomized model.

        For example, we might set the `model.geom_friction` values by returning
        `{"geom_friction": new_frictions, ...}`.

        The default behavior is to return an empty dictionary, which means no
        randomization is applied.

        Args:
            rng: A random number generator key.

        Returns:
            A dictionary of randomized model parameters.
        """
        return {}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Generate randomized data elements for domain randomization.

        This is the place where we could randomize the initial state and other
        `data` elements. Like `domain_randomize_model`, this method should
        return a dictionary that can be used with `mjx.Data.tree_replace`.

        Args:
            data: The base data instance holding the current state.
            rng: A random number generator key.

        Returns:
            A dictionary of randomized data elements.
        """
        return {}

    def step(self, model: mjx.Model, state: mjx.Data) -> mjx.Data:
        """Custom step function to advance the state.

        By default, this uses the standard MuJoCo MJX step function, repeated
        n_substeps times with the same control. Override this method to use a
        custom dynamics model (e.g., a neural network).

        Args:
            model: The MuJoCo MJX model (may be unused if using custom dynamics).
            state: The current state xₜ.

        Returns:
            The next state xₜ₊₁ after applying the dynamics.
        """
        # Perform n_substeps simulation steps with the same control
        # This matches mujoco_playground's mjx_env.step behavior
        def single_step(data, _):
            return mjx.step(model, data), None
        
        return jax.lax.scan(single_step, state, None, self.n_substeps)[0]
