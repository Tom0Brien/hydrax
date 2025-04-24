from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class CCEMParams(SamplingParams):
    """Policy parameters for the cross-entropy method.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
        cov: The (diagonal) covariance of the control distribution.
    """

    cov: jax.Array


class CCEM(SamplingBasedController):
    """Constrained Cross-entropy method with diagonal covariance."""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        num_elites: int,
        sigma_start: float,
        sigma_min: float,
        num_randomizations: int = 1,
        explore_fraction: float = 0.0,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            num_elites: The number of elite samples to keep at each iteration.
            sigma_start: The initial standard deviation for the controls.
            sigma_min: The minimum standard deviation for the controls.
            num_randomizations: The number of domain randomizations to use.
            explore_fraction: Fraction of samples to keep at sigma_start.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
        """
        if not 0 <= explore_fraction <= 1:
            raise ValueError(
                f"explore_fraction must be between 0 and 1, got {explore_fraction}"
            )
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
        )
        self.num_samples = num_samples
        self.sigma_min = sigma_min
        self.sigma_start = sigma_start
        self.num_elites = num_elites
        self.num_explore = int(self.num_samples * explore_fraction)

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> CCEMParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        cov = jnp.full_like(_params.mean, self.sigma_start)
        return CCEMParams(
            tk=_params.tk, mean=_params.mean, cov=cov, rng=_params.rng
        )

    def sample_knots(self, params: CCEMParams) -> Tuple[jax.Array, CCEMParams]:
        """Sample a control sequence.

        A fraction of samples (determined by explore_fraction) will use the initial
        high variance for better exploration, while the rest use the current covariance.
        """
        rng, sample_rng, explore_rng = jax.random.split(params.rng, 3)

        # Pre-compute shapes for both main and exploration samples
        main_shape = (
            self.num_samples - self.num_explore,
            self.num_knots,
            self.task.model.nu,
        )
        explore_shape = (
            self.num_explore,
            self.num_knots,
            self.task.model.nu,
        )

        # Sample main knots with current covariance
        main_controls = (
            params.mean + params.cov * jax.random.normal(sample_rng, main_shape)
            if main_shape[0] > 0
            else jnp.empty(main_shape)
        )

        # Sample exploration knots with initial covariance
        explore_controls = (
            params.mean
            + self.sigma_start * jax.random.normal(explore_rng, explore_shape)
            if explore_shape[0] > 0
            else jnp.empty(explore_shape)
        )

        # Combine both sets of controls
        controls = jnp.concatenate([main_controls, explore_controls])
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: CCEMParams, rollouts: Trajectory
    ) -> CCEMParams:
        """Update the distribution parameters based on elite samples.

        Selects elite samples by prioritizing feasible solutions (constraint_cost <= 0).
        If there are enough feasible samples, only those are used. Otherwise, all feasible
        samples are selected plus the least-violating infeasible samples.
        """
        # Sum costs across time steps
        costs = jnp.sum(rollouts.costs, axis=1)
        constraint_costs = jnp.sum(
            jnp.maximum(rollouts.constraint_costs, 0), axis=1
        )

        # Identify feasible samples
        is_feasible = constraint_costs <= 0

        # Create a combined score for sorting, prioritizing constraint feasibility over cost
        # For feasible samples: use their cost
        # For infeasible samples: use constraint violation, offset by max cost + 1
        combined_score = jnp.where(
            is_feasible,
            costs,
            constraint_costs + jnp.max(costs) + 1.0,
        )

        # Sort all samples in one pass and take top elite_count
        elite_indices = jnp.argsort(combined_score)[: self.num_elites]

        # Compute new distribution parameters from elites
        elite_samples = rollouts.knots[elite_indices]
        mean = jnp.mean(elite_samples, axis=0)
        cov = jnp.maximum(jnp.std(elite_samples, axis=0), self.sigma_min)

        return params.replace(mean=mean, cov=cov)
