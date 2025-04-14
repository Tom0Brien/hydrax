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
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
        """
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

    def init_params(self, seed: int = 0) -> CCEMParams:
        """Initialize the policy parameters."""
        _params = super().init_params(seed)
        cov = jnp.full_like(_params.mean, self.sigma_start)
        return CCEMParams(
            tk=_params.tk, mean=_params.mean, cov=cov, rng=_params.rng
        )

    def sample_knots(self, params: CCEMParams) -> Tuple[jax.Array, CCEMParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
        )
        controls = params.mean + params.cov * noise
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: CCEMParams, rollouts: Trajectory
    ) -> CCEMParams:
        """Update the mean with an exponentially weighted average using the constrained approach.

        This implementation selects elites based on:
        1. If there are enough feasible samples (constraint_cost <= 0), select elites from them
        2. Otherwise, select all feasible samples and the remaining from the lowest constraint violations
        """
        # Sum over time steps for both cost and constraint cost
        costs = jnp.sum(rollouts.costs, axis=1)
        constraint_costs = jnp.sum(rollouts.constraint_costs, axis=1)

        # Identify feasible samples (where constraint_cost <= 0)
        is_feasible = constraint_costs <= 0
        num_feasible = jnp.sum(is_feasible)

        # Create large sentinel values to push undesired samples to the end after sorting
        LARGE_VALUE = 1e9

        # Get masks for feasible and infeasible solutions
        feasible_mask = is_feasible
        infeasible_mask = ~is_feasible

        # Sort by cost for feasible samples (adding large value to infeasible to push them back)
        # We create two arrays of indices - one sorted by cost, one by constraint violation
        feasible_sort_values = costs + LARGE_VALUE * infeasible_mask
        feasible_sorted_indices = jnp.argsort(feasible_sort_values)

        # Sort by constraint violation for infeasible samples
        infeasible_sort_values = constraint_costs + LARGE_VALUE * feasible_mask
        infeasible_sorted_indices = jnp.argsort(infeasible_sort_values)

        # Function to select elites when we have enough feasible solutions
        def select_from_feasible():
            return feasible_sorted_indices[: self.num_elites]

        # Function to select mixed elites - uses masking instead of dynamic slicing
        def select_mixed():
            # Create a mask of the form [1,1,...,1,0,0,...] where there are num_feasible 1s
            # This mask will be used to combine feasible and infeasible solutions
            idx_range = jnp.arange(self.num_elites)
            feasible_idx_mask = idx_range < num_feasible

            # Where the mask is True, take values from feasible_sorted_indices
            # Where the mask is False, take values from infeasible_sorted_indices
            # We take the first num_feasible items from feasible_sorted_indices
            # And the first (num_elites - num_feasible) items from infeasible_sorted_indices
            mixed_elites = jnp.where(
                feasible_idx_mask,
                feasible_sorted_indices[idx_range],
                infeasible_sorted_indices[idx_range - num_feasible],
            )
            return mixed_elites

        # If we have enough feasible samples, use those; otherwise, use mixed selection
        elites = jax.lax.cond(
            num_feasible >= self.num_elites, select_from_feasible, select_mixed
        )

        # The new proposal distribution is a Gaussian fit to the elites
        mean = jnp.mean(rollouts.knots[elites], axis=0)
        cov = jnp.maximum(
            jnp.std(rollouts.knots[elites], axis=0), self.sigma_min
        )
        return params.replace(mean=mean, cov=cov)
