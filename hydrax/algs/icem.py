"""Improved Cross-Entropy Method (iCEM) for trajectory optimization.

This module implements iCEM, an enhanced version of CEM with several key
improvements for sample efficiency and performance:
- Temporally-correlated colored noise for smoother action sequences
- Elite memory across iterations to reduce variance
- Population decay to reduce computational cost
- Shifted elites for warm-starting next timestep
- Best action execution rather than mean action
"""

from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
from hydrax.utils.colored_noise import powerlaw_psd_gaussian


@dataclass
class iCEMParams(SamplingParams):
    """Policy parameters for the improved cross-entropy method.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
        cov: The (diagonal) covariance of the control distribution.
        elite_knots: Elite action sequences from previous timestep for warm-starting.
                     Shape: (num_elites_to_keep, num_knots, nu)
        best_knots: Best action sequence from previous timestep.
                    Shape: (num_knots, nu)
    """

    cov: jax.Array
    elite_knots: jax.Array  # Shape: (num_kept_elites, num_knots, nu)
    best_knots: jax.Array  # Shape: (num_knots, nu)


class iCEM(SamplingBasedController):
    """Improved Cross-Entropy Method with colored noise and elite memory.

    iCEM enhances the standard CEM algorithm with several improvements:
    1. Colored noise (β) for temporally-correlated action sequences
    2. Elite memory: keep best samples across iterations
    3. Shifted elites: warm-start next timestep with shifted trajectories
    4. Best action execution: execute best action, not mean
    5. Momentum smoothing: smooth distribution updates with α
    6. Mean injection: add mean as sample in last iteration

    Reference: https://arxiv.org/abs/2008.06389
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        num_elites: int,
        sigma_start: float,
        sigma_min: float,
        alpha: float = 0.1,
        noise_beta: float = 2.0,
        fraction_elites_reused: float = 0.3,
        population_decay_factor: float = 2.0,
        shift_elites: bool = True,
        keep_elites: bool = False,
        use_best_action: bool = True,
        add_mean_sample: bool = True,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 3,
    ) -> None:
        """Initialize the iCEM controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample (initial population).
            num_elites: The number of elite samples to keep at each iteration.
            sigma_start: The initial standard deviation for the controls.
            sigma_min: The minimum standard deviation for the controls.
            alpha: Momentum for distribution smoothing (0=no momentum, 1=no update).
            noise_beta: Colored noise exponent. Higher values create smoother trajectories.
                       Typical values: 0.25 (high-freq), 2.0 (low-freq), 3.5 (very smooth).
            fraction_elites_reused: Fraction of elites to reuse across iterations (0.0-1.0).
            population_decay_factor: Factor to decay population each iteration (e.g., 2.0).
            shift_elites: Whether to shift elite trajectories forward for warm-starting.
            keep_elites: Whether to keep elites across iterations within a timestep.
            use_best_action: Execute best action (True) or mean action (False).
            add_mean_sample: Add mean to samples at the last iteration.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combine costs from different randomizations.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
        """
        if not 0 <= fraction_elites_reused <= 1:
            raise ValueError(
                f"fraction_elites_reused must be in [0, 1], got {fraction_elites_reused}"
            )
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        self.num_samples = num_samples
        self.sigma_min = sigma_min
        self.sigma_start = sigma_start
        self.num_elites = num_elites
        self.alpha = alpha
        self.noise_beta = noise_beta
        self.fraction_elites_reused = fraction_elites_reused
        self.population_decay_factor = population_decay_factor
        self.shift_elites = shift_elites
        self.keep_elites = keep_elites
        self.use_best_action = use_best_action
        self.add_mean_sample = add_mean_sample

        # Calculate how many elites to keep for warm-starting
        self.num_elites_to_keep = int(num_elites * fraction_elites_reused)

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> iCEMParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        cov = jnp.full_like(_params.mean, self.sigma_start)

        # Initialize empty elite buffer
        elite_knots = jnp.zeros((self.num_elites_to_keep, self.num_knots, self.task.nu))
        best_knots = _params.mean.copy()

        return iCEMParams(
            tk=_params.tk,
            mean=_params.mean,
            cov=cov,
            rng=_params.rng,
            elite_knots=elite_knots,
            best_knots=best_knots,
        )



    def _shift_elite_knots(
        self, elite_knots: jax.Array, params: iCEMParams
    ) -> Tuple[jax.Array, jax.Array]:
        """Shift elite knots forward in time and append a random action.

        This creates warm-start samples for the next timestep by taking the
        elite trajectories, removing the first action, and adding a random
        action at the end.

        Args:
            elite_knots: Elite knot sequences, shape (num_elites, num_knots, nu).
            params: Current parameter state (for RNG).

        Returns:
            Tuple of:
            - Shifted elite knots, shape (num_elites, num_knots, nu)
            - Updated RNG key
        """
        num_to_shift = elite_knots.shape[0]

        if num_to_shift == 0:
            return elite_knots, params.rng

        # Shift: take actions from index 1 onwards
        shifted = elite_knots[:, 1:, :]  # Shape: (num_elites, num_knots-1, nu)

        # Generate random last actions
        rng, sample_rng = jax.random.split(params.rng)
        last_actions = (
            params.mean[-1:, :]  # Use last mean as center
            + params.cov[-1:, :]
            * jax.random.normal(sample_rng, (num_to_shift, 1, self.task.nu))
        )

        # Concatenate shifted trajectory with new random action
        shifted_elites = jnp.concatenate([shifted, last_actions], axis=1)

        return shifted_elites, rng

    def sample_knots(
        self, params: iCEMParams, is_first_iter: bool = False
    ) -> Tuple[jax.Array, iCEMParams]:
        """Sample control sequences with colored noise and elite reuse.

        Args:
            params: Current policy parameters.
            is_first_iter: Whether this is the first iteration (for shifted elites).

        Returns:
            Tuple of:
            - Sampled control knots, shape (num_samples, num_knots, nu)
            - Updated parameters
        """
        rng = params.rng

        # Generate colored noise samples
        # Note: colored_noise expects shape where temporal axis is LAST
        # We want (pop_size, num_knots, nu) with correlation along num_knots
        # So we generate with shape (pop_size, nu, num_knots) and transpose
        rng, noise_rng = jax.random.split(rng)
        noise = powerlaw_psd_gaussian(
            self.noise_beta,
            shape=(self.num_samples, self.task.nu, self.num_knots),
            key=noise_rng,
        )
        # Transpose to (pop_size, num_knots, nu)
        noise = jnp.transpose(noise, (0, 2, 1))

        # Scale and shift by mean and covariance
        samples = params.mean + params.cov * noise

        # On first iteration, add shifted elites from previous timestep
        # Check if we have valid elites to shift (sum > 0 indicates non-zero elites)
        has_elites = jnp.sum(jnp.abs(params.elite_knots)) > 0

        def add_shifted_elites(operands):
            _samples, _params = operands
            shifted_elites, new_rng = self._shift_elite_knots(
                _params.elite_knots, _params
            )
            combined = jnp.concatenate([_samples, shifted_elites], axis=0)
            return combined, new_rng

        def no_shifted_elites(operands):
            _samples, _params = operands
            # Pad with zeros to match shape of shifted_elites branch
            # This ensures both branches return the same shape
            num_elites = _params.elite_knots.shape[0]
            padding = jnp.zeros((num_elites, self.num_knots, self.task.nu))
            combined = jnp.concatenate([_samples, padding], axis=0)
            return combined, _params.rng

        # Use shifted elites only on first iteration if enabled
        use_shifted = is_first_iter & self.shift_elites & has_elites
        samples, rng = jax.lax.cond(
            use_shifted,
            add_shifted_elites,
            no_shifted_elites,
            (samples, params),
        )

        return samples, params.replace(rng=rng)

    def update_params(
        self, params: iCEMParams, rollouts: Trajectory, iteration: int = 0
    ) -> iCEMParams:
        """Update the distribution parameters using elite samples.

        Args:
            params: Current policy parameters.
            rollouts: Trajectory rollouts from sampled controls.
            iteration: Current iteration number.

        Returns:
            Updated policy parameters.
        """
        costs = jnp.sum(rollouts.costs, axis=1)  # Sum over time steps

        # Sort costs and get elite indices
        sorted_indices = jnp.argsort(costs)
        elite_indices = sorted_indices[: self.num_elites]

        # Extract elite knots
        elite_knots_all = rollouts.knots[elite_indices]

        # Keep a subset for warm-starting next timestep
        elites_to_keep = elite_knots_all[: self.num_elites_to_keep]

        # Store best knots for action execution
        best_knots = rollouts.knots[sorted_indices[0]]

        # Compute new mean and covariance from elites
        new_mean = jnp.mean(elite_knots_all, axis=0)
        new_cov = jnp.maximum(
            jnp.std(elite_knots_all, axis=0), self.sigma_min
        )

        # Apply momentum smoothing (exponential moving average)
        mean = (1 - self.alpha) * new_mean + self.alpha * params.mean
        cov = (1 - self.alpha) * new_cov + self.alpha * params.cov

        return params.replace(
            mean=mean,
            cov=cov,
            elite_knots=elites_to_keep,
            best_knots=best_knots,
        )

    def optimize(self, state, params: iCEMParams) -> Tuple[iCEMParams, Trajectory]:
        """Perform optimization with iteration-aware sampling and updates.

        This override is needed to handle shifted elites.
        """

        def _optimize_scan_body(carry, iteration):
            _params = carry

            # Sample knots (first iteration gets shifted elites)
            is_first = iteration == 0
            knots, _params = self.sample_knots(_params, is_first_iter=is_first)

            # Interpolate to get controls
            tk = _params.tk
            tq = jnp.linspace(0.0, self.plan_horizon - self.dt, self.ctrl_steps)
            controls = self.interp_func(tq, tk, knots)

            # Roll out trajectories and compute costs
            rollouts = self.rollout_with_randomizations(
                state, tk, knots, _params.rng
            )

            # Update parameters
            _params = self.update_params(_params, rollouts, iteration=iteration)

            return _params, rollouts

        # Run optimization iterations
        params_final, all_rollouts = jax.lax.scan(
            _optimize_scan_body,
            params,
            jnp.arange(self.iterations),
        )

        # Return final params and last iteration's rollouts
        final_rollouts = jax.tree.map(lambda x: x[-1], all_rollouts)

        return params_final, final_rollouts

    def get_action(self, params: iCEMParams, t: jax.Array) -> jax.Array:
        """Get the control action at time t.

        Overrides parent to use best action instead of mean when use_best_action=True.
        """
        if self.use_best_action:
            # Use the best knots instead of mean
            knots_to_use = params.best_knots
        else:
            knots_to_use = params.mean

        # Interpolate to get action at time t
        return self.interp_func(t, params.tk, knots_to_use[None, :, :])[0]
