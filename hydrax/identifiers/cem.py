from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from hydrax.identifier_base import (
    ApplyParamsFn, 
    IdentifierParams, 
    SamplingBasedIdentifier
)

Array = jax.Array


@dataclass
class CEMIdentifierParams(IdentifierParams):
    """CEM-specific identifier parameters.
    
    Attributes:
        mean: The mean of the parameter distribution.
        cov: The (diagonal) covariance of the parameter distribution.
        rng: The pseudo-random number generator key.
        iteration: The current iteration number.
    """
    cov: Array


class CEMIdentifier(SamplingBasedIdentifier):
    """Cross-entropy method for system identification."""
    
    def __init__(
        self,
        model_template: mjx.Model,
        apply_params_fn: ApplyParamsFn,
        *,
        param_dim: int,
        buffer_size: int,
        num_samples: int,
        num_elites: int,
        sigma_start: float,
        sigma_min: float,
        explore_fraction: float = 0.0,
        seed: int = 0,
        iterations: int = 1,
    ) -> None:
        """Initialize the CEM identifier.
        
        Args:
            model_template: The MuJoCo model template to use for identification.
            apply_params_fn: Function to apply parameters to the model.
            param_dim: Dimension of the parameter vector to identify.
            buffer_size: Size of the rolling buffer for state-control history.
            num_samples: Number of parameter samples to use per iteration.
            num_elites: Number of elite samples to keep at each iteration.
            sigma_start: Initial standard deviation for parameters.
            sigma_min: Minimum standard deviation for parameters.
            explore_fraction: Fraction of samples to keep at initial covariance.
            seed: Random seed for parameter sampling.
            iterations: Number of optimization iterations per update.
        """
        if not 0 <= explore_fraction <= 1:
            raise ValueError(
                f"explore_fraction must be between 0 and 1, got {explore_fraction}"
            )
        
        # Set CEM-specific attributes BEFORE calling super().__init__
        self.num_elites = int(num_elites)
        self.sigma_start = float(sigma_start)
        self.sigma_min = float(sigma_min)
        self.num_explore = int(num_samples * explore_fraction)
        
        super().__init__(
            model_template=model_template,
            apply_params_fn=apply_params_fn,
            param_dim=param_dim,
            buffer_size=buffer_size,
            num_samples=num_samples,
            seed=seed,
            iterations=iterations,
        )
    
    def init_params(self, rng: Array) -> CEMIdentifierParams:
        """Initialize CEM parameter distribution."""
        return CEMIdentifierParams(
            mean=jnp.zeros((self.param_dim,)),
            cov=jnp.full((self.param_dim,), self.sigma_start),
            rng=rng,
            iteration=0,
        )
    
    def sample_params(self, params: CEMIdentifierParams) -> Tuple[Array, CEMIdentifierParams]:
        """Sample parameter vectors using CEM strategy."""
        # Split random keys for main samples and exploration samples
        rng, sample_rng, explore_rng = jax.random.split(params.rng, 3)
        
        # Calculate the number of main samples
        num_main = self.num_samples - self.num_explore
        
        # Sample main population with current covariance
        main_eps = (
            jax.random.normal(sample_rng, (num_main, self.param_dim))
            if num_main > 0
            else jnp.empty((0, self.param_dim))
        )
        main_params = params.mean + params.cov * main_eps
        
        # Sample exploration population with initial wide covariance
        explore_eps = (
            jax.random.normal(explore_rng, (self.num_explore, self.param_dim))
            if self.num_explore > 0
            else jnp.empty((0, self.param_dim))
        )
        explore_params = params.mean + self.sigma_start * explore_eps
        
        # Combine both sets of samples
        param_samples = jnp.concatenate([main_params, explore_params])
        
        return param_samples, params.replace(rng=rng)
    
    def update_params(
        self, 
        params: CEMIdentifierParams, 
        param_samples: Array, 
        losses: Array
    ) -> CEMIdentifierParams:
        """Update CEM parameter distribution using elite samples."""
        # Get elite parameter samples
        elite_indices = jnp.argsort(losses)[:self.num_elites]
        elite_params = param_samples[elite_indices]
        
        # Update mean and covariance
        new_mean = jnp.mean(elite_params, axis=0)
        new_cov = jnp.maximum(
            jnp.std(elite_params, axis=0), 
            self.sigma_min
        )
        
        return params.replace(mean=new_mean, cov=new_cov)
    
    def init_with_params(self, initial_params: CEMIdentifierParams) -> None:
        """Initialize with specific parameter values."""
        self._params = initial_params 