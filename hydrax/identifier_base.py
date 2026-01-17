from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Deque, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

Array = jax.Array
ApplyParamsFn = Callable[[mjx.Model, Array], mjx.Model]


@dataclass
class IdentifierParams:
    """Base parameters for sampling-based identifiers.
    
    Attributes:
        mean: The mean of the parameter distribution.
        rng: The pseudo-random number generator key.
        iteration: The current iteration number.
    """
    mean: Array
    rng: Array
    iteration: int


class SamplingBasedIdentifier(ABC):
    """An abstract sampling-based system identification interface."""
    
    def __init__(
        self,
        model_template: mjx.Model,
        apply_params_fn: ApplyParamsFn,
        *,
        param_dim: int,
        buffer_size: int,
        num_samples: int,
        seed: int = 0,
        iterations: int = 1,
    ) -> None:
        """Initialize the system identifier.
        
        Args:
            model_template: The MuJoCo model template to use for identification.
            apply_params_fn: Function to apply parameters to the model.
            param_dim: Dimension of the parameter vector to identify.
            buffer_size: Size of the rolling buffer for state-control history.
            num_samples: Number of parameter samples to use per iteration.
            seed: Random seed for parameter sampling.
            iterations: Number of optimization iterations per update.
        """
        self.model_template = model_template
        self.apply_params_fn = apply_params_fn
        self.param_dim = param_dim
        self.num_samples = num_samples
        self.iterations = iterations
        self.seed = seed
        
        # Buffer length B+1 → B transitions
        self._buf_x: Deque[Array] = deque(maxlen=buffer_size + 1)
        self._buf_u: Deque[Array] = deque(maxlen=buffer_size + 1)
        
        # Cache model dimensions
        self._nq = model_template.nq
        self._nv = model_template.nv
        
        # Initialize parameters
        rng = jax.random.key(seed)
        self._params = self.init_params(rng)
    
    def observe(self, data: mjx.Data) -> None:
        """Push current (state, control) sample to the rolling buffer.
        
        Args:
            data: MuJoCo data containing current state and control.
        """
        x = jnp.concatenate([jnp.asarray(data.qpos), jnp.asarray(data.qvel)])
        u = jnp.asarray(data.ctrl)
        self._buf_x.append(x)
        self._buf_u.append(u)
    
    def ready(self) -> bool:
        """Check if the buffer is ready for parameter updates."""
        return len(self._buf_x) == self._buf_x.maxlen
    
    def get_params(self) -> IdentifierParams:
        """Get current parameter distribution."""
        return self._params
    
    def set_params(self, params: IdentifierParams) -> None:
        """Set parameter distribution."""
        self._params = params
    
    def get_buffer_data(self) -> Tuple[Array, Array]:
        """Extract current buffer data as JAX arrays.
        
        Returns:
            x_stack: State trajectory of shape (B+1, nx)
            u_stack: Control trajectory of shape (B+1, nu)
        """
        if not self.ready():
            raise RuntimeError("Buffer under-filled; cannot get buffer data.")
        
        x_stack = jnp.stack(tuple(self._buf_x))  # (B+1, nx)
        u_stack = jnp.stack(tuple(self._buf_u))  # (B+1, nu)
        return x_stack, u_stack
    
    def update(self, params: IdentifierParams, x_stack: Array, u_stack: Array) -> IdentifierParams:
        """JIT-compilable update function."""
        def _update_scan_body(params: IdentifierParams, _: Any):
            params = params.replace(iteration=params.iteration + 1)
            
            # Sample parameter vectors
            param_samples, params = self.sample_params(params)
            
            # Evaluate loss for each parameter sample
            losses = self.evaluate_params(param_samples, x_stack, u_stack)
            
            # Update parameter distribution
            params = self.update_params(params, param_samples, losses)
            
            return params, losses
        
        params, _ = jax.lax.scan(
            f=_update_scan_body, 
            init=params, 
            xs=jnp.arange(self.iterations)
        )
        return params
    
    def evaluate_params(
        self, 
        param_samples: Array, 
        x_stack: Array, 
        u_stack: Array
    ) -> Array:
        """Evaluate loss for each parameter sample.
        
        Args:
            param_samples: Parameter samples of shape (num_samples, param_dim)
            x_stack: State trajectory of shape (B+1, nx)
            u_stack: Control trajectory of shape (B+1, nu)
            
        Returns:
            losses: Loss for each parameter sample, shape (num_samples,)
        """
        x_in, x_tgt = x_stack[:-1], x_stack[1:]
        u_in = u_stack[:-1]
        qpos_in, qvel_in = jnp.split(x_in, [self._nq], axis=1)
        
        def loss_single(theta: Array) -> Array:
            """Compute prediction loss for a single parameter vector."""
            # Apply parameters to model
            model_updates = self.apply_params_fn(self.model_template, theta)
            model = self.model_template.tree_replace(model_updates)
            
            def one_step(qpos_i, qvel_i, u_i):
                """Single dynamics step."""
                x = mjx.make_data(model)
                x = x.replace(qpos=qpos_i, qvel=qvel_i, ctrl=u_i)
                x = mjx.step(model, x)
                return jnp.concatenate([x.qpos, x.qvel])
            
            # Predict next states
            x_pred = jax.vmap(one_step)(qpos_in, qvel_in, u_in)
            
            # Compute prediction error
            err = x_pred - x_tgt
            return jnp.mean(jnp.square(err))
        
        return jax.vmap(loss_single)(param_samples)
    
    @abstractmethod
    def init_params(self, rng: Array) -> IdentifierParams:
        """Initialize parameter distribution.
        
        Args:
            rng: Random number generator key.
            
        Returns:
            Initial parameter distribution.
        """
        pass
    
    @abstractmethod 
    def sample_params(self, params: IdentifierParams) -> Tuple[Array, IdentifierParams]:
        """Sample parameter vectors from current distribution.
        
        Args:
            params: Current parameter distribution.
            
        Returns:
            param_samples: Sampled parameters of shape (num_samples, param_dim)
            updated_params: Updated parameters (e.g., with new RNG key)
        """
        pass
    
    @abstractmethod
    def update_params(
        self, 
        params: IdentifierParams, 
        param_samples: Array, 
        losses: Array
    ) -> IdentifierParams:
        """Update parameter distribution based on losses.
        
        Args:
            params: Current parameter distribution.
            param_samples: Parameter samples of shape (num_samples, param_dim)
            losses: Loss for each sample of shape (num_samples,)
            
        Returns:
            Updated parameter distribution.
        """
        pass 