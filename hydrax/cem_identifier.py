from __future__ import annotations
from collections import deque
from typing import Callable, Deque, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

Array = jax.Array
ApplyParamsFn = Callable[[mjx.Model, Array], mjx.Model]


# ---------------------------------------------------------------------------
#  Proposal distribution parameters
# ---------------------------------------------------------------------------
@dataclass
class IDParams:
    mean: Array  # (p,)
    cov: Array  # (p,) – diagonal std‑dev σ
    rng: Array  # PRNGKey


# ---------------------------------------------------------------------------
#  Identifier
# ---------------------------------------------------------------------------
class CEMIdentifier:
    """CEM system ID."""

    # ---------------------------------------------------------
    #  Construction
    # ---------------------------------------------------------
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
    ) -> None:
        self._model_template = model_template
        self._apply_params = apply_params_fn

        # Buffer length B+1 ➔ B transitions
        self._buf_x: Deque[Array] = deque(maxlen=buffer_size + 1)
        self._buf_u: Deque[Array] = deque(maxlen=buffer_size + 1)

        if not 0 <= explore_fraction <= 1:
            raise ValueError(
                f"explore_fraction must be between 0 and 1, got {explore_fraction}"
            )

        self._num_samples = int(num_samples)
        self._num_elites = int(num_elites)
        self._sigma_min = float(sigma_min)
        self._sigma_start = float(sigma_start)
        self._num_explore = int(self._num_samples * explore_fraction)

        rng = jax.random.key(seed)
        self._params = IDParams(
            mean=jnp.zeros((param_dim,)),
            cov=jnp.full((param_dim,), sigma_start),
            rng=rng,
        )

        self._nq = model_template.nq
        self._nv = model_template.nv

    # ---------------------------------------------------------
    #  Public API
    # ---------------------------------------------------------
    def observe(self, data) -> None:
        """Push current `(state, control)` sample to the rolling buffer."""
        x = jnp.concatenate([jnp.asarray(data.qpos), jnp.asarray(data.qvel)])
        u = jnp.asarray(data.ctrl)
        self._buf_x.append(x)
        self._buf_u.append(u)

    def ready(self) -> bool:
        return len(self._buf_x) == self._buf_x.maxlen

    def params(self) -> Tuple[Array, Array]:
        return self._params

    def init_params(self, initial_params: IDParams) -> None:
        self._params = initial_params

    def get_buffer_data(self) -> Tuple[Array, Array]:
        """Extract current buffer data as JAX arrays."""
        if not self.ready():
            raise RuntimeError("Buffer under‑filled; cannot get buffer data.")
        
        x_stack = jnp.stack(tuple(self._buf_x))  # (B+1, nx)
        u_stack = jnp.stack(tuple(self._buf_u))  # (B+1, nu)
        return x_stack, u_stack
    
    def update(self, params: IDParams, x_stack: Array, u_stack: Array) -> IDParams:
        """JIT-compilable update function that takes buffer data as input."""
        x_in, x_tgt = x_stack[:-1], x_stack[1:]
        u_in = u_stack[:-1]
        qpos_in, qvel_in = jnp.split(x_in, [self._nq], axis=1)

        def loss_single(theta: Array) -> Array:
            model = self._model_template.tree_replace(
                self._apply_params(self._model_template, theta)
            )

            def one_step(qpos_i, qvel_i, u_i):
                x = mjx.make_data(model)
                x = x.replace(qpos=qpos_i, qvel=qvel_i, ctrl=u_i)
                x = mjx.step(model, x)
                return jnp.concatenate([x.qpos, x.qvel])

            x_pred = jax.vmap(one_step)(qpos_in, qvel_in, u_in)
            err = x_pred - x_tgt
            return jnp.mean(jnp.square(err))

        # Split the random keys for main samples and exploration samples
        rng, sample_rng, explore_rng = jax.random.split(params.rng, 3)

        # Calculate the number of main samples
        num_main = self._num_samples - self._num_explore

        # Sample main population
        main_eps = (
            jax.random.normal(sample_rng, (num_main, params.mean.size))
            if num_main > 0
            else jnp.empty((0, params.mean.size))
        )
        main_thetas = params.mean + params.cov * main_eps

        # Sample exploration population with initial wide covariance
        explore_eps = (
            jax.random.normal(
                explore_rng, (self._num_explore, params.mean.size)
            )
            if self._num_explore > 0
            else jnp.empty((0, params.mean.size))
        )
        explore_thetas = params.mean + self._sigma_start * explore_eps

        # Combine both sets of samples
        thetas = jnp.concatenate([main_thetas, explore_thetas])

        # Roll out thetas and compute losses
        losses = jax.vmap(loss_single)(thetas)

        elite_theta = thetas[jnp.argsort(losses)[: self._num_elites]]
        new_mean = jnp.mean(elite_theta, axis=0)
        new_cov = jnp.maximum(jnp.std(elite_theta, axis=0), self._sigma_min)
        params = params.replace(mean=new_mean, cov=new_cov, rng=rng)
        return params
