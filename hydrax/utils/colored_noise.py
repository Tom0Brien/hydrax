"""Colored noise generation for temporally-correlated sampling.

This module provides JAX-compatible colored noise generation using FFT-based
methods to create temporally-correlated sequences for trajectory optimization.
"""

import jax
import jax.numpy as jnp


def powerlaw_psd_gaussian(
    beta: float, shape: tuple[int, ...], key: jax.Array
) -> jax.Array:
    """Generate colored noise with a powerlaw power spectral density.

    This function generates Gaussian noise with temporal correlations defined by
    a power law in the frequency domain: S(f) ∝ 1/f^β

    Args:
        beta: The exponent of the power spectral density. Common values:
              - β=0: White noise (no correlation)
              - β=1: Pink noise (1/f noise)
              - β=2: Brown/Brownian noise
              Higher β values create smoother, more correlated sequences.
        shape: The shape of the output array. The temporal correlation is
               applied along the LAST axis. For example, shape=(num_samples,
               time_steps, action_dim) will create time_steps correlated
               samples for each of num_samples trajectories and action_dim
               dimensions.
        key: JAX random key for generating the base white noise.

    Returns:
        Array of colored noise with the specified shape. Values are approximately
        zero-mean with unit variance (after normalization).

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> # Generate 10 trajectories of 100 timesteps with 3 action dimensions
        >>> noise = powerlaw_psd_gaussian(2.0, (10, 3, 100), key)
        >>> noise.shape
        (10, 3, 100)
    """
    if len(shape) < 1:
        raise ValueError("shape must have at least one dimension")

    # The temporal axis is the last dimension
    temporal_size = shape[-1]

    # Generate white noise in the time domain
    white_noise = jax.random.normal(key, shape)

    # If beta is 0, return white noise (no correlation)
    if beta == 0:
        return white_noise

    # Apply FFT along the temporal (last) axis
    fft_noise = jnp.fft.rfft(white_noise, axis=-1)

    # Create frequency array (positive frequencies only for rfft)
    # Shape: (temporal_size // 2 + 1,)
    freqs = jnp.arange(temporal_size // 2 + 1)

    # Avoid division by zero at DC component (f=0)
    # We set f=0 to 1, which gives it the same weight as f=1
    freqs = jnp.where(freqs == 0, 1.0, freqs.astype(jnp.float32))

    # Apply power law scaling: divide by f^(β/2)
    # We use β/2 because power spectral density is amplitude^2
    scaling = freqs ** (-beta / 2.0)

    # Broadcast scaling to match fft_noise shape
    # scaling needs to be broadcast to shape[:-1] + (temporal_size // 2 + 1,)
    for _ in range(len(shape) - 1):
        scaling = jnp.expand_dims(scaling, 0)

    # Apply scaling in frequency domain
    scaled_fft = fft_noise * scaling

    # Transform back to time domain
    colored_noise = jnp.fft.irfft(scaled_fft, n=temporal_size, axis=-1)

    # Normalize to approximately unit variance
    # The normalization factor depends on beta, but we'll use empirical std
    std = jnp.std(colored_noise)
    # Avoid division by zero
    std = jnp.where(std > 1e-10, std, 1.0)
    colored_noise = colored_noise / std

    return colored_noise
