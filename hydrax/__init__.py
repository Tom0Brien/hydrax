import os
import jax
from pathlib import Path

# package root
ROOT = str(Path(__file__).parent.absolute())

# Set XLA flags for better performance and determinism
# --xla_gpu_deterministic_ops=true ensures reproducible GPU operations
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=true "
    "--xla_gpu_deterministic_ops=true"
)

# Enable persistent compilation cache
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

# Use highest precision for matmul operations to ensure reproducibility
jax.config.update("jax_default_matmul_precision", "highest")
