
import jax
import jax.numpy as jnp
import numpy as np
import json
from pathlib import Path
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground.config import manipulation_params
import functools
from etils import epath
from orbax import checkpoint as ocp

# Checkpoint path for trained RL policy
CHECKPOINT_PATH = (
    Path("/home/tom/OneDrive/Phd/Papers/spc_rl/mujoco_playground/logs/PandaRobotiqPushCube-20260108-150347/checkpoints")
)

def _load_checkpoint_compat(path):
    path = epath.Path(path)
    metadata = ocp.PyTreeCheckpointer().metadata(path).item_metadata
    restore_args = jax.tree.map(
        lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata
    )
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    target = orbax_checkpointer.restore(
        path, ocp.args.PyTreeRestore(restore_args=restore_args), item=None
    )
    stats_dict = target[0]
    policy_params = target[1]
    return {'policy_params': policy_params}

def inspect():
    ENV_NAME = "PandaRobotiqPushCube"
    ppo_params = manipulation_params.brax_ppo_config(ENV_NAME)
    
    network_fn = ppo_networks.make_ppo_networks
    if hasattr(ppo_params, "network_factory"):
        network_factory = functools.partial(network_fn, **ppo_params.network_factory)
    else:
        network_factory = network_fn
        

    # Find latest checkpoint
    checkpoint_path = epath.Path(CHECKPOINT_PATH).resolve()
    print(f"Searching in: {checkpoint_path}")
    if not checkpoint_path.exists():
        print(f"Path does not exist: {checkpoint_path}")
        return

    latest_ckpts = list(checkpoint_path.glob("*"))
    print(f"Found {len(latest_ckpts)} items")
    for c in latest_ckpts:
        print(f" - {c.name} (isdigit: {c.name.isdigit()})")

    numeric_ckpts = [c for c in latest_ckpts if c.name.isdigit()]
    numeric_ckpts.sort(key=lambda x: int(x.name))
    
    if not numeric_ckpts:
        print("No numeric checkpoints found!")
        return

    restore_checkpoint_path = numeric_ckpts[-1]
    
    print(f"Loading from {restore_checkpoint_path}")
    
    config_path = restore_checkpoint_path / "ppo_network_config.json"
    if not config_path.exists():
        # Try looking in parent dir
        config_path = checkpoint_path / "config.json"
        
    if not config_path.exists():
         print("Config not found")
         # Create dummy config
         net_config = {
             "observation_size": {"shape": (26,)}, # Guessing based on code
             "action_size": 7,
             "normalize_observations": True
         }
    else:
        with open(config_path) as f:
            net_config = json.load(f)
    
    observation_size = net_config.get("observation_size", {}).get("shape", (26,))
    action_size = net_config.get("action_size", 7)
    
    print(f"Obs size: {observation_size}, Action size: {action_size}")

    ppo_network = network_factory(
        observation_size,
        action_size,
        preprocess_observations_fn=lambda x, y: x,
    )
    
    ckpt = _load_checkpoint_compat(restore_checkpoint_path)
    policy_params = ckpt['policy_params']
    
    # Create dummy observation
    obs = jnp.zeros(observation_size)
    
    # Run policy
    output = ppo_network.policy_network.apply({}, policy_params, obs)
    
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Check parametric action distribution
    print(f"Parametric Action Distribution: {ppo_network.parametric_action_distribution}")
    
    # Try to get distribution
    dist = ppo_network.parametric_action_distribution.create_dist(output)
    print(f"Distribution: {dist}")
    print(f"Dir(dist): {dir(dist)}")
    try:
        if hasattr(dist, 'loc'):
            print(f"Loc: {dist.loc}")
        if hasattr(dist, 'scale'):
            print(f"Scale: {dist.scale}")
        if hasattr(dist, 'mode'):
            print(f"Mode: {dist.mode()}")
    except Exception as e:
        print(f"Error inspecting dist: {e}")

if __name__ == "__main__":
    inspect()
