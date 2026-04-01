import numpy as np
import gymnasium as gym
from nasim.envs.utils import AccessLevel

# Custom action mask function for ActionMasker wrapper (returns a boolean mask of valid actions based on the current state of the environment)
def custom_mask_fn(env: gym.Env) -> np.ndarray:
    base_env = env.unwrapped
    action_space = base_env.action_space

    mask = np.zeros(action_space.n, dtype=bool)
    state = base_env.current_state

    for i in range(action_space.n):
        action = action_space.get_action(i)
        target = action.target

        if target not in state.host_num_map:
            mask[i] = False
            continue
            
        host_vec = state.get_host(target)
        action_name = action.name.lower() if hasattr(action, "name") else ""

        # Subnet scans require a compromised host (USER+ access).
        if "subnet_scan" in action_name:
            mask[i] = host_vec.access >= AccessLevel.USER
            continue

        if "pe_" in action_name:
            mask[i] = host_vec.access >= AccessLevel.USER
            continue

        mask[i] = host_vec.discovered

    if not np.any(mask):
        mask.fill(True)

    return mask