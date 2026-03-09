import numpy as np

class IntActionWrapper(gym.ActionWrapper):
    def action(self, action):
        print(f"Original action: {action}")
        if isinstance(action, np.ndarray):
            if action.size == 1:
                return int(action.item())
        if isinstance(action, np.integer):
            return int(action)
        
        print(f"Converted action: {action}")
        return action