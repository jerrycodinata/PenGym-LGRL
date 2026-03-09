from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class ConvergenceCallback(BaseCallback):
    def __init__(self, ideal_steps, window_size=100, margin=2, verbose=0):
        super().__init__(verbose)
        self.ideal_steps = ideal_steps
        self.window_size = window_size
        self.margin = margin

        self.convergence_timestep = -1
        self.convergence_episode = -1

        self.episode_lengths = []
        self.current_ep_length = 0

    def _on_step(self) -> bool:
        self.current_ep_length += 1

        done = bool(self.locals["dones"][0])
        info = self.locals["infos"][0] if "infos" in self.locals else {}
        truncated = bool(info.get("TimeLimit.truncated", False))

        if done:
            if not truncated:
                self.episode_lengths.append(self.current_ep_length)

                if len(self.episode_lengths) >= self.window_size:
                    mean_length = float(np.mean(self.episode_lengths[-self.window_size:]))

                    if mean_length <= (self.ideal_steps + self.margin) and self.convergence_timestep == -1:
                        self.convergence_timestep = self.num_timesteps
                        self.convergence_episode = len(self.episode_lengths)
                        print(f"Convergence achieved at timestep {self.convergence_timestep} with average length {mean_length:.2f}")

            self.current_ep_length = 0
        
        return True