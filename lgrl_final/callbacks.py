import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional


class ConvergenceCallback(BaseCallback):
    def __init__(self, ideal_steps: Optional[int], window_size=100, margin=2, verbose=0):
        super().__init__(verbose)
        self.ideal_steps = ideal_steps
        self.window_size = window_size
        self.margin = margin

        self.convergence_timestep = -1
        self.convergence_episode = -1

        self.episode_lengths = []
        self.episode_returns = []
        self.current_ep_length = 0
        self.current_ep_return = 0.0

    @property
    def average_return(self) -> float:
        if not self.episode_returns:
            return 0.0
        return float(np.mean(self.episode_returns))

    @property
    def num_recorded_episodes(self) -> int:
        return len(self.episode_returns)

    def _on_step(self) -> bool:
        self.current_ep_length += 1
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.current_ep_return += float(rewards[0])

        done = bool(self.locals["dones"][0])
        info = self.locals["infos"][0] if "infos" in self.locals else {}
        truncated = bool(info.get("TimeLimit.truncated", False))

        if done:
            ep_info = info.get("episode") if isinstance(info, dict) else None
            if isinstance(ep_info, dict) and "r" in ep_info:
                episode_return = float(ep_info["r"])
            else:
                episode_return = self.current_ep_return
            self.episode_returns.append(episode_return)

            if not truncated:
                self.episode_lengths.append(self.current_ep_length)

                if self.ideal_steps is not None and len(self.episode_lengths) >= self.window_size:
                    mean_length = float(np.mean(self.episode_lengths[-self.window_size :]))
                    threshold = self.ideal_steps + self.margin
                    if mean_length <= threshold and self.convergence_timestep == -1:
                        self.convergence_timestep = self.num_timesteps
                        self.convergence_episode = len(self.episode_lengths)
                        print(
                            f"Convergence achieved at timestep {self.convergence_timestep} "
                            f"with average length {mean_length:.2f}"
                        )

            self.current_ep_length = 0
            self.current_ep_return = 0.0

        return True
