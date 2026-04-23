import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional


class ConvergenceCallback(BaseCallback):
    def __init__(
        self,
        ideal_steps: Optional[int] = None,
        window_size: int = 100,
        margin: int = 2,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        # The codebase used to compute convergence speed based on ideal episode
        # length. That logic is no longer used, but we keep the constructor
        # signature and attributes for compatibility.
        self.ideal_steps = ideal_steps
        self.window_size = window_size
        self.margin = margin

        self.convergence_timestep = -1
        self.convergence_episode = -1

        self.episode_lengths: list[int] = []
        self.episode_returns: list[float] = []
        self.episode_end_steps: list[int] = []
        self.rolling_average_returns: list[float] = []
        self.total_training_steps = 0
        self.current_ep_length = 0
        self.current_ep_return = 0.0

    @property
    def average_return_per_training_episodes(self) -> float:
        if not self.episode_returns:
            return 0.0
        return float(np.mean(self.episode_returns))

    @property
    def average_return_over_training_steps(self) -> float:
        if not self.rolling_average_returns:
            return 0.0
        return float(self.rolling_average_returns[-1])

    @property
    def convergence_speed_over_training_steps(self) -> float:
        if self.convergence_timestep <= 0 or self.total_training_steps <= 0:
            return -1.0
        return float(self.convergence_timestep / self.total_training_steps)

    @property
    def num_recorded_episodes(self) -> int:
        return len(self.episode_returns)

    def _update_convergence(self):
        # Convergence is the first step where the step-indexed rolling average
        # reaches 90% of its peak value.
        if self.convergence_timestep != -1:
            return
        if len(self.rolling_average_returns) < 2:
            return

        peak_return = max(self.rolling_average_returns)
        target_return = 0.9 * peak_return

        for idx, avg_return in enumerate(self.rolling_average_returns):
            if avg_return >= target_return:
                self.convergence_episode = idx + 1
                self.convergence_timestep = self.episode_end_steps[idx]
                return

    def _on_step(self) -> bool:
        self.total_training_steps += 1
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
            self.episode_end_steps.append(self.total_training_steps)

            window = self.episode_returns[-self.window_size :]
            self.rolling_average_returns.append(float(np.mean(window)))
            self._update_convergence()

            if not truncated:
                self.episode_lengths.append(self.current_ep_length)

            self.current_ep_length = 0
            self.current_ep_return = 0.0

        return True
