import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional


class ConvergenceCallback(BaseCallback):
    def __init__(
        self,
        ideal_steps: Optional[int] = None,
        window_size: int = 100,
        margin: int = 2,
        reward_history_interval_steps: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.ideal_steps = ideal_steps
        self.window_size = window_size
        self.margin = margin
        self.reward_history_interval_steps = reward_history_interval_steps

        self.convergence_timestep = -1
        self.convergence_episode = -1

        self.episode_lengths: list[int] = []
        self.episode_returns: list[float] = []
        self.episode_end_steps: list[int] = []
        self.rolling_average_returns: list[float] = []
        self.reward_history: list[dict[str, float | int]] = []
        self.total_training_steps = 0
        self.current_ep_length = 0
        self.current_ep_return = 0.0
        self.total_reward_sum = 0.0
        self.total_reward_count = 0
        self.current_reward_window_sum = 0.0
        self.current_reward_window_count = 0

    @property
    def average_return_per_training_episodes(self) -> float:
        if not self.episode_returns:
            return 0.0
        return float(np.mean(self.episode_returns))

    @property
    def average_return_over_training_steps(self) -> float:
        if self.total_reward_count <= 0:
            return 0.0
        return float(self.total_reward_sum / self.total_reward_count)

    @property
    def average_reward_over_training_steps(self) -> float:
        return self.average_return_over_training_steps

    @property
    def convergence_speed_over_training_steps(self) -> float:
        if self.convergence_timestep <= 0 or self.total_training_steps <= 0:
            return -1.0
        return float(self.convergence_timestep / self.total_training_steps)

    @property
    def num_recorded_episodes(self) -> int:
        return len(self.episode_returns)

    def _record_reward_history(self, force: bool = False):
        if self.current_reward_window_count <= 0:
            return

        if not force and self.current_reward_window_count < self.reward_history_interval_steps:
            return

        mean_reward = self.current_reward_window_sum / self.current_reward_window_count
        self.reward_history.append(
            {
                "training_step": int(self.total_training_steps),
                "mean_reward": float(mean_reward),
                "reward_sum": float(self.current_reward_window_sum),
                "reward_count": int(self.current_reward_window_count),
            }
        )

        self.current_reward_window_sum = 0.0
        self.current_reward_window_count = 0

    def _update_convergence(self):
        if self.convergence_timestep != -1:
            return
        if len(self.reward_history) < 2:
            return

        peak_reward = max(entry["mean_reward"] for entry in self.reward_history)
        target_reward = 0.9 * peak_reward

        for idx, entry in enumerate(self.reward_history):
            if entry["mean_reward"] >= target_reward:
                self.convergence_episode = idx + 1
                self.convergence_timestep = int(entry["training_step"])
                return

    def _on_step(self) -> bool:
        self.total_training_steps += 1
        self.current_ep_length += 1
        rewards = self.locals.get("rewards")
        if rewards is not None:
            step_reward = float(rewards[0])
            self.current_ep_return += step_reward
            self.total_reward_sum += step_reward
            self.total_reward_count += 1
            self.current_reward_window_sum += step_reward
            self.current_reward_window_count += 1

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

            if not truncated:
                self.episode_lengths.append(self.current_ep_length)

            self.current_ep_length = 0
            self.current_ep_return = 0.0

        self._record_reward_history()
        self._update_convergence()

        return True

    def _on_training_end(self) -> None:
        self._record_reward_history(force=True)
        self._update_convergence()
