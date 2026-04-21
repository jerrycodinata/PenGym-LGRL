import gymnasium as gym
import numpy as np

from lgrl_final.subgoal_manager import SUBGOAL_TO_IDX


NUM_SUBGOALS = len(SUBGOAL_TO_IDX)


def one_hot_subgoal(subgoal):
    vec = np.zeros(NUM_SUBGOALS, dtype=np.float32)
    idx = SUBGOAL_TO_IDX.get(subgoal)
    if idx is not None:
        vec[idx] = 1.0
    return vec


class IntActionWrapper(gym.ActionWrapper):
    def action(self, action):
        if isinstance(action, np.ndarray) and action.size == 1:
            return int(action.item())
        if isinstance(action, np.integer):
            return int(action)
        return action


class SubgoalObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, subgoal_manager):
        super().__init__(env)
        self.subgoal_manager = subgoal_manager

        obs_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=100,
            shape=(obs_dim + NUM_SUBGOALS,),
            dtype=np.float32,
        )

    def observation(self, obs):
        g = self.subgoal_manager.get()
        g_vec = one_hot_subgoal(g)
        return np.concatenate([obs.astype(np.float32), g_vec])


class SubgoalRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, subgoal_manager, lambda_=0.5):
        super().__init__(env)
        self.subgoal_manager = subgoal_manager
        self.lambda_ = lambda_

    def reward(self, reward):
        if self.subgoal_manager.just_completed:
            return reward + self.lambda_
        return reward


class SubgoalUpdateWrapper(gym.Wrapper):
    def __init__(self, env, subgoal_manager):
        super().__init__(env)
        self.subgoal_manager = subgoal_manager

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.subgoal_manager.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.subgoal_manager.update()
        return obs, reward, done, truncated, info
