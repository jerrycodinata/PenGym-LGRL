import numpy as np
import gymnasium as gym

# Define subgoals and mapping to indices (SUBGOAL_TO_IDX = {"DISCOVER_HOST": 0, "ENUM_SERVICE": 1, "EXPLOIT_ACCESS": 2, "PRIV_ESC": 3})
SUBGOALS = [
    "DISCOVER_HOST",
    "ENUM_SERVICE",
    "EXPLOIT_ACCESS",
    "PRIV_ESC"
]

SUBGOAL_TO_IDX = {g: i for i, g in enumerate(SUBGOALS)}
NUM_SUBGOALS = len(SUBGOALS)

# Function to convert subgoal name to one-hot vector (vec = DISCOVER_HOST -> [1, 0, 0, 0], ENUM_SERVICE -> [0, 1, 0, 0], etc.)
def one_hot_subgoal(subgoal):
    vec = np.zeros(NUM_SUBGOALS, dtype=np.float32)
    vec[SUBGOAL_TO_IDX[subgoal]] = 1.0
    return vec

# LGRL Subgoal Wrapper that adds the current subgoal as a one-hot vector to the observation (obs = [original_obs, g_vec], where g_vec is the one-hot vector for the current subgoal)
class SubgoalObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, subgoal_manager):
        super().__init__(env)
        self.subgoal_manager = subgoal_manager

        # Dimension of the original observation space (e.g., if original obs is a vector of length 10, obs_dim = 10)
        obs_dim = env.observation_space.shape[0]

        # Create a new Box with low=0, high=100, and shape=(obs_dim + NUM_SUBGOALS,) to accommodate the original observation and the one-hot subgoal vector
        # The low and high values can be adjusted based on the expected range of the original observations; here we use 0 and 100 as placeholders
        self.observation_space = gym.spaces.Box(
            low=0,
            high=100,
            shape=(obs_dim + NUM_SUBGOALS,),
            dtype=np.float32
        )

    # Update subgoal manager based on the new state after each action and then return the augmented observation
    def observation(self, obs):
        g = self.subgoal_manager.get()
        g_vec = one_hot_subgoal(g)
        return np.concatenate([obs.astype(np.float32), g_vec])
    
# LGRL Reward Wrapper that adds an intrinsic reward (lambda_) when a subgoal is just completed (i.e., when the subgoal manager indicates that the current subgoal was just completed)
class SubgoalRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, subgoal_manager, lambda_=0.5):
        super().__init__(env)
        self.subgoal_manager = subgoal_manager
        self.lambda_ = lambda_

    def reward(self, reward):
        if self.subgoal_manager.just_completed:
            return reward + self.lambda_
        return reward

# LGRL Subgoal Update Wrapper that updates the subgoal manager after each action (i.e., after each step, it calls subgoal_manager.update(state) to check if the subgoal should be updated based on the new state)
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