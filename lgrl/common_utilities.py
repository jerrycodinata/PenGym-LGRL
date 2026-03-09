import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from run import create_pengym_custom_environment
from lgrl.common_wrappers import IntActionWrapper
from lgrl.action_mask import custom_mask_fn
from lgrl.subgoal_wrappers import SubgoalObsWrapper, SubgoalRewardWrapper, SubgoalUpdateWrapper

def make_env(scenario_path, llm_guidance=False, subgoal_manager=None, intrinsic_reward=False, intrinsic_reward_lambda=0.5):
    env = create_pengym_custom_environment(scenario_path)
    env = IntActionWrapper(env)
    env = ActionMasker(env, custom_mask_fn)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=30)

    if llm_guidance:
        env = SubgoalUpdateWrapper(env, subgoal_manager)
        env = SubgoalObsWrapper(env, subgoal_manager)

        if intrinsic_reward:
            env = SubgoalRewardWrapper(env, subgoal_manager, lambda_=intrinsic_reward_lambda)

    return Monitor(env)