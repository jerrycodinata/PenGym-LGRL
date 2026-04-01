import os
import random
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import pengym.utilities as utils
from lgrl.common_utilities import create_pengym_custom_environment
from lgrl.common_wrappers import IntActionWrapper
from lgrl.action_mask import custom_mask_fn
from lgrl.subgoal_managers import OracleSubgoalManager
from lgrl.subgoal_wrappers import SubgoalObsWrapper, SubgoalUpdateWrapper
from pengym.storyboard import Storyboard

FRAME_MEMORY = 1
MAX_EPISODE_STEPS = 30
GAMMA = 0.99
CLIP_RANGE = 0.2


def make_wrapped_env(scenario_path, subgoal_manager=None):
	env = create_pengym_custom_environment(scenario_path)
	env = IntActionWrapper(env)
	env = ActionMasker(env, custom_mask_fn)
	env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
	if subgoal_manager is not None:
		env = SubgoalUpdateWrapper(env, subgoal_manager)
		env = SubgoalObsWrapper(env, subgoal_manager)
	return Monitor(env)


def set_seeds(seed=1):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def print_tensor(label, tensor):
	print(f"{label}: {tensor}")


def run_single_step_analysis():
	scenario_path = os.path.join("database", "scenarios", "tiny.yml")
	utils.ENABLE_PENGYM = False
	utils.ENABLE_NASIM = True
	set_seeds(1)

	storyboard = Storyboard()
	subgoal_manager = OracleSubgoalManager(utils=utils, storyboard=storyboard)

	env_fn = lambda: make_wrapped_env(scenario_path, subgoal_manager=subgoal_manager)
	vec_env = DummyVecEnv([env_fn])
	vec_env = VecFrameStack(vec_env, n_stack=FRAME_MEMORY)

	model = MaskablePPO(
		"MlpPolicy",
		vec_env,
		learning_rate=3e-4,
		n_steps=128,
		batch_size=64,
		n_epochs=1,
		gamma=GAMMA,
		verbose=0
	)

	obs = vec_env.reset()
	print(f"current_subgoal: {subgoal_manager.get()}")
	action_masks = np.array([custom_mask_fn(vec_env.envs[0])])

	obs_tensor = torch.as_tensor(obs).to(model.policy.device)
	mask_tensor = torch.as_tensor(action_masks).to(model.policy.device)

	with torch.no_grad():
		distribution = model.policy.get_distribution(obs_tensor, action_masks=mask_tensor)
		actions = distribution.get_actions(deterministic=False)
		log_prob = distribution.log_prob(actions)
		values = model.policy.predict_values(obs_tensor)

		dist = distribution.distribution if hasattr(distribution, "distribution") else distribution
		logits = getattr(dist, "logits", None)
		probs = getattr(dist, "probs", None)

	print("=== Training Step (single step) ===")
	print_tensor("obs", obs_tensor)
	print_tensor("action_masks", mask_tensor)
	if logits is not None:
		print_tensor("logits", logits)
	if probs is not None:
		print_tensor("probs", probs)
	print_tensor("sampled_action", actions)
	print_tensor("log_prob", log_prob)
	print_tensor("value", values)

	next_obs, reward, done, info = vec_env.step(actions.cpu().numpy())
	print(f"updated_subgoal: {subgoal_manager.get()}")

	next_obs_tensor = torch.as_tensor(next_obs).to(model.policy.device)
	reward_tensor = torch.as_tensor(reward).to(model.policy.device)
	done_tensor = torch.as_tensor(done).to(model.policy.device)

	with torch.no_grad():
		next_values = model.policy.predict_values(next_obs_tensor)

	not_done = 1.0 - done_tensor.float()
	advantage = reward_tensor + (GAMMA * not_done * next_values) - values
	returns = advantage + values

	old_log_prob = log_prob
	ratio = torch.exp(log_prob - old_log_prob)
	clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
	policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
	value_loss = (returns - values) ** 2
	entropy = distribution.entropy()

	print_tensor("reward", reward_tensor)
	print_tensor("done", done_tensor)
	print_tensor("next_value", next_values)
	print_tensor("advantage", advantage)
	print_tensor("returns", returns)
	print_tensor("ratio", ratio)
	print_tensor("clipped_ratio", clipped_ratio)
	print_tensor("policy_loss", policy_loss)
	print_tensor("value_loss", value_loss)
	print_tensor("entropy", entropy)

	print("\n=== Evaluation Step (single step) ===")
	obs = vec_env.reset()
	print(f"current_subgoal: {subgoal_manager.get()}")
	action_masks = np.array([custom_mask_fn(vec_env.envs[0])])
	action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
	print(f"deterministic_action: {action}")
	next_obs, reward, done, info = vec_env.step(action)
	print(f"updated_subgoal: {subgoal_manager.get()}")
	print(f"reward: {reward}, done: {done}")

	vec_env.close()


if __name__ == "__main__":
	run_single_step_analysis()
