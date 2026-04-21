import random
from typing import Callable, Iterable, Optional

import gymnasium as gym
import numpy as np
import pengym
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from lgrl_final.action_mask import CustomActionMask
from lgrl_final.wrappers import (
    IntActionWrapper,
    SubgoalObsWrapper,
    SubgoalRewardWrapper,
    SubgoalUpdateWrapper,
)


MaskFn = Callable[[gym.Env], np.ndarray]


class EnvFactory:
    AGENT_TYPE_LGRL = "lgrl"

    def __init__(
        self,
        agent_type: str,
        subgoal_manager,
        max_steps: int,
        action_mask_fn: MaskFn = CustomActionMask.mask_fn,
        enable_action_masking: bool = True,
        intrinsic_reward: bool = False,
        intrinsic_reward_lambda: float = 10.0,
    ):
        self.agent_type = agent_type
        self.subgoal_manager = subgoal_manager
        self.max_steps = max_steps
        self.action_mask_fn = action_mask_fn
        self.enable_action_masking = enable_action_masking
        self.intrinsic_reward = intrinsic_reward
        self.intrinsic_reward_lambda = intrinsic_reward_lambda

    @staticmethod
    def create_pengym_env(scenario_name, seed=None):
        return pengym.create_environment(scenario_name, seed=seed)

    @staticmethod
    def create_pengym_custom_environment(scenario_path):
        env = pengym.load(scenario_path)
        seed = 1
        np.random.seed(seed)
        env.action_space.seed(seed)
        return env

    def _build_env_kwargs(self) -> dict:
        kwargs = {
            "llm_guidance": self.agent_type == self.AGENT_TYPE_LGRL,
            "subgoal_manager": self.subgoal_manager,
            "intrinsic_reward": self.intrinsic_reward,
            "intrinsic_reward_lambda": self.intrinsic_reward_lambda,
        }
        return kwargs

    def _apply_wrappers(self, env: gym.Env, llm_guidance: bool, subgoal_manager, intrinsic_reward: bool, intrinsic_reward_lambda: float):
        env = IntActionWrapper(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=self.max_steps)

        if llm_guidance:
            env = SubgoalUpdateWrapper(env, subgoal_manager)
            env = SubgoalObsWrapper(env, subgoal_manager)

            if intrinsic_reward:
                env = SubgoalRewardWrapper(env, subgoal_manager, lambda_=intrinsic_reward_lambda)

        # Keep ActionMasker outermost so VecEnv lookups can discover action_masks().
        env = Monitor(env)
        if self.enable_action_masking:
            env = ActionMasker(env, self.action_mask_fn)
        return env

    def normalize_firewall_collections(self, env: gym.Env) -> gym.Env:
        """Normalize firewall values to list for generated NASim scenarios."""
        try:
            base_env = env.unwrapped
            network = getattr(base_env, "network", None)
            firewall = getattr(network, "firewall", None)

            if isinstance(firewall, dict):
                for link, services in firewall.items():
                    if not isinstance(services, list):
                        if isinstance(services, (set, tuple)):
                            firewall[link] = list(services)
                        else:
                            try:
                                firewall[link] = list(services)
                            except TypeError:
                                firewall[link] = [services]
        except Exception:
            # Best effort only.
            pass

        return env

    def build_env(
        self,
        scenario_name: Optional[str] = None,
        scenario_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> gym.Env:
        if scenario_name is not None and scenario_path is not None:
            raise ValueError(
                "Use scenario_name (dynamic env) OR scenario_path (static env), not both."
            )

        kwargs = self._build_env_kwargs()
        if scenario_path is not None:
            env = self.create_pengym_custom_environment(scenario_path)
        else:
            if scenario_name is None:
                raise ValueError("scenario_name is required when scenario_path is not provided")
            env = self.create_pengym_env(scenario_name, seed=seed)

        env = self._apply_wrappers(env, **kwargs)
        return self.normalize_firewall_collections(env)

    def make_eval_env(self, scenario_name: str, seed: int):
        return lambda: self.build_env(scenario_name=scenario_name, seed=seed)

    def make_static_env(self, scenario_path: str):
        return lambda: self.build_env(scenario_path=scenario_path)

    def make_env_reseedable(self, scenario_name: str, seed_pool: Iterable[int]):
        seeds = list(seed_pool)
        if not seeds:
            raise ValueError("seed_pool must not be empty")

        factory = self

        class _ReseedableEnv(gym.Env):
            def __init__(self):
                self.scenario_name = scenario_name
                self.seed_pool = seeds
                self.env = factory.build_env(
                    scenario_name=self.scenario_name,
                    seed=random.choice(self.seed_pool),
                )

                self.action_space = self.env.action_space
                self.observation_space = self.env.observation_space

            def reset(self, **kwargs):
                self.env = factory.build_env(
                    scenario_name=self.scenario_name,
                    seed=random.choice(self.seed_pool),
                )
                return self.env.reset(**kwargs)

            def step(self, action):
                return self.env.step(action)

            def render(self, *args, **kwargs):
                return self.env.render(*args, **kwargs)

            def action_masks(self):
                if factory.enable_action_masking:
                    return self.env.get_wrapper_attr("action_masks")()
                return np.ones(self.action_space.n, dtype=bool)

            def __getattr__(self, name):
                return self.env.get_wrapper_attr(name)

        return _ReseedableEnv

    def build_train_env_factory(
        self,
        scenario_name: Optional[str],
        scenario_path: Optional[str],
        train_seeds: Optional[Iterable[int]],
    ):
        if scenario_name is not None and scenario_path is not None:
            raise ValueError(
                "Use scenario_name (dynamic env) OR scenario_path (static env), not both."
            )

        if scenario_path is not None:
            return self.make_static_env(scenario_path)

        if scenario_name is None:
            raise ValueError("scenario_name is required when scenario_path is not provided")

        seeds = list(train_seeds) if train_seeds is not None else list(range(100))
        return self.make_env_reseedable(scenario_name=scenario_name, seed_pool=seeds)
