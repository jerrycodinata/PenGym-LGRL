from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

import gymnasium as gym
import numpy as np
import pengym.utilities as utils
from pengym.storyboard import Storyboard
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from lgrl_final.action_mask import CustomActionMask
from lgrl_final.callbacks import ConvergenceCallback
from lgrl_final.env_factory import EnvFactory
from lgrl_final.subgoal_manager import DeterministicSubgoalManager, LLMSubgoalManager


MaskFn = Callable[[gym.Env], np.ndarray]


class PPOTrainer:
    AGENT_TYPE_PPO = "ppo"
    AGENT_TYPE_LGRL = "lgrl"
    SUBGOAL_MANAGER_DETERMINISTIC = "deterministic"
    SUBGOAL_MANAGER_LLM = "llm"

    IDEAL_STEPS = {
        "tiny": 6,
        "tiny-small": 7,
        "tiny-hard": 5,
        "small-linear": 12,
        "small-honeypot": 8,
        "medium": 8,
        "medium-single-site": 4,
        "medium-multi-site": 7,
    }

    def __init__(
        self,
        agent_type: str = AGENT_TYPE_PPO,
        action_mask_fn: MaskFn = CustomActionMask.mask_fn,
        use_action_masking: bool = True,
        subgoal_manager=None,
        subgoal_manager_type: str = SUBGOAL_MANAGER_DETERMINISTIC,
        llm_client=None,
        translator=None,
        max_steps: int = 150,
        total_episodes: int = 100,
        eval_episodes: int = 100,
        frame_memory: int = 4,
        window_size: int = 5,
        margin: int = 2,
        render_obs_state: bool = False,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        verbose: int = 1,
    ):
        if agent_type not in {self.AGENT_TYPE_PPO, self.AGENT_TYPE_LGRL}:
            raise ValueError(f"Unsupported agent type: {agent_type}")

        self.storyboard = Storyboard()
        self.agent_type = agent_type
        self.action_mask_fn = action_mask_fn
        self.use_action_masking = use_action_masking

        self.max_steps = max_steps
        self.total_episodes = total_episodes
        self.eval_episodes = eval_episodes
        self.frame_memory = frame_memory
        self.window_size = window_size
        self.margin = margin
        self.render_obs_state = render_obs_state

        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.verbose = verbose

        self.model: Optional[MaskablePPO] = None
        self.scenario_name: Optional[str] = None
        self.scenario_path: Optional[str] = None
        self.convergence_speed: int = -1
        self.subgoal_manager_type = subgoal_manager_type

        valid_manager_types = {
            self.SUBGOAL_MANAGER_DETERMINISTIC,
            self.SUBGOAL_MANAGER_LLM,
        }
        if self.subgoal_manager_type not in valid_manager_types:
            raise ValueError(
                f"Unsupported subgoal_manager_type: {self.subgoal_manager_type}. "
                f"Use one of {sorted(valid_manager_types)}"
            )

        if self.agent_type == self.AGENT_TYPE_LGRL:
            if subgoal_manager is not None:
                self.subgoal_manager = subgoal_manager
            elif self.subgoal_manager_type == self.SUBGOAL_MANAGER_LLM:
                self.subgoal_manager = LLMSubgoalManager(
                    utils=utils,
                    storyboard=self.storyboard,
                    llm_client=llm_client,
                    translator=translator,
                )
            else:
                self.subgoal_manager = DeterministicSubgoalManager(utils=utils, storyboard=self.storyboard)
        else:
            self.subgoal_manager = None

        self.env_factory = EnvFactory(
            agent_type=self.agent_type,
            subgoal_manager=self.subgoal_manager,
            max_steps=self.max_steps,
            action_mask_fn=self.action_mask_fn,
            enable_action_masking=self.use_action_masking,
        )

    def _resolve_scenario_inputs(
        self,
        scenario_name: Optional[str],
        scenario_path: Optional[str],
        allow_fallback: bool = False,
    ):
        resolved_name = scenario_name
        resolved_path = scenario_path

        if allow_fallback:
            if resolved_name is None:
                resolved_name = self.scenario_name
            if resolved_path is None:
                resolved_path = self.scenario_path

        if resolved_name is not None and resolved_path is not None:
            raise ValueError(
                "Use exactly one scenario selector: scenario_name (dynamic env) OR "
                "scenario_path (static env), not both."
            )

        if resolved_name is None and resolved_path is None:
            raise ValueError(
                "Scenario is not specified. Provide scenario_name for dynamic env "
                "or scenario_path for static env."
            )

        scenario_key = resolved_name if resolved_name is not None else Path(resolved_path).stem
        return scenario_key, resolved_name, resolved_path

    def _build_model(self, vec_env) -> MaskablePPO:
        return MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            verbose=self.verbose,
        )

    def load(self, model_path: str):
        self.model = MaskablePPO.load(model_path)
        return self.model

    def train(
        self,
        scenario_name: Optional[str] = None,
        scenario_path: Optional[str] = None,
        train_seeds: Optional[Iterable[int]] = None,
        model_path: Optional[str] = None,
        total_timesteps: Optional[int] = None,
    ):
        scenario_key, resolved_name, resolved_path = self._resolve_scenario_inputs(
            scenario_name,
            scenario_path,
            allow_fallback=False,
        )
        self.scenario_name = resolved_name
        self.scenario_path = resolved_path
        self.convergence_speed = -1

        if model_path is not None:
            return self.load(model_path)

        env_fn = self.env_factory.build_train_env_factory(
            scenario_name=resolved_name,
            scenario_path=resolved_path,
            train_seeds=train_seeds,
        )
        vec_env = DummyVecEnv([env_fn])
        vec_env = VecFrameStack(vec_env, n_stack=self.frame_memory)

        self.model = self._build_model(vec_env)

        ideal_step = self.IDEAL_STEPS.get(scenario_key)
        convergence_cb = None
        if ideal_step is not None:
            convergence_cb = ConvergenceCallback(
                ideal_steps=ideal_step,
                window_size=self.window_size,
                margin=self.margin,
            )
        else:
            print(f"* WARNING: No IDEAL_STEPS configured for scenario '{scenario_key}'. Convergence metric disabled.")

        print("=================STARTING TRAINING=================")
        target_timesteps = total_timesteps if total_timesteps is not None else self.max_steps * self.total_episodes
        if convergence_cb is not None:
            self.model.learn(
                total_timesteps=target_timesteps,
                callback=convergence_cb,
                use_masking=self.use_action_masking,
            )
            self.convergence_speed = convergence_cb.convergence_timestep
        else:
            self.model.learn(
                total_timesteps=target_timesteps,
                use_masking=self.use_action_masking,
            )
        print("=================TRAINING COMPLETE=================")

        return self.model

    def evaluate(
        self,
        scenario_name: Optional[str] = None,
        scenario_path: Optional[str] = None,
        num_episodes: Optional[int] = None,
        seeds: Optional[Iterable[int]] = None,
    ):
        if self.model is None:
            raise ValueError("Model is not loaded/trained. Call train() or load() first.")

        _, resolved_name, resolved_path = self._resolve_scenario_inputs(
            scenario_name,
            scenario_path,
            allow_fallback=True,
        )
        episodes_per_seed = num_episodes if num_episodes is not None else self.eval_episodes
        eval_seeds = list(seeds) if seeds is not None else [1000, 1001, 1002, 1003]

        print("\n---------------------------------------")
        print("Evaluation phase:")
        print("---------------------------------------")

        success_count = 0
        total_tokens_used = 0
        total_cumulative_reward = 0.0
        done = False
        truncated = False
        ep_steps = 0

        for seed in eval_seeds:
            if resolved_path is not None:
                eval_env_fn = self.env_factory.make_static_env(resolved_path)
            else:
                eval_env_fn = self.env_factory.make_eval_env(resolved_name, seed)

            eval_env = DummyVecEnv([eval_env_fn])
            eval_vec_env = VecFrameStack(eval_env, n_stack=self.frame_memory)

            for ep in range(episodes_per_seed):
                print(f"\n=== Evaluation Episode {ep + 1}/{episodes_per_seed} (seed={seed}) ===")

                obs = eval_vec_env.reset()

                terminated = False
                truncated = False
                ep_steps = 0
                ep_reward = 0.0
                ep_token_usage = 0

                while not terminated and not truncated and ep_steps < self.max_steps:
                    if self.use_action_masking:
                        action_masks = np.array([self.action_mask_fn(eval_vec_env.envs[0])])
                        action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
                    else:
                        action, _ = self.model.predict(obs, deterministic=True)

                    base_env = eval_vec_env.envs[0].env.unwrapped
                    action_idx = int(action[0]) if isinstance(action, np.ndarray) else int(action)
                    action_obj = base_env.action_space.get_action(action_idx)
                    print(f"- Step {ep_steps + 1}: {action_obj}")

                    obs, reward, done_arr, info = eval_vec_env.step(action)
                    terminated = done_arr[0]
                    truncated = info[0].get("TimeLimit.truncated", False)
                    ep_steps += 1
                    ep_reward += reward[0]

                    if self.render_obs_state:
                        base_env.render()
                        base_env.render_state()

                    print(f"  Reward: {reward[0]:.2f}, Cumulative: {ep_reward:.2f}")

                done = bool(terminated and not truncated)
                success_count += 1 if done else 0
                total_cumulative_reward += ep_reward
                total_tokens_used += ep_token_usage

                print(f"\nEvaluation {ep + 1}/{episodes_per_seed} complete:")
                print(f"  - Steps: {ep_steps}")
                print(f"  - Total reward: {ep_reward:.2f}")
                print(f"  - Goal reached: {done}")
                print(f"  - Truncated: {truncated}")

            eval_vec_env.close()

        total_eval_episodes = episodes_per_seed * len(eval_seeds)
        success_rate = (success_count / total_eval_episodes) * 100
        avg_cumulative_reward = total_cumulative_reward / total_eval_episodes
        avg_tokens_used = total_tokens_used / total_eval_episodes

        print("\n=======================================")
        print("Evaluation Summary:")
        print("=======================================")
        print(f"Agent Type                 : {self.agent_type.upper()}")
        print(f"Total Evaluation Episodes  : {total_eval_episodes}")
        print(f"Convergence Speed          : {self.convergence_speed} timesteps")
        print(f"Success Rate               : {success_rate:.2f}%")
        print(f"Average Cumulative Reward  : {avg_cumulative_reward:.2f}")
        print(f"Average Tokens Used        : {avg_tokens_used:.2f}")

        return done, truncated, ep_steps

    def save(self, output_dir: str = "models") -> str:
        if self.model is None:
            raise ValueError("Model is not loaded/trained. Call train() or load() first.")

        scenario_name_for_model, _, _ = self._resolve_scenario_inputs(
            self.scenario_name,
            self.scenario_path,
            allow_fallback=False,
        )
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        model_name = f"{self.agent_type}_{scenario_name_for_model}_({self.max_steps}_{self.total_episodes}ep)_{timestamp}"
        model_path = Path(output_dir) / model_name

        self.model.save(str(model_path))
        return str(model_path)