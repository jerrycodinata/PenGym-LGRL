import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pengym.utilities as utils

from lgrl_final.ppo_trainer import PPOTrainer


@dataclass
class ExperimentConfig:
    agent_type: str = PPOTrainer.AGENT_TYPE_PPO
    # Use scenario_name for dynamic/reseeded env mode.
    scenario_name: Optional[str] = None
    # Use scenario_path for static/custom env mode.
    scenario_path: Optional[str] = None
    config_path: Optional[str] = None
    model_path: Optional[str] = None
    train_seeds: Optional[Iterable[int]] = None
    eval_seeds: Optional[Iterable[int]] = None
    total_timesteps: Optional[int] = None
    eval_episodes: Optional[int] = None
    save_after_train: bool = True
    model_output_dir: str = "models"
    subgoal_manager_type: str = PPOTrainer.SUBGOAL_MANAGER_DETERMINISTIC
    use_action_masking: bool = True
    enable_pengym: Optional[bool] = None
    enable_nasim: Optional[bool] = None
    llm_client: Optional[object] = None
    translator: Optional[object] = None


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._configure_backends()
        self.trainer = PPOTrainer(
            agent_type=config.agent_type,
            subgoal_manager_type=config.subgoal_manager_type,
            use_action_masking=config.use_action_masking,
            llm_client=config.llm_client,
            translator=config.translator,
        )

    def _configure_backends(self):
        # Default behavior by scenario mode:
        # - dynamic/reseeded (scenario_name): NASIM simulation mode
        # - static/custom (scenario_path): PenGym mode
        dynamic_mode = self.config.scenario_name is not None and self.config.scenario_path is None

        if self.config.enable_pengym is None:
            enable_pengym = not dynamic_mode
        else:
            enable_pengym = self.config.enable_pengym

        if self.config.enable_nasim is None:
            enable_nasim = dynamic_mode
        else:
            enable_nasim = self.config.enable_nasim

        if not (enable_pengym or enable_nasim):
            raise ValueError("Invalid backend selection: both PenGym and NASIM are disabled.")

        utils.ENABLE_PENGYM = enable_pengym
        utils.ENABLE_NASIM = enable_nasim

        if enable_pengym:
            default_config_path = Path(__file__).resolve().parents[1] / "pengym" / "CONFIG.yml"
            config_path = self.config.config_path or str(default_config_path)
            utils.init_config_info(config_path)
            utils.init_service_port_map()

    def run(self):
        model = self.trainer.train(
            scenario_name=self.config.scenario_name,
            scenario_path=self.config.scenario_path,
            train_seeds=self.config.train_seeds,
            model_path=self.config.model_path,
            total_timesteps=self.config.total_timesteps,
        )

        model_path = None
        if self.config.save_after_train and self.config.model_path is None:
            model_path = self.trainer.save(output_dir=self.config.model_output_dir)

        done, truncated, steps = self.trainer.evaluate(
            scenario_name=self.config.scenario_name,
            scenario_path=self.config.scenario_path,
            num_episodes=self.config.eval_episodes,
            seeds=self.config.eval_seeds,
        )

        metrics = {
            "success_rate": self.trainer.last_eval_metrics.get("success_rate"),
            "average_steps": self.trainer.last_eval_metrics.get("average_steps"),
            "average_return_per_training_episodes": self.trainer.last_train_metrics.get("average_return_per_training_episodes"),
            "average_return_over_training_steps": self.trainer.last_train_metrics.get("average_return_over_training_steps"),
            "convergence_timestep": self.trainer.last_train_metrics.get("convergence_timestep"),
            "convergence_speed_over_training_steps": self.trainer.last_train_metrics.get("convergence_speed_over_training_steps"),
            "average_token_usage": self.trainer.last_eval_metrics.get("average_token_usage"),
        }

        return {
            "model": model,
            "model_path": model_path,
            "done": done,
            "truncated": truncated,
            "steps": steps,
            "metrics": metrics,
        }


def _parse_seed_list(seed_text: Optional[str]) -> Optional[list[int]]:
    if seed_text is None:
        return None

    stripped = seed_text.strip()
    if not stripped:
        return None

    seeds: list[int] = []
    for token in stripped.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            seeds.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid seed '{item}'. Seeds must be integers.") from exc

    return seeds or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate PPO or LGRL agents using lgrl_final OOP runner."
    )

    parser.add_argument(
        "--agent-type",
        choices=[PPOTrainer.AGENT_TYPE_PPO, PPOTrainer.AGENT_TYPE_LGRL],
        default=PPOTrainer.AGENT_TYPE_PPO,
        help="Agent type to run.",
    )

    scenario_group = parser.add_mutually_exclusive_group(required=True)
    scenario_group.add_argument(
        "--scenario-name",
        help="Dynamic/reseeded mode scenario name.",
    )
    scenario_group.add_argument(
        "--scenario-path",
        help="Static/custom mode scenario file path.",
    )

    default_config_path = Path(__file__).resolve().parents[1] / "pengym" / "CONFIG.yml"
    parser.add_argument(
        "--config-path",
        default=str(default_config_path),
        help="Path to PenGym CONFIG.yml used to initialize service-port mappings.",
    )

    parser.add_argument(
        "--model-path",
        help="Optional pre-trained model path to load (without changing model name).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        help="Optional training timestep override.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        help="Evaluation episodes per seed.",
    )
    parser.add_argument(
        "--train-seeds",
        help="Comma-separated training seeds, e.g. '0,1,2,3'.",
    )
    parser.add_argument(
        "--eval-seeds",
        help="Comma-separated evaluation seeds, e.g. '1000,1001'.",
    )
    parser.add_argument(
        "--subgoal-manager-type",
        choices=[
            PPOTrainer.SUBGOAL_MANAGER_DETERMINISTIC,
            PPOTrainer.SUBGOAL_MANAGER_LLM,
        ],
        default=PPOTrainer.SUBGOAL_MANAGER_DETERMINISTIC,
        help=(
            "Legacy compatibility option. LGRL pipeline now always uses deterministic "
            "subgoal manager for training and LLM subgoal manager for evaluation."
        ),
    )
    parser.add_argument(
        "--disable-action-masking",
        action="store_true",
        help="Disable action masking in both training and evaluation (ablation mode).",
    )
    parser.add_argument(
        "--disable-pengym",
        action="store_true",
        help="Disable PenGym execution backend.",
    )
    parser.add_argument(
        "--nasim-simulation",
        action="store_true",
        help="Enable NASIM simulation backend.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save model after training.",
    )
    parser.add_argument(
        "--model-output-dir",
        default="models",
        help="Directory to save models into.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print final summary as JSON.",
    )

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        train_seeds = _parse_seed_list(args.train_seeds)
        eval_seeds = _parse_seed_list(args.eval_seeds)
    except ValueError as err:
        parser.error(str(err))

    if args.agent_type == PPOTrainer.AGENT_TYPE_LGRL:
        print(
            "* NOTE: LGRL pipeline uses deterministic subgoal manager for training "
            "and LLM subgoal manager for evaluation."
        )
        print(
            "* NOTE: CLI does not inject a custom llm_client yet. LLM evaluation "
            "will use fallback behavior unless configured in Python API."
        )

    config = ExperimentConfig(
        agent_type=args.agent_type,
        scenario_name=args.scenario_name,
        scenario_path=args.scenario_path,
        config_path=args.config_path,
        model_path=args.model_path,
        train_seeds=train_seeds,
        eval_seeds=eval_seeds,
        total_timesteps=args.total_timesteps,
        eval_episodes=args.eval_episodes,
        save_after_train=not args.no_save,
        model_output_dir=args.model_output_dir,
        subgoal_manager_type=args.subgoal_manager_type,
        use_action_masking=not args.disable_action_masking,
        enable_pengym=(None if not args.disable_pengym else False),
        enable_nasim=(True if args.nasim_simulation else None),
    )

    result = ExperimentRunner(config).run()

    summary = {
        "model_path": result["model_path"],
        "done": result["done"],
        "truncated": result["truncated"],
        "steps": result["steps"],
        "use_action_masking": config.use_action_masking,
        "metrics": result["metrics"],
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("\nRun Summary")
        print(f"  model_path: {summary['model_path']}")
        print(f"  done: {summary['done']}")
        print(f"  truncated: {summary['truncated']}")
        print(f"  steps: {summary['steps']}")
        print(f"  use_action_masking: {summary['use_action_masking']}")
        print(f"  success_rate: {summary['metrics']['success_rate']}")
        print(f"  average_steps: {summary['metrics']['average_steps']}")
        print(
            "  average_return_per_training_episodes: "
            f"{summary['metrics']['average_return_per_training_episodes']}"
        )
        print(
            "  average_return_over_training_steps: "
            f"{summary['metrics']['average_return_over_training_steps']}"
        )
        print(f"  convergence_timestep: {summary['metrics']['convergence_timestep']}")
        print(
            "  convergence_speed_over_training_steps: "
            f"{summary['metrics']['convergence_speed_over_training_steps']}"
        )
        print(f"  average_token_usage: {summary['metrics']['average_token_usage']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

