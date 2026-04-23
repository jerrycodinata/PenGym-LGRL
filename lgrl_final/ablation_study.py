"""
Ablation study entry point for LGRL pipeline.
Runs 6 experimental configurations across randomized training seeds.

Configurations:
1. Plain PPO: agent_type=ppo, masking=disabled
2. PPO + Action Masking: agent_type=ppo, masking=enabled
3. Deterministic LGRL: agent_type=lgrl, masking=disabled (training uses deterministic subgoal)
4. Deterministic LGRL + Action Masking: agent_type=lgrl, masking=enabled (training uses deterministic subgoal)
5. Pure LGRL: agent_type=lgrl, masking=disabled (eval uses LLM subgoal; training still deterministic)
6. LGRL + Action Masking: agent_type=lgrl, masking=enabled (eval uses LLM subgoal; training still deterministic)
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pengym.utilities as utils

from lgrl_final.main import ExperimentConfig, ExperimentRunner
from lgrl_final.ppo_trainer import PPOTrainer


@dataclass
class AblationStudyConfig:
    """Configuration for ablation study sweep."""
    scenario_name: Optional[str] = None
    scenario_path: Optional[str] = None
    config_path: Optional[str] = None
    num_seeds: int = 5
    seeds: Optional[List[int]] = None
    eval_seeds: Optional[List[int]] = None
    total_timesteps: Optional[int] = None
    eval_episodes: Optional[int] = None
    model_output_dir: str = "models_ablation"
    save_after_train: bool = True
    enable_pengym: Optional[bool] = None
    enable_nasim: Optional[bool] = None
    llm_client: Optional[object] = None
    translator: Optional[object] = None


class AblationStudyRunner:
    """Runs ablation study across all 6 configurations."""

    CONFIGURATIONS = [
        {
            "name": "Plain PPO",
            "agent_type": PPOTrainer.AGENT_TYPE_PPO,
            "use_action_masking": False,
        },
        {
            "name": "PPO + Action Masking",
            "agent_type": PPOTrainer.AGENT_TYPE_PPO,
            "use_action_masking": True,
        },
        {
            "name": "Deterministic LGRL",
            "agent_type": PPOTrainer.AGENT_TYPE_LGRL,
            "use_action_masking": False,
        },
        {
            "name": "Deterministic LGRL + Action Masking",
            "agent_type": PPOTrainer.AGENT_TYPE_LGRL,
            "use_action_masking": True,
        },
        {
            "name": "Pure LGRL",
            "agent_type": PPOTrainer.AGENT_TYPE_LGRL,
            "use_action_masking": False,
        },
        {
            "name": "LGRL + Action Masking",
            "agent_type": PPOTrainer.AGENT_TYPE_LGRL,
            "use_action_masking": True,
        },
    ]

    def __init__(self, ablation_config: AblationStudyConfig):
        self.ablation_config = ablation_config
        self.results = {}

        if self.ablation_config.seeds is None:
            random.seed(42)
            self.ablation_config.seeds = [random.randint(0, 10000) for _ in range(ablation_config.num_seeds)]

        if self.ablation_config.eval_seeds is None:
            self.ablation_config.eval_seeds = [1000 + i for i in range(4)]

    def run_all(self) -> dict:
        """Run all 6 configurations."""
        print("=" * 80)
        print("LGRL ABLATION STUDY")
        print("=" * 80)
        print(f"Scenario: {self.ablation_config.scenario_name or self.ablation_config.scenario_path}")
        print(f"Training seeds: {self.ablation_config.seeds}")
        print(f"Evaluation seeds: {self.ablation_config.eval_seeds}")
        print(f"Number of configurations: {len(self.CONFIGURATIONS)}")
        print("=" * 80)

        for config_idx, config_spec in enumerate(self.CONFIGURATIONS, 1):
            print(f"\n{'='*80}")
            print(f"[{config_idx}/{len(self.CONFIGURATIONS)}] {config_spec['name']}")
            print(f"{'='*80}")

            result = self._run_configuration(config_spec)
            self.results[config_spec["name"]] = result

        return self._aggregate_results()

    def _run_configuration(self, config_spec: dict) -> dict:
        """Run a single ablation configuration."""
        config_name = config_spec["name"]

        config = ExperimentConfig(
            agent_type=config_spec["agent_type"],
            scenario_name=self.ablation_config.scenario_name,
            scenario_path=self.ablation_config.scenario_path,
            config_path=self.ablation_config.config_path,
            train_seeds=self.ablation_config.seeds,
            eval_seeds=self.ablation_config.eval_seeds,
            total_timesteps=self.ablation_config.total_timesteps,
            eval_episodes=self.ablation_config.eval_episodes,
            save_after_train=self.ablation_config.save_after_train,
            model_output_dir=self.ablation_config.model_output_dir,
            use_action_masking=config_spec["use_action_masking"],
            enable_pengym=self.ablation_config.enable_pengym,
            enable_nasim=self.ablation_config.enable_nasim,
            llm_client=self.ablation_config.llm_client,
            translator=self.ablation_config.translator,
        )

        runner = ExperimentRunner(config)
        result = runner.run()

        return {
            "config_name": config_name,
            "agent_type": config_spec["agent_type"],
            "use_action_masking": config_spec["use_action_masking"],
            "model_path": result["model_path"],
            "metrics": result["metrics"],
            "done": result["done"],
            "truncated": result["truncated"],
            "steps": result["steps"],
        }

    def _aggregate_results(self) -> dict:
        """Aggregate results across all configurations."""
        print(f"\n\n{'='*80}")
        print("ABLATION STUDY SUMMARY")
        print(f"{'='*80}\n")

        summary = {}
        for config_name, result in self.results.items():
            metrics = result["metrics"]
            summary[config_name] = {
                "agent_type": result["agent_type"],
                "use_action_masking": result["use_action_masking"],
                "success_rate": metrics.get("success_rate"),
                "average_steps": metrics.get("average_steps"),
                "average_return_per_training_episodes": metrics.get("average_return_per_training_episodes"),
                "average_return_over_training_steps": metrics.get("average_return_over_training_steps"),
                "convergence_timestep": metrics.get("convergence_timestep"),
                "convergence_speed_over_training_steps": metrics.get("convergence_speed_over_training_steps"),
                "average_token_usage": metrics.get("average_token_usage"),
            }

            print(f"Configuration: {config_name}")
            print(f"  Agent Type: {result['agent_type']}")
            print(f"  Action Masking: {'Enabled' if result['use_action_masking'] else 'Disabled'}")
            print(f"  Success Rate: {metrics.get('success_rate', 'N/A')}")
            print(f"  Average Steps: {metrics.get('average_steps', 'N/A')}")
            print(f"  Avg Return per Episodes: {metrics.get('average_return_per_training_episodes', 'N/A')}")
            print(f"  Avg Return over Steps: {metrics.get('average_return_over_training_steps', 'N/A')}")
            print(f"  Convergence Timestep: {metrics.get('convergence_timestep', 'N/A')}")
            print(f"  Convergence Speed: {metrics.get('convergence_speed_over_training_steps', 'N/A')}")
            print(f"  Avg Token Usage: {metrics.get('average_token_usage', 'N/A')}")
            print()

        return summary


def _parse_seed_list(seed_text: Optional[str]) -> Optional[list]:
    """Parse comma-separated seed list."""
    if seed_text is None:
        return None
    stripped = seed_text.strip()
    if not stripped:
        return None
    seeds = []
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
        description="Run ablation study across 6 LGRL configurations."
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
        help="Path to PenGym CONFIG.yml.",
    )

    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of random seeds to generate for training (default: 5).",
    )
    parser.add_argument(
        "--seeds",
        help="Comma-separated training seeds, e.g. '0,1,2,3,4'. Overrides --num-seeds.",
    )
    parser.add_argument(
        "--eval-seeds",
        help="Comma-separated evaluation seeds, e.g. '1000,1001,1002,1003'.",
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
        "--model-output-dir",
        default="models_ablation",
        help="Directory to save models into.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save models after training.",
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
        "--json",
        action="store_true",
        help="Output final summary as JSON.",
    )

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        train_seeds = _parse_seed_list(args.seeds)
        eval_seeds = _parse_seed_list(args.eval_seeds)
    except ValueError as err:
        parser.error(str(err))

    ablation_config = AblationStudyConfig(
        scenario_name=args.scenario_name,
        scenario_path=args.scenario_path,
        config_path=args.config_path,
        num_seeds=args.num_seeds if train_seeds is None else len(train_seeds),
        seeds=train_seeds,
        eval_seeds=eval_seeds,
        total_timesteps=args.total_timesteps,
        eval_episodes=args.eval_episodes,
        model_output_dir=args.model_output_dir,
        save_after_train=not args.no_save,
        enable_pengym=(None if not args.disable_pengym else False),
        enable_nasim=(True if args.nasim_simulation else None),
    )

    runner = AblationStudyRunner(ablation_config)
    summary = runner.run_all()

    if args.json:
        print("\n" + json.dumps(summary, indent=2))
    else:
        print("\nAblation study complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
