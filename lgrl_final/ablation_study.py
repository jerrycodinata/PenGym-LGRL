"""
Ablation study entry point for LGRL pipeline.
Runs 6 experimental configurations across randomized training seeds.

Configurations:
1. Plain PPO: agent_type=ppo, masking=disabled
2. PPO + Action Masking: agent_type=ppo, masking=enabled
3. Deterministic LGRL: agent_type=lgrl, masking=disabled, deterministic eval subgoal manager
4. Deterministic LGRL + Action Masking: agent_type=lgrl, masking=enabled, deterministic eval subgoal manager
5. Pure LGRL: agent_type=lgrl, masking=disabled, LLM eval subgoal manager
6. LGRL + Action Masking: agent_type=lgrl, masking=enabled, LLM eval subgoal manager
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pengym.utilities as utils

from lgrl_final.llm_clients import build_llm_client
from lgrl_final.observation_translator import ObservationTranslator
from lgrl_final.main import ExperimentConfig, ExperimentRunner
from lgrl_final.ppo_trainer import PPOTrainer


def _slugify_label(label: str) -> str:
    return "".join(char if char.isalnum() or char in {".", "_", "-"} else "_" for char in label).strip("_") or "run"


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
    run_label: Optional[str] = None
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
            "subgoal_manager_type": PPOTrainer.SUBGOAL_MANAGER_DETERMINISTIC,
        },
        {
            "name": "Deterministic LGRL + Action Masking",
            "agent_type": PPOTrainer.AGENT_TYPE_LGRL,
            "use_action_masking": True,
            "subgoal_manager_type": PPOTrainer.SUBGOAL_MANAGER_DETERMINISTIC,
        },
        {
            "name": "Pure LGRL",
            "agent_type": PPOTrainer.AGENT_TYPE_LGRL,
            "use_action_masking": False,
            "subgoal_manager_type": PPOTrainer.SUBGOAL_MANAGER_LLM,
        },
        {
            "name": "LGRL + Action Masking",
            "agent_type": PPOTrainer.AGENT_TYPE_LGRL,
            "use_action_masking": True,
            "subgoal_manager_type": PPOTrainer.SUBGOAL_MANAGER_LLM,
        },
    ]

    def __init__(self, ablation_config: AblationStudyConfig):
        self.ablation_config = ablation_config
        self.results = {}
        self.generated_root = Path(self.ablation_config.model_output_dir) / "generated"

        if self.ablation_config.seeds is None:
            random.seed(42)
            self.ablation_config.seeds = [random.randint(0, 10000) for _ in range(ablation_config.num_seeds)]

        if self.ablation_config.eval_seeds is None:
            self.ablation_config.eval_seeds = [1000 + i for i in range(10)]

    def run_all(self) -> dict:
        """Run all 6 configurations."""
        print("=" * 80)
        print("LGRL ABLATION STUDY")
        print("=" * 80)
        print(f"Scenario: {self.ablation_config.scenario_name or self.ablation_config.scenario_path}")
        if self.ablation_config.total_timesteps is not None:
            print(f"Max timesteps: {self.ablation_config.total_timesteps}")
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

    @staticmethod
    def _format_value(value):
        if value is None:
            return "N/A"
        if isinstance(value, bool):
            return "Enabled" if value else "Disabled"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _print_results_table(self, summary: dict) -> None:
        columns = [
            ("Configuration", "configuration"),
            ("Agent", "agent_type"),
            ("Mask", "use_action_masking"),
            ("Success Rate", "success_rate"),
            ("Avg Steps", "average_steps"),
            ("Avg Return/Ep", "average_return_per_training_episodes"),
            ("Avg Return/Step", "average_return_over_training_steps"),
            ("Avg Reward/Step", "average_reward_over_training_steps"),
            ("Convergence Timestep", "convergence_timestep"),
            ("Convergence Speed", "convergence_speed_over_training_steps"),
            ("Avg Token Usage", "average_token_usage"),
        ]

        widths = []
        for header, key in columns:
            max_width = len(header)
            for row in summary.values():
                value = row.get(key)
                max_width = max(max_width, len(self._format_value(value)))
            widths.append(max_width)

        def render_row(values):
            return " | ".join(str(value).ljust(width) for value, width in zip(values, widths))

        headers = [header for header, _ in columns]
        separator = "-+-".join("-" * width for width in widths)

        print("\nAblation Study Results Table")
        print(render_row(headers))
        print(separator)
        for row in summary.values():
            rendered = [self._format_value(row.get(key)) for _, key in columns]
            print(render_row(rendered))

    def _write_run_manifest(self, run_output_dir: Path, config_name: str, config_spec: dict, run_folder_name: str) -> Path:
        """Persist the effective configuration for a single ablation run."""
        manifest_path = run_output_dir / "run_config.json"
        run_manifest = {
            "config_name": config_name,
            "scenario_name": self.ablation_config.scenario_name,
            "scenario_path": self.ablation_config.scenario_path,
            "run_label": run_folder_name,
            "agent_type": config_spec["agent_type"],
            "subgoal_manager_type": config_spec.get(
                "subgoal_manager_type",
                PPOTrainer.SUBGOAL_MANAGER_DETERMINISTIC,
            ),
            "use_action_masking": config_spec["use_action_masking"],
            "train_seeds": self.ablation_config.seeds,
            "eval_seeds": self.ablation_config.eval_seeds,
            "num_seeds": len(self.ablation_config.seeds or []),
            "eval_episodes": self.ablation_config.eval_episodes,
            "total_timesteps": self.ablation_config.total_timesteps,
            "save_after_train": self.ablation_config.save_after_train,
            "config_path": self.ablation_config.config_path,
            "enable_pengym": self.ablation_config.enable_pengym,
            "enable_nasim": self.ablation_config.enable_nasim,
            "model_output_dir": str(run_output_dir),
        }
        manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        return manifest_path

    def _run_configuration(self, config_spec: dict) -> dict:
        """Run a single ablation configuration."""
        config_name = config_spec["name"]
        scenario_label = self.ablation_config.scenario_name or (
            Path(self.ablation_config.scenario_path).stem if self.ablation_config.scenario_path else "scenario"
        )
        run_folder_name = _slugify_label(f"{scenario_label}_{config_name}")
        run_output_dir = self.generated_root / run_folder_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        run_manifest_path = self._write_run_manifest(run_output_dir, config_name, config_spec, run_folder_name)

        config = ExperimentConfig(
            agent_type=config_spec["agent_type"],
            scenario_name=self.ablation_config.scenario_name,
            scenario_path=self.ablation_config.scenario_path,
            config_path=self.ablation_config.config_path,
            run_label=run_folder_name,
            subgoal_manager_type=config_spec.get(
                "subgoal_manager_type",
                PPOTrainer.SUBGOAL_MANAGER_DETERMINISTIC,
            ),
            train_seeds=self.ablation_config.seeds,
            eval_seeds=self.ablation_config.eval_seeds,
            total_timesteps=self.ablation_config.total_timesteps,
            eval_episodes=self.ablation_config.eval_episodes,
            save_after_train=self.ablation_config.save_after_train,
            model_output_dir=str(run_output_dir),
            use_action_masking=config_spec["use_action_masking"],
            enable_pengym=self.ablation_config.enable_pengym,
            enable_nasim=self.ablation_config.enable_nasim,
            llm_client=self.ablation_config.llm_client,
            translator=self.ablation_config.translator,
        )

        runner = ExperimentRunner(config)
        result = runner.run()

        metrics_path = run_output_dir / f"{run_folder_name}_metrics.json"
        metrics_payload = {
            "config_name": config_name,
            "scenario_name": self.ablation_config.scenario_name,
            "scenario_path": self.ablation_config.scenario_path,
            "run_label": run_folder_name,
            "run_config_path": str(run_manifest_path),
            "agent_type": config_spec["agent_type"],
            "use_action_masking": config_spec["use_action_masking"],
            "model_path": result["model_path"],
            "reward_history_artifacts": result.get("reward_history_artifacts"),
            "metrics": result["metrics"],
            "done": result["done"],
            "truncated": result["truncated"],
            "steps": result["steps"],
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        return {
            "config_name": config_name,
            "agent_type": config_spec["agent_type"],
            "use_action_masking": config_spec["use_action_masking"],
            "model_path": result["model_path"],
            "reward_history_artifacts": result.get("reward_history_artifacts"),
            "run_config_path": str(run_manifest_path),
            "metrics_path": str(metrics_path),
            "run_output_dir": str(run_output_dir),
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
                "configuration": config_name,
                "agent_type": result["agent_type"],
                "use_action_masking": result["use_action_masking"],
                "run_output_dir": result.get("run_output_dir"),
                "metrics_path": result.get("metrics_path"),
                "reward_history_artifacts": result.get("reward_history_artifacts"),
                "success_rate": metrics.get("success_rate"),
                "average_steps": metrics.get("average_steps"),
                "average_return_per_training_episodes": metrics.get("average_return_per_training_episodes"),
                "average_return_over_training_steps": metrics.get("average_return_over_training_steps"),
                "average_reward_over_training_steps": metrics.get("average_reward_over_training_steps"),
                "convergence_timestep": metrics.get("convergence_timestep"),
                "convergence_speed_over_training_steps": metrics.get("convergence_speed_over_training_steps"),
                "average_token_usage": metrics.get("average_token_usage"),
            }

            print(f"Configuration: {config_name}")
            print(f"  Agent Type: {result['agent_type']}")
            print(f"  Action Masking: {'Enabled' if result['use_action_masking'] else 'Disabled'}")
            print(f"  Output Dir: {result.get('run_output_dir', 'N/A')}")
            print(f"  Run Config: {result.get('run_config_path', 'N/A')}")
            print(f"  Metrics File: {result.get('metrics_path', 'N/A')}")
            print(f"  Reward Artifacts: {result.get('reward_history_artifacts', 'N/A')}")
            print(f"  Success Rate: {metrics.get('success_rate', 'N/A')}")
            print(f"  Average Steps: {metrics.get('average_steps', 'N/A')}")
            print(f"  Avg Return per Episodes: {metrics.get('average_return_per_training_episodes', 'N/A')}")
            print(f"  Avg Return over Steps: {metrics.get('average_return_over_training_steps', 'N/A')}")
            print(f"  Avg Reward over Training Steps: {metrics.get('average_reward_over_training_steps', 'N/A')}")
            print(f"  Convergence Timestep: {metrics.get('convergence_timestep', 'N/A')}")
            print(f"  Convergence Speed: {metrics.get('convergence_speed_over_training_steps', 'N/A')}")
            print(f"  Avg Token Usage: {metrics.get('average_token_usage', 'N/A')}")
            print()

        self._print_results_table(summary)

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
        "--llm-provider",
        choices=["deepseek"],
        default="deepseek",
        help="LLM provider to use for LLM-guided ablation runs.",
    )
    parser.add_argument(
        "--llm-api-key",
        help="API key for the selected LLM provider. Defaults to DEEPSEEK_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--llm-base-url",
        help="Override the provider base URL. Defaults to https://api.deepseek.com for DeepSeek.",
    )
    parser.add_argument(
        "--llm-model",
        default="deepseek-v4-pro",
        help="Model name to call for LLM-guided evaluation.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the LLM provider.",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=16,
        help="Maximum tokens to request from the LLM provider.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds for the LLM provider.",
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

    try:
        llm_client = build_llm_client(
            provider=args.llm_provider,
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            model=args.llm_model,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            timeout=args.llm_timeout,
        )
    except ValueError as err:
        parser.error(str(err))

    translator = ObservationTranslator()

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
        llm_client=llm_client,
        translator=translator,
    )

    runner = AblationStudyRunner(ablation_config)
    summary = runner.run_all()

    if args.json:
        print("\n" + json.dumps(summary, indent=2))
    else:
        print("\nAblation study complete.")
        print(f"Generated outputs root: {Path(args.model_output_dir) / 'generated'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
