#!/usr/bin/env python3
"""Generate ablation plots from a single run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


CONFIG_ORDER = [
    "Plain PPO",
    "PPO + Action Masking",
    "Deterministic LGRL",
    "Deterministic LGRL + Action Masking",
    "Pure LGRL",
    "LGRL + Action Masking",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_config_dirs(run_dir: Path) -> List[Path]:
    config_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir())
    if not config_dirs:
        raise FileNotFoundError(f"No configuration directories found in {run_dir}")
    return config_dirs


def _filter_reward_history(
    entries: List[dict],
    target_interval: int,
    include_partial: bool,
) -> Tuple[List[int], List[float]]:
    steps: List[int] = []
    returns: List[float] = []

    for entry in entries:
        step = int(entry.get("training_step", 0))
        count = int(entry.get("reward_count", target_interval))
        if step <= 0:
            continue
        if not include_partial:
            # Drop partial windows so points align to the target interval.
            if count < target_interval or step % target_interval != 0:
                continue
        steps.append(step)
        returns.append(float(entry.get("mean_reward", 0.0)))

    return steps, returns


def _sort_configs(configs: List[dict]) -> List[dict]:
    order_index = {name: idx for idx, name in enumerate(CONFIG_ORDER)}
    return sorted(
        configs,
        key=lambda item: (order_index.get(item["config_name"], len(CONFIG_ORDER)), item["config_name"]),
    )


def _plot_learning_curves(
    configs: List[dict],
    output_path: Path,
    title: str,
    target_interval: int,
    include_partial: bool,
) -> None:
    plt.figure(figsize=(12, 6))

    plotted_any = False
    for config in configs:
        label = config["config_name"]
        reward_history = config.get("reward_history", [])
        steps, returns = _filter_reward_history(reward_history, target_interval, include_partial)
        if not steps:
            continue
        plotted_any = True
        plt.plot(steps, returns, marker="o", linewidth=1.6, label=label)

    if not plotted_any:
        raise ValueError("No reward history points available after filtering.")

    plt.title(title)
    plt.xlabel("Training Timesteps")
    plt.ylabel("Average Return")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_average_steps(configs: List[dict], output_path: Path, title: str) -> None:
    labels = [config["config_name"] for config in configs]
    values = [config["average_steps"] for config in configs]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(values)), values, color="#4C78A8")
    plt.xticks(range(len(values)), labels, rotation=20, ha="right")
    plt.ylabel("Average Steps")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate learning curve and average steps plots for an ablation run.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to ablation run directory, e.g. models_ablation/generated/tiny-gen_2026-05-26_142619",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write plots (default: run dir).",
    )
    parser.add_argument(
        "--interval-steps",
        type=int,
        default=10000,
        help="Target reward history interval in timesteps (default: 10000).",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Include partial reward history windows.",
    )
    parser.add_argument(
        "--image-prefix",
        default="ablation",
        help="Prefix for generated image filenames.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    run_dir = (repo_root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dirs = _collect_config_dirs(run_dir)
    configs: List[dict] = []
    for config_dir in config_dirs:
        run_config_path = config_dir / "run_config.json"
        metrics_candidates = sorted(config_dir.glob("*_metrics.json"))
        reward_history_candidates = sorted((config_dir / "reward_history").glob("*_reward_history.json"))

        if not run_config_path.exists() or not metrics_candidates or not reward_history_candidates:
            continue

        run_config = _load_json(run_config_path)
        metrics_payload = _load_json(metrics_candidates[0])
        reward_payload = _load_json(reward_history_candidates[0])

        config_name = run_config.get("config_name", metrics_payload.get("config_name", config_dir.name))
        avg_steps = metrics_payload.get("metrics", {}).get("average_steps")
        if avg_steps is None:
            continue
        reward_history = reward_payload.get("reward_history", [])

        configs.append(
            {
                "config_name": config_name,
                "average_steps": float(avg_steps),
                "reward_history": reward_history,
                "scenario_name": run_config.get("scenario_name") or metrics_payload.get("scenario_name"),
            }
        )

    if not configs:
        raise ValueError("No configs with reward history and average steps found.")

    configs = _sort_configs(configs)
    scenario_name = configs[0].get("scenario_name") or run_dir.name

    learning_curve_path = output_dir / f"{args.image_prefix}_learning_curve.png"
    avg_steps_path = output_dir / f"{args.image_prefix}_average_steps.png"

    _plot_learning_curves(
        configs,
        learning_curve_path,
        title=f"Average Return Over Training Steps ({scenario_name})",
        target_interval=args.interval_steps,
        include_partial=args.include_partial,
    )
    _plot_average_steps(
        configs,
        avg_steps_path,
        title=f"Average Steps by Configuration ({scenario_name})",
    )

    print(f"Saved learning curve to: {learning_curve_path}")
    print(f"Saved average steps chart to: {avg_steps_path}")


if __name__ == "__main__":
    main()
