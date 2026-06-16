#!/usr/bin/env python3
"""Generate mean ablation plots from the latest N tiny-gen runs using training logs only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib.pyplot as plt


CONFIG_ORDER = [
    "PPO",
    "PPO + Action Masking",
    "Deterministic LGRL",
    "Deterministic LGRL + Action Masking",
    "LGRL",
    "LGRL + Action Masking",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest_run_dirs(runs_root: Path, run_prefix: str, num_runs: int) -> List[Path]:
    candidates = [
        path
        for path in runs_root.iterdir()
        if path.is_dir() and path.name.startswith(run_prefix)
    ]
    if not candidates:
        raise ValueError(
            f"No runs found with prefix '{run_prefix}' under {runs_root}."
        )

    # Name format is sortable: tiny-gen_YYYY-MM-DD_HHMMSS
    candidates.sort(key=lambda p: p.name)
    return candidates[-num_runs:]


def _parse_explicit_run_dirs(run_dirs_text: str, repo_root: Path) -> List[Path]:
    selected: List[Path] = []
    for token in run_dirs_text.split(","):
        item = token.strip()
        if not item:
            continue
        path = Path(item)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Run directory does not exist or is not a directory: {path}")
        selected.append(path)

    if not selected:
        raise ValueError("No valid run directories were provided to --run-dirs.")

    return selected


def _collect_run_data(run_dir: Path) -> List[dict]:
    configs: List[dict] = []
    for config_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
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
            }
        )

    return configs


def _aggregate_mean_data(
    run_payloads: List[List[dict]],
    interval_steps: int,
) -> tuple[Dict[str, Dict[int, float]], Dict[str, float]]:
    curve_accumulator: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    as_accumulator: Dict[str, List[float]] = defaultdict(list)

    for run_configs in run_payloads:
        for config in run_configs:
            config_name = config["config_name"]
            as_accumulator[config_name].append(config["average_steps"])

            for row in config.get("reward_history", []):
                step = int(row.get("training_step", 0))
                reward_count = int(row.get("reward_count", 0))
                if step <= 0:
                    continue
                if step % interval_steps != 0:
                    continue
                if reward_count < interval_steps:
                    continue
                curve_accumulator[config_name][step].append(float(row.get("mean_reward", 0.0)))

    mean_curves: Dict[str, Dict[int, float]] = {}
    for config_name, step_map in curve_accumulator.items():
        mean_curves[config_name] = {step: mean(vals) for step, vals in sorted(step_map.items()) if vals}

    mean_average_steps = {
        config_name: mean(values) for config_name, values in as_accumulator.items() if values
    }

    return mean_curves, mean_average_steps


def _sorted_config_names(config_names: List[str]) -> List[str]:
    order_index = {name: idx for idx, name in enumerate(CONFIG_ORDER)}
    return sorted(config_names, key=lambda x: (order_index.get(x, len(CONFIG_ORDER)), x))


def _plot_mean_learning_curves(
    mean_curves: Dict[str, Dict[int, float]],
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(12, 6))
    plotted_any = False

    for config_name in _sorted_config_names(list(mean_curves.keys())):
        step_map = mean_curves[config_name]
        if not step_map:
            continue
        plotted_any = True
        steps = list(step_map.keys())
        values = [step_map[s] for s in steps]
        plt.plot(steps, values, marker="o", linewidth=1.8, label=config_name)

    if not plotted_any:
        raise ValueError("No learning-curve points found after aggregation.")

    plt.title(title)
    plt.xlabel("Training Timesteps")
    plt.ylabel("Average Return")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_mean_average_steps(mean_average_steps: Dict[str, float], output_path: Path, title: str) -> None:
    config_names = _sorted_config_names(list(mean_average_steps.keys()))
    if not config_names:
        raise ValueError("No average-steps data found after aggregation.")

    values = [mean_average_steps[name] for name in config_names]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(values)), values, color="#4C78A8")
    plt.xticks(range(len(values)), config_names, rotation=20, ha="right")
    plt.ylabel("Average Steps")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
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
        description="Generate mean learning curve and mean average-steps plots from latest tiny-gen runs.",
    )
    parser.add_argument(
        "--runs-root",
        default="models_ablation/generated",
        help="Root directory containing generated ablation run folders.",
    )
    parser.add_argument(
        "--run-dirs",
        default=None,
        help=(
            "Comma-separated explicit run directories to use. "
            "If set, overrides --runs-root/--run-prefix/--num-runs."
        ),
    )
    parser.add_argument(
        "--run-prefix",
        default="tiny-gen_2026-",
        help="Run directory prefix used to select runs.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of latest runs to aggregate (default: 3).",
    )
    parser.add_argument(
        "--interval-steps",
        type=int,
        default=10000,
        help="Timesteps interval for learning-curve points (default: 10000).",
    )
    parser.add_argument(
        "--output-dir",
        default="models_ablation/generated/images",
        help="Output directory for generated images.",
    )
    parser.add_argument(
        "--image-prefix",
        default="tiny_gen_mean_last3",
        help="Filename prefix for generated images.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_root = (repo_root / args.runs_root).resolve() if not Path(args.runs_root).is_absolute() else Path(args.runs_root)
    output_dir = (repo_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_dirs:
        latest_runs = _parse_explicit_run_dirs(args.run_dirs, repo_root)
    else:
        latest_runs = _find_latest_run_dirs(runs_root, args.run_prefix, args.num_runs)
        if len(latest_runs) < args.num_runs:
            print(
                f"Warning: requested {args.num_runs} runs but found {len(latest_runs)} for prefix '{args.run_prefix}'. "
                "Using available runs."
            )
    run_payloads = [_collect_run_data(run_dir) for run_dir in latest_runs]

    mean_curves, mean_average_steps = _aggregate_mean_data(run_payloads, args.interval_steps)

    curve_out = output_dir / f"{args.image_prefix}_learning_curve.png"
    as_out = output_dir / f"{args.image_prefix}_average_steps.png"

    _plot_mean_learning_curves(
        mean_curves,
        curve_out,
        title=f"Mean Average Return Over Training Steps ({args.num_runs} latest runs)",
    )
    _plot_mean_average_steps(
        mean_average_steps,
        as_out,
        title=f"Mean Average Steps by Configuration ({args.num_runs} latest runs)",
    )

    selected = ", ".join(run.name for run in latest_runs)
    print(f"Used runs: {selected}")
    print(f"Saved mean learning curve to: {curve_out}")
    print(f"Saved mean average steps chart to: {as_out}")


if __name__ == "__main__":
    main()
