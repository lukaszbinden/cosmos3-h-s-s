#!/usr/bin/env python3
"""Aggregate paired held-out inverse-dynamics metrics across CMR subsets/seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


METRICS = (
    "mean_mse_robot_30d",
    "mean_mse_44d",
    "mean_mse_state_conditioning_14d",
    "mean_mae_robot_30d",
    "mean_mae_44d",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    return parser.parse_args()


def _load(root: Path, subset: str, seed: int) -> dict[str, Any]:
    path = root / subset / f"seed{seed}" / "c3hss_id_results.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _checkpoint_iteration(payload: dict[str, Any]) -> int:
    """Read the iteration from the checkpoint dirname.

    Weight-only inference intentionally does not restore trainer state, so the
    checkpointer's returned trainer iteration is zero even when model weights
    came from (for example) iter_000008400.
    """
    name = Path(payload["checkpoint"]).name
    return int(name.removeprefix("iter_"))


def _paired_stats(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    delta = after - before
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1))
    half_width = 4.3026527297 * sd / math.sqrt(delta.size)
    return {
        "before_seed_values": before.tolist(),
        "after_seed_values": after.tolist(),
        "before_mean": float(before.mean()),
        "after_mean": float(after.mean()),
        "paired_delta_mean": mean,
        "paired_delta_sd": sd,
        "paired_delta_95pct_t_interval": [mean - half_width, mean + half_width],
    }


def _seed_curves(payloads: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for payload in payloads:
        window_values = [
            window[key]
            for episode in payload["episode_results"]
            for window in episode["windows"]
        ]
        values.append(np.asarray(window_values, dtype=np.float64).mean(axis=0))
    return np.asarray(values, dtype=np.float64).mean(axis=0)


def main() -> None:
    args = parse_args()
    before_payloads: dict[tuple[int, str], dict[str, Any]] = {}
    after_payloads: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in args.seeds:
        for subset in args.subsets:
            before_payloads[(seed, subset)] = _load(args.before_root, subset, seed)
            after_payloads[(seed, subset)] = _load(args.after_root, subset, seed)

    first_before = before_payloads[(args.seeds[0], args.subsets[0])]
    first_after = after_payloads[(args.seeds[0], args.subsets[0])]
    before_iteration = _checkpoint_iteration(first_before)
    after_iteration = _checkpoint_iteration(first_after)

    seed_metric_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    metric_summary: dict[str, Any] = {}
    for metric in METRICS:
        before_values = np.asarray(
            [
                np.mean([before_payloads[(seed, subset)][metric] for subset in args.subsets])
                for seed in args.seeds
            ],
            dtype=np.float64,
        )
        after_values = np.asarray(
            [
                np.mean([after_payloads[(seed, subset)][metric] for subset in args.subsets])
                for seed in args.seeds
            ],
            dtype=np.float64,
        )
        seed_metric_values[metric] = (before_values, after_values)
        metric_summary[metric] = _paired_stats(before_values, after_values)

    per_subset: dict[str, Any] = {}
    for subset in args.subsets:
        per_subset[subset] = {}
        for metric in METRICS:
            before_values = np.asarray(
                [before_payloads[(seed, subset)][metric] for seed in args.seeds], dtype=np.float64
            )
            after_values = np.asarray(
                [after_payloads[(seed, subset)][metric] for seed in args.seeds], dtype=np.float64
            )
            per_subset[subset][metric] = _paired_stats(before_values, after_values)

    before_timestep = np.asarray(
        [
            np.mean(
                [
                    _seed_curves([before_payloads[(seed, subset)]], "mse_per_timestep")
                    for subset in args.subsets
                ],
                axis=0,
            )
            for seed in args.seeds
        ]
    )
    after_timestep = np.asarray(
        [
            np.mean(
                [
                    _seed_curves([after_payloads[(seed, subset)]], "mse_per_timestep")
                    for subset in args.subsets
                ],
                axis=0,
            )
            for seed in args.seeds
        ]
    )
    before_channel = np.asarray(
        [
            np.mean(
                [
                    _seed_curves([before_payloads[(seed, subset)]], "mse_per_channel")
                    for subset in args.subsets
                ],
                axis=0,
            )
            for seed in args.seeds
        ]
    )
    after_channel = np.asarray(
        [
            np.mean(
                [
                    _seed_curves([after_payloads[(seed, subset)]], "mse_per_channel")
                    for subset in args.subsets
                ],
                axis=0,
            )
            for seed in args.seeds
        ]
    )

    summary = {
        "protocol": (
            "For each checkpoint and seed, average 17 inverse-dynamics windows from five held-out "
            "episodes in each of four operative CMR subsets; balance subsets; compare the same three seeds."
        ),
        "action_space": first_before["action_space"],
        "before_checkpoint": first_before["checkpoint"],
        "after_checkpoint": first_after["checkpoint"],
        "before_iteration": before_iteration,
        "after_iteration": after_iteration,
        "subsets": args.subsets,
        "seeds": args.seeds,
        "metrics": metric_summary,
        "per_subset": per_subset,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "id_convergence_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.output_dir / "id_convergence_curves.npz",
        seeds=np.asarray(args.seeds),
        before_timestep=before_timestep,
        after_timestep=after_timestep,
        before_channel=before_channel,
        after_channel=after_channel,
    )

    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for curves, label, color in (
            (before_timestep, f"ID @ {before_iteration}", "#4c78a8"),
            (after_timestep, f"ID @ {after_iteration} post-fix", "#e45756"),
        ):
            mean = curves.mean(axis=0)
            sd = curves.std(axis=0, ddof=1)
            x = np.arange(1, mean.size + 1)
            axes[0].plot(x, mean, label=label, color=color)
            axes[0].fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18)
        axes[0].set_title("Inverse-dynamics MSE by action timestep")
        axes[0].set_xlabel("Predicted action timestep")
        axes[0].set_ylabel("Normalized 44-D MSE")
        axes[0].grid(alpha=0.25)
        axes[0].legend()
        for curves, label, color in (
            (before_channel, f"ID @ {before_iteration}", "#4c78a8"),
            (after_channel, f"ID @ {after_iteration} post-fix", "#e45756"),
        ):
            axes[1].plot(np.arange(curves.shape[1]), curves.mean(axis=0), label=label, color=color)
        axes[1].axvline(29.5, color="black", linestyle="--", linewidth=1)
        axes[1].set_title("Inverse-dynamics MSE by channel")
        axes[1].set_xlabel("Channel (0–29 robot action; 30–43 state conditioning)")
        axes[1].set_ylabel("Normalized MSE")
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "id_convergence.png", dpi=180)
        plt.close(fig)

    display_metrics = (
        ("Robot-action MSE (30D)", "mean_mse_robot_30d"),
        ("All supervised MSE (44D)", "mean_mse_44d"),
        ("State-conditioning MSE (14D)", "mean_mse_state_conditioning_14d"),
        ("Robot-action MAE (30D)", "mean_mae_robot_30d"),
    )
    lines = [
        "# C3-H-S-S inverse-dynamics convergence",
        "",
        f"Paired held-out comparison of checkpoint `{before_iteration}` and post-fix checkpoint `{after_iteration}`.",
        "Metrics are in the normalized hybrid-relative action space used during training. Lower is better.",
        "",
        "| Metric | Before | After | Paired after-before delta | 95% t interval (n=3) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in display_metrics:
        item = metric_summary[key]
        low, high = item["paired_delta_95pct_t_interval"]
        lines.append(
            f"| {label} | {item['before_mean']:.6f} | {item['after_mean']:.6f} | "
            f"{item['paired_delta_mean']:+.6f} | [{low:+.6f}, {high:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "| Operative CMR subset | Robot-action MSE before | Robot-action MSE after | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for subset in args.subsets:
        item = per_subset[subset]["mean_mse_robot_30d"]
        lines.append(
            f"| {subset} | {item['before_mean']:.6f} | {item['after_mean']:.6f} | "
            f"{item['paired_delta_mean']:+.6f} |"
        )
    if plt is not None:
        lines.extend(["", "![Inverse-dynamics convergence](id_convergence.png)"])
    lines.append("")
    (args.output_dir / "id_convergence_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
