#!/usr/bin/env python3
"""Combine paired convergence reports into one fixed-baseline checkpoint trajectory."""

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


T_CRIT_DF2_95 = 4.3026527297


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-8400-9400", type=Path, required=True)
    parser.add_argument("--policy-8400-9400", type=Path, required=True)
    parser.add_argument("--id-9400-10000", type=Path, required=True)
    parser.add_argument("--policy-9400-10000", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def three_checkpoint_seed_values(
    earlier: dict[str, Any], later: dict[str, Any], key: str
) -> np.ndarray:
    early_metric = earlier["metrics"][key]
    late_metric = later["metrics"][key]
    middle_a = np.asarray(early_metric["after_seed_values"], dtype=np.float64)
    middle_b = np.asarray(late_metric["before_seed_values"], dtype=np.float64)
    np.testing.assert_allclose(middle_a, middle_b, rtol=0.0, atol=1e-12)
    return np.asarray(
        [
            early_metric["before_seed_values"],
            early_metric["after_seed_values"],
            late_metric["after_seed_values"],
        ],
        dtype=np.float64,
    )


def paired_stats(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    delta = after - before
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1))
    half_width = T_CRIT_DF2_95 * sd / math.sqrt(delta.size)
    before_mean = float(before.mean())
    return {
        "before_mean": before_mean,
        "after_mean": float(after.mean()),
        "paired_delta_mean": mean,
        "relative_change_pct": 100.0 * mean / before_mean,
        "paired_delta_95pct_t_interval": [mean - half_width, mean + half_width],
    }


def series_summary(iterations: list[int], values: np.ndarray) -> dict[str, Any]:
    return {
        "seed_values_by_checkpoint": {
            str(iteration): values[index].tolist() for index, iteration in enumerate(iterations)
        },
        "mean_by_checkpoint": {
            str(iteration): float(values[index].mean())
            for index, iteration in enumerate(iterations)
        },
        "sample_sd_by_checkpoint": {
            str(iteration): float(values[index].std(ddof=1))
            for index, iteration in enumerate(iterations)
        },
        "paired_intervals": {
            f"{iterations[0]}_to_{iterations[1]}": paired_stats(values[0], values[1]),
            f"{iterations[1]}_to_{iterations[2]}": paired_stats(values[1], values[2]),
        },
    }


def subset_means(
    earlier: dict[str, Any], later: dict[str, Any], subset: str, key: str
) -> list[float]:
    early_metric = earlier["per_subset"][subset][key]
    late_metric = later["per_subset"][subset][key]
    if not math.isclose(
        float(early_metric["after_mean"]),
        float(late_metric["before_mean"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Linked checkpoint mismatch for {subset} / {key}")
    return [
        float(early_metric["before_mean"]),
        float(early_metric["after_mean"]),
        float(late_metric["after_mean"]),
    ]


def main() -> None:
    args = parse_args()
    id_early = load(args.id_8400_9400)
    policy_early = load(args.policy_8400_9400)
    id_late = load(args.id_9400_10000)
    policy_late = load(args.policy_9400_10000)

    iterations = [
        int(id_early["before_iteration"]),
        int(id_early["after_iteration"]),
        int(id_late["after_iteration"]),
    ]
    expected_links = (
        int(id_late["before_iteration"]),
        int(policy_early["before_iteration"]),
        int(policy_early["after_iteration"]),
        int(policy_late["before_iteration"]),
        int(policy_late["after_iteration"]),
    )
    if expected_links != (iterations[1], iterations[0], iterations[1], iterations[1], iterations[2]):
        raise ValueError(f"Checkpoint summaries do not form one trajectory: {iterations}, {expected_links}")

    id_action = three_checkpoint_seed_values(
        id_early, id_late, "mean_mse_robot_30d"
    )
    policy_action = three_checkpoint_seed_values(
        policy_early, policy_late, "robot_action_mse_30d"
    )
    policy_video = three_checkpoint_seed_values(
        policy_early, policy_late, "rollout_video_l1"
    )

    action_weight = float(policy_early["composite_definition"]["action_weight"])
    video_weight = float(policy_early["composite_definition"]["video_weight"])
    baseline_action = float(policy_early["metrics"]["robot_action_mse_30d"]["before_mean"])
    baseline_video = float(policy_early["metrics"]["rollout_video_l1"]["before_mean"])
    policy_composite = (
        action_weight * policy_action / baseline_action
        + video_weight * policy_video / baseline_video
    )

    subsets = list(id_early["subsets"])
    per_subset: dict[str, Any] = {}
    for subset in subsets:
        per_subset[subset] = {
            "id_robot_action_mse_30d": subset_means(
                id_early, id_late, subset, "mean_mse_robot_30d"
            ),
            "policy_robot_action_mse_30d": subset_means(
                policy_early, policy_late, subset, "robot_action_mse_30d"
            ),
            "policy_rollout_video_l1": subset_means(
                policy_early, policy_late, subset, "rollout_video_l1"
            ),
        }

    series = {
        "id_robot_action_mse_30d": series_summary(iterations, id_action),
        "policy_robot_action_mse_30d": series_summary(iterations, policy_action),
        "policy_rollout_video_l1": series_summary(iterations, policy_video),
        "policy_fixed_8400_composite_2_to_1": series_summary(iterations, policy_composite),
    }
    summary = {
        "protocol": (
            "Same paired held-out protocol at every checkpoint: three seeds, five episodes from each "
            "of all four operative CMR subsets, and 17 chunks / 204 predicted frames for policy rollout."
        ),
        "checkpoint_iterations": iterations,
        "lower_is_better": True,
        "composite_definition": {
            "action_weight": action_weight,
            "video_weight": video_weight,
            "fixed_normalization_checkpoint": iterations[0],
            "baseline_action_mse": baseline_action,
            "baseline_video_l1": baseline_video,
        },
        "metrics": series,
        "per_subset_means_in_checkpoint_order": per_subset,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoint_trajectory_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    plot_path = args.output_dir / "checkpoint_trajectory.png"
    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        panels = (
            (axes[0, 0], id_action, "ID robot-action MSE", "Normalized MSE"),
            (axes[0, 1], policy_action, "Policy robot-action MSE", "Normalized MSE"),
            (axes[1, 0], policy_video, "Closed-loop rollout-video L1", "L1"),
            (axes[1, 1], policy_composite, "Policy composite (fixed @ 8400)", "Composite"),
        )
        for axis, values, title, ylabel in panels:
            axis.errorbar(
                iterations,
                values.mean(axis=1),
                yerr=values.std(axis=1, ddof=1),
                marker="o",
                capsize=4,
                linewidth=2,
            )
            axis.set_title(title)
            axis.set_xlabel("Checkpoint iteration")
            axis.set_ylabel(ylabel)
            axis.set_xticks(iterations)
            axis.grid(alpha=0.25)
        fig.suptitle("C3-H-S-S validation trajectory across all four operative CMR subsets")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)

    metric_rows = (
        ("ID robot-action MSE", "id_robot_action_mse_30d"),
        ("Policy robot-action MSE", "policy_robot_action_mse_30d"),
        ("Policy rollout-video L1", "policy_rollout_video_l1"),
        ("Policy fixed-8400 composite (2:1)", "policy_fixed_8400_composite_2_to_1"),
    )
    lines = [
        "# C3-H-S-S checkpoint validation trajectory",
        "",
        "All checkpoints use the same three seeds, five held-out episodes per operative CMR subset, "
        "four subsets, and a 17-chunk policy horizon. Lower is better.",
        "",
        "| Checkpoint | ID action MSE | Policy action MSE | Policy video L1 | Fixed-8400 composite |",
        "|---:|---:|---:|---:|---:|",
    ]
    for index, iteration in enumerate(iterations):
        lines.append(
            f"| {iteration} | {id_action[index].mean():.6f} | {policy_action[index].mean():.6f} | "
            f"{policy_video[index].mean():.6f} | {policy_composite[index].mean():.6f} |"
        )

    lines.extend(
        [
            "",
            f"## Paired change from {iterations[1]} to {iterations[2]}",
            "",
            "| Metric | Before | After | Relative change | 95% paired t interval (n=3) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, key in metric_rows:
        item = series[key]["paired_intervals"][f"{iterations[1]}_to_{iterations[2]}"]
        low, high = item["paired_delta_95pct_t_interval"]
        lines.append(
            f"| {label} | {item['before_mean']:.6f} | {item['after_mean']:.6f} | "
            f"{item['relative_change_pct']:+.2f}% | [{low:+.6f}, {high:+.6f}] |"
        )

    lines.extend(
        [
            "",
            "## Per-subset action MSE trajectory",
            "",
            "| Operative CMR subset | ID @ 8400 | ID @ 9400 | ID @ 10000 | "
            "Policy @ 8400 | Policy @ 9400 | Policy @ 10000 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for subset in subsets:
        id_values = per_subset[subset]["id_robot_action_mse_30d"]
        policy_values = per_subset[subset]["policy_robot_action_mse_30d"]
        lines.append(
            f"| {subset} | {id_values[0]:.6f} | {id_values[1]:.6f} | {id_values[2]:.6f} | "
            f"{policy_values[0]:.6f} | {policy_values[1]:.6f} | {policy_values[2]:.6f} |"
        )
    lines.extend(
        [
            "",
            "The composite is normalized once to checkpoint 8400, so values remain directly comparable "
            "across the full trajectory.",
        ]
    )
    if plt is not None:
        lines.extend(["", "![Checkpoint validation trajectory](checkpoint_trajectory.png)"])
    lines.append("")
    (args.output_dir / "checkpoint_trajectory_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
