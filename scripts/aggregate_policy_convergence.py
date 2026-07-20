#!/usr/bin/env python3
"""Aggregate paired joint-policy metrics across operative CMR subsets and seeds.

The headline composite normalizes robot-action MSE and rollout-video L1 by
their checkpoint-before means, then combines them with configurable weights.
Raw components remain first-class outputs so the composite is interpretable.
"""

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


T_CRITICAL_DF2_95PCT = 4.3026527297
ACTION_METRIC = "mean_mse_robot_30d"
VIDEO_METRIC = "mean_l1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--episodes-per-subset", type=int, default=5)
    parser.add_argument("--action-weight", type=float, default=2.0 / 3.0)
    return parser.parse_args()


def _load(root: Path, subset: str, seed: int) -> dict[str, Any]:
    path = root / subset / f"seed{seed}" / "c3hss_policy_results.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload["inference_mode"] != "policy":
        raise RuntimeError(f"Expected policy payload in {path}, got {payload['inference_mode']!r}")
    if int(payload["seed"]) != seed:
        raise RuntimeError(f"Seed mismatch in {path}: {payload['seed']} != {seed}")
    if payload.get("dataset_name", Path(payload["dataset"]).name) != subset:
        raise RuntimeError(f"Dataset mismatch in {path}")
    return payload


def _checkpoint_iteration(payload: dict[str, Any]) -> int:
    # Weight-only inference intentionally does not restore trainer state.
    return int(Path(payload["checkpoint"]).name.removeprefix("iter_"))


def _paired_stats(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    if before.shape != after.shape or before.ndim != 1:
        raise ValueError(f"Invalid paired arrays: {before.shape} vs {after.shape}")
    delta = after - before
    mean = float(delta.mean())
    if delta.size > 1:
        sd = float(delta.std(ddof=1))
        half_width = T_CRITICAL_DF2_95PCT * sd / math.sqrt(delta.size)
    else:
        sd = float("nan")
        half_width = float("nan")
    before_mean = float(before.mean())
    after_mean = float(after.mean())
    return {
        "before_seed_values": before.tolist(),
        "after_seed_values": after.tolist(),
        "before_mean": before_mean,
        "after_mean": after_mean,
        "relative_change_pct": 100.0 * (after_mean / before_mean - 1.0),
        "paired_delta_mean": mean,
        "paired_delta_sd": sd,
        "paired_delta_95pct_t_interval": [mean - half_width, mean + half_width],
    }


def _episode_ids(payload: dict[str, Any]) -> list[int]:
    return [int(result["episode_id"]) for result in payload["results"]]


def _subset_scalar(payload: dict[str, Any], metric: str) -> float:
    if metric == VIDEO_METRIC:
        values = [float(result["fds"][metric]) for result in payload["results"]]
    else:
        values = [float(result[metric]) for result in payload["results"]]
    return float(np.mean(values))


def _balanced_seed_values(
    payloads: dict[tuple[int, str], dict[str, Any]],
    seeds: list[int],
    subsets: list[str],
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [
            np.mean([_subset_scalar(payloads[(seed, subset)], metric) for subset in subsets])
            for seed in seeds
        ],
        dtype=np.float64,
    )


def _video_curve(payload: dict[str, Any]) -> np.ndarray:
    curves = np.asarray(
        [result["fds"]["l1_per_frame"] for result in payload["results"]], dtype=np.float64
    )
    if curves.ndim != 2:
        raise RuntimeError(f"Invalid policy video curve shape: {curves.shape}")
    return curves.mean(axis=0)


def _action_chunk_curve(payload: dict[str, Any]) -> np.ndarray:
    curves = np.asarray(
        [
            [float(window["mse_robot_30d"]) for window in result["action_windows"]]
            for result in payload["results"]
        ],
        dtype=np.float64,
    )
    if curves.ndim != 2:
        raise RuntimeError(f"Invalid policy action curve shape: {curves.shape}")
    return curves.mean(axis=0)


def _balanced_curves(
    payloads: dict[tuple[int, str], dict[str, Any]],
    seeds: list[int],
    subsets: list[str],
    curve_fn: Any,
) -> np.ndarray:
    return np.asarray(
        [np.mean([curve_fn(payloads[(seed, subset)]) for subset in subsets], axis=0) for seed in seeds]
    )


def _composite(
    action: np.ndarray,
    video: np.ndarray,
    action_baseline: float,
    video_baseline: float,
    action_weight: float,
) -> np.ndarray:
    return action_weight * action / action_baseline + (1.0 - action_weight) * video / video_baseline


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.action_weight <= 1.0:
        raise ValueError("--action-weight must be in [0, 1]")
    if len(args.seeds) != 3:
        raise ValueError("This paired protocol requires exactly three seeds")

    before: dict[tuple[int, str], dict[str, Any]] = {}
    after: dict[tuple[int, str], dict[str, Any]] = {}
    episode_ids: dict[str, list[int]] = {}
    for seed in args.seeds:
        for subset in args.subsets:
            before_payload = _load(args.before_root, subset, seed)
            after_payload = _load(args.after_root, subset, seed)
            before[(seed, subset)] = before_payload
            after[(seed, subset)] = after_payload
            before_ids = _episode_ids(before_payload)
            after_ids = _episode_ids(after_payload)
            if len(before_ids) != args.episodes_per_subset:
                raise RuntimeError(
                    f"{subset}/seed{seed}: expected {args.episodes_per_subset} episodes, got {len(before_ids)}"
                )
            if before_ids != after_ids:
                raise RuntimeError(f"Checkpoint episode mismatch for {subset}/seed{seed}")
            if subset not in episode_ids:
                episode_ids[subset] = before_ids
            elif episode_ids[subset] != before_ids:
                raise RuntimeError(f"Seed episode mismatch for {subset}")

    first_before = before[(args.seeds[0], args.subsets[0])]
    first_after = after[(args.seeds[0], args.subsets[0])]
    before_iteration = _checkpoint_iteration(first_before)
    after_iteration = _checkpoint_iteration(first_after)

    action_before = _balanced_seed_values(before, args.seeds, args.subsets, ACTION_METRIC)
    action_after = _balanced_seed_values(after, args.seeds, args.subsets, ACTION_METRIC)
    video_before = _balanced_seed_values(before, args.seeds, args.subsets, VIDEO_METRIC)
    video_after = _balanced_seed_values(after, args.seeds, args.subsets, VIDEO_METRIC)
    action_baseline = float(action_before.mean())
    video_baseline = float(video_before.mean())

    composite_before = _composite(
        action_before, video_before, action_baseline, video_baseline, args.action_weight
    )
    composite_after = _composite(
        action_after, video_after, action_baseline, video_baseline, args.action_weight
    )
    equal_before = _composite(action_before, video_before, action_baseline, video_baseline, 0.5)
    equal_after = _composite(action_after, video_after, action_baseline, video_baseline, 0.5)

    scalar_summary = {
        "robot_action_mse_30d": _paired_stats(action_before, action_after),
        "rollout_video_l1": _paired_stats(video_before, video_after),
        "normalized_composite_action_2_video_1": _paired_stats(composite_before, composite_after),
        "normalized_composite_equal_weight_sensitivity": _paired_stats(equal_before, equal_after),
    }

    per_subset: dict[str, Any] = {}
    for subset in args.subsets:
        b_action = np.asarray(
            [_subset_scalar(before[(seed, subset)], ACTION_METRIC) for seed in args.seeds]
        )
        a_action = np.asarray(
            [_subset_scalar(after[(seed, subset)], ACTION_METRIC) for seed in args.seeds]
        )
        b_video = np.asarray(
            [_subset_scalar(before[(seed, subset)], VIDEO_METRIC) for seed in args.seeds]
        )
        a_video = np.asarray(
            [_subset_scalar(after[(seed, subset)], VIDEO_METRIC) for seed in args.seeds]
        )
        b_composite = _composite(
            b_action, b_video, float(b_action.mean()), float(b_video.mean()), args.action_weight
        )
        a_composite = _composite(
            a_action, a_video, float(b_action.mean()), float(b_video.mean()), args.action_weight
        )
        per_subset[subset] = {
            "robot_action_mse_30d": _paired_stats(b_action, a_action),
            "rollout_video_l1": _paired_stats(b_video, a_video),
            "normalized_composite": _paired_stats(b_composite, a_composite),
        }

    before_video_curves = _balanced_curves(before, args.seeds, args.subsets, _video_curve)
    after_video_curves = _balanced_curves(after, args.seeds, args.subsets, _video_curve)
    before_action_curves = _balanced_curves(before, args.seeds, args.subsets, _action_chunk_curve)
    after_action_curves = _balanced_curves(after, args.seeds, args.subsets, _action_chunk_curve)

    summary = {
        "protocol": (
            f"Closed-loop policy inference for {args.episodes_per_subset} held-out episodes from each "
            "of all four operative CMR subsets. Each seed is balanced across subsets; uncertainty and "
            "paired intervals use the same three diffusion seeds."
        ),
        "composite_definition": {
            "formula": (
                f"{args.action_weight:.6f} * (robot_action_mse_30d / before_mean_action_mse) + "
                f"{1.0 - args.action_weight:.6f} * (rollout_video_l1 / before_mean_video_l1)"
            ),
            "lower_is_better": True,
            "normalization_reference": f"checkpoint {before_iteration} aggregate across the same validation set",
            "action_weight": args.action_weight,
            "video_weight": 1.0 - args.action_weight,
            "before_mean_action_mse": action_baseline,
            "before_mean_video_l1": video_baseline,
            "caution": "This is a protocol-relative composite, not an absolute metric transferable to another dataset.",
        },
        "before_checkpoint": first_before["checkpoint"],
        "after_checkpoint": first_after["checkpoint"],
        "before_iteration": before_iteration,
        "after_iteration": after_iteration,
        "subsets": args.subsets,
        "seeds": args.seeds,
        "episodes_per_subset_per_seed": args.episodes_per_subset,
        "episodes_per_seed": args.episodes_per_subset * len(args.subsets),
        "episode_ids": episode_ids,
        "predicted_video_frames": int(before_video_curves.shape[1]),
        "policy_chunks": int(before_action_curves.shape[1]),
        "metrics": scalar_summary,
        "per_subset": per_subset,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "policy_convergence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "policy_convergence_curves.npz",
        seeds=np.asarray(args.seeds),
        before_video_l1=before_video_curves,
        after_video_l1=after_video_curves,
        before_action_mse_by_chunk=before_action_curves,
        after_action_mse_by_chunk=after_action_curves,
        before_composite=composite_before,
        after_composite=composite_after,
    )

    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
        series = (
            (before_iteration, before_video_curves, before_action_curves, composite_before, "#4c78a8"),
            (after_iteration, after_video_curves, after_action_curves, composite_after, "#e45756"),
        )
        for iteration, video_curves, action_curves, composite, color in series:
            label = f"Policy @ {iteration}"
            video_mean = video_curves.mean(axis=0)
            video_sd = video_curves.std(axis=0, ddof=1)
            frame_x = np.arange(1, video_mean.size + 1)
            axes[0].plot(frame_x, video_mean, label=label, color=color)
            axes[0].fill_between(frame_x, video_mean - video_sd, video_mean + video_sd, color=color, alpha=0.18)
            action_mean = action_curves.mean(axis=0)
            action_sd = action_curves.std(axis=0, ddof=1)
            chunk_x = np.arange(1, action_mean.size + 1)
            axes[1].plot(chunk_x, action_mean, label=label, color=color, marker="o", markersize=3)
            axes[1].fill_between(
                chunk_x, action_mean - action_sd, action_mean + action_sd, color=color, alpha=0.18
            )
            position = 0 if iteration == before_iteration else 1
            axes[2].bar(position, composite.mean(), color=color, alpha=0.75, width=0.55)
            axes[2].errorbar(
                position,
                composite.mean(),
                yerr=composite.std(ddof=1),
                fmt="none",
                ecolor="black",
                capsize=4,
            )
            axes[2].scatter(
                np.full(composite.shape, position) + np.linspace(-0.08, 0.08, composite.size),
                composite,
                color="black",
                s=20,
                zorder=3,
            )
        axes[0].set_title("Closed-loop video L1")
        axes[0].set_xlabel("Predicted frame")
        axes[0].set_ylabel("L1 error")
        axes[1].set_title("Policy robot-action MSE")
        axes[1].set_xlabel("Rollout chunk")
        axes[1].set_ylabel("Normalized 30-D MSE")
        axes[2].set_title("Normalized joint policy metric")
        axes[2].set_xticks([0, 1], [str(before_iteration), str(after_iteration)])
        axes[2].set_xlabel("Checkpoint")
        axes[2].set_ylabel("2/3 action + 1/3 video (lower is better)")
        for ax in axes:
            ax.grid(alpha=0.25)
        axes[0].legend()
        axes[1].legend()
        fig.suptitle(
            "C3-H-S-S policy evaluation across all four operative CMR subsets\n"
            f"{args.episodes_per_subset} episodes/subset × 3 seeds; bands/error bars are ±1 seed SD"
        )
        fig.tight_layout()
        fig.savefig(args.output_dir / "policy_convergence.png", dpi=180)
        plt.close(fig)

    def metric_row(label: str, key: str) -> str:
        item = scalar_summary[key]
        low, high = item["paired_delta_95pct_t_interval"]
        return (
            f"| {label} | {item['before_mean']:.6f} | {item['after_mean']:.6f} | "
            f"{item['paired_delta_mean']:+.6f} | {item['relative_change_pct']:+.2f}% | "
            f"[{low:+.6f}, {high:+.6f}] |"
        )

    lines = [
        "# C3-H-S-S joint policy convergence",
        "",
        f"Paired closed-loop comparison of checkpoint `{before_iteration}` and checkpoint `{after_iteration}`.",
        f"Each seed averages {args.episodes_per_subset} held-out episodes from each of all four operative CMR subsets; uncertainty is across the same three seeds.",
        "",
        "The primary composite first normalizes each component by its checkpoint-"
        f"{before_iteration} mean, then uses **2/3 robot-action MSE + 1/3 rollout-video L1**. Lower is better. Raw components are shown to keep the score interpretable.",
        "",
        "| Metric | Before | After | Paired delta | Relative change | 95% t interval (n=3) |",
        "|---|---:|---:|---:|---:|---:|",
        metric_row("Robot-action MSE (normalized 30D)", "robot_action_mse_30d"),
        metric_row("Closed-loop rollout-video L1", "rollout_video_l1"),
        metric_row("Primary normalized composite (2:1)", "normalized_composite_action_2_video_1"),
        metric_row("Equal-weight composite sensitivity", "normalized_composite_equal_weight_sensitivity"),
        "",
        "| Operative CMR subset | Action MSE before | Action MSE after | Video L1 before | Video L1 after | Composite before | Composite after |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for subset in args.subsets:
        item = per_subset[subset]
        lines.append(
            f"| {subset} | {item['robot_action_mse_30d']['before_mean']:.6f} | "
            f"{item['robot_action_mse_30d']['after_mean']:.6f} | "
            f"{item['rollout_video_l1']['before_mean']:.6f} | "
            f"{item['rollout_video_l1']['after_mean']:.6f} | "
            f"{item['normalized_composite']['before_mean']:.6f} | "
            f"{item['normalized_composite']['after_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The composite is relative to this fixed validation protocol and baseline checkpoint; it should not be compared numerically with composites normalized on another dataset or baseline.",
        ]
    )
    if plt is not None:
        lines.extend(["", "![Joint policy convergence](policy_convergence.png)"])
    lines.append("")
    (args.output_dir / "policy_convergence_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
