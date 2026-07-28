#!/usr/bin/env python3
"""Aggregate a matched multi-condition C3-H-S-S autoregressive sweep.

The episode, action windows, diffusion seed, and rollout horizon must match
across conditions. Statistical units are episodes, never frames.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = (
    "arm_a_h0",
    "arm_b_history_real",
    "arm_b_history_zero",
    "arm_c_real",
    "arm_c_memory_zero",
)
CONDITION_LABELS = {
    "arm_a_h0": "H0 parent",
    "arm_b_history_real": "H16 real",
    "arm_b_history_zero": "H16 history-zero",
    "arm_c_real": "CAMP real",
    "arm_c_memory_zero": "CAMP memory-zero",
}
COMPARISONS = {
    "h16_vs_h0": ("arm_b_history_real", "arm_a_h0"),
    "h16_real_vs_history_zero": ("arm_b_history_real", "arm_b_history_zero"),
    "camp_vs_h16": ("arm_c_real", "arm_b_history_real"),
    "camp_real_vs_memory_zero": ("arm_c_real", "arm_c_memory_zero"),
}
LOWER_IS_BETTER = {
    "mean_l1",
    "l1_slope",
    "endpoint_l1",
    "early_l1",
    "late_l1",
    "l1_drift",
}
METRICS = (
    "mean_l1",
    "mean_ssim",
    "l1_slope",
    "endpoint_l1",
    "endpoint_ssim",
    "early_l1",
    "late_l1",
    "l1_drift",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def _metrics(result: dict[str, Any]) -> dict[str, float]:
    fds = result["fds"]
    l1 = np.asarray(fds["l1_per_frame"], dtype=np.float64)
    ssim = np.asarray(fds["ssim_per_frame"], dtype=np.float64)
    if l1.shape != (120,) or ssim.shape != (120,):
        raise ValueError(
            f"expected 120 scored frames for episode {result['episode_id']}, "
            f"got L1={l1.shape}, SSIM={ssim.shape}"
        )
    early = slice(0, 36)
    late = slice(84, 120)
    early_l1 = float(l1[early].mean())
    late_l1 = float(l1[late].mean())
    return {
        "mean_l1": float(fds["mean_l1"]),
        "mean_ssim": float(fds["mean_ssim"]),
        "l1_slope": float(fds["l1_slope"]),
        "endpoint_l1": float(l1[-1]),
        "endpoint_ssim": float(ssim[-1]),
        "early_l1": early_l1,
        "late_l1": late_l1,
        "l1_drift": late_l1 - early_l1,
    }


def _load(root: Path) -> dict[str, dict[tuple[str, int], dict[str, float]]]:
    records: dict[str, dict[tuple[str, int], dict[str, float]]] = {}
    for condition in CONDITIONS:
        condition_records: dict[tuple[str, int], dict[str, float]] = {}
        result_paths = sorted((root / condition / "raw").glob("*/c3hss_results.json"))
        if len(result_paths) != 3:
            raise ValueError(
                f"{condition}: expected three dataset result files, found "
                f"{len(result_paths)}"
            )
        for result_path in result_paths:
            with result_path.open() as handle:
                payload = json.load(handle)
            if payload["rollout_conditioning"] != "autoregressive":
                raise ValueError(f"{result_path} is not autoregressive")
            if payload["max_chunks"] != 10 or payload["fps"] != 10:
                raise ValueError(f"{result_path} has the wrong horizon or FPS")
            subset = result_path.parent.name
            for result in payload["results"]:
                if result["num_frames"] != 121:
                    raise ValueError(
                        f"{condition}/{subset}/{result['episode_id']} has "
                        f"{result['num_frames']} frames"
                    )
                key = (subset, int(result["episode_id"]))
                if key in condition_records:
                    raise ValueError(f"duplicate record: {condition}/{key}")
                condition_records[key] = _metrics(result)
        if len(condition_records) != 10:
            raise ValueError(
                f"{condition}: expected ten episode records, found "
                f"{len(condition_records)}"
            )
        records[condition] = condition_records

    identities = set(records[CONDITIONS[0]])
    for condition in CONDITIONS[1:]:
        if set(records[condition]) != identities:
            raise ValueError(f"{condition}: episode identities do not match")
    return records


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _paired_effect(
    left: np.ndarray,
    right: np.ndarray,
    metric: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    # Positive always means the named left condition is better.
    effects = right - left if metric in LOWER_IS_BETTER else left - right
    sample_indices = rng.integers(
        0, len(effects), size=(bootstrap_samples, len(effects))
    )
    boot_means = effects[sample_indices].mean(axis=1)

    observed = float(effects.mean())
    sign_means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(effects)):
        sign_means.append(float((effects * np.asarray(signs)).mean()))
    sign_means_array = np.asarray(sign_means)
    p_value = float(
        (np.abs(sign_means_array) >= abs(observed) - 1e-15).mean()
    )
    tolerance = 1e-12
    return {
        "effect_definition": (
            "right_minus_left" if metric in LOWER_IS_BETTER else "left_minus_right"
        ),
        "positive_means_left_is_better": True,
        "mean_effect": observed,
        "bootstrap_95_ci": [
            float(np.quantile(boot_means, 0.025)),
            float(np.quantile(boot_means, 0.975)),
        ],
        "exact_paired_sign_flip_p_two_sided": p_value,
        "left_wins": int(np.sum(effects > tolerance)),
        "right_wins": int(np.sum(effects < -tolerance)),
        "ties": int(np.sum(np.abs(effects) <= tolerance)),
        "per_episode_effects": [float(value) for value in effects],
    }


def main() -> None:
    args = parse_args()
    records = _load(args.root)
    identities = sorted(records[CONDITIONS[0]])
    rng = np.random.default_rng(args.seed)

    condition_summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        overall = {}
        by_subset = {}
        for metric in METRICS:
            values = np.asarray(
                [records[condition][identity][metric] for identity in identities]
            )
            overall[metric] = _summary(values)
        for subset in sorted({identity[0] for identity in identities}):
            subset_ids = [identity for identity in identities if identity[0] == subset]
            by_subset[subset] = {
                metric: _summary(
                    np.asarray(
                        [
                            records[condition][identity][metric]
                            for identity in subset_ids
                        ]
                    )
                )
                for metric in METRICS
            }
        condition_summary[condition] = {
            "label": CONDITION_LABELS[condition],
            "overall": overall,
            "by_subset": by_subset,
        }

    comparisons = {}
    for name, (left_condition, right_condition) in COMPARISONS.items():
        metrics = {}
        for metric in METRICS:
            left = np.asarray(
                [records[left_condition][identity][metric] for identity in identities]
            )
            right = np.asarray(
                [records[right_condition][identity][metric] for identity in identities]
            )
            metrics[metric] = _paired_effect(
                left, right, metric, rng, args.bootstrap_samples
            )
        comparisons[name] = {
            "left": left_condition,
            "right": right_condition,
            "metrics": metrics,
        }

    payload = {
        "analysis": "matched 10-episode 12.1-second autoregressive sweep",
        "statistical_unit": "episode",
        "num_episodes": len(identities),
        "identities": [
            {"subset": subset, "episode_id": episode_id}
            for subset, episode_id in identities
        ],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "condition_summary": condition_summary,
        "paired_comparisons": comparisons,
        "limitations": [
            "Ten episodes provide paired variance estimates but remain a small sample.",
            "Every episode uses one diffusion seed.",
            "Metrics measure whole-frame fidelity, not tool-localized action response.",
            "Visual state is autoregressive; actions, raw history, and memory codes come from the recorded episode.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    lines = [
        "# C3-H-S-S CAMP 12.1-second autoregressive sweep",
        "",
        "Ten matched held-out episodes; the statistical unit is the episode.",
        "Positive paired effects mean the left condition is better.",
        "",
        "## Condition means",
        "",
        "| Condition | Mean L1 ↓ | Mean SSIM ↑ | Late L1 ↓ | L1 drift ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        overall = condition_summary[condition]["overall"]
        lines.append(
            f"| {CONDITION_LABELS[condition]} "
            f"| {overall['mean_l1']['mean']:.6f} "
            f"| {overall['mean_ssim']['mean']:.6f} "
            f"| {overall['late_l1']['mean']:.6f} "
            f"| {overall['l1_drift']['mean']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired effects",
            "",
            "| Comparison | Metric | Mean effect | 95% bootstrap CI | Wins | Exact p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for comparison_name in COMPARISONS:
        comparison = comparisons[comparison_name]
        label = (
            f"{CONDITION_LABELS[comparison['left']]} vs "
            f"{CONDITION_LABELS[comparison['right']]}"
        )
        for metric in ("mean_l1", "mean_ssim", "late_l1", "l1_drift"):
            result = comparison["metrics"][metric]
            lower, upper = result["bootstrap_95_ci"]
            lines.append(
                f"| {label} | {metric} | {result['mean_effect']:+.6f} "
                f"| [{lower:+.6f}, {upper:+.6f}] "
                f"| {result['left_wins']}/{len(identities)} "
                f"| {result['exact_paired_sign_flip_p_two_sided']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- One diffusion seed per episode.",
            "- Whole-frame fidelity is not a tool-localized action-following metric.",
            "- Actions, history, and memory are recorded; only visual state is fed back autoregressively.",
            "",
        ]
    )
    args.output_markdown.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
