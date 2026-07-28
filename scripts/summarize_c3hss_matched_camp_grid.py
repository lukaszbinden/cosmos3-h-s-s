#!/usr/bin/env python3
"""Summarize paired fidelity and action-sensitivity metrics for the CAMP grid."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

CONDITIONS = (
    "arm_a_h0",
    "arm_b_history_real",
    "arm_b_history_zero",
    "arm_b_history_permute",
    "arm_c_real",
    "arm_c_memory_zero",
    "arm_c_memory_shuffle",
    "arm_c_history_zero",
    "arm_c_history_permute",
)
COMPARISONS = (
    ("arm_b_history_real", "arm_a_h0", "raw_history_vs_h0"),
    ("arm_c_real", "arm_b_history_real", "learned_memory_vs_raw_history"),
    ("arm_b_history_zero", "arm_b_history_real", "arm_b_zero_history"),
    ("arm_b_history_permute", "arm_b_history_real", "arm_b_permuted_history"),
    ("arm_c_memory_zero", "arm_c_real", "arm_c_zero_memory"),
    ("arm_c_memory_shuffle", "arm_c_real", "arm_c_shuffled_memory"),
    ("arm_c_history_zero", "arm_c_real", "arm_c_zero_history"),
    ("arm_c_history_permute", "arm_c_real", "arm_c_permuted_history"),
)


def _load(root: Path, condition: str) -> dict[tuple, dict[str, float]]:
    records: dict[tuple, dict[str, float]] = {}
    condition_root = root / condition
    for path in sorted(
        condition_root.glob("raw/**/*_action_intervention_episode.json")
    ):
        relative = path.relative_to(condition_root)
        subset, target = relative.parts[1:3]
        manifest = json.loads(path.read_text())
        identity = (
            subset,
            target,
            int(manifest["episode_id"]),
            int(manifest["base_index"]),
            int(manifest["sampling_seed"]),
        )
        variants = {item["name"]: item for item in manifest["variants"]}
        correct = variants["correct"]
        response_values = [
            float(item["paired_output_delta_from_correct_motion_roi_l1"])
            for name, item in variants.items()
            if name != "correct"
        ]
        ratio = max(float(correct["generated_to_gt_motion_energy_ratio"]), 1e-8)
        records[identity] = {
            "mean_l1": float(correct["fds"]["mean_l1"]),
            "mean_ssim": float(correct["fds"]["mean_ssim"]),
            "motion_roi_l1": float(correct["motion_roi_l1"]),
            "motion_roi_endpoint_l1": float(correct["motion_roi_endpoint_l1"]),
            "motion_energy_abs_log_ratio": abs(math.log(ratio)),
            "action_response_sensitivity": float(np.mean(response_values)),
        }
    return records


def _window_means(
    records: dict[tuple, dict[str, float]], metric: str
) -> dict[tuple, float]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for identity, values in records.items():
        grouped[identity[:-1]].append(values[metric])
    return {identity: float(np.mean(values)) for identity, values in grouped.items()}


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    seed: int = 950,
    draws: int = 20_000,
) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    samples = np.asarray([statistic(values[index]) for index in indices])
    return [float(item) for item in np.quantile(samples, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    loaded = {condition: _load(args.root, condition) for condition in CONDITIONS}
    expected = set(loaded[CONDITIONS[0]])
    for condition, records in loaded.items():
        if set(records) != expected:
            raise ValueError(f"{condition} does not contain the matched identity set")
    if len(expected) != 60:
        raise ValueError(f"Expected 60 records per condition, found {len(expected)}")

    metrics = tuple(next(iter(next(iter(loaded.values())).values())).keys())
    condition_summaries: dict[str, Any] = {}
    for condition, records in loaded.items():
        condition_summaries[condition] = {
            metric: _summary([record[metric] for record in records.values()])
            for metric in metrics
        }

    comparisons = []
    for candidate, reference, label in COMPARISONS:
        metric_deltas = {}
        for metric in metrics:
            candidate_windows = _window_means(loaded[candidate], metric)
            reference_windows = _window_means(loaded[reference], metric)
            identities = sorted(candidate_windows)
            deltas = np.asarray(
                [
                    candidate_windows[identity] - reference_windows[identity]
                    for identity in identities
                ],
                dtype=np.float64,
            )
            metric_deltas[metric] = {
                "candidate_minus_reference_mean": float(deltas.mean()),
                "window_cluster_bootstrap_95_ci": _bootstrap_ci(deltas),
                "candidate_better_window_count": int(
                    np.sum(
                        deltas > 0
                        if metric in {"mean_ssim", "action_response_sensitivity"}
                        else deltas < 0
                    )
                ),
                "window_count": len(deltas),
                "direction": (
                    "higher_is_better_or_more_sensitive"
                    if metric in {"mean_ssim", "action_response_sensitivity"}
                    else "lower_is_better"
                ),
            }
        comparisons.append(
            {
                "label": label,
                "candidate": candidate,
                "reference": reference,
                "metrics": metric_deltas,
            }
        )

    payload = {
        "diagnostic": "matched CAMP grid aggregate",
        "comparison_step": 950,
        "records_per_condition": len(expected),
        "independent_window_clusters": len({identity[:-1] for identity in expected}),
        "diffusion_seeds": sorted({identity[-1] for identity in expected}),
        "condition_summaries": condition_summaries,
        "paired_comparisons": comparisons,
        "interpretation_note": (
            "Full-frame and motion-ROI fidelity measure correct-prediction quality. "
            "Action-response sensitivity measures output change under counterfactual "
            "commands, not whether the intended tool moved; use the shared-mask SAM "
            "analysis for tool-localized causality."
        ),
    }
    output = args.output or args.root / "matched_grid_metric_summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
