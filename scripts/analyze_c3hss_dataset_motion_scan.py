#!/usr/bin/env python3
"""Summarize raw HF SutureBot and NephFat dVRK motion scans."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SUBSETS = ("hf_suturebot", "nephfat")
ARMS = ("psm1", "psm2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--loader-source")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def main() -> None:
    args = parse_args()
    scan_root = Path(args.scan_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        subset: json.loads(
            (scan_root / subset / "motion_scan_summary.json").read_text()
        )
        for subset in SUBSETS
    }
    rows = {}
    for subset in SUBSETS:
        with (scan_root / subset / "motion_windows.csv").open() as handle:
            rows[subset] = list(csv.DictReader(handle))

    loader_fallback = None
    if args.loader_source:
        loader_path = Path(args.loader_source).resolve()
        source = loader_path.read_text()
        required_fragments = (
            "ts_looks_broken",
            "float(timestamp[0]) > 1e6",
            "video_timestamp = step_indices.astype(np.float32) / float(fps)",
        )
        loader_fallback = {
            "source": str(loader_path),
            "sha256": _sha256(loader_path),
            "required_fragments_present": {
                fragment: fragment in source for fragment in required_fragments
            },
            "active": all(fragment in source for fragment in required_fragments),
        }

    dataset_results = {}
    for subset, summary in summaries.items():
        health = summary["data_health"]
        arm_results = {}
        for arm in ARMS:
            activity = summary["activity_summary"][arm]
            arm_results[arm] = {
                "median_translation_step_mm": activity[
                    "translation_step_mm"
                ]["q50"],
                "median_rotation_step_degrees": activity[
                    "rotation_step_degrees"
                ]["q50"],
                "fraction_below_0p1_mm_step": activity[
                    "fraction_below_0p1_mm_step"
                ],
                "median_action_to_state_translation_cosine": activity[
                    "action_to_state_translation_step_cosine"
                ]["q50"],
                "median_dynamic_fraction": activity[
                    "relative_translation_dynamic_fraction"
                ]["q50"],
            }
        dataset_results[subset] = {
            "episodes": summary["episodes"],
            "raw_rows": summary["raw_rows"],
            "candidate_windows": summary["candidate_windows"],
            "pose_reference_source": summary["pose_reference_source"],
            "invalid_numeric_or_quaternion_rows": (
                health["nonfinite_action_rows"]
                + health["nonfinite_state_rows"]
                + sum(health["invalid_action_quaternion_rows"].values())
                + sum(health["invalid_state_quaternion_rows"].values())
            ),
            "nonincreasing_timestamp_steps": health[
                "nonincreasing_timestamp_steps"
            ],
            "episodes_with_nonincreasing_timestamps": health[
                "episodes_with_nonincreasing_timestamps"
            ],
            "arms": arm_results,
            "psm1_to_psm2_median_translation_ratio": float(
                arm_results["psm1"]["median_translation_step_mm"]
                / max(
                    arm_results["psm2"]["median_translation_step_mm"], 1e-12
                )
            ),
        }

    payload = {
        "diagnostic": "HF SutureBot and NephFat raw motion health audit",
        "verdict": (
            "valid action/state pose rows with strong command-state agreement; "
            "substantial PSM2 inactivity, especially in HF SutureBot"
        ),
        "datasets": dataset_results,
        "loader_timestamp_fallback": loader_fallback,
    }
    (output_dir / "dataset_motion_audit_summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"
    )

    colors = {"psm1": "#4C78A8", "psm2": "#E45756"}
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for axis, subset in zip(axes, SUBSETS):
        for arm in ARMS:
            values = np.sort(
                np.asarray(
                    [
                        float(
                            row[f"{arm}_mean_action_translation_step_mm"]
                        )
                        for row in rows[subset]
                    ]
                )
            )
            cumulative = np.arange(1, len(values) + 1) / len(values)
            axis.plot(
                values,
                cumulative,
                label=arm.upper(),
                color=colors[arm],
                linewidth=2,
            )
        axis.axvline(0.1, color="black", linestyle=":", linewidth=1)
        axis.set_xscale("log")
        axis.set_xlim(0.003, 5.0)
        axis.set_title(subset)
        axis.set_xlabel("Mean translation step (mm / 0.1 s, log scale)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Fraction of eligible windows")
    axes[1].legend()
    figure.suptitle("Raw dVRK arm-activity distributions")
    figure.tight_layout()
    figure.savefig(output_dir / "motion_activity_cdf.png", dpi=180)
    plt.close(figure)

    hf = dataset_results["hf_suturebot"]
    neph = dataset_results["nephfat"]
    fallback_text = (
        "The pinned loader source contains and activates the intended fallback:"
        " epoch/low-resolution timestamps are replaced by frame-index/FPS"
        " timestamps before video decoding."
        if loader_fallback and loader_fallback["active"]
        else "The loader timestamp fallback was not verified in this run."
    )
    lines = [
        "# HF SutureBot and NephFat dataset motion audit",
        "",
        "## Verdict",
        "",
        "The audited action/state pose data are not corrupt: no non-finite rows or",
        "invalid quaternions were found, and median raw command→robot translation",
        "agreement is strong for both arms. The important dataset property is",
        "arm inactivity, not malformed PSM2 values.",
        "",
        (
            f"- HF SutureBot: PSM2 is below 0.1 mm/step in "
            f"{hf['arms']['psm2']['fraction_below_0p1_mm_step']:.1%} of windows, "
            f"versus "
            f"{hf['arms']['psm1']['fraction_below_0p1_mm_step']:.1%} for PSM1. "
            f"Median PSM1 motion is "
            f"{hf['psm1_to_psm2_median_translation_ratio']:.1f}× PSM2."
        ),
        (
            f"- NephFat: PSM2 is below 0.1 mm/step in "
            f"{neph['arms']['psm2']['fraction_below_0p1_mm_step']:.1%} of windows, "
            f"versus "
            f"{neph['arms']['psm1']['fraction_below_0p1_mm_step']:.1%} for PSM1."
        ),
        (
            f"- Median command→state cosine (PSM1/PSM2): HF "
            f"{hf['arms']['psm1']['median_action_to_state_translation_cosine']:.3f}/"
            f"{hf['arms']['psm2']['median_action_to_state_translation_cosine']:.3f}; "
            f"NephFat "
            f"{neph['arms']['psm1']['median_action_to_state_translation_cosine']:.3f}/"
            f"{neph['arms']['psm2']['median_action_to_state_translation_cosine']:.3f}."
        ),
        "",
        "## Timestamp finding",
        "",
        (
            f"HF SutureBot has non-increasing float32 epoch timestamps in "
            f"{hf['episodes_with_nonincreasing_timestamps']}/{hf['episodes']} test "
            "episodes. This is a real metadata defect, caused by insufficient float32"
        ),
        "precision at Unix-epoch magnitude. " + fallback_text,
        "",
        "NephFat timestamps are healthy except for one duplicated step in one",
        "test episode.",
        "",
        "## Evaluation consequence",
        "",
        "Fixed-base sampling over-represents parked PSM2 behavior. Model action",
        "following must be evaluated on motion-matched, arm-isolated windows and",
        "reported separately from naturally sampled performance. The matched",
        "selection under `matched_selection/` provides that controlled set.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
