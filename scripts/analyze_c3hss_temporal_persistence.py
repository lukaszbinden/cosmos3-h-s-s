#!/usr/bin/env python3
"""Summarize whether an action-binding failure persists in nearby windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiseed-summary", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--earlier-legacy-summary")
    parser.add_argument("--earlier-tight-summary")
    parser.add_argument(
        "--measurement-focus",
        help=(
            "Optional subset:arm:episode:base focus used to quantify the effect "
            "of corrected tool masks on an earlier evaluation."
        ),
    )
    return parser.parse_args()


def _classify(records: list[dict[str, Any]]) -> tuple[str, list[int]]:
    focus_position = next(
        index for index, record in enumerate(records) if record["relative_base"] == 0
    )
    failed = [float(record["mean"]) < 0.5 for record in records]
    if not failed[focus_position]:
        return "center_localizes", []

    left = focus_position
    right = focus_position
    while left > 0 and failed[left - 1]:
        left -= 1
    while right + 1 < len(failed) and failed[right + 1]:
        right += 1
    region = [int(records[index]["base_index"]) for index in range(left, right + 1)]
    if len(region) == 1:
        return "isolated_center_failure", region
    if all(failed):
        return "episode_wide_failure", region
    return "temporally_persistent_local_failure", region


def _focus_record(
    summary: dict[str, Any], focus: tuple[str, str, int, int]
) -> dict[str, Any]:
    subset, target_arm, episode_id, base_index = focus
    matches = [
        record
        for record in summary["episode_groups"]
        if record["subset"] == subset
        and record["target_arm"] == target_arm
        and int(record["episode_id"]) == episode_id
        and int(record["base_index"]) == base_index
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one summary record for {focus}; got {len(matches)}")
    return matches[0]


def main() -> None:
    args = parse_args()
    summary_path = Path(args.multiseed_summary).resolve()
    design_path = Path(args.design).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text())
    design = json.loads(design_path.read_text())

    if int(summary["iteration"]) != int(design["checkpoint_iteration"]):
        raise ValueError("Summary iteration does not match temporal design")

    measurement_args = (
        args.earlier_legacy_summary,
        args.earlier_tight_summary,
        args.measurement_focus,
    )
    if any(measurement_args) and not all(measurement_args):
        raise ValueError(
            "The two earlier summaries and --measurement-focus must be supplied together"
        )

    measured = {
        (int(record["episode_id"]), int(record["base_index"])): record
        for record in summary["episode_groups"]
    }
    episode_results = []
    expected_pairs = set()
    for episode_text, episode_design in design["episodes"].items():
        episode_id = int(episode_text)
        records = []
        for window in episode_design["windows"]:
            base_index = int(window["base_index"])
            expected_pairs.add((episode_id, base_index))
            measured_record = measured.get((episode_id, base_index))
            if measured_record is None:
                raise ValueError(f"Missing measured window {episode_id}:{base_index}")
            records.append(
                {
                    **window,
                    "mean": float(measured_record["mean"]),
                    "population_std": float(measured_record["population_std"]),
                    "min": float(measured_record["min"]),
                    "max": float(measured_record["max"]),
                    "localized_seed_count": int(
                        measured_record["localized_seed_count"]
                    ),
                    "seed_count": int(measured_record["seed_count"]),
                    "by_seed": measured_record["by_seed"],
                }
            )
        records.sort(key=lambda record: int(record["base_index"]))
        classification, failure_region = _classify(records)
        episode_results.append(
            {
                "episode_id": episode_id,
                "split_role": episode_design["split_role"],
                "focus_base": int(episode_design["focus_base"]),
                "classification": classification,
                "center_failure_region_bases": failure_region,
                "failed_window_count": sum(
                    float(record["mean"]) < 0.5 for record in records
                ),
                "systematic_failure_window_count": sum(
                    int(record["localized_seed_count"]) == 0 for record in records
                ),
                "window_count": len(records),
                "windows": records,
            }
        )
    unexpected = set(measured) - expected_pairs
    if unexpected:
        raise ValueError(f"Summary contains unexpected windows: {sorted(unexpected)}")

    classifications = {
        result["split_role"]: result["classification"] for result in episode_results
    }
    if all(value == "center_localizes" for value in classifications.values()):
        verdict = "neither_focus_failure_reproduced"
    elif all(value == "isolated_center_failure" for value in classifications.values()):
        verdict = "both_failures_are_temporally_isolated"
    elif all(
        value in {"temporally_persistent_local_failure", "episode_wide_failure"}
        for value in classifications.values()
    ):
        verdict = "both_failures_are_temporally_persistent"
    else:
        verdict = "training_and_heldout_temporal_patterns_differ"

    measurement_control = None
    if all(measurement_args):
        focus_parts = args.measurement_focus.split(":")
        if len(focus_parts) != 4:
            raise ValueError("--measurement-focus must be subset:arm:episode:base")
        focus = (
            focus_parts[0],
            focus_parts[1],
            int(focus_parts[2]),
            int(focus_parts[3]),
        )
        legacy_path = Path(args.earlier_legacy_summary).resolve()
        earlier_tight_path = Path(args.earlier_tight_summary).resolve()
        legacy_summary = json.loads(legacy_path.read_text())
        earlier_tight_summary = json.loads(earlier_tight_path.read_text())
        if int(legacy_summary["iteration"]) != int(earlier_tight_summary["iteration"]):
            raise ValueError("Earlier mask-control summaries use different iterations")
        legacy = _focus_record(legacy_summary, focus)
        earlier_tight = _focus_record(earlier_tight_summary, focus)
        current_tight = _focus_record(summary, focus)
        measurement_control = {
            "focus": {
                "subset": focus[0],
                "target_arm": focus[1],
                "episode_id": focus[2],
                "base_index": focus[3],
            },
            "earlier_iteration": int(legacy_summary["iteration"]),
            "current_iteration": int(summary["iteration"]),
            "earlier_legacy_masks": legacy,
            "earlier_tight_masks": earlier_tight,
            "current_tight_masks": current_tight,
            "mask_correction_mean_delta": (
                float(earlier_tight["mean"]) - float(legacy["mean"])
            ),
            "later_checkpoint_mean_delta": (
                float(current_tight["mean"]) - float(earlier_tight["mean"])
            ),
            "interpretation": (
                "The legacy-to-tight comparison re-scores identical generated "
                "videos and isolates measurement sensitivity. The earlier-to-current "
                "comparison is not paired because the evaluations sampled different "
                "generated outputs and spatially augmented camera crops."
            ),
            "sources": {
                "earlier_legacy_summary": str(legacy_path),
                "earlier_tight_summary": str(earlier_tight_path),
            },
        }

    payload = {
        "diagnostic": "C3-H-S-S temporal action-binding persistence",
        "model": summary["model"],
        "iteration": int(summary["iteration"]),
        "localization_threshold": 0.5,
        "verdict": verdict,
        "episodes": episode_results,
        "measurement_control": measurement_control,
        "sources": {
            "multiseed_summary": str(summary_path),
            "design": str(design_path),
        },
    }
    (output_dir / "temporal_persistence_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    fieldnames = [
        "split_role",
        "episode_id",
        "base_index",
        "relative_base",
        "progress",
        "phase",
        "mean",
        "population_std",
        "min",
        "max",
        "localized_seed_count",
        "seed_count",
    ]
    with (output_dir / "temporal_persistence_windows.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for episode in episode_results:
            for window in episode["windows"]:
                writer.writerow(
                    {
                        "split_role": episode["split_role"],
                        "episode_id": episode["episode_id"],
                        **{key: window[key] for key in fieldnames[3:]},
                    }
                )

    figure, axes = plt.subplots(
        len(episode_results),
        1,
        figsize=(8.5, 3.6 * len(episode_results)),
        sharex=True,
        squeeze=False,
    )
    for axis, episode in zip(axes[:, 0], episode_results, strict=True):
        windows = episode["windows"]
        x = [int(window["relative_base"]) for window in windows]
        y = [float(window["mean"]) for window in windows]
        yerr = [float(window["population_std"]) for window in windows]
        axis.errorbar(x, y, yerr=yerr, marker="o", capsize=4, linewidth=2)
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.axvline(0, color="gray", linestyle=":", linewidth=1)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Intended-tool response")
        axis.set_title(
            f"{episode['split_role'].replace('_', ' ').title()} episode "
            f"{episode['episode_id']}: {episode['classification']}"
        )
        for relative_base, mean, window in zip(x, y, windows, strict=True):
            axis.annotate(
                f"{window['localized_seed_count']}/{window['seed_count']}",
                (relative_base, mean),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axes[-1, 0].set_xlabel("Raw-frame offset from focus (36 frames ≈ 1.2 s)")
    figure.suptitle("Temporal persistence of PSM1 intervention localization")
    figure.tight_layout()
    figure.savefig(output_dir / "temporal_persistence.png", dpi=180)
    plt.close(figure)

    lines = [
        "# Temporal action-binding persistence",
        "",
        f"Verdict: `{verdict}`.",
        "",
        (
            "Response fractions above 0.5 mean translation interventions changed "
            "the commanded PSM1 more than the other visible tool. Error bars in the "
            "plot are population standard deviations across diffusion seeds."
        ),
    ]
    for episode in episode_results:
        lines.extend(
            [
                "",
                (
                    f"## {episode['split_role'].replace('_', ' ').title()} episode "
                    f"{episode['episode_id']}"
                ),
                "",
                f"Classification: `{episode['classification']}`.",
                "",
                "| Base | Relative | Progress | Phase | Mean ± std | Localized seeds |",
                "|---:|---:|---:|---|---:|---:|",
            ]
        )
        for window in episode["windows"]:
            lines.append(
                f"| {window['base_index']} | {window['relative_base']:+d} | "
                f"{window['progress']:.3f} | {window['phase']} | "
                f"{window['mean']:.3f} ± {window['population_std']:.3f} | "
                f"{window['localized_seed_count']}/{window['seed_count']} |"
            )
    if measurement_control is not None:
        legacy = measurement_control["earlier_legacy_masks"]
        earlier_tight = measurement_control["earlier_tight_masks"]
        current_tight = measurement_control["current_tight_masks"]
        focus = measurement_control["focus"]
        lines.extend(
            [
                "",
                "## Measurement control",
                "",
                (
                    f"Focus: `{focus['subset']}:{focus['target_arm']}:"
                    f"{focus['episode_id']}:{focus['base_index']}`."
                ),
                "",
                "| Evaluation | Mean ± std | Localized seeds |",
                "|---|---:|---:|",
                (
                    f"| Iteration {measurement_control['earlier_iteration']}, "
                    f"legacy masks | {legacy['mean']:.3f} ± "
                    f"{legacy['population_std']:.3f} | "
                    f"{legacy['localized_seed_count']}/{legacy['seed_count']} |"
                ),
                (
                    f"| Iteration {measurement_control['earlier_iteration']}, "
                    f"tight jaw masks | {earlier_tight['mean']:.3f} ± "
                    f"{earlier_tight['population_std']:.3f} | "
                    f"{earlier_tight['localized_seed_count']}/"
                    f"{earlier_tight['seed_count']} |"
                ),
                (
                    f"| Iteration {measurement_control['current_iteration']}, "
                    f"tight jaw masks | {current_tight['mean']:.3f} ± "
                    f"{current_tight['population_std']:.3f} | "
                    f"{current_tight['localized_seed_count']}/"
                    f"{current_tight['seed_count']} |"
                ),
                "",
                (
                    "Re-scoring the identical iteration-"
                    f"{measurement_control['earlier_iteration']} videos with tight "
                    "jaw masks changes the mean by "
                    f"{measurement_control['mask_correction_mean_delta']:+.3f}. "
                    "That directly identifies the earlier failure as a segmentation/"
                    "ROI confound. The cross-checkpoint difference is not a paired "
                    "estimate because those evaluations used different generated "
                    "samples and spatially augmented camera crops."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "The windows are adjacent in time within each episode and deliberately "
                "share visual context. This diagnoses temporal persistence around two "
                "known failures; it is not an unbiased estimate over the dataset."
            ),
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
