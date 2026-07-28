#!/usr/bin/env python3
"""Analyze full-split dVRK coupling scans and rank a held-out failure window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

DATASETS = ("hf_suturebot", "nephfat")
ARMS = ("psm1", "psm2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--nephfat-dir", required=True)
    parser.add_argument("--mapping-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--focus-episode", type=int, default=1382)
    parser.add_argument("--focus-base", type=int, default=381)
    parser.add_argument("--nearest-neighbors", type=int, default=20)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _finite_percentile(values: pd.Series, score: float) -> float | None:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array) or not np.isfinite(score):
        return None
    return float(percentileofscore(array, score, kind="mean"))


def _record(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _lag_value(metric: dict[str, Any], lag: int = 0) -> float:
    return float(
        next(
            item["value"]
            for item in metric["by_lag"]
            if int(item["lag_model_frames"]) == lag
        )
    )


def _best(metric: dict[str, Any]) -> tuple[int, float]:
    item = metric["best"]
    return int(item["lag_model_frames"]), float(item["value"])


def _group_summary(frame: pd.DataFrame) -> dict[str, Any]:
    result = {
        "windows": int(len(frame)),
        "episodes": int(frame["episode_id"].nunique()),
    }
    for arm in ARMS:
        for column in (
            f"{arm}_mean_action_translation_step_mm",
            f"{arm}_mean_state_translation_step_mm",
            f"{arm}_same_arm_vector_cosine",
            f"{arm}_cross_arm_vector_cosine",
        ):
            values = pd.to_numeric(frame[column], errors="coerce")
            result[column] = {
                "median": float(values.median()),
                "q10": float(values.quantile(0.10)),
                "q90": float(values.quantile(0.90)),
            }
    if "endoscope_flow_magnitude_mean" in frame:
        for column in (
            "endoscope_flow_magnitude_mean",
            "endoscope_flow_per_combined_state_mm",
            "endoscope_psm1_state_speed_correlation",
            "endoscope_psm2_state_speed_correlation",
        ):
            values = pd.to_numeric(frame[column], errors="coerce")
            result[column] = {
                "median": float(values.median()),
                "q10": float(values.quantile(0.10)),
                "q90": float(values.quantile(0.90)),
            }
    return result


def _nearest_neighbors(
    pool: pd.DataFrame,
    focus: pd.Series,
    count: int,
) -> list[dict[str, Any]]:
    features = [
        "progress",
        "psm1_mean_action_translation_step_mm",
        "psm2_mean_action_translation_step_mm",
        "psm1_same_arm_vector_cosine",
        "psm2_same_arm_vector_cosine",
        "psm1_same_arm_speed_correlation",
        "psm2_same_arm_speed_correlation",
        "psm1_psm2_action_speed_correlation",
        "endoscope_flow_magnitude_mean",
        "endoscope_flow_per_combined_state_mm",
        "endoscope_psm1_state_speed_correlation",
        "endoscope_psm2_state_speed_correlation",
    ]
    features = [column for column in features if column in pool]
    matrix = pool[features].apply(pd.to_numeric, errors="coerce")
    medians = matrix.median()
    matrix = matrix.fillna(medians)
    centre = matrix.median()
    scale = matrix.quantile(0.75) - matrix.quantile(0.25)
    scale = scale.mask(scale.abs() < 1e-12, 1.0)
    focus_vector = pd.to_numeric(focus[features], errors="coerce").fillna(medians)
    standardized = (matrix - centre) / scale
    focus_standardized = (focus_vector - centre) / scale
    distances = np.sqrt(
        np.mean((standardized.to_numpy() - focus_standardized.to_numpy()) ** 2, axis=1)
    )
    ranked = pool.copy()
    ranked["robust_feature_distance"] = distances
    records = []
    keep = [
        "subset",
        "episode_id",
        "base_index",
        "task",
        "phase",
        "dominant_arm",
        "robust_feature_distance",
        *features,
    ]
    for _, row in ranked.nsmallest(count, "robust_feature_distance")[keep].iterrows():
        records.append({key: _record(value) for key, value in row.items()})
    return records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_dirs = {
        "hf_suturebot": Path(args.hf_dir).resolve(),
        "nephfat": Path(args.nephfat_dir).resolve(),
    }
    summaries = {
        subset: _json(path / "coupling_scan_summary.json")
        for subset, path in scan_dirs.items()
    }
    frames = {}
    for subset, path in scan_dirs.items():
        frame = pd.read_csv(path / "coupling_windows.csv")
        frame["subset"] = subset
        frames[subset] = frame
    all_windows = pd.concat(frames.values(), ignore_index=True)
    focus_path = scan_dirs["hf_suturebot"] / "extra_windows.csv"
    focus_candidates = pd.read_csv(focus_path)
    focus_match = focus_candidates[
        (focus_candidates["episode_id"] == args.focus_episode)
        & (focus_candidates["base_index"] == args.focus_base)
    ]
    if len(focus_match) != 1:
        raise ValueError(
            f"expected one focus row {args.focus_episode}:{args.focus_base}, "
            f"found {len(focus_match)}"
        )
    focus = focus_match.iloc[0].copy()
    focus["subset"] = "hf_suturebot"
    mapping = _json(Path(args.mapping_audit).resolve())

    lag_summary = {}
    for subset, summary in summaries.items():
        action_to_state = summary["lag_diagnostics"]["action_to_state"]
        pair_summary = {}
        for source in ARMS:
            for target in ARMS:
                pair = f"{source}_to_{target}"
                vector_metric = action_to_state[pair]["vector_cosine"]
                speed_metric = action_to_state[pair]["speed_correlation"]
                pair_summary[pair] = {
                    "vector_cosine_at_zero": _lag_value(vector_metric),
                    "vector_cosine_best_lag": _best(vector_metric)[0],
                    "vector_cosine_best": _best(vector_metric)[1],
                    "speed_correlation_at_zero": _lag_value(speed_metric),
                    "speed_correlation_best_lag": _best(speed_metric)[0],
                    "speed_correlation_best": _best(speed_metric)[1],
                }
        same_zero = np.mean(
            [
                pair_summary["psm1_to_psm1"]["vector_cosine_at_zero"],
                pair_summary["psm2_to_psm2"]["vector_cosine_at_zero"],
            ]
        )
        swapped_zero = np.mean(
            [
                pair_summary["psm1_to_psm2"]["vector_cosine_at_zero"],
                pair_summary["psm2_to_psm1"]["vector_cosine_at_zero"],
            ]
        )
        visual_summary = {}
        for view, arms in summary["lag_diagnostics"]["visual"].items():
            visual_summary[view] = {}
            for arm, metrics in arms.items():
                action_lag, action_value = _best(metrics["action_speed_to_flow"])
                state_lag, state_value = _best(metrics["state_speed_to_flow"])
                visual_summary[view][arm] = {
                    "action_best_lag": action_lag,
                    "action_best_correlation": action_value,
                    "state_best_lag": state_lag,
                    "state_best_correlation": state_value,
                }
        lag_summary[subset] = {
            "pairs": pair_summary,
            "same_arm_vector_cosine_at_zero_mean": float(same_zero),
            "swapped_arm_vector_cosine_at_zero_mean": float(swapped_zero),
            "same_minus_swapped_margin": float(same_zero - swapped_zero),
            "mapping_separation_passed": bool(same_zero - swapped_zero > 0.25),
            "visual": visual_summary,
        }

    stratified = {}
    for subset, frame in frames.items():
        stratified[subset] = {
            "overall": _group_summary(frame),
            "by_task": {
                str(task): _group_summary(group)
                for task, group in frame.groupby("task", dropna=False)
            },
            "by_phase": {
                str(phase): _group_summary(group)
                for phase, group in frame.groupby("phase", dropna=False)
            },
            "by_dominant_arm": {
                str(arm): _group_summary(group)
                for arm, group in frame.groupby("dominant_arm", dropna=False)
            },
        }

    hf = frames["hf_suturebot"]
    same_task_phase = hf[
        (hf["task"] == focus["task"])
        & (hf["phase"] == focus["phase"])
        & (hf["dominant_arm"] == focus["dominant_arm"])
    ]
    motion_matched = same_task_phase[
        (
            same_task_phase["psm1_mean_action_translation_step_mm"].between(
                0.55, 0.75
            )
        )
        & (same_task_phase["psm2_mean_action_translation_step_mm"] <= 0.15)
    ]
    if len(motion_matched) < 20:
        motion_matched = hf[
            (hf["dominant_arm"] == "psm1")
            & (
                hf["psm1_mean_action_translation_step_mm"].between(
                    0.55, 0.75
                )
            )
            & (hf["psm2_mean_action_translation_step_mm"] <= 0.15)
        ]
    comparison_pools = {
        "all_hf_train": hf,
        "same_task_phase_dominant_arm": same_task_phase,
        "motion_matched_psm1_isolated": motion_matched,
    }
    rank_columns = [
        "psm1_mean_action_translation_step_mm",
        "psm2_mean_action_translation_step_mm",
        "psm1_to_psm2_action_motion_ratio",
        "psm1_same_arm_vector_cosine",
        "psm1_cross_arm_vector_cosine",
        "psm1_same_arm_speed_correlation",
        "psm1_cross_arm_speed_correlation",
        "psm1_psm2_action_speed_correlation",
        "endoscope_flow_magnitude_mean",
        "endoscope_flow_per_combined_state_mm",
        "endoscope_psm1_state_speed_correlation",
        "endoscope_psm2_state_speed_correlation",
        "wrist_left_psm1_state_speed_correlation",
        "wrist_left_psm2_state_speed_correlation",
        "wrist_right_psm1_state_speed_correlation",
        "wrist_right_psm2_state_speed_correlation",
    ]
    focus_ranks = {}
    for pool_name, pool in comparison_pools.items():
        focus_ranks[pool_name] = {
            "windows": int(len(pool)),
            "episodes": int(pool["episode_id"].nunique()),
            "percentiles": {
                column: _finite_percentile(pool[column], float(focus[column]))
                for column in rank_columns
                if column in pool
            },
        }
    nearest_pool = (
        same_task_phase
        if len(same_task_phase) >= args.nearest_neighbors
        else hf[hf["dominant_arm"] == "psm1"]
    )
    nearest = _nearest_neighbors(
        nearest_pool, focus, args.nearest_neighbors
    )
    nearest_frame = pd.DataFrame(nearest)
    nearest_frame.to_csv(output_dir / "focus_nearest_training_windows.csv", index=False)

    focus_record = {key: _record(value) for key, value in focus.items()}
    focus_interpretation = {
        "raw_command_is_psm1_dominant": bool(
            float(focus["psm1_to_psm2_action_motion_ratio"]) >= 3.0
        ),
        "psm2_is_parked": bool(
            float(focus["psm2_mean_action_translation_step_mm"]) < 0.1
        ),
        "psm1_command_tracks_psm1_state": bool(
            float(focus["psm1_same_arm_vector_cosine"]) >= 0.8
        ),
        "psm1_command_tracks_psm1_more_than_psm2": bool(
            float(focus["psm1_same_arm_vector_cosine"])
            > float(focus["psm1_cross_arm_vector_cosine"])
        ),
    }
    payload = {
        "diagnostic": "HF SutureBot and NephFat full-training coupling audit",
        "mapping_audit": mapping,
        "datasets": {
            subset: {
                "episodes": int(summary["episodes"]),
                "raw_rows": int(summary["raw_rows"]),
                "candidate_windows": int(summary["candidate_windows"]),
                "step_transitions": int(summary["step_transitions"]),
                "split_spec": summary["split_spec"],
            }
            for subset, summary in summaries.items()
        },
        "lag_and_mapping": lag_summary,
        "stratified": stratified,
        "focus_window": {
            "record": focus_record,
            "interpretation": focus_interpretation,
            "training_distribution_ranks": focus_ranks,
            "nearest_training_windows": nearest,
        },
        "visual_metric_caveat": (
            "Endoscope/wrist metrics are low-resolution motion proxies, not "
            "semantic tool masks; use SAM overlays for individual windows."
        ),
    }
    (output_dir / "training_coupling_audit_summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    colors = {
        "psm1_to_psm1": "#4C78A8",
        "psm2_to_psm2": "#E45756",
        "psm1_to_psm2": "#72B7B2",
        "psm2_to_psm1": "#F2CF5B",
    }
    for axis, subset in zip(axes, DATASETS):
        metrics = summaries[subset]["lag_diagnostics"]["action_to_state"]
        for pair, color in colors.items():
            points = metrics[pair]["vector_cosine"]["by_lag"]
            axis.plot(
                [point["lag_model_frames"] for point in points],
                [point["value"] for point in points],
                marker="o",
                label=pair.replace("_to_", "→"),
                color=color,
            )
        axis.axvline(0, color="black", linestyle=":", linewidth=1)
        axis.set_title(subset)
        axis.set_xlabel("State lag after action (model frames; 0.1 s)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean action→state step-vector cosine")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "action_state_lag_curves.png", dpi=180)
    plt.close(figure)

    rank_pool = focus_ranks["motion_matched_psm1_isolated"]["percentiles"]
    rank_plot_columns = [
        "psm1_same_arm_vector_cosine",
        "psm1_cross_arm_vector_cosine",
        "endoscope_flow_per_combined_state_mm",
        "endoscope_psm1_state_speed_correlation",
        "endoscope_psm2_state_speed_correlation",
    ]
    rank_plot_columns = [
        column for column in rank_plot_columns if rank_pool.get(column) is not None
    ]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.barh(
        [column.replace("_", " ") for column in rank_plot_columns],
        [rank_pool[column] for column in rank_plot_columns],
        color="#4C78A8",
    )
    axis.axvline(50, color="black", linestyle=":", linewidth=1)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Percentile among motion-matched isolated HF PSM1 windows")
    axis.set_title(f"HF episode {args.focus_episode}:{args.focus_base}")
    figure.tight_layout()
    figure.savefig(output_dir / "focus_training_percentiles.png", dpi=180)
    plt.close(figure)

    mapping_verdict = mapping["verdict"]
    focus_pool = focus_ranks["motion_matched_psm1_isolated"]
    lines = [
        "# C3-H-S-S full-training action/vision coupling audit",
        "",
        "## Scope",
        "",
        (
            f"Scanned {summaries['hf_suturebot']['episodes']} HF SutureBot and "
            f"{summaries['nephfat']['episodes']} NephFat training episodes, "
            f"{len(all_windows):,} eligible H=16/N=12 windows total."
        ),
        "",
        "## Mapping and timing",
        "",
        (
            f"- Raw-to-model arm mapping: **{mapping_verdict['training_arm_mapping']}**. "
            f"Translation reconstruction error was "
            f"{mapping['live_archive_comparison']['translation_max_abs_error']:.2e}."
        ),
        (
            "- Rotational eval helper: **"
            f"{mapping_verdict['rotation_intervention_helper']}**. Training uses "
            "column-based rotation-6D while the helper decodes rows. This does not "
            "affect training or the translation-only multi-seed probe."
        ),
    ]
    for subset in DATASETS:
        lag = lag_summary[subset]
        lines.append(
            f"- {subset}: same-arm zero-lag vector cosine "
            f"{lag['same_arm_vector_cosine_at_zero_mean']:.3f}, swapped-arm "
            f"{lag['swapped_arm_vector_cosine_at_zero_mean']:.3f}, margin "
            f"{lag['same_minus_swapped_margin']:.3f}."
        )
    lines.extend(
        [
            "",
            f"## Held-out failure window {args.focus_episode}:{args.focus_base}",
            "",
            (
                f"- Task/phase: {focus['task']} / {focus['phase']}; dominant arm "
                f"{focus['dominant_arm']}."
            ),
            (
                f"- PSM1 command motion {focus['psm1_mean_action_translation_step_mm']:.3f} "
                f"mm/step versus PSM2 {focus['psm2_mean_action_translation_step_mm']:.3f} "
                f"mm/step ({focus['psm1_to_psm2_action_motion_ratio']:.1f}× isolation)."
            ),
            (
                f"- PSM1 command→PSM1 state cosine "
                f"{focus['psm1_same_arm_vector_cosine']:.3f}; command→PSM2 state "
                f"cosine {focus['psm1_cross_arm_vector_cosine']:.3f}."
            ),
            (
                f"- Ranked against {focus_pool['windows']:,} motion-matched isolated "
                "HF PSM1 training windows; see `focus_training_percentiles.png` and "
                "`focus_nearest_training_windows.csv`."
            ),
            "",
            "The visual fields are motion/flow proxies rather than semantic tool masks.",
            "They are appropriate for full-split timing and outlier ranking; the",
            "existing SAM tracking overlays remain the tool-specific evidence.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"WROTE {output_dir / 'training_coupling_audit_summary.json'}")
    print(f"WROTE {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
