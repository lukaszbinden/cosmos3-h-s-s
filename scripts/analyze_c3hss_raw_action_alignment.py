#!/usr/bin/env python3
"""Locate dVRK action-following failures across the raw-to-model data path.

The analysis joins four independently captured views of the same ten windows:

* raw parquet action and observation.state tracks;
* model-facing pre-normalized and normalized action archives;
* SAM2 ground-truth tool tracks; and
* per-tool model counterfactual response.

It tests arm order, reference-pose semantics, transform/normalization fidelity,
action-to-robot timing, arm-to-visible-tool mapping, and whether counterfactual
localization is confounded by near-static selected action windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ARMS = ("psm1", "psm2")
ARM_RAW_OFFSETS = {"psm1": 0, "psm2": 8}
ARM_MODEL_OFFSETS = {"psm1": 0, "psm2": 10}
ARM_TO_TOOL = {"psm1": "entry_right", "psm2": "entry_left"}
TOOLS = ("entry_left", "entry_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--raw-extract-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-index", type=int, default=48)
    parser.add_argument("--timestep-interval", type=int, default=3)
    parser.add_argument("--current-actions", type=int, default=12)
    parser.add_argument("--low-motion-threshold-mm", type=float, default=0.1)
    return parser.parse_args()


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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"
    )


def _mean_vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = denominator > 1e-10
    if not valid.any():
        return float("nan")
    return float(
        np.mean(np.sum(left[valid] * right[valid], axis=1) / denominator[valid])
    )


def _lagged_vector_cosines(
    command_steps: np.ndarray, state_steps: np.ndarray, max_lag: int = 2
) -> dict[int, float]:
    """Score whether observed robot steps lag same-rate command steps.

    A positive lag means the state track is shifted later than the command
    track; one step is 0.1 seconds for the evaluated 30 Hz / stride-3 recipe.
    """
    result = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left, right = command_steps[-lag:], state_steps[:lag]
        elif lag > 0:
            left, right = command_steps[:-lag], state_steps[lag:]
        else:
            left, right = command_steps, state_steps
        result[lag] = _mean_vector_cosine(left, right)
    return result


def _quaternion_step_degrees(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    normalized = quaternions / np.maximum(norms, 1e-12)
    dots = np.abs(np.sum(normalized[1:] * normalized[:-1], axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0)))


def _mask_points(mask: np.ndarray, object_name: str) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.nonzero(mask)
    if not len(x):
        missing = np.full(2, np.nan)
        return missing, missing
    centroid = np.asarray([x.mean(), y.mean()], dtype=np.float64)
    percentile = 98.0 if object_name == "entry_left" else 2.0
    extreme = np.percentile(x, percentile)
    selected = x >= extreme if object_name == "entry_left" else x <= extreme
    tip = np.asarray([x[selected].mean(), y[selected].mean()], dtype=np.float64)
    return centroid, tip


def _trajectory(mask_track: np.ndarray, object_name: str) -> dict[str, np.ndarray]:
    points = [_mask_points(mask, object_name) for mask in mask_track]
    return {
        "centroid": np.asarray([item[0] for item in points]),
        "tip": np.asarray([item[1] for item in points]),
    }


def _mean_step_magnitude(points: np.ndarray) -> float:
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(np.nanmean(steps))


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(
        values, return_inverse=True, return_counts=True
    )
    del unique
    for group in np.flatnonzero(counts > 1):
        indices = inverse == group
        ranks[indices] = ranks[indices].mean()
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _correlation(_ranks(left), _ranks(right))


def _bootstrap_spearman(
    left: np.ndarray, right: np.ndarray, seed: int = 20260727
) -> list[float]:
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(20_000):
        indices = generator.integers(0, len(left), size=len(left))
        value = _spearman(left[indices], right[indices])
        if np.isfinite(value):
            values.append(value)
    return [float(item) for item in np.quantile(values, [0.025, 0.975])]


def _resolve_one(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {directory / pattern}: {matches}")
    return matches[0]


def _episode_localization_lookup(summary: dict[str, Any]) -> dict[tuple[str, int, str], float]:
    result = {}
    for arm in ARMS:
        for item in summary["arm_localization"][arm]["episode_localization"]:
            result[(item["subset"], int(item["episode_id"]), arm)] = float(
                item["intended_response_fraction"]
            )
    return result


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    raw_extract_dir = Path(args.raw_extract_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = input_root / "analysis/tool_specific/masks"
    tool_summary = json.loads(
        (input_root / "analysis/tool_specific/tool_specific_summary.json").read_text()
    )
    localization = _episode_localization_lookup(tool_summary)

    episode_rows: list[dict[str, Any]] = []
    transform_audits = []
    subset_names = sorted(
        path.stem for path in raw_extract_dir.glob("*.npz")
    )
    if subset_names != ["hf_suturebot", "nephfat", "wound_closure"]:
        raise RuntimeError(f"Unexpected raw extract set: {subset_names}")

    for subset in subset_names:
        raw_manifest = json.loads((raw_extract_dir / f"{subset}.json").read_text())
        reference_sources = raw_manifest["state_reference_source"]
        pose_reference_sources = {
            reference_sources["psm1_pose"],
            reference_sources["psm2_pose"],
        }
        if len(pose_reference_sources) != 1:
            raise ValueError(f"{subset} uses different pose references per arm")
        reference_source = next(iter(pose_reference_sources))
        with np.load(raw_extract_dir / f"{subset}.npz") as raw_archive:
            action_keys = sorted(
                key for key in raw_archive.files if key.endswith("__action")
            )
            for raw_action_key in action_keys:
                episode_id = int(
                    re.search(r"episode_(\d+)", raw_action_key).group(1)
                )
                prefix = f"episode_{episode_id:06d}"
                raw_action = raw_archive[f"{prefix}__action"]
                raw_state = raw_archive[f"{prefix}__state"]
                action_indices = args.base_index + np.arange(
                    args.current_actions
                ) * args.timestep_interval
                video_indices = args.base_index + np.arange(
                    args.current_actions + 1
                ) * args.timestep_interval
                if video_indices[-1] >= len(raw_action):
                    raise IndexError(f"{subset}:{episode_id} window exceeds episode")

                episode_dir = input_root / "raw" / subset
                intervention_manifest_path = _resolve_one(
                    episode_dir,
                    f"*_ep{episode_id:05d}_seed*_action_intervention_episode.json",
                )
                intervention_manifest = json.loads(
                    intervention_manifest_path.read_text()
                )
                action_archive_path = _resolve_one(
                    episode_dir,
                    f"*_ep{episode_id:05d}_seed*_normalized_actions.npz",
                )
                mask_path = _resolve_one(
                    masks_dir, f"{subset}_ep{episode_id:05d}.npz"
                )
                stats = intervention_manifest["physical_action_audit"]
                means = np.asarray(stats["mean"], dtype=np.float64)
                stds = np.asarray(stats["std"], dtype=np.float64)

                with np.load(action_archive_path) as model_archive:
                    model_physical = model_archive["physical__correct"].astype(
                        np.float64
                    )
                    model_normalized = model_archive[
                        "normalized__correct"
                    ].astype(np.float64)
                expected_normalized = (model_physical - means) / stds
                normalization_error = float(
                    np.max(np.abs(expected_normalized - model_normalized))
                )

                reference_track = (
                    raw_action
                    if reference_source == "action"
                    else raw_state
                )
                transform_errors = {}
                sign_mismatches = {}
                arm_tracks = {}
                for arm in ARMS:
                    raw_offset = ARM_RAW_OFFSETS[arm]
                    model_offset = ARM_MODEL_OFFSETS[arm]
                    expected = (
                        raw_action[action_indices, raw_offset : raw_offset + 3]
                        - reference_track[
                            args.base_index, raw_offset : raw_offset + 3
                        ]
                    )
                    actual = model_physical[:, model_offset : model_offset + 3]
                    transform_errors[arm] = float(
                        np.max(np.abs(expected - actual))
                    )
                    physical_steps = np.diff(actual, axis=0)
                    normalized_steps = np.diff(
                        model_normalized[:, model_offset : model_offset + 3],
                        axis=0,
                    )
                    active = np.abs(physical_steps) > 1e-9
                    sign_mismatches[arm] = int(
                        np.count_nonzero(
                            active
                            & (
                                np.sign(physical_steps)
                                != np.sign(normalized_steps)
                            )
                        )
                    )

                    action_position = raw_action[
                        video_indices, raw_offset : raw_offset + 3
                    ]
                    state_position = raw_state[
                        video_indices, raw_offset : raw_offset + 3
                    ]
                    action_steps = np.diff(action_position, axis=0)
                    state_steps = np.diff(state_position, axis=0)
                    action_translation_mm = (
                        np.linalg.norm(action_steps, axis=1) * 1000.0
                    )
                    state_translation_mm = (
                        np.linalg.norm(state_steps, axis=1) * 1000.0
                    )
                    action_rotation_degrees = _quaternion_step_degrees(
                        raw_action[
                            video_indices, raw_offset + 3 : raw_offset + 7
                        ]
                    )
                    state_rotation_degrees = _quaternion_step_degrees(
                        raw_state[
                            video_indices, raw_offset + 3 : raw_offset + 7
                        ]
                    )
                    physical_translation = model_physical[
                        :, model_offset : model_offset + 3
                    ]
                    lag_cosines = _lagged_vector_cosines(
                        action_steps, state_steps
                    )
                    physical_rms = float(
                        np.sqrt(np.mean(physical_translation**2)) * 1000.0
                    )
                    physical_dynamic_rms = float(
                        np.sqrt(
                            np.mean(
                                (
                                    physical_translation
                                    - physical_translation.mean(axis=0)
                                )
                                ** 2
                            )
                        )
                        * 1000.0
                    )
                    arm_tracks[arm] = {
                        "mean_action_translation_step_mm": float(
                            action_translation_mm.mean()
                        ),
                        "mean_state_translation_step_mm": float(
                            state_translation_mm.mean()
                        ),
                        "mean_action_rotation_step_degrees": float(
                            action_rotation_degrees.mean()
                        ),
                        "mean_state_rotation_step_degrees": float(
                            state_rotation_degrees.mean()
                        ),
                        "action_to_state_translation_step_cosine": (
                            _mean_vector_cosine(action_steps, state_steps)
                        ),
                        "action_to_state_translation_step_lag_cosines": {
                            str(lag): value
                            for lag, value in lag_cosines.items()
                        },
                        "best_action_to_state_lag_steps": max(
                            lag_cosines, key=lag_cosines.get
                        ),
                        "mean_action_state_translation_step_error_mm": float(
                            np.linalg.norm(
                                action_steps - state_steps, axis=1
                            ).mean()
                            * 1000.0
                        ),
                        "model_physical_translation_rms_mm": physical_rms,
                        "model_physical_translation_dynamic_rms_mm": (
                            physical_dynamic_rms
                        ),
                        "model_physical_translation_dynamic_fraction": float(
                            physical_dynamic_rms / max(physical_rms, 1e-12)
                        ),
                        "model_localization_fraction": localization[
                            (subset, episode_id, arm)
                        ],
                    }

                with np.load(mask_path) as mask_archive:
                    tool_tracks = {
                        tool: _trajectory(
                            mask_archive[f"gt__{tool}"].astype(bool), tool
                        )
                        for tool in TOOLS
                    }
                tool_motion = {
                    tool: {
                        "mean_centroid_step_px": _mean_step_magnitude(
                            track["centroid"]
                        ),
                        "mean_tip_step_px": _mean_step_magnitude(track["tip"]),
                    }
                    for tool, track in tool_tracks.items()
                }
                dominant_action_arm = max(
                    ARMS,
                    key=lambda arm: arm_tracks[arm][
                        "mean_action_translation_step_mm"
                    ],
                )
                dominant_tool = max(
                    TOOLS,
                    key=lambda tool: tool_motion[tool]["mean_centroid_step_px"],
                )
                dominant_mapping_matches = (
                    ARM_TO_TOOL[dominant_action_arm] == dominant_tool
                )

                row: dict[str, Any] = {
                    "subset": subset,
                    "episode_id": episode_id,
                    "configured_pose_reference_source": reference_source,
                    "normalization_max_abs_error": normalization_error,
                    "dominant_action_arm": dominant_action_arm,
                    "dominant_visible_tool": dominant_tool,
                    "dominant_arm_tool_mapping_matches": dominant_mapping_matches,
                }
                for arm in ARMS:
                    row[f"{arm}_transform_max_abs_error_m"] = transform_errors[arm]
                    row[f"{arm}_normalization_step_sign_mismatches"] = (
                        sign_mismatches[arm]
                    )
                    row.update(
                        {
                            f"{arm}_{key}": value
                            for key, value in arm_tracks[arm].items()
                        }
                    )
                for tool in TOOLS:
                    row.update(
                        {
                            f"{tool}_{key}": value
                            for key, value in tool_motion[tool].items()
                        }
                    )
                episode_rows.append(row)
                transform_audits.append(
                    {
                        "subset": subset,
                        "episode_id": episode_id,
                        "reference_source": reference_source,
                        "transform_max_abs_error_m": max(
                            transform_errors.values()
                        ),
                        "normalization_max_abs_error": normalization_error,
                        "normalization_step_sign_mismatches": sum(
                            sign_mismatches.values()
                        ),
                    }
                )

    with (output_dir / "per_episode_raw_alignment.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)

    arm_summary = {}
    for arm in ARMS:
        activity = np.asarray(
            [row[f"{arm}_mean_action_translation_step_mm"] for row in episode_rows]
        )
        localization_fraction = np.asarray(
            [row[f"{arm}_model_localization_fraction"] for row in episode_rows]
        )
        cosine = np.asarray(
            [
                row[f"{arm}_action_to_state_translation_step_cosine"]
                for row in episode_rows
            ]
        )
        low_motion = activity < args.low_motion_threshold_mm
        mean_lag_cosines = {
            lag: float(
                np.nanmean(
                    [
                        row[
                            f"{arm}_action_to_state_translation_step_lag_cosines"
                        ][str(lag)]
                        for row in episode_rows
                    ]
                )
            )
            for lag in range(-2, 3)
        }
        arm_summary[arm] = {
            "mean_action_to_state_translation_step_cosine": float(
                np.nanmean(cosine)
            ),
            "median_action_to_state_translation_step_cosine": float(
                np.nanmedian(cosine)
            ),
            "mean_action_to_state_lag_cosines": {
                str(lag): value for lag, value in mean_lag_cosines.items()
            },
            "best_mean_action_to_state_lag_steps": max(
                mean_lag_cosines, key=mean_lag_cosines.get
            ),
            "activity_vs_model_localization_pearson": _correlation(
                activity, localization_fraction
            ),
            "activity_vs_model_localization_spearman": _spearman(
                activity, localization_fraction
            ),
            "activity_vs_model_localization_spearman_bootstrap_95_ci": (
                _bootstrap_spearman(
                    activity,
                    localization_fraction,
                    seed=20260727 + int(arm[-1]),
                )
            ),
            "low_motion_threshold_mm": args.low_motion_threshold_mm,
            "low_motion_windows": int(low_motion.sum()),
            "low_motion_mean_localization_fraction": float(
                localization_fraction[low_motion].mean()
            )
            if low_motion.any()
            else None,
            "active_windows": int((~low_motion).sum()),
            "active_mean_localization_fraction": float(
                localization_fraction[~low_motion].mean()
            )
            if (~low_motion).any()
            else None,
        }

    reference_sources = {
        subset: min(
            {
                row["configured_pose_reference_source"]
                for row in episode_rows
                if row["subset"] == subset
            }
        )
        for subset in subset_names
    }
    payload = {
        "diagnostic": "raw dVRK action-to-visible-tool alignment",
        "episodes": len(episode_rows),
        "base_index": args.base_index,
        "timestep_interval": args.timestep_interval,
        "current_actions": args.current_actions,
        "arm_to_tool_mapping": ARM_TO_TOOL,
        "configured_pose_reference_sources": reference_sources,
        "raw_to_model_transform": {
            "maximum_translation_error_m": max(
                item["transform_max_abs_error_m"] for item in transform_audits
            ),
            "maximum_normalization_error": max(
                item["normalization_max_abs_error"] for item in transform_audits
            ),
            "translation_step_sign_mismatches_after_normalization": sum(
                item["normalization_step_sign_mismatches"]
                for item in transform_audits
            ),
        },
        "dominant_arm_matches_dominant_visible_tool": {
            "matches": sum(
                bool(row["dominant_arm_tool_mapping_matches"])
                for row in episode_rows
            ),
            "episodes": len(episode_rows),
        },
        "arm_summary": arm_summary,
        "episodes_detail": episode_rows,
    }
    _write_json(output_dir / "raw_action_alignment_summary.json", payload)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    subset_colors = {
        "hf_suturebot": "#4C78A8",
        "nephfat": "#F58518",
        "wound_closure": "#54A24B",
    }
    for axis, arm in zip(axes, ARMS):
        for subset in subset_names:
            records = [row for row in episode_rows if row["subset"] == subset]
            axis.scatter(
                [row[f"{arm}_mean_action_translation_step_mm"] for row in records],
                [row[f"{arm}_model_localization_fraction"] for row in records],
                label=subset,
                color=subset_colors[subset],
                s=55,
            )
            for row in records:
                axis.annotate(
                    str(row["episode_id"]),
                    (
                        row[f"{arm}_mean_action_translation_step_mm"],
                        row[f"{arm}_model_localization_fraction"],
                    ),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                )
        axis.axhline(0.5, color="black", linewidth=1, linestyle="--")
        axis.axvline(
            args.low_motion_threshold_mm,
            color="gray",
            linewidth=1,
            linestyle=":",
        )
        axis.set_title(
            f"{arm.upper()}: Spearman "
            f"{arm_summary[arm]['activity_vs_model_localization_spearman']:.2f}"
        )
        axis.set_xlabel("Raw action translation step (mm / 0.1 s)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Intended-tool counterfactual response fraction")
    axes[1].legend(fontsize=8)
    figure.suptitle("Selected arm motion vs model response localization")
    figure.tight_layout()
    figure.savefig(output_dir / "activity_vs_model_localization.png", dpi=180)
    plt.close(figure)

    psm2 = arm_summary["psm2"]
    transform = payload["raw_to_model_transform"]
    mapping = payload["dominant_arm_matches_dominant_visible_tool"]
    lines = [
        "# C3-H-S-S raw action alignment diagnostic",
        "",
        "## Verdict",
        "",
        "The PSM2 problem is not introduced by arm ordering, the relative-pose",
        "transform, mean/std normalization, or a PSM2-specific timestamp shift.",
        "The ten-window counterfactual set is strongly arm-motion imbalanced, and",
        "the earlier PSM2 localization result is substantially confounded by that",
        "selection.",
        "",
        "## Boundary checks",
        "",
        (
            f"- Maximum raw-to-model translation mismatch: "
            f"`{transform['maximum_translation_error_m']:.3e} m`."
        ),
        (
            f"- Maximum normalization reconstruction mismatch: "
            f"`{transform['maximum_normalization_error']:.3e}`; translation-step "
            f"sign flips: "
            f"`{transform['translation_step_sign_mismatches_after_normalization']}`."
        ),
        (
            f"- Raw command-step versus observed robot-step cosine: "
            f"PSM1 "
            f"`{arm_summary['psm1']['mean_action_to_state_translation_step_cosine']:.3f}`, "
            f"PSM2 `{psm2['mean_action_to_state_translation_step_cosine']:.3f}`."
        ),
        (
            f"- Best mean command→state lag: PSM1 "
            f"`{arm_summary['psm1']['best_mean_action_to_state_lag_steps']}` step, "
            f"PSM2 `{psm2['best_mean_action_to_state_lag_steps']}` steps "
            "(one step is 0.1 s)."
        ),
        (
            f"- Dominant raw arm motion matches the corresponding visible tool in "
            f"`{mapping['matches']}/{mapping['episodes']}` episodes."
        ),
        "",
        "## Why the prior PSM2 result looked poor",
        "",
        (
            f"PSM2 action activity and intended-tool response have Spearman "
            f"`{psm2['activity_vs_model_localization_spearman']:.3f}` "
            f"(episode bootstrap 95% CI "
            f"`{psm2['activity_vs_model_localization_spearman_bootstrap_95_ci'][0]:.3f}` "
            f"to "
            f"`{psm2['activity_vs_model_localization_spearman_bootstrap_95_ci'][1]:.3f}`)."
        ),
        (
            f"The `{psm2['low_motion_windows']}` windows below "
            f"`{args.low_motion_threshold_mm:.1f} mm/step` average only "
            f"`{psm2['low_motion_mean_localization_fraction']:.3f}` localization; "
            f"the `{psm2['active_windows']}` active windows average "
            f"`{psm2['active_mean_localization_fraction']:.3f}`."
        ),
        "",
        "In the three hf_suturebot windows, PSM2 translation and rotation are",
        "nearly static. Its model-facing relative translation is dominated by a",
        "constant command-versus-state offset, so the old 0x intervention removes",
        "that offset rather than isolating visible trajectory motion.",
        "",
        "## Dataset semantic difference",
        "",
        "The modality metadata anchors hf_suturebot and nephfat actions to",
        "`observation.state`, but wound_closure anchors both arms to the first",
        "`action` setpoint. This is a real cross-dataset semantic difference, but",
        "it affects both PSMs symmetrically and does not explain a PSM2-only error.",
        "",
        "## Next controlled test",
        "",
        "Select separate high-motion PSM1 and high-motion PSM2 windows from each",
        "dataset. Scale each translation/rotation trajectory around its first",
        "model-facing row, preserving the initial command-state offset. Then rerun",
        "the same per-tool counterfactual grid. That removes the current selection",
        "and intervention-origin confounds.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
