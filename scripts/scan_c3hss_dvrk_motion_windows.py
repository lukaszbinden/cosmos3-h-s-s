#!/usr/bin/env python3
"""Scan a dVRK LeRobot split for motion-balanced CAMP evaluation windows.

The scanner reads raw parquet action/state rows without importing Cosmos.  It
emits one CSV row per eligible window and a data-health summary covering raw
layout, timestamps, quaternion validity, arm activity, command/state agreement,
and the configured reference source.  These outputs support auditable selection
of active PSM1 and PSM2 windows instead of relying on one fixed base index.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ARMS = ("psm1", "psm2")
ARM_OFFSETS = {"psm1": 0, "psm2": 8}
EXPECTED_MODALITY_SLICES = {
    "psm1_pose": (0, 7),
    "psm1_gripper": (7, 8),
    "psm2_pose": (8, 15),
    "psm2_gripper": (15, 16),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--timestep-interval", type=int, default=3)
    parser.add_argument("--current-actions", type=int, default=12)
    parser.add_argument("--history-actions", type=int, default=16)
    parser.add_argument(
        "--base-step",
        type=int,
        default=3,
        help="Raw-frame increment between scanned candidate anchors.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _parse_split(spec: str) -> list[int]:
    result = []
    for section in spec.split(","):
        bounds = section.strip().split(":")
        if len(bounds) != 2:
            raise ValueError(f"Unsupported split section {section!r}")
        start, stop = (int(value) for value in bounds)
        result.extend(range(start, stop))
    return result


def _validate_modality(modality: dict[str, Any]) -> dict[str, str]:
    for section_name in ("state", "action"):
        section = modality.get(section_name, {})
        for key, expected in EXPECTED_MODALITY_SLICES.items():
            entry = section.get(key)
            actual = (
                None
                if entry is None
                else (int(entry["start"]), int(entry["end"]))
            )
            if actual != expected:
                raise ValueError(
                    f"{section_name}.{key} has slice {actual}, expected {expected}"
                )
    return {
        key: modality["state"][key].get(
            "original_key", "observation.state"
        )
        for key in EXPECTED_MODALITY_SLICES
    }


def _episode_path(root: Path, info: dict[str, Any], episode_id: int) -> Path:
    relative = info["data_path"].format(
        episode_chunk=episode_id // int(info["chunks_size"]),
        episode_index=episode_id,
    )
    return root / relative


def _mean_vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = denominator > 1e-10
    if not valid.any():
        return float("nan")
    return float(
        np.mean(np.sum(left[valid] * right[valid], axis=1) / denominator[valid])
    )


def _quaternion_steps_degrees(quaternions: np.ndarray) -> np.ndarray:
    normalized = quaternions / np.maximum(
        np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-12
    )
    dots = np.abs(np.sum(normalized[1:] * normalized[:-1], axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0)))


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            key: float("nan") for key in ("q05", "q25", "q50", "q75", "q95")
        }
    return {
        key: float(value)
        for key, value in zip(
            ("q05", "q25", "q50", "q75", "q95"),
            np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95]),
        )
    }


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
    root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    info_path = root / "meta/info.json"
    modality_path = root / "meta/modality.json"
    info = _load_json(info_path)
    modality = _load_json(modality_path)
    reference_sources = _validate_modality(modality)
    pose_reference_sources = {
        reference_sources["psm1_pose"],
        reference_sources["psm2_pose"],
    }
    if len(pose_reference_sources) != 1:
        raise ValueError("PSM1 and PSM2 use different pose reference sources")
    pose_reference_source = next(iter(pose_reference_sources))
    episode_ids = _parse_split(info["splits"][args.split])

    rows: list[dict[str, Any]] = []
    episodes_detail = []
    nonfinite_action_rows = 0
    nonfinite_state_rows = 0
    invalid_action_quaternion_rows = {arm: 0 for arm in ARMS}
    invalid_state_quaternion_rows = {arm: 0 for arm in ARMS}
    nonincreasing_timestamp_steps = 0
    episodes_with_nonincreasing_timestamps = 0
    total_rows = 0
    timestep_seconds = args.timestep_interval / float(info["fps"])

    for progress_index, episode_id in enumerate(episode_ids, start=1):
        parquet_path = _episode_path(root, info, episode_id)
        table = pq.read_table(
            parquet_path,
            columns=[
                "action",
                "observation.state",
                "timestamp",
                "frame_index",
                "episode_index",
            ],
        )
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        state = np.asarray(
            table["observation.state"].to_pylist(), dtype=np.float32
        )
        timestamp = table["timestamp"].to_numpy().astype(np.float64)
        frame_index = table["frame_index"].to_numpy().astype(np.int64)
        episode_index = table["episode_index"].to_numpy().astype(np.int64)
        if action.shape != state.shape or action.ndim != 2 or action.shape[1] != 16:
            raise ValueError(
                f"{parquet_path} action/state shapes {action.shape}/{state.shape}"
            )
        if not np.array_equal(frame_index, np.arange(len(frame_index))):
            raise ValueError(f"{parquet_path} has non-contiguous frame_index")
        if not np.all(episode_index == episode_id):
            raise ValueError(f"{parquet_path} contains a wrong episode_index")
        timestamp_steps = np.diff(timestamp)
        episode_nonincreasing_timestamp_steps = int(
            np.count_nonzero(timestamp_steps <= 0)
        )
        nonincreasing_timestamp_steps += episode_nonincreasing_timestamp_steps
        episodes_with_nonincreasing_timestamps += int(
            episode_nonincreasing_timestamp_steps > 0
        )

        total_rows += len(action)
        nonfinite_action_rows += int(np.count_nonzero(~np.isfinite(action).all(axis=1)))
        nonfinite_state_rows += int(np.count_nonzero(~np.isfinite(state).all(axis=1)))
        for arm in ARMS:
            offset = ARM_OFFSETS[arm]
            action_quaternion_norm = np.linalg.norm(
                action[:, offset + 3 : offset + 7], axis=1
            )
            state_quaternion_norm = np.linalg.norm(
                state[:, offset + 3 : offset + 7], axis=1
            )
            invalid_action_quaternion_rows[arm] += int(
                np.count_nonzero(np.abs(action_quaternion_norm - 1.0) > 1e-3)
            )
            invalid_state_quaternion_rows[arm] += int(
                np.count_nonzero(np.abs(state_quaternion_norm - 1.0) > 1e-3)
            )

        first_base = args.history_actions * args.timestep_interval
        last_base_exclusive = (
            len(action) - args.current_actions * args.timestep_interval
        )
        candidate_count = 0
        for base_index in range(
            first_base, last_base_exclusive, args.base_step
        ):
            indices = base_index + np.arange(
                args.current_actions + 1
            ) * args.timestep_interval
            action_indices = indices[:-1]
            reference = (
                action[base_index]
                if pose_reference_source == "action"
                else state[base_index]
            )
            row: dict[str, Any] = {
                "episode_id": episode_id,
                "base_index": base_index,
                "window_start_seconds": float(timestamp[base_index]),
                "window_duration_seconds": (
                    args.current_actions * timestep_seconds
                ),
                "pose_reference_source": pose_reference_source,
            }
            for arm in ARMS:
                offset = ARM_OFFSETS[arm]
                action_positions = action[indices, offset : offset + 3]
                state_positions = state[indices, offset : offset + 3]
                action_steps = np.diff(action_positions, axis=0)
                state_steps = np.diff(state_positions, axis=0)
                action_step_mm = np.linalg.norm(action_steps, axis=1) * 1000.0
                state_step_mm = np.linalg.norm(state_steps, axis=1) * 1000.0
                action_rotation_step = _quaternion_steps_degrees(
                    action[indices, offset + 3 : offset + 7]
                )
                state_rotation_step = _quaternion_steps_degrees(
                    state[indices, offset + 3 : offset + 7]
                )
                relative_translation = (
                    action[action_indices, offset : offset + 3]
                    - reference[offset : offset + 3]
                )
                relative_rms = float(
                    np.sqrt(np.mean(relative_translation**2)) * 1000.0
                )
                dynamic_rms = float(
                    np.sqrt(
                        np.mean(
                            (
                                relative_translation
                                - relative_translation.mean(axis=0)
                            )
                            ** 2
                        )
                    )
                    * 1000.0
                )
                row.update(
                    {
                        f"{arm}_mean_action_translation_step_mm": float(
                            action_step_mm.mean()
                        ),
                        f"{arm}_mean_state_translation_step_mm": float(
                            state_step_mm.mean()
                        ),
                        f"{arm}_action_translation_path_mm": float(
                            action_step_mm.sum()
                        ),
                        f"{arm}_mean_action_rotation_step_degrees": float(
                            action_rotation_step.mean()
                        ),
                        f"{arm}_mean_state_rotation_step_degrees": float(
                            state_rotation_step.mean()
                        ),
                        f"{arm}_action_rotation_path_degrees": float(
                            action_rotation_step.sum()
                        ),
                        f"{arm}_action_to_state_translation_step_cosine": (
                            _mean_vector_cosine(action_steps, state_steps)
                        ),
                        f"{arm}_mean_action_state_step_error_mm": float(
                            np.linalg.norm(
                                action_steps - state_steps, axis=1
                            ).mean()
                            * 1000.0
                        ),
                        f"{arm}_initial_command_state_offset_mm": float(
                            np.linalg.norm(
                                action[base_index, offset : offset + 3]
                                - state[base_index, offset : offset + 3]
                            )
                            * 1000.0
                        ),
                        f"{arm}_relative_translation_rms_mm": relative_rms,
                        f"{arm}_relative_translation_dynamic_rms_mm": dynamic_rms,
                        f"{arm}_relative_translation_dynamic_fraction": float(
                            dynamic_rms / max(relative_rms, 1e-12)
                        ),
                    }
                )
            rows.append(row)
            candidate_count += 1
        episodes_detail.append(
            {
                "episode_id": episode_id,
                "rows": len(action),
                "candidate_windows": candidate_count,
                "mean_timestamp_step_seconds": float(np.mean(timestamp_steps)),
                "nonincreasing_timestamp_steps": (
                    episode_nonincreasing_timestamp_steps
                ),
                "minimum_timestamp_step_seconds": float(
                    np.min(timestamp_steps)
                ),
                "parquet_sha256": _sha256(parquet_path),
            }
        )
        if progress_index % 25 == 0 or progress_index == len(episode_ids):
            print(
                f"SCANNED {progress_index}/{len(episode_ids)} episodes; "
                f"windows={len(rows)}",
                flush=True,
            )

    csv_path = output_dir / "motion_windows.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    activity_summary = {}
    for arm in ARMS:
        translation = [
            row[f"{arm}_mean_action_translation_step_mm"] for row in rows
        ]
        rotation = [
            row[f"{arm}_mean_action_rotation_step_degrees"] for row in rows
        ]
        cosine = [
            row[f"{arm}_action_to_state_translation_step_cosine"]
            for row in rows
        ]
        dynamic_fraction = [
            row[f"{arm}_relative_translation_dynamic_fraction"] for row in rows
        ]
        activity_summary[arm] = {
            "translation_step_mm": _quantiles(translation),
            "rotation_step_degrees": _quantiles(rotation),
            "action_to_state_translation_step_cosine": _quantiles(cosine),
            "relative_translation_dynamic_fraction": _quantiles(
                dynamic_fraction
            ),
            "windows_below_0p1_mm_step": int(
                np.count_nonzero(np.asarray(translation) < 0.1)
            ),
            "fraction_below_0p1_mm_step": float(
                np.mean(np.asarray(translation) < 0.1)
            ),
        }

    summary = {
        "diagnostic": "dataset-wide raw dVRK motion-window scan",
        "dataset_root": str(root),
        "dataset_robot_type": info.get("robot_type"),
        "dataset_fps": info.get("fps"),
        "split": args.split,
        "split_spec": info["splits"][args.split],
        "episodes": len(episode_ids),
        "raw_rows": total_rows,
        "candidate_windows": len(rows),
        "window_recipe": {
            "timestep_interval": args.timestep_interval,
            "current_actions": args.current_actions,
            "history_actions": args.history_actions,
            "base_step": args.base_step,
        },
        "validated_raw_layout": {
            key: list(value) for key, value in EXPECTED_MODALITY_SLICES.items()
        },
        "state_reference_source": reference_sources,
        "pose_reference_source": pose_reference_source,
        "data_health": {
            "nonfinite_action_rows": nonfinite_action_rows,
            "nonfinite_state_rows": nonfinite_state_rows,
            "invalid_action_quaternion_rows": invalid_action_quaternion_rows,
            "invalid_state_quaternion_rows": invalid_state_quaternion_rows,
            "nonincreasing_timestamp_steps": nonincreasing_timestamp_steps,
            "episodes_with_nonincreasing_timestamps": (
                episodes_with_nonincreasing_timestamps
            ),
        },
        "activity_summary": activity_summary,
        "info_sha256": _sha256(info_path),
        "modality_sha256": _sha256(modality_path),
        "motion_windows_csv": str(csv_path),
        "episodes_detail": episodes_detail,
    }
    (output_dir / "motion_scan_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n"
    )
    print(f"WROTE {csv_path}")
    print(f"WROTE {output_dir / 'motion_scan_summary.json'}")


if __name__ == "__main__":
    main()
