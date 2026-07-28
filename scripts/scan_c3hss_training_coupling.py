#!/usr/bin/env python3
"""Scan dVRK training episodes for action/state/video coupling.

This diagnostic works directly from LeRobot parquet and MP4 files.  It avoids
the Cosmos training stack so that arm ordering, timing, and cross-arm coupling
can be checked independently of model inference.

For every eligible CAMP H=16 + N=12 window it records:

* PSM1/PSM2 action and measured-state motion;
* same-arm and cross-arm command/state agreement;
* arm co-activity and task/episode phase;
* low-resolution visual motion in the endoscope and both wrist views; and
* a transparent visibility proxy (visual motion per millimetre of state motion).

The visual metric is not a semantic tool segmentation.  It is a cheap,
dataset-wide motion proxy based on brightness-centred frame differences and
dense optical flow at 160x90.  Tool-specific SAM analysis remains the right
follow-up for individual outliers.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

ARMS = ("psm1", "psm2")
ARM_OFFSETS = {"psm1": 0, "psm2": 8}
VIEWS = {
    "endoscope": "observation.images.endoscope.left",
    "wrist_left": "observation.images.wrist.left",
    "wrist_right": "observation.images.wrist.right",
}
EXPECTED_ACTION_NAMES = {
    "psm1": (
        "psm1_sp.position.x",
        "psm1_sp.position.y",
        "psm1_sp.position.z",
        "psm1_sp.orientation.x",
        "psm1_sp.orientation.y",
        "psm1_sp.orientation.z",
        "psm1_sp.orientation.w",
        "psm1_jaw_sp",
    ),
    "psm2": (
        "psm2_sp.position.x",
        "psm2_sp.position.y",
        "psm2_sp.position.z",
        "psm2_sp.orientation.x",
        "psm2_sp.orientation.y",
        "psm2_sp.orientation.z",
        "psm2_sp.orientation.w",
        "psm2_jaw_sp",
    ),
}
EXPECTED_STATE_NAMES = {
    "psm1": (
        "psm1_pose.position.x",
        "psm1_pose.position.y",
        "psm1_pose.position.z",
        "psm1_pose.orientation.x",
        "psm1_pose.orientation.y",
        "psm1_pose.orientation.z",
        "psm1_pose.orientation.w",
        "psm1_jaw",
    ),
    "psm2": (
        "psm2_pose.position.x",
        "psm2_pose.position.y",
        "psm2_pose.position.z",
        "psm2_pose.orientation.x",
        "psm2_pose.orientation.y",
        "psm2_pose.orientation.z",
        "psm2_pose.orientation.w",
        "psm2_jaw",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--timestep-interval", type=int, default=3)
    parser.add_argument("--current-actions", type=int, default=12)
    parser.add_argument("--history-actions", type=int, default=16)
    parser.add_argument("--base-step", type=int, default=3)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--visual-width", type=int, default=160)
    parser.add_argument("--visual-height", type=int, default=90)
    parser.add_argument("--maximum-lag", type=int, default=5)
    parser.add_argument(
        "--extra-window",
        action="append",
        default=[],
        help="Episode:base pair outside the selected split, e.g. 1382:381.",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Run the action/state portion without decoding MP4s.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_split(spec: str) -> list[int]:
    result = []
    for section in spec.split(","):
        start, stop = (int(value) for value in section.strip().split(":"))
        result.extend(range(start, stop))
    return result


def _parse_extra_windows(specs: list[str]) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for spec in specs:
        episode, base = (int(value) for value in spec.split(":"))
        result.setdefault(episode, set()).add(base)
    return result


def _feature_names(info: dict[str, Any], key: str) -> tuple[str, ...]:
    names = info["features"][key]["names"]
    if len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    return tuple(names)


def _validate_layout(
    info: dict[str, Any], modality: dict[str, Any]
) -> dict[str, Any]:
    action_names = _feature_names(info, "action")
    state_names = _feature_names(info, "observation.state")
    expected_action = EXPECTED_ACTION_NAMES["psm1"] + EXPECTED_ACTION_NAMES["psm2"]
    expected_state = EXPECTED_STATE_NAMES["psm1"] + EXPECTED_STATE_NAMES["psm2"]
    if action_names != expected_action:
        raise ValueError(f"raw action names differ from expected dVRK order: {action_names}")
    if state_names != expected_state:
        raise ValueError(f"raw state names differ from expected dVRK order: {state_names}")
    expected_slices = {
        "psm1_pose": (0, 7),
        "psm1_gripper": (7, 8),
        "psm2_pose": (8, 15),
        "psm2_gripper": (15, 16),
    }
    original_keys = {}
    for section_name in ("action", "state"):
        section = modality[section_name]
        original_keys[section_name] = {}
        for key, expected in expected_slices.items():
            entry = section[key]
            actual = (int(entry["start"]), int(entry["end"]))
            if actual != expected:
                raise ValueError(
                    f"{section_name}.{key} slice {actual} != expected {expected}"
                )
            original_keys[section_name][key] = entry.get(
                "original_key",
                "action" if section_name == "action" else "observation.state",
            )
    return {
        "raw_action_names": list(action_names),
        "raw_state_names": list(state_names),
        "raw_slices": {key: list(value) for key, value in expected_slices.items()},
        "original_keys": original_keys,
        "model_action_layout": {
            "psm1_translation": [0, 3],
            "psm1_rotation_6d": [3, 9],
            "psm1_gripper": [9, 10],
            "psm2_translation": [10, 13],
            "psm2_rotation_6d": [13, 19],
            "psm2_gripper": [19, 20],
        },
    }


def _episode_path(root: Path, info: dict[str, Any], episode_id: int) -> Path:
    relative = info["data_path"].format(
        episode_chunk=episode_id // int(info["chunks_size"]),
        episode_index=episode_id,
    )
    return root / relative


def _video_path(
    root: Path, info: dict[str, Any], episode_id: int, video_key: str
) -> Path:
    relative = info["video_path"].format(
        episode_chunk=episode_id // int(info["chunks_size"]),
        episode_index=episode_id,
        video_key=video_key,
    )
    return root / relative


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    left = left[valid]
    right = right[valid]
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _mean_vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = np.isfinite(denominator) & (denominator > 1e-12)
    if not np.any(valid):
        return float("nan")
    cosine = np.sum(left[valid] * right[valid], axis=1) / denominator[valid]
    return float(np.mean(cosine))


def _phase(progress: float) -> str:
    if progress < 1.0 / 3.0:
        return "early"
    if progress < 2.0 / 3.0:
        return "middle"
    return "late"


def _dominant_arm(psm1_mm: float, psm2_mm: float) -> str:
    threshold = 0.1
    if psm1_mm < threshold and psm2_mm < threshold:
        return "parked"
    if psm1_mm >= 3.0 * max(psm2_mm, 1e-12):
        return "psm1"
    if psm2_mm >= 3.0 * max(psm1_mm, 1e-12):
        return "psm2"
    return "coactive"


def _align(left: np.ndarray, right: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Align left(t) with right(t + lag); positive lag means right is later."""
    if lag > 0:
        return left[:-lag], right[lag:]
    if lag < 0:
        return left[-lag:], right[:lag]
    return left, right


def _decode_visual_steps(
    path: Path,
    raw_rows: int,
    interval: int,
    width: int,
    height: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cv2.setNumThreads(1)
    sampled = []
    decoded_frames = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame_index, frame in enumerate(container.decode(stream)):
            decoded_frames += 1
            if frame_index % interval:
                continue
            gray = frame.reformat(width=width, height=height, format="gray")
            sampled.append(gray.to_ndarray().astype(np.float32))
    if len(sampled) < 2:
        raise ValueError(f"{path} decoded only {len(sampled)} sampled frames")
    energies = []
    flow_magnitude = []
    flow_x = []
    flow_y = []
    previous = sampled[0]
    for current in sampled[1:]:
        previous_centered = previous - float(np.mean(previous))
        current_centered = current - float(np.mean(current))
        energies.append(
            float(np.mean(np.abs(current_centered - previous_centered)) / 255.0)
        )
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=15,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )
        magnitudes = np.linalg.norm(flow, axis=2)
        flow_magnitude.append(float(np.median(magnitudes)))
        flow_x.append(float(np.median(flow[:, :, 0])))
        flow_y.append(float(np.median(flow[:, :, 1])))
        previous = current
    expected_sampled = (raw_rows - 1) // interval + 1
    audit = {
        "path": str(path),
        "raw_rows": raw_rows,
        "decoded_frames": decoded_frames,
        "sampled_frames": len(sampled),
        "expected_sampled_frames": expected_sampled,
        "sample_count_error": len(sampled) - expected_sampled,
    }
    return {
        "energy": np.asarray(energies, dtype=np.float32),
        "flow_magnitude": np.asarray(flow_magnitude, dtype=np.float32),
        "flow_x": np.asarray(flow_x, dtype=np.float32),
        "flow_y": np.asarray(flow_y, dtype=np.float32),
    }, audit


def _episode_worker(payload: dict[str, Any]) -> dict[str, Any]:
    root = Path(payload["root"])
    info = payload["info"]
    episode_id = int(payload["episode_id"])
    interval = int(payload["interval"])
    current_actions = int(payload["current_actions"])
    history_actions = int(payload["history_actions"])
    base_step = int(payload["base_step"])
    extra_bases = set(payload["extra_bases"])
    parquet_path = _episode_path(root, info, episode_id)
    table = pq.read_table(
        parquet_path,
        columns=[
            "action",
            "observation.state",
            "timestamp",
            "frame_index",
            "episode_index",
            "task_index",
        ],
    )
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    frame_index = table["frame_index"].to_numpy().astype(np.int64)
    episode_index = table["episode_index"].to_numpy().astype(np.int64)
    task_indices = table["task_index"].to_numpy().astype(np.int64)
    if action.shape != state.shape or action.ndim != 2 or action.shape[1] != 16:
        raise ValueError(f"{parquet_path}: action/state {action.shape}/{state.shape}")
    if not np.array_equal(frame_index, np.arange(len(frame_index))):
        raise ValueError(f"{parquet_path}: non-contiguous frame_index")
    if not np.all(episode_index == episode_id):
        raise ValueError(f"{parquet_path}: wrong episode_index")
    if len(set(task_indices.tolist())) != 1:
        raise ValueError(f"{parquet_path}: task_index changes inside episode")
    task_index = int(task_indices[0])

    sampled_indices = np.arange(0, len(action), interval, dtype=np.int64)
    action_steps = {}
    state_steps = {}
    for arm in ARMS:
        offset = ARM_OFFSETS[arm]
        action_steps[arm] = np.diff(action[sampled_indices, offset : offset + 3], axis=0)
        state_steps[arm] = np.diff(state[sampled_indices, offset : offset + 3], axis=0)

    visual: dict[str, dict[str, np.ndarray]] = {}
    video_audit = {}
    if not payload["skip_video"]:
        for view_name, video_key in VIEWS.items():
            values, audit = _decode_visual_steps(
                _video_path(root, info, episode_id, video_key),
                len(action),
                interval,
                int(payload["visual_width"]),
                int(payload["visual_height"]),
            )
            visual[view_name] = values
            video_audit[view_name] = audit

    common_steps = min(
        [len(action_steps[arm]) for arm in ARMS]
        + [
            len(values["energy"])
            for values in visual.values()
        ]
    )
    sampled_transition_raw_indices = sampled_indices[:common_steps]
    step_arrays: dict[str, np.ndarray] = {
        "episode_id": np.full(common_steps, episode_id, dtype=np.int32),
        "task_index": np.full(common_steps, task_index, dtype=np.int16),
        "raw_index": sampled_transition_raw_indices.astype(np.int32),
        "progress": (
            sampled_transition_raw_indices / max(len(action) - 1, 1)
        ).astype(np.float32),
    }
    for arm in ARMS:
        step_arrays[f"{arm}_action_velocity"] = action_steps[arm][:common_steps]
        step_arrays[f"{arm}_state_velocity"] = state_steps[arm][:common_steps]
        step_arrays[f"{arm}_action_speed_mm"] = (
            np.linalg.norm(action_steps[arm][:common_steps], axis=1) * 1000.0
        ).astype(np.float32)
        step_arrays[f"{arm}_state_speed_mm"] = (
            np.linalg.norm(state_steps[arm][:common_steps], axis=1) * 1000.0
        ).astype(np.float32)
    for view_name, values in visual.items():
        for metric_name, metric_values in values.items():
            step_arrays[f"{view_name}_{metric_name}"] = metric_values[:common_steps]

    first_base = history_actions * interval
    last_base_exclusive = len(action) - current_actions * interval
    bases = list(range(first_base, last_base_exclusive, base_step))
    for extra_base in extra_bases:
        if extra_base not in bases:
            bases.append(extra_base)
    windows = []
    for base in sorted(bases):
        if base < 0 or base + current_actions * interval >= len(action):
            raise ValueError(f"{episode_id}:{base} cannot fit the configured window")
        sample_start = base // interval
        sample_stop = sample_start + current_actions
        if base % interval or sample_stop > common_steps:
            raise ValueError(f"{episode_id}:{base} does not align to decoded steps")
        action_window = {
            arm: action_steps[arm][sample_start:sample_stop] for arm in ARMS
        }
        state_window = {
            arm: state_steps[arm][sample_start:sample_stop] for arm in ARMS
        }
        action_speed = {
            arm: np.linalg.norm(value, axis=1) * 1000.0
            for arm, value in action_window.items()
        }
        state_speed = {
            arm: np.linalg.norm(value, axis=1) * 1000.0
            for arm, value in state_window.items()
        }
        progress = base / max(len(action) - 1, 1)
        psm1_mean = float(np.mean(action_speed["psm1"]))
        psm2_mean = float(np.mean(action_speed["psm2"]))
        row: dict[str, Any] = {
            "episode_id": episode_id,
            "base_index": base,
            "is_extra_window": int(base in extra_bases),
            "task_index": task_index,
            "progress": progress,
            "phase": _phase(progress),
            "dominant_arm": _dominant_arm(psm1_mean, psm2_mean),
            "psm1_to_psm2_action_motion_ratio": psm1_mean
            / max(psm2_mean, 1e-12),
            "psm2_to_psm1_action_motion_ratio": psm2_mean
            / max(psm1_mean, 1e-12),
            "psm1_psm2_action_speed_correlation": _safe_corr(
                action_speed["psm1"], action_speed["psm2"]
            ),
            "psm1_psm2_state_speed_correlation": _safe_corr(
                state_speed["psm1"], state_speed["psm2"]
            ),
        }
        for arm in ARMS:
            other = "psm2" if arm == "psm1" else "psm1"
            row.update(
                {
                    f"{arm}_mean_action_translation_step_mm": float(
                        np.mean(action_speed[arm])
                    ),
                    f"{arm}_mean_state_translation_step_mm": float(
                        np.mean(state_speed[arm])
                    ),
                    f"{arm}_action_path_mm": float(np.sum(action_speed[arm])),
                    f"{arm}_state_path_mm": float(np.sum(state_speed[arm])),
                    f"{arm}_same_arm_vector_cosine": _mean_vector_cosine(
                        action_window[arm], state_window[arm]
                    ),
                    f"{arm}_cross_arm_vector_cosine": _mean_vector_cosine(
                        action_window[arm], state_window[other]
                    ),
                    f"{arm}_same_arm_speed_correlation": _safe_corr(
                        action_speed[arm], state_speed[arm]
                    ),
                    f"{arm}_cross_arm_speed_correlation": _safe_corr(
                        action_speed[arm], state_speed[other]
                    ),
                }
            )
        for view_name, values in visual.items():
            visual_slice = slice(sample_start, sample_stop)
            energy = values["energy"][visual_slice]
            flow = values["flow_magnitude"][visual_slice]
            combined_state_motion = state_speed["psm1"] + state_speed["psm2"]
            row[f"{view_name}_energy_mean"] = float(np.mean(energy))
            row[f"{view_name}_flow_magnitude_mean"] = float(np.mean(flow))
            row[f"{view_name}_flow_per_combined_state_mm"] = float(
                np.mean(flow) / max(float(np.mean(combined_state_motion)), 1e-12)
            )
            for arm in ARMS:
                row[f"{view_name}_{arm}_state_speed_correlation"] = _safe_corr(
                    flow, state_speed[arm]
                )
                row[f"{view_name}_{arm}_action_speed_correlation"] = _safe_corr(
                    flow, action_speed[arm]
                )
        windows.append(row)

    return {
        "episode_id": episode_id,
        "task_index": task_index,
        "raw_rows": len(action),
        "window_rows": windows,
        "step_arrays": step_arrays,
        "video_audit": video_audit,
        "parquet_sha256": _sha256(parquet_path),
    }


def _lag_metric(
    left: np.ndarray,
    right: np.ndarray,
    episode_ids: np.ndarray,
    maximum_lag: int,
    vector: bool,
) -> dict[str, Any]:
    values = []
    for lag in range(-maximum_lag, maximum_lag + 1):
        aligned_left, aligned_right = _align(left, right, lag)
        aligned_left_episodes, aligned_right_episodes = _align(
            episode_ids, episode_ids, lag
        )
        same_episode = aligned_left_episodes == aligned_right_episodes
        aligned_left = aligned_left[same_episode]
        aligned_right = aligned_right[same_episode]
        value = (
            _mean_vector_cosine(aligned_left, aligned_right)
            if vector
            else _safe_corr(aligned_left, aligned_right)
        )
        values.append({"lag_model_frames": lag, "value": value})
    finite = [item for item in values if math.isfinite(item["value"])]
    best = max(finite, key=lambda item: item["value"]) if finite else None
    return {"by_lag": values, "best": best}


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


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    info_path = root / "meta/info.json"
    modality_path = root / "meta/modality.json"
    info = _load_json(info_path)
    modality = _load_json(modality_path)
    layout_audit = _validate_layout(info, modality)
    task_records = [
        json.loads(line)
        for line in (root / "meta/tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    task_names = {
        int(record["task_index"]): record["task"] for record in task_records
    }
    split_episodes = _parse_split(info["splits"][args.split])
    extra_windows = _parse_extra_windows(args.extra_window)
    all_episodes = split_episodes + sorted(set(extra_windows) - set(split_episodes))
    payloads = [
        {
            "root": str(root),
            "info": info,
            "episode_id": episode_id,
            "interval": args.timestep_interval,
            "current_actions": args.current_actions,
            "history_actions": args.history_actions,
            "base_step": args.base_step,
            "extra_bases": sorted(extra_windows.get(episode_id, set())),
            "skip_video": args.skip_video,
            "visual_width": args.visual_width,
            "visual_height": args.visual_height,
        }
        for episode_id in all_episodes
    ]

    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.workers)
    ) as executor:
        futures = {
            executor.submit(_episode_worker, payload): payload["episode_id"]
            for payload in payloads
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            episode_id = futures[future]
            try:
                result = future.result()
            except Exception as error:
                raise RuntimeError(f"episode {episode_id} failed") from error
            results.append(result)
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"SCANNED {completed}/{len(futures)} episodes "
                    f"(latest={episode_id})",
                    flush=True,
                )
    results.sort(key=lambda item: item["episode_id"])

    train_set = set(split_episodes)
    window_rows = []
    focus_rows = []
    for result in results:
        for row in result["window_rows"]:
            row["subset"] = args.subset_name
            row["task"] = task_names.get(int(row["task_index"]), "unknown")
            if result["episode_id"] in train_set and not row["is_extra_window"]:
                window_rows.append(row)
            if row["is_extra_window"]:
                focus_rows.append(row)
    if not window_rows:
        raise RuntimeError("scan produced no training windows")
    fieldnames = list(window_rows[0])
    with (output_dir / "coupling_windows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(window_rows)
    if focus_rows:
        with (output_dir / "extra_windows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(focus_rows)

    train_results = [item for item in results if item["episode_id"] in train_set]
    step_keys = list(train_results[0]["step_arrays"])
    step_arrays = {
        key: np.concatenate([item["step_arrays"][key] for item in train_results])
        for key in step_keys
    }
    np.savez_compressed(output_dir / "coupling_steps.npz", **step_arrays)

    lag_diagnostics: dict[str, Any] = {"action_to_state": {}, "visual": {}}
    for action_arm in ARMS:
        for state_arm in ARMS:
            pair = f"{action_arm}_to_{state_arm}"
            lag_diagnostics["action_to_state"][pair] = {
                "vector_cosine": _lag_metric(
                    step_arrays[f"{action_arm}_action_velocity"],
                    step_arrays[f"{state_arm}_state_velocity"],
                    step_arrays["episode_id"],
                    args.maximum_lag,
                    vector=True,
                ),
                "speed_correlation": _lag_metric(
                    step_arrays[f"{action_arm}_action_speed_mm"],
                    step_arrays[f"{state_arm}_state_speed_mm"],
                    step_arrays["episode_id"],
                    args.maximum_lag,
                    vector=False,
                ),
            }
    if not args.skip_video:
        for view_name in VIEWS:
            lag_diagnostics["visual"][view_name] = {}
            for arm in ARMS:
                lag_diagnostics["visual"][view_name][arm] = {
                    "action_speed_to_flow": _lag_metric(
                        step_arrays[f"{arm}_action_speed_mm"],
                        step_arrays[f"{view_name}_flow_magnitude"],
                        step_arrays["episode_id"],
                        args.maximum_lag,
                        vector=False,
                    ),
                    "state_speed_to_flow": _lag_metric(
                        step_arrays[f"{arm}_state_speed_mm"],
                        step_arrays[f"{view_name}_flow_magnitude"],
                        step_arrays["episode_id"],
                        args.maximum_lag,
                        vector=False,
                    ),
                }

    episode_audit = []
    for result in results:
        episode_audit.append(
            {
                "episode_id": result["episode_id"],
                "in_split": result["episode_id"] in train_set,
                "task_index": result["task_index"],
                "task": task_names.get(result["task_index"], "unknown"),
                "raw_rows": result["raw_rows"],
                "parquet_sha256": result["parquet_sha256"],
                "video": result["video_audit"],
            }
        )
    summary = {
        "diagnostic": "full-training-split dVRK action/state/video coupling scan",
        "subset": args.subset_name,
        "dataset_root": str(root),
        "split": args.split,
        "split_spec": info["splits"][args.split],
        "episodes": len(split_episodes),
        "extra_windows": args.extra_window,
        "raw_rows": int(sum(item["raw_rows"] for item in train_results)),
        "candidate_windows": len(window_rows),
        "step_transitions": int(len(step_arrays["episode_id"])),
        "window_recipe": {
            "timestep_interval": args.timestep_interval,
            "current_actions": args.current_actions,
            "history_actions": args.history_actions,
            "base_step": args.base_step,
        },
        "visual_recipe": None
        if args.skip_video
        else {
            "views": VIEWS,
            "resolution": [args.visual_height, args.visual_width],
            "energy": "mean absolute brightness-centred frame difference / 255",
            "flow": "median Farneback magnitude",
            "semantic_tool_segmentation": False,
        },
        "layout_audit": layout_audit,
        "lag_convention": (
            "positive lag means state/video occurs later than the action; "
            "one model frame is 0.1 seconds"
        ),
        "lag_diagnostics": lag_diagnostics,
        "info_sha256": _sha256(info_path),
        "modality_sha256": _sha256(modality_path),
        "episodes_detail": episode_audit,
    }
    (output_dir / "coupling_scan_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n"
    )
    print(f"WROTE {output_dir / 'coupling_windows.csv'}")
    print(f"WROTE {output_dir / 'coupling_steps.npz'}")
    print(f"WROTE {output_dir / 'coupling_scan_summary.json'}")


if __name__ == "__main__":
    main()
