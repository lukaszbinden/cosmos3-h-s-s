#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build versioned CMR clean-window catalogs for Cosmos3 training.

The CMR release has no failure/recovery, manual-intervention, or tool-exchange
annotations.  This builder is consequently fail-closed:

* ``--tier candidate`` applies conservative telemetry rules and writes a review
  queue, but marks the artifact ``training_authorized: false``.
* ``--tier strict`` additionally requires a JSONL review entry for every
  included episode/segment and is the only tier accepted by the training
  loader.

Review JSONL schema (raw frame intervals are half-open)::

    {"procedure":"cholecystectomy","episode_index":12,
     "review_status":"verified","outcome":"success",
     "manual_activity":"absent","tool_exchange":"absent",
     "clean_intervals":[[0,1800]],"reviewer":"initials",
     "source":"visual-review-v1"}

An episode labelled ``failure``, ``recovery``, ``manual_activity=present``, or
``tool_exchange=present`` contributes no strict windows.  ``unknown`` also
contributes none.  A window is included only when its entire raw 60 Hz span is
inside a reviewed clean interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ARROW_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import numpy as np

_CMR_BASE = (
    "/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/"
    "datasets/Open-H/cmr-surgical-60hz-fixed"
)
DEFAULT_DATASET_PATHS = [
    f"{_CMR_BASE}/cholecystectomy",
    f"{_CMR_BASE}/hysterectomy",
    f"{_CMR_BASE}/inguinal_hernia",
    f"{_CMR_BASE}/prostatectomy",
]
SCHEMA_VERSION = 1
CATALOG_KIND = "cmr_clean_windows"
REQUIRED_REVIEW_FIELDS = [
    "outcome",
    "manual_activity",
    "tool_exchange",
    "clean_intervals",
]
VALID_OUTCOMES = {"success", "failure", "recovery", "unknown"}
VALID_PRESENCE = {"absent", "present", "unknown"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_delta_indices(num_frames: int, timestep_interval: int) -> list[int]:
    if num_frames < 2:
        raise ValueError("--num-frames must be at least 2")
    if timestep_interval < 1:
        raise ValueError("--timestep-interval must be positive")
    return [index * timestep_interval for index in range(num_frames - 1)]


def _feature_indices(
    info: dict[str, Any],
    feature: str,
    required_names: list[str],
) -> list[int]:
    names = info.get("features", {}).get(feature, {}).get("names", [])
    missing = [name for name in required_names if name not in names]
    if missing:
        raise ValueError(
            f"{feature} is missing required CMR fields {missing}; found {names}"
        )
    return [names.index(name) for name in required_names]


def _window_has_event(event: np.ndarray, width: int, num_starts: int) -> np.ndarray:
    """Return whether each contiguous window contains any True event."""

    prefix = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(event, dtype=np.int64))
    )
    starts = np.arange(num_starts)
    return (prefix[starts + width] - prefix[starts]) > 0


def _window_has_change(
    values: np.ndarray,
    width: int,
    num_starts: int,
) -> np.ndarray:
    """Return whether any selected column changes inside each window."""

    if width <= 1:
        return np.zeros(num_starts, dtype=bool)
    changes = np.any(values[1:] != values[:-1], axis=1)
    return _window_has_event(changes, width - 1, num_starts)


def _compress_indices(indices: np.ndarray) -> list[tuple[int, int]]:
    """Compress sorted integer indices into half-open contiguous ranges."""

    if not len(indices):
        return []
    boundaries = np.flatnonzero(np.diff(indices) != 1) + 1
    groups = np.split(indices, boundaries)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def _review_allowed_starts(
    *,
    num_starts: int,
    max_delta: int,
    clean_intervals: list[list[int]],
) -> np.ndarray:
    allowed = np.zeros(num_starts, dtype=bool)
    for raw_interval in clean_intervals:
        if not isinstance(raw_interval, list) or len(raw_interval) != 2:
            raise ValueError(f"Invalid clean interval {raw_interval!r}")
        start, stop = (int(raw_interval[0]), int(raw_interval[1]))
        if start < 0 or stop <= start:
            raise ValueError(f"Invalid clean interval [{start}, {stop})")
        base_stop = min(num_starts, stop - max_delta)
        if base_stop > start:
            allowed[start:base_stop] = True
    return allowed


def _load_reviews(path: Path | None) -> tuple[dict[tuple[str, int], dict], str | None]:
    if path is None:
        return {}, None
    reviews: dict[tuple[str, int], dict] = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            try:
                key = (str(entry["procedure"]), int(entry["episode_index"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_number}: procedure and integer episode_index "
                    "are required"
                ) from exc
            if key in reviews:
                raise ValueError(f"{path}:{line_number}: duplicate review for {key}")
            reviews[key] = entry
    return reviews, sha256_file(path)


def _strict_intervals(entry: dict | None) -> tuple[list[list[int]], str]:
    if entry is None:
        return [], "unknown"
    status = str(entry.get("review_status", "unknown")).lower()
    outcome = str(entry.get("outcome", "unknown")).lower()
    manual = str(entry.get("manual_activity", "unknown")).lower()
    tool = str(entry.get("tool_exchange", "unknown")).lower()
    if status != "verified":
        return [], "unknown"
    if outcome not in VALID_OUTCOMES:
        raise ValueError(f"Invalid review outcome {outcome!r}")
    if manual not in VALID_PRESENCE or tool not in VALID_PRESENCE:
        raise ValueError(
            f"manual_activity/tool_exchange must be one of {sorted(VALID_PRESENCE)}"
        )
    if outcome in {"failure", "recovery"}:
        return [], outcome
    if outcome != "success" or manual != "absent" or tool != "absent":
        if manual == "present":
            return [], "manual_activity"
        if tool == "present":
            return [], "tool_exchange"
        return [], "unknown"
    intervals = entry.get("clean_intervals")
    if not isinstance(intervals, list) or not intervals:
        return [], "unknown"
    return intervals, "verified_clean"


def _episode_parquet_path(
    dataset_path: Path,
    info: dict[str, Any],
    episode_index: int,
) -> Path:
    return dataset_path / info["data_path"].format(
        episode_chunk=episode_index // int(info["chunks_size"]),
        episode_index=episode_index,
    )


def _scan_episode(
    *,
    dataset_path_str: str,
    procedure: str,
    info: dict[str, Any],
    episode: dict[str, Any],
    max_delta: int,
    state_event_indices: list[int],
    state_constant_indices: list[int],
    action_clutch_indices: list[int],
    tier: str,
    review: dict | None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path_str)
    episode_index = int(episode["episode_index"])
    declared_length = int(episode["length"])
    parquet_path = _episode_parquet_path(dataset_path, info, episode_index)
    result: dict[str, Any] = {
        "episode_index": episode_index,
        "ranges": [],
        "effective_windows": max(0, declared_length - max_delta),
        "telemetry_candidate_windows": 0,
        "valid_windows": 0,
        "rule_hits": {},
        "semantic_status": "not_applicable",
        "error": None,
    }
    if not parquet_path.is_file():
        result["error"] = f"missing parquet: {parquet_path}"
        return result
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(
            parquet_path,
            columns=["observation.state", "action"],
            use_threads=False,
        )
        state = np.asarray(
            table.column("observation.state").to_pylist(), dtype=np.float64
        )
        action = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
    # Arrow surfaces schema, IO, and native decoder failures through several
    # unrelated exception classes. Return the episode-scoped error so the
    # parent can refuse the entire partial catalog with useful context.
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    usable_length = min(declared_length, len(state), len(action))
    num_starts = max(0, usable_length - max_delta)
    result["effective_windows"] = num_starts
    if not num_starts:
        return result

    width = max_delta + 1
    selected_state_events = state[:usable_length, state_event_indices]
    selected_action_clutch = action[:usable_length, action_clutch_indices]
    selected_constants = state[:usable_length, state_constant_indices]

    nonfinite = (
        ~np.all(np.isfinite(selected_state_events), axis=1)
        | ~np.all(np.isfinite(selected_action_clutch), axis=1)
        | ~np.all(np.isfinite(selected_constants), axis=1)
    )
    disengaged_left = selected_state_events[:, 0] < 0.5
    disengaged_right = selected_state_events[:, 1] < 0.5
    unengageable_left = selected_state_events[:, 2] < 0.5
    unengageable_right = selected_state_events[:, 3] < 0.5
    clutch_pressed = np.any(selected_action_clutch > 0.5, axis=1)

    hit_nonfinite = _window_has_event(nonfinite, width, num_starts)
    hit_left = _window_has_event(disengaged_left, width, num_starts)
    hit_right = _window_has_event(disengaged_right, width, num_starts)
    hit_unengageable_left = _window_has_event(unengageable_left, width, num_starts)
    hit_unengageable_right = _window_has_event(unengageable_right, width, num_starts)
    hit_clutch = _window_has_event(clutch_pressed, width, num_starts)
    hit_mapping_change = _window_has_change(
        selected_constants[:, :2], width, num_starts
    )
    hit_instrument_change = _window_has_change(
        selected_constants[:, 2:], width, num_starts
    )

    valid = ~(
        hit_nonfinite
        | hit_left
        | hit_right
        | hit_unengageable_left
        | hit_unengageable_right
        | hit_clutch
        | hit_mapping_change
        | hit_instrument_change
    )
    result["telemetry_candidate_windows"] = int(valid.sum())
    result["rule_hits"] = {
        "nonfinite_telemetry": int(hit_nonfinite.sum()),
        "left_disengaged_any_raw_frame": int(hit_left.sum()),
        "right_disengaged_any_raw_frame": int(hit_right.sum()),
        "left_unengageable_any_raw_frame": int(hit_unengageable_left.sum()),
        "right_unengageable_any_raw_frame": int(hit_unengageable_right.sum()),
        "clutch_pressed_any_raw_frame": int(hit_clutch.sum()),
        "arm_mapping_changed": int(hit_mapping_change.sum()),
        "instrument_type_changed": int(hit_instrument_change.sum()),
    }

    if tier == "strict":
        clean_intervals, semantic_status = _strict_intervals(review)
        result["semantic_status"] = semantic_status
        if semantic_status != "verified_clean":
            valid[:] = False
        else:
            valid &= _review_allowed_starts(
                num_starts=num_starts,
                max_delta=max_delta,
                clean_intervals=clean_intervals,
            )

    indices = np.flatnonzero(valid)
    result["ranges"] = _compress_indices(indices)
    result["valid_windows"] = len(indices)
    return result


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _build_procedure(
    *,
    dataset_path: Path,
    output_dir: Path,
    tier: str,
    reviews: dict[tuple[str, int], dict],
    max_delta: int,
    num_workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    procedure = dataset_path.name
    info_path = dataset_path / "meta/info.json"
    episodes_path = dataset_path / "meta/episodes.jsonl"
    with info_path.open() as stream:
        info = json.load(stream)
    with episodes_path.open() as stream:
        episodes = [json.loads(line) for line in stream if line.strip()]

    state_event_indices = _feature_indices(
        info,
        "observation.state",
        [
            "hapticengaged_left",
            "hapticengaged_right",
            "haptic_left_armengageable",
            "haptic_right_armengageable",
        ],
    )
    state_constant_indices = _feature_indices(
        info,
        "observation.state",
        [
            "armlinkedtohaptic_left",
            "armlinkedtohaptic_right",
            "arm_0_instrtype",
            "arm_1_instrtype",
            "arm_2_instrtype",
            "arm_3_instrtype",
            "arm_4_instrtype",
            "instrtype_left",
            "instrtype_right",
        ],
    )
    action_clutch_indices = _feature_indices(
        info,
        "action",
        ["clutchBtn_left", "clutchBtn_right"],
    )

    results: list[dict[str, Any]] = []
    kwargs = {
        "dataset_path_str": str(dataset_path),
        "procedure": procedure,
        "info": info,
        "max_delta": max_delta,
        "state_event_indices": state_event_indices,
        "state_constant_indices": state_constant_indices,
        "action_clutch_indices": action_clutch_indices,
        "tier": tier,
    }
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _scan_episode,
                episode=episode,
                review=reviews.get((procedure, int(episode["episode_index"]))),
                **kwargs,
            ): int(episode["episode_index"])
            for episode in episodes
        }
        for completed, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if completed % 500 == 0 or completed == len(futures):
                print(
                    f"[{procedure}] scanned {completed:,}/{len(futures):,} episodes",
                    flush=True,
                )
    results.sort(key=lambda item: item["episode_index"])

    errors = [item for item in results if item["error"]]
    if errors:
        preview = "\n".join(
            f"  episode {item['episode_index']}: {item['error']}"
            for item in errors[:10]
        )
        raise RuntimeError(
            f"{procedure}: {len(errors)} episode scan(s) failed; refusing to "
            f"publish a partial catalog:\n{preview}"
        )

    episode_ids: list[int] = []
    starts: list[int] = []
    stops: list[int] = []
    for item in results:
        for start, stop in item["ranges"]:
            episode_ids.append(int(item["episode_index"]))
            starts.append(int(start))
            stops.append(int(stop))

    ranges_name = f"{procedure}.npz"
    ranges_path = output_dir / ranges_name
    np.savez_compressed(
        ranges_path,
        episode_id=np.asarray(episode_ids, dtype=np.int32),
        start=np.asarray(starts, dtype=np.int32),
        stop=np.asarray(stops, dtype=np.int32),
    )
    effective_windows = sum(item["effective_windows"] for item in results)
    eligible_windows = sum(item["valid_windows"] for item in results)
    rule_hits = {
        key: sum(item["rule_hits"].get(key, 0) for item in results)
        for key in (
            "nonfinite_telemetry",
            "left_disengaged_any_raw_frame",
            "right_disengaged_any_raw_frame",
            "left_unengageable_any_raw_frame",
            "right_unengageable_any_raw_frame",
            "clutch_pressed_any_raw_frame",
            "arm_mapping_changed",
            "instrument_type_changed",
        )
    }
    semantic_counts: dict[str, int] = {}
    for item in results:
        status = item["semantic_status"]
        semantic_counts[status] = semantic_counts.get(status, 0) + 1

    entry = {
        "dataset_path": str(dataset_path),
        "dataset_fingerprint": {
            "meta/info.json": sha256_file(info_path),
            "meta/episodes.jsonl": sha256_file(episodes_path),
        },
        "episodes": len(episodes),
        "effective_windows": effective_windows,
        "eligible_windows": eligible_windows,
        "eligible_fraction": (
            eligible_windows / effective_windows if effective_windows else 0.0
        ),
        "eligible_episodes": sum(item["valid_windows"] > 0 for item in results),
        "telemetry_candidate_episodes": sum(
            item["telemetry_candidate_windows"] > 0 for item in results
        ),
        "unreviewed_candidate_episodes": sum(
            item["telemetry_candidate_windows"] > 0
            and item["semantic_status"] == "unknown"
            for item in results
        ),
        "ranges": len(episode_ids),
        "ranges_file": ranges_name,
        "ranges_sha256": sha256_file(ranges_path),
        "rule_hit_counts_overlapping": rule_hits,
        "semantic_episode_counts": semantic_counts,
    }
    review_queue = [
        {
            "procedure": procedure,
            "episode_index": item["episode_index"],
            "telemetry_candidate_windows": item["telemetry_candidate_windows"],
            "effective_windows": item["effective_windows"],
            "review_status": "unknown",
            "outcome": "unknown",
            "manual_activity": "unknown",
            "tool_exchange": "unknown",
            "clean_intervals": [],
        }
        for item in results
        if item["telemetry_candidate_windows"] > 0
    ]
    return entry, review_queue


def build_catalog(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output)
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Output already exists: {output_dir}; pass --force to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )

    review_path = Path(args.review_labels) if args.review_labels else None
    reviews, review_sha = _load_reviews(review_path)
    if args.tier == "strict" and not reviews:
        raise ValueError(
            "--tier strict requires non-empty --review-labels; CMR has no native "
            "failure/recovery/manual/tool-exchange annotations"
        )

    deltas = action_delta_indices(args.num_frames, args.timestep_interval)
    max_delta = max(deltas)
    procedures: dict[str, Any] = {}
    review_queue: list[dict[str, Any]] = []
    try:
        for raw_path in args.dataset_path:
            dataset_path = Path(raw_path)
            print(f"Building {args.tier} catalog for {dataset_path}", flush=True)
            entry, queue = _build_procedure(
                dataset_path=dataset_path,
                output_dir=temporary_dir,
                tier=args.tier,
                reviews=reviews,
                max_delta=max_delta,
                num_workers=args.workers,
            )
            procedures[dataset_path.name] = entry
            review_queue.extend(queue)

        reviewed_keys = {
            (procedure, episode)
            for (procedure, episode), entry in reviews.items()
            if str(entry.get("review_status", "")).lower() == "verified"
        }
        known_episode_keys = {
            (procedure, int(item["episode_index"]))
            for procedure in procedures
            for item in review_queue
            if item["procedure"] == procedure
        }
        unknown_episodes = 0
        if args.tier == "strict":
            # Only episodes with telemetry-clean candidates require semantic
            # review. Fully telemetry-rejected episodes cannot contribute a
            # training window regardless of their semantic label.
            unknown_episodes = sum(
                entry["unreviewed_candidate_episodes"] for entry in procedures.values()
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "catalog_kind": CATALOG_KIND,
            "catalog_tier": args.tier,
            "training_authorized": args.tier == "strict" and unknown_episodes == 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "builder_script_sha256": sha256_file(Path(__file__).resolve()),
            "horizon": {
                "num_video_frames": args.num_frames,
                "timestep_interval": args.timestep_interval,
                "action_delta_indices": deltas,
                "raw_window_width": max_delta + 1,
                "policy": "every raw 60 Hz frame, not only sampled action frames",
            },
            "criteria": {
                "both_haptics_engaged_every_raw_frame": True,
                "both_selected_arms_engageable_every_raw_frame": True,
                "both_clutch_buttons_released_every_raw_frame": True,
                "arm_mapping_constant": True,
                "all_instrument_types_constant": True,
                "finite_required_telemetry": True,
                "failure_recovery_exclusion": "explicit review labels required",
                "manual_activity_exclusion": "explicit visual/review labels required",
                "tool_exchange_exclusion": (
                    "instrument telemetry transition filter plus explicit "
                    "visual/review labels"
                ),
            },
            "review": {
                "labels_path": str(review_path) if review_path else None,
                "labels_sha256": review_sha,
                "required_fields": REQUIRED_REVIEW_FIELDS,
                "verified_entries": len(reviewed_keys),
                "unknown_episodes": unknown_episodes,
                "known_candidate_episode_keys": len(known_episode_keys),
            },
            "procedures": procedures,
        }
        identity_payload = {
            key: manifest[key]
            for key in (
                "schema_version",
                "catalog_kind",
                "catalog_tier",
                "training_authorized",
                "builder_script_sha256",
                "horizon",
                "criteria",
                "review",
                "procedures",
            )
        }
        manifest["catalog_id"] = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        _atomic_write_json(temporary_dir / "manifest.json", manifest)
        if args.tier == "candidate":
            with (temporary_dir / "review_queue.jsonl").open("w") as stream:
                for entry in review_queue:
                    stream.write(json.dumps(entry, sort_keys=True) + "\n")
        temporary_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=None,
        help="CMR procedure leaf; repeat for multiple leaves (default: all four)",
    )
    parser.add_argument("--output", required=True, help="Catalog output directory")
    parser.add_argument(
        "--tier",
        choices=["candidate", "strict"],
        default="candidate",
        help="candidate is review-only; strict requires verified labels",
    )
    parser.add_argument(
        "--review-labels",
        help="JSONL semantic/visual reviews (required for strict tier)",
    )
    parser.add_argument("--num-frames", type=int, default=13)
    parser.add_argument("--timestep-interval", type=int, default=6)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 8, 64),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.dataset_path is None:
        args.dataset_path = list(DEFAULT_DATASET_PATHS)
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main() -> None:
    try:
        output = build_catalog(parse_args())
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise
    print(f"[SUCCESS] catalog: {output}")


if __name__ == "__main__":
    main()
