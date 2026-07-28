#!/usr/bin/env python3
"""Extract auditable raw dVRK action/state tracks from LeRobot parquet files.

This intentionally bypasses the Cosmos dataset and transform stack.  It copies
the original 16-D action and observation.state rows, timestamps, and frame
indices for selected episodes, together with the dataset's modality mapping and
source-file hashes.  The resulting NPZ/JSON pair is small enough to pull from a
cluster and compare against model-facing transformed actions and video tracks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

RAW_DIM = 16
EXPECTED_MODALITY_SLICES = {
    "psm1_pose": (0, 7),
    "psm1_gripper": (7, 8),
    "psm2_pose": (8, 15),
    "psm2_gripper": (15, 16),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _validate_modality(modality: dict[str, Any]) -> None:
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


def _episode_path(root: Path, info: dict[str, Any], episode_id: int) -> Path:
    chunks_size = int(info["chunks_size"])
    relative = info["data_path"].format(
        episode_chunk=episode_id // chunks_size,
        episode_index=episode_id,
    )
    return root / relative


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    info_path = root / "meta/info.json"
    modality_path = root / "meta/modality.json"
    info = _load_json(info_path)
    modality = _load_json(modality_path)
    _validate_modality(modality)

    arrays: dict[str, np.ndarray] = {}
    episodes_detail = []
    for episode_id in args.episodes:
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
        if action.ndim != 2 or action.shape[1] != RAW_DIM:
            raise ValueError(
                f"{parquet_path} action shape {action.shape}, expected (T, {RAW_DIM})"
            )
        if state.shape != action.shape:
            raise ValueError(
                f"{parquet_path} state shape {state.shape} != action {action.shape}"
            )
        if not np.all(episode_index == episode_id):
            raise ValueError(f"{parquet_path} contains the wrong episode_index")
        if not np.array_equal(frame_index, np.arange(len(frame_index))):
            raise ValueError(f"{parquet_path} frame_index is not contiguous from zero")

        prefix = f"episode_{episode_id:06d}"
        arrays[f"{prefix}__action"] = action
        arrays[f"{prefix}__state"] = state
        arrays[f"{prefix}__timestamp"] = timestamp
        arrays[f"{prefix}__frame_index"] = frame_index
        episodes_detail.append(
            {
                "episode_id": episode_id,
                "rows": len(action),
                "parquet_path": str(parquet_path),
                "parquet_sha256": _sha256(parquet_path),
                "timestamp_start": float(timestamp[0]),
                "timestamp_end": float(timestamp[-1]),
            }
        )

    np.savez_compressed(output, **arrays)
    manifest = {
        "diagnostic": "raw dVRK action/state extraction",
        "dataset_root": str(root),
        "dataset_robot_type": info.get("robot_type"),
        "dataset_fps": info.get("fps"),
        "info_path": str(info_path),
        "info_sha256": _sha256(info_path),
        "modality_path": str(modality_path),
        "modality_sha256": _sha256(modality_path),
        "validated_raw_layout": {
            key: list(value) for key, value in EXPECTED_MODALITY_SLICES.items()
        },
        "state_reference_source": {
            key: modality["state"][key].get(
                "original_key", "observation.state"
            )
            for key in EXPECTED_MODALITY_SLICES
        },
        "npz": str(output),
        "episodes": episodes_detail,
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(f"WROTE {output} ({len(arrays)} arrays)")
    print(f"WROTE {manifest_path}")


if __name__ == "__main__":
    main()
