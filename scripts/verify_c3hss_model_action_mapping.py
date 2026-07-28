#!/usr/bin/env python3
"""Verify raw dVRK arm ordering against a model-facing action archive.

The archive is emitted after the real Open-H dataset transform and statistics
denormalization.  Reconstructing it independently from raw parquet rows checks
both arm slices and the temporal/reference convention without importing Cosmos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--base-index", type=int, required=True)
    parser.add_argument("--timestep-interval", type=int, default=3)
    parser.add_argument("--current-actions", type=int, default=12)
    parser.add_argument("--model-action-archive", required=True)
    parser.add_argument("--registry-source", required=True)
    parser.add_argument("--physical-helper-source", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_path(root: Path, episode: int) -> Path:
    info = json.loads((root / "meta/info.json").read_text())
    return root / info["data_path"].format(
        episode_chunk=episode // int(info["chunks_size"]),
        episode_index=episode,
    )


def _rot6d_columns(matrices: np.ndarray) -> np.ndarray:
    return matrices[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)


def _rot6d_rows(matrices: np.ndarray) -> np.ndarray:
    return matrices[:, :2, :].reshape(-1, 6)


def _build_expected(
    action: np.ndarray,
    state: np.ndarray,
    base: int,
    interval: int,
    current_actions: int,
    column_convention: bool,
) -> np.ndarray:
    indices = base + np.arange(current_actions) * interval
    arms = []
    for raw_offset in (0, 8):
        reference_rotation = Rotation.from_quat(
            state[base, raw_offset + 3 : raw_offset + 7]
        ).as_matrix()
        action_rotation = Rotation.from_quat(
            action[indices, raw_offset + 3 : raw_offset + 7]
        ).as_matrix()
        relative_rotation = np.einsum(
            "ij,tjk->tik", reference_rotation.T, action_rotation
        )
        rotation_6d = (
            _rot6d_columns(relative_rotation)
            if column_convention
            else _rot6d_rows(relative_rotation)
        )
        arms.append(
            np.concatenate(
                [
                    action[indices, raw_offset : raw_offset + 3]
                    - state[base, raw_offset : raw_offset + 3],
                    rotation_6d,
                    action[indices, raw_offset + 7 : raw_offset + 8],
                ],
                axis=1,
            )
        )
    return np.concatenate(arms, axis=1)


def _maximum_error(left: np.ndarray, right: np.ndarray, slices: list[slice]) -> float:
    return float(
        max(np.max(np.abs(left[:, section] - right[:, section])) for section in slices)
    )


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
    parquet_path = _episode_path(root, args.episode)
    table = pq.read_table(parquet_path, columns=["action", "observation.state"])
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    archive_path = Path(args.model_action_archive).resolve()
    with np.load(archive_path) as archive:
        model_physical = archive["physical__correct"].astype(np.float64)
    expected_columns = _build_expected(
        action,
        state,
        args.base_index,
        args.timestep_interval,
        args.current_actions,
        column_convention=True,
    )
    expected_rows = _build_expected(
        action,
        state,
        args.base_index,
        args.timestep_interval,
        args.current_actions,
        column_convention=False,
    )
    if model_physical.shape != expected_columns.shape:
        raise ValueError(
            f"archive shape {model_physical.shape} != expected {expected_columns.shape}"
        )

    translation_slices = [slice(0, 3), slice(10, 13)]
    rotation_slices = [slice(3, 9), slice(13, 19)]
    gripper_slices = [slice(9, 10), slice(19, 20)]
    translation_error = _maximum_error(
        model_physical, expected_columns, translation_slices
    )
    rotation_column_error = _maximum_error(
        model_physical, expected_columns, rotation_slices
    )
    rotation_row_error = _maximum_error(
        model_physical, expected_rows, rotation_slices
    )
    gripper_error = _maximum_error(model_physical, expected_columns, gripper_slices)

    registry_path = Path(args.registry_source).resolve()
    helper_path = Path(args.physical_helper_source).resolve()
    registry_source = registry_path.read_text()
    helper_source = helper_path.read_text()
    registry_order_fragments = [
        '"action.psm1_pose"',
        '"action.psm1_gripper"',
        '"action.psm2_pose"',
        '"action.psm2_gripper"',
    ]
    fragment_offsets = [
        registry_source.find(fragment) for fragment in registry_order_fragments
    ]
    registry_order_present = all(offset >= 0 for offset in fragment_offsets)
    registry_ordered = registry_order_present and fragment_offsets == sorted(
        fragment_offsets
    )
    training_column_fragment = "rotation_matrices[:, :, :2].transpose(0, 2, 1)"
    helper_row_fragment = "matrices[..., :2, :]"
    training_uses_columns = training_column_fragment in registry_source
    # The actual transform implementation is normally in state_action.py, not
    # groot_configs.py. Resolve it relative to the registry when available.
    transform_path = (
        registry_path.parent
        / "data"
        / "transform"
        / "state_action.py"
    )
    if transform_path.exists():
        transform_source = transform_path.read_text()
        training_uses_columns = training_column_fragment in transform_source
    else:
        transform_source = ""
    helper_uses_rows = helper_row_fragment in helper_source

    tolerance = 1e-6
    payload = {
        "diagnostic": "raw-to-model dVRK action mapping verification",
        "episode_id": args.episode,
        "base_index": args.base_index,
        "timestep_interval": args.timestep_interval,
        "model_action_layout": {
            "psm1": {
                "translation": [0, 3],
                "rotation_6d": [3, 9],
                "gripper": [9, 10],
            },
            "psm2": {
                "translation": [10, 13],
                "rotation_6d": [13, 19],
                "gripper": [19, 20],
            },
        },
        "live_archive_comparison": {
            "shape": list(model_physical.shape),
            "translation_max_abs_error": translation_error,
            "rotation_column_convention_max_abs_error": rotation_column_error,
            "rotation_row_convention_max_abs_error": rotation_row_error,
            "gripper_max_abs_error": gripper_error,
            "translation_mapping_passed": translation_error <= tolerance,
            "rotation_column_mapping_passed": rotation_column_error <= tolerance,
            "gripper_mapping_passed": gripper_error <= tolerance,
        },
        "source_audit": {
            "registry_source": str(registry_path),
            "registry_sha256": _sha256(registry_path),
            "transform_source": str(transform_path) if transform_path.exists() else None,
            "transform_sha256": _sha256(transform_path)
            if transform_path.exists()
            else None,
            "physical_helper_source": str(helper_path),
            "physical_helper_sha256": _sha256(helper_path),
            "registry_action_key_order_present": registry_order_present,
            "registry_action_key_order_is_psm1_then_psm2": registry_ordered,
            "training_rotation_6d_uses_columns": training_uses_columns,
            "physical_intervention_helper_uses_rows": helper_uses_rows,
            "rotation_intervention_convention_matches_training": not (
                training_uses_columns and helper_uses_rows
            ),
        },
        "verdict": {
            "training_arm_mapping": (
                "pass"
                if translation_error <= tolerance
                and rotation_column_error <= tolerance
                and gripper_error <= tolerance
                else "fail"
            ),
            "translation_intervention_mapping": (
                "pass" if translation_error <= tolerance else "fail"
            ),
            "rotation_intervention_helper": (
                "mismatch"
                if training_uses_columns and helper_uses_rows
                else "no_mismatch_detected"
            ),
            "scope": (
                "The rotation helper mismatch affects rx/ry/rz evaluation variants, "
                "not training and not the tx/ty/tz-only multi-seed diagnostic."
            ),
        },
        "sources": {
            "dataset_root": str(root),
            "parquet": str(parquet_path),
            "parquet_sha256": _sha256(parquet_path),
            "model_action_archive": str(archive_path),
            "model_action_archive_sha256": _sha256(archive_path),
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"
    )
    print(
        "MAPPING "
        f"translation_error={translation_error:.3e} "
        f"rotation_column_error={rotation_column_error:.3e} "
        f"rotation_row_error={rotation_row_error:.3e} "
        f"helper_mismatch={payload['verdict']['rotation_intervention_helper']}"
    )
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
