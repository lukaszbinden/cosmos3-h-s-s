#!/usr/bin/env python3
"""Physical-space action interventions for the 20-D C3-H-S-S dVRK layout.

The dataset emits mean/std-normalized actions after converting each PSM pose
to a translation relative to the conditioning frame and a relative SO(3)
rotation represented by the first two rows of its rotation matrix.  This
module reverses only the normalization, applies physically meaningful edits,
and then restores the exact normalized representation expected by Cosmos.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

CHUNK_SIZE = 12
ACTION_DIM = 20
ARM_ACTION_DIM = 10
ACTION_KEYS = (
    "action.psm1_pose",
    "action.psm1_gripper",
    "action.psm2_pose",
    "action.psm2_gripper",
)
AXES = ("x", "y", "z")
GAINS = ((0.0, "0x"), (1.5, "1p5x"))
ANCHOR_MODES = ("reference", "first_row")


def load_action_stats(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load action means/stds in the exact concatenation order used in training."""
    path = Path(path)
    payload = json.loads(path.read_text())
    missing = [key for key in ACTION_KEYS if key not in payload]
    if missing:
        raise KeyError(f"{path} is missing action statistics for {missing}")

    means = np.concatenate(
        [np.asarray(payload[key]["mean"], dtype=np.float64) for key in ACTION_KEYS]
    )
    stds = np.concatenate(
        [np.asarray(payload[key]["std"], dtype=np.float64) for key in ACTION_KEYS]
    )
    if means.shape != (ACTION_DIM,) or stds.shape != (ACTION_DIM,):
        raise ValueError(
            f"{path} produced mean/std shapes {means.shape}/{stds.shape}, "
            f"expected ({ACTION_DIM},)"
        )
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(stds)):
        raise ValueError(f"{path} contains non-finite action statistics")
    if np.any(stds <= 0):
        raise ValueError(f"{path} contains non-positive action standard deviations")
    return means, stds


def denormalize_action(
    normalized: np.ndarray, means: np.ndarray, stds: np.ndarray
) -> np.ndarray:
    normalized = np.asarray(normalized, dtype=np.float64)
    return normalized * stds + means


def normalize_action(
    physical: np.ndarray, means: np.ndarray, stds: np.ndarray
) -> np.ndarray:
    physical = np.asarray(physical, dtype=np.float64)
    return (physical - means) / stds


def rotation_6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Match the training transform's row-based Gram-Schmidt conversion."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    rows = rot6d.reshape(*rot6d.shape[:-1], 2, 3)
    row1 = rows[..., 0, :]
    row1 /= np.maximum(np.linalg.norm(row1, axis=-1, keepdims=True), 1e-12)
    row2 = rows[..., 1, :]
    row2 -= np.sum(row1 * row2, axis=-1, keepdims=True) * row1
    row2 /= np.maximum(np.linalg.norm(row2, axis=-1, keepdims=True), 1e-12)
    row3 = np.cross(row1, row2)
    matrices = np.stack([row1, row2, row3], axis=-2)
    _validate_rotation_matrices(matrices)
    return matrices


def matrix_to_rotation_6d(matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float64)
    _validate_rotation_matrices(matrices)
    return matrices[..., :2, :].reshape(*matrices.shape[:-2], 6)


def _validate_rotation_matrices(matrices: np.ndarray) -> None:
    identity = np.eye(3, dtype=np.float64)
    gram = np.swapaxes(matrices, -1, -2) @ matrices
    max_orthogonality_error = float(np.max(np.abs(gram - identity)))
    max_determinant_error = float(np.max(np.abs(np.linalg.det(matrices) - 1.0)))
    if max_orthogonality_error > 1e-6 or max_determinant_error > 1e-6:
        raise ValueError(
            "Invalid SO(3) intervention: "
            f"orthogonality_error={max_orthogonality_error:.3e}, "
            f"determinant_error={max_determinant_error:.3e}"
        )


def _scaled_axis_rotation(
    physical: np.ndarray,
    arm_offset: int,
    axis: int,
    gain: float,
    anchor_mode: str,
) -> np.ndarray:
    result = physical.copy()
    rotation_slice = slice(arm_offset + 3, arm_offset + 9)
    matrices = rotation_6d_to_matrix(result[:, rotation_slice])
    if anchor_mode == "first_row":
        anchor = matrices[0]
        delta_matrices = np.einsum("ij,tjk->tik", anchor.T, matrices)
        rotation_vectors = Rotation.from_matrix(delta_matrices).as_rotvec()
    else:
        anchor = np.eye(3, dtype=np.float64)
        rotation_vectors = Rotation.from_matrix(matrices).as_rotvec()
    rotation_vectors[:, axis] *= gain
    scaled_delta = Rotation.from_rotvec(rotation_vectors).as_matrix()
    result[:, rotation_slice] = matrix_to_rotation_6d(
        np.einsum("ij,tjk->tik", anchor, scaled_delta)
    )
    return result


def build_physical_axis_variants(
    correct: torch.Tensor,
    stats_path: str | Path,
    anchor_mode: str = "reference",
) -> tuple[
    OrderedDict[str, torch.Tensor],
    OrderedDict[str, np.ndarray],
    dict[str, Any],
]:
    """Build correct plus 0x/1.5x interventions for 14 physical components.

    With ``anchor_mode="reference"`` (legacy behavior), translation and
    rotation are scaled around the conditioning-state origin. With
    ``anchor_mode="first_row"``, each trajectory is scaled around its first
    model-facing row, preserving the initial command-state offset and changing
    only subsequent motion. Jaw motion is always scaled around its first row.
    """
    if anchor_mode not in ANCHOR_MODES:
        raise ValueError(
            f"anchor_mode must be one of {ANCHOR_MODES}, got {anchor_mode!r}"
        )
    if tuple(correct.shape) != (CHUNK_SIZE, ACTION_DIM):
        raise ValueError(
            f"correct action has shape {tuple(correct.shape)}, "
            f"expected ({CHUNK_SIZE}, {ACTION_DIM})"
        )
    correct_numpy = correct.detach().cpu().double().numpy()
    means, stds = load_action_stats(stats_path)
    correct_physical = denormalize_action(correct_numpy, means, stds)

    roundtrip = normalize_action(correct_physical, means, stds)
    roundtrip_max_abs_error = float(np.max(np.abs(roundtrip - correct_numpy)))
    if roundtrip_max_abs_error > 1e-10:
        raise ValueError(f"action stats round-trip failed: {roundtrip_max_abs_error:.3e}")

    normalized_variants: OrderedDict[str, torch.Tensor] = OrderedDict()
    physical_variants: OrderedDict[str, np.ndarray] = OrderedDict()

    def add(name: str, physical: np.ndarray) -> None:
        normalized = normalize_action(physical, means, stds)
        reconstructed = denormalize_action(normalized, means, stds)
        error = float(np.max(np.abs(reconstructed - physical)))
        if error > 1e-10:
            raise ValueError(f"{name} physical round-trip failed: {error:.3e}")
        physical_variants[name] = physical.astype(np.float32)
        normalized_variants[name] = torch.as_tensor(
            normalized, dtype=correct.dtype, device=correct.device
        )

    add("correct", correct_physical.copy())
    for arm_index, arm_name in enumerate(("psm1", "psm2")):
        arm_offset = arm_index * ARM_ACTION_DIM
        for axis_index, axis_name in enumerate(AXES):
            for gain, gain_name in GAINS:
                variant = correct_physical.copy()
                axis_values = variant[:, arm_offset + axis_index]
                if anchor_mode == "first_row":
                    anchor = axis_values[0]
                    variant[:, arm_offset + axis_index] = anchor + gain * (
                        axis_values - anchor
                    )
                else:
                    variant[:, arm_offset + axis_index] *= gain
                add(f"{arm_name}_t{axis_name}_{gain_name}", variant)

        for axis_index, axis_name in enumerate(AXES):
            for gain, gain_name in GAINS:
                add(
                    f"{arm_name}_r{axis_name}_{gain_name}",
                    _scaled_axis_rotation(
                        correct_physical,
                        arm_offset,
                        axis_index,
                        gain,
                        anchor_mode,
                    ),
                )

        jaw_index = arm_offset + 9
        jaw_anchor = correct_physical[0, jaw_index]
        for gain, gain_name in GAINS:
            variant = correct_physical.copy()
            variant[:, jaw_index] = jaw_anchor + gain * (
                correct_physical[:, jaw_index] - jaw_anchor
            )
            add(f"{arm_name}_jaw_{gain_name}", variant)

    if len(normalized_variants) != 29:
        raise AssertionError(f"expected 29 variants, built {len(normalized_variants)}")
    first_row_max_abs_error = max(
        float(np.max(np.abs(value[0] - correct_physical[0])))
        for value in physical_variants.values()
    )
    if anchor_mode == "first_row" and first_row_max_abs_error > 1e-6:
        raise ValueError(
            "first-row anchoring failed: "
            f"maximum first-row error={first_row_max_abs_error:.3e}"
        )

    audit = {
        "stats_path": str(Path(stats_path)),
        "action_keys": list(ACTION_KEYS),
        "mean": means.tolist(),
        "std": stds.tolist(),
        "normalized_physical_roundtrip_max_abs_error": roundtrip_max_abs_error,
        "anchor_mode": anchor_mode,
        "variant_first_row_max_abs_error": first_row_max_abs_error,
        "rotation_parameterization": (
            "relative SO(3) matrix -> anchor-local axis-angle; selected rotvec "
            "component scaled; converted back to first-two-rows rot6d"
        ),
        "jaw_parameterization": "absolute command motion around the first command",
        "variant_count": len(normalized_variants),
    }
    return normalized_variants, physical_variants, audit
