# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CAMP Phase-3 memory-row injection — the framework-free core.

``inject_memory_rows`` performs the actual injection on the OUTPUT of the
base ``ActionTransformPipeline``: reshape the 132D VQ code to (3, 44),
prepend it to the already-normalized-and-padded action tensor, and rebuild
the sequence plan so the memory rows are clean conditioning.

Deliberately dependency-light (torch + camp_data_contract only): the
sequence-plan builder is passed IN as a callable rather than imported from
``cosmos_framework...transforms`` (which needs the framework's full dep set),
so this logic unit-tests anywhere against the pinned builder's AST-extracted
source. The framework-facing subclass lives in ``camp_transforms.py``.

Ordering rationale (why injection happens AFTER the base pipeline): the
memory code is NOT a robot action. It has no native width, no pose to anchor
relatively, and no per-channel stats — running it through the per-embodiment
normalize/pad path would corrupt it. The base pipeline finishes native
processing (history concat, normalization, padding to 44D, channel-mask
bookkeeping) first; the code rows are then seated onto the padded lattice.

Phase-3b dependency (arm C blocker on narrow embodiments): the pinned model
zeroes padded channels of the action INPUT per sample
(``xt_action[i][:, raw_action_dim[i]:] = 0`` in omni_mot_model.py's noising
path, and the equivalent site in the sampling-init path). Native rows are
unaffected (their padded channels are zero anyway), but memory rows are
DENSE 44-wide — for an embodiment with raw_action_dim < 44 the code would be
silently truncated. Until the guarded model-side exemption lands (skip the
first ``num_memory_action_rows`` rows at those two sites; the loss and
velocity sites are already row-masked for conditioning rows),
``inject_memory_rows`` FAILS CLOSED on raw_action_dim < 44.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import torch

from cosmos_framework.data.vfm.action.camp_data_contract import (
    ACTION_DIM,
    CODE_DIM,
    NUM_MEMORY_SLOTS,
)


def model_memory_row_patch_present() -> bool:
    """True when the Phase-3b model-side exemption is installed.

    Detects the overlay-patched ``GenerationDataClean.num_memory_action_rows``
    field. On an unpatched framework the model would channel-truncate dense
    memory rows for raw_action_dim < 44, so injection must fail closed there.
    """
    try:
        from cosmos_framework.model.vfm.utils.data_and_condition import GenerationDataClean
    except Exception:
        return False
    return any(f.name == "num_memory_action_rows" for f in dataclasses.fields(GenerationDataClean))


def inject_memory_rows(
    action: torch.Tensor,
    memory_code: torch.Tensor,
    *,
    mode: str,
    video_length: int,
    num_history_actions: int,
    sequence_plan_builder: Callable[..., Any],
    video_temporal_downsample: int = 4,
    raw_action_dim: torch.Tensor | int | None = None,
) -> tuple[torch.Tensor, Any]:
    """Prepend the reshaped memory code and rebuild the sequence plan.

    Args:
        action: (T, ACTION_DIM) tensor as produced by the base pipeline —
            history (if any) already concatenated, normalized, padded.
        memory_code: (CODE_DIM,) or (NUM_MEMORY_SLOTS, ACTION_DIM) VQ code.
        mode: "forward_dynamics" | "policy" | "inverse_dynamics".
        video_length: Video frame count (sequence-plan input; unchanged).
        num_history_actions: H rows already prepended by the base pipeline
            (0 for a memory-only arm; 16 for full CAMP).
        sequence_plan_builder: ``build_sequence_plan_from_mode`` from the
            (pinned) framework transforms module.
        video_temporal_downsample: Forwarded to the builder.
        raw_action_dim: The per-sample channel-mask width recorded by the
            base pipeline. Values < ACTION_DIM raise until the Phase-3b
            model-side exemption lands (see module docstring). ``None``
            (masking disabled) passes.

    Returns:
        ``(new_action, new_sequence_plan)`` where ``new_action`` is
        ``(NUM_MEMORY_SLOTS + T, ACTION_DIM)`` and the plan marks
        ``num_history_actions + NUM_MEMORY_SLOTS`` leading rows as clean
        conditioning.
    """
    if action.ndim != 2 or action.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected action (T, {ACTION_DIM}), got {tuple(action.shape)}")

    if raw_action_dim is not None:
        raw = int(raw_action_dim.item() if isinstance(raw_action_dim, torch.Tensor) else raw_action_dim)
        if raw < ACTION_DIM and not model_memory_row_patch_present():
            raise ValueError(
                f"Memory injection with raw_action_dim={raw} < {ACTION_DIM} on an UNPATCHED "
                "framework would let the model-side padded-channel zeroing truncate the "
                "dense memory rows (omni_mot_model.py noising/sampling-init sites). "
                "Apply the Phase-3b overlay (framework_patch/cosmos_framework/model/vfm/"
                "omni_mot_model.py + utils/data_and_condition.py) — detected via "
                "GenerationDataClean.num_memory_action_rows. Failing closed rather than "
                "training on a silently truncated code."
            )

    code = memory_code
    if code.ndim == 1:
        if code.shape[0] != CODE_DIM:
            raise ValueError(f"expected memory_code ({CODE_DIM},), got {tuple(code.shape)}")
        code = code.view(NUM_MEMORY_SLOTS, ACTION_DIM)
    elif code.shape != (NUM_MEMORY_SLOTS, ACTION_DIM):
        raise ValueError(
            f"expected memory_code ({CODE_DIM},) or ({NUM_MEMORY_SLOTS}, {ACTION_DIM}), "
            f"got {tuple(code.shape)}"
        )
    if not torch.isfinite(code).all():
        raise ValueError("memory_code contains non-finite values")

    code = code.to(dtype=action.dtype, device=action.device)
    new_action = torch.cat([code, action], dim=0)

    new_plan = sequence_plan_builder(
        mode=mode,
        video_length=video_length,
        action_length=int(new_action.shape[0]),
        video_temporal_downsample=video_temporal_downsample,
        num_history_actions=int(num_history_actions) + NUM_MEMORY_SLOTS,
    )
    return new_action, new_plan
