# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``CampActionTransformPipeline`` — first-class memory injection (Phase 3).

Thin framework-facing subclass of the pinned ``ActionTransformPipeline``.
Per sample it:

1. pops ``"memory_code"`` (the 132D VQ code from the Phase-2 exporter or the
   Phase-5 online state manager) BEFORE the base pipeline runs,
2. lets the base pipeline do everything it already does — history concat,
   per-embodiment normalization, padding to 44D, channel-mask bookkeeping,
   sequence-plan construction,
3. prepends the reshaped (3, 44) code rows to the padded action tensor and
   REBUILDS the sequence plan with ``num_history_actions + 3`` conditioning
   rows (framework-free core: ``camp_injection.inject_memory_rows``),
4. records ``data_dict["num_memory_action_rows"] = 3`` for the Phase-3b
   model-side padded-channel-zeroing exemption.

The memory rows therefore never pass through native-action transforms — a
132D learned code is not a 9D/20D/44D robot action, and treating it as one
(the in-dataset concatenation the SutureBot implementation used) would
corrupt it under the per-embodiment normalize/pad stack.

Modes without an action rail (``image2video``) pass through untouched.
``require_memory_code=True`` (default) fails closed if an action-mode sample
arrives without a code — an arm-C run silently degrading to arm B would
poison the comparison.
"""

from __future__ import annotations

import torch

from cosmos_framework.data.vfm.action.camp_data_contract import NUM_MEMORY_SLOTS
from cosmos_framework.data.vfm.action.camp_injection import inject_memory_rows
from cosmos_framework.data.vfm.action.transforms import (
    ActionTransformPipeline,
    build_sequence_plan_from_mode,
)


class CampActionTransformPipeline(ActionTransformPipeline):
    """ActionTransformPipeline + CAMP memory-row injection.

    Args:
        memory_code_key: Data-dict key holding the per-sample 132D code.
        require_memory_code: When True (default), raise if an action-mode
            sample has no code. Set False only for mixed pipelines where
            some sources legitimately have no memory tracks.
        Remaining args: forwarded verbatim to ``ActionTransformPipeline``.
    """

    def __init__(
        self,
        *args,
        memory_code_key: str = "memory_code",
        require_memory_code: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.memory_code_key = memory_code_key
        self.require_memory_code = require_memory_code

    def __call__(self, data_dict: dict, resolution, action_normalizer=None) -> dict:
        memory_code = data_dict.pop(self.memory_code_key, None)
        history = data_dict.get("history_action")
        num_history_actions = int(history.shape[0]) if isinstance(history, torch.Tensor) else 0

        data_dict = super().__call__(data_dict, resolution, action_normalizer=action_normalizer)

        sequence_plan = data_dict.get("sequence_plan")
        action = data_dict.get("action")
        has_action = (
            sequence_plan is not None
            and getattr(sequence_plan, "has_action", False)
            and isinstance(action, torch.Tensor)
        )
        if not has_action:
            # e.g. image2video — no action rail, nothing to inject.
            return data_dict
        if memory_code is None:
            if self.require_memory_code:
                raise ValueError(
                    f"CampActionTransformPipeline: sample in mode {data_dict.get('mode')!r} "
                    f"has no {self.memory_code_key!r}. An arm-C pipeline must not silently "
                    "degrade to history-only conditioning; wire the memory-track joiner or "
                    "construct with require_memory_code=False."
                )
            return data_dict

        video = data_dict["video"]  # (C, T, H, W)
        new_action, new_plan = inject_memory_rows(
            action,
            memory_code,
            mode=data_dict["mode"],
            video_length=int(video.shape[1]),
            num_history_actions=num_history_actions,
            sequence_plan_builder=build_sequence_plan_from_mode,
            video_temporal_downsample=self.video_temporal_downsample,
            raw_action_dim=data_dict.get("raw_action_dim"),
        )
        data_dict["action"] = new_action
        data_dict["sequence_plan"] = new_plan
        # Consumed by the Phase-3b model patch: exempt the first M action rows
        # from per-sample padded-channel zeroing (they carry the dense code).
        data_dict["num_memory_action_rows"] = NUM_MEMORY_SLOTS
        return data_dict
