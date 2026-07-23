# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Map-style Open-H multi-embodiment action SFT dataset.

``OpenHMixedLeRobotDataset`` (the Cosmos Framework port of the
Cosmos-H-Surgical-Simulator surgical mixture) -> ``ActionTransformPipeline``.

This is the surgical counterpart of
``cosmos_framework.data.vfm.action.datasets.action_sft_dataset.get_action_droid_sft_dataset``:
the base ``OpenHMixedLeRobotDataset.__getitem__`` returns the raw per-sample
dict (``video``/``action``/``ai_caption``/``viewpoint``/``mode``/``domain_id``/
``conditioning_fps``) at the embodiment's native (un-padded) action width, and
this wrapper composes it with ``ActionTransformPipeline`` (spatial resize/pad,
text tokenization, action padding to ``max_action_dim``=44 with channel masking,
and ``sequence_plan`` construction) so the experiment can hand a single
map-style dataset to ``RankPartitionedDataLoader`` -- exactly like the DROID
action SFT recipe.

The unified 44D action contract (CMR Versius ceiling: 30D actions + 14D state
conditioning; all other embodiments zero-padded) is enforced by
``OpenHMixedLeRobotDataset(max_action_dim=44)`` plus
``ActionTransformPipeline(max_action_dim=44, action_channel_masking=True)``.
"""
from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from cosmos_framework.data.vfm.action.camp_memory_tracks import CampMemoryTrackJoiner
from cosmos_framework.data.vfm.action.camp_transforms import CampActionTransformPipeline
from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import (
    ActionIterableShuffleDataset,
    ActionSFTDataset,
)
from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
    get_open_h_multi_train_specs,
)
from cosmos_framework.data.vfm.action.open_h_dataset import OpenHMixedLeRobotDataset
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline


class _OpenHShuffleBlockAdapter(Dataset):
    """Wrap ``OpenHMixedLeRobotDataset`` to expose ``get_shuffle_blocks``.

    ``ActionIterableShuffleDataset`` (used for decorrelated, sequential-read
    streaming in the DROID recipe) expects the inner dataset to expose
    ``get_shuffle_blocks() -> list[(start, length)]`` so each
    ``(rank, worker)`` can stream a disjoint, episode-order-shuffled subset.

    ``OpenHMixedLeRobotDataset`` indexes a virtual space partitioned into
    contiguous per-sub-dataset blocks (``virtual_sizes``); we expose those
    blocks so shuffling preserves within-sub-dataset locality (good for COW /
    sequential parquet reads) while decorrelating across sub-datasets.
    """

    def __init__(self, dataset: OpenHMixedLeRobotDataset | CampMemoryTrackJoiner) -> None:
        super().__init__()
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._dataset[idx]

    # Target block length (in windows). Each sub-dataset's contiguous virtual
    # span is split into blocks of about this many windows. This must yield a
    # block COUNT that is comfortably larger than the training world_size:
    # ``ActionIterableShuffleDataset`` shards blocks via ``order[global_shard::
    # total_shards]``, so if #blocks < world_size the trailing ranks get an EMPTY
    # shard and spin forever in the ``while True`` reshuffle loop, never yielding
    # a sample and never reaching the pre-warm barrier (this is exactly what hung
    # the 48-GPU runs: the previous "one block per sub-dataset" gave only
    # len(virtual_sizes)=36 blocks < 48 ranks, so ranks 36-47 got nothing).
    # ~512 windows/block over a ~10^8-window mixture yields ~10^5 blocks — far
    # more than any plausible world_size — while keeping each block CONTIGUOUS
    # (sequential parquet reads + copy-on-write locality, like the DROID recipe's
    # per-episode blocks).
    _BLOCK_LEN = 512

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Contiguous ``(start, length)`` blocks over the mixture's virtual index
        space, fine-grained enough that #blocks >> world_size.

        Each sub-dataset occupies a contiguous virtual range; we split each range
        into <= _BLOCK_LEN chunks. Splitting per sub-dataset (never spanning a
        sub-dataset boundary) keeps a block within ONE embodiment/leaf, so reads
        stay sequential within the source.
        """
        blocks: list[tuple[int, int]] = []
        start = 0
        for size in self._dataset.virtual_sizes:
            size = int(size)
            end = start + size
            pos = start
            while pos < end:
                length = min(self._BLOCK_LEN, end - pos)
                blocks.append((pos, length))
                pos += length
            start = end
        return blocks


def get_action_openh_sft_dataset(
    *,
    base_path: str | None = None,
    cmr_base_path: str | None = None,
    path_layout: str | None = None,
    openh_lz_base_path: str | None = None,
    num_frames: int = 13,
    data_split: str = "train",
    mode: str = "forward_dynamics",
    viewpoint: str = "third_person_view",
    test_split_ratio: float = 0.02,
    default_storage_fps: float = 30.0,
    downscaled_res: bool = False,
    max_retries_per_sample: int = 16,
    # ActionTransformPipeline knobs (mirror the cosmos3-internal YAML)
    resolution: str | int = "480",
    max_action_dim: int = 44,
    action_channel_masking: bool = True,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    keep_aspect_ratio: bool = True,
    caption_key: str = "ai_caption",
    video_temporal_downsample: int = 4,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    idle_frames_dropout: float = 0.05,
    format_prompt_as_json: bool = False,
    iterable_shuffle: bool = True,
    episode_shuffle_seed: int = 42,
    num_history_actions: int = 0,
    history_ablation: str | None = None,
    camp_memory_tracks_root: str | None = None,
    camp_memory_ablation: str | None = None,
) -> Dataset:
    """Build the Open-H multi-embodiment 44D action SFT dataset.

    Args mirror ``get_action_droid_sft_dataset`` where they overlap. The
    Open-H mixture itself (paths, per-embodiment transforms, mix ratios) is
    defined by ``OPEN_H_DATASET_SPECS`` in ``groot_configs.py``; ``base_path``
    re-roots the public surgical tree (``DATASET_PATH`` /
    ``OPENH_SURGICAL_ROOT``), while ``cmr_base_path`` independently points the
    four CMR leaves at Draco's fixed 60 Hz mirror. ``path_layout`` selects the
    relative naming convention (canonical public/EOS or ``draco_internal``).

    CAMP Phase 1: ``num_history_actions`` > 0 makes the base dataset emit a
    per-sample ``"history_action"`` tensor (H rows at the embodiment's native
    width, same transforms/normalization as the current window).  The
    ``ActionTransformPipeline`` (pinned framework) concatenates it ahead of
    ``"action"``, pads the combined tensor to ``max_action_dim``, and builds
    the sequence plan with the H rows as clean conditioning (zero noise, zero
    loss).  ``history_ablation`` (``zero`` / ``permute``) is an eval-arm knob.

    CAMP Phase 3 (arm C): ``camp_memory_tracks_root`` switches the stack to
    memory-conditioned mode — the base dataset emits retry-safe step ids, a
    ``CampMemoryTrackJoiner`` attaches the per-sample 132D ``memory_code``
    (from exported tracks, or deterministic debug codes when the root is
    ``"__random__"``), and ``CampActionTransformPipeline`` injects the 3 x 44
    code rows with 19-row conditioning plans.  ``camp_memory_ablation``
    (``zero`` / ``shuffle_episode``) is the Phase-5 eval grid.
    """
    specs = get_open_h_multi_train_specs(
        base_path=base_path,
        cmr_base_path=cmr_base_path,
        path_layout=path_layout,
        openh_lz_base_path=openh_lz_base_path,
    )
    base = OpenHMixedLeRobotDataset(
        dataset_specs=specs,
        num_frames=num_frames,
        data_split=data_split,
        max_action_dim=max_action_dim,
        downscaled_res=downscaled_res,
        test_split_ratio=test_split_ratio,
        mode=mode,
        viewpoint=viewpoint,
        default_storage_fps=default_storage_fps,
        max_retries_per_sample=max_retries_per_sample,
        num_history_actions=num_history_actions,
        history_ablation=history_ablation,
        emit_step_ids=camp_memory_tracks_root is not None,
    )
    inner: Dataset = base
    if camp_memory_tracks_root is not None:
        inner = CampMemoryTrackJoiner(
            base,
            tracks_root=camp_memory_tracks_root,
            memory_ablation=camp_memory_ablation,
        )
    elif camp_memory_ablation is not None:
        raise ValueError(
            "camp_memory_ablation requires camp_memory_tracks_root — an ablation "
            "with no memory source would silently be a no-op."
        )
    pipeline_cls = (
        CampActionTransformPipeline if camp_memory_tracks_root is not None else ActionTransformPipeline
    )
    transform = pipeline_cls(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        keep_aspect_ratio=keep_aspect_ratio,
        caption_key=caption_key,
        video_temporal_downsample=video_temporal_downsample,
        max_action_dim=max_action_dim,
        action_channel_masking=action_channel_masking,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        idle_frames_dropout=idle_frames_dropout,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(_OpenHShuffleBlockAdapter(inner), transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft


__all__ = ["get_action_openh_sft_dataset"]
