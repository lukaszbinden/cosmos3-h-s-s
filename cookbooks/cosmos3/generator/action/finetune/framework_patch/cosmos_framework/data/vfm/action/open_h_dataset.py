# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cosmos Framework adapter around the Open-H ``MixedLeRobotDataset`` from
``cosmos_framework.data.vfm.action.gr00t_dreams``.

The gr00t_dreams pipeline (ported from the Cosmos3 internal
Cosmos-H-Surgical-Simulator surgical stack) handles:

* multi-embodiment LeRobot dataset loading from local parquet shards,
* per-embodiment action transforms (``RelativeActionTransform``,
  ``CMRVersiusRelativeActionTransform``, ``GenericRelativeActionTransform``,
  ``ConcatTransform``, …) declared in
  :data:`cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs.EMBODIMENT_REGISTRY`,
* train/val/full splitting via a trailing test partition,
* weighted mixture sampling with integer repeat factors.

This adapter wraps the same per-spec :class:`WrappedLeRobotSingleDataset`
instances and the same Hare-Niemeyer-style repeat-factor logic that
:class:`MixedLeRobotDataset` uses, but produces samples in the schema
expected by the Cosmos Framework :class:`ActionTransformPipeline`
(``video``, ``action``, ``ai_caption``, ``mode``, ``domain_id``,
``conditioning_fps``, ``viewpoint``).  In particular:

* The action tensor is returned at its **raw** per-embodiment dimension
  (e.g. 20D for ``jhu_dvrk_mono``, 30D for ``cmr_versius``) — *not*
  pre-padded — so the pipeline can correctly populate
  ``raw_action_dim`` for channel-masked loss/noise/velocity on the
  zero-padded extras.
* No CUDA tensors are produced inside ``__getitem__``; the dataloader's
  ``spawn`` workers therefore never need a CUDA context just to fetch
  data.
* The video is returned as ``(C, T, H, W)`` ``uint8`` at whatever
  per-embodiment resolution :data:`EMBODIMENT_REGISTRY` declares (e.g.
  544x960 for ``jhu_dvrk_mono``); the transform pipeline then
  reflection-pads / resizes to the closest ``VIDEO_RES_SIZE_INFO`` tier.

The returned dict matches the contract of the base action datasets in
``cosmos_framework.data.vfm.action.datasets`` (e.g.
``DROIDLeRobotDataset``), so it can be wrapped by
:class:`cosmos_framework.data.vfm.action.datasets.action_sft_dataset.ActionSFTDataset`
exactly like the DROID action SFT recipe.  See
``get_action_openh_sft_dataset`` in
``cosmos_framework.data.vfm.action.datasets.openh_sft_dataset`` and the
experiment config
``cosmos_framework.configs.base.experiment.action.posttrain_config.action_fdm_open_h_sft_nano``
for the training entry point.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from random import randint
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_framework.utils import log
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.gr00t_dreams.data.dataset import (
    LeRobotSingleDataset,
    ModalityConfig,
    WrappedLeRobotSingleDataset,
)
from cosmos_framework.data.vfm.action.gr00t_dreams.data.embodiment_tags import EmbodimentTag
from cosmos_framework.data.vfm.action.gr00t_dreams.data.history_utils import (
    apply_history_ablation,
    history_ablation_seed,
    split_history_and_current,
    validate_history_args,
)


_FwdInvPol = Literal["forward_dynamics", "inverse_dynamics", "policy", "joint"]


# Per-embodiment effective FPS for embodiments that use a dedicated code path
# in construct_modality_config_and_transforms and therefore do not appear in
# EMBODIMENT_REGISTRY.  Without this table the fallback would report the raw
# storage FPS (30 Hz default) instead of the strided training rate.
#
# Values must match the ``timestep_interval`` hard-coded in each embodiment's
# dedicated branch of ``construct_modality_config_and_transforms``
# (groot_configs.py):
#   cmr_versius: 60 Hz storage / stride 6 = 10 Hz effective.
#   suturebot:   30 Hz storage / stride 3 = 10 Hz effective (dedicated
#                pre-concatenated-action branch; currently unused by the
#                Open-H specs, which load SutureBot data as jhu_dvrk_mono).
_OUT_OF_REGISTRY_EFFECTIVE_FPS: dict[str, int] = {
    "cmr_versius": 10,
    "suturebot": 10,
}


def _effective_fps_from_registry(registry_entry: dict, default_storage_fps: float = 30.0) -> int:
    """Effective sample-time FPS after the gr00t_dreams ``timestep_interval`` stride.

    Each entry in :data:`EMBODIMENT_REGISTRY` declares a ``timestep_interval``
    (e.g. ``3`` for the 30 Hz JHU dVRK monocular dataset → 10 Hz effective
    training rate).  ``ModalityConfig.delta_indices`` are then ``range(0,
    num_frames) * timestep_interval`` so successive sampled frames are spaced
    ``timestep_interval`` storage frames apart.

    Args:
        registry_entry: One value of :data:`EMBODIMENT_REGISTRY`.
        default_storage_fps: Fallback storage frame rate when the registry
            entry doesn't carry an explicit ``raw_fps`` field (the original
            registry didn't track raw FPS — see
            ``scripts/compute_openh_action_stats.py`` for the per-subset
            raw rates).  30 Hz is the most common; specific embodiments
            (LSCR_MIRACLE @ 15 Hz, LSCR_SMARTS @ 10 Hz) can override via
            ``raw_fps`` in the registry.

    Returns:
        Effective FPS at which the model sees the video.
    """
    raw_fps = float(registry_entry.get("raw_fps", default_storage_fps))
    stride = int(registry_entry["timestep_interval"])
    return int(round(raw_fps / max(1, stride)))


class OpenHMixedLeRobotDataset(Dataset):
    """Cosmos Framework-compatible multi-embodiment Open-H LeRobot mixture dataset.

    This is the Cosmos Framework counterpart of
    :class:`cosmos_framework.data.vfm.action.gr00t_dreams.data.dataset.MixedLeRobotDataset`.
    It reuses the same :class:`WrappedLeRobotSingleDataset` sub-datasets and
    repeat-factor mixing logic, but produces samples in the schema expected
    by :class:`cosmos_framework.data.vfm.action.transforms.ActionTransformPipeline`.

    The output of :meth:`__getitem__` matches the per-sample contract of the
    base action datasets in ``cosmos_framework.data.vfm.action.datasets``
    (``video``/``action``/``ai_caption``/``viewpoint``/``mode``/``domain_id``/
    ``conditioning_fps``), so this dataset can be wrapped by
    :class:`cosmos_framework.data.vfm.action.datasets.action_sft_dataset.ActionSFTDataset`
    (see ``get_action_openh_sft_dataset``).

    Args:
        dataset_specs: List of dicts; see
            :class:`MixedLeRobotDataset` for the full key reference.
            Required keys: ``path``, ``embodiment``.  Optional:
            ``mix_ratio``, ``data_split_override``,
            ``test_split_ratio_override``, ``exclude_splits``.
        num_frames: Number of video frames per sample
            (default 13 = 1 conditional + 12 prediction).
        data_split: One of ``"train"`` / ``"test"`` / ``"full"``.
        max_action_dim: Maximum action dimension used for **assertions**
            only — actions are returned at their per-embodiment raw
            width; downstream
            :class:`ActionTransformPipeline.pad_action_to_max_dim` performs
            the zero-pad to ``max_action_dim``.  Defaults to 44 (CMR Versius
            ceiling).
        downscaled_res: Forwarded to
            ``construct_modality_config_and_transforms`` (256x256 mode).
        test_split_ratio: Default trailing-test fraction; overridable
            per-spec via ``test_split_ratio_override``.
        mode: Training mode — ``"forward_dynamics"`` (video conditioned on
            action + initial frame), ``"inverse_dynamics"``, ``"policy"``,
            or ``"joint"`` (uniformly randomly choose one per sample).
            Defaults to ``"forward_dynamics"`` (the surgical simulator
            use-case).
        viewpoint: Cosmos3 viewpoint tag for the camera.  Must be one of
            the labels declared in
            ``cosmos_framework.data.vfm.action.viewpoint_utils.Viewpoint``:
            ``"ego_view"``, ``"third_person_view"``, ``"wrist_view"``,
            ``"concat_view"``.  Any other string will cause
            ``ViewpointTextInfo`` to emit a per-sample WARNING and drop
            the viewpoint conditioning sentence from the caption.
            Surgical endoscope footage is third-person.
        default_storage_fps: Fallback storage FPS for
            :func:`_effective_fps_from_registry` when the registry entry
            doesn't declare ``raw_fps``.  30 Hz matches the majority of
            Open-H datasets.
        max_retries_per_sample: Bound on how many random retries
            ``__getitem__`` will attempt before giving up on a malformed
            sample.  Mirrors the predict2.5 ``MixedLeRobotDataset`` behavior
            which retries indefinitely; we bound it to avoid silently
            hiding systemic data issues.
        num_history_actions: CAMP Phase 1 — number of recent EXECUTED action
            rows (H) to sample before the current window, at the embodiment's
            effective timestep interval, through the same per-embodiment
            transform + normalization stack.  When > 0, each sample gains a
            ``"history_action"`` key of shape ``(H, D_native)`` which the
            (pinned) framework ``ActionTransformPipeline`` concatenates ahead
            of ``"action"``, pads to ``max_action_dim``, and marks as clean
            conditioning in the sequence plan.  ``0`` (default) is
            byte-identical to the pre-CAMP pipeline — no new keys, identical
            delta indices.
        history_ablation: ``None`` (real history), ``"zero"`` (history rows
            zeroed), or ``"permute"`` (rows time-shuffled with a stable
            per-sample seed).  Evaluation-arm knob; requires
            ``num_history_actions > 0``.
    """

    def __init__(
        self,
        dataset_specs: list[dict],
        num_frames: int = 13,
        data_split: str = "train",
        max_action_dim: int = 44,
        downscaled_res: bool = False,
        test_split_ratio: float = 0.05,
        mode: _FwdInvPol = "forward_dynamics",
        viewpoint: str = "third_person_view",
        default_storage_fps: float = 30.0,
        max_retries_per_sample: int = 16,
        num_history_actions: int = 0,
        history_ablation: str | None = None,
    ) -> None:
        from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
            EMBODIMENT_REGISTRY,
            construct_modality_config_and_transforms,
        )

        super().__init__()
        if not dataset_specs:
            raise ValueError("dataset_specs must be a non-empty list of {path, embodiment, ...} dicts")

        self.num_frames = num_frames
        self.max_action_dim = int(max_action_dim)
        self.mode = mode
        self.viewpoint = viewpoint
        self._default_storage_fps = float(default_storage_fps)
        self._max_retries_per_sample = int(max_retries_per_sample)

        validate_history_args(num_history_actions, history_ablation)
        self.num_history_actions = int(num_history_actions)
        self.history_ablation = history_ablation

        self.sub_datasets: list[WrappedLeRobotSingleDataset] = []
        self.mix_ratios: list[float] = []
        self.embodiment_tags: list[str] = []
        self.domain_ids: list[int] = []
        self.dataset_paths: list[str] = []
        self.effective_fps_per_dataset: list[int] = []

        log.info("=" * 80)
        log.info("INITIALIZING OPEN-H MIXED LEROBOT DATASET (Cosmos3 adapter)")
        log.info("=" * 80)
        for i, spec in enumerate(dataset_specs):
            path = spec["path"]
            raw_embodiment = spec["embodiment"]
            embodiment = raw_embodiment.value if isinstance(raw_embodiment, EmbodimentTag) else raw_embodiment
            mix_ratio = float(spec.get("mix_ratio", 1.0))

            spec_data_split = spec.get("data_split_override", data_split)
            spec_test_split_ratio = spec.get("test_split_ratio_override", test_split_ratio)
            exclude_splits = spec.get("exclude_splits", None)

            log.info(
                f"[{i}] {embodiment} mix_ratio={mix_ratio:.3f} "
                f"data_split={spec_data_split} test_ratio={spec_test_split_ratio:.3f}\n    path={path}"
            )

            config, train_tf, test_tf = construct_modality_config_and_transforms(
                num_frames=num_frames,
                embodiment=embodiment,
                downscaled_res=downscaled_res,
                num_history_actions=self.num_history_actions,
            )

            modality_filename = None
            if isinstance(config, dict) and "modality_filename" in config:
                modality_filename = config.pop("modality_filename")

            # Language / caption conditioning.
            #
            # ``_build_generic_config_and_transforms`` (the per-embodiment config
            # used by every ``jhu_dvrk_mono`` leaf) requests only video/state/action
            # modalities, so no ``annotation.*`` key ever reaches ``__getitem__`` and
            # every window would otherwise get ``ai_caption=""``.  Where a leaf's
            # ``modality.json`` declares the episode-level ``human.task_description``
            # annotation (currently ``jhu/imerse/suturebot`` — "knot tying" /
            # "needle pickup" / "needle throw"), request it as a single-frame
            # ``language`` modality so ``LeRobotSingleDataset.get_language`` resolves
            # it via ``original_key=task_index`` → ``meta/tasks.jsonl``.
            #
            # This is deliberately best-effort and per-leaf: leaves that do NOT
            # declare ``human.task_description`` (other Open-H surgical leaves use a
            # ``task`` / ``instruction`` annotation, or none) are left exactly as
            # before and never get a ``language`` key added — so the sub-dataset's
            # ``_check_integrity`` and ``get_language`` never assert on a missing key.
            language_key = self._leaf_task_description_key(path, modality_filename)
            if language_key is not None:
                config["language"] = ModalityConfig(delta_indices=[0], modality_keys=[language_key])
                log.info(f"    language conditioning: {language_key} (episode-level task description)")

            transform = train_tf if spec_data_split in ("train", "full") else test_tf

            sub = WrappedLeRobotSingleDataset(
                dataset_path=path,
                modality_configs=config,
                transforms=transform,
                embodiment_tag=embodiment,
                data_split=spec_data_split,
                test_split_ratio=spec_test_split_ratio,
                modality_filename=modality_filename,
                exclude_splits=exclude_splits,
            )

            self.sub_datasets.append(sub)
            self.mix_ratios.append(mix_ratio)
            self.embodiment_tags.append(embodiment)
            self.domain_ids.append(get_domain_id(embodiment))
            self.dataset_paths.append(str(path))
            registry_entry = EMBODIMENT_REGISTRY.get(embodiment, {})
            if registry_entry:
                effective_fps = _effective_fps_from_registry(
                    registry_entry, default_storage_fps=self._default_storage_fps
                )
            elif embodiment in _OUT_OF_REGISTRY_EFFECTIVE_FPS:
                # Embodiments with dedicated code paths (e.g. cmr_versius) are
                # not in EMBODIMENT_REGISTRY but have known stride/FPS values.
                effective_fps = _OUT_OF_REGISTRY_EFFECTIVE_FPS[embodiment]
            else:
                effective_fps = int(round(self._default_storage_fps))
                log.warning(
                    f"Embodiment {embodiment!r} is neither in EMBODIMENT_REGISTRY nor "
                    f"_OUT_OF_REGISTRY_EFFECTIVE_FPS; falling back to the raw storage "
                    f"FPS ({effective_fps} Hz). If this embodiment strides frames "
                    f"(timestep_interval > 1), conditioning_fps will be WRONG — add it "
                    f"to _OUT_OF_REGISTRY_EFFECTIVE_FPS."
                )
            self.effective_fps_per_dataset.append(effective_fps)

            log.info(f"    => {len(sub):,} samples loaded (effective_fps={effective_fps} Hz)")

        self._compute_repeat_factors()
        self._print_summary()

    # ------------------------------------------------------------------
    # Mixture indexing — mirrors ``MixedLeRobotDataset._compute_repeat_factors``
    # ------------------------------------------------------------------

    def _compute_repeat_factors(self) -> None:
        """Compute integer repeat factors so virtual sample-share matches mix_ratios.

        Per dataset i: ``per_sample_weight_i = mix_ratio_i / len(ds_i)``.
        Normalize to the smallest weight, round to ``int``, ensure ≥ 1.
        Virtual indices ``[0, total_virtual_len)`` are partitioned into
        contiguous per-dataset blocks of length ``len(ds_i) * repeat_i``.
        """
        per_sample_weights: list[float] = [
            ratio / max(len(ds), 1) for ds, ratio in zip(self.sub_datasets, self.mix_ratios)
        ]
        min_weight = min(per_sample_weights)
        raw_factors = [w / min_weight for w in per_sample_weights]
        self.repeat_factors: list[int] = [max(1, round(f)) for f in raw_factors]
        self.virtual_sizes: list[int] = [
            len(ds) * rf for ds, rf in zip(self.sub_datasets, self.repeat_factors)
        ]
        self._total_virtual_len: int = int(sum(self.virtual_sizes))
        self._cumulative_sizes: np.ndarray = np.cumsum(self.virtual_sizes)

    def _print_summary(self) -> None:
        log.info("=" * 80)
        log.info("OPEN-H MIXTURE SUMMARY (Cosmos3 adapter)")
        log.info("=" * 80)
        for ds, tag, ratio, rf, vs in zip(
            self.sub_datasets,
            self.embodiment_tags,
            self.mix_ratios,
            self.repeat_factors,
            self.virtual_sizes,
        ):
            path_short = Path(ds.dataset_path).name
            pct = 100.0 * vs / max(self._total_virtual_len, 1)
            log.info(
                f"  {tag:<22} {path_short:<38} real={len(ds):>10,} "
                f"mix={ratio:>6.3f} repeat={rf:>3} virtual={vs:>10,} ({pct:>5.1f}%)"
            )
        log.info(
            f"  Total virtual={self._total_virtual_len:,}  "
            f"max_action_dim={self.max_action_dim}  "
            f"num_embodiments={len(set(self.embodiment_tags))}"
        )
        log.info("=" * 80)

    def __len__(self) -> int:
        return self._total_virtual_len

    # ------------------------------------------------------------------
    # Cosmos3 schema construction
    # ------------------------------------------------------------------

    def _format_video(self, video_array: Any) -> torch.Tensor:
        """Convert gr00t_dreams transform output to ``(C, T, H, W)`` ``uint8``.

        The gr00t_dreams ``VideoToTensor`` transform produces a numpy
        ``float32`` array in ``[0, 1]`` with layout ``(T, C, H, W)`` or
        ``(T, V, C, H, W)`` (with ``V==1`` for single-view, since
        :data:`EMBODIMENT_REGISTRY` declares one ``video_keys`` entry
        per embodiment).  Cosmos3 expects ``(C, T, H, W)`` ``uint8``
        (see ``BaseActionLeRobotDataset._convert_video``).
        """
        if isinstance(video_array, np.ndarray):
            video = torch.from_numpy(video_array)
        elif isinstance(video_array, torch.Tensor):
            video = video_array
        else:
            raise TypeError(f"Unsupported video type {type(video_array).__name__}")
        # Collapse a singleton view dimension if present
        if video.ndim == 5:
            t, v, c, h, w = video.shape
            if v != 1:
                raise ValueError(f"OpenHMixedLeRobotDataset expects monocular video, got {v} views")
            video = video.squeeze(1)
        if video.ndim != 4:
            raise ValueError(
                f"OpenHMixedLeRobotDataset expects video with shape (T,C,H,W) or (T,1,C,H,W), "
                f"got {tuple(video.shape)}"
            )
        if torch.is_floating_point(video):
            video = torch.clamp(video * 255.0, 0.0, 255.0).to(torch.uint8)
        # (T, C, H, W) → (C, T, H, W)
        return video.permute(1, 0, 2, 3).contiguous()

    # Annotation keys carrying a natural-language caption, in priority order.
    # ``coarse_action`` is the predict2.5 per-window key; ``task_description`` is
    # the episode-level Open-H surgical annotation (resolves through
    # ``meta/tasks.jsonl``).  ``__getitem__`` takes the first one present.
    _CAPTION_ANNOTATION_KEYS = (
        "annotation.human.coarse_action",
        "annotation.human.task_description",
    )

    @staticmethod
    def _leaf_task_description_key(path: str, modality_filename: str | None) -> str | None:
        """Annotation modality key for a leaf's episode-level task description.

        Returns ``"annotation.human.task_description"`` iff the leaf's
        ``modality.json`` declares that annotation subkey, else ``None``.  Only
        ``human.task_description`` is wired here because it is the one annotation
        scheme known to resolve cleanly through ``meta/tasks.jsonl`` (via
        ``original_key=task_index``); the ``task`` / ``instruction`` schemes seen on
        other Open-H surgical leaves are intentionally left untouched so their
        windows keep their current (empty-caption) behavior rather than risk an
        ``_check_integrity`` / ``get_language`` assertion mid-training.

        Best-effort: any I/O or JSON error yields ``None`` (no language modality).
        """
        modality_path = Path(path) / (modality_filename or "meta/modality.json")
        try:
            with open(modality_path) as f:
                annotation = json.load(f).get("annotation") or {}
        except (OSError, ValueError):
            return None
        subkey = "human.task_description"
        return f"annotation.{subkey}" if subkey in annotation else None

    def _get_raw_sample(self, dataset_idx: int, real_idx: int) -> dict[str, Any]:
        """Pull a raw transformed sample from the underlying gr00t_dreams dataset.

        We deliberately skip the predict2.5-specific
        ``WrappedLeRobotSingleDataset.__getitem__`` (which manufactures
        ``t5_text_embeddings`` / ``image_size`` / ``padding_mask`` dummy
        tensors directly on the CUDA device — fatal in spawn-mode
        DataLoader workers without an initialised CUDA context).  Instead
        we call the grandparent :meth:`LeRobotSingleDataset.__getitem__`
        binding to obtain the post-transform dict on CPU and reshape it
        ourselves.
        """
        sub = self.sub_datasets[dataset_idx]
        return LeRobotSingleDataset.__getitem__(sub, real_idx)

    def get_step_ids(self, idx: int) -> dict[str, Any]:
        """Stable provenance identifiers for the sample at virtual index ``idx``.

        Mirrors ``__getitem__``'s virtual-index resolution exactly, then reads
        the underlying ``(trajectory_id, base_index)`` from the sub-dataset's
        step table.  Used by the CAMP Phase-2 memory-track exporter to key
        exported codes collision-proof — ``dataset_path`` (not its basename;
        36 leaves share names) + ``episode_id`` + ``base_index``.

        Deliberately NOT part of the training sample dict, so the training
        data contract is untouched.
        """
        idx = int(idx) % len(self)
        dataset_idx = int(np.searchsorted(self._cumulative_sizes, idx, side="right"))
        local_idx = idx if dataset_idx == 0 else idx - int(self._cumulative_sizes[dataset_idx - 1])
        real_idx = local_idx % len(self.sub_datasets[dataset_idx])
        trajectory_id, base_index = self.sub_datasets[dataset_idx].all_steps[real_idx]
        return {
            "dataset_idx": dataset_idx,
            "dataset_path": self.dataset_paths[dataset_idx],
            "embodiment": self.embodiment_tags[dataset_idx],
            "episode_id": int(trajectory_id),
            "base_index": int(base_index),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:  # noqa: C901
        idx = int(idx) % len(self)

        attempt = 0
        while True:
            try:
                dataset_idx = int(np.searchsorted(self._cumulative_sizes, idx, side="right"))
                local_idx = idx if dataset_idx == 0 else idx - int(self._cumulative_sizes[dataset_idx - 1])
                real_idx = local_idx % len(self.sub_datasets[dataset_idx])

                outputs = self._get_raw_sample(dataset_idx, real_idx)

                if "action" not in outputs:
                    raise KeyError(f"sub-dataset[{dataset_idx}] returned dict missing 'action' key")
                action = outputs["action"]
                if not isinstance(action, torch.Tensor):
                    action = torch.as_tensor(action)
                action = action.detach().to(dtype=torch.float32, device="cpu")
                if action.ndim != 2:
                    raise ValueError(
                        f"sub-dataset[{dataset_idx}] returned action with shape {tuple(action.shape)}; "
                        "expected [T, D]"
                    )
                if action.shape[-1] > self.max_action_dim:
                    raise ValueError(
                        f"sub-dataset[{dataset_idx}] ({self.embodiment_tags[dataset_idx]}) "
                        f"returned action with D={action.shape[-1]} > max_action_dim={self.max_action_dim}"
                    )

                # CAMP Phase 1: the transformed action tensor is (H + N, D) —
                # split off the H history rows (clean conditioning) from the N
                # current rows (the public 12x44 contract, unchanged). The
                # ablation seed is stable per (sub-dataset, sample) so eval
                # permutations are reproducible across epochs and workers.
                history_action: torch.Tensor | None = None
                if self.num_history_actions > 0:
                    history_action, action = split_history_and_current(
                        action, self.num_history_actions
                    )
                    history_action = apply_history_ablation(
                        history_action,
                        self.history_ablation,
                        seed=history_ablation_seed(dataset_idx, real_idx),
                    )

                video = self._format_video(outputs["video"])

                ai_caption = ""
                for ann_key in self._CAPTION_ANNOTATION_KEYS:
                    if ann_key not in outputs:
                        continue
                    raw_text = outputs[ann_key]
                    if isinstance(raw_text, list) and raw_text:
                        raw_text = raw_text[0]
                    if isinstance(raw_text, str) and raw_text.strip():
                        ai_caption = raw_text.split(":")[-1].strip()
                        break

                conditioning_fps = torch.tensor(
                    self.effective_fps_per_dataset[dataset_idx], dtype=torch.long
                )
                domain_id = torch.tensor(self.domain_ids[dataset_idx], dtype=torch.long)

                sample: dict[str, Any] = {
                    "ai_caption": ai_caption,
                    "video": video,
                    "action": action,
                    "conditioning_fps": conditioning_fps,
                    "mode": self.mode,
                    "domain_id": domain_id,
                    "viewpoint": self.viewpoint,
                }
                # Only present when H > 0 so the H=0 sample dict stays
                # byte-identical to the pre-CAMP contract. The framework
                # ActionTransformPipeline pops this key, prepends it to
                # "action", and marks the rows as clean conditioning.
                if history_action is not None:
                    sample["history_action"] = history_action
                return sample
            except Exception as e:
                attempt += 1
                if attempt > self._max_retries_per_sample:
                    raise RuntimeError(
                        f"OpenHMixedLeRobotDataset gave up after {self._max_retries_per_sample} retries "
                        f"on idx={idx}: {e!r}"
                    ) from e
                # Don't log spam in production; emit a warning on first retry only.
                if attempt == 1:
                    log.warning(
                        f"OpenHMixedLeRobotDataset retry (idx={idx}, pid={os.getpid()}): {e!r}"
                    )
                idx = randint(0, len(self) - 1)


__all__ = ["OpenHMixedLeRobotDataset"]
