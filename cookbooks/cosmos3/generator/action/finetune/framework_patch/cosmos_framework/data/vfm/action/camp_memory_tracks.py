# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CAMP memory-track store + joiner: per-sample ``memory_code`` for arm C.

This module defines the on-disk TRACK CONTRACT the Phase-2b exporter writes
and the joiner that attaches codes to training samples:

    <tracks_root>/
      <leaf-basename>-<sha1(dataset_path)[:10]>/     one dir per leaf
        manifest.json                                provenance + validation
        episode_<episode_id>.npy                     (T_episode, CODE_DIM)

``manifest.json`` REQUIRED fields (validated at first lookup):
    dataset_path      full path of the source leaf — the collision-proof key
                      (36 leaves share basenames); must match exactly.
    code_dim          must equal camp_data_contract.CODE_DIM.
Recommended provenance fields (written by the exporter, not validated here):
    framework_sha, repo_sha, memory_checkpoint_hash, action_schema_version,
    stats_hashes, episode_count, effective_fps, embodiment.

Joining is keyed by the sample's ``"_step_ids"`` (see
``OpenHMixedLeRobotDataset(emit_step_ids=True)``) — NEVER by re-resolving the
virtual index, because the base dataset's retry loop rerolls indices on
per-sample failures.

The code for a sample anchored at storage frame ``base_index`` is
``track[base_index]`` — the causal encoder's summary of everything up to the
anchor, matching the SutureBot reference (m_t conditions the window at t) and
the Phase-5 online state manager.

Memory sources / ablations (Phase-5 grid):
    tracks_root=<dir>                    real exported codes
    tracks_root="__random__"             RANDOM DEBUG codes, uniform [-1, 1],
                                         deterministic per (leaf, episode,
                                         frame). For packing/plumbing smokes
                                         ONLY — never a science arm.
    memory_ablation="zero"               zero codes (does the model use the
                                         memory at all?)
    memory_ablation="shuffle_episode"    code from a DIFFERENT episode of the
                                         same leaf at the (clamped) same
                                         frame — in-distribution but
                                         behaviorally wrong; the strongest
                                         null detector.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.vfm.action.camp_data_contract import CODE_DIM

RANDOM_DEBUG_SENTINEL = "__random__"

MEMORY_ABLATIONS: tuple[str | None, ...] = (None, "zero", "shuffle_episode")

# Bounded per-worker cache of mmap'd episode arrays.
_EPISODE_CACHE_SIZE = 64


def track_dir_name(dataset_path: str) -> str:
    """Collision-proof, human-readable track directory name for a leaf.

    Basename alone collides (e.g. two ``suturebot`` leaves); the full-path
    sha1 suffix disambiguates while keeping the name greppable.
    """
    path = str(dataset_path).rstrip("/")
    digest = hashlib.sha1(path.encode()).hexdigest()[:10]
    return f"{Path(path).name}-{digest}"


class MemoryTrackStore:
    """Read-side access to exported memory tracks (see module docstring)."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"CAMP memory-tracks root {self.root} does not exist. Run the "
                "Phase-2b exporter first, or use tracks_root='__random__' for a "
                "plumbing/packing smoke."
            )
        self._validated_leaves: dict[str, Path] = {}
        self._episode_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._episode_ids: dict[str, list[int]] = {}

    # -- leaf resolution ----------------------------------------------------

    def _leaf_dir(self, dataset_path: str) -> Path:
        key = str(dataset_path)
        cached = self._validated_leaves.get(key)
        if cached is not None:
            return cached
        leaf = self.root / track_dir_name(key)
        manifest_path = leaf / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No memory track for leaf {key!r}: expected {manifest_path}. "
                "Export it with the Phase-2b exporter."
            )
        with open(manifest_path) as f:
            manifest = json.load(f)
        if str(manifest.get("dataset_path", "")).rstrip("/") != key.rstrip("/"):
            raise ValueError(
                f"Memory-track manifest mismatch in {leaf}: manifest dataset_path="
                f"{manifest.get('dataset_path')!r} != requested {key!r}. Tracks are "
                "keyed by FULL dataset path; a hash collision or a mis-exported "
                "track directory is fatal."
            )
        if int(manifest.get("code_dim", -1)) != CODE_DIM:
            raise ValueError(
                f"Memory track {leaf} has code_dim={manifest.get('code_dim')} but this "
                f"contract requires {CODE_DIM}. Re-export with the current memory "
                "checkpoint."
            )
        self._validated_leaves[key] = leaf
        return leaf

    def _episode_array(self, dataset_path: str, episode_id: int) -> np.ndarray:
        cache_key = (str(dataset_path), int(episode_id))
        cached = self._episode_cache.get(cache_key)
        if cached is not None:
            self._episode_cache.move_to_end(cache_key)
            return cached
        leaf = self._leaf_dir(dataset_path)
        ep_path = leaf / f"episode_{int(episode_id)}.npy"
        if not ep_path.is_file():
            raise FileNotFoundError(
                f"Missing memory-track episode file {ep_path} (leaf exported "
                "incompletely? re-run the exporter for this leaf)."
            )
        arr = np.load(ep_path, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[1] != CODE_DIM:
            raise ValueError(
                f"Memory-track episode {ep_path} has shape {arr.shape}; expected "
                f"(T, {CODE_DIM})."
            )
        self._episode_cache[cache_key] = arr
        if len(self._episode_cache) > _EPISODE_CACHE_SIZE:
            self._episode_cache.popitem(last=False)
        return arr

    # -- public API -----------------------------------------------------------

    def episode_ids(self, dataset_path: str) -> list[int]:
        """Sorted episode ids with exported tracks for a leaf."""
        key = str(dataset_path)
        if key not in self._episode_ids:
            leaf = self._leaf_dir(key)
            ids = sorted(
                int(p.stem.split("_", 1)[1]) for p in leaf.glob("episode_*.npy")
            )
            if not ids:
                raise FileNotFoundError(f"Memory track {leaf} contains no episodes.")
            self._episode_ids[key] = ids
        return self._episode_ids[key]

    def lookup(self, dataset_path: str, episode_id: int, base_index: int) -> np.ndarray:
        """Code at the sample's anchor frame, clamped to the track length."""
        arr = self._episode_array(dataset_path, episode_id)
        idx = min(max(int(base_index), 0), arr.shape[0] - 1)
        return np.asarray(arr[idx], dtype=np.float32)


class CampMemoryTrackJoiner(Dataset):
    """Attach a per-sample 132D ``memory_code`` to an Open-H dataset.

    Wraps an ``OpenHMixedLeRobotDataset`` constructed with
    ``emit_step_ids=True`` and pops each sample's ``"_step_ids"`` (computed
    inside the base dataset's successful fetch attempt — retry-reroll safe).
    The added ``"memory_code"`` is consumed downstream by
    ``CampActionTransformPipeline``.

    Args:
        base: The Open-H mixture dataset (``emit_step_ids=True`` required).
        tracks_root: Exported-tracks directory, or ``"__random__"`` for
            deterministic debug codes (plumbing/packing smokes only).
        memory_ablation: ``None`` | ``"zero"`` | ``"shuffle_episode"``
            (see module docstring). ``shuffle_episode`` requires real tracks.
    """

    def __init__(
        self,
        base: Dataset,
        tracks_root: str,
        memory_ablation: str | None = None,
    ) -> None:
        super().__init__()
        if memory_ablation not in MEMORY_ABLATIONS:
            raise ValueError(
                f"memory_ablation must be one of {MEMORY_ABLATIONS}, got {memory_ablation!r}"
            )
        if not getattr(base, "emit_step_ids", False):
            raise ValueError(
                "CampMemoryTrackJoiner requires the base dataset to be constructed "
                "with emit_step_ids=True — joining via get_step_ids(idx) after the "
                "fact is UNSAFE (the base retry loop rerolls indices)."
            )
        self._base = base
        self.memory_ablation = memory_ablation
        self._random_debug = str(tracks_root) == RANDOM_DEBUG_SENTINEL
        if self._random_debug:
            if memory_ablation == "shuffle_episode":
                raise ValueError(
                    "memory_ablation='shuffle_episode' is meaningless with "
                    "tracks_root='__random__' (debug codes are already episode-free)."
                )
            self._store = None
            print(
                "=" * 78
                + "\nCAMP MEMORY: tracks_root='__random__' — feeding DETERMINISTIC RANDOM "
                "codes.\nThis is a plumbing/packing smoke mode, NOT a science arm.\n"
                + "=" * 78
            )
        else:
            self._store = MemoryTrackStore(tracks_root)

    # -- dataset protocol ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._base)

    @property
    def virtual_sizes(self) -> list[int]:
        """Pass-through for _OpenHShuffleBlockAdapter.get_shuffle_blocks."""
        return self._base.virtual_sizes

    def get_step_ids(self, idx: int) -> dict[str, Any]:
        return self._base.get_step_ids(idx)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._base[idx]
        ids = sample.pop("_step_ids", None)
        if ids is None:
            raise KeyError(
                "Base sample carries no '_step_ids' — was the base dataset built "
                "with emit_step_ids=True?"
            )
        sample["memory_code"] = self._code_for(ids)
        return sample

    # -- code sources ------------------------------------------------------------

    @staticmethod
    def _sample_seed(ids: dict[str, Any], salt: int = 0) -> int:
        key = f"{ids['dataset_path']}|{ids['episode_id']}|{ids['base_index']}|{salt}"
        return int.from_bytes(hashlib.sha1(key.encode()).digest()[:4], "little") & 0x7FFF_FFFF

    def _code_for(self, ids: dict[str, Any]) -> torch.Tensor:
        if self.memory_ablation == "zero":
            return torch.zeros(CODE_DIM, dtype=torch.float32)

        if self._random_debug:
            generator = torch.Generator()
            generator.manual_seed(self._sample_seed(ids))
            # Uniform in [-1, 1] — the tanh range real codes live in.
            return torch.rand(CODE_DIM, generator=generator, dtype=torch.float32) * 2.0 - 1.0

        assert self._store is not None
        episode_id = int(ids["episode_id"])
        if self.memory_ablation == "shuffle_episode":
            candidates = self._store.episode_ids(ids["dataset_path"])
            if len(candidates) < 2:
                raise ValueError(
                    f"shuffle_episode ablation needs >= 2 exported episodes for leaf "
                    f"{ids['dataset_path']!r}, found {len(candidates)}."
                )
            # Deterministic donor pick, never the sample's own episode.
            pool = [e for e in candidates if e != episode_id]
            episode_id = pool[self._sample_seed(ids, salt=1) % len(pool)]

        code = self._store.lookup(ids["dataset_path"], episode_id, int(ids["base_index"]))
        return torch.from_numpy(code)
