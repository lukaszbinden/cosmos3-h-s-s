# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for the CAMP memory-track store + joiner (arm-C data wiring).

Pure CPU with synthetic track directories in tmp_path — also serves as the
executable specification of the track format the Phase-2b exporter must
write (directory naming, manifest fields, per-episode .npy layout).
"""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
import torch

mt = importlib.import_module("cosmos_framework.data.vfm.action.camp_memory_tracks")
contract = importlib.import_module("cosmos_framework.data.vfm.action.camp_data_contract")

LEAF_A = "/data/open-h/jhu/imerse/suturebot"
LEAF_B = "/data/open-h/LeRobot_540x960/hf_suturebot"  # same basename family, different path


def _write_track(root, dataset_path: str, episodes: dict[int, int], code_dim: int | None = None):
    """Write a synthetic track dir: {episode_id: length} -> deterministic codes."""
    code_dim = code_dim or contract.CODE_DIM
    leaf = root / mt.track_dir_name(dataset_path)
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "manifest.json").write_text(
        json.dumps({"dataset_path": dataset_path, "code_dim": code_dim})
    )
    for ep, length in episodes.items():
        arr = (
            np.arange(length * code_dim, dtype=np.float32).reshape(length, code_dim)
            + 1000.0 * ep
        )
        np.save(leaf / f"episode_{ep}.npy", arr)
    return leaf


class _StubBase(torch.utils.data.Dataset):
    """Minimal stand-in for OpenHMixedLeRobotDataset(emit_step_ids=True)."""

    emit_step_ids = True
    virtual_sizes = [4]

    def __init__(self, dataset_path: str = LEAF_A):
        self._path = dataset_path

    def __len__(self):
        return 4

    def get_step_ids(self, idx):
        return self._ids(idx)

    def _ids(self, idx):
        return {
            "dataset_idx": 0,
            "dataset_path": self._path,
            "embodiment": "jhu_dvrk_mono",
            "episode_id": idx % 2,          # episodes 0 and 1
            "base_index": 3 + idx,          # distinct frames
        }

    def __getitem__(self, idx):
        return {"action": torch.zeros(12, 20), "mode": "policy", "_step_ids": self._ids(idx)}


class TestTrackDirName:
    def test_same_basename_different_paths_do_not_collide(self):
        assert mt.track_dir_name(LEAF_A) != mt.track_dir_name(LEAF_B)

    def test_trailing_slash_insensitive(self):
        assert mt.track_dir_name(LEAF_A) == mt.track_dir_name(LEAF_A + "/")


class TestMemoryTrackStore:
    def test_lookup_returns_anchor_frame_code(self, tmp_path):
        _write_track(tmp_path, LEAF_A, {0: 10})
        store = mt.MemoryTrackStore(str(tmp_path))
        code = store.lookup(LEAF_A, episode_id=0, base_index=4)
        expected = np.arange(contract.CODE_DIM, dtype=np.float32) + 4 * contract.CODE_DIM
        np.testing.assert_array_equal(code, expected)
        assert code.dtype == np.float32

    def test_lookup_clamps_to_track_end(self, tmp_path):
        _write_track(tmp_path, LEAF_A, {0: 5})
        store = mt.MemoryTrackStore(str(tmp_path))
        end = store.lookup(LEAF_A, 0, 999)
        np.testing.assert_array_equal(end, store.lookup(LEAF_A, 0, 4))

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            mt.MemoryTrackStore(str(tmp_path / "nope"))

    def test_missing_leaf_raises_with_exporter_hint(self, tmp_path):
        _write_track(tmp_path, LEAF_A, {0: 5})
        store = mt.MemoryTrackStore(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="exporter"):
            store.lookup(LEAF_B, 0, 0)

    def test_manifest_path_mismatch_raises(self, tmp_path):
        leaf = _write_track(tmp_path, LEAF_A, {0: 5})
        (leaf / "manifest.json").write_text(
            json.dumps({"dataset_path": "/somewhere/else", "code_dim": contract.CODE_DIM})
        )
        store = mt.MemoryTrackStore(str(tmp_path))
        with pytest.raises(ValueError, match="manifest mismatch"):
            store.lookup(LEAF_A, 0, 0)

    def test_wrong_code_dim_raises(self, tmp_path):
        _write_track(tmp_path, LEAF_A, {0: 5}, code_dim=63)
        store = mt.MemoryTrackStore(str(tmp_path))
        with pytest.raises(ValueError, match="code_dim"):
            store.lookup(LEAF_A, 0, 0)

    def test_missing_episode_raises(self, tmp_path):
        _write_track(tmp_path, LEAF_A, {0: 5})
        store = mt.MemoryTrackStore(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="episode"):
            store.lookup(LEAF_A, 7, 0)

    def test_episode_ids_sorted(self, tmp_path):
        _write_track(tmp_path, LEAF_A, {3: 4, 0: 4, 11: 4})
        store = mt.MemoryTrackStore(str(tmp_path))
        assert store.episode_ids(LEAF_A) == [0, 3, 11]


class TestJoiner:
    def test_joins_code_for_the_returned_sample(self, tmp_path):
        _write_track(tmp_path, LEAF_A, {0: 20, 1: 20})
        joiner = mt.CampMemoryTrackJoiner(_StubBase(), tracks_root=str(tmp_path))
        sample = joiner[1]  # episode 1, base_index 4
        assert "_step_ids" not in sample  # consumed, never reaches the batch
        expected = (
            np.arange(contract.CODE_DIM, dtype=np.float32)
            + 4 * contract.CODE_DIM
            + 1000.0
        )
        np.testing.assert_array_equal(sample["memory_code"].numpy(), expected)
        assert sample["memory_code"].shape == (contract.CODE_DIM,)
        assert sample["memory_code"].dtype == torch.float32

    def test_requires_emit_step_ids(self, tmp_path):
        _write_track(tmp_path, LEAF_A, {0: 5})

        class _NoIds(_StubBase):
            emit_step_ids = False

        with pytest.raises(ValueError, match="emit_step_ids=True"):
            mt.CampMemoryTrackJoiner(_NoIds(), tracks_root=str(tmp_path))

    def test_zero_ablation(self, tmp_path):
        _write_track(tmp_path, LEAF_A, {0: 5, 1: 5})
        joiner = mt.CampMemoryTrackJoiner(
            _StubBase(), tracks_root=str(tmp_path), memory_ablation="zero"
        )
        assert torch.count_nonzero(joiner[0]["memory_code"]) == 0

    def test_shuffle_episode_ablation_uses_other_episode(self, tmp_path):
        _write_track(tmp_path, LEAF_A, {0: 20, 1: 20})
        joiner = mt.CampMemoryTrackJoiner(
            _StubBase(), tracks_root=str(tmp_path), memory_ablation="shuffle_episode"
        )
        sample = joiner[0]  # own episode 0, base 3 → donor must be episode 1
        expected = (
            np.arange(contract.CODE_DIM, dtype=np.float32)
            + 3 * contract.CODE_DIM
            + 1000.0
        )
        np.testing.assert_array_equal(sample["memory_code"].numpy(), expected)
        # Deterministic across calls.
        assert torch.equal(sample["memory_code"], joiner[0]["memory_code"])

    def test_shuffle_episode_needs_two_episodes(self, tmp_path):
        _write_track(tmp_path, LEAF_A, {0: 5})
        joiner = mt.CampMemoryTrackJoiner(
            _StubBase(), tracks_root=str(tmp_path), memory_ablation="shuffle_episode"
        )
        with pytest.raises(ValueError, match=">= 2 exported episodes"):
            joiner[0]

    def test_random_debug_source(self):
        joiner = mt.CampMemoryTrackJoiner(_StubBase(), tracks_root=mt.RANDOM_DEBUG_SENTINEL)
        a, b = joiner[0]["memory_code"], joiner[0]["memory_code"]
        assert torch.equal(a, b)  # deterministic per sample
        assert not torch.equal(a, joiner[1]["memory_code"])  # varies across samples
        assert a.shape == (contract.CODE_DIM,)
        assert a.min() >= -1.0 and a.max() <= 1.0  # tanh range

    def test_random_debug_rejects_shuffle_ablation(self):
        with pytest.raises(ValueError, match="meaningless"):
            mt.CampMemoryTrackJoiner(
                _StubBase(), tracks_root=mt.RANDOM_DEBUG_SENTINEL,
                memory_ablation="shuffle_episode",
            )

    def test_unknown_ablation_rejected(self, tmp_path):
        _write_track(tmp_path, LEAF_A, {0: 5})
        with pytest.raises(ValueError, match="memory_ablation"):
            mt.CampMemoryTrackJoiner(
                _StubBase(), tracks_root=str(tmp_path), memory_ablation="noise"
            )

    def test_passthroughs(self, tmp_path):
        _write_track(tmp_path, LEAF_A, {0: 5, 1: 5})
        joiner = mt.CampMemoryTrackJoiner(_StubBase(), tracks_root=str(tmp_path))
        assert len(joiner) == 4
        assert joiner.virtual_sizes == [4]
        assert joiner.get_step_ids(2)["episode_id"] == 0


class TestDatasetEmitsStepIds:
    """The base dataset must compute '_step_ids' INSIDE the successful fetch
    attempt (the retry loop rerolls indices)."""

    def test_stub_dataset_emits_matching_ids(self):
        oh = pytest.importorskip("torch") and None
        try:
            import cosmos_framework.data.vfm.action.open_h_dataset as ohmod
        except ImportError as e:  # pragma: no cover
            pytest.skip(f"open_h_dataset not importable: {e}")
        from tests.test_phase1_history import _make_stub_dataset

        ds, _ = _make_stub_dataset(ohmod, h=16, ablation=None, raw_t=28)
        ds.emit_step_ids = True
        sample = ds[3]
        assert sample["_step_ids"] == {
            "dataset_idx": 0,
            "dataset_path": "/data/fake/leaf",
            "embodiment": "jhu_dvrk_mono",
            "episode_id": 1,
            "base_index": 0,
        }
        # And absent by default (arms A/B contract untouched).
        ds.emit_step_ids = False
        assert "_step_ids" not in ds[3]
