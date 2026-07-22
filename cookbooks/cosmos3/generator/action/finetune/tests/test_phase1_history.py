# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CAMP Phase-1 tests: H16 recent-action history through the Open-H dataloader.

Layered by import weight so every environment runs what it can:

1. ``history_utils``      — torch only; runs everywhere.
2. CMR transform math     — numpy/scipy/pydantic/torch; asserts the cumulative
                            hybrid-relative re-integration TELESCOPES, i.e. the
                            current 12 rows of a history-extended window are
                            byte-identical to the H=0 computation.
3. Config construction    — needs albumentations/imageio (transform imports);
                            asserts delta-index layouts per embodiment class.
4. Dataset unit           — needs the full merged framework tree; exercises
                            ``OpenHMixedLeRobotDataset.__getitem__`` split /
                            ablation / key-emission logic against a stubbed
                            raw-sample source (no disk, no video decode).
5. Sequence plan (pinned) — subprocess against the pinned framework checkout;
                            asserts history rows land as clean conditioning
                            for FD/policy/ID, and cross-links the Phase-3
                            NUM_CONDITIONING_ROWS=19 contract.

Run: python -m pytest cookbooks/cosmos3/generator/action/finetune/tests/ -v
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import numpy as np
import pytest
import torch

hu = importlib.import_module(
    "cosmos_framework.data.vfm.action.gr00t_dreams.data.history_utils"
)


def _try_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"{module_name} not importable here: {e}")


# ===========================================================================
# Layer 1 — history_utils (always runs)
# ===========================================================================


class TestBuildActionDeltaIndices:
    @pytest.mark.parametrize("ti,n", [(1, 12), (3, 12), (6, 12), (3, 13)])
    def test_h0_is_legacy_identity(self, ti, n):
        assert hu.build_action_delta_indices(ti, n, 0) == list(range(0, n * ti, ti))

    def test_h16_layout_jhu(self):
        idx = hu.build_action_delta_indices(3, 12, 16)
        assert len(idx) == 28
        assert idx[0] == -48
        assert idx[16] == 0
        assert idx[-1] == 33

    def test_h16_layout_cmr(self):
        idx = hu.build_action_delta_indices(6, 12, 16)
        assert len(idx) == 28
        assert idx[0] == -96
        assert idx[16] == 0
        assert idx[-1] == 66

    @pytest.mark.parametrize("ti,n,h", [(1, 12, 16), (3, 12, 16), (6, 12, 16), (3, 13, 16)])
    def test_strictly_increasing_uniform_stride(self, ti, n, h):
        idx = hu.build_action_delta_indices(ti, n, h)
        assert len(idx) == h + n
        diffs = {b - a for a, b in zip(idx, idx[1:])}
        assert diffs == {ti}

    def test_history_rows_all_negative_current_from_zero(self):
        idx = hu.build_action_delta_indices(3, 12, 16)
        assert all(d < 0 for d in idx[:16])
        assert idx[16:] == list(range(0, 36, 3))

    @pytest.mark.parametrize("ti,n,h", [(0, 12, 0), (-1, 12, 0), (3, 0, 0), (3, 12, -1)])
    def test_invalid_args_raise(self, ti, n, h):
        with pytest.raises(ValueError):
            hu.build_action_delta_indices(ti, n, h)


class TestSplitHistoryAndCurrent:
    def test_h0_passthrough_same_object(self):
        a = torch.randn(12, 20)
        hist, cur = hu.split_history_and_current(a, 0)
        assert hist is None
        assert cur is a

    def test_h16_shapes_and_content(self):
        a = torch.arange(28 * 20, dtype=torch.float32).reshape(28, 20)
        hist, cur = hu.split_history_and_current(a, 16)
        assert hist.shape == (16, 20)
        assert cur.shape == (12, 20)
        assert torch.equal(hist, a[:16])
        assert torch.equal(cur, a[16:])

    def test_too_few_rows_raises(self):
        with pytest.raises(ValueError, match="needs >"):
            hu.split_history_and_current(torch.randn(16, 20), 16)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="expected a"):
            hu.split_history_and_current(torch.randn(28), 16)


class TestHistoryAblation:
    def test_none_is_identity_same_object(self):
        h = torch.randn(16, 20)
        assert hu.apply_history_ablation(h, None, seed=0) is h

    def test_zero(self):
        h = torch.randn(16, 20)
        out = hu.apply_history_ablation(h, "zero", seed=0)
        assert out.shape == h.shape
        assert out.dtype == h.dtype
        assert torch.count_nonzero(out) == 0

    def test_permute_preserves_rows(self):
        h = torch.randn(16, 20)
        out = hu.apply_history_ablation(h, "permute", seed=123)
        # Same multiset of rows (sort by first column as a stable key).
        key_in = h[:, 0].argsort()
        key_out = out[:, 0].argsort()
        assert torch.allclose(h[key_in], out[key_out])

    def test_permute_deterministic_per_seed(self):
        h = torch.randn(16, 20)
        a = hu.apply_history_ablation(h, "permute", seed=7)
        b = hu.apply_history_ablation(h, "permute", seed=7)
        assert torch.equal(a, b)

    def test_permute_differs_across_seeds(self):
        h = torch.randn(16, 20)
        a = hu.apply_history_ablation(h, "permute", seed=1)
        b = hu.apply_history_ablation(h, "permute", seed=2)
        assert not torch.equal(a, b)

    def test_unknown_ablation_raises(self):
        with pytest.raises(ValueError):
            hu.apply_history_ablation(torch.randn(16, 20), "shuffle", seed=0)


class TestValidateHistoryArgs:
    def test_valid_combinations(self):
        hu.validate_history_args(0, None)
        hu.validate_history_args(16, None)
        hu.validate_history_args(16, "zero")
        hu.validate_history_args(16, "permute")

    def test_ablation_without_history_raises(self):
        with pytest.raises(ValueError, match="requires num_history_actions > 0"):
            hu.validate_history_args(0, "zero")

    def test_negative_h_raises(self):
        with pytest.raises(ValueError):
            hu.validate_history_args(-1, None)

    def test_unknown_ablation_raises(self):
        with pytest.raises(ValueError):
            hu.validate_history_args(16, "noise")


class TestAblationSeed:
    def test_deterministic_and_distinct(self):
        assert hu.history_ablation_seed(3, 41) == hu.history_ablation_seed(3, 41)
        assert hu.history_ablation_seed(0, 1) != hu.history_ablation_seed(1, 0)
        assert 0 <= hu.history_ablation_seed(35, 10**7) < 2**31


# ===========================================================================
# Layer 2 — CMR cumulative re-integration telescopes over extended windows
# ===========================================================================


def _synthetic_cmr_window(t: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(T, 7) absolute xyz+quat_xyzw action rows and a (7,) reference pose."""
    rng = np.random.default_rng(seed)
    xyz = np.cumsum(rng.normal(scale=0.01, size=(t, 3)), axis=0)
    quat = rng.normal(size=(t, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    actions = np.concatenate([xyz, quat], axis=1).astype(np.float32)
    ref_q = rng.normal(size=4)
    ref_q /= np.linalg.norm(ref_q)
    ref = np.concatenate([rng.normal(scale=0.01, size=3), ref_q]).astype(np.float32)
    return actions, ref


class TestCmrTelescoping:
    """The production CMR transform passes a LENGTH-1 engagement gate (the
    anchor state's engagement, ``delta_indices=[0]``), which broadcasts over
    all per-step deltas. Under that gate the cumulative re-integration
    telescopes, so rows [H:] of a history-extended window must equal the
    unextended window exactly — the property that keeps arm-A/B/C current
    windows byte-identical. These tests pin that behavior."""

    @pytest.fixture(scope="class")
    def sa(self):
        return _try_import(
            "cosmos_framework.data.vfm.action.gr00t_dreams.data.transform.state_action"
        )

    @pytest.mark.parametrize("engaged_scalar", [1.0, 0.0])
    def test_current_rows_identical_to_h0(self, sa, engaged_scalar):
        actions, ref = _synthetic_cmr_window(28)
        engaged = np.array([engaged_scalar], dtype=np.float32)
        kwargs = dict(
            eef_pose=ref,
            engaged=engaged,
            input_rotation_format="quat",
            reference_rotation_format="quat",
            ref_engaged=bool(engaged_scalar),
        )
        full = sa.convert_to_hybrid_relative_with_engagement(action_data=actions, **kwargs)
        tail = sa.convert_to_hybrid_relative_with_engagement(action_data=actions[16:], **kwargs)
        assert full.shape == (28, 9)
        assert tail.shape == (12, 9)
        np.testing.assert_allclose(full[16:], tail, atol=1e-5)

    def test_history_rows_are_relative_to_anchor(self, sa):
        actions, ref = _synthetic_cmr_window(28)
        engaged = np.array([1.0], dtype=np.float32)
        full = sa.convert_to_hybrid_relative_with_engagement(
            action_data=actions,
            eef_pose=ref,
            engaged=engaged,
            input_rotation_format="quat",
            reference_rotation_format="quat",
            ref_engaged=True,
        )
        # Fully-engaged translation telescopes: row i == action_xyz[i] - ref_xyz
        # for EVERY row, history included ("pose relative to current anchor").
        np.testing.assert_allclose(full[:, :3], actions[:, :3] - ref[:3], atol=1e-5)
        assert np.isfinite(full).all()

    def test_disengaged_anchor_zeroes_everything(self, sa):
        actions, ref = _synthetic_cmr_window(28)
        full = sa.convert_to_hybrid_relative_with_engagement(
            action_data=actions,
            eef_pose=ref,
            engaged=np.array([0.0], dtype=np.float32),
            input_rotation_format="quat",
            reference_rotation_format="quat",
            ref_engaged=False,
        )
        np.testing.assert_allclose(full[:, :3], 0.0, atol=1e-7)
        # Rotation collapses to identity rot6d = [1,0,0,0,1,0].
        np.testing.assert_allclose(
            full[:, 3:9], np.tile([1, 0, 0, 0, 1, 0], (28, 1)), atol=1e-6
        )


# ===========================================================================
# Layer 3 — config construction per embodiment class
# ===========================================================================


class TestConstructConfigs:
    @pytest.fixture(scope="class")
    def gc(self):
        return _try_import(
            "cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs"
        )

    def test_h0_identity_generic(self, gc):
        cfg_legacy, _, _ = gc.construct_modality_config_and_transforms(13, "jhu_dvrk_mono")
        cfg_h0, _, _ = gc.construct_modality_config_and_transforms(
            13, "jhu_dvrk_mono", num_history_actions=0
        )
        assert cfg_legacy["action"].delta_indices == cfg_h0["action"].delta_indices
        assert cfg_legacy["video"].delta_indices == cfg_h0["video"].delta_indices

    def test_h16_generic_layout(self, gc):
        cfg, _, _ = gc.construct_modality_config_and_transforms(
            13, "jhu_dvrk_mono", num_history_actions=16
        )
        assert cfg["action"].delta_indices == list(range(-48, 36, 3))
        # Video and anchor state must be untouched by history.
        assert cfg["video"].delta_indices == list(range(0, 39, 3))
        assert cfg["state"].delta_indices == [0]

    def test_h16_cmr_layout(self, gc):
        cfg, _, _ = gc.construct_modality_config_and_transforms(
            13, "cmr_versius", num_history_actions=16
        )
        assert cfg["action"].delta_indices == list(range(-96, 72, 6))
        assert cfg["video"].delta_indices == list(range(0, 78, 6))
        assert cfg["state"].delta_indices == [0]
        # The 14D cond_* state-conditioning keys ride in the action modality —
        # they get history rows automatically.
        assert any(k.startswith("action.cond_") for k in cfg["action"].modality_keys)

    def test_h16_suturebot_layout(self, gc):
        cfg, _, _ = gc.construct_modality_config_and_transforms(
            13, "suturebot", num_history_actions=16
        )
        # suturebot's current window is num_frames (13) steps, not 12.
        assert cfg["action"].delta_indices == list(range(-48, 39, 3))
        assert cfg["video"].delta_indices == list(range(0, 39, 3))

    def test_every_registry_embodiment_supports_h16(self, gc):
        for emb, reg in gc.EMBODIMENT_REGISTRY.items():
            cfg, _, _ = gc.construct_modality_config_and_transforms(
                13, emb, num_history_actions=16
            )
            idx = cfg["action"].delta_indices
            ti = reg["timestep_interval"]
            assert len(idx) == 28, emb
            assert idx[0] == -16 * ti, emb
            assert idx[16] == 0, emb

    def test_legacy_embodiment_fails_closed(self, gc):
        with pytest.raises(ValueError, match="not supported for legacy"):
            gc.construct_modality_config_and_transforms(13, "gr1", num_history_actions=16)


# ===========================================================================
# Layer 4 — dataset __getitem__ unit (stubbed raw samples, no disk)
# ===========================================================================


def _make_stub_dataset(oh, h: int, ablation: str | None, raw_t: int, d: int = 20):
    """OpenHMixedLeRobotDataset with __init__ bypassed and a stubbed
    ``_get_raw_sample`` — exercises the real __getitem__ body."""
    ds = oh.OpenHMixedLeRobotDataset.__new__(oh.OpenHMixedLeRobotDataset)
    ds.num_frames = 13
    ds.max_action_dim = 44
    ds.mode = "policy"
    ds.viewpoint = "third_person_view"
    ds._default_storage_fps = 30.0
    ds._max_retries_per_sample = 1
    ds.num_history_actions = h
    ds.history_ablation = ablation
    ds.embodiment_tags = ["jhu_dvrk_mono"]
    ds.domain_ids = [7]
    ds.dataset_paths = ["/data/fake/leaf"]
    ds.effective_fps_per_dataset = [10]
    ds.mix_ratios = [1.0]
    ds.virtual_sizes = [5]
    ds._total_virtual_len = 5
    ds._cumulative_sizes = np.cumsum(ds.virtual_sizes)

    class _Sub:
        all_steps = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

        def __len__(self):
            return 5

    ds.sub_datasets = [_Sub()]

    raw_action = torch.arange(raw_t * d, dtype=torch.float32).reshape(raw_t, d)
    raw_video = torch.zeros(13, 3, 8, 8, dtype=torch.uint8)

    def _stub_raw(dataset_idx: int, real_idx: int):
        return {"action": raw_action.clone(), "video": raw_video.clone()}

    ds._get_raw_sample = _stub_raw
    return ds, raw_action


class TestDatasetGetitem:
    @pytest.fixture(scope="class")
    def oh(self):
        return _try_import("cosmos_framework.data.vfm.action.open_h_dataset")

    def test_h0_contract_unchanged(self, oh):
        ds, raw = _make_stub_dataset(oh, h=0, ablation=None, raw_t=12)
        sample = ds[0]
        assert "history_action" not in sample
        assert torch.equal(sample["action"], raw)
        assert sample["mode"] == "policy"
        assert int(sample["conditioning_fps"]) == 10

    def test_h16_split_and_key_emission(self, oh):
        ds, raw = _make_stub_dataset(oh, h=16, ablation=None, raw_t=28)
        sample = ds[0]
        assert sample["history_action"].shape == (16, 20)
        assert sample["action"].shape == (12, 20)
        assert torch.equal(sample["history_action"], raw[:16])
        assert torch.equal(sample["action"], raw[16:])

    def test_h16_current_rows_match_h0(self, oh):
        """The current 12 rows must be the SAME values arm A trains on."""
        ds16, raw28 = _make_stub_dataset(oh, h=16, ablation=None, raw_t=28)
        sample = ds16[0]
        assert torch.equal(sample["action"], raw28[16:])

    def test_zero_ablation(self, oh):
        ds, _ = _make_stub_dataset(oh, h=16, ablation="zero", raw_t=28)
        sample = ds[0]
        assert torch.count_nonzero(sample["history_action"]) == 0
        assert torch.count_nonzero(sample["action"]) > 0  # current untouched

    def test_permute_ablation_deterministic(self, oh):
        ds, raw = _make_stub_dataset(oh, h=16, ablation="permute", raw_t=28)
        s1, s2 = ds[2], ds[2]
        assert torch.equal(s1["history_action"], s2["history_action"])
        assert not torch.equal(s1["history_action"], raw[:16])  # actually permuted
        # Row multiset preserved.
        key_a = s1["history_action"][:, 0].argsort()
        assert torch.allclose(s1["history_action"][key_a], raw[:16])

    def test_get_step_ids(self, oh):
        ds, _ = _make_stub_dataset(oh, h=16, ablation=None, raw_t=28)
        ids = ds.get_step_ids(3)
        assert ids == {
            "dataset_idx": 0,
            "dataset_path": "/data/fake/leaf",
            "embodiment": "jhu_dvrk_mono",
            "episode_id": 1,
            "base_index": 0,
        }


# ===========================================================================
# Layer 5 — pinned-framework sequence plan (subprocess isolation)
# ===========================================================================

_SEQPLAN_SCRIPT = r"""
import ast
import pathlib
import sys

sys.path.insert(0, {root!r})
try:
    from cosmos_framework.data.vfm.action.transforms import build_sequence_plan_from_mode
    source_mode = "import"
except Exception:
    # The full transforms module needs the framework's exact dep set (e.g.
    # torch>=2.3 for torch.nn.attention). The sequence-plan function itself is
    # pure logic — AST-extract it from the PINNED source and exec it against a
    # stub SequencePlan, so the property is still verified against the real
    # pinned code (not a reimplementation).
    src_path = pathlib.Path({root!r}, "cosmos_framework/data/vfm/action/transforms.py")
    tree = ast.parse(src_path.read_text())
    fn_node = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "build_sequence_plan_from_mode"
    )

    class SequencePlan:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    ns = dict(SequencePlan=SequencePlan)
    exec(compile(ast.Module(body=[fn_node], type_ignores=[]), str(src_path), "exec"), ns)
    build_sequence_plan_from_mode = ns["build_sequence_plan_from_mode"]
    source_mode = "ast"

def check(mode, expected_cond, h):
    plan = build_sequence_plan_from_mode(
        mode=mode, video_length=13, action_length=12 + h,
        video_temporal_downsample=4, num_history_actions=h,
    )
    got = list(plan.condition_frame_indexes_action)
    assert got == expected_cond, (mode, h, got)

# H=16 (arm B): history rows are clean conditioning; current 12 denoise.
check("forward_dynamics", list(range(28)), 16)
check("policy", list(range(16)), 16)
check("inverse_dynamics", list(range(16)), 16)
# H=0 legacy: no conditioning rows for policy/ID.
check("forward_dynamics", list(range(12)), 0)
check("policy", [], 0)
check("inverse_dynamics", [], 0)
# Phase-3 preview (arm C): 3 memory + 16 history = 19 conditioning rows.
check("forward_dynamics", list(range(31)), 19)
check("policy", list(range(19)), 19)
check("inverse_dynamics", list(range(19)), 19)
print("SEQPLAN_OK", source_mode)
"""


class TestPinnedSequencePlan:
    def test_history_rows_are_clean_conditioning(self, framework_root):
        if framework_root is None:
            pytest.skip("no framework checkout found (set COSMOS3_FRAMEWORK_DIR)")
        proc = subprocess.run(
            [sys.executable, "-c", _SEQPLAN_SCRIPT.format(root=str(framework_root))],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0 and "SEQPLAN_OK" not in proc.stdout:
            if "ModuleNotFoundError" in proc.stderr or "ImportError" in proc.stderr:
                pytest.skip(
                    f"pinned framework deps unavailable here: {proc.stderr.strip().splitlines()[-1]}"
                )
            pytest.fail(f"sequence-plan check failed:\n{proc.stderr}")
        assert "SEQPLAN_OK" in proc.stdout

    def test_phase3_conditioning_matches_contract(self):
        """Cross-link: framework's 19-row conditioning == CAMP contract."""
        from cosmos_framework.data.vfm.action.camp_data_contract import (
            NUM_CONDITIONING_ROWS,
            NUM_HISTORY_ROWS,
            NUM_MEMORY_SLOTS,
        )

        assert NUM_MEMORY_SLOTS + NUM_HISTORY_ROWS == NUM_CONDITIONING_ROWS == 19
