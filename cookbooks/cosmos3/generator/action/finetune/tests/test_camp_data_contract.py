# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for the canonical multi-embodiment CAMP data contract.

These tests are deliberately free of heavy dependencies (no PyTorch, no
cosmos_framework imports) so they can run in a base Python environment as a
pre-flight check before any training or data-pipeline work.

Run with:
    python -m pytest cookbooks/cosmos3/generator/action/finetune/tests/ -v
"""

from __future__ import annotations

import sys
import os

# Make the framework_patch tree importable without installing it.  NOTE: if a
# real cosmos_framework package is installed in the active environment, Python
# resolves the regular (installed) package before this namespace-package tree —
# in that case run apply_overlay.sh first so camp_data_contract.py exists in
# the installed package too.
_PATCH_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "framework_patch")
)
if _PATCH_ROOT not in sys.path:
    sys.path.insert(0, _PATCH_ROOT)

import pytest

from cosmos_framework.data.vfm.action.camp_data_contract import (
    ACTION_DIM,
    CODE_DIM,
    CURRENT_SLICE,
    HISTORY_SLICE,
    MEMORY_SLICE,
    NUM_CONDITIONING_ROWS,
    NUM_CURRENT_ROWS,
    NUM_HISTORY_ROWS,
    NUM_MEMORY_SLOTS,
    TOTAL_ACTION_ROWS,
    assert_contract_invariants,
    make_conditioning_mask,
)


# ---------------------------------------------------------------------------
# Layout invariants
# ---------------------------------------------------------------------------


class TestLayoutConstants:
    def test_total_rows(self):
        assert TOTAL_ACTION_ROWS == 31

    def test_row_decomposition(self):
        assert NUM_MEMORY_SLOTS + NUM_HISTORY_ROWS + NUM_CURRENT_ROWS == TOTAL_ACTION_ROWS

    def test_memory_slots(self):
        assert NUM_MEMORY_SLOTS == 3

    def test_history_rows(self):
        assert NUM_HISTORY_ROWS == 16

    def test_current_rows_unchanged(self):
        # Public contract: 12 current rows must never change.
        assert NUM_CURRENT_ROWS == 12

    def test_action_dim_unchanged(self):
        # Public contract: 44D must never change.
        assert ACTION_DIM == 44

    def test_conditioning_rows(self):
        assert NUM_CONDITIONING_ROWS == NUM_MEMORY_SLOTS + NUM_HISTORY_ROWS
        assert NUM_CONDITIONING_ROWS == 19

    def test_code_dim(self):
        assert CODE_DIM == NUM_MEMORY_SLOTS * ACTION_DIM
        assert CODE_DIM == 132


class TestSlices:
    def test_memory_slice_start(self):
        assert MEMORY_SLICE.start == 0

    def test_memory_slice_stop(self):
        assert MEMORY_SLICE.stop == NUM_MEMORY_SLOTS

    def test_history_slice_start(self):
        assert HISTORY_SLICE.start == NUM_MEMORY_SLOTS

    def test_history_slice_stop(self):
        assert HISTORY_SLICE.stop == NUM_MEMORY_SLOTS + NUM_HISTORY_ROWS

    def test_current_slice_start(self):
        assert CURRENT_SLICE.start == NUM_MEMORY_SLOTS + NUM_HISTORY_ROWS

    def test_current_slice_stop(self):
        assert CURRENT_SLICE.stop == TOTAL_ACTION_ROWS

    def test_slices_cover_full_sequence(self):
        indices = (
            list(range(*MEMORY_SLICE.indices(TOTAL_ACTION_ROWS)))
            + list(range(*HISTORY_SLICE.indices(TOTAL_ACTION_ROWS)))
            + list(range(*CURRENT_SLICE.indices(TOTAL_ACTION_ROWS)))
        )
        assert sorted(indices) == list(range(TOTAL_ACTION_ROWS)), (
            "Memory + history + current slices must partition 0..TOTAL_ACTION_ROWS-1 "
            "without gaps or overlaps."
        )

    def test_slices_do_not_overlap(self):
        mem = set(range(*MEMORY_SLICE.indices(TOTAL_ACTION_ROWS)))
        hist = set(range(*HISTORY_SLICE.indices(TOTAL_ACTION_ROWS)))
        cur = set(range(*CURRENT_SLICE.indices(TOTAL_ACTION_ROWS)))
        assert mem.isdisjoint(hist)
        assert mem.isdisjoint(cur)
        assert hist.isdisjoint(cur)


# ---------------------------------------------------------------------------
# Conditioning masks
# ---------------------------------------------------------------------------


class TestConditioningMasks:
    def test_fd_mask_length(self):
        mask = make_conditioning_mask("forward_dynamics")
        assert len(mask) == TOTAL_ACTION_ROWS

    def test_fd_all_conditioning(self):
        mask = make_conditioning_mask("forward_dynamics")
        assert all(mask), "FD: every row must be clean conditioning (True)."

    def test_fd_no_denoising_targets(self):
        mask = make_conditioning_mask("forward_dynamics")
        assert sum(1 for m in mask if not m) == 0

    def test_policy_mask_length(self):
        mask = make_conditioning_mask("policy")
        assert len(mask) == TOTAL_ACTION_ROWS

    def test_policy_conditioning_region(self):
        mask = make_conditioning_mask("policy")
        assert all(mask[:NUM_CONDITIONING_ROWS]), (
            "Policy: memory + history rows must be clean conditioning."
        )

    def test_policy_denoising_region(self):
        mask = make_conditioning_mask("policy")
        assert not any(mask[NUM_CONDITIONING_ROWS:]), (
            "Policy: current action rows must be denoising targets."
        )

    def test_policy_denoising_count(self):
        mask = make_conditioning_mask("policy")
        assert sum(1 for m in mask if not m) == NUM_CURRENT_ROWS

    def test_id_matches_policy(self):
        assert make_conditioning_mask("inverse_dynamics") == make_conditioning_mask("policy")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            make_conditioning_mask("joint")

    def test_memory_rows_always_conditioning(self):
        for mode in ("forward_dynamics", "policy", "inverse_dynamics"):
            mask = make_conditioning_mask(mode)
            assert all(mask[MEMORY_SLICE]), (
                f"Mode {mode!r}: memory rows must always be conditioning."
            )

    def test_history_rows_always_conditioning(self):
        for mode in ("forward_dynamics", "policy", "inverse_dynamics"):
            mask = make_conditioning_mask(mode)
            hist_mask = mask[HISTORY_SLICE.start : HISTORY_SLICE.stop]
            assert all(hist_mask), (
                f"Mode {mode!r}: history rows must always be conditioning."
            )

    def test_current_rows_denoised_only_for_policy_and_id(self):
        fd_mask = make_conditioning_mask("forward_dynamics")
        assert all(fd_mask[CURRENT_SLICE.start : CURRENT_SLICE.stop]), (
            "FD: current rows must be conditioning (not denoising)."
        )
        for mode in ("policy", "inverse_dynamics"):
            m = make_conditioning_mask(mode)
            assert not any(m[CURRENT_SLICE.start : CURRENT_SLICE.stop]), (
                f"Mode {mode!r}: current rows must be denoising targets."
            )


# ---------------------------------------------------------------------------
# assert_contract_invariants re-entry
# ---------------------------------------------------------------------------


class TestContractInvariants:
    def test_assert_contract_invariants_passes(self):
        # Should not raise; invariants are satisfied by the module constants.
        assert_contract_invariants()

    def test_invariant_check_ran_at_import(self):
        # The module calls assert_contract_invariants() at import time.
        # If we reached this point the import succeeded, which means it passed.
        pass


# ---------------------------------------------------------------------------
# Shape helpers (pure Python, no torch)
# ---------------------------------------------------------------------------


class TestSequenceShape:
    def test_full_sequence_shape(self):
        rows = TOTAL_ACTION_ROWS
        cols = ACTION_DIM
        # Simulate a flat list representing (rows, cols) and verify indexing.
        seq = [[0.0] * cols for _ in range(rows)]
        assert len(seq) == 31
        assert len(seq[0]) == 44

    def test_memory_code_reshape(self):
        # 132D code reshaped to (3, 44) must match NUM_MEMORY_SLOTS × ACTION_DIM.
        flat_code = [0.0] * CODE_DIM
        reshaped = [
            flat_code[i * ACTION_DIM : (i + 1) * ACTION_DIM]
            for i in range(NUM_MEMORY_SLOTS)
        ]
        assert len(reshaped) == NUM_MEMORY_SLOTS
        assert all(len(row) == ACTION_DIM for row in reshaped)

    def test_history_native_dim_padded_to_44(self):
        # History for a 20D embodiment (e.g. jhu_dvrk_mono) must pad to ACTION_DIM.
        native_dim = 20
        padded_row = [0.0] * native_dim + [0.0] * (ACTION_DIM - native_dim)
        assert len(padded_row) == ACTION_DIM

    def test_history_full_block(self):
        history = [[0.0] * ACTION_DIM for _ in range(NUM_HISTORY_ROWS)]
        assert len(history) == 16
        assert all(len(r) == 44 for r in history)


# ---------------------------------------------------------------------------
# Cross-checks against the real data stack (skipped when torch/albumentations
# are unavailable — they run in the workspace training environment)
# ---------------------------------------------------------------------------


class TestContractMatchesDataStack:
    def test_action_dim_matches_max_action_dim(self):
        """ACTION_DIM here must equal groot_configs.MAX_ACTION_DIM.

        The contract module deliberately avoids importing the (torch-heavy)
        groot_configs, so the two constants can drift.  This test pins them
        together wherever the full stack is importable.
        """
        pytest.importorskip("torch")
        try:
            from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
                MAX_ACTION_DIM,
            )
        except ImportError as e:
            pytest.skip(f"cosmos_framework data stack not importable: {e}")
        assert MAX_ACTION_DIM == ACTION_DIM

    def test_draco_cmr_root_rebases_to_original_resolution_leaves(self):
        """The separate Draco CMR mirror must not inherit DATASET_PATH."""
        pytest.importorskip("torch")
        try:
            from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
                get_open_h_multi_train_specs,
            )
        except ImportError as e:
            pytest.skip(f"cosmos_framework data stack not importable: {e}")

        cmr_root = "/draco/cmr-surgical-60hz-fixed"
        specs = get_open_h_multi_train_specs(
            base_path="/draco/public/Surgical",
            cmr_base_path=cmr_root,
        )
        cmr_paths = sorted(
            spec["path"] for spec in specs if spec["embodiment"].value == "cmr_versius"
        )
        assert cmr_paths == sorted(
            [
                f"{cmr_root}/cholecystectomy",
                f"{cmr_root}/hysterectomy",
                f"{cmr_root}/inguinal_hernia",
                f"{cmr_root}/prostatectomy",
            ]
        )

    def test_clean_catalog_scales_nominal_cmr_share_to_30_percent(self):
        pytest.importorskip("torch")
        try:
            from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
                get_open_h_multi_train_specs,
            )
        except ImportError as e:
            pytest.skip(f"cosmos_framework data stack not importable: {e}")

        specs = get_open_h_multi_train_specs(
            cmr_clean_catalog_root="/catalog/cmr-clean-v1",
            cmr_target_share=0.30,
        )
        cmr = [
            spec
            for spec in specs
            if spec["embodiment"].value == "cmr_versius"
        ]
        non_cmr = [
            spec
            for spec in specs
            if spec["embodiment"].value != "cmr_versius"
        ]
        cmr_weight = sum(float(spec["mix_ratio"]) for spec in cmr)
        non_cmr_weight = sum(float(spec["mix_ratio"]) for spec in non_cmr)
        assert cmr_weight / (cmr_weight + non_cmr_weight) == pytest.approx(0.30)
        assert all(
            spec["cmr_clean_catalog_root"] == "/catalog/cmr-clean-v1"
            for spec in cmr
        )
        assert all("cmr_clean_catalog_root" not in spec for spec in non_cmr)

    def test_cmr_target_share_without_catalog_fails_closed(self):
        pytest.importorskip("torch")
        try:
            from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
                get_open_h_multi_train_specs,
            )
        except ImportError as e:
            pytest.skip(f"cosmos_framework data stack not importable: {e}")

        with pytest.raises(ValueError, match="requires cmr_clean_catalog_root"):
            get_open_h_multi_train_specs(cmr_target_share=0.30)

    def test_draco_internal_layout_maps_every_non_cmr_leaf(self):
        """Every EOS-authored leaf must have an explicit Draco counterpart."""
        pytest.importorskip("torch")
        try:
            from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
                get_open_h_multi_train_specs,
            )
        except ImportError as e:
            pytest.skip(f"cosmos_framework data stack not importable: {e}")

        root = "/draco/Open-H/Surgical"
        cmr_root = "/draco/Open-H/cmr-surgical-60hz-fixed"
        lz_root = "/draco/Open-H-lz"
        specs = get_open_h_multi_train_specs(
            base_path=root,
            cmr_base_path=cmr_root,
            path_layout="draco_internal",
            openh_lz_base_path=lz_root,
        )
        assert len(specs) == 36
        paths = {spec["path"] for spec in specs}
        assert f"{lz_root}/Surgical/JHU/Imerse/previously_collected_data/srth_porcine_chole_fix" in paths
        assert f"{lz_root}/Surgical/JHU/Imerse/previously_collected_data/hf_suturebot" in paths
        assert f"{root}/JHU/LSCR/MIRACLE /Prepare to Pierce" in paths
        assert f"{root}/Obuda/FRS_Dome_1" in paths
        assert (
            f"{root}/Stanford/Collaborative Haptics and Robotics in Medicine Lab/"
            "Real Robot (dVRK)/Needle Transfer"
        ) in paths
        assert f"{root}/Turin/mitic_lerobot_plastic_pad_3DMED" in paths
        assert f"{root}/TUD/260131_TUNDRA_dataset/grasping_retraction" in paths
        assert f"{cmr_root}/cholecystectomy" in paths

    def test_every_spec_embodiment_has_effective_fps_source(self):
        """Every spec'd embodiment must resolve its effective FPS explicitly.

        Regression net for the CMR 30-vs-10 Hz bug: an embodiment handled by
        a dedicated code path (not in EMBODIMENT_REGISTRY) silently fell back
        to the raw storage FPS.  Every embodiment reachable from the Open-H
        specs must be in the registry or in _OUT_OF_REGISTRY_EFFECTIVE_FPS.
        """
        pytest.importorskip("torch")
        try:
            from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
                EMBODIMENT_REGISTRY,
                OPEN_H_DATASET_SPECS,
            )
            from cosmos_framework.data.vfm.action.gr00t_dreams.data.embodiment_tags import (
                EmbodimentTag,
            )
            from cosmos_framework.data.vfm.action.open_h_dataset import (
                _OUT_OF_REGISTRY_EFFECTIVE_FPS,
            )
        except ImportError as e:
            pytest.skip(f"cosmos_framework data stack not importable: {e}")

        unresolved = []
        for spec in OPEN_H_DATASET_SPECS:
            emb = spec["embodiment"]
            emb = emb.value if isinstance(emb, EmbodimentTag) else emb
            if emb not in EMBODIMENT_REGISTRY and emb not in _OUT_OF_REGISTRY_EFFECTIVE_FPS:
                unresolved.append(emb)
        assert not unresolved, (
            f"Embodiments with no explicit effective-FPS source (would silently "
            f"fall back to storage FPS): {sorted(set(unresolved))}"
        )
