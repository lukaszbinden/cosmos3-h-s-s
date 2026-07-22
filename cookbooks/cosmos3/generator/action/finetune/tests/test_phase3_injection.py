# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CAMP Phase-3 tests: memory-row injection.

``inject_memory_rows`` (framework-free) is exercised against the PINNED
framework's ``build_sequence_plan_from_mode`` — AST-extracted from the pinned
source so the plan semantics tested are the real ones, not a reimplementation
(the full transforms module needs the cluster dep set, e.g. torch>=2.3).

The framework-facing ``CampActionTransformPipeline`` subclass is validated by
import + behavior wherever the full framework stack is available (cluster
workspace env) and skips locally.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
import torch

ci = importlib.import_module("cosmos_framework.data.vfm.action.camp_injection")
contract = importlib.import_module("cosmos_framework.data.vfm.action.camp_data_contract")


class _StubSequencePlan:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _load_pinned_plan_builder(framework_root: Path):
    src_path = framework_root / "cosmos_framework/data/vfm/action/transforms.py"
    tree = ast.parse(src_path.read_text())
    fn_node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "build_sequence_plan_from_mode"
    )
    ns = {"SequencePlan": _StubSequencePlan}
    exec(compile(ast.Module(body=[fn_node], type_ignores=[]), str(src_path), "exec"), ns)
    return ns["build_sequence_plan_from_mode"]


@pytest.fixture(scope="session")
def plan_builder(framework_root):
    if framework_root is None:
        pytest.skip("no framework checkout found (set COSMOS3_FRAMEWORK_DIR)")
    return _load_pinned_plan_builder(framework_root)


def _padded_action(t: int = 28) -> torch.Tensor:
    return torch.arange(t * 44, dtype=torch.float32).reshape(t, 44)


def _code() -> torch.Tensor:
    return torch.linspace(-1, 1, contract.CODE_DIM)


class TestInjectMemoryRows:
    def test_shapes_and_content(self, plan_builder):
        action = _padded_action()
        code = _code()
        new_action, plan = ci.inject_memory_rows(
            action, code, mode="policy", video_length=13,
            num_history_actions=16, sequence_plan_builder=plan_builder,
        )
        assert new_action.shape == (31, 44)
        assert torch.equal(new_action[:3], code.view(3, 44))
        assert torch.equal(new_action[3:], action)

    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("forward_dynamics", list(range(31))),
            ("policy", list(range(19))),
            ("inverse_dynamics", list(range(19))),
        ],
    )
    def test_sequence_plans_match_contract(self, plan_builder, mode, expected):
        _, plan = ci.inject_memory_rows(
            _padded_action(), _code(), mode=mode, video_length=13,
            num_history_actions=16, sequence_plan_builder=plan_builder,
        )
        assert list(plan.condition_frame_indexes_action) == expected
        if mode in ("policy", "inverse_dynamics"):
            assert len(plan.condition_frame_indexes_action) == contract.NUM_CONDITIONING_ROWS

    def test_plan_matches_contract_mask(self, plan_builder):
        """Framework plan == camp_data_contract.make_conditioning_mask."""
        for mode in ("forward_dynamics", "policy", "inverse_dynamics"):
            _, plan = ci.inject_memory_rows(
                _padded_action(), _code(), mode=mode, video_length=13,
                num_history_actions=16, sequence_plan_builder=plan_builder,
            )
            mask = contract.make_conditioning_mask(mode)
            expected_condition_rows = [i for i, clean in enumerate(mask) if clean]
            assert list(plan.condition_frame_indexes_action) == expected_condition_rows

    def test_memory_only_no_history(self, plan_builder):
        """H=0 + memory: 12 current rows + 3 code rows = 15; 3 conditioning."""
        new_action, plan = ci.inject_memory_rows(
            _padded_action(12), _code(), mode="policy", video_length=13,
            num_history_actions=0, sequence_plan_builder=plan_builder,
        )
        assert new_action.shape == (15, 44)
        assert list(plan.condition_frame_indexes_action) == [0, 1, 2]

    def test_denoising_rows_unchanged_by_injection(self, plan_builder):
        """Injection must not shift WHICH values are denoised: the last 12
        rows of the new tensor are the original current window."""
        action = _padded_action()
        new_action, plan = ci.inject_memory_rows(
            action, _code(), mode="policy", video_length=13,
            num_history_actions=16, sequence_plan_builder=plan_builder,
        )
        n_cond = len(plan.condition_frame_indexes_action)
        assert torch.equal(new_action[n_cond:], action[16:])

    def test_accepts_pre_reshaped_code(self, plan_builder):
        code = _code().view(3, 44)
        new_action, _ = ci.inject_memory_rows(
            _padded_action(), code, mode="policy", video_length=13,
            num_history_actions=16, sequence_plan_builder=plan_builder,
        )
        assert torch.equal(new_action[:3], code)

    def test_dtype_and_device_follow_action(self, plan_builder):
        action = _padded_action().to(torch.bfloat16)
        new_action, _ = ci.inject_memory_rows(
            action, _code().to(torch.float64), mode="policy", video_length=13,
            num_history_actions=16, sequence_plan_builder=plan_builder,
        )
        assert new_action.dtype == torch.bfloat16

    def test_raw_action_dim_guard_fails_closed_on_unpatched_framework(
        self, plan_builder, monkeypatch
    ):
        monkeypatch.setattr(ci, "model_memory_row_patch_present", lambda: False)
        with pytest.raises(ValueError, match="Phase-3b"):
            ci.inject_memory_rows(
                _padded_action(), _code(), mode="policy", video_length=13,
                num_history_actions=16, sequence_plan_builder=plan_builder,
                raw_action_dim=torch.tensor(20),
            )

    def test_narrow_raw_dim_passes_when_model_patched(self, plan_builder, monkeypatch):
        monkeypatch.setattr(ci, "model_memory_row_patch_present", lambda: True)
        new_action, _ = ci.inject_memory_rows(
            _padded_action(), _code(), mode="policy", video_length=13,
            num_history_actions=16, sequence_plan_builder=plan_builder,
            raw_action_dim=torch.tensor(20),
        )
        assert new_action.shape == (31, 44)

    def test_patch_detection_against_overlaid_tree(self):
        """In the overlaid checkout the dataclass carries the new field, so
        detection must return True (data_and_condition is torch-only)."""
        try:
            assert ci.model_memory_row_patch_present() is True
        except AssertionError:
            raise
        except Exception as e:  # pragma: no cover - environment dependent
            pytest.skip(f"data_and_condition not importable here: {e}")

    @pytest.mark.parametrize("raw", [44, torch.tensor(44), None])
    def test_raw_action_dim_full_width_or_disabled_passes(self, plan_builder, raw):
        new_action, _ = ci.inject_memory_rows(
            _padded_action(), _code(), mode="policy", video_length=13,
            num_history_actions=16, sequence_plan_builder=plan_builder,
            raw_action_dim=raw,
        )
        assert new_action.shape == (31, 44)

    def test_bad_code_shapes_raise(self, plan_builder):
        for bad in (torch.randn(131), torch.randn(4, 44), torch.randn(3, 43)):
            with pytest.raises(ValueError, match="memory_code"):
                ci.inject_memory_rows(
                    _padded_action(), bad, mode="policy", video_length=13,
                    num_history_actions=16, sequence_plan_builder=plan_builder,
                )

    def test_nonfinite_code_raises(self, plan_builder):
        code = _code()
        code[7] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            ci.inject_memory_rows(
                _padded_action(), code, mode="policy", video_length=13,
                num_history_actions=16, sequence_plan_builder=plan_builder,
            )

    def test_bad_action_shape_raises(self, plan_builder):
        with pytest.raises(ValueError, match="expected action"):
            ci.inject_memory_rows(
                torch.randn(28, 20), _code(), mode="policy", video_length=13,
                num_history_actions=16, sequence_plan_builder=plan_builder,
            )


class TestCampPipelineClass:
    """Framework-facing subclass — runs only where the full stack imports
    (cluster workspace env); local envs skip on the torch>=2.3 dep chain."""

    def test_subclass_and_call_wiring(self):
        try:
            ct = importlib.import_module("cosmos_framework.data.vfm.action.camp_transforms")
            tf = importlib.import_module("cosmos_framework.data.vfm.action.transforms")
        except Exception as e:  # pragma: no cover - environment dependent
            pytest.skip(f"framework transforms not importable here: {e}")
        assert issubclass(ct.CampActionTransformPipeline, tf.ActionTransformPipeline)
        pipe = ct.CampActionTransformPipeline(max_action_dim=44)
        assert pipe.memory_code_key == "memory_code"
        assert pipe.require_memory_code is True
