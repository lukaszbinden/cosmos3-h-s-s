# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from cosmos_framework.data.vfm.action.gr00t_dreams.data.cmr_clean_catalog import (
    load_strict_cmr_catalog_steps,
    sha256_file,
)


def _builder_module():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "build_cmr_clean_catalog.py"
    )
    spec = importlib.util.spec_from_file_location("build_cmr_clean_catalog", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_catalog(
    tmp_path: Path,
    *,
    tier: str = "strict",
    training_authorized: bool = True,
    unknown_episodes: int = 0,
) -> tuple[Path, Path]:
    dataset = tmp_path / "cholecystectomy"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text('{"splits":{"train":"0:0"}}\n')
    (meta / "episodes.jsonl").write_text('{"episode_index":0,"length":20,"tasks":[]}\n')

    catalog = tmp_path / "catalog"
    catalog.mkdir()
    ranges = catalog / "cholecystectomy.npz"
    np.savez_compressed(
        ranges,
        episode_id=np.array([0, 0], dtype=np.int32),
        start=np.array([1, 8], dtype=np.int32),
        stop=np.array([3, 11], dtype=np.int32),
    )
    manifest = {
        "schema_version": 1,
        "catalog_kind": "cmr_clean_windows",
        "catalog_tier": tier,
        "training_authorized": training_authorized,
        "builder_script_sha256": "test-builder",
        "horizon": {"action_delta_indices": [0, 2, 4]},
        "criteria": {},
        "review": {
            "unknown_episodes": unknown_episodes,
            "required_fields": [
                "outcome",
                "manual_activity",
                "tool_exchange",
                "clean_intervals",
            ],
        },
        "procedures": {
            "cholecystectomy": {
                "dataset_fingerprint": {
                    "meta/info.json": sha256_file(meta / "info.json"),
                    "meta/episodes.jsonl": sha256_file(meta / "episodes.jsonl"),
                },
                "ranges_file": ranges.name,
                "ranges_sha256": sha256_file(ranges),
                "eligible_windows": 5,
            }
        },
    }
    identity_keys = (
        "schema_version",
        "catalog_kind",
        "catalog_tier",
        "training_authorized",
        "builder_script_sha256",
        "horizon",
        "criteria",
        "review",
        "procedures",
    )
    manifest["catalog_id"] = hashlib.sha256(
        json.dumps(
            {key: manifest[key] for key in identity_keys},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    (catalog / "manifest.json").write_text(json.dumps(manifest))
    return catalog, dataset


def test_strict_catalog_expands_verified_ranges(tmp_path: Path):
    catalog, dataset = _write_catalog(tmp_path)
    steps = load_strict_cmr_catalog_steps(
        catalog_root=catalog,
        dataset_path=dataset,
        action_delta_indices=[0, 2, 4],
    )
    assert steps == [(0, 1), (0, 2), (0, 8), (0, 9), (0, 10)]


@pytest.mark.parametrize(
    ("tier", "authorized", "unknown"),
    [
        ("candidate", False, 0),
        ("strict", False, 0),
        ("strict", True, 1),
    ],
)
def test_loader_rejects_any_non_strict_or_unreviewed_catalog(
    tmp_path: Path,
    tier: str,
    authorized: bool,
    unknown: int,
):
    catalog, dataset = _write_catalog(
        tmp_path,
        tier=tier,
        training_authorized=authorized,
        unknown_episodes=unknown,
    )
    with pytest.raises(ValueError, match="Invalid CMR clean catalog"):
        load_strict_cmr_catalog_steps(
            catalog_root=catalog,
            dataset_path=dataset,
            action_delta_indices=[0, 2, 4],
        )


def test_loader_rejects_dataset_metadata_drift(tmp_path: Path):
    catalog, dataset = _write_catalog(tmp_path)
    (dataset / "meta/info.json").write_text('{"splits":{"train":"0:1"}}\n')
    with pytest.raises(ValueError, match="different dataset metadata"):
        load_strict_cmr_catalog_steps(
            catalog_root=catalog,
            dataset_path=dataset,
            action_delta_indices=[0, 2, 4],
        )


def test_window_rules_cover_every_raw_frame_not_only_sampled_frames():
    builder = _builder_module()
    # A one-frame event at raw frame 3 invalidates starts whose contiguous
    # five-frame span includes it. A sampled-only [0,2,4] check would miss it.
    event = np.array([False, False, False, True, False, False, False])
    assert builder._window_has_event(event, width=5, num_starts=3).tolist() == [
        True,
        True,
        True,
    ]


def test_change_and_review_interval_are_full_horizon_constraints():
    builder = _builder_module()
    values = np.array([[1], [1], [2], [2], [2], [2]])
    assert builder._window_has_change(values, width=3, num_starts=4).tolist() == [
        True,
        True,
        False,
        False,
    ]
    allowed = builder._review_allowed_starts(
        num_starts=8,
        max_delta=4,
        clean_intervals=[[2, 9]],
    )
    # Only bases 2..4 have [base, base+4] wholly inside [2, 9).
    assert np.flatnonzero(allowed).tolist() == [2, 3, 4]


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        (
            {
                "review_status": "verified",
                "outcome": "success",
                "manual_activity": "absent",
                "tool_exchange": "absent",
                "clean_intervals": [[0, 10]],
            },
            "verified_clean",
        ),
        (
            {
                "review_status": "verified",
                "outcome": "failure",
                "manual_activity": "absent",
                "tool_exchange": "absent",
                "clean_intervals": [[0, 10]],
            },
            "failure",
        ),
        (
            {
                "review_status": "verified",
                "outcome": "recovery",
                "manual_activity": "absent",
                "tool_exchange": "absent",
                "clean_intervals": [[0, 10]],
            },
            "recovery",
        ),
        (
            {
                "review_status": "verified",
                "outcome": "success",
                "manual_activity": "present",
                "tool_exchange": "absent",
                "clean_intervals": [[0, 10]],
            },
            "manual_activity",
        ),
        (
            {
                "review_status": "verified",
                "outcome": "success",
                "manual_activity": "absent",
                "tool_exchange": "present",
                "clean_intervals": [[0, 10]],
            },
            "tool_exchange",
        ),
        (None, "unknown"),
    ],
)
def test_semantic_review_is_fail_closed(entry: dict | None, expected: str):
    builder = _builder_module()
    _, status = builder._strict_intervals(entry)
    assert status == expected


def test_catalog_identity_payload_is_stable():
    payload_a = {"procedures": {"b": 2, "a": 1}, "horizon": [0, 6]}
    payload_b = {"horizon": [0, 6], "procedures": {"a": 1, "b": 2}}
    digest = lambda value: hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest(payload_a) == digest(payload_b)
