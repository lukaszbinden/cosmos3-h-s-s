# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail-closed loader for reviewed CMR training-window catalogs.

The public CMR LeRobot leaves expose only a single ``train`` split.  They do
not annotate failure/recovery, manual intervention, or physical tool-exchange
segments.  The catalog builder therefore emits two artifact tiers:

``candidate``
    Telemetry-clean windows suitable for review.  Never accepted here.

``strict``
    Telemetry-clean windows intersected with explicitly reviewed clean
    intervals.  Only this tier is training-authorized.

Keeping the gate in the data loader is intentional: a typo, missing review
file, or accidentally supplied candidate catalog must stop training rather
than silently widen the data mixture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CMR_CLEAN_CATALOG_SCHEMA_VERSION = 1
CMR_CLEAN_CATALOG_KIND = "cmr_clean_windows"
CMR_CLEAN_CATALOG_MANIFEST = "manifest.json"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of *path* without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Invalid CMR clean catalog: {message}")


def _load_manifest(catalog_root: Path) -> dict[str, Any]:
    manifest_path = catalog_root / CMR_CLEAN_CATALOG_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"CMR clean catalog manifest not found: {manifest_path}"
        )
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    _require(
        manifest.get("schema_version") == CMR_CLEAN_CATALOG_SCHEMA_VERSION,
        "unsupported schema_version "
        f"{manifest.get('schema_version')!r}; expected "
        f"{CMR_CLEAN_CATALOG_SCHEMA_VERSION}",
    )
    _require(
        manifest.get("catalog_kind") == CMR_CLEAN_CATALOG_KIND,
        f"catalog_kind must be {CMR_CLEAN_CATALOG_KIND!r}",
    )
    _require(
        manifest.get("catalog_tier") == "strict",
        "only catalog_tier='strict' may be used for training "
        f"(got {manifest.get('catalog_tier')!r})",
    )
    _require(
        manifest.get("training_authorized") is True,
        "training_authorized must be true",
    )
    review = manifest.get("review", {})
    _require(
        review.get("unknown_episodes") == 0,
        "strict catalog contains episodes without semantic review",
    )
    _require(
        review.get("required_fields")
        == ["outcome", "manual_activity", "tool_exchange", "clean_intervals"],
        "review required_fields contract does not match the strict schema",
    )
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
    identity_payload = {key: manifest[key] for key in identity_keys}
    actual_catalog_id = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _require(
        manifest.get("catalog_id") == actual_catalog_id,
        "catalog_id does not match canonical manifest contents",
    )
    return manifest


def _validate_dataset_fingerprint(
    dataset_path: Path,
    procedure_entry: dict[str, Any],
) -> None:
    expected = procedure_entry.get("dataset_fingerprint", {})
    for relative in ("meta/info.json", "meta/episodes.jsonl"):
        path = dataset_path / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"CMR dataset metadata required by catalog is missing: {path}"
            )
        actual_sha = sha256_file(path)
        _require(
            actual_sha == expected.get(relative),
            f"{dataset_path.name}/{relative} SHA-256 mismatch; catalog was "
            "built against different dataset metadata",
        )


def _validate_ranges(
    episode_ids: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
) -> None:
    _require(
        episode_ids.ndim == starts.ndim == stops.ndim == 1,
        "range arrays must be one-dimensional",
    )
    _require(
        len(episode_ids) == len(starts) == len(stops),
        "range arrays have different lengths",
    )
    if not len(episode_ids):
        raise ValueError(
            "Invalid CMR clean catalog: strict catalog has zero eligible ranges"
        )
    _require(bool(np.all(episode_ids >= 0)), "negative episode id")
    _require(bool(np.all(starts >= 0)), "negative range start")
    _require(bool(np.all(stops > starts)), "empty or reversed range")
    if len(episode_ids) > 1:
        ordered = (episode_ids[1:] > episode_ids[:-1]) | (
            (episode_ids[1:] == episode_ids[:-1]) & (starts[1:] >= stops[:-1])
        )
        _require(
            bool(np.all(ordered)),
            "ranges must be sorted and non-overlapping within each episode",
        )


def load_strict_cmr_catalog_steps(
    *,
    catalog_root: str | Path,
    dataset_path: str | Path,
    action_delta_indices: list[int],
) -> list[tuple[int, int]]:
    """Load and verify strict CMR ``(episode_id, base_index)`` windows.

    Verification covers the catalog tier, review completeness, action horizon,
    dataset metadata fingerprints, compressed range-file checksum, and range
    ordering.  Candidate/unreviewed catalogs fail before any samples are
    returned.
    """

    root = Path(catalog_root)
    dataset = Path(dataset_path)
    manifest = _load_manifest(root)

    cached_deltas = [
        int(value)
        for value in manifest.get("horizon", {}).get("action_delta_indices", [])
    ]
    current_deltas = [int(value) for value in action_delta_indices]
    _require(
        cached_deltas == current_deltas,
        f"action horizon mismatch: catalog={cached_deltas}, current={current_deltas}",
    )

    procedure = dataset.name
    procedures = manifest.get("procedures", {})
    _require(
        procedure in procedures,
        f"procedure {procedure!r} is absent from manifest",
    )
    entry = procedures[procedure]
    _validate_dataset_fingerprint(dataset, entry)

    ranges_path = root / entry["ranges_file"]
    if not ranges_path.is_file():
        raise FileNotFoundError(
            f"CMR catalog range file not found for {procedure}: {ranges_path}"
        )
    _require(
        sha256_file(ranges_path) == entry.get("ranges_sha256"),
        f"{procedure} range-file SHA-256 mismatch",
    )

    with np.load(ranges_path, allow_pickle=False) as arrays:
        episode_ids = np.asarray(arrays["episode_id"], dtype=np.int64)
        starts = np.asarray(arrays["start"], dtype=np.int64)
        stops = np.asarray(arrays["stop"], dtype=np.int64)
    _validate_ranges(episode_ids, starts, stops)

    range_windows = int(np.sum(stops - starts, dtype=np.int64))
    _require(
        range_windows == int(entry.get("eligible_windows", -1)),
        f"{procedure} eligible-window count mismatch",
    )

    steps: list[tuple[int, int]] = []
    for episode_id, start, stop in zip(episode_ids, starts, stops):
        ep = int(episode_id)
        steps.extend((ep, base_index) for base_index in range(int(start), int(stop)))
    return steps
