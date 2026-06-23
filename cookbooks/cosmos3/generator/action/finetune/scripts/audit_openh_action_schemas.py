#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail-closed audit of the Open-H registry schemas against on-disk metadata.

For every dataset in ``OPEN_H_DATASET_SPECS`` (or a single ``--dataset-path``),
load the LeRobot ``meta/modality.json`` (and ``meta/info.json``) and verify
that every ``action_keys`` / ``state_keys`` / ``video_keys`` the registry
declares for that embodiment actually exists in the dataset's modality file.

This is the gate the cosmos3-h-s-s mixture needs because it ADDS embodiments
whose exact modality keys could not be confirmed offline — chiefly the new
``jhu_imerse`` (STAR-IL) and ``virtual_incision_mira`` entries, plus the
re-rooted Obuda / LSCR / TUD leaves. If the audit reports a missing key, FIX
``groot_configs.py`` (the ``EMBODIMENT_REGISTRY`` entry) before training:
training would otherwise fail at first batch.

Mirrors the philosophy of the sean-cosmos3_surgical_fd
``audit_surgical_action_schemas.py`` (validate the schema against the actual
metadata before any manifest/stats/training).

Must run inside the patched Cosmos Framework venv.

Usage::

    python scripts/audit_openh_action_schemas.py --root "$OPENH_SURGICAL_ROOT"
    python scripts/audit_openh_action_schemas.py \\
        --dataset-path /path/to/Surgical/jhu/imerse/star_il/star_il \\
        --embodiment jhu_imerse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_modality_keys(dataset_path: Path, modality_filename: str) -> set[str]:
    """Return the set of fully-qualified modality keys (e.g. ``action.psm1_pose``)."""
    mpath = dataset_path / modality_filename
    if not mpath.exists():
        # Fall back to info.json feature names if no modality.json.
        info = dataset_path / "meta" / "info.json"
        if not info.exists():
            raise FileNotFoundError(f"neither {mpath} nor {info} exists")
        with open(info) as f:
            features = json.load(f).get("features", {})
        return set(features.keys())

    with open(mpath) as f:
        modality = json.load(f)
    keys: set[str] = set()
    # modality.json is typically {"state": {<subkey>: {...}}, "action": {...}, "video": {...}}
    for top, sub in modality.items():
        if isinstance(sub, dict):
            for subkey in sub:
                keys.add(f"{top}.{subkey}")
    return keys


def _iter_specs(args):
    from cosmos_framework.data.vfm.action.gr00t_dreams.data.embodiment_tags import EmbodimentTag
    from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
        get_open_h_multi_train_specs,
    )

    if args.dataset_path:
        if not args.embodiment:
            raise SystemExit("--embodiment is required when --dataset-path is given")
        yield Path(args.dataset_path), args.embodiment
        return
    for spec in get_open_h_multi_train_specs(base_path=args.root):
        emb = spec["embodiment"]
        emb = emb.value if isinstance(emb, EmbodimentTag) else emb
        yield Path(spec["path"]), emb


def audit_one(dataset_path: Path, embodiment: str) -> list[str]:
    from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import EMBODIMENT_REGISTRY

    errors: list[str] = []
    if embodiment == "cmr_versius":
        # CMR has a bespoke (non-registry) code path; only check the dir exists.
        if not dataset_path.exists():
            errors.append(f"{embodiment}: dataset path missing: {dataset_path}")
        return errors

    reg = EMBODIMENT_REGISTRY.get(embodiment)
    if reg is None:
        errors.append(f"{embodiment}: no EMBODIMENT_REGISTRY entry")
        return errors
    if not dataset_path.exists():
        errors.append(f"{embodiment}: dataset path missing: {dataset_path}")
        return errors

    modality_filename = reg.get("modality_filename", "meta/modality.json")
    try:
        present = _load_modality_keys(dataset_path, modality_filename)
    except Exception as e:  # noqa: BLE001
        errors.append(f"{embodiment}: cannot read modality metadata: {e!r}")
        return errors

    required = list(reg.get("video_keys", [])) + list(reg.get("state_keys", [])) + list(reg.get("action_keys", []))
    for key in required:
        if key not in present:
            errors.append(
                f"{embodiment}: registry key {key!r} NOT in {modality_filename} "
                f"(present sample: {sorted(present)[:8]}...)"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default=None, help="OPENH_SURGICAL_ROOT / DATASET_PATH to re-root specs")
    parser.add_argument("--dataset-path", default=None, help="single dataset path (requires --embodiment)")
    parser.add_argument("--embodiment", default=None, help="embodiment tag for --dataset-path")
    parser.add_argument("--output", default=None, help="optional JSON report path")
    args = parser.parse_args()

    report: dict[str, list[str]] = {}
    total_errors = 0
    for dataset_path, embodiment in _iter_specs(args):
        errs = audit_one(dataset_path, embodiment)
        key = f"{embodiment}:{dataset_path}"
        report[key] = errs
        status = "OK" if not errs else f"{len(errs)} ERROR(S)"
        print(f"[{status}] {key}")
        for e in errs:
            print(f"    - {e}")
        total_errors += len(errs)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {args.output}")

    print(f"\nAudit complete: {total_errors} error(s) across {len(report)} dataset(s).")
    if total_errors:
        print("FIX groot_configs.py EMBODIMENT_REGISTRY / OPEN_H_DATASET_SPECS before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
