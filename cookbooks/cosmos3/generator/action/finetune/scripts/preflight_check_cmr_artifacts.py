#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pre-flight: verify the per-dataset artifacts the stats run / training need.

Run this BEFORE ``compute_openh_action_stats.py`` (Step 3) to confirm every
dataset leaf in ``OPEN_H_DATASET_SPECS`` has the files the loader will demand,
so the long stats job doesn't die halfway through on a missing file.

For EVERY leaf it checks:
  * ``meta/modality.json``     (the gr00t loader requires it)
  * ``meta/info.json``
  * ``meta/episodes.jsonl``    (``_get_trajectories`` requires it)

For CMR Versius leaves it ALSO checks the clutch filter cache that
``_get_all_steps_cmr_filtered`` will look for. The cache filename embeds an
md5 hash of ``f"{dataset_name}_{split}_{sorted(action_delta_indices)}"`` — this
script recomputes that hash EXACTLY as both
``compute_cmr_filtered_episodes_cache.py`` and ``dataset.py`` do, for the
requested ``--num-frames`` (default 13 → action deltas ``[0,6,…,66]``).

Which split's cache matters:
  * the stats run builds the BASE ``LeRobotSingleDataset`` → ``data_split="full"``
    → it needs ``cmr_filter_cache_full_<hash>-44D.json``;
  * training via ``WrappedLeRobotSingleDataset`` uses ``train`` / ``test``.
By default this checks ``full`` (the Step-3 prerequisite). Pass
``--splits full,train,test`` to also assert the training caches exist.

This is READ-ONLY (stat() calls only); it imports ``OPEN_H_DATASET_SPECS`` so it
sees exactly the mixture training will use.

Usage::

    python scripts/preflight_check_cmr_artifacts.py --root "$OPENH_SURGICAL_ROOT"
    python scripts/preflight_check_cmr_artifacts.py --root "$OPENH_SURGICAL_ROOT" \
        --splits full,train,test
    # also verify the stats files exist (i.e. AFTER Step 3):
    python scripts/preflight_check_cmr_artifacts.py --root "$OPENH_SURGICAL_ROOT" \
        --postfix c3hss-v1 --require-stats
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

# Cap BLAS/OpenMP threads — this only stat()s files but importing numpy via the
# framework can still spin pools on a busy login node.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

# Stats-compute mode so importing the registry / specs never tries to load stats.
os.environ.setdefault("COSMOS_OPENH_STATS_COMPUTE_MODE", "1")

CMR_TAG = "cmr_versius"
TIMESTEP_INTERVAL_CMR = 6


def _action_delta_indices(num_frames: int, timestep_interval: int = TIMESTEP_INTERVAL_CMR) -> list[int]:
    """Mirror compute_cmr_filtered_episodes_cache.compute_action_delta_indices."""
    num_action_frames = num_frames - 1
    return [i * timestep_interval for i in range(num_action_frames)]


def _cmr_cache_filename(dataset_name: str, split: str, action_delta_indices: list[int]) -> str:
    """Mirror BOTH the cache script's get_cache_path and the loader's hash."""
    cache_key = f"{dataset_name}_{split}_{sorted(action_delta_indices)}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    return f"cmr_filter_cache_{split}_{cache_hash}-44D.json"


def _cmr_stats_filename(postfix: str) -> str:
    return f"stats_cosmos-44D-{postfix}.json" if postfix else "stats_cosmos-44D.json"


def _openh_stats_filename(postfix: str) -> str:
    return f"stats_cosmos-{postfix}.json" if postfix else "stats_cosmos.json"


def _load_specs(root: str | None):
    from cosmos_framework.data.vfm.action.gr00t_dreams.data.embodiment_tags import EmbodimentTag
    from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
        get_open_h_multi_train_specs,
    )

    specs = []
    for spec in get_open_h_multi_train_specs(base_path=root):
        emb = spec["embodiment"]
        emb = emb.value if isinstance(emb, EmbodimentTag) else emb
        specs.append((Path(spec["path"]), emb))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="OPENH_SURGICAL_ROOT to re-root specs (matches the stats run).")
    ap.add_argument("--num-frames", type=int, default=13, help="VIDEO frames (default 13 → 12 action deltas).")
    ap.add_argument(
        "--splits",
        default="full",
        help="Comma list of CMR cache splits to require (default 'full' = the Step-3 prerequisite).",
    )
    ap.add_argument("--postfix", default=None, help="Stats postfix (only used with --require-stats).")
    ap.add_argument(
        "--require-stats",
        action="store_true",
        help="Also require the stats_cosmos[-44D][-postfix].json files (use AFTER Step 3).",
    )
    args = ap.parse_args()

    postfix = (args.postfix or os.environ.get("COSMOS_OPENH_STATS_POSTFIX", "")).strip()
    if args.require_stats and not postfix:
        print("[FATAL] --require-stats needs --postfix or COSMOS_OPENH_STATS_POSTFIX", file=sys.stderr)
        sys.exit(2)

    cmr_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    adi = _action_delta_indices(args.num_frames)

    try:
        specs = _load_specs(args.root)
    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] could not import OPEN_H_DATASET_SPECS (is the overlay applied + venv active?): {e!r}")
        sys.exit(2)

    print(
        f"[info] {len(specs)} leaves | num_frames={args.num_frames} "
        f"action_deltas={adi[:3]}...{adi[-1]} (len {len(adi)}) | cmr_splits={cmr_splits} "
        f"| postfix={postfix!r} require_stats={args.require_stats}"
    )

    missing: list[str] = []
    n_cmr = 0
    n_ok = 0
    for path, emb in specs:
        meta = path / "meta"
        leaf_missing: list[str] = []

        if not path.exists():
            missing.append(f"{path}  ::  DATASET DIR MISSING")
            continue

        # Common required files for every leaf.
        for rel in ("modality.json", "info.json", "episodes.jsonl"):
            if not (meta / rel).exists():
                leaf_missing.append(f"meta/{rel}")

        # CMR-specific clutch caches.
        if emb == CMR_TAG:
            n_cmr += 1
            for split in cmr_splits:
                fn = _cmr_cache_filename(path.name, split, adi)
                if not (meta / fn).exists():
                    leaf_missing.append(f"meta/{fn}  (CMR {split} clutch cache)")

        # Stats files (optional, post-Step-3).
        if args.require_stats:
            stats_fn = _cmr_stats_filename(postfix) if emb == CMR_TAG else _openh_stats_filename(postfix)
            if not (meta / stats_fn).exists():
                leaf_missing.append(f"meta/{stats_fn}  (stats)")

        if leaf_missing:
            for m in leaf_missing:
                missing.append(f"{emb:22s} {path.name:42s} {m}")
        else:
            n_ok += 1
            print(f"[ok]   {emb:22s} {path.name}")

    print("\n" + "=" * 80)
    if missing:
        print(f"MISSING ARTIFACTS ({len(missing)}):")
        for m in missing:
            print(f"  [MISS] {m}")
        print("=" * 80)
        print(f"{n_ok}/{len(specs)} leaves complete; {n_cmr} CMR leaves checked. NOT ready for Step 3.")
        sys.exit(1)
    print(f"ALL GOOD: {n_ok}/{len(specs)} leaves complete ({n_cmr} CMR). Ready for Step 3.")


if __name__ == "__main__":
    main()
