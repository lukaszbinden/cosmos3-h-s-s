#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compute per-embodiment post-transform action/state normalization stats.

Writes the ``meta/stats_cosmos.json`` (generic Open-H embodiments) and
``meta/stats_cosmos-44D.json`` (CMR Versius) files that the training pipeline
(``cosmos_framework.data.vfm.action.gr00t_dreams.data.dataset
.WrappedLeRobotSingleDataset``) REQUIRES — without them training fails at
startup with "Open-H embodiments require stats_cosmos.json".

Why a Cosmos-specific stats file (not the raw LeRobot ``stats.json``): the
per-embodiment transforms convert raw absolute poses (e.g. 7D quaternion) into
the model-facing relative representation (9D xyz + rot6d), so the normalization
statistics MUST be computed on the *transformed* values, keyed per action/state
key. Mirrors the philosophy of the sean-cosmos3_surgical_fd
``compute_action_stats.py`` (use the same action-building code as training).

This must run inside the patched Cosmos Framework venv (it imports
``cosmos_framework.data.vfm.action.gr00t_dreams``). It computes stats on the
transformed-but-PRE-normalization tensors (ToTensor -> delta/relative
conversion -> collect per key), then writes per-key ``{mean, std, min, max}``
plus a top-level ``timestep_interval`` stamp (the dataset loader verifies this
stamp against ``EMBODIMENT_REGISTRY[tag]["timestep_interval"]``).

Usage::

    # All datasets in OPEN_H_DATASET_SPECS (uses registry strides):
    python scripts/compute_openh_action_stats.py --root "$OPENH_SURGICAL_ROOT"

    # A single dataset / embodiment:
    python scripts/compute_openh_action_stats.py \\
        --dataset-path /path/to/Surgical/obuda/<task> --embodiment dvrk_obuda

    # Limit sampling for a fast pass:
    python scripts/compute_openh_action_stats.py --max-windows 50000

NOTE: For the newly added (B) embodiments (jhu_imerse, virtual_incision_mira)
the registry keys are assumptions — run audit_openh_action_schemas.py first and
fix groot_configs.py if the modality keys differ, otherwise stats here will
KeyError on the missing keys (which is the intended fail-closed behavior).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _iter_specs(args):
    """Yield (dataset_path, embodiment_tag) pairs to process."""
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


def _stats_filename(embodiment: str) -> str:
    return "stats_cosmos-44D.json" if embodiment == "cmr_versius" else "stats_cosmos.json"


def _collect_pre_norm_transform(num_frames: int, embodiment: str, downscaled_res: bool):
    """Build a transform that stops BEFORE normalization, so we can measure it.

    Reuses ``construct_modality_config_and_transforms`` to get the per-embodiment
    modality config, then rebuilds the transform list up to (and including) the
    delta/relative action conversion but EXCLUDING the ``StateActionTransform``
    normalization and the final concat — leaving per-key transformed arrays we
    can accumulate.
    """
    from cosmos_framework.data.vfm.action.gr00t_dreams.data.transform.base import (
        ComposedModalityTransform,
    )
    from cosmos_framework.data.vfm.action.gr00t_dreams.data.transform.state_action import (
        StateActionTransform,
    )
    from cosmos_framework.data.vfm.action.gr00t_dreams.data.transform.concat import ConcatTransform
    from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
        construct_modality_config_and_transforms,
    )

    config, train_tf, _test_tf = construct_modality_config_and_transforms(
        num_frames=num_frames, embodiment=embodiment, downscaled_res=downscaled_res
    )
    modality_filename = None
    if isinstance(config, dict) and "modality_filename" in config:
        modality_filename = config.pop("modality_filename")

    # Strip the normalization + concat tail so per-key transformed arrays remain.
    kept = [
        t
        for t in train_tf.transforms
        if not isinstance(t, (StateActionTransform, ConcatTransform))
    ]
    return config, modality_filename, ComposedModalityTransform(transforms=kept)


def compute_for_dataset(
    dataset_path: Path,
    embodiment: str,
    num_frames: int,
    max_windows: int,
    downscaled_res: bool,
    force: bool,
) -> Path | None:
    from cosmos_framework.data.vfm.action.gr00t_dreams.data.dataset import LeRobotSingleDataset
    from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import EMBODIMENT_REGISTRY

    out_path = dataset_path / "meta" / _stats_filename(embodiment)
    if out_path.exists() and not force:
        print(f"[SKIP] {out_path} exists (use --force to recompute)")
        return out_path
    if not dataset_path.exists():
        print(f"[ERROR] dataset path missing: {dataset_path}")
        return None

    config, modality_filename, transform = _collect_pre_norm_transform(
        num_frames, embodiment, downscaled_res
    )
    ds = LeRobotSingleDataset(
        dataset_path=str(dataset_path),
        modality_configs=config,
        transforms=transform,
        embodiment_tag=embodiment,
        modality_filename=modality_filename,
    )

    n = len(ds)
    if n == 0:
        print(f"[WARN] {dataset_path} has 0 samples; skipping")
        return None
    stride = max(1, n // max_windows) if max_windows and n > max_windows else 1
    idxs = range(0, n, stride)

    # Accumulate per-key concatenated-over-time arrays.
    buckets: dict[str, list[np.ndarray]] = {}
    used = 0
    for i in idxs:
        try:
            sample = LeRobotSingleDataset.__getitem__(ds, i)
        except Exception as e:  # noqa: BLE001
            if used == 0:
                print(f"[warn] sample {i} failed: {e!r}")
            continue
        for key, val in sample.items():
            if not (key.startswith("action.") or key.startswith("state.")):
                continue
            arr = np.asarray(val, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[None, :]
            arr = arr.reshape(-1, arr.shape[-1])
            buckets.setdefault(key, []).append(arr)
        used += 1

    if not buckets:
        print(f"[ERROR] no action/state keys collected for {dataset_path}")
        return None

    stats: dict = {"timestep_interval": int(EMBODIMENT_REGISTRY.get(embodiment, {}).get("timestep_interval", 1))}
    for key, chunks in buckets.items():
        data = np.concatenate(chunks, axis=0)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)  # avoid divide-by-zero on constant channels
        stats[key] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": data.min(axis=0).tolist(),
            "max": data.max(axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f)
    print(f"[OK] wrote {out_path}  ({used} windows, {len(buckets)} keys)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default=None, help="OPENH_SURGICAL_ROOT / DATASET_PATH to re-root specs")
    parser.add_argument("--dataset-path", default=None, help="single dataset path (requires --embodiment)")
    parser.add_argument("--embodiment", default=None, help="embodiment tag for --dataset-path")
    parser.add_argument("--num-frames", type=int, default=13, help="video frames (1 context + N pred); default 13")
    parser.add_argument("--max-windows", type=int, default=200000, help="max sampled windows per dataset")
    parser.add_argument("--downscaled-res", action="store_true", help="use 256x256 transform path")
    parser.add_argument("--force", action="store_true", help="recompute even if stats exist")
    args = parser.parse_args()

    errors = 0
    for dataset_path, embodiment in _iter_specs(args):
        try:
            res = compute_for_dataset(
                dataset_path=dataset_path,
                embodiment=embodiment,
                num_frames=args.num_frames,
                max_windows=args.max_windows,
                downscaled_res=args.downscaled_res,
                force=args.force,
            )
            if res is None:
                errors += 1
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {dataset_path} ({embodiment}): {e!r}")
            errors += 1

    if errors:
        print(f"\nCompleted with {errors} dataset error(s).")
        sys.exit(1)
    print("\nAll stats computed.")


if __name__ == "__main__":
    main()
