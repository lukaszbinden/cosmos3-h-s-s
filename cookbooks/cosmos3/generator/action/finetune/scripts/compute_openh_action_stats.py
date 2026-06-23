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

Collision avoidance (IMPORTANT)
-------------------------------
Stats are written INTO each dataset's shared ``meta/`` dir on the canonical
Open-H tree, so independent experiments (e.g. this 44D run vs a colleague's 54D
run) collide on ``stats_cosmos.json`` / ``stats_cosmos-44D.json``. Pass a
``--postfix`` (or set ``COSMOS_OPENH_STATS_POSTFIX``) so files are written as:

    CMR Versius : meta/stats_cosmos-44D-<postfix>.json
    other Open-H: meta/stats_cosmos-<postfix>.json

The training loader reads the SAME postfix via ``COSMOS_OPENH_STATS_POSTFIX``
(see ``gr00t_dreams/data/dataset.py``), strict-matching it. Each file embeds a
``_provenance`` block (experiment id, action rep, exact dataset-set hash +
leaf list, horizon, stride, sampling, git rev, timestamp), an archival sidecar
``stats_cosmos[-44D].<experiment_id>.json`` is written alongside, and an
existing file whose provenance DIFFERS is NOT overwritten unless ``--force``.

Note: the CMR filename already carries ``-44D-``, so the postfix need NOT repeat
the dimension (use e.g. ``c3hss-v1``, giving CMR
``stats_cosmos-44D-c3hss-v1.json`` and others ``stats_cosmos-c3hss-v1.json``).

Usage::

    # All datasets in OPEN_H_DATASET_SPECS, with an experiment postfix:
    python scripts/compute_openh_action_stats.py --root "$OPENH_SURGICAL_ROOT" \\
        --postfix c3hss-v1 --experiment-id c3hss_openh_44d_v1

    # equivalently via env (matches how training reads it):
    COSMOS_OPENH_STATS_POSTFIX=c3hss-v1 \\
        python scripts/compute_openh_action_stats.py --root "$OPENH_SURGICAL_ROOT"

    # A single dataset / embodiment (fast smoke):
    python scripts/compute_openh_action_stats.py --postfix c3hss-v1 \\
        --dataset-path /path/to/Surgical/obuda/<task> --embodiment dvrk_obuda --max-windows 5000

NOTE: run audit_openh_action_schemas.py first; if a registry key doesn't match a
dataset's modality.json the run fails closed (KeyError) rather than mis-normalizing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# Default action-representation tag recorded in provenance / used in the dim
# component of the postfix convention. This cookbook is 44D.
ACTION_REP = "44D"


def _resolve_postfix(args) -> str:
    """Postfix precedence: --postfix > COSMOS_OPENH_STATS_POSTFIX env > ''."""
    if getattr(args, "postfix", None):
        return str(args.postfix).strip()
    return os.environ.get("COSMOS_OPENH_STATS_POSTFIX", "").strip()


def _cmr_stats_filename(postfix: str) -> str:
    return f"stats_cosmos-44D-{postfix}.json" if postfix else "stats_cosmos-44D.json"


def _openh_stats_filename(postfix: str) -> str:
    return f"stats_cosmos-{postfix}.json" if postfix else "stats_cosmos.json"


def _stats_filename(embodiment: str, postfix: str) -> str:
    """Loader-matching stats filename (MUST match dataset.py helpers)."""
    if embodiment == "cmr_versius":
        return _cmr_stats_filename(postfix)
    return _openh_stats_filename(postfix)


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _dataset_set_signature(args) -> tuple[list[str], str]:
    """Return (sorted leaf list, short hash) of the FULL included dataset set.

    Recorded in provenance so a stats file is traceable to the exact mixture it
    was computed for — a different OPEN_H_DATASET_SPECS membership yields a
    different hash, flagging stale stats even when the filename is reused.
    """
    from cosmos_framework.data.vfm.action.gr00t_dreams.data.embodiment_tags import EmbodimentTag
    from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
        get_open_h_multi_train_specs,
    )

    leaves = []
    for spec in get_open_h_multi_train_specs(base_path=None):
        emb = spec["embodiment"]
        emb = emb.value if isinstance(emb, EmbodimentTag) else emb
        leaves.append(f"{emb}:{Path(spec['path']).name}")
    leaves = sorted(leaves)
    h = hashlib.sha1("\n".join(leaves).encode()).hexdigest()[:12]
    return leaves, h


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


def _read_existing_provenance(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f).get("_provenance")
    except Exception:  # noqa: BLE001
        return None


def compute_for_dataset(
    dataset_path: Path,
    embodiment: str,
    num_frames: int,
    max_windows: int,
    downscaled_res: bool,
    force: bool,
    postfix: str,
    experiment_id: str,
    dataset_set_hash: str,
    dataset_set_leaves: list[str],
    write_sidecar: bool = True,
) -> Path | None:
    from cosmos_framework.data.vfm.action.gr00t_dreams.data.dataset import LeRobotSingleDataset
    from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import EMBODIMENT_REGISTRY

    out_path = dataset_path / "meta" / _stats_filename(embodiment, postfix)
    # No-clobber guard: refuse to overwrite an existing stats file whose embedded
    # provenance disagrees with this run (different experiment_id or dataset set),
    # unless --force. A matching-provenance file is treated as already-done.
    if out_path.exists():
        prov = _read_existing_provenance(out_path)
        same = (
            prov is not None
            and prov.get("experiment_id") == experiment_id
            and prov.get("dataset_set_hash") == dataset_set_hash
            and prov.get("action_rep") == ACTION_REP
        )
        if same and not force:
            print(f"[SKIP] {out_path} already computed for this experiment (provenance matches)")
            return out_path
        if not same and not force:
            print(
                f"[REFUSE] {out_path} exists but its provenance differs from this run "
                f"(existing={prov!r}). NOT overwriting. Use --force to override, or set a "
                f"distinct --postfix/--experiment-id."
            )
            return None
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

    timestep_interval = int(EMBODIMENT_REGISTRY.get(embodiment, {}).get("timestep_interval", 1))
    stats: dict = {"timestep_interval": timestep_interval}
    # Provenance block (top-level ``_provenance`` key; the loader skips
    # underscore-prefixed keys during stats validation). Full traceability:
    # which experiment, action rep, exact dataset mixture, horizon, stride,
    # sampling, code rev, and time.
    stats["_provenance"] = {
        "experiment_id": experiment_id,
        "postfix": postfix,
        "action_rep": ACTION_REP,
        "embodiment": embodiment,
        "num_frames": num_frames,
        "timestep_interval": timestep_interval,
        "max_windows": max_windows,
        "windows_used": used,
        "downscaled_res": downscaled_res,
        "dataset_set_hash": dataset_set_hash,
        "dataset_set_size": len(dataset_set_leaves),
        "dataset_set_leaves": dataset_set_leaves,
        "stats_filename": out_path.name,
        "generated_by": "compute_openh_action_stats.py",
        "git_rev": _git_rev(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
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

    # Archival sidecar: an immutable, postfix+experiment-tagged copy kept
    # alongside the live file so prior runs' stats are never lost even if the
    # live file is later regenerated. Always includes the experiment id.
    if write_sidecar:
        eid = experiment_id or postfix or "noid"
        if embodiment == "cmr_versius":
            sidecar_name = f"stats_cosmos-44D.{eid}.json"
        else:
            sidecar_name = f"stats_cosmos.{eid}.json"
        sidecar = out_path.parent / sidecar_name
        if sidecar.resolve() != out_path.resolve():
            with open(sidecar, "w") as f:
                json.dump(stats, f)
            print(f"[OK] archived sidecar {sidecar}")
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
    parser.add_argument("--force", action="store_true", help="overwrite even if a differing stats file exists")
    parser.add_argument(
        "--postfix",
        default=None,
        help=(
            "Experiment postfix for the stats filename (else COSMOS_OPENH_STATS_POSTFIX env). "
            "Writes stats_cosmos-44D-<postfix>.json (CMR) / stats_cosmos-<postfix>.json (others) "
            "so independent experiments don't collide in the shared meta/ dir. The training "
            "loader reads the SAME postfix via COSMOS_OPENH_STATS_POSTFIX."
        ),
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Human experiment id recorded in _provenance + sidecar name (default: the postfix).",
    )
    parser.add_argument("--no-sidecar", action="store_true", help="do not write the archival sidecar copy")
    args = parser.parse_args()

    postfix = _resolve_postfix(args)
    experiment_id = (args.experiment_id or postfix or "").strip()
    if not postfix:
        print(
            "[WARN] No --postfix / COSMOS_OPENH_STATS_POSTFIX set: writing the BARE "
            "stats_cosmos.json / stats_cosmos-44D.json names, which can COLLIDE with another "
            "experiment's stats in the shared meta/ dir. Strongly recommended to pass --postfix "
            "(e.g. 44D-c3hss-v1)."
        )
    dataset_set_leaves, dataset_set_hash = _dataset_set_signature(args)
    print(
        f"[info] postfix={postfix!r} experiment_id={experiment_id!r} "
        f"dataset_set_hash={dataset_set_hash} ({len(dataset_set_leaves)} leaves) action_rep={ACTION_REP}"
    )

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
                postfix=postfix,
                experiment_id=experiment_id,
                dataset_set_hash=dataset_set_hash,
                dataset_set_leaves=dataset_set_leaves,
                write_sidecar=not args.no_sidecar,
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
