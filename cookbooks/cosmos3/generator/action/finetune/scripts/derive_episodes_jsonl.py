#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reconstruct a missing LeRobot ``meta/episodes.jsonl`` for Open-H datasets.

The gr00t_dreams loader (``LeRobotSingleDataset._get_trajectories``) needs
``meta/episodes.jsonl`` — one JSON line per episode with at least::

    {"episode_index": <int>, "tasks": [<str>], "length": <int>}

Some open-h-embodiment leaves on EOS are missing this file (they ship only
``episodes_stats.jsonl``, or — for ``jhu/imerse/srth_porcine_chole`` — neither).
This script derives ``episodes.jsonl`` from what IS available, in priority:

  1. ``meta/episodes_stats.jsonl`` (fast, no parquet reads): each record is
     ``{"episode_index": i, "stats": {<key>: {"count": [N], ...}, ...,
     "task_index": {"min": [t], ...}}}``. We take ``length = count[0]`` from a
     per-frame key and ``tasks = [tasks.jsonl[task_index]]``.
  2. parquet row counts: enumerate ``data/chunk-*/episode_*.parquet`` (path
     pattern from ``info.json::data_path``), take ``episode_index`` from the
     filename and ``length`` = number of rows; ``tasks`` from the parquet
     ``task_index`` column (if present) mapped via ``tasks.jsonl``.

It validates the derived episode count / total frames against
``info.json::total_episodes`` / ``total_frames`` and, by default, refuses to
overwrite an existing ``episodes.jsonl`` (use ``--force``).

Dependencies: stdlib (+ pyarrow only for the parquet fallback).

Usage::

    # one dataset:
    python scripts/derive_episodes_jsonl.py \
        --dataset-path "$OPENH_SURGICAL_ROOT/cmr_surgical/hysterectomy"

    # the known-affected EOS leaves at once:
    python scripts/derive_episodes_jsonl.py --root "$OPENH_SURGICAL_ROOT" --affected

    # dry-run (compute + validate, write nothing):
    python scripts/derive_episodes_jsonl.py --dataset-path <DIR> --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Leaves observed missing episodes.jsonl on the EOS open-h-embodiment tree.
AFFECTED_RELS = [
    "cmr_surgical/hysterectomy",
    "cmr_surgical/inguinal_hernia",
    "cmr_surgical/prostatectomy",
    "jhu/imerse/srth_porcine_chole",
]

EPISODES_FILENAME = "meta/episodes.jsonl"
EPISODES_STATS_FILENAME = "meta/episodes_stats.jsonl"
INFO_FILENAME = "meta/info.json"
TASKS_FILENAME = "meta/tasks.jsonl"

# Per-frame stat keys to read ``count`` (=episode length) from, in order of
# preference. All of these are per-frame in LeRobot episodes_stats.
_LENGTH_KEYS = ["index", "frame_index", "timestamp", "action", "observation.state"]


def _read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def _load_tasks(dataset_path: Path) -> dict[int, str]:
    """Map task_index -> task string from meta/tasks.jsonl (best-effort)."""
    tasks_path = dataset_path / TASKS_FILENAME
    mapping: dict[int, str] = {}
    if not tasks_path.exists():
        return mapping
    with tasks_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # LeRobot tasks.jsonl: {"task_index": i, "task": "..."}
            if "task_index" in rec:
                mapping[int(rec["task_index"])] = rec.get("task", "")
    return mapping


def _length_from_stats(stats: dict) -> int | None:
    for k in _LENGTH_KEYS:
        node = stats.get(k)
        if isinstance(node, dict) and isinstance(node.get("count"), list) and node["count"]:
            return int(node["count"][0])
    # Fallback: any key with a count.
    for node in stats.values():
        if isinstance(node, dict) and isinstance(node.get("count"), list) and node["count"]:
            return int(node["count"][0])
    return None


def _task_index_from_stats(stats: dict) -> int | None:
    node = stats.get("task_index")
    if isinstance(node, dict) and isinstance(node.get("min"), list) and node["min"]:
        return int(node["min"][0])
    return None


def _derive_from_stats(dataset_path: Path) -> list[dict] | None:
    stats_path = dataset_path / EPISODES_STATS_FILENAME
    if not stats_path.exists():
        return None
    task_map = _load_tasks(dataset_path)
    episodes: list[dict] = []
    with stats_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ei = int(rec["episode_index"])
            stats = rec.get("stats", {})
            length = _length_from_stats(stats)
            if length is None:
                print(f"  [warn] episode {ei}: no count found in stats; skipping derive-from-stats")
                return None
            ti = _task_index_from_stats(stats)
            tasks = [task_map[ti]] if (ti is not None and ti in task_map) else []
            episodes.append({"episode_index": ei, "tasks": tasks, "length": length})
    episodes.sort(key=lambda e: e["episode_index"])
    return episodes


def _derive_from_parquet(dataset_path: Path) -> list[dict] | None:
    try:
        import pyarrow.parquet as pq
    except Exception as e:  # noqa: BLE001
        print(f"  [error] pyarrow needed for parquet fallback: {e!r}")
        return None
    info = _read_json(dataset_path / INFO_FILENAME)
    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    files = sorted((dataset_path / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        print(f"  [error] no parquet files under {dataset_path / 'data'}")
        return None
    task_map = _load_tasks(dataset_path)
    episodes: list[dict] = []
    for pf in files:
        # episode_index from filename (episode_######.parquet)
        stem = pf.stem  # episode_000123
        try:
            ei = int(stem.split("_")[-1])
        except ValueError:
            print(f"  [warn] cannot parse episode index from {pf.name}; skipping")
            continue
        try:
            t = pq.read_table(pf, columns=None)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] failed reading {pf.name}: {e!r}")
            return None
        length = t.num_rows
        tasks = []
        if "task_index" in t.column_names:
            try:
                ti = int(t.column("task_index")[0].as_py())
                if ti in task_map:
                    tasks = [task_map[ti]]
            except Exception:  # noqa: BLE001
                pass
        episodes.append({"episode_index": ei, "tasks": tasks, "length": length})
    episodes.sort(key=lambda e: e["episode_index"])
    return episodes


def _validate(dataset_path: Path, episodes: list[dict]) -> list[str]:
    warnings: list[str] = []
    try:
        info = _read_json(dataset_path / INFO_FILENAME)
    except Exception:  # noqa: BLE001
        return ["could not read info.json for validation"]
    n_exp = info.get("total_episodes")
    f_exp = info.get("total_frames")
    n_got = len(episodes)
    f_got = sum(e["length"] for e in episodes)
    if n_exp is not None and int(n_exp) != n_got:
        warnings.append(f"episode count {n_got} != info.json total_episodes {n_exp}")
    if f_exp is not None and int(f_exp) != f_got:
        warnings.append(f"total frames {f_got} != info.json total_frames {f_exp}")
    # episode_index contiguity (warn only)
    idxs = [e["episode_index"] for e in episodes]
    if idxs != list(range(len(idxs))):
        warnings.append("episode_index values are not a contiguous 0..N-1 range")
    return warnings


def derive_one(dataset_path: Path, force: bool, dry_run: bool) -> bool:
    out_path = dataset_path / EPISODES_FILENAME
    print(f"=== {dataset_path} ===")
    if not dataset_path.exists():
        print("  [error] dataset path does not exist")
        return False
    if out_path.exists() and not force:
        print(f"  [skip] {out_path.name} already exists (use --force to overwrite)")
        return True

    episodes = _derive_from_stats(dataset_path)
    source = "episodes_stats.jsonl"
    if episodes is None:
        episodes = _derive_from_parquet(dataset_path)
        source = "parquet row counts"
    if not episodes:
        print("  [error] could not derive episodes (no episodes_stats.jsonl and no usable parquet)")
        return False

    warnings = _validate(dataset_path, episodes)
    print(f"  derived {len(episodes)} episodes from {source}; total_frames={sum(e['length'] for e in episodes):,}")
    for w in warnings:
        print(f"  [warn] {w}")

    if dry_run:
        print(f"  [dry-run] would write {out_path} (sample: {episodes[0]})")
        return True

    # Safety: a non-contiguous episode_index almost always means the dataset is
    # only partially staged on disk (missing parquet chunks). The gr00t_dreams
    # loader assumes episode_index is a contiguous 0..N-1 range, so writing such
    # a file would silently corrupt sampling. Refuse unless explicitly forced.
    idxs = [e["episode_index"] for e in episodes]
    if idxs != list(range(len(idxs))) and not force:
        print(
            "  [error] refusing to write: episode_index is not a contiguous "
            "0..N-1 range (dataset likely partially staged). Re-stage the full "
            "dataset, or pass --force to write anyway (NOT recommended)."
        )
        return False

    with out_path.open("w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    print(f"  [ok] wrote {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", default=None, help="single dataset dir (contains meta/)")
    parser.add_argument("--root", default=None, help="OPENH_SURGICAL_ROOT (with --affected)")
    parser.add_argument(
        "--affected",
        action="store_true",
        help="process the known-affected EOS leaves under --root (CMR hysterectomy/inguinal/prostatectomy + JHU srth_porcine_chole)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing episodes.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="compute + validate, write nothing")
    args = parser.parse_args()

    targets: list[Path] = []
    if args.dataset_path:
        targets.append(Path(args.dataset_path))
    if args.affected:
        if not args.root:
            raise SystemExit("--affected requires --root")
        targets.extend(Path(args.root) / rel for rel in AFFECTED_RELS)
    if not targets:
        raise SystemExit("Provide --dataset-path <DIR> and/or --affected --root <ROOT>")

    ok = True
    for dp in targets:
        ok = derive_one(dp, force=args.force, dry_run=args.dry_run) and ok
    if not ok:
        sys.exit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
