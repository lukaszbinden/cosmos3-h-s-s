#!/usr/bin/env python3
"""Dump C3-H-S-S episode spans in Justin's cross-checkpoint JSON schema.

This is deliberately an index-only CPU tool. It instantiates Lukas's native
Open-H dataset adapter, so split membership and embodiment-specific strides
come from the same code used by the Lukas evaluator. No video is decoded.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
    OPEN_H_DATASET_SPECS,
)
from cosmos_framework.data.vfm.action.open_h_dataset import OpenHMixedLeRobotDataset


def embodiment_for(path: str):
    wanted = path.rstrip("/")
    for spec in OPEN_H_DATASET_SPECS:
        if str(spec["path"]).rstrip("/") == wanted:
            return spec["embodiment"]
    raise ValueError(f"{wanted} is absent from OPEN_H_DATASET_SPECS")


def canonical_root(path: str, openh_root: str) -> str:
    path = path.rstrip("/")
    openh_root = openh_root.rstrip("/")
    cmr_root = f"{openh_root}/cmr_surgical"
    if path == cmr_root or path.startswith(cmr_root + "/"):
        return "CMR" + path[len(cmr_root) :]
    if path == openh_root or path.startswith(openh_root + "/"):
        return "OPENH" + path[len(openh_root) :]
    raise ValueError(f"cannot canonicalize {path!r} under {openh_root!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="TSV: family, shard path")
    p.add_argument("--label", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=("train", "test", "full"), required=True)
    p.add_argument("--test-split-ratio", type=float, default=0.02)
    p.add_argument("--families", nargs="+", default=("cmr_clinical", "dvrk_jhu"))
    p.add_argument(
        "--openh-root",
        default="/lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment/Surgical",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    families = set(args.families)
    entries: dict[str, list[dict]] = defaultdict(list)
    failures: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()

    for raw in Path(args.manifest).read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        family, shard = raw.split("\t")[:2]
        family, shard = family.strip(), shard.strip().rstrip("/")
        if family not in families or (family, shard) in seen:
            continue
        seen.add((family, shard))
        try:
            mixed = OpenHMixedLeRobotDataset(
                dataset_specs=[
                    {"path": shard, "embodiment": embodiment_for(shard), "mix_ratio": 1.0}
                ],
                num_frames=13,
                data_split=args.split,
                test_split_ratio=args.test_split_ratio,
                max_action_dim=44,
                mode="forward_dynamics",
                viewpoint="third_person_view",
            )
            steps = mixed.sub_datasets[0]._all_steps
            by_episode: dict[int, list[int]] = defaultdict(list)
            for episode_id, base_index in steps:
                by_episode[int(episode_id)].append(int(base_index))
            root = canonical_root(shard, args.openh_root)
            for episode_id in sorted(by_episode):
                bases = by_episode[episode_id]
                entries[family].append(
                    {
                        "root": root,
                        "episode_id": episode_id,
                        "flat_start": min(bases),
                        "valid_len": len(bases),
                        "base_index_min": min(bases),
                        "base_index_max": max(bases),
                    }
                )
            print(
                f"{args.label}: {family} {Path(shard).name}: "
                f"{len(by_episode)} episodes, {len(steps)} windows"
            )
        except Exception as exc:  # keep the complete audit even if one leaf is broken
            failures[f"{family}\t{shard}"] = f"{type(exc).__name__}: {exc}"

    payload = {
        "label": args.label,
        "config_file": "Lukas native OpenHMixedLeRobotDataset",
        "split": args.split,
        "test_split_ratio": args.test_split_ratio,
        "root_aliases": json.dumps({args.openh_root: "OPENH", f"{args.openh_root}/cmr_surgical": "CMR"}),
        "entries": dict(entries),
        "failures": failures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(
        f"wrote {output}: {sum(len(v) for v in entries.values())} episode spans, "
        f"{len(failures)} failures"
    )
    if failures:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
