#!/usr/bin/env python3
"""Fail once with a complete Open-H artifact inventory before torchrun."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from cosmos_framework.data.vfm.action.gr00t_dreams.data.embodiment_tags import (
    EmbodimentTag,
)
from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
    get_open_h_multi_train_specs,
)


def _is_cmr(spec: dict) -> bool:
    embodiment = spec["embodiment"]
    value = embodiment.value if isinstance(embodiment, EmbodimentTag) else str(embodiment)
    return value == EmbodimentTag.CMR_VERSIUS.value


def _cmr_cache_name(dataset_name: str, split: str, num_frames: int, stride: int) -> str:
    action_indices = [i * stride for i in range(num_frames - 1)]
    key = f"{dataset_name}_{split}_{sorted(action_indices)}"
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    return f"cmr_filter_cache_{split}_{digest}-44D.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--cmr-root", required=True)
    parser.add_argument("--openh-lz-root")
    parser.add_argument("--path-layout", default="draco_internal")
    parser.add_argument(
        "--stats-postfix",
        default=os.environ.get("COSMOS_OPENH_STATS_POSTFIX", "").strip(),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-frames", type=int, default=13)
    parser.add_argument("--timestep-interval", type=int, default=6)
    args = parser.parse_args()

    if not args.stats_postfix:
        raise SystemExit("FATAL: --stats-postfix/COSMOS_OPENH_STATS_POSTFIX is required")

    specs = get_open_h_multi_train_specs(
        base_path=args.root,
        cmr_base_path=args.cmr_root,
        path_layout=args.path_layout,
        openh_lz_base_path=args.openh_lz_root,
    )
    missing: list[str] = []
    print(f"[Open-H preflight] checking {len(specs)} dataset leaves")
    for index, spec in enumerate(specs):
        dataset_path = Path(spec["path"])
        cmr = _is_cmr(spec)
        required = [
            dataset_path,
            dataset_path / "data",
            dataset_path / "meta/info.json",
            dataset_path / "meta/episodes.jsonl",
            dataset_path / ("meta/modality-44D.json" if cmr else "meta/modality.json"),
            dataset_path
            / (
                f"meta/stats_cosmos-44D-{args.stats_postfix}.json"
                if cmr
                else f"meta/stats_cosmos-{args.stats_postfix}.json"
            ),
        ]
        if cmr:
            required.append(
                dataset_path
                / "meta"
                / _cmr_cache_name(
                    dataset_path.name,
                    args.split,
                    args.num_frames,
                    args.timestep_interval,
                )
            )
        absent = [path for path in required if not path.exists()]
        if absent:
            print(f"  [MISSING] [{index:02d}] {dataset_path}")
            missing.extend(str(path) for path in absent)
        else:
            print(f"  [ok] [{index:02d}] {dataset_path}")

    if missing:
        print("\n[Open-H preflight] missing required paths/artifacts:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(64)
    print("[Open-H preflight] all dataset paths and artifacts are present")


if __name__ == "__main__":
    main()
