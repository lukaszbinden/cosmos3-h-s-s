#!/usr/bin/env python3
"""Verify that only the diffusion seed changes in a multi-seed probe."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4]
    )
    return parser.parse_args()


def _correct_action_sha(manifest: dict[str, Any]) -> str:
    correct = [record for record in manifest["variants"] if record["name"] == "correct"]
    if len(correct) != 1:
        raise ValueError("Manifest must contain exactly one correct variant")
    return str(correct[0]["action"]["sha256"])


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    manifests = sorted(
        (input_root / "raw").glob("*/*/*_action_intervention_episode.json")
    )
    expected_count = 12 * len(args.expected_seeds)
    if len(manifests) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} manifests, found {len(manifests)}"
        )

    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for path in manifests:
        relative = path.relative_to(input_root / "raw")
        if len(relative.parts) != 3:
            raise ValueError(f"Unexpected manifest layout: {relative}")
        subset, target_arm = relative.parts[:2]
        payload = json.loads(path.read_text())
        groups[
            (
                subset,
                target_arm,
                int(payload["episode_id"]),
                int(payload["base_index"]),
            )
        ].append(
            {
                "path": str(path),
                "sampling_seed": int(payload["sampling_seed"]),
                "data_seed": int(payload["data_seed"]),
                "ground_truth_frames_sha256": payload["ground_truth_frames_sha256"],
                "history_action_sha256": payload["history_action_sha256"],
                "correct_action_sha256": _correct_action_sha(payload),
                "first_row_max_abs_error": float(
                    payload["physical_action_audit"]["variant_first_row_max_abs_error"]
                ),
            }
        )

    group_records = []
    all_passed = True
    expected_seeds = sorted(args.expected_seeds)
    for identity, records in sorted(groups.items()):
        records.sort(key=lambda record: record["sampling_seed"])
        checks = {
            "sampling_seeds_exact": [record["sampling_seed"] for record in records]
            == expected_seeds,
            "single_data_seed": len({record["data_seed"] for record in records}) == 1,
            "ground_truth_identical": len(
                {record["ground_truth_frames_sha256"] for record in records}
            )
            == 1,
            "history_action_identical": len(
                {record["history_action_sha256"] for record in records}
            )
            == 1,
            "correct_action_identical": len(
                {record["correct_action_sha256"] for record in records}
            )
            == 1,
            "first_row_preserved": max(
                record["first_row_max_abs_error"] for record in records
            )
            <= 1e-6,
        }
        passed = all(checks.values())
        all_passed &= passed
        subset, target_arm, episode_id, base_index = identity
        group_records.append(
            {
                "subset": subset,
                "target_arm": target_arm,
                "episode_id": episode_id,
                "base_index": base_index,
                "passed": passed,
                "checks": checks,
                "records": records,
            }
        )

    result = {
        "diagnostic": "multi-seed input identity",
        "passed": all_passed,
        "expected_seeds": expected_seeds,
        "manifest_count": len(manifests),
        "group_count": len(group_records),
        "groups": group_records,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"MULTISEED_INPUT_IDENTITY passed={all_passed} "
        f"groups={len(group_records)} manifests={len(manifests)}"
    )
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
