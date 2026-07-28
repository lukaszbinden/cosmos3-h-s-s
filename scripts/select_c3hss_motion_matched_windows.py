#!/usr/bin/env python3
"""Select balanced, arm-isolated dVRK windows from motion-scan CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ARMS = ("psm1", "psm2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--subsets", nargs="+", default=["hf_suturebot", "nephfat"]
    )
    parser.add_argument("--windows-per-arm", type=int, default=3)
    parser.add_argument("--target-translation-mm", type=float, default=0.65)
    parser.add_argument("--minimum-translation-mm", type=float, default=0.55)
    parser.add_argument("--maximum-translation-mm", type=float, default=0.75)
    parser.add_argument("--maximum-other-translation-mm", type=float, default=0.15)
    parser.add_argument("--minimum-isolation-ratio", type=float, default=3.0)
    parser.add_argument("--minimum-dynamic-fraction", type=float, default=0.4)
    parser.add_argument("--minimum-command-state-cosine", type=float, default=0.8)
    parser.add_argument("--minimum-rotation-degrees", type=float, default=0.3)
    parser.add_argument("--maximum-rotation-degrees", type=float, default=3.0)
    return parser.parse_args()


def _number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _eligible(
    row: dict[str, str],
    arm: str,
    other: str,
    args: argparse.Namespace,
) -> bool:
    target_translation = _number(
        row, f"{arm}_mean_action_translation_step_mm"
    )
    other_translation = _number(
        row, f"{other}_mean_action_translation_step_mm"
    )
    cosine = _number(
        row, f"{arm}_action_to_state_translation_step_cosine"
    )
    rotation = _number(row, f"{arm}_mean_action_rotation_step_degrees")
    return (
        args.minimum_translation_mm
        <= target_translation
        <= args.maximum_translation_mm
        and other_translation <= args.maximum_other_translation_mm
        and target_translation
        >= args.minimum_isolation_ratio * max(other_translation, 1e-12)
        and _number(
            row, f"{arm}_relative_translation_dynamic_fraction"
        )
        >= args.minimum_dynamic_fraction
        and (not np.isfinite(cosine) or cosine >= args.minimum_command_state_cosine)
        and args.minimum_rotation_degrees
        <= rotation
        <= args.maximum_rotation_degrees
    )


def _selection_record(
    row: dict[str, str], subset: str, target_arm: str, other_arm: str
) -> dict[str, Any]:
    return {
        "subset": subset,
        "target_arm": target_arm,
        "other_arm": other_arm,
        "episode_id": int(row["episode_id"]),
        "base_index": int(row["base_index"]),
        "target_translation_step_mm": _number(
            row, f"{target_arm}_mean_action_translation_step_mm"
        ),
        "other_translation_step_mm": _number(
            row, f"{other_arm}_mean_action_translation_step_mm"
        ),
        "isolation_ratio": _number(
            row, f"{target_arm}_mean_action_translation_step_mm"
        )
        / max(
            _number(row, f"{other_arm}_mean_action_translation_step_mm"),
            1e-12,
        ),
        "target_rotation_step_degrees": _number(
            row, f"{target_arm}_mean_action_rotation_step_degrees"
        ),
        "target_dynamic_fraction": _number(
            row, f"{target_arm}_relative_translation_dynamic_fraction"
        ),
        "target_command_state_cosine": _number(
            row, f"{target_arm}_action_to_state_translation_step_cosine"
        ),
        "pose_reference_source": row["pose_reference_source"],
    }


def main() -> None:
    args = parse_args()
    scan_root = Path(args.scan_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    candidate_counts = {}
    for subset in args.subsets:
        with (scan_root / subset / "motion_windows.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        for target_arm, other_arm in (("psm1", "psm2"), ("psm2", "psm1")):
            candidates = [
                row
                for row in rows
                if _eligible(row, target_arm, other_arm, args)
            ]
            candidate_counts[f"{subset}:{target_arm}"] = {
                "windows": len(candidates),
                "episodes": len({row["episode_id"] for row in candidates}),
            }
            candidates.sort(
                key=lambda row: (
                    abs(
                        _number(
                            row,
                            f"{target_arm}_mean_action_translation_step_mm",
                        )
                        - args.target_translation_mm
                    ),
                    _number(
                        row, f"{other_arm}_mean_action_translation_step_mm"
                    ),
                    -_number(
                        row,
                        f"{target_arm}_relative_translation_dynamic_fraction",
                    ),
                    int(row["episode_id"]),
                    int(row["base_index"]),
                )
            )
            seen_episodes = set()
            arm_selection = []
            for row in candidates:
                episode_id = int(row["episode_id"])
                if episode_id in seen_episodes:
                    continue
                seen_episodes.add(episode_id)
                arm_selection.append(
                    _selection_record(
                        row, subset, target_arm, other_arm
                    )
                )
                if len(arm_selection) == args.windows_per_arm:
                    break
            if len(arm_selection) != args.windows_per_arm:
                raise RuntimeError(
                    f"{subset}:{target_arm} yielded only "
                    f"{len(arm_selection)}/{args.windows_per_arm} unique episodes"
                )
            selected.extend(arm_selection)

    fieldnames = list(selected[0])
    with (output_dir / "selected_windows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    criteria = {
        "windows_per_arm_per_subset": args.windows_per_arm,
        "target_translation_mm": args.target_translation_mm,
        "translation_range_mm": [
            args.minimum_translation_mm,
            args.maximum_translation_mm,
        ],
        "maximum_other_translation_mm": args.maximum_other_translation_mm,
        "minimum_isolation_ratio": args.minimum_isolation_ratio,
        "minimum_dynamic_fraction": args.minimum_dynamic_fraction,
        "minimum_command_state_cosine": args.minimum_command_state_cosine,
        "rotation_range_degrees": [
            args.minimum_rotation_degrees,
            args.maximum_rotation_degrees,
        ],
        "ranking": (
            "closest target translation, then least opposite-arm translation, "
            "then highest target dynamic fraction; unique episodes"
        ),
    }
    manifest = {
        "diagnostic": "motion-matched anchor-preserving counterfactual selection",
        "subsets": args.subsets,
        "criteria": criteria,
        "candidate_counts": candidate_counts,
        "selected_windows": selected,
    }
    (output_dir / "selected_windows.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Motion-matched dVRK evaluation windows",
        "",
        f"Selected {len(selected)} windows: {args.windows_per_arm} per arm per",
        "dataset, each from a distinct episode.",
        "",
        "| Dataset | Target | Episode:base | Target mm/step | Other mm/step | Isolation | Rotation deg/step | Dynamic fraction |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in selected:
        lines.append(
            f"| {item['subset']} | {item['target_arm']} | "
            f"{item['episode_id']}:{item['base_index']} | "
            f"{item['target_translation_step_mm']:.3f} | "
            f"{item['other_translation_step_mm']:.3f} | "
            f"{item['isolation_ratio']:.1f}x | "
            f"{item['target_rotation_step_degrees']:.2f} | "
            f"{item['target_dynamic_fraction']:.2f} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
