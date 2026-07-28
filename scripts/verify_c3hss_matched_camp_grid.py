#!/usr/bin/env python3
"""Verify the provenance and tensor contracts of the matched CAMP eval grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

CONDITIONS = (
    "arm_a_h0",
    "arm_b_history_real",
    "arm_b_history_zero",
    "arm_b_history_permute",
    "arm_c_real",
    "arm_c_memory_zero",
    "arm_c_memory_shuffle",
    "arm_c_history_zero",
    "arm_c_history_permute",
)
EXPECTED_SHAPES = {
    "arm_a_h0": ([12, 44], 0),
    "arm_b_history_real": ([28, 44], 0),
    "arm_b_history_zero": ([28, 44], 0),
    "arm_b_history_permute": ([28, 44], 0),
    "arm_c_real": ([31, 44], 3),
    "arm_c_memory_zero": ([31, 44], 3),
    "arm_c_memory_shuffle": ([31, 44], 3),
    "arm_c_history_zero": ([31, 44], 3),
    "arm_c_history_permute": ([31, 44], 3),
}
EXPECTED_CHECKPOINT_ITERATIONS = {
    condition: (15000 if condition == "arm_a_h0" else 950) for condition in CONDITIONS
}


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _identity(path: Path, record: dict[str, Any], condition_root: Path) -> tuple:
    relative = path.relative_to(condition_root)
    if len(relative.parts) < 4 or relative.parts[0] != "raw":
        raise ValueError(f"Unexpected episode-manifest location: {path}")
    subset, target = relative.parts[1:3]
    return (
        subset,
        target,
        int(record["episode_id"]),
        int(record["base_index"]),
        int(record["sampling_seed"]),
    )


def _correct_action_hash(record: dict[str, Any]) -> str:
    variants = {item["name"]: item for item in record["variants"]}
    return variants["correct"]["action"]["sha256"]


def _load_condition(root: Path, condition: str) -> dict[tuple, tuple[Path, dict]]:
    condition_root = root / condition
    paths = sorted(condition_root.glob("raw/**/*_action_intervention_episode.json"))
    loaded: dict[tuple, tuple[Path, dict]] = {}
    for path in paths:
        record = json.loads(path.read_text())
        identity = _identity(path, record, condition_root)
        if identity in loaded:
            raise ValueError(f"Duplicate identity in {condition}: {identity}")
        loaded[identity] = (path, record)
    return loaded


def _check_first_row_anchor(path: Path) -> float:
    archive_name = json.loads(path.read_text())["normalized_actions_archive"]
    archive = path.parent / Path(archive_name).name
    with np.load(archive) as arrays:
        correct = arrays["normalized__correct"]
        variants = [
            arrays[name]
            for name in arrays.files
            if name.startswith("normalized__") and name != "normalized__correct"
        ]
        if not variants:
            raise ValueError(f"No physical intervention variants in {archive}")
        return max(float(np.max(np.abs(item[0] - correct[0]))) for item in variants)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--expected-windows-per-condition",
        type=int,
        default=60,
        help="Use 4 for the two-condition, one-seed smoke.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    parser.add_argument(
        "--skip-provenance",
        action="store_true",
        help="Allow smoke artifacts produced before per-episode provenance fields.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    observations: list[str] = []
    ablation_effectiveness: dict[str, dict[str, int]] = {}
    conditions = {
        condition: _load_condition(args.root, condition)
        for condition in args.conditions
    }
    for condition, records in conditions.items():
        if len(records) != args.expected_windows_per_condition:
            failures.append(
                f"{condition}: expected {args.expected_windows_per_condition} "
                f"records, found {len(records)}"
            )
        expected_shape, expected_memory_rows = EXPECTED_SHAPES[condition]
        for identity, (path, record) in records.items():
            if record["eval_condition"] != condition:
                failures.append(
                    f"{condition} {identity}: embedded condition is "
                    f"{record['eval_condition']!r}"
                )
            if not args.skip_provenance and record.get("comparison_step") != 950:
                failures.append(
                    f"{condition} {identity}: comparison step is "
                    f"{record.get('comparison_step')!r}, expected 950"
                )
            if (
                not args.skip_provenance
                and record.get("checkpoint_requested_iteration")
                != EXPECTED_CHECKPOINT_ITERATIONS[condition]
            ):
                failures.append(
                    f"{condition} {identity}: requested checkpoint iteration is "
                    f"{record.get('checkpoint_requested_iteration')!r}, expected "
                    f"{EXPECTED_CHECKPOINT_ITERATIONS[condition]}"
                )
            audit = record.get("conditioning_audit") or {}
            if audit.get("model_action_shape") != expected_shape:
                failures.append(
                    f"{condition} {identity}: action shape "
                    f"{audit.get('model_action_shape')} != {expected_shape}"
                )
            if audit.get("num_memory_action_rows") != expected_memory_rows:
                failures.append(
                    f"{condition} {identity}: memory rows "
                    f"{audit.get('num_memory_action_rows')} != "
                    f"{expected_memory_rows}"
                )
            try:
                anchor_error = _check_first_row_anchor(path)
            except (KeyError, OSError, ValueError) as error:
                failures.append(f"{condition} {identity}: anchor audit failed: {error}")
            else:
                if anchor_error > 1e-6:
                    failures.append(
                        f"{condition} {identity}: first-row anchor error "
                        f"{anchor_error:.3e}"
                    )

    identity_sets = [set(records) for records in conditions.values()]
    common = set.intersection(*identity_sets) if identity_sets else set()
    union = set.union(*identity_sets) if identity_sets else set()
    if common != union:
        for condition, records in conditions.items():
            missing = sorted(union - set(records))
            if missing:
                failures.append(f"{condition}: missing identities {missing}")

    for identity in sorted(common):
        records = {
            condition: condition_records[identity][1]
            for condition, condition_records in conditions.items()
        }
        for field in ("ground_truth_frames_sha256", "current_action_sha256"):
            values = {record[field] for record in records.values()}
            if len(values) != 1:
                failures.append(f"{identity}: mismatched {field}: {values}")
        correct_hashes = {_correct_action_hash(record) for record in records.values()}
        if len(correct_hashes) != 1:
            failures.append(
                f"{identity}: correct current-action variant hashes differ: "
                f"{correct_hashes}"
            )
        seeds = {record["data_seed"] for record in records.values()}
        if seeds != {1729}:
            failures.append(f"{identity}: unexpected data seeds {seeds}")

        # Real history must be byte-identical across both trained arms and all
        # counterfactuals that alter only memory.
        history_real_names = (
            "arm_b_history_real",
            "arm_c_real",
            "arm_c_memory_zero",
            "arm_c_memory_shuffle",
        )
        present_history_real = [
            records[name]["history_action_sha256"]
            for name in history_real_names
            if name in records
        ]
        if present_history_real and (
            None in present_history_real or len(set(present_history_real)) != 1
        ):
            failures.append(
                f"{identity}: real-history hashes differ: {present_history_real}"
            )

        for left, right in (
            ("arm_b_history_zero", "arm_c_history_zero"),
            ("arm_b_history_permute", "arm_c_history_permute"),
        ):
            if (
                left in records
                and right in records
                and records[left]["history_action_sha256"]
                != records[right]["history_action_sha256"]
            ):
                failures.append(f"{identity}: {left}/{right} history mismatch")

        memory_real_names = (
            "arm_c_real",
            "arm_c_history_zero",
            "arm_c_history_permute",
        )
        present_memory_real = [
            records[name]["memory_code_sha256"]
            for name in memory_real_names
            if name in records
        ]
        if present_memory_real and (
            None in present_memory_real or len(set(present_memory_real)) != 1
        ):
            failures.append(
                f"{identity}: real-memory hashes differ: {present_memory_real}"
            )
        if "arm_a_h0" in records:
            if records["arm_a_h0"]["history_action_sha256"] is not None:
                failures.append(f"{identity}: Arm A unexpectedly has history")
            if records["arm_a_h0"]["memory_code_sha256"] is not None:
                failures.append(f"{identity}: Arm A unexpectedly has memory")

    ablation_pairs = (
        (
            "arm_b_history_zero",
            "arm_b_history_real",
            "history_action_sha256",
            True,
        ),
        (
            "arm_b_history_permute",
            "arm_b_history_real",
            "history_action_sha256",
            True,
        ),
        ("arm_c_memory_zero", "arm_c_real", "memory_code_sha256", True),
        ("arm_c_memory_shuffle", "arm_c_real", "memory_code_sha256", False),
        (
            "arm_c_history_zero",
            "arm_c_real",
            "history_action_sha256",
            True,
        ),
        (
            "arm_c_history_permute",
            "arm_c_real",
            "history_action_sha256",
            True,
        ),
    )
    for ablated, reference, field, require_every_change in ablation_pairs:
        if ablated not in conditions or reference not in conditions:
            continue
        changed = sum(
            conditions[ablated][identity][1][field]
            != conditions[reference][identity][1][field]
            for identity in common
        )
        label = f"{ablated}_vs_{reference}"
        ablation_effectiveness[label] = {
            "changed_records": changed,
            "total_records": len(common),
        }
        if require_every_change and changed != len(common):
            failures.append(
                f"{label}: only {changed}/{len(common)} tensors actually changed"
            )
        elif not require_every_change and changed < len(common):
            observations.append(
                f"{label}: {changed}/{len(common)} records changed; repeated "
                "quantized codes make the remaining shuffled donors identical"
            )
        if not require_every_change and changed < len(common) / 2:
            failures.append(
                f"{label}: fewer than half of shuffled-memory records changed"
            )

    # Zero-history and zero-memory hashes should each collapse to one constant
    # across all examples; shuffled tensors should retain example variation.
    relational: dict[str, dict[str, int]] = defaultdict(dict)
    for condition, records in conditions.items():
        history_hashes = {
            record["history_action_sha256"] for _, record in records.values()
        }
        memory_hashes = {record["memory_code_sha256"] for _, record in records.values()}
        relational[condition] = {
            "unique_history_hashes": len(history_hashes),
            "unique_memory_hashes": len(memory_hashes),
        }
    for condition in ("arm_b_history_zero", "arm_c_history_zero"):
        if (
            condition in conditions
            and relational[condition]["unique_history_hashes"] != 1
        ):
            failures.append(f"{condition}: zero history is not a single constant hash")
    if (
        "arm_c_memory_zero" in conditions
        and relational["arm_c_memory_zero"]["unique_memory_hashes"] != 1
    ):
        failures.append("arm_c_memory_zero: zero memory is not a single constant hash")

    summary = {
        "root": str(args.root),
        "conditions": {
            name: {
                "record_count": len(records),
                **relational[name],
            }
            for name, records in conditions.items()
        },
        "common_identity_count": len(common),
        "ablation_effectiveness": ablation_effectiveness,
        "checks_passed": not failures,
        "failures": failures,
        "observations": observations,
    }
    output = args.output or args.root / "matched_grid_verification.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
