#!/usr/bin/env python3
"""Compare CAMP conditions with shared, reviewed ground-truth tool ROIs.

Unlike the SAM tracking analysis, this diagnostic never derives a region from
the generated video. Every condition therefore receives exactly the same
spatiotemporal tool masks for a matched dataset window.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

OBJECTS = ("entry_left", "entry_right")
ARM_TO_TOOL = {"psm1": "entry_right", "psm2": "entry_left"}
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
COMPARISONS = (
    ("arm_b_history_real", "arm_a_h0", "raw_history_vs_h0"),
    ("arm_c_real", "arm_b_history_real", "learned_memory_vs_raw_history"),
    ("arm_b_history_zero", "arm_b_history_real", "arm_b_zero_history"),
    ("arm_b_history_permute", "arm_b_history_real", "arm_b_permuted_history"),
    ("arm_c_memory_zero", "arm_c_real", "arm_c_zero_memory"),
    ("arm_c_memory_shuffle", "arm_c_real", "arm_c_shuffled_memory"),
    ("arm_c_history_zero", "arm_c_real", "arm_c_zero_history"),
    ("arm_c_history_permute", "arm_c_real", "arm_c_permuted_history"),
)
VARIANT_PATTERN = re.compile(r"^(psm[12])_(t[xyz])_(0x|1p5x)$")
METRICS = (
    "intended_response_fraction",
    "intended_response_l1",
    "other_response_l1",
    "tool_localization_margin_l1",
    "correct_intended_tool_l1",
    "correct_other_tool_l1",
)


def _read_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"Could not decode {path}")
    return np.asarray(frames)


def _resolve(path: Path, manifest_path: Path) -> Path:
    return manifest_path.parent / Path(path).name


def _exclusive_tool_rois(
    masks: dict[str, np.ndarray], dilation: int
) -> dict[str, np.ndarray]:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1)
    )
    rois = {
        name: np.asarray(
            [
                cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
                for mask in masks[name]
            ]
        )
        for name in OBJECTS
    }
    for frame_index in range(len(rois[OBJECTS[0]])):
        overlap = rois[OBJECTS[0]][frame_index] & rois[OBJECTS[1]][frame_index]
        if not overlap.any():
            continue
        left_core = masks[OBJECTS[0]][frame_index]
        right_core = masks[OBJECTS[1]][frame_index]
        left_distance = cv2.distanceTransform(
            (~left_core).astype(np.uint8), cv2.DIST_L2, 5
        )
        right_distance = cv2.distanceTransform(
            (~right_core).astype(np.uint8), cv2.DIST_L2, 5
        )
        rois[OBJECTS[0]][frame_index][overlap & (left_distance > right_distance)] = (
            False
        )
        rois[OBJECTS[1]][frame_index][overlap & (right_distance >= left_distance)] = (
            False
        )
    return rois


def _masked_l1(left: np.ndarray, right: np.ndarray, masks: np.ndarray) -> float:
    values = []
    for left_frame, right_frame, mask in zip(left, right, masks):
        if mask.any():
            difference = (
                np.abs(left_frame.astype(np.float32) - right_frame.astype(np.float32))
                / 255.0
            )
            values.append(float(difference[mask].mean()))
    return float(np.mean(values))


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _window_means(records: dict[tuple, dict[str, float]], metric: str) -> dict:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for identity, values in records.items():
        grouped[identity[:-1]].append(values[metric])
    return {identity: float(np.mean(values)) for identity, values in grouped.items()}


def _bootstrap_ci(values: np.ndarray, draws: int = 20_000) -> list[float]:
    rng = np.random.default_rng(950)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _select(records: dict, subset: str, target: str) -> dict:
    return {
        identity: values
        for identity, values in records.items()
        if identity[:2] == (subset, target)
    }


def _analyze_manifest(
    manifest_path: Path,
    condition_root: Path,
    masks_root: Path,
    reference_hashes: dict[str, str],
    dilation: int,
) -> tuple[tuple, dict[str, float]]:
    manifest = json.loads(manifest_path.read_text())
    relative = manifest_path.relative_to(condition_root)
    subset, target = relative.parts[1:3]
    episode = int(manifest["episode_id"])
    base = int(manifest["base_index"])
    seed = int(manifest["sampling_seed"])
    identity = (subset, target, episode, base, seed)
    intended_tool = ARM_TO_TOOL[target]
    other_tool = next(name for name in OBJECTS if name != intended_tool)

    mask_path = masks_root / (f"{subset}_{target}_ep{episode:05d}_base{base:05d}.npz")
    expected_gt_hash = reference_hashes.get(mask_path.name)
    if expected_gt_hash is None:
        raise KeyError(f"No reference provenance for {mask_path.name}")
    if manifest["ground_truth_frames_sha256"] != expected_gt_hash:
        raise ValueError(
            f"{identity}: ground-truth hash does not match the reviewed mask source"
        )
    with np.load(mask_path) as archive:
        masks = {name: archive[f"gt__{name}"].astype(bool) for name in OBJECTS}
    rois = _exclusive_tool_rois(masks, dilation)

    variants = {item["name"]: item for item in manifest["variants"]}
    correct = _read_video(
        _resolve(Path(variants["correct"]["generated_video"]), manifest_path)
    )
    ground_truth = _read_video(
        _resolve(Path(manifest["ground_truth_video"]), manifest_path)
    )
    if correct.shape != ground_truth.shape:
        raise ValueError(f"{identity}: correct/ground-truth shapes differ")
    for name, mask in masks.items():
        if mask.shape != ground_truth.shape[:3]:
            raise ValueError(
                f"{identity}: {name} mask {mask.shape} != video {ground_truth.shape[:3]}"
            )

    audit = manifest["physical_action_audit"]
    if audit["anchor_mode"] != "first_row":
        raise ValueError(f"{identity}: intervention is not first-row anchored")
    if audit["variant_first_row_max_abs_error"] > 1e-6:
        raise ValueError(f"{identity}: intervention changes the first action row")

    action_archive = _resolve(
        Path(manifest["normalized_actions_archive"]), manifest_path
    )
    with np.load(action_archive) as arrays:
        correct_action = arrays["normalized__correct"]
        action_arrays = {
            name.removeprefix("normalized__"): arrays[name]
            for name in arrays.files
            if name.startswith("normalized__")
        }

    intended_responses = []
    other_responses = []
    for name, variant in variants.items():
        if name == "correct":
            continue
        match = VARIANT_PATTERN.match(name)
        if match is None or match.group(1) != target:
            raise ValueError(f"{identity}: unexpected intervention {name}")
        input_delta = float(
            np.sqrt(np.mean((action_arrays[name] - correct_action) ** 2))
        )
        if input_delta < 0.02:
            continue
        generated = _read_video(
            _resolve(Path(variant["generated_video"]), manifest_path)
        )
        intended_responses.append(
            _masked_l1(correct[1:], generated[1:], rois[intended_tool][1:])
        )
        other_responses.append(
            _masked_l1(correct[1:], generated[1:], rois[other_tool][1:])
        )
    if not intended_responses:
        raise ValueError(f"{identity}: no eligible physical interventions")
    intended = float(np.mean(intended_responses))
    other = float(np.mean(other_responses))
    return identity, {
        "intended_response_fraction": intended / max(intended + other, 1e-8),
        "intended_response_l1": intended,
        "other_response_l1": other,
        "tool_localization_margin_l1": intended - other,
        "correct_intended_tool_l1": _masked_l1(
            ground_truth[1:], correct[1:], rois[intended_tool][1:]
        ),
        "correct_other_tool_l1": _masked_l1(
            ground_truth[1:], correct[1:], rois[other_tool][1:]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--reference-gt-masks-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--roi-dilation", type=int, default=24)
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    parser.add_argument("--expected-manifests-per-condition", type=int, default=60)
    args = parser.parse_args()

    provenance_path = args.reference_gt_masks_dir / "REFERENCE_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text())
    reference_hashes = {
        record["mask_file"]: record["ground_truth_frames_sha256"]
        for record in provenance["records"]
    }
    loaded: dict[str, dict[tuple, dict[str, float]]] = {}
    for condition in args.conditions:
        condition_root = args.root / condition
        manifests = sorted(
            condition_root.glob("raw/**/*_action_intervention_episode.json")
        )
        if len(manifests) != args.expected_manifests_per_condition:
            raise ValueError(
                f"{condition}: expected {args.expected_manifests_per_condition} "
                f"manifests, got {len(manifests)}"
            )
        condition_records = {}
        for path in manifests:
            identity, metrics = _analyze_manifest(
                path,
                condition_root,
                args.reference_gt_masks_dir,
                reference_hashes,
                args.roi_dilation,
            )
            condition_records[identity] = metrics
        loaded[condition] = condition_records

    identities = set(loaded[args.conditions[0]])
    if any(set(records) != identities for records in loaded.values()):
        raise ValueError("Conditions do not contain the same matched identities")
    strata = sorted({identity[:2] for identity in identities})

    condition_summaries: dict[str, Any] = {}
    for condition, records in loaded.items():
        condition_summaries[condition] = {
            "overall": {
                metric: _summary([values[metric] for values in records.values()])
                for metric in METRICS
            },
            "strata": [
                {
                    "subset": subset,
                    "target_arm": target,
                    "metrics": {
                        metric: _summary(
                            [
                                values[metric]
                                for values in _select(records, subset, target).values()
                            ]
                        )
                        for metric in METRICS
                    },
                }
                for subset, target in strata
            ],
        }

    comparisons = []
    for candidate, reference, label in COMPARISONS:
        if candidate not in loaded or reference not in loaded:
            continue
        comparison: dict[str, Any] = {
            "label": label,
            "candidate": candidate,
            "reference": reference,
            "metrics": {},
        }
        for metric in METRICS:
            candidate_windows = _window_means(loaded[candidate], metric)
            reference_windows = _window_means(loaded[reference], metric)
            differences = np.asarray(
                [
                    candidate_windows[identity] - reference_windows[identity]
                    for identity in sorted(candidate_windows)
                ]
            )
            comparison["metrics"][metric] = {
                "candidate_minus_reference_mean": float(differences.mean()),
                "window_cluster_bootstrap_95_ci": _bootstrap_ci(differences),
                "candidate_higher_window_count": int(np.sum(differences > 0)),
                "window_count": len(differences),
            }
        comparisons.append(comparison)

    payload = {
        "diagnostic": "shared reviewed GT-ROI tool-localized CAMP comparison",
        "comparison_step": 950,
        "roi_dilation_pixels": args.roi_dilation,
        "reference_gt_masks_dir": str(args.reference_gt_masks_dir.resolve()),
        "reference_mask_provenance": str(provenance_path.resolve()),
        "records_per_condition": len(identities),
        "independent_window_clusters": len({identity[:-1] for identity in identities}),
        "condition_summaries": condition_summaries,
        "paired_comparisons": comparisons,
        "interpretation": {
            "training_match": (
                "Arm B and Arm C are compute-matched at 950 CAMP fine-tuning steps. "
                "Arm A is their pre-CAMP parent checkpoint, so Arm-A/Arm-B deltas "
                "also include those 950 adaptation steps."
            ),
            "intended_response_fraction": (
                ">0.5 means the counterfactual action changes the commanded tool ROI "
                "more than the other tool ROI"
            ),
            "tool_localization_margin_l1": (
                ">0 means greater change in the commanded tool ROI"
            ),
            "limitation": (
                "ROIs follow the reviewed ground-truth tool trajectories and are fixed "
                "across conditions. A generated tool displaced beyond the 24-pixel "
                "dilation can be undercounted; tracking-based SAM analysis is the "
                "confirmatory companion."
            ),
        },
    }
    output = args.output or args.root / "matched_grid_tool_response.json"
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
