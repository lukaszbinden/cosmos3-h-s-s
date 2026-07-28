#!/usr/bin/env python3
"""Per-tool analysis for motion-matched, first-row-anchored dVRK probes."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from pathlib import Path
from typing import Any

import analyze_c3hss_tool_specific_response as tools
import matplotlib.pyplot as plt
import numpy as np
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

ARMS = ("psm1", "psm2")
OBJECTS = ("entry_left", "entry_right")
ARM_TO_TOOL = {"psm1": "entry_right", "psm2": "entry_left"}
COMPONENTS = ("tx", "ty", "tz", "rx", "ry", "rz", "jaw")
GAINS = ("0x", "1p5x")
VARIANT_PATTERN = re.compile(r"^(psm[12])_(t[xyz]|r[xyz]|jaw)_(0x|1p5x)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument(
        "--sam-code-sha",
        default="2b90b9f5ceec907a1c18123530e92e794ad901a4",
    )
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--roi-dilation", type=int, default=24)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--iteration", type=int, default=700)
    parser.add_argument(
        "--reference-gt-masks-dir",
        default=None,
        help=(
            "Reuse gt__ tool tracks from this masks directory while tracking "
            "only the seed-specific correct-action prediction."
        ),
    )
    parser.add_argument("--reuse-masks", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Analyze only manifests generated with this diffusion seed.",
    )
    parser.add_argument(
        "--expected-manifests",
        type=int,
        default=12,
        help="Expected episode manifests after optional seed filtering.",
    )
    return parser.parse_args()


def _mean(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


def _centroid_step_pixels(masks: np.ndarray) -> float:
    centroids = []
    for mask in masks:
        rows, columns = np.nonzero(mask)
        if not len(rows):
            raise RuntimeError("Tracked tool mask is empty")
        centroids.append([float(columns.mean()), float(rows.mean())])
    return float(np.linalg.norm(np.diff(np.asarray(centroids), axis=0), axis=1).mean())


def _manifest_identity(
    input_root: Path, manifest_path: Path, manifest: dict[str, Any]
) -> tuple[str, str, int, int]:
    relative = manifest_path.relative_to(input_root / "raw")
    if len(relative.parts) != 3:
        raise ValueError(f"Unexpected manifest layout: {relative}")
    subset, target_arm = relative.parts[:2]
    if target_arm not in ARMS:
        raise ValueError(f"Unexpected target arm in {relative}")
    return (
        subset,
        target_arm,
        int(manifest["episode_id"]),
        int(manifest["base_index"]),
    )


def _mask_path(
    masks_dir: Path, subset: str, arm: str, episode_id: int, base_index: int
) -> Path:
    return masks_dir / (f"{subset}_{arm}_ep{episode_id:05d}_base{base_index:05d}.npz")


def _tracking_path(
    overlays_dir: Path,
    subset: str,
    arm: str,
    episode_id: int,
    base_index: int,
) -> Path:
    return overlays_dir / (
        f"{subset}_{arm}_ep{episode_id:05d}_base{base_index:05d}_tracking.mp4"
    )


def _resolve_video(
    directory: Path,
    episode_id: int,
    base_index: int,
    suffix: str,
    seed: int | None,
) -> Path:
    seed_pattern = "*" if seed is None else str(seed)
    matches = list(
        directory.glob(
            f"*ep{episode_id:05d}_base{base_index:05d}_seed{seed_pattern}_{suffix}.mp4"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {suffix} video for episode {episode_id}, "
            f"base {base_index}; got {matches}"
        )
    return matches[0]


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "tracking_overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(args.prompts).read_text())
    reference_size = tuple(prompts["reference_size"])
    manifests = sorted(
        (input_root / "raw").glob("*/*/*_action_intervention_episode.json")
    )
    if args.seed is not None:
        manifests = [path for path in manifests if f"_seed{args.seed}_" in path.name]
    if len(manifests) != args.expected_manifests:
        raise RuntimeError(
            f"Expected {args.expected_manifests} episode manifests for seed {args.seed}, "
            f"found {len(manifests)}"
        )

    manifest_records = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        identity = _manifest_identity(input_root, manifest_path, manifest)
        manifest_records.append((manifest_path, manifest, identity))
    mask_paths = {
        identity: _mask_path(masks_dir, *identity)
        for _, _, identity in manifest_records
    }
    need_tracking = any(
        not (args.reuse_masks and path.exists()) for path in mask_paths.values()
    )
    prompt_audits: dict[str, Any] = {}
    reference_gt_masks: dict[tuple[str, str, int, int], dict[str, np.ndarray]] = {}
    if args.reference_gt_masks_dir is not None:
        reference_dir = Path(args.reference_gt_masks_dir).resolve()
        for identity in mask_paths:
            reference_path = _mask_path(reference_dir, *identity)
            if not reference_path.exists():
                raise FileNotFoundError(reference_path)
            with np.load(reference_path) as archive:
                reference_gt_masks[identity] = {
                    name: archive[f"gt__{name}"].astype(bool) for name in OBJECTS
                }
    previous_summary_path = output_dir / "matched_response_summary.json"
    if args.reuse_masks and previous_summary_path.exists():
        previous_summary = json.loads(previous_summary_path.read_text())
        for episode in previous_summary.get("episodes_detail", []):
            prompt_key = (
                f"{episode['subset']}:{episode['episode_id']}:{episode['base_index']}"
            )
            if episode.get("prompt_audit") is not None:
                prompt_audits[prompt_key] = episode["prompt_audit"]

    if need_tracking:
        initial_by_identity = {}
        if reference_gt_masks:
            initial_by_identity = {
                identity: {name: masks[name][0] for name in OBJECTS}
                for identity, masks in reference_gt_masks.items()
            }
        else:
            image_model = build_sam2(
                args.sam_config, args.sam_checkpoint, device=args.device
            )
            image_predictor = SAM2ImagePredictor(image_model)
            for manifest_path, _, identity in manifest_records:
                subset, target_arm, episode_id, base_index = identity
                if args.reuse_masks and mask_paths[identity].exists():
                    continue
                ground_truth = tools._read_video(
                    _resolve_video(
                        manifest_path.parent,
                        episode_id,
                        base_index,
                        "ground_truth",
                        args.seed,
                    )
                )
                prompt_key = f"{subset}:{episode_id}:{base_index}"
                if prompt_key not in prompts["episodes"]:
                    raise KeyError(f"Missing reviewed prompts for {prompt_key}")
                initial, audit = tools._initial_masks(
                    image_predictor,
                    ground_truth[0],
                    prompts["episodes"][prompt_key],
                    reference_size,
                )
                initial_by_identity[identity] = initial
                prompt_audits[prompt_key] = audit
                print(
                    f"PROMPTED subset={subset} arm={target_arm} "
                    f"episode={episode_id} base={base_index}",
                    flush=True,
                )
            del image_predictor, image_model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        video_predictor = build_sam2_video_predictor(
            args.sam_config, args.sam_checkpoint, device=args.device
        )
        for manifest_path, _, identity in manifest_records:
            subset, target_arm, episode_id, base_index = identity
            if args.reuse_masks and mask_paths[identity].exists():
                continue
            ground_truth = tools._read_video(
                _resolve_video(
                    manifest_path.parent,
                    episode_id,
                    base_index,
                    "ground_truth",
                    args.seed,
                )
            )
            correct = tools._read_video(
                _resolve_video(
                    manifest_path.parent,
                    episode_id,
                    base_index,
                    "correct",
                    args.seed,
                )
            )
            initial = initial_by_identity[identity]
            gt_masks = (
                reference_gt_masks[identity]
                if reference_gt_masks
                else tools._track_video(video_predictor, ground_truth, initial)
            )
            correct_masks = tools._track_video(video_predictor, correct, initial)
            np.savez_compressed(
                mask_paths[identity],
                **{f"gt__{name}": gt_masks[name].astype(np.uint8) for name in OBJECTS},
                **{
                    f"correct__{name}": correct_masks[name].astype(np.uint8)
                    for name in OBJECTS
                },
            )
            tools._write_tracking_video(
                _tracking_path(overlays_dir, *identity),
                ground_truth,
                correct,
                gt_masks,
                correct_masks,
                args.fps,
            )
            print(
                f"TRACKED subset={subset} arm={target_arm} "
                f"episode={episode_id} base={base_index}",
                flush=True,
            )
        del video_predictor
        gc.collect()

    episode_records = []
    variant_rows = []
    for manifest_path, manifest, identity in manifest_records:
        subset, target_arm, episode_id, base_index = identity
        intended_tool = ARM_TO_TOOL[target_arm]
        other_tool = next(tool for tool in OBJECTS if tool != intended_tool)
        ground_truth = tools._read_video(
            _resolve_video(
                manifest_path.parent,
                episode_id,
                base_index,
                "ground_truth",
                args.seed,
            )
        )
        correct = tools._read_video(
            _resolve_video(
                manifest_path.parent,
                episode_id,
                base_index,
                "correct",
                args.seed,
            )
        )
        with np.load(mask_paths[identity]) as archive:
            gt_masks = {name: archive[f"gt__{name}"].astype(bool) for name in OBJECTS}
            correct_masks = {
                name: archive[f"correct__{name}"].astype(bool) for name in OBJECTS
            }
        tracking_overlay = _tracking_path(overlays_dir, *identity)
        if not tracking_overlay.exists():
            tools._write_tracking_video(
                tracking_overlay,
                ground_truth,
                correct,
                gt_masks,
                correct_masks,
                args.fps,
            )
        union_masks = {name: gt_masks[name] | correct_masks[name] for name in OBJECTS}
        rois = tools._exclusive_tool_rois(union_masks, args.roi_dilation)
        gt_flow = tools._flow(ground_truth)
        correct_flow = tools._flow(correct)
        gt_target_step_pixels = _centroid_step_pixels(gt_masks[intended_tool])
        gt_other_step_pixels = _centroid_step_pixels(gt_masks[other_tool])
        correct_tool_metrics = {}
        for object_name in OBJECTS:
            trajectory = tools._trajectory_metrics(
                gt_masks[object_name], correct_masks[object_name], object_name
            )
            correct_tool_metrics[object_name] = {
                "l1": tools._masked_l1(
                    ground_truth[1:], correct[1:], rois[object_name][1:]
                ),
                **tools._flow_metrics(gt_flow, correct_flow, rois[object_name]),
                **{
                    key: value
                    for key, value in trajectory.items()
                    if not isinstance(value, np.ndarray)
                },
            }

        action_matches = list(
            manifest_path.parent.glob(
                f"*ep{episode_id:05d}_base{base_index:05d}_seed"
                f"{'*' if args.seed is None else args.seed}_normalized_actions.npz"
            )
        )
        if len(action_matches) != 1:
            raise RuntimeError(f"Expected one action archive, got {action_matches}")
        with np.load(action_matches[0]) as archive:
            correct_action = archive["normalized__correct"]
            action_arrays = {
                key.removeprefix("normalized__"): archive[key]
                for key in archive.files
                if key.startswith("normalized__")
            }

        for variant in manifest["variants"]:
            variant_name = variant["name"]
            if variant_name == "correct":
                continue
            match = VARIANT_PATTERN.match(variant_name)
            if not match or match.group(1) != target_arm:
                raise ValueError(
                    f"Unexpected variant {variant_name} for target {target_arm}"
                )
            generated = tools._read_video(
                _resolve_video(
                    manifest_path.parent,
                    episode_id,
                    base_index,
                    variant_name,
                    args.seed,
                )
            )
            generated_flow = tools._flow(generated)
            normalized_delta_rms = float(
                np.sqrt(np.mean((action_arrays[variant_name] - correct_action) ** 2))
            )
            row: dict[str, Any] = {
                "subset": subset,
                "target_arm": target_arm,
                "episode_id": episode_id,
                "base_index": base_index,
                "seed": args.seed,
                "variant": variant_name,
                "component": match.group(2),
                "gain": match.group(3),
                "normalized_input_delta_rms": normalized_delta_rms,
                "intended_tool": intended_tool,
                "other_tool": other_tool,
            }
            for object_name in OBJECTS:
                l1 = tools._masked_l1(
                    ground_truth[1:], generated[1:], rois[object_name][1:]
                )
                paired_delta = tools._masked_l1(
                    correct[1:], generated[1:], rois[object_name][1:]
                )
                flow_metrics = tools._flow_metrics(
                    gt_flow, generated_flow, rois[object_name]
                )
                row[f"{object_name}_l1"] = l1
                row[f"{object_name}_l1_delta_vs_correct"] = (
                    l1 - correct_tool_metrics[object_name]["l1"]
                )
                row[f"{object_name}_paired_output_delta_l1"] = paired_delta
                for key, value in flow_metrics.items():
                    row[f"{object_name}_{key}"] = value
                    row[f"{object_name}_{key}_delta_vs_correct"] = (
                        value - correct_tool_metrics[object_name][key]
                    )
            intended_response = row[f"{intended_tool}_paired_output_delta_l1"]
            other_response = row[f"{other_tool}_paired_output_delta_l1"]
            row["intended_response_fraction"] = float(
                intended_response / max(intended_response + other_response, 1e-8)
            )
            variant_rows.append(row)

        stats_audit = manifest["physical_action_audit"]
        if stats_audit["anchor_mode"] != "first_row":
            raise ValueError(f"{identity} did not use first_row anchoring")
        if stats_audit["variant_first_row_max_abs_error"] > 1e-6:
            raise ValueError(f"{identity} failed first-row preservation")
        episode_records.append(
            {
                "subset": subset,
                "target_arm": target_arm,
                "episode_id": episode_id,
                "base_index": base_index,
                "seed": args.seed,
                "intended_tool": intended_tool,
                "correct_tool_metrics": correct_tool_metrics,
                "ground_truth_tool_motion": {
                    "intended_mean_centroid_step_pixels": gt_target_step_pixels,
                    "other_mean_centroid_step_pixels": gt_other_step_pixels,
                    "intended_to_other_ratio": float(
                        gt_target_step_pixels / max(gt_other_step_pixels, 1e-8)
                    ),
                },
                "anchor_audit": stats_audit,
                "mask_archive": str(mask_paths[identity]),
                "tracking_overlay": str(tracking_overlay),
                "prompt_audit": prompt_audits.get(
                    f"{subset}:{episode_id}:{base_index}"
                ),
            }
        )

    with (output_dir / "per_variant_tool_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(variant_rows[0]))
        writer.writeheader()
        writer.writerows(variant_rows)

    episode_groups = []
    for episode in episode_records:
        group = [
            row
            for row in variant_rows
            if row["subset"] == episode["subset"]
            and row["target_arm"] == episode["target_arm"]
            and row["episode_id"] == episode["episode_id"]
            and row["base_index"] == episode["base_index"]
            and row["normalized_input_delta_rms"] >= 0.02
        ]
        intended_tool = episode["intended_tool"]
        other_tool = next(tool for tool in OBJECTS if tool != intended_tool)
        intended_response = _mean(group, f"{intended_tool}_paired_output_delta_l1")
        other_response = _mean(group, f"{other_tool}_paired_output_delta_l1")
        episode_groups.append(
            {
                "subset": episode["subset"],
                "target_arm": episode["target_arm"],
                "episode_id": episode["episode_id"],
                "base_index": episode["base_index"],
                "eligible_interventions": len(group),
                "intended_response_fraction": float(
                    intended_response / max(intended_response + other_response, 1e-8)
                ),
                "intended_response_l1": intended_response,
                "other_response_l1": other_response,
                **episode["ground_truth_tool_motion"],
            }
        )

    aggregate_groups = []
    for subset in sorted({row["subset"] for row in variant_rows}):
        present_arms = {
            row["target_arm"] for row in variant_rows if row["subset"] == subset
        }
        for target_arm in (arm for arm in ARMS if arm in present_arms):
            group = [
                row
                for row in variant_rows
                if row["subset"] == subset
                and row["target_arm"] == target_arm
                and row["normalized_input_delta_rms"] >= 0.02
            ]
            intended_tool = ARM_TO_TOOL[target_arm]
            other_tool = next(tool for tool in OBJECTS if tool != intended_tool)
            intended_response = _mean(group, f"{intended_tool}_paired_output_delta_l1")
            other_response = _mean(group, f"{other_tool}_paired_output_delta_l1")
            aggregate_groups.append(
                {
                    "subset": subset,
                    "target_arm": target_arm,
                    "intended_tool": intended_tool,
                    "eligible_interventions": len(group),
                    "intended_response_l1": intended_response,
                    "other_response_l1": other_response,
                    "intended_response_fraction": float(
                        intended_response
                        / max(intended_response + other_response, 1e-8)
                    ),
                    "mean_intended_l1_delta_vs_correct": _mean(
                        group, f"{intended_tool}_l1_delta_vs_correct"
                    ),
                }
            )

    component_groups = []
    keys = sorted(
        {
            (
                row["subset"],
                row["target_arm"],
                row["component"],
                row["gain"],
            )
            for row in variant_rows
        }
    )
    for subset, target_arm, component, gain in keys:
        group = [
            row
            for row in variant_rows
            if row["subset"] == subset
            and row["target_arm"] == target_arm
            and row["component"] == component
            and row["gain"] == gain
        ]
        intended_tool = ARM_TO_TOOL[target_arm]
        other_tool = next(tool for tool in OBJECTS if tool != intended_tool)
        intended_response = _mean(group, f"{intended_tool}_paired_output_delta_l1")
        other_response = _mean(group, f"{other_tool}_paired_output_delta_l1")
        component_groups.append(
            {
                "subset": subset,
                "target_arm": target_arm,
                "component": component,
                "gain": gain,
                "intended_response_fraction": float(
                    intended_response / max(intended_response + other_response, 1e-8)
                ),
                "intended_response_l1": intended_response,
                "other_response_l1": other_response,
                "intended_l1_delta_vs_correct": _mean(
                    group, f"{intended_tool}_l1_delta_vs_correct"
                ),
            }
        )

    subsets = sorted({item["subset"] for item in aggregate_groups})
    arms = [
        arm
        for arm in ARMS
        if any(item["target_arm"] == arm for item in aggregate_groups)
    ]
    response_matrix = np.asarray(
        [
            [
                next(
                    item["intended_response_fraction"]
                    for item in aggregate_groups
                    if item["subset"] == subset and item["target_arm"] == arm
                )
                for arm in arms
            ]
            for subset in subsets
        ]
    )
    figure, axis = plt.subplots(
        figsize=(max(4.5, 2.6 * len(arms)), max(3.5, 1.8 * len(subsets)))
    )
    image = axis.imshow(response_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(arms)), [f"{arm.upper()}-active" for arm in arms])
    axis.set_yticks(
        range(len(subsets)),
        [subset.replace("_", " ").title() for subset in subsets],
    )
    axis.set_title("Intended-tool response fraction")
    for row_index in range(len(subsets)):
        for column_index in range(len(arms)):
            axis.text(
                column_index,
                row_index,
                f"{response_matrix[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                color="white"
                if response_matrix[row_index, column_index] < 0.35
                else "black",
            )
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(output_dir / "matched_response_matrix.png", dpi=180)
    plt.close(figure)

    payload = {
        "diagnostic": ("motion-matched first-row-anchored per-tool action response"),
        "model": f"C3-H-S-S CAMP-lite H16 iter {args.iteration}",
        "iteration": args.iteration,
        "seed": args.seed,
        "episodes": len(episode_records),
        "sam": {
            "checkpoint": str(Path(args.sam_checkpoint)),
            "checkpoint_sha256": tools._file_sha256(Path(args.sam_checkpoint)),
            "config": args.sam_config,
            "code_sha": args.sam_code_sha,
            "device": args.device,
            "prompt_file": str(Path(args.prompts)),
        },
        "anchor_mode": "first_row",
        "reference_gt_masks_dir": (
            str(Path(args.reference_gt_masks_dir).resolve())
            if args.reference_gt_masks_dir is not None
            else None
        ),
        "arm_to_tool_mapping": ARM_TO_TOOL,
        "aggregate_groups": aggregate_groups,
        "episode_groups": episode_groups,
        "component_groups": component_groups,
        "episodes_detail": episode_records,
    }
    tools._write_json(output_dir / "matched_response_summary.json", payload)

    lines = [
        "# Motion-matched, anchor-preserving action response",
        "",
        "Each cell measures first-row-anchored physical-axis interventions",
        "against the correct-action generation from the same fixed input.",
        "Ground-truth tool masks are tracked separately from generated videos.",
        "",
        "| Dataset | Target arm | Intended tool | Eligible probes | Intended response fraction | Intended Δ | Other Δ | Accuracy Δ |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in aggregate_groups:
        lines.append(
            f"| {item['subset']} | {item['target_arm']} | "
            f"{item['intended_tool']} | {item['eligible_interventions']} | "
            f"{item['intended_response_fraction']:.3f} | "
            f"{item['intended_response_l1']:.4f} | "
            f"{item['other_response_l1']:.4f} | "
            f"{item['mean_intended_l1_delta_vs_correct']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "A response fraction of 0.5 is unlocalized; values above 0.5",
            "concentrate the counterfactual change on the commanded tool.",
            "",
            "## Episode replication and ground-truth motion",
            "",
            "| Dataset | Target | Episode:base | Eligible | Response fraction | GT target/other tool motion |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for item in episode_groups:
        lines.append(
            f"| {item['subset']} | {item['target_arm']} | "
            f"{item['episode_id']}:{item['base_index']} | "
            f"{item['eligible_interventions']} | "
            f"{item['intended_response_fraction']:.3f} | "
            f"{item['intended_to_other_ratio']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "Ground-truth target/other ratios describe visible motion in the",
            "recorded window; response fractions describe where the model",
            "places the counterfactual change. A strong ground-truth ratio",
            "therefore does not guarantee correct model localization.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
