#!/usr/bin/env python3
"""Tool-specific action-following diagnostic for C3-H-S-S dVRK videos.

The script uses reviewed first-frame prompts and SAM2 video propagation to
track the two instruments in the ground-truth and correct-action clips.  The
union of those trajectories defines exclusive per-tool evaluation regions for
all physical action interventions.  This separates intended-tool response
from response on the other arm, tissue, and camera/background motion.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

OBJECTS = ("entry_left", "entry_right")
COLORS = {"entry_left": (80, 255, 80), "entry_right": (255, 80, 80)}
COMPONENTS = ("tx", "ty", "tz", "rx", "ry", "rz", "jaw")
GAINS = ("0x", "1p5x")
VARIANT_PATTERN = re.compile(r"^(psm[12])_(t[xyz]|r[xyz]|jaw)_(0x|1p5x)$")
FLOW_SIZE = (480, 272)


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
    parser.add_argument(
        "--sam-config", default="configs/sam2.1/sam2.1_hiera_t.yaml"
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--roi-dilation", type=int, default=24)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--reuse-masks",
        action="store_true",
        help="Skip SAM2 propagation when every expected mask archive already exists.",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _resolve_video(directory: Path, episode_id: int, suffix: str) -> Path:
    matches = list(directory.glob(f"*ep{episode_id:05d}_seed*_{suffix}.mp4"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one episode {episode_id} video ending in {suffix}, got {matches}"
        )
    return matches[0]


def _episode_key(subset: str, episode_id: int) -> str:
    return f"{subset}:{episode_id}"


def _scale_prompt(
    prompt: dict[str, Any],
    reference_size: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    reference_width, reference_height = reference_size
    width, height = image_size
    scale = np.asarray(
        [width / reference_width, height / reference_height], dtype=np.float32
    )
    box = np.asarray(prompt["box"], dtype=np.float32).reshape(2, 2) * scale
    positive = np.asarray(prompt["positive_points"], dtype=np.float32) * scale
    negative = np.asarray(prompt["negative_points"], dtype=np.float32) * scale
    polygon = prompt.get("manual_polygon")
    scaled_polygon = (
        None
        if polygon is None
        else np.rint(np.asarray(polygon, dtype=np.float32) * scale).astype(np.int32)
    )
    return box.reshape(4), positive, negative, scaled_polygon


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask.astype(bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def _initial_masks(
    predictor: SAM2ImagePredictor,
    frame: np.ndarray,
    episode_prompts: dict[str, Any],
    reference_size: tuple[int, int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    height, width = frame.shape[:2]
    predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    masks: dict[str, np.ndarray] = {}
    audit: dict[str, Any] = {}
    for object_name in OBJECTS:
        box, positive, negative, polygon = _scale_prompt(
            episode_prompts[object_name],
            reference_size,
            (width, height),
        )
        if polygon is not None:
            selected = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(selected, [polygon], 1)
            score = None
            source = "reviewed manual polygon"
        else:
            points = np.concatenate([positive, negative])
            labels = np.concatenate(
                [
                    np.ones(len(positive), dtype=np.int32),
                    np.zeros(len(negative), dtype=np.int32),
                ]
            )
            candidates, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=True,
            )
            candidate_index = int(np.argmax(scores))
            selected = np.asarray(candidates[candidate_index]) > 0
            score = float(scores[candidate_index])
            source = f"SAM2 multimask candidate {candidate_index}"
        selected = _largest_component(selected)
        if selected.mean() < 0.001 or selected.mean() > 0.25:
            raise RuntimeError(
                f"Implausible initial {object_name} mask area {selected.mean():.4f}"
            )
        masks[object_name] = selected
        audit[object_name] = {
            "source": source,
            "sam_score": score,
            "area_fraction": float(selected.mean()),
            "box": box.tolist(),
            "positive_points": positive.tolist(),
            "negative_points": negative.tolist(),
        }
    overlap = masks[OBJECTS[0]] & masks[OBJECTS[1]]
    if overlap.any():
        # Initial prompts are identity-defining. Do not give an overlapping
        # pixel to both objects; assign it to the closest non-overlap core.
        left_only = masks[OBJECTS[0]] & ~overlap
        right_only = masks[OBJECTS[1]] & ~overlap
        left_distance = cv2.distanceTransform((~left_only).astype(np.uint8), cv2.DIST_L2, 5)
        right_distance = cv2.distanceTransform(
            (~right_only).astype(np.uint8), cv2.DIST_L2, 5
        )
        masks[OBJECTS[0]][overlap & (left_distance > right_distance)] = False
        masks[OBJECTS[1]][overlap & (right_distance >= left_distance)] = False
    return masks, audit


def _track_video(
    predictor: Any,
    video: np.ndarray,
    initial_masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="c3hss_tool_frames_") as temporary:
        frame_dir = Path(temporary)
        for frame_index, frame in enumerate(video):
            if not cv2.imwrite(
                str(frame_dir / f"{frame_index:05d}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise RuntimeError("Failed to stage a SAM2 video frame")
        state = predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
        )
        for object_id, object_name in enumerate(OBJECTS, start=1):
            predictor.add_new_mask(
                state,
                frame_idx=0,
                obj_id=object_id,
                mask=initial_masks[object_name],
            )
        tracked = {
            object_name: np.zeros(
                (len(video), *video.shape[1:3]), dtype=bool
            )
            for object_name in OBJECTS
        }
        for frame_index, object_ids, logits in predictor.propagate_in_video(state):
            for object_position, object_id in enumerate(object_ids):
                object_name = OBJECTS[int(object_id) - 1]
                tracked[object_name][frame_index] = _largest_component(
                    (logits[object_position, 0] > 0).cpu().numpy()
                )
        del state
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return tracked


def _mask_point(mask: np.ndarray, object_name: str) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.nonzero(mask)
    if not len(x):
        return np.full(2, np.nan), np.full(2, np.nan)
    centroid = np.asarray([x.mean(), y.mean()], dtype=np.float64)
    percentile = 98.0 if object_name == "entry_left" else 2.0
    extreme = np.percentile(x, percentile)
    selected = x >= extreme if object_name == "entry_left" else x <= extreme
    tip = np.asarray([x[selected].mean(), y[selected].mean()], dtype=np.float64)
    return centroid, tip


def _trajectory_metrics(
    gt_masks: np.ndarray,
    correct_masks: np.ndarray,
    object_name: str,
) -> dict[str, Any]:
    ious = []
    centroid_errors = []
    tip_errors = []
    gt_centroids = []
    correct_centroids = []
    gt_tips = []
    correct_tips = []
    for gt_mask, correct_mask in zip(gt_masks, correct_masks):
        union = gt_mask | correct_mask
        ious.append(float((gt_mask & correct_mask).sum() / max(union.sum(), 1)))
        gt_centroid, gt_tip = _mask_point(gt_mask, object_name)
        correct_centroid, correct_tip = _mask_point(correct_mask, object_name)
        gt_centroids.append(gt_centroid)
        correct_centroids.append(correct_centroid)
        gt_tips.append(gt_tip)
        correct_tips.append(correct_tip)
        centroid_errors.append(float(np.linalg.norm(gt_centroid - correct_centroid)))
        tip_errors.append(float(np.linalg.norm(gt_tip - correct_tip)))
    return {
        "mean_mask_iou": float(np.nanmean(ious[1:])),
        "mean_centroid_error_px": float(np.nanmean(centroid_errors[1:])),
        "endpoint_centroid_error_px": centroid_errors[-1],
        "mean_tip_error_px": float(np.nanmean(tip_errors[1:])),
        "endpoint_tip_error_px": tip_errors[-1],
        "gt_centroids": np.asarray(gt_centroids),
        "correct_centroids": np.asarray(correct_centroids),
        "gt_tips": np.asarray(gt_tips),
        "correct_tips": np.asarray(correct_tips),
    }


def _exclusive_tool_rois(
    masks: dict[str, np.ndarray], dilation: int
) -> dict[str, np.ndarray]:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1)
    )
    rois = {
        object_name: np.asarray(
            [
                cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
                for mask in masks[object_name]
            ]
        )
        for object_name in OBJECTS
    }
    for frame_index in range(len(rois[OBJECTS[0]])):
        overlap = rois[OBJECTS[0]][frame_index] & rois[OBJECTS[1]][frame_index]
        if not overlap.any():
            continue
        left_core = masks[OBJECTS[0]][frame_index]
        right_core = masks[OBJECTS[1]][frame_index]
        left_distance = cv2.distanceTransform((~left_core).astype(np.uint8), cv2.DIST_L2, 5)
        right_distance = cv2.distanceTransform(
            (~right_core).astype(np.uint8), cv2.DIST_L2, 5
        )
        rois[OBJECTS[0]][frame_index][
            overlap & (left_distance > right_distance)
        ] = False
        rois[OBJECTS[1]][frame_index][
            overlap & (right_distance >= left_distance)
        ] = False
    return rois


def _masked_l1(left: np.ndarray, right: np.ndarray, masks: np.ndarray) -> float:
    values = []
    for left_frame, right_frame, mask in zip(left, right, masks):
        difference = (
            np.abs(left_frame.astype(np.float32) - right_frame.astype(np.float32))
            / 255.0
        )
        if mask.any():
            values.append(float(difference[mask].mean()))
    return float(np.mean(values)) if values else float("nan")


def _flow(video: np.ndarray) -> np.ndarray:
    resized = np.asarray([cv2.resize(frame, FLOW_SIZE) for frame in video])
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in resized]
    return np.asarray(
        [
            cv2.calcOpticalFlowFarneback(
                left, right, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            for left, right in zip(gray[:-1], gray[1:])
        ]
    )


def _flow_metrics(
    gt_flow: np.ndarray, generated_flow: np.ndarray, masks: np.ndarray
) -> dict[str, float]:
    resized_masks = np.asarray(
        [
            cv2.resize(mask.astype(np.uint8), FLOW_SIZE, interpolation=cv2.INTER_NEAREST)
            > 0
            for mask in masks[1:]
        ]
    )
    gt = gt_flow[resized_masks]
    generated = generated_flow[resized_masks]
    gt_magnitude = np.linalg.norm(gt, axis=-1)
    generated_magnitude = np.linalg.norm(generated, axis=-1)
    positive = gt_magnitude[gt_magnitude > 1e-6]
    threshold = float(np.quantile(positive, 0.25)) if positive.size else 0.0
    active = gt_magnitude > threshold
    if not active.any():
        active = np.ones(gt_magnitude.shape, dtype=bool)
    gt = gt[active]
    generated = generated[active]
    gt_magnitude = gt_magnitude[active]
    generated_magnitude = generated_magnitude[active]
    cosine = np.sum(gt * generated, axis=-1) / np.maximum(
        gt_magnitude * generated_magnitude, 1e-8
    )
    return {
        "flow_cosine": float(np.mean(cosine)),
        "flow_epe": float(np.mean(np.linalg.norm(gt - generated, axis=-1))),
        "flow_magnitude_ratio": float(
            generated_magnitude.mean() / max(gt_magnitude.mean(), 1e-8)
        ),
    }


def _overlay_masks(
    frame: np.ndarray,
    masks: dict[str, np.ndarray],
    tips: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    output = frame.copy()
    for object_name in OBJECTS:
        mask = masks[object_name]
        if not mask.any():
            continue
        color = COLORS[object_name]
        layer = np.zeros_like(output)
        layer[:] = color
        output[mask] = cv2.addWeighted(output[mask], 0.55, layer[mask], 0.45, 0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, color, 2)
        if tips is not None and np.all(np.isfinite(tips[object_name])):
            cv2.circle(output, tuple(np.rint(tips[object_name]).astype(int)), 5, color, -1)
    return output


def _write_tracking_video(
    path: Path,
    gt: np.ndarray,
    correct: np.ndarray,
    gt_masks: dict[str, np.ndarray],
    correct_masks: dict[str, np.ndarray],
    fps: int,
) -> None:
    height, width = gt.shape[1:3]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (2 * width, height + 28),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {path}")
    for frame_index in range(min(len(gt), len(correct))):
        gt_tips = {
            name: _mask_point(gt_masks[name][frame_index], name)[1]
            for name in OBJECTS
        }
        correct_tips = {
            name: _mask_point(correct_masks[name][frame_index], name)[1]
            for name in OBJECTS
        }
        cells = [
            _overlay_masks(
                gt[frame_index],
                {name: gt_masks[name][frame_index] for name in OBJECTS},
                gt_tips,
            ),
            _overlay_masks(
                correct[frame_index],
                {name: correct_masks[name][frame_index] for name in OBJECTS},
                correct_tips,
            ),
        ]
        canvas = np.zeros((height + 28, 2 * width, 3), dtype=np.uint8)
        canvas[:height] = np.concatenate(cells, axis=1)
        cv2.putText(
            canvas,
            "Ground truth masks/tips",
            (8, height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Correct-action prediction masks/tips",
            (width + 8, height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(canvas)
    writer.release()


def _mean(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


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
    manifests = sorted(input_root.glob("raw/*/*_action_intervention_episode.json"))
    if len(manifests) != 10:
        raise RuntimeError(f"Expected 10 episode manifests, found {len(manifests)}")

    mask_paths = {
        (
            manifest.parent.name,
            int(json.loads(manifest.read_text())["episode_id"]),
        ): masks_dir
        / f"{manifest.parent.name}_ep{int(json.loads(manifest.read_text())['episode_id']):05d}.npz"
        for manifest in manifests
    }
    need_tracking = any(
        not (args.reuse_masks and path.exists()) for path in mask_paths.values()
    )
    prompt_audits: dict[str, Any] = {}

    if need_tracking:
        image_model = build_sam2(
            args.sam_config,
            args.sam_checkpoint,
            device=args.device,
        )
        image_predictor = SAM2ImagePredictor(image_model)
        initial_by_episode: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text())
            subset = manifest_path.parent.name
            episode_id = int(manifest["episode_id"])
            if args.reuse_masks and mask_paths[(subset, episode_id)].exists():
                continue
            gt_video = _read_video(
                _resolve_video(manifest_path.parent, episode_id, "ground_truth")
            )
            key = _episode_key(subset, episode_id)
            if key not in prompts["episodes"]:
                raise KeyError(f"Missing reviewed prompts for {key}")
            initial_masks, audit = _initial_masks(
                image_predictor,
                gt_video[0],
                prompts["episodes"][key],
                reference_size,
            )
            initial_by_episode[(subset, episode_id)] = initial_masks
            prompt_audits[key] = audit
        del image_predictor, image_model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        video_predictor = build_sam2_video_predictor(
            args.sam_config,
            args.sam_checkpoint,
            device=args.device,
        )
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text())
            subset = manifest_path.parent.name
            episode_id = int(manifest["episode_id"])
            if args.reuse_masks and mask_paths[(subset, episode_id)].exists():
                print(f"REUSED subset={subset} episode={episode_id}", flush=True)
                continue
            gt_video = _read_video(
                _resolve_video(manifest_path.parent, episode_id, "ground_truth")
            )
            correct_video = _read_video(
                _resolve_video(manifest_path.parent, episode_id, "correct")
            )
            initial_masks = initial_by_episode[(subset, episode_id)]
            gt_masks = _track_video(video_predictor, gt_video, initial_masks)
            correct_masks = _track_video(video_predictor, correct_video, initial_masks)
            np.savez_compressed(
                mask_paths[(subset, episode_id)],
                **{
                    f"gt__{name}": gt_masks[name].astype(np.uint8)
                    for name in OBJECTS
                },
                **{
                    f"correct__{name}": correct_masks[name].astype(np.uint8)
                    for name in OBJECTS
                },
            )
            _write_tracking_video(
                overlays_dir / f"{subset}_ep{episode_id:05d}_tracking.mp4",
                gt_video,
                correct_video,
                gt_masks,
                correct_masks,
                args.fps,
            )
            print(f"TRACKED subset={subset} episode={episode_id}", flush=True)
        del video_predictor
        gc.collect()

    episode_records = []
    variant_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        subset = manifest_path.parent.name
        episode_id = int(manifest["episode_id"])
        gt_video = _read_video(
            _resolve_video(manifest_path.parent, episode_id, "ground_truth")
        )
        correct_video = _read_video(
            _resolve_video(manifest_path.parent, episode_id, "correct")
        )
        with np.load(mask_paths[(subset, episode_id)]) as archive:
            gt_masks = {
                name: archive[f"gt__{name}"].astype(bool) for name in OBJECTS
            }
            correct_masks = {
                name: archive[f"correct__{name}"].astype(bool) for name in OBJECTS
            }
        tracking_overlay = (
            overlays_dir / f"{subset}_ep{episode_id:05d}_tracking.mp4"
        )
        if not tracking_overlay.exists():
            _write_tracking_video(
                tracking_overlay,
                gt_video,
                correct_video,
                gt_masks,
                correct_masks,
                args.fps,
            )
        union_masks = {
            name: gt_masks[name] | correct_masks[name] for name in OBJECTS
        }
        rois = _exclusive_tool_rois(union_masks, args.roi_dilation)
        gt_flow = _flow(gt_video)
        action_archive_path = next(
            manifest_path.parent.glob(f"*ep{episode_id:05d}*.npz")
        )
        with np.load(action_archive_path) as archive:
            action_archive = {key: archive[key] for key in archive.files}

        trajectory = {
            name: _trajectory_metrics(gt_masks[name], correct_masks[name], name)
            for name in OBJECTS
        }
        correct_tool_metrics = {}
        correct_flow = _flow(correct_video)
        for object_name in OBJECTS:
            correct_tool_metrics[object_name] = {
                "l1": _masked_l1(
                    gt_video[1:], correct_video[1:], rois[object_name][1:]
                ),
                **_flow_metrics(gt_flow, correct_flow, rois[object_name]),
                **{
                    key: value
                    for key, value in trajectory[object_name].items()
                    if not isinstance(value, np.ndarray)
                },
                "mean_gt_mask_fraction": float(gt_masks[object_name].mean()),
                "mean_correct_mask_fraction": float(correct_masks[object_name].mean()),
                "mean_roi_fraction": float(rois[object_name].mean()),
            }

        record_by_name = {record["name"]: record for record in manifest["variants"]}
        episode_variant_rows = []
        for variant_name in record_by_name:
            video = (
                correct_video
                if variant_name == "correct"
                else _read_video(
                    _resolve_video(manifest_path.parent, episode_id, variant_name)
                )
            )
            flow = correct_flow if variant_name == "correct" else _flow(video)
            normalized_delta_rms = float(
                np.sqrt(
                    np.mean(
                        (
                            action_archive[f"normalized__{variant_name}"]
                            - action_archive["normalized__correct"]
                        )
                        ** 2
                    )
                )
            )
            row: dict[str, Any] = {
                "subset": subset,
                "episode_id": episode_id,
                "variant": variant_name,
                "normalized_input_delta_rms": normalized_delta_rms,
            }
            match = VARIANT_PATTERN.match(variant_name)
            row["arm"] = match.group(1) if match else None
            row["component"] = match.group(2) if match else None
            row["gain"] = match.group(3) if match else None
            for object_name in OBJECTS:
                l1 = _masked_l1(gt_video[1:], video[1:], rois[object_name][1:])
                paired_delta = _masked_l1(
                    correct_video[1:], video[1:], rois[object_name][1:]
                )
                flow_metrics = _flow_metrics(gt_flow, flow, rois[object_name])
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
            variant_rows.append(row)
            episode_variant_rows.append(row)

        episode_record = {
            "subset": subset,
            "episode_id": episode_id,
            "prompt_audit": prompt_audits.get(_episode_key(subset, episode_id)),
            "correct_tool_metrics": correct_tool_metrics,
            "mask_archive": str(mask_paths[(subset, episode_id)]),
            "tracking_overlay": str(tracking_overlay),
        }
        episode_records.append(episode_record)

        for arm in ("psm1", "psm2"):
            for component in COMPONENTS:
                for gain in GAINS:
                    variant_name = f"{arm}_{component}_{gain}"
                    source = next(
                        row
                        for row in episode_variant_rows
                        if row["variant"] == variant_name
                    )
                    component_rows.append(source.copy())

    for filename, rows in (
        ("per_variant_tool_metrics.csv", variant_rows),
        ("per_component_tool_metrics.csv", component_rows),
    ):
        with (output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    arm_localization = {}
    response_by_arm = {}
    eligible_interventions_by_arm = {}
    for arm in ("psm1", "psm2"):
        records = [
            row
            for row in component_rows
            if row["arm"] == arm and row["normalized_input_delta_rms"] >= 0.02
        ]
        responses = {
            object_name: _mean(records, f"{object_name}_paired_output_delta_l1")
            for object_name in OBJECTS
        }
        response_by_arm[arm] = responses
        eligible_interventions_by_arm[arm] = len(records)

    assignment_scores = {
        "psm1_left__psm2_right": (
            response_by_arm["psm1"]["entry_left"]
            + response_by_arm["psm2"]["entry_right"]
        ),
        "psm1_right__psm2_left": (
            response_by_arm["psm1"]["entry_right"]
            + response_by_arm["psm2"]["entry_left"]
        ),
    }
    if (
        assignment_scores["psm1_right__psm2_left"]
        > assignment_scores["psm1_left__psm2_right"]
    ):
        inferred_mapping = {"psm1": "entry_right", "psm2": "entry_left"}
    else:
        inferred_mapping = {"psm1": "entry_left", "psm2": "entry_right"}
    assignment_margin = float(
        abs(
            assignment_scores["psm1_right__psm2_left"]
            - assignment_scores["psm1_left__psm2_right"]
        )
    )

    for arm in ("psm1", "psm2"):
        responses = response_by_arm[arm]
        intended = inferred_mapping[arm]
        other = next(name for name in OBJECTS if name != intended)
        episode_localization = []
        episode_keys = sorted(
            {
                (row["subset"], row["episode_id"])
                for row in component_rows
                if row["arm"] == arm
            }
        )
        for subset, episode_id in episode_keys:
            records = [
                row
                for row in component_rows
                if row["arm"] == arm
                and row["subset"] == subset
                and row["episode_id"] == episode_id
                and row["normalized_input_delta_rms"] >= 0.02
            ]
            if not records:
                continue
            intended_response = _mean(
                records, f"{intended}_paired_output_delta_l1"
            )
            other_response = _mean(records, f"{other}_paired_output_delta_l1")
            episode_localization.append(
                {
                    "subset": subset,
                    "episode_id": int(episode_id),
                    "eligible_interventions": len(records),
                    "intended_response_l1": intended_response,
                    "other_response_l1": other_response,
                    "intended_response_fraction": float(
                        intended_response
                        / max(intended_response + other_response, 1e-8)
                    ),
                    "response_margin_l1": intended_response - other_response,
                }
            )
        episode_margins = np.asarray(
            [item["response_margin_l1"] for item in episode_localization]
        )
        generator = np.random.default_rng(20260727 + int(arm[-1]))
        bootstrap_means = generator.choice(
            episode_margins,
            size=(20_000, len(episode_margins)),
            replace=True,
        ).mean(axis=1)
        arm_localization[arm] = {
            "eligible_interventions": eligible_interventions_by_arm[arm],
            "mean_response_l1": responses,
            "inferred_intended_tool": intended,
            "independent_max_response_tool": max(responses, key=responses.get),
            "intended_response_fraction": float(
                responses[intended] / max(sum(responses.values()), 1e-8)
            ),
            "episodes_intended_response_dominant": int(
                np.count_nonzero(episode_margins > 0)
            ),
            "episodes_total": len(episode_localization),
            "mean_episode_response_fraction": float(
                np.mean(
                    [
                        item["intended_response_fraction"]
                        for item in episode_localization
                    ]
                )
            ),
            "mean_episode_response_margin_l1": float(episode_margins.mean()),
            "episode_bootstrap_95_ci_response_margin_l1": [
                float(value)
                for value in np.quantile(bootstrap_means, [0.025, 0.975])
            ],
            "episode_localization": episode_localization,
        }

    aggregate_components = []
    for arm in ("psm1", "psm2"):
        intended = inferred_mapping[arm]
        other = next(name for name in OBJECTS if name != intended)
        for component in COMPONENTS:
            for gain in GAINS:
                records = [
                    row
                    for row in component_rows
                    if row["arm"] == arm
                    and row["component"] == component
                    and row["gain"] == gain
                ]
                intended_response = _mean(
                    records, f"{intended}_paired_output_delta_l1"
                )
                other_response = _mean(records, f"{other}_paired_output_delta_l1")
                aggregate_components.append(
                    {
                        "arm": arm,
                        "component": component,
                        "gain": gain,
                        "intended_tool": intended,
                        "other_tool": other,
                        "mean_normalized_input_delta_rms": _mean(
                            records, "normalized_input_delta_rms"
                        ),
                        "intended_paired_output_delta_l1": intended_response,
                        "other_paired_output_delta_l1": other_response,
                        "intended_response_fraction": float(
                            intended_response
                            / max(intended_response + other_response, 1e-8)
                        ),
                        "intended_l1_delta_vs_correct": _mean(
                            records, f"{intended}_l1_delta_vs_correct"
                        ),
                        "intended_flow_cosine_delta_vs_correct": _mean(
                            records, f"{intended}_flow_cosine_delta_vs_correct"
                        ),
                        "intended_flow_magnitude_ratio_delta_vs_correct": _mean(
                            records,
                            f"{intended}_flow_magnitude_ratio_delta_vs_correct",
                        ),
                    }
                )

    response_matrix = np.asarray(
        [
            [
                arm_localization[arm]["mean_response_l1"][object_name]
                for object_name in OBJECTS
            ]
            for arm in ("psm1", "psm2")
        ]
    )
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    image = axis.imshow(response_matrix, cmap="magma")
    axis.set_xticks(range(2), ["Screen-entry left", "Screen-entry right"])
    axis.set_yticks(range(2), ["PSM1 interventions", "PSM2 interventions"])
    axis.set_title("Mean tool-region output response")
    for row_index in range(2):
        for column_index in range(2):
            axis.text(
                column_index,
                row_index,
                f"{response_matrix[row_index, column_index]:.4f}",
                ha="center",
                va="center",
                color="white"
                if response_matrix[row_index, column_index]
                > response_matrix.max() / 2
                else "black",
            )
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(output_dir / "arm_to_tool_response_matrix.png", dpi=180)
    plt.close(figure)

    payload = {
        "diagnostic": "SAM2 tool-specific physical-axis action response",
        "model": "C3-H-S-S CAMP-lite H16 iter 700",
        "episodes": len(episode_records),
        "sam": {
            "checkpoint": str(Path(args.sam_checkpoint)),
            "checkpoint_sha256": _file_sha256(Path(args.sam_checkpoint)),
            "config": args.sam_config,
            "code_sha": args.sam_code_sha,
            "device": args.device,
            "prompt_file": str(Path(args.prompts)),
        },
        "roi_dilation_pixels": args.roi_dilation,
        "arm_localization": arm_localization,
        "inferred_arm_to_tool_mapping": inferred_mapping,
        "arm_to_tool_assignment": {
            "method": "maximum-response one-to-one assignment",
            "scores": assignment_scores,
            "winning_margin_l1": assignment_margin,
        },
        "correct_tool_metrics": {
            object_name: {
                key: float(
                    np.mean(
                        [
                            record["correct_tool_metrics"][object_name][key]
                            for record in episode_records
                        ]
                    )
                )
                for key in (
                    "l1",
                    "flow_cosine",
                    "flow_magnitude_ratio",
                    "mean_mask_iou",
                    "mean_centroid_error_px",
                    "mean_tip_error_px",
                )
            }
            for object_name in OBJECTS
        },
        "aggregate_components": aggregate_components,
        "episodes_detail": episode_records,
    }
    _write_json(output_dir / "tool_specific_summary.json", payload)

    lines = [
        "# C3-H-S-S tool-specific action response",
        "",
        "SAM2 tracks the screen-entry-left and screen-entry-right instruments in",
        "the ground truth and correct-action prediction. Exclusive dilated unions",
        "of those tracks define the evaluation regions for every intervention.",
        "",
        "## Arm-to-tool localization",
        "",
        "The mapping is the maximum-response one-to-one assignment: each robot",
        "arm must map to a distinct screen-entry tool. The response fraction then",
        "measures localization to that assigned tool; 0.5 is unlocalized.",
        "",
        "| Action arm | Inferred tool | Response fraction | Episodes localized | Episode-margin 95% CI |",
        "|---|---|---:|---:|---:|",
    ]
    for arm in ("psm1", "psm2"):
        item = arm_localization[arm]
        confidence = item["episode_bootstrap_95_ci_response_margin_l1"]
        lines.append(
            f"| {arm.upper()} | {item['inferred_intended_tool']} | "
            f"{item['intended_response_fraction']:.3f} | "
            f"{item['episodes_intended_response_dominant']}/"
            f"{item['episodes_total']} | "
            f"[{confidence[0]:+.4f}, {confidence[1]:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Correct-action trajectory quality",
            "",
            "| Tool | RGB L1 ↓ | Flow cosine ↑ | Flow magnitude / GT | Mask IoU ↑ | Tip error px ↓ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for object_name in OBJECTS:
        item = payload["correct_tool_metrics"][object_name]
        lines.append(
            f"| {object_name} | {item['l1']:.4f} | {item['flow_cosine']:.3f} | "
            f"{item['flow_magnitude_ratio']:.3f} | {item['mean_mask_iou']:.3f} | "
            f"{item['mean_tip_error_px']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Component counterfactuals",
            "",
            "| Arm/component | Gain | Intended fraction | Intended output Δ | Other output Δ | Intended accuracy Δ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in aggregate_components:
        lines.append(
            f"| {item['arm']} {item['component']} | {item['gain']} | "
            f"{item['intended_response_fraction']:.3f} | "
            f"{item['intended_paired_output_delta_l1']:.4f} | "
            f"{item['other_paired_output_delta_l1']:.4f} | "
            f"{item['intended_l1_delta_vs_correct']:+.4f} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
