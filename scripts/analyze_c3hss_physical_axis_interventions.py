#!/usr/bin/env python3
"""Analyze and visualize the C3-H-S-S physical per-axis diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

VARIANT_PATTERN = re.compile(
    r"^(psm[12])_(t[xyz]|r[xyz]|jaw)_(0x|1p5x)$"
)
COMPONENTS = ("tx", "ty", "tz", "rx", "ry", "rz", "jaw")
GAINS = ("0x", "1p5x")
FLOW_SIZE = (480, 272)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n")


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


def _flow(video: np.ndarray) -> np.ndarray:
    resized = np.asarray([cv2.resize(frame, FLOW_SIZE) for frame in video])
    grayscale = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in resized]
    return np.asarray(
        [
            cv2.calcOpticalFlowFarneback(
                left,
                right,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            for left, right in zip(grayscale[:-1], grayscale[1:])
        ]
    )


def _flow_metrics(
    gt_flow: np.ndarray, generated_flow: np.ndarray, full_resolution_mask: np.ndarray
) -> dict[str, float]:
    mask = cv2.resize(
        full_resolution_mask.astype(np.uint8), FLOW_SIZE, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    gt = gt_flow[:, mask]
    generated = generated_flow[:, mask]
    gt_magnitude = np.linalg.norm(gt, axis=-1)
    generated_magnitude = np.linalg.norm(generated, axis=-1)
    positive = gt_magnitude[gt_magnitude > 1e-6]
    threshold = float(np.quantile(positive, 0.25)) if positive.size else 0.0
    active = gt_magnitude > threshold
    if not np.any(active):
        active = np.ones(gt_magnitude.shape, dtype=bool)
    gt = gt[active]
    generated = generated[active]
    gt_magnitude = gt_magnitude[active]
    generated_magnitude = generated_magnitude[active]
    point_cosine = np.sum(gt * generated, axis=-1) / np.maximum(
        gt_magnitude * generated_magnitude, 1e-8
    )
    return {
        "motion_roi_flow_cosine": float(np.mean(point_cosine)),
        "motion_roi_flow_epe": float(np.mean(np.linalg.norm(gt - generated, axis=-1))),
        "motion_roi_flow_magnitude_ratio": float(
            generated_magnitude.mean() / max(gt_magnitude.mean(), 1e-8)
        ),
        "active_flow_fraction": float(active.mean()),
        "active_flow_threshold": threshold,
    }


def _resolve_video(directory: Path, episode_id: int, suffix: str) -> Path:
    matches = list(directory.glob(f"*ep{episode_id:05d}_seed*_{suffix}.mp4"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one episode {episode_id} video ending in {suffix}, got {matches}"
        )
    return matches[0]


def _component_excitation(
    archive: dict[str, np.ndarray], arm: str, component: str
) -> tuple[float, str]:
    correct = archive["physical__correct"].astype(np.float64)
    offset = 0 if arm == "psm1" else 10
    if component.startswith("t"):
        values = correct[:, offset + "xyz".index(component[1])]
        return float(np.sqrt(np.mean(values**2)) * 1000.0), "mm_rms"
    if component.startswith("r"):
        rows = correct[:, offset + 3 : offset + 9].reshape(-1, 2, 3)
        row1 = rows[:, 0]
        row1 /= np.maximum(np.linalg.norm(row1, axis=-1, keepdims=True), 1e-12)
        row2 = rows[:, 1] - np.sum(rows[:, 1] * row1, axis=-1, keepdims=True) * row1
        row2 /= np.maximum(np.linalg.norm(row2, axis=-1, keepdims=True), 1e-12)
        matrices = np.stack([row1, row2, np.cross(row1, row2)], axis=1)
        rotation_vectors, _ = cv2.Rodrigues(matrices[0])
        # cv2.Rodrigues is scalar-only; convert the remaining matrices in a
        # short loop because each diagnostic sample has only 12 rows.
        vectors = [rotation_vectors.reshape(3)]
        vectors.extend(cv2.Rodrigues(matrix)[0].reshape(3) for matrix in matrices[1:])
        values = np.asarray(vectors)[:, "xyz".index(component[1])]
        return float(np.sqrt(np.mean(values**2)) * 180.0 / np.pi), "degrees_rms"
    values = correct[:, offset + 9] - correct[0, offset + 9]
    return float(np.sqrt(np.mean(values**2))), "jaw_units_rms_from_first"


def _panel(
    videos: list[np.ndarray],
    labels: list[str],
    output_path: Path,
    fps: int,
    panel_width: int = 240,
) -> None:
    frames = min(len(video) for video in videos)
    source_height, source_width = videos[0].shape[1:3]
    panel_height = round(source_height * panel_width / source_width)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (panel_width * len(videos), panel_height + 28),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output_path}")
    for frame_index in range(frames):
        cells = []
        for video, label in zip(videos, labels):
            frame = cv2.resize(video[frame_index], (panel_width, panel_height))
            cell = np.zeros((panel_height + 28, panel_width, 3), dtype=np.uint8)
            cell[:panel_height] = frame
            cv2.putText(
                cell,
                label,
                (8, panel_height + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cells.append(cell)
        writer.write(np.concatenate(cells, axis=1))
    writer.release()


def _mean(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    manifests = sorted(input_root.glob("raw/*/*_action_intervention_episode.json"))
    if not manifests:
        raise RuntimeError(f"No per-episode manifests found below {input_root}")

    per_variant: list[dict[str, Any]] = []
    per_component: list[dict[str, Any]] = []
    episode_count = 0
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        if "physical_action_audit" not in manifest:
            raise RuntimeError(f"{manifest_path} is not a physical-axis result")
        episode_count += 1
        episode_id = int(manifest["episode_id"])
        subset = manifest_path.parent.name
        gt_path = _resolve_video(manifest_path.parent, episode_id, "ground_truth")
        gt_video = _read_video(gt_path)
        gt_flow = _flow(gt_video)
        archive_path = next(manifest_path.parent.glob(f"*ep{episode_id:05d}*.npz"))
        with np.load(archive_path) as payload:
            archive = {key: payload[key] for key in payload.files}
        motion_mask = archive["motion_roi"].astype(bool)

        videos: dict[str, np.ndarray] = {"ground_truth": gt_video}
        record_by_name = {record["name"]: record for record in manifest["variants"]}
        for variant_name, record in record_by_name.items():
            video_path = _resolve_video(manifest_path.parent, episode_id, variant_name)
            video = _read_video(video_path)
            videos[variant_name] = video
            flow_metrics = _flow_metrics(gt_flow, _flow(video), motion_mask)
            row = {
                "subset": subset,
                "episode_id": episode_id,
                "variant": variant_name,
                "motion_roi_l1": record["motion_roi_l1"],
                "motion_roi_endpoint_l1": record["motion_roi_endpoint_l1"],
                "paired_output_delta_from_correct_motion_roi_l1": record[
                    "paired_output_delta_from_correct_motion_roi_l1"
                ],
                "normalized_input_delta_rms": float(
                    np.sqrt(
                        np.mean(
                            (
                                archive[f"normalized__{variant_name}"]
                                - archive["normalized__correct"]
                            )
                            ** 2
                        )
                    )
                ),
                **flow_metrics,
            }
            per_variant.append(row)

        for arm in ("psm1", "psm2"):
            for component in COMPONENTS:
                excitation, excitation_unit = _component_excitation(archive, arm, component)
                correct_record = next(
                    record
                    for record in per_variant
                    if record["subset"] == subset
                    and record["episode_id"] == episode_id
                    and record["variant"] == "correct"
                )
                row = {
                    "subset": subset,
                    "episode_id": episode_id,
                    "arm": arm,
                    "component": component,
                    "physical_excitation": excitation,
                    "physical_excitation_unit": excitation_unit,
                }
                for gain in GAINS:
                    variant = f"{arm}_{component}_{gain}"
                    variant_record = next(
                        record
                        for record in per_variant
                        if record["subset"] == subset
                        and record["episode_id"] == episode_id
                        and record["variant"] == variant
                    )
                    for key in (
                        "motion_roi_l1",
                        "motion_roi_endpoint_l1",
                        "motion_roi_flow_cosine",
                        "motion_roi_flow_epe",
                        "motion_roi_flow_magnitude_ratio",
                        "paired_output_delta_from_correct_motion_roi_l1",
                        "normalized_input_delta_rms",
                    ):
                        row[f"{gain}_{key}"] = variant_record[key]
                        if key in (
                            "motion_roi_l1",
                            "motion_roi_endpoint_l1",
                            "motion_roi_flow_cosine",
                            "motion_roi_flow_epe",
                            "motion_roi_flow_magnitude_ratio",
                        ):
                            row[f"{gain}_{key}_delta_vs_correct"] = (
                                variant_record[key] - correct_record[key]
                            )
                per_component.append(row)

            for group_name, components in (
                ("translation", COMPONENTS[:3]),
                ("rotation", COMPONENTS[3:6]),
            ):
                group_variants = [
                    f"{arm}_{component}_{gain}"
                    for component in components
                    for gain in GAINS
                ]
                _panel(
                    [videos["ground_truth"], videos["correct"]]
                    + [videos[name] for name in group_variants],
                    ["Ground truth", "Correct"]
                    + [name.replace("_", " ") for name in group_variants],
                    panel_dir / f"{subset}_ep{episode_id:05d}_{arm}_{group_name}.mp4",
                    args.fps,
                    panel_width=240,
                )
        jaw_variants = [
            f"{arm}_jaw_{gain}" for arm in ("psm1", "psm2") for gain in GAINS
        ]
        _panel(
            [videos["ground_truth"], videos["correct"]]
            + [videos[name] for name in jaw_variants],
            ["Ground truth", "Correct"]
            + [name.replace("_", " ") for name in jaw_variants],
            panel_dir / f"{subset}_ep{episode_id:05d}_jaw.mp4",
            args.fps,
            panel_width=320,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("per_variant_metrics.csv", per_variant),
        ("per_component_metrics.csv", per_component),
    ):
        with (output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    aggregate = []
    for arm in ("psm1", "psm2"):
        for component in COMPONENTS:
            records = [
                row
                for row in per_component
                if row["arm"] == arm and row["component"] == component
            ]
            row: dict[str, Any] = {
                "arm": arm,
                "component": component,
                "episodes": len(records),
                "mean_physical_excitation": _mean(records, "physical_excitation"),
                "physical_excitation_unit": records[0]["physical_excitation_unit"],
            }
            for gain in GAINS:
                for key in (
                    "motion_roi_l1_delta_vs_correct",
                    "motion_roi_flow_cosine_delta_vs_correct",
                    "motion_roi_flow_magnitude_ratio_delta_vs_correct",
                    "paired_output_delta_from_correct_motion_roi_l1",
                    "normalized_input_delta_rms",
                ):
                    row[f"{gain}_{key}"] = _mean(records, f"{gain}_{key}")
            aggregate.append(row)

    output_delta = np.asarray(
        [
            [
                next(
                    row
                    for row in aggregate
                    if row["arm"] == arm and row["component"] == component
                )[f"{gain}_paired_output_delta_from_correct_motion_roi_l1"]
                for component in COMPONENTS
            ]
            for arm in ("psm1", "psm2")
            for gain in GAINS
        ]
    )
    figure, axis = plt.subplots(figsize=(11, 4.5))
    image = axis.imshow(output_delta, cmap="magma")
    axis.set_xticks(range(len(COMPONENTS)), COMPONENTS)
    axis.set_yticks(
        range(4),
        [f"{arm} {gain}" for arm in ("PSM1", "PSM2") for gain in GAINS],
    )
    axis.set_title("Generated motion-region change from correct output (RGB L1)")
    for row_index in range(output_delta.shape[0]):
        for column_index in range(output_delta.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{output_delta[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                color="white" if output_delta[row_index, column_index] > output_delta.max() / 2 else "black",
                fontsize=9,
            )
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(output_dir / "component_output_sensitivity.png", dpi=180)
    plt.close(figure)

    ranked = sorted(
        aggregate,
        key=lambda row: row["0x_paired_output_delta_from_correct_motion_roi_l1"],
        reverse=True,
    )
    payload = {
        "diagnostic": "physical per-axis action interventions",
        "episodes": episode_count,
        "flow_estimator": (
            "Farneback at 480x272; top 75% of positive GT-flow magnitudes inside "
            "the evaluator's GT temporal-motion ROI"
        ),
        "aggregate": aggregate,
        "ranked_by_zeroing_output_effect": ranked,
    }
    _write_json(output_dir / "physical_axis_summary.json", payload)

    lines = [
        "# C3-H-S-S physical per-axis action diagnostic",
        "",
        f"Analyzed {episode_count} matched episodes. Each row changes one physical",
        "component while holding image, history, caption, checkpoint, sampler, and",
        "seed fixed.",
        "",
        "## Strongest output effects when zeroed",
        "",
        "| Component | Output Δ L1 | Accuracy Δ L1 | Flow-cosine Δ | Excitation |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['arm']} {row['component']} | "
            f"{row['0x_paired_output_delta_from_correct_motion_roi_l1']:.4f} | "
            f"{row['0x_motion_roi_l1_delta_vs_correct']:+.4f} | "
            f"{row['0x_motion_roi_flow_cosine_delta_vs_correct']:+.3f} | "
            f"{row['mean_physical_excitation']:.3f} {row['physical_excitation_unit']} |"
        )
    lines.extend(
        [
            "",
            "Positive accuracy Δ L1 and negative flow-cosine Δ mean the intervention",
            "made the prediction worse than the correct action. Small output changes",
            "are only evidence of insensitivity when that component was materially",
            "excited in the selected episodes.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
