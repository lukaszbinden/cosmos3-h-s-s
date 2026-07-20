#!/usr/bin/env python3
"""Validate matched results, create triptych videos, and plot long-horizon FDS."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapy
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--chss-json", required=True)
    p.add_argument("--c3hss-json", required=True)
    p.add_argument("--baseline-c3hss-json", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args()


def _resize_video(video: np.ndarray, height: int, width: int) -> np.ndarray:
    if video.shape[1:3] == (height, width):
        return video
    return np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in video])


def _label(video: np.ndarray, text: str) -> np.ndarray:
    result = video.copy()
    for frame in result:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(frame, text, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def _mean_ci(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def _boundary_discontinuity(video: np.ndarray) -> dict[str, float]:
    """Compare frame-to-frame pixel motion at autoregressive seams vs interiors."""
    delta = np.mean(np.abs(video[1:].astype(np.float32) - video[:-1].astype(np.float32)), axis=(1, 2, 3))
    boundary_idx = np.asarray([i for i in range(12, len(delta), 12)], dtype=np.int64)
    interior_mask = np.ones(len(delta), dtype=bool)
    interior_mask[boundary_idx] = False
    boundary_mean = float(delta[boundary_idx].mean())
    interior_mean = float(delta[interior_mask].mean())
    return {
        "boundary_mean_abs_uint8": boundary_mean,
        "interior_mean_abs_uint8": interior_mean,
        "boundary_to_interior_ratio": boundary_mean / interior_mean,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "comparison_videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    chss = json.loads(Path(args.chss_json).read_text())
    c3hss = json.loads(Path(args.c3hss_json).read_text())
    ch_by_ep = {entry["episode_id"]: entry for entry in chss["results"]}
    c3_by_ep = {entry["episode_id"]: entry for entry in c3hss["results"]}
    episodes = sorted(ch_by_ep)
    if episodes != sorted(c3_by_ep) or len(episodes) < 7:
        raise RuntimeError(f"Episode mismatch or fewer than seven episodes: CHSS={episodes}, C3HSS={sorted(c3_by_ep)}")

    seam_metrics = {"chss": {}, "c3hss": {}}
    for episode_id in episodes:
        ch = ch_by_ep[episode_id]
        c3 = c3_by_ep[episode_id]
        if ch["num_frames"] < 100 or c3["num_frames"] < 100:
            raise RuntimeError(f"Episode {episode_id} has a rollout shorter than 100 frames")
        gt = np.asarray(mediapy.read_video(ch["ground_truth_video"]), dtype=np.uint8)
        ch_video = np.asarray(mediapy.read_video(ch["generated_video"]), dtype=np.uint8)
        c3_video = np.asarray(mediapy.read_video(c3["generated_video"]), dtype=np.uint8)
        n = min(len(gt), len(ch_video), len(c3_video))
        height, width = gt.shape[1:3]
        triptych = np.concatenate(
            [
                _label(gt[:n], "Ground truth"),
                _label(_resize_video(ch_video[:n], height, width), "C-H-S-S"),
                _label(_resize_video(c3_video[:n], height, width), "C3-H-S-S"),
            ],
            axis=2,
        )
        mediapy.write_video(videos_dir / f"prostatectomy_ep{episode_id:05d}_comparison.mp4", triptych, fps=args.fps)
        seam_metrics["chss"][str(episode_id)] = _boundary_discontinuity(ch_video[:n])
        seam_metrics["c3hss"][str(episode_id)] = _boundary_discontinuity(c3_video[:n])

    for model_metrics in seam_metrics.values():
        model_metrics["aggregate"] = {
            key: float(np.mean([entry[key] for name, entry in model_metrics.items() if name != "aggregate"]))
            for key in ("boundary_mean_abs_uint8", "interior_mean_abs_uint8", "boundary_to_interior_ratio")
        }
    if args.baseline_c3hss_json:
        baseline = json.loads(Path(args.baseline_c3hss_json).read_text())
        baseline_metrics = {}
        for entry in baseline["results"]:
            video = np.asarray(mediapy.read_video(entry["generated_video"]), dtype=np.uint8)
            baseline_metrics[str(entry["episode_id"])] = _boundary_discontinuity(video)
        baseline_metrics["aggregate"] = {
            key: float(np.mean([entry[key] for name, entry in baseline_metrics.items() if name != "aggregate"]))
            for key in ("boundary_mean_abs_uint8", "interior_mean_abs_uint8", "boundary_to_interior_ratio")
        }
        seam_metrics["c3hss_pre_fix"] = baseline_metrics
    (output_dir / "boundary_discontinuity.json").write_text(json.dumps(seam_metrics, indent=2))

    ch_curves = np.asarray([ch_by_ep[ep]["fds"]["l1_per_frame"] for ep in episodes], dtype=np.float64)
    c3_curves = np.asarray([c3_by_ep[ep]["fds"]["l1_per_frame"] for ep in episodes], dtype=np.float64)
    horizon = min(ch_curves.shape[1], c3_curves.shape[1])
    ch_curves, c3_curves = ch_curves[:, :horizon], c3_curves[:, :horizon]
    x = np.arange(1, horizon + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for curves, label, color in ((ch_curves, "C-H-S-S", "#1f77b4"), (c3_curves, "C3-H-S-S", "#d62728")):
        mean = curves.mean(axis=0)
        ci = 1.96 * curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])
        ax.plot(x, mean, label=label, color=color, linewidth=2)
        ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.18, linewidth=0)
    for boundary in range(12, horizon, 12):
        ax.axvline(boundary, color="0.75", linewidth=0.7, linestyle="--")
    ax.set(title="Long-horizon Frame Decay Score (mean L1; 95% CI)", xlabel="Predicted frame", ylabel="L1 error in [-1, 1] space")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fds_comparison.png", dpi=180)
    plt.close(fig)

    rows = []
    for ep in episodes:
        rows.append(
            {
                "episode_id": ep,
                "chss_mean_l1": ch_by_ep[ep]["fds"]["mean_l1"],
                "chss_mean_ssim": ch_by_ep[ep]["fds"]["mean_ssim"],
                "chss_l1_slope": ch_by_ep[ep]["fds"]["l1_slope"],
                "c3hss_mean_l1": c3_by_ep[ep]["fds"]["mean_l1"],
                "c3hss_mean_ssim": c3_by_ep[ep]["fds"]["mean_ssim"],
                "c3hss_l1_slope": c3_by_ep[ep]["fds"]["l1_slope"],
            }
        )
    with (output_dir / "fds_per_episode.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ch_mean, ch_ci = _mean_ci([row["chss_mean_l1"] for row in rows])
    c3_mean, c3_ci = _mean_ci([row["c3hss_mean_l1"] for row in rows])
    summary = {
        "episodes": episodes,
        "num_episodes": len(episodes),
        "predicted_frames": horizon,
        "chss_mean_l1": ch_mean,
        "chss_mean_l1_95ci": ch_ci,
        "c3hss_mean_l1": c3_mean,
        "c3hss_mean_l1_95ci": c3_ci,
        "lower_mean_l1_model": "C-H-S-S" if ch_mean < c3_mean else "C3-H-S-S",
    }
    (output_dir / "fds_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
