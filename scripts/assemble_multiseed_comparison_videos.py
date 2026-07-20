#!/usr/bin/env python3
"""Assemble labeled GT | C-H-S-S | C3-H-S-S videos from saved rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapy
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def resize(video: np.ndarray, height: int, width: int) -> np.ndarray:
    if video.shape[1:3] == (height, width):
        return video
    return np.stack(
        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in video]
    )


def label(video: np.ndarray, text: str) -> np.ndarray:
    result = video.copy()
    for frame in result:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (14, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return result


def load_results(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text())
    return {int(entry["episode_id"]): entry for entry in payload["results"]}


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    comparison_dir = run_dir / "comparison"
    manifest: dict[str, object] = {
        "layout": ["Ground truth", "C-H-S-S", "C3-H-S-S (corrected rollout)"],
        "fps": args.fps,
        "seeds": {},
    }

    for seed in args.seeds:
        chss = load_results(run_dir / "chss" / f"seed{seed}" / "chss_results.json")
        c3hss = load_results(run_dir / "c3hss" / f"seed{seed}" / "c3hss_results.json")
        if sorted(chss) != sorted(c3hss):
            raise RuntimeError(f"Seed {seed} has mismatched episodes")

        seed_dir = comparison_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_outputs = []
        for episode_id in sorted(chss):
            ch_entry = chss[episode_id]
            c3_entry = c3hss[episode_id]
            gt = np.asarray(mediapy.read_video(ch_entry["ground_truth_video"]), dtype=np.uint8)
            ch_video = np.asarray(mediapy.read_video(ch_entry["generated_video"]), dtype=np.uint8)
            c3_video = np.asarray(mediapy.read_video(c3_entry["generated_video"]), dtype=np.uint8)
            n = min(len(gt), len(ch_video), len(c3_video))
            if n < 100:
                raise RuntimeError(f"Seed {seed}, episode {episode_id} has only {n} frames")
            height, width = gt.shape[1:3]
            triptych = np.concatenate(
                [
                    label(gt[:n], "Ground truth"),
                    label(resize(ch_video[:n], height, width), "C-H-S-S"),
                    label(resize(c3_video[:n], height, width), "C3-H-S-S"),
                ],
                axis=2,
            )
            output = seed_dir / f"prostatectomy_ep{episode_id:05d}_seed{seed}_comparison.mp4"
            mediapy.write_video(output, triptych, fps=args.fps)
            seed_outputs.append({"episode_id": episode_id, "frames": n, "video": str(output)})
            print(output, flush=True)
        manifest["seeds"][str(seed)] = seed_outputs

    comparison_dir.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
