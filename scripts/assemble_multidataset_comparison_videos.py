#!/usr/bin/env python3
"""Assemble GT | C-H-S-S | C3-H-S-S videos for multiple datasets and seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapy
import numpy as np


DEFAULT_SUBSETS = ["cholecystectomy", "hysterectomy", "inguinal_hernia", "prostatectomy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--chss-root", type=Path, default=None)
    parser.add_argument("--c3hss-root", type=Path, default=None)
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--c3-label", default="C3-H-S-S")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse already completed comparison videos and encode only missing/corrupt outputs.",
    )
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
    root = args.run_dir.resolve()
    chss_root = args.chss_root.resolve() if args.chss_root else root / "chss"
    c3hss_root = args.c3hss_root.resolve() if args.c3hss_root else root / "c3hss"
    comparison = root / "comparison"
    manifest: dict[str, object] = {
        "layout": ["Ground truth", "C-H-S-S", args.c3_label],
        "subsets": args.subsets,
        "fps": args.fps,
        "videos": {},
    }

    for subset in args.subsets:
        manifest["videos"][subset] = {}
        for seed in args.seeds:
            chss = load_results(chss_root / subset / f"seed{seed}" / "chss_results.json")
            c3hss = load_results(c3hss_root / subset / f"seed{seed}" / "c3hss_results.json")
            if sorted(chss) != sorted(c3hss):
                raise RuntimeError(f"{subset}/seed{seed} has mismatched episodes")
            output_dir = comparison / subset / f"seed{seed}"
            output_dir.mkdir(parents=True, exist_ok=True)
            records = []
            for episode_id in sorted(chss):
                output = output_dir / f"{subset}_ep{episode_id:05d}_seed{seed}_comparison.mp4"
                if args.resume_existing and output.is_file() and output.stat().st_size > 1024:
                    try:
                        existing_frames = len(mediapy.read_video(output))
                    except Exception:
                        existing_frames = 0
                    if existing_frames >= 200:
                        records.append(
                            {"episode_id": episode_id, "frames": existing_frames, "video": str(output)}
                        )
                        print(f"{output} (reused)", flush=True)
                        continue
                ch_entry = chss[episode_id]
                c3_entry = c3hss[episode_id]
                gt = np.asarray(mediapy.read_video(ch_entry["ground_truth_video"]), dtype=np.uint8)
                ch_video = np.asarray(mediapy.read_video(ch_entry["generated_video"]), dtype=np.uint8)
                c3_video = np.asarray(mediapy.read_video(c3_entry["generated_video"]), dtype=np.uint8)
                n = min(len(gt), len(ch_video), len(c3_video))
                if n < 200:
                    raise RuntimeError(f"{subset}/seed{seed}/episode{episode_id} has only {n} frames")
                height, width = gt.shape[1:3]
                triptych = np.concatenate(
                    [
                        label(gt[:n], "Ground truth"),
                        label(resize(ch_video[:n], height, width), "C-H-S-S"),
                        label(resize(c3_video[:n], height, width), args.c3_label),
                    ],
                    axis=2,
                )
                mediapy.write_video(output, triptych, fps=args.fps)
                records.append({"episode_id": episode_id, "frames": n, "video": str(output)})
                print(output, flush=True)
            manifest["videos"][subset][str(seed)] = records

    comparison.mkdir(parents=True, exist_ok=True)
    (comparison / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
