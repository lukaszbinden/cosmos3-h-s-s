#!/usr/bin/env python3
"""Compare pre/post-fix C3 rollout seams and render a boundary contact sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mediapy
import numpy as np


def stats(video: np.ndarray) -> dict[str, float]:
    delta = np.mean(np.abs(video[1:].astype(np.float32) - video[:-1].astype(np.float32)), axis=(1, 2, 3))
    idx = np.asarray([i for i in range(12, len(delta), 12)], dtype=np.int64)
    keep = np.ones(len(delta), dtype=bool)
    keep[idx] = False
    boundary = float(delta[idx].mean())
    interior = float(delta[keep].mean())
    return {
        "boundary_mean_abs_uint8": boundary,
        "interior_mean_abs_uint8": interior,
        "boundary_to_interior_ratio": boundary / interior,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pre-json", required=True)
    p.add_argument("--post-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--contact-sheet", required=True)
    args = p.parse_args()
    payloads = {
        "pre_fix": json.loads(Path(args.pre_json).read_text()),
        "post_fix": json.loads(Path(args.post_json).read_text()),
    }
    output = {}
    videos = {}
    for label, payload in payloads.items():
        per_episode = {}
        for entry in payload["results"]:
            video = np.asarray(mediapy.read_video(entry["generated_video"]), dtype=np.uint8)
            per_episode[str(entry["episode_id"])] = stats(video)
            if entry["episode_id"] == payload["episodes"][0]:
                videos[label] = video
        aggregate = {
            key: float(np.mean([entry[key] for entry in per_episode.values()]))
            for key in ("boundary_mean_abs_uint8", "interior_mean_abs_uint8", "boundary_to_interior_ratio")
        }
        output[label] = {"aggregate": aggregate, "per_episode": per_episode}
    (Path(args.output_json)).write_text(json.dumps(output, indent=2))

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for row, (label, video) in enumerate(videos.items()):
        for col, frame_idx in enumerate((11, 12, 13, 14)):
            axes[row, col].imshow(video[frame_idx])
            axes[row, col].set_title(f"{label.replace('_', ' ')}: frame {frame_idx}")
            axes[row, col].axis("off")
    fig.suptitle("C3-H-S-S frames around first autoregressive seam (12 → 13)")
    fig.tight_layout()
    fig.savefig(args.contact_sheet, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
