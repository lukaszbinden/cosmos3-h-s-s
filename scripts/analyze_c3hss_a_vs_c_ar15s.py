#!/usr/bin/env python3
"""Analyze and render the matched Arm-A-versus-Arm-C 15-second sweep."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any


CONDITIONS = ("arm_a_h0", "arm_c_real")
LOWER_IS_BETTER = {"mean_l1", "endpoint_l1", "late_l1", "l1_drift"}
METRICS = ("mean_l1", "mean_ssim", "endpoint_l1", "endpoint_ssim", "late_l1", "l1_drift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--skip-videos", action="store_true")
    return parser.parse_args()


def _metrics(result: dict[str, Any]) -> dict[str, float]:
    fds = result["fds"]
    l1 = [float(value) for value in fds["l1_per_frame"]]
    ssim = [float(value) for value in fds["ssim_per_frame"]]
    if len(l1) != 149 or len(ssim) != 149:
        raise ValueError(
            f"episode {result['episode_id']}: expected 149 scored transitions, "
            f"got L1={len(l1)} SSIM={len(ssim)}"
        )
    band = 45
    early_l1 = statistics.fmean(l1[:band])
    late_l1 = statistics.fmean(l1[-band:])
    return {
        "mean_l1": float(fds["mean_l1"]),
        "mean_ssim": float(fds["mean_ssim"]),
        "endpoint_l1": l1[-1],
        "endpoint_ssim": ssim[-1],
        "late_l1": late_l1,
        "l1_drift": late_l1 - early_l1,
    }


def _load(
    root: Path,
) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    records: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for condition in CONDITIONS:
        condition_records: dict[tuple[str, int], dict[str, Any]] = {}
        result_paths = sorted((root / condition / "raw").glob("*/c3hss_results.json"))
        if len(result_paths) != 3:
            raise ValueError(
                f"{condition}: expected three result files, found {len(result_paths)}"
            )
        for result_path in result_paths:
            payload = json.loads(result_path.read_text())
            expected = {
                "rollout_conditioning": "autoregressive",
                "max_chunks": 13,
                "generated_rollout_frames": 157,
                "output_frames": 150,
                "fps": 10,
            }
            for key, value in expected.items():
                if payload.get(key) != value:
                    raise ValueError(
                        f"{result_path}: expected {key}={value!r}, "
                        f"got {payload.get(key)!r}"
                    )
            subset = result_path.parent.name
            for result in payload["results"]:
                if result["num_frames"] != 150:
                    raise ValueError(
                        f"{condition}/{subset}/{result['episode_id']}: "
                        f"expected 150 frames, got {result['num_frames']}"
                    )
                key = (subset, int(result["episode_id"]))
                if key in condition_records:
                    raise ValueError(f"duplicate record: {condition}/{key}")
                condition_records[key] = {
                    "metrics": _metrics(result),
                    "directory": result_path.parent,
                    "ground_truth_name": Path(result["ground_truth_video"]).name,
                    "generated_name": Path(result["generated_video"]).name,
                }
        if len(condition_records) != 10:
            raise ValueError(
                f"{condition}: expected ten episode records, "
                f"found {len(condition_records)}"
            )
        records[condition] = condition_records

    identities = set(records[CONDITIONS[0]])
    if set(records[CONDITIONS[1]]) != identities:
        raise ValueError("Arm A and Arm C episode identities do not match")
    return records


def _paired_effect(
    arm_a: list[float],
    arm_c: list[float],
    metric: str,
    rng: random.Random,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if metric in LOWER_IS_BETTER:
        effects = [a - c for a, c in zip(arm_a, arm_c, strict=True)]
    else:
        effects = [c - a for a, c in zip(arm_a, arm_c, strict=True)]
    observed = statistics.fmean(effects)
    bootstrap = sorted(
        statistics.fmean(rng.choices(effects, k=len(effects)))
        for _ in range(bootstrap_samples)
    )
    lower = bootstrap[int(0.025 * (bootstrap_samples - 1))]
    upper = bootstrap[int(0.975 * (bootstrap_samples - 1))]
    sign_flips = [
        statistics.fmean(effect * sign for effect, sign in zip(effects, signs, strict=True))
        for signs in itertools.product((-1.0, 1.0), repeat=len(effects))
    ]
    return {
        "effect_definition": (
            "arm_a_minus_arm_c" if metric in LOWER_IS_BETTER else "arm_c_minus_arm_a"
        ),
        "positive_means_arm_c_is_better": True,
        "mean_effect": observed,
        "bootstrap_95_ci": [lower, upper],
        "exact_paired_sign_flip_p_two_sided": sum(
            abs(value) >= abs(observed) - 1e-15 for value in sign_flips
        )
        / len(sign_flips),
        "arm_c_wins": sum(value > 1e-12 for value in effects),
        "arm_a_wins": sum(value < -1e-12 for value in effects),
        "ties": sum(abs(value) <= 1e-12 for value in effects),
        "per_episode_effects": effects,
    }


def _render_panel(
    ground_truth: Path,
    arm_a: Path,
    arm_c: Path,
    output: Path,
) -> None:
    filter_graph = (
        "[0:v]scale=640:360:force_original_aspect_ratio=decrease,"
        "pad=640:360:(ow-iw)/2:(oh-ih)/2:black,"
        "drawbox=x=0:y=0:w=iw:h=38:color=black@0.65:t=fill,"
        "drawtext=text='Ground Truth':fontcolor=white:fontsize=24:"
        "x=(w-text_w)/2:y=7[gt];"
        "[1:v]scale=640:360:force_original_aspect_ratio=decrease,"
        "pad=640:360:(ow-iw)/2:(oh-ih)/2:black,"
        "drawbox=x=0:y=0:w=iw:h=38:color=black@0.65:t=fill,"
        "drawtext=text='Arm A (H=0)':fontcolor=white:fontsize=24:"
        "x=(w-text_w)/2:y=7[a];"
        "[2:v]scale=640:360:force_original_aspect_ratio=decrease,"
        "pad=640:360:(ow-iw)/2:(oh-ih)/2:black,"
        "drawbox=x=0:y=0:w=iw:h=38:color=black@0.65:t=fill,"
        "drawtext=text='Arm C (full CAMP)':fontcolor=white:fontsize=24:"
        "x=(w-text_w)/2:y=7[c];"
        "[gt][a][c]hstack=inputs=3[v]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(ground_truth),
            "-i",
            str(arm_a),
            "-i",
            str(arm_c),
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-r",
            "10",
            "-frames:v",
            "150",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    if not args.skip_videos and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required unless --skip-videos is set")

    records = _load(args.root)
    identities = sorted(records["arm_a_h0"])
    rng = random.Random(args.seed)
    paired = {}
    for metric in METRICS:
        arm_a = [
            records["arm_a_h0"][identity]["metrics"][metric] for identity in identities
        ]
        arm_c = [
            records["arm_c_real"][identity]["metrics"][metric] for identity in identities
        ]
        paired[metric] = _paired_effect(
            arm_a, arm_c, metric, rng, args.bootstrap_samples
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos = []
    if not args.skip_videos:
        video_dir = args.output_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        for subset, episode_id in identities:
            arm_a_record = records["arm_a_h0"][(subset, episode_id)]
            arm_c_record = records["arm_c_real"][(subset, episode_id)]
            ground_truth = (
                arm_a_record["directory"] / arm_a_record["ground_truth_name"]
            )
            arm_a = arm_a_record["directory"] / arm_a_record["generated_name"]
            arm_c = arm_c_record["directory"] / arm_c_record["generated_name"]
            output = video_dir / f"{subset}_ep{episode_id:05d}_gt_arm_a_arm_c.mp4"
            _render_panel(ground_truth, arm_a, arm_c, output)
            videos.append(
                {
                    "subset": subset,
                    "episode_id": episode_id,
                    "video": str(output),
                    "ground_truth": str(ground_truth),
                    "arm_a": str(arm_a),
                    "arm_c": str(arm_c),
                }
            )

    payload = {
        "analysis": "matched Arm A versus Arm C 15-second autoregressive sweep",
        "statistical_unit": "episode",
        "num_episodes": 10,
        "fps": 10,
        "frames_per_video": 150,
        "duration_seconds": 15.0,
        "arm_a_checkpoint": "iter_000015000",
        "arm_c_checkpoint": "iter_000001700",
        "identities": [
            {"subset": subset, "episode_id": episode_id}
            for subset, episode_id in identities
        ],
        "paired_effects": paired,
        "videos": videos,
        "limitations": [
            "One diffusion seed per episode.",
            "Actions, raw history, and memory codes come from the recorded episode.",
            "Only visual state is fed back autoregressively.",
            "Whole-frame metrics are not tool-localized action-following metrics.",
        ],
    }
    (args.output_dir / "comparison_manifest.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )

    lines = [
        "# Arm A versus Arm C: 15-second autoregressive rollouts",
        "",
        "Ten matched held-out episodes; positive effects mean Arm C is better.",
        "",
        "| Metric | Arm-C effect | 95% bootstrap CI | C wins | Exact p |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        result = paired[metric]
        lower, upper = result["bootstrap_95_ci"]
        lines.append(
            f"| {metric} | {result['mean_effect']:+.6f} "
            f"| [{lower:+.6f}, {upper:+.6f}] "
            f"| {result['arm_c_wins']}/10 "
            f"| {result['exact_paired_sign_flip_p_two_sided']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Actions/history/memory are recorded; only visual state is autoregressive.",
            "",
        ]
    )
    (args.output_dir / "comparison_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
