#!/usr/bin/env python3
"""Validate a completed four-subset, three-seed long-horizon evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUBSETS = ("cholecystectomy", "hysterectomy", "inguinal_hernia", "prostatectomy")
SEEDS = (0, 1, 2)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_nonempty(path: Path, minimum_bytes: int = 1) -> None:
    require(path.is_file(), f"missing file: {path}")
    require(path.stat().st_size >= minimum_bytes, f"file too small: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes-per-subset", type=int, default=5)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--report-must-contain")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    expected_checkpoint = str(Path(args.checkpoint).resolve())
    native_videos: set[Path] = set()
    episode_ids: dict[str, list[int]] = {}

    for subset in SUBSETS:
        for seed in SEEDS:
            result_path = run_dir / "c3hss" / subset / f"seed{seed}" / "c3hss_results.json"
            require_nonempty(result_path)
            data = json.loads(result_path.read_text())
            require(str(Path(data["checkpoint"]).resolve()) == expected_checkpoint,
                    f"wrong checkpoint in {result_path}: {data['checkpoint']}")
            require(data["dataset_name"] == subset, f"wrong subset in {result_path}")
            require(data["seed"] == seed, f"wrong seed in {result_path}")
            require(data["inference_mode"] == "forward_dynamics", f"not FD mode: {result_path}")
            require(data["max_chunks"] == 17, f"wrong chunk count: {result_path}")
            require(data["chunk_size"] == 12, f"wrong chunk size: {result_path}")
            results = data["results"]
            require(len(results) == args.episodes_per_subset,
                    f"expected {args.episodes_per_subset} episodes in {result_path}, got {len(results)}")
            ids = sorted(int(item["episode_id"]) for item in results)
            if seed == 0:
                episode_ids[subset] = ids
            else:
                require(ids == episode_ids[subset], f"episode IDs vary across seeds for {subset}")
            for item in results:
                require(item["num_frames"] == 205, f"wrong frame count in {result_path}")
                require((item["frame_width"], item["frame_height"]) == (832, 480),
                        f"wrong native resolution in {result_path}")
                require((item["score_frame_width"], item["score_frame_height"]) == (512, 288),
                        f"wrong scoring resolution in {result_path}")
                require(str(Path(item["checkpoint"]).resolve()) == expected_checkpoint,
                        f"wrong per-episode checkpoint in {result_path}")
                fds = item["fds"]
                require(fds["num_frames"] == 204, f"wrong predicted-frame count in {result_path}")
                require(len(fds["l1_per_frame"]) == 204, f"wrong L1 curve length in {result_path}")
                require(len(fds["ssim_per_frame"]) == 204, f"wrong SSIM curve length in {result_path}")
                for key in ("ground_truth_video", "generated_video"):
                    video = Path(item[key])
                    require_nonempty(video, 1024)
                    native_videos.add(video.resolve())

    require(len(native_videos) == 120, f"expected 120 native videos, got {len(native_videos)}")

    summary_path = run_dir / "fds_4subset_3seed_summary.json"
    require_nonempty(summary_path)
    summary = json.loads(summary_path.read_text())
    require(summary["subsets"] == list(SUBSETS), "summary has wrong subsets")
    require(summary["seeds"] == list(SEEDS), "summary has wrong seeds")
    require(summary["episodes_per_subset_per_seed"] == args.episodes_per_subset,
            "summary has wrong episode count")
    require(summary["episodes_per_seed"] == len(SUBSETS) * args.episodes_per_subset,
            "summary has wrong balanced episode count")
    require(summary["predicted_frames"] == 204, "summary has wrong horizon")
    require(str(Path(summary["c3hss"]["checkpoint"]).resolve()) == expected_checkpoint,
            "summary has wrong C3 checkpoint")
    for model in ("chss", "c3hss"):
        for metric in ("l1", "ssim"):
            require(len(summary[model][metric]["seed_values"]) == 3,
                    f"{model} {metric} error bars are not seed-level")

    require_nonempty(run_dir / "fds_4subset_3seed_mean_curve.png", 1024)
    require_nonempty(run_dir / "fds_per_seed_and_subset.csv")
    require_nonempty(run_dir / "fds_seed_level_curves.npz", 1024)
    report_path = run_dir / "c-h-s-s_vs_c3-h-s-s_report.md"
    require_nonempty(report_path)
    report = report_path.read_text()
    require("all four operative CMR subsets" in report, "report does not state four-subset scope")
    if args.report_must_contain:
        require(args.report_must_contain in report,
                f"report lacks required text: {args.report_must_contain}")

    manifest_path = run_dir / "comparison" / "manifest.json"
    require_nonempty(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    require(manifest["subsets"] == list(SUBSETS), "comparison manifest has wrong subsets")
    comparison_videos: set[Path] = set()
    for subset in SUBSETS:
        by_seed = manifest["videos"][subset]
        for seed in SEEDS:
            entries = by_seed[str(seed)]
            require(len(entries) == args.episodes_per_subset,
                    f"wrong comparison count for {subset} seed {seed}")
            require(sorted(int(item["episode_id"]) for item in entries) == episode_ids[subset],
                    f"comparison episode IDs mismatch for {subset} seed {seed}")
            for item in entries:
                require(item["frames"] == 205, "comparison has wrong frame count")
                video = Path(item["video"])
                require_nonempty(video, 1024)
                comparison_videos.add(video.resolve())
    require(len(comparison_videos) == 60,
            f"expected 60 comparison videos, got {len(comparison_videos)}")

    status = {
        "status": "validated",
        "checkpoint": expected_checkpoint,
        "subsets": list(SUBSETS),
        "seeds": list(SEEDS),
        "episodes_per_subset_per_seed": args.episodes_per_subset,
        "native_videos": len(native_videos),
        "comparison_videos": len(comparison_videos),
        "predicted_frames": 204,
        "c3hss_l1_mean": summary["c3hss"]["l1"]["mean_across_seeds"],
        "c3hss_l1_std": summary["c3hss"]["l1"]["std_across_seeds"],
        "c3hss_ssim_mean": summary["c3hss"]["ssim"]["mean_across_seeds"],
        "c3hss_ssim_std": summary["c3hss"]["ssim"]["std_across_seeds"],
    }
    status_path = args.status_file or (run_dir / "validation_status.json")
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
