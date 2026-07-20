#!/usr/bin/env python3
"""FDS-only autoregressive long-horizon evaluator for C-H-S-S."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import mediapy
import numpy as np
import torch


CHSS_REPO = Path("/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/git/cosmos-h-surgical-simulator")
sys.path.insert(0, str(CHSS_REPO))
sys.path.insert(0, str(CHSS_REPO / "scripts"))
import cosmos_h_surgical_simulator_quant_eval as quant_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    p.add_argument("--max-chunks", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--guidance", type=float, default=0.0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument(
        "--experiment",
        default="cosmos_predict2p5_2B_action_conditioned_open_h-fixed_13frame_8nodes_release_oss",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = Path(args.dataset).name
    torch.set_grad_enabled(False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quant_eval.MAX_CHUNKS = args.max_chunks
    quant_eval.find_chunk_indices.__defaults__ = (
        quant_eval.CHUNK_SIZE,
        quant_eval.TIMESTEP_INTERVAL,
        args.max_chunks,
    )
    quant_eval.CHUNK_RANGES = {
        "early_c1c3": (0, 3 * quant_eval.CHUNK_SIZE),
        "mid_c4c6": (3 * quant_eval.CHUNK_SIZE, 6 * quant_eval.CHUNK_SIZE),
        "late_c7c9": (6 * quant_eval.CHUNK_SIZE, args.max_chunks * quant_eval.CHUNK_SIZE),
    }

    # The shared dataset carries the versioned C3-era 44D stats file.  The
    # launcher bind-mounts it at the legacy filename expected by C-H-S-S.  Its
    # per-key values use the same schema, with one additional provenance object
    # that the older loader mistakes for a statistical field.  Ignore only that
    # metadata record while leaving every numerical statistic untouched.
    dataset_module = importlib.import_module(quant_eval.LeRobotDataset.__module__)
    original_json_load = dataset_module.json.load

    def load_stats_without_provenance(file_obj, *load_args, **load_kwargs):
        value = original_json_load(file_obj, *load_args, **load_kwargs)
        if Path(getattr(file_obj, "name", "")).name == "stats_cosmos-44D.json" and isinstance(value, dict):
            value.pop("_provenance", None)
        return value

    dataset_module.json.load = load_stats_without_provenance

    # The repository pins the Wan2.1 VAE to a historical HF revision where
    # tokenizer.pth has since disappeared (404).  The same official NVIDIA
    # repository still publishes tokenizer.pth on its main revision.  Update
    # only this asset locator; model/checkpoint weights remain exactly as given.
    from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_by_uuid

    wan_vae = get_checkpoint_by_uuid("685afcaa-4de2-42fe-b7b9-69f7a2dee4d8")
    object.__setattr__(wan_vae.hf, "revision", "main")

    dataset = quant_eval.load_dataset(args.dataset, "test")
    episode_map = quant_eval.build_episode_index_map(dataset)
    missing = []
    for episode_id in args.episodes:
        indices = quant_eval.find_chunk_indices(episode_map, episode_id, max_chunks=args.max_chunks)
        if indices is None or len(indices) != args.max_chunks:
            missing.append(episode_id)
    if missing:
        raise RuntimeError(f"Episodes missing a complete {args.max_chunks}-chunk horizon: {missing}")

    model = quant_eval.setup_inference_pipeline(args.experiment, args.checkpoint, "", 1)
    results = []
    try:
        for episode_id in args.episodes:
            gt, generated = quant_eval.generate_episode_video(
                model, dataset, episode_map, episode_id, args.seed, args.guidance
            )
            if gt is None or generated is None:
                raise RuntimeError(f"Generation returned no video for episode {episode_id}")
            fds = quant_eval.compute_frame_decay(gt, generated)
            tag = f"{dataset_name}_ep{episode_id:05d}_seed{args.seed}"
            mediapy.write_video(output_dir / f"{tag}_ground_truth.mp4", gt, fps=args.fps)
            mediapy.write_video(output_dir / f"{tag}_chss.mp4", generated, fps=args.fps)
            results.append(
                {
                    "episode_id": episode_id,
                    "seed": args.seed,
                    "num_frames": int(len(gt)),
                    "checkpoint": args.checkpoint,
                    "fds": fds,
                    "ground_truth_video": str(output_dir / f"{tag}_ground_truth.mp4"),
                    "generated_video": str(output_dir / f"{tag}_chss.mp4"),
                }
            )
            print(f"CHSS episode={episode_id} frames={len(gt)} mean_l1={fds['mean_l1']:.6f}", flush=True)
    finally:
        model.cleanup()

    payload = {
        "model": "C-H-S-S",
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "dataset_name": dataset_name,
        "data_split": "test",
        "episodes": args.episodes,
        "max_chunks": args.max_chunks,
        "chunk_size": quant_eval.CHUNK_SIZE,
        "timestep_interval": quant_eval.TIMESTEP_INTERVAL,
        "seed": args.seed,
        "guidance": args.guidance,
        "results": results,
    }
    (output_dir / "chss_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
