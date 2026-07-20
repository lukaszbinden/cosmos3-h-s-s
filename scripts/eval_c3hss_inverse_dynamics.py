#!/usr/bin/env python3
"""Held-out inverse-dynamics evaluation for mixed-mode C3-H-S-S checkpoints.

The model observes each complete 13-frame CMR clip and predicts the 12 aligned
44-D normalized action targets.  Metrics separately report the 30 robot-action
channels and the 14 state-conditioning channels appended during CMR training.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.data.vfm.action.gr00t_dreams.data.embodiment_tags import EmbodimentTag
from cosmos_framework.data.vfm.action.open_h_dataset import OpenHMixedLeRobotDataset
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline
from cosmos_framework.data.vfm.joint_dataloader import PackingDataLoader, custom_collate_fn
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.context_managers import distributed_init, model_init
from cosmos_framework.utils.lazy_config import instantiate


CHUNK_SIZE = 12
TIMESTEP_INTERVAL = 6
ROBOT_ACTION_DIM = 30
CMR_ACTION_DIM = 44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-toml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--max-chunks", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--num-sampling-step", type=int, default=16)
    return parser.parse_args()


def _memory_format(value: Any) -> torch.memory_format:
    if isinstance(value, torch.memory_format):
        return value
    if isinstance(value, str):
        return getattr(torch, value, torch.preserve_format)
    return torch.preserve_format


def _pack_one(sample: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    loader = torch.utils.data.DataLoader([sample], batch_size=1, collate_fn=custom_collate_fn)
    packed = PackingDataLoader(
        dataloader=loader,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_sequence_length=None,
        max_samples_per_batch=1,
        dataset_name=f"matched_{dataset_name}_id_test",
    )
    return next(iter(packed))


def _metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(f"Expected [T,D] actions, got pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if pred.shape != target.shape:
        raise ValueError(f"Action shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if pred.shape[-1] != CMR_ACTION_DIM:
        raise ValueError(f"Expected CMR {CMR_ACTION_DIM}D actions, got {tuple(pred.shape)}")
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Predicted actions contain NaN or Inf")

    error = pred - target
    sq = error.square()
    absolute = error.abs()
    return {
        "mse_44d": float(sq.mean()),
        "mae_44d": float(absolute.mean()),
        "rmse_44d": float(sq.mean().sqrt()),
        "mse_robot_30d": float(sq[:, :ROBOT_ACTION_DIM].mean()),
        "mae_robot_30d": float(absolute[:, :ROBOT_ACTION_DIM].mean()),
        "mse_state_conditioning_14d": float(sq[:, ROBOT_ACTION_DIM:].mean()),
        "mae_state_conditioning_14d": float(absolute[:, ROBOT_ACTION_DIM:].mean()),
        "mse_per_timestep": sq.mean(dim=1).tolist(),
        "mse_per_channel": sq.mean(dim=0).tolist(),
    }


def main() -> None:
    args = parse_args()
    dataset_name = Path(args.dataset).name
    with distributed_init():
        distributed.init()
    if not distributed.is_rank0():
        raise RuntimeError("This evaluator is intentionally single-GPU/rank-0 only")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overrides = [
        f"job.name=matched_id_eval_iter_{args.iteration:09d}",
        f"checkpoint.load_path={args.checkpoint}",
        "checkpoint.load_training_state=false",
        "checkpoint.strict_resume=false",
        'checkpoint.keys_not_to_resume=["optim","scheduler","trainer","dataloader"]',
        "checkpoint.keys_to_skip_loading=[]",
        "job.wandb_mode=offline",
        "model.config.ema.enabled=false",
        "model.config.compile.enabled=false",
        "model.config.parallelism.data_parallel_shard_degree=1",
        "model.config.parallelism.data_parallel_replicate_degree=1",
        "trainer.distributed_parallelism=fsdp",
        "trainer.run_validation=false",
    ]
    config = load_experiment_from_toml(args.sft_toml, extra_overrides=overrides)
    config.trainer.callbacks = {}
    config.validate()
    config.freeze()
    trainer = config.trainer.type(config)
    with model_init():
        model = instantiate(config.model)
    mem_fmt = _memory_format(getattr(config.trainer, "memory_format", None))
    model = model.to("cuda", memory_format=mem_fmt)
    model.on_train_start(mem_fmt)
    model.eval()
    loaded_iteration = trainer.checkpointer.load(model, optimizer=None, scheduler=None, grad_scaler=None)
    log.info(f"Loaded C3-H-S-S checkpoint for ID evaluation; loader iteration={loaded_iteration}")

    base = OpenHMixedLeRobotDataset(
        dataset_specs=[{"path": args.dataset, "embodiment": EmbodimentTag.CMR_VERSIUS, "mix_ratio": 1.0}],
        num_frames=13,
        data_split="test",
        test_split_ratio=0.05,
        max_action_dim=CMR_ACTION_DIM,
        mode="inverse_dynamics",
        viewpoint="third_person_view",
    )
    transform = ActionTransformPipeline(
        tokenizer_config=config.model.config.vlm_config.tokenizer,
        cfg_dropout_rate=0.0,
        keep_aspect_ratio=True,
        caption_key="ai_caption",
        video_temporal_downsample=4,
        max_action_dim=CMR_ACTION_DIM,
        action_channel_masking=True,
        append_viewpoint_info=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=False,
    )

    required = {
        (episode_id, chunk * CHUNK_SIZE * TIMESTEP_INTERVAL)
        for episode_id in args.episodes
        for chunk in range(args.max_chunks)
    }
    index_by_pair = {
        pair: idx for idx, pair in enumerate(base.sub_datasets[0]._all_steps) if tuple(pair) in required
    }
    missing = sorted(required - set(index_by_pair))
    if missing:
        raise RuntimeError(f"Missing matched episode/base-index pairs: {missing}")

    torch.set_grad_enabled(False)
    episode_results = []
    all_window_metrics = []
    for episode_id in args.episodes:
        window_results = []
        for chunk in range(args.max_chunks):
            base_index = chunk * CHUNK_SIZE * TIMESTEP_INTERVAL
            raw_sample = base[index_by_pair[(episode_id, base_index)]]
            target_action = raw_sample["action"].detach().float().cpu().clone()
            model_sample = deepcopy(raw_sample)
            sample = transform(model_sample, config.model.config.resolution)
            data_batch = _pack_one(sample, dataset_name)
            with torch.inference_mode():
                generated = model.generate_samples_from_batch(
                    data_batch,
                    guidance=args.guidance,
                    n_sample=1,
                    num_steps=args.num_sampling_step,
                    seed=[args.seed + chunk],
                )
            if "action" not in generated or len(generated["action"]) != 1:
                raise RuntimeError(f"ID generation did not return one action tensor: {generated.keys()}")
            pred_action = generated["action"][0]
            metrics = _metrics(pred_action, target_action)
            window = {
                "chunk": chunk,
                "base_index": base_index,
                **metrics,
            }
            window_results.append(window)
            all_window_metrics.append(window)
            print(
                f"C3HSS-ID episode={episode_id} chunk={chunk + 1}/{args.max_chunks} "
                f"mse44={metrics['mse_44d']:.6f} mse30={metrics['mse_robot_30d']:.6f}",
                flush=True,
            )

        episode_results.append(
            {
                "episode_id": episode_id,
                "seed": args.seed,
                "num_windows": len(window_results),
                "mean_mse_44d": float(np.mean([item["mse_44d"] for item in window_results])),
                "mean_mae_44d": float(np.mean([item["mae_44d"] for item in window_results])),
                "mean_mse_robot_30d": float(
                    np.mean([item["mse_robot_30d"] for item in window_results])
                ),
                "mean_mse_state_conditioning_14d": float(
                    np.mean([item["mse_state_conditioning_14d"] for item in window_results])
                ),
                "windows": window_results,
            }
        )

    payload = {
        "model": "C3-H-S-S",
        "checkpoint": args.checkpoint,
        "checkpoint_loader_iteration": int(loaded_iteration),
        "dataset": args.dataset,
        "dataset_name": dataset_name,
        "data_split": "test (trailing 5%)",
        "episodes": args.episodes,
        "max_chunks": args.max_chunks,
        "chunk_size": CHUNK_SIZE,
        "timestep_interval": TIMESTEP_INTERVAL,
        "seed": args.seed,
        "guidance": args.guidance,
        "num_sampling_step": args.num_sampling_step,
        "inference_mode": "inverse_dynamics",
        "action_space": (
            "normalized CMR hybrid-relative 44D training space: 30 robot-action + "
            "14 state-conditioning channels"
        ),
        "mean_mse_44d": float(np.mean([item["mse_44d"] for item in all_window_metrics])),
        "mean_mae_44d": float(np.mean([item["mae_44d"] for item in all_window_metrics])),
        "mean_mse_robot_30d": float(
            np.mean([item["mse_robot_30d"] for item in all_window_metrics])
        ),
        "mean_mae_robot_30d": float(
            np.mean([item["mae_robot_30d"] for item in all_window_metrics])
        ),
        "mean_mse_state_conditioning_14d": float(
            np.mean([item["mse_state_conditioning_14d"] for item in all_window_metrics])
        ),
        "episode_results": episode_results,
    }
    (output_dir / "c3hss_id_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
