#!/usr/bin/env python3
"""Closed-loop joint policy evaluation for mixed-mode C3-H-S-S checkpoints.

Policy mode predicts both the next 12 actions and 12 future video frames from
the current frame. The last generated frame is fed into the following chunk,
so video error measures the same long-horizon drift as FD evaluation while the
aligned action predictions are compared with the held-out trajectory actions.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import mediapy
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

from fds_metrics import compute_frame_decay, resize_video_uint8, video_tensor_to_uint8


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
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--score-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None)
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
        dataset_name=f"matched_{dataset_name}_policy_test",
    )
    return next(iter(packed))


def _tensor_video_uint8(video: torch.Tensor) -> np.ndarray:
    return video.permute(1, 2, 3, 0).contiguous().cpu().numpy().astype(np.uint8)


def _score_pair(
    gt: np.ndarray, generated: np.ndarray, score_size: list[int] | None
) -> tuple[np.ndarray, np.ndarray]:
    if score_size is None:
        return gt, generated
    width, height = score_size
    target_hw = (height, width)
    return resize_video_uint8(gt, target_hw), resize_video_uint8(generated, target_hw)


def _action_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError(f"Action shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if pred.shape[-1] != CMR_ACTION_DIM:
        raise ValueError(f"Expected CMR {CMR_ACTION_DIM}D actions, got {tuple(pred.shape)}")
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Predicted actions contain NaN or Inf")
    error = pred - target
    sq = error.square()
    absolute = error.abs()
    return {
        "mse_robot_30d": float(sq[:, :ROBOT_ACTION_DIM].mean()),
        "mae_robot_30d": float(absolute[:, :ROBOT_ACTION_DIM].mean()),
        "mse_44d": float(sq.mean()),
        "mae_44d": float(absolute.mean()),
        "mse_state_conditioning_14d": float(sq[:, ROBOT_ACTION_DIM:].mean()),
        "mse_per_timestep": sq[:, :ROBOT_ACTION_DIM].mean(dim=1).tolist(),
        "mse_per_channel": sq.mean(dim=0).tolist(),
    }


def _json_finite(value: Any) -> Any:
    """Replace unavailable/non-finite scalar metrics with JSON null."""
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


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
        f"job.name=matched_policy_eval_iter_{args.iteration:09d}",
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
    log.info(f"Loaded C3-H-S-S checkpoint for policy evaluation; loader iteration={loaded_iteration}")

    base = OpenHMixedLeRobotDataset(
        dataset_specs=[{"path": args.dataset, "embodiment": EmbodimentTag.CMR_VERSIUS, "mix_ratio": 1.0}],
        num_frames=13,
        data_split="test",
        test_split_ratio=0.05,
        max_action_dim=CMR_ACTION_DIM,
        mode="policy",
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
    results = []
    for episode_id in args.episodes:
        tag = f"{dataset_name}_ep{episode_id:05d}_seed{args.seed}"
        gt_chunks: list[np.ndarray] = []
        generated_chunks: list[np.ndarray] = []
        action_windows = []
        current_frame: np.ndarray | None = None

        for chunk in range(args.max_chunks):
            base_index = chunk * CHUNK_SIZE * TIMESTEP_INTERVAL
            raw_sample = base[index_by_pair[(episode_id, base_index)]]
            gt_chunk = _tensor_video_uint8(raw_sample["video"])
            target_action = raw_sample["action"].detach().float().cpu().clone()
            gt_chunks.append(gt_chunk)

            model_sample = deepcopy(raw_sample)
            if current_frame is not None:
                raw_h, raw_w = model_sample["video"].shape[-2:]
                if current_frame.shape[:2] != (raw_h, raw_w):
                    current_frame = resize_video_uint8(current_frame[None], (raw_h, raw_w))[0]
                model_sample["video"][:, 0] = torch.from_numpy(current_frame).permute(2, 0, 1)
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
                decoded = model.decode(generated["vision"][0])
            if "action" not in generated or len(generated["action"]) != 1:
                raise RuntimeError(f"Policy generation did not return one action tensor: {generated.keys()}")

            generated_content = video_tensor_to_uint8(decoded)
            current_frame = generated_content[-1].copy()
            generated_for_score = generated_content
            if generated_for_score.shape[1:3] != gt_chunk.shape[1:3]:
                generated_for_score = resize_video_uint8(generated_for_score, gt_chunk.shape[1:3])
            generated_chunks.append(generated_for_score)
            action_metric = _action_metrics(generated["action"][0], target_action)
            action_windows.append({"chunk": chunk, "base_index": base_index, **action_metric})
            print(
                f"C3HSS-POLICY episode={episode_id} chunk={chunk + 1}/{args.max_chunks} "
                f"action_mse30={action_metric['mse_robot_30d']:.6f}",
                flush=True,
            )

        gt = np.concatenate([gt_chunks[0]] + [chunk[1:] for chunk in gt_chunks[1:]], axis=0)
        generated_video = np.concatenate(
            [generated_chunks[0]] + [chunk[1:] for chunk in generated_chunks[1:]], axis=0
        )
        n = min(len(gt), len(generated_video))
        gt, generated_video = gt[:n], generated_video[:n]
        gt_score, generated_score = _score_pair(gt, generated_video, args.score_size)
        fds = compute_frame_decay(gt_score, generated_score)
        gt_path = output_dir / f"{tag}_ground_truth.mp4"
        generated_path = output_dir / f"{tag}_policy.mp4"
        mediapy.write_video(gt_path, gt, fps=args.fps)
        mediapy.write_video(generated_path, generated_video, fps=args.fps)
        results.append(
            {
                "episode_id": episode_id,
                "seed": args.seed,
                "num_frames": int(n),
                "frame_height": int(gt.shape[1]),
                "frame_width": int(gt.shape[2]),
                "score_frame_height": int(gt_score.shape[1]),
                "score_frame_width": int(gt_score.shape[2]),
                "fds": fds,
                "mean_mse_robot_30d": float(np.mean([x["mse_robot_30d"] for x in action_windows])),
                "mean_mae_robot_30d": float(np.mean([x["mae_robot_30d"] for x in action_windows])),
                "mean_mse_44d": float(np.mean([x["mse_44d"] for x in action_windows])),
                "mean_mse_state_conditioning_14d": float(
                    np.mean([x["mse_state_conditioning_14d"] for x in action_windows])
                ),
                "action_windows": action_windows,
                "ground_truth_video": str(gt_path),
                "generated_video": str(generated_path),
            }
        )
        print(
            f"C3HSS-POLICY episode={episode_id} frames={n} video_l1={fds['mean_l1']:.6f} "
            f"action_mse30={results[-1]['mean_mse_robot_30d']:.6f}",
            flush=True,
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
        "inference_mode": "policy",
        "rollout_space": "native unpadded 832x480 content with autoregressive generated-frame feedback",
        "scoring_space": (
            f"common {args.score_size[0]}x{args.score_size[1]} video grid"
            if args.score_size
            else "native unpadded video content"
        ),
        "action_space": (
            "normalized CMR hybrid-relative 44D training space: 30 robot-action + "
            "14 state-conditioning channels"
        ),
        "results": results,
    }
    (output_dir / "c3hss_policy_results.json").write_text(
        json.dumps(_json_finite(payload), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
