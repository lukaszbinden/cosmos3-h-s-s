#!/usr/bin/env python3
"""Matched-episode long-horizon evaluator for C3-H-S-S.

Supports both the production-style autoregressive rollout and a diagnostic
teacher-forced mode that resets every chunk from its matched ground-truth
initial frame while leaving actions and history unchanged.
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
from cosmos_framework.data.vfm.action.camp_memory_tracks import CampMemoryTrackJoiner
from cosmos_framework.data.vfm.action.camp_transforms import CampActionTransformPipeline
from cosmos_framework.data.vfm.action.gr00t_dreams.data.embodiment_tags import EmbodimentTag
from cosmos_framework.data.vfm.action.open_h_dataset import OpenHMixedLeRobotDataset
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline
from cosmos_framework.data.vfm.joint_dataloader import PackingDataLoader, custom_collate_fn
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.context_managers import distributed_init, model_init
from cosmos_framework.utils.lazy_config import instantiate

from fds_metrics import compute_frame_decay, resize_video_uint8, video_tensor_to_uint8


CHUNK_SIZE = 12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sft-toml", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    p.add_argument(
        "--embodiment",
        choices=sorted(tag.value for tag in EmbodimentTag),
        default=EmbodimentTag.CMR_VERSIUS.value,
        help="Dataset embodiment tag. Use jhu_dvrk_mono for canonical JHU dVRK data.",
    )
    p.add_argument(
        "--data-split",
        choices=("train", "test", "full"),
        default="test",
        help="Dataset partition to enumerate.",
    )
    p.add_argument("--test-split-ratio", type=float, default=0.05)
    p.add_argument(
        "--timestep-interval",
        type=int,
        default=6,
        help=(
            "Raw dataset-frame stride between 10 Hz model timesteps "
            "(CMR=6, canonical JHU dVRK mono=3)."
        ),
    )
    p.add_argument(
        "--start-base-index",
        type=int,
        default=0,
        help=(
            "Raw-frame base index for the first rollout window. A positive value "
            "allows the first CAMP window to use real, non-boundary-padded history."
        ),
    )
    p.add_argument("--iteration", type=int, default=8000)
    p.add_argument("--max-chunks", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--num-sampling-step", type=int, default=16)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument(
        "--num-history-actions",
        type=int,
        default=0,
        help=(
            "Number of executed actions preceding each forward-dynamics window "
            "to condition on. Use 16 for CAMP Arm B."
        ),
    )
    p.add_argument(
        "--history-ablation",
        choices=("zero", "permute"),
        default=None,
        help="Optional raw-history null intervention. Requires history actions.",
    )
    p.add_argument(
        "--camp-memory-tracks-root",
        default=None,
        help="Exported CAMP memory-track root. Omit for arms A/B.",
    )
    p.add_argument(
        "--camp-memory-ablation",
        choices=("zero", "shuffle_episode"),
        default=None,
        help="Optional learned-memory null intervention. Requires memory tracks.",
    )
    p.add_argument(
        "--score-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional common FDS grid; rollout and saved videos remain at native resolution.",
    )
    p.add_argument("--resume-existing-videos", action="store_true")
    p.add_argument(
        "--rollout-conditioning",
        choices=("autoregressive", "teacher_forced"),
        default="autoregressive",
        help=(
            "How to condition chunks after the first. 'autoregressive' feeds "
            "the previous generated endpoint back into the model; "
            "'teacher_forced' retains each chunk's matched ground-truth first "
            "frame. Actions and action history are identical in both modes."
        ),
    )
    return p.parse_args()


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
        dataset_name=f"matched_{dataset_name}_test",
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


def main() -> None:
    args = parse_args()
    if args.num_history_actions < 0:
        raise ValueError("--num-history-actions must be non-negative")
    if args.history_ablation is not None and args.num_history_actions == 0:
        raise ValueError("--history-ablation requires --num-history-actions > 0")
    if args.camp_memory_ablation is not None and args.camp_memory_tracks_root is None:
        raise ValueError("--camp-memory-ablation requires --camp-memory-tracks-root")
    if args.timestep_interval <= 0:
        raise ValueError("--timestep-interval must be positive")
    if args.start_base_index < 0:
        raise ValueError("--start-base-index must be non-negative")
    if not 0.0 < args.test_split_ratio < 1.0:
        raise ValueError("--test-split-ratio must be in (0, 1)")
    dataset_name = Path(args.dataset).name
    with distributed_init():
        distributed.init()
    if not distributed.is_rank0():
        raise RuntimeError("This evaluator is intentionally single-GPU/rank-0 only")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_job_name = (
        f"matched_long_horizon_eval_{args.rollout_conditioning}_iter_{args.iteration:09d}"
    )
    overrides = [
        f"job.name={eval_job_name}",
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
    config.trainer.run_validation = False
    config.validate()
    config.freeze()

    base = OpenHMixedLeRobotDataset(
        dataset_specs=[{"path": args.dataset, "embodiment": args.embodiment, "mix_ratio": 1.0}],
        num_frames=13,
        data_split=args.data_split,
        test_split_ratio=args.test_split_ratio,
        max_action_dim=44,
        mode="forward_dynamics",
        viewpoint="third_person_view",
        num_history_actions=args.num_history_actions,
        history_ablation=args.history_ablation,
        emit_step_ids=args.camp_memory_tracks_root is not None,
    )
    dataset = base
    if args.camp_memory_tracks_root is not None:
        dataset = CampMemoryTrackJoiner(
            base,
            tracks_root=args.camp_memory_tracks_root,
            memory_ablation=args.camp_memory_ablation,
        )
    transform_cls = (
        CampActionTransformPipeline
        if args.camp_memory_tracks_root is not None
        else ActionTransformPipeline
    )
    transform = transform_cls(
        tokenizer_config=config.model.config.vlm_config.tokenizer,
        cfg_dropout_rate=0.0,
        keep_aspect_ratio=True,
        caption_key="ai_caption",
        video_temporal_downsample=4,
        max_action_dim=44,
        action_channel_masking=True,
        append_viewpoint_info=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=False,
    )
    required = {
        (
            episode_id,
            args.start_base_index + chunk * CHUNK_SIZE * args.timestep_interval,
        )
        for episode_id in args.episodes
        for chunk in range(args.max_chunks)
    }
    index_by_pair = {
        pair: idx for idx, pair in enumerate(base.sub_datasets[0]._all_steps) if tuple(pair) in required
    }
    missing = sorted(required - set(index_by_pair))
    if missing:
        raise RuntimeError(f"Missing matched episode/base-index pairs: {missing}")

    # Validate dataset metadata, normalization provenance, and every requested
    # episode/base-index pair before reading the large distributed checkpoint.
    # A bad eval selection should fail in seconds rather than after minutes of
    # Lustre I/O.
    trainer = config.trainer.type(config)
    with model_init():
        model = instantiate(config.model)
    mem_fmt = _memory_format(getattr(config.trainer, "memory_format", None))
    model = model.to("cuda", memory_format=mem_fmt)
    model.on_train_start(mem_fmt)
    model.eval()
    loaded_iteration = trainer.checkpointer.load(model, optimizer=None, scheduler=None, grad_scaler=None)
    log.info(f"Loaded C3-H-S-S checkpoint; loader iteration={loaded_iteration}")

    torch.set_grad_enabled(False)
    results = []
    for episode_id in args.episodes:
        tag = f"{dataset_name}_ep{episode_id:05d}_seed{args.seed}"
        ground_truth_path = output_dir / f"{tag}_ground_truth.mp4"
        generated_path = output_dir / f"{tag}_c3hss.mp4"
        if args.resume_existing_videos and ground_truth_path.is_file() and generated_path.is_file():
            gt = np.asarray(mediapy.read_video(ground_truth_path), dtype=np.uint8)
            generated = np.asarray(mediapy.read_video(generated_path), dtype=np.uint8)
            n = min(len(gt), len(generated))
            if n == 1 + args.max_chunks * CHUNK_SIZE:
                gt, generated = gt[:n], generated[:n]
                gt_score, generated_score = _score_pair(gt, generated, args.score_size)
                fds = compute_frame_decay(gt_score, generated_score)
                results.append(
                    {
                        "episode_id": episode_id,
                        "seed": args.seed,
                        "num_frames": int(n),
                        "frame_height": int(gt.shape[1]),
                        "frame_width": int(gt.shape[2]),
                        "score_frame_height": int(gt_score.shape[1]),
                        "score_frame_width": int(gt_score.shape[2]),
                        "checkpoint": args.checkpoint,
                        "fds": fds,
                        "ground_truth_video": str(ground_truth_path),
                        "generated_video": str(generated_path),
                    }
                )
                print(
                    f"C3HSS episode={episode_id} frames={n} mean_l1={fds['mean_l1']:.6f} "
                    "(reused existing videos)",
                    flush=True,
                )
                continue
        gt_chunks: list[np.ndarray] = []
        generated_chunks: list[np.ndarray] = []
        current_frame: np.ndarray | None = None
        for chunk in range(args.max_chunks):
            base_index = (
                args.start_base_index + chunk * CHUNK_SIZE * args.timestep_interval
            )
            # Work in the raw, unpadded content space for autoregressive feedback.
            # ActionTransformPipeline pads CMR video to the model canvas and records
            # the true content bounds in image_size.  The decoder removes that
            # padding again.  Feeding a decoded frame back *after stretching it to
            # the padded canvas* makes the next encode crop its upper-left content
            # region, causing a progressive zoom/shift at every chunk boundary.
            raw_sample = dataset[index_by_pair[(episode_id, base_index)]]
            if args.num_history_actions:
                history_action = raw_sample.get("history_action")
                if not isinstance(history_action, torch.Tensor):
                    raise RuntimeError(
                        "CAMP evaluation requested action history, but the dataset "
                        "sample did not emit history_action"
                    )
                if history_action.shape[0] != args.num_history_actions:
                    raise RuntimeError(
                        "history_action row count does not match the requested "
                        f"history length: {history_action.shape[0]} != "
                        f"{args.num_history_actions}"
                    )
            memory_code = raw_sample.get("memory_code")
            if args.camp_memory_tracks_root is not None:
                if not isinstance(memory_code, torch.Tensor) or tuple(
                    memory_code.shape
                ) != (132,):
                    raise RuntimeError(
                        "CAMP memory evaluation requested exported tracks, but the "
                        f"sample emitted memory_code={type(memory_code).__name__} "
                        f"shape={getattr(memory_code, 'shape', None)}"
                    )
            elif memory_code is not None:
                raise RuntimeError(
                    "Non-memory rollout unexpectedly emitted a memory_code"
                )
            # Work at the dataset's native preprocessed content size (for
            # example, 832x480 CMR or 960x544 JHU dVRK), matching each
            # embodiment's training pipeline. Any common-grid resizing happens
            # only after the complete native-resolution rollout, before FDS.
            gt_chunk = _tensor_video_uint8(raw_sample["video"])
            gt_chunks.append(gt_chunk)

            model_sample = deepcopy(raw_sample)
            if args.rollout_conditioning == "autoregressive" and current_frame is not None:
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
            generated_content = video_tensor_to_uint8(decoded)
            # In autoregressive mode, preserve the decoder's unpadded endpoint
            # for the next chunk. Teacher-forced mode deliberately keeps
            # ``current_frame`` unset, so the next raw sample retains its
            # matched ground-truth first frame while actions/history remain
            # unchanged. This isolates rollout-state mismatch from action
            # timing without altering the model or checkpoint.
            if args.rollout_conditioning == "autoregressive":
                current_frame = generated_content[-1].copy()
            # A divisibility edge case may make decoded content differ by a few
            # pixels; resize only to the native GT content shape, never to the
            # temporary padded model canvas.
            generated_for_score = generated_content
            if generated_for_score.shape[1:3] != gt_chunk.shape[1:3]:
                generated_for_score = resize_video_uint8(generated_for_score, gt_chunk.shape[1:3])
            generated_chunks.append(generated_for_score)
            print(f"C3HSS episode={episode_id} chunk={chunk + 1}/{args.max_chunks}", flush=True)

        gt = np.concatenate([gt_chunks[0]] + [chunk[1:] for chunk in gt_chunks[1:]], axis=0)
        generated = np.concatenate(
            [generated_chunks[0]] + [chunk[1:] for chunk in generated_chunks[1:]], axis=0
        )
        n = min(len(gt), len(generated))
        gt, generated = gt[:n], generated[:n]
        gt_score, generated_score = _score_pair(gt, generated, args.score_size)
        fds = compute_frame_decay(gt_score, generated_score)
        mediapy.write_video(output_dir / f"{tag}_ground_truth.mp4", gt, fps=args.fps)
        mediapy.write_video(output_dir / f"{tag}_c3hss.mp4", generated, fps=args.fps)
        results.append(
            {
                "episode_id": episode_id,
                "seed": args.seed,
                "num_frames": int(n),
                "frame_height": int(gt.shape[1]),
                "frame_width": int(gt.shape[2]),
                "score_frame_height": int(gt_score.shape[1]),
                "score_frame_width": int(gt_score.shape[2]),
                "checkpoint": args.checkpoint,
                "fds": fds,
                "ground_truth_video": str(output_dir / f"{tag}_ground_truth.mp4"),
                "generated_video": str(output_dir / f"{tag}_c3hss.mp4"),
            }
        )
        print(f"C3HSS episode={episode_id} frames={n} mean_l1={fds['mean_l1']:.6f}", flush=True)

    payload = {
        "model": "C3-H-S-S",
        "checkpoint": args.checkpoint,
        "checkpoint_loader_iteration": int(loaded_iteration),
        "dataset": args.dataset,
        "dataset_name": dataset_name,
        "embodiment": args.embodiment,
        "data_split": args.data_split,
        "test_split_ratio": args.test_split_ratio,
        "episodes": args.episodes,
        "max_chunks": args.max_chunks,
        "chunk_size": CHUNK_SIZE,
        "timestep_interval": args.timestep_interval,
        "start_base_index": args.start_base_index,
        "seed": args.seed,
        "guidance": args.guidance,
        "num_sampling_step": args.num_sampling_step,
        "fps": args.fps,
        "num_history_actions": args.num_history_actions,
        "history_ablation": args.history_ablation,
        "history_source": (
            "dataset executed actions immediately preceding each rollout window"
            if args.num_history_actions
            else None
        ),
        "camp_memory_tracks_root": args.camp_memory_tracks_root,
        "camp_memory_ablation": args.camp_memory_ablation,
        "rollout_conditioning": args.rollout_conditioning,
        "chunk_initial_frame_source": (
            "previous generated endpoint"
            if args.rollout_conditioning == "autoregressive"
            else "matched ground-truth frame at each chunk boundary"
        ),
        "inference_mode": "forward_dynamics",
        "rollout_space": "native unpadded dataset content",
        "scoring_space": (
            f"common {args.score_size[0]}x{args.score_size[1]} grid"
            if args.score_size
            else "native unpadded dataset content"
        ),
        "results": results,
    }
    (output_dir / "c3hss_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
