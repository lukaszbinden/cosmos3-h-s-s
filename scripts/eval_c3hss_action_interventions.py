#!/usr/bin/env python3
"""Paired one-chunk action-following diagnostic for C3-H-S-S.

Every intervention uses the same initial frame, executed H-action history,
caption, diffusion seed, and checkpoint. Only the normalized 12-row current
action tensor changes. This isolates whether the generated tools respond to
the commanded action window without long-rollout compounding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

import mediapy
import numpy as np
import torch
from c3hss_physical_action_interventions import build_physical_axis_variants
from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.data.vfm.action.gr00t_dreams.data.embodiment_tags import (
    EmbodimentTag,
)
from cosmos_framework.data.vfm.action.open_h_dataset import OpenHMixedLeRobotDataset
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline
from cosmos_framework.data.vfm.joint_dataloader import (
    PackingDataLoader,
    custom_collate_fn,
)
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.context_managers import distributed_init, model_init
from cosmos_framework.utils.lazy_config import instantiate
from fds_metrics import compute_frame_decay, resize_video_uint8, video_tensor_to_uint8

CHUNK_SIZE = 12
ACTION_DIM = 20
ARM_ACTION_DIM = 10
SHIFT_RAW_FRAMES = (-6, -3, 3, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-toml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument(
        "--episode-windows",
        nargs="+",
        help=(
            "Explicit episode:base_index pairs. This is mutually exclusive "
            "with --episodes/--start-base-index."
        ),
    )
    parser.add_argument(
        "--embodiment",
        choices=sorted(tag.value for tag in EmbodimentTag),
        default=EmbodimentTag.JHU_DVRK_MONO.value,
    )
    parser.add_argument(
        "--data-split", choices=("train", "test", "full"), default="test"
    )
    parser.add_argument("--test-split-ratio", type=float, default=0.05)
    parser.add_argument("--timestep-interval", type=int, default=3)
    parser.add_argument("--start-base-index", type=int, default=48)
    parser.add_argument("--iteration", type=int, default=700)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--num-sampling-step", type=int, default=16)
    parser.add_argument("--num-history-actions", type=int, default=16)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--motion-roi-quantile", type=float, default=0.85)
    parser.add_argument(
        "--variant-set",
        choices=("normalized_probes", "physical_axes"),
        default="normalized_probes",
        help="Use legacy normalized probes or valid physical-space per-axis interventions.",
    )
    parser.add_argument(
        "--stats-filename",
        default="meta/stats_cosmos-c3hss-v1.json",
        help="Dataset-relative mean/std file used after the relative-action transform.",
    )
    parser.add_argument(
        "--physical-anchor-mode",
        choices=("reference", "first_row"),
        default="reference",
        help=(
            "Scale physical pose components around the conditioning reference "
            "(legacy) or the first model-facing row (motion-only diagnostic)."
        ),
    )
    parser.add_argument(
        "--physical-intervention-arms",
        nargs="+",
        choices=("psm1", "psm2"),
        default=["psm1", "psm2"],
        help="Generate physical-axis variants only for these robot arms.",
    )
    parser.add_argument(
        "--physical-intervention-components",
        nargs="+",
        choices=("tx", "ty", "tz", "rx", "ry", "rz", "jaw"),
        default=["tx", "ty", "tz", "rx", "ry", "rz", "jaw"],
        help="Generate physical-axis variants only for these components.",
    )
    return parser.parse_args()


def _memory_format(value: Any) -> torch.memory_format:
    if isinstance(value, torch.memory_format):
        return value
    if isinstance(value, str):
        return getattr(torch, value, torch.preserve_format)
    return torch.preserve_format


def _pack_one(sample: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    loader = torch.utils.data.DataLoader(
        [sample], batch_size=1, collate_fn=custom_collate_fn
    )
    packed = PackingDataLoader(
        dataloader=loader,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_sequence_length=None,
        max_samples_per_batch=1,
        dataset_name=f"action_intervention_{dataset_name}_test",
    )
    return next(iter(packed))


def _tensor_video_uint8(video: torch.Tensor) -> np.ndarray:
    return video.permute(1, 2, 3, 0).contiguous().cpu().numpy().astype(np.uint8)


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    """Replace undefined numeric metrics with JSON ``null`` recursively."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")


def build_action_variants(
    correct: torch.Tensor,
    donor: torch.Tensor,
    shifted: dict[int, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Construct paired normalized-action interventions.

    ``correct`` and all source tensors are already relative-transformed and
    normalized by the exact dataset pipeline used in training. Zero and scale
    interventions therefore operate in normalized model space; they are
    conditioning probes, not claims of a physically exact dVRK no-op.
    """
    expected_shape = (CHUNK_SIZE, ACTION_DIM)
    sources = {
        "correct": correct,
        "donor": donor,
        **{f"shift_{k}": v for k, v in shifted.items()},
    }
    for name, value in sources.items():
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name} action has shape {tuple(value.shape)}, expected {expected_shape}"
            )

    variants: OrderedDict[str, torch.Tensor] = OrderedDict()
    variants["correct"] = correct.clone()
    variants["normalized_zero"] = torch.zeros_like(correct)
    variants["donor_episode"] = donor.clone()
    variants["shift_m0p2s"] = shifted[-6].clone()
    variants["shift_m0p1s"] = shifted[-3].clone()
    variants["shift_p0p1s"] = shifted[3].clone()
    variants["shift_p0p2s"] = shifted[6].clone()
    variants["psm1_psm2_swap"] = torch.cat(
        [correct[:, ARM_ACTION_DIM:], correct[:, :ARM_ACTION_DIM]], dim=-1
    )
    variants["normalized_scale_0p5"] = correct * 0.5
    variants["normalized_scale_1p5"] = correct * 1.5
    return variants


def _motion_roi(gt: np.ndarray, quantile: float) -> np.ndarray:
    """Ground-truth motion support used for instrument-sensitive image metrics."""
    frame_delta = np.abs(gt[1:].astype(np.float32) - gt[:-1].astype(np.float32)).mean(
        axis=-1
    )
    score = frame_delta.max(axis=0)
    positive = score[score > 0]
    if positive.size == 0:
        return np.ones(score.shape, dtype=bool)
    threshold = float(np.quantile(positive, quantile))
    mask = score >= threshold
    if mask.mean() < 0.01:
        threshold = float(np.quantile(positive, 0.75))
        mask = score >= threshold
    return mask


def _masked_l1(gt: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> float:
    difference = np.abs(gt.astype(np.float32) - generated.astype(np.float32)) / 255.0
    return float(difference[:, mask, :].mean())


def _motion_energy(video: np.ndarray, mask: np.ndarray) -> np.ndarray:
    difference = (
        np.abs(video[1:].astype(np.float32) - video[:-1].astype(np.float32)) / 255.0
    )
    return difference[:, mask, :].mean(axis=(1, 2))


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or left.std() < 1e-8 or right.std() < 1e-8:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _best_motion_lag(
    gt_energy: np.ndarray, generated_energy: np.ndarray, max_lag: int = 2
) -> dict[str, float | int | None]:
    best: tuple[float, int] | None = None
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left, right = gt_energy[-lag:], generated_energy[:lag]
        elif lag > 0:
            left, right = gt_energy[:-lag], generated_energy[lag:]
        else:
            left, right = gt_energy, generated_energy
        correlation = _safe_corr(left, right)
        if correlation is not None and (best is None or correlation > best[0]):
            best = (correlation, lag)
    return {
        "best_lag_model_frames": None if best is None else best[1],
        "best_lag_seconds": None if best is None else best[1] / 10.0,
        "best_lag_correlation": None if best is None else best[0],
        "zero_lag_correlation": _safe_corr(gt_energy, generated_energy),
    }


def _action_summary(action: torch.Tensor, correct: torch.Tensor) -> dict[str, Any]:
    array = action.detach().cpu().float().numpy()
    delta = array - correct.detach().cpu().float().numpy()
    return {
        "shape": list(array.shape),
        "sha256": _tensor_sha256(action),
        "normalized_rms": float(np.sqrt(np.mean(array**2))),
        "normalized_mean_abs": float(np.mean(np.abs(array))),
        "normalized_max_abs": float(np.max(np.abs(array))),
        "mean_abs_delta_from_correct": float(np.mean(np.abs(delta))),
        "psm1_normalized_rms": float(np.sqrt(np.mean(array[:, :ARM_ACTION_DIM] ** 2))),
        "psm2_normalized_rms": float(np.sqrt(np.mean(array[:, ARM_ACTION_DIM:] ** 2))),
    }


def main() -> None:
    args = parse_args()
    if args.embodiment != EmbodimentTag.JHU_DVRK_MONO.value:
        raise ValueError(
            "This diagnostic currently encodes the 20-D jhu_dvrk_mono arm layout"
        )
    if args.num_history_actions <= 0:
        raise ValueError(
            "--num-history-actions must be positive for the CAMP-lite diagnostic"
        )
    if bool(args.episodes) == bool(args.episode_windows):
        raise ValueError("Provide exactly one of --episodes or --episode-windows")
    if args.episode_windows:
        if args.variant_set != "physical_axes":
            raise ValueError(
                "--episode-windows currently requires --variant-set=physical_axes"
            )
        requested_windows = []
        for value in args.episode_windows:
            try:
                episode_text, base_text = value.split(":", maxsplit=1)
                requested_windows.append((int(episode_text), int(base_text)))
            except ValueError as error:
                raise ValueError(
                    f"Invalid --episode-windows value {value!r}; expected episode:base"
                ) from error
    else:
        requested_windows = [
            (episode_id, args.start_base_index) for episode_id in args.episodes
        ]
    if len({episode_id for episode_id, _ in requested_windows}) != len(
        requested_windows
    ):
        raise ValueError("Each requested window must use a distinct episode")
    episode_ids = [episode_id for episode_id, _ in requested_windows]
    if args.variant_set == "normalized_probes" and args.start_base_index < max(
        abs(value) for value in SHIFT_RAW_FRAMES
    ):
        raise ValueError(
            "--start-base-index is too small for the negative temporal interventions"
        )
    if not 0.0 < args.motion_roi_quantile < 1.0:
        raise ValueError("--motion-roi-quantile must be in (0, 1)")

    with distributed_init():
        distributed.init()
    if not distributed.is_rank0():
        raise RuntimeError("This evaluator is intentionally single-GPU/rank-0 only")

    dataset_name = Path(args.dataset).name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overrides = [
        f"job.name=action_intervention_{dataset_name}_iter_{args.iteration:09d}",
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
        dataset_specs=[
            {"path": args.dataset, "embodiment": args.embodiment, "mix_ratio": 1.0}
        ],
        num_frames=13,
        data_split=args.data_split,
        test_split_ratio=args.test_split_ratio,
        max_action_dim=44,
        mode="forward_dynamics",
        viewpoint="third_person_view",
        num_history_actions=args.num_history_actions,
    )
    transform = ActionTransformPipeline(
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

    required_shifts = (
        (0, *SHIFT_RAW_FRAMES) if args.variant_set == "normalized_probes" else (0,)
    )
    required = {
        (episode_id, base_index + shift)
        for episode_id, base_index in requested_windows
        for shift in required_shifts
    }
    index_by_pair = {
        pair: idx
        for idx, pair in enumerate(base.sub_datasets[0]._all_steps)
        if tuple(pair) in required
    }
    missing = sorted(required - set(index_by_pair))
    if missing:
        raise RuntimeError(f"Missing episode/base-index pairs: {missing}")

    # Fetch and validate every action source before paying the checkpoint load.
    sample_cache: dict[tuple[int, int], dict[str, Any]] = {}
    for pair, index in index_by_pair.items():
        sample = base[index]
        if tuple(sample["action"].shape) != (CHUNK_SIZE, ACTION_DIM):
            raise RuntimeError(
                f"{pair} emitted action shape {tuple(sample['action'].shape)}"
            )
        history = sample.get("history_action")
        if (
            not isinstance(history, torch.Tensor)
            or history.shape[0] != args.num_history_actions
        ):
            raise RuntimeError(f"{pair} emitted invalid history_action")
        sample_cache[pair] = sample

    trainer = config.trainer.type(config)
    with model_init():
        model = instantiate(config.model)
    mem_fmt = _memory_format(getattr(config.trainer, "memory_format", None))
    model = model.to("cuda", memory_format=mem_fmt)
    model.on_train_start(mem_fmt)
    model.eval()
    loaded_iteration = trainer.checkpointer.load(
        model, optimizer=None, scheduler=None, grad_scaler=None
    )
    log.info(
        f"Loaded action-intervention checkpoint; loader iteration={loaded_iteration}"
    )

    torch.set_grad_enabled(False)
    episode_records = []
    for episode_index, (episode_id, base_index) in enumerate(requested_windows):
        donor_episode = episode_ids[(episode_index + 1) % len(episode_ids)]
        raw_sample = sample_cache[(episode_id, base_index)]
        correct_action = raw_sample["action"]
        physical_variants: OrderedDict[str, np.ndarray] | None = None
        stats_audit: dict[str, Any] | None = None
        if args.variant_set == "physical_axes":
            variants, physical_variants, stats_audit = build_physical_axis_variants(
                correct_action,
                Path(args.dataset) / args.stats_filename,
                anchor_mode=args.physical_anchor_mode,
            )
            selected_arms = set(args.physical_intervention_arms)
            selected_components = set(args.physical_intervention_components)
            variants = OrderedDict(
                (name, value)
                for name, value in variants.items()
                if name == "correct"
                or (
                    any(name.startswith(f"{arm}_") for arm in selected_arms)
                    and name.split("_")[1] in selected_components
                )
            )
            physical_variants = OrderedDict(
                (name, value)
                for name, value in physical_variants.items()
                if name == "correct"
                or (
                    any(name.startswith(f"{arm}_") for arm in selected_arms)
                    and name.split("_")[1] in selected_components
                )
            )
            stats_audit["evaluated_intervention_arms"] = sorted(selected_arms)
            stats_audit["evaluated_intervention_components"] = sorted(
                selected_components
            )
            stats_audit["evaluated_variant_count"] = len(variants)
        else:
            donor_action = sample_cache[(donor_episode, base_index)]["action"]
            shifted_actions = {
                shift: sample_cache[(episode_id, base_index + shift)]["action"]
                for shift in SHIFT_RAW_FRAMES
            }
            variants = build_action_variants(
                correct_action, donor_action, shifted_actions
            )
        tag = (
            f"{dataset_name}_ep{episode_id:05d}_base{base_index:05d}_seed{args.seed}"
            if args.episode_windows
            else f"{dataset_name}_ep{episode_id:05d}_seed{args.seed}"
        )
        gt = _tensor_video_uint8(raw_sample["video"])
        ground_truth_path = output_dir / f"{tag}_ground_truth.mp4"
        mediapy.write_video(ground_truth_path, gt, fps=args.fps)
        motion_mask = _motion_roi(gt, args.motion_roi_quantile)
        gt_energy = _motion_energy(gt, motion_mask)
        action_archive = output_dir / f"{tag}_normalized_actions.npz"
        archive_arrays = {
            (
                f"normalized__{name}" if args.variant_set == "physical_axes" else name
            ): value.detach().cpu().numpy()
            for name, value in variants.items()
        }
        if physical_variants is not None:
            archive_arrays.update(
                {
                    f"physical__{name}": value
                    for name, value in physical_variants.items()
                }
            )
        np.savez_compressed(
            action_archive,
            **archive_arrays,
            history_action=raw_sample["history_action"].detach().cpu().numpy(),
            motion_roi=motion_mask,
        )

        generated_by_variant: dict[str, np.ndarray] = {}
        variant_records = []
        for variant_name, action in variants.items():
            model_sample = deepcopy(raw_sample)
            model_sample["action"] = action.clone()
            transformed = transform(model_sample, config.model.config.resolution)
            data_batch = _pack_one(transformed, dataset_name)
            with torch.inference_mode():
                generated = model.generate_samples_from_batch(
                    data_batch,
                    guidance=args.guidance,
                    n_sample=1,
                    num_steps=args.num_sampling_step,
                    seed=[args.seed],
                )
                decoded = model.decode(generated["vision"][0])
            generated_content = video_tensor_to_uint8(decoded)
            if generated_content.shape[1:3] != gt.shape[1:3]:
                generated_content = resize_video_uint8(generated_content, gt.shape[1:3])
            generated_content = generated_content[: len(gt)]
            generated_by_variant[variant_name] = generated_content
            generated_path = output_dir / f"{tag}_{variant_name}.mp4"
            mediapy.write_video(generated_path, generated_content, fps=args.fps)

            fds = compute_frame_decay(gt, generated_content)
            generated_energy = _motion_energy(generated_content, motion_mask)
            motion_lag = _best_motion_lag(gt_energy, generated_energy)
            variant_records.append(
                {
                    "name": variant_name,
                    "generated_video": str(generated_path),
                    "action": _action_summary(action, correct_action),
                    "fds": fds,
                    "motion_roi_l1": _masked_l1(
                        gt[1:], generated_content[1:], motion_mask
                    ),
                    "motion_roi_endpoint_l1": _masked_l1(
                        gt[-1:], generated_content[-1:], motion_mask
                    ),
                    "generated_to_gt_motion_energy_ratio": float(
                        generated_energy.mean() / max(gt_energy.mean(), 1e-8)
                    ),
                    **motion_lag,
                }
            )
            print(
                f"ACTION_INTERVENTION episode={episode_id} variant={variant_name} "
                f"mean_l1={fds['mean_l1']:.6f}",
                flush=True,
            )

        correct_generated = generated_by_variant["correct"]
        for record in variant_records:
            generated_content = generated_by_variant[record["name"]]
            record["paired_output_delta_from_correct_l1"] = float(
                np.abs(
                    generated_content.astype(np.float32)
                    - correct_generated.astype(np.float32)
                ).mean()
                / 255.0
            )
            record["paired_output_delta_from_correct_motion_roi_l1"] = _masked_l1(
                correct_generated[1:], generated_content[1:], motion_mask
            )

        episode_record = {
            "episode_id": episode_id,
            "base_index": base_index,
            "donor_episode_id": (
                donor_episode if args.variant_set == "normalized_probes" else None
            ),
            "ground_truth_video": str(ground_truth_path),
            "normalized_actions_archive": str(action_archive),
            "history_action_sha256": _tensor_sha256(raw_sample["history_action"]),
            "motion_roi_fraction": float(motion_mask.mean()),
            "variants": variant_records,
        }
        if stats_audit is not None:
            episode_record["physical_action_audit"] = stats_audit
        episode_records.append(episode_record)
        # Persist each completed episode immediately. A later episode or final
        # aggregate failure must never discard already-computed GPU results.
        _write_json(
            output_dir / f"{tag}_action_intervention_episode.json", episode_record
        )

    payload = {
        "diagnostic": "paired one-chunk current-action interventions",
        "variant_set": args.variant_set,
        "model": "C3-H-S-S CAMP-lite H16",
        "checkpoint": args.checkpoint,
        "checkpoint_loader_iteration": int(loaded_iteration),
        "dataset": args.dataset,
        "dataset_name": dataset_name,
        "embodiment": args.embodiment,
        "episodes": episode_ids,
        "episode_windows": [
            {"episode_id": episode_id, "base_index": base_index}
            for episode_id, base_index in requested_windows
        ],
        "start_base_index": (None if args.episode_windows else args.start_base_index),
        "timestep_interval": args.timestep_interval,
        "chunk_size": CHUNK_SIZE,
        "num_history_actions": args.num_history_actions,
        "seed": args.seed,
        "guidance": args.guidance,
        "num_sampling_step": args.num_sampling_step,
        "motion_roi_definition": (
            "top ground-truth temporal-change pixels at the configured positive-score quantile"
        ),
        "motion_roi_quantile": args.motion_roi_quantile,
        "normalization_note": (
            "physical-axis conditions invert the dataset-specific mean/std after "
            "the relative-action transform, edit translation/rotation/jaw in "
            f"physical space around the {args.physical_anchor_mode} anchor, then "
            "reapply the same statistics"
            if args.variant_set == "physical_axes"
            else "zero and scale conditions operate after the dataset's relative-action "
            "transform and mean/std normalization; they are model-space causal probes"
        ),
        "physical_anchor_mode": (
            args.physical_anchor_mode if args.variant_set == "physical_axes" else None
        ),
        "physical_intervention_arms": (
            args.physical_intervention_arms
            if args.variant_set == "physical_axes"
            else None
        ),
        "physical_intervention_components": (
            args.physical_intervention_components
            if args.variant_set == "physical_axes"
            else None
        ),
        "temporal_shift_note": (
            "shift conditions borrow the normalized current-action profile from the same "
            "episode at base_index +/-3 or +/-6 raw frames while retaining the target "
            "initial image and executed history"
            if args.variant_set == "normalized_probes"
            else None
        ),
        "results": episode_records,
    }
    _write_json(output_dir / "action_intervention_results.json", payload)


if __name__ == "__main__":
    main()
