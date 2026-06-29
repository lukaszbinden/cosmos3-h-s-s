#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate held-out validation video samples from an Open-H 44D training checkpoint.

Out-of-band, single-GPU evaluation for the cosmos3-h-s-s Open-H multi-embodiment
44D action SFT run. Mirrors sean-cosmos3_surgical_fd's
``generate_checkpoint_samples.py`` (load DCP ckpt -> pull a few val batches ->
run the ``EveryNDrawSample`` diffusion sampler -> save generated-vs-ground-truth
comparison mp4s + ``sample_metadata.json``), adapted to THIS repo:

  * dataset is the gr00t ``OpenHMixedLeRobotDataset`` (``get_action_openh_sft_dataset``),
    NOT the colleague's manifest dataset -- so we BUILD a val dataloader over the
    held-out per-leaf TEST split (``data_split="test"``, ``cfg_dropout_rate=0.0``)
    in-script, because the experiment config ships ``dataloader_val=None``.
  * 44D action space; EMA off, compile off, dp=1 for a 1-GPU eval.

It does NOT touch the running training job (separate checkpoint read, separate
output dir). Pick the checkpoint via --checkpoint-path / CHECKPOINT_ITER.

Usage (via slurm_eval_checkpoint.sbatch):
    python generate_checkpoint_samples_openh.py \
        --sft-toml <action_fdm_open_h_sft_nano.toml> \
        --checkpoint-path <run_dir>/checkpoints/iter_000004000 \
        --iteration 4000 --output-dir <eval_out>/iter_000004000_<ts> \
        --batches 2 --n-viz-sample 2 --num-sampling-step 16 --guidance 1.5 --fps 10
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.callbacks.every_n_draw_sample import EveryNDrawSample
from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.data.vfm.action.datasets.openh_sft_dataset import get_action_openh_sft_dataset
from cosmos_framework.data.vfm.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.context_managers import data_loader_init, distributed_init, model_init
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import instantiate


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() <= 32:
            return value.detach().cpu().tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _memory_format(value: Any) -> torch.memory_format:
    if isinstance(value, torch.memory_format):
        return value
    if isinstance(value, str):
        return getattr(torch, value, torch.preserve_format)
    return torch.preserve_format


def _build_val_dataloader(config: Any, *, num_frames: int, resolution: Any, max_action_dim: int):
    """Build a held-out TEST-split val dataloader for the gr00t Open-H mixture.

    The experiment ships dataloader_val=None, so we construct one here that mirrors
    dataloader_train but with data_split="test", cfg_dropout_rate=0.0, and
    iterable_shuffle=False (deterministic, in-order eval). Token packing matches
    training (max_sequence_length=45056) so the model sees the same shapes.
    """
    tok_cfg = config.model.config.tokenizer
    val_dl = L(PackingDataLoader)(
        audio_sample_rate=48000,
        dataset_name="action_open_h_val",
        max_samples_per_batch=None,
        max_sequence_length=45056,
        patch_spatial=2,
        sound_latent_fps=0,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        dataloader=L(RankPartitionedDataLoader)(
            batch_size=1,
            in_order=True,
            num_workers=0,  # eval: inline decode (no spawn/fork), small + simple
            persistent_workers=False,
            pin_memory=True,
            prefetch_factor=None,
            sampler=None,
            datasets=dict(
                open_h=dict(
                    ratio=1,
                    dataset=L(get_action_openh_sft_dataset)(
                        base_path="${oc.env:DATASET_PATH,null}",
                        num_frames=num_frames,
                        data_split="test",  # the held-out 0.02 split
                        mode="forward_dynamics",
                        viewpoint="third_person_view",
                        resolution=resolution,
                        max_action_dim=max_action_dim,
                        tokenizer_config=tok_cfg,
                        cfg_dropout_rate=0.0,  # no CFG dropout at eval
                        iterable_shuffle=False,  # deterministic eval order
                    ),
                ),
            ),
        ),
    )
    with data_loader_init():
        return instantiate(val_dl)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-toml", required=True, help="Open-H SFT TOML used for training.")
    parser.add_argument("--checkpoint-path", required=True, help="DCP checkpoint directory to load.")
    parser.add_argument("--iteration", type=int, default=0, help="Checkpoint iteration label for outputs.")
    parser.add_argument(
        "--sample-iteration",
        type=int,
        default=None,
        help="Iteration value passed to the sampler (seeding). Defaults to --iteration. "
        "Set equal across checkpoints to compare with the same diffusion seeds.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for copied generated videos + metadata.")
    parser.add_argument("--num-frames", type=int, default=13, help="Video frames (1 ctx + 12 pred); matches training.")
    parser.add_argument("--batches", type=int, default=2, help="Number of val batches to sample.")
    parser.add_argument("--n-viz-sample", type=int, default=2, help="Samples per generated comparison grid.")
    parser.add_argument("--num-sampling-step", type=int, default=16, help="Diffusion sampling steps.")
    parser.add_argument("--guidance", type=float, nargs="+", default=[1.5], help="Classifier-free guidance value(s).")
    parser.add_argument("--fps", type=int, default=10, help="Saved video FPS.")
    parser.add_argument(
        "--compute-fds",
        action="store_true",
        help="Compute Frame Decay Score (L1/SSIM/slope) generated-vs-GT and write to metadata.",
    )
    parser.add_argument(
        "--fds-guidance",
        type=float,
        default=None,
        help="Guidance value to use for the FDS generation pass (defaults to the first --guidance).",
    )
    parser.add_argument(
        "--wandb-id-file",
        default=None,
        help="Path to the training run's wandb_id.txt; if set (+ WANDB_MODE=online), aggregated FDS is "
        "logged to that SAME wandb run at step=--iteration (overlays the loss curves). Optional.",
    )
    parser.add_argument(
        "--wandb-project", default="cosmos3_action_surgical", help="W&B project (must match the training run)."
    )
    parser.add_argument(
        "--wandb-group", default="action_open_h", help="W&B group (must match the training run)."
    )
    parser.add_argument(
        "--wandb-name", default="action_fdm_open_h_sft_nano", help="W&B run name (must match the training run)."
    )
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Optional Hydra-style config overrides.")
    return parser.parse_args()


def _compute_fds_for_batch(model: Any, data_batch: dict[str, Any], *, guidance: float, num_sampling_step: int,
                           n_sample: int, iteration: int) -> list[dict[str, Any]]:
    """Generate samples for a batch and compute per-sample FDS (gen vs GT).

    Mirrors EveryNDrawSample.sample internals to obtain the decoded generated
    video (sample_vision_decoded) and the GT (data_clean.raw_state_vision), then
    runs compute_frame_decay on each. Returns a list of FDS dicts (one per sample).
    """
    from fds_metrics import compute_frame_decay, video_tensor_to_uint8

    data_clean = model.get_data_and_condition(data_batch)
    raw_gt = data_clean.raw_state_vision  # list of [1,C,T,H,W] (or [C,T,H,W]) in [-1,1]
    n = min(n_sample, data_clean.batch_size)

    sample = model.generate_samples_from_batch(
        data_batch,
        guidance=guidance,
        n_sample=n,
        num_steps=num_sampling_step,
        seed=list(range(iteration, iteration + n)),
    )
    sample_vision = sample["vision"]
    decoded = [model.decode(s_i) for s_i in sample_vision]

    out: list[dict[str, Any]] = []
    for i in range(min(n, len(decoded), len(raw_gt))):
        gen_u8 = video_tensor_to_uint8(decoded[i])
        gt_u8 = video_tensor_to_uint8(raw_gt[i])
        out.append(compute_frame_decay(gt_u8, gen_u8))
    return out


def main() -> None:
    args = parse_args()
    overrides = list(args.overrides)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]

    with distributed_init():
        distributed.init()

    rank = distributed.get_rank()
    output_dir = Path(args.output_dir)
    if distributed.is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load the experiment, then force a light/isolated single-GPU eval config.
    config_overrides = [
        f"checkpoint.load_path={args.checkpoint_path}",
        "checkpoint.load_training_state=false",
        "checkpoint.strict_resume=false",
        # MODEL-ONLY load. The checkpointer treats a run dir containing
        # latest_checkpoint.txt as a same-run RESUME and then tries to load all of
        # [model, optim, scheduler, trainer, dataloader] regardless of
        # load_training_state (dcp.py:433-437). We pass optimizer=None for eval, so
        # loading "optim" does optimizer.state_dict() on None -> AttributeError
        # (job 5529588). keys_not_to_resume filters those out (dcp.py:477-480), so
        # only the model weights load.
        'checkpoint.keys_not_to_resume=["optim","scheduler","trainer","dataloader"]',
        "job.wandb_mode=offline",
        "model.config.ema.enabled=false",
        "model.config.compile.enabled=false",
        "model.config.parallelism.data_parallel_shard_degree=1",
        "model.config.parallelism.data_parallel_replicate_degree=1",
        "trainer.distributed_parallelism=fsdp",
        "trainer.run_validation=false",
    ]
    config = load_experiment_from_toml(args.sft_toml, extra_overrides=[*config_overrides, *overrides])

    # Eval is a one-shot sampler; training callbacks are unneeded and can spin up
    # unrelated services.
    config.trainer.callbacks = {}
    config.trainer.run_validation = False

    config.validate()
    config.freeze()

    trainer = config.trainer.type(config)

    with model_init():
        model = instantiate(config.model)
    mem_fmt = _memory_format(getattr(config.trainer, "memory_format", None))
    model = model.to("cuda", memory_format=mem_fmt)
    model.on_train_start(mem_fmt)
    model.eval()

    iteration_loaded = trainer.checkpointer.load(model, optimizer=None, scheduler=None, grad_scaler=None)
    log.info(f"Loaded checkpoint for sampling; trainer iteration from checkpoint loader: {iteration_loaded}")

    # Build the held-out TEST-split val loader (config ships dataloader_val=None).
    dataloader_val = _build_val_dataloader(
        config,
        num_frames=args.num_frames,
        resolution=config.model.config.resolution,
        max_action_dim=config.model.config.max_action_dim,
    )

    sampler = EveryNDrawSample(
        every_n=1,
        n_viz_sample=args.n_viz_sample,
        n_sample_to_save=1,
        num_sampling_step=args.num_sampling_step,
        guidance=args.guidance,
        do_x0_prediction=False,
        save_s3=False,
        save_local=True,
        fps=args.fps,
    )
    sampler.config = config
    sampler.trainer = trainer
    sampler.on_train_start(model, iteration=args.iteration)

    sample_iteration_base = args.sample_iteration if args.sample_iteration is not None else args.iteration
    metadata: dict[str, Any] = {
        "checkpoint_path": args.checkpoint_path,
        "iteration_label": args.iteration,
        "sample_iteration_base": sample_iteration_base,
        "data_split": "test",
        "batches": [],
        "sampler_local_dir": getattr(sampler, "local_dir", None),
    }

    all_fds: list[dict[str, Any]] = []  # per-sample FDS dicts across batches (for aggregation)
    torch.set_grad_enabled(False)
    dataloader_iter = iter(dataloader_val)
    for batch_idx in range(args.batches):
        data_batch = next(dataloader_iter)
        iter_label = sample_iteration_base + batch_idx
        batch_meta = {
            "batch_idx": batch_idx,
            "checkpoint_iteration": args.iteration,
            "callback_iteration": iter_label,
            "keys": sorted(str(key) for key in data_batch.keys()),
            "dataset_name": _as_jsonable(data_batch.get("dataset_name")),
            "num_samples": _as_jsonable(data_batch.get("_num_samples")),
        }
        with torch.inference_mode():
            sampler.sample(
                trainer,
                model,
                data_batch,
                output_batch={},
                loss=torch.zeros((), device="cuda"),
                iteration=iter_label,
            )

        # Frame Decay Score (gen vs GT) on this batch, if requested.
        if args.compute_fds:
            fds_g = args.fds_guidance if args.fds_guidance is not None else float(args.guidance[0])
            try:
                with torch.inference_mode():
                    per_sample_fds = _compute_fds_for_batch(
                        model,
                        data_batch,
                        guidance=fds_g,
                        num_sampling_step=args.num_sampling_step,
                        n_sample=args.n_viz_sample,
                        iteration=iter_label,
                    )
                # strip the bulky per-frame arrays from the per-batch record (keep summaries)
                batch_meta["fds"] = [
                    {k: v for k, v in d.items() if k not in ("l1_per_frame", "ssim_per_frame")}
                    for d in per_sample_fds
                ]
                all_fds.extend(per_sample_fds)
            except Exception as e:  # noqa: BLE001
                log.warning(f"FDS computation failed on batch {batch_idx}: {type(e).__name__}: {e}")
                batch_meta["fds_error"] = f"{type(e).__name__}: {e}"

        expected = (
            Path(sampler.local_dir)
            / f"Iter{iter_label:09d}"
            / f"reg_ReplicateID{rank:04d}_Sample_Iter{iter_label:09d}.mp4"
        )
        batch_meta["callback_video"] = str(expected)
        if distributed.is_rank0() and expected.exists():
            stem = f"checkpoint_iter_{args.iteration:09d}"
            if args.sample_iteration is not None:
                stem += f"_sample_iter_{sample_iteration_base:09d}"
            copied = output_dir / f"{stem}_batch_{batch_idx:02d}_comparison.mp4"
            shutil.copy2(expected, copied)
            batch_meta["copied_video"] = str(copied)
        metadata["batches"].append(batch_meta)
        distributed.barrier()

    # --- Aggregate FDS across all sampled clips + (optionally) push to wandb ---
    if args.compute_fds and all_fds:
        import numpy as np

        def _agg(key: str) -> float:
            vals = [float(d[key]) for d in all_fds if d.get(key) is not None and np.isfinite(d[key])]
            return float(np.mean(vals)) if vals else float("nan")

        fds_summary = {
            "fds/mean_l1": _agg("mean_l1"),  # headline FDS (lower better)
            "fds/mean_ssim": _agg("mean_ssim"),  # structural sim (higher better)
            "fds/l1_slope": _agg("l1_slope"),  # decay across horizon
            "fds/early_c1_l1": float(
                np.nanmean([d["per_chunk"]["early_c1"]["l1"] for d in all_fds])
            ),
            "fds/mid_c2c3_l1": float(
                np.nanmean([d["per_chunk"]["mid_c2c3"]["l1"] for d in all_fds])
            ),
            "fds/late_c4c6_l1": float(
                np.nanmean([d["per_chunk"]["late_c4c6"]["l1"] for d in all_fds])
            ),
            "fds/num_clips": len(all_fds),
        }
        metadata["fds_summary"] = fds_summary
        if distributed.is_rank0():
            log.success(
                f"FDS @ iter {args.iteration}: mean_l1={fds_summary['fds/mean_l1']:.4f} "
                f"mean_ssim={fds_summary['fds/mean_ssim']:.4f} l1_slope={fds_summary['fds/l1_slope']:.5f} "
                f"(n={len(all_fds)} clips)"
            )
            _maybe_log_fds_to_wandb(args, fds_summary)

    if distributed.is_rank0():
        (output_dir / "sample_metadata.json").write_text(
            json.dumps(_as_jsonable(metadata), indent=2, sort_keys=True)
        )
        log.success(f"Saved Open-H checkpoint samples to {output_dir}")

    distributed.barrier()


def _maybe_log_fds_to_wandb(args: Any, fds_summary: dict[str, float]) -> None:
    """Log aggregated FDS to the SAME training wandb run (resume by id) at
    step=checkpoint iteration, so FDS overlays the training loss curves.

    No-op if --wandb-id-file is unset/missing or WANDB_MODE is offline/disabled.
    Best-effort: never raises into the eval (just warns).
    """
    import os

    id_file = args.wandb_id_file
    if not id_file:
        return
    mode = os.environ.get("WANDB_MODE", "online").strip().lower()
    if mode in ("offline", "disabled", ""):
        log.info(f"FDS wandb push skipped (WANDB_MODE={mode!r}).")
        return
    if not os.path.isfile(id_file):
        log.warning(f"FDS wandb push skipped: wandb id file not found: {id_file}")
        return
    try:
        run_id = open(id_file).read().strip()
        if not run_id:
            log.warning(f"FDS wandb push skipped: empty wandb id in {id_file}")
            return
        import wandb

        run = wandb.init(
            id=run_id,
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name,
            resume="allow",
            mode="online",
        )
        # step = checkpoint iteration -> FDS lands at the same x as the loss curve.
        wandb.log({**fds_summary, "trainer/global_step": args.iteration}, step=int(args.iteration))
        wandb.finish()
        log.success(f"Logged FDS to wandb run {run_id} at step {args.iteration}.")
    except Exception as e:  # noqa: BLE001
        log.warning(f"FDS wandb push failed (non-fatal): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
