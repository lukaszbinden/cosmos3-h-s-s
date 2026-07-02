# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_mixed_open_h_sft_nano`` — Cosmos3-Nano Open-H surgical MIXED-MODE mid-training.

Produces **Cosmos-Surg-3-Base**: a surgical world-foundation base mid-trained on
the SAME Open-H multi-embodiment mixture as ``action_fdm_open_h_sft_nano`` (the
FD-only recipe), but trained on **all three action modes jointly** —
**forward dynamics (FD) + inverse dynamics (ID) + policy** — exactly as the
official Cosmos 3 generator mid-training does (Cosmos 3 paper Table 6,
"Generator mid-training data mixture", Action row = "Forward dynamics, inverse
dynamics, policy"; and the "Action sequence configurations" figure, which shows
each mode is just a different clean/noisy token pattern of the ONE sequence
model over video + action [+ text] tokens).

Why mixed-mode (vs the FD-only recipe)
--------------------------------------
The three modes are conditioning patterns of a single MoT over one multimodal
sequence; only the clean-vs-noised mask changes:

    FD     : p(video | frame0, action [, text])       # denoise video; actions are clean
    ID     : p(action | video [, text])               # denoise actions; video is clean
    policy : p(action, video | frame0 [, text])       # denoise BOTH

Crucially, in **FD mode every action token is conditioning (clean, sigma=0)**, so
the action-PREDICTION head (``llm2action``) receives NO denoising loss — an
FD-only base learns to *use* actions, not to *predict* them. Only ID and policy
train action prediction. A mixed-mode base therefore learns both directions of
the world-action relationship and is a stronger starting point for the
downstream single-mode specialists:

    Cosmos-Surg-3-Base (this recipe)
        --FD-only fine-tune-->     Cosmos-Surg-3-Simulator
        --policy-only fine-tune--> Cosmos-Surg-3-Policy
        --ID-only fine-tune-->     Cosmos-Surg-3-ID   (optional)

Everything else (44D unified action space, 480 tier, 13 frames = 1 context + 12
prediction, model config, optimizer/scheduler, token packing, warm-start with
fresh 44D action heads) is IDENTICAL to ``action_fdm_open_h_sft_nano`` so the
only intended difference vs that baseline is the mode mixture. This keeps the
two directly comparable for the mixed-vs-single-mode ablation (H1-ablation).

Mode mixture (rank partitioning)
--------------------------------
``RankPartitionedDataLoader`` assigns each rank to exactly ONE mode by ratio
(this recipe runs on 6 nodes x 8 = 48 GPUs at FD:ID:policy = ``_MODE_RATIOS``
below = 1:1:1, so 16/16/16 ranks each). The *global* batch is therefore
mixed-mode (gradients average across modes), while each rank streams a single
mode (so token packing never mixes modes within one packed sequence).
``_MODE_RATIOS`` is a HYPERPARAMETER — see its docstring.

Configured for 6 nodes x 8 GPU = 48 GPUs, IDENTICAL to the FD recipe (same LR
3.0e-5, same 45056 token cap, same schedule) so this run is a clean mirror of
the FD baseline differing ONLY in the mode mixture. Launch with the dedicated
launcher (does NOT rely on env-var toml switching)::

    OPENH_SURGICAL_ROOT=/path/to/open-h-embodiment/Surgical \\
    BASE_CHECKPOINT_PATH=<warm-start DCP dir> \\
    WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    sbatch cookbooks/.../scripts/slurm_train_mixed.sbatch

    # or directly (data_parallel_shard_degree set to WORLD_SIZE at launch):
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml toml/sft_config/action_mixed_open_h_sft_nano.toml \\
        --config-overrides \\
            "checkpoint.load_path=$BASE_CHECKPOINT_PATH" \\
            "model.config.parallelism.data_parallel_shard_degree=$WORLD_SIZE"
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.vfm.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.data.vfm.action.datasets.openh_sft_dataset import get_action_openh_sft_dataset

cs = ConfigStore.instance()


# Unified Open-H action dimension (CMR Versius ceiling). Mirrors the FD recipe.
_OPEN_H_MAX_ACTION_DIM = 44
# 480 tier: officially pretrained Cosmos3-Nano resolution.
_OPEN_H_RESOLUTION = "480"
# 13 frames = 1 conditional + 12 prediction; 12 action timesteps (div. by 4).
_OPEN_H_NUM_FRAMES = 13
# Packed-sequence TOKEN cap = 45056, IDENTICAL to the FD recipe. This is safe at
# the 6-node/48-GPU shape (FSDP shards params+grads+optimizer+EMA across 48 ranks,
# same per-GPU model-state as the FD run, which trains + resumes fine at 45056).
# History: an earlier 4-node/32-GPU attempt OOM'd on RESUME in the backward
# ("Failed to CUDA calloc async", job 5541390) because 32-way sharding held ~1.5x
# more model-state per GPU with no headroom. Going to 48 GPUs restores that
# headroom, so we keep the FD token cap for max throughput + a clean ablation.
_OPEN_H_MAX_SEQ_LEN = 45056

# ---------------------------------------------------------------------------
# Mode mixture: HYPERPARAMETER. RankPartitionedDataLoader allocates ranks to
# modes proportionally to these weights (each rank runs ONE mode; the global
# batch is the weighted mix).
#
# DEFAULT = 1:1:1 (equal), following the Cosmos 3 paper. The paper does NOT
# publish a FD:ID:policy sub-split for the full mid-training mixture (Table 6
# only lists Action = 25% of the overall mixture with modes "forward dynamics,
# inverse dynamics, policy"), but its own joint-mode study (App. "Synergy
# Between Action Modes", tab:pusht_fd_id_policy_open_loop) trains the joint
# FD/ID/policy checkpoint so that "each mode is trained for the same amount of
# optimization steps" -- i.e. EQUAL per-mode allocation. We mirror that here.
#
# That ablation is the paper's own version of our H1(ablation): the joint
# checkpoint cut ID MSE by 72% and raised policy coverage (74.1->77.3%), with a
# modest FD PSNR tradeoff (27.13->26.22) vs single-mode. So the equal mix helps
# ID/policy most and costs FD slightly -- reweight toward FD (e.g. 2:1:1) ONLY
# if the FD Simulator is the priority and you accept less ID/policy synergy.
#
# Constraints:
#   - world_size must be >= number of NONZERO-ratio modes (>=3 here).
#   - set a mode's weight to 0 to drop it (RankPartitionedDataLoader skips it).
# NOTE: changing these changes the compute split across modes, not total FLOPs.
# ---------------------------------------------------------------------------
_MODE_RATIOS = {
    "forward_dynamics": 1,
    "inverse_dynamics": 1,
    "policy": 1,
}


def _openh_dataset(mode: str, *, data_split: str, cfg_dropout_rate: float, iterable_shuffle: bool):
    """One Open-H mixture dataset bound to a single action ``mode``.

    All knobs are IDENTICAL to ``action_fdm_open_h_sft_nano`` except ``mode``
    (and the split/dropout/shuffle that differ between train and eval). This is
    the single source of truth for both train and val entries below.
    """
    return L(get_action_openh_sft_dataset)(
        # ``None`` -> use OPEN_H_DATASET_SPECS' absolute paths verbatim. Set
        # DATASET_PATH only to re-root ALL specs elsewhere.
        base_path="${oc.env:DATASET_PATH,null}",
        num_frames=_OPEN_H_NUM_FRAMES,
        data_split=data_split,
        mode=mode,
        viewpoint="third_person_view",  # surgical endoscope (third-person)
        test_split_ratio=0.02,
        default_storage_fps=30.0,
        resolution=_OPEN_H_RESOLUTION,
        max_action_dim=_OPEN_H_MAX_ACTION_DIM,
        action_channel_masking=True,
        cfg_dropout_rate=cfg_dropout_rate,
        keep_aspect_ratio=True,
        caption_key="ai_caption",
        video_temporal_downsample=4,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=False,
        idle_frames_dropout=0.05 if data_split == "train" else 0.0,
        format_prompt_as_json=False,
        iterable_shuffle=iterable_shuffle,
        episode_shuffle_seed=42,
        tokenizer_config="${model.config.vlm_config.tokenizer}",
    )


def _mode_datasets(*, data_split: str, cfg_dropout_rate: float, iterable_shuffle: bool) -> dict:
    """Build the ``datasets=dict(...)`` block: one entry per nonzero-ratio mode.

    Entry keys are ``open_h_<mode-abbrev>`` so per-mode allocation is legible in
    the RankPartitionedDataLoader startup log.
    """
    abbrev = {"forward_dynamics": "fd", "inverse_dynamics": "id", "policy": "policy"}
    out: dict = {}
    for mode, ratio in _MODE_RATIOS.items():
        if ratio <= 0:
            continue
        out[f"open_h_{abbrev[mode]}"] = dict(
            ratio=ratio,
            dataset=_openh_dataset(
                mode, data_split=data_split, cfg_dropout_rate=cfg_dropout_rate, iterable_shuffle=iterable_shuffle
            ),
        )
    return out


action_mixed_open_h_sft_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3_action_surgical",
            group="action_open_h",
            # Run/output name = mm-C3-H-S-S-base (multi-mode C3-H-S-S base). Drives
            # the W&B run name and the checkpoint run dir
            # (.../action_open_h/mm-C3-H-S-S-base), distinct from the FD run's dir.
            # NOTE: the Hydra EXPERIMENT key stays action_mixed_open_h_sft_nano
            # (referenced by the TOML experiment= and apply_overlay registration);
            # only this human/run name is mm-C3-H-S-S-base.
            name="mm-C3-H-S-S-base",  # -> Cosmos-Surg-3-Base
            wandb_mode="disabled",  # real runs set wandb_mode via TOML/env
        ),
        model=dict(
            config=copy.deepcopy(NANO_MODEL_CONFIG),  # action_gen=True; max_action_dim overridden below
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            # Train the generation + action heads. action2llm/llm2action/
            # action_modality_embed matter MORE here than in the FD recipe:
            # ID + policy actually denoise action tokens, so llm2action gets real
            # gradient (FD alone left it untrained).
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            # Peak LR 3.0e-5 (base/shared weights) — IDENTICAL to the FD recipe at
            # the same 6-node/48-GPU shape. Keeping the FD LR (not a scaled value)
            # makes this mixed run a clean mirror of the FD baseline: same GPUs,
            # same token cap, same LR/schedule — the ONLY difference is the mode
            # mixture. That is exactly the controlled setup H1(ablation) needs.
            # If you relaunch at a different world size, rescale lr ~= 3.0e-5 *
            # (WORLD_SIZE / 48).
            lr=3.0e-05,
            lr_multipliers={
                # Fresh 44D action heads (base ckpt is 64D; skipped on load) get a
                # 5x boost to catch up to the warm-started tower.
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            # PyTorch-native fused AdamW (FSDP/DTensor-safe); NOT apex FusedAdam.
            optimizer_type="AdamW",
            weight_decay=0.1,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[20000],
            f_max=[1.0],
            f_min=[0.1],
            f_start=[0.1],
            verbosity_interval=0,
            warm_up_steps=[1000],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=50,
            # 20k steps to match the FD baseline for the "same data" comparison.
            # NOTE: this base spreads updates across 3 modes, so each mode sees
            # fewer effective steps than a pure run at the same max_iter. If you
            # want a stronger BASE (not a controlled ablation), consider raising
            # max_iter (e.g. 40000) via TOML; keep it at 20000 when comparing
            # head-to-head against the FD-only baseline.
            max_iter=20000,
            max_val_iter=None,
            run_validation=False,  # OmniMoTModel.validation_step is a no-op; see FD recipe
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=0,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=100, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=50, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=200, gc_level=1, warm_up=5),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=True,
            dcp_async_mode_enabled=True,
            enable_gcs_patch_in_boto3=False,
            keys_not_to_resume=[],
            # Action-projection heads init fresh from the base (base is 64D vs our
            # 44D); EMA warm-starts from net.
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Nano DCP dir; supply via TOML/env
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=True,
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_open_h_mixed",
            # TOKEN-based packing (max_sequence_length), NOT sample-count. See the
            # FD recipe for the full OOM rationale; identical here. Each rank is a
            # single mode (RankPartitioned), so packing only ever combines
            # same-mode samples.
            max_samples_per_batch=None,
            max_sequence_length=_OPEN_H_MAX_SEQ_LEN,
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                # One entry per mode (FD/policy/ID); ranks are partitioned across
                # them by _MODE_RATIOS. world_size must be >= #nonzero modes.
                datasets=_mode_datasets(
                    data_split="train", cfg_dropout_rate=0.1, iterable_shuffle=True
                ),
            ),
        ),
        # Held-out TEST-split loader, mirroring train but deterministic and with
        # no CFG dropout. Not used in-training (run_validation=False); defined for
        # reuse / split documentation. Mirrors all three modes.
        dataloader_val=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_open_h_mixed_val",
            max_samples_per_batch=None,
            max_sequence_length=_OPEN_H_MAX_SEQ_LEN,
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=True,
                num_workers=2,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=2,
                sampler=None,
                datasets=_mode_datasets(
                    data_split="test", cfg_dropout_rate=0.0, iterable_shuffle=False
                ),
            ),
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


# === Surgical Open-H overrides on the deep-copied NANO_MODEL_CONFIG ===
# (identical to action_fdm_open_h_sft_nano — mixed-mode changes ONLY the data)
action_mixed_open_h_sft_nano["model"]["config"]["max_action_dim"] = _OPEN_H_MAX_ACTION_DIM
action_mixed_open_h_sft_nano["model"]["config"]["resolution"] = _OPEN_H_RESOLUTION
action_mixed_open_h_sft_nano["model"]["config"]["diffusion_expert_config"][
    "max_vae_latent_side_after_patchify"
] = 52
# 17-frame exact VAE encode duration (matches num_frames=13 video + action window).
action_mixed_open_h_sft_nano["model"]["config"]["tokenizer"]["encode_exact_durations"] = [17]
# Weight the vision flow-matching loss 10x (mirror cosmos3-internal loss_scale=10.0).
action_mixed_open_h_sft_nano["model"]["config"]["rectified_flow_training_config"]["loss_scale"] = 10.0


for _item in [action_mixed_open_h_sft_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
