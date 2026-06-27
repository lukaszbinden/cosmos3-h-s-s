# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_fdm_open_h_sft_nano`` — Cosmos3-Nano Open-H surgical forward-dynamics SFT.

Cosmos Framework port of the cosmos3-internal experiment
``action_fdm_open_h_sft_nano`` (YAML:
``cosmos3/configs/experiment/action_fdm_open_h_sft_nano.yaml``). Post-trains
Cosmos3-Nano (Qwen3-VL-8B + diffusion expert, MoT) into a surgical
world-foundation model on the full Open-H multi-embodiment mixture under the
unified **44D** action space (CMR Versius ceiling: 30D actions + 14D state
conditioning; all other embodiments zero-padded to 44D with channel masking).

The dataset mixture (``OPEN_H_DATASET_SPECS`` in ``groot_configs.py``) targets
maximum non-synthetic surgical coverage of the PUBLIC open-h-embodiment release
that fits the 44D contract: CMR Versius, JHU (IMERSE + LCSR ARCADE/MIRACLE/
SMARTS + STAR-IL), Obuda, Stanford, Turin, UC Berkeley, UCSD, TUD
grasping_retraction, and Virtual Incision MIRA. It is a superset of the
released C-H-S-S mixture (paper Table S4) minus Hamlyn (no endoscope camera in
the public release), USTC/Tuodao (absent from the public tree), and Moon
(delta-xyz only). All embodiments are zero-padded to 44D. See the cookbook
README for the full Table S1 cross-reference.

Why 44D (not 54D): continuity with the already-released Cosmos-H-Surgical-
Simulator checkpoint (which the community already consumes) and with the
cosmos3-internal recipe. See the cookbook README for the full 44D-vs-54D
rationale.

Resolution: 480 tier (832x480 16:9), an officially pretrained Cosmos3-Nano
tier whose pixel budget aligns ~perfectly with the weighted Open-H source
resolution distribution.

Usage (6 nodes x 8 GPU = 48 GPUs, matching the sean reference; data_parallel_shard_degree
set to WORLD_SIZE at launch)::

    OPENH_SURGICAL_ROOT=/path/to/open-h-embodiment/Surgical \\
    BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir> \\
    WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml toml/sft_config/action_fdm_open_h_sft_nano.toml \\
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


# Unified Open-H action dimension (CMR Versius ceiling). Mirrors the
# cosmos3-internal recipe; overrides the NANO default of 64.
_OPEN_H_MAX_ACTION_DIM = 44
# 480 tier: officially pretrained Cosmos3-Nano resolution; pixel-budget aligned
# with the weighted Open-H source distribution (~410k px).
_OPEN_H_RESOLUTION = "480"
# 13 frames = 1 conditional + 12 prediction; 12 action timesteps (div. by the
# temporal compression factor 4).
_OPEN_H_NUM_FRAMES = 13


action_fdm_open_h_sft_nano = LazyDict(
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
            name="action_fdm_open_h_sft_nano",
            wandb_mode="disabled",  # real runs set wandb_mode via TOML/env
        ),
        model=dict(
            config=copy.deepcopy(NANO_MODEL_CONFIG),  # action_gen=True; max_action_dim overridden below
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            # Train the generation + action heads.
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            # Peak LR = 5.0e-5 (base/shared weights).
            #
            # Anchored on THIS framework's own action-posttrain recipe
            # (cosmos_framework .../action/posttrain_config/action_policy_droid_nano.py),
            # which is the closest analog (same framework, same Nano model, same
            # action-conditioning heads + identical 5x head multiplier). It pins:
            #     lr = 2.0e-4  "for the 8192 global batch"
            # and documents max_samples_per_batch as PER-RANK, so
            # global_batch = max_samples_per_batch x world_size.
            #
            # Our run: max_samples_per_batch=64 (per rank) x 48 GPUs = global
            # batch 3072. Linear-scaling the droid anchor to our batch:
            #     2.0e-4 x (3072 / 8192) = 7.5e-5.
            # We then discount ~30% -> 5.0e-5 for the warm-start: we resume from
            # the 54D *surgical* 8k checkpoint (not the bare Cosmos3-Nano base),
            # so the shared tower/diffusion-expert weights are ALREADY adapted to
            # surgical FD. A gentler base LR avoids perturbing those inherited
            # features early while staying in the regime the framework expects
            # for action SFT (notably HIGHER than the prior 2.0e-5 guess, which
            # was anchored on the internal recipe's per-GPU batch assumption that
            # this framework's per-rank batching does not match).
            # Override via TOML (optimizer.lr=...) if throughput/loss says otherwise.
            lr=5.0e-05,
            lr_multipliers={
                # Action-projection heads re-init fresh (base/54D ckpt had a
                # different max_action_dim; keys_to_skip_loading drops them, so
                # they start from scratch at 44D). 5x is the SAME multiplier the
                # framework's droid action recipe uses; here it also lets the
                # fresh 44D heads catch up to the already-surgical tower.
                # 5x * 5.0e-5 = 2.5e-4 effective on the heads (~the droid base LR).
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.1,
        ),
        scheduler=dict(
            # 20k-step integrated warmup + (lambda)linear schedule, mirroring the
            # cosmos3-internal recipe (1000-step warmup, decay to 10% of peak).
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
            max_iter=20000,
            max_val_iter=None,
            run_validation=False,
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
            # Action-projection heads init fresh from the base (base has
            # max_action_dim=64 vs our 44); EMA warm-starts from net.
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
            # Save every 100 iters for fine-grained recovery against 4h SLURM
            # array-task preemption (each ckpt ~30GB).
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
            dataset_name="action_open_h",
            # per-rank micro-batch; effective bs = max_samples_per_batch x world_size.
            max_samples_per_batch=64,
            max_sequence_length=None,  # token packing disabled (TOML can't express null)
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                # num_workers=4 with the DEFAULT spawn context (NOT fork).
                #
                # The 48-GPU "hang" was NOT a worker problem -- it was an empty-shard
                # bug: ActionIterableShuffleDataset shards blocks via
                # order[global_shard::total_shards], and _OpenHShuffleBlockAdapter
                # used to return only len(virtual_sizes)=36 blocks (one per leaf).
                # With 48 ranks, ranks 36-47 got an EMPTY shard and spun forever in
                # the while-True reshuffle loop, never yielding a sample, never
                # reaching the pre-warm barrier (confirmed by py-spy: 100% of CPU in
                # action_sft_dataset.py:79-83). That is fixed in
                # _OpenHShuffleBlockAdapter.get_shuffle_blocks (now ~10^5 fine-grained
                # blocks >> world_size). With that fixed, plain spawn workers are
                # fine: each re-constructs the 36 leaves once at startup (a few min on
                # cold Lustre) then prefetches normally. Do NOT use
                # multiprocessing_context="fork" here -- it deadlocks (fork after the
                # numpy/BLAS/libav init leaves locked mutexes; job 5522818).
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                datasets=dict(
                    open_h=dict(
                        ratio=1,
                        dataset=L(get_action_openh_sft_dataset)(
                            # ``None`` -> use OPEN_H_DATASET_SPECS' absolute
                            # paths verbatim (already rooted at the EOS
                            # open-h-embodiment/Surgical tree). Set DATASET_PATH
                            # only to re-root ALL specs elsewhere (rebases by the
                            # full relative path under the surgical root).
                            base_path="${oc.env:DATASET_PATH,null}",
                            num_frames=_OPEN_H_NUM_FRAMES,
                            data_split="train",
                            mode="forward_dynamics",
                            viewpoint="third_person_view",  # surgical endoscope (third-person)
                            test_split_ratio=0.02,
                            default_storage_fps=30.0,
                            resolution=_OPEN_H_RESOLUTION,
                            max_action_dim=_OPEN_H_MAX_ACTION_DIM,
                            action_channel_masking=True,
                            cfg_dropout_rate=0.1,
                            keep_aspect_ratio=True,
                            caption_key="ai_caption",
                            video_temporal_downsample=4,
                            append_duration_fps_timestamps=True,
                            append_resolution_info=True,
                            append_idle_frames=False,
                            idle_frames_dropout=0.05,
                            format_prompt_as_json=False,
                            iterable_shuffle=True,
                            episode_shuffle_seed=42,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


# === Surgical Open-H overrides on the deep-copied NANO_MODEL_CONFIG ===
# Unified 44D action space (CMR Versius ceiling) — overrides the NANO default 64.
action_fdm_open_h_sft_nano["model"]["config"]["max_action_dim"] = _OPEN_H_MAX_ACTION_DIM
# 480 tier (officially pretrained); pixel-aligned with the Open-H mixture.
action_fdm_open_h_sft_nano["model"]["config"]["resolution"] = _OPEN_H_RESOLUTION
# Uncap packed-sequence length (mirror cosmos3-internal max_num_tokens_after_packing=-1).
action_fdm_open_h_sft_nano["model"]["config"]["max_num_tokens_after_packing"] = -1
# 480 tier needs a larger VAE-latent side budget than the NANO 720-policy default.
action_fdm_open_h_sft_nano["model"]["config"]["diffusion_expert_config"][
    "max_vae_latent_side_after_patchify"
] = 52
# 17-frame exact VAE encode duration (matches num_frames=13 video + action window).
action_fdm_open_h_sft_nano["model"]["config"]["tokenizer"]["encode_exact_durations"] = [17]
# Weight the vision flow-matching loss 10x to balance against action_loss_weight=10
# (mirror cosmos3-internal loss_scale=10.0); image_loss_scale left to default.
action_fdm_open_h_sft_nano["model"]["config"]["rectified_flow_training_config"]["loss_scale"] = 10.0


for _item in [action_fdm_open_h_sft_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
