#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Single-node smoke test: validates that the overlaid Open-H 44D surgical
# experiment instantiates end-to-end (trainer + model + Open-H mixture
# dataloader) and trains a handful of iterations. Exercises every dataset in
# OPEN_H_DATASET_SPECS, which surfaces missing meta/stats_cosmos.json files and
# any schema mismatch in the newly added (B) leaves.
#
# Exit 0 -> cleared for sbatch scripts/slurm_train.sbatch.
#
#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:40:00
#SBATCH --constraint=h100
#SBATCH --exclusive
#SBATCH --job-name=cosmos3_openh_surgical_smoke

set -euo pipefail
export TMPDIR=/tmp

WORKSPACE="${WORKSPACE:-$HOME/cosmos3_openh_surgical_fd}"
source "$WORKSPACE/env.sh"
if [[ -f "$WORKSPACE/env.private.sh" ]]; then
  source "$WORKSPACE/env.private.sh"
fi

COOKBOOK_DIR="${COOKBOOK_DIR:-$WORKSPACE/cookbook}"
[[ -d "$COOKBOOK_DIR/scripts" ]] || COOKBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 30000))}"
CONTAINER_IMAGE="${COSMOS3_CONTAINER:-docker://nvcr.io#nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04}"
HOST_HOME="${HOST_HOME:-$HOME}"
OUTPUTS="$WORKSPACE/outputs/smoke"
mkdir -p "$WORKSPACE/logs" "$OUTPUTS" "$WORKSPACE/.cache" "$WORKSPACE/.uv-cache" "$WORKSPACE/.uv-python" "$HOST_HOME/.local/share/uv"

DATA_ROOT_MOUNT="${DATA_ROOT_MOUNT:-/lustre/fsw/healthcareeng_holoscan:/lustre/fsw/healthcareeng_holoscan}"
MOUNTS="$DATA_ROOT_MOUNT"
MOUNTS="$MOUNTS,$WORKSPACE/packages/cosmos3:/workspace"
MOUNTS="$MOUNTS,$OUTPUTS:/outputs"
MOUNTS="$MOUNTS,$BASE_CHECKPOINT_PATH:/ckpt"
MOUNTS="$MOUNTS,$WORKSPACE/.cache:/root/.cache"
MOUNTS="$MOUNTS,$WORKSPACE/.uv-cache:$WORKSPACE/.uv-cache"
MOUNTS="$MOUNTS,$WORKSPACE/.uv-python:$WORKSPACE/.uv-python"
MOUNTS="$MOUNTS,$HOST_HOME/.local/share/uv:$HOST_HOME/.local/share/uv"

srun --container-image="$CONTAINER_IMAGE" bash -c 'echo pyxis-warmup-ok on $(hostname)' || true

# Small-iter overrides for a fast end-to-end smoke. 8-GPU single-node shard.
srun --ntasks=1 --ntasks-per-node=1 \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$MOUNTS" \
  --container-workdir=/workspace \
  --export=ALL,MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$MASTER_PORT",BASE_CHECKPOINT_PATH=/ckpt,IMAGINAIRE_OUTPUT_ROOT=/outputs,WANDB_MODE=offline,OPENH_SURGICAL_ROOT="${OPENH_SURGICAL_ROOT:-}",DATASET_PATH="${DATASET_PATH:-}" \
  "$COOKBOOK_DIR/scripts/_eos_torchrun_inner.sh" \
  -m cosmos_framework.scripts.train \
  --sft-toml="$WORKSPACE/configs/action_fdm_open_h_sft_nano.toml" \
  -- \
  job.name=action_fdm_open_h_smoke \
  job.wandb_mode=offline \
  trainer.max_iter=10 \
  checkpoint.save_iter=10 \
  model.parallelism.data_parallel_shard_degree=8 \
  dataloader_train.max_samples_per_batch=8
