#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Main resumable training job: Cosmos3-Nano Open-H multi-embodiment surgical
# forward-dynamics SFT under the unified 44D action space.
#
# Drives cosmos_framework.scripts.train against the staged
# action_fdm_open_h_sft_nano.toml (registered `action_fdm_open_h_sft_nano`
# experiment; res480, forward_dynamics, third_person_view, 44D). One Slurm
# task per node + torchrun --nproc_per_node=8 inside each (EOS/Pyxis
# convention). Re-running with the same job.name resumes from
# latest_checkpoint.txt.
#
#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=batch
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:55:00
#SBATCH --constraint=h100
#SBATCH --exclusive
#SBATCH --job-name=cosmos3_hss_openh_44d
#SBATCH --requeue

set -euo pipefail
export TMPDIR=/tmp

WORKSPACE="${WORKSPACE:-$HOME/cosmos3-h-s-s-workspace}"
source "$WORKSPACE/env.sh"
if [[ -f "$WORKSPACE/env.private.sh" ]]; then
  source "$WORKSPACE/env.private.sh"
fi

# Cookbook dir (staged under $WORKSPACE/cookbook by setup, or run in place).
COOKBOOK_DIR="${COOKBOOK_DIR:-$WORKSPACE/cookbook}"
[[ -d "$COOKBOOK_DIR/scripts" ]] || COOKBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 30000))}"
CONTAINER_IMAGE="${COSMOS3_CONTAINER:-docker://nvcr.io#nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04}"
HOST_HOME="${HOST_HOME:-$HOME}"
OUTPUTS="$WORKSPACE/outputs/train"
WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "$WORKSPACE/logs" "$OUTPUTS" "$WORKSPACE/.cache" "$WORKSPACE/.uv-cache" "$WORKSPACE/.uv-python" "$HOST_HOME/.local/share/uv"

# Bind-mounts: the Open-H surgical data root must be visible inside the
# container at the same path the specs reference (OPENH_SURGICAL_ROOT and/or
# the fs11 absolute paths). Adjust the first mount for your cluster.
DATA_ROOT_MOUNT="${DATA_ROOT_MOUNT:-/lustre/fsw/healthcareeng_holoscan:/lustre/fsw/healthcareeng_holoscan}"
MOUNTS="$DATA_ROOT_MOUNT"
MOUNTS="$MOUNTS,$WORKSPACE/packages/cosmos3:/workspace"
MOUNTS="$MOUNTS,$OUTPUTS:/outputs"
MOUNTS="$MOUNTS,$BASE_CHECKPOINT_PATH:/ckpt"
MOUNTS="$MOUNTS,$WORKSPACE/.cache:/root/.cache"
MOUNTS="$MOUNTS,$WORKSPACE/.uv-cache:$WORKSPACE/.uv-cache"
MOUNTS="$MOUNTS,$WORKSPACE/.uv-python:$WORKSPACE/.uv-python"
MOUNTS="$MOUNTS,$HOST_HOME/.local/share/uv:$HOST_HOME/.local/share/uv"

# Pyxis warmup (pull image once).
srun --container-image="$CONTAINER_IMAGE" bash -c 'echo pyxis-warmup-ok on $(hostname)' || true

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$MOUNTS" \
  --container-workdir=/workspace \
  --export=ALL,MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$MASTER_PORT",BASE_CHECKPOINT_PATH=/ckpt,IMAGINAIRE_OUTPUT_ROOT=/outputs,WANDB_MODE="$WANDB_MODE",OPENH_SURGICAL_ROOT="${OPENH_SURGICAL_ROOT:-}",DATASET_PATH="${DATASET_PATH:-}",COSMOS_OPENH_STATS_POSTFIX="${COSMOS_OPENH_STATS_POSTFIX:-}" \
  "$COOKBOOK_DIR/scripts/_eos_torchrun_inner.sh" \
  -m cosmos_framework.scripts.train \
  --sft-toml="$WORKSPACE/configs/action_fdm_open_h_sft_nano.toml" \
  -- \
  job.wandb_mode="$WANDB_MODE" \
  model.parallelism.data_parallel_shard_degree="$((SLURM_NNODES * 8))"
