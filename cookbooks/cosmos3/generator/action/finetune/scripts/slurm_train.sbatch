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

# Repo root = 6 levels above scripts/ (.../cookbooks/cosmos3/generator/action/finetune/scripts).
_SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_SBATCH_DIR/../../../../../.." && pwd)"
# Default WORKSPACE is the repo root (framework checkout at packages/cosmos3,
# env.sh at the root; all gitignored). Override WORKSPACE to relocate.
WORKSPACE="${WORKSPACE:-$REPO_ROOT}"
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

# --- Pre-flight: the two artifacts the run REQUIRES on the host ------------
# Fail fast (before burning an 8-node allocation) with a clear message.
if [[ -z "${BASE_CHECKPOINT_PATH:-}" || ! -e "$BASE_CHECKPOINT_PATH" ]]; then
  echo "FATAL: BASE_CHECKPOINT_PATH not set or missing on host: '${BASE_CHECKPOINT_PATH:-}'." >&2
  echo "       Stage the Cosmos3-Nano DCP checkpoint dir there (see env.sh)." >&2
  exit 64
fi
if [[ -z "${WAN_VAE_PATH:-}" || ! -f "$WAN_VAE_PATH" ]]; then
  echo "FATAL: WAN_VAE_PATH not set or missing on host: '${WAN_VAE_PATH:-}'." >&2
  echo "       setup_workspace.sh downloads it to \$WORKSPACE/checkpoints/wan22_vae/Wan2.2_VAE.pth." >&2
  exit 64
fi

# In-container target for the VAE file. The TOML reads the VAE location from
# \$WAN_VAE_PATH (``[model.tokenizer] vae_path = ${oc.env:WAN_VAE_PATH}``), so we
# mount the host file at a FIXED in-container path and export that path below
# (mirrors BASE_CHECKPOINT_PATH -> /ckpt).
WAN_VAE_MOUNT="/wan_vae/Wan2.2_VAE.pth"

# Bind-mounts: the Open-H surgical data root must be visible inside the
# container at the same path the specs reference (OPENH_SURGICAL_ROOT and/or
# the fs11 absolute paths). Adjust the first mount for your cluster.
DATA_ROOT_MOUNT="${DATA_ROOT_MOUNT:-/lustre/fsw/healthcareeng_holoscan:/lustre/fsw/healthcareeng_holoscan}"
MOUNTS="$DATA_ROOT_MOUNT"
MOUNTS="$MOUNTS,$WORKSPACE/packages/cosmos3:/workspace"
MOUNTS="$MOUNTS,$OUTPUTS:/outputs"
MOUNTS="$MOUNTS,$BASE_CHECKPOINT_PATH:/ckpt"
MOUNTS="$MOUNTS,$WAN_VAE_PATH:$WAN_VAE_MOUNT"
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
  --export=ALL,MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$MASTER_PORT",BASE_CHECKPOINT_PATH=/ckpt,WAN_VAE_PATH="$WAN_VAE_MOUNT",IMAGINAIRE_OUTPUT_ROOT=/outputs,WANDB_MODE="$WANDB_MODE",OPENH_SURGICAL_ROOT="${OPENH_SURGICAL_ROOT:-}",DATASET_PATH="${DATASET_PATH:-}",COSMOS_OPENH_STATS_POSTFIX="${COSMOS_OPENH_STATS_POSTFIX:-}" \
  "$COOKBOOK_DIR/scripts/_eos_torchrun_inner.sh" \
  -m cosmos_framework.scripts.train \
  --sft-toml="$WORKSPACE/configs/action_fdm_open_h_sft_nano.toml" \
  -- \
  job.wandb_mode="$WANDB_MODE" \
  model.parallelism.data_parallel_shard_degree="$((SLURM_NNODES * 8))"
