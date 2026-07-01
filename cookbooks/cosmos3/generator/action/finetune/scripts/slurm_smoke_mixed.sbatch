#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Single-node smoke test for the MIXED-MODE (FD+ID+policy) recipe: validates that
# the overlaid Open-H 44D surgical MIXED experiment instantiates end-to-end
# (trainer + model + per-mode Open-H mixture dataloaders) and trains a handful of
# iterations. Exercises every dataset in OPEN_H_DATASET_SPECS across all three
# modes, surfacing missing stats files or per-mode sequence-plan issues.
# NOTE: single node = 8 GPUs >= 3 modes, so RankPartitionedDataLoader can give
# each mode >=1 rank (FD/ID/policy get ~3/~3/~2 ranks at 1:1:1).
#
# Exit 0 -> cleared for sbatch scripts/slurm_train_mixed.sbatch.
#
#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:40:00
#SBATCH --constraint=h100
#SBATCH --exclusive
#SBATCH --job-name=mm-C3-H-S-S-base-smoke
# Per-job log under logs/ (resolved at submit time relative to the submit CWD —
# submit from the repo root, which has a gitignored logs/). %j = job id.
#SBATCH --output=logs/mm-C3-H-S-S-base-smoke_%j.out
#SBATCH --error=logs/mm-C3-H-S-S-base-smoke_%j.out

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

# Cookbook dir — derive from WORKSPACE (repo root), NOT from ${BASH_SOURCE[0]}:
# under SLURM the script body runs from a COPY (e.g. /var/spool/slurm/...), so
# BASH_SOURCE-relative paths resolve wrong and the /cookbook mount source would
# be bogus (-> "execve(): /cookbook/scripts/...: No such file or directory").
# The cookbook is always at this fixed sub-path of the repo root.
COOKBOOK_DIR="${COOKBOOK_DIR:-$WORKSPACE/cookbooks/cosmos3/generator/action/finetune}"
if [[ ! -d "$COOKBOOK_DIR/scripts" ]]; then
  echo "FATAL: cookbook dir not found at '$COOKBOOK_DIR' (WORKSPACE='$WORKSPACE')." >&2
  echo "       Set COOKBOOK_DIR explicitly or fix WORKSPACE (must be the repo root)." >&2
  exit 64
fi

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 30000))}"
CONTAINER_IMAGE="${COSMOS3_CONTAINER:-docker://nvcr.io#nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04}"
HOST_HOME="${HOST_HOME:-$HOME}"
OUTPUTS="$WORKSPACE/outputs/smoke"
mkdir -p "$WORKSPACE/logs" "$OUTPUTS" "$WORKSPACE/.cache" "$WORKSPACE/.uv-cache" "$WORKSPACE/.uv-python" "$HOST_HOME/.local/share/uv"

# --- Pre-flight: the two artifacts the run REQUIRES on the host ------------
# Fail fast with a clear message instead of an opaque in-container error.
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

# In-container target for the VAE file (a single .pth). The TOML reads the VAE
# location from \$WAN_VAE_PATH (``[model.tokenizer] vae_path = ${oc.env:WAN_VAE_PATH}``),
# so we mount the host file at a FIXED in-container path and export that path
# below — mirroring how BASE_CHECKPOINT_PATH is mounted at /ckpt.
WAN_VAE_MOUNT="/wan_vae/Wan2.2_VAE.pth"

# In-container mount point for this cookbook, so the inner launcher
# ($COOKBOOK_DIR/scripts/_eos_torchrun_inner.sh) is reachable INSIDE the
# container. Without this, srun is handed a HOST path (e.g.
# ~/git/cosmos3-h-s-s/cookbooks/.../scripts/...) that doesn't exist in the
# container's mount namespace -> "execve(): ... No such file or directory".
COOKBOOK_MOUNT="/cookbook"

DATA_ROOT_MOUNT="${DATA_ROOT_MOUNT:-/lustre/fsw/healthcareeng_holoscan:/lustre/fsw/healthcareeng_holoscan}"
MOUNTS="$DATA_ROOT_MOUNT"
MOUNTS="$MOUNTS,$WORKSPACE/packages/cosmos3:/workspace"
MOUNTS="$MOUNTS,$COOKBOOK_DIR:$COOKBOOK_MOUNT"
MOUNTS="$MOUNTS,$OUTPUTS:/outputs"
MOUNTS="$MOUNTS,$BASE_CHECKPOINT_PATH:/ckpt"
MOUNTS="$MOUNTS,$WAN_VAE_PATH:$WAN_VAE_MOUNT"
MOUNTS="$MOUNTS,$WORKSPACE/.cache:/root/.cache"
MOUNTS="$MOUNTS,$WORKSPACE/.uv-cache:$WORKSPACE/.uv-cache"
MOUNTS="$MOUNTS,$WORKSPACE/.uv-python:$WORKSPACE/.uv-python"
MOUNTS="$MOUNTS,$HOST_HOME/.local/share/uv:$HOST_HOME/.local/share/uv"

srun --container-image="$CONTAINER_IMAGE" bash -c 'echo pyxis-warmup-ok on $(hostname)' || true

# Small-iter overrides for a fast end-to-end smoke. 8-GPU single-node shard.
# Inner launcher is referenced at its IN-CONTAINER path ($COOKBOOK_MOUNT), and
# we point apply_overlay at the same mount via COSMOS3_APPLY_OVERLAY.
srun --ntasks=1 --ntasks-per-node=1 \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$MOUNTS" \
  --container-workdir=/workspace \
  --export=ALL,MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$MASTER_PORT",BASE_CHECKPOINT_PATH=/ckpt,WAN_VAE_PATH="$WAN_VAE_MOUNT",IMAGINAIRE_OUTPUT_ROOT=/outputs,WANDB_MODE=offline,OPENH_SURGICAL_ROOT="${OPENH_SURGICAL_ROOT:-}",DATASET_PATH="${DATASET_PATH:-}",COSMOS_OPENH_STATS_POSTFIX="${COSMOS_OPENH_STATS_POSTFIX:-}",COSMOS3_APPLY_OVERLAY="$COOKBOOK_MOUNT/scripts/apply_overlay.sh" \
  "$COOKBOOK_MOUNT/scripts/_eos_torchrun_inner.sh" \
  -m cosmos_framework.scripts.train \
  --sft-toml="$COOKBOOK_MOUNT/toml/sft_config/action_mixed_open_h_sft_nano.toml" \
  -- \
  job.name=mm-C3-H-S-S-base-smoke \
  job.wandb_mode=offline \
  trainer.max_iter=10 \
  checkpoint.save_iter=10 \
  model.config.parallelism.data_parallel_shard_degree=8
# NOTE: do NOT override dataloader_train.max_samples_per_batch here. The experiment
# now uses TOKEN packing (max_sequence_length=45056, max_samples_per_batch=None);
# setting max_samples_per_batch would make BOTH non-None and trip the batcher's
# "exactly one of max_sequence_length / max_samples_per_batch must be None" assert.
# The 45056 token cap already fits in memory (it's the colleague's proven value),
# and 1-node smoke has more per-rank headroom than the 48-GPU run anyway.
