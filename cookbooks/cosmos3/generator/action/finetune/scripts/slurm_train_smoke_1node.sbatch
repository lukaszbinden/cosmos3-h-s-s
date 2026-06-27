#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Single-node TRAINING smoke: a faithful mini-run of the real 6-node training
# (slurm_train.sbatch) on ONE node / 8 GPUs. Unlike slurm_smoke.sbatch (which
# uses a toy batch of 8 just to prove the experiment instantiates), this runs at
# the REAL per-rank batch (max_samples_per_batch=64) and the REAL warm-start
# from the 54D surgical checkpoint (BASE_CHECKPOINT_PATH), so it validates the
# things that only got changed AFTER the last smoke passed:
#   * lr=5.0e-5 (from the experiment config) actually loads,
#   * the 54D -> 44D warm-start loads with keys_to_skip_loading (fresh action
#     heads), optimizer/scheduler/heads build,
#   * training actually steps at the real batch (one iter_speed/mfu/loss line at
#     iter 50) and writes a checkpoint (save_iter=50) -> resume path works.
#
# It uses a SEPARATE job.name (action_fdm_open_h_smoke_1node) and a SEPARATE
# output dir (outputs/smoke_train), so it NEVER touches the real run's
# checkpoints — the real 6-node job still cold-starts cleanly from the 54D ckpt.
#
# NOTE re: the prior 6-node crash — that was a dist.barrier() NCCL timeout during
# dataset pre-warm, caused by a cold-Lustre straggler spread across 48 ranks
# (fixed by TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC in _eos_torchrun_inner.sh). That
# failure mode is MULTI-NODE-specific (needs the many-rank straggler spread), so
# this 1-node smoke is NOT expected to reproduce it; its purpose is to confirm
# the warm-start + real-batch stepping that the multi-node run never reached.
#
# Exit 0 -> the warm-start + training path is healthy; the 6-node run only adds
# scale (and the already-fixed barrier timeout).
#
#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:55:00
#SBATCH --constraint=h100
#SBATCH --exclusive
#SBATCH --job-name=healthcareeng_holoscan-cosmos3.openh44d_smoke1n
# Per-job log under logs/ (resolved at submit time relative to the submit CWD —
# submit from the repo root, which has a gitignored logs/). %j = job id.
#SBATCH --output=logs/openh44d_smoke1n_%j.out
#SBATCH --error=logs/openh44d_smoke1n_%j.out

set -euo pipefail
export TMPDIR=/tmp

# Resolve WORKSPACE (repo root: framework checkout at packages/cosmos3, env.sh at
# the root). Prefer $WORKSPACE from the environment (set by sourcing env.sh before
# sbatch, propagated via --export=ALL), else $SLURM_SUBMIT_DIR if it has env.sh.
# Do NOT derive from ${BASH_SOURCE[0]}: under SLURM the script runs from a COPY
# (/var/spool/slurm/...), so BASH_SOURCE path math yields '' or '/' -> a broken
# `source "//env.sh"`.
if [[ -z "${WORKSPACE:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/env.sh" ]]; then
    WORKSPACE="$SLURM_SUBMIT_DIR"
  fi
fi
if [[ -z "${WORKSPACE:-}" || ! -f "$WORKSPACE/env.sh" ]]; then
  echo "FATAL: cannot locate WORKSPACE/env.sh (WORKSPACE='${WORKSPACE:-}' SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}')." >&2
  echo "  Fix: 'source env.sh' before sbatch, OR submit from the repo root, OR pass WORKSPACE=... explicitly." >&2
  exit 64
fi
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
OUTPUTS="$WORKSPACE/outputs/smoke_train"
WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "$WORKSPACE/logs" "$OUTPUTS" "$WORKSPACE/.cache" "$WORKSPACE/.uv-cache" "$WORKSPACE/.uv-python" "$HOST_HOME/.local/share/uv"

# --- Startup banner + resume status ----------------------------------------
# Separate run dir from the real 6-node job (job.name=action_fdm_open_h_smoke_1node).
# Report whether this will RESUME (a prior smoke left a checkpoint) or COLD-START
# (warm-start from the 54D base ckpt).
_RUN_DIR="$OUTPUTS/cosmos3_action_surgical/action_open_h/action_fdm_open_h_smoke_1node"
_LATEST_CKPT_FILE="$_RUN_DIR/checkpoints/latest_checkpoint.txt"
echo "===== SLURM TRAIN-SMOKE (1 node) START $(date -Is) ====="
echo "  job_id=${SLURM_JOB_ID:-?}  nodes=${SLURM_NNODES:-?}  job_name=${SLURM_JOB_NAME:-?}  workspace=$WORKSPACE"
echo "  run_dir=$_RUN_DIR  (separate from the real 6-node run)"
if [[ -f "$_LATEST_CKPT_FILE" ]]; then
  echo "  RESUME: latest_checkpoint.txt -> $(cat "$_LATEST_CKPT_FILE" 2>/dev/null) (base ckpt ignored)"
else
  echo "  COLD-START: no latest_checkpoint.txt yet -> warm-starts from BASE_CHECKPOINT_PATH"
fi
echo "  base_ckpt=${BASE_CHECKPOINT_PATH:-<unset>}"
echo "  log: $WORKSPACE/logs/openh44d_smoke1n_${SLURM_JOB_ID:-0}.out"

# --- Pre-flight: the two artifacts the run REQUIRES on the host ------------
# Fail fast with a clear message instead of an opaque in-container error.
if [[ -z "${BASE_CHECKPOINT_PATH:-}" || ! -e "$BASE_CHECKPOINT_PATH" ]]; then
  echo "FATAL: BASE_CHECKPOINT_PATH not set or missing on host: '${BASE_CHECKPOINT_PATH:-}'." >&2
  echo "       Stage the warm-start DCP checkpoint dir there (see env.sh)." >&2
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

# In-container mount point for this cookbook, so the inner launcher
# ($COOKBOOK_DIR/scripts/_eos_torchrun_inner.sh) + apply_overlay.sh are reachable
# INSIDE the container. Without this, srun is handed a HOST path that doesn't
# exist in the container's mount namespace -> "execve(): ... No such file or
# directory".
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

# Pyxis warmup (pull image once).
srun --container-image="$CONTAINER_IMAGE" bash -c 'echo pyxis-warmup-ok on $(hostname)' || true

# Faithful mini-run on 1 node / 8 GPUs: REAL per-rank batch (64) + REAL warm-start
# (BASE_CHECKPOINT_PATH=/ckpt) + the experiment lr (5.0e-5), but a short
# max_iter with a single checkpoint save. job.name + output dir are isolated from
# the real 6-node run. Inner launcher + apply_overlay referenced at their
# IN-CONTAINER paths ($COOKBOOK_MOUNT).
srun --ntasks=1 --ntasks-per-node=1 \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$MOUNTS" \
  --container-workdir=/workspace \
  --export=ALL,MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$MASTER_PORT",BASE_CHECKPOINT_PATH=/ckpt,WAN_VAE_PATH="$WAN_VAE_MOUNT",IMAGINAIRE_OUTPUT_ROOT=/outputs,WANDB_MODE="$WANDB_MODE",OPENH_SURGICAL_ROOT="${OPENH_SURGICAL_ROOT:-}",DATASET_PATH="${DATASET_PATH:-}",COSMOS_OPENH_STATS_POSTFIX="${COSMOS_OPENH_STATS_POSTFIX:-}",COSMOS3_APPLY_OVERLAY="$COOKBOOK_MOUNT/scripts/apply_overlay.sh" \
  "$COOKBOOK_MOUNT/scripts/_eos_torchrun_inner.sh" \
  -m cosmos_framework.scripts.train \
  --sft-toml="$COOKBOOK_MOUNT/toml/sft_config/action_fdm_open_h_sft_nano.toml" \
  -- \
  job.name=action_fdm_open_h_smoke_1node \
  job.wandb_mode="$WANDB_MODE" \
  trainer.max_iter=50 \
  checkpoint.save_iter=50 \
  model.config.parallelism.data_parallel_shard_degree=8
