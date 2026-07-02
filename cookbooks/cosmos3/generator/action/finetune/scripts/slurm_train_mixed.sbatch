#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Main resumable training job: Cosmos3-Nano Open-H multi-embodiment surgical
# MIXED-MODE mid-training under the unified 44D action space -> Cosmos-Surg-3-Base.
#
# Drives cosmos_framework.scripts.train against the staged
# action_mixed_open_h_sft_nano.toml (registered `action_mixed_open_h_sft_nano`
# experiment; res480, mixed FD+ID+policy modes, third_person_view, 44D). One
# Slurm task per node + torchrun --nproc_per_node=8 inside each (EOS/Pyxis
# convention). Re-running with the same job.name resumes from
# latest_checkpoint.txt.
#
#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=batch
# 6 nodes x 8 H100 = 48 GPUs, IDENTICAL to the FD run (matching GPUs gives the
# same per-GPU FSDP shard, avoiding the 32-GPU resume-OOM, and makes this a clean
# LR/token-cap mirror of FD for the H1 ablation). data_parallel_shard_degree is
# set to SLURM_NNODES*8 below, so it auto-follows this node count (=48). NOTE:
# 48 >= 3 (the number of action modes), as RankPartitionedDataLoader requires.
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:55:00
#SBATCH --constraint=h100
#SBATCH --exclusive
#SBATCH --job-name=healthcareeng_holoscan-cosmos3.mm-C3-H-S-S-base
# --- 4h-cap auto-chaining ---------------------------------------------------
# EOS `batch` caps wall-time at ~4h, but a 20k-step run needs much longer. Use a
# SLURM ARRAY that runs tasks SEQUENTIALLY (%1 = at most 1 concurrent), each ≤4h.
# The Cosmos Framework trainer auto-resumes from the run dir's
# latest_checkpoint.txt (run-dir checkpoint beats checkpoint.load_path), so each
# array task continues iter/optimizer/scheduler/EMA/dataloader from the previous
# one — no manual resubmission, no extra flags. 12 tasks x ~3.9h ≈ 47h budget;
# tasks past convergence exit immediately (target iter already reached).
# `--dependency=singleton` also serializes vs any other job of the same name.
# `--requeue` lets a preempted task come back. Sequential (%1) is REQUIRED:
# parallel tasks would race on checkpoint I/O in the shared run dir.
#SBATCH --array=0-11%1
#SBATCH --dependency=singleton
#SBATCH --requeue
# Per-array-task logs: %A=array-job-id, %a=array-task-index. Lands in
# $WORKSPACE/logs (WORKSPACE defaults to the repo root). NOTE: SLURM resolves
# this path at submit time relative to the submit CWD, so submit from the repo
# root (or pass --chdir). The script also re-derives WORKSPACE below.
#SBATCH --output=logs/mm-C3-H-S-S-base_%A_%a.out
#SBATCH --error=logs/mm-C3-H-S-S-base_%A_%a.out

set -euo pipefail
export TMPDIR=/tmp

# Resolve WORKSPACE (repo root: framework checkout at packages/cosmos3, env.sh at
# the root; all gitignored). Order of preference:
#   1. $WORKSPACE from the environment (set by sourcing env.sh before sbatch, and
#      propagated via --export=ALL) — this is how prior runs resolved it.
#   2. $SLURM_SUBMIT_DIR (the dir you ran `sbatch` from) if it contains env.sh.
# Do NOT derive from ${BASH_SOURCE[0]}: under SLURM the script body runs from a
# COPY (e.g. /var/spool/slurm/d/jobNNN/slurm_script), so BASH_SOURCE-relative
# path math walks past '/' and yields '' or '/', making `source "$WORKSPACE/env.sh"`
# become `source "//env.sh"` -> "No such file or directory" (the 5522803 failure).
if [[ -z "${WORKSPACE:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/env.sh" ]]; then
    WORKSPACE="$SLURM_SUBMIT_DIR"
  fi
fi
if [[ -z "${WORKSPACE:-}" || ! -f "$WORKSPACE/env.sh" ]]; then
  echo "FATAL: cannot locate WORKSPACE/env.sh." >&2
  echo "  WORKSPACE='${WORKSPACE:-}'  SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}'" >&2
  echo "  Fix: 'source env.sh' (sets WORKSPACE) before sbatch, OR submit from the" >&2
  echo "  repo root (so SLURM_SUBMIT_DIR has env.sh), OR pass WORKSPACE=... explicitly." >&2
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
OUTPUTS="$WORKSPACE/outputs/train"
WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "$WORKSPACE/logs" "$OUTPUTS" "$WORKSPACE/.cache" "$WORKSPACE/.uv-cache" "$WORKSPACE/.uv-python" "$HOST_HOME/.local/share/uv"

# --- Startup banner + resume status ----------------------------------------
# Report whether this array task will RESUME (latest_checkpoint.txt present) or
# COLD-START from the base ckpt. The run dir is rooted at the HOST
# IMAGINAIRE_OUTPUT_ROOT that env.sh exports (checkpoints actually land there;
# verified iter_* dirs under it), NOT $OUTPUTS. (Earlier this checked $OUTPUTS
# and always mis-reported COLD-START even while the trainer correctly resumed
# from $IMAGINAIRE_OUTPUT_ROOT.) Fall back to $OUTPUTS only if the env var is
# somehow unset. project/group/name come from the experiment config.
_CKPT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUTS}"
# Run dir = project/group/name from the MIXED experiment config. name is
# mm-C3-H-S-S-base (set in the toml/experiment), distinct from the FD run's dir
# (action_fdm_open_h_sft_nano), so the two never collide / cross-resume.
_RUN_DIR="$_CKPT_ROOT/cosmos3_action_surgical/action_open_h/mm-C3-H-S-S-base"
_LATEST_CKPT_FILE="$_RUN_DIR/checkpoints/latest_checkpoint.txt"
echo "===== SLURM TRAIN START $(date -Is) ====="
echo "  job_id=${SLURM_JOB_ID:-?}  array_job=${SLURM_ARRAY_JOB_ID:-NA}  array_task=${SLURM_ARRAY_TASK_ID:-NA}"
echo "  nodes=${SLURM_NNODES:-?}  job_name=${SLURM_JOB_NAME:-?}  workspace=$WORKSPACE"
echo "  run_dir=$_RUN_DIR"
if [[ -f "$_LATEST_CKPT_FILE" ]]; then
  echo "  RESUME: latest_checkpoint.txt -> $(cat "$_LATEST_CKPT_FILE" 2>/dev/null) (base ckpt ignored)"
else
  echo "  COLD-START: no latest_checkpoint.txt yet -> warm-starts from BASE_CHECKPOINT_PATH"
fi
echo "  log: $WORKSPACE/logs/mm-C3-H-S-S-base_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.out"

# --- Pre-flight: the two artifacts the run REQUIRES on the host ------------
# Fail fast (before burning a 6-node allocation) with a clear message.
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

# In-container mount point for this cookbook, so the inner launcher
# ($COOKBOOK_DIR/scripts/_eos_torchrun_inner.sh) + apply_overlay.sh are reachable
# INSIDE the container. Without this, srun is handed a HOST path that doesn't
# exist in the container's mount namespace -> "execve(): ... No such file or
# directory".
COOKBOOK_MOUNT="/cookbook"

# Bind-mounts: the Open-H surgical data root must be visible inside the
# container at the same path the specs reference (OPENH_SURGICAL_ROOT and/or
# the fs11 absolute paths). Adjust the first mount for your cluster.
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

# W&B auth: wandb runs INSIDE the container (whose $HOME is /root), so it looks
# for /root/.netrc. Mount the host ~/.netrc there (read-only) so online logging
# can authenticate without baking a token into env/git. Only mount if present
# (an absent source path makes enroot fail). Requires WANDB_MODE=online (set via
# env / the --export below); the TOML's wandb_mode is overridden by job.wandb_mode.
if [[ -f "$HOST_HOME/.netrc" ]]; then
  MOUNTS="$MOUNTS,$HOST_HOME/.netrc:/root/.netrc:ro"
else
  echo "WARN: $HOST_HOME/.netrc not found -> wandb online auth will fail; logging will be offline/anonymous." >&2
fi

# Pyxis warmup (pull image once).
srun --container-image="$CONTAINER_IMAGE" bash -c 'echo pyxis-warmup-ok on $(hostname)' || true

# Inner launcher referenced at its IN-CONTAINER path ($COOKBOOK_MOUNT); overlay
# script pointed at the same mount.
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$MOUNTS" \
  --container-workdir=/workspace \
  --export=ALL,MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$MASTER_PORT",BASE_CHECKPOINT_PATH=/ckpt,WAN_VAE_PATH="$WAN_VAE_MOUNT",IMAGINAIRE_OUTPUT_ROOT=/outputs,WANDB_MODE="$WANDB_MODE",OPENH_SURGICAL_ROOT="${OPENH_SURGICAL_ROOT:-}",DATASET_PATH="${DATASET_PATH:-}",COSMOS_OPENH_STATS_POSTFIX="${COSMOS_OPENH_STATS_POSTFIX:-}",COSMOS_OPENH_DEBUG_INIT="${COSMOS_OPENH_DEBUG_INIT:-0}",COSMOS3_APPLY_OVERLAY="$COOKBOOK_MOUNT/scripts/apply_overlay.sh" \
  "$COOKBOOK_MOUNT/scripts/_eos_torchrun_inner.sh" \
  -m cosmos_framework.scripts.train \
  --sft-toml="$COOKBOOK_MOUNT/toml/sft_config/action_mixed_open_h_sft_nano.toml" \
  -- \
  job.wandb_mode="$WANDB_MODE" \
  model.config.parallelism.data_parallel_shard_degree="$((SLURM_NNODES * 8))"
