#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Out-of-band, single-GPU evaluation of an Open-H 44D training checkpoint:
# generates generated-vs-ground-truth comparison videos from the held-out TEST
# split (+ sample_metadata.json), via generate_checkpoint_samples_openh.py.
#
# Runs SEPARATELY from the training job (reads a chosen checkpoint, writes to a
# separate eval output dir) -- safe to run while training continues. Mirrors
# sean-cosmos3_surgical_fd/cookbook/scripts/slurm_eval_checkpoint.sbatch, adapted
# to this repo's overlay / paths / gr00t dataset.
#
# Pick the checkpoint via CHECKPOINT_ITER (default reads latest) or CHECKPOINT_PATH:
#   CHECKPOINT_ITER=4000 sbatch scripts/slurm_eval_checkpoint.sbatch
#   CHECKPOINT_PATH=/path/to/iter_000004000 sbatch scripts/slurm_eval_checkpoint.sbatch
#
#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:30:00
#SBATCH --constraint=h100
#SBATCH --exclusive
#SBATCH --job-name=healthcareeng_holoscan-cosmos3.openh44d_eval
#SBATCH --output=logs/openh44d_eval_%j.out
#SBATCH --error=logs/openh44d_eval_%j.out

set -euo pipefail
export TMPDIR=/tmp

# Resolve WORKSPACE (repo root). Prefer $WORKSPACE from env (sourced env.sh),
# else $SLURM_SUBMIT_DIR if it has env.sh. Do NOT use ${BASH_SOURCE[0]} (under
# SLURM the body runs from a copy -> "//env.sh"). Mirrors slurm_train.sbatch.
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

COOKBOOK_DIR="${COOKBOOK_DIR:-$WORKSPACE/cookbooks/cosmos3/generator/action/finetune}"
if [[ ! -d "$COOKBOOK_DIR/scripts" ]]; then
  echo "FATAL: cookbook dir not found at '$COOKBOOK_DIR' (WORKSPACE='$WORKSPACE')." >&2
  exit 64
fi

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 30000))}"
CONTAINER_IMAGE="${COSMOS3_CONTAINER:-docker://nvcr.io#nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04}"
HOST_HOME="${HOST_HOME:-$HOME}"

# --- Resolve which checkpoint to evaluate ----------------------------------
# Training writes checkpoints under the HOST IMAGINAIRE_OUTPUT_ROOT (env.sh),
# project/group/name from the experiment config.
CKPT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$WORKSPACE/outputs/train}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-$CKPT_ROOT/cosmos3_action_surgical/action_open_h/action_fdm_open_h_sft_nano}"
if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
  if [[ -n "${CHECKPOINT_ITER:-}" ]]; then
    CHECKPOINT_NAME="$(printf 'iter_%09d' "$CHECKPOINT_ITER")"
  else
    # default: the latest checkpoint the trainer recorded
    _LATEST="$TRAIN_RUN_DIR/checkpoints/latest_checkpoint.txt"
    if [[ -f "$_LATEST" ]]; then
      CHECKPOINT_NAME="$(cat "$_LATEST" | tr -d '[:space:]')"
      CHECKPOINT_ITER="${CHECKPOINT_NAME#iter_}"; CHECKPOINT_ITER="$((10#${CHECKPOINT_ITER:-0}))"
    else
      echo "FATAL: no CHECKPOINT_PATH/CHECKPOINT_ITER given and no $_LATEST. Set CHECKPOINT_ITER=N." >&2
      exit 64
    fi
  fi
  CHECKPOINT_PATH="$TRAIN_RUN_DIR/checkpoints/$CHECKPOINT_NAME"
fi
CHECKPOINT_ITER="${CHECKPOINT_ITER:-0}"
if [[ ! -d "$CHECKPOINT_PATH" ]]; then
  echo "FATAL: checkpoint dir not found: $CHECKPOINT_PATH" >&2
  exit 64
fi

EVAL_TAG="${EVAL_TAG:-iter_$(printf '%09d' "$CHECKPOINT_ITER")_$(date +%Y%m%d_%H%M%S)}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$CKPT_ROOT/eval}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$EVAL_OUTPUT_ROOT/$EVAL_TAG}"
mkdir -p "$WORKSPACE/logs" "$EVAL_OUTPUT_DIR" "$WORKSPACE/.cache" "$WORKSPACE/.uv-cache" "$WORKSPACE/.uv-python" "$HOST_HOME/.local/share/uv"

echo "===== SLURM EVAL START $(date -Is) ====="
echo "  checkpoint_path=$CHECKPOINT_PATH  iter=$CHECKPOINT_ITER"
echo "  eval_output_dir=$EVAL_OUTPUT_DIR"

WAN_VAE_MOUNT="/wan_vae/Wan2.2_VAE.pth"
COOKBOOK_MOUNT="/cookbook"

# Pre-flight (same artifacts the model needs).
if [[ -z "${WAN_VAE_PATH:-}" || ! -f "$WAN_VAE_PATH" ]]; then
  echo "FATAL: WAN_VAE_PATH not set or missing on host: '${WAN_VAE_PATH:-}'." >&2
  exit 64
fi

# FDS (Frame Decay Score: L1/SSIM/slope, gen-vs-GT) on the test split by default.
COMPUTE_FDS="${COMPUTE_FDS:-1}"
# Push aggregated FDS to the SAME training wandb run (overlays loss curves) when
# online. The training run's wandb id is at $TRAIN_RUN_DIR/wandb_id.txt (under the
# already-mounted lustre root, so its in-container path == host path).
WANDB_ID_FILE="${WANDB_ID_FILE:-$TRAIN_RUN_DIR/wandb_id.txt}"
EVAL_WANDB_MODE="${EVAL_WANDB_MODE:-${WANDB_MODE:-online}}"  # online -> FDS reaches wandb

DATA_ROOT_MOUNT="${DATA_ROOT_MOUNT:-/lustre/fsw/healthcareeng_holoscan:/lustre/fsw/healthcareeng_holoscan}"
MOUNTS="$DATA_ROOT_MOUNT"
MOUNTS="$MOUNTS,$WORKSPACE/packages/cosmos3:/workspace"
MOUNTS="$MOUNTS,$COOKBOOK_DIR:$COOKBOOK_MOUNT"
MOUNTS="$MOUNTS,$EVAL_OUTPUT_ROOT:/outputs_eval"
MOUNTS="$MOUNTS,$WAN_VAE_PATH:$WAN_VAE_MOUNT"
MOUNTS="$MOUNTS,$WORKSPACE/.cache:/root/.cache"
MOUNTS="$MOUNTS,$WORKSPACE/.uv-cache:$WORKSPACE/.uv-cache"
MOUNTS="$MOUNTS,$WORKSPACE/.uv-python:$WORKSPACE/.uv-python"
MOUNTS="$MOUNTS,$HOST_HOME/.local/share/uv:$HOST_HOME/.local/share/uv"
# W&B auth (online FDS push): mount ~/.netrc -> /root/.netrc (container $HOME).
if [[ -f "$HOST_HOME/.netrc" ]]; then
  MOUNTS="$MOUNTS,$HOST_HOME/.netrc:/root/.netrc:ro"
elif [[ "$EVAL_WANDB_MODE" == "online" ]]; then
  echo "WARN: $HOST_HOME/.netrc not found -> FDS wandb push will fail auth (falls back to offline/json-only)." >&2
fi
# CHECKPOINT_PATH + WANDB_ID_FILE are under the lustre data root (already mounted),
# so in-container paths equal host paths -- pass them through verbatim.

srun --container-image="$CONTAINER_IMAGE" bash -c 'echo pyxis-warmup-ok on $(hostname)' || true

FDS_ARGS=()
if [[ "$COMPUTE_FDS" == "1" ]]; then
  FDS_ARGS=(--compute-fds --wandb-id-file="$WANDB_ID_FILE")
fi

# 1-GPU eval: NPROC_PER_NODE=1 (the inner launcher defaults to 8 for training).
# WANDB_MODE here drives the FDS push (online to reach the training run); the
# in-script config still forces job.wandb_mode=offline so the SAMPLER doesn't
# open its own run -- only the explicit FDS wandb.init(resume) logs.
#
# FDS clip count = EVAL_BATCHES x EVAL_N_VIZ_SAMPLE (default 8 x 2 = 16 clips).
# To reduce FDS noise, raise EVAL_BATCHES (each batch is a DIFFERENT episode ->
# more diversity) in preference to EVAL_N_VIZ_SAMPLE (more samples of the same
# batch's episodes, and it also enlarges the comparison-video grid). Each clip is
# one diffusion sample (~3-7s warm), so 16 clips ~= 1-2 min of sampling per eval.
# e.g. EVAL_BATCHES=16 -> 32 clips for an even smoother curve.
srun --ntasks=1 --ntasks-per-node=1 \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$MOUNTS" \
  --container-workdir=/workspace \
  --export=ALL,MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$MASTER_PORT",NPROC_PER_NODE=1,WAN_VAE_PATH="$WAN_VAE_MOUNT",IMAGINAIRE_OUTPUT_ROOT=/outputs_eval,WANDB_MODE="$EVAL_WANDB_MODE",OPENH_SURGICAL_ROOT="${OPENH_SURGICAL_ROOT:-}",DATASET_PATH="${DATASET_PATH:-}",COSMOS_OPENH_STATS_POSTFIX="${COSMOS_OPENH_STATS_POSTFIX:-}",COSMOS_OPENH_DEBUG_INIT="${COSMOS_OPENH_DEBUG_INIT:-0}",COSMOS3_APPLY_OVERLAY="$COOKBOOK_MOUNT/scripts/apply_overlay.sh" \
  "$COOKBOOK_MOUNT/scripts/_eos_torchrun_inner.sh" \
  "$COOKBOOK_MOUNT/scripts/generate_checkpoint_samples_openh.py" \
  --sft-toml="$COOKBOOK_MOUNT/toml/sft_config/action_fdm_open_h_sft_nano.toml" \
  --checkpoint-path="$CHECKPOINT_PATH" \
  --iteration="$CHECKPOINT_ITER" \
  --output-dir="$EVAL_OUTPUT_DIR" \
  --num-frames="${EVAL_NUM_FRAMES:-13}" \
  --batches="${EVAL_BATCHES:-8}" \
  --n-viz-sample="${EVAL_N_VIZ_SAMPLE:-2}" \
  --num-sampling-step="${EVAL_NUM_SAMPLING_STEP:-16}" \
  --guidance="${EVAL_GUIDANCE:-1.5}" \
  --fps="${EVAL_FPS:-10}" \
  "${FDS_ARGS[@]}" \
  ${EVAL_SAMPLE_ITERATION:+--sample-iteration="$EVAL_SAMPLE_ITERATION"}
