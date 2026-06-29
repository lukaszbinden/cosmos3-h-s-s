#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Watch the training run's checkpoint dir and auto-submit the FDS eval job
# (slurm_eval_checkpoint.sbatch) for each NEW checkpoint at the configured stride
# (default every 1000 steps). Each eval samples test-split clips, computes FDS
# (L1/SSIM/slope), and pushes it to the SAME training wandb run (overlaying loss).
#
# Run this on the login node (it just polls + sbatches; cheap). Leave it running
# in tmux/screen alongside training:
#   source env.sh
#   bash scripts/watch_eval_checkpoints.sh
#
# Tunables (env):
#   EVAL_STRIDE     evaluate only iters divisible by this (default 1000)
#   POLL_SECONDS    how often to poll for new checkpoints (default 300)
#   MAX_CONCURRENT_EVAL  don't submit if >= this many eval jobs are queued/running (default 1)
#   START_FROM_ITER skip checkpoints below this iter (default 0)
# Stop with Ctrl-C.

set -euo pipefail

# Resolve WORKSPACE (repo root) like the sbatch scripts.
if [[ -z "${WORKSPACE:-}" ]]; then
  _SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # scripts/ -> .../finetune -> ... 6 up = repo root
  WORKSPACE="$(cd "$_SELF/../../../../../.." && pwd)"
fi
if [[ -f "$WORKSPACE/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$WORKSPACE/env.sh"
fi

COOKBOOK_DIR="${COOKBOOK_DIR:-$WORKSPACE/cookbooks/cosmos3/generator/action/finetune}"
EVAL_SBATCH="$COOKBOOK_DIR/scripts/slurm_eval_checkpoint.sbatch"
[[ -f "$EVAL_SBATCH" ]] || { echo "FATAL: eval sbatch not found: $EVAL_SBATCH" >&2; exit 64; }

CKPT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$WORKSPACE/outputs/train}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-$CKPT_ROOT/cosmos3_action_surgical/action_open_h/action_fdm_open_h_sft_nano}"
CKPT_DIR="$TRAIN_RUN_DIR/checkpoints"

EVAL_STRIDE="${EVAL_STRIDE:-1000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_CONCURRENT_EVAL="${MAX_CONCURRENT_EVAL:-1}"
START_FROM_ITER="${START_FROM_ITER:-0}"
EVAL_JOB_NAME="healthcareeng_holoscan-cosmos3.openh44d_eval"
STATE_FILE="${STATE_FILE:-$TRAIN_RUN_DIR/.fds_eval_submitted.txt}"

echo "===== FDS eval watcher ====="
echo "  ckpt_dir=$CKPT_DIR"
echo "  stride=$EVAL_STRIDE  poll=${POLL_SECONDS}s  max_concurrent=$MAX_CONCURRENT_EVAL  start_from=$START_FROM_ITER"
echo "  state_file=$STATE_FILE"
mkdir -p "$TRAIN_RUN_DIR"
touch "$STATE_FILE"

_already_submitted() { grep -qxF "$1" "$STATE_FILE"; }
_mark_submitted() { echo "$1" >> "$STATE_FILE"; }
_n_eval_jobs() { squeue -u "$USER" -h -n "$EVAL_JOB_NAME" 2>/dev/null | wc -l | tr -d ' '; }

while true; do
  if [[ -d "$CKPT_DIR" ]]; then
    # iter_000004000 -> 4000, sorted ascending
    for d in $(ls -1 "$CKPT_DIR" 2>/dev/null | grep -E '^iter_[0-9]+$' | sort); do
      iter="$((10#${d#iter_}))"
      (( iter < START_FROM_ITER )) && continue
      (( iter % EVAL_STRIDE != 0 )) && continue       # only the stride boundaries
      _already_submitted "$d" && continue
      # require the checkpoint to be complete (model/.metadata present)
      [[ -f "$CKPT_DIR/$d/model/.metadata" ]] || { echo "[skip] $d not finished writing yet"; continue; }
      # throttle concurrent eval jobs
      n="$(_n_eval_jobs)"
      if (( n >= MAX_CONCURRENT_EVAL )); then
        echo "[wait] $n eval job(s) active >= MAX_CONCURRENT_EVAL=$MAX_CONCURRENT_EVAL; will retry $d next poll"
        break
      fi
      echo "[submit] FDS eval for $d (iter $iter)"
      if CHECKPOINT_ITER="$iter" sbatch "$EVAL_SBATCH"; then
        _mark_submitted "$d"
      else
        echo "[warn] sbatch failed for $d; will retry next poll"
      fi
    done
  else
    echo "[wait] checkpoint dir not present yet: $CKPT_DIR"
  fi
  sleep "$POLL_SECONDS"
done
