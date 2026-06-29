#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Backfill FDS for EXISTING (earlier) checkpoints by submitting the eval job
# (slurm_eval_checkpoint.sbatch) for each requested iteration, in ASCENDING order
# and (by default) with a FIXED diffusion seed so FDS is apples-to-apples across
# checkpoints. Each eval logs fds/* to the SAME training wandb run at step=iter,
# so you get a clean FDS-over-training curve overlaying the loss.
#
# Use this for checkpoints that ALREADY exist; use watch_eval_checkpoints.sh for
# new ones going forward.
#
# Usage:
#   source env.sh
#   # explicit list:
#   bash scripts/backfill_fds.sh 1000 2000 3000 4000
#   # inclusive range start:stop[:step]  (step default = EVAL_STRIDE or 1000):
#   bash scripts/backfill_fds.sh 1000:4700:1000
#   # all available checkpoints at the stride (no args -> auto-discover):
#   bash scripts/backfill_fds.sh
#
# Key env (with defaults):
#   EVAL_STRIDE=1000              stride for range / auto-discover
#   FIXED_SEED=1                  1 = same diffusion seed for all (comparable FDS);
#                                 0 = each eval seeds from its own iter
#   EVAL_SAMPLE_ITERATION=1000000 the fixed seed value used when FIXED_SEED=1
#   EVAL_BATCHES / EVAL_N_VIZ_SAMPLE   #clips per eval (more = less noisy FDS)
#   SUBMIT_SLEEP=2               seconds between sbatch submits (gentle on the scheduler)
#   DRY_RUN=0                    1 = print what would be submitted, don't sbatch
#   ONLY_EXISTING=1             1 = skip iters with no checkpoint dir on disk
#
# NOTE on wandb step order: submit ascending (this script sorts) so the FDS curve
# is monotonic in step. Jobs still RUN in scheduler order, but each logs at its
# own step, so ordering is best-effort; the per-eval sample_metadata.json always
# has the numbers regardless.

set -euo pipefail

# Resolve WORKSPACE (repo root) like the other scripts.
if [[ -z "${WORKSPACE:-}" ]]; then
  _SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
FIXED_SEED="${FIXED_SEED:-1}"
EVAL_SAMPLE_ITERATION="${EVAL_SAMPLE_ITERATION:-1000000}"
SUBMIT_SLEEP="${SUBMIT_SLEEP:-2}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_EXISTING="${ONLY_EXISTING:-1}"

# --- Build the list of iterations to evaluate ------------------------------
iters=()
if [[ "$#" -eq 0 ]]; then
  # auto-discover: all iter_* dirs whose iter is divisible by EVAL_STRIDE
  if [[ ! -d "$CKPT_DIR" ]]; then
    echo "FATAL: no args and checkpoint dir not found: $CKPT_DIR" >&2; exit 64
  fi
  while IFS= read -r d; do
    it="$((10#${d#iter_}))"
    (( it % EVAL_STRIDE == 0 )) && iters+=("$it")
  done < <(ls -1 "$CKPT_DIR" 2>/dev/null | grep -E '^iter_[0-9]+$' | sort)
elif [[ "$#" -eq 1 && "$1" == *:* ]]; then
  # range start:stop[:step]
  IFS=':' read -r start stop step <<< "$1"
  step="${step:-$EVAL_STRIDE}"
  (( start > 0 && stop >= start && step > 0 )) || { echo "FATAL: bad range '$1'" >&2; exit 64; }
  for ((it = start; it <= stop; it += step)); do iters+=("$it"); done
else
  # explicit list of iters
  for a in "$@"; do iters+=("$((10#$a))"); done
fi

# sort ascending + unique
mapfile -t iters < <(printf '%s\n' "${iters[@]}" | sort -n -u)
[[ "${#iters[@]}" -gt 0 ]] || { echo "Nothing to do (no iterations resolved)." >&2; exit 0; }

echo "===== FDS backfill ====="
echo "  ckpt_dir=$CKPT_DIR"
echo "  iters=${iters[*]}"
echo "  fixed_seed=$FIXED_SEED (EVAL_SAMPLE_ITERATION=$EVAL_SAMPLE_ITERATION)  only_existing=$ONLY_EXISTING  dry_run=$DRY_RUN"

submitted=0
skipped=0
for it in "${iters[@]}"; do
  name="$(printf 'iter_%09d' "$it")"
  if [[ "$ONLY_EXISTING" == "1" ]]; then
    if [[ ! -f "$CKPT_DIR/$name/model/.metadata" ]]; then
      echo "[skip] $name: no complete checkpoint on disk"
      skipped=$((skipped + 1))
      continue
    fi
  fi
  # Per-iter env for the eval job.
  declare -a envvars=(CHECKPOINT_ITER="$it")
  if [[ "$FIXED_SEED" == "1" ]]; then
    envvars+=(EVAL_SAMPLE_ITERATION="$EVAL_SAMPLE_ITERATION")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] ${envvars[*]} sbatch $EVAL_SBATCH"
  else
    echo "[submit] FDS eval $name"
    env "${envvars[@]}" sbatch "$EVAL_SBATCH" || echo "[warn] sbatch failed for $name"
    submitted=$((submitted + 1))
    sleep "$SUBMIT_SLEEP"
  fi
done

echo "Done: submitted=$submitted skipped=$skipped (of ${#iters[@]} requested)."
echo "Watch results in wandb (fds/mean_l1, fds/mean_ssim, fds/l1_slope) + each eval's sample_metadata.json."
