#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Resubmit slurm_train.sbatch until TARGET_ITER is reached. EOS `batch` jobs
# cap at ~4h, and the experiment resumes from latest_checkpoint.txt, so this
# loops: submit -> wait -> check latest_checkpoint.txt advanced -> repeat.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-$HOME/cosmos3-h-s-s-workspace}"
# Default run dir follows IMAGINAIRE_OUTPUT_ROOT/{project}/{group}/{name}.
RUN_DIR="${RUN_DIR:-$WORKSPACE/outputs/train/cosmos3_action_surgical/action_open_h/action_fdm_open_h_sft_nano}"
TARGET_ITER="${TARGET_ITER:-20000}"
WAIT_FOR_JOB="${WAIT_FOR_JOB:-}"
MAX_NO_PROGRESS_FAILURES="${MAX_NO_PROGRESS_FAILURES:-1}"

current_iter() {
    local latest_file="$RUN_DIR/checkpoints/latest_checkpoint.txt"
    local latest iter
    if [[ ! -f "$latest_file" ]]; then
        echo 0
        return
    fi
    latest="$(cat "$latest_file")"
    iter="${latest#iter_}"
    iter="${iter%%/*}"
    if [[ "$iter" =~ ^[0-9]+$ ]]; then
        echo "$((10#$iter))"
    else
        echo 0
    fi
}

if [[ -n "$WAIT_FOR_JOB" ]]; then
    echo "Waiting for Slurm job $WAIT_FOR_JOB before starting training resubmits."
    while squeue -j "$WAIT_FOR_JOB" -h | grep -q .; do
        sleep 60
    done
fi

no_progress_failures=0

while true; do
    previous_iter="$(current_iter)"
    if (( previous_iter >= TARGET_ITER )); then
        echo "Target iteration reached: $previous_iter >= $TARGET_ITER"
        exit 0
    fi
    echo "Current checkpoint iter: $previous_iter (target $TARGET_ITER)"

    jobid="$(sbatch --parsable "$SCRIPT_DIR/slurm_train.sbatch")"
    echo "Submitted $jobid"
    while squeue -j "$jobid" -h | grep -q .; do
        sleep 60
    done

    state="$(sacct -j "$jobid" -X --format=State --parsable2 --noheader | head -n1 || true)"
    next_iter="$(current_iter)"
    echo "Job $jobid ended with state: ${state:-unknown}; checkpoint iter: $previous_iter -> $next_iter"

    if (( next_iter > previous_iter )); then
        no_progress_failures=0
    elif [[ "$state" != COMPLETED* && "$state" != TIMEOUT* ]]; then
        no_progress_failures=$((no_progress_failures + 1))
        if (( no_progress_failures > MAX_NO_PROGRESS_FAILURES )); then
            echo "Stopping after $no_progress_failures no-progress non-timeout failures."
            exit 1
        fi
    fi
done
