#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# One-node, 10-iteration Draco smoke for CAMP Arm B. This reuses the production
# launch body so mounts, the 44D warm-start, H16 data path, packing, and model
# step are identical to the six-node experiment.
#
#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=batch_block1,batch_block3,batch_block4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=1:00:00
#SBATCH --job-name=healthcareeng_holoscan-cosmos3.camp-arm-b-h16-smoke
#SBATCH --output=logs/camp-arm-b-h16-smoke_%j.out
#SBATCH --error=logs/camp-arm-b-h16-smoke_%j.out

set -euo pipefail

if [[ -z "${WORKSPACE:-}" && -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/env.sh" ]]; then
    WORKSPACE="$SLURM_SUBMIT_DIR"
fi
if [[ -z "${WORKSPACE:-}" || ! -f "$WORKSPACE/env.sh" ]]; then
    echo "FATAL: submit from the c3-h-s-o-b_camp repo root after preparing env.sh." >&2
    exit 64
fi

TRAIN_LAUNCHER="$WORKSPACE/cookbooks/cosmos3/generator/action/finetune/scripts/slurm_train_camp_arm_b_draco.sbatch"
if [[ ! -f "$TRAIN_LAUNCHER" ]]; then
    echo "FATAL: Arm-B production launcher missing: $TRAIN_LAUNCHER" >&2
    exit 64
fi

export ARM_A_CHECKPOINT_PATH="${ARM_A_CHECKPOINT_PATH:-/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden/checkpoints/cosmos3-h-surgical-omni-base/iter_000015000}"
_ARM_A_CHECKPOINT_TAG="$(basename "$ARM_A_CHECKPOINT_PATH")"
export CAMP_ARM_B_RUN_NAME="${CAMP_ARM_B_SMOKE_RUN_NAME:-camp-arm-b-h16-smoke-${_ARM_A_CHECKPOINT_TAG}}"
export CAMP_ARM_B_RUN_GROUP="${CAMP_ARM_B_SMOKE_RUN_GROUP:-action_open_h_camp_smoke}"
export CAMP_ARM_B_MAX_ITER=10
export CAMP_ARM_B_SAVE_ITER=10
export WANDB_MODE=offline

exec bash "$TRAIN_LAUNCHER"
