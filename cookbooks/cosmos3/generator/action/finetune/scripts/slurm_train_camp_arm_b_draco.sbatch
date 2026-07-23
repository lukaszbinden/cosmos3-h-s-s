#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# CAMP Arm B on Draco: continue the trained 44D mixed-mode C3-H-S-S base with
# H=16 exact executed-action history and no learned-memory rows.
#
# Warm-start default:
#   /lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden/checkpoints/cosmos3-h-surgical-omni-base/iter_000015000
#
# Override ARM_A_CHECKPOINT_PATH only when a newer approved Arm-A checkpoint
# supersedes it. Then:
#   source env.sh
#   sbatch cookbooks/cosmos3/generator/action/finetune/scripts/slurm_train_camp_arm_b_draco.sbatch
#
# The base checkpoint MUST be the trained 44D mixed-mode checkpoint, not the
# public 64D Cosmos3-Nano checkpoint. All action heads are loaded.
#
#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=batch_block1,batch_block3,batch_block4
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH --job-name=healthcareeng_holoscan-cosmos3.camp-arm-b-h16
#SBATCH --array=0-11%1
#SBATCH --dependency=singleton
#SBATCH --requeue
#SBATCH --output=logs/camp-arm-b-h16_%A_%a.out
#SBATCH --error=logs/camp-arm-b-h16_%A_%a.out

set -euo pipefail
export TMPDIR=/tmp

# Resolve the checkout from which this script was submitted. Preserve it across
# sourcing env.sh in case that file was copied from another workspace.
if [[ -z "${WORKSPACE:-}" && -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/env.sh" ]]; then
    WORKSPACE="$SLURM_SUBMIT_DIR"
fi
if [[ -z "${WORKSPACE:-}" || ! -f "$WORKSPACE/env.sh" ]]; then
    echo "FATAL: submit from the c3-h-s-o-b_camp repo root after preparing env.sh." >&2
    echo "       WORKSPACE='${WORKSPACE:-}' SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}'" >&2
    exit 64
fi
_TARGET_WORKSPACE="$WORKSPACE"
source "$_TARGET_WORKSPACE/env.sh"
WORKSPACE="$_TARGET_WORKSPACE"
if [[ -f "$WORKSPACE/env.private.sh" ]]; then
    source "$WORKSPACE/env.private.sh"
fi

COOKBOOK_DIR="${COOKBOOK_DIR:-$WORKSPACE/cookbooks/cosmos3/generator/action/finetune}"
SFT_TOML="$COOKBOOK_DIR/toml/sft_config/action_mixed_open_h_sft_nano.toml"
INNER_LAUNCHER="$COOKBOOK_DIR/scripts/_draco_torchrun_inner.sh"
for required in "$SFT_TOML" "$INNER_LAUNCHER" "$COOKBOOK_DIR/scripts/apply_overlay.sh"; do
    if [[ ! -f "$required" ]]; then
        echo "FATAL: required Arm-B file missing: $required" >&2
        exit 64
    fi
done

# Arm identity and data semantics. Do not permit an ablation setting on the
# production Arm-B run.
CAMP_ARM_B_RUN_GROUP="${CAMP_ARM_B_RUN_GROUP:-action_open_h_camp}"
export COSMOS_OPENH_NUM_HISTORY_ACTIONS=16
if [[ -n "${COSMOS_OPENH_HISTORY_ABLATION:-}" ]]; then
    echo "FATAL: Arm B requires real history; unset COSMOS_OPENH_HISTORY_ABLATION." >&2
    exit 64
fi
unset COSMOS_OPENH_HISTORY_ABLATION

# Never fall back to env.sh's BASE_CHECKPOINT_PATH: it commonly points to the
# public 64D Nano checkpoint. Arms B/C must warm-start from this trained 44D
# Arm-A checkpoint and load all of its action heads.
ARM_A_CHECKPOINT_PATH="${ARM_A_CHECKPOINT_PATH:-/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden/checkpoints/cosmos3-h-surgical-omni-base/iter_000015000}"
CAMP_ARM_B_BASE_CHECKPOINT_PATH="${CAMP_ARM_B_BASE_CHECKPOINT_PATH:-$ARM_A_CHECKPOINT_PATH}"
if [[ -z "$CAMP_ARM_B_BASE_CHECKPOINT_PATH" || ! -d "$CAMP_ARM_B_BASE_CHECKPOINT_PATH" ]]; then
    echo "FATAL: CAMP_ARM_B_BASE_CHECKPOINT_PATH must name the trained 44D" >&2
    echo "       Arm-A DCP checkpoint directory." >&2
    echo "       Got: '${CAMP_ARM_B_BASE_CHECKPOINT_PATH:-}'" >&2
    exit 64
fi
if [[ ! -f "$CAMP_ARM_B_BASE_CHECKPOINT_PATH/model/.metadata" ]]; then
    echo "FATAL: incomplete DCP checkpoint: $CAMP_ARM_B_BASE_CHECKPOINT_PATH" >&2
    echo "       Expected model/.metadata in the iter checkpoint." >&2
    exit 64
fi
_ARM_A_CHECKPOINT_TAG="$(basename "$CAMP_ARM_B_BASE_CHECKPOINT_PATH")"
CAMP_ARM_B_RUN_NAME="${CAMP_ARM_B_RUN_NAME:-camp-arm-b-h16-${_ARM_A_CHECKPOINT_TAG}}"
if [[ -z "${WAN_VAE_PATH:-}" || ! -f "$WAN_VAE_PATH" ]]; then
    echo "FATAL: WAN_VAE_PATH is unset or missing: '${WAN_VAE_PATH:-}'." >&2
    exit 64
fi

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 30000))}"
DRACO_PROJECT_ROOT="${DRACO_PROJECT_ROOT:-/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan}"
DRACO_USERS_ROOT="${DRACO_USERS_ROOT:-/lustre/fs11/portfolios/healthcareeng/users}"
DRACO_USER_ROOT="${DRACO_USER_ROOT:-$DRACO_PROJECT_ROOT/users/$USER}"
DRACO_FSW_PROJECT_ALIAS="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan"
# Do not inherit setup_workspace.sh's EOS /lustre/fsw Open-H default. An
# alternate Draco staging root must be opted into explicitly with this var.
DRACO_OPENH_SURGICAL_ROOT="${DRACO_OPENH_SURGICAL_ROOT:-$DRACO_PROJECT_ROOT/datasets/Open-H/Surgical}"
OPENH_SURGICAL_ROOT="$DRACO_OPENH_SURGICAL_ROOT"
# Always re-root the EOS-authored specs onto Draco's canonical fs11 tree.
DATASET_PATH="$OPENH_SURGICAL_ROOT"
# Translate the canonical EOS relative leaves to Draco's case-sensitive
# internal staging layout (JHU/Imerse, Obuda/FRS_Dome_1, etc.).
COSMOS_OPENH_PATH_LAYOUT="draco_internal"
# Draco's clinical CMR mirror is outside the public Surgical tree. Use the
# unsuffixed original-resolution leaves; the data pipeline downsamples them
# once to the model's 832x480 training tier.
COSMOS_OPENH_CMR_ROOT="${COSMOS_OPENH_CMR_ROOT:-$DRACO_PROJECT_ROOT/datasets/Open-H/cmr-surgical-60hz-fixed}"
CONTAINER_IMAGE="${COSMOS3_CONTAINER:-$DRACO_USER_ROOT/images/cosmos3.sqsh}"
DRACO_NVML_PATH="${DRACO_NVML_PATH:-$DRACO_USER_ROOT/images/nvidia-pinned}"
HOST_HOME="${HOST_HOME:-$HOME}"
HF_HOME="${HF_HOME:-$DRACO_USER_ROOT/.hf_cache}"
WANDB_MODE="${WANDB_MODE:-offline}"
# Likewise, keep Arm B out of a stale EOS/local output root from env.sh.
OUTPUTS="${CAMP_ARM_B_OUTPUT_ROOT:-$DRACO_USER_ROOT/imaginaire/output}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$DRACO_USER_ROOT/.uv_cache}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WORKSPACE/.uv-python}"
COSMOS_OPENH_STATS_POSTFIX="${COSMOS_OPENH_STATS_POSTFIX:-c3hss-v1}"

if [[ ! -d "$OPENH_SURGICAL_ROOT" ]]; then
    echo "FATAL: Draco Open-H surgical root missing: $OPENH_SURGICAL_ROOT" >&2
    exit 64
fi
for cmr_leaf in \
    cholecystectomy \
    hysterectomy \
    inguinal_hernia \
    prostatectomy; do
    if [[ ! -d "$COSMOS_OPENH_CMR_ROOT/$cmr_leaf" ]]; then
        echo "FATAL: Draco CMR leaf missing: $COSMOS_OPENH_CMR_ROOT/$cmr_leaf" >&2
        exit 64
    fi
done
if [[ ! -f "$CONTAINER_IMAGE" ]]; then
    echo "FATAL: Draco Cosmos3 image missing: $CONTAINER_IMAGE" >&2
    exit 64
fi
if [[ ! -f "$DRACO_NVML_PATH/libnvidia-ml.so.1" ]]; then
    echo "FATAL: Draco pinned NVML missing: $DRACO_NVML_PATH/libnvidia-ml.so.1" >&2
    exit 64
fi
_QWEN_SNAPSHOTS="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"
if ! compgen -G "$_QWEN_SNAPSHOTS/*/config.json" >/dev/null \
   || ! compgen -G "$_QWEN_SNAPSHOTS/*/*.safetensors" >/dev/null; then
    echo "FATAL: Qwen3-VL-8B-Instruct is not fully staged under HF_HOME=$HF_HOME." >&2
    echo "       Draco compute nodes run offline; stage the model on a login node." >&2
    exit 64
fi

mkdir -p \
    "$WORKSPACE/logs" \
    "$OUTPUTS" \
    "$WORKSPACE/.cache" \
    "$UV_CACHE_DIR" \
    "$UV_PYTHON_INSTALL_DIR"

UV_CACHE_MOUNT_SOURCE="$(readlink -f "$UV_CACHE_DIR")"
UV_PYTHON_MOUNT_SOURCE="$(readlink -f "$UV_PYTHON_INSTALL_DIR")"
if [[ ! -d "$UV_PYTHON_MOUNT_SOURCE" ]]; then
    echo "FATAL: uv Python install directory missing: $UV_PYTHON_INSTALL_DIR" >&2
    exit 64
fi
VENV_PYTHON_LINK="$WORKSPACE/packages/cosmos3/.venv/bin/python"
if [[ ! -e "$VENV_PYTHON_LINK" ]]; then
    echo "FATAL: framework venv Python is unresolved: $VENV_PYTHON_LINK" >&2
    exit 64
fi
VENV_PYTHON_TARGET="$(readlink "$VENV_PYTHON_LINK" || true)"
VENV_UV_INSTALL_DIR=""
VENV_UV_MOUNT_SOURCE=""
if [[ "$VENV_PYTHON_TARGET" == /*/cpython-*/bin/python* ]]; then
    VENV_UV_INSTALL_DIR="${VENV_PYTHON_TARGET%%/cpython-*}"
    VENV_UV_MOUNT_SOURCE="$(readlink -f "$VENV_UV_INSTALL_DIR")"
fi

_RUN_DIR="$OUTPUTS/cosmos3_action_surgical/$CAMP_ARM_B_RUN_GROUP/$CAMP_ARM_B_RUN_NAME"
_LATEST_CKPT_FILE="$_RUN_DIR/checkpoints/latest_checkpoint.txt"
_WARMSTART_MARKER="$_RUN_DIR/arm_a_warmstart_attempt.env"
_ARM_A_METADATA_SHA256="$(sha256sum "$CAMP_ARM_B_BASE_CHECKPOINT_PATH/model/.metadata" | awk '{print $1}')"
mkdir -p "$_RUN_DIR"

# Exactly-once warm-start guard:
# - The first segment records an immutable attempt marker, then loads Arm A.
# - Once Arm B has written latest_checkpoint.txt, every later 4h segment resumes
#   Arm B's own complete training state and does not apply Arm A again.
# - A marker without an Arm-B checkpoint means the first attempt failed before
#   producing resumable state. Refuse to silently warm-start a second time; an
#   operator must inspect the first log and explicitly remove the marker.
if [[ -s "$_LATEST_CKPT_FILE" ]]; then
    CAMP_WARMSTART_DECISION="RESUME_ARM_B"
    _RESUME_TARGET="$(<"$_LATEST_CKPT_FILE")"
    if [[ -f "$_WARMSTART_MARKER" ]]; then
        _RECORDED_ARM_A_SHA="$(awk -F= '$1 == "arm_a_model_metadata_sha256" {print $2}' "$_WARMSTART_MARKER")"
        if [[ -n "$_RECORDED_ARM_A_SHA" && "$_RECORDED_ARM_A_SHA" != "$_ARM_A_METADATA_SHA256" ]]; then
            echo "FATAL: Arm-B resume provenance does not match the configured Arm-A checkpoint." >&2
            echo "       recorded_sha=$_RECORDED_ARM_A_SHA" >&2
            echo "       configured_sha=$_ARM_A_METADATA_SHA256" >&2
            exit 65
        fi
    else
        echo "WARN: Arm-B checkpoint exists without a warm-start marker (legacy run?)." >&2
    fi
elif [[ -e "$_WARMSTART_MARKER" ]]; then
    echo "FATAL: Arm-A warm-start was already attempted, but Arm B has no" >&2
    echo "       latest_checkpoint.txt. Refusing to warm-start a second time." >&2
    echo "       marker=$_WARMSTART_MARKER" >&2
    echo "       Inspect the first segment's log. If a deliberate retry from Arm A" >&2
    echo "       is required, remove the marker manually and resubmit." >&2
    exit 65
else
    CAMP_WARMSTART_DECISION="WARMSTART_ARM_A_ONCE"
    (
        set -o noclobber
        umask 027
        {
            printf 'created_at=%q\n' "$(date -Is)"
            printf 'slurm_job_id=%q\n' "${SLURM_JOB_ID:-unset}"
            printf 'slurm_array_job_id=%q\n' "${SLURM_ARRAY_JOB_ID:-unset}"
            printf 'slurm_array_task_id=%q\n' "${SLURM_ARRAY_TASK_ID:-unset}"
            printf 'arm_a_checkpoint=%q\n' "$CAMP_ARM_B_BASE_CHECKPOINT_PATH"
            printf 'arm_a_model_metadata_sha256=%q\n' "$_ARM_A_METADATA_SHA256"
            printf 'load_all_44d_heads=%q\n' "true"
            printf 'load_training_state=%q\n' "false"
        } > "$_WARMSTART_MARKER"
    ) || {
        echo "FATAL: could not atomically create warm-start marker: $_WARMSTART_MARKER" >&2
        exit 65
    }
fi

echo "===== CAMP ARM B (H16) TRAIN START $(date -Is) ====="
echo "  job_id=${SLURM_JOB_ID:-?} array_job=${SLURM_ARRAY_JOB_ID:-NA} array_task=${SLURM_ARRAY_TASK_ID:-NA}"
echo "  nodes=${SLURM_NNODES:-?} gpus=$((SLURM_NNODES * 8)) workspace=$WORKSPACE"
echo "  base_44d=$CAMP_ARM_B_BASE_CHECKPOINT_PATH"
echo "  base_model_metadata_sha256=$_ARM_A_METADATA_SHA256"
echo "  openh_root=$OPENH_SURGICAL_ROOT"
echo "  cmr_root=$COSMOS_OPENH_CMR_ROOT"
echo "  stats_postfix=$COSMOS_OPENH_STATS_POSTFIX"
echo "  history_actions=$COSMOS_OPENH_NUM_HISTORY_ACTIONS"
echo "  run_dir=$_RUN_DIR"
echo "  warmstart_marker=$_WARMSTART_MARKER"
if [[ -f "$_WARMSTART_MARKER" ]]; then
    while IFS= read -r provenance_line; do
        echo "  warmstart_origin.$provenance_line"
    done < "$_WARMSTART_MARKER"
fi
echo "  WARMSTART_DECISION=$CAMP_WARMSTART_DECISION"
if [[ "$CAMP_WARMSTART_DECISION" == "RESUME_ARM_B" ]]; then
    echo "  WARMSTART_CONFIRMED_BY_ARM_B_CHECKPOINT=true"
    echo "  RESUME_SOURCE=$_RESUME_TARGET"
    echo "  ARM_A_CHECKPOINT_REAPPLIED=false"
else
    echo "  WARMSTART_CONFIRMED_BY_ARM_B_CHECKPOINT=pending_first_save"
    echo "  MODEL_LOAD_SOURCE=$CAMP_ARM_B_BASE_CHECKPOINT_PATH"
    echo "  LOAD_ALL_44D_ACTION_HEADS=true (checkpoint.keys_to_skip_loading=[])"
    echo "  LOAD_ARM_A_TRAINING_STATE=false"
fi
trap 'rc=$?; echo "===== CAMP ARM B OUTER EXIT status=$rc decision=${CAMP_WARMSTART_DECISION:-unset} time=$(date -Is) ====="' EXIT

WAN_VAE_MOUNT="/wan_vae/Wan2.2_VAE.pth"
COOKBOOK_MOUNT="/cookbook"

MOUNTS="$DRACO_PROJECT_ROOT:$DRACO_PROJECT_ROOT"
MOUNTS="$MOUNTS,$DRACO_PROJECT_ROOT:$DRACO_FSW_PROJECT_ALIAS:ro"
MOUNTS="$MOUNTS,$DRACO_USERS_ROOT:$DRACO_USERS_ROOT:ro"
MOUNTS="$MOUNTS,$DRACO_NVML_PATH:/opt/nvidia-pinned:ro"
MOUNTS="$MOUNTS,$WORKSPACE/packages/cosmos3:/workspace"
MOUNTS="$MOUNTS,$COOKBOOK_DIR:$COOKBOOK_MOUNT"
MOUNTS="$MOUNTS,$OUTPUTS:/outputs"
MOUNTS="$MOUNTS,$CAMP_ARM_B_BASE_CHECKPOINT_PATH:/ckpt:ro"
MOUNTS="$MOUNTS,$WAN_VAE_PATH:$WAN_VAE_MOUNT:ro"
MOUNTS="$MOUNTS,$WORKSPACE/.cache:/root/.cache"
MOUNTS="$MOUNTS,$UV_CACHE_MOUNT_SOURCE:$UV_CACHE_DIR"
MOUNTS="$MOUNTS,$UV_PYTHON_MOUNT_SOURCE:$UV_PYTHON_INSTALL_DIR:ro"
if [[ -n "$VENV_UV_INSTALL_DIR" && "$VENV_UV_INSTALL_DIR" != "$UV_PYTHON_INSTALL_DIR" ]]; then
    MOUNTS="$MOUNTS,$VENV_UV_MOUNT_SOURCE:$VENV_UV_INSTALL_DIR:ro"
fi
if [[ -f "$HOST_HOME/.netrc" ]]; then
    MOUNTS="$MOUNTS,$HOST_HOME/.netrc:/root/.netrc:ro"
elif [[ "$WANDB_MODE" == "online" ]]; then
    echo "WARN: $HOST_HOME/.netrc missing; W&B online authentication may fail." >&2
fi

srun --no-container-mount-home --container-image="$CONTAINER_IMAGE" \
    bash -c 'echo pyxis-warmup-ok on $(hostname)' || true

TRAIN_OVERRIDES=(
    "job.name=$CAMP_ARM_B_RUN_NAME"
    "job.group=$CAMP_ARM_B_RUN_GROUP"
    "job.wandb_mode=$WANDB_MODE"
    "checkpoint.keys_to_skip_loading=[]"
    "checkpoint.load_training_state=false"
    "model.config.parallelism.data_parallel_shard_degree=$((SLURM_NNODES * 8))"
)
if [[ -n "${CAMP_ARM_B_MAX_ITER:-}" ]]; then
    TRAIN_OVERRIDES+=("trainer.max_iter=$CAMP_ARM_B_MAX_ITER")
fi
if [[ -n "${CAMP_ARM_B_SAVE_ITER:-}" ]]; then
    TRAIN_OVERRIDES+=("checkpoint.save_iter=$CAMP_ARM_B_SAVE_ITER")
fi

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    --no-container-mount-home \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace \
    --export=ALL,MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$MASTER_PORT",BASE_CHECKPOINT_PATH=/ckpt,WAN_VAE_PATH="$WAN_VAE_MOUNT",IMAGINAIRE_OUTPUT_ROOT=/outputs,WANDB_MODE="$WANDB_MODE",HF_HOME="$HF_HOME",OPENH_SURGICAL_ROOT="$OPENH_SURGICAL_ROOT",DATASET_PATH="$DATASET_PATH",COSMOS_OPENH_CMR_ROOT="$COSMOS_OPENH_CMR_ROOT",COSMOS_OPENH_PATH_LAYOUT="$COSMOS_OPENH_PATH_LAYOUT",COSMOS_OPENH_STATS_POSTFIX="$COSMOS_OPENH_STATS_POSTFIX",COSMOS_OPENH_NUM_HISTORY_ACTIONS=16,COSMOS_OPENH_DEBUG_INIT="${COSMOS_OPENH_DEBUG_INIT:-0}",COSMOS3_APPLY_OVERLAY="$COOKBOOK_MOUNT/scripts/apply_overlay.sh",CAMP_WARMSTART_DECISION="$CAMP_WARMSTART_DECISION",ARM_A_CHECKPOINT_SOURCE_HOST="$CAMP_ARM_B_BASE_CHECKPOINT_PATH",ARM_A_MODEL_METADATA_SHA256="$_ARM_A_METADATA_SHA256" \
    bash "$COOKBOOK_MOUNT/scripts/_draco_torchrun_inner.sh" \
    -m cosmos_framework.scripts.train \
    --sft-toml="$COOKBOOK_MOUNT/toml/sft_config/action_mixed_open_h_sft_nano.toml" \
    -- \
    "${TRAIN_OVERRIDES[@]}"
