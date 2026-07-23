#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# CAMP smoke on Draco (1 node, 8 GPUs): short mixed-mode training run to
# behaviorally verify the CAMP rails before real arm launches:
#
#   SMOKE_ARM=C (default)  H=16 history + '__random__' debug memory codes —
#                          exercises the 31-row packing question AND the
#                          Phase-3b padded-channel exemption end to end.
#                          Debug codes are deterministic junk: loss VALUES are
#                          meaningless; the smoke passes if training steps run
#                          (packing accepts 31 rows, no shape/mask crashes)
#                          and loss is finite.
#   SMOKE_ARM=B            H=16 only (28 rows, divisible packing) — the arm-B
#                          preflight.
#   SMOKE_ARM=A            no CAMP env at all — baseline control that the
#                          workspace itself is healthy.
#
# PREREQS (once, before first submit — see README "CAMP port handoff"):
#   1. Repo cloned under $LUSTRE_HOME, workspace prepared INSIDE the container
#      (interactive srun per the Draco runbook):
#          bash cookbooks/cosmos3/generator/action/finetune/scripts/setup_workspace.sh
#      (pins cosmos-framework to 300faa14, builds the venv, writes env.sh)
#   2. The trained 44D Arm-A checkpoint exists at $ARM_A_CHECKPOINT_PATH
#      (default: iter_000015000 below), and the Wan2.2 VAE is at $WAN_VAE_PATH.
#   3. Per-embodiment stats + CMR filter caches present (README "Pre-flight");
#      set OPENH_SURGICAL_ROOT / DATASET_PATH if the Open-H staging path on
#      Draco differs from the EOS paths baked into groot_configs.py.
#
# Submit from the REPO ROOT on a Draco login node:
#   mkdir -p logs && sbatch cookbooks/cosmos3/generator/action/finetune/scripts/slurm_camp_smoke_draco.sbatch
#   SMOKE_ARM=B SMOKE_ITERS=30 sbatch .../slurm_camp_smoke_draco.sbatch
#
#SBATCH -A healthcareeng_holoscan
#SBATCH --job-name=camp_smoke
#SBATCH --partition=batch_block1,batch_block3,batch_block4,backfill_block1,backfill_block3,backfill_block4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=01:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

export LUSTRE_HOME="${LUSTRE_HOME:-/lustre/fsw/portfolios/healthcareeng/users/${USER}}"
export LUSTRE_TEAM_HOME="${LUSTRE_TEAM_HOME:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan}"
export WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR}}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-$LUSTRE_HOME/images/pytorch_image.sqsh}"
export ARM_A_CHECKPOINT_PATH="${ARM_A_CHECKPOINT_PATH:-/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden/checkpoints/cosmos3-h-surgical-omni-base/iter_000015000}"
export COSMOS_OPENH_CMR_ROOT="${COSMOS_OPENH_CMR_ROOT:-/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/datasets/Open-H/cmr-surgical-60hz-fixed}"

# Keep every cache off /home (Draco runbook).
export UV_CACHE_DIR="${UV_CACHE_DIR:-$LUSTRE_HOME/.cache/uv}"
export TMPDIR="${TMPDIR:-$LUSTRE_HOME/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$LUSTRE_HOME/.cache}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-$TORCH_HOME/kernels}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$XDG_CACHE_HOME/nv/ComputeCache}"
mkdir -p logs "$UV_CACHE_DIR" "$TMPDIR" "$TORCH_HOME" "$PYTORCH_KERNEL_CACHE_PATH" "$CUDA_CACHE_PATH"
if [[ ! -f "$ARM_A_CHECKPOINT_PATH/model/.metadata" ]]; then
    echo "ERROR: Arm-A checkpoint is missing model/.metadata: $ARM_A_CHECKPOINT_PATH" >&2
    exit 64
fi
if [[ ! -d "$COSMOS_OPENH_CMR_ROOT/cholecystectomy" ]]; then
    echo "ERROR: Draco CMR mirror is missing: $COSMOS_OPENH_CMR_ROOT" >&2
    exit 64
fi
ARM_A_CHECKPOINT_TAG="$(basename "$ARM_A_CHECKPOINT_PATH")"
ARM_A_MODEL_METADATA_SHA256="$(sha256sum "$ARM_A_CHECKPOINT_PATH/model/.metadata" | awk '{print $1}')"

# Arm selection (see header). Exported into the container environment below.
SMOKE_ARM="${SMOKE_ARM:-C}"
SMOKE_ITERS="${SMOKE_ITERS:-50}"
case "$SMOKE_ARM" in
    A) CAMP_H=""; CAMP_MEM="" ;;
    B) CAMP_H="16"; CAMP_MEM="" ;;
    C) CAMP_H="16"; CAMP_MEM="__random__" ;;
    *) echo "ERROR: SMOKE_ARM must be A, B, or C; got '$SMOKE_ARM'" >&2; exit 2 ;;
esac
echo "[camp-smoke] arm=$SMOKE_ARM iters=$SMOKE_ITERS work_dir=$WORK_DIR"
echo "[camp-smoke] arm_a_checkpoint=$ARM_A_CHECKPOINT_PATH"
echo "[camp-smoke] cmr_root=$COSMOS_OPENH_CMR_ROOT"
echo "[camp-smoke] arm_a_model_metadata_sha256=$ARM_A_MODEL_METADATA_SHA256"
echo "[camp-smoke] WARMSTART_DECISION=WARMSTART_ARM_A_ONCE"
echo "[camp-smoke] LOAD_ALL_44D_ACTION_HEADS=true"
echo "[camp-smoke] LOAD_ARM_A_TRAINING_STATE=false"

srun --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$LUSTRE_HOME:$LUSTRE_HOME,$LUSTRE_TEAM_HOME:$LUSTRE_TEAM_HOME,$COSMOS_OPENH_CMR_ROOT:$COSMOS_OPENH_CMR_ROOT:ro,$ARM_A_CHECKPOINT_PATH:/arm_a_checkpoint:ro,/tmp:/tmp,/dev/shm:/dev/shm" \
  bash -c "
    set -euo pipefail
    cd '$WORK_DIR'

    # Workspace env (WORKSPACE, BASE_CHECKPOINT_PATH, WAN_VAE_PATH, stats
    # postfix, pinned-framework SHA) — written by setup_workspace.sh.
    source ./env.sh
    # Arms A/B/C all branch from the same trained 44D Arm-A checkpoint. Do not
    # inherit env.sh's public 64D Cosmos3-Nano BASE_CHECKPOINT_PATH.
    export BASE_CHECKPOINT_PATH=/arm_a_checkpoint
    export COSMOS_OPENH_CMR_ROOT='$COSMOS_OPENH_CMR_ROOT'

    # Framework venv built by setup_workspace.sh (uv sync inside the container).
    source \"\$WORKSPACE/packages/cosmos3/.venv/bin/activate\"

    # Job-start overlay re-stamp: the running env must match framework_patch/.
    bash '$WORK_DIR/cookbooks/cosmos3/generator/action/finetune/scripts/apply_overlay.sh' \
        --framework-dir \"\$WORKSPACE/packages/cosmos3\" --no-deps

    # Fast red/green gate before burning GPU time: the full local suite,
    # including the stale-overlay guard and the 3b diff-discipline tests.
    COSMOS3_FRAMEWORK_DIR=\"\$WORKSPACE/packages/cosmos3\" \
        python -m pytest '$WORK_DIR/cookbooks/cosmos3/generator/action/finetune/tests/' -q

    # Arm env (empty = unset, matching arm A semantics).
    if [ -n '$CAMP_H' ]; then export COSMOS_OPENH_NUM_HISTORY_ACTIONS='$CAMP_H'; fi
    if [ -n '$CAMP_MEM' ]; then export COSMOS_OPENH_CAMP_MEMORY_TRACKS='$CAMP_MEM'; fi
    # Global config registration imports the FD module too. This explicit marker
    # permits that import while retaining the FD launcher's fail-closed guard.
    export COSMOS_OPENH_MIXED_EXPERIMENT_WITH_CAMP=1
    echo '[camp-smoke] MODEL_LOAD_SOURCE=/arm_a_checkpoint'
    echo '[camp-smoke] ARM_A_MODEL_METADATA_SHA256=$ARM_A_MODEL_METADATA_SHA256'
    echo '[camp-smoke] checkpoint.keys_to_skip_loading=[]'
    echo '[camp-smoke] checkpoint.load_training_state=false'

    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
    export MASTER_PORT=\$((29500 + (\${SLURM_JOB_ID:-0} % 1000)))

    torchrun --nproc_per_node=8 --master_port=\${MASTER_PORT} \
      -m cosmos_framework.scripts.train \
      --sft-toml='$WORK_DIR/cookbooks/cosmos3/generator/action/finetune/toml/sft_config/action_mixed_open_h_sft_nano.toml' \
      -- \
      job.name=camp-smoke-arm$SMOKE_ARM-$ARM_A_CHECKPOINT_TAG \
      job.wandb_mode=offline \
      trainer.max_iter=$SMOKE_ITERS \
      trainer.logging_iter=10 \
      checkpoint.save_iter=1000000 \
      checkpoint.keys_to_skip_loading=[] \
      checkpoint.load_training_state=false \
      model.config.parallelism.data_parallel_shard_degree=8

    echo '[camp-smoke] ARM $SMOKE_ARM SMOKE OK'
  "
