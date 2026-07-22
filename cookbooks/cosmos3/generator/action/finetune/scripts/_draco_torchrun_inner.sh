#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

echo "===== DRACO TORCHRUN INNER START host=$(hostname) time=$(date -Is) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset} SLURM_NODEID=${SLURM_NODEID:-unset} SLURM_NNODES=${SLURM_NNODES:-unset}"
echo "MASTER_ADDR=${MASTER_ADDR:-unset} MASTER_PORT=${MASTER_PORT:-unset}"
echo "COSMOS_OPENH_NUM_HISTORY_ACTIONS=${COSMOS_OPENH_NUM_HISTORY_ACTIONS:-unset}"
echo "COSMOS_OPENH_STATS_POSTFIX=${COSMOS_OPENH_STATS_POSTFIX:-unset}"
echo "CAMP_WARMSTART_DECISION=${CAMP_WARMSTART_DECISION:-unset}"
echo "ARM_A_CHECKPOINT_SOURCE_HOST=${ARM_A_CHECKPOINT_SOURCE_HOST:-unset}"
echo "ARM_A_MODEL_METADATA_SHA256=${ARM_A_MODEL_METADATA_SHA256:-unset}"
if [[ "${CAMP_WARMSTART_DECISION:-}" == "RESUME_ARM_B" ]]; then
    echo "ARM_A_CHECKPOINT_REAPPLIED=false (resuming Arm-B run state)"
elif [[ "${CAMP_WARMSTART_DECISION:-}" == "WARMSTART_ARM_A_ONCE" ]]; then
    echo "ARM_A_CHECKPOINT_LOAD_ATTEMPT=true (first segment only)"
    echo "ARM_A_CHECKPOINT_REAPPLIED=pending_model_load"
fi

# Draco compute nodes are offline. The squashfs image must already contain
# ffmpeg; trying apt-get here would only waste the allocation before failing.
if ! command -v ffmpeg >/dev/null; then
    echo "FATAL: ffmpeg is missing from the Draco squashfs image." >&2
    echo "       Rebuild/stage cosmos3.sqsh with ffmpeg before submitting." >&2
    exit 78
fi

export HOME=/root
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/outputs}"
export AOT_TOKENIZER_CACHE_DIR="${AOT_TOKENIZER_CACHE_DIR:-/outputs/aot_tokenizer_cache}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_TIMEOUT_MS="${NCCL_TIMEOUT_MS:-7200000}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export TORCHDYNAMO_SUPPRESS_ERRORS="${TORCHDYNAMO_SUPPRESS_ERRORS:-1}"
export TORCHINDUCTOR_PERSISTENT_REDUCTIONS="${TORCHINDUCTOR_PERSISTENT_REDUCTIONS:-0}"
export TORCHINDUCTOR_MIX_ORDER_REDUCTION="${TORCHINDUCTOR_MIX_ORDER_REDUCTION:-0}"
export TORCHINDUCTOR_COOPERATIVE_REDUCTIONS="${TORCHINDUCTOR_COOPERATIVE_REDUCTIONS:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$PYTORCH_CUDA_ALLOC_CONF}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_${USER}_${SLURM_JOB_ID:-local}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_${USER}_${SLURM_JOB_ID:-local}}"

if [[ ! -f /workspace/.venv/bin/activate ]]; then
    echo "FATAL: /workspace/.venv/bin/activate missing inside container." >&2
    exit 78
fi
source /workspace/.venv/bin/activate

# Draco's CUDA-13 image needs the cluster-version-pinned NVML library plus the
# image's forward-compat driver libraries. Do not add /usr/local/cuda/lib64:
# torch's uv environment supplies the CUDA runtime used by training.
if [[ ! -f /opt/nvidia-pinned/libnvidia-ml.so.1 ]]; then
    echo "FATAL: /opt/nvidia-pinned/libnvidia-ml.so.1 missing." >&2
    exit 78
fi
export LD_LIBRARY_PATH=/opt/nvidia-pinned:/usr/local/cuda/compat
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/compat:${LIBRARY_PATH:-}

# Stamp this branch's CAMP/Open-H overlay onto the pinned framework checkout on
# every node immediately before launch.
_INNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_APPLY_OVERLAY="${COSMOS3_APPLY_OVERLAY:-$_INNER_DIR/apply_overlay.sh}"
if [[ ! -f "$_APPLY_OVERLAY" ]]; then
    echo "FATAL: apply_overlay.sh not found at $_APPLY_OVERLAY." >&2
    exit 78
fi
bash "$_APPLY_OVERLAY" \
    --framework-dir "${COSMOS3_FRAMEWORK_DIR:-/workspace}" \
    --no-deps

python - <<'PY'
import torch
import torchcodec

print(
    "torch", torch.__version__,
    "torchcodec", torchcodec.__version__,
    "cuda_available", torch.cuda.is_available(),
    "device_count", torch.cuda.device_count(),
)
PY

exec torchrun \
    --nnodes="$SLURM_NNODES" \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="${NPROC_PER_NODE:-8}" \
    "$@"
