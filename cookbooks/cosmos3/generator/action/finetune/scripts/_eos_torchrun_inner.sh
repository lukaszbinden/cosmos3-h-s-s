#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

echo "===== EOS TORCHRUN INNER START host=$(hostname) time=$(date -Is) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset} SLURM_NODEID=${SLURM_NODEID:-unset} SLURM_NNODES=${SLURM_NNODES:-unset}"
echo "MASTER_ADDR=${MASTER_ADDR:-unset} MASTER_PORT=${MASTER_PORT:-unset}"

if ! command -v ffmpeg >/dev/null; then
    apt-get update -qq
    apt-get install -y --no-install-recommends ffmpeg >/dev/null
    ldconfig
fi

unset HF_HUB_OFFLINE
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/.cache/huggingface/hub}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/outputs}"
export AOT_TOKENIZER_CACHE_DIR="${AOT_TOKENIZER_CACHE_DIR:-/outputs/aot_tokenizer_cache}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_TIMEOUT_MS="${NCCL_TIMEOUT_MS:-7200000}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export TORCHDYNAMO_SUPPRESS_ERRORS="${TORCHDYNAMO_SUPPRESS_ERRORS:-1}"
export TORCHINDUCTOR_PERSISTENT_REDUCTIONS="${TORCHINDUCTOR_PERSISTENT_REDUCTIONS:-0}"
export TORCHINDUCTOR_MIX_ORDER_REDUCTION="${TORCHINDUCTOR_MIX_ORDER_REDUCTION:-0}"
export TORCHINDUCTOR_COOPERATIVE_REDUCTIONS="${TORCHINDUCTOR_COOPERATIVE_REDUCTIONS:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$PYTORCH_CUDA_ALLOC_CONF}"

if [[ ! -f /workspace/.venv/bin/activate ]]; then
    echo "FATAL: /workspace/.venv/bin/activate missing inside container."
    exit 78
fi

source /workspace/.venv/bin/activate
# uv-created venvs carry the CUDA libraries we want; avoid host/container
# library-path leakage shadowing them.
export LD_LIBRARY_PATH=

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
try:
    import torchcodec
    print("torchcodec", torchcodec.__version__)
except Exception as exc:
    print("torchcodec_import_error", type(exc).__name__, exc)
    raise
PY

exec torchrun \
    --nnodes="$SLURM_NNODES" \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node=8 \
    "$@"
