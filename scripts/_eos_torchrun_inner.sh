#!/usr/bin/env bash
# Container entrypoint: activate the venv, verify torch+GPU, then exec torchrun.
# Called by slurm_train.sbatch via srun inside the pyxis container.
# /workspace is mounted from $WORKSPACE/packages/cosmos3 by the sbatch script.
#
# Usage (from sbatch):
#   srun ... _eos_torchrun_inner.sh -m cosmos_framework.scripts.train --sft-toml=...

set -euo pipefail

echo "===== EOS TORCHRUN INNER START host=$(hostname) time=$(date -Is) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset} SLURM_NODEID=${SLURM_NODEID:-unset} SLURM_NNODES=${SLURM_NNODES:-unset}"
echo "MASTER_ADDR=${MASTER_ADDR:-unset} MASTER_PORT=${MASTER_PORT:-unset}"

# Install system libs needed by guardrails / OpenCV if missing (bare CUDA image)
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "--- Installing system packages ---"
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        curl ffmpeg git-lfs libgl1 libglib2.0-0 libx11-dev libxcb1 tree wget >/dev/null
    ldconfig
fi

# HF and output roots. IMAGINAIRE_OUTPUT_ROOT is injected by the sbatch --export
# as a real lustre path (mounted identically in-container).
unset HF_HUB_OFFLINE
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/.cache/huggingface/hub}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:?IMAGINAIRE_OUTPUT_ROOT must be set by the launcher}"
export AOT_TOKENIZER_CACHE_DIR="${AOT_TOKENIZER_CACHE_DIR:-$IMAGINAIRE_OUTPUT_ROOT/aot_tokenizer_cache}"

# NCCL / PyTorch tuning
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

# Activate the framework venv (built by setup_workspace.sh on the host)
if [[ ! -f /workspace/.venv/bin/activate ]]; then
    echo "FATAL: /workspace/.venv/bin/activate not found inside container."
    echo "Run scripts/setup_workspace.sh on the host first."
    exit 78
fi
source /workspace/.venv/bin/activate
# Clear LD_LIBRARY_PATH so the container's bundled libs don't shadow the venv's torch
export LD_LIBRARY_PATH=

# Sanity-check torch + GPU visibility
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available(), "| device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  device {i}:", torch.cuda.get_device_name(i))
try:
    import torchcodec
    print("torchcodec:", torchcodec.__version__)
except Exception as exc:
    print("WARNING: torchcodec not available:", exc)
PY

exec torchrun \
    --nnodes="$SLURM_NNODES" \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node=8 \
    "$@"
