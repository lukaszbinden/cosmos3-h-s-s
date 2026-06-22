#!/usr/bin/env bash
#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:55:00
#SBATCH --constraint=h100
#SBATCH --exclusive
#SBATCH --job-name=cosmos3-h-s-s
#SBATCH --output=/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/git/cosmos3-h-s-s/logs/%x-%j.out

# Generic training launcher for cosmos3-h-s-s finetuning.
# Requires env.sh to be sourced or the key vars set before sbatch:
#
#   source env.sh
#   SFT_TOML=configs/my_config.toml sbatch scripts/slurm_train.sbatch
#
# Adjust --nodes to match 8 GPUs per node × desired GPU count
# (e.g. --nodes=6 for 48 GPUs with H100 80GB nodes).

set -euo pipefail
export TMPDIR=/tmp

WORKSPACE="${WORKSPACE:-/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/git/cosmos3-h-s-s}"
source "$WORKSPACE/env.sh"   # also sources env.private.sh (HF_TOKEN) if present

# SFT_TOML must be set before submission (absolute path or relative to WORKSPACE)
SFT_TOML="${SFT_TOML:?Set SFT_TOML=<path/to/config.toml> before sbatch}"

# HF_TOKEN is needed inside the container to fetch gated datasets/models.
# Fail fast with a clear message rather than 401-ing mid-run.
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "FATAL: HF_TOKEN is not set. Export it in ~/.bashrc, env.private.sh, or before sbatch." >&2
    exit 1
fi

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 30000))}"
CONTAINER_IMAGE="${COSMOS3_CONTAINER:-docker://nvcr.io#nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04}"

# IMAGINAIRE_OUTPUT_ROOT comes from env.sh (single source of truth). It lives
# under the lustre tree, which is mounted at the identical path in-container, so
# no remap is needed — the env var is passed straight through.
mkdir -p "$WORKSPACE/logs" "$IMAGINAIRE_OUTPUT_ROOT" "$WORKSPACE/.cache"

# Container mounts:
#   /lustre/...   — full lustre tree (data, checkpoints, IMAGINAIRE_OUTPUT_ROOT)
#   /workspace    — the cosmos-framework checkout (venv lives here)
#   /ckpt         — base DCP checkpoint (read-only)
#   /root/.cache  — HF model cache
MOUNTS="/lustre/fsw/healthcareeng_holoscan:/lustre/fsw/healthcareeng_holoscan"
MOUNTS="$MOUNTS,$WORKSPACE/packages/cosmos3:/workspace"
MOUNTS="$MOUNTS,$BASE_CHECKPOINT_PATH:/ckpt"
MOUNTS="$MOUNTS,$WORKSPACE/.cache:/root/.cache"
MOUNTS="$MOUNTS,$UV_CACHE_DIR:$UV_CACHE_DIR"
MOUNTS="$MOUNTS,$UV_PYTHON_INSTALL_DIR:$UV_PYTHON_INSTALL_DIR"
MOUNTS="$MOUNTS,$HOST_HOME/.local/share/uv:$HOST_HOME/.local/share/uv"

# Pyxis container warmup (pulls/caches the image on all nodes before training starts)
srun --container-image="$CONTAINER_IMAGE" bash -c 'echo pyxis-warmup-ok on $(hostname)' || true

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace \
    --export=ALL,MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$MASTER_PORT",BASE_CHECKPOINT_PATH=/ckpt,IMAGINAIRE_OUTPUT_ROOT="$IMAGINAIRE_OUTPUT_ROOT",HF_TOKEN="$HF_TOKEN" \
    "$WORKSPACE/scripts/_eos_torchrun_inner.sh" \
    -m cosmos_framework.scripts.train \
    --sft-toml="$SFT_TOML"
