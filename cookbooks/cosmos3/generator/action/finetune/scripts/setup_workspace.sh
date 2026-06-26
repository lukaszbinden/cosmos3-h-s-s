#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Stage a Cosmos Framework checkout, overlay the Open-H 44D surgical
# forward-dynamics finetune patch, register the experiment, and (optionally)
# compute per-embodiment action stats.
#
# This mirrors the sean-cosmos3_surgical_fd cookbook's setup_eos_workspace.sh,
# but installs the cosmos3-internal-derived 44D registry/specs overlay
# (framework_patch/) instead of the 54D manifest stack, and registers the
# `action_fdm_open_h_sft_nano` experiment.
#
# Usage (from this cookbook dir):
#   bash scripts/setup_workspace.sh
#
# Override the CUDA group / container for older drivers:
#   COSMOS3_UV_GROUP=cu128-train \
#   COSMOS3_CONTAINER=docker://nvcr.io#nvidia/pytorch:25.06-py3 \
#   bash scripts/setup_workspace.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOKBOOK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Configurable paths (override via env) ---------------------------------
WORKSPACE="${WORKSPACE:-$HOME/cosmos3-h-s-s-workspace}"
# Public Open-H-Embodiment surgical tree (the folder names the cluster
# currently exposes: cmr_surgical, hamlyn, jhu, obuda, stanford, tud, turin,
# ucberkeley, ucsd, virtual_incision, ...). Used to re-root the (B) specs.
OPENH_SURGICAL_ROOT="${OPENH_SURGICAL_ROOT:-/lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment/Surgical}"
# Experiment-specific stats-filename postfix. Stats live in the shared dataset
# meta/ dirs, so set a unique postfix to avoid colliding with other experiments'
# stats_cosmos*.json (e.g. a colleague's 54D run). The training loader and
# compute_openh_action_stats.py both read this var. Empty = legacy bare names.
COSMOS_OPENH_STATS_POSTFIX="${COSMOS_OPENH_STATS_POSTFIX:-c3hss-v1}"
COSMOS3_UV_GROUP="${COSMOS3_UV_GROUP:-cu130-train}"
COSMOS3_REPO_URL="${COSMOS3_REPO_URL:-https://github.com/NVIDIA/cosmos-framework.git}"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$WORKSPACE/checkpoints/Cosmos3-Nano}"
WAN_VAE_PATH="${WAN_VAE_PATH:-$WORKSPACE/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE/.uv-cache}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WORKSPACE/.uv-python}"
# Extra Python deps the ported gr00t_dreams stack needs that are not in the
# cosmos-framework pyproject (albumentations: VideoCrop/VideoResize backends;
# imageio: debug video writer in dataset.py; dm-tree: utils/misc.py).
COSMOS3_EXTRA_DEPS="${COSMOS3_EXTRA_DEPS:-albumentations imageio dm-tree}"
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR

mkdir -p "$WORKSPACE"/{packages,configs,manifests,checkpoints,logs,outputs}
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$WORKSPACE/.cache"

# --- uv ---------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --user uv
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- Cosmos Framework checkout ---------------------------------------------
FRAMEWORK_DIR="$WORKSPACE/packages/cosmos3"
if [[ ! -d "$FRAMEWORK_DIR/.git" ]]; then
    git clone "$COSMOS3_REPO_URL" "$FRAMEWORK_DIR"
fi

cd "$FRAMEWORK_DIR"
export GIT_LFS_SKIP_SMUDGE=1
uv sync --all-extras --group="$COSMOS3_UV_GROUP"

# --- Overlay the framework patch + register experiment + extra deps --------
# Delegated to apply_overlay.sh (single source of truth for the overlay step;
# re-run it standalone any time framework_patch/ changes). It rsyncs the 44D
# Open-H surgical data stack + experiment config into the checkout, registers
# the experiment in config.py, and installs the extra Python deps. Activate the
# checkout venv first so its internal `python` calls (import-verification) and
# `uv pip install` target this environment.
# shellcheck disable=SC1091
source "$FRAMEWORK_DIR/.venv/bin/activate"
COSMOS3_EXTRA_DEPS="$COSMOS3_EXTRA_DEPS" \
    bash "$SCRIPT_DIR/apply_overlay.sh" --framework-dir "$FRAMEWORK_DIR"

# --- Stage the TOML --------------------------------------------------------
cp "$COOKBOOK_DIR/toml/sft_config/action_fdm_open_h_sft_nano.toml" \
    "$WORKSPACE/configs/action_fdm_open_h_sft_nano.toml"

# --- Wan2.2 VAE ------------------------------------------------------------
if [[ ! -f "$WAN_VAE_PATH" ]]; then
    mkdir -p "$(dirname "$WAN_VAE_PATH")"
    uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
        --local-dir "$(dirname "$WAN_VAE_PATH")"
fi

# --- Persist env for the launchers -----------------------------------------
cat > "$WORKSPACE/env.sh" <<EOF
export WORKSPACE="$WORKSPACE"
export OPENH_SURGICAL_ROOT="$OPENH_SURGICAL_ROOT"
export BASE_CHECKPOINT_PATH="$BASE_CHECKPOINT_PATH"
export WAN_VAE_PATH="$WAN_VAE_PATH"
export UV_CACHE_DIR="$UV_CACHE_DIR"
export UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR"
export IMAGINAIRE_OUTPUT_ROOT="$WORKSPACE/outputs/train"
# Experiment stats postfix — MUST match what compute_openh_action_stats.py was
# run with (the loader reads stats_cosmos-<postfix>.json / -44D-<postfix>.json).
export COSMOS_OPENH_STATS_POSTFIX="$COSMOS_OPENH_STATS_POSTFIX"
# The specs already use absolute open-h-embodiment/Surgical paths, so a default
# run needs neither var. Set DATASET_PATH only to re-root ALL specs elsewhere
# (rebases by the full relative path under the surgical root).
# export DATASET_PATH="$OPENH_SURGICAL_ROOT"
EOF

echo ""
echo "============================================================"
echo "Prepared $WORKSPACE"
echo ""
echo "NEXT STEPS (before a production run):"
echo "  1. Stage the Cosmos3-Nano DCP base checkpoint at:"
echo "       $BASE_CHECKPOINT_PATH"
echo "     (python -m cosmos_framework.scripts.convert_model_to_dcp ...)"
echo "  2. Audit the newly added (B) Open-H leaves and FIX any schema"
echo "     mismatches in groot_configs.py:"
echo "       python $COOKBOOK_DIR/scripts/audit_openh_action_schemas.py \\"
echo "           --root \$OPENH_SURGICAL_ROOT"
echo "  3. Compute per-embodiment action stats (writes postfixed"
echo "     meta/stats_cosmos-\$COSMOS_OPENH_STATS_POSTFIX.json and CMR"
echo "     meta/stats_cosmos-44D-\$COSMOS_OPENH_STATS_POSTFIX.json):"
echo "       COSMOS_OPENH_STATS_POSTFIX=$COSMOS_OPENH_STATS_POSTFIX \\"
echo "       python $COOKBOOK_DIR/scripts/compute_openh_action_stats.py \\"
echo "           --root \$OPENH_SURGICAL_ROOT --experiment-id <id>"
echo "  4. Build the CMR filtered-episode caches:"
echo "       python $COOKBOOK_DIR/scripts/compute_cmr_filtered_episodes_cache.py"
echo "  5. Launch: sbatch $COOKBOOK_DIR/scripts/slurm_train.sbatch"
echo "============================================================"
