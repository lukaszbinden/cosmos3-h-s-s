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
WORKSPACE="${WORKSPACE:-$HOME/cosmos3_openh_surgical_fd}"
# Public Open-H-Embodiment surgical tree (the folder names the cluster
# currently exposes: cmr_surgical, hamlyn, jhu, obuda, stanford, tud, turin,
# ucberkeley, ucsd, virtual_incision, ...). Used to re-root the (B) specs.
OPENH_SURGICAL_ROOT="${OPENH_SURGICAL_ROOT:-/lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment/Surgical}"
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

# Extra deps for the ported surgical data stack.
if [[ -n "$COSMOS3_EXTRA_DEPS" ]]; then
    uv pip install $COSMOS3_EXTRA_DEPS
fi

# --- Overlay the framework patch -------------------------------------------
# Copies the 44D Open-H surgical data stack + experiment config into the
# framework checkout (cosmos_framework/data/vfm/action/{gr00t_dreams,
# open_h_dataset.py,domain_utils.py,datasets/openh_sft_dataset.py} and
# configs/.../action/posttrain_config/action_fdm_open_h_sft_nano.py).
rsync -a "$COOKBOOK_DIR/framework_patch/" "$FRAMEWORK_DIR/"

# --- Register the experiment in config.py ----------------------------------
"$FRAMEWORK_DIR/.venv/bin/python" - <<'PY'
from pathlib import Path

path = Path("cosmos_framework/configs/base/config.py")
text = path.read_text()
line = (
    "    import cosmos_framework.configs.base.experiment.action."
    "posttrain_config.action_fdm_open_h_sft_nano  # noqa: F401\n"
)
needle = (
    "    import cosmos_framework.configs.base.experiment.action."
    "posttrain_config.action_policy_droid_nano  # noqa: F401\n"
)
if line not in text:
    if needle not in text:
        raise SystemExit(f"Could not find experiment import insertion point in {path}")
    text = text.replace(needle, needle + line)
    path.write_text(text)
    print(f"Registered action_fdm_open_h_sft_nano in {path}")
else:
    print(f"action_fdm_open_h_sft_nano already registered in {path}")
PY

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
echo "  3. Compute per-embodiment action stats (writes meta/stats_cosmos.json"
echo "     and CMR meta/stats_cosmos-44D.json) and recompute (B) mix ratios:"
echo "       python $COOKBOOK_DIR/scripts/compute_openh_action_stats.py \\"
echo "           --root \$OPENH_SURGICAL_ROOT"
echo "  4. Build the CMR filtered-episode caches:"
echo "       python $COOKBOOK_DIR/scripts/compute_cmr_filtered_episodes_cache.py"
echo "  5. Launch: sbatch $COOKBOOK_DIR/scripts/slurm_train.sbatch"
echo "============================================================"
