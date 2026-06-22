#!/usr/bin/env bash
# Set up the cosmos3-h-s-s workspace for finetuning:
#   1. Create runtime directories
#   2. Install uv (if absent)
#   3. Clone cosmos-framework into packages/cosmos3 (if absent)
#   4. Install framework dependencies (cu130-train or cu128-train)
#
# Run once from the repo root on a login node (no GPU required for setup):
#   source env.sh && bash scripts/setup_workspace.sh
#
# Prerequisites: env.sh must be sourced so WORKSPACE, UV_CACHE_DIR,
# UV_PYTHON_INSTALL_DIR, and COSMOS3_UV_GROUP are set.

set -euo pipefail

WORKSPACE="${WORKSPACE:?source env.sh first}"
UV_CACHE_DIR="${UV_CACHE_DIR:?source env.sh first}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:?source env.sh first}"
COSMOS3_UV_GROUP="${COSMOS3_UV_GROUP:-cu130-train}"

echo "=== cosmos3-h-s-s workspace setup ==="
echo "WORKSPACE=$WORKSPACE"
echo "COSMOS3_UV_GROUP=$COSMOS3_UV_GROUP"

# 1. Runtime directories (workspace dirs are gitignored; output root is on lustre)
mkdir -p \
    "$WORKSPACE/packages" \
    "$WORKSPACE/logs" \
    "$WORKSPACE/.cache" \
    "${IMAGINAIRE_OUTPUT_ROOT:?source env.sh first}" \
    "$UV_CACHE_DIR" \
    "$UV_PYTHON_INSTALL_DIR" \
    "$HOST_HOME/.local/share/uv"

# 2. uv
if ! command -v uv >/dev/null 2>&1; then
    echo "--- Installing uv ---"
    python3 -m pip install --user uv
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv $(uv --version)"

# 3. cosmos-framework
COSMOS3_DIR="$WORKSPACE/packages/cosmos3"
if [[ ! -d "$COSMOS3_DIR/.git" ]]; then
    echo "--- Cloning cosmos-framework ---"
    export GIT_LFS_SKIP_SMUDGE=1
    git clone https://github.com/NVIDIA/cosmos-framework.git "$COSMOS3_DIR"
fi

# 4. Install framework dependencies
echo "--- Running uv sync (group=$COSMOS3_UV_GROUP) ---"
cd "$COSMOS3_DIR"
export GIT_LFS_SKIP_SMUDGE=1
uv sync --all-extras --group="$COSMOS3_UV_GROUP"

echo ""
echo "=== Setup complete ==="
echo "Framework venv: $COSMOS3_DIR/.venv"
echo "Activate with:  source $COSMOS3_DIR/.venv/bin/activate"
echo ""
echo "Next: submit a training job with scripts/slurm_train.sbatch"
