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
COSMOS3_REPO_URL="${COSMOS3_REPO_URL:-https://github.com/NVIDIA/cosmos-framework.git}"
# Exact legacy data.vfm framework revision used by mm-C3-H-S-S-base on EOS.
# A deliberate override is allowed, but it must remain an immutable full SHA.
COSMOS3_FRAMEWORK_REF="${COSMOS3_FRAMEWORK_REF:-300faa14daab3910be9d303c31708a0a1d6e4371}"

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
    git clone "$COSMOS3_REPO_URL" "$COSMOS3_DIR"
fi

if [[ ! "$COSMOS3_FRAMEWORK_REF" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: COSMOS3_FRAMEWORK_REF must be a full 40-character lowercase SHA;" >&2
    echo "  got: $COSMOS3_FRAMEWORK_REF" >&2
    exit 1
fi
git -C "$COSMOS3_DIR" fetch --quiet origin \
    || echo "[framework] WARN: fetch failed (offline?); trying local objects"
if ! git -C "$COSMOS3_DIR" rev-parse --verify --quiet \
        "${COSMOS3_FRAMEWORK_REF}^{commit}" >/dev/null; then
    echo "ERROR: COSMOS3_FRAMEWORK_REF=$COSMOS3_FRAMEWORK_REF does not" >&2
    echo "  resolve to a commit in $COSMOS3_REPO_URL." >&2
    exit 1
fi
git -C "$COSMOS3_DIR" checkout --quiet --force --detach "$COSMOS3_FRAMEWORK_REF"
COSMOS3_FRAMEWORK_SHA="$(git -C "$COSMOS3_DIR" rev-parse HEAD)"
if [[ "$COSMOS3_FRAMEWORK_SHA" != "$COSMOS3_FRAMEWORK_REF" ]]; then
    echo "ERROR: framework checkout resolved to $COSMOS3_FRAMEWORK_SHA, expected" >&2
    echo "  $COSMOS3_FRAMEWORK_REF" >&2
    exit 1
fi
cat > "$WORKSPACE/packages/cosmos3-framework.lock" <<LOCKEOF
repo=$COSMOS3_REPO_URL
ref=$COSMOS3_FRAMEWORK_REF
sha=$COSMOS3_FRAMEWORK_SHA
LOCKEOF
echo "[framework] pinned to $COSMOS3_FRAMEWORK_SHA"

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
