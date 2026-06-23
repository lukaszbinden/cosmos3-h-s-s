#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Apply this cookbook's framework_patch/ overlay onto an installed
# cosmos_framework: rsync the 44D Open-H surgical data stack + experiment
# config, register the experiment in config.py, and install the extra Python
# deps the data stack needs.
#
# Use this whenever you change anything under framework_patch/ (e.g. the
# registry / dataset.py), or to fix a fresh venv that errors with
#   ModuleNotFoundError: No module named 'cosmos_framework.data.vfm.action.gr00t_dreams'
#
# The target cosmos_framework install is resolved in this order:
#   1. --framework-dir <DIR>           (DIR contains the 'cosmos_framework/' pkg)
#   2. $COSMOS3_FRAMEWORK_DIR          (same meaning)
#   3. auto-detect from the ACTIVE python (import cosmos_framework)
#   4. $WORKSPACE/packages/cosmos3     (the setup_workspace.sh checkout)
#
# Usage:
#   # overlay onto whatever cosmos_framework your current venv imports:
#   bash scripts/apply_overlay.sh
#
#   # overlay onto an explicit checkout:
#   bash scripts/apply_overlay.sh --framework-dir /path/to/cosmos-framework
#
#   # skip the extra-deps pip install (e.g. offline / already installed):
#   bash scripts/apply_overlay.sh --no-deps
#
#   # dry-run (show what rsync would copy, change nothing):
#   bash scripts/apply_overlay.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOKBOOK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PATCH_DIR="$COOKBOOK_DIR/framework_patch"

FRAMEWORK_DIR=""
INSTALL_DEPS=1
DRY_RUN=0
COSMOS3_EXTRA_DEPS="${COSMOS3_EXTRA_DEPS:-albumentations imageio dm-tree}"

# --- args ------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --framework-dir) FRAMEWORK_DIR="$2"; shift 2 ;;
        --framework-dir=*) FRAMEWORK_DIR="${1#*=}"; shift ;;
        --no-deps) INSTALL_DEPS=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -d "$PATCH_DIR/cosmos_framework" ]] || {
    echo "ERROR: overlay source not found: $PATCH_DIR/cosmos_framework" >&2
    exit 1
}

# --- resolve the target framework package root -----------------------------
# We want the directory that CONTAINS the 'cosmos_framework' package dir, so
# that rsync of framework_patch/cosmos_framework/... lands on <root>/cosmos_framework/...
resolve_from_python() {
    python - <<'PY' 2>/dev/null || true
import importlib.util, os
spec = importlib.util.find_spec("cosmos_framework")
if spec and spec.origin:
    # .../cosmos_framework/__init__.py -> parent of the package dir
    print(os.path.dirname(os.path.dirname(spec.origin)))
PY
}

if [[ -z "$FRAMEWORK_DIR" && -n "${COSMOS3_FRAMEWORK_DIR:-}" ]]; then
    FRAMEWORK_DIR="$COSMOS3_FRAMEWORK_DIR"
fi
if [[ -z "$FRAMEWORK_DIR" ]]; then
    FRAMEWORK_DIR="$(resolve_from_python)"
    [[ -n "$FRAMEWORK_DIR" ]] && echo "[info] auto-detected cosmos_framework via active python: $FRAMEWORK_DIR"
fi
if [[ -z "$FRAMEWORK_DIR" && -n "${WORKSPACE:-}" && -d "$WORKSPACE/packages/cosmos3/cosmos_framework" ]]; then
    FRAMEWORK_DIR="$WORKSPACE/packages/cosmos3"
    echo "[info] falling back to \$WORKSPACE checkout: $FRAMEWORK_DIR"
fi

[[ -n "$FRAMEWORK_DIR" ]] || {
    echo "ERROR: could not resolve a cosmos_framework install." >&2
    echo "  Activate the venv that has cosmos_framework, or pass --framework-dir <DIR>" >&2
    echo "  (DIR must contain the 'cosmos_framework/' package directory)." >&2
    exit 1
}
[[ -d "$FRAMEWORK_DIR/cosmos_framework" ]] || {
    echo "ERROR: $FRAMEWORK_DIR does not contain a 'cosmos_framework/' package dir." >&2
    echo "  Pass the directory that CONTAINS cosmos_framework/ (its parent)." >&2
    exit 1
}

CONFIG_PY="$FRAMEWORK_DIR/cosmos_framework/configs/base/config.py"
[[ -f "$CONFIG_PY" ]] || {
    echo "ERROR: expected $CONFIG_PY not found (is this a cosmos_framework checkout?)" >&2
    exit 1
}

echo "============================================================"
echo "Overlay source : $PATCH_DIR/"
echo "Target install : $FRAMEWORK_DIR/"
echo "Extra deps     : $([[ $INSTALL_DEPS -eq 1 ]] && echo "$COSMOS3_EXTRA_DEPS" || echo '(skipped)')"
echo "Mode           : $([[ $DRY_RUN -eq 1 ]] && echo 'DRY-RUN' || echo 'apply')"
echo "============================================================"

# --- rsync the overlay -----------------------------------------------------
RSYNC_FLAGS=(-a -v)
[[ $DRY_RUN -eq 1 ]] && RSYNC_FLAGS+=(--dry-run)
# Never copy compiled caches.
RSYNC_FLAGS+=(--exclude='__pycache__' --exclude='*.pyc')
rsync "${RSYNC_FLAGS[@]}" "$PATCH_DIR/" "$FRAMEWORK_DIR/"

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "[dry-run] No changes made (rsync, config.py registration, and deps skipped)."
    exit 0
fi

# --- register the experiment in config.py (idempotent) ---------------------
python - "$CONFIG_PY" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
line = (
    "    import cosmos_framework.configs.base.experiment.action."
    "posttrain_config.action_fdm_open_h_sft_nano  # noqa: F401\n"
)
needle = (
    "    import cosmos_framework.configs.base.experiment.action."
    "posttrain_config.action_policy_droid_nano  # noqa: F401\n"
)
if line in text:
    print(f"[ok] experiment already registered in {path}")
elif needle in text:
    path.write_text(text.replace(needle, needle + line))
    print(f"[ok] registered action_fdm_open_h_sft_nano in {path}")
else:
    raise SystemExit(
        f"ERROR: could not find the import insertion point in {path}.\n"
        f"  Expected a line importing action_policy_droid_nano to anchor after.\n"
        f"  Add this line manually inside the experiment-imports block:\n"
        f"  {line.strip()}"
    )
PY

# --- extra deps ------------------------------------------------------------
if [[ $INSTALL_DEPS -eq 1 && -n "$COSMOS3_EXTRA_DEPS" ]]; then
    if command -v uv >/dev/null 2>&1; then
        uv pip install $COSMOS3_EXTRA_DEPS || pip install $COSMOS3_EXTRA_DEPS
    else
        pip install $COSMOS3_EXTRA_DEPS
    fi
fi

# --- verify ----------------------------------------------------------------
echo ""
echo "[verify] importing overlaid modules..."
python - <<'PY'
import importlib
mods = [
    "cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs",
    "cosmos_framework.data.vfm.action.open_h_dataset",
    "cosmos_framework.data.vfm.action.datasets.openh_sft_dataset",
    "cosmos_framework.configs.base.experiment.action.posttrain_config.action_fdm_open_h_sft_nano",
]
for m in mods:
    importlib.import_module(m)
    print(f"  [ok] import {m}")
print("[verify] overlay import check passed.")
PY

echo ""
echo "Overlay applied to $FRAMEWORK_DIR"
echo "Next: run the audit / stats, e.g."
echo "  python $COOKBOOK_DIR/scripts/audit_openh_action_schemas.py --root \"\$OPENH_SURGICAL_ROOT\""
