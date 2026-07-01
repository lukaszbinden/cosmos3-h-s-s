#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Stamp this cookbook's framework_patch/ overlay onto the INSTALLED
# cosmos_framework package — using a purely LOCAL file copy (cp, no rsync, no
# network).
#
# Why: cosmos_framework is an installed dependency (NOT source you edit/commit
# in git). Your overlay edits live in git under framework_patch/; this script
# copies them onto the installed package at job start so the running env picks
# them up. Re-run it after every git change to framework_patch/ (and at the top
# of each job), since reinstalling/syncing the dependency reverts the package to
# its pristine state.
#
# It (1) copies framework_patch/cosmos_framework/* over the installed package,
# (2) registers the action_fdm_open_h_sft_nano experiment in config.py
# (idempotent), (3) installs the extra Python deps the data stack needs, and
# (4) verifies the overlaid modules import.
#
# Target cosmos_framework install is resolved in this order:
#   1. --framework-dir <DIR>           (DIR contains the 'cosmos_framework/' pkg)
#   2. $COSMOS3_FRAMEWORK_DIR          (same meaning)
#   3. auto-detect from the ACTIVE python (import cosmos_framework)
#   4. $WORKSPACE/packages/cosmos3     (the setup_workspace.sh checkout)
#
# Usage:
#   # stamp whatever cosmos_framework your current venv imports:
#   bash scripts/apply_overlay.sh
#
#   # target an explicit install root (parent of cosmos_framework/):
#   bash scripts/apply_overlay.sh --framework-dir /path/to/site-packages
#
#   bash scripts/apply_overlay.sh --no-deps     # skip the extra-deps install
#   bash scripts/apply_overlay.sh --dry-run     # list files that would be copied

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
        -h|--help) sed -n '2,37p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -d "$PATCH_DIR/cosmos_framework" ]] || {
    echo "ERROR: overlay source not found: $PATCH_DIR/cosmos_framework" >&2
    exit 1
}

# --- resolve the target framework package root -----------------------------
# We want the directory that CONTAINS the 'cosmos_framework' package dir, so a
# copy of framework_patch/cosmos_framework/... lands on <root>/cosmos_framework/...
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

# Guard: don't copy a tree onto itself (e.g. if someone points --framework-dir
# at the cookbook's own framework_patch).
if [[ "$(cd "$PATCH_DIR" && pwd -P)" == "$(cd "$FRAMEWORK_DIR" && pwd -P)" ]]; then
    echo "ERROR: source and target are the same directory ($FRAMEWORK_DIR)." >&2
    exit 1
fi

CONFIG_PY="$FRAMEWORK_DIR/cosmos_framework/configs/base/config.py"
[[ -f "$CONFIG_PY" ]] || {
    echo "ERROR: expected $CONFIG_PY not found (is this a cosmos_framework install?)" >&2
    exit 1
}

echo "============================================================"
echo "Overlay source : $PATCH_DIR/  (git-managed)"
echo "Target install : $FRAMEWORK_DIR/  (installed dependency)"
echo "Extra deps     : $([[ $INSTALL_DEPS -eq 1 ]] && echo "$COSMOS3_EXTRA_DEPS" || echo '(skipped)')"
echo "Mode           : $([[ $DRY_RUN -eq 1 ]] && echo 'DRY-RUN' || echo 'apply (local cp)')"
echo "============================================================"

# --- copy the overlay (local, no rsync/network) ----------------------------
# Enumerate regular files under the overlay (excluding pycache) and cp each to
# the mirrored path under the target, creating parent dirs as needed.
copied=0
while IFS= read -r -d '' src; do
    rel="${src#"$PATCH_DIR"/}"          # e.g. cosmos_framework/data/.../dataset.py
    dst="$FRAMEWORK_DIR/$rel"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  would copy  $rel"
    else
        mkdir -p "$(dirname "$dst")"
        cp -f "$src" "$dst"
    fi
    copied=$((copied + 1))
done < <(find "$PATCH_DIR" -type f -not -path '*/__pycache__/*' -not -name '*.pyc' -print0)

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "[dry-run] $copied file(s) would be copied; no changes made."
    exit 0
fi
echo "[ok] copied $copied overlay file(s) into $FRAMEWORK_DIR"

# --- register the experiments in config.py (idempotent) --------------------
python - "$CONFIG_PY" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()

def _imp(mod: str) -> str:
    return (
        "    import cosmos_framework.configs.base.experiment.action."
        f"posttrain_config.{mod}  # noqa: F401\n"
    )

needle = _imp("action_policy_droid_nano")
# Register both the FD-only and the mixed-mode Open-H experiments, each anchored
# after the droid import. Insert in order so the block reads fd, then mixed.
to_register = ["action_fdm_open_h_sft_nano", "action_mixed_open_h_sft_nano"]

if needle not in text:
    raise SystemExit(
        f"ERROR: could not find the import insertion point in {path}.\n"
        f"  Expected a line importing action_policy_droid_nano to anchor after.\n"
        f"  Add these lines manually inside the experiment-imports block:\n"
        + "".join(f"  {_imp(m).strip()}\n" for m in to_register)
    )

for mod in to_register:
    line = _imp(mod)
    if line in text:
        print(f"[ok] {mod} already registered in {path}")
        continue
    text = text.replace(needle, needle + line)
    print(f"[ok] registered {mod} in {path}")

path.write_text(text)
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
    "cosmos_framework.configs.base.experiment.action.posttrain_config.action_mixed_open_h_sft_nano",
]
for m in mods:
    importlib.import_module(m)
    print(f"  [ok] import {m}")
print("[verify] overlay import check passed.")
PY

echo ""
echo "Overlay stamped onto $FRAMEWORK_DIR"
echo "Re-run this after any framework_patch/ change (and at job start, since"
echo "reinstalling the dependency reverts it). Next: audit / stats, e.g."
echo "  python $COOKBOOK_DIR/scripts/audit_openh_action_schemas.py --root \"\$OPENH_SURGICAL_ROOT\""
