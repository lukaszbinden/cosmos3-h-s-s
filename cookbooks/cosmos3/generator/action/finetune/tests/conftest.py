# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared sys.path setup for the cookbook test suite.

Resolution order for the ``cosmos_framework`` package:

1. A full framework checkout WITH the overlay applied (regular package —
   wins over namespace portions regardless of sys.path order per PEP 420).
   Located via, in order: ``$COSMOS3_FRAMEWORK_DIR``,
   ``$WORKSPACE/packages/cosmos3``, ``~/cosmos-framework-pinned``.
   Re-run ``scripts/apply_overlay.sh --framework-dir <root> --no-deps``
   after any ``framework_patch/`` change or the overlaid copy goes stale.
2. The cookbook's ``framework_patch/`` tree as a PEP-420 namespace package —
   sufficient for the dependency-light test layers (camp_data_contract,
   history_utils, state_action math); dataset-/config-level tests skip.

Inside the cluster workspace venv the installed (overlaid) package resolves
by itself and both inserts are inert.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
PATCH_ROOT = (_HERE / ".." / "framework_patch").resolve()


def _framework_root() -> Path | None:
    candidates = [
        os.environ.get("COSMOS3_FRAMEWORK_DIR"),
        os.path.join(os.environ["WORKSPACE"], "packages", "cosmos3")
        if os.environ.get("WORKSPACE")
        else None,
        str(Path.home() / "cosmos-framework-pinned"),
    ]
    for cand in candidates:
        if cand and (Path(cand) / "cosmos_framework" / "__init__.py").is_file():
            return Path(cand)
    return None


FRAMEWORK_ROOT = _framework_root()

for _p in [str(PATCH_ROOT)] + ([str(FRAMEWORK_ROOT)] if FRAMEWORK_ROOT else []):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _warn_if_overlay_stale() -> None:
    """A stale overlaid checkout makes tests silently validate OLD code —
    the regular package wins import resolution over the patch tree, so an
    un-restamped checkout is the worst kind of green. Compare every overlay
    file byte-for-byte and shout if the merged tree is out of date."""
    if FRAMEWORK_ROOT is None:
        return
    stale: list[str] = []
    for src in PATCH_ROOT.rglob("*.py"):
        rel = src.relative_to(PATCH_ROOT)
        dst = FRAMEWORK_ROOT / rel
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            stale.append(str(rel))
    if stale:
        banner = (
            "\n" + "!" * 78 + "\n"
            f"STALE OVERLAY: {len(stale)} framework_patch file(s) differ from the\n"
            f"checkout at {FRAMEWORK_ROOT} — tests import the CHECKOUT, so they are\n"
            "validating outdated code. Re-stamp before trusting results:\n"
            f"  bash scripts/apply_overlay.sh --framework-dir {FRAMEWORK_ROOT} --no-deps\n"
            "Stale: " + ", ".join(stale[:8]) + (" ..." if len(stale) > 8 else "") + "\n"
            + "!" * 78
        )
        # NOTE: pytest captures stderr during conftest import, so the banner
        # alone can vanish from quiet runs — the warning below surfaces in
        # pytest's warnings summary regardless.
        print(banner, file=sys.stderr)
        import warnings

        warnings.warn(
            f"STALE OVERLAY at {FRAMEWORK_ROOT} — tests validate outdated code; "
            f"re-run apply_overlay.sh. Differing: {stale[:4]}",
            stacklevel=1,
        )


_warn_if_overlay_stale()


@pytest.fixture(scope="session")
def framework_root() -> Path | None:
    """Full framework checkout root (or None). NOTE: exposed as a fixture —
    do NOT ``import conftest``; a framework checkout on sys.path ships its
    own top-level conftest.py that would shadow this one."""
    return FRAMEWORK_ROOT
