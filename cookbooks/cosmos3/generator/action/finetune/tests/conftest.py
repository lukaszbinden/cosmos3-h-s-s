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


@pytest.fixture(scope="session")
def framework_root() -> Path | None:
    """Full framework checkout root (or None). NOTE: exposed as a fixture —
    do NOT ``import conftest``; a framework checkout on sys.path ships its
    own top-level conftest.py that would shadow this one."""
    return FRAMEWORK_ROOT
