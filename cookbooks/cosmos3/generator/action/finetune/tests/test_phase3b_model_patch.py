# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CAMP Phase-3b tests: diff discipline for the overlaid model files.

The Phase-3b exemption ships as overlay copies of two PINNED framework files
(``omni_mot_model.py``, ``utils/data_and_condition.py``) with surgical edits.
Behavioral verification needs the cluster (the model imports torch>=2.3 and
runs on GPUs), so these tests enforce the strongest properties available
statically:

1. DIFF DISCIPLINE — the overlay copies differ from the pinned originals
   ONLY in lines mentioning ``num_memory_action_rows`` (or the CAMP comment
   block around them). Any other drift means the overlay is no longer
   "pinned file + exactly this patch" and must be rejected.
2. STRUCTURE — both zeroing sites use the row-offset form, the dataclass
   field exists, the batch-ingestion and subset-slicing plumbing is present.
3. Both files parse (AST) — no syntax risk to the training job.

The pinned originals are read from the framework checkout's git object store
(``git show HEAD:<path>``), NOT its working tree, so a stamped overlay cannot
mask drift.
"""

from __future__ import annotations

import ast
import difflib
import subprocess
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
PATCH_ROOT = (_HERE / ".." / "framework_patch").resolve()

_PATCHED_FILES = [
    "cosmos_framework/model/vfm/omni_mot_model.py",
    "cosmos_framework/model/vfm/utils/data_and_condition.py",
]

_MARKER = "num_memory_action_rows"


def _pinned_source(framework_root: Path, rel: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(framework_root), "show", f"HEAD:{rel}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if out.returncode != 0:
        pytest.skip(f"cannot read pinned {rel} from git: {out.stderr.strip()[:120]}")
    return out.stdout


@pytest.fixture(scope="session")
def pinned_root(framework_root):
    if framework_root is None or not (framework_root / ".git").exists():
        pytest.skip("no git framework checkout found (set COSMOS3_FRAMEWORK_DIR)")
    return framework_root


class TestDiffDiscipline:
    @pytest.mark.parametrize("rel", _PATCHED_FILES)
    def test_only_marker_lines_changed(self, pinned_root, rel):
        pinned = _pinned_source(pinned_root, rel).splitlines()
        patched = (PATCH_ROOT / rel).read_text().splitlines()
        changed = [
            line
            for line in difflib.unified_diff(pinned, patched, lineterm="", n=0)
            if (line.startswith("+") or line.startswith("-"))
            and not line.startswith(("+++", "---"))
        ]
        assert changed, f"{rel}: overlay copy is identical to pinned — patch missing?"
        # Every removed line must reappear (indent-shift only) or be a pure
        # context-free deletion of nothing; every added line must belong to
        # the CAMP patch: mention the marker, be a comment in the CAMP block,
        # or be a re-indented original line adjacent to the guard.
        removed = [l[1:] for l in changed if l.startswith("-")]
        added = [l[1:] for l in changed if l.startswith("+")]
        # Removed lines must either be re-added verbatim, or be one of the two
        # sanctioned zeroing-statement replacements: full-row `[:, ...]` form
        # rewritten to the row-offset `[_camp_mem_rows:, ...]` form. Nothing
        # else may be deleted from upstream.
        for line in removed:
            stripped = line.strip()
            offset_form = stripped.replace("[:, ", "[_camp_mem_rows:, ", 1)
            assert any(
                a.strip() in (stripped, offset_form) for a in added
            ), f"{rel}: patch removed an upstream line outright: {line!r}"
        for line in added:
            stripped = line.strip()
            ok = (
                _MARKER in stripped
                or stripped.startswith("#")
                or "_camp_mem_rows" in stripped
                or any(stripped == r.strip() for r in removed)
            )
            assert ok, f"{rel}: unexpected non-CAMP addition: {line!r}"

    @pytest.mark.parametrize("rel", _PATCHED_FILES)
    def test_patched_file_parses(self, rel):
        ast.parse((PATCH_ROOT / rel).read_text())


class TestPatchStructure:
    def test_dataclass_field_present(self):
        src = (PATCH_ROOT / _PATCHED_FILES[1]).read_text()
        assert "num_memory_action_rows: list[torch.Tensor] | None = None" in src

    def test_both_zeroing_sites_row_offset(self):
        src = (PATCH_ROOT / _PATCHED_FILES[0]).read_text()
        assert src.count("_camp_mem_rows:, gen_data_clean.raw_action_dim[i] :] = 0") == 2, (
            "expected the row-offset zeroing form at exactly the xt-noising and "
            "sampling-init sites"
        )
        # The pristine full-row form must be GONE (both sites patched).
        assert "xt_action[i][:, gen_data_clean.raw_action_dim[i] :] = 0" not in src
        assert "noise_action_i[:, gen_data_clean.raw_action_dim[i] :] = 0" not in src

    def test_plumbing_present(self):
        src = (PATCH_ROOT / _PATCHED_FILES[0]).read_text()
        assert 'data_batch.get("num_memory_action_rows", None)' in src
        assert src.count("num_memory_action_rows=num_memory_action_rows") == 2, (
            "expected the field in BOTH GenerationDataClean constructions "
            "(batch ingestion + subset slicing)"
        )

    def test_velocity_and_loss_sites_untouched(self):
        """The velocity site stays pristine (already row-masked for
        conditioning rows) and flow_matching.py is NOT overlaid at all —
        the loss slice is row-masked via noisy_mask."""
        src = (PATCH_ROOT / _PATCHED_FILES[0]).read_text()
        assert "v[:, gen_data_clean.raw_action_dim[i] :] = 0" in src
        assert not (PATCH_ROOT / "cosmos_framework/model/vfm/algorithm").exists()
