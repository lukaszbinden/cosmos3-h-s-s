# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Canonical CAMP (Context-Aware Memory Policy) data contract for the
multi-embodiment Open-H port.

This module is the single source of truth for every integer and slice that
describes a CAMP action sequence.  Import from here; never copy-paste these
constants into dataset, transform, or model code.

Sequence layout (per sample):

    position  0 –  2  (NUM_MEMORY_SLOTS = 3 rows)
        Learned VQ memory: the 132D code is reshaped to (3, 44).
        These rows are produced by CampActionTransformPipeline *after* the
        native action/history have been padded to ACTION_DIM.  They are
        never treated as native robot actions by any transform.

    position  3 – 18  (NUM_HISTORY_ROWS = 16 rows)
        Raw recent history: 16 executed actions sampled at each embodiment's
        effective timestep interval, normalized and padded to ACTION_DIM with
        per-embodiment channel masking identical to the current action window.
        Episode-start clamping fills missing history with the first available
        frame (not zeros) to avoid spurious gradients.

    position 19 – 30  (NUM_CURRENT_ROWS = 12 rows)
        Current action window: the public 12 × 44D contract (unchanged).
        This is the only denoising target for policy and ID modes.

Full shape: (TOTAL_ACTION_ROWS, ACTION_DIM) = (31, 44).

Conditioning masks (True = clean conditioning, False = denoising target;
see :func:`make_conditioning_mask` — note the polarity is CONDITIONING, i.e.
the inverse of a "noise this row" mask):

    FD    : all 31 rows clean → conditioning_mask = [True] * 31
    policy: rows 0-18 clean, rows 19-30 denoised → conditioning_mask[19:] all False
    ID    : same as policy

The 12-step × 44D public action space is unchanged by CAMP.

Memory encoder reference (not imported here to keep this module lightweight):

    codebook_size  = 512
    code_dim       = CODE_DIM  (= 132D = NUM_MEMORY_SLOTS × ACTION_DIM)
    num_coeffs     = 32        (DCT reconstruction head)
    recon_len      = 512       (at verified 10 Hz)
    num_slots      = NUM_MEMORY_SLOTS

Embodiment coverage: 9 top-level embodiment tags, ≤ 36 dataset leaves.
All embodiments share one memory encoder with a learned embodiment embedding.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Action-space constants (must match MAX_ACTION_DIM in groot_configs.py)
# ---------------------------------------------------------------------------

ACTION_DIM: int = 44
"""Unified per-channel action dimension.  CMR Versius uses all 44; every other
embodiment is zero-padded and channel-masked up to this width."""

# ---------------------------------------------------------------------------
# CAMP sequence layout
# ---------------------------------------------------------------------------

NUM_MEMORY_SLOTS: int = 3
"""Learned VQ memory slots prepended to every CAMP sample."""

NUM_HISTORY_ROWS: int = 16
"""Recent executed-action history rows (fixed H=16 for all embodiments)."""

NUM_CURRENT_ROWS: int = 12
"""Current action window rows — the public 44D contract, unchanged."""

TOTAL_ACTION_ROWS: int = NUM_MEMORY_SLOTS + NUM_HISTORY_ROWS + NUM_CURRENT_ROWS
"""Total action positions in a CAMP sequence: 3 + 16 + 12 = 31."""

# Derived slice indices into the (TOTAL_ACTION_ROWS, ACTION_DIM) tensor.
MEMORY_SLICE: slice = slice(0, NUM_MEMORY_SLOTS)
HISTORY_SLICE: slice = slice(NUM_MEMORY_SLOTS, NUM_MEMORY_SLOTS + NUM_HISTORY_ROWS)
CURRENT_SLICE: slice = slice(NUM_MEMORY_SLOTS + NUM_HISTORY_ROWS, TOTAL_ACTION_ROWS)

NUM_CONDITIONING_ROWS: int = NUM_MEMORY_SLOTS + NUM_HISTORY_ROWS
"""Rows that are always clean conditioning (memory + history = 19 rows)."""

# ---------------------------------------------------------------------------
# Memory encoder sizing
# ---------------------------------------------------------------------------

CODE_DIM: int = NUM_MEMORY_SLOTS * ACTION_DIM
"""Flat VQ code dimension output by the memory encoder: 3 × 44 = 132D."""

MEMORY_CODEBOOK_SIZE: int = 512
MEMORY_NUM_COEFFS: int = 32
MEMORY_RECON_LEN: int = 512  # at verified 10 Hz effective training rate

# ---------------------------------------------------------------------------
# Memory-encoder embodiment vocabulary
# ---------------------------------------------------------------------------
# Dense ids for the shared memory encoder's embodiment embedding. These are
# NOT the dataset domain_ids (domain_utils assigns sparse values like 31);
# feeding a domain_id into an Embedding(9, E) would index-error or silently
# demand an oversized vocabulary. Always map through
# :func:`memory_embodiment_id`.
#
# STABILITY: these ids are baked into every memory checkpoint and every
# exported code track. NEVER renumber or remove an entry — append new
# embodiments with the next free id. (Initial assignment: the 9 tags of
# OPEN_H_DATASET_SPECS, alphabetical.)
MEMORY_EMBODIMENT_IDS: dict[str, int] = {
    "cmr_versius": 0,
    "dvrk_obuda": 1,
    "dvrk_stanford_real": 2,
    "dvrk_ucb": 3,
    "dvrk_ucsd": 4,
    "jhu_dvrk_mono": 5,
    "jhu_lscr_miracle": 6,
    "tud_tundra": 7,
    "turin_mitic_ex_vivo": 8,
}

NUM_MEMORY_EMBODIMENTS: int = len(MEMORY_EMBODIMENT_IDS)
"""Embedding vocabulary size for the shared memory encoder (9 Open-H tags)."""


def memory_embodiment_id(embodiment_tag: str) -> int:
    """Dense memory-encoder id for an embodiment tag.

    Raises:
        KeyError: For tags outside the frozen vocabulary — extend
            ``MEMORY_EMBODIMENT_IDS`` (append-only) rather than passing
            arbitrary ints.
    """
    try:
        return MEMORY_EMBODIMENT_IDS[embodiment_tag]
    except KeyError:
        raise KeyError(
            f"Unknown memory embodiment tag {embodiment_tag!r}; known: "
            f"{sorted(MEMORY_EMBODIMENT_IDS)}. Do NOT pass dataset domain_ids "
            "here — extend MEMORY_EMBODIMENT_IDS (append-only) for new "
            "embodiments."
        ) from None

# ---------------------------------------------------------------------------
# Conditioning masks
# ---------------------------------------------------------------------------


def make_conditioning_mask(mode: str) -> list[bool]:
    """Return a per-row conditioning mask for a CAMP sequence.

    True = clean conditioning (zero noise, excluded from loss).
    False = denoising target (normal diffusion noise + loss).

    Args:
        mode: One of ``"forward_dynamics"``, ``"policy"``, ``"inverse_dynamics"``.

    Returns:
        List of ``TOTAL_ACTION_ROWS`` booleans.

    Raises:
        ValueError: If ``mode`` is not a recognised action training mode.
    """
    if mode == "forward_dynamics":
        # FD: all 31 rows are clean conditioning — the video is the target.
        return [True] * TOTAL_ACTION_ROWS
    elif mode in ("policy", "inverse_dynamics"):
        # Policy / ID: memory + history clean; current 12 are denoising targets.
        return [True] * NUM_CONDITIONING_ROWS + [False] * NUM_CURRENT_ROWS
    else:
        raise ValueError(
            f"Unknown mode {mode!r}; expected one of "
            "'forward_dynamics', 'policy', 'inverse_dynamics'. "
            "(The dataset-level 'joint' mode must be resolved to a concrete "
            "per-sample mode before a conditioning mask is constructed.)"
        )


def assert_contract_invariants() -> None:
    """Raise AssertionError if any layout constant is internally inconsistent.

    Uses explicit raises rather than ``assert`` statements so the check
    survives ``python -O`` (which strips asserts).  Runs at import time and
    can be re-invoked from pre-flight checks.
    """
    checks: list[tuple[bool, str]] = [
        (
            NUM_MEMORY_SLOTS + NUM_HISTORY_ROWS + NUM_CURRENT_ROWS == TOTAL_ACTION_ROWS,
            f"Row count mismatch: {NUM_MEMORY_SLOTS} + {NUM_HISTORY_ROWS} + "
            f"{NUM_CURRENT_ROWS} != {TOTAL_ACTION_ROWS}",
        ),
        (
            NUM_CONDITIONING_ROWS == NUM_MEMORY_SLOTS + NUM_HISTORY_ROWS,
            f"Conditioning-row count mismatch: {NUM_CONDITIONING_ROWS} != "
            f"{NUM_MEMORY_SLOTS} + {NUM_HISTORY_ROWS}",
        ),
        (
            CODE_DIM == NUM_MEMORY_SLOTS * ACTION_DIM,
            f"CODE_DIM mismatch: {CODE_DIM} != {NUM_MEMORY_SLOTS} x {ACTION_DIM}",
        ),
        (MEMORY_SLICE.start == 0, "MEMORY_SLICE must start at row 0"),
        (MEMORY_SLICE.stop == NUM_MEMORY_SLOTS, "MEMORY_SLICE stop mismatch"),
        (HISTORY_SLICE.start == NUM_MEMORY_SLOTS, "HISTORY_SLICE start mismatch"),
        (
            HISTORY_SLICE.stop == NUM_MEMORY_SLOTS + NUM_HISTORY_ROWS,
            "HISTORY_SLICE stop mismatch",
        ),
        (
            CURRENT_SLICE.start == NUM_MEMORY_SLOTS + NUM_HISTORY_ROWS,
            "CURRENT_SLICE start mismatch",
        ),
        (CURRENT_SLICE.stop == TOTAL_ACTION_ROWS, "CURRENT_SLICE stop mismatch"),
        # Public action contract: the current window must stay 12 x 44.
        (
            NUM_CURRENT_ROWS == 12,
            "NUM_CURRENT_ROWS must remain 12 to preserve the public 44D action contract.",
        ),
        (
            ACTION_DIM == 44,
            "ACTION_DIM must remain 44 to preserve the public 44D action contract.",
        ),
        (
            sorted(MEMORY_EMBODIMENT_IDS.values()) == list(range(NUM_MEMORY_EMBODIMENTS)),
            "MEMORY_EMBODIMENT_IDS must be dense and contiguous 0..N-1 "
            "(append-only; never renumber).",
        ),
    ]
    for ok, message in checks:
        if not ok:
            raise AssertionError(f"CAMP data contract violated: {message}")


# Run the invariant check at import time so any accidental edit is caught
# immediately rather than at training time.
assert_contract_invariants()
