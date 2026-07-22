# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CAMP Phase-1 history helpers: delta-index construction, history/current
splitting, and history ablations.

Kept deliberately dependency-light (numpy/torch only — no albumentations, no
framework imports) so these functions unit-test in a bare Python environment.

Design notes (see README_COSMOS3_H_S_S.md, "CAMP port handoff"):

* History rows are the H most recent EXECUTED actions, sampled at the same
  per-embodiment ``timestep_interval`` as the current window, and flow through
  the SAME per-embodiment transform + normalization stack.  This module only
  computes indices and post-transform tensor splits — the semantics live in
  the existing transforms.
* Episode-start clamping is inherited from
  ``LeRobotSingleDataset.retrieve_data_and_pad``: negative step indices are
  padded with the episode's first frame for absolute-format keys ("resting at
  the first pose before the episode starts") and with zeros for
  delta-format keys ("no motion before the episode starts").
* ``H = 0`` must be byte-identical to the pre-CAMP pipeline:
  ``build_action_delta_indices(ti, N, 0)`` returns exactly the legacy
  ``range(0, N * ti, ti)`` and ``split_history_and_current(a, 0)`` returns
  the input tensor untouched.
"""

from __future__ import annotations

import torch

# Valid values for the ``history_ablation`` knob.
#   None      — real history (training + the primary eval arm).
#   "zero"    — history replaced with zeros (eval: does the model use history
#               at all? should revert toward the H=0 baseline).
#   "permute" — history rows time-shuffled within the sample (eval: does the
#               model use the ORDER of history, or just its marginal stats?).
HISTORY_ABLATIONS: tuple[str | None, ...] = (None, "zero", "permute")

# Large prime for mixing (dataset_idx, sample_idx) into a stable per-sample
# ablation seed. Any odd prime > max plausible sample count works.
_PERMUTE_SEED_PRIME = 1_000_003


def validate_history_args(num_history_actions: int, history_ablation: str | None) -> None:
    """Validate the (num_history_actions, history_ablation) pair.

    Raises:
        ValueError: On a negative H, an unknown ablation name, or an ablation
            requested with H=0 (which would silently do nothing — fail loudly
            instead so a mis-configured eval arm cannot masquerade as real).
    """
    if num_history_actions < 0:
        raise ValueError(f"num_history_actions must be >= 0, got {num_history_actions}")
    if history_ablation not in HISTORY_ABLATIONS:
        raise ValueError(
            f"history_ablation must be one of {HISTORY_ABLATIONS}, got {history_ablation!r}"
        )
    if history_ablation is not None and num_history_actions == 0:
        raise ValueError(
            f"history_ablation={history_ablation!r} requires num_history_actions > 0 "
            "(an ablation with no history rows would silently be a no-op)"
        )


def build_action_delta_indices(
    timestep_interval: int,
    num_current_actions: int,
    num_history_actions: int = 0,
) -> list[int]:
    """Action delta indices for a history-extended window.

    Layout: ``[-H*ti, ..., -ti,  0, ti, ..., (N-1)*ti]`` — H history steps
    strictly before the anchor frame (delta 0), then the legacy N-step current
    window starting AT the anchor.  With ``num_history_actions=0`` this is
    byte-identical to the legacy ``range(0, N*ti, ti)``.

    Negative indices are clamped/padded by ``retrieve_data_and_pad`` at
    episode starts (first-frame repeat for absolute keys, zeros for delta
    keys), so callers never need boundary handling.

    Args:
        timestep_interval: Per-embodiment frame stride (e.g. 3 for 30 Hz
            storage → 10 Hz effective, 6 for CMR's 60 Hz → 10 Hz).
        num_current_actions: Length of the current (public-contract) action
            window — 12 for the standard 13-frame recipe.
        num_history_actions: H, the number of history rows to prepend.

    Returns:
        List of ``H + N`` ints, strictly increasing, spaced ``timestep_interval``.
    """
    ti = int(timestep_interval)
    n = int(num_current_actions)
    h = int(num_history_actions)
    if ti <= 0:
        raise ValueError(f"timestep_interval must be > 0, got {timestep_interval}")
    if n <= 0:
        raise ValueError(f"num_current_actions must be > 0, got {num_current_actions}")
    if h < 0:
        raise ValueError(f"num_history_actions must be >= 0, got {num_history_actions}")
    return list(range(-h * ti, n * ti, ti))


def split_history_and_current(
    action: torch.Tensor, num_history_actions: int
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Split a transformed ``(H + N, D)`` action tensor into history and current.

    With ``num_history_actions=0`` the input is returned untouched (H=0
    byte-identity) with ``None`` history.

    Returns:
        ``(history, current)`` where ``history`` is ``(H, D)`` (or ``None``)
        and ``current`` is ``(N, D)``.

    Raises:
        ValueError: If the tensor has too few rows to contain H history rows
            plus a non-empty current window.
    """
    h = int(num_history_actions)
    if h == 0:
        return None, action
    if action.ndim != 2:
        raise ValueError(f"expected a (T, D) action tensor, got shape {tuple(action.shape)}")
    if action.shape[0] <= h:
        raise ValueError(
            f"action tensor has {action.shape[0]} rows; needs > num_history_actions={h} "
            "(history rows plus a non-empty current window)"
        )
    return action[:h], action[h:]


def history_ablation_seed(dataset_idx: int, sample_idx: int) -> int:
    """Stable per-sample seed for the ``permute`` ablation.

    Depends only on (sub-dataset index, sample index within it), so a given
    sample permutes identically across epochs, workers, and re-runs — the
    ablation measures the model, not dataloader nondeterminism.
    """
    return (int(dataset_idx) * _PERMUTE_SEED_PRIME + int(sample_idx)) & 0x7FFF_FFFF


def apply_history_ablation(
    history: torch.Tensor, history_ablation: str | None, seed: int
) -> torch.Tensor:
    """Apply the configured ablation to an ``(H, D)`` history tensor.

    * ``None``      → returned unchanged.
    * ``"zero"``    → zeros of the same shape/dtype.
    * ``"permute"`` → rows time-shuffled with a ``torch.Generator`` seeded by
      ``seed`` (deterministic per sample; see :func:`history_ablation_seed`).
    """
    if history_ablation is None:
        return history
    if history_ablation == "zero":
        return torch.zeros_like(history)
    if history_ablation == "permute":
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        perm = torch.randperm(history.shape[0], generator=generator)
        return history[perm]
    raise ValueError(
        f"history_ablation must be one of {HISTORY_ABLATIONS}, got {history_ablation!r}"
    )
