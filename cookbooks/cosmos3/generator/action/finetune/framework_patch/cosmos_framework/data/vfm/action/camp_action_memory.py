# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CAMP behavioral action memory (arXiv 2606.21188), multi-embodiment port.

Ported from the SutureBot CAMP-v2 implementation
(``cosmos3-dvrk-camp/cosmos3/_src/vfm/models/action_memory.py``). The DCT
scaffold, losses, hardened ``VectorQuantizer`` recipe (data init, EMA updates,
dead-code restarts, commitment-only loss), and the LSTM ``ActionMemoryEncoder``
are carried over VERBATIM — preserving the production recipe is the point.

New here (the Open-H generalization; see camp_data_contract.py):

* ``MultiEmbodimentActionMemoryEncoder`` — ONE shared encoder for all nine
  embodiment tags. Per-step input assembles:
      canonical two-arm state (20D, masked) + state mask (20D)
      + previous normalized 44D action (masked) + active-channel mask (44D)
      + a learned embodiment embedding.
  The wrapper delegates to an internal ``ActionMemoryEncoder`` so every knob
  of the validated recipe is unchanged; only the input width grows.
* ``code_dim`` defaults to camp_data_contract.CODE_DIM = 132 = 3 x 44
  (the SutureBot 63 = 3 x 21 was the same formula at 21-wide rows).
* ``reconstruction_loss`` gains an optional per-channel validity mask so the
  DCT head reconstructs only channels that exist for the embodiment.

Deviations from the paper carried over from SutureBot (both deliberate):
inputs are proprio+action only (no visual features — executed kinematics
encode outcomes vision can't see, e.g. an arm out of frame), and code_dim is
a lattice-packing choice, not an information-capacity one.

Pure torch — no framework dependencies — so every piece unit-tests on CPU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_framework.data.vfm.action.camp_data_contract import (
    ACTION_DIM,
    CODE_DIM,
    MEMORY_CODEBOOK_SIZE,
    MEMORY_NUM_COEFFS,
    MEMORY_RECON_LEN,
)

# Canonical two-arm kinematic state: 2 arms x (xyz 3 + rot6d 6 + gripper 1).
CANONICAL_STATE_DIM = 20

# ---------------------------------------------------------------------------
# DCT utilities  (verbatim from SutureBot CAMP-v2)
# ---------------------------------------------------------------------------


def dct_basis(length: int, num_coeffs: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """First ``num_coeffs`` orthonormal type-II DCT basis vectors.

    Returns:
        (length, num_coeffs) matrix B with orthonormal columns, so a
        trajectory x (length,) has coefficients ``c = B^T x`` and lossy
        reconstruction ``x_hat = B c``.
    """
    assert 1 <= num_coeffs <= length, f"num_coeffs {num_coeffs} must be in [1, {length}]"
    n = torch.arange(length, dtype=dtype).unsqueeze(1)  # (L, 1)
    k = torch.arange(num_coeffs, dtype=dtype).unsqueeze(0)  # (1, K)
    basis = torch.cos(math.pi / length * (n + 0.5) * k)  # (L, K)
    scale = torch.full((1, num_coeffs), math.sqrt(2.0 / length), dtype=dtype)
    scale[0, 0] = math.sqrt(1.0 / length)
    return basis * scale


def dct_encode(traj: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project trajectories onto the basis. traj (..., L, D) -> (..., K, D)."""
    return torch.einsum("lk,...ld->...kd", basis, traj)


def dct_decode(coeffs: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Reconstruct trajectories from coefficients. (..., K, D) -> (..., L, D)."""
    return torch.einsum("lk,...kd->...ld", basis, coeffs)


def build_dct_targets(
    pose_traj: torch.Tensor, anchor_steps: torch.Tensor, basis: torch.Tensor
) -> torch.Tensor:
    """DCT coefficients of the past-L window ending at each anchor step.

    For anchor t the window covers steps [t - L + 1, t]; history shorter than
    L is left-padded by repeating the trajectory's first frame — matching the
    LeRobot clamp semantics the hist16 rail already trains with ("resting at
    the episode's first pose" before the episode starts).

    Args:
        pose_traj: (B, T, D) normalized canonical-state trajectories.
        anchor_steps: (M,) long tensor of anchor indices into T (same anchors
            for every batch element).
        basis: (L, K) DCT basis from :func:`dct_basis`.

    Returns:
        (B, M, K, D) target coefficients.
    """
    B, T, D = pose_traj.shape
    L = basis.shape[0]
    windows = []
    for t in anchor_steps.tolist():
        start = t - L + 1
        if start >= 0:
            win = pose_traj[:, start : t + 1]  # (B, L, D)
        else:
            pad = pose_traj[:, :1].expand(B, -start, D)  # repeat first frame
            win = torch.cat([pad, pose_traj[:, : t + 1]], dim=1)
        windows.append(win)
    windows = torch.stack(windows, dim=1)  # (B, M, L, D)
    return dct_encode(windows, basis)  # (B, M, K, D)


# ---------------------------------------------------------------------------
# Losses  (verbatim, except the optional channel mask on reconstruction_loss)
# ---------------------------------------------------------------------------


def frequency_weights(num_coeffs: int, gamma: float = 3.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """w_k = exp(-gamma * k / (K - 1)): down-weight high-frequency coefficients."""
    if num_coeffs == 1:
        return torch.ones(1, dtype=dtype)
    k = torch.arange(num_coeffs, dtype=dtype)
    return torch.exp(-gamma * k / (num_coeffs - 1))


def reconstruction_loss(
    pred_coeffs: torch.Tensor,
    target_coeffs: torch.Tensor,
    gamma: float = 3.0,
    channel_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Frequency-weighted MSE in DCT coefficient space (paper eq. 1).

    pred/target: (..., K, D).

    Args:
        channel_mask: Optional (..., D) or (D,) validity mask (1 = channel
            exists for this embodiment). Masked channels contribute nothing —
            required in the multi-embodiment setting where e.g. a single-arm
            robot has 10 dead canonical-state channels. ``None`` reproduces
            the SutureBot behavior exactly.
    """
    w = frequency_weights(pred_coeffs.shape[-2], gamma, dtype=pred_coeffs.dtype).to(pred_coeffs.device)
    sq = (pred_coeffs - target_coeffs).pow(2)  # (..., K, D)
    if channel_mask is not None:
        m = channel_mask.to(sq.dtype)
        if m.dim() < sq.dim():  # broadcast (D,) or (..., D) over the K axis
            m = m.unsqueeze(-2)
        sq = sq * m
        denom = m.expand_as(sq).sum(dim=-1).clamp_min(1e-8)  # (..., K)
        sq = sq.sum(dim=-1) / denom
    else:
        sq = sq.mean(dim=-1)  # (..., K)
    return (sq * w).sum(dim=-1).mean() / w.sum()


def temporal_consistency_loss(
    coeffs_a: torch.Tensor, coeffs_b: torch.Tensor, offset: int, basis: torch.Tensor
) -> torch.Tensor:
    """Reconstructions of the same past from anchors ``offset`` steps apart must agree.

    Slot k of the window decoded at time t refers to the same physical action
    as slot k + offset decoded at time t + offset (paper eq. 2); we penalize
    disagreement on the decoded overlap.

    Args:
        coeffs_a: (..., K, D) coefficients predicted at anchor t.
        coeffs_b: (..., K, D) coefficients predicted at anchor t + offset.
        offset: N > 0, the anchor separation (in steps, < L).
        basis: (L, K) DCT basis.
    """
    L = basis.shape[0]
    assert 0 < offset < L, f"offset {offset} must be in (0, {L})"
    recon_a = dct_decode(coeffs_a, basis)  # window ending at t
    recon_b = dct_decode(coeffs_b, basis)  # window ending at t + offset
    # Overlap: recon_a spans [t-L+1, t], recon_b spans [t-L+1+offset, t+offset];
    # shared steps are [t-L+1+offset, t].
    return F.mse_loss(recon_a[..., offset:, :], recon_b[..., : L - offset, :])


# ---------------------------------------------------------------------------
# Vector quantizer  (verbatim from SutureBot CAMP-v2)
# ---------------------------------------------------------------------------


class VectorQuantizer(nn.Module):
    """Nearest-neighbor VQ hardened against codebook collapse.

    The paper's minimal loss-based VQ (eqs. 4-5) collapsed on real SutureBot
    pretraining (perplexity pinned at 1-6 of 512 while reconstruction kept
    improving): the classic tiny-uniform init leaves every entry in a ball
    near the origin, one entry wins the first assignments, and the never-
    selected rest receive no gradient — permanently dead. This version uses
    the standard production recipe instead:

      - data-dependent init: the codebook is seeded from the first training
        batch's encoder outputs (k-means-free variant);
      - EMA codebook updates (decay 0.99, Laplace-smoothed) instead of the
        codebook MSE loss — the returned vq_loss is the commitment term only;
      - dead-code restarts: entries unused for ``dead_steps`` consecutive
        training forwards are re-seeded from the current batch;
      - optional validity mask so padded steps never pollute the statistics.

    Single-process training only (EMA statistics are not all-reduced).
    """

    def __init__(
        self,
        codebook_size: int = 512,
        dim: int = 128,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        dead_steps: int = 100,
    ) -> None:
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)
        self.codebook.weight.requires_grad_(False)  # EMA-updated, not SGD-updated
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.dead_steps = dead_steps
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed", torch.zeros(codebook_size, dim))
        self.register_buffer("steps_since_use", torch.zeros(codebook_size, dtype=torch.long))

    @torch.no_grad()
    def _seed_from(self, flat: torch.Tensor, rows: torch.Tensor | None = None) -> None:
        """(Re-)seed codebook rows from encoder outputs (+ jitter if too few)."""
        K = self.codebook.num_embeddings
        target = rows if rows is not None else torch.arange(K, device=flat.device)
        picks = flat[torch.randint(0, flat.shape[0], (target.numel(),), device=flat.device)]
        if flat.shape[0] < target.numel():
            picks = picks + torch.randn_like(picks) * (flat.std() * 0.01 + 1e-6)
        self.codebook.weight.data[target] = picks
        self.ema_embed[target] = picks
        self.ema_cluster_size[target] = 1.0
        self.steps_since_use[target] = 0

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize (..., dim) -> (quantized_st, indices, vq_loss, perplexity).

        ``mask`` (matching h's leading dims) marks valid positions; EMA
        statistics, restarts, perplexity, and the commitment loss use valid
        positions only. All positions are still quantized in the output.
        """
        flat = h.reshape(-1, h.shape[-1])  # (N, dim)
        flat_mask = (
            mask.reshape(-1).to(torch.bool)
            if mask is not None
            else torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
        )
        valid = flat[flat_mask].detach()

        if not bool(self.initialized):
            if not self.training:
                raise RuntimeError(
                    "VectorQuantizer used in eval mode before initialization — "
                    "load a trained checkpoint or run a training forward first"
                )
            if valid.numel() > 0:
                self._seed_from(valid)
                self.initialized.fill_(True)

        dists = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1)
        )  # (N, Kq)
        indices = dists.argmin(dim=1)  # (N,)
        quantized = self.codebook(indices).view_as(h)

        # Commitment only — the codebook itself learns by EMA below.
        if flat_mask.any():
            vq_loss = self.beta * F.mse_loss(
                flat[flat_mask], self.codebook(indices[flat_mask]).detach()
            )
        else:
            vq_loss = h.sum() * 0.0

        # Straight-through: forward uses the code, gradient flows to the encoder.
        quantized_st = h + (quantized - h).detach()

        onehot = F.one_hot(indices[flat_mask], self.codebook.num_embeddings).to(flat.dtype)
        probs = onehot.mean(dim=0) if flat_mask.any() else onehot.sum(dim=0)
        perplexity = torch.exp(-(probs * torch.log(probs.clamp_min(1e-10))).sum())

        if self.training and flat_mask.any():
            with torch.no_grad():
                counts = onehot.sum(dim=0)  # (Kq,)
                self.ema_cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                self.ema_embed.mul_(self.decay).add_(onehot.t() @ valid, alpha=1 - self.decay)
                n = self.ema_cluster_size.sum()
                smoothed = (
                    (self.ema_cluster_size + self.eps)
                    / (n + self.codebook.num_embeddings * self.eps)
                    * n
                )
                self.codebook.weight.data.copy_(self.ema_embed / smoothed.unsqueeze(1))
                # Dead-code restarts.
                self.steps_since_use += 1
                self.steps_since_use[counts > 0] = 0
                dead = (self.steps_since_use >= self.dead_steps).nonzero(as_tuple=True)[0]
                if dead.numel() > 0:
                    self._seed_from(valid, rows=dead)

        return quantized_st, indices.view(h.shape[:-1]), vq_loss, perplexity


# ---------------------------------------------------------------------------
# The memory encoder  (verbatim core from SutureBot CAMP-v2)
# ---------------------------------------------------------------------------


class ActionMemoryEncoder(nn.Module):
    """LSTM behavioral memory: kinematic history -> discrete code m_t.

    Args:
        input_dim: per-step input width.
        hidden_dim: LSTM hidden size d_h.
        num_layers: LSTM depth (paper: 2).
        num_coeffs: K, DCT coefficients reconstructed by the pretraining head.
        recon_dim: D, dimensionality of the reconstructed pose trajectory.
        recon_len: L, length of the past window the head reconstructs.
        code_dim: memory code width (132 = 3 pseudo-rows x 44 action dims here;
            63 = 3 x 21 in the SutureBot original).
        codebook_size: VQ codebook entries.
        beta: VQ commitment weight.
    """

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_coeffs: int = MEMORY_NUM_COEFFS,
        recon_dim: int = CANONICAL_STATE_DIM,
        recon_len: int = MEMORY_RECON_LEN,
        code_dim: int = CODE_DIM,
        codebook_size: int = MEMORY_CODEBOOK_SIZE,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_coeffs = num_coeffs
        self.recon_dim = recon_dim
        self.recon_len = recon_len
        self.code_dim = code_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.recon_head = nn.Linear(hidden_dim, num_coeffs * recon_dim)
        self.vq = VectorQuantizer(codebook_size, hidden_dim, beta)
        self.code_head = nn.Sequential(nn.Linear(hidden_dim, code_dim), nn.Tanh())
        # Pretraining-only auxiliary: reconstruct the same DCT targets FROM the
        # code, so the code bottleneck (and code_head, which otherwise receives
        # no pretraining gradient) is forced to carry the behavioral summary
        # the frozen-memory SFT will condition on.
        self.code_recon_head = nn.Linear(code_dim, num_coeffs * recon_dim)
        self.register_buffer("basis", dct_basis(recon_len, num_coeffs), persistent=False)

    # -- core passes ------------------------------------------------------

    def forward(
        self, inputs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the LSTM. inputs (B, T, input_dim) -> hidden (B, T, hidden_dim), state."""
        hidden, state = self.lstm(inputs, state)
        return hidden, state

    def reconstruct(self, hidden: torch.Tensor) -> torch.Tensor:
        """Pretraining head: hidden (..., hidden_dim) -> DCT coeffs (..., K, recon_dim)."""
        out = self.recon_head(hidden)
        return out.view(*hidden.shape[:-1], self.num_coeffs, self.recon_dim)

    def code(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize hidden state and emit the memory code.

        Returns (code (..., code_dim), vq_loss, perplexity).
        """
        quantized, _, vq_loss, perplexity = self.vq(hidden)
        return self.code_head(quantized), vq_loss, perplexity

    def quantize(
        self, hidden: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full-sequence quantization for pretraining.

        Returns (quantized (..., hidden_dim), vq_loss, perplexity); ``mask``
        keeps padded steps out of the codebook statistics.
        """
        quantized, _, vq_loss, perplexity = self.vq(hidden, mask)
        return quantized, vq_loss, perplexity

    def reconstruct_from_code(self, code: torch.Tensor) -> torch.Tensor:
        """Auxiliary pretraining head: code (..., code_dim) -> (..., K, recon_dim)."""
        out = self.code_recon_head(code)
        return out.view(*code.shape[:-1], self.num_coeffs, self.recon_dim)

    # -- inference conveniences --------------------------------------------

    @torch.no_grad()
    def encode_episode(self, inputs: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """Codes for every step of one episode. inputs (T, input_dim) -> (T, code_dim)."""
        state = None
        codes = []
        for start in range(0, inputs.shape[0], chunk):
            hidden, state = self.forward(inputs[start : start + chunk].unsqueeze(0), state)
            codes.append(self.code(hidden)[0].squeeze(0))
        return torch.cat(codes, dim=0)

    @torch.no_grad()
    def step(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single incremental step for closed-loop serving.

        x (input_dim,) -> (code (code_dim,), new state).
        """
        hidden, state = self.forward(x.view(1, 1, -1), state)
        return self.code(hidden)[0].view(-1), state


# ---------------------------------------------------------------------------
# Multi-embodiment generalization (new in the Open-H port)
# ---------------------------------------------------------------------------


def assemble_memory_inputs(
    canonical_state: torch.Tensor,
    state_mask: torch.Tensor,
    prev_action: torch.Tensor,
    action_mask: torch.Tensor,
    embodiment_embedding: torch.Tensor,
) -> torch.Tensor:
    """Assemble the canonical per-step encoder input.

    Values are multiplied by their masks (absent channels are EXACTLY zero)
    and the masks are also concatenated as inputs, so the encoder can
    distinguish "channel is zero" from "channel does not exist".

    Args:
        canonical_state: (..., T, 20) two-arm canonical kinematic state.
        state_mask: (..., 20) or (..., T, 20) — 1 where the channel exists
            (missing arm → 10 zeros; missing gripper → 1 zero).
        prev_action: (..., T, 44) previous normalized action, zero-padded.
        action_mask: (..., 44) or (..., T, 44) active-channel mask
            (1 for channels within the embodiment's raw_action_dim).
        embodiment_embedding: (..., E) or (..., T, E) learned embedding,
            constant over T for a given sample.

    Returns:
        (..., T, 20 + 20 + 44 + 44 + E) input tensor.
    """
    t = canonical_state.shape[-2]

    def _timewise(x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.shape[-1] != dim:
            raise ValueError(f"expected trailing dim {dim}, got {tuple(x.shape)}")
        if x.dim() == canonical_state.dim():
            return x
        return x.unsqueeze(-2).expand(*x.shape[:-1], t, dim)

    sm = _timewise(state_mask, canonical_state.shape[-1]).to(canonical_state.dtype)
    am = _timewise(action_mask, prev_action.shape[-1]).to(prev_action.dtype)
    emb = _timewise(embodiment_embedding, embodiment_embedding.shape[-1])
    return torch.cat([canonical_state * sm, sm, prev_action * am, am, emb], dim=-1)


class MultiEmbodimentActionMemoryEncoder(nn.Module):
    """ONE shared CAMP memory encoder across all Open-H embodiment tags.

    A thin assembly wrapper around :class:`ActionMemoryEncoder` — the entire
    validated recipe (LSTM, VQ, DCT heads, losses) is delegated unchanged;
    this class only owns the embodiment embedding and the input contract.

    Per-step input (see :func:`assemble_memory_inputs`):
        [state*mask (20) | state_mask (20) | prev_action*mask (44) |
         action_mask (44) | embodiment_embedding (E)]

    Args:
        num_embodiments: Size of the embodiment vocabulary (9 Open-H tags;
            pass the domain-id vocabulary size used by the dataset).
        embed_dim: Learned embodiment-embedding width.
        Remaining args: forwarded to :class:`ActionMemoryEncoder`.
    """

    def __init__(
        self,
        num_embodiments: int,
        embed_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_coeffs: int = MEMORY_NUM_COEFFS,
        recon_len: int = MEMORY_RECON_LEN,
        code_dim: int = CODE_DIM,
        codebook_size: int = MEMORY_CODEBOOK_SIZE,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_embodiments = int(num_embodiments)
        self.embed_dim = int(embed_dim)
        self.embodiment_embedding = nn.Embedding(self.num_embodiments, self.embed_dim)
        self.input_dim = 2 * CANONICAL_STATE_DIM + 2 * ACTION_DIM + self.embed_dim
        self.encoder = ActionMemoryEncoder(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_coeffs=num_coeffs,
            recon_dim=CANONICAL_STATE_DIM,
            recon_len=recon_len,
            code_dim=code_dim,
            codebook_size=codebook_size,
            beta=beta,
        )

    @property
    def basis(self) -> torch.Tensor:
        return self.encoder.basis

    @property
    def code_dim(self) -> int:
        return self.encoder.code_dim

    def assemble(
        self,
        canonical_state: torch.Tensor,
        state_mask: torch.Tensor,
        prev_action: torch.Tensor,
        action_mask: torch.Tensor,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        """Look up the embedding and assemble (B, T, input_dim) inputs.

        ``embodiment_id``: (B,) long tensor (constant per sequence).
        """
        emb = self.embodiment_embedding(embodiment_id)  # (B, E)
        return assemble_memory_inputs(canonical_state, state_mask, prev_action, action_mask, emb)

    def forward(
        self,
        inputs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """LSTM over pre-assembled inputs (see :meth:`assemble`)."""
        return self.encoder(inputs, state)

    def reconstruct(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.encoder.reconstruct(hidden)

    def quantize(
        self, hidden: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder.quantize(hidden, mask)

    def code(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder.code(hidden)

    def reconstruct_from_code(self, code: torch.Tensor) -> torch.Tensor:
        return self.encoder.reconstruct_from_code(code)

    @torch.no_grad()
    def encode_episode(
        self,
        canonical_state: torch.Tensor,
        state_mask: torch.Tensor,
        prev_action: torch.Tensor,
        action_mask: torch.Tensor,
        embodiment_id: torch.Tensor,
        chunk: int = 4096,
    ) -> torch.Tensor:
        """Codes for every step of one episode.

        canonical_state (T, 20), prev_action (T, 44), masks (20,)/(44,),
        embodiment_id () or (1,) -> (T, code_dim). Strictly causal: the code
        at step t depends only on inputs [0..t].
        """
        inputs = self.assemble(
            canonical_state.unsqueeze(0),
            state_mask.unsqueeze(0),
            prev_action.unsqueeze(0),
            action_mask.unsqueeze(0),
            embodiment_id.view(1),
        ).squeeze(0)
        return self.encoder.encode_episode(inputs, chunk=chunk)

    @torch.no_grad()
    def step(
        self,
        canonical_state: torch.Tensor,
        state_mask: torch.Tensor,
        prev_action: torch.Tensor,
        action_mask: torch.Tensor,
        embodiment_id: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single incremental step for the Phase-5 online state manager.

        canonical_state (20,), prev_action (44,), masks (20,)/(44,) ->
        (code (code_dim,), new LSTM state). Reset ``state=None`` at episode
        boundaries.
        """
        x = self.assemble(
            canonical_state.view(1, 1, -1),
            state_mask.view(1, -1),
            prev_action.view(1, 1, -1),
            action_mask.view(1, -1),
            embodiment_id.view(1),
        )
        hidden, state = self.encoder(x, state)
        return self.encoder.code(hidden)[0].view(-1), state
