# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CAMP Phase-2 tests: the multi-embodiment action-memory module.

Everything here is pure torch on CPU. The module under test is the verbatim
SutureBot CAMP-v2 port plus the multi-embodiment wrapper — these tests pin:

* the hardened VQ recipe (data init, EMA updates, dead-code restarts,
  commitment-only loss, mask-aware statistics);
* straight-through gradient flow to the encoder;
* STRICT CAUSALITY — the Phase-2 gate "changing future episode rows cannot
  change codes at time t";
* the multi-embodiment input contract (masking, embedding, dims);
* masked DCT reconstruction loss semantics;
* contract wiring (code_dim == 132 == camp_data_contract.CODE_DIM).
"""

from __future__ import annotations

import importlib

import pytest
import torch

am = importlib.import_module("cosmos_framework.data.vfm.action.camp_action_memory")
contract = importlib.import_module("cosmos_framework.data.vfm.action.camp_data_contract")

torch.manual_seed(0)


def _train_forward(encoder: am.ActionMemoryEncoder, b=4, t=32) -> None:
    """One training forward to data-initialize the VQ codebook."""
    encoder.train()
    inputs = torch.randn(b, t, encoder.input_dim)
    hidden, _ = encoder(inputs)
    encoder.quantize(hidden)


# ---------------------------------------------------------------------------
# DCT scaffold
# ---------------------------------------------------------------------------


class TestDct:
    def test_basis_orthonormal(self):
        basis = am.dct_basis(64, 32)
        gram = basis.t() @ basis
        assert torch.allclose(gram, torch.eye(32), atol=1e-5)

    def test_full_basis_roundtrip_exact(self):
        basis = am.dct_basis(16, 16)
        traj = torch.randn(2, 16, 5)
        recon = am.dct_decode(am.dct_encode(traj, basis), basis)
        assert torch.allclose(recon, traj, atol=1e-5)

    def test_build_dct_targets_shapes(self):
        basis = am.dct_basis(8, 4)
        traj = torch.randn(2, 20, 5)
        anchors = torch.tensor([3, 10, 19])
        targets = am.build_dct_targets(traj, anchors, basis)
        assert targets.shape == (2, 3, 4, 5)

    def test_build_dct_targets_clamps_episode_start(self):
        """Anchor earlier than L-1: window left-pads with the FIRST frame —
        the LeRobot clamp semantics the H16 rail trains with."""
        basis = am.dct_basis(8, 8)  # full basis → exact reconstruction
        traj = torch.randn(1, 20, 3)
        targets = am.build_dct_targets(traj, torch.tensor([2]), basis)
        window = am.dct_decode(targets, basis)[0, 0]  # (8, 3)
        expected = torch.cat([traj[0, :1].expand(5, 3), traj[0, :3]], dim=0)
        assert torch.allclose(window, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


class TestLosses:
    def test_reconstruction_loss_zero_on_match(self):
        c = torch.randn(2, 4, 32, 20)
        assert am.reconstruction_loss(c, c).item() == pytest.approx(0.0, abs=1e-8)

    def test_reconstruction_loss_unmasked_matches_suturebot_formula(self):
        pred, target = torch.randn(2, 32, 20), torch.randn(2, 32, 20)
        w = am.frequency_weights(32)
        expected = ((pred - target).pow(2).mean(dim=-1) * w).sum(dim=-1).mean() / w.sum()
        assert am.reconstruction_loss(pred, target).item() == pytest.approx(expected.item(), rel=1e-6)

    def test_reconstruction_loss_channel_mask_ignores_masked(self):
        pred = torch.randn(2, 32, 20)
        target = pred.clone()
        # Corrupt masked channels only — loss must stay zero.
        mask = torch.ones(20)
        mask[10:] = 0.0
        pred_corrupt = pred.clone()
        pred_corrupt[..., 10:] += 100.0
        assert am.reconstruction_loss(pred_corrupt, target, channel_mask=mask).item() == pytest.approx(
            0.0, abs=1e-8
        )
        # Corrupting a VALID channel must register.
        pred_corrupt2 = pred.clone()
        pred_corrupt2[..., 0] += 1.0
        assert am.reconstruction_loss(pred_corrupt2, target, channel_mask=mask).item() > 0

    def test_temporal_consistency_zero_for_shifted_windows(self):
        """Exact coefficients of two overlapping windows of ONE trajectory
        (full basis) must agree on the decoded overlap."""
        L, offset = 16, 4
        basis = am.dct_basis(L, L)
        traj = torch.randn(40, 6)
        win_a = traj[10 : 10 + L]  # ends at t
        win_b = traj[10 + offset : 10 + offset + L]  # ends at t + offset
        ca, cb = am.dct_encode(win_a, basis), am.dct_encode(win_b, basis)
        assert am.temporal_consistency_loss(ca, cb, offset, basis).item() == pytest.approx(0.0, abs=1e-8)

    def test_frequency_weights_monotone_decreasing(self):
        w = am.frequency_weights(32)
        assert (w[:-1] >= w[1:]).all()
        assert w[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Vector quantizer
# ---------------------------------------------------------------------------


class TestVectorQuantizer:
    def test_eval_before_init_raises(self):
        vq = am.VectorQuantizer(codebook_size=8, dim=4)
        vq.eval()
        with pytest.raises(RuntimeError, match="before initialization"):
            vq(torch.randn(4, 4))

    def test_data_init_seeds_from_batch(self):
        vq = am.VectorQuantizer(codebook_size=8, dim=4)
        vq.train()
        h = torch.randn(64, 4) * 5 + 3
        vq(h)
        assert bool(vq.initialized)
        # Codebook lives near the data, not near the origin.
        assert (vq.codebook.weight.norm(dim=1) > 1.0).any()

    def test_straight_through_gradient(self):
        vq = am.VectorQuantizer(codebook_size=8, dim=4)
        vq.train()
        h = torch.randn(16, 4, requires_grad=True)
        quantized, _, vq_loss, _ = vq(h)
        (quantized.sum() + vq_loss).backward()
        assert h.grad is not None
        assert h.grad.abs().sum() > 0

    def test_ema_moves_codebook(self):
        vq = am.VectorQuantizer(codebook_size=4, dim=4, decay=0.5)
        vq.train()
        vq(torch.randn(32, 4))
        before = vq.codebook.weight.clone()
        vq(torch.randn(32, 4) + 2.0)
        assert not torch.allclose(before, vq.codebook.weight)

    def test_dead_code_restart(self):
        vq = am.VectorQuantizer(codebook_size=4, dim=4, dead_steps=2)
        vq.train()
        # Seed, then park a far-away code no batch will select.
        vq(torch.randn(64, 4))
        vq.codebook.weight.data[0] = torch.full((4,), 1e4)
        vq.ema_embed[0] = vq.codebook.weight.data[0]
        for _ in range(3):
            vq(torch.randn(64, 4))
        # The dead entry must have been re-seeded near the data.
        assert vq.codebook.weight[0].norm() < 1e3

    def test_mask_excludes_positions_from_stats(self):
        vq = am.VectorQuantizer(codebook_size=8, dim=4)
        vq.train()
        h = torch.randn(32, 4)
        vq(h)  # init from clean data
        book_before = vq.codebook.weight.clone()
        # Garbage at masked positions must not move the codebook...
        garbage = torch.full((32, 4), 1e6)
        mask = torch.zeros(32)
        vq(garbage, mask=mask)
        # ...beyond the pure-decay drift an all-masked batch applies.
        assert vq.codebook.weight.abs().max() < 1e5
        assert torch.allclose(book_before, vq.codebook.weight, atol=1e-3)

    def test_perplexity_healthy_on_diverse_data(self):
        vq = am.VectorQuantizer(codebook_size=32, dim=8)
        vq.train()
        perp = None
        for _ in range(20):
            _, _, _, perp = vq(torch.randn(256, 8))
        assert perp.item() > 4.0  # far from collapse (1.0)


# ---------------------------------------------------------------------------
# ActionMemoryEncoder core
# ---------------------------------------------------------------------------


class TestActionMemoryEncoder:
    @pytest.fixture()
    def enc(self):
        torch.manual_seed(1)
        return am.ActionMemoryEncoder(
            input_dim=12, hidden_dim=16, num_coeffs=4, recon_dim=5, recon_len=8,
            code_dim=6, codebook_size=8,
        )

    def test_shapes(self, enc):
        _train_forward(enc)
        hidden, _ = enc(torch.randn(2, 10, 12))
        assert hidden.shape == (2, 10, 16)
        assert enc.reconstruct(hidden).shape == (2, 10, 4, 5)
        code, vq_loss, perp = enc.code(hidden)
        assert code.shape == (2, 10, 6)
        assert vq_loss.ndim == 0 and perp.ndim == 0
        assert enc.reconstruct_from_code(code).shape == (2, 10, 4, 5)

    def test_code_bounded_by_tanh(self, enc):
        _train_forward(enc)
        hidden, _ = enc(torch.randn(2, 10, 12))
        code, _, _ = enc.code(hidden)
        assert code.abs().max() <= 1.0

    def test_encode_episode_chunked_equals_full(self, enc):
        """Chunked streaming must equal the one-shot pass — no state leaks."""
        _train_forward(enc)
        enc.eval()
        inputs = torch.randn(50, 12)
        full = enc.encode_episode(inputs, chunk=4096)
        chunked = enc.encode_episode(inputs, chunk=7)
        assert torch.allclose(full, chunked, atol=1e-6)

    def test_causality_future_cannot_change_past_codes(self, enc):
        """THE Phase-2 gate: codes at time t depend only on inputs [0..t]."""
        _train_forward(enc)
        enc.eval()
        inputs = torch.randn(40, 12)
        codes_before = enc.encode_episode(inputs)
        mutated = inputs.clone()
        mutated[25:] = torch.randn(15, 12) * 10  # rewrite the future
        codes_after = enc.encode_episode(mutated)
        assert torch.allclose(codes_before[:25], codes_after[:25], atol=1e-6)
        assert not torch.allclose(codes_before[25:], codes_after[25:], atol=1e-3)

    def test_step_matches_episode_encoding(self, enc):
        """Incremental serving path must reproduce the offline path exactly."""
        _train_forward(enc)
        enc.eval()
        inputs = torch.randn(12, 12)
        offline = enc.encode_episode(inputs)
        state = None
        online = []
        for t in range(inputs.shape[0]):
            code, state = enc.step(inputs[t], state)
            online.append(code)
        assert torch.allclose(offline, torch.stack(online), atol=1e-6)

    def test_pretraining_gradients_flow(self, enc):
        enc.train()
        inputs = torch.randn(2, 20, 12)
        traj = torch.randn(2, 20, 5)
        hidden, _ = enc(inputs)
        anchors = torch.tensor([7, 19])
        targets = am.build_dct_targets(traj, anchors, enc.basis)
        quantized, vq_loss, _ = enc.quantize(hidden[:, anchors])
        pred = enc.reconstruct(hidden[:, anchors])
        code = enc.code_head(quantized)
        pred_from_code = enc.reconstruct_from_code(code)
        loss = (
            am.reconstruction_loss(pred, targets)
            + am.reconstruction_loss(pred_from_code, targets)
            + vq_loss
        )
        loss.backward()
        grads = [p.grad for p in enc.lstm.parameters()]
        assert all(g is not None for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
        assert next(enc.code_head.parameters()).grad is not None


# ---------------------------------------------------------------------------
# Multi-embodiment wrapper
# ---------------------------------------------------------------------------


class TestMultiEmbodiment:
    @pytest.fixture()
    def menc(self):
        torch.manual_seed(2)
        return am.MultiEmbodimentActionMemoryEncoder(
            num_embodiments=9, embed_dim=16, hidden_dim=32,
            num_coeffs=4, recon_len=8, code_dim=contract.CODE_DIM, codebook_size=16,
        )

    def _inputs(self, b=2, t=10):
        state = torch.randn(b, t, am.CANONICAL_STATE_DIM)
        smask = torch.ones(b, am.CANONICAL_STATE_DIM)
        smask[:, 10:] = 0.0  # single-arm: second arm absent
        act = torch.randn(b, t, am.ACTION_DIM)
        amask = torch.ones(b, am.ACTION_DIM)
        amask[:, 20:] = 0.0
        eid = torch.tensor([3] * b)
        return state, smask, act, amask, eid

    def test_input_dim_wiring(self, menc):
        assert menc.input_dim == 20 + 20 + 44 + 44 + 16 == 144
        assert menc.encoder.input_dim == 144

    def test_default_code_dim_is_contract(self):
        m = am.MultiEmbodimentActionMemoryEncoder(num_embodiments=9)
        assert m.code_dim == contract.CODE_DIM == 132

    def test_assemble_masks_zero_absent_channels(self, menc):
        state, smask, act, amask, eid = self._inputs()
        x = menc.assemble(state, smask, act, amask, eid)
        assert x.shape == (2, 10, 144)
        # Masked state channels (10:20) and action channels (20:44) are zero.
        assert torch.count_nonzero(x[..., 10:20]) == 0
        assert torch.count_nonzero(x[..., 60 : 60 + 24]) == 0  # 40 + 20 = start of masked action span
        # The mask segments themselves carry the mask values.
        assert torch.equal(x[..., 20:40], smask.unsqueeze(1).expand(2, 10, 20))

    def test_embodiment_id_changes_codes(self, menc):
        state, smask, act, amask, _ = self._inputs()
        # Initialize VQ.
        menc.train()
        x = menc.assemble(state, smask, act, amask, torch.tensor([0, 1]))
        hidden, _ = menc(x)
        menc.quantize(hidden)
        menc.eval()
        h0, _ = menc(menc.assemble(state, smask, act, amask, torch.tensor([0, 0])))
        h1, _ = menc(menc.assemble(state, smask, act, amask, torch.tensor([5, 5])))
        assert not torch.allclose(h0, h1, atol=1e-4)

    def test_episode_and_step_agree(self, menc):
        _t = 9
        menc.train()
        state, smask, act, amask, eid = self._inputs(b=1, t=32)
        x = menc.assemble(state, smask, act, amask, eid[:1])
        hidden, _ = menc(x)
        menc.quantize(hidden)
        menc.eval()

        st = torch.randn(_t, am.CANONICAL_STATE_DIM)
        sm = torch.ones(am.CANONICAL_STATE_DIM)
        ac = torch.randn(_t, am.ACTION_DIM)
        aM = torch.ones(am.ACTION_DIM)
        eid1 = torch.tensor(4)
        offline = menc.encode_episode(st, sm, ac, aM, eid1)
        assert offline.shape == (_t, contract.CODE_DIM)
        state_lstm = None
        online = []
        for t in range(_t):
            code, state_lstm = menc.step(st[t], sm, ac[t], aM, eid1, state_lstm)
            online.append(code)
        assert torch.allclose(offline, torch.stack(online), atol=1e-6)

    def test_causality_multi_embodiment(self, menc):
        menc.train()
        state, smask, act, amask, eid = self._inputs(b=1, t=32)
        hidden, _ = menc(menc.assemble(state, smask, act, amask, eid[:1]))
        menc.quantize(hidden)
        menc.eval()

        st = torch.randn(30, am.CANONICAL_STATE_DIM)
        sm = torch.ones(am.CANONICAL_STATE_DIM)
        ac = torch.randn(30, am.ACTION_DIM)
        aM = torch.ones(am.ACTION_DIM)
        eid1 = torch.tensor(2)
        before = menc.encode_episode(st, sm, ac, aM, eid1)
        st2, ac2 = st.clone(), ac.clone()
        st2[20:] += 5.0
        ac2[20:] -= 5.0
        after = menc.encode_episode(st2, sm, ac2, aM, eid1)
        assert torch.allclose(before[:20], after[:20], atol=1e-6)

    def test_code_reshapes_to_memory_rows(self, menc):
        """The 132D code must tile exactly onto (3, 44) — the Phase-3 rows."""
        code = torch.randn(contract.CODE_DIM)
        rows = code.view(contract.NUM_MEMORY_SLOTS, contract.ACTION_DIM)
        assert rows.shape == (3, 44)
