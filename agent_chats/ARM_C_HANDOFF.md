# Arm C build handoff — Open-H forward-dynamics CAMP (Phase-2b producer + launch workflow)

**For:** the agent with DRACO SSH access.
**From:** Sean (shuver@nvidia.com) via Claude Code, 2026-07-24.
**Owner of the plan:** Lukas. This handoff implements *his* 6-item plan, on *his* branch. Coordinate with him; do not fork a divergent copy. New work flows as commits on his branch (or format-patches against it), never bundles.

## 1. Context

Arms A + B (no-memory / raw-history baselines) are running. Arm C is the CAMP memory arm for the **Open-H multi-embodiment surgical forward-dynamics** finetune (this is distinct from the SutureBot *policy-side* CAMP on EOS, which already ran end-to-end and serves as the reference implementation).

Lukas's current branch has the **consumer** side only: memory encoder, joiner, injection, model patch, and a `camp_memory_tracks.py` contract with a `__random__` synthetic-track mode. It has **no producer**: no canonical-state extraction, no Phase-2b pretraining script/checkpoint, no track exporter, no real tracks. Your job is the producer plus the launch workflow.

Settled design (manager's choices — do NOT substitute the paper-literal joint-finetuning variant):
- H16 history; **frozen offline 132D tracks** (memory encoder is pretrained separately and frozen; tracks are exported artifacts consumed at SFT time)
- **Canonical 20D state**, shared across all 9 embodiment schemas
- **Shared multi-embodiment causal encoder** (single encoder pretrained across all embodiments)
- Arm C is **not gated** on A/B results — Sean's explicit call (2026-07-24). Build and run as if needed.

Reference paper: "Remember what you did?" (CAMP), arXiv 2606.21188 — LSTM memory pretrained to reconstruct past action trajectory in DCT-coefficient space (frequency-weighted recon loss + temporal-consistency loss), VQ-quantized code conditions the head.

## 2. Phase 0 — Discover and verify (do this first; do not trust guesses)

Facts I could NOT verify from the Mac checkout — establish each one and record it at the top of your worklog:

1. **Locate Lukas's branch and repo path on DRACO.** Unknown to me. Ask Lukas, or search working copies under `/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/` and home dirs for `camp_action_memory.py` / `camp_memory_tracks.py`. Record repo path + branch + tip SHA.
2. **Get the design doc** Lukas calls "**5.6-sol**" (also defines "31-row packing" and the "Phase-3b exemption"). Ask Lukas directly. Do not implement the exporter contract from inference — read `camp_memory_tracks.py` on his branch as the ground-truth contract and the doc for intent.
3. **Confirm the exact contract numbers** from the branch source: 132D track layout, 20D canonical state table, H16 semantics, 31-row packing, what the Phase-3b exemption exempts.
4. **Locate the Arm-A checkpoint** to warm-start from (Lukas's plan: warm-start **exactly once**, same ckpt as the other arms' lineage). Record its path + iteration + hash.
5. **Locate Arm B's output/checkpoint roots** — you must prove Arm C's paths are isolated from them (see §8, incident 3).
6. **Enumerate the 36 Draco data leaves / 9 embodiment schemas.** The schema definitions exist in the cosmos cookbooks repo: `cookbooks/cosmos3/generator/action/finetune/scripts/surgical_action_schemas.py` (+ `audit_surgical_action_schemas.py`, `build_surgical_manifest.py`, `validate_surgical_manifest.py`, tests). Reuse; don't rewrite.
7. **Reference implementation for the pretrain half:** branch `camp-v2-memory` (also on the shared bare repo `/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/shared/cosmos3-dvrk.git`): `cosmos3/scripts/pretrain_action_memory.py` (train + export modes, parquet-direct), `cosmos3/_src/vfm/models/action_memory.py`. Borrow: DCT recon objective, hardened VQ (data-dependent init, EMA decay 0.99, dead-code restarts, masked full-seq stats), bucketed batching, min-episode-length filter, coverage asserts, provenance hashing.

DRACO Slurm facts (verified 2026-07-17): account `healthcareeng_holoscan`; GPU partitions `batch_block1/3/4` with a **4-hour wall cap** → chained resubmit via `--dependency=afterany`; `backfill` allows 16 h; enroot sqsh containers (`$HOST_HOME/enroot/cuda-base.sqsh`). login-01 rejected Sean's key at least once; login-02 worked.

## 3. Phase 1 — Canonical episode extraction + validation

- Implement canonical 20D-state extraction for **all 9 embodiment schemas / all 36 leaves**, reusing the existing Open-H transforms and the `c3hss-v1` normalization already in `camp_action_memory.py` on Lukas's branch. The extraction table itself must come from the design doc + branch, not invention.
- Validation harness, complete-or-fail: per-leaf episode counts, per-schema dim checks, NaN/range checks against c3hss-v1 stats, min-episode-length filter (log the drop count — the policy side dropped 88/2389 episodes under its length floor; expect a tail here too).
- Emit a machine-readable extraction manifest (per-leaf: episodes in/out/dropped + reason) — the preflight in Phase 3 consumes it.

## 4. Phase 2 — Phase-2b pretraining (shared causal memory encoder)

- Training script + a DRACO sbatch. Resumable checkpoints; provenance (config, data manifest hash, git SHA) written into the checkpoint dir.
- **Health gate before any export** (this is where the policy side failed silently first): log per-iteration VQ perplexity and used-code count (or the equivalent for this encoder). Go/no-go: sustained perplexity well off the floor and used-code count near the codebook size, not just falling recon loss. The policy-side first run had recon 69.7→~3 while the codebook collapsed to 1–6 of 512 codes; the hardened-VQ rerun passed with 512/512 used. **Do not export from a collapsed run.**
- Expect the pretrain itself to be cheap (minutes on the policy side) — the long pole is data plumbing. Run a small real-data pretrain **early** to shake out schema surprises across the 36 leaves; hparam sweeps are essentially free.

## 5. Phase 3 — Exporter + preflight

- Exporter writes the **exact** `camp_memory_tracks.py` contract (dims, dtype, packing, naming, directory layout — read the consumer source, byte-level).
- Hash and record: encoder checkpoint, normalization stats, export manifest. The Arm-C training config must record which track artifacts it consumed (checkpoint hash in the manifest).
- Preflight: complete-or-fail across all 36 leaves and every episode — any missing leaf/episode is a hard failure, not a warning. Verify pack shape (31-row packing per the contract) and the Phase-3b exemption logic explicitly.

## 6. Phase 4 — Smokes (production-identical config)

1. `__random__` tracks first: verify 31-row packing, Phase-3b exemption behavior, join alignment (track↔episode↔window indexing — off-by-one here is fatal and silent), finite loss, and a short train run to non-NaN convergence.
2. Then real tracks, same checks.
3. Smoke on the production node shape if queue allows; delete smoke checkpoints afterward (a prior smoke left 411 GB behind).

## 7. Phase 5 — Production launcher

- Dedicated **6-node / 48-GPU** Arm-C launcher, fully isolated from Arm B (separate sbatch, separate output/ckpt roots, distinct job names).
- Warm-start **exactly once** from the Arm-A checkpoint identified in Phase 0. If the framework here shares the cosmos3 checkpoint loader semantics, check whether an EMA-reinit override analogous to `checkpoint.keys_to_skip_loading=[net_ema.]` is needed so warm-start doesn't reinitialize trained modules — verify against this repo's loader, don't assume.
- Save every **50** iterations (Lukas's spec); if a pruner keeps only every-Nth permanent checkpoints, keep the permanent cadence **iteration-matched with Arms A/B** so the eval grid compares like with like.
- Sequential 4-hour array chained with `--dependency=afterany`, array length sized from **measured** Arm-C throughput (from the smoke), conservatively.
- **Verify the second link's log resumes from iter > 0.** A silent restart-from-zero has happened before; if it restarts, cancel and resubmit with an explicit resume path.

## 8. Hard-won rules (each cost us real time/storage on the policy side)

1. **VQ/codebook collapse** hides behind good recon loss. Gate on code-usage metrics (see §4).
2. **Provenance or it didn't happen**: hash best.pt / stats / manifest; the arm config records what it consumed.
3. **Pruner glob collision**: a cleanup glob (`*hist16*`) matched both a baseline dir and the memory-variant dir (`hist16_mem`) → wrong dir → silent no-op → **~22 TB**. Make Arm-C names un-globbable from A/B's and test every pruning/cleanup glob against all three arms' dirs.
4. **Chained-array resume** silently restarted from iter 0 once. Check every link.
5. **Preflight warns are worthless** — hard-fail on incomplete coverage.
6. **Container env mounts**: two separate launches broke on the shared venv's absolute UV-interpreter path needing an explicit mount (`.local/share/uv/python`). Bake required mounts into the launcher, and treat a missing data/track root mount as a launch blocker.
7. **Anything needing CUDA runs on the cluster, never on the Mac** (Sean's standing rule).

## 9. Deliverables / acceptance

- [ ] Extraction + validation for 9 schemas / 36 leaves, with manifest (Phase 1)
- [ ] Phase-2b pretrain script + DRACO sbatch, resumable, provenance'd; health gate documented and **passed** on real data (Phase 2)
- [ ] Exporter + complete-or-fail 36-leaf preflight (Phase 3)
- [ ] `__random__` smoke green, then real-track smoke green, production-identical config (Phase 4)
- [ ] 6-node/48-GPU launcher: isolated paths, single Arm-A warm-start, save-every-50, sized chained array, resume verified (Phase 5)
- [ ] Tests extended; README/runbook with the exact DRACO command sequence: **pretrain → export → preflight → random smoke → real-track smoke → production** (Phase 6)
- [ ] Worklog records every Phase-0 fact (paths, SHAs, hashes) and every gate result

Code + workflow ready is the deliverable; the scientific Arm-C finetune itself starts only after the real Phase-2b run and export pass their gates. Report back: Phase-0 findings first (especially anything that contradicts this doc), then gate results as they land.
