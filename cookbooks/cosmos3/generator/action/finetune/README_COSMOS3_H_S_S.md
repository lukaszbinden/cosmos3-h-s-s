# Cosmos3 Open-H Surgical Forward-Dynamics Fine-Tuning (44D)

Post-train Cosmos3-Nano (Qwen3-VL-8B + diffusion expert, MoT) into a surgical
world-foundation model on the [Open-H](https://huggingface.co/datasets/nvidia/Open-H)
multi-embodiment surgical robotics benchmark, using a unified **44-dimensional**
action-conditioning space.

This cookbook is the Cosmos Framework (`cosmos_framework`) port of the
cosmos3-internal experiment `action_fdm_open_h_sft_nano` (originally a YAML
recipe in the `cosmos3` package). It is delivered as a **cookbook +
`framework_patch/` overlay** — nothing in the released `cosmos_framework`
package is edited in place; the setup script clones a framework checkout and
rsyncs the patch over it (the same delivery pattern used by the
`sean-cosmos3_surgical_fd` cookbook).

The training mixture targets **maximum non-synthetic surgical coverage** of the
public Open-H-Embodiment release that fits the 1-/2-arm 44D Cartesian-pose
contract: CMR Versius, JHU (IMERSE + LCSR MIRACLE), Obuda, Stanford, Turin,
UC Berkeley, UCSD, and TU Dresden (grasping_retraction)
— **36 dataset leaves across 9 embodiment tags**. All video/state/action keys
are verified against each dataset's `meta/modality.json` (see
[Dataset mixture](#dataset-mixture)).

Relative to the released C-H-S-S checkpoint mixture (paper Table S4) it drops
datasets that are unavailable or incompatible in the public release — **Hamlyn**
(no endoscope camera), **USTC/Tuodao** (absent from the public tree), and
**Moon** (delta-xyz only) — and, after the modality audit, three leaves with no
`meta/modality.json` (**STAR-IL**, **ARCADE cholecystectomy**, **MIRA**).

> Status: the data stack was authored offline then reconciled against the live
> EOS `meta/modality.json` files (`scripts/inspect_openh_modality.py`). Before a
> production run still complete the cluster [Pre-flight](#pre-flight-must-run-on-the-cluster)
> (stats, CMR filter caches, smoke test).

## Table of Contents

- [Why 44D (and not 54D)](#why-44d-and-not-54d)
- [The 44D action space](#the-44d-action-space)
- [Dataset mixture](#dataset-mixture)
  - [Included (36 dataset leaves, 9 embodiment tags)](#included-36-dataset-leaves-9-embodiment-tags)
  - [Dropped after the modality audit](#dropped-after-the-modality-audit-no-usable-metamodalityjson)
  - [Excluded (and why)](#excluded-and-why)
- [Layout](#layout)
- [Setup](#setup)
- [Pre-flight (must run on the cluster)](#pre-flight-must-run-on-the-cluster)
- [Stats file naming & collision avoidance](#stats-file-naming--collision-avoidance)
- [Launch](#launch)
- [Licensing / provenance](#licensing--provenance)
- [Decision log and handoff notes](#decision-log-and-handoff-notes)
  - [CAMP port handoff (start here)](#camp-port-handoff-start-here)
  - [Embodiment tag vs institution (important)](#embodiment-tag-vs-institution-important)
  - [Modality-audit findings (per embodiment)](#modality-audit-findings-per-embodiment)
  - [Chronology of decisions](#chronology-of-decisions)
  - [Known gaps and open items](#known-gaps-and-open-items)
  - [How to re-derive / re-validate](#how-to-re-derive--re-validate)

## Why 44D (and not 54D)

Two prior efforts mapped Open-H surgical actions into a fixed vector:

- **cosmos3-internal** and the **publicly released Cosmos-H-Surgical-Simulator
  (C-H-S-S) checkpoint** use **44D** (CMR Versius ceiling: 30D actions + 14D
  state conditioning).
- **`sean-cosmos3_surgical_fd`** chose **54D** to carry richer CMR control /
  physical-arm context channels (20D dual-haptic pose/gripper + 10 CMR controls
  + 24 `observation.state` context).

**Framing fact:** `44 vs 54` affects **only CMR Versius**. Every other
embodiment is ≤20D natively and zero-pads up to the ceiling either way, and both
schemes are subsets of the same 100-wide CMR raw `action`/`observation.state`
columns. The two carry the **same 20D pose/gripper and the same 10D
hand-controller channels**; the entire delta is the CMR **context tail — 14D
(44D) vs 24D (54D)**. 54D's extra ~10 channels are a 5th arm slot, ICG state,
explicit linked-instrument types, exposed motion-scaling, and engageable flags.

Short comparison:

| | 44D | 54D |
| --- | --- | --- |
| CMR pose+gripper / controls | 20D / 10D | 20D / 10D (same) |
| CMR context tail | 14D | 24D (+5th arm, ICG, linked-instr types, scaling) |
| Non-CMR embodiments | ≤20D, zero-padded | identical (just wider padding) |
| Inert padding across ≈50% non-CMR half | smaller | larger |
| Hand-mapped categorical channels to audit | fewer | more (higher mis-label risk) |
| Extra info value | — | CMR-only, mostly categorical/quasi-static, partly redundant with the image |

This cookbook uses **44D**. Rationale:

1. **Community continuity.** The already-released C-H-S-S checkpoint is 44D, so
   the community is already building tooling, inference specs, and action
   viewers around 44D. A successor that silently changes the action dimension
   creates a migration burden and confusion. Changing it needs a *strong*
   justification; absent that, 44D is the safer choice on behalf of the
   community.
2. **The technical trade is asymmetric and favors 44D.** 54D's benefit is
   **speculative, CMR-only, and partly redundant** with the conditioning image
   (its extra channels are mostly categorical/quasi-static scene-state); its
   cost — wider inert action tail across the ≈50% non-CMR half, more compute, and
   more hand-mapped channels that can be silently mis-labeled — is **certain and
   global**. No ablation (here or in GR00T-H) has shown the extra channels help
   generation. (Full per-channel pros/cons in the standalone note below.)
3. **Matches the design source.** The data stack ported here (the
   `gr00t_dreams` registry + `OPEN_H_DATASET_SPECS` + per-embodiment transforms)
   is cosmos3-internal's 44D design verbatim. Re-deriving it at 54D would mean
   re-authoring the CMR transform, re-computing all stats, and re-validating.
4. **The 54D gain is CMR-only and additive-later.** The 44D layout already
   includes a 14D CMR state-conditioning tail (engagement, arm link, instrument
   type, color, electrosurgery mode). If the additional CMR context proves
   valuable, introduce it later as a **separate, clearly-versioned model
   variant** rather than a breaking change to the default.

**Conclusion / recommendation:** ship the public successor at **44D** — it wins
on both the technical merits (asymmetric trade: speculative CMR-local gain vs
certain global cost) and community continuity (the released checkpoint and its
tooling are 44D). Treat 54D as a **falsifiable, CMR-only experiment**: adopt it
only if a CMR-only 44-vs-54 ablation (same data/seed, scoring CMR FDS/PSNR)
shows a real margin, and then only as a separately versioned variant. A cheap
middle path first: keep 44D but swap which context channels fill the 14D tail
(e.g. ICG state or the 5th-arm slot in place of HUD color). If a future run does
adopt 54D it must: (a) re-author the CMR action transform to emit the 10
controls + 24 context channels, (b) bump `MAX_ACTION_DIM` and the model
`max_action_dim`, (c) recompute all `stats_cosmos*.json`, and (d) re-export
under a new checkpoint name to avoid confusing 44D consumers.

For the in-depth comparison **on technical merits only** (ignoring the
released-checkpoint argument) — exact per-channel decomposition, full pros/cons,
the "crux", and the ablation/middle-path proposal — see the standalone note
[`NOTE_44D_vs_54D_action_space.md`](./NOTE_44D_vs_54D_action_space.md).

## The 44D action space

Per-timestep vector of shape `(44,)`; 12 timesteps per training sample
(`(12, 44)`). Only CMR Versius uses all 44 dims; every other embodiment is
zero-padded up to 44 and the model masks loss/noise/velocity on the padded
channels (`action_channel_masking=True`).

| Block | Dims | Contents |
| --- | --- | --- |
| Actions | 0–29 (30D) | 2x EEF pose (xyz_rel 3 + rot6d_rel 6) + 2x gripper + energy(2) + thumbstick(6) + clutch(2) |
| State conditioning | 30–43 (14D) | haptic engaged(2) + arm-linked-to-haptic(2) + instrument type(4) + HUD color(4) + electrosurgery mode(2) |

Per-embodiment native widths (pre-pad), for the embodiments in this mixture:
CMR 44 - dual-arm dVRK 20 - Turin 18 (no grippers) - single-arm/STAR-IL 9–10 -
MIRA ~10 (delta). See
`framework_patch/cosmos_framework/data/vfm/action/gr00t_dreams/groot_configs.py`
(`EMBODIMENT_REGISTRY`, `MAX_ACTION_DIM = 44`) and the upstream
`cosmos-h-surgical-simulator-public/scripts/README_ACTION_SPACE.md`.

## Dataset mixture

The single source of truth is `OPEN_H_DATASET_SPECS` in
[`groot_configs.py`](framework_patch/cosmos_framework/data/vfm/action/gr00t_dreams/groot_configs.py).
Every path is grounded in the **public Open-H-Embodiment tree** on the EOS
cluster (`/lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment/Surgical`,
verified against [`doc/open-h-embodiment_dataset_folder_structure.txt`](../../../../../doc/open-h-embodiment_dataset_folder_structure.txt)),
NOT the C-H-S-S internal re-converted mirror.

Goal: **maximum non-synthetic surgical coverage** that fits the 1-/2-arm 44D
Cartesian-pose contract and has a usable endoscope/stereo/scope view. CMR
Versius is held at exactly 50%; the rest is frame-proportional (real
`meta/info.json::total_frames`), normalized so the non-CMR pool sums to ~4.0.

All video/state/action keys below have been **verified against each dataset's
`meta/modality.json` on EOS** (via `scripts/inspect_openh_modality.py` →
`doc/openh_modality_report.json`), and an automated cross-check confirms every
registry key matches a real modality subkey. The `video_keys` use the modality
**subkey name** (e.g. `video.endoscope_left`), which can differ from the
on-disk video folder (`observation.images.endoscope.left`).

### Included (36 dataset leaves, 9 embodiment tags)

| Group | Leaves | Native dim | modality video key |
| --- | --- | --- | --- |
| CMR Versius | cholecystectomy, hysterectomy, inguinal_hernia, prostatectomy | 44 | `video.endoscope` |
| JHU IMERSE (dVRK-Si) | wound_closure, srth_porcine_chole, suturebot, nephfat, srt_needle_pickup_handover, cao_cautery_combined, srt_tissue_lift | 20 | `video.endoscope_left` |
| JHU LCSR MIRACLE | miracle/{prepare_to_pierce, needle_pick_up, needle_regrasp} | 20 | `video.camera_left` |
| Obuda dVRK | all 11 task leaves (frs_dome, pork, pegtransfer×2, rollercoaster, needlethreading×2, seaspike×3, skinphantom) | 20 | `video.endoscope_left` |
| Stanford real dVRK | needle_transfer, tissue_retraction, peg_transfer | 20 (Euler w6) | `video.endoscope_left` |
| Turin MITIC | ex_vivo, plastic_pad, plastic_pad_3dmed, plastic_tube | 18 (no grippers) | `video.endoscope_left` |
| UCSD | surgical_learning_dataset, surgical_learning_dataset2 | 20 | `video.camera_left` |
| UC Berkeley | debridement_lerobot | 20 | `video.camera_left` |
| TUD TUNDRA | 260131_tundra_dataset/grasping_retraction | 10 | `video.laparoscope_left` |

All native widths are zero-padded to the unified 44D.

### Dropped (no usable `meta/modality.json`, malformed schema, or partial staging)

These were in an earlier draft but are removed from the mixture because the gr00t
LeRobot loader cannot consume them as staged — missing/malformed `modality.json`,
a per-DOF column layout the loader can't slice, or incomplete video staging.
Where applicable, registry entries (`jhu_imerse`, `virtual_incision_mira`) and
enum tags are kept **dormant** for easy re-add once the data/metadata is fixed:

| Dataset / leaf | Status |
| --- | --- |
| `jhu/imerse/star_il/star_il` (STAR-IL) | no modality.json |
| `jhu/lcsr/arcade/cholecystectomy` | no modality.json on disk |
| `jhu/lcsr/arcade/cautery` | **half-staged**: 22 parquet episodes but only 12 `endoscope.left` .mp4 (episodes 12–21 have raw frames under `images/` but no video) → ~half the windows FileNotFoundError. Smallest leaf (5,288 fr, 0.05% of pool) → dropped rather than re-stage. |
| `jhu/lcsr/smarts/SurgSync-stitch-coldcut/{P1,P2,P3}` | malformed modality.json + per-DOF scalar columns (`observation.cartesian_state.psm1.pose.position.x` … not a single sliceable vector); the gr00t loader can't assemble them without a parquet re-conversion. ≈103k fr. |
| `jhu/lcsr/smarts/SurgSync-multitask/{P1,P2,P3,P4}` | no modality.json — raw dVRK-Si ROS dump (per-DOF dotted columns + ECM arm), ≈64k fr |
| `virtual_incision/150_episodes_mira_needle_lift` (MIRA) | no modality.json (bespoke column layout) |

### Excluded (and why)

| Dataset / leaf | Reason |
| --- | --- |
| Hamlyn/Imperial (whole group) | public release has NO endoscope camera (only `color`/`depth`/`wrist_{left,right}`) — incompatible with the endoscope-conditioned FD setup |
| USTC/Tuodao (whole group) | NOT present in the public open-h-embodiment tree (was only in C-H-S-S v1's internal mirror) |
| Moon Surgical | delta-xyz only, no verified rotation action |
| UTenn (all 4 leaves) | NO endoscope-left camera (only `rgb`/`color`/`depth`/`tool_segmentation`/`part_id`) — incompatible with the endoscope-conditioned FD setup (same reason as Hamlyn). Newly staged 2026-06-24; re-checked and still no paired endoscope view. |
| Rob Surgical | 3-arm / 27D layout, unsupported by the 1-/2-arm 44D contract |
| UIC (`uic_crcd_lerobot`) | joint-only action, no verified Cartesian schema |
| Semaphor | manual laparoscopic tools, third-person view only, no robot kinematics |
| HK PolyU | synthetic (Sim. dVRK) |
| UT Austin (`colonoscope-lerobot`) | flexible colonoscopy navigation, not a pose-arm manipulation dataset |
| Balgrist, CUHK, HKBU, ImFusion, TUM CAMP | ultrasound (US) modality |
| SanoScience | synthetic (XR simulator) |
| CMR dry_box / peg_transfer | benchtop / unverified schema |
| UCSD retraction_dataset3 / retraction_failurecase | unverified schema |
| TUD endoscope_guidance | 4D delta-tip schema, incompatible with the pose-arm contract |

This set is the available `open-h-embodiment` surgical coverage that fits the
44D contract and has verified modality metadata. See
[Decision log and handoff notes](#decision-log-and-handoff-notes) for the full
Table S1 cross-reference reasoning and the chronology of inclusion/exclusion
decisions.

## Layout

```text
finetune/
  README_COSMOS3_H_S_S.md        # this file (cookbook guide + decision log)
  framework_patch/               # rsync'd over a cosmos-framework checkout at setup
    cosmos_framework/
      data/vfm/action/
        gr00t_dreams/            # ported 44D registry + transforms + LeRobot dataset
        open_h_dataset.py        # OpenHMixedLeRobotDataset (multi-embodiment mixture)
        domain_utils.py          # framework domain_utils + surgical embodiment IDs
        datasets/openh_sft_dataset.py   # get_action_openh_sft_dataset (-> ActionTransformPipeline)
      configs/base/experiment/action/posttrain_config/
        action_fdm_open_h_sft_nano.py   # the registered experiment (44D, 480 res, FD)
  toml/sft_config/
    action_fdm_open_h_sft_nano.toml     # run-level scalars
  scripts/
    setup_workspace.sh                  # clone + uv sync + overlay + register + stage
    apply_overlay.sh                    # local cp of framework_patch/ onto the installed cosmos_framework (run at job start)
    _eos_torchrun_inner.sh              # in-container torchrun wrapper (stamps the overlay, then torchrun)
    slurm_smoke.sbatch                  # 1-node, 10-iter end-to-end smoke
    slurm_train.sbatch                  # 6-node (48 GPU) resumable main train
    resubmit_until_done.sh              # resubmit until TARGET_ITER
    inspect_openh_modality.py           # dump modality.json/info.json/parquet checks (EOS)
    audit_openh_action_schemas.py       # fail-closed schema audit (run before training)
    compute_openh_action_stats.py       # per-embodiment stats_cosmos.json
    compute_cmr_filtered_episodes_cache.py  # CMR clutch-aware filter caches
    derive_episodes_jsonl.py            # rebuild missing meta/episodes.jsonl
    estimate_training_compute.py        # cumulative training FLOP (EU AI Act 6ND)
```

The registry's video/state/action keys were reconciled against the live EOS
`meta/modality.json` files using `inspect_openh_modality.py`; the resulting
report is saved at `doc/openh_modality_report.json`.

## Setup

```bash
# WORKSPACE defaults to the REPO ROOT (this repo) — the framework checkout goes
# to $WORKSPACE/packages/cosmos3 and env.sh to $WORKSPACE/env.sh, both gitignored.
# Override WORKSPACE only to put the runtime tree on a different filesystem:
#   export WORKSPACE=/scratch/$USER/cosmos3-h-s-s-workspace
export OPENH_SURGICAL_ROOT=/lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment/Surgical
export BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir>
bash scripts/setup_workspace.sh
```

`setup_workspace.sh` clones Cosmos Framework into `$WORKSPACE/packages/cosmos3`,
installs the CUDA 13 training environment (`cu130-train`; use
`COSMOS3_UV_GROUP=cu128-train` + a PyTorch container for older drivers), applies
the overlay (via `apply_overlay.sh`), stages the TOML, and downloads the Wan2.2
VAE.

### Apply / re-apply the overlay (`apply_overlay.sh`)

`cosmos_framework` is an **installed dependency**, not source you edit/commit.
Your overlay edits live in git under `framework_patch/`; `scripts/apply_overlay.sh`
**stamps them onto the installed package with a local file copy** (`cp`, no
rsync, no network). Run it:

- **at the start of every job/session**, because reinstalling or re-syncing the
  dependency reverts the package to pristine; and
- **after any git change to `framework_patch/`** (e.g. `groot_configs.py`,
  `dataset.py`).

It also fixes a venv that errors with
`ModuleNotFoundError: ... cosmos_framework.data.vfm.action.gr00t_dreams`. The
script: copies `framework_patch/cosmos_framework/*` over the installed package,
registers the `action_fdm_open_h_sft_nano` experiment in `config.py`
(idempotent), installs the extra deps (`albumentations`, `imageio`, `dm-tree`),
and verifies the overlaid modules import.

```bash
# activate the venv that has cosmos_framework, then:
bash scripts/apply_overlay.sh                              # auto-detects the install
bash scripts/apply_overlay.sh --framework-dir /path/to/site-packages   # explicit
bash scripts/apply_overlay.sh --dry-run                   # list files that would copy
bash scripts/apply_overlay.sh --no-deps                   # skip pip install
```

Target resolution order: `--framework-dir` → `$COSMOS3_FRAMEWORK_DIR` →
auto-detect from the active `python` (`import cosmos_framework`) →
`$WORKSPACE/packages/cosmos3`. The copy is purely local; you manage the overlay
*source* via git in `framework_patch/` and never edit the installed package by
hand.

## Pre-flight (must run on the cluster)

These steps need the real Open-H data and the patched venv. Do them before a
production run:

1. **Stage the Cosmos3-Nano DCP base checkpoint** at `$BASE_CHECKPOINT_PATH`
   (`python -m cosmos_framework.scripts.convert_model_to_dcp ...`).
2. **Re-confirm the schemas** (the registry keys were already reconciled against
   the modality report; this is a fail-closed re-check, e.g. after a data
   refresh):
   ```bash
   source "$WORKSPACE/packages/cosmos3/.venv/bin/activate"
   python scripts/audit_openh_action_schemas.py --root "$OPENH_SURGICAL_ROOT"
   # to regenerate the modality report itself:
   python scripts/inspect_openh_modality.py --root "$OPENH_SURGICAL_ROOT" \
       --included-only --sample-rows 256 --output doc/openh_modality_report.json
   ```
   If it reports a missing modality key, FIX the `EMBODIMENT_REGISTRY` entry in
   `groot_configs.py`.
3. **Compute normalization stats** — writes **experiment-postfixed** files into
   each dataset's `meta/` (see [Stats file naming & collision avoidance](#stats-file-naming--collision-avoidance)).
   With `COSMOS_OPENH_STATS_POSTFIX=c3hss-v1` (set by `setup_workspace.sh` /
   `env.sh`) this writes `meta/stats_cosmos-c3hss-v1.json` (and CMR
   `meta/stats_cosmos-44D-c3hss-v1.json`):
   ```bash
   source "$WORKSPACE/env.sh"   # exports COSMOS_OPENH_STATS_POSTFIX
   python scripts/compute_openh_action_stats.py --root "$OPENH_SURGICAL_ROOT" \
       --postfix "$COSMOS_OPENH_STATS_POSTFIX" --experiment-id c3hss_openh_44d_v1
   ```
   The training loader reads the same `COSMOS_OPENH_STATS_POSTFIX` and
   strict-matches it. Each file embeds a `_provenance` block and an archival
   sidecar is written; an existing file with differing provenance is NOT
   overwritten without `--force`.
4. **Build the CMR clutch-aware filter caches** (content-addressed by
   horizon/stride; safe to share across experiments — not postfixed):
   ```bash
   for proc in cholecystectomy hysterectomy inguinal_hernia prostatectomy; do
     python scripts/compute_cmr_filtered_episodes_cache.py \
       --dataset-path "$OPENH_SURGICAL_ROOT/cmr_surgical/$proc"
   done
   ```
5. **Smoke test** (1 node, 10 iters; exercises every dataset + stats files):
   ```bash
   sbatch scripts/slurm_smoke.sbatch
   ```

## Stats file naming & collision avoidance

Post-transform normalization stats are written **into each dataset's shared
`meta/` directory on the canonical Open-H tree** (the action transforms change
dimensionality — e.g. 7D quat → 9D rot6d — so raw `stats.json` is unusable).
Because that tree is shared, **independent experiments collide** on the default
`stats_cosmos.json` / `stats_cosmos-44D.json` filenames — concretely, a
colleague's 54D run already placed `stats_cosmos.json` in those `meta/` dirs, so
a naive 44D run would either overwrite theirs or silently read theirs.

To prevent this, the stats filename carries an **experiment postfix** from the
`COSMOS_OPENH_STATS_POSTFIX` env var (mirrors the od-hamlyn-cmr
`CMR_28D_EXP_POSTFIX` pattern). With `COSMOS_OPENH_STATS_POSTFIX=c3hss-v1`:

| Embodiment | File written / read |
| --- | --- |
| CMR Versius | `meta/stats_cosmos-44D-c3hss-v1.json` |
| other Open-H | `meta/stats_cosmos-c3hss-v1.json` |

Mechanics (both sides read the SAME env var, so they always agree):

- **A postfix is REQUIRED on both sides** — to prevent silent collisions with
  another experiment's stats in the shared `meta/` dir, an unset postfix is a
  **hard error**, not a warning:
  - Loader: raises if `COSMOS_OPENH_STATS_POSTFIX` is unset (escape hatch:
    `COSMOS_OPENH_ALLOW_BARE_STATS=1`).
  - Generator: exits non-zero if neither `--postfix` nor the env var is set
    (escape hatch: `--allow-bare-stats-filename`).
- **Loader** (`gr00t_dreams/data/dataset.py`): builds the filename from
  `COSMOS_OPENH_STATS_POSTFIX` and **strict-matches** it (a missing postfixed
  file is a hard error — it never silently falls back to the bare,
  possibly-someone-else's, `stats_cosmos.json`).
- **Generator** (`scripts/compute_openh_action_stats.py`): `--postfix` (or the
  env var) selects the same names, and additionally:
  - embeds a `_provenance` block in every file — `experiment_id`, `action_rep`
    (`44D`), the exact **dataset-set hash + leaf list**, horizon, stride,
    sampling, git rev, UTC timestamp (the loader skips `_`-prefixed keys);
  - writes an immutable **archival sidecar** `stats_cosmos[-44D].<experiment_id>.json`
    next to the live file so prior runs are never lost;
  - **refuses to overwrite** an existing file whose provenance differs (different
    experiment id or dataset set) unless `--force` — no silent clobber.

`setup_workspace.sh` sets `COSMOS_OPENH_STATS_POSTFIX` (default `c3hss-v1`) into
`env.sh`, and the Slurm launchers propagate it into the container, so training
reads the matching files. To run a *different* mixture/experiment later, just
pick a new postfix + `--experiment-id`.

Note: the **CMR clutch filter caches** (`cmr_filter_cache_*-44D.json`) are
content-addressed by horizon/stride only (independent of action space and
mixture) and are **not** produced by the colleague's stack, so they don't
collide and are intentionally **not** postfixed.

## Launch

EOS `batch` caps wall-time at ~4 h, but a 20k-step run needs far longer.
`slurm_train.sbatch` handles this with a **self-chaining SLURM array**
(`--array=0-11%1 --dependency=singleton --requeue`): up to 12 tasks run
**sequentially** (`%1` = one at a time), each ≤4 h. The Cosmos Framework trainer
**auto-resumes** from the run dir's `latest_checkpoint.txt` (the run-dir
checkpoint takes precedence over `checkpoint.load_path`), so each task continues
iteration / optimizer / scheduler / EMA / dataloader from the previous one — no
manual resubmission, no extra flags. Tasks that start past the target iteration
exit immediately.

```bash
# Submit FROM THE REPO ROOT (so the relative `logs/` output path resolves and
# WORKSPACE defaults correctly). data_parallel_shard_degree auto-set to
# nodes*8; resume is automatic across the array tasks.
cd "$WORKSPACE"   # = repo root by default
sbatch cookbooks/cosmos3/generator/action/finetune/scripts/slurm_train.sbatch
```

Per-task logs land in `$WORKSPACE/logs/openh44d_<arrayJobId>_<taskIdx>.out`
(each task logs a startup banner showing whether it RESUMED or COLD-STARTED).

Alternative (equivalent) chaining via an external loop — **use one OR the other,
not both** (they'd double-chain): comment out the `#SBATCH --array=...` line in
`slurm_train.sbatch`, then:

```bash
bash cookbooks/cosmos3/generator/action/finetune/scripts/resubmit_until_done.sh
```

To **cold-start** (ignore prior checkpoints and warm-start fresh from the base):
delete the run dir's `checkpoints/` tree first (there is no `--resume=false` flag
on this entrypoint).

Training output defaults to
`$WORKSPACE/outputs/train/cosmos3_action_surgical/action_open_h/action_fdm_open_h_sft_nano/`.

### Mixed-mode mid-training (Cosmos-Surg-3-Base)

Besides the FD-only recipe, there is a **mixed-mode mid-training** experiment,
`action_mixed_open_h_sft_nano`, that trains on the **same Open-H mixture** but on
**all three action modes jointly** — forward dynamics + inverse dynamics +
policy — exactly as the Cosmos 3 generator mid-training does (paper Table 6
"Generator mid-training data mixture", Action row; and the "Action sequence
configurations" figure). It produces the shared base **Cosmos-Surg-3-Base** that
the single-mode specialists (`Cosmos-Surg-3-Simulator` = FD, `Cosmos-Surg-3-Policy`
= policy, optional `Cosmos-Surg-3-ID`) then fine-tune from.

Everything (44D space, 480 tier, 13-frame/12-step horizon, model config,
optimizer/scheduler, token packing, warm-start with fresh 44D action heads) is
**identical** to the FD recipe — the *only* difference is the mode mixture — so
the two are directly comparable for the mixed-vs-single-mode ablation.

- **Files:** experiment `.../posttrain_config/action_mixed_open_h_sft_nano.py`,
  TOML `toml/sft_config/action_mixed_open_h_sft_nano.toml`, and **dedicated
  launchers** `scripts/slurm_train_mixed.sbatch` + `scripts/slurm_smoke_mixed.sbatch`
  (separate files so the FD launchers are untouched; no env-var toml switching).
- **Cluster shape:** **6 nodes × 8 = 48 GPUs**, identical to the FD run (same LR
  3.0e-5, same 45056 token cap) so the mixed run is a clean mirror of FD for the
  H1 ablation. `data_parallel_shard_degree` auto-follows the node count.
  `world_size` must be ≥ 3 (one rank min per mode). (A 32-GPU attempt OOM'd on
  resume — 32-way FSDP held ~1.5× more model-state per GPU with no headroom.)
- **Mode ratios** (`_MODE_RATIOS` in the experiment): default FD:ID:policy =
  **1:1:1** (equal), following the Cosmos 3 paper — Table 6 gives no sub-split,
  but the paper's own joint-mode study (App. "Synergy Between Action Modes")
  trains all three modes for equal per-mode steps. `RankPartitionedDataLoader`
  gives each mode its own ranks, so the global batch is mixed while each rank
  streams one mode (packing never mixes modes). Reweight toward FD only if the
  Simulator is the priority (the paper's ablation shows equal mixing helps ID/
  policy most and costs FD PSNR slightly).
- **Why mixed:** in FD mode all action tokens are conditioning (clean), so the
  action-**prediction** head (`llm2action`) gets no loss — FD-only trains the
  model to *use* actions, not *predict* them. ID and policy are what train action
  prediction, so a mixed base is a stronger start for policy/ID specialists.

Launch (submit from the repo root, same as the FD launcher):

```bash
cd "$WORKSPACE"
sbatch cookbooks/cosmos3/generator/action/finetune/scripts/slurm_smoke_mixed.sbatch  # 1-node smoke first
sbatch cookbooks/cosmos3/generator/action/finetune/scripts/slurm_train_mixed.sbatch  # 4-node run
```

The mixed run's output/run name is **`mm-C3-H-S-S-base`** (the Hydra experiment
key stays `action_mixed_open_h_sft_nano`), so it writes to its own run dir
(`.../action_open_h/mm-C3-H-S-S-base/`) and never collides with the FD run.
`apply_overlay.sh` registers both experiments; the same stats / filter-cache
artifacts are reused (mode does not affect normalization stats).

## Training compute (EU AI Act 6ND)

For regulatory reporting (`doc/Cummulative Compute Calculation.pdf`), cumulative
training compute is estimated as `C ≈ 6·N·D` (N = total params, D = training
examples seen). `scripts/estimate_training_compute.py` computes it from the
committed run shape and prints both the **compute basis** (`D_seen` = examples
the run processes) and the **epoch basis** (`D_dataset` = unique windows in the
mixture):

```bash
# EVERYTHING at once (compute + dataset/epochs + energy + CO2) via --all:
python scripts/estimate_training_compute.py --root "$OPENH_SURGICAL_ROOT" --all

# Just the headline compute number from the committed run config:
python scripts/estimate_training_compute.py --no-dataset

# With D_dataset / epoch count (inside the patched venv; reads info.json totals):
python scripts/estimate_training_compute.py --root "$OPENH_SURGICAL_ROOT" \
    --from-filter-cache --num-frames 13
```

`--all` implies `--energy` and re-enables the dataset section, so one command
prints the full report. Energy details below.

With the defaults (`N=8e9`, `max_samples_per_batch=64`, 6×8=48 GPUs,
`max_iter=20000`): `D_seen = 64·48·20000 ≈ 6.14e7` examples →
**C ≈ 2.9e18 FLOP** (~20× the Cosmos-Predict2.5 Open-H reference of 1.48e17,
driven mainly by the 2B→8B model size; ~6 orders of magnitude below the EU AI
Act 1e25 GPAISR threshold). The same script reproduces the Predict2.5 number as
a self-check: `--examples 12333333 --n-params 2e9` → 1.48e17.

> N is the **total** model parameters (Qwen3-VL-8B backbone) per the EU AI Act
> convention, even though this recipe only updates the gen-tower + action
> adapters. Recompute whenever `max_iter`, the cluster shape, or
> `max_samples_per_batch` change.

### Energy & CO2

Add `--energy` for an energy-first estimate (the FLOP figure is **not** used —
FLOP→energy needs an efficiency assumption equivalent to just knowing
GPU-hours):

```
Energy(kWh) = GPU_hours × (GPU_TDP_kW × util) × PUE
CO2e(kg)    = Energy(kWh) × grid_carbon_intensity(kgCO2e/kWh)
```

```bash
# From config (uses an ASSUMED sec_per_iter — see caveat):
python scripts/estimate_training_compute.py --no-dataset --energy

# Reportable, easiest: sum the WHOLE resubmit chain by job name (no ids needed).
# Bare --sacct-name uses the slurm_train.sbatch name (healthcareeng_holoscan-cosmos3.openh44d):
python scripts/estimate_training_compute.py --no-dataset --energy \
    --sacct-name --sacct-since 2026-06-20 --pue 1.1 --carbon-intensity 0.05

# Or pass explicit job id(s) — comma-separated for a resubmit chain:
python scripts/estimate_training_compute.py --no-dataset --energy \
    --from-sacct 5487578,5487612 --pue 1.1 --carbon-intensity 0.05

# Or supply GPU-hours directly if you already have them:
python scripts/estimate_training_compute.py --no-dataset --energy --gpu-hours 4200
```

Both query `sacct -X` and compute `Σ ElapsedRaw/3600 × AllocNodes ×
gpus_per_node` (run on a Slurm login node).

**How many job ids?** This run is resumable (`--time=03:55:00`, `--requeue`,
`resubmit_until_done.sh`), so a full 20k-step training is a **chain of many ~4h
jobs**, not one. You need GPU-hours from *all* of them:

- **`--sacct-name` (recommended)** — they all share `--job-name`, so this sums
  the entire chain with **zero ids** to enumerate. Add `--sacct-since` if the run
  predates the accounting window, and `--sacct-states COMPLETED,TIMEOUT` to drop
  failed retries.
- **`--from-sacct`** — pass **one id per `sbatch` in the chain** (a single
  requeued id already sums its own rows; separate resubmits get new ids). Get the
  list with `sacct --name=healthcareeng_holoscan-cosmos3.openh44d -X -n -o JobID | paste -sd,`.

With defaults (H100 700 W @ 70 %, PUE 1.2, 0.35 kgCO2e/kWh world-avg) and an
**assumed 6 s/iter** over 20k steps × 48 GPUs ≈ 1,600 GPU-hours →
**≈ 0.94 MWh, ≈ 0.33 tCO2e**. This is dominated by two unknowns:

- **`sec_per_iter`** (no throughput logged yet): at 3 / 6 / 12 s/iter →
  0.16 / 0.33 / 0.66 tCO2e. **Replace with `--gpu-hours <measured>`** for a real
  number.
- **Carbon intensity**: a low-carbon (hydro/nuclear) DC at 0.05 kgCO2e/kWh drops
  it to **≈ 0.06 tCO2e**.

> The Predict2.5 **2.65 tCO2e** is ~9,000 GPU-hours — that's *pretraining*
> scale, **not** the 1.48e17 Open-H finetune delta (~0.1 GPU-hr). So this
> finetune's footprint (sub-1 tCO2e) is **not** comparable to 2.65 tCO2e; the
> right comparison is finetune-to-finetune. Confirm what scope 2.65 covers
> before quoting them side by side.

## Licensing / provenance

The files under `framework_patch/.../gr00t_dreams/` retain their original
Apache-2.0 headers (they originate from the Cosmos-Predict2.5 /
Cosmos-H-Surgical-Simulator gr00t_dreams lineage). New files authored for this
cookbook carry the `cosmos_framework` package license (`OpenMDW-1.1`). Align
headers per your release policy before publishing.

## Decision log and handoff notes

This section records the choices made while building this cookbook and the
reasoning behind them, so a future maintainer can pick up without re-deriving
everything. Three source repos were involved:

- **cosmos3-internal** — the 44D Open-H surgical FD recipe (YAML +
  `cosmos3._src.vfm` package) that this cookbook is ported from (the *design
  source*).
- **cosmos3-h-s-s** (this repo) — the public OSS release (`cosmos_framework`
  package, Python LazyConfig configs) that this cookbook *targets*.
- **sean-cosmos3_surgical_fd** — a parallel effort against the same
  `cosmos_framework`, using a 54D CMR schema and a manifest-based dataset stack;
  the source of the *cookbook + `framework_patch/` overlay delivery pattern* and
  of several additional dataset candidates.

### CAMP port handoff (start here)

*Handoff 2026-07-22, Wes Huver → @lukaszbinden. Branch `c3-h-s-o-b_camp`
carries Gate 0 of the CAMP (learned action-memory) port onto this cookbook.
Questions: shuver@nvidia.com.*

**What this branch adds** on top of the 44D mixed-mode cookbook above:

- `a3d6f89` — framework-SHA pinning in `scripts/setup_workspace.sh`
  (`COSMOS3_FRAMEWORK_REF`), the canonical CAMP data contract
  (`framework_patch/cosmos_framework/data/vfm/action/camp_data_contract.py`),
  and its test suite (`tests/test_camp_data_contract.py`).
- `662cc8f` — CMR `conditioning_fps` fix: the adapter stamped the 30 Hz
  storage rate on `cmr_versius` samples (it bypasses `EMBODIMENT_REGISTRY`,
  so it hit the storage-FPS fallback); the actual training rate is
  60 Hz / stride 6 = **10 Hz**. Also covers the latent `suturebot` case and
  warns loudly if any future out-of-registry embodiment would silently fall
  back.

**Step 1 — framework SHA recovered and pinned.** Upstream cosmos-framework
has reorganised `data.vfm` → `data.generator`; this overlay still targets
`data.vfm`, so a fresh **unpinned** `setup_workspace.sh` clone will not even
import. The exact checkout that trained `mm-C3-H-S-S-base` was recovered from
the original EOS workspace on 2026-07-22:

```bash
300faa14daab3910be9d303c31708a0a1d6e4371
# 2026-06-22: action dataloader: episode-shuffle stream
# (fix DROID grad-norm instability) (#37)
```

Both setup helpers now use that SHA by default. A deliberate
`COSMOS3_FRAMEWORK_REF=<other-full-sha>` override remains possible; branch
names and abbreviated SHAs are rejected. Setup records the resolved SHA in
`packages/cosmos3-framework.lock` and the canonical cookbook helper exports
`COSMOS3_FRAMEWORK_SHA` via `env.sh`; stamp it into every training/eval
manifest from here on. W&B run `mm-C3-H-S-S-base` (project
`cosmos3_action_surgical`, entity `holoscan-advanced-workflow-team`) remains
the independent provenance cross-check.

**Step 2 — byte-identical H=0 reproduction check.** Run it at commit
`a3d6f89`, NOT at branch head: the FPS fix in `662cc8f` deliberately changes
CMR samples' `conditioning_fps` 30 → 10, so the pipeline is only
byte-identical to the base run *before* that commit. Once reproduction and a
successful base-checkpoint load are confirmed, everything launches from
branch head so all experimental arms inherit the FPS correction uniformly.

**The CAMP sequence layout** — single source of truth is
`camp_data_contract.py`; import it, never copy the integers:

| Rows | Content | Conditioning |
| --- | --- | --- |
| 0–2 (3) | learned memory: 132D VQ code reshaped to 3 × 44 | always clean |
| 3–18 (16) | recent history at effective rate, native-D padded to 44D | always clean |
| 19–30 (12) | current action window — public 12 × 44 contract, unchanged | FD clean; policy/ID denoise |

There is deliberately **no task-progress channel** (Open-H has no single
trustworthy completion definition across tasks/failures/endpoints);
early/middle/late episode bins are for evaluation stratification only and
are never exposed to the model.

**Planned training arms (Phase 4)** — all three start from the same trained
44D `mm-C3-H-S-S-base` checkpoint with identical data, mode ratios
(equal FD/ID/policy), seed, LR schedule, steps, and cluster shape. Load ALL
action heads — the "skip action heads" path was only for converting a 64D
base to fresh 44D heads and must not be reused here.

- **Arm A** — H0 continued-training control
- **Arm B** — + 16-row real action history
- **Arm C** — + history + learned memory (full CAMP)

**Tests**:
`python -m pytest cookbooks/cosmos3/generator/action/finetune/tests/ -v` —
34 tests run in any Python env; 2 more (contract vs `groot_configs`
cross-check, effective-FPS source coverage) activate automatically inside
the workspace training env.

**Implementation status (2026-07-22)** — Phases 1–3 are CODE-COMPLETE on
this branch (129 local tests passing; run the suite with the command above):

- **Phase 1 (`619419f`)** — `num_history_actions` / `history_ablation`
  through the full data path (`history_utils.py`, `groot_configs.py`,
  `open_h_dataset.py`, SFT wrapper, experiment config). Launch arms via env:
  `COSMOS_OPENH_NUM_HISTORY_ACTIONS=16` (+ `COSMOS_OPENH_HISTORY_ABLATION=
  zero|permute` for eval arms). The pinned framework rail does the rest
  (`history_action` → concat → pad → clean-conditioning sequence plan).
  CMR clutch filter + caches key on the CURRENT window only, so H=0/H16
  sample sets are identical and existing caches stay valid. The CMR pose
  re-integration provably telescopes: current rows are byte-identical
  across arms (tested).
- **Phase 2 (`9d5dc14`)** — `camp_action_memory.py`: verbatim SutureBot
  CAMP-v2 port (DCT scaffold, losses, hardened VQ) +
  `MultiEmbodimentActionMemoryEncoder` (one shared encoder, embodiment
  embedding, masked canonical 20D state + prev 44D action input,
  code_dim=132). Strict causality and offline==online agreement tested.
- **Phase 3 (`dad8657`)** — `camp_injection.py` + `camp_transforms.py`:
  `CampActionTransformPipeline` injects the reshaped (3, 44) code AFTER
  the base normalize/pad, rebuilds the plan with 19 conditioning rows.

**Remaining, in order (all cluster-side):**

1. **Gate-0 checks** — byte-identical H=0 reproduction at `a3d6f89`; base
   checkpoint loads under the pinned framework.
2. **Phase 1 on real leaves** — one smoke pass per leaf class (dual-arm,
   single-arm, MIRA deltas, CMR clutch, no-gripper, episode starts) with
   `COSMOS_OPENH_NUM_HISTORY_ACTIONS=16`; then arms A/B can launch.
3. **Phase 2b** — per-leaf canonical-state extraction table (verify against
   the modality audit), memory pretraining run, code exporter keyed by
   `OpenHMixedLeRobotDataset.get_step_ids()` (full dataset paths — 36
   leaves share basenames) with the manifest fields from the plan.
4. **Phase 3b — REQUIRED BEFORE ARM C** on narrow embodiments: the pinned
   model zeroes padded action-INPUT channels per sample
   (`omni_mot_model.py` xt-noising site ~L1557 and sampling-init site
   ~L1804: `xt_action[i][:, raw_action_dim[i]:] = 0`). Native rows are
   unaffected (padded channels already zero) but dense 44-wide memory rows
   would be silently truncated for raw_action_dim < 44. Fix: guarded
   exemption skipping the first `num_memory_action_rows[i]` rows at those
   two sites (field already emitted by `CampActionTransformPipeline`;
   plumb through `data_and_condition.py` like `raw_action_dim`). The loss
   (`flow_matching.py` sqerr slice) and velocity sites are already
   row-masked for conditioning rows — no change needed there.
   `inject_memory_rows` FAILS CLOSED on raw_action_dim < 44 until this
   lands, so a mis-wired arm C crashes rather than trains on a truncated
   code. Arms A/B are unaffected.

**Roadmap after Gate 0** (agreed plan, condensed):

1. **Phase 1 — history**: `num_history_actions=16` +
   `history_ablation=None|zero|permute` on `OpenHMixedLeRobotDataset`;
   history sampled at each embodiment's effective timestep interval with
   the same relative transform/normalization as the current window;
   episode-start clamping; stable `dataset_id`/`episode_id`/`base_index`.
   Test per leaf class: dual-arm poses, single-arm, MIRA deltas, CMR
   clutch/engagement, missing grippers, episode boundaries.
2. **Phase 2 — memory**: ONE shared encoder with a learned embodiment
   embedding (not nine); per-step input = canonical two-arm 20D state
   (+ state mask) + previous normalized 44D action + active-channel mask;
   production VQ recipe unchanged (data init, EMA, dead-code restarts,
   commitment loss); `recon_len=512` @ verified 10 Hz, 32 DCT coeffs,
   codebook 512, `code_dim=132`; collision-proof dataset IDs (never root
   basenames — 36 leaves share names) in every export manifest.
3. **Phase 3 — injection**: `CampActionTransformPipeline` wrapper prepends
   the reshaped 3 × 44 code AFTER the existing normalize/pad, then rebuilds
   the sequence plan with 19 conditioning positions. Never feed the 132D
   code through native-action transforms.
4. **Phase 4 — arms**: A/B/C above at matched iterations.
5. **Phase 5 — eval + serving**: real/zero/shuffled history;
   real/zero/**cross-episode-shuffled memory** (the strongest null detector
   — in-distribution but behaviorally wrong); long-horizon FDM across
   embodiments; policy MSE over active channels only; ID metrics; episode
   bins; occlusion/arm-leaves-frame cases. Then the online state manager:
   per-env LSTM hidden state, H16 executed-action buffer, episode-boundary
   resets, 132D code emitted online with no dependency on pre-exported
   tracks.

### Embodiment tag vs institution (important)

"JHU IMERSE" the **institution / dataset family** is NOT the same as
`jhu_imerse` the **embodiment registry tag**. This trips people up:

- The 7 JHU **IMERSE datasets** (`srth_porcine_chole`, `wound_closure`,
  `suturebot`, `nephfat`, `srt_needle_pickup_handover`, `cao_cautery_combined`,
  `srt_tissue_lift`) ARE in the mixture — they all use the standard dual-arm
  dVRK layout (`psm1/psm2_pose`(7) + grippers, action width 16), so they are
  tagged **`jhu_dvrk_mono`** (the shared dVRK embodiment), not `jhu_imerse`.
  `srth_porcine_chole` and `wound_closure` are the two heaviest non-CMR
  datasets (~0.87 mix each).
- The **`jhu_imerse` tag** was created specifically for **STAR-IL** (a single
  KUKA arm, no gripper, 9D — a genuinely different kinematic config). STAR-IL
  was dropped (no `modality.json`), so the `jhu_imerse` *tag* is currently
  **dormant** — but the IMERSE *datasets* are fully trained on.
- `virtual_incision_mira` is fully gone: both the tag and its only dataset
  (MIRA) were dropped.

If renaming for clarity later, consider `jhu_kuka_star_il` for the dormant tag.

### Modality-audit findings (per embodiment)

After dumping every dataset's `meta/modality.json` + `info.json` + a parquet
sample on EOS (`scripts/inspect_openh_modality.py` →
`doc/openh_modality_report.json`), the registry was corrected to ground truth.
Notable points a maintainer should know:

- **`modality.json` does NOT declare `rotation_type`** for these datasets — the
  pose representation (quaternion vs Euler) is inferred from the pose sub-slice
  width: **w7 = quaternion**, **w6 = Euler**. The registry's
  `input_rotation_format` / quat-order choices are therefore authored, not read.
- **video keys = modality SUBKEY name, not the folder.** e.g. JHU/Obuda/Turin
  expose subkey `endoscope_left` whose `original_key` folder is
  `observation.images.endoscope.left` → registry key is `video.endoscope_left`
  (underscore), NOT `video.endoscope.left`. Stanford's subkey is also
  `endoscope_left` even though its folder is `observation.images.camera_left`.
  UCSD/UCB/MIRACLE use subkey `camera_left` (folder `observation.images.left`).
- **CMR Versius**: `action` width 100; pose w7 @ [0:7]/[13:20], grippers
  [10:11]/[23:24], energy [8:9]/[21:22] + the CMR state-conditioning keys —
  matches the dedicated CMR transform path.
- **JHU IMERSE / Obuda**: identical 16-wide dual-arm dVRK layout →
  `jhu_dvrk_mono` / `dvrk_obuda` (quat xyzw). (ARCADE `cautery` shared this
  schema but was later dropped for incomplete video staging — see the dropped
  table; its schema was never the problem.)
- **Stanford**: pose is **w6 Euler**, gripper-first (`gripper[0:1]`,
  `pose[1:7]`), action width 14.
- **Turin**: **pose-only w7, no grippers**, action width 14.
- **UC Berkeley**: action comes from `action.cartesian_state` (width 16), with
  joint state pass-through.
- **TUD grasping_retraction**: `eef_pose` reference comes from
  `observation.state[26:33]`; gripper from `action[4:5]`. There is **no
  `state.gripper`** — the registry's earlier `state.gripper` key was removed
  (it would have failed at load).
- **JHU LCSR SMARTS**: `SurgSync-stitch-coldcut/{P1,P2,P3}` have a curated
  `modality.json` (pose subkeys + `action.psm{1,2}.gripper`, `endoscope_left`)
  and are included. `SurgSync-multitask/{P1..P4}` were re-audited 2026-06-24 and
  have **NO `modality.json`** — they ship the raw dVRK-Si ROS dump (per-DOF
  dotted columns like `action.psm1.pose.orientation.w` + an ECM camera arm), so
  they are **blocked** until a `modality.json` is authored for them.
- **JHU LCSR MIRACLE**: pose from `observation.state[18:25]/[32:39]` (quat
  xyzw), grippers from `action[6]/[13]`; camera `left` (`video.camera_left`).
  All three leaves — `prepare_to_pierce`, `needle_pick_up`, `needle_regrasp` —
  share this schema (verified 2026-06-24) and are included.

An automated cross-check (re-run any time) confirms every registry
video/state/action key matches a real `modality.json` subkey for all 8 non-CMR
used embodiments (after the SMARTS and ARCADE-cautery drops).

### Chronology of decisions

1. **Port target & dimension.** Implement the cosmos3-internal Open-H FD recipe
   into cosmos3-h-s-s at **44D** (not 54D), via cookbook + `framework_patch/`
   overlay (user-confirmed).
2. **First draft mixture** = cosmos3-internal specs ∪ sean-repo additions
   (Obuda, LSCR MIRACLE/SMARTS, TUD, STAR-IL, MIRA), all 44D.
3. **Table S1 / folder-tree audit.** Cross-referenced the paper's Table S1, the
   on-disk `doc/open-h-embodiment_dataset_folder_structure.txt`, and the
   colleague's exclusion list. Findings: PolyU is synthetic, UT Austin is
   colonoscopy, USTC is absent from the public tree.
4. **User decisions:** drop **USTC** (not public), **Moon** (delta-xyz only, no
   rotation — C-H-S-S inclusion alone doesn't justify it), and **Hamlyn**
   (public release has no endoscope camera, only wrist/color/depth). Keep
   Semaphor/PolyU/UT-Austin excluded.
5. **Re-root to the public tree.** All paths re-based on
   `/lustre/fsw/.../open-h-embodiment/Surgical` with real on-disk leaf names;
   `_rebase_specs` fixed to preserve the full nested relative path.
6. **Modality audit on EOS** (`inspect_openh_modality.py`). Corrected all
   `video_keys`, fixed TUD `state` keys, recomputed ratios from real
   `total_frames`.
7. **Dropped 3 leaves with no `modality.json`** (STAR-IL, ARCADE
   cholecystectomy, MIRA); SMARTS kept to stitch-coldcut P1-P3 (user-confirmed).
8. **Mixture after the first audit: 38 leaves, 10 embodiments, CMR exactly 50%.**
9. **2026-06-24 delta-download re-audit** (`open-h-embodiment` v2 tree +
   `inspect_openh_modality.py --included-only`, report
   `doc/openh_modality_report_v2.json`). The delta added several orgs/leaves;
   re-checked each against the endoscope-left + `modality.json` gates:
   - **Added** `jhu/lcsr/miracle/needle_pick_up` (291 fr) and
     `jhu/lcsr/miracle/needle_regrasp` (300 fr) — both have a `modality.json`
     identical to the already-included `prepare_to_pierce`. → **+2 leaves**.
   - **Blocked** `jhu/lcsr/smarts/SurgSync-multitask/{P1..P4}` — no
     `modality.json` (raw ROS dump). Moved from "not audited" to the
     dropped-after-audit list.
   - **Skipped** `utenn/*` (newly staged) — no endoscope-left camera
     (user-confirmed, same gate as Hamlyn).
   - **Skipped** `polyu/*`, `sanoscience/*`, `stanford/.../simulation/*` —
     synthetic (user-confirmed).
   - **No-op** `stanford/.../real_robot_dvrk/*` — already in the mixture since
     step 5; the delta merely staged the data those specs referenced.
   - **Unchanged** `arcade/cholecystectomy` — still no `modality.json`.
10. **Mixture after the v2 re-audit: 40 leaves, 10 embodiments, CMR ~50%.** (The 2
    new MIRACLE leaves add 591 fr — 0.026% of the non-CMR pool — so CMR's share
    and all existing `mix_ratio`s are effectively unchanged; both new ratios sit
    at the 0.001 floor. `compute_openh_action_stats.py` re-derives ratios from
    real `total_frames` at pre-flight regardless.)
11. **2026-06-25/26 stats-run drops (data reality vs the offline registry):**
    - **SMARTS `stitch-coldcut/{P1,P2,P3}` dropped** — at stats time they failed
      every window (`AssertionError: No observation.state found`). The on-disk
      parquet stores per-DOF scalar columns
      (`observation.cartesian_state.psm1.pose.position.x` …), not a single
      sliceable `observation.state`/`action` vector, so the loader can't build
      the pose without a re-conversion. ≈103k fr. → **−3 leaves**.
    - **ARCADE `cautery` dropped** — half-staged: 22 parquet episodes but only 12
      `endoscope.left` .mp4 (episodes 12–21 lack video), so ~half its windows hit
      FileNotFoundError. Smallest leaf (5,288 fr, 0.05% of pool). → **−1 leaf**.
12. **Final mixture: 36 leaves, 9 embodiment tags, CMR ~50%.** Stats
    (`stats_cosmos[-44D]-c3hss-v1.json`), CMR `modality-44D.json`, and CMR
    `cmr_filter_cache_train_*-44D.json` are all present for these 36 leaves
    (verified via `scripts/preflight_check_cmr_artifacts.py --require-stats`).

### Known gaps and open items

- **`state_keys` / `action_key_configs` are still partly authored**, not fully
  read from `modality.json` (quat order, Euler convention, gripper open/close
  sign). The video/state/action key NAMES are verified; the per-key
  representation details should be confirmed by the parquet sanity checks
  (quat-norm ≈ 1, plausible Euler magnitudes, gripper ranges) before a long run.
- **Normalization stats not yet computed** — `compute_openh_action_stats.py`
  must be run on EOS to produce `meta/stats_cosmos.json` (and CMR's
  `stats_cosmos-44D.json`); training fails without them.
- **CMR filter caches not yet computed** (`compute_cmr_filtered_episodes_cache.py`).
- **Mix ratios are frame-proportional estimates**; fine, but re-derive if the
  dataset set changes.
- **Dormant tags** (`jhu_imerse`, `virtual_incision_mira`) remain in the
  registry/enum for re-add if STAR-IL / MIRA publish a `modality.json`.
- **SMARTS multitask P1-P4** (≈64k fr) are blocked on a missing `modality.json`
  (raw dVRK-Si ROS dump). To add: author a `meta/modality.json` mapping the
  per-DOF dotted columns into `psm1_pose`/`psm2_pose` (7D xyz+quat) + grippers,
  then they slot into `JHU_LSCR_SMARTS` as-is.
- **License headers** on ported `gr00t_dreams` files are Apache-2.0 (provenance)
  vs `OpenMDW-1.1` for new files — reconcile before publishing.
- The cookbook is currently **untracked / not on EOS**; sync via git or
  `setup_workspace.sh` staging before running there.

### How to re-derive / re-validate

```bash
# 1. Re-dump modality/info/parquet from EOS (the ground truth):
python scripts/inspect_openh_modality.py \
    --root /lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment/Surgical \
    --included-only --sample-rows 256 --output doc/openh_modality_report.json

# 2. Fail-closed schema re-check (every registry key must exist in modality.json):
python scripts/audit_openh_action_schemas.py --root "$OPENH_SURGICAL_ROOT"
```

`OPEN_H_DATASET_SPECS` and `EMBODIMENT_REGISTRY` in
`framework_patch/cosmos_framework/data/vfm/action/gr00t_dreams/groot_configs.py`
are the single source of truth; everything else (embodiment tag set, stats-file
checks, the experiment dataloader) derives from them.
