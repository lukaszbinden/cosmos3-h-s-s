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
contract: CMR Versius, JHU (IMERSE + LCSR ARCADE cautery / MIRACLE / SMARTS),
Obuda, Stanford, Turin, UC Berkeley, UCSD, and TU Dresden (grasping_retraction)
— **38 dataset leaves across 10 embodiment tags**. All video/state/action keys
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
  - [Included (38 dataset leaves, 10 embodiment tags)](#included-38-dataset-leaves-10-embodiment-tags)
  - [Dropped after the modality audit](#dropped-after-the-modality-audit-no-usable-metamodalityjson)
  - [Excluded (and why)](#excluded-and-why)
- [Layout](#layout)
- [Setup](#setup)
- [Pre-flight (must run on the cluster)](#pre-flight-must-run-on-the-cluster)
- [Launch](#launch)
- [Licensing / provenance](#licensing--provenance)
- [Decision log and handoff notes](#decision-log-and-handoff-notes)
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

### Included (38 dataset leaves, 10 embodiment tags)

| Group | Leaves | Native dim | modality video key |
| --- | --- | --- | --- |
| CMR Versius | cholecystectomy, hysterectomy, inguinal_hernia, prostatectomy | 44 | `video.endoscope` |
| JHU IMERSE (dVRK-Si) | wound_closure, srth_porcine_chole, suturebot, nephfat, srt_needle_pickup_handover, cao_cautery_combined, srt_tissue_lift | 20 | `video.endoscope_left` |
| JHU LCSR ARCADE | arcade/cautery | 20 | `video.endoscope_left` |
| JHU LCSR MIRACLE | miracle/prepare_to_pierce | 20 | `video.camera_left` |
| JHU LCSR SMARTS | smarts/SurgSync-stitch-coldcut/{P1,P2,P3} | 20 | `video.endoscope_left` |
| Obuda dVRK | all 11 task leaves (frs_dome, pork, pegtransfer×2, rollercoaster, needlethreading×2, seaspike×3, skinphantom) | 20 | `video.endoscope_left` |
| Stanford real dVRK | needle_transfer, tissue_retraction, peg_transfer | 20 (Euler w6) | `video.endoscope_left` |
| Turin MITIC | ex_vivo, plastic_pad, plastic_pad_3dmed, plastic_tube | 18 (no grippers) | `video.endoscope_left` |
| UCSD | surgical_learning_dataset, surgical_learning_dataset2 | 20 | `video.camera_left` |
| UC Berkeley | debridement_lerobot | 20 | `video.camera_left` |
| TUD TUNDRA | 260131_tundra_dataset/grasping_retraction | 10 | `video.laparoscope_left` |

All native widths are zero-padded to the unified 44D.

### Dropped after the modality audit (no usable `meta/modality.json`)

These were in the earlier draft but have **no `modality.json`** in the public
release, so the gr00t LeRobot loader cannot consume them. They are removed from
the mixture; their registry entries (`jhu_imerse`, `virtual_incision_mira`) and
enum tags are kept **dormant** for easy re-add if the metadata is published:

| Dataset / leaf | Status |
| --- | --- |
| `jhu/imerse/star_il/star_il` (STAR-IL) | no modality.json |
| `jhu/lcsr/arcade/cholecystectomy` | no modality.json (ARCADE `cautery` is kept) |
| `virtual_incision/150_episodes_mira_needle_lift` (MIRA) | no modality.json (bespoke column layout) |

### Excluded (and why)

| Dataset / leaf | Reason |
| --- | --- |
| Hamlyn/Imperial (whole group) | public release has NO endoscope camera (only `color`/`depth`/`wrist_{left,right}`) — incompatible with the endoscope-conditioned FD setup |
| USTC/Tuodao (whole group) | NOT present in the public open-h-embodiment tree (was only in C-H-S-S v1's internal mirror) |
| Moon Surgical | delta-xyz only, no verified rotation action |
| UTenn (all leaves) | video / segmentation / action-label only — no paired Cartesian action |
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
| JHU SMARTS SurgSync-multitask P1-P4 | not audited yet (only stitch-coldcut P1-P3 included) |

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
    _eos_torchrun_inner.sh              # in-container torchrun wrapper
    slurm_smoke.sbatch                  # 1-node, 10-iter end-to-end smoke
    slurm_train.sbatch                  # 8-node resumable main train
    resubmit_until_done.sh              # resubmit until TARGET_ITER
    inspect_openh_modality.py           # dump modality.json/info.json/parquet checks (EOS)
    audit_openh_action_schemas.py       # fail-closed schema audit (run before training)
    compute_openh_action_stats.py       # per-embodiment stats_cosmos.json
    compute_cmr_filtered_episodes_cache.py  # CMR clutch-aware filter caches
```

The registry's video/state/action keys were reconciled against the live EOS
`meta/modality.json` files using `inspect_openh_modality.py`; the resulting
report is saved at `doc/openh_modality_report.json`.

## Setup

```bash
export WORKSPACE=$HOME/cosmos3_openh_surgical_fd
export OPENH_SURGICAL_ROOT=/lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment/Surgical
export BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir>
bash scripts/setup_workspace.sh
```

`setup_workspace.sh` clones Cosmos Framework into `$WORKSPACE/packages/cosmos3`,
installs the CUDA 13 training environment (`cu130-train`; use
`COSMOS3_UV_GROUP=cu128-train` + a PyTorch container for older drivers),
installs the extra deps the ported data stack needs (`albumentations`,
`imageio`, `dm-tree`), rsyncs `framework_patch/` over the checkout, registers
the `action_fdm_open_h_sft_nano` experiment in
`cosmos_framework/configs/base/config.py`, stages the TOML, and downloads the
Wan2.2 VAE.

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
3. **Compute normalization stats** (writes `meta/stats_cosmos.json`, and CMR's
   `meta/stats_cosmos-44D.json`); ratios are already frame-proportional from the
   modality report but can be refreshed from each `meta/info.json::total_frames`:
   ```bash
   python scripts/compute_openh_action_stats.py --root "$OPENH_SURGICAL_ROOT"
   ```
4. **Build the CMR clutch-aware filter caches**:
   ```bash
   python scripts/compute_cmr_filtered_episodes_cache.py
   ```
5. **Smoke test** (1 node, 10 iters; exercises every dataset + stats files):
   ```bash
   sbatch scripts/slurm_smoke.sbatch
   ```

## Launch

```bash
# Single resumable 8-node job (data_parallel_shard_degree auto-set to WORLD_SIZE):
sbatch scripts/slurm_train.sbatch

# Or resubmit until max_iter (EOS 4h wall-time loop):
bash scripts/resubmit_until_done.sh
```

Training output defaults to
`$WORKSPACE/outputs/train/cosmos3_action_surgical/action_open_h/action_fdm_open_h_sft_nano/`.

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
- **JHU IMERSE / ARCADE cautery / Obuda**: identical 16-wide dual-arm dVRK
  layout → `jhu_dvrk_mono` / `dvrk_obuda` (quat xyzw).
- **Stanford**: pose is **w6 Euler**, gripper-first (`gripper[0:1]`,
  `pose[1:7]`), action width 14.
- **Turin**: **pose-only w7, no grippers**, action width 14.
- **UC Berkeley**: action comes from `action.cartesian_state` (width 16), with
  joint state pass-through.
- **TUD grasping_retraction**: `eef_pose` reference comes from
  `observation.state[26:33]`; gripper from `action[4:5]`. There is **no
  `state.gripper`** — the registry's earlier `state.gripper` key was removed
  (it would have failed at load).
- **JHU LCSR SMARTS**: scalar per-joint columns + `action.psm{1,2}.gripper`;
  pose subkeys defined in modality. Only `SurgSync-stitch-coldcut/{P1,P2,P3}`
  are audited and included; `SurgSync-multitask/{P1..P4}` exist but are not
  audited.
- **JHU LCSR MIRACLE**: pose from `observation.state`, grippers from
  `action[6]/[13]`; cameras `left`/`right`.

An automated cross-check (re-run any time) confirms every registry
video/state/action key matches a real `modality.json` subkey for all 9 non-CMR
used embodiments.

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
8. **Final mixture: 38 leaves, 10 embodiments, CMR exactly 50%.**

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
- **SMARTS multitask P1-P4** could be added after auditing.
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
