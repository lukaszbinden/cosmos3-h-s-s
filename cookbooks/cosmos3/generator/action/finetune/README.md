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
contract: CMR Versius, JHU (IMERSE + LCSR ARCADE/MIRACLE/SMARTS + STAR-IL),
Obuda, Stanford, Turin, UC Berkeley, UCSD, TU Dresden (grasping_retraction),
and Virtual Incision MIRA. It is a superset of the released C-H-S-S checkpoint
mixture (paper Table S4), minus three datasets that are unavailable or
incompatible in the public release — **Hamlyn** (no endoscope camera),
**USTC/Tuodao** (absent from the public tree), and **Moon** (delta-xyz only,
no verified rotation). See [Dataset mixture](#dataset-mixture).

> Status: this is a training recipe + data stack authored OFFLINE against the
> folder tree. The public open-h-embodiment LeRobot datasets use different
> modality keys than the C-H-S-S mirror this registry was first written for:
> `video_keys` are set from the on-disk camera folders, but the action/state
> keys (and the two new embodiment schemas `jhu_imerse`,
> `virtual_incision_mira`) MUST be validated against each `meta/modality.json`
> on the cluster before a production run — see
> [Pre-flight](#pre-flight-must-run-on-the-cluster).

## Table of Contents

- [Why 44D (and not 54D)](#why-44d-and-not-54d)
- [The 44D action space](#the-44d-action-space)
- [Dataset mixture](#dataset-mixture)
  - [Included (41 dataset leaves, 12 embodiment tags)](#included-41-dataset-leaves-12-embodiment-tags)
  - [Excluded (and why)](#excluded-and-why)
- [Layout](#layout)
- [Setup](#setup)
- [Pre-flight (must run on the cluster)](#pre-flight-must-run-on-the-cluster)
- [Launch](#launch)
- [Licensing / provenance](#licensing--provenance)

## Why 44D (and not 54D)

Two prior efforts mapped Open-H surgical actions into a fixed vector:

- **cosmos3-internal** and the **publicly released Cosmos-H-Surgical-Simulator
  (C-H-S-S) checkpoint** use **44D** (CMR Versius ceiling: 30D actions + 14D
  state conditioning).
- **`sean-cosmos3_surgical_fd`** chose **54D** to carry richer CMR control /
  physical-arm context channels (20D dual-haptic pose/gripper + 10 CMR controls
  + 24 `observation.state` context).

This cookbook uses **44D**. Rationale:

1. **Community continuity.** The already-released C-H-S-S checkpoint is 44D, so
   the community is already building tooling, inference specs, and action
   viewers around 44D. A successor that silently changes the action dimension
   creates a migration burden and confusion. Changing it needs a *strong*
   justification; absent that, 44D is the safer choice on behalf of the
   community.
2. **Matches the design source.** The data stack ported here (the
   `gr00t_dreams` registry + `OPEN_H_DATASET_SPECS` + per-embodiment transforms)
   is cosmos3-internal's 44D design verbatim. Re-deriving it at 54D would mean
   re-authoring the CMR transform, re-computing all stats, and re-validating.
3. **The 54D gain is CMR-only and additive-later.** 54D's extra 10–24 channels
   are exclusively CMR control/context (clutch/energy/thumbstick + arm
   colors/instrument-types/links/electrosurgery/ICG). The 44D layout already
   includes the 14D CMR state-conditioning tail (engagement, arm link,
   instrument type, color, electrosurgery mode). If the additional CMR context
   proves valuable, it can be introduced later as a **separate, clearly-versioned
   model variant** rather than as a breaking change to the default.

Full analysis is in the chat handoff that accompanied this cookbook; the short
version: **44D is the recommended default for the public successor.**

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
Versius is held at ~50%; the rest is frame-proportional (Table S1 counts),
normalized so the non-CMR pool sums to ~4.0. Recompute ratios at setup from
each dataset's `meta/info.json::total_frames`.

### Included (41 dataset leaves, 12 embodiment tags)

| Group | Leaves | Native dim | On-disk camera (left/mono) |
| --- | --- | --- | --- |
| CMR Versius | cholecystectomy, hysterectomy, inguinal_hernia, prostatectomy | 44 | `endoscope` |
| JHU IMERSE (dVRK-Si) | srth_porcine_chole, wound_closure, suturebot, nephfat, srt_needle_pickup_handover, cao_cautery_combined, srt_tissue_lift | 20 | `endoscope.left` |
| JHU LCSR ARCADE | arcade/cholecystectomy, arcade/cautery | 20 | `endoscope.left` |
| JHU LCSR MIRACLE | miracle/prepare_to_pierce | 20 | `left` |
| JHU LCSR SMARTS | smarts/SurgSync-stitch-coldcut/{P1,P2,P3} | 20 | `left` |
| JHU IMERSE STAR-IL | star_il/star_il | 9 (single KUKA, no gripper) | `endoscope.left` |
| Obuda dVRK | all 11 task leaves (frs_dome, pork, pegtransfer×2, rollercoaster, needlethreading×2, seaspike×3, skinphantom) | 20 | `endoscope.left` |
| Stanford real dVRK | needle_transfer, tissue_retraction, peg_transfer | 20 | `camera_left` |
| Turin MITIC | ex_vivo, plastic_pad, plastic_pad_3dmed, plastic_tube | 18 (no grippers) | `endoscope.left` |
| UCSD | surgical_learning_dataset, surgical_learning_dataset2 | 20 | `left` |
| UC Berkeley | debridement_lerobot | 20 | `left` |
| TUD TUNDRA | 260131_tundra_dataset/grasping_retraction | 10 | `laparoscope_left` |
| Virtual Incision MIRA | 150_episodes_mira_needle_lift | ~10 (delta-command) | `endoscope` |

All native widths are zero-padded to the unified 44D. `jhu_imerse` (STAR-IL)
and `virtual_incision_mira` are NEW embodiment entries (transforms authored
here); the rest reuse the ported registry.

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

This list is the **superset of the released C-H-S-S mixture (paper Table S4)
plus available `open-h-embodiment` surgical datasets** that fit the contract,
minus the three removals (Hamlyn, USTC, Moon) that are either unavailable or
incompatible with the public release. See the analysis chat for the full Table
S1 cross-reference.

## Layout

```text
finetune/
  README.md                      # this file
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
    audit_openh_action_schemas.py       # fail-closed schema audit (run before training)
    compute_openh_action_stats.py       # per-embodiment stats_cosmos.json
    compute_cmr_filtered_episodes_cache.py  # CMR clutch-aware filter caches
```

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
2. **Audit the schemas** (fail-closed) — especially the new `jhu_imerse` and
   `virtual_incision_mira` entries and the re-rooted (B) leaves:
   ```bash
   source "$WORKSPACE/packages/cosmos3/.venv/bin/activate"
   python scripts/audit_openh_action_schemas.py --root "$OPENH_SURGICAL_ROOT"
   ```
   If it reports a missing modality key, FIX the `EMBODIMENT_REGISTRY` entry in
   `groot_configs.py` (the new-embodiment keys are documented assumptions).
3. **Compute normalization stats** (writes `meta/stats_cosmos.json`, and CMR's
   `meta/stats_cosmos-44D.json`) and then update the (B) `mix_ratio`s from each
   dataset's `meta/info.json::total_frames`:
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
