# Note: 44D vs 54D Surgical Action Space — Technical Analysis

Standalone analysis of the two candidate unified action encodings for the
Open-H surgical forward-dynamics finetune, **on technical merits only**. The
"the released Cosmos-H-Surgical-Simulator (C-H-S-S) checkpoint is already 44D,
so keep 44D for community continuity" argument is deliberately **excluded
here** — that is a real and arguably decisive consideration, but it is covered
in `README_COSMOS3_H_S_S.md` § "Why 44D (and not 54D)". This note asks only:
*as encodings, which is better, and why?*

- **44D** = the cosmos3-internal / this-cookbook scheme.
- **54D** = the `sean-cosmos3_surgical_fd` scheme.

## TL;DR

Default to **44D**. `44 vs 54` affects **only CMR Versius** (every other
embodiment is ≤20D and zero-pads either way). The two schemes carry the **same
20D pose/gripper and the same 10D hand-controller channels**; the entire delta
is the CMR **context tail: 14D (44D) vs 24D (54D)**. 54D's extra ~10 channels
are CMR-only, mostly categorical / quasi-static scene-state that is partly
redundant with the conditioning image. The cost (wider inert tail across the
50% non-CMR half, more compute, more hand-mapped channels to mis-label) is
certain and global; the benefit is speculative and embodiment-local. Adopt 54D
only if a CMR-only ablation shows a real generation-quality gain, and then as a
separately versioned variant.

## The framing fact

Both schemes are **identical for every non-CMR embodiment.** All the
dVRK / UR5e / KUKA datasets are ≤20D natively and are zero-padded up to the
ceiling regardless of whether the ceiling is 44 or 54. So this is **not** a
general "more action fidelity" question — it is purely:

1. *How much of the CMR Versius signal does the model see?* and
2. *What padding width does that ceiling impose on the other 9 embodiments
   (≈50% of training)?*

Both schemes read from the **same raw CMR columns** — `action` and
`observation.state` are each **100-wide** in the CMR parquet (verified, see
`doc/openh_modality_report.json`). Neither scheme invents data; each is a
*subset selection* of those 100+100 channels.

## Exactly what each scheme exposes for CMR

| Block | 44D (this cookbook) | 54D (sean) |
| --- | --- | --- |
| Dual-arm pose + gripper | 2×(xyz 3 + rot6d 6) + 2 grippers = **20D** | identical **20D** |
| Hand-controller inputs | energy(2) + thumbstick x/y(4) + thumbstickBtn(2) + clutchBtn(2) = **10D** (inside the 30D "action" block) | clutch + energy + thumbstick btn/x/y, ×2 sides = **10D** (as explicit "controls") |
| CMR scene/system context | **14D**: hapticengaged(2), arm→haptic link(2), instr type(4), HUD color(4), electrosurgery mode(2) | **24D**: haptic engageable(2), arm color ×5, instr type ×5, translation+rotation scaling(2), electrosurgery(2), haptic engaged(2), ICG mode+enabled(2), arm→haptic link(2), linked instr type L/R(2) |
| **CMR total** | **44** | **54** |

So the **10-channel difference** is entirely in the **context tail**. Relative
to 44D's 14D tail, 54D's 24D tail adds, concretely:

- a **5th arm** color + instrument-type slot (44D covers arms 0–3; 54D 0–4),
- **ICG** fluorescence mode + enabled (2),
- **explicit linked-instrument types** per side (2),
- **motion-scaling factors** exposed as model inputs (44D uses these inside the
  transform but does not feed them to the MLP) (2),
- haptic **engageable** flags (distinct from "engaged") (2),

minus a couple 44D chooses differently — net ≈ +10.

## Pros and cons

### 54D — pros
- **Better CMR scene disambiguation.** Clinical CMR frames often show 3–4
  physical arms while only 2 haptic pose streams exist; the extra 5th-arm slot,
  full instrument-type/color set, and explicit linked-instrument identity give
  the model more signal to explain on-screen content the 2 pose streams don't
  account for — directly targeting CMR's known "phantom/unexplained arm motion"
  ambiguity.
- **Predicts large appearance changes** driven by mode, not motion: **ICG**
  fluorescence switches and **electrosurgery/energy** (cautery smoke/char) are
  hard to predict from pose alone; as explicit inputs they could sharpen those
  transitions.
- **More faithful conditioning** by exposing motion-scaling to the model rather
  than hiding it in the transform.

### 54D — cons
- **All extra channels are CMR-only.** For the other 9 embodiments (≈50% of
  training, and the entire multi-embodiment research value of Open-H), dims
  44–53 are **always zero**. You widen the action MLP, per-sample tensors, and
  loss/noise/velocity masks for channels almost nothing uses.
- **Weak learning-signal density** on the tail dims (non-zero only on CMR) for
  marginal benefit; more action-head parameters/compute.
- **Larger hand-mapped schema surface = more mis-label risk.** The 24D context
  is assembled from institution-specific `observation.state` slices (color
  codes, instrument-type enums, ICG flags). Each is a silent-corruption
  opportunity — exactly the bug class the modality audit repeatedly surfaced.
- **Most extra channels are categorical / quasi-static** (arm color, instrument
  type, ICG on/off) — they change per-episode or rarely, so their marginal value
  for *frame-to-frame forward dynamics* is questionable, and they overlap with
  what the conditioning image already shows. They are closer to "scene id" than
  "action."

### 44D — pros
- **Tighter, denser action vector**: 30D of genuine per-timestep action +
  a compact 14D tail that still captures the highest-value context (engagement,
  arm link, instrument type, color, electrosurgery). Less inert padding across
  the non-CMR half.
- **Lower audit/mis-label risk** — 10 fewer hand-mapped categorical channels per
  CMR procedure.
- **Cleaner action/scene separation**: the MLP focuses on signals that vary at
  action timesteps; slowly-varying scene identity is carried by the image.

### 44D — cons
- **Strictly less CMR information**: drops the 5th arm, ICG state, explicit
  linked-instrument types, and exposed scaling. If CMR clinical realism is the
  generation bottleneck, 44D is the limiter.

## The crux

The decisive unknown is empirical and untested: **do ~10 mostly-categorical,
CMR-only context channels measurably improve forward-dynamics generation?**
Nothing we have answers it — neither GR00T-H's mixture nor any ablation here
compared 44 vs 54. Meanwhile the trade is **asymmetric**:

- 54D's upside is **speculative, CMR-local, and partly redundant** with the image.
- 54D's cost is **certain and global** (padding, compute, audit/mis-label risk).

You should not pay a certain global cost for a speculative local gain by default.

## Recommendation

1. **Default to 44D.** On technical merits the extra 10 channels don't justify a
   wider shared action space for a 10-embodiment mixture.
2. **Treat 54D as a falsifiable experiment, not the default.** If CMR generation
   quality becomes the measured bottleneck (e.g. the model can't disambiguate
   multi-arm scenes or ICG transitions), run a **CMR-only 44-vs-54 ablation** —
   same data and seed, score CMR FDS / PSNR — and adopt 54D only on a meaningful
   margin. Because 54D changes `MAX_ACTION_DIM`, the model `max_action_dim`, the
   CMR transform, and all `stats_cosmos*.json`, ship it as a **separate
   versioned variant**, not a silent change to the shared default.
3. **Cheap middle path first:** keep 44D but **swap which context channels fill
   the 14D tail** — e.g. include ICG state or the 5th-arm slot in place of a
   lower-value channel like HUD color. This tests "which context matters" for the
   highest-value signals without widening the vector or touching the non-CMR
   half.

## If 54D is ever adopted — required changes

- Re-author the CMR action transform to emit the 10 controls + 24 context
  channels (the non-CMR transforms are unaffected).
- Bump `MAX_ACTION_DIM` (groot_configs) and the model `max_action_dim` to 54.
- Recompute every `meta/stats_cosmos.json` and CMR `meta/stats_cosmos-44D.json`
  (rename to `-54D`).
- Re-export under a new checkpoint name so 44D consumers are not silently broken.

## Sources

- 44D layout: `framework_patch/cosmos_framework/data/vfm/action/gr00t_dreams/groot_configs.py`
  (CMR path; `MAX_ACTION_DIM = 44`) and
  `cosmos-h-surgical-simulator-public/scripts/README_ACTION_SPACE.md`.
- 54D layout: `sean-cosmos3_surgical_fd/cookbook/scripts/surgical_action_schemas.py`
  (`_cmr_clinical_layout`) and `OPENH_SURGICAL_FINETUNE_HANDOFF.md`.
- CMR raw column widths (100/100) and per-channel slices: `doc/openh_modality_report.json`.
