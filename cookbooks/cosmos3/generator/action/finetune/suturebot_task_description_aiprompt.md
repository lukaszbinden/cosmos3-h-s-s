# SutureBot `task_description` → `ai_caption` wiring (handoff for an LLM)

**Status:** implemented + smoke-tested (CPU) on 2026-07-09. Picked up by the finetune
job `healthcareeng_holoscan-cosmos3.mm-C3-H-S-S-base` on its next restart/resume
(the overlay is re-stamped at job start).

**2026-07-10 correction:** the per-leaf widening exposed a second annotation
representation: the Fausto wound-closure leaf maps `human.task_description` to
the direct string column `instruction.text`, whereas SutureBot and CMR map it to
numeric `task_index`. `LeRobotSingleDataset.get_language()` previously called
`.item()` unconditionally and silently forced every Fausto sample through the
mixed-dataset retry/substitution path. It now accepts homogeneous direct-text
columns while retaining the numeric `task_index -> tasks.jsonl` lookup. Both
maintained `dataset.py` copies were patched and regression-tested.

This note is written for another language model / engineer to read in cold. It is
self-contained: it states the problem, the exact change, why it is shaped the way
it is, how it was verified, and how to extend or revert it.

---

## 1. Problem

Open-H surgical action-SFT windows were training with **empty captions**. The dataset
adapter `OpenHMixedLeRobotDataset.__getitem__` built `ai_caption` only from the
annotation key `annotation.human.coarse_action`:

```python
ai_caption = ""
if "annotation.human.coarse_action" in outputs:
    ...
```

But the surgical leaves (e.g. `jhu/imerse/suturebot`) never emit that key, so every
window got `ai_caption = ""` (only the viewpoint/fps/res appendices survived).
Language-steerability was therefore dead for these leaves.

### Root cause (traced end-to-end)

- All IMERSE/LCSR surgical leaves load under `EmbodimentTag.JHU_DVRK_MONO`, whose
  config is built by `_build_generic_config_and_transforms()` in
  `.../gr00t_dreams/groot_configs.py`. That generic config requests **only**
  `video` / `state` / `action` modalities — **no `language` modality** — so
  `LeRobotSingleDataset.get_step_data()` never produces any `annotation.*` key, and
  the `if ... in outputs` check above is always false.
- The suturebot leaf's `meta/modality.json` declares the annotation as
  `human.task_description` (mapped to `original_key: task_index`), **not**
  `coarse_action`:
  ```json
  "annotation": { "human.task_description": { "original_key": "task_index" } }
  ```
- The row-level parquet has **no `task_description` column**. The per-episode task
  lives in the int column `task_index`, plus a convenience `instruction.text`
  string column. `task_index` resolves to text via `meta/tasks.jsonl`:
  ```
  {"task_index": 0, "task": "knot tying"}
  {"task_index": 1, "task": "needle pickup"}
  {"task_index": 2, "task": "needle throw"}
  ```
- The framework already knows how to resolve this: `LeRobotSingleDataset.get_language()`
  (in `gr00t_dreams/data/dataset.py`) takes an annotation key, follows
  `original_key = task_index`, and looks the ints up in `tasks.jsonl`. It just was
  never being **asked** for a language modality on these leaves.

So the fix is: **request the `human.task_description` annotation as a single-frame
`language` modality, then read it in the caption builder.** These are episode-level
(coarse) labels — one string per trajectory — not the fine-grained per-window
`coarse_action` the loader was originally designed around.

---

## 2. Why the fix must be per-leaf (not a shared registry key)

All `jhu_dvrk_mono` leaves share ONE `EMBODIMENT_REGISTRY["jhu_dvrk_mono"]` entry,
but their annotation schemes are **heterogeneous**. Scanning
`/lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment/Surgical/**/modality.json`:

| annotation block            | # leaves |
|-----------------------------|----------|
| `human.task_description`    | 13       |
| `task`                      | 27       |
| `instruction`               | 5        |
| (no annotation)             | 21       |

`LeRobotSingleDataset._check_integrity()` and `get_language()` both **hard-assert**
if a configured modality key is missing from a given leaf's `modality.json`. So
hard-coding `annotation.human.task_description` into the shared registry/config
would crash every sibling leaf that lacks it, at dataset-init time, taking down the
whole mixture.

**Therefore the wiring is best-effort and per-leaf: it is keyed off what each leaf's
own `modality.json` actually declares.** Only `human.task_description` is wired,
because it is the scheme known to resolve cleanly through `tasks.jsonl`. Leaves with
`task` / `instruction` / no annotation are left exactly as before (empty caption) —
no crash, no regression.

> User's steer (2026-07-09): the extra prompt is expected on `jhu/imerse/suturebot`
> specifically; unsure about others. The per-leaf detection handles that
> automatically — it happened to also light up ~13 IMERSE leaves, which is a safe
> bonus. If you want it scoped **strictly** to suturebot, gate the injection on the
> leaf path instead of on annotation presence (see §6).

---

## 3. The change

**One file, two identical copies (keep them byte-identical):**

- `packages/cosmos3/cosmos_framework/data/vfm/action/open_h_dataset.py`  (vendored install)
- `cookbooks/cosmos3/generator/action/finetune/framework_patch/cosmos_framework/data/vfm/action/open_h_dataset.py`  (git-managed overlay; `scripts/apply_overlay.sh` stamps it onto the install at job start)

**No changes to `groot_configs.py`, `_check_integrity`, `get_language`, the CMR path,
or any other embodiment.**

### 3a. Imports
Add `import json` and add `ModalityConfig` to the existing gr00t_dreams import:
```python
import json
import os
...
from cosmos_framework.data.vfm.action.gr00t_dreams.data.dataset import (
    LeRobotSingleDataset,
    ModalityConfig,
    WrappedLeRobotSingleDataset,
)
```

### 3b. Inject a `language` modality per-leaf (in `__init__`, right after the
`modality_filename` pop, before `transform = train_tf ...`):
```python
# Language / caption conditioning.
# The generic per-embodiment config requests only video/state/action, so no
# annotation.* key reaches __getitem__ and every window would get ai_caption="".
# Where a leaf's modality.json declares the episode-level human.task_description
# annotation (currently jhu/imerse/suturebot: "knot tying"/"needle pickup"/
# "needle throw"), request it as a single-frame language modality so
# LeRobotSingleDataset.get_language resolves it via original_key=task_index ->
# meta/tasks.jsonl. Best-effort + per-leaf: leaves without that key are left
# untouched (never added), so _check_integrity / get_language never assert.
language_key = self._leaf_task_description_key(path, modality_filename)
if language_key is not None:
    config["language"] = ModalityConfig(delta_indices=[0], modality_keys=[language_key])
    log.info(f"    language conditioning: {language_key} (episode-level task description)")
```

### 3c. The detection helper + caption-key list (added just before `_get_raw_sample`):
```python
# Annotation keys carrying a natural-language caption, in priority order.
# coarse_action is the predict2.5 per-window key; task_description is the
# episode-level Open-H surgical annotation (resolves through meta/tasks.jsonl).
_CAPTION_ANNOTATION_KEYS = (
    "annotation.human.coarse_action",
    "annotation.human.task_description",
)

@staticmethod
def _leaf_task_description_key(path: str, modality_filename: str | None) -> str | None:
    """Return "annotation.human.task_description" iff the leaf's modality.json
    declares that annotation subkey, else None. Only human.task_description is
    wired (it resolves cleanly via original_key=task_index -> meta/tasks.jsonl);
    other schemes (task / instruction / none) are intentionally left untouched.
    Best-effort: any I/O or JSON error yields None (no language modality)."""
    modality_path = Path(path) / (modality_filename or "meta/modality.json")
    try:
        with open(modality_path) as f:
            annotation = json.load(f).get("annotation") or {}
    except (OSError, ValueError):
        return None
    subkey = "human.task_description"
    return f"annotation.{subkey}" if subkey in annotation else None
```

### 3d. Broaden caption extraction (in `__getitem__`, replacing the old
single-key `if` block):
```python
ai_caption = ""
for ann_key in self._CAPTION_ANNOTATION_KEYS:
    if ann_key not in outputs:
        continue
    raw_text = outputs[ann_key]
    if isinstance(raw_text, list) and raw_text:
        raw_text = raw_text[0]
    if isinstance(raw_text, str) and raw_text.strip():
        ai_caption = raw_text.split(":")[-1].strip()
        break
```
`coarse_action` keeps priority; `task_description` is the fallback. `.split(":")[-1]`
is a no-op on colon-free strings like `"knot tying"`.

### 3e. Support both task-index and direct-text annotations

The two maintained copies of
`cosmos_framework/data/vfm/action/gr00t_dreams/data/dataset.py` now unwrap
numpy/pandas scalars only when they expose `.item()`. If all selected annotation
values are strings, they are returned directly; otherwise the existing numeric
task-index lookup is used. Mixed string/index values in one window fail visibly
with a `TypeError` rather than being interpreted ambiguously.

**Why it survives the transform pipeline:** the leaf's transforms (`VideoToTensor`,
`VideoResize`, `StateActionToTensor`, `ConcatTransform`, …) all have explicit
`apply_to` lists over video/state/action keys only, and `ConcatTransform` groups the
`annotation.*` key separately and never pops it. So `annotation.human.task_description`
passes through untouched into `outputs` as a `list[str]` (e.g. `["knot tying"]`).

---

## 4. Verification (already done)

CPU-only smoke test, run in the training venv
(`packages/cosmos3/.venv`, torch 2.10+cu130), single-threaded per site policy,
against the real suturebot leaf. It exercised the whole caption path **minus video
decode** (see caveat below):

```
[1] _leaf_task_description_key -> 'annotation.human.task_description'
[2] leaf loaded: 1452 episodes, 516334 steps          # matches groot_configs "# 516,334 fr"
[3] sampled windows -> ai_caption:
    ep    0 -> 'knot tying'
    ep  293 -> 'knot tying'
    ep  615 -> 'knot tying'
    ep  886 -> 'needle throw'      # genuinely per-episode, not constant
    ep 1202 -> 'knot tying'
    ep 1451 -> 'knot tying'
[summary] distinct non-empty captions: {'knot tying': 5, 'needle throw': 1}
[summary] empty captions in sample: 0/6
RESULT: PASS
```

The 2026-07-10 direct-text regression test additionally covers:

```
Fausto instruction.text -> ["Wound Closure"]
SutureBot task_index=0   -> ["knot tying"]
2 passed
```

Method: built a **language-only** `LeRobotSingleDataset` on the real leaf (config =
`{"language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"])}`,
`transforms=None`), which routes `get_step_data` straight to `get_language` with no
video decode. Then ran the exact `_CAPTION_ANNOTATION_KEYS` extraction loop on the
result. Env needed: `COSMOS_OPENH_STATS_POSTFIX=c3hss-v1` (matches
`meta/stats_cosmos-c3hss-v1.json`) and thread caps
(`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`)
to respect the shared login node's process limit + single-threaded site policy.

### Caveat
- **Video decode (`_format_video`) was NOT exercised** — `torchcodec`/`ffmpeg` isn't
  on the login node (the container installs ffmpeg via apt at job start). That path
  is unchanged by this work and already runs in production, so it's orthogonal to
  captions. To exercise the entire `__getitem__` incl. decode, use the container
  (`scripts/slurm_smoke.sbatch`) — but that needs a GPU + base checkpoint + VAE,
  heavier than warranted just to confirm captions.
- Captions are **coarse / episode-level** (one of 3 strings per trajectory), giving
  task-level steerability, not sub-action granularity.

---

## 5. How to re-run the smoke test

```bash
W=/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/git/cosmos3-h-s-s
V=$W/packages/cosmos3/.venv/bin/python
cd $W/packages/cosmos3
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export COSMOS_OPENH_STATS_POSTFIX=c3hss-v1 COSMOS_OPENH_DEBUG_INIT=0
"$V" - <<'PY'
from cosmos_framework.data.vfm.action.open_h_dataset import OpenHMixedLeRobotDataset
from cosmos_framework.data.vfm.action.gr00t_dreams.data.dataset import LeRobotSingleDataset, ModalityConfig
from cosmos_framework.data.vfm.action.gr00t_dreams.data.embodiment_tags import EmbodimentTag
LEAF="/lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment/Surgical/jhu/imerse/suturebot"
key=OpenHMixedLeRobotDataset._leaf_task_description_key(LEAF,"meta/modality.json"); assert key=="annotation.human.task_description"
ds=LeRobotSingleDataset(dataset_path=LEAF,modality_configs={"language":ModalityConfig(delta_indices=[0],modality_keys=[key])},embodiment_tag=EmbodimentTag.JHU_DVRK_MONO,transforms=None,modality_filename="meta/modality.json")
for i in [0,len(ds)//2,len(ds)-1]:
    o=ds[i]; t=o[key][0] if isinstance(o.get(key),list) and o[key] else ""
    print(i, "->", repr(t.split(":")[-1].strip()))
PY
```

---

## 6. Extending / reverting

- **Scope strictly to suturebot:** in `_leaf_task_description_key`, additionally gate
  on `path`, e.g. `if not str(path).rstrip("/").endswith("jhu/imerse/suturebot"): return None`.
- **Support the `task` / `instruction` schemes** (the other 27 + 5 leaves): first
  verify each scheme's `original_key` resolves through `tasks.jsonl` (or is a raw
  string column) without tripping `get_language`'s asserts; only then add its subkey
  to the detection helper. Do NOT blindly widen — mismatched keys crash sibling
  leaves at init.
- **Revert:** restore the original single-key block in `__getitem__`, drop the
  helper / `_CAPTION_ANNOTATION_KEYS` / injection / imports. No other file touched.

## 7. Invariants for whoever edits next
- Keep the two `open_h_dataset.py` copies byte-identical (`diff` them; the overlay
  copy is authoritative and stamped onto the install at job start).
- Never put a modality key in a **shared** config/registry unless ALL leaves sharing
  that embodiment declare it — `_check_integrity` / `get_language` assert on absence.
- `delta_indices=[0]` = one label at the window's base frame = episode-level. Don't
  widen it expecting per-frame captions; these leaves only carry one task per episode.
