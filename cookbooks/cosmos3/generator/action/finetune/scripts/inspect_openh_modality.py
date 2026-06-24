#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Inspect the public Open-H-Embodiment surgical datasets on EOS.

Run this ON THE CLUSTER (login node is fine) to dump, for each dataset leaf:

  * ``meta/modality.json``  -> video/state/action subkeys with index ranges,
    rotation_type, absolute flag, and original LeRobot column.
  * ``meta/info.json``      -> per-feature shapes/dtypes, total_frames,
    chunks_size, fps, robot_type.
  * a small parquet sample  -> per-action-key width, quaternion norm stats,
    Euler magnitude stats, gripper ranges (sanity checks).

The output is a single JSON report (stdout and/or ``--output``) that the
registry author consumes to correct
``gr00t_dreams/groot_configs.py::EMBODIMENT_REGISTRY`` (state_keys /
action_keys / action_key_configs / video_keys) for the PUBLIC schema, and to
recompute mix ratios from ``total_frames``.

No video/parquet data leaves the cluster — only this compact JSON summary.

Dependencies: stdlib + pyarrow (already in the framework venv). pandas optional.

Usage (from anywhere on EOS)::

    python inspect_openh_modality.py \
        --root /lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment/Surgical \
        --output openh_modality_report.json \
        --sample-rows 256

    # restrict to the leaves this cookbook trains on:
    python inspect_openh_modality.py --root <ROOT> --included-only \
        --output openh_modality_report.json

Then paste ``openh_modality_report.json`` (or its printed contents) back.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Leaf paths (relative to ROOT) this cookbook's OPEN_H_DATASET_SPECS trains on.
# Keep in sync with groot_configs.OPEN_H_DATASET_SPECS.
INCLUDED_LEAVES = [
    "cmr_surgical/cholecystectomy",
    "cmr_surgical/hysterectomy",
    "cmr_surgical/inguinal_hernia",
    "cmr_surgical/prostatectomy",
    "jhu/imerse/srth_porcine_chole",
    "jhu/imerse/wound_closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled",
    "jhu/imerse/suturebot",
    "jhu/imerse/nephfat/nephfat",
    "jhu/imerse/srt_needle_pickup_handover",
    "jhu/imerse/cao_cautery_combined",
    "jhu/imerse/srt_tissue_lift",
    "jhu/lcsr/arcade/cholecystectomy",
    "jhu/lcsr/arcade/cautery",
    "jhu/lcsr/miracle/prepare_to_pierce",
    # Newly staged JHU LCSR leaves (2026-06-24 delta download) — candidates to
    # add to OPEN_H_DATASET_SPECS once their modality.json is verified here.
    "jhu/lcsr/miracle/needle_pick_up",
    "jhu/lcsr/miracle/needle_regrasp",
    "jhu/lcsr/smarts/SurgSync-multitask/P1",
    "jhu/lcsr/smarts/SurgSync-multitask/P2",
    "jhu/lcsr/smarts/SurgSync-multitask/P3",
    "jhu/lcsr/smarts/SurgSync-multitask/P4",
    "jhu/lcsr/smarts/SurgSync-stitch-coldcut/P1",
    "jhu/lcsr/smarts/SurgSync-stitch-coldcut/P2",
    "jhu/lcsr/smarts/SurgSync-stitch-coldcut/P3",
    "jhu/imerse/star_il/star_il",
    "obuda/frs_dome_1",
    "obuda/pork_1",
    "obuda/pegtransfer_1",
    "obuda/rollercoaster_1",
    "obuda/needlethreading_1",
    "obuda/needlethreading_2",
    "obuda/seaspike_3",
    "obuda/seaspike_1",
    "obuda/pegtransfer_2",
    "obuda/seaspike_2",
    "obuda/skinphantom_1",
    "stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/needle_transfer",
    "stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/tissue_retraction",
    "stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/peg_transfer",
    "turin/mitic_lerobot_ex_vivo",
    "turin/mitic_lerobot_plastic_pad_3dmed",
    "turin/mitic_lerobot_plastic_tube",
    "turin/mitic_lerobot_plastic_pad",
    "ucsd/surgical_learning_dataset",
    "ucsd/surgical_learning_dataset2",
    "ucberkeley/debridement_lerobot",
    "tud/260131_tundra_dataset/grasping_retraction",
    "virtual_incision/150_episodes_mira_needle_lift",
]


def _read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def _find_leaves(root: Path) -> list[Path]:
    """All dataset leaves under ``root`` (dirs with a meta/modality.json)."""
    return sorted(p.parent.parent for p in root.rglob("meta/modality.json"))


def _summarize_modality(modality: dict) -> dict:
    """Flatten modality.json into {video:[...], state:[...], action:[...]}."""
    out: dict = {}
    for mod in ("video", "state", "action", "annotation"):
        entries = modality.get(mod)
        if not isinstance(entries, dict):
            continue
        out[mod] = {}
        for subkey, meta in entries.items():
            if mod == "video":
                out[mod][subkey] = {"original_key": meta.get("original_key")}
            else:
                out[mod][subkey] = {
                    "start": meta.get("start"),
                    "end": meta.get("end"),
                    "width": (meta.get("end") - meta.get("start"))
                    if isinstance(meta.get("end"), int) and isinstance(meta.get("start"), int)
                    else None,
                    "rotation_type": meta.get("rotation_type"),
                    "absolute": meta.get("absolute", True),
                    "original_key": meta.get("original_key"),
                    "dtype": meta.get("dtype"),
                    "range": meta.get("range"),
                }
    return out


def _summarize_info(info: dict) -> dict:
    feats = info.get("features", {})
    feat_summary = {}
    for name, meta in feats.items():
        feat_summary[name] = {
            "dtype": meta.get("dtype"),
            "shape": meta.get("shape"),
            "names": meta.get("names") if isinstance(meta.get("names"), list) and len(meta.get("names", [])) <= 64 else None,
        }
    return {
        "robot_type": info.get("robot_type"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "chunks_size": info.get("chunks_size"),
        "data_path": info.get("data_path"),
        "features": feat_summary,
    }


def _first_parquet(dataset_path: Path, info: dict) -> Path | None:
    pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    # Try the documented pattern for the first episode, else glob.
    try:
        cand = dataset_path / pattern.format(episode_chunk=0, episode_index=0)
        if cand.exists():
            return cand
    except Exception:
        pass
    hits = sorted(dataset_path.glob("data/**/*.parquet"))
    return hits[0] if hits else None


def _parquet_checks(dataset_path: Path, info: dict, modality: dict, sample_rows: int) -> dict:
    """Sample one parquet file; report widths + quaternion/euler/gripper stats."""
    try:
        import pyarrow.parquet as pq
    except Exception as e:  # noqa: BLE001
        return {"error": f"pyarrow unavailable: {e!r}"}

    pf = _first_parquet(dataset_path, info)
    if pf is None:
        return {"error": "no parquet file found"}

    try:
        table = pq.read_table(pf)
    except Exception as e:  # noqa: BLE001
        return {"error": f"failed to read {pf.name}: {e!r}"}

    n = min(sample_rows, table.num_rows)
    cols = {name: table.column(name).slice(0, n).to_pylist() for name in table.column_names}

    def _col_width(colname):
        vals = cols.get(colname)
        if not vals:
            return None
        first = vals[0]
        return len(first) if isinstance(first, (list, tuple)) else 1

    checks: dict = {
        "parquet_file": pf.name,
        "sampled_rows": n,
        "column_widths": {name: _col_width(name) for name in table.column_names},
    }

    # Action / state vector widths per the LeRobot raw columns referenced by modality.
    def _raw_keys(mod):
        ks = set()
        for meta in (modality.get(mod) or {}).values():
            ok = meta.get("original_key")
            if ok:
                ks.add(ok)
        return ks

    checks["action_raw_keys"] = sorted(_raw_keys("action"))
    checks["state_raw_keys"] = sorted(_raw_keys("state"))

    # Quaternion-norm + euler-magnitude sanity per action subkey that declares a
    # rotation_type, sliced out of its original column by [start:end].
    rot_report = {}
    for subkey, meta in (modality.get("action") or {}).items():
        rtype = meta.get("rotation_type")
        if not rtype:
            continue
        ok = meta.get("original_key")
        s, e = meta.get("start"), meta.get("end")
        vals = cols.get(ok)
        if not vals or not isinstance(s, int) or not isinstance(e, int):
            continue
        # Pull the rotation sub-slice for each sampled row.
        try:
            sub = [row[s:e] for row in vals if isinstance(row, (list, tuple)) and len(row) >= e]
        except Exception:
            continue
        if not sub:
            continue
        info_r = {"rotation_type": rtype, "slice": [s, e], "slice_width": e - s}
        if "quaternion" in rtype:
            norms = [math.sqrt(sum(c * c for c in v[-4:])) for v in sub if len(v) >= 4]
            if norms:
                info_r["quat_norm_min"] = round(min(norms), 4)
                info_r["quat_norm_max"] = round(max(norms), 4)
                info_r["quat_norm_mean"] = round(sum(norms) / len(norms), 4)
        elif "euler" in rtype:
            mags = [max(abs(c) for c in v) for v in sub if v]
            if mags:
                info_r["euler_absmax"] = round(max(mags), 4)
        rot_report[subkey] = info_r
    checks["rotation_checks"] = rot_report

    # Gripper range per action subkey whose name contains 'grip'.
    grip_report = {}
    for subkey, meta in (modality.get("action") or {}).items():
        if "grip" not in subkey.lower():
            continue
        ok = meta.get("original_key")
        s, e = meta.get("start"), meta.get("end")
        vals = cols.get(ok)
        if not vals or not isinstance(s, int):
            continue
        flat = []
        for row in vals:
            if isinstance(row, (list, tuple)) and len(row) > s:
                flat.extend(row[s:e])
            elif not isinstance(row, (list, tuple)):
                flat.append(row)
        flat = [x for x in flat if isinstance(x, (int, float))]
        if flat:
            grip_report[subkey] = {"min": round(min(flat), 4), "max": round(max(flat), 4)}
    checks["gripper_checks"] = grip_report
    return checks


def inspect_dataset(dataset_path: Path, sample_rows: int) -> dict:
    rec: dict = {"path": str(dataset_path)}
    mod_path = dataset_path / "meta" / "modality.json"
    info_path = dataset_path / "meta" / "info.json"
    try:
        modality = _read_json(mod_path)
        rec["modality"] = _summarize_modality(modality)
    except Exception as e:  # noqa: BLE001
        rec["modality_error"] = repr(e)
        modality = {}
    info = {}
    try:
        info = _read_json(info_path)
        rec["info"] = _summarize_info(info)
    except Exception as e:  # noqa: BLE001
        rec["info_error"] = repr(e)
    if sample_rows > 0:
        rec["parquet"] = _parquet_checks(dataset_path, info, modality, sample_rows)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        default="/lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment/Surgical",
        help="Open-H-Embodiment Surgical root on EOS.",
    )
    parser.add_argument("--output", default=None, help="Write the JSON report here (else stdout only).")
    parser.add_argument("--sample-rows", type=int, default=256, help="Parquet rows to sample (0 = skip parquet).")
    parser.add_argument(
        "--included-only",
        action="store_true",
        help="Only inspect the leaves this cookbook trains on (INCLUDED_LEAVES); else all leaves with a modality.json.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        sys.exit(2)

    if args.included_only:
        leaves = [root / rel for rel in INCLUDED_LEAVES]
    else:
        leaves = _find_leaves(root)

    report = {"root": str(root), "sample_rows": args.sample_rows, "datasets": {}}
    for leaf in leaves:
        rel = str(leaf.relative_to(root)) if str(leaf).startswith(str(root)) else str(leaf)
        if not leaf.exists():
            report["datasets"][rel] = {"error": "path not found"}
            print(f"[MISS] {rel}", file=sys.stderr)
            continue
        report["datasets"][rel] = inspect_dataset(leaf, args.sample_rows)
        print(f"[ok]   {rel}", file=sys.stderr)

    text = json.dumps(report, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(text)
        print(f"\nWrote report: {args.output}  ({len(report['datasets'])} datasets)", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
