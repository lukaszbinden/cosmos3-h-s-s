#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Author ``meta/modality-44D.json`` for the public Open-H CMR Versius leaves.

WHY THIS EXISTS
---------------
The 44D CMR loader (``gr00t_dreams/data/dataset.py``) prefers a CMR-specific
modality file, ``meta/modality-44D.json`` (constant
``LE_ROBOT_CMR_MODALITY_FILENAME``), and only falls back to the public
``meta/modality.json`` if it's absent. The 44D action layout needs 30 action
channels (dual-arm pose+gripper, energies, thumbsticks, clutch buttons) plus a
14-channel ``cond_*`` state-conditioning tail — but the PUBLIC
``meta/modality.json`` shipped on the open-h-embodiment tree only declares 8
action subkeys (pose/gripper/energy + the two hapticengaged). So
``construct_modality_config_and_transforms(embodiment="cmr_versius")`` fails at
dataset construction with e.g.::

    ValueError: Unable to find key action.thumbstick_x_left in modality metadata

The required channels DO exist in the raw 100-wide ``action`` /
``observation.state`` parquet columns; they're just not mapped by the public
modality file. The authoritative 44D mapping lives only as an un-versioned
``modality-44D.json`` on the internal CMR mirror (``cmr-surgical-60hz-fixed``)
and as code in ``groot_configs.py`` / the sean repo's
``surgical_action_schemas._cmr_clinical_layout``. This script writes that file
for each public CMR leaf so the loader can build the full 44D representation.

VERIFIED: the JSON this emits is BYTE-IDENTICAL to a real Draco
``cmr-surgical-60hz-fixed/hysterectomy_360p/meta/modality-44D.json`` (18 state
keys, 30 action keys, same key order, same indenting). The raw column layout is
identical across all four CMR procedures, so one mapping serves every leaf.

INDEX MAP (verified against the public ``meta/info.json`` feature names AND
``sean-cosmos3_surgical_fd/.../surgical_action_schemas.py::_cmr_clinical_layout``;
both agree, and the slices match the public ``modality.json`` for the keys it
does declare):

Raw ``action`` column (per arm, left block 0..12 / right block 13..25):
    x,y,z, quat_x,quat_y,quat_z,quat_w (0:7),  clutchBtn (7:8), energyBtn (8:9),
    thumbstickBtn (9:10), pince=gripper (10:11), thumbstick_x (11:12),
    thumbstick_y (12:13)   [right block = same, +13]
Raw ``observation.state`` column (24 named):
    haptic_{l,r}_armengageable (0:2), arm_0..4_color (2:7),
    arm_0..4_instrtype (7:12), translationscaling (12:13), rotationscaling
    (13:14), electroSurgeryMode_{l,r} (14:16), hapticengaged_{l,r} (16:18),
    icgmode/icgenabled (18:20), armlinkedtohaptic_{l,r} (20:22),
    instrtype_{l,r} (22:24)

This emits EXACTLY the keys the ``cmr_versius`` branch of
``construct_modality_config_and_transforms`` reads (state + action), so the
loader's ``get_key_meta`` lookups all resolve.

Safety: by default refuses to overwrite an existing ``modality-44D.json`` unless
``--force`` (the internal mirror's hand-authored file should win if present). Use
``--dry-run`` to preview. Validates each leaf's ``info.json`` action/state widths
are >= the max index we slice (catches a wrong/older column layout).

Usage::

    # all CMR leaves in OPEN_H_DATASET_SPECS, under the root:
    python scripts/generate_cmr_modality_44d.py --root "$OPENH_SURGICAL_ROOT"

    # one leaf:
    python scripts/generate_cmr_modality_44d.py \
        --dataset-path "$OPENH_SURGICAL_ROOT/cmr_surgical/cholecystectomy"

    # preview only:
    python scripts/generate_cmr_modality_44d.py --root "$OPENH_SURGICAL_ROOT" --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Stats-compute mode so importing the specs never tries to load a stats file.
os.environ.setdefault("COSMOS_OPENH_STATS_COMPUTE_MODE", "1")

CMR_TAG = "cmr_versius"
MODALITY_44D_FILENAME = "meta/modality-44D.json"
MODALITY_PUBLIC_FILENAME = "meta/modality.json"
INFO_FILENAME = "meta/info.json"

ACT = "action"
OBS = "observation.state"


def _sa(start: int, end: int, original_key: str | None = None) -> dict:
    """One state/action modality entry.

    Byte-matches the authoritative Draco ``modality-44D.json`` style: minimal
    ``{start, end}`` (+ ``original_key`` only when it differs from the section
    default). We deliberately DO NOT emit ``absolute`` or ``rotation_type`` —
    the Draco file omits them and the loader's pydantic schema defaults
    ``absolute=True`` / ``rotation_type=None`` anyway (the CMR transform is told
    the pose is quaternion-xyzw via groot_configs.py, not via this file).

    ``original_key=None`` -> omit the field (caller passes it only when the entry
    reads from a non-default raw column).
    """
    d: dict = {"start": start, "end": end}
    if original_key is not None:
        d["original_key"] = original_key
    return d


# ---- The authoritative 44D CMR mapping (raw column slices) -------------------
# NOTE: rotation_type is left null/None for pose slices to match BOTH the public
# modality.json (rotation_type=null everywhere) and the internal 44D file; the
# CMR transform is told the rotation is quaternion-xyzw via groot_configs.py
# (CMRVersiusRelativeActionTransform(input_rotation_format="quat")), NOT via this
# file. Do not add rotation_type here or you risk double-handling the rotation.

# state.* keys. The cmr_versius config only READS 8 of these (left/right
# pose+gripper, hapticengaged_{l,r}, translation_scaling, rotation_scaling), but
# we emit the FULL 18-key section to byte-match the authoritative Draco
# ``modality-44D.json`` (cmr-surgical-60hz-fixed/*/meta/) used by the original
# C-H-S-S / cosmos3-internal runs. The extra keys (electroSurgeryMode,
# armlinkedtohaptic, instrtype, arm_*_color) are harmless supersets — present in
# the metadata but not concatenated into the model's state input. Order matches
# the Draco file. NOTE pose/gripper come from the raw ``action`` column (not
# ``observation.state``) — that's the CMR convention the Draco file uses.
_STATE: dict[str, dict] = {
    "left_pose": _sa(0, 7, ACT),
    "left_gripper": _sa(10, 11, ACT),
    "right_pose": _sa(13, 20, ACT),
    "right_gripper": _sa(23, 24, ACT),
    "electroSurgeryMode_left": _sa(14, 15, OBS),
    "armlinkedtohaptic_left": _sa(20, 21, OBS),
    "instrtype_left": _sa(22, 23, OBS),
    "electroSurgeryMode_right": _sa(15, 16, OBS),
    "armlinkedtohaptic_right": _sa(21, 22, OBS),
    "instrtype_right": _sa(23, 24, OBS),
    "translation_scaling": _sa(12, 13, OBS),
    "rotation_scaling": _sa(13, 14, OBS),
    "hapticengaged_left": _sa(16, 17, OBS),
    "hapticengaged_right": _sa(17, 18, OBS),
    "arm_0_color": _sa(2, 3, OBS),
    "arm_1_color": _sa(3, 4, OBS),
    "arm_2_color": _sa(4, 5, OBS),
    "arm_3_color": _sa(5, 6, OBS),
}

# action.* keys read by the cmr_versius config (30 keys: 16 control + 14 cond_*).
# Section default original_key is "action", so action-sourced entries omit it
# (pass None) and only the observation.state-sourced entries name it — matching
# the Draco file's style exactly.
_ACTION: dict[str, dict] = {
    # Dual-arm pose / gripper / energy (the 20D + 2 energy block).
    "left_pose": _sa(0, 7),
    "left_gripper": _sa(10, 11),
    "right_pose": _sa(13, 20),
    "right_gripper": _sa(23, 24),
    "left_energy": _sa(8, 9),
    "right_energy": _sa(21, 22),
    # Thumbsticks (continuous) + thumbstick/clutch buttons. Order (per-side
    # x,y,btn then the other side, then clutch) matches the Draco file.
    "thumbstick_x_left": _sa(11, 12),
    "thumbstick_y_left": _sa(12, 13),
    "thumbstickBtn_left": _sa(9, 10),
    "thumbstick_x_right": _sa(24, 25),
    "thumbstick_y_right": _sa(25, 26),
    "thumbstickBtn_right": _sa(22, 23),
    "clutchBtn_left": _sa(7, 8),
    "clutchBtn_right": _sa(20, 21),
    # Engagement (passthrough for clutch-aware transform; from observation.state).
    "hapticengaged_left": _sa(16, 17, OBS),
    "hapticengaged_right": _sa(17, 18, OBS),
    # ---- 14D state-conditioning tail (cond_*), all from observation.state ----
    "cond_hapticengaged_left": _sa(16, 17, OBS),
    "cond_hapticengaged_right": _sa(17, 18, OBS),
    "cond_armlinkedtohaptic_left": _sa(20, 21, OBS),
    "cond_armlinkedtohaptic_right": _sa(21, 22, OBS),
    "cond_arm_0_instrtype": _sa(7, 8, OBS),
    "cond_arm_1_instrtype": _sa(8, 9, OBS),
    "cond_arm_2_instrtype": _sa(9, 10, OBS),
    "cond_arm_3_instrtype": _sa(10, 11, OBS),
    "cond_arm_0_color": _sa(2, 3, OBS),
    "cond_arm_1_color": _sa(3, 4, OBS),
    "cond_arm_2_color": _sa(4, 5, OBS),
    "cond_arm_3_color": _sa(5, 6, OBS),
    "cond_electroSurgeryMode_left": _sa(14, 15, OBS),
    "cond_electroSurgeryMode_right": _sa(15, 16, OBS),
}


def _read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def _video_section(public_modality: dict) -> dict:
    """Reuse the public modality.json's video section verbatim (it's correct)."""
    vid = public_modality.get("video")
    if isinstance(vid, dict) and vid:
        return vid
    # Fallback: the CMR endoscope camera.
    return {"endoscope": {"original_key": "observation.images.endoscope"}}


def _annotation_section(public_modality: dict) -> dict | None:
    return public_modality.get("annotation")


def _validate_against_info(dataset_path: Path) -> list[str]:
    """Check the raw action/observation.state columns are wide enough."""
    warnings: list[str] = []
    info_path = dataset_path / INFO_FILENAME
    try:
        info = _read_json(info_path)
    except Exception as e:  # noqa: BLE001
        return [f"could not read info.json: {e!r}"]
    feats = info.get("features", {})

    def _width(col: str) -> int | None:
        meta = feats.get(col)
        if not isinstance(meta, dict):
            return None
        shape = meta.get("shape")
        if isinstance(shape, list) and shape:
            return int(shape[0])
        return None

    # Each entry's source column is ``entry["original_key"]`` when present, else
    # the section default ("observation.state" for state entries, "action" for
    # action entries) — matching the loader's LeRobotState/ActionMetadata schema.
    # (We omit original_key on action entries that read the default "action"
    # column, to byte-match the Draco file, so don't assume the field exists.)
    need = {ACT: 0, OBS: 0}
    for section, default_col in ((_STATE, OBS), (_ACTION, ACT)):
        for entry in section.values():
            col = entry.get("original_key", default_col)
            need[col] = max(need.get(col, 0), int(entry["end"]))
    for col, max_end in need.items():
        w = _width(col)
        if w is None:
            warnings.append(f"info.json has no feature '{col}' (cannot verify width >= {max_end})")
        elif w < max_end:
            warnings.append(f"raw column '{col}' width {w} < required {max_end} — WRONG layout for this leaf!")
    # robot_type sanity (informational).
    rt = info.get("robot_type")
    if rt and "versius" not in str(rt).lower():
        warnings.append(f"robot_type={rt!r} doesn't look like CMR Versius")
    return warnings


def _build_modality_44d(public_modality: dict) -> dict:
    return {
        "state": dict(_STATE),
        "action": dict(_ACTION),
        "video": _video_section(public_modality),
        **({"annotation": _annotation_section(public_modality)} if _annotation_section(public_modality) else {}),
    }


def generate_one(dataset_path: Path, force: bool, dry_run: bool) -> bool:
    print(f"=== {dataset_path} ===")
    if not dataset_path.exists():
        print("  [error] dataset path does not exist")
        return False
    out_path = dataset_path / MODALITY_44D_FILENAME
    public_path = dataset_path / MODALITY_PUBLIC_FILENAME

    if out_path.exists() and not force:
        print(f"  [skip] {out_path.name} already exists (use --force to overwrite)")
        return True
    if not public_path.exists():
        print(f"  [error] {public_path} not found (need it for the video section)")
        return False

    public_modality = _read_json(public_path)
    warnings = _validate_against_info(dataset_path)
    fatal = [w for w in warnings if "WRONG layout" in w]
    for w in warnings:
        print(f"  [warn] {w}")
    if fatal and not force:
        print("  [error] refusing to write: raw column layout doesn't match the 44D index map (use --force to override)")
        return False

    modality_44d = _build_modality_44d(public_modality)
    n_state = len(modality_44d["state"])
    n_action = len(modality_44d["action"])
    print(f"  built modality-44D.json: {n_state} state keys, {n_action} action keys")

    if dry_run:
        print(f"  [dry-run] would write {out_path}")
        print(f"  [dry-run] action keys: {sorted(modality_44d['action'].keys())}")
        return True

    with out_path.open("w") as f:
        json.dump(modality_44d, f, indent=4)
        f.write("\n")
    print(f"  [ok] wrote {out_path}")
    return True


def _cmr_leaves_from_specs(root: str | None) -> list[Path]:
    from cosmos_framework.data.vfm.action.gr00t_dreams.data.embodiment_tags import EmbodimentTag
    from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
        get_open_h_multi_train_specs,
    )

    leaves: list[Path] = []
    for spec in get_open_h_multi_train_specs(base_path=root):
        emb = spec["embodiment"]
        emb = emb.value if isinstance(emb, EmbodimentTag) else emb
        if emb == CMR_TAG:
            leaves.append(Path(spec["path"]))
    return leaves


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-path", default=None, help="single CMR leaf dir (contains meta/)")
    ap.add_argument("--root", default=None, help="OPENH_SURGICAL_ROOT; process all CMR leaves in OPEN_H_DATASET_SPECS")
    ap.add_argument("--force", action="store_true", help="overwrite an existing modality-44D.json")
    ap.add_argument("--dry-run", action="store_true", help="build + validate, write nothing")
    args = ap.parse_args()

    targets: list[Path] = []
    if args.dataset_path:
        targets.append(Path(args.dataset_path))
    if args.root or (not args.dataset_path):
        try:
            targets.extend(_cmr_leaves_from_specs(args.root))
        except Exception as e:  # noqa: BLE001
            if not args.dataset_path:
                raise SystemExit(
                    f"could not import OPEN_H_DATASET_SPECS to find CMR leaves "
                    f"(overlay applied + venv active?): {e!r}"
                )
    # De-dup while preserving order.
    seen = set()
    uniq: list[Path] = []
    for t in targets:
        if str(t) not in seen:
            seen.add(str(t))
            uniq.append(t)
    if not uniq:
        raise SystemExit("no CMR leaves to process (pass --dataset-path or --root)")

    ok = True
    for dp in uniq:
        ok = generate_one(dp, force=args.force, dry_run=args.dry_run) and ok
    if not ok:
        sys.exit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
