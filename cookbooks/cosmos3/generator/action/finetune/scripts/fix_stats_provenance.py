#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Rewrite the embedded ``_provenance.dataset_set_*`` fields in existing stats files.

WHY: ``compute_openh_action_stats.py`` stamps each ``stats_cosmos[-44D]-<postfix>.json``
with a ``_provenance`` block that records the FULL dataset-set signature
(``dataset_set_hash`` / ``dataset_set_leaves`` / ``dataset_set_size``) at the time
it was generated. If ``OPEN_H_DATASET_SPECS`` later changes membership (e.g. the
SMARTS stitch-coldcut leaves were dropped, taking the set from 40 -> 37 leaves),
previously-written stats files carry the STALE signature. The stats VALUES are
per-dataset (each leaf's mean/std depends only on its own data), so they remain
correct — and the training loader never reads ``_provenance`` (it skips
underscore-prefixed keys). This mismatch is therefore COSMETIC, but it (a) makes
the no-clobber guard ``[REFUSE]`` future re-runs and (b) misleads anyone auditing
which mixture a file belongs to.

This script fixes that WITHOUT recomputing stats: it recomputes the current
dataset-set signature EXACTLY as the generator does
(``_dataset_set_signature``: ``f"{emb}:{Path(path).name}"`` sorted, sha1[:12])
and rewrites only the three ``dataset_set_*`` provenance fields in each existing
stats file (and its archival sidecar). Everything else — per-key stats,
``timestep_interval``, ``experiment_id``, timestamps — is left untouched, except
an audit breadcrumb ``_provenance.dataset_set_rewritten_utc``.

It only touches files whose embedded hash differs from the current one
(idempotent), supports ``--dry-run``, and matches by ``--postfix`` so it won't
touch a colleague's differently-postfixed files.

Usage::

    # preview which files would be updated:
    python scripts/fix_stats_provenance.py --root "$OPENH_SURGICAL_ROOT" \
        --postfix c3hss-v1 --dry-run

    # apply:
    python scripts/fix_stats_provenance.py --root "$OPENH_SURGICAL_ROOT" \
        --postfix c3hss-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("COSMOS_OPENH_STATS_COMPUTE_MODE", "1")

CMR_TAG = "cmr_versius"


def _current_signature(root: str | None) -> tuple[list[str], str]:
    """Recompute (sorted leaf list, sha1[:12]) for the CURRENT specs.

    MUST mirror compute_openh_action_stats._dataset_set_signature exactly so the
    rewritten hash equals what a fresh ``--force`` run would stamp.
    """
    from cosmos_framework.data.vfm.action.gr00t_dreams.data.embodiment_tags import EmbodimentTag
    from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
        get_open_h_multi_train_specs,
    )

    leaves = []
    # base_path=None on purpose: the signature is computed over the registry's
    # canonical leaf NAMES, not the rebased paths (same as the generator).
    for spec in get_open_h_multi_train_specs(base_path=None):
        emb = spec["embodiment"]
        emb = emb.value if isinstance(emb, EmbodimentTag) else emb
        leaves.append(f"{emb}:{Path(spec['path']).name}")
    leaves = sorted(leaves)
    h = hashlib.sha1("\n".join(leaves).encode()).hexdigest()[:12]
    return leaves, h


def _iter_stat_files(root: str | None, postfix: str):
    """Yield (dataset_path, embodiment, [stats_file_paths]) for each spec leaf.

    Includes the live postfixed file AND the archival sidecar, if present.
    """
    from cosmos_framework.data.vfm.action.gr00t_dreams.data.embodiment_tags import EmbodimentTag
    from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
        get_open_h_multi_train_specs,
    )

    for spec in get_open_h_multi_train_specs(base_path=root):
        emb = spec["embodiment"]
        emb = emb.value if isinstance(emb, EmbodimentTag) else emb
        dp = Path(spec["path"])
        meta = dp / "meta"
        cmr = emb == CMR_TAG
        live = meta / (f"stats_cosmos-44D-{postfix}.json" if cmr else f"stats_cosmos-{postfix}.json")
        files = [live]
        # Archival sidecars are named by experiment_id, which we don't know a
        # priori — discover them by glob.
        sidecar_glob = "stats_cosmos-44D.*.json" if cmr else "stats_cosmos.*.json"
        for sc in sorted(meta.glob(sidecar_glob)):
            # Exclude the postfixed live files (different shape) and the bare names.
            name = sc.name
            if name in {f"stats_cosmos-44D-{postfix}.json", f"stats_cosmos-{postfix}.json"}:
                continue
            if name in {"stats_cosmos.json", "stats_cosmos-44D.json"}:
                continue
            files.append(sc)
        yield dp, emb, files


def _rewrite_file(path: Path, leaves: list[str], hash_: str, dry_run: bool) -> str:
    """Return a status string; update the file in place unless dry_run."""
    if not path.exists():
        return "absent"
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        return f"unreadable ({e!r})"
    prov = data.get("_provenance")
    if not isinstance(prov, dict):
        return "no _provenance"
    old_hash = prov.get("dataset_set_hash")
    if old_hash == hash_:
        return "already-current"
    if dry_run:
        return f"would update ({old_hash} -> {hash_})"
    prov["dataset_set_hash"] = hash_
    prov["dataset_set_leaves"] = leaves
    prov["dataset_set_size"] = len(leaves)
    prov["dataset_set_rewritten_utc"] = datetime.now(timezone.utc).isoformat()
    # Preserve the file's existing formatting style as best we can: stats files
    # are written compact (no indent) by compute_openh_action_stats, sidecars too.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f)
    tmp.replace(path)  # atomic
    return f"updated ({old_hash} -> {hash_})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="OPENH_SURGICAL_ROOT to locate each leaf's meta/ dir")
    ap.add_argument("--postfix", default=None, help="stats postfix (else COSMOS_OPENH_STATS_POSTFIX)")
    ap.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args()

    postfix = (args.postfix or os.environ.get("COSMOS_OPENH_STATS_POSTFIX", "")).strip()
    if not postfix:
        print("[FATAL] need --postfix or COSMOS_OPENH_STATS_POSTFIX", file=sys.stderr)
        sys.exit(2)

    try:
        leaves, hash_ = _current_signature(args.root)
    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] could not import specs (overlay applied + venv active?): {e!r}", file=sys.stderr)
        sys.exit(2)

    print(f"[info] current dataset set: {len(leaves)} leaves, hash={hash_}, postfix={postfix!r}")
    print(f"[info] {'DRY-RUN — ' if args.dry_run else ''}rewriting stale _provenance.dataset_set_* fields\n")

    root_p = Path(str(args.root)) if args.root else None

    def _disp(fp: Path) -> str:
        if root_p is not None:
            try:
                return str(fp.relative_to(root_p))
            except ValueError:
                pass
        return str(fp)

    n_updated = n_current = n_other = 0
    for dp, emb, files in _iter_stat_files(args.root, postfix):
        for fp in files:
            status = _rewrite_file(fp, leaves, hash_, args.dry_run)
            if status.startswith(("updated", "would update")):
                n_updated += 1
                print(f"  [{'DRY' if args.dry_run else 'OK'}] {emb:22s} {_disp(fp)} : {status}")
            elif status == "already-current":
                n_current += 1
            else:
                n_other += 1
                if status != "absent":
                    print(f"  [skip] {emb:22s} {fp.name} : {status}")

    print("\n" + "=" * 72)
    print(
        f"{'Would update' if args.dry_run else 'Updated'}: {n_updated} | "
        f"already-current: {n_current} | skipped/absent: {n_other}"
    )
    if args.dry_run and n_updated:
        print("Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
