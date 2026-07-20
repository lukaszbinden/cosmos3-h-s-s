#!/usr/bin/env python3
"""Re-upload fragmented offline wandb runs as a single continuous run.

Usage (inside container, with WANDB_MODE=online and WANDB_API_KEY set):
    python3 wandb_reupload.py
"""

import glob
import json
import os

WANDB_DIR = (
    "/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/imaginaire/output"
    "/cosmos3_action_surgical/action_open_h/action_fdm_open_h_sft_nano/wandb"
)
PROJECT = "cosmos3_action_surgical"
GROUP   = "action_open_h"
NAME    = "action_fdm_open_h_sft_nano_merged"


def read_wandb_file(path):
    """Read history rows from a .wandb binary file using wandb's DataStore."""
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2 as pb

    rows = []
    ds = DataStore()
    try:
        ds.open_for_scan(path)
    except Exception as e:
        print(f"    open_for_scan error: {e}")
        return rows

    while True:
        try:
            data = ds.scan_data()
        except Exception:
            # skip corrupt block, advance to next block boundary
            try:
                # align to next 32768-byte block
                offset = ds.get_offset()
                block_size = 32768
                next_block = ((offset // block_size) + 1) * block_size
                ds.seek(next_block)
                continue
            except Exception:
                break
        if data is None:
            break
        try:
            rec = pb.Record()
            rec.ParseFromString(data)
            if rec.HasField("history"):
                row = {}
                for item in rec.history.item:
                    key = "/".join(item.nested_key) if item.nested_key else item.key
                    try:
                        row[key] = json.loads(item.value_json)
                    except Exception:
                        row[key] = item.value_json
                if "_step" in row:
                    rows.append(row)
        except Exception:
            continue

    try:
        ds.close()
    except Exception:
        pass
    return rows



def main():
    import wandb

    run_dirs = sorted(glob.glob(os.path.join(WANDB_DIR, "offline-run-*")))
    print(f"Found {len(run_dirs)} offline run directories\n")

    # Collect all history rows
    all_rows = []
    for d in run_dirs:
        wfile = os.path.join(d, "run-njcmmy74.wandb")
        if not os.path.exists(wfile):
            print(f"  SKIP {os.path.basename(d)} — no .wandb file")
            continue
        rows = read_wandb_file(wfile)
        if rows:
            steps = [int(r["_step"]) for r in rows if "_step" in r]
            print(f"  {os.path.basename(d)}: {len(rows)} rows, steps {min(steps)}-{max(steps)}")
            all_rows.extend(rows)
        else:
            print(f"  {os.path.basename(d)}: no history rows (size={os.path.getsize(wfile)})")

    if not all_rows:
        print("\nERROR: Could not parse any history rows.")
        print("Try running: python3 -c \"from wandb.sdk.internal.datastore import DataStore; print('ok')\"")
        return

    # Deduplicate by step, sort
    by_step = {}
    for row in all_rows:
        step = row.get("_step")
        if step is not None:
            by_step[int(step)] = row
    sorted_rows = [by_step[s] for s in sorted(by_step.keys())]
    print(f"\nTotal unique steps: {len(sorted_rows)}, "
          f"range: {sorted_rows[0]['_step']}-{sorted_rows[-1]['_step']}")

    # Load config from a real run
    config = {}
    for d in run_dirs[5:]:
        config_path = os.path.join(d, "files", "config.yaml")
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}
                break
            except Exception:
                pass

    # Create new run and upload
    run = wandb.init(project=PROJECT, group=GROUP, name=NAME,
                     config=config, resume="never")
    print(f"\nCreated new run: {run.url}")
    print("Uploading history...")

    SKIP_KEYS = {"_step", "_runtime", "_timestamp", "_wandb"}
    for i, row in enumerate(sorted_rows):
        step = int(row["_step"])
        metrics = {k: v for k, v in row.items() if k not in SKIP_KEYS}
        if metrics:
            wandb.log(metrics, step=step)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(sorted_rows)} rows uploaded (step={step})...")

    wandb.finish()
    print(f"\nDone. View at: {run.url}")


if __name__ == "__main__":
    main()
