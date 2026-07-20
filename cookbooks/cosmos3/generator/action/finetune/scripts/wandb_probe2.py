#!/usr/bin/env python3
"""Probe the structure of history records in wandb 0.25.0 binary files."""
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore

WFILE = (
    "/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/imaginaire/output"
    "/cosmos3_action_surgical/action_open_h/action_fdm_open_h_sft_nano/wandb"
    "/offline-run-20260627_063934-njcmmy74/run-njcmmy74.wandb"
)

ds = DataStore()
ds.open_for_scan(WFILE)

found = 0
while found < 2:
    data = ds.scan_data()
    if data is None:
        break
    rec = pb.Record()
    rec.ParseFromString(data)
    if not rec.HasField("history"):
        continue
    found += 1
    h = rec.history
    print(f"=== History record {found} ===")
    print(f"History proto fields: {[f.name for f, _ in h.ListFields()]}")
    print(f"num items: {len(h.item)}")
    if h.item:
        item = h.item[0]
        print(f"  item[0] fields: {[f.name for f, _ in item.ListFields()]}")
        print(f"  item[0] repr: {repr(item)[:300]}")
    # Check for json_data or step fields at record level
    print(f"Record-level fields: {[f.name for f, _ in rec.ListFields()]}")
    # Try printing the full history repr
    print(f"Full history repr: {repr(h)[:500]}")
    print()

ds.close()
