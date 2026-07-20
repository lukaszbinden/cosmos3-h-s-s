#!/usr/bin/env python3
"""Debug why read_wandb_file returns no rows despite probe2 finding history."""
import json
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

WFILE = (
    "/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/imaginaire/output"
    "/cosmos3_action_surgical/action_open_h/action_fdm_open_h_sft_nano/wandb"
    "/offline-run-20260627_063934-njcmmy74/run-njcmmy74.wandb"
)

ds = DataStore()
ds.open_for_scan(WFILE)

total = 0
history_total = 0
has_step = 0
exceptions = []

while True:
    try:
        data = ds.scan_data()
    except Exception as e:
        exceptions.append(str(e))
        try:
            offset = ds.get_offset()
            ds.seek(((offset // 32768) + 1) * 32768)
        except Exception:
            break
        continue
    if data is None:
        break
    total += 1
    try:
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.HasField("history"):
            history_total += 1
            row = {}
            for item in rec.history.item:
                key = item.nested_key if item.nested_key else item.key
                try:
                    row[key] = json.loads(item.value_json)
                except Exception:
                    row[key] = item.value_json
            if "_step" in row:
                has_step += 1
                if has_step <= 3:
                    print(f"  Row with _step={row['_step']}: keys={[k for k in row if not k.startswith('stats/')][:8]}")
            else:
                if history_total <= 3:
                    sample_keys = [item.nested_key or item.key for item in rec.history.item[:3]]
                    print(f"  History record {history_total} WITHOUT _step, sample keys: {sample_keys}")
    except Exception as e:
        exceptions.append(f"parse error: {e}")

ds.close()
print(f"\nTotal blobs: {total}")
print(f"History records: {history_total}")
print(f"History records with _step: {has_step}")
print(f"Exceptions: {len(exceptions)}")
for e in exceptions[:5]:
    print(f"  {e}")
