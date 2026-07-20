#!/usr/bin/env python3
"""Probe wandb binary format to find correct proto version."""
import json
from wandb.sdk.internal.datastore import DataStore

WFILE = (
    "/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/imaginaire/output"
    "/cosmos3_action_surgical/action_open_h/action_fdm_open_h_sft_nano/wandb"
    "/offline-run-20260627_063934-njcmmy74/run-njcmmy74.wandb"
)

# Read raw data blobs from the leveldb log
print("Reading raw data blobs via DataStore.scan_data()...")
ds = DataStore()
ds.open_for_scan(WFILE)
blobs = []
errors = 0
while True:
    try:
        data = ds.scan_data()
    except Exception as e:
        errors += 1
        try:
            offset = ds.get_offset()
            ds.seek(((offset // 32768) + 1) * 32768)
        except Exception:
            break
        continue
    if data is None:
        break
    blobs.append(data)
ds.close()
print(f"Got {len(blobs)} blobs, {errors} errors\n")

if not blobs:
    print("No blobs — DataStore read failed completely")
    raise SystemExit(1)

# Show first blob raw bytes
print(f"First blob ({len(blobs[0])} bytes, hex): {blobs[0][:32].hex()}\n")

# Try each proto version
proto_modules = []
try:
    from wandb.proto import wandb_internal_pb2 as pb_default
    proto_modules.append(("default", pb_default))
except Exception as e:
    print(f"default proto import failed: {e}")
for ver in ("v3", "v4", "v6"):
    try:
        mod = __import__(f"wandb.proto.{ver}.wandb_internal_pb2", fromlist=["Record"])
        proto_modules.append((ver, mod))
    except Exception as e:
        print(f"{ver} proto import failed: {e}")

print(f"Trying {len(proto_modules)} proto versions on first blob:")
for ver, pb in proto_modules:
    try:
        rec = pb.Record()
        rec.ParseFromString(blobs[0])
        fields = [f.name for f, _ in rec.ListFields()]
        print(f"  {ver}: parsed OK, fields={fields}")
    except Exception as e:
        print(f"  {ver}: parse error: {e}")

# Try all blobs with default proto to count history records
print()
if proto_modules:
    ver, pb = proto_modules[0]
    history_count = 0
    for blob in blobs:
        try:
            rec = pb.Record()
            rec.ParseFromString(blob)
            if rec.HasField("history"):
                history_count += 1
                if history_count == 1:
                    # Print first history record keys
                    row = {item.key: item.value_json for item in rec.history.item}
                    print(f"First history record keys: {list(row.keys())[:10]}")
                    print(f"Sample _step value: {row.get('_step', 'NOT FOUND')}")
        except Exception:
            pass
    print(f"\nTotal history records found with '{ver}' proto: {history_count}/{len(blobs)} blobs")
