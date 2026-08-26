#!/bin/bash
#SBATCH --job-name=cosmos3-cross-owner-offsets
#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

set -euo pipefail
EOSU=/lustre/fsw/healthcareeng_holoscan/user_data/shuver
OUT=${OUT:-$EOSU/outputs/cosmos3_lukas_vs_justin_v1}
J=/lustre/fsw/healthcareeng_holoscan/user_data/justchen/projects/cosmos3-internal

for B in justin_heldout lukas_heldout; do
  ROOT=$OUT/pilot/$B/justin_14280/iter_000014280
  python3 "$J/scripts/build_lzb_start_offsets.py" --ours-root "$ROOT" \
    --fds-subdir . --output "$OUT/targets/$B/lzb_start_offsets.json"
  test "$(python3 -c 'import json,sys; print(sum(map(len,json.load(open(sys.argv[1])).values())))' "$OUT/targets/$B/lzb_start_offsets.json")" -eq 2
done
echo OFFSETS_PASS
