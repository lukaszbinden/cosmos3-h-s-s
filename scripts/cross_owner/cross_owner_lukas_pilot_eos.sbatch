#!/bin/bash
#SBATCH --job-name=cosmos3-cross-owner-lukas-pilot
#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --constraint=h100
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.out

set -euo pipefail
EOSU=/lustre/fsw/healthcareeng_holoscan/user_data/shuver
OUT=${OUT:-$EOSU/outputs/cosmos3_lukas_vs_justin_v1}
J=/lustre/fsw/healthcareeng_holoscan/user_data/justchen/projects/cosmos3-internal
LHOME=/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden
L=$LHOME/git/cosmos3-h-s-s
LPKG=$L/packages/cosmos3
CKPT=$LHOME/imaginaire/output/cosmos3_action_surgical/action_open_h/mm-C3-H-S-S-base/checkpoints/iter_000026500
TOML=$L/cookbooks/cosmos3/generator/action/finetune/toml/sft_config/action_mixed_open_h_sft_nano.toml
OPENH=/lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment/Surgical
WAN=/lustre/fsw/healthcareeng_holoscan/user_data/shuver/cosmos3_surgical_fd/checkpoints/wan22_vae/Wan2.2_VAE.pth
CONTAINER='docker://nvcr.io#nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04'

case "${SLURM_ARRAY_TASK_ID:-0}" in
  1) BRACKET=justin_heldout; SPLIT=train ;;
  2) BRACKET=lukas_heldout; SPLIT=test ;;
  *) echo "array index must be 1..2"; exit 64 ;;
esac
TARGETS=$OUT/targets/$BRACKET
MANIFEST=$OUT/manifests/$BRACKET.tsv
OFFSETS=$TARGETS/lzb_start_offsets.json
DEST=$OUT/pilot/$BRACKET/lukas_26500/iter_000026500
mkdir -p "$DEST" "$OUT/_imaginaire"

MOUNTS="$L:$L:ro,$J:$J:ro,$OUT:$OUT,$OPENH:$OPENH:ro,$CKPT:$CKPT:ro,$WAN:/wan_vae/Wan2.2_VAE.pth:ro,$LHOME/.uv_cache:$LHOME/.uv_cache:ro,$LHOME/.uv_python:$LHOME/.uv_python:ro,/lustre/fsw/healthcareeng_holoscan/user_data/justchen/.cache:/root/.cache"
srun --container-image="$CONTAINER" --container-mounts="$MOUNTS" --container-workdir="$LPKG" \
  --export=ALL,L="$L",LPKG="$LPKG",J="$J",OUT="$OUT",CKPT="$CKPT",TOML="$TOML",OPENH="$OPENH",TARGETS="$TARGETS",MANIFEST="$MANIFEST",OFFSETS="$OFFSETS",DEST="$DEST",SPLIT="$SPLIT" \
  bash -lc '
    set -euo pipefail
    sha256sum -c <<EOF
8cbd6b55469b0ecd735a085979b6068161cb81bf701f46c931bbe6182691134c  $TOML
f9bafa9845383419de64673286e3a2bd93504dd22b44147ceb1115cb391fea58  $J/scripts/lzb_long_horizon_fds.py
eb80dc029a75ce45508f02626339ecfbfcc7ec63f06cc385069ab2cedd54880e  $J/scripts/fds_metrics.py
57c66852a462af11d3e76bfffdc38a47599254c63dfcd2f10f6b031f0eb12ef2  $CKPT/model/.metadata
EOF
    source "$LPKG/.venv/bin/activate"
    export LD_LIBRARY_PATH= PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
    export WAN_VAE_PATH=/wan_vae/Wan2.2_VAE.pth OPENH_SURGICAL_ROOT="$OPENH"
    export IMAGINAIRE_OUTPUT_ROOT="$OUT/_imaginaire" COSMOS_OPENH_STATS_POSTFIX=c3hss-v1
    export PYTHONPATH="$J/scripts:$L/cookbooks/cosmos3/generator/action/finetune/scripts:${PYTHONPATH:-}"
    torchrun --nproc_per_node=1 "$J/scripts/lzb_long_horizon_fds.py" \
      --sft-toml "$TOML" --checkpoint "$CKPT" --manifest "$MANIFEST" \
      --output-root "$DEST" --targets-dir "$TARGETS" --openh-root "$OPENH" \
      --data-split "$SPLIT" --test-split-ratio 0.02 --iteration 26500 \
      --max-chunks 2 --min-chunks 2 --guidance 1.5 --num-sampling-step 16 \
      --score-size 320 192 --start-offsets "$OFFSETS" --emit-eval-layout
    test -s "$DEST/_run_summary.json"
  '
