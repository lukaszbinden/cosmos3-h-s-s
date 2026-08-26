#!/bin/bash
# Array rows: Justin-held-out CMR/JHU, then Lukas-held-out CMR/JHU.
#SBATCH --job-name=cosmos3-cross-owner-justin-pilot
#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --constraint=h100
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.out

set -euo pipefail
EOSU=/lustre/fsw/healthcareeng_holoscan/user_data/shuver
OUT=${OUT:-$EOSU/outputs/cosmos3_lukas_vs_justin_v1}
J=/lustre/fsw/healthcareeng_holoscan/user_data/justchen/projects/cosmos3-internal
JH=/lustre/fsw/healthcareeng_holoscan/user_data/justchen
JX=$JH/outputs/crossckpt_eval
OPENH=/lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment/Surgical
CKPT=$JH/outputs/exp_cmr_clinical_train/cosmos3_surgical/action_exp232_mix/action_exp232_mix_v3_latentcache256/checkpoints/iter_000014280
CFG=$JX/_configs/ours_14280.yaml
CONTAINER='docker://nvcr.io#nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04'

case "${SLURM_ARRAY_TASK_ID:-0}" in
  1) BRACKET=justin_heldout; FAMILY=cmr_clinical; SPLIT=val ;;
  2) BRACKET=justin_heldout; FAMILY=dvrk_jhu; SPLIT=val ;;
  3) BRACKET=lukas_heldout;  FAMILY=cmr_clinical; SPLIT=train ;;
  4) BRACKET=lukas_heldout;  FAMILY=dvrk_jhu; SPLIT=train ;;
  *) echo "array index must be 1..4"; exit 64 ;;
esac
TARGET=$OUT/targets/$BRACKET/$FAMILY.json
DEST=$OUT/pilot/$BRACKET/justin_14280/iter_000014280/$FAMILY
mkdir -p "$DEST" "$J/slurm_logs"

export COSMOS_EPISODE_ROOT_ALIASES="{\"$JH/datasets/openh_v30\":\"OPENH\",\"$OPENH\":\"OPENH\",\"$OPENH/cmr_surgical\":\"CMR\",\"$JH/datasets/cmr_clinical_v30\":\"CMR\"}"
MOUNTS="$J:$J,$J:/workspace,$OUT:$OUT,$OPENH:$OPENH:ro,$JH/datasets:$JH/datasets:ro,$CKPT:/ckpt:ro,$JH/.cache:/root/.cache,$JH/.local/share/uv:$JH/.local/share/uv"

srun --container-image="$CONTAINER" --container-mounts="$MOUNTS" --container-workdir=/workspace \
  --export=ALL,CFG="$CFG",FAMILY="$FAMILY",SPLIT="$SPLIT",TARGET="$TARGET",DEST="$DEST",COSMOS_EPISODE_ROOT_ALIASES \
  bash -lc '
    set -euo pipefail
    sha256sum -c <<EOF
e9e81810e21678c261ea2683c07031c1611f85d1a07a9ae36c06a7e982fe8f5f  $CFG
9905b186a23e5e293c5aaf8225c424ceb11b4ef8e47527f5626a29c6a276a0c2  /workspace/scripts/long_horizon_fds.py
eb80dc029a75ce45508f02626339ecfbfcc7ec63f06cc385069ab2cedd54880e  /workspace/scripts/fds_metrics.py
EOF
    source /workspace/.venv/bin/activate
    export LD_LIBRARY_PATH= HF_HOME=/root/.cache/huggingface WANDB_MODE=offline
    torchrun --nproc_per_node=8 /workspace/scripts/long_horizon_fds.py \
      --checkpoint-path /ckpt --config-file "$CFG" --output-dir "$DEST" \
      --dataset "$FAMILY" --dataset-split "$SPLIT" --episode-targets "$TARGET" \
      --max-chunks 2 --guidance 1.5 --num-steps 16 --seed 0 --score-size 320 192
    test -s "$DEST/fds_results.json"
  '
