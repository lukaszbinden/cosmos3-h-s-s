#!/usr/bin/env bash

#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --job-name=c3hss-dvrk-training-coupling-audit
#SBATCH --output=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/shuver/outputs/c3hss_dvrk_training_coupling_audit/slurm_%j.out
#SBATCH --error=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/shuver/outputs/c3hss_dvrk_training_coupling_audit/slurm_%j.out

set -euo pipefail

LUKAS_ROOT=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden
LUKAS_WORKSPACE=$LUKAS_ROOT/git/c3-h-s-o-b_camp
FRAMEWORK_DIR=$LUKAS_WORKSPACE/packages/cosmos3
CONTAINER_IMAGE=$LUKAS_ROOT/images/cosmos3.sqsh
SHUVER_ROOT=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/shuver
JOB_SCRIPTS=$SHUVER_ROOT/eval_jobs
OUTPUT_ROOT=${OUTPUT_ROOT:-$SHUVER_ROOT/outputs/c3hss_dvrk_training_coupling_audit}
HF_ROOT=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/datasets/Open-H-lz/Surgical/JHU/Imerse/previously_collected_data/hf_suturebot
NEPH_ROOT=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/datasets/Open-H/Surgical/JHU/Imerse/NephFat_extracted/nephfat
WORKERS_PER_DATASET=${WORKERS_PER_DATASET:-24}

for required in \
    "$FRAMEWORK_DIR/.venv/bin/activate" \
    "$CONTAINER_IMAGE" \
    "$JOB_SCRIPTS/scan_c3hss_training_coupling.py" \
    "$HF_ROOT/meta/info.json" \
    "$NEPH_ROOT/meta/info.json"; do
    if [[ ! -e "$required" ]]; then
        echo "FATAL: missing required path: $required" >&2
        exit 64
    fi
done

mkdir -p "$OUTPUT_ROOT/hf_suturebot" "$OUTPUT_ROOT/nephfat"
MOUNTS="/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan:/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan"
MOUNTS="$MOUNTS,$FRAMEWORK_DIR:/workspace"
MOUNTS="$MOUNTS,$JOB_SCRIPTS:/eval"

echo "===== C3-H-S-S FULL TRAINING COUPLING AUDIT $(date -Is) ====="
echo "output_root=$OUTPUT_ROOT"
echo "workers_per_dataset=$WORKERS_PER_DATASET"

srun --ntasks=1 --ntasks-per-node=1 \
    --no-container-mount-home \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace \
    --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",HF_ROOT="$HF_ROOT",NEPH_ROOT="$NEPH_ROOT",WORKERS_PER_DATASET="$WORKERS_PER_DATASET" \
    bash -lc '
        set -euo pipefail
        source /workspace/.venv/bin/activate
        export HOME=/tmp/c3hss-audit-home
        export TMPDIR=/tmp
        export PYTHONPATH=/eval
        mkdir -p "$HOME"

        python /eval/scan_c3hss_training_coupling.py \
            --dataset-root "$HF_ROOT" \
            --output-dir "$OUTPUT_ROOT/hf_suturebot" \
            --subset-name hf_suturebot \
            --split train \
            --extra-window 1382:381 \
            --workers "$WORKERS_PER_DATASET" \
            > "$OUTPUT_ROOT/hf_suturebot/scan.log" 2>&1 &
        hf_pid=$!

        python /eval/scan_c3hss_training_coupling.py \
            --dataset-root "$NEPH_ROOT" \
            --output-dir "$OUTPUT_ROOT/nephfat" \
            --subset-name nephfat \
            --split train \
            --workers "$WORKERS_PER_DATASET" \
            > "$OUTPUT_ROOT/nephfat/scan.log" 2>&1 &
        neph_pid=$!

        status=0
        wait "$hf_pid" || status=1
        wait "$neph_pid" || status=1
        if [[ "$status" -ne 0 ]]; then
            echo "FATAL: at least one dataset scan failed"
            tail -80 "$OUTPUT_ROOT/hf_suturebot/scan.log" || true
            tail -80 "$OUTPUT_ROOT/nephfat/scan.log" || true
            exit "$status"
        fi
    '

echo "===== AUDIT COMPLETE $(date -Is) ====="
for subset in hf_suturebot nephfat; do
    echo "--- $subset"
    tail -10 "$OUTPUT_ROOT/$subset/scan.log"
    ls -lh "$OUTPUT_ROOT/$subset"/coupling_*
done
