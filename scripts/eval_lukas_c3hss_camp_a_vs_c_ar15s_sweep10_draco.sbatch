#!/usr/bin/env bash

# Arm A versus Arm C over ten matched held-out 15-second autoregressive
# trajectories. Each condition fans its three dataset groups across three GPUs;
# both conditions may run concurrently (six GPUs total).

#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=backfill_block1,backfill_block3,backfill_block4,batch_block1,batch_block3,batch_block4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --time=01:30:00
#SBATCH --array=0-1%2
#SBATCH --job-name=cosmos3-camp-a-vs-c-iter1700-ar15s-sweep10
#SBATCH --output=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/shuver/outputs/c3hss_camp_a_vs_c_iter1700_ar15s_sweep10/slurm_%A_%a.out
#SBATCH --error=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/shuver/outputs/c3hss_camp_a_vs_c_iter1700_ar15s_sweep10/slurm_%A_%a.out

set -euo pipefail
export TMPDIR=/tmp

LUKAS_ROOT=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden
LUKAS_WORKSPACE=$LUKAS_ROOT/git/c3-h-s-o-b_camp
FRAMEWORK_DIR=$LUKAS_WORKSPACE/packages/cosmos3
COOKBOOK_DIR=$LUKAS_WORKSPACE/cookbooks/cosmos3/generator/action/finetune
SHUVER_ROOT=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/shuver
JOB_SCRIPTS=$SHUVER_ROOT/eval_jobs
OUTPUT_ROOT=$SHUVER_ROOT/outputs/c3hss_camp_a_vs_c_iter1700_ar15s_sweep10
DVRK_OPENH_ROOT=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/datasets/Open-H/Surgical
DVRK_LZ_ROOT=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/datasets/Open-H-lz
CONTAINER_IMAGE=$LUKAS_ROOT/images/cosmos3.sqsh
DRACO_NVML_PATH=$LUKAS_ROOT/images/nvidia-pinned
ARM_A_CHECKPOINT=$LUKAS_ROOT/checkpoints/cosmos3-h-surgical-omni-base/iter_000015000
ARM_C_CHECKPOINT=$LUKAS_ROOT/imaginaire/output-arm-c-memory/cosmos3_action_surgical/action_open_h_memory_c/camp-arm-c-memory-iter_000015000-tracks_2150c4e6ac6a-encoder_3da03042327a/checkpoints/iter_000001700
MEMORY_TRACKS_HOST=$LUKAS_ROOT/imaginaire/camp-arm-c-memory-artifacts/tracks-shared-v2-log500
MEMORY_TRACKS_CONTAINER=/camp_tracks

source "$LUKAS_WORKSPACE/env.sh"

case "${SLURM_ARRAY_TASK_ID:?array task id is required}" in
    0)
        EVAL_CONDITION=arm_a_h0
        CHECKPOINT=$ARM_A_CHECKPOINT
        CHECKPOINT_ITER=15000
        NUM_HISTORY_ACTIONS=0
        MEMORY_TRACKS_ROOT=
        ;;
    1)
        EVAL_CONDITION=arm_c_real
        CHECKPOINT=$ARM_C_CHECKPOINT
        CHECKPOINT_ITER=1700
        NUM_HISTORY_ACTIONS=16
        MEMORY_TRACKS_ROOT=$MEMORY_TRACKS_CONTAINER
        ;;
    *)
        echo "FATAL: unsupported array task $SLURM_ARRAY_TASK_ID" >&2
        exit 64
        ;;
esac

for required in \
    "$FRAMEWORK_DIR/.venv/bin/activate" \
    "$COOKBOOK_DIR/toml/sft_config/action_mixed_open_h_sft_nano.toml" \
    "$JOB_SCRIPTS/eval_c3hss_long_horizon.py" \
    "$JOB_SCRIPTS/run_lukas_c3hss_camp_a_vs_c_ar15s_sweep10_draco.sh" \
    "$DVRK_OPENH_ROOT/JHU/Imerse/Wound_Closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled/meta/stats_cosmos-c3hss-v1.json" \
    "$DVRK_OPENH_ROOT/JHU/Imerse/NephFat_extracted/nephfat/meta/stats_cosmos-c3hss-v1.json" \
    "$DVRK_LZ_ROOT/Surgical/JHU/Imerse/previously_collected_data/hf_suturebot/meta/stats_cosmos-c3hss-v1.json" \
    "$CHECKPOINT/model/.metadata" \
    "$WAN_VAE_PATH" \
    "$CONTAINER_IMAGE" \
    "$DRACO_NVML_PATH/libnvidia-ml.so.1" \
    "$MEMORY_TRACKS_HOST/manifest.json"; do
    if [[ ! -e "$required" ]]; then
        echo "FATAL: missing required path: $required" >&2
        exit 64
    fi
done

TRACKS_MANIFEST_SHA256=$(sha256sum "$MEMORY_TRACKS_HOST/manifest.json" | awk '{print $1}')
if [[ "$TRACKS_MANIFEST_SHA256" != "2150c4e6ac6af47a72299b073db53bcb0b883cb835fd73a7f7852ab5e6f4c7ea" ]]; then
    echo "FATAL: memory-track manifest hash changed: $TRACKS_MANIFEST_SHA256" >&2
    exit 65
fi

mkdir -p "$OUTPUT_ROOT" "$JOB_SCRIPTS" "$SHUVER_ROOT/.cache"

OVERLAY_ROOT=$COOKBOOK_DIR/framework_patch
while IFS= read -r -d '' source_file; do
    relative_path="${source_file#"$OVERLAY_ROOT/"}"
    target_file="$FRAMEWORK_DIR/$relative_path"
    if [[ ! -f "$target_file" ]] || ! cmp -s "$source_file" "$target_file"; then
        echo "FATAL: installed framework does not match CAMP overlay: $relative_path" >&2
        exit 65
    fi
done < <(
    find "$OVERLAY_ROOT" -type f \
        ! -path '*/__pycache__/*' \
        ! -name '*.pyc' \
        -print0
)
echo "CAMP overlay verification passed (read-only)"

MOUNTS="/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan:/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan"
MOUNTS="$MOUNTS,/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan:/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan"
MOUNTS="$MOUNTS,$FRAMEWORK_DIR:/workspace"
MOUNTS="$MOUNTS,$COOKBOOK_DIR:/cookbook"
MOUNTS="$MOUNTS,$JOB_SCRIPTS:/eval"
MOUNTS="$MOUNTS,$DRACO_NVML_PATH:/opt/nvidia-pinned:ro"
MOUNTS="$MOUNTS,$WAN_VAE_PATH:/wan_vae/Wan2.2_VAE.pth:ro"
MOUNTS="$MOUNTS,$LUKAS_ROOT/.hf_cache/hub:/root/.cache/huggingface/hub:ro"
MOUNTS="$MOUNTS,$SHUVER_ROOT/.cache:/root/.cache/cosmos-a-vs-c-ar15"
MOUNTS="$MOUNTS,$MEMORY_TRACKS_HOST:$MEMORY_TRACKS_CONTAINER:ro"

echo "===== MATCHED CAMP A VS C AR15 SWEEP10 CONDITION $(date -Is) ====="
echo "condition=$EVAL_CONDITION checkpoint=$CHECKPOINT checkpoint_iter=$CHECKPOINT_ITER"
echo "history_rows=$NUM_HISTORY_ACTIONS memory_root=${MEMORY_TRACKS_ROOT:-absent}"
echo "episodes=wound:[209,212,214,215] hf:[1420,1426,1435] nephfat:[2046,2049,2052]"
echo "chunks=13 generated_frames=157 output_frames=150 fps=10 duration=15.0s output_root=$OUTPUT_ROOT"

srun --no-container-mount-home --container-image="$CONTAINER_IMAGE" \
    bash -c 'echo pyxis-warmup-ok on $(hostname)' || true

srun --ntasks=1 --ntasks-per-node=1 \
    --no-container-mount-home \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace \
    --export=ALL,CHECKPOINT="$CHECKPOINT",CHECKPOINT_ITER="$CHECKPOINT_ITER",EVAL_CONDITION="$EVAL_CONDITION",NUM_HISTORY_ACTIONS="$NUM_HISTORY_ACTIONS",MEMORY_TRACKS_ROOT="$MEMORY_TRACKS_ROOT",OUTPUT_ROOT="$OUTPUT_ROOT",DVRK_OPENH_ROOT="$DVRK_OPENH_ROOT",DVRK_LZ_ROOT="$DVRK_LZ_ROOT",COSMOS_OPENH_PATH_LAYOUT=draco_internal,COSMOS_OPENH_STATS_POSTFIX=c3hss-v1 \
    bash /eval/run_lukas_c3hss_camp_a_vs_c_ar15s_sweep10_draco.sh
