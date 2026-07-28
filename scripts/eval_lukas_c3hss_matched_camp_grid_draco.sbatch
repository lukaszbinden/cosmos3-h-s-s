#!/usr/bin/env bash

#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=backfill_block1,backfill_block3,backfill_block4,batch_block1,batch_block3,batch_block4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --array=0-8%9
#SBATCH --job-name=cosmos3-camp-matched-iter950
#SBATCH --output=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/shuver/outputs/c3hss_camp_matched_iter950/slurm_%A_%a.out
#SBATCH --error=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/shuver/outputs/c3hss_camp_matched_iter950/slurm_%A_%a.out

set -euo pipefail
export TMPDIR=/tmp

LUKAS_ROOT=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden
LUKAS_WORKSPACE=$LUKAS_ROOT/git/c3-h-s-o-b_camp
FRAMEWORK_DIR=$LUKAS_WORKSPACE/packages/cosmos3
COOKBOOK_DIR=$LUKAS_WORKSPACE/cookbooks/cosmos3/generator/action/finetune
SHUVER_ROOT=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/shuver
JOB_SCRIPTS=$SHUVER_ROOT/eval_jobs
OUTPUT_ROOT=${OUTPUT_ROOT:-$SHUVER_ROOT/outputs/c3hss_camp_matched_iter950}
DVRK_OPENH_ROOT=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/datasets/Open-H/Surgical
DVRK_LZ_ROOT=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/datasets/Open-H-lz
CONTAINER_IMAGE=$LUKAS_ROOT/images/cosmos3.sqsh
DRACO_NVML_PATH=$LUKAS_ROOT/images/nvidia-pinned
ARM_A_CHECKPOINT=$LUKAS_ROOT/checkpoints/cosmos3-h-surgical-omni-base/iter_000015000
ARM_B_CHECKPOINT=$LUKAS_ROOT/imaginaire/output/cosmos3_action_surgical/action_open_h_camp/camp-arm-b-h16-iter_000015000/checkpoints/iter_000000950
ARM_C_CHECKPOINT=$LUKAS_ROOT/imaginaire/output-arm-c-memory/cosmos3_action_surgical/action_open_h_memory_c/camp-arm-c-memory-iter_000015000-tracks_2150c4e6ac6a-encoder_3da03042327a/checkpoints/iter_000000950
MEMORY_TRACKS_HOST=$LUKAS_ROOT/imaginaire/camp-arm-c-memory-artifacts/tracks-shared-v2-log500
MEMORY_TRACKS_CONTAINER=/camp_tracks
COMPARISON_STEP=950
SEEDS=${SEEDS:-"0 1 2 3 4"}
DATA_SEED=${DATA_SEED:-1729}
PHYSICAL_COMPONENTS=${PHYSICAL_COMPONENTS:-"tx ty tz"}

source "$LUKAS_WORKSPACE/env.sh"

case "${SLURM_ARRAY_TASK_ID:?array task id is required}" in
    0)
        EVAL_CONDITION=arm_a_h0
        MODEL_LABEL="Arm A H0 base"
        CHECKPOINT=$ARM_A_CHECKPOINT
        CHECKPOINT_ITER=15000
        NUM_HISTORY_ACTIONS=0
        HISTORY_ABLATION=
        MEMORY_ABLATION=
        MEMORY_TRACKS_ROOT=
        ;;
    1)
        EVAL_CONDITION=arm_b_history_real
        MODEL_LABEL="Arm B H16"
        CHECKPOINT=$ARM_B_CHECKPOINT
        CHECKPOINT_ITER=950
        NUM_HISTORY_ACTIONS=16
        HISTORY_ABLATION=
        MEMORY_ABLATION=
        MEMORY_TRACKS_ROOT=
        ;;
    2)
        EVAL_CONDITION=arm_b_history_zero
        MODEL_LABEL="Arm B H16"
        CHECKPOINT=$ARM_B_CHECKPOINT
        CHECKPOINT_ITER=950
        NUM_HISTORY_ACTIONS=16
        HISTORY_ABLATION=zero
        MEMORY_ABLATION=
        MEMORY_TRACKS_ROOT=
        ;;
    3)
        EVAL_CONDITION=arm_b_history_permute
        MODEL_LABEL="Arm B H16"
        CHECKPOINT=$ARM_B_CHECKPOINT
        CHECKPOINT_ITER=950
        NUM_HISTORY_ACTIONS=16
        HISTORY_ABLATION=permute
        MEMORY_ABLATION=
        MEMORY_TRACKS_ROOT=
        ;;
    4)
        EVAL_CONDITION=arm_c_real
        MODEL_LABEL="Arm C H16+memory"
        CHECKPOINT=$ARM_C_CHECKPOINT
        CHECKPOINT_ITER=950
        NUM_HISTORY_ACTIONS=16
        HISTORY_ABLATION=
        MEMORY_ABLATION=
        MEMORY_TRACKS_ROOT=$MEMORY_TRACKS_CONTAINER
        ;;
    5)
        EVAL_CONDITION=arm_c_memory_zero
        MODEL_LABEL="Arm C H16+memory"
        CHECKPOINT=$ARM_C_CHECKPOINT
        CHECKPOINT_ITER=950
        NUM_HISTORY_ACTIONS=16
        HISTORY_ABLATION=
        MEMORY_ABLATION=zero
        MEMORY_TRACKS_ROOT=$MEMORY_TRACKS_CONTAINER
        ;;
    6)
        EVAL_CONDITION=arm_c_memory_shuffle
        MODEL_LABEL="Arm C H16+memory"
        CHECKPOINT=$ARM_C_CHECKPOINT
        CHECKPOINT_ITER=950
        NUM_HISTORY_ACTIONS=16
        HISTORY_ABLATION=
        MEMORY_ABLATION=shuffle_episode
        MEMORY_TRACKS_ROOT=$MEMORY_TRACKS_CONTAINER
        ;;
    7)
        EVAL_CONDITION=arm_c_history_zero
        MODEL_LABEL="Arm C H16+memory"
        CHECKPOINT=$ARM_C_CHECKPOINT
        CHECKPOINT_ITER=950
        NUM_HISTORY_ACTIONS=16
        HISTORY_ABLATION=zero
        MEMORY_ABLATION=
        MEMORY_TRACKS_ROOT=$MEMORY_TRACKS_CONTAINER
        ;;
    8)
        EVAL_CONDITION=arm_c_history_permute
        MODEL_LABEL="Arm C H16+memory"
        CHECKPOINT=$ARM_C_CHECKPOINT
        CHECKPOINT_ITER=950
        NUM_HISTORY_ACTIONS=16
        HISTORY_ABLATION=permute
        MEMORY_ABLATION=
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
    "$JOB_SCRIPTS/eval_c3hss_action_interventions.py" \
    "$JOB_SCRIPTS/c3hss_physical_action_interventions.py" \
    "$JOB_SCRIPTS/run_lukas_c3hss_matched_camp_grid_draco.sh" \
    "$DVRK_OPENH_ROOT/JHU/Imerse/NephFat_extracted/nephfat/meta/stats_cosmos-c3hss-v1.json" \
    "$DVRK_LZ_ROOT/Surgical/JHU/Imerse/previously_collected_data/hf_suturebot/meta/stats_cosmos-c3hss-v1.json" \
    "$WAN_VAE_PATH" \
    "$CONTAINER_IMAGE" \
    "$DRACO_NVML_PATH/libnvidia-ml.so.1" \
    "$MEMORY_TRACKS_HOST/manifest.json"; do
    if [[ ! -e "$required" ]]; then
        echo "FATAL: missing required path: $required" >&2
        exit 64
    fi
done
for state in model optim scheduler trainer; do
    if [[ ! -f "$CHECKPOINT/$state/.metadata" ]]; then
        echo "FATAL: incomplete checkpoint state: $CHECKPOINT/$state/.metadata" >&2
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
MOUNTS="$MOUNTS,$SHUVER_ROOT/.cache:/root/.cache/cosmos-matched-camp"
MOUNTS="$MOUNTS,$MEMORY_TRACKS_HOST:$MEMORY_TRACKS_CONTAINER:ro"

echo "===== MATCHED CAMP GRID CONDITION $(date -Is) ====="
echo "condition=$EVAL_CONDITION model=$MODEL_LABEL"
echo "checkpoint=$CHECKPOINT checkpoint_iter=$CHECKPOINT_ITER comparison_step=$COMPARISON_STEP"
echo "history_rows=$NUM_HISTORY_ACTIONS history_ablation=${HISTORY_ABLATION:-real_or_absent}"
echo "memory_root=${MEMORY_TRACKS_ROOT:-absent} memory_ablation=${MEMORY_ABLATION:-real_or_absent}"
echo "seeds=$SEEDS data_seed=$DATA_SEED components=$PHYSICAL_COMPONENTS"
echo "tracks_manifest_sha256=$TRACKS_MANIFEST_SHA256"
echo "output_root=$OUTPUT_ROOT"

srun --no-container-mount-home --container-image="$CONTAINER_IMAGE" \
    bash -c 'echo pyxis-warmup-ok on $(hostname)' || true

srun --ntasks=1 --ntasks-per-node=1 \
    --no-container-mount-home \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace \
    --export=ALL,CHECKPOINT="$CHECKPOINT",CHECKPOINT_ITER="$CHECKPOINT_ITER",COMPARISON_STEP="$COMPARISON_STEP",EVAL_CONDITION="$EVAL_CONDITION",MODEL_LABEL="$MODEL_LABEL",NUM_HISTORY_ACTIONS="$NUM_HISTORY_ACTIONS",HISTORY_ABLATION="$HISTORY_ABLATION",MEMORY_ABLATION="$MEMORY_ABLATION",MEMORY_TRACKS_ROOT="$MEMORY_TRACKS_ROOT",OUTPUT_ROOT="$OUTPUT_ROOT",DVRK_OPENH_ROOT="$DVRK_OPENH_ROOT",DVRK_LZ_ROOT="$DVRK_LZ_ROOT",SEEDS="$SEEDS",DATA_SEED="$DATA_SEED",PHYSICAL_COMPONENTS="$PHYSICAL_COMPONENTS",COSMOS_OPENH_PATH_LAYOUT=draco_internal,COSMOS_OPENH_STATS_POSTFIX=c3hss-v1 \
    bash /eval/run_lukas_c3hss_matched_camp_grid_draco.sh
