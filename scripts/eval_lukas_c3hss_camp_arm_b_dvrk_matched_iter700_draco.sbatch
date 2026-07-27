#!/usr/bin/env bash

# Reproduces the six-GPU matched diagnostic submitted as jobs 11194583 and
# 11194584. Submit once per ROLLOUT_MODE with an isolated OUTPUT_ROOT.

#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=backfill_block1,backfill_block3,backfill_block4,batch_block1,batch_block3,batch_block4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --time=01:00:00
#SBATCH --job-name=cosmos3-camp-arm-b-dvrk-matched-iter700
#SBATCH --output=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/shuver/outputs/c3hss_camp_arm_b_iter700_dvrk_matched/slurm_%j.out
#SBATCH --error=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/shuver/outputs/c3hss_camp_arm_b_iter700_dvrk_matched/slurm_%j.out

set -euo pipefail
export TMPDIR=/tmp

: "${ROLLOUT_MODE:?set ROLLOUT_MODE=autoregressive or teacher_forced}"
: "${OUTPUT_ROOT:?set an isolated OUTPUT_ROOT}"
if [[ "$ROLLOUT_MODE" == "autoregressive" ]]; then
    RUN_LAUNCHER=run_lukas_c3hss_camp_arm_b_dvrk_fd20s_sweep_draco.sh
elif [[ "$ROLLOUT_MODE" == "teacher_forced" ]]; then
    RUN_LAUNCHER=run_lukas_c3hss_camp_arm_b_dvrk_teacher_forced_draco.sh
else
    echo "FATAL: unsupported ROLLOUT_MODE=$ROLLOUT_MODE" >&2
    exit 64
fi

LUKAS_ROOT=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden
LUKAS_WORKSPACE=$LUKAS_ROOT/git/c3-h-s-o-b_camp
FRAMEWORK_DIR=$LUKAS_WORKSPACE/packages/cosmos3
COOKBOOK_DIR=$LUKAS_WORKSPACE/cookbooks/cosmos3/generator/action/finetune
SHUVER_ROOT=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/shuver
JOB_SCRIPTS=$SHUVER_ROOT/eval_jobs
DVRK_OPENH_ROOT=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/datasets/Open-H/Surgical
DVRK_LZ_ROOT=/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/datasets/Open-H-lz
CONTAINER_IMAGE=$LUKAS_ROOT/images/cosmos3.sqsh
DRACO_NVML_PATH=$LUKAS_ROOT/images/nvidia-pinned
CHECKPOINT_ITER=700
CHECKPOINT=$LUKAS_ROOT/imaginaire/output/cosmos3_action_surgical/action_open_h_camp/camp-arm-b-h16-iter_000015000/checkpoints/iter_000000700

source "$LUKAS_WORKSPACE/env.sh"

for required in \
    "$FRAMEWORK_DIR/.venv/bin/activate" \
    "$COOKBOOK_DIR/toml/sft_config/action_mixed_open_h_sft_nano.toml" \
    "$JOB_SCRIPTS/eval_c3hss_long_horizon.py" \
    "$JOB_SCRIPTS/$RUN_LAUNCHER" \
    "$DVRK_OPENH_ROOT/JHU/Imerse/Wound_Closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled/meta/stats_cosmos-c3hss-v1.json" \
    "$DVRK_OPENH_ROOT/JHU/Imerse/NephFat_extracted/nephfat/meta/stats_cosmos-c3hss-v1.json" \
    "$DVRK_LZ_ROOT/Surgical/JHU/Imerse/previously_collected_data/hf_suturebot/meta/stats_cosmos-c3hss-v1.json" \
    "$CHECKPOINT/model/.metadata" \
    "$CHECKPOINT/optim/.metadata" \
    "$CHECKPOINT/scheduler/.metadata" \
    "$CHECKPOINT/trainer/.metadata" \
    "$WAN_VAE_PATH" \
    "$CONTAINER_IMAGE" \
    "$DRACO_NVML_PATH/libnvidia-ml.so.1"; do
    if [[ ! -e "$required" ]]; then
        echo "FATAL: missing required path: $required" >&2
        exit 65
    fi
done

mkdir -p "$OUTPUT_ROOT" "$JOB_SCRIPTS" "$SHUVER_ROOT/.cache"

OVERLAY_ROOT=$COOKBOOK_DIR/framework_patch
while IFS= read -r -d '' source_file; do
    relative_path="${source_file#"$OVERLAY_ROOT/"}"
    target_file="$FRAMEWORK_DIR/$relative_path"
    if [[ ! -f "$target_file" ]] || ! cmp -s "$source_file" "$target_file"; then
        echo "FATAL: installed framework does not match CAMP overlay: $relative_path" >&2
        exit 66
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
MOUNTS="$MOUNTS,$SHUVER_ROOT/.cache:/root/.cache/cosmos-eval"

echo "===== MATCHED dVRK ITER-700 EVAL $(date -Is) ====="
echo "rollout_mode=$ROLLOUT_MODE"
echo "checkpoint=$CHECKPOINT"
echo "checkpoint_iter=$CHECKPOINT_ITER"
echo "history_actions=16"
echo "effective_fps=10 raw_timestep_interval=3 start_base_index=48"
echo "episodes=wound_closure:[209,212,214,215] hf_suturebot:[1420,1426,1435] nephfat:[2046,2049,2052]"
echo "output_root=$OUTPUT_ROOT"

srun --no-container-mount-home --container-image="$CONTAINER_IMAGE" \
    bash -c 'echo pyxis-warmup-ok on $(hostname)' || true

srun --ntasks=1 --ntasks-per-node=1 \
    --no-container-mount-home \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace \
    --export=ALL,CHECKPOINT="$CHECKPOINT",CHECKPOINT_ITER="$CHECKPOINT_ITER",OUTPUT_ROOT="$OUTPUT_ROOT",DVRK_OPENH_ROOT="$DVRK_OPENH_ROOT",DVRK_LZ_ROOT="$DVRK_LZ_ROOT",DVRK_GROUPS=all,COSMOS_OPENH_PATH_LAYOUT=draco_internal,COSMOS_OPENH_STATS_POSTFIX=c3hss-v1,COSMOS_OPENH_NUM_HISTORY_ACTIONS=16 \
    bash "/eval/$RUN_LAUNCHER"
