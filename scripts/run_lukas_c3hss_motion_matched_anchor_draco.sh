#!/usr/bin/env bash
# Container entrypoint for motion-matched, first-row-anchored dVRK probes.

set -euo pipefail

: "${CHECKPOINT:?launcher must pin a complete CAMP-lite checkpoint}"
: "${CHECKPOINT_ITER:?launcher must provide the fine-tune iteration}"
: "${OUTPUT_ROOT:?launcher must provide an output root}"
: "${DVRK_OPENH_ROOT:?launcher must provide the DRACO Open-H root}"
: "${DVRK_LZ_ROOT:?launcher must provide the Lucas Open-H-lz root}"

SEEDS=${SEEDS:-0}
PHYSICAL_COMPONENTS=${PHYSICAL_COMPONENTS:-"tx ty tz rx ry rz jaw"}
read -r -a physical_components <<<"$PHYSICAL_COMPONENTS"

RUNTIME_ROOT="$OUTPUT_ROOT/runtime"
mkdir -p "$OUTPUT_ROOT" "$RUNTIME_ROOT"

source /workspace/.venv/bin/activate
export HOME=/root
export TMPDIR=/tmp
export LD_LIBRARY_PATH=/opt/nvidia-pinned:/usr/local/cuda/compat
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/compat:${LIBRARY_PATH:-}
export PYTHONPATH=/eval:/cookbook/scripts:/workspace
export WAN_VAE_PATH=/wan_vae/Wan2.2_VAE.pth
export COSMOS_OPENH_NUM_HISTORY_ACTIONS=16
export COSMOS_OPENH_MIXED_EXPERIMENT_WITH_CAMP=1
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_TOKEN_PATH=/tmp/cosmos_eval_no_hf_token
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6

run_group() {
    local gpu=$1
    local subset=$2
    local target_arm=$3
    local dataset=$4
    shift 4
    local out=$OUTPUT_ROOT/raw/$subset/$target_arm
    local runtime=$RUNTIME_ROOT/$subset/$target_arm
    local log=$OUTPUT_ROOT/logs/${subset}_${target_arm}.log
    mkdir -p "$out" "$runtime" "$(dirname "$log")"
    echo "START subset=$subset target=$target_arm gpu=$gpu windows=$* time=$(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        export IMAGINAIRE_OUTPUT_ROOT=$runtime
        # Episode IDs are exclusively from meta/info.json's official test split.
        # Cosmos data_split=test instead re-samples 5% of individual steps, so
        # use full only to make those pinned episode/base pairs addressable.
        for seed in $SEEDS; do
            echo "SEED_START subset=$subset target=$target_arm seed=$seed time=$(date -Is)"
            torchrun --standalone --nnodes=1 --nproc-per-node=1 \
                /eval/eval_c3hss_action_interventions.py \
                --sft-toml=/cookbook/toml/sft_config/action_mixed_open_h_sft_nano.toml \
                --checkpoint="$CHECKPOINT" \
                --dataset="$dataset" \
                --output-dir="$out" \
                --episode-windows "$@" \
                --embodiment=jhu_dvrk_mono \
                --data-split=full \
                --test-split-ratio=0.05 \
                --timestep-interval=3 \
                --iteration "$CHECKPOINT_ITER" \
                --seed "$seed" \
                --guidance 1.5 \
                --num-sampling-step 16 \
                --num-history-actions 16 \
                --fps 10 \
                --variant-set=physical_axes \
                --physical-anchor-mode=first_row \
                --physical-intervention-arms="$target_arm" \
                --physical-intervention-components "${physical_components[@]}"
            echo "SEED_DONE subset=$subset target=$target_arm seed=$seed time=$(date -Is)"
        done
    ) >"$log" 2>&1
    echo "DONE subset=$subset target=$target_arm gpu=$gpu time=$(date -Is)"
}

pids=()
run_group 0 hf_suturebot psm1 \
    "$DVRK_LZ_ROOT/Surgical/JHU/Imerse/previously_collected_data/hf_suturebot" \
    1401:246 1445:159 1382:381 &
pids+=("$!")
run_group 1 hf_suturebot psm2 \
    "$DVRK_LZ_ROOT/Surgical/JHU/Imerse/previously_collected_data/hf_suturebot" \
    1361:246 1314:60 1318:108 &
pids+=("$!")
run_group 2 nephfat psm1 \
    "$DVRK_OPENH_ROOT/JHU/Imerse/NephFat_extracted/nephfat" \
    2091:63 1954:108 2106:129 &
pids+=("$!")
run_group 3 nephfat psm2 \
    "$DVRK_OPENH_ROOT/JHU/Imerse/NephFat_extracted/nephfat" \
    1914:240 1908:186 1906:315 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

echo "===== GROUP STATUS ====="
for log in "$OUTPUT_ROOT"/logs/*.log; do
    echo "--- $log"
    grep -E "ACTION_INTERVENTION episode=.*variant=|Traceback|RuntimeError|ChildFailedError" "$log" || true
done
if [[ "$status" -ne 0 ]]; then
    echo "FATAL: at least one motion-matched intervention group failed" >&2
    exit "$status"
fi

echo "===== MOTION-MATCHED ANCHOR PROBES COMPLETE $(date -Is) ====="
find "$OUTPUT_ROOT/raw" -type f -printf '%P %s bytes\n' | sort
