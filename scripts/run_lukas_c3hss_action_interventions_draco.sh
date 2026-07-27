#!/usr/bin/env bash
# Container entrypoint for the paired one-chunk dVRK action intervention grid.

set -euo pipefail

: "${CHECKPOINT:?launcher must pin a complete CAMP-lite checkpoint}"
: "${CHECKPOINT_ITER:?launcher must provide the fine-tune iteration}"
: "${OUTPUT_ROOT:?launcher must provide an output root}"
: "${DVRK_OPENH_ROOT:?launcher must provide the DRACO Open-H root}"
: "${DVRK_LZ_ROOT:?launcher must provide the Lucas Open-H-lz root}"

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
    local dataset=$3
    shift 3
    local out=$OUTPUT_ROOT/raw/$subset
    local runtime=$RUNTIME_ROOT/$subset
    local log=$OUTPUT_ROOT/logs/$subset.log
    mkdir -p "$out" "$runtime" "$(dirname "$log")"
    echo "START subset=$subset gpu=$gpu episodes=$* time=$(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        export IMAGINAIRE_OUTPUT_ROOT=$runtime
        torchrun --standalone --nnodes=1 --nproc-per-node=1 \
            /eval/eval_c3hss_action_interventions.py \
            --sft-toml=/cookbook/toml/sft_config/action_mixed_open_h_sft_nano.toml \
            --checkpoint="$CHECKPOINT" \
            --dataset="$dataset" \
            --output-dir="$out" \
            --episodes "$@" \
            --embodiment=jhu_dvrk_mono \
            --data-split=test \
            --test-split-ratio=0.05 \
            --timestep-interval=3 \
            --start-base-index=48 \
            --iteration "$CHECKPOINT_ITER" \
            --seed 0 \
            --guidance 1.5 \
            --num-sampling-step 16 \
            --num-history-actions 16 \
            --fps 10
    ) >"$log" 2>&1
    echo "DONE subset=$subset gpu=$gpu time=$(date -Is)"
}

pids=()
run_group 0 wound_closure \
    "$DVRK_OPENH_ROOT/JHU/Imerse/Wound_Closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled" \
    209 212 214 215 &
pids+=("$!")
run_group 1 hf_suturebot \
    "$DVRK_LZ_ROOT/Surgical/JHU/Imerse/previously_collected_data/hf_suturebot" \
    1420 1426 1435 &
pids+=("$!")
run_group 2 nephfat \
    "$DVRK_OPENH_ROOT/JHU/Imerse/NephFat_extracted/nephfat" \
    2046 2049 2052 &
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
    echo "FATAL: at least one action-intervention group failed" >&2
    exit "$status"
fi

echo "===== dVRK ACTION INTERVENTIONS COMPLETE $(date -Is) ====="
find "$OUTPUT_ROOT/raw" -maxdepth 2 -type f -printf '%P %s bytes\n' | sort
