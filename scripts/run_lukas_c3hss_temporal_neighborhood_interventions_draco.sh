#!/usr/bin/env bash
# Container entrypoint for temporal PSM1 probes around two dVRK failures.

set -euo pipefail

: "${CHECKPOINT_ROOT:?launcher must provide the CAMP-lite checkpoint root}"
: "${CHECKPOINT_ITER:?launcher must provide the fine-tune iteration}"
: "${OUTPUT_ROOT:?launcher must provide an output root}"
: "${DVRK_LZ_ROOT:?launcher must provide the Lucas Open-H-lz root}"

SEEDS=${SEEDS:-"0 1 2 3 4"}
DATA_SEED=${DATA_SEED:-1729}
PHYSICAL_COMPONENTS=${PHYSICAL_COMPONENTS:-"tx ty tz"}
read -r -a physical_components <<<"$PHYSICAL_COMPONENTS"

CHECKPOINT="$CHECKPOINT_ROOT/iter_$(printf '%09d' "$CHECKPOINT_ITER")"
CHECKPOINT_TAG="iter_$(printf '%09d' "$CHECKPOINT_ITER")"
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
    shift
    local windows=("$@")
    local dataset="$DVRK_LZ_ROOT/Surgical/JHU/Imerse/previously_collected_data/hf_suturebot"
    local out="$OUTPUT_ROOT/$CHECKPOINT_TAG/raw/hf_suturebot/psm1"
    local runtime="$RUNTIME_ROOT/$CHECKPOINT_TAG/gpu$gpu"
    mkdir -p "$out" "$runtime" "$OUTPUT_ROOT/$CHECKPOINT_TAG/logs"
    echo "GROUP_START gpu=$gpu checkpoint=$CHECKPOINT windows=${windows[*]} time=$(date -Is)"
    for seed in $SEEDS; do
        echo "SEED_START gpu=$gpu seed=$seed time=$(date -Is)"
        (
            export CUDA_VISIBLE_DEVICES=$gpu
            export IMAGINAIRE_OUTPUT_ROOT=$runtime
            torchrun --standalone --nnodes=1 --nproc-per-node=1 \
                /eval/eval_c3hss_action_interventions.py \
                --sft-toml=/cookbook/toml/sft_config/action_mixed_open_h_sft_nano.toml \
                --checkpoint="$CHECKPOINT" \
                --dataset="$dataset" \
                --output-dir="$out" \
                --episode-windows "${windows[@]}" \
                --embodiment=jhu_dvrk_mono \
                --data-split=full \
                --test-split-ratio=0.05 \
                --timestep-interval=3 \
                --iteration "$CHECKPOINT_ITER" \
                --seed "$seed" \
                --data-seed "$DATA_SEED" \
                --guidance 1.5 \
                --num-sampling-step 16 \
                --num-history-actions 16 \
                --fps 10 \
                --variant-set=physical_axes \
                --physical-anchor-mode=first_row \
                --physical-intervention-arms=psm1 \
                --physical-intervention-components "${physical_components[@]}"
        )
        echo "SEED_DONE gpu=$gpu seed=$seed time=$(date -Is)"
    done
    echo "GROUP_DONE gpu=$gpu checkpoint=$CHECKPOINT time=$(date -Is)"
}

# Each N=12 window spans 36 raw frames at the dVRK interval of 3. Consecutive
# centers are separated by 36, so action targets do not overlap.
pids=()
run_group 0 141:144 141:180 141:216 >"$OUTPUT_ROOT/gpu0.log" 2>&1 &
pids+=("$!")
run_group 1 141:252 141:288 >"$OUTPUT_ROOT/gpu1.log" 2>&1 &
pids+=("$!")
run_group 2 1382:309 1382:345 1382:381 >"$OUTPUT_ROOT/gpu2.log" 2>&1 &
pids+=("$!")
run_group 3 1382:417 1382:453 >"$OUTPUT_ROOT/gpu3.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

echo "===== GROUP STATUS ====="
for log in "$OUTPUT_ROOT"/gpu*.log; do
    echo "--- $log"
    grep -E "ACTION_INTERVENTION episode=.*variant=|Traceback|RuntimeError|ChildFailedError" "$log" || true
done
if [[ "$status" -ne 0 ]]; then
    echo "FATAL: at least one temporal-neighborhood group failed" >&2
    exit "$status"
fi

echo "===== TEMPORAL-NEIGHBORHOOD INTERVENTIONS COMPLETE $(date -Is) ====="
find "$OUTPUT_ROOT" -type f -printf '%P %s bytes\n' | sort
