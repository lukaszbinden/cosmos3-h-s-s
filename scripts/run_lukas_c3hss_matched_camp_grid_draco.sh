#!/usr/bin/env bash
# Container entrypoint for the matched Arm-A/Arm-B/Arm-C CAMP evaluation grid.

set -euo pipefail

: "${CHECKPOINT:?launcher must pin a complete checkpoint}"
: "${CHECKPOINT_ITER:?launcher must provide the checkpoint iteration}"
: "${COMPARISON_STEP:?launcher must provide the matched comparison step}"
: "${EVAL_CONDITION:?launcher must name the evaluation condition}"
: "${MODEL_LABEL:?launcher must label the model arm}"
: "${NUM_HISTORY_ACTIONS:?launcher must set the history row count}"
: "${OUTPUT_ROOT:?launcher must provide an output root}"
: "${DVRK_OPENH_ROOT:?launcher must provide the DRACO Open-H root}"
: "${DVRK_LZ_ROOT:?launcher must provide the Lucas Open-H-lz root}"

SEEDS=${SEEDS:-"0 1 2 3 4"}
DATA_SEED=${DATA_SEED:-1729}
PHYSICAL_COMPONENTS=${PHYSICAL_COMPONENTS:-"tx ty tz"}
HISTORY_ABLATION=${HISTORY_ABLATION:-}
MEMORY_ABLATION=${MEMORY_ABLATION:-}
MEMORY_TRACKS_ROOT=${MEMORY_TRACKS_ROOT:-}
read -r -a physical_components <<<"$PHYSICAL_COMPONENTS"

CONDITION_ROOT="$OUTPUT_ROOT/$EVAL_CONDITION"
RUNTIME_ROOT="$CONDITION_ROOT/runtime"
mkdir -p "$CONDITION_ROOT" "$RUNTIME_ROOT"

source /workspace/.venv/bin/activate
export HOME=/root
export TMPDIR=/tmp
export LD_LIBRARY_PATH=/opt/nvidia-pinned:/usr/local/cuda/compat
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/compat:${LIBRARY_PATH:-}
export PYTHONPATH=/eval:/cookbook/scripts:/workspace
export WAN_VAE_PATH=/wan_vae/Wan2.2_VAE.pth
export COSMOS_OPENH_NUM_HISTORY_ACTIONS="$NUM_HISTORY_ACTIONS"
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

if [[ -n "$HISTORY_ABLATION" ]]; then
    export COSMOS_OPENH_HISTORY_ABLATION="$HISTORY_ABLATION"
else
    unset COSMOS_OPENH_HISTORY_ABLATION
fi
if [[ -n "$MEMORY_TRACKS_ROOT" ]]; then
    export COSMOS_OPENH_CAMP_MEMORY_TRACKS="$MEMORY_TRACKS_ROOT"
else
    unset COSMOS_OPENH_CAMP_MEMORY_TRACKS
fi
if [[ -n "$MEMORY_ABLATION" ]]; then
    export COSMOS_OPENH_MEMORY_ABLATION="$MEMORY_ABLATION"
else
    unset COSMOS_OPENH_MEMORY_ABLATION
fi

extra_args=(--num-history-actions "$NUM_HISTORY_ACTIONS")
if [[ -n "$HISTORY_ABLATION" ]]; then
    extra_args+=(--history-ablation "$HISTORY_ABLATION")
fi
if [[ -n "$MEMORY_TRACKS_ROOT" ]]; then
    extra_args+=(--camp-memory-tracks-root "$MEMORY_TRACKS_ROOT")
fi
if [[ -n "$MEMORY_ABLATION" ]]; then
    extra_args+=(--camp-memory-ablation "$MEMORY_ABLATION")
fi

run_group() {
    local gpu=$1
    local subset=$2
    local target_arm=$3
    local dataset=$4
    shift 4
    local out="$CONDITION_ROOT/raw/$subset/$target_arm"
    local runtime="$RUNTIME_ROOT/$subset/$target_arm"
    local log="$CONDITION_ROOT/logs/${subset}_${target_arm}.log"
    mkdir -p "$out" "$runtime" "$(dirname "$log")"
    echo "GROUP_START condition=$EVAL_CONDITION gpu=$gpu subset=$subset target=$target_arm windows=$* time=$(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        export IMAGINAIRE_OUTPUT_ROOT=$runtime
        for seed in $SEEDS; do
            echo "SEED_START condition=$EVAL_CONDITION subset=$subset target=$target_arm seed=$seed time=$(date -Is)"
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
                --comparison-step "$COMPARISON_STEP" \
                --eval-condition "$EVAL_CONDITION" \
                --model-label "$MODEL_LABEL" \
                --seed "$seed" \
                --data-seed "$DATA_SEED" \
                --guidance 1.5 \
                --num-sampling-step 16 \
                --fps 10 \
                --variant-set=physical_axes \
                --physical-anchor-mode=first_row \
                --physical-intervention-arms="$target_arm" \
                --physical-intervention-components "${physical_components[@]}" \
                "${extra_args[@]}"
            echo "SEED_DONE condition=$EVAL_CONDITION subset=$subset target=$target_arm seed=$seed time=$(date -Is)"
        done
    ) >"$log" 2>&1
    echo "GROUP_DONE condition=$EVAL_CONDITION subset=$subset target=$target_arm gpu=$gpu time=$(date -Is)"
}

pids=()
if [[ "${MATCHED_SMOKE:-0}" == "1" ]]; then
    run_group 0 hf_suturebot psm1 \
        "$DVRK_LZ_ROOT/Surgical/JHU/Imerse/previously_collected_data/hf_suturebot" \
        1401:246 &
    pids+=("$!")
    run_group 1 hf_suturebot psm2 \
        "$DVRK_LZ_ROOT/Surgical/JHU/Imerse/previously_collected_data/hf_suturebot" \
        1361:246 &
    pids+=("$!")
    run_group 2 nephfat psm1 \
        "$DVRK_OPENH_ROOT/JHU/Imerse/NephFat_extracted/nephfat" \
        2091:63 &
    pids+=("$!")
    run_group 3 nephfat psm2 \
        "$DVRK_OPENH_ROOT/JHU/Imerse/NephFat_extracted/nephfat" \
        1914:240 &
    pids+=("$!")
else
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
fi

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

echo "===== CONDITION STATUS: $EVAL_CONDITION ====="
for log in "$CONDITION_ROOT"/logs/*.log; do
    echo "--- $log"
    grep -E "ACTION_INTERVENTION episode=.*variant=|Traceback|RuntimeError|ChildFailedError" "$log" || true
done
if [[ "$status" -ne 0 ]]; then
    echo "FATAL: at least one group failed for $EVAL_CONDITION" >&2
    exit "$status"
fi

echo "===== MATCHED CAMP CONDITION COMPLETE: $EVAL_CONDITION $(date -Is) ====="
find "$CONDITION_ROOT/raw" -type f -printf '%P %s bytes\n' | sort
