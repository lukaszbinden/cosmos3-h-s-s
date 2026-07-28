#!/usr/bin/env bash
# Container entrypoint for one condition of the matched 10-episode,
# 12.1-second autoregressive CAMP sweep.

set -euo pipefail

: "${CHECKPOINT:?launcher must pin a checkpoint}"
: "${CHECKPOINT_ITER:?launcher must provide its iteration}"
: "${EVAL_CONDITION:?launcher must name the comparison condition}"
: "${NUM_HISTORY_ACTIONS:?launcher must set the history row count}"
: "${OUTPUT_ROOT:?launcher must provide the output root}"
: "${DVRK_OPENH_ROOT:?launcher must provide the canonical Open-H root}"
: "${DVRK_LZ_ROOT:?launcher must provide the Lucas Open-H root}"

HISTORY_ABLATION=${HISTORY_ABLATION:-}
MEMORY_ABLATION=${MEMORY_ABLATION:-}
MEMORY_TRACKS_ROOT=${MEMORY_TRACKS_ROOT:-}
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

extra_args=()
if [[ -n "$HISTORY_ABLATION" ]]; then
    export COSMOS_OPENH_HISTORY_ABLATION="$HISTORY_ABLATION"
    extra_args+=(--history-ablation "$HISTORY_ABLATION")
else
    unset COSMOS_OPENH_HISTORY_ABLATION
fi
if [[ -n "$MEMORY_TRACKS_ROOT" ]]; then
    export COSMOS_OPENH_CAMP_MEMORY_TRACKS="$MEMORY_TRACKS_ROOT"
    extra_args+=(--camp-memory-tracks-root "$MEMORY_TRACKS_ROOT")
else
    unset COSMOS_OPENH_CAMP_MEMORY_TRACKS
fi
if [[ -n "$MEMORY_ABLATION" ]]; then
    export COSMOS_OPENH_MEMORY_ABLATION="$MEMORY_ABLATION"
    extra_args+=(--camp-memory-ablation "$MEMORY_ABLATION")
else
    unset COSMOS_OPENH_MEMORY_ABLATION
fi

run_group() {
    local gpu=$1
    local subset=$2
    local dataset=$3
    shift 3
    local out="$CONDITION_ROOT/raw/$subset"
    local runtime="$RUNTIME_ROOT/$subset"
    local group_log="$CONDITION_ROOT/${subset}.log"
    mkdir -p "$out" "$runtime"
    echo "AR12_GROUP_START condition=$EVAL_CONDITION subset=$subset gpu=$gpu episodes=$* time=$(date -Is)"
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        export IMAGINAIRE_OUTPUT_ROOT=$runtime
        torchrun --standalone --nnodes=1 --nproc-per-node=1 \
            /eval/eval_c3hss_long_horizon.py \
            --sft-toml=/cookbook/toml/sft_config/action_mixed_open_h_sft_nano.toml \
            --checkpoint="$CHECKPOINT" \
            --dataset="$dataset" \
            --output-dir="$out" \
            --episodes "$@" \
            --embodiment=jhu_dvrk_mono \
            --data-split=full \
            --test-split-ratio=0.05 \
            --timestep-interval=3 \
            --start-base-index=48 \
            --iteration="$CHECKPOINT_ITER" \
            --max-chunks=10 \
            --seed=0 \
            --guidance=1.5 \
            --num-sampling-step=16 \
            --num-history-actions="$NUM_HISTORY_ACTIONS" \
            --fps=10 \
            --rollout-conditioning=autoregressive \
            "${extra_args[@]}"
    ) >"$group_log" 2>&1
    echo "AR12_GROUP_DONE condition=$EVAL_CONDITION subset=$subset gpu=$gpu time=$(date -Is)"
}

run_group 0 wound_closure \
    "$DVRK_OPENH_ROOT/JHU/Imerse/Wound_Closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled" \
    209 212 214 215 &
pids=("$!")
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

for group_log in "$CONDITION_ROOT"/*.log; do
    echo "--- $group_log"
    grep -E "C3HSS episode=.*frames=|Traceback|RuntimeError|FATAL|ChildFailedError" \
        "$group_log" || true
done
if [[ "$status" -ne 0 ]]; then
    echo "FATAL: at least one dataset group failed" >&2
    exit "$status"
fi

python - "$CONDITION_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
records = []
for result_path in sorted(root.glob("raw/*/c3hss_results.json")):
    with result_path.open() as handle:
        payload = json.load(handle)
    if payload["rollout_conditioning"] != "autoregressive":
        raise SystemExit(f"{result_path} was not autoregressive")
    records.extend(payload["results"])
if len(records) != 10:
    raise SystemExit(f"expected 10 results, found {len(records)}")
bad = [record["episode_id"] for record in records if record["num_frames"] != 121]
if bad:
    raise SystemExit(f"episodes with non-121 frame rollouts: {bad}")
print(f"AR12_CONDITION_DONE results={len(records)} frames_each=121")
PY

