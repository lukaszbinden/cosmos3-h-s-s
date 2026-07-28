#!/usr/bin/env bash
# Container entrypoint for one matched 12.1-second autoregressive rollout.

set -euo pipefail

: "${CHECKPOINT:?launcher must pin a checkpoint}"
: "${CHECKPOINT_ITER:?launcher must provide its iteration}"
: "${EVAL_CONDITION:?launcher must name the comparison arm}"
: "${NUM_HISTORY_ACTIONS:?launcher must set the history row count}"
: "${OUTPUT_ROOT:?launcher must provide the output root}"
: "${DATASET:?launcher must provide the held-out dataset leaf}"

MEMORY_TRACKS_ROOT=${MEMORY_TRACKS_ROOT:-}
EVAL_EPISODE=${EVAL_EPISODE:-2049}
START_BASE_INDEX=${START_BASE_INDEX:-48}
CONDITION_ROOT="$OUTPUT_ROOT/$EVAL_CONDITION"
RUNTIME_ROOT="$CONDITION_ROOT/runtime"
RAW_ROOT="$CONDITION_ROOT/raw"
mkdir -p "$RUNTIME_ROOT" "$RAW_ROOT"

source /workspace/.venv/bin/activate
export HOME=/root
export TMPDIR=/tmp
export LD_LIBRARY_PATH=/opt/nvidia-pinned:/usr/local/cuda/compat
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/compat:${LIBRARY_PATH:-}
export PYTHONPATH=/eval:/cookbook/scripts:/workspace
export WAN_VAE_PATH=/wan_vae/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT="$RUNTIME_ROOT"
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
if [[ -n "$MEMORY_TRACKS_ROOT" ]]; then
    export COSMOS_OPENH_CAMP_MEMORY_TRACKS="$MEMORY_TRACKS_ROOT"
    extra_args+=(--camp-memory-tracks-root "$MEMORY_TRACKS_ROOT")
else
    unset COSMOS_OPENH_CAMP_MEMORY_TRACKS
fi

echo "AR12_START condition=$EVAL_CONDITION checkpoint_iter=$CHECKPOINT_ITER time=$(date -Is)"
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    /eval/eval_c3hss_long_horizon.py \
    --sft-toml=/cookbook/toml/sft_config/action_mixed_open_h_sft_nano.toml \
    --checkpoint="$CHECKPOINT" \
    --dataset="$DATASET" \
    --output-dir="$RAW_ROOT" \
    --episodes "$EVAL_EPISODE" \
    --embodiment=jhu_dvrk_mono \
    --data-split=full \
    --test-split-ratio=0.05 \
    --timestep-interval=3 \
    --start-base-index="$START_BASE_INDEX" \
    --iteration="$CHECKPOINT_ITER" \
    --max-chunks=10 \
    --seed=0 \
    --guidance=1.5 \
    --num-sampling-step=16 \
    --num-history-actions="$NUM_HISTORY_ACTIONS" \
    --fps=10 \
    --rollout-conditioning=autoregressive \
    "${extra_args[@]}"

python - "$RAW_ROOT/c3hss_results.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    payload = json.load(handle)
result = payload["results"][0]
if result["num_frames"] != 121:
    raise SystemExit(f"expected 121 frames, got {result['num_frames']}")
if payload["rollout_conditioning"] != "autoregressive":
    raise SystemExit("rollout was not autoregressive")
print(
    "AR12_DONE",
    f"checkpoint={payload['checkpoint']}",
    f"frames={result['num_frames']}",
    f"duration_seconds={result['num_frames'] / payload['fps']:.1f}",
    f"mean_l1={result['fds']['mean_l1']:.6f}",
)
PY
