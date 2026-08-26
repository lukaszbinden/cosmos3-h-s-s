#!/bin/bash
# Build the two held-out brackets and fail before any GPU allocation if episode
# identities, split membership, or pinned evaluator inputs have drifted.
#SBATCH --job-name=cosmos3-cross-owner-cpu-gate
#SBATCH --account=healthcareeng_holoscan
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

set -euo pipefail

EOSU=/lustre/fsw/healthcareeng_holoscan/user_data/shuver
OURS=${OURS:-$EOSU/projects/cosmos3-h-s-s}
OUT=${OUT:-$EOSU/outputs/cosmos3_lukas_vs_justin_v1}
J=/lustre/fsw/healthcareeng_holoscan/user_data/justchen/projects/cosmos3-internal
L=/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/git/cosmos3-h-s-s
JX=/lustre/fsw/healthcareeng_holoscan/user_data/justchen/outputs/crossckpt_eval
OPENH=/lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment/Surgical
JCKPT=/lustre/fsw/healthcareeng_holoscan/user_data/justchen/outputs/exp_cmr_clinical_train/cosmos3_surgical/action_exp232_mix/action_exp232_mix_v3_latentcache256/checkpoints/iter_000014280
LCKPT=/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/imaginaire/output/cosmos3_action_surgical/action_open_h/mm-C3-H-S-S-base/checkpoints/iter_000026500
CONTAINER='docker://nvcr.io#nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04'
MANIFEST=$J/scripts/fds_lzb_manifest.tsv

mkdir -p "$OUT/_spans" "$OUT/targets/justin_heldout" "$OUT/targets/lukas_heldout" \
  "$OUT/manifests" "$OURS/slurm_logs"

MOUNTS="$OURS:$OURS:ro,$OUT:$OUT,$J:$J:ro,$L:$L:ro,$OPENH:$OPENH:ro,/lustre/fsw/healthcareeng_holoscan/user_data/justchen:/lustre/fsw/healthcareeng_holoscan/user_data/justchen:ro,/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden:/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden:ro,$JCKPT:$JCKPT:ro,$LCKPT:$LCKPT:ro"

srun --container-image="$CONTAINER" --container-mounts="$MOUNTS" --container-workdir="$OURS" \
  --export=ALL,OURS="$OURS",OUT="$OUT",J="$J",L="$L",JX="$JX",OPENH="$OPENH",MANIFEST="$MANIFEST",JCKPT="$JCKPT",LCKPT="$LCKPT" \
  bash -lc '
    set -euo pipefail
    sha256sum -c <<EOF
e9e81810e21678c261ea2683c07031c1611f85d1a07a9ae36c06a7e982fe8f5f  $JX/_configs/ours_14280.yaml
021a860819714772d1a73c4dbbd6151fd4746f98cf2591494881446c0259e4c3  $J/scripts/build_episode_targets.py
7f2ab57fa900ba17df6268b5fcdb0baf836176ea63fa3f3f97e1873cf105dc46  $MANIFEST
a2cc413b98371d2bc887bdf130bfecf52b9570f75a24ec248f2c7ab3ed3a20e0  $JCKPT/model/.metadata
57c66852a462af11d3e76bfffdc38a47599254c63dfcd2f10f6b031f0eb12ef2  $LCKPT/model/.metadata
d0b86e3ffa7348ff9a9331e7f2402e9a37f9142296d2be04adc2f4babbf5de20  /lustre/fsw/healthcareeng_holoscan/user_data/justchen/outputs/exp_cmr_clinical_train/cosmos3_surgical/action_exp232_mix/action_exp232_mix_v3_latentcache256/config.yaml
a5f3764164829adac07ca8677df20f48611f652baa73d6c6441c173a29c17a64  /lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/imaginaire/output/cosmos3_action_surgical/action_open_h/mm-C3-H-S-S-base/config.yaml
EOF
    source "$J/.venv/bin/activate"
    export LD_LIBRARY_PATH= PYTHONPATH="$J" COSMOS_TRAINING=1
    export COSMOS_EPISODE_ROOT_ALIASES="{\"/lustre/fsw/healthcareeng_holoscan/user_data/justchen/datasets/openh_v30\":\"OPENH\",\"$OPENH\":\"OPENH\",\"$OPENH/cmr_surgical\":\"CMR\",\"/lustre/fsw/healthcareeng_holoscan/user_data/justchen/datasets/cmr_clinical_v30\":\"CMR\"}"
    python "$J/scripts/build_episode_targets.py" dump \
      --config-file "$JX/_configs/ours_14280.yaml" --label justin_val \
      --output-dir "$OUT/_spans" --split val --datasets cmr_clinical dvrk_jhu
    python "$J/scripts/build_episode_targets.py" dump \
      --config-file "$JX/_configs/ours_14280.yaml" --label justin_train \
      --output-dir "$OUT/_spans" --split train --datasets cmr_clinical dvrk_jhu
    deactivate || true

    source "$L/packages/cosmos3/.venv/bin/activate"
    export LD_LIBRARY_PATH= PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="$L/packages/cosmos3:$L/cookbooks/cosmos3/generator/action/finetune/scripts"
    python "$OURS/scripts/cross_owner/dump_lukas_spans.py" \
      --manifest "$MANIFEST" --label lukas_train --output "$OUT/_spans/lukas_train.json" \
      --split train --test-split-ratio 0.02 --families cmr_clinical dvrk_jhu
    python "$OURS/scripts/cross_owner/dump_lukas_spans.py" \
      --manifest "$MANIFEST" --label lukas_test --output "$OUT/_spans/lukas_test.json" \
      --split test --test-split-ratio 0.02 --families cmr_clinical dvrk_jhu
    deactivate || true

    source "$J/.venv/bin/activate"
    export LD_LIBRARY_PATH= PYTHONPATH="$J" COSMOS_TRAINING=1
    python "$J/scripts/build_episode_targets.py" intersect \
      --spans "$OUT/_spans/justin_val.json" --spans "$OUT/_spans/lukas_train.json" \
      --episodes-per-family 1 --output-dir "$OUT/targets/justin_heldout"
    python "$J/scripts/build_episode_targets.py" intersect \
      --spans "$OUT/_spans/justin_train.json" --spans "$OUT/_spans/lukas_test.json" \
      --episodes-per-family 1 --output-dir "$OUT/targets/lukas_heldout"
    python "$J/scripts/build_episode_targets.py" verify \
      --config-file "$JX/_configs/ours_14280.yaml" --targets-dir "$OUT/targets/justin_heldout" \
      --label justin_heldout --split val --full-split-families
    python "$J/scripts/build_episode_targets.py" verify \
      --config-file "$JX/_configs/ours_14280.yaml" --targets-dir "$OUT/targets/lukas_heldout" \
      --label lukas_heldout --split train --full-split-families
    python "$J/scripts/build_episode_targets.py" lzb-manifest \
      --targets-dir "$OUT/targets/justin_heldout" --output "$OUT/manifests/justin_heldout.tsv" \
      --cmr-root "$OPENH/cmr_surgical"
    python "$J/scripts/build_episode_targets.py" lzb-manifest \
      --targets-dir "$OUT/targets/lukas_heldout" --output "$OUT/manifests/lukas_heldout.tsv" \
      --cmr-root "$OPENH/cmr_surgical"
    test "$(wc -l < "$OUT/manifests/justin_heldout.tsv")" -eq 2
    test "$(wc -l < "$OUT/manifests/lukas_heldout.tsv")" -eq 2
    cp "$OURS/scripts/cross_owner/checkpoint_manifest.json" "$OUT/checkpoint_manifest.json"
    echo "CPU_GATE_PASS"
  '
