# EOS Lukas-versus-Justin checkpoint comparison

This harness compares the latest durable checkpoints without pretending the two
training recipes are the same model. The primary endpoint is forward-dynamics
quality on identical physical clips. Raw policy/IDM action errors are not a valid
cross-owner scalar comparison yet: Lukas uses a 44-D, 13-frame action schema and
Justin uses a 64-D, 24-frame schema with different pose normalization. Those
outputs remain diagnostic until both are decoded into the same physical action
coordinates.

## Frozen primary checkpoints

The full paths, metadata/config hashes, workspace revisions, and evaluator hashes
are in `scripts/cross_owner/checkpoint_manifest.json`. Jobs fail closed if a pinned
config or evaluator has changed.

- Lukas: `mm-C3-H-S-S-base`, iteration 26,500, native 13-frame/44-D recipe.
- Justin: `action_exp232_mix_v3_latentcache256`, iteration 14,280, native
  24-frame/64-D recipe. Its evaluation config disables the latent cache because an
  autoregressive rollout must feed decoded pixels back into the next window.

## Split design

There is no common held-out episode set. Justin validation is the leading 5% and
Lukas test is the trailing 2%, so the comparison reports two brackets rather than
averaging them:

1. `justin_heldout`: Justin `val` versus Lukas `train`.
2. `lukas_heldout`: Justin `train` versus Lukas `test`.

For every target, Justin's evaluator first chooses the highest-motion valid window
and records its actual source-row start. `build_lzb_start_offsets.py` converts that
row to Lukas's stride grid, and the Lukas evaluator is forced to start there. This
is the alignment proof; episode ID alone is insufficient.

## Pilot gate

The pilot uses one CMR and one JHU episode in each split bracket, seed 0, two
autoregressive chunks (4.8 seconds), 16 diffusion steps, guidance 1.5, and a neutral
320x192 scoring grid. It runs in four logical cells: two owners by two split
brackets. Each cell contains both dataset families.

```bash
EOSU=/lustre/fsw/healthcareeng_holoscan/user_data/shuver
cd $EOSU/projects/cosmos3-h-s-s

gate=$(sbatch --parsable scripts/cross_owner/cross_owner_cpu_gate_eos.sbatch)
justin=$(sbatch --parsable --dependency=afterok:$gate --array=1-4%4 \
  scripts/cross_owner/cross_owner_justin_pilot_eos.sbatch)
offsets=$(sbatch --parsable --dependency=afterok:$justin \
  scripts/cross_owner/cross_owner_build_offsets_eos.sbatch)
lukas=$(sbatch --parsable --dependency=afterok:$offsets --array=1-2%2 \
  scripts/cross_owner/cross_owner_lukas_pilot_eos.sbatch)
echo "gate=$gate justin=$justin offsets=$offsets lukas=$lukas"
```

Do not launch the full matrix until all pilot jobs exit zero, both target manifests
contain exactly CMR and JHU, both offset files contain two episodes, every rollout
has exactly two chunks, and the reported source row/ground-truth first frame agrees
between owners. Only then expand to 10 episodes per family and three seeds for
single-window metrics, followed by six-chunk (14.4-second) autoregressive rollouts
on five episodes per family.
