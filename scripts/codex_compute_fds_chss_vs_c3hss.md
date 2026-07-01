# Your task:
Evaluate C-H-S-S vs. C3-H-S-S including side-by-side comparison videos (3 videos: left is ground truth, center is C-H-S-S roll-out and right is C3-H-S-S roll-out) and an FDS plot comparing C-H-S-S vs C3-H-S-S on a long roll-out horizon (at least 100 frames) on at least 7 test episodes. You perform all these tasks on the EOS cluster, that has H100 GPUs.

# Cosmos3-H-Surigcal-Simulator (C3-H-S-S) pointers:
## git repo / code:
/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/git/cosmos3-h-s-s

## checkpoint:
/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/imaginaire/output/cosmos3_action_surgical/action_open_h/action_fdm_open_h_sft_nano/checkpoints/iter_000008000
(notice: not a .pt file, but a distributed checkpoint, you mave have to convert it first)

# Cosmos-H-Surgical-Simulator (C-H-S-S) pointers:
## git repo / code:
/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/git/cosmos-h-surgical-simulator

## checkpoint:
/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/checkpoints/Cosmos-H-Surgical-Simulator/checkpoints/iter_000012000-v2


# notes
because the two models use different containers, you probably need to run them in separate slurm jobs. 

# you must implement the following at all times:
- never ever delete any file except under /tmp where you created any temporary files

# FDS calculation and ground truth-based roll-out (video) generation
## for C-H-S-S
there is script ./scripts/cosmos_h_surgical_simulator_quant_eval.py for FDS calculation and roll-out video generation based on a ground truth videos/kinematics. You'll find the implementation for FDS (frame decay score) also in that script.
A sample slurm script for C-H-S-S is here: 
/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/git/cosmos-h-surgical-simulator/scripts/run_cosmos_h_quant_eval_cmr_only_eos.sh
important: the script has also code for computing GATC and TCD metrics, but don't compute them as they require an installed Med-SAM3 model which you have not. You can create an adapted version of that script to only compute FDS and generate videos.
note: in order to run the slurm job successfully you may have to install the environment according to the repo's instructions in the README.md, which will includea .venv.

## For C3-H-S-S 
there is a slurm script 
cookbooks/cosmos3/generator/action/finetune/scripts/slurm_eval_checkpoint.sbatch
and
cookbooks/cosmos3/generator/action/finetune/scripts/generate_checkpoint_samples_openh.py
you may have to write a new script similar to cosmos_h_surgical_simulator_quant_eval.py to compute both FDS and the video roll-outs. 
A sample slurm script for C3-H-S-S is here: 
cookbooks/cosmos3/generator/action/finetune/scripts/slurm_eval_checkpoint.sbatch


# roll-out horizon
take at least 100 frames (i.e. 8 or 9 chunks, each 12 frames) for the long-horizon FDS comparison, and at least 7 epsiodes from the dataloader test split. Ensure the 7 episodes are identical for both models.

# output directory for final results
use this path for any final outputs:
/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/git/cosmos3-h-s-s/outputs/evaluation
note: the C-H-S-S evaluation may have different output paths, e.g. folder /lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/git/cosmos-h-surgical-simulator/output

# The Open-H dataset is here:
/lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment


# important
if you encounter any errors you cannot fix autonomously, report back to me for guidance.

# Logging
write all your steps and progress into a log file codex_chss-vs-c3hss.log under
/home/lzbinden/git/cosmos3-h-s-s/outputs/evaluation

# FINAL DELIVERABLES:
a c-h-s-s_vs_c3-h-s-s_report.md file that briefly describes the results and comparison parameters, with pointers to the directory containing the comparison videos (as described) and the FDS plot. The output directory must be timestamp-based folder in base output path
/lustre/fsw/healthcareeng_holoscan/user_data/lzbinden/git/cosmos3-h-s-s/outputs/evaluation
so that repeated runs do not overwrite each other. Also, the report must contain the used checkpoints for both models, as they may vary. 
