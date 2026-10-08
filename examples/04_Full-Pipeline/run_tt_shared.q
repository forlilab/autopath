#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-0
#SBATCH --array=1-5
# TTMetaD validation vs run_k100_shared.q: same CV, funnel and hills, transition-tempered
# with basins at d = 0.4 nm (S1) and 2.5 nm (bulk), gamma 5. One walker per milestone.
# Runs the ttmetad worktree code from the main example directory (3ptb data lives there).
# usage (from examples/04_Full-Pipeline of the main checkout):
#   sbatch -p alphafold,forli,a100 --exclude=nodea0110,nodea0111,nodeb201 \
#     -J tt_shr -o 3ptb_tts_%a.out -e 3ptb_tts_%a.err \
#     ../../.claude/worktrees/ttmetad/examples/04_Full-Pipeline/run_tt_shared.q [NS] [OUT] [V_TH kJ/mol]
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
source ~/.bashrc
micromamba activate autopath
WT=/gpfs/home/mllanos/forlilab/autopath/.claude/worktrees/ttmetad
export PYTHONPATH=$WT
SEEDS=(milestone_1_frame_0 milestone_2_frame_115 milestone_3_frame_189 milestone_4_frame_250 milestone_5_frame_1116)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID-1))]}
NS=${1:-40}; OUT=${2:-3ptb/metadynamics_tt_shared}; TH=${3:-0}
python $WT/examples/04_Full-Pipeline/run_single_walker_control.py --cv dist --ns "$NS" --seed-from "$SEED" \
    --hill-width 0.01 --hill-height 2.0 --bias-factor 5 --dist-grid 0.2 3.6 1701 \
    --tt-basins 0.4 2.5 --tt-threshold "$TH" \
    --funnel-from 3ptb/funnel_params_narrow_k100.npz --out "$OUT" --bias-dir "$OUT" \
    --run-id "${SEED}_w0" --rct-ustride 5 --dcd-interval 50
