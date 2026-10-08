#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-0
#SBATCH --array=1-5
# Shared-walker metadynamics on 3ptb with transition tempering (optionally on top of well
# tempering): distance CV Asp189 CG - C7, narrow k100 funnel, basins d = 0.4 / 2.5 nm.
# Seeding: path = one walker per milestone (default); s1 = all walkers in S1.
# Env: HILL (kJ/mol), CV, HW (nm), DGRID, BASINS ("none" = no TT), BF (bias factor),
#      TTBF (TT on top of WT), SYNC=1 (all walkers start together), ISOLATED=1 (one bias per
#      walker), RIDSUF (run-id suffix), WT (code dir; default: the submitting repository).
# usage (from examples/04_Full-Pipeline):
#   sbatch -p alphafold,forli,a100 --exclude=<slow GPUs> -J tt_shr \
#     -o 3ptb_tts_%a.out -e 3ptb_tts_%a.err run_tt_shared.q [NS] [OUT] [V_TH kJ/mol] [path|s1]
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
source ~/.bashrc
micromamba activate autopath
WT=${WT:-$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)}
export PYTHONPATH=$WT
SEEDS=(milestone_1_frame_0 milestone_2_frame_115 milestone_3_frame_189 milestone_4_frame_250 milestone_5_frame_1116)
[ "${4:-path}" = s1 ] && SEEDS=(equil milestone_1_frame_0 equil milestone_1_frame_0 equil)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID-1))]}
RID="${SEED}_w0"; [ "${4:-path}" = s1 ] && RID="s1_${SLURM_ARRAY_TASK_ID}_${SEED}"
RID="${RID}${RIDSUF}"      # RIDSUF: new names for resubmitted walkers
NS=${1:-40}; OUT=${2:-3ptb/metadynamics_tt_shared}; TH=${3:-0}
HILL=${HILL:-2.0}; CV=${CV:-dist}; HW=${HW:-0.01}; DGRID=${DGRID:-"0.2 3.6 1701"}; BASINS=${BASINS:-"0.4 2.5"}; BF=${BF:-5}
TTARG="--tt-basins $BASINS --tt-threshold $TH"; [ -n "$TTBF" ] && TTARG="$TTARG --tt-bias-factor $TTBF"
[ "$BASINS" = none ] && TTARG=""                 # BASINS=none: plain well tempering
BIASDIR="$OUT"; [ -n "$ISOLATED" ] && BIASDIR="$OUT/bias_${RID}"   # ISOLATED=1: one bias per walker
# SYNC=1: every walker waits until all array tasks have started (shared bias filled in parallel)
if [ -n "$SYNC" ]; then
    mkdir -p "$OUT"; touch "$OUT/.ready_${SLURM_ARRAY_TASK_ID}"
    N=${SLURM_ARRAY_TASK_COUNT:-5}; W=0
    until [ "$(ls "$OUT"/.ready_* 2>/dev/null | wc -l)" -ge "$N" ]; do
        sleep 30; W=$((W + 30))
        [ "$W" -ge "${MAXWAIT:-21600}" ] && { echo "only $(ls "$OUT"/.ready_* | wc -l)/$N walkers started in ${W}s: abort"; exit 3; }
    done
    echo "all $N walkers ready after ${W}s"
fi
python $WT/examples/04_Full-Pipeline/run_single_walker_control.py --cv "$CV" --ns "$NS" --seed-from "$SEED" \
    --hill-width "$HW" --hill-height "$HILL" --bias-factor "$BF" --dist-grid $DGRID \
    $TTARG \
    --funnel-from 3ptb/funnel_params_narrow_k100.npz --out "$OUT" --bias-dir "$BIASDIR" \
    --run-id "$RID" --rct-ustride 5 --dcd-interval 50
