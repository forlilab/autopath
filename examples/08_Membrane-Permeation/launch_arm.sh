#!/bin/bash
# Submit one arm: RUNS independent runs x WALKERS walkers, one GPU job per walker.
#   ./launch_arm.sh ARM_DIR RUNS WALKERS [sbatch options --] run_permeation.py options
# e.g.
#   ./launch_arm.sh runs/wt_shared 3 5 -p alphafold,forli,a100 -- \
#       --system system.xml --pdb system.pdb --axis-atoms O1x C2x --method wt --layout shared --ns 100
# Runs land in ARM_DIR/r1..rRUNS; logs in ARM_DIR/slurm. Env: MIN_NS_DAY, GUARD_AFTER_S, ENV_NAME.
set -eu
if (( $# < 4 )); then sed -n 2,8p "$0"; exit 1; fi
ARM_DIR=$(realpath -m "$1"); RUNS=$2; WALKERS=$3; shift 3
SB=()
if [[ " $* " == *" -- "* ]]; then
    while [[ $1 != "--" ]]; do SB+=("$1"); shift; done
    shift
fi
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ARM_DIR/slurm"
ARGS=()
for x in "$@"; do                      # absolute paths, so jobs do not depend on the submit dir
    if [[ -e $x ]]; then ARGS+=("$(realpath "$x")"); else ARGS+=("$x"); fi
done
ARM_DIR=$ARM_DIR WALKERS=$WALKERS AUTOPATH_SRC=$(cd "$HERE/../.." && pwd) \
sbatch --export=ALL --array=1-$(( RUNS * WALKERS )) -J "perm_$(basename "$ARM_DIR")" \
    -o "$ARM_DIR/slurm/%a.out" -e "$ARM_DIR/slurm/%a.err" "${SB[@]}" \
    "$HERE/permeation.sbatch" "${ARGS[@]}"
