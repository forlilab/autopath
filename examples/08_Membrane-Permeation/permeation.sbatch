#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=2-0
#SBATCH --exclude=nodea0110,nodea0111,nodeb0417,nodeb0419,nodeb0423,nodeb0425,nodeb0427,nodeb0429,nodeb0431,nodeb0433,nodeb1201,nodeb1203,nodeb1205,nodeb1207,nodeb1209,nodeb1215,nodeb1217
# One walker per array task: task k -> run r$(( (k-1)/WALKERS + 1 )), walker $(( (k-1)%WALKERS + 1 )).
# Submit through launch_arm.sh, which sets ARM_DIR, WALKERS, AUTOPATH_SRC and the array range.
# Arguments after the script name go to run_permeation.py unchanged.
# Throughput guard: after GUARD_AFTER_S the walker is killed if it runs below MIN_NS_DAY
# (slow cards or CPU contention), so a bad node fails fast instead of holding a run back.
source ~/.bashrc
micromamba activate "${ENV_NAME:-autopath}"
set -u
export PYTHONPATH="$AUTOPATH_SRC${PYTHONPATH:+:$PYTHONPATH}"

K=$SLURM_ARRAY_TASK_ID
R=$(( (K - 1) / WALKERS + 1 ))
W=$(( (K - 1) % WALKERS + 1 ))
RUN_DIR="$ARM_DIR/r$R"
LOG="$RUN_DIR/w$W.log"
mkdir -p "$RUN_DIR"
echo "$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  run r$R walker $W"

python -u "$AUTOPATH_SRC/examples/08_Membrane-Permeation/run_permeation.py" \
    --run-dir "$RUN_DIR" --walker "$W" "$@" > "$LOG" 2>&1 &
PID=$!

MIN=${MIN_NS_DAY:-300}
AFTER=${GUARD_AFTER_S:-900}
t=0
while kill -0 $PID 2>/dev/null; do
    sleep 60; t=$(( t + 60 ))
    if (( t >= AFTER && (t - AFTER) % 1800 == 0 )); then     # at AFTER, then every 30 min
        # StateDataReporter columns: step, time, progress, speed (ns/day), remaining
        speed=$(awk -F'\t' '$4 ~ /^[0-9.]+$/ {s=$4} END {print s}' "$LOG")
        if [[ -n "$speed" ]] && awk -v s="$speed" -v m="$MIN" 'BEGIN {exit !(s < m)}'; then
            echo "throughput guard: $speed ns/day < $MIN on $(hostname); killing walker" | tee -a "$LOG"
            kill $PID; wait $PID; exit 3
        fi
    fi
done
wait $PID
