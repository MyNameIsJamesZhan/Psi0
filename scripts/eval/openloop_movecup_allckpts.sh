#!/bin/bash
# Open-loop eval of all checkpoints for both bedroom-move-cup runs (past0 + past3)
# on the test split (12 episodes). One serial queue per GPU; queues run in parallel.
cd /data2/yunkai/HumanoidEverywhere/third_party/Psi0
# NB: source .env BEFORE `set -u` — .env appends to $LD_LIBRARY_PATH which is
# unset in a fresh shell, and set -u would make that a fatal (silenced) error.
source .env 2>/dev/null
source .venv-psi/bin/activate
set -u

GPUS=(0 1 2 5 7 8)
RUNS=(
  ".runs/finetune/bedroom-move-cup-past0.2606170047"
  ".runs/finetune/bedroom-move-cup-past3.2606170024"
)
STEPS=(10000 20000 30000 36000 38000 40000)
SPLIT=test
STRIDE=4

JOBS=()
for run in "${RUNS[@]}"; do
  for step in "${STEPS[@]}"; do
    JOBS+=("$run|$step")
  done
done

run_one() {
  local job="$1" gpu="$2"
  local run="${job%|*}" step="${job#*|}"
  local outdir="$run/openloop"
  mkdir -p "$outdir"
  local log="$outdir/openloop_${SPLIT}_ckpt${step}.log"
  echo "[GPU $gpu] START $run ckpt $step"
  python examples/simple/openloop_eval.py \
    --run-dir "$run" \
    --ckpt-step "$step" \
    --split "$SPLIT" \
    --num-episodes 12 \
    --stride "$STRIDE" \
    --gpu "$gpu" \
    --output-dir "$outdir" > "$log" 2>&1
  echo "[GPU $gpu] DONE  $run ckpt $step (exit $?)"
}

NG=${#GPUS[@]}
for g in "${!GPUS[@]}"; do
  (
    gpu=${GPUS[$g]}
    for ((i=g; i<${#JOBS[@]}; i+=NG)); do
      run_one "${JOBS[$i]}" "$gpu"
    done
  ) &
done
wait
echo "ALL DONE"
