#!/bin/bash
# Closed-loop eval of a Psi0 checkpoint in the SIMPLE simulator (sibling simple-3dgs checkout).
#
# Spins up the RTC policy server (Psi0 .venv-psi, serve_psi0_simple.sh -> :PORT) and then runs
# the SIMPLE eval client (simple-3dgs/.venv, `eval` console script) which builds the env, connects
# to the server via Psi0Agent/HttpActionClient, rolls out the policy, and records rollout videos.
#
# The server is torn down on exit. Rollout videos + eval_stats land under
#   $SG_DIR/data/evals/psi0/$SPLIT/
#
# Usage:
#   RUN_DIR=.runs/finetune/<run> CKPT_STEP=26000 bash scripts/deploy/eval_closeloop_psi0_simple.sh
#
# Override anything via env vars (defaults target the bedroom move-thermos task on held-out test):
#   TASK, DATA_DIR, SPLIT, SERVE_GPU, SIM_GPU, PORT, NUM_EPISODES, MAX_EPISODE_STEPS, SIM_MODE
set -uo pipefail

PSI_DIR=/data2/yunkai/HumanoidEverywhere/third_party/Psi0
SG_DIR=/data2/yunkai/HumanoidEverywhere/third_party/simple-3dgs

RUN_DIR="${RUN_DIR:?set RUN_DIR to the .runs/.../<run> directory}"
CKPT_STEP="${CKPT_STEP:?set CKPT_STEP (e.g. 26000)}"

TASK="${TASK:-simple/G1WholebodyBedroomMoveThermos-v0}"
SPLIT="${SPLIT:-test}"
DATA_DIR="${DATA_DIR:-$SG_DIR/data/training/bedroom_v2_move_thermos_final/$SPLIT}"
SIM_MODE="${SIM_MODE:-mujoco_gsplat}"
PORT="${PORT:-22085}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-24}"
SERVE_GPU="${SERVE_GPU:-8}"
SIM_GPU="${SIM_GPU:-8}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"

LOGDIR="${LOGDIR:-/tmp/psi0_closeloop}"
mkdir -p "$LOGDIR"
SERVER_LOG="$LOGDIR/server_ckpt${CKPT_STEP}.log"
EVAL_LOG="$LOGDIR/eval_ckpt${CKPT_STEP}_${SPLIT}.log"

echo "[closeloop] RUN_DIR=$RUN_DIR  CKPT_STEP=$CKPT_STEP"
echo "[closeloop] TASK=$TASK  SPLIT=$SPLIT  DATA_DIR=$DATA_DIR"
echo "[closeloop] SERVE_GPU=$SERVE_GPU  SIM_GPU=$SIM_GPU  PORT=$PORT  sim_mode=$SIM_MODE"
echo "[closeloop] server log: $SERVER_LOG   eval log: $EVAL_LOG"

# ----------------------------------------------------------------------------
# 1) start the RTC policy server in its own process group
#    NOTE: call serve_psi0 directly (not serve_psi0_simple.sh). That wrapper uses
#    `uv run`, which re-resolves the project deps and fails on the nvidia-curobo
#    editable that points at the (intentionally uninitialized) SIMPLE submodule.
#    The .venv-psi already has everything installed, so the console script runs fine.
# ----------------------------------------------------------------------------
( cd "$PSI_DIR" && source .venv-psi/bin/activate \
    && CUDA_VISIBLE_DEVICES="$SERVE_GPU" exec serve_psi0 \
        --host 0.0.0.0 --port "$PORT" --policy=psi0 \
        --run-dir="$RUN_DIR" --ckpt-step="$CKPT_STEP" \
        --action-exec-horizon="$ACTION_EXEC_HORIZON" --rtc \
) > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[closeloop] server pid=$SERVER_PID (gpu $SERVE_GPU)"

cleanup() {
    echo "[closeloop] tearing down server (pid $SERVER_PID)"
    kill -TERM "$SERVER_PID" 2>/dev/null
    pkill -TERM -f "serve_psi0 .*--port $PORT" 2>/dev/null
}
trap cleanup EXIT INT TERM

# ----------------------------------------------------------------------------
# 2) wait for the server to start listening on PORT
# ----------------------------------------------------------------------------
echo "[closeloop] waiting for server on :$PORT (model load ~1-2 min) ..."
ready=0
for _ in $(seq 1 90); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null && ! pgrep -f "serve_psi0 .*--port $PORT" >/dev/null; then
        echo "[closeloop] ERROR: server process exited before opening port; see $SERVER_LOG"; tail -20 "$SERVER_LOG"; exit 1
    fi
    if (exec 3<>/dev/tcp/localhost/"$PORT") 2>/dev/null; then ready=1; exec 3>&- 3<&-; break; fi
    sleep 5
done
if [ "$ready" != 1 ]; then
    echo "[closeloop] ERROR: server did not open :$PORT in time; see $SERVER_LOG"; tail -20 "$SERVER_LOG"; exit 1
fi
echo "[closeloop] server is listening on :$PORT"

# ----------------------------------------------------------------------------
# 3) run the SIMPLE eval client (simple-3dgs venv)
# ----------------------------------------------------------------------------
cd "$SG_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
# Headless offscreen rendering: MuJoCo must use EGL (no X11 display). Without this
# the renderer dies with "gladLoadGL error". Per simple-3dgs docs/eval.md.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
# gsplat JIT-compiles its CUDA kernels at first import — needs a >=12.x toolkit
# (system default nvcc is 11.5, too old). Point at CUDA 12.8 for the build.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# NOTE: call the venv's `eval` console script by full path — `eval` is a bash
# builtin and would otherwise shadow it (treating the task id as a command).
EVAL_BIN="$SG_DIR/.venv/bin/eval"
CUDA_VISIBLE_DEVICES="$SIM_GPU" "$EVAL_BIN" "$TASK" psi0 "$SPLIT" \
    --host=localhost --port="$PORT" \
    --data-format=lerobot --data-dir="$DATA_DIR" \
    --sim-mode="$SIM_MODE" --headless \
    --num-episodes="$NUM_EPISODES" --max-episode-steps="$MAX_EPISODE_STEPS" \
    --save-video 2>&1 | tee "$EVAL_LOG"

echo "[closeloop] eval finished. rollout videos under $SG_DIR/data/evals/psi0/$SPLIT/"
