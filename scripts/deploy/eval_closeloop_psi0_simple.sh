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
# Episodes (incl. per-episode environment_config) are loaded from here. Must be
# the SAME dataset the checkpoint trained on, else the restored scene/initial
# state won't match what the policy saw. Default = the bedroom grasp-thermos
# held-out test split (what the past=N finetunes train on).
DATA_DIR="${DATA_DIR:-$SG_DIR/data/training/bedroom_v2_grasp_thermos_psi0/$SPLIT}"

# --- Train/eval initial-state alignment (direction B) ----------------------
# The recorded trajectories start with the robot ALREADY at the table (datagen
# drops the walk-up), but env_conf restores the far spawn (X≈1.2). Without this,
# the policy is dropped 1 m back into a state it never saw in training and just
# flails. SIMPLE_TABLE_SPAWN_X teleports the robot to a table-front X at reset so
# the grasp skill is evaluated from the trained initial distribution. Unset it
# (SIMPLE_TABLE_SPAWN_X= ) to fall back to the raw env_conf spawn.
# 0.60 was calibrated by rendering head_stereo_left at reset for several X and
# matching the framing of the recorded training frame 0 (near table edge at the
# bottom of frame). Override to retune if the robot/table geometry changes.
export SIMPLE_TABLE_SPAWN_X="${SIMPLE_TABLE_SPAWN_X-0.60}"

# Instruction alignment: this task natively emits the *move* instruction, but the
# bedroom grasp finetunes train on "pick up the thermos from the nightstand."
# The System-2 VLM is language-conditioned, so eval must feed the trained prompt.
# Unset (SIMPLE_INSTRUCTION_OVERRIDE= ) to use the task's native instruction.
export SIMPLE_INSTRUCTION_OVERRIDE="${SIMPLE_INSTRUCTION_OVERRIDE-pick up the thermos from the nightstand.}"

# Where rollout videos + eval_stats land: $SG_DIR/$EVAL_DIR/psi0/$EXP_NAME/
#   EVAL_DIR  - root (relative to the SIMPLE client CWD = $SG_DIR)
#   EXP_NAME  - per-experiment leaf; defaults to the run-dir name minus the
#               trailing .<timestamp>, so each checkpoint's eval gets its own
#               dir instead of all runs clobbering a single split folder.
# For data_format=lerobot the SIMPLE client uses this leaf ONLY for the output
# path (episodes are loaded from --data-dir), so it's safe to put EXP_NAME here.
EVAL_DIR="${EVAL_DIR:-data/evals}"
EXP_NAME="${EXP_NAME:-$(basename "$RUN_DIR" | sed -E 's/\.[0-9]{10}$//')}"
SIM_MODE="${SIM_MODE:-mujoco_gsplat}"
PORT="${PORT:-22085}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-24}"
SERVE_GPU="${SERVE_GPU:-8}"
SIM_GPU="${SIM_GPU:-8}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
# Global index of the first dataset episode to evaluate. Episodes evaluated are
# range(EPISODE_START, dataset_size)[:NUM_EPISODES]. Output dirs are named by the
# GLOBAL episode index (episode_<idx>), so e.g. EPISODE_START=3 resumes at
# episode_3 without clobbering an earlier episode_0..2 run. Default 0.
EPISODE_START="${EPISODE_START:-0}"

LOGDIR="${LOGDIR:-/tmp/psi0_closeloop}"
mkdir -p "$LOGDIR"
SERVER_LOG="$LOGDIR/server_ckpt${CKPT_STEP}.log"
EVAL_LOG="$LOGDIR/eval_ckpt${CKPT_STEP}_${SPLIT}.log"

echo "[closeloop] RUN_DIR=$RUN_DIR  CKPT_STEP=$CKPT_STEP"
echo "[closeloop] TASK=$TASK  SPLIT=$SPLIT  DATA_DIR=$DATA_DIR"
echo "[closeloop] SERVE_GPU=$SERVE_GPU  SIM_GPU=$SIM_GPU  PORT=$PORT  sim_mode=$SIM_MODE"
echo "[closeloop] output dir: $SG_DIR/$EVAL_DIR/psi0/$EXP_NAME"
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
CUDA_VISIBLE_DEVICES="$SIM_GPU" "$EVAL_BIN" "$TASK" psi0 "$EXP_NAME" \
    --host=localhost --port="$PORT" \
    --data-format=lerobot --data-dir="$DATA_DIR" \
    --eval-dir="$EVAL_DIR" \
    --sim-mode="$SIM_MODE" --headless \
    --num-episodes="$NUM_EPISODES" --episode-start="$EPISODE_START" \
    --max-episode-steps="$MAX_EPISODE_STEPS" \
    --save-video 2>&1 | tee "$EVAL_LOG"

echo "[closeloop] eval finished. rollout videos under $SG_DIR/$EVAL_DIR/psi0/$EXP_NAME/"
