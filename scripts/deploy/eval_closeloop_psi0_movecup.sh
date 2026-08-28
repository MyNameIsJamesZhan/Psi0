#!/bin/bash
# Closed-loop eval of a Psi0 *move_cup* checkpoint in the SIMPLE simulator.
#
# There is no dedicated move_cup task in simple-3dgs, so we REUSE the bedroom
# move-thermos env (same bedroom_v2 nightstand) and let SIMPLE rebuild the cup
# scene from the move_cup dataset's per-episode `environment_config`
# (--data-format=lerobot). The thermos task's eval-only reset() overrides
# (instruction / table-spawn-X / target-Y recouple) operate on whatever target
# the data loads — here, the cup — driven by the SIMPLE_* env vars below.
#
# Thin wrapper over eval_closeloop_psi0_simple.sh: it only sets move_cup-specific
# defaults, then delegates serve+sim orchestration to that script.
#
# Differences from the thermos default:
#   * RUN_DIR / DATA_DIR / instruction point at move_cup
#   * SIMPLE_EVAL_CAMERAS=head_stereo_left  -> render only the cam the policy uses
#                                              (7 cams -> 1, ~7x faster gsplat)
#   * MAX_EPISODE_STEPS=1500                -> move_cup trajectories are longer
#
# Usage:
#   CKPT_STEP=38000 bash scripts/deploy/eval_closeloop_psi0_movecup.sh
#   # other run / more episodes:
#   RUN_DIR=.runs/finetune/bedroom-move-cup-past3.2606170024 CKPT_STEP=40000 \
#       NUM_EPISODES=3 bash scripts/deploy/eval_closeloop_psi0_movecup.sh
#   # render all cameras again:  SIMPLE_EVAL_CAMERAS= ...
set -uo pipefail

PSI_DIR=/data2/yunkai/HumanoidEverywhere/third_party/Psi0
SG_DIR=/data2/yunkai/HumanoidEverywhere/third_party/simple-3dgs

# --- which checkpoint --------------------------------------------------------
# Default run = the "no past frame" finetune (past0). The other run is
# .runs/finetune/bedroom-move-cup-past3.2606170024 (num_past_frames=3).
export RUN_DIR="${RUN_DIR:-.runs/finetune/bedroom-move-cup-past0.2606170047}"
export CKPT_STEP="${CKPT_STEP:?set CKPT_STEP (e.g. 38000)}"

# --- task / data (move_cup) --------------------------------------------------
# Reuse the move-thermos env as the wrapper; the cup scene comes from the data.
export TASK="${TASK:-simple/G1WholebodyBedroomMoveThermos-v0}"
export SPLIT="${SPLIT:-test}"
export DATA_DIR="${DATA_DIR:-$SG_DIR/data/training/bedroom_v2_move_cup_psi0/$SPLIT}"
# The instruction the move_cup finetune trained on (from its meta/tasks.jsonl).
export SIMPLE_INSTRUCTION_OVERRIDE="${SIMPLE_INSTRUCTION_OVERRIDE-move the cup from the side of the nightstand to the bedside.}"
# Same bedroom_v2 nightstand + +Y-end grasp as thermos -> inherit its table-front
# spawn calibration. Retune (or set empty) if the cup rollout starts misplaced.
export SIMPLE_TABLE_SPAWN_X="${SIMPLE_TABLE_SPAWN_X-0.60}"

# --- rendering + horizon (the two move_cup tweaks) ---------------------------
# Render ONLY head_stereo_left (the single view the policy consumes). Honored by
# simple core (Task.observation_space + Layout.add_camera). Empty -> all cameras.
export SIMPLE_EVAL_CAMERAS="${SIMPLE_EVAL_CAMERAS-head_stereo_left}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1500}"
export NUM_EPISODES="${NUM_EPISODES:-1}"

# --- output / gpu / port -----------------------------------------------------
# Per-checkpoint output leaf so evals don't clobber: data/evals/psi0/$EXP_NAME/
_run_leaf="$(basename "$RUN_DIR" | sed -E 's/\.[0-9]{10}$//')"   # e.g. bedroom-move-cup-past0
export EXP_NAME="${EXP_NAME:-${_run_leaf}-ckpt${CKPT_STEP}}"
export SERVE_GPU="${SERVE_GPU:-4}"
export SIM_GPU="${SIM_GPU:-4}"
export PORT="${PORT:-22085}"

echo "[movecup-closeloop] RUN_DIR=$RUN_DIR CKPT_STEP=$CKPT_STEP"
echo "[movecup-closeloop] cameras=${SIMPLE_EVAL_CAMERAS:-<all>} max_steps=$MAX_EPISODE_STEPS num_episodes=$NUM_EPISODES"
echo "[movecup-closeloop] delegating to eval_closeloop_psi0_simple.sh ..."

exec bash "$PSI_DIR/scripts/deploy/eval_closeloop_psi0_simple.sh"
