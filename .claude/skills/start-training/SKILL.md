---
name: start-training
description: >-
  Convert a freshly generated SIMPLE bedroom_v2_<task>_final LeRobot dataset into
  Psi0 _psi0 training format (downscale egocentric video to 320x180, build
  self-contained train/val/test splits), generate a 2-GPU finetune launcher, and
  start the training job. Use when the user wants to take a *_final dataset under
  simple-3dgs/data/training and kick off a Psi0 finetune ("start training on
  <task>", "convert <task> and train", "/start-training move_cup").
---

# start-training

End-to-end: `bedroom_v2_<task>_final`  →  `bedroom_v2_<task>_psi0`  →  launcher  →  2-GPU finetune.

## Inputs (ask only if ambiguous)
- **TASK** — slug, e.g. `move_cup`. Resolves the source dir `${TRAIN_ROOT}/bedroom_v2_${TASK}_final`
  and output `${TRAIN_ROOT}/bedroom_v2_${TASK}_psi0`. (You can also pass an explicit `_final` path.)
- **TRAIN_ROOT** — default `/data2/yunkai/HumanoidEverywhere/third_party/simple-3dgs/data/training`.
- **GPUS** — default `0,1` (the task asks for a 2-GPU job).
- **VAL_EPISODES / TEST_EPISODES** — default `6` / `12` (rest → train).

Always work from the Psi0 repo root and run data/training inside `.venv-psi`
(`source .env && source .venv-psi/bin/activate`). `pyarrow`/`ffmpeg` are only present there.

## Why a conversion is needed: `_final` vs `_psi0`
The `_final` dataset is raw SIMPLE datagen output; the Psi0 trainer needs the `_psi0`
schema. Verify with `meta/info.json` before running — do not assume, the metadata can be stale.
Differences this pipeline handles:

1. **Resolution (the "lower resolution" bit).** Since the 3DGS render-time replay
   augmentation landed, `_final` egocentric video is high-res (e.g. **1280×720**, check
   with `ffprobe`). The psi0 datasets store **320×180** (= the trainer's `resize 180 320`
   target, [H,W]). The `_final` `info.json` often *claims* 180×320 while the actual mp4 is
   1280×720 — trust `ffprobe`, not the json. We downscale during the single re-encode.
2. **Schema.** `_final` has `observation.joint_qpos`, `observation.amo_policy_*`,
   `action[43]`, video key `observation.rgb_head_stereo_left`. `_psi0` has 36-dim
   `states`/`action`, split `observation.{hand,arm,leg}_joints`, `observation.prev_*`, and
   video key `observation.images.egocentric`. `postprocess_psi0.py` does this remap.
3. **Splits.** The trainer loads each split as its OWN complete dataset at
   `<root>/<repo_id>` (`--data.train-repo-ids=train --data.val-repo-ids=val`) and reads
   stats from `<root>/train/meta/stats_psi0.json`. `postprocess_psi0.py` only writes a flat
   dataset, so we materialize `train/ val/ test/` sub-datasets (re-indexed from 0, global
   stats copied into each — matching the existing psi0 datasets).

## Pipeline

```bash
cd /data2/yunkai/HumanoidEverywhere/third_party/Psi0
source .env && source .venv-psi/bin/activate

TASK=move_cup
TRAIN_ROOT=/data2/yunkai/HumanoidEverywhere/third_party/simple-3dgs/data/training
SRC=$TRAIN_ROOT/bedroom_v2_${TASK}_final
PSI0=$TRAIN_ROOT/bedroom_v2_${TASK}_psi0
SIMPLE=/data2/yunkai/HumanoidEverywhere/third_party/simple-3dgs

# 0. Sanity: episode count, that every episode length >= skip, source video resolution.
#    (ffprobe a sample mp4; py-read a parquet for required cols + action/joint dims = 43.)

# 1. Convert _final -> flat _psi0, downscaling video to 320x180 in the single re-encode.
#    fps MUST match the source info.json fps (recent datagen = 30, not the default 50);
#    with downsample=1, output fps = source fps. skip=60 trims the warmup frames.
N=$(ls "$SRC"/data/chunk-*/episode_*.parquet | wc -l)
python "$SIMPLE/scripts/postprocess_psi0.py" \
    --sim-root "$SRC" \
    --out-dir "$PSI0" \
    --skip 60 --downsample 1 --fps 30 \
    --total_episodes "$N" \
    --video-key observation.rgb_head_stereo_left \
    --video-hw 180 320

# 2. Split the flat dataset into self-contained train/val/test (in place).
python scripts/data/split_psi0_dataset.py \
    --in-dir "$PSI0" \
    --val-episodes 6 --test-episodes 12

# 3. Verify: train/ val/ test/ each have meta/info.json + meta/stats_psi0.json + data + videos,
#    a converted mp4 is 320x180, and the trainer can construct the dataset (LeRobotDatasetMetadata).
```

## Launcher + launch
Copy `scripts/train/psi0/finetune-bedroom-thermos.sh` to
`scripts/train/psi0/finetune-bedroom-${TASK//_/-}.sh`, then set: `CUDA_VISIBLE_DEVICES`
default to the two GPUs, `DATA_ROOT` → `bedroom_v2_${TASK}_psi0`, `exp` →
`bedroom-${TASK//_/-}`. With 2 GPUs and `TRAIN_BATCH_SIZE=64`, `GRAD_ACCUM` auto-derives to
1 (global batch stays 128). Everything else stays identical to the thermos launcher.

Launch (long-running; start in the background and report the run dir + how to tail logs):
```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train/psi0/finetune-bedroom-move-cup.sh
```
Runs land in `.runs/<exp>/...`. Note: `cfg.auto_tag_run` may `git add/commit/tag` at startup.

## Notes / gotchas
- Run data steps inside `.venv-psi`; system python has no pyarrow.
- `postprocess_psi0.py` lives in the simple-3dgs submodule; the `--video-hw` downscale is a
  local addition there. `scale=W:H` (so `--video-hw 180 320` → `scale=320:180`).
- The psi0 dataloader decodes actual video frames and resizes to 180×320 itself, so
  `info.json` video shape is cosmetic — but we still write it correctly (320×180).
- For a different GPU pair or batch, pass `CUDA_VISIBLE_DEVICES=` / `TRAIN_BATCH_SIZE=` env
  vars to the launcher; resume with `RESUME_TIMESTAMP=<ts>`.
