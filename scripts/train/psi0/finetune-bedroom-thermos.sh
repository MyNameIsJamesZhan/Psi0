#!/bin/bash

set -e

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,0,7,1}

source .env
source .venv-psi/bin/activate

NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
TRAIN_BATCH_SIZE=64
GLOBAL_BATCH_SIZE=128
# grad_accum derived to hit global batch = train_batch_size * NPROC * grad_accum
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (TRAIN_BATCH_SIZE * NPROC_PER_NODE)))
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES), batch=$TRAIN_BATCH_SIZE, grad_accum=$GRAD_ACCUM, global=$((TRAIN_BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM))"

DATA_ROOT="${DATA_ROOT:-/data2/yunkai/HumanoidEverywhere/third_party/simple-3dgs/data/training/bedroom_v2_move_thermos_final}"
exp="${exp:-bedroom-thermos-v2}"
# To resume an interrupted run, set RESUME_TIMESTAMP to that run's timestamp
# (the trailing .XXXXXXXXXX in its run dir name). It reuses the same run dir and
# loads the latest ckpt_* in it (restoring step/optimizer/scheduler/RNG).
RESUME_TIMESTAMP="${RESUME_TIMESTAMP:-}"

echo "Data root: $DATA_ROOT"
echo "Experiment name: $exp"

args="
finetune_simple_psi0_config \
--seed=292285 \
--exp=$exp \
--train.name=finetune \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=$TRAIN_BATCH_SIZE \
--train.max_checkpoints_to_keep=3 \
--train.gradient_accumulation_steps=$GRAD_ACCUM \
--train.learning_rate=1e-4 \
--train.max_training_steps=40000 \
--train.warmup_ratio=None \
--train.warmup_steps=1000 \
--train.checkpointing_steps=2000 \
--train.validation_steps=500 \
--train.val_num_batches=20 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=cosine \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--log.report_to=wandb \
--log.log_freq=10 \
--train.no-empty-cache-at-val \
--data.root_dir=$DATA_ROOT \
--data.train-repo-ids=train \
--data.val-repo-ids=val \
--data.transform.repack.pad-action-dim=36 \
--data.transform.repack.pad-state-dim=36 \
--data.transform.repack.state-key=states \
--data.transform.field.stat-path=meta/stats_psi0.json \
--data.transform.field.stat-action-key=action \
--data.transform.field.stat-state-key=states \
--data.transform.field.action_norm_type=bounds \
--data.transform.field.no-use-norm-mask \
--data.transform.field.normalize-state \
--data.transform.field.pad-action-dim=36 \
--data.transform.field.pad-state-dim=36 \
--data.transform.model.img-aug \
--data.transform.model.resize.size 180 320 \
--data.transform.model.center_crop.size 180 320 \
--model.model_name_or_path=$PSI_HOME/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
--model.pretrained-action-header-path=$PSI_HOME/cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k \
--model.noise-scheduler=flow \
--model.train-diffusion-steps=1000 \
--model.n_conditions=0 \
--model.action-chunk-size=30 \
--model.action-dim=36 \
--model.action-exec-horizon=30 \
--model.observation-horizon=1 \
--model.odim=36 \
--model.view_feature_dim=2048 \
--model.no-tune-vlm \
--model.no-use_film \
--model.no-combined_temb \
--model.rtc \
--model.max-delay=8
"

if [ -n "$RESUME_TIMESTAMP" ]; then
    echo "Resuming run with timestamp $RESUME_TIMESTAMP (latest ckpt in its run dir)"
    args="$args --train.resume_from_checkpoint=latest --timestamp=$RESUME_TIMESTAMP"
fi

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=29500 scripts/train.py \
    ${args}
