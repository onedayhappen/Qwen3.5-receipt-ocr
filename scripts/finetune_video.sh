#!/bin/bash

# Qwen3.5-only fork -- video SFT.
# Qwen3.5 is natively multimodal (no `-VL` suffix): dense -> Qwen/Qwen3.5-4B
MODEL_NAME="Qwen/Qwen3.5-4B"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-1}
NUM_DEVICES=${NUM_DEVICES:-1}
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# If your dataset is mixed with images and videos, you need to use zero2.
#
# Video is the expensive modality: `--video_max_pixels` is PER FRAME, and Qwen3.5 uses
# temporal_patch 2, so N frames cost N * (pixels / 1024) vision tokens. The old 360*420
# value is not a multiple of 1024 and silently rounds down; use N * 32 * 32 instead.
# `--fps` samples frames, so fps x duration x per-frame cost is your real token bill.
#
# Qwen3.5 also needs `video_metadata` (3D RoPE) instead of Qwen2.x `second_per_grid_ts`;
# the dataset layer handles that switch, so do not pass `second_per_grid_ts` manually.
#
# `--disable_flash_attn2 True` is REQUIRED for Qwen3.5 (FA2 raises CUDA errors; use SDPA).

deepspeed --num_gpus "$NUM_DEVICES" src/train/train_sft.py \
    --use_liger_kernel False \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path data/receipt/exports/video_train.json \
    --image_folder data/receipt \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir output/test_train \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_max_pixels $((128 * 32 * 32)) \
    --fps 1.0 \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 10 \
    --dataloader_num_workers 4
