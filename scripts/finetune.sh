#!/bin/bash

# Qwen3.5-only fork.
# Qwen3.5 is natively multimodal (no `-VL` suffix): dense -> Qwen/Qwen3.5-4B, MoE -> Qwen/Qwen3.5-*-A*-Instruct
MODEL_NAME="Qwen/Qwen3.5-4B"

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-1}
NUM_DEVICES=${NUM_DEVICES:-1}
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH

# Qwen3.5 uses ViT patch 16 with a 2x2 spatial merge => one vision token = 32x32 px.
# Every image pixel budget MUST be a multiple of 32*32 = 1024, otherwise the processor
# resizes to the nearest valid grid and you silently get a different token count.
#
# `--disable_flash_attn2 True` is REQUIRED for Qwen3.5: FA2 raised CUDA errors on the
# Qwen3.5 series in local tests, so SDPA is the stable path for now.

deepspeed --num_gpus "$NUM_DEVICES" src/train/train_sft.py \
    --use_liger_kernel False \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path data/receipt/exports/sft_train.json \
    --image_folder data/receipt \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm False \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir output/test_fft \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((512 * 32 * 32)) \
    --image_max_pixels $((1280 * 32 * 32)) \
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
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4
