#!/bin/bash

# Qwen3.5-only fork -- sequence classification (adds a bridge + linear head on the last token).
# Qwen3.5 is natively multimodal (no `-VL` suffix): dense -> Qwen/Qwen3.5-4B
MODEL_NAME="Qwen/Qwen3.5-4B"

# loss_type should be one of "cross_entropy", "focal_loss",
# "class_balanced_cross_entropy", or "class_balanced_focal_loss".
#
# Image pixel budgets must be a multiple of 32*32 = 1024 (patch 16 x merge 2).
# `--disable_flash_attn2 True` is REQUIRED for Qwen3.5 (FA2 raises CUDA errors; use SDPA).

export PYTHONPATH=src:$PYTHONPATH

NUM_DEVICES=${NUM_DEVICES:-1}
deepspeed --num_gpus "$NUM_DEVICES" src/train/train_cls.py \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path data/receipt/exports/cls_train.json \
    --image_folder data/receipt \
    --eval_path data/receipt/exports/cls_dev.json \
    --eval_image_folder data/receipt \
    --freeze_llm True \
    --freeze_vision_tower False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --loss_type "cross_entropy" \
    --num_labels 2 \
    --disable_flash_attn2 True \
    --output_dir output/qwen_cls \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --head_lr 4e-5 \
    --vision_lr 6e-6 \
    --merger_lr 2e-5 \
    --weight_decay 0.02 \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --eval_strategy "epoch" \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_f1" \
    --greater_is_better True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "epoch" \
    --dataloader_num_workers 4
