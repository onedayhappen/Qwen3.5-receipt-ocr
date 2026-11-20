#!/bin/bash
set -euo pipefail
# 票据SFT主线：只训练LLM LoRA，视觉塔和merger保持冻结。
NUM_DEVICES="${NUM_DEVICES:-1}"
export PYTHONPATH="src:${PYTHONPATH:-}"
deepspeed --num_gpus "$NUM_DEVICES" src/train/train_sft.py \
    --model_id "${MODEL_ID:-Qwen/Qwen3.5-4B}" \
    --data_path data/receipt/exports/sft_train.json \
    --eval_path data/receipt/exports/sft_dev.json \
    --image_folder data/receipt --eval_image_folder data/receipt \
    --output_dir models/sft-adapter \
    --lora_enable True --vision_lora False --use_dora False \
    --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens', 'in_proj_qkv', 'visual']" \
    --freeze_llm True --freeze_vision_tower True --freeze_merger True \
    --bf16 True --fp16 False --disable_flash_attn2 True --use_liger_kernel False \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 --gradient_checkpointing True \
    --max_seq_length 8192 --image_min_pixels 65536 --image_max_pixels 1048576 \
    --learning_rate 1e-4 --num_train_epochs 1 --logging_steps 1 \
    --eval_strategy epoch --save_strategy epoch --save_total_limit 2 \
    --generation_eval True --generation_max_new_tokens 2048 \
    --remove_unused_columns False --report_to none "$@"
